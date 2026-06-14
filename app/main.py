import asyncio
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from gemini_webapi import GeminiClient
from gemini_webapi.exceptions import AuthError
from pydantic import BaseModel, Field


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str | None = None
    temporary: bool = True


class GenerateResponse(BaseModel):
    text: str | None
    images: list[dict[str, Any]]
    videos: list[dict[str, Any]]
    media: list[dict[str, Any]]
    metadata: Any | None = None


class ErrorResponse(BaseModel):
    detail: str



class ChatMessage(BaseModel):
    role: Literal["developer", "system", "user", "assistant", "tool"]
    content: Any | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(..., min_length=1)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = "auto"
    temperature: float | None = None  # Accepted for compatibility; gemini_webapi does not expose this reliably.
    max_tokens: int | None = None     # Accepted for compatibility; gemini_webapi does not expose this reliably.
    top_p: float | None = None        # Accepted for compatibility.
    stop: str | list[str] | None = None
    response_format: dict[str, Any] | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    temporary: bool = True


API_KEY = os.getenv("APP_API_KEY")
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_COMPAT_DEFAULT_MODEL", "gemini-web")
COMPAT_MODEL_IDS = [
    model.strip()
    for model in os.getenv(
        "OPENAI_COMPAT_MODELS",
        "gemini-web,gemini-3-flash-thinking-advanced",
    ).split(",")
    if model.strip()
]
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
REQUEST_TIMEOUT = float(os.getenv("GEMINI_REQUEST_TIMEOUT", "300"))
INIT_TIMEOUT = float(os.getenv("GEMINI_INIT_TIMEOUT", "30"))
AUTO_CLOSE_DELAY = int(os.getenv("GEMINI_AUTO_CLOSE_DELAY", "300"))

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


def require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    if not API_KEY:
        return

    bearer_token = None
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            bearer_token = token

    if x_api_key != API_KEY and bearer_token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _get_attr(obj: Any, name: str) -> Any:
    try:
        value = getattr(obj, name, None)
        if callable(value):
            return None
        return value
    except Exception:
        return None


def _serialize_items(items: Any) -> list[dict[str, Any]]:
    if not items:
        return []

    serialized: list[dict[str, Any]] = []
    for item in items:
        data = {
            "type": item.__class__.__name__,
            "url": _get_attr(item, "url"),
            "title": _get_attr(item, "title"),
            "alt": _get_attr(item, "alt"),
            "description": _get_attr(item, "description"),
        }
        serialized.append({k: v for k, v in data.items() if v is not None})
    return serialized


def serialize_output(output: Any) -> dict[str, Any]:
    return {
        "text": _get_attr(output, "text"),
        "images": _serialize_items(_get_attr(output, "images")),
        "videos": _serialize_items(_get_attr(output, "videos")),
        "media": _serialize_items(_get_attr(output, "media")),
        "metadata": _get_attr(output, "metadata"),
    }



def _extract_json_object(text: str | None) -> dict[str, Any] | None:
    """Best-effort parser for model JSON output."""
    if not text:
        return None

    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _content_to_prompt_text(content: Any) -> Any:
    if content is None or isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                parts.append(json.dumps(part, ensure_ascii=False))
                continue

            part_type = part.get("type")
            if part_type in {"text", "input_text"}:
                text = part.get("text")
                if text:
                    parts.append(str(text))
            elif part_type == "image_url":
                image_url = part.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                parts.append(f"[image_url: {url}]")
            elif part_type in {"file", "input_file"}:
                parts.append("[file attachment omitted]")
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
        return "\n".join(parts)

    return json.dumps(content, ensure_ascii=False)


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        item: dict[str, Any] = {
            "role": msg.role,
            "content": _content_to_prompt_text(msg.content),
        }
        if msg.name:
            item["name"] = msg.name
        if msg.tool_call_id:
            item["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            item["tool_calls"] = msg.tool_calls
        normalized.append({k: v for k, v in item.items() if v is not None})

    return json.dumps(normalized, ensure_ascii=False, indent=2)


def _should_prompt_for_tools(req: ChatCompletionRequest) -> bool:
    return bool(req.tools) and req.tool_choice != "none"


def _build_chat_prompt(req: ChatCompletionRequest) -> str:
    base = [
        "You are a chat completion model behind an OpenAI-compatible HTTP wrapper.",
        "Conversation messages are provided as JSON. Follow the latest user request and respect system messages.",
    ]

    if req.response_format:
        base.extend([
            "The client requested this response_format:",
            json.dumps(req.response_format, ensure_ascii=False),
        ])

    if req.stop:
        base.extend([
            "Stop sequences were provided. Do not include these stop sequences in the final answer:",
            json.dumps(req.stop, ensure_ascii=False),
        ])

    if _should_prompt_for_tools(req):
        base.extend([
            "The client has provided callable tools. You must decide whether to call a tool or answer normally.",
            "Return ONLY valid JSON. No markdown, no prose outside JSON.",
            "If a tool is needed, return this exact shape:",
            '{"type":"tool_calls","tool_calls":[{"id":"call_<unique>","type":"function","function":{"name":"tool_name","arguments":{"arg":"value"}}}]}',
            "If no tool is needed, return this exact shape:",
            '{"type":"message","content":"your assistant response"}',
            "function.arguments MUST be a JSON object, not a JSON string.",
            f"tool_choice: {json.dumps(req.tool_choice, ensure_ascii=False)}",
            "Tools:",
            json.dumps(req.tools, ensure_ascii=False, indent=2),
        ])
        if any(msg.role == "tool" for msg in req.messages):
            base.append("Tool results are already present in the conversation. Prefer returning a final message unless another tool call is clearly required.")
    else:
        base.append("Answer naturally. Do not mention this wrapper unless asked.")

    base.append("Conversation messages:")
    base.append(_messages_to_prompt(req.messages))
    return "\n\n".join(base)


def _estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _usage(prompt: str, completion: str | None = None) -> dict[str, int]:
    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = _estimate_tokens(completion)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        args = fn.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        normalized_calls.append({
            "id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": call.get("type") or "function",
            "function": {
                "name": fn.get("name"),
                "arguments": args,
            },
        })
    return normalized_calls


def _openai_chat_response(
    *,
    model: str,
    prompt: str,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = _normalize_tool_calls(tool_calls)
        message["content"] = None

    completion_text = content
    if tool_calls:
        completion_text = json.dumps(message["tool_calls"], ensure_ascii=False)

    return {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": _usage(prompt, completion_text),
        "system_fingerprint": None,
    }


def _openai_chat_chunk(
    *,
    chunk_id: str,
    created: int,
    model: str,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": None,
    }
    if delta is None and usage is not None:
        payload["choices"] = []
    else:
        payload["choices"] = [{
            "index": 0,
            "delta": delta or {},
            "finish_reason": finish_reason,
        }]
    if usage is not None:
        payload["usage"] = usage
    return payload


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    secure_1psid = os.getenv("GEMINI_SECURE_1PSID")
    secure_1psidts = os.getenv("GEMINI_SECURE_1PSIDTS", "")
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    account_index_raw = os.getenv("GEMINI_ACCOUNT_INDEX")
    account_index = int(account_index_raw) if account_index_raw else None

    if not secure_1psid:
        raise RuntimeError("Missing GEMINI_SECURE_1PSID in environment")

    client = GeminiClient(
        secure_1psid=secure_1psid,
        secure_1psidts=secure_1psidts,
        proxy=proxy,
        account_index=account_index,
        verify=not _env_bool("GEMINI_SKIP_VERIFY", False),
    )

    try:
        await client.init(
            timeout=INIT_TIMEOUT,
            auto_close=True,
            close_delay=AUTO_CLOSE_DELAY,
            auto_refresh=True,
            verbose=_env_bool("GEMINI_VERBOSE", False),
        )
    except AuthError as exc:
        raise RuntimeError("Gemini authentication failed. Refresh your cookies.") from exc

    app.state.gemini_client = client
    yield
    await client.close()


app = FastAPI(
    title="Self-hosted Gemini Web API Wrapper",
    version="0.1.0",
    lifespan=lifespan,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models() -> dict[str, Any]:
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": created,
                "owned_by": "gemini-webapi-wrapper",
            }
            for model in COMPAT_MODEL_IDS
        ],
    }


@app.get("/v1/models/{model_id}", dependencies=[Depends(require_api_key)])
async def retrieve_model(model_id: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "gemini-webapi-wrapper",
    }


@app.post("/v1/generate", dependencies=[Depends(require_api_key)], response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"temporary": req.temporary}
    if req.model:
        kwargs["model"] = req.model

    async with semaphore:
        try:
            output = await app.state.gemini_client.generate_content(
                req.prompt,
                timeout=REQUEST_TIMEOUT,
                **kwargs,
            )
            return serialize_output(output)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/generate/stream", dependencies=[Depends(require_api_key)])
async def generate_stream(req: GenerateRequest) -> StreamingResponse:
    kwargs: dict[str, Any] = {"temporary": req.temporary}
    if req.model:
        kwargs["model"] = req.model

    async def event_stream() -> AsyncIterator[str]:
        async with semaphore:
            try:
                async for chunk in app.state.gemini_client.generate_content_stream(
                    req.prompt,
                    timeout=REQUEST_TIMEOUT,
                    **kwargs,
                ):
                    delta = _get_attr(chunk, "text_delta")
                    if delta:
                        yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"

                yield "event: done\ndata: {}\n\n"
            except Exception as exc:
                payload = json.dumps({"detail": str(exc)}, ensure_ascii=False)
                yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")



async def _run_chat_completion(req: ChatCompletionRequest, prompt: str, model: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"temporary": req.temporary}
    if req.model:
        kwargs["model"] = req.model

    async with semaphore:
        output = await app.state.gemini_client.generate_content(
            prompt,
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
        text = _get_attr(output, "text") or ""

    if _should_prompt_for_tools(req):
        parsed = _extract_json_object(text)
        if parsed and parsed.get("type") == "tool_calls" and isinstance(parsed.get("tool_calls"), list):
            return _openai_chat_response(
                model=model,
                prompt=prompt,
                tool_calls=parsed["tool_calls"],
                finish_reason="tool_calls",
            )
        if parsed and parsed.get("type") == "message":
            return _openai_chat_response(
                model=model,
                prompt=prompt,
                content=str(parsed.get("content", "")),
                finish_reason="stop",
            )

    return _openai_chat_response(
        model=model,
        prompt=prompt,
        content=text,
        finish_reason="stop",
    )


async def _chat_completion_stream(req: ChatCompletionRequest, prompt: str, model: str) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl_{uuid.uuid4().hex}"
    created = int(time.time())
    include_usage = bool(req.stream_options and req.stream_options.get("include_usage"))

    try:
        yield _sse(_openai_chat_chunk(
            chunk_id=chunk_id,
            created=created,
            model=model,
            delta={"role": "assistant"},
        ))

        if _should_prompt_for_tools(req):
            result = await _run_chat_completion(req, prompt, model)
            choice = result["choices"][0]
            message = choice["message"]

            if message.get("tool_calls"):
                delta_tool_calls = []
                for index, call in enumerate(message["tool_calls"]):
                    delta_tool_calls.append({
                        "index": index,
                        "id": call["id"],
                        "type": call["type"],
                        "function": call["function"],
                    })
                yield _sse(_openai_chat_chunk(
                    chunk_id=chunk_id,
                    created=created,
                    model=model,
                    delta={"tool_calls": delta_tool_calls},
                ))
            elif message.get("content"):
                yield _sse(_openai_chat_chunk(
                    chunk_id=chunk_id,
                    created=created,
                    model=model,
                    delta={"content": message["content"]},
                ))

            yield _sse(_openai_chat_chunk(
                chunk_id=chunk_id,
                created=created,
                model=model,
                delta={},
                finish_reason=choice["finish_reason"],
            ))
            if include_usage:
                yield _sse(_openai_chat_chunk(
                    chunk_id=chunk_id,
                    created=created,
                    model=model,
                    usage=result["usage"],
                ))
            yield "data: [DONE]\n\n"
            return

        kwargs: dict[str, Any] = {"temporary": req.temporary}
        if req.model:
            kwargs["model"] = req.model

        completion_parts: list[str] = []
        async with semaphore:
            async for chunk in app.state.gemini_client.generate_content_stream(
                prompt,
                timeout=REQUEST_TIMEOUT,
                **kwargs,
            ):
                delta = _get_attr(chunk, "text_delta")
                if not delta:
                    continue
                completion_parts.append(delta)
                yield _sse(_openai_chat_chunk(
                    chunk_id=chunk_id,
                    created=created,
                    model=model,
                    delta={"content": delta},
                ))

        yield _sse(_openai_chat_chunk(
            chunk_id=chunk_id,
            created=created,
            model=model,
            delta={},
            finish_reason="stop",
        ))
        if include_usage:
            completion = "".join(completion_parts)
            yield _sse(_openai_chat_chunk(
                chunk_id=chunk_id,
                created=created,
                model=model,
                usage=_usage(prompt, completion),
            ))
        yield "data: [DONE]\n\n"
    except Exception as exc:
        payload = {
            "error": {
                "message": str(exc),
                "type": "server_error",
                "code": None,
            }
        }
        yield _sse(payload)
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)], response_model=None)
async def chat_completions(req: ChatCompletionRequest) -> Any:
    """
    OpenAI-compatible chat endpoint with prompt-based tool-call shim.

    Limitation: gemini_webapi does not expose native Gemini function calling. This endpoint asks Gemini
    to emit structured JSON and converts it to OpenAI-style tool_calls. Your client still executes the
    tool and sends the tool result back as a `role: "tool"` message.
    """
    prompt = _build_chat_prompt(req)
    model = req.model or DEFAULT_CHAT_MODEL

    if req.stream:
        return StreamingResponse(_chat_completion_stream(req, prompt, model), media_type="text/event-stream")

    try:
        return await _run_chat_completion(req, prompt, model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/generate-with-files", dependencies=[Depends(require_api_key)])
async def generate_with_files(
    prompt: str = Form(...),
    model: str | None = Form(default=None),
    temporary: bool = Form(default=True),
    files: list[UploadFile] = File(default=[]),
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"temporary": temporary}
    if model:
        kwargs["model"] = model

    async with semaphore:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_paths: list[str] = []
                for upload in files:
                    safe_name = Path(upload.filename or "upload.bin").name
                    path = Path(tmp_dir) / safe_name
                    path.write_bytes(await upload.read())
                    file_paths.append(str(path))

                output = await app.state.gemini_client.generate_content(
                    prompt,
                    files=file_paths or None,
                    timeout=REQUEST_TIMEOUT,
                    **kwargs,
                )
                return serialize_output(output)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
