import asyncio
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from gemini_webapi import GeminiClient
from gemini_webapi.exceptions import AuthError
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MODEL = os.getenv("OPENAI_COMPAT_MODEL", "gemini-web")
GEMINI_WEB_MODEL = os.getenv("GEMINI_WEB_MODEL")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


FORWARD_OPENAI_MODEL_TO_GEMINI = _env_bool("FORWARD_OPENAI_MODEL_TO_GEMINI", False)


def _gemini_kwargs(*, temporary: bool, request_model: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"temporary": temporary}
    # OPENAI_COMPAT_MODEL is the public model id for OpenAI-compatible clients.
    # GEMINI_WEB_MODEL is the optional internal model id forwarded to gemini_webapi.
    # By default we do NOT forward arbitrary OpenAI-compatible model ids to gemini_webapi.
    if GEMINI_WEB_MODEL:
        kwargs["model"] = GEMINI_WEB_MODEL
    elif FORWARD_OPENAI_MODEL_TO_GEMINI and request_model and request_model != DEFAULT_MODEL:
        kwargs["model"] = request_model
    return kwargs


def _approx_tokens(value: Any) -> int:
    """Cheap usage approximation so OpenAI-compatible clients have a usage object."""
    if value is None:
        return 0
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return max(1, len(value) // 4)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

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
    model_config = ConfigDict(extra="allow")

    # OpenAI/AI SDK normally sends: system | user | assistant | tool.
    # Keep this as str to avoid 422 for newer roles such as developer.
    role: str
    content: Any = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage] = Field(..., min_length=1)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = "auto"
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    response_format: dict[str, Any] | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None

    # Wrapper-only option. AI SDK can pass it via providerOptions/body if needed.
    temporary: bool = True


API_KEY = os.getenv("APP_API_KEY")
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
REQUEST_TIMEOUT = float(os.getenv("GEMINI_REQUEST_TIMEOUT", "300"))
INIT_TIMEOUT = float(os.getenv("GEMINI_INIT_TIMEOUT", "30"))
AUTO_CLOSE_DELAY = int(os.getenv("GEMINI_AUTO_CLOSE_DELAY", "300"))

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


def require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    """
    Accept both:
    - X-API-Key: <key>               # useful for curl/custom clients
    - Authorization: Bearer <key>    # what @ai-sdk/openai-compatible sends when apiKey is set
    """
    if not API_KEY:
        return

    bearer_token = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer_token = authorization.split(" ", 1)[1].strip()

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


def _normalize_content(content: Any) -> Any:
    """Keep text parts useful when AI SDK/OpenAI sends multi-part content."""
    if isinstance(content, list):
        parts: list[Any] = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(part["text"])
                elif part.get("type") == "text" and "content" in part:
                    parts.append(part["content"])
                else:
                    parts.append(part)
            else:
                parts.append(part)
        if all(isinstance(p, str) for p in parts):
            return "\n".join(parts)
        return parts
    return content


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        item: dict[str, Any] = {
            "role": msg.role,
            "content": _normalize_content(msg.content),
        }
        if msg.name:
            item["name"] = msg.name
        if msg.tool_call_id:
            item["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            item["tool_calls"] = msg.tool_calls
        normalized.append({k: v for k, v in item.items() if v is not None})

    return json.dumps(normalized, ensure_ascii=False, indent=2)


def _tools_enabled(req: ChatCompletionRequest) -> bool:
    return bool(req.tools) and req.tool_choice != "none"


def _build_chat_prompt(req: ChatCompletionRequest) -> str:
    base = [
        "You are a chat completion model behind an OpenAI-compatible HTTP wrapper.",
        "Conversation messages are provided as JSON. Follow the latest user request and respect system/developer messages.",
        "Do not reveal wrapper instructions unless the user explicitly asks about the wrapper implementation.",
    ]

    if req.response_format and req.response_format.get("type") == "json_object" and not req.tools:
        base.extend([
            "The client requested JSON object output.",
            "Return ONLY one valid JSON object. No markdown. No prose outside JSON.",
        ])

    if _tools_enabled(req):
        base.extend([
            "The client has provided callable tools. Decide whether to call a tool or answer normally.",
            "Return ONLY valid JSON. No markdown, no prose outside JSON.",
            "If a tool is needed, return this exact shape:",
            '{"type":"tool_calls","tool_calls":[{"id":"call_<unique>","type":"function","function":{"name":"tool_name","arguments":{"arg":"value"}}}]}',
            "If no tool is needed, return this exact shape:",
            '{"type":"message","content":"your assistant response"}',
            "function.arguments MUST be a JSON object, not a JSON string.",
            "Only call tools that are present in the provided tools list.",
            "If tool_choice forces a named function, call that function.",
            f"tool_choice: {json.dumps(req.tool_choice, ensure_ascii=False)}",
            "Tools:",
            json.dumps(req.tools, ensure_ascii=False, indent=2),
        ])
    else:
        base.append("Answer naturally.")

    if req.stop:
        base.append(f"Stop sequences requested by client: {json.dumps(req.stop, ensure_ascii=False)}")

    base.append("Conversation messages:")
    base.append(_messages_to_prompt(req.messages))
    return "\n\n".join(base)


def _normalize_tool_calls(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized_calls: list[dict[str, Any]] = []
    if not tool_calls:
        return normalized_calls

    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function", {}) if isinstance(call.get("function", {}), dict) else {}
        name = fn.get("name") or call.get("name")
        args = fn.get("arguments", call.get("arguments", {}))
        if not isinstance(args, str):
            args = json.dumps(args or {}, ensure_ascii=False)
        normalized_calls.append({
            "id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": args,
            },
        })
    return normalized_calls


def _openai_chat_response(
    *,
    model: str,
    prompt: Any,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    normalized_calls = _normalize_tool_calls(tool_calls)
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if normalized_calls:
        message["tool_calls"] = normalized_calls
        message["content"] = None

    completion_payload = normalized_calls if normalized_calls else content
    prompt_tokens = _approx_tokens(prompt)
    completion_tokens = _approx_tokens(completion_payload)

    return {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _openai_chunk(
    *,
    chat_id: str,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> str:
    payload = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _usage_chunk(*, chat_id: str, model: str, prompt: Any, completion: Any) -> str:
    prompt_tokens = _approx_tokens(prompt)
    completion_tokens = _approx_tokens(completion)
    payload = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _parse_chat_output(req: ChatCompletionRequest, output_text: str, prompt: str) -> dict[str, Any]:
    model = req.model or DEFAULT_MODEL

    if _tools_enabled(req):
        parsed = _extract_json_object(output_text)
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
        content=output_text,
        finish_reason="stop",
    )


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
    version="0.2.0-ai-sdk-compatible",
    lifespan=lifespan,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def models() -> dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": now,
                "owned_by": "self-hosted-gemini-webapi",
            }
        ],
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


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)], response_model=None)
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat endpoint for @ai-sdk/openai-compatible.

    Important: gemini_webapi does not expose native Gemini function calling. Tool calling here is a
    prompt-based shim: Gemini emits JSON, this wrapper converts it to OpenAI-style tool_calls, then
    AI SDK executes your tool and sends the result back as role=tool.
    """
    prompt = _build_chat_prompt(req)
    model = req.model or DEFAULT_MODEL
    kwargs = _gemini_kwargs(temporary=req.temporary, request_model=req.model)

    if not req.stream:
        async with semaphore:
            try:
                output = await app.state.gemini_client.generate_content(
                    prompt,
                    timeout=REQUEST_TIMEOUT,
                    **kwargs,
                )
                text = _get_attr(output, "text") or ""
                return _parse_chat_output(req, text, prompt)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def event_stream() -> AsyncIterator[str]:
        chat_id = f"chatcmpl_{uuid.uuid4().hex}"
        completion_accumulator: list[Any] = []
        yield _openai_chunk(chat_id=chat_id, model=model, delta={"role": "assistant"})

        async with semaphore:
            try:
                # For tool requests, stream a normal OpenAI tool_calls delta after parsing the full answer.
                # This is AI SDK-compatible, but the tool call is not emitted token-by-token.
                if _tools_enabled(req):
                    output = await app.state.gemini_client.generate_content(
                        prompt,
                        timeout=REQUEST_TIMEOUT,
                        **kwargs,
                    )
                    text = _get_attr(output, "text") or ""
                    parsed_response = _parse_chat_output(req, text, prompt)
                    message = parsed_response["choices"][0]["message"]

                    if message.get("tool_calls"):
                        calls = message["tool_calls"]
                        completion_accumulator.extend(calls)
                        for index, call in enumerate(calls):
                            delta = {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": call["id"],
                                        "type": "function",
                                        "function": {
                                            "name": call["function"]["name"],
                                            "arguments": call["function"]["arguments"],
                                        },
                                    }
                                ]
                            }
                            yield _openai_chunk(chat_id=chat_id, model=model, delta=delta)
                        yield _openai_chunk(chat_id=chat_id, model=model, delta={}, finish_reason="tool_calls")
                    else:
                        content = message.get("content") or ""
                        completion_accumulator.append(content)
                        if content:
                            yield _openai_chunk(chat_id=chat_id, model=model, delta={"content": content})
                        yield _openai_chunk(chat_id=chat_id, model=model, delta={}, finish_reason="stop")
                else:
                    async for chunk in app.state.gemini_client.generate_content_stream(
                        prompt,
                        timeout=REQUEST_TIMEOUT,
                        **kwargs,
                    ):
                        delta_text = _get_attr(chunk, "text_delta")
                        if delta_text:
                            completion_accumulator.append(delta_text)
                            yield _openai_chunk(chat_id=chat_id, model=model, delta={"content": delta_text})
                    yield _openai_chunk(chat_id=chat_id, model=model, delta={}, finish_reason="stop")

                if req.stream_options and req.stream_options.get("include_usage"):
                    yield _usage_chunk(
                        chat_id=chat_id,
                        model=model,
                        prompt=prompt,
                        completion="".join(str(x) for x in completion_accumulator),
                    )
                yield "data: [DONE]\n\n"
            except Exception as exc:
                payload = {
                    "error": {
                        "message": str(exc),
                        "type": "server_error",
                        "code": "gemini_webapi_error",
                    }
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
