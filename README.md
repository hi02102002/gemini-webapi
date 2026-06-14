# Self-hosted Gemini Web API wrapper

This wraps `gemini_webapi` with a small FastAPI server and Docker Compose.

## 1. Prepare env

```bash
cp .env.example .env
nano .env
```

Fill:

- `APP_API_KEY`
- `GEMINI_SECURE_1PSID`
- `GEMINI_SECURE_1PSIDTS`

Get cookies from a browser logged in to `https://gemini.google.com`: DevTools → Network → any request → Request Headers → Cookie.

Use a separate Google account/browser session when possible.

## 2. Run

```bash
docker compose up -d --build
docker compose logs -f gemini-api
```

## 3. Test

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me-long-random-key" \
  -d '{"prompt":"Hello, reply in Vietnamese", "temporary": true}'
```

Streaming:

```bash
curl -N -X POST http://localhost:8000/v1/generate/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me-long-random-key" \
  -d '{"prompt":"Explain Docker in Vietnamese", "temporary": true}'
```

File upload:

```bash
curl -X POST http://localhost:8000/v1/generate-with-files \
  -H "X-API-Key: change-me-long-random-key" \
  -F "prompt=Summarize this file" \
  -F "files=@./sample.pdf"
```

## 4. Nginx

Edit `nginx/gemini-api.conf`, replace `gemini-api.example.com`, then:

```bash
sudo cp nginx/gemini-api.conf /etc/nginx/sites-available/gemini-api.conf
sudo ln -s /etc/nginx/sites-available/gemini-api.conf /etc/nginx/sites-enabled/gemini-api.conf
sudo nginx -t
sudo systemctl reload nginx
```

Add TLS with Certbot:

```bash
sudo certbot --nginx -d gemini-api.example.com
```

## Notes

- Do not expose this without auth. Cookies grant access to your Gemini web session.
- This is a reverse-engineered web wrapper, so it can break when Google changes Gemini web internals.
- If auth fails, refresh the cookies and restart the container.

## 5. OpenAI-compatible chat + tool-call shim

This wrapper also includes a best-effort `/v1/chat/completions` endpoint.

Important limitation: `gemini_webapi` talks to the Gemini web app and does **not** expose native Gemini API `tools` / function-calling. This endpoint simulates tool calling by asking Gemini to return strict JSON, then converts that JSON into OpenAI-style `tool_calls`.

Your client must still execute the function and send the result back as a `role: "tool"` message, same as OpenAI's flow.

Example first request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: abcd1234-5678-90ef-ghij-klmnopqrstuv" \
  -d '{
    "model": "gemini-3-flash-thinking-advanced",
    "messages": [
      {"role": "user", "content": "Weather in Ho Chi Minh City?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather by city name",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'
```

Possible response:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_xxx",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Ho Chi Minh City\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

Then your app executes `get_weather`, and sends a second request with the tool result:

```json
{
  "messages": [
    {"role": "user", "content": "Weather in Ho Chi Minh City?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_xxx",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\":\"Ho Chi Minh City\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_xxx",
      "content": "{\"city\":\"Ho Chi Minh City\",\"temp_c\":31,\"condition\":\"Cloudy\"}"
    }
  ],
  "tools": [/* same tools */]
}
```
