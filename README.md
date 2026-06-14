# Gemini WebAPI self-host wrapper — AI SDK compatible

This wraps `gemini_webapi` with a small FastAPI server and exposes an OpenAI-compatible endpoint that works with `@ai-sdk/openai-compatible`.

> Important: `gemini_webapi` is a reverse-engineered wrapper around the Gemini web app. Tool calling in this server is a prompt-based shim, not native Gemini function calling.

## Supported endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `GET /v1/models` | OpenAI-compatible model list |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions for AI SDK |
| `POST /v1/generate` | Simple direct wrapper endpoint |
| `POST /v1/generate/stream` | Simple direct SSE wrapper endpoint |
| `POST /v1/generate-with-files` | Simple direct file upload endpoint |

`/v1/chat/completions` accepts both non-streaming and streaming requests:

- `stream: false` returns a normal OpenAI-style `chat.completion` response.
- `stream: true` returns OpenAI-style `text/event-stream` chunks ending with `data: [DONE]`.
- When tools are provided, the server asks Gemini to emit JSON and converts it into OpenAI-style `tool_calls`.

## 1. Prepare env

```bash
cp .env.example .env
nano .env
```

Fill:

```env
APP_API_KEY=your-long-random-key
GEMINI_SECURE_1PSID=...
GEMINI_SECURE_1PSIDTS=...
OPENAI_COMPAT_MODEL=gemini-web
# Optional: GEMINI_WEB_MODEL=<supported-gemini_webapi-model>
```

Get Gemini cookies from a browser logged in to `https://gemini.google.com`:

DevTools → Network → any request → Request Headers → Cookie.

Use a separate Google account/browser session when possible.

## 2. Run

```bash
docker compose up -d --build
docker compose logs -f gemini-api
```

## 3. Test with curl

Health:

```bash
curl http://localhost:8000/health
```

Models:

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer abcd1234-5678-90ef-ghij-klmnopqrstuv"
```

Chat completion:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer abcd1234-5678-90ef-ghij-klmnopqrstuv" \
  -d '{
    "model": "gemini-web",
    "messages": [{"role":"user","content":"Reply in Vietnamese: hello"}]
  }'
```

Streaming:

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer abcd1234-5678-90ef-ghij-klmnopqrstuv" \
  -d '{
    "model": "gemini-web",
    "stream": true,
    "messages": [{"role":"user","content":"Explain Docker in Vietnamese"}]
  }'
```

Tool call:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer abcd1234-5678-90ef-ghij-klmnopqrstuv" \
  -d '{
    "model": "gemini-web",
    "messages": [{"role":"user","content":"Weather in Ho Chi Minh City?"}],
    "tools": [
      {
        "type":"function",
        "function":{
          "name":"get_weather",
          "description":"Get current weather by city",
          "parameters":{
            "type":"object",
            "properties":{"city":{"type":"string"}},
            "required":["city"]
          }
        }
      }
    ],
    "tool_choice":"auto"
  }'
```

## 4. Use with AI SDK

Install:

```bash
pnpm add ai @ai-sdk/openai-compatible zod
```

Create provider:

```ts
// lib/gemini-web.ts
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';

export const geminiWeb = createOpenAICompatible({
  name: 'gemini-web',
  baseURL: process.env.GEMINI_WEB_BASE_URL ?? 'http://localhost:8000/v1',
  apiKey: process.env.GEMINI_WEB_API_KEY,
});
```

Generate text:

```ts
import { generateText } from 'ai';
import { geminiWeb } from './lib/gemini-web';

const result = await generateText({
  model: geminiWeb('gemini-web'),
  prompt: 'Viết 3 ý tưởng micro SaaS cho developer Việt Nam',
});

console.log(result.text);
```

Use tools:

```ts
import { generateText, stepCountIs, tool } from 'ai';
import { z } from 'zod';
import { geminiWeb } from './lib/gemini-web';

const result = await generateText({
  model: geminiWeb('gemini-web'),
  prompt: 'Tìm phòng trọ Quận 7 dưới 5 triệu',
  tools: {
    searchRooms: tool({
      description: 'Search rental rooms from database',
      inputSchema: z.object({
        district: z.string(),
        maxPrice: z.number(),
      }),
      execute: async ({ district, maxPrice }) => {
        return [
          { title: 'Studio Tân Thuận', price: 4_500_000, district },
          { title: 'Phòng gần Lotte Q7', price: 5_000_000, district },
        ];
      },
    }),
  },
  stopWhen: stepCountIs(3),
});

console.log(result.text);
```

Stream in a Next.js route:

```ts
// app/api/chat/route.ts
import { streamText } from 'ai';
import { geminiWeb } from '@/lib/gemini-web';

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: geminiWeb('gemini-web'),
    messages,
  });

  return result.toUIMessageStreamResponse();
}
```

## 5. Nginx

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

Then AI SDK config becomes:

```ts
export const geminiWeb = createOpenAICompatible({
  name: 'gemini-web',
  baseURL: 'https://gemini-api.example.com/v1',
  apiKey: process.env.GEMINI_WEB_API_KEY,
});
```

## Notes and limitations

- Do not expose this without auth. Cookies grant access to your Gemini web session.
- Prefer `Authorization: Bearer <APP_API_KEY>` for AI SDK.
- `OPENAI_COMPAT_MODEL` is just the model id shown to OpenAI-compatible clients. Leave `GEMINI_WEB_MODEL` empty unless you know the internal model values supported by `gemini_webapi`.
- `X-API-Key` is still supported for curl/custom clients.
- Tool calling is prompt-based. Always validate tool arguments in your own backend.
- For dangerous actions such as payment, deletion, or sending real emails, require human confirmation.
- This wrapper can break when Google changes Gemini web internals.
- If auth fails, refresh the cookies and restart the container.
