<div align="center">

# grok-newapi-compat

**English** | [简体中文](./README.zh-CN.md)

</div>

---

A small, single-auth compatibility bridge that lets **Grok CLI** talk to **NewAPI** as if NewAPI exposed an xAI/Grok-compatible Responses API.

It accepts Grok-style `POST /v1/responses` traffic, routes it to NewAPI, and normalizes the response/SSE wire format back into the strict shape Grok CLI expects. NewAPI remains the only authentication, quota, model-access, logging, and rate-limit authority.

> Not affiliated with xAI, OpenAI, Grok, or NewAPI.

---

## Why

Grok CLI speaks a strict xAI/Grok Responses dialect. Many OpenAI-compatible gateways speak OpenAI Responses or Chat Completions, but the wire details are not always identical:

- some required xAI response fields are omitted;
- streamed `output_item.added` shapes may be incomplete;
- citation annotations may omit fields Grok requires;
- hosted/server-side tools are lost if everything is converted to Chat Completions.

This bridge keeps both worlds usable:

```text
Grok CLI
  -> grok-newapi-compat
       -> NewAPI /v1/responses   for server-side tools
       -> NewAPI /v1/chat/completions
  -> strict xAI/Grok Responses normalization
  -> Grok CLI
```

---

## Features

- **Single authentication model**
  - no bridge-side API keys;
  - no secondary token store;
  - `Authorization` is passed through unchanged to NewAPI.
- **Responses API compatibility**
  - accepts `POST /v1/responses`, `/responses`, and `/openai/v1/responses`;
  - normalizes non-stream and SSE responses.
- **Hybrid routing**
  - server-side tools are passed to NewAPI's native `/v1/responses`;
  - ordinary requests can be converted to `/v1/chat/completions`.
- **Strict xAI/Grok schema normalization**
  - fills required `ModelResponse` fields;
  - repairs malformed streamed output items;
  - adds required `annotations`;
  - adds citation `start_index` / `end_index` when missing;
  - preserves monotonic SSE `sequence_number`.
- **Tool-call mapping**
  - Responses function tools to Chat tools;
  - Chat tool calls to Responses `function_call`;
  - xAI local `shell` tool mapping.
- **Zero Python dependencies**
  - implemented with the Python standard library only;
  - no LiteLLM Proxy or other sidecar is required.

---

## Supported request routing

Requests are classified by the `tools` array.

| Request contains | Upstream path | Why |
|---|---|---|
| `web_search`, `web_search_preview`, `x_search`, `file_search`, `code_interpreter`, `mcp`, `computer_use_preview`, `image_generation`, `tool_search` | `POST /v1/responses` | These are server-side tools and must remain in native Responses traffic. |
| Only client-side function tools, or no tools | `POST /v1/chat/completions` | Broad provider compatibility. |

The bridge does not make unsupported hosted tools work by itself. Your NewAPI channel/upstream must support the tool you request.

---

## Requirements

- Python 3.9+
- A reachable NewAPI instance
- A NewAPI token accepted by NewAPI
- A model/channel that supports the requested input type:
  - ordinary text and tools: OpenAI-compatible Chat Completions or Responses-compatible channel;
  - `web_search` and other server tools: a NewAPI model/channel that supports native Responses server tools.

---

## Installation

Copy these files into a dedicated directory:

```bash
mkdir -p /opt/grok-newapi-compat
cp grok_compat.py serialization_selftest.py /opt/grok-newapi-compat/
```

Start it directly:

```bash
NEWAPI_HOST=127.0.0.1 \
NEWAPI_PORT=3000 \
GROK_COMPAT_PORT=3001 \
python3 /opt/grok-newapi-compat/grok_compat.py
```

The service listens on `127.0.0.1:3001` by default and forwards to `127.0.0.1:3000`.

---

## Configuration

All options are environment variables. There is no bridge API key.

| Variable | Default | Description |
|---|---|---|
| `GROK_COMPAT_HOST` | `127.0.0.1` | Listen address. Keep private behind a reverse proxy. |
| `GROK_COMPAT_PORT` | `3001` | Listen port. |
| `NEWAPI_HOST` | `127.0.0.1` | NewAPI host. |
| `NEWAPI_PORT` | `3000` | NewAPI port. |
| `GROK_COMPAT_UPSTREAM_TIMEOUT` | `3600` | Upstream timeout in seconds. |
| `GROK_COMPAT_HISTORY_SIZE` | `512` | Bounded local history for Chat-path `previous_response_id`. |
| `GROK_COMPAT_LOG_LEVEL` | `INFO` | Python log level. |
| `GROK_COMPAT_DEBUG_CAPTURE` | `0` | Set to `1` to opt in to request/SSE body capture. |
| `GROK_COMPAT_DEBUG_DIR` | `/tmp/grok-compat` | Directory for opt-in debug captures. |

Debug capture may contain request and response content. Never enable it on a shared or production host unless you understand the contents being written.

---

## systemd example

```ini
[Unit]
Description=Grok Responses compatibility bridge for NewAPI
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/grok-newapi-compat
ExecStart=/usr/bin/python3 /opt/grok-newapi-compat/grok_compat.py
Environment=GROK_COMPAT_HOST=127.0.0.1
Environment=GROK_COMPAT_PORT=3001
Environment=NEWAPI_HOST=127.0.0.1
Environment=NEWAPI_PORT=3000
Restart=always
RestartSec=2
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now grok-newapi-compat
```

---

## Reverse proxy example

Only the bridge needs to be reachable by Grok CLI. Keep `3001` private.

If NewAPI and the Grok compatibility path share one public port, Caddy can expose `/grok` as a dedicated prefix:

```caddy
:8443 {
    handle_path /grok/* {
        reverse_proxy 127.0.0.1:3001
    }

    handle {
        reverse_proxy 127.0.0.1:3000
    }
}
```

`handle_path` strips `/grok`, so Grok CLI can be configured with:

```text
Base URL: https://your-domain.example:8443/grok/v1
API key:  <NewAPI token>
```

---

## Testing

Run the built-in strict-schema self test:

```bash
python3 serialization_selftest.py
```

For a stronger check against the official xAI schema, download it first and point the test at it:

```bash
curl -o /tmp/xai-openapi.json https://docs.x.ai/openapi.json
XAI_OPENAPI=/tmp/xai-openapi.json python3 serialization_selftest.py
```

Quick smoke test:

```bash
curl -sS http://127.0.0.1:3001/v1/responses \
  -H "Authorization: Bearer <NewAPI-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "input": "Return OK only.",
    "stream": false,
    "store": false
  }'
```

Web-search smoke test, if your NewAPI model supports it:

```bash
curl -sS http://127.0.0.1:3001/v1/responses \
  -H "Authorization: Bearer <NewAPI-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "input": "Search the web and return one source.",
    "tools": [{"type": "web_search"}],
    "stream": false,
    "store": false
  }'
```

---

## What is normalized

Examples of compatibility issues handled by this bridge:

```json
{
  "type": "function_call",
  "call_id": "call_123",
  "name": "run_terminal_command",
  "arguments": "",
  "status": "in_progress"
}
```

```json
{
  "type": "output_text",
  "text": "",
  "annotations": []
}
```

```json
{
  "type": "url_citation",
  "url": "https://example.com",
  "start_index": 0,
  "end_index": 0
}
```

The bridge also adds the required top-level xAI `ModelResponse` fields, including strict fields such as `service_tier`, `truncation`, `top_logprobs`, `presence_penalty`, and `frequency_penalty`.

---

## Important distinctions

### `web_search` is not MCP

Grok's `search_tool` searches configured MCP servers. It is not the server-side `web_search` tool.

- No MCP servers configured means `search_tool` will correctly report that no MCP server is connected.
- `web_search` is requested as a native server-side tool and is not provided by MCP.

### `previous_response_id` support is path-dependent

On the Chat Completions route, the bridge keeps a small local replay history. On the native Responses route, `previous_response_id` is left to NewAPI/upstream semantics. This project does not provide durable server-side storage.

---

## Security notes

- Do not expose port `3001` directly.
- Put the bridge behind TLS.
- Keep NewAPI responsible for keys, quotas, model access, and audit logs.
- Do not enable `GROK_COMPAT_DEBUG_CAPTURE` on hosts with untrusted local users.
- The bridge intentionally forwards the caller's `Authorization` header unchanged.

---

## License

MIT. See [LICENSE](./LICENSE).
