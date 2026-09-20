<div align="center">

# grok-newapi-compat

[English](./README.md) | **简体中文**

</div>

---

一个轻量的 **Grok CLI ↔ NewAPI Responses 兼容桥**。

它让 Grok CLI 以 xAI/Grok 风格的 `POST /v1/responses` 访问 NewAPI，并在返回时把响应流规范化为 Grok CLI 期待的 strict schema。鉴权、配额、模型访问、审计与限流仍然完全由 NewAPI 负责，兼容层本身不引入第二套认证体系。

> 与 xAI、OpenAI、Grok 或 NewAPI 均无隶属关系。

---

## 为什么需要它

Grok CLI 使用的是严格模式的 xAI/Grok Responses 协议。很多 OpenAI 兼容网关虽然支持 Responses 或 Chat Completions，但协议细节不完全一致：

- 部分必需的 xAI response 字段缺失；
- 流式 `output_item.added` 结构可能不完整；
- citation annotation 可能缺少 Grok 需要的字段；
- 服务端工具如果被强制转换成 Chat Completions 会丢失。

本项目同时保留两边能力：

```text
Grok CLI
  -> grok-newapi-compat
       -> NewAPI /v1/responses
       -> NewAPI /v1/chat/completions
  -> strict xAI/Grok Responses normalization
  -> Grok CLI
```

---

## 特性

- **单一认证体系**
  - 无桥接层 API key；
  - 无第二套 token 存储；
  - `Authorization` 透传给 NewAPI。
- **Responses API 兼容**
  - 接受 `POST /v1/responses`、`/responses`、`/openai/v1/responses`；
  - 支持非流式与 SSE 规范化。
- **混合路由**
  - 服务端工具走 NewAPI 原生 `/v1/responses`；
  - 普通请求可转换到 `/v1/chat/completions`。
- **严格 schema 修复**
  - 补齐 `ModelResponse` 必需字段；
  - 修复流式输出项缺失字段；
  - 补齐 `annotations`；
  - 补齐 citation 的 `start_index` / `end_index`；
  - 保持 SSE `sequence_number` 单调递增。
- **工具调用映射**
  - Responses function tools → Chat tools；
  - Chat tool calls → Responses `function_call`；
  - 支持本地 `shell` 工具映射。
- **零 Python 依赖**
  - 仅使用 Python 标准库；
  - 不需要 LiteLLM Proxy 或其他 sidecar。

---

## 路由策略

请求会根据 `tools` 数组分类。

| 请求包含 | 上游路径 | 原因 |
|---|---|---|
| `web_search`, `web_search_preview`, `x_search`, `file_search`, `code_interpreter`, `mcp`, `computer_use_preview`, `image_generation`, `tool_search` | `POST /v1/responses` | 服务端工具必须保留在原生 Responses 流量中。 |
| 只有客户端函数工具，或没有工具 | `POST /v1/chat/completions` | 兼容更多上游 provider。 |

兼容层本身不会凭空实现服务端工具。实际能力取决于 NewAPI channel/upstream 是否支持对应工具。

---

## 环境要求

- Python 3.9+
- 可访问的 NewAPI 实例
- NewAPI token
- 对应模型/通道支持你要使用的输入类型：
  - 普通文本与工具：OpenAI-compatible Chat Completions 或 Responses-compatible channel；
  - `web_search` 等服务端工具：NewAPI model/channel 需支持原生 Responses server tools。

---

## 安装

```bash
mkdir -p /opt/grok-newapi-compat
cp grok_compat.py serialization_selftest.py /opt/grok-newapi-compat/
```

直接启动：

```bash
NEWAPI_HOST=127.0.0.1 \
NEWAPI_PORT=3000 \
GROK_COMPAT_PORT=3001 \
python3 /opt/grok-newapi-compat/grok_compat.py
```

默认监听 `127.0.0.1:3001`，并转发到 `127.0.0.1:3000`。

---

## 配置

所有配置都通过环境变量提供。兼容层没有 API key 配置。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GROK_COMPAT_HOST` | `127.0.0.1` | 监听地址，应保持在内网或反代后面。 |
| `GROK_COMPAT_PORT` | `3001` | 监听端口。 |
| `NEWAPI_HOST` | `127.0.0.1` | NewAPI 地址。 |
| `NEWAPI_PORT` | `3000` | NewAPI 端口。 |
| `GROK_COMPAT_UPSTREAM_TIMEOUT` | `3600` | 上游超时时间，单位秒。 |
| `GROK_COMPAT_HISTORY_SIZE` | `512` | Chat 路径下用于 `previous_response_id` 的本地历史长度。 |
| `GROK_COMPAT_LOG_LEVEL` | `INFO` | 日志级别。 |
| `GROK_COMPAT_DEBUG_CAPTURE` | `0` | 设为 `1` 才会开启请求/SSE 抓包。 |
| `GROK_COMPAT_DEBUG_DIR` | `/tmp/grok-compat` | 抓包输出目录。 |

调试抓包会包含请求与响应内容。除非你清楚其含义，否则不要在生产或共享主机上开启。

---

## systemd 示例

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

---

## 反向代理示例

只暴露反代端口，不要直接暴露 `3001`。

如果 NewAPI 与 Grok 兼容路径共用同一个公网端口，可以用 Caddy 把 `/grok` 前缀转发到兼容层：

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

Grok CLI 可配置为：

```text
Base URL: https://your-domain.example:8443/grok/v1
API key:  <NewAPI token>
```

---

## 测试

运行内置 strict-schema 自查：

```bash
python3 serialization_selftest.py
```

如果想对官方 xAI schema 做更强校验：

```bash
curl -o /tmp/xai-openapi.json https://docs.x.ai/openapi.json
XAI_OPENAPI=/tmp/xai-openapi.json python3 serialization_selftest.py
```

快速冒烟测试：

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

---

## 安全说明

- 不要直接暴露 `3001`。
- 反代层启用 TLS。
- key、配额、模型访问、审计日志仍由 NewAPI 管理。
- 除非明确需要，否则不要开启调试抓包。
- `Authorization` 会透传给 NewAPI。

---

## License

MIT. See [LICENSE](./LICENSE).
