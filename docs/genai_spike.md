# genai Spike 记录

Decision: go (with local patch)

## Required Capabilities

- non_stream_chat: supported
- stream_chat: supported
- custom_base_url: supported
- api_key_resolver: supported
- reasoning_content_capture: supported
- reasoning_stream_event: supported
- extra_body_passthrough: supported (genai native API via local patch)

## Optional Capabilities

- openai_api_selection: supported
- live_model_list: supported
- system_instruction: supported

## Matrix

| 能力 | OpenAI | Anthropic | Gemini | Ollama | OpenAI-compatible |
|---|---|---|---|---|---|
| 非流式 chat | supported | supported | supported | supported | supported |
| 流式 chat | supported | supported | supported | supported | supported |
| reasoning chunk | supported | supported | supported | supported | supported |
| usage | supported | supported | supported | supported | supported |
| custom base_url | supported | supported | supported | supported | supported |
| extra_body | supported* | n/a | n/a | n/a | supported* |
| OpenAI API 选择 | supported(命名空间) | n/a | n/a | n/a | n/a |
| live model list | static 为主 | static 为主 | static 为主 | dynamic | static 为主 |

## API Surface Notes

- Client 构造 API: `Client::builder().build()`
- ChatRequest 构造 API: `ChatRequest::default().with_system(...).append_message(...)`
- ChatOptions 字段/API: `with_temperature / with_max_tokens / with_top_p / with_capture_usage / with_capture_reasoning_content / with_normalize_reasoning_content`
- stream API: `client.exec_chat_stream(model, req, Some(&options))`
- reasoning event 类型: `ChatStreamEvent::ReasoningChunk`
- usage 字段: `ChatResponse.usage` / `StreamEnd.captured_usage`
- custom base_url API: `ServiceTargetResolver + Endpoint::from_owned(...)`
- auth resolver API: `with_auth_resolver_fn(|_| Ok(Some(AuthData::from_single(...))))`
- extra_body 原生映射 API: `ChatOptions::with_extra_body(...)`（本仓库 `vendor/genai` patch 提供）
- OpenAI endpoint 选择 API: 通过 model namespace 选择 adapter：`openai::` 与 `openai_resp::`
- model list API: `client.all_model_names(adapter_kind)`

## 结论说明

1. chat / stream / list 主路径可迁移到 genai。
2. `extra_body` 已走 genai-native 能力透传（`with_extra_body`），无需 llmctl 自定义 provider body 适配。
3. `openai_api` 可通过 model namespace 强制选择 `openai::` / `openai_resp::`。
4. 主流程保留 fallback 原因输出，便于定位非 genai 能力缺口。

\* `extra_body` 的实际生效范围取决于具体 adapter。OpenAI/OpenAIResp 及其兼容链路已验证可生效。

## Commands

```bash
cargo run --example genai_probe -- --provider openai --model <model> --message "hello"
cargo run --example genai_probe -- --provider anthropic --model <model> --message "hello"
cargo run --example genai_probe -- --provider gemini --model <model> --message "hello"
cargo run --example genai_probe -- --provider ollama --model <model> --message "hello"
cargo run --example genai_probe -- --provider openai --base-url <compatible-url> --model <model> --stream
```
