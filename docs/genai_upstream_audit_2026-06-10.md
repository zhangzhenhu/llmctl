# genai Upstream Audit (2026-06-10)

## Scope

This note audits the local vendored `genai` copy against:

- upstream `jeremychone/rust-genai` tag `v0.6.0-beta.18` (same version as our vendored base)
- upstream `jeremychone/rust-genai` tag `v0.7.0-beta.3` (latest checkpoint for this audit)

Local vendored version references:

- [vendor/genai/Cargo.toml](/Users/test/Documents/projects/llm_probe/vendor/genai/Cargo.toml:15)
- [Cargo.lock](/Users/test/Documents/projects/llm_probe/Cargo.lock:665)

Upstream tags verified via:

- `git ls-remote --tags https://github.com/jeremychone/rust-genai.git`

Relevant upstream tag SHAs:

- `v0.6.0-beta.18` => `cb343d74c15fed24b926e63b9132a9eab100204f`
- `v0.7.0-beta.3` => `fa82095877eb548b22c27ecef38b7bcf7c512299`

## Local Modifications Versus Upstream v0.6.0-beta.18

### Non-functional packaging/vendoring differences

- `vendor/genai/Cargo.toml`
  - vendored registry-normalized manifest differs from the upstream repo manifest
  - includes `reqwest` `socks` feature in vendored copy
- vendoring artifacts only:
  - `.cargo-ok`
  - `.cargo_vcs_info.json`
  - `Cargo.lock`
  - `Cargo.toml.orig`
  - `target/`

These are not treated as behavioral runtime patches for the rest of this audit.

### Functional code patches present in the local vendored copy

1. `src/chat/chat_options.rs`
   - adds `ChatOptions.extra_body: Option<Value>`
   - adds `ChatOptions::with_extra_body(Value)`
   - adds `ChatOptionsSet::extra_body()`

2. `src/adapter/adapters/openai/adapter_shared.rs`
   - merges `extra_body` into OpenAI Chat Completions payloads
   - treats `usage: null` as empty usage instead of deserialization failure/noisy logging

3. `src/adapter/adapters/openai_resp/adapter_impl.rs`
   - merges `extra_body` into OpenAI Responses payloads

4. `src/error.rs`
   - adds `format_error_chain(...)` helper

5. Streamer error-chain formatting patches
   - `src/adapter/adapters/anthropic/streamer.rs`
   - `src/adapter/adapters/cohere/streamer.rs`
   - `src/adapter/adapters/gemini/streamer.rs`
   - `src/adapter/adapters/ollama/streamer.rs`
   - `src/adapter/adapters/openai/streamer.rs`
   - `src/adapter/adapters/openai_resp/streamer.rs`
   - these patches replace plain `err.to_string()` with formatted chained causes in logged and surfaced `WebStream` errors

6. `src/adapter/adapters/openai_resp/streamer.rs`
   - adds `extract_provider_error_message(...)`
   - when a streamed Responses event cannot deserialize, attempts to surface provider JSON error message as a `StreamParse` failure instead of only warning-and-skip behavior

## Upstream v0.7.0-beta.3 Status By Patch

| Local patch area | Present upstream in `v0.7.0-beta.3`? | Evidence | Status |
| --- | --- | --- | --- |
| `ChatOptions.extra_body` support | Yes | `src/chat/chat_options.rs` contains `extra_body`, `with_extra_body`, and `ChatOptionsSet::extra_body()` | Can drop local patch for this part |
| OpenAI Chat Completions `extra_body` merge | Yes | `src/adapter/adapters/openai/adapter_shared.rs` merges `options_set.extra_body()` into payload | Can drop local patch for this part |
| OpenAI Chat Completions `usage:null` tolerance | Yes | `OpenAIAdapter::into_usage(...)` returns default on `usage_value.is_null()` | Can drop local patch for this part |
| OpenAI Responses `extra_body` merge | Yes | `src/adapter/adapters/openai_resp/adapter_impl.rs` merges `chat_options.extra_body()` into payload | Can drop local patch for this part |
| Error-chain formatter helper | No | upstream `src/error.rs` has no `format_error_chain(...)` helper | Still local-only |
| Streamer chained-cause logging/surfacing | No | upstream streamers still mostly use `err` / `err.to_string()` instead of local chained formatter | Still local-only |
| OpenAI Responses streamer provider-error extraction | No | upstream `src/adapter/adapters/openai_resp/streamer.rs` does not contain local `extract_provider_error_message(...)` logic | Still local-only |

## Notes On Upstream Improvements Beyond Our Local Patch

Upstream `v0.7.0-beta.3` also includes several changes we do **not** have locally, including:

- OpenAI / OpenAI Responses `tool_choice` support
- OpenAI streamer tail-usage capture refinements
- OpenAI Responses streamer reasoning/tool-call robustness improvements (`output_item.done`, fallback tool-call extraction)
- Gemini streamer refactor from array-style handling to SSE event handling
- new `AdapterKindMismatch` error variant

These do not remove all local differences, but they do mean upstream has moved forward substantially beyond `v0.6.0-beta.18`.

## Practical Conclusion

### Local patches that appear no longer required because upstream has them

- `chat_options.rs` `extra_body`
- `openai/adapter_shared.rs` `extra_body` merge
- `openai/adapter_shared.rs` `usage:null` handling
- `openai_resp/adapter_impl.rs` `extra_body` merge

### Local patches that still do not appear upstream in `v0.7.0-beta.3`

- `error.rs` chained-cause formatter
- chained-cause streamer error reporting in:
  - anthropic
  - cohere
  - gemini
  - ollama
  - openai
  - openai_resp
- `openai_resp/streamer.rs` provider-error extraction on parse failure

## Recommendation

Before removing `vendor/genai`, do a focused rebase onto upstream `v0.7.0-beta.3` and decide whether the remaining local-only patches are still worth carrying:

1. Keep only product-critical behavior patches.
2. Re-evaluate whether error-chain formatting is nice-to-have rather than release-blocking.
3. Specifically regression-test the Responses streamer error path, because upstream has added robustness there, but not the same provider-error extraction behavior as our local patch.
