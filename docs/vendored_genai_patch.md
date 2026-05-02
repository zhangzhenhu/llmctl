# Vendored genai Patch

Date: 2026-05-02

llmctl currently builds with:

```toml
[patch.crates-io]
genai = { path = "vendor/genai" }
```

This is intentional. llmctl no longer publishes to crates.io while this patch is required. GitHub Release, Homebrew, and `cargo install --git` all build from the repository and therefore include `vendor/genai`.

## Why

Aliyun/DashScope is OpenAI-compatible, but some thinking models require provider-specific request fields. For example, `glm-5` keeps streaming `delta.reasoning_content` unless the request body contains:

```json
{
  "enable_thinking": false
}
```

Published `genai 0.6.0-beta.18` does not expose a request-body passthrough for chat options, so llmctl cannot express this through the normal genai runtime without a patch.

Aliyun streaming chunks can also include:

```json
{
  "usage": null
}
```

Published genai tries to deserialize that null value as a usage object and logs an error for stream chunks even though the response itself is valid.

## Patch Inventory

Keep this list updated whenever `vendor/genai` changes.

1. `vendor/genai/src/chat/chat_options.rs`
   - Adds `ChatOptions.extra_body: Option<Value>`.
   - Adds `ChatOptions::with_extra_body(Value)`.
   - Adds `ChatOptionsSet::extra_body()`.

2. `vendor/genai/src/adapter/adapters/openai/adapter_shared.rs`
   - Merges `extra_body` into OpenAI Chat Completions request payloads.
   - Treats `usage:null` as empty usage instead of logging a deserialization error.

3. `vendor/genai/src/adapter/adapters/openai_resp/adapter_impl.rs`
   - Merges `extra_body` into OpenAI Responses request payloads.

## llmctl Behavior Depending On This Patch

`--reasoning off` for `adapter: aliyun` injects:

```json
{
  "enable_thinking": false
}
```

This is resolved in `src/config/resolver.rs` and forwarded in `src/runtime/genai_runtime.rs` through `ChatOptions::with_extra_body`.

## Upgrade Checklist

When upgrading genai:

1. Check whether upstream genai has native equivalents for every item in the patch inventory.
2. If upstream supports all items, remove `[patch.crates-io]` and delete `vendor/genai`.
3. If upstream only supports part of the inventory, re-apply the missing changes to the new `vendor/genai`.
4. Run:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test -q
./scripts/run-cli-dry-run-tests.sh
cargo build --release
```

5. Run a real Aliyun smoke test with a local key, without printing the key:

```bash
llmctl -p aliyun -m glm-5 --message "从1数到8，每个数字单独一行" --reasoning off --stream
```

Expected result: normal answer text streams, no thinking content is printed, and no `usage:null` deserialization error is logged.
