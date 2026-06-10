# Vendored genai Patch

Date: 2026-06-10

llmctl currently builds with:

```toml
[dependencies]
genai = "0.7.0-beta.3"

[patch.crates-io]
genai = { path = "vendor/genai" }
```

This is intentional. llmctl no longer publishes to crates.io while this patch is required. GitHub Release, Homebrew, and `cargo install --git` all build from the repository and therefore include `vendor/genai`.

## Base Upstream Version

The vendored copy now tracks upstream:

- repository: `jeremychone/rust-genai`
- tag: `v0.7.0-beta.3`
- commit: `fa82095877eb548b22c27ecef38b7bcf7c512299`

The remaining local changes are kept as two focused patch files:

- [genai-v0.7.0-beta.3-error-diagnostics.patch](/Users/test/Documents/projects/llm_probe/patches/genai-v0.7.0-beta.3-error-diagnostics.patch)
- [genai-v0.7.0-beta.3-stream-provider-model.patch](/Users/test/Documents/projects/llm_probe/patches/genai-v0.7.0-beta.3-stream-provider-model.patch)

Both patch files are normalized to upstream-style `src/...` paths so they can be reviewed or proposed upstream more easily.

The comparison audit used for this upgrade is recorded at:

- [genai_upstream_audit_2026-06-10.md](/Users/test/Documents/projects/llm_probe/docs/genai_upstream_audit_2026-06-10.md)

## Why A Patch Still Exists

Upstream `v0.7.0-beta.3` already includes the earlier llmctl-required runtime features:

- `ChatOptions::with_extra_body(...)`
- OpenAI Chat Completions `extra_body` payload merge
- OpenAI Responses `extra_body` payload merge
- tolerant handling for OpenAI-compatible `usage: null`

So those earlier functional patches are no longer carried locally.

The remaining vendored differences are now split into two focused areas:

1. Preserve full chained causes in streamed adapter/web errors.
2. Surface provider JSON error messages during OpenAI Responses stream parse failures instead of silently warning-and-skipping the bad event.
3. Capture the provider-reported model name at stream end so downstream callers can display the actual server-selected model instead of only the requested model alias.

## Patch Inventory

Keep this list updated whenever `vendor/genai` changes.

1. `genai-v0.7.0-beta.3-error-diagnostics.patch`
   - Focused patch for chained error reporting and OpenAI Responses stream parse diagnostics.

2. `vendor/genai/src/error.rs`
   - Adds `format_error_chain(...)`.
   - Renders nested error causes into a stable multi-line string.

3. Streamer error-chain formatting
   - `vendor/genai/src/adapter/adapters/anthropic/streamer.rs`
   - `vendor/genai/src/adapter/adapters/cohere/streamer.rs`
   - `vendor/genai/src/adapter/adapters/gemini/streamer.rs`
   - `vendor/genai/src/adapter/adapters/ollama/streamer.rs`
   - `vendor/genai/src/adapter/adapters/openai/streamer.rs`
   - `vendor/genai/src/adapter/adapters/openai_resp/streamer.rs`
   - Replaces plain `err.to_string()` with formatted chained causes in `Error::WebStream`.

4. `vendor/genai/src/adapter/adapters/openai_resp/streamer.rs`
   - Adds `extract_provider_error_message(...)`.
   - When a Responses SSE event cannot be deserialized, tries to extract provider-side JSON error details and surface them as `Error::StreamParse`.

5. `genai-v0.7.0-beta.3-stream-provider-model.patch`
   - Focused upstream-candidate patch for carrying provider-reported model names through streaming end events.

6. Stream end provider-model capture
   - `vendor/genai/src/adapter/inter_stream.rs`
   - `vendor/genai/src/chat/chat_stream.rs`
   - `vendor/genai/src/adapter/adapters/support.rs`
   - `vendor/genai/src/adapter/adapters/anthropic/streamer.rs`
   - `vendor/genai/src/adapter/adapters/gemini/streamer.rs`
   - `vendor/genai/src/adapter/adapters/ollama/streamer.rs`
   - `vendor/genai/src/adapter/adapters/openai/streamer.rs`
   - `vendor/genai/src/adapter/adapters/openai_resp/streamer.rs`
   - plus `captured_provider_model_name: None` plumbing in other streamers that construct `InterStreamEnd`
   - Adds a small optional field that carries the provider-reported model name through the stream end event.
   - Anthropic streams capture `message.model` from `message_start`.
   - Gemini streams capture `modelVersion` from SSE payloads.
   - Ollama ndjson streams capture `model` from each streamed JSON object.
   - OpenAI Chat Completions streams capture `model` from SSE payloads.
   - OpenAI Responses streams capture `response.model` from terminal events.

## llmctl Behavior Depending On This Patch

The patch no longer exists for request-body passthrough or `usage:null`; upstream covers those now.

The current llmctl benefit is operational:

- streamed provider failures include deeper cause chains in logs and surfaced errors
- malformed or non-standard OpenAI Responses stream events can still expose provider error messages when the server returns JSON-shaped failure payloads
- streamed responses can report the provider-returned model name, allowing llmctl to present compact `Profile / Adapter / Model` output while still showing `Requested Model` when the server-selected name differs

## Upgrade Checklist

When upgrading genai again:

1. Re-check whether upstream has native equivalents for:
   - chained-cause stream error rendering
   - Responses provider-error extraction on stream parse failure
2. If upstream supports both areas, remove `[patch.crates-io]`, delete `vendor/genai`, and delete both focused patch files.
3. If upstream supports only part of the inventory, regenerate only the still-needed focused patch file(s) from the new upstream tag.
4. Run:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test -q
./scripts/run-cli-dry-run-tests.sh
cargo build --release
```

5. Run at least one real streamed smoke test on an OpenAI-compatible provider that can emit non-trivial stream failures.

## Regenerating The Patch Files

Assuming clean upstream source is available in `/tmp/rust-genai-upstream`:

```bash
diff -ru /tmp/rust-genai-upstream vendor/genai > /tmp/genai-full.patch
```

From that full diff, split the result into focused artifacts:

- `patches/genai-v<version>-error-diagnostics.patch`
- `patches/genai-v<version>-stream-provider-model.patch`

Each patch file should stay focused. If either starts growing beyond the inventory above, run a fresh upstream audit before carrying more local changes forward.
