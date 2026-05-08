# llmctl

A CLI tool for testing and validating LLM (Large Language Model) services. Supports multiple providers including OpenAI, Gemini, Claude, Ollama, DeepSeek, and any OpenAI-compatible APIs.

## What's New in v2.1.0

- Unified CLI/config naming around `adapter`, `profile`, `base_url`, and `api_mode`
- Added `--list-adapters` plus richer built-in adapter defaults and aliases
- Added explicit proxy control with `--no-proxy`, `defaults.no_proxy`, and `profiles.<name>.no_proxy`
- Simplified the runtime to the current genai-based path and updated the docs accordingly

## Features

- **Multiple Provider Support**: OpenAI, Gemini (Google), Anthropic (Claude), Ollama, DeepSeek, XAI, Groq, Cohere and more
- **OpenAI-Compatible API**: Works with any service that implements the OpenAI API format (Aliyun, DashScope, local deployments, etc.)
- **Model Listing**: List all available models from your provider
- **Streaming Responses**: Real-time streaming output for chat responses
- **Thinking/Reasoning Support**: Uses genai-native reasoning capture with vendored `extra_body` passthrough for provider-specific controls such as Aliyun `enable_thinking`
- **Flexible Configuration**: Configure via YAML/JSON files or command-line arguments
- **Proxy Control**: Default behavior is `inherit reqwest/system proxy settings`; you can disable proxies per config profile or per run with `--no-proxy`

## Installation

### From Homebrew (Recommended)

```bash
# 1) Add tap first
brew tap zhangzhenhu/llmctl

# 2) Install prebuilt binary
brew install llmctl
```

Supported platforms: macOS (Apple Silicon & Intel), Linux (arm64 & x86_64)

Note:

- In some environments, running `brew install zhangzhenhu/llmctl/llmctl` directly may trigger an extra GitHub auth prompt during implicit tap resolution.
- Explicit `brew tap` first usually avoids that prompt.

### From Cargo Git Install

```bash
# Install from source because llmctl uses a vendored genai patch
cargo install --git https://github.com/zhangzhenhu/llmctl.git
```

llmctl is not currently published to crates.io. It uses a small vendored genai patch to support OpenAI-compatible provider behavior that upstream genai has not released yet:

- request `extra_body` passthrough, used for controls such as Aliyun/DashScope `enable_thinking=false`;
- tolerant streaming usage parsing for chunks that contain `usage:null`.

This keeps the runtime on genai while avoiding a custom OpenAI adapter in llmctl. See `docs/vendored_genai_patch.md` for the patch inventory and upgrade checklist.

If you only want the local source version:

```bash
cargo install --path .
```

If your platform is not supported, build from source:

```bash
brew install --build-from-source zhangzhenhu/llmctl/llmctl
```

### From Source

```bash
git clone https://github.com/zhangzhenhu/llmctl.git
cd llmctl
cargo build --release
./target/release/llmctl --help
```

### Pre-built Binaries

Download pre-built binaries for macOS, Linux, and Windows from [GitHub Releases](https://github.com/zhangzhenhu/llmctl/releases), extract them, and add the llmctl binary to your $PATH.

## Quick Start

### 1. Create a Configuration File

```bash
# Initialize a YAML config file
llmctl --init yaml

# Or JSON format
llmctl --init json
```

### 2. Edit the Configuration File

```yaml
# llm.yaml
version: 2
active_profile: openai_main
defaults:
  no_proxy: false
profiles:
  openai_main:
    adapter: openai
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
    # no_proxy: true
context:
  - role: system
    content: You are a helpful assistant.
```

### 3. Run a Chat

```bash
llmctl -c llm.yaml
```

## Usage

### Command-Line Options

```bash
llmctl [OPTIONS]

Options:
  -c, --config <PATH>          Config file path (v2 YAML or JSON)
  -m, --model <STRING>         Model name
  -l, --list                   List available models
      --list-adapters          List supported adapters, aliases, and built-in defaults
      --message <STRING>       Append user message (repeatable)
  -p, --adapter <STRING>       Adapter name or alias
  -P, --profile <NAME>         Profile name from config (v2)
  -u, --base-url <STRING>      API base URL
  -s, --secret <STRING>        API key
  -k, --key <STRING>           API key alias for --secret
      --stream                 Enable streaming response
      --no-stream              Disable streaming response for this run
      --no-proxy               Disable all proxies for llmctl-managed HTTP clients
  -v, --version                Show version information
  -i, --init <FORMAT>          Initialize config file: yaml/json
      --init-path <PATH>       Custom config file path
  -t, --convert <INPUT>        Convert v2 config between YAML and JSON
      --api-mode <MODE>        API mode: auto|responses|chat-completions
      --reasoning <MODE>       Unified reasoning: off|auto|low|medium|high|xhigh|max|budget:<n>
      --dry-run                Print resolved execution plan without request
      --doctor-config          Validate config and print diagnostics
```

Proxy resolution is unified as:

- CLI `--no-proxy`
- `profiles.<name>.no_proxy`
- `defaults.no_proxy`
- otherwise inherit reqwest/system proxy settings

On macOS this inherited mode can still use the system proxy even when shell `*_proxy` environment variables are unset. Use `--no-proxy` or set `no_proxy: true` in config to force direct connections for llmctl-managed HTTP clients.

The current runtime architecture is documented in [docs/当前架构.md](docs/当前架构.md). Older design and stabilization docs in `docs/` are kept as migration history.

## Automated Tests

Centralized CLI regression cases live in:

- `tests/cases/cli_dry_run_cases.yaml`
- `tests/cli_dry_run_cases.rs`

Run the suite with:

```bash
./scripts/run-cli-dry-run-tests.sh
```

or:

```bash
cargo test --test cli_dry_run_cases -- --nocapture
```

### Examples

#### Quick Start Without Config File

```bash
# Show supported adapters and aliases
llmctl --list-adapters

# OpenAI quick start (reads OPENAI_API_KEY)
llmctl --adapter openai --message "hello"

# Aliyun quick start (reads ALIYUN_API_KEY)
llmctl --adapter aliyun --message "你好"

# Alias mode also works:
llmctl --adapter ds --message "你好"
```

#### Select Config Profile (v2)

```bash
# Use profiles.anthropic_main from config
llmctl -c llm.yaml -P anthropic_main --message "hello"

# Keep the profile, but override the adapter identity for one run
llmctl -c llm.yaml -P openai_main --adapter aliyun --message "你好"
```

#### List Available Models

```bash
llmctl -c llm.yaml -l
```

`--list` now prints the source of the returned catalog, so you can distinguish a live provider `/models` result from a static fallback.

#### Chat with a Specific Model

```bash
llmctl -c llm.yaml -m gpt-4-turbo
```

#### Stream Response

```bash
# Streaming is enabled by default
llmctl -c llm.yaml

# Disable streaming for one run
llmctl -c llm.yaml --no-stream
```

#### Unified Reasoning Control

```bash
# Force higher reasoning effort when supported
llmctl --adapter openai --model gpt-5 --reasoning high --message "hello"

# Budget style reasoning
llmctl --adapter gemini --reasoning budget:8000 --message "hello"
```

#### Use with Environment Variable for API Key

```bash
export LLM_API_KEY="your-api-key"
llmctl -c llm.yaml
```

### Supported Adapters

| Adapter | Common aliases | Notes |
|---------|----------------|-------|
| `openai` | `oi`, `oai` | OpenAI-compatible protocol family |
| `aliyun` | `ali`, `dashscope`, `ds` | DashScope / Aliyun |
| `anthropic` | `claude`, `anth` | |
| `gemini` | `google`, `gmi` | |
| `ollama` | `ol` | Local deployment |
| `deepseek` | `dsk` | |
| `xai` | `grok` | |
| `groq` | `gq` | |
| `cohere` | `co` | |
| `fireworks` | `fw` | |
| `together` | `tg` | |
| `zai` | `zhipu`, `zhi` | Z.ai / Zhipu |

## Configuration Reference

### YAML Format

`v2` is the current config schema version for `llmctl`.

```yaml
version: 2
active_profile: openai_main
defaults:
  stream: true
  timeout_seconds: 60
  api_mode: auto
  reasoning: auto
profiles:
  openai_main:
    adapter: openai
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
    reasoning: high
    max_tokens: 2048
    temperature: 0.7
    top_p: 1.0
context:
  - role: "user"
    content: "Your message here"
```

### Reasoning Configuration

Use unified `reasoning`:

```yaml
defaults:
  reasoning: auto
profiles:
  openai_main:
    reasoning: high
  gemini_main:
    reasoning: budget:8000
```

Allowed values:

- `off` (disable reasoning capture and apply known provider controls, for example Aliyun `enable_thinking=false`)
- `auto` (provider default behavior with normalized parsing)
- `low|medium|high|xhigh|max`
- `budget:<n>` (numeric budget for providers that support budget mapping)

Config notes:

- Only the v2 config schema is supported.
- `reasoning_effort` is still accepted as a deprecated alias inside v2 profiles.

## Error Handling

The tool provides user-friendly error messages for common issues:

- Invalid API key
- Network errors (DNS resolution failure, connection refused, timeout)
- Model not found
- Rate limiting
- Server errors

## Development

### Build

```bash
cargo build
```

### Run Tests

```bash
cargo test
```

### Release Build

```bash
cargo build --release
```

## License

MIT License
