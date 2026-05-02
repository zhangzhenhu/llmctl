# llmctl

A CLI tool for testing and validating LLM (Large Language Model) services. Supports multiple providers including OpenAI, Gemini, Claude, Ollama, DeepSeek, and any OpenAI-compatible APIs.

## Features

- **Multiple Provider Support**: OpenAI, Gemini (Google), Anthropic (Claude), Ollama, DeepSeek, XAI, Groq, Mistral and more
- **OpenAI-Compatible API**: Works with any service that implements the OpenAI API format (Aliyun, DashScope, local deployments, etc.)
- **Model Listing**: List all available models from your provider
- **Streaming Responses**: Real-time streaming output for chat responses
- **Thinking/Reasoning Support**: Uses genai-native capabilities with fallback compatibility for `extra_body.enable_thinking` and `reasoning_content`
- **Flexible Configuration**: Configure via YAML/JSON files or command-line arguments

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
active_provider: openai_main
providers:
  openai_main:
    adapter: openai
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
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
  -c, --config <PATH>          Config file path (YAML or JSON)
  -m, --model <STRING>         Model name
  -l, --list                   List available models
      --list-presets           List built-in provider presets and exit
      --message <STRING>       Append user message (repeatable)
  -p, --provider <STRING>      Provider adapter or alias
  -P, --profile <NAME>         Provider profile name from config (v2)
  -u, --url <STRING>           API base URL
  -s, --secret <STRING>        API key
  -k, --key <STRING>           API key alias for --secret
      --stream                 Enable streaming response
      --no-stream              Disable streaming response for this run
  -v, --version                Show version information
  -i, --init <FORMAT>          Initialize config file: yaml/json
      --init-path <PATH>       Custom config file path
  -t, --convert <INPUT>        Convert config format
      --endpoint <MODE>        OpenAI API mode: auto|responses|chat-completions
      --reasoning <MODE>       Unified reasoning: off|auto|low|medium|high|xhigh|max|budget:<n>
      --dry-run                Print resolved execution plan without request
      --doctor-config          Validate config and print diagnostics
      --allow-sdk-default-api  Allow OpenAI endpoint fallback to SDK default
```

### Examples

#### Quick Start Without Config File

```bash
# Show all built-in presets
llmctl --list-presets

# OpenAI quick start (reads OPENAI_API_KEY)
llmctl --provider openai --message "hello"

# Aliyun quick start (reads ALIYUN_API_KEY)
llmctl --provider aliyun --message "你好"

# Alias mode also works:
llmctl --provider dashscope --message "你好"
```

#### Select Config Profile (v2)

```bash
# Use providers.anthropic_main from config
llmctl -c llm.yaml -P anthropic_main --message "hello"

# Keep profile, but temporarily override adapter/preset
llmctl -c llm.yaml -P openai_main --provider dashscope --message "你好"
```

#### List Available Models

```bash
llmctl -c llm.yaml -l
```

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
llmctl --provider openai --model gpt-5 --reasoning high --message "hello"

# Budget style reasoning
llmctl --provider gemini --reasoning budget:8000 --message "hello"
```

#### Use with Environment Variable for API Key

```bash
export LLM_API_KEY="your-api-key"
llmctl -c llm.yaml
```

### Supported Providers

| Provider | Value | Notes |
|----------|-------|-------|
| OpenAI | `openai` | |
| Google Gemini | `gemini` or `google` | |
| Anthropic Claude | `anthropic` or `claude` | |
| Ollama | `ollama` | Local deployment |
| DeepSeek | `deepseek` | |
| XAI | `xai` | |
| Groq | `groq` | |
| Mistral | `mistral` | |
| OpenAI-Compatible | `openai-compatible`, `aliyun`, `dashscope` | Custom endpoints |

## Configuration Reference

### YAML Format

```yaml
version: 2
active_provider: openai_main
defaults:
  stream: true
  timeout_seconds: 60
  openai_api: auto
  reasoning: auto
providers:
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
providers:
  openai_main:
    reasoning: high
  gemini_main:
    reasoning: budget:8000
```

Allowed values:

- `off` (disable reasoning capture; sends best-effort disable signal)
- `auto` (provider default behavior with normalized parsing)
- `low|medium|high|xhigh|max`
- `budget:<n>` (numeric budget for providers that support budget mapping)

Backward compatibility:

- `reasoning_effort` is still accepted as a legacy alias in provider profiles.

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
