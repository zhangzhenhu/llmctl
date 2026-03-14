# llmctl

A CLI tool for testing and validating LLM (Large Language Model) services. Supports multiple providers including OpenAI, Gemini, Claude, Ollama, DeepSeek, and any OpenAI-compatible APIs.

## Features

- **Multiple Provider Support**: OpenAI, Gemini (Google), Anthropic (Claude), Ollama, DeepSeek, XAI, Groq, Mistral and more
- **OpenAI-Compatible API**: Works with any service that implements the OpenAI API format (Aliyun, DashScope, local deployments, etc.)
- **Model Listing**: List all available models from your provider
- **Streaming Responses**: Real-time streaming output for chat responses
- **Thinking/Reasoning Support**: Enable reasoning mode for compatible providers
- **Flexible Configuration**: Configure via YAML/JSON files or command-line arguments

## Installation

### From Source

```bash
git clone https://github.com/zhangzhenhu/llmctl.git
cd llmctl
cargo build --release
./target/release/llmctl --help
```

### Pre-built Binaries

Download pre-built binaries for macOS, Linux, and Windows from [GitHub Releases](https://github.com/zhangzhenhu/llmct/releases), extract them, and add the aichat binary to your $PATH.

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
provider: "openai-compatible"
base_url: "https://api.openai.com/v1"
api_key: "your-api-key-here"
model: "gpt-4o"
stream: true
context:
  - role: "system"
    content: "You are a helpful assistant."
  - role: "user"
    content: "Hello, world!"
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
  -c, --config <PATH>          Config file path
  -m, --model <STRING>        Model name
  -l, --list                  List available models
  -p, --provider <STRING>     Provider name
  -u, --url <STRING>          API base URL
  -s, --secret <STRING>       API key
      --stream                Enable streaming response
      --init <FORMAT>         Initialize config file (yaml/json)
      --init-path <PATH>      Custom config file path
  -t, --convert <INPUT>       Convert config format
```

### Examples

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
llmctl -c llm.yaml --stream
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
provider: "openai-compatible"    # Provider name
base_url: "https://api.openai.com/v1"  # API base URL
api_key: "sk-..."                 # API key
model: "gpt-4o"                   # Model name
stream: false                     # Enable streaming
max_tokens: 2048                  # Max tokens to generate
temperature: 0.7                  # Sampling temperature (0-2)
top_p: 1.0                        # Top-p sampling
top_k: 40                         # Top-k sampling
timeout_seconds: 60               # Request timeout
system: "You are a helpful assistant."  # System prompt
context:                          # Conversation history
  - role: "user"
    content: "Your message here"
```

### Reasoning Configuration

For providers that support thinking/reasoning:

**OpenAI** (uses `reasoning_effort`):
```yaml
enable_thinking: true
reasoning_effort: "high"          # low, medium, high
```

**Anthropic** (uses `reasoning_budget_tokens`):
```yaml
enable_thinking: true
reasoning_budget_tokens: 1024      # Max tokens for thinking
```

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
