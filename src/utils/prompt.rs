#![allow(dead_code)]
use std::fs;
use std::io::{self, Write};
use std::path::Path;

pub fn prompt_overwrite(path: &Path) -> bool {
    print!("File {} already exists, overwrite? [y/N]: ", path.display());
    io::stdout().flush().ok();

    let mut answer = String::new();
    if io::stdin().read_line(&mut answer).is_ok() {
        answer.trim().eq_ignore_ascii_case("y")
    } else {
        false
    }
}

pub fn prompt_confirm(message: &str) -> bool {
    print!("{} [y/N]: ", message);
    io::stdout().flush().ok();

    let mut answer = String::new();
    if io::stdin().read_line(&mut answer).is_ok() {
        answer.trim().eq_ignore_ascii_case("y")
    } else {
        false
    }
}

pub fn init_config_file(path: &Path, format: &str) -> Result<(), String> {
    if path.exists() {
        if !prompt_overwrite(path) {
            return Err("Operation cancelled".to_string());
        }
    }

    let content = match format {
        "json" => DEFAULT_CONFIG_JSON,
        _ => DEFAULT_CONFIG_YAML,
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    fs::write(path, content).map_err(|e| e.to_string())?;

    Ok(())
}

const DEFAULT_CONFIG_YAML: &str = r#"# llmctl Config File
#
# Supported Providers (provider value):
#   - openai         : https://api.openai.com/v1
#   - google/gemini  : https://generativelanguage.googleapis.com/v1beta
#   - anthropic/claude: https://api.anthropic.com
#   - ollama         : http://localhost:11434 (local)
#   - deepseek       : https://api.deepseek.com/v1
#   - xai            : https://api.x.ai/v1
#   - groq           : https://api.groq.com/openai/v1
#   - mistral        : https://api.mistral.ai/v1
#   - openai-compatible (for Aliyun/DashScope/custom endpoints):
#       provider: "openai-compatible"
#       base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
#
provider: "openai"

# API Base URL (optional, defaults to provider's default URL)
# Examples:
#   - https://api.openai.com/v1
#   - https://generativelanguage.googleapis.com/v1beta
#   - https://api.anthropic.com
#   - http://localhost:11434
#   - https://dashscope.aliyuncs.com/compatible-mode/v1 (Aliyun)
base_url: ""

# API Key (recommended: set via environment variable LLM_API_KEY)
api_key: ""

# Model name (use -l to list available models for your provider)
model: "gpt-4o"

# Enable streaming response
stream: false

# Request timeout in seconds (default: 60)
# timeout_seconds: 60

# Maximum tokens to generate
# max_tokens: 2048

# Sampling temperature (0.0 - 2.0, higher = more random)
# temperature: 0.7

# Top-p sampling (0.0 - 1.0)
# top_p: 1.0

# Top-k sampling
# top_k: 40

# System prompt
# system: "You are a helpful assistant."

# Reasoning/Thinking Configuration:
# For OpenAI (o1 series) - put directly in config:
#   reasoning_effort: "high"  # low, medium, high
#
# For Anthropic - put directly in config:
# enable_thinking: true
# thinking_budget_tokens: 1024
#
# For openai-compatible (Aliyun/DashScope) - put in extra_body:
# extra_body:
#   enable_thinking: true

# Conversation context (messages history)
context:
  - role: "system"
    content: "You are a helpful assistant."
  - role: "user"
    content: "Hello!"
"#;

const DEFAULT_CONFIG_JSON: &str = r#"{
  "provider": "openai",
  "base_url": "",
  "api_key": "",
  "model": "gpt-4o",
  "stream": false,
  "context": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello!"
    }
  ]
}
"#;
