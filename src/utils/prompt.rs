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

const DEFAULT_CONFIG_YAML: &str = r#"# llmctl v2 config
#
# Quick usage:
#   llmctl -c llmctl.yaml --message "hello"
#   llmctl -c llmctl.yaml --profile openai_main --message "hello"
#   llmctl --list-presets
#
# Versioned schema (current: 2)
version: 2

# Default profile used when --profile is not provided
active_provider: openai_main

# Global defaults (can be overridden by provider profile or CLI args)
defaults:
  stream: true                        # stream output by default
  timeout_seconds: 60                # request timeout
  capture_usage: true                # collect prompt/completion token usage
  capture_reasoning_content: true    # collect reasoning content when provider supports it
  normalize_reasoning_content: true  # normalize provider-specific reasoning field
  openai_api: auto                   # auto | responses | chat_completions (CLI also accepts chat-completions)
  # reasoning: auto                  # off | auto | low | medium | high | xhigh | max | budget:2048

# Multiple provider profiles in one file
providers:
  # Profile name: openai_main
  openai_main:
    adapter: openai                  # openai | aliyun | anthropic | gemini | ollama | deepseek | groq | mistral
    model: gpt-4o
    api_key_env: OPENAI_API_KEY      # read API key from env var
    # api_key: ""                    # optional, not recommended
    # base_url: https://api.openai.com/v1
    # temperature: 0.7
    # max_tokens: 2048
    # top_p: 1.0
    # reasoning: high                # unified reasoning control (recommended)
    # reasoning_effort: medium       # legacy alias, still supported
    # openai_api: responses

  # Example: Aliyun / DashScope profile
  # aliyun_qwen:
  #   adapter: aliyun
  #   model: glm-5
  #   base_url: https://dashscope.aliyuncs.com/compatible-mode/v1/
  #   api_key_env: ALIYUN_API_KEY
  #   # Some Aliyun models do not expose reasoning. For reasoning tests,
  #   # prefer models such as: glm-5, deepseek-v3.2
  #   extra_body:
  #     enable_thinking: true        # fallback for some OpenAI-compatible providers

# Shared context messages (appended before CLI --message)
context:
  - role: system
    content: You are a helpful assistant.
  # - role: user
  #   content: Hello
"#;

const DEFAULT_CONFIG_JSON: &str = r#"{
  "version": 2,
  "active_provider": "openai_main",
  "defaults": {
    "stream": true,
    "timeout_seconds": 60,
    "capture_usage": true,
    "capture_reasoning_content": true,
    "normalize_reasoning_content": true,
    "openai_api": "auto",
    "reasoning": "auto"
  },
  "providers": {
    "openai_main": {
      "adapter": "openai",
      "model": "gpt-4o",
      "api_key_env": "OPENAI_API_KEY",
      "reasoning": "high",
      "temperature": 0.7,
      "max_tokens": 2048,
      "top_p": 1.0
    },
    "aliyun_qwen": {
      "adapter": "aliyun",
      "model": "glm-5",
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/",
      "api_key_env": "ALIYUN_API_KEY",
      "extra_body": {
        "enable_thinking": true
      }
    }
  },
  "context": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello"
    }
  ]
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn default_yaml_template_uses_v2_shape() {
        let parsed: Value = serde_yaml::from_str(DEFAULT_CONFIG_YAML).expect("valid yaml");
        assert_eq!(parsed.get("version"), Some(&Value::from(2)));
        assert!(parsed.get("active_provider").is_some());
        assert!(parsed.get("providers").is_some());
    }

    #[test]
    fn default_json_template_uses_v2_shape() {
        let parsed: Value = serde_json::from_str(DEFAULT_CONFIG_JSON).expect("valid json");
        assert_eq!(parsed.get("version"), Some(&Value::from(2)));
        assert!(parsed.get("active_provider").is_some());
        assert!(parsed.get("providers").is_some());
    }

    #[test]
    fn init_config_file_writes_v2_yaml_template() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("llmctl_init_test_{stamp}.yaml"));

        init_config_file(&path, "yaml").expect("init config should succeed");
        let content = std::fs::read_to_string(&path).expect("file should be readable");
        let parsed: Value = serde_yaml::from_str(&content).expect("written yaml should be valid");

        assert_eq!(parsed.get("version"), Some(&Value::from(2)));
        assert!(parsed.get("providers").is_some());

        let _ = std::fs::remove_file(path);
    }
}
