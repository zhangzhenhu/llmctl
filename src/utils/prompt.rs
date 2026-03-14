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

const DEFAULT_CONFIG_YAML: &str = r#"# LLM Probe Config File
# Provider: openai / gemini (extensible)
provider: "openai"

# API Base URL
# OpenAI compatible example: https://api.openai.com/v1
base_url: "https://api.openai.com/v1"

# API Key
# Recommended: set via environment variable LLM_API_KEY to avoid storing in plain text
api_key: ""

# Default model
model: "gpt-4"

# Enable streaming
stream: false

# Conversation context
context:
  - role: "system"
    content: "You are a helpful assistant."
  - role: "user"
    content: "Hello!"
"#;

const DEFAULT_CONFIG_JSON: &str = r#"{
  "provider": "openai",
  "base_url": "https://api.openai.com/v1",
  "api_key": "",
  "model": "gpt-4",
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
