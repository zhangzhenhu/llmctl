#![allow(dead_code)]
use std::fs;
use std::io::{self, Write};
use std::path::Path;

pub fn prompt_overwrite(path: &Path) -> bool {
    print!("文件 {} 已存在，是否覆盖？[y/N]: ", path.display());
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
            return Err("操作已取消".to_string());
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

const DEFAULT_CONFIG_YAML: &str = r#"# LLM Probe 配置文件
# 服务商：openai / gemini（可扩展）
provider: "openai"

# API 基础地址
# OpenAI 兼容接口示例：https://api.openai.com/v1
base_url: "https://api.openai.com/v1"

# API 密钥
# 建议通过环境变量 LLM_API_KEY 设置，避免明文存储
api_key: ""

# 默认模型
model: "gpt-4"

# 是否流式返回
stream: false

# 对话上下文
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
