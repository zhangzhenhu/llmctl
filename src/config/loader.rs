use crate::config::schema::{Args, FileConfig, Message, RuntimeConfig};
use crate::error::LlmProbeError;
use std::fs;
use std::path::Path;

pub fn load_config(path: &Path) -> Result<FileConfig, LlmProbeError> {
    let content = fs::read_to_string(path)
        .map_err(|_| LlmProbeError::ConfigFileNotFound(path.display().to_string()))?;

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext.to_lowercase().as_str() {
        "yaml" | "yml" => {
            serde_yaml::from_str(&content).map_err(|_| LlmProbeError::ConfigFormatError)
        }
        "json" => serde_json::from_str(&content).map_err(|_| LlmProbeError::ConfigFormatError),
        _ => Err(LlmProbeError::ConfigFormatError),
    }
}

pub fn merge_configs(file_config: Option<FileConfig>, args: &Args) -> RuntimeConfig {
    let mut config = RuntimeConfig::new();

    if let Some(fc) = file_config {
        if let Some(provider) = fc.provider {
            config.provider = provider;
        }
        if let Some(base_url) = fc.base_url {
            config.base_url = base_url;
        }
        if let Some(api_key) = fc.api_key {
            config.api_key = api_key;
        }
        if let Some(model) = fc.model {
            config.model = model;
        }
        if let Some(stream) = fc.stream {
            config.stream = stream;
        }
        if let Some(context) = fc.context {
            config.context = context;
        }
        config.max_tokens = fc.max_tokens;
        config.temperature = fc.temperature;
        config.top_p = fc.top_p;
        config.top_k = fc.top_k;
        config.system = fc.system;
        config.timeout_seconds = fc.timeout_seconds;
        config.reasoning = fc.reasoning;
        config.reasoning_effort = fc.reasoning_effort;
        config.reasoning_budget_tokens = fc.reasoning_budget_tokens;
        config.extra_body = fc.extra_body;
    }

    config.provider = args.provider.clone();
    if let Some(base_url) = &args.url {
        config.base_url = base_url.clone();
    }
    // Support both --secret (-s) and --key (-k) for API key
    if let Some(api_key) = &args.secret {
        config.api_key = api_key.clone();
    }
    if let Some(api_key) = &args.key {
        config.api_key = api_key.clone();
    }
    if let Some(model) = &args.model {
        config.model = model.clone();
    }
    if args.stream {
        config.stream = true;
    }

    for msg in &args.message {
        config.context.push(Message {
            role: "user".to_string(),
            content: msg.clone(),
        });
    }

    if let Ok(api_key) = std::env::var("LLM_API_KEY") {
        if config.api_key.is_empty() {
            config.api_key = api_key;
        }
    }

    config
}

pub fn validate_config(config: &RuntimeConfig) -> Result<(), LlmProbeError> {
    validate_config_with_list(config, false)
}

pub fn validate_config_with_list(
    config: &RuntimeConfig,
    is_list_mode: bool,
) -> Result<(), LlmProbeError> {
    if config.provider.is_empty() {
        return Err(LlmProbeError::MissingRequiredField("provider".to_string()));
    }
    if config.base_url.is_empty() {
        return Err(LlmProbeError::MissingRequiredField("base_url".to_string()));
    }
    if config.api_key.is_empty() {
        return Err(LlmProbeError::MissingRequiredField("api_key".to_string()));
    }
    // Model is not required when listing models
    if !is_list_mode && config.model.is_empty() {
        return Err(LlmProbeError::MissingRequiredField("model".to_string()));
    }

    Ok(())
}

pub fn search_config_file() -> Option<std::path::PathBuf> {
    let search_paths = [
        std::path::PathBuf::from("./llmctl.yaml"),
        std::path::PathBuf::from("./llmctl.json"),
        std::path::PathBuf::from("./llm.yaml"),
        std::path::PathBuf::from("./llm.json"),
    ];

    for path in &search_paths {
        if path.exists() {
            return Some(path.clone());
        }
    }

    if let Some(config_dir) = dirs::config_dir() {
        let app_config_dir = config_dir.join("llmctl");
        for ext in ["yaml", "json"] {
            let path = app_config_dir.join(format!("config.{}", ext));
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}
