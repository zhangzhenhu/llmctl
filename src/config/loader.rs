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

    if let Some(provider) = &args.provider {
        config.provider = provider.clone();
    } else if config.provider.is_empty() {
        config.provider = "openai-compatible".to_string();
    }
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

    if args.secret.is_none() && args.key.is_none() {
        if let Ok(api_key) = std::env::var("LLM_API_KEY") {
            if !api_key.is_empty() {
                config.api_key = api_key;
            }
        }
    }

    extract_system_messages(&mut config);

    config
}

fn extract_system_messages(config: &mut RuntimeConfig) {
    let mut system_messages = Vec::new();
    config.context.retain(|message| {
        if message.role.eq_ignore_ascii_case("system") {
            system_messages.push(message.content.clone());
            false
        } else {
            true
        }
    });

    if system_messages.is_empty() {
        return;
    }

    let joined = system_messages.join("\n");
    config.system = Some(match config.system.take() {
        Some(existing) if !existing.is_empty() => format!("{}\n{}", existing, joined),
        _ => joined,
    });
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

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args {
            config: None,
            model: None,
            list: false,
            list_presets: false,
            message: Vec::new(),
            provider: None,
            profile: None,
            url: None,
            secret: None,
            key: None,
            stream: false,
            no_stream: false,
            version: false,
            init: None,
            init_path: None,
            convert: None,
            endpoint: None,
            reasoning: None,
            dry_run: false,
            doctor_config: false,
            legacy_runtime: false,
            allow_sdk_default_api: false,
        }
    }

    #[test]
    fn merge_preserves_file_provider_when_cli_provider_is_absent() {
        let file_config = FileConfig {
            provider: Some("anthropic".to_string()),
            base_url: Some("https://api.anthropic.com".to_string()),
            api_key: Some("key".to_string()),
            model: Some("claude".to_string()),
            ..FileConfig::default()
        };

        let config = merge_configs(Some(file_config), &args());

        assert_eq!(config.provider, "anthropic");
    }

    #[test]
    fn merge_uses_default_provider_when_no_provider_is_configured() {
        let config = merge_configs(None, &args());

        assert_eq!(config.provider, "openai-compatible");
    }

    #[test]
    fn merge_moves_system_context_messages_to_system_prompt() {
        let file_config = FileConfig {
            provider: Some("openai".to_string()),
            context: Some(vec![
                Message {
                    role: "system".to_string(),
                    content: "be concise".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                },
            ]),
            system: Some("base system".to_string()),
            ..FileConfig::default()
        };

        let config = merge_configs(Some(file_config), &args());

        assert_eq!(config.system.as_deref(), Some("base system\nbe concise"));
        assert_eq!(config.context.len(), 1);
        assert_eq!(config.context[0].role, "user");
    }

    #[test]
    fn validate_allows_empty_base_url() {
        let mut config = RuntimeConfig::new();
        config.provider = "openai".to_string();
        config.api_key = "key".to_string();
        config.model = "gpt".to_string();

        assert!(validate_config(&config).is_ok());
    }
}
