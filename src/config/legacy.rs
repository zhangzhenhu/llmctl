use crate::config::schema::{AppConfigV2, FileConfig, OpenAiApiMode, ProviderProfile};
use crate::error::LlmProbeError;
use serde_json::Value;
use std::fs;
use std::path::Path;

const DEFAULT_PROFILE: &str = "default";

/// Load any supported config shape and normalize it to v2.
///
/// v2 files are parsed directly. Legacy flat files are migrated to one
/// provider profile so the rest of the runtime can operate on a single schema.
pub fn load_app_config(path: &Path) -> Result<AppConfigV2, LlmProbeError> {
    let content = fs::read_to_string(path)
        .map_err(|_| LlmProbeError::ConfigFileNotFound(path.display().to_string()))?;
    let value = parse_config_value(path, &content)?;

    if is_v2_config_value(&value) {
        serde_json::from_value(value)
            .map_err(|e| LlmProbeError::ConfigError(format!("Invalid v2 config schema: {e}")))
    } else {
        let legacy: FileConfig = serde_json::from_value(value).map_err(|e| {
            LlmProbeError::ConfigError(format!("Invalid legacy config schema: {e}"))
        })?;
        Ok(legacy_file_config_to_v2(legacy))
    }
}

pub fn is_v2_config_value(value: &Value) -> bool {
    match value {
        Value::Object(map) => map.contains_key("providers") || map.contains_key("active_provider"),
        _ => false,
    }
}

pub fn legacy_file_config_to_v2(config: FileConfig) -> AppConfigV2 {
    let provider_name = config
        .provider
        .unwrap_or_else(|| "openai-compatible".to_string());
    let adapter = provider_alias_to_adapter(&provider_name);
    let openai_api = default_openai_api_for_legacy_provider(&provider_name);

    // Legacy config stored both system and context together. We preserve context
    // as-is and let runtime normalization extract system messages later.
    let mut extra_body = config.extra_body;
    // Preserve the legacy boolean reasoning toggle by mapping it to the same
    // `extra_body.enable_thinking` key used by OpenAI-compatible providers.
    if config.reasoning.unwrap_or(false) {
        extra_body
            .entry("enable_thinking".to_string())
            .or_insert(Value::Bool(true));
    }

    let profile = ProviderProfile {
        adapter,
        model: config.model,
        base_url: config.base_url,
        api_key: config.api_key,
        api_key_env: None,
        stream: config.stream,
        timeout_seconds: config.timeout_seconds,
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        top_p: config.top_p,
        reasoning: None,
        reasoning_effort: config.reasoning_effort,
        openai_api,
        extra_body,
        context: Vec::new(),
    };

    let mut app = AppConfigV2 {
        version: Some(2),
        active_provider: Some(DEFAULT_PROFILE.to_string()),
        defaults: Default::default(),
        providers: Default::default(),
        context: config.context.unwrap_or_default(),
    };
    app.providers.insert(DEFAULT_PROFILE.to_string(), profile);
    app
}

fn parse_config_value(path: &Path, content: &str) -> Result<Value, LlmProbeError> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext.to_lowercase().as_str() {
        "yaml" | "yml" => {
            serde_yaml::from_str::<Value>(content).map_err(|_| LlmProbeError::ConfigFormatError)
        }
        "json" => {
            serde_json::from_str::<Value>(content).map_err(|_| LlmProbeError::ConfigFormatError)
        }
        _ => Err(LlmProbeError::ConfigFormatError),
    }
}

fn provider_alias_to_adapter(provider: &str) -> String {
    match provider.to_lowercase().as_str() {
        "anthropic" | "claude" => "anthropic".to_string(),
        "gemini" | "google" => "gemini".to_string(),
        "ollama" => "ollama".to_string(),
        // Dashscope/Aliyun has a first-class adapter in genai 0.6+.
        "dashscope" | "aliyun" => "aliyun".to_string(),
        // All OpenAI-compatible aliases route through the OpenAI adapter with
        // a custom base_url in v2.
        "openai-compatible" | "openai_compatible" => "openai".to_string(),
        _ => "openai".to_string(),
    }
}

fn default_openai_api_for_legacy_provider(provider: &str) -> Option<OpenAiApiMode> {
    match provider.to_lowercase().as_str() {
        "openai-compatible" | "openai_compatible" => Some(OpenAiApiMode::ChatCompletions),
        _ => Some(OpenAiApiMode::Auto),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::Message;
    use serde_json::json;

    #[test]
    fn migrates_flat_config_to_default_provider_profile() {
        let legacy = FileConfig {
            provider: Some("dashscope".to_string()),
            base_url: Some("https://dashscope.aliyuncs.com/compatible-mode/v1".to_string()),
            api_key: Some("key".to_string()),
            model: Some("qwen3-max".to_string()),
            stream: Some(true),
            ..FileConfig::default()
        };

        let app = legacy_file_config_to_v2(legacy);
        let profile = app.providers.get(DEFAULT_PROFILE).expect("profile missing");

        assert_eq!(app.version, Some(2));
        assert_eq!(app.active_provider.as_deref(), Some(DEFAULT_PROFILE));
        assert_eq!(profile.adapter, "aliyun");
        assert_eq!(profile.openai_api, Some(OpenAiApiMode::Auto));
        assert_eq!(profile.model.as_deref(), Some("qwen3-max"));
    }

    #[test]
    fn preserves_context_and_extra_body() {
        let mut extra_body = std::collections::HashMap::new();
        extra_body.insert("enable_thinking".to_string(), json!(true));

        let legacy = FileConfig {
            provider: Some("openai-compatible".to_string()),
            context: Some(vec![
                Message {
                    role: "system".to_string(),
                    content: "You are concise.".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                },
            ]),
            extra_body,
            ..FileConfig::default()
        };

        let app = legacy_file_config_to_v2(legacy);
        let profile = app.providers.get(DEFAULT_PROFILE).expect("profile missing");

        assert_eq!(app.context.len(), 2);
        assert!(profile.extra_body.contains_key("enable_thinking"));
        assert_eq!(
            profile.extra_body.get("enable_thinking"),
            Some(&Value::Bool(true))
        );
    }
}
