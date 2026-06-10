//! Validate resolved runtime plan and report actionable diagnostics.
//!
//! Validation is separate from resolution so we can reuse it for normal
//! execution, `--dry-run`, and `--doctor-config`.

use crate::config::resolver::{is_builtin_adapter_name, ResolvedRuntimeConfig};
use crate::config::schema::{Args, OpenAiApiMode};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigDiagnostic {
    pub severity: DiagnosticSeverity,
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    pub diagnostics: Vec<ConfigDiagnostic>,
}

impl ValidationReport {
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == DiagnosticSeverity::Error)
    }

    pub fn errors(&self) -> Vec<&ConfigDiagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .collect()
    }
}

pub fn validate_resolved_config(resolved: &ResolvedRuntimeConfig, args: &Args) -> ValidationReport {
    let mut report = ValidationReport::default();

    if !is_supported_adapter(&resolved.adapter) {
        report.diagnostics.push(ConfigDiagnostic {
            severity: DiagnosticSeverity::Error,
            code: "unsupported_adapter".to_string(),
            message: format!("Unsupported adapter: {}", resolved.adapter),
        });
    }

    if resolved.api_key.is_empty() && resolved.adapter != "ollama" {
        report.diagnostics.push(ConfigDiagnostic {
            severity: DiagnosticSeverity::Error,
            code: "missing_api_key".to_string(),
            message: "API key is required for this adapter".to_string(),
        });
    }

    if resolved.model.trim().is_empty() && !args.list {
        report.diagnostics.push(ConfigDiagnostic {
            severity: DiagnosticSeverity::Error,
            code: "missing_model".to_string(),
            message: "Model is required unless --list is set".to_string(),
        });
    }

    if let Some(base_url) = &resolved.base_url {
        if reqwest::Url::parse(base_url).is_err() {
            report.diagnostics.push(ConfigDiagnostic {
                severity: DiagnosticSeverity::Error,
                code: "invalid_base_url".to_string(),
                message: format!("Invalid base_url: {base_url}"),
            });
        }
    }

    if resolved.top_k.is_some() {
        report.diagnostics.push(ConfigDiagnostic {
            severity: DiagnosticSeverity::Warning,
            code: "top_k_not_applied".to_string(),
            message: "top_k is parsed but not applied by the current genai runtime".to_string(),
        });
    }

    if resolved.adapter == "openai"
        && resolved.api_mode != OpenAiApiMode::Auto
        && !resolved.api_mode_enforced
    {
        report.diagnostics.push(ConfigDiagnostic {
            severity: DiagnosticSeverity::Error,
            code: "api_mode_not_enforced".to_string(),
            message: "api_mode is requested but runtime did not enforce a protocol namespace"
                .to_string(),
        });
    }

    if !matches!(resolved.adapter.as_str(), "openai" | "aliyun")
        && resolved.api_mode != OpenAiApiMode::Auto
    {
        report.diagnostics.push(ConfigDiagnostic {
            severity: DiagnosticSeverity::Warning,
            code: "api_mode_ignored".to_string(),
            message: "api_mode is ignored because adapter is not OpenAI-compatible".to_string(),
        });
    }

    report
}

fn is_supported_adapter(adapter: &str) -> bool {
    is_builtin_adapter_name(adapter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::resolver::{ApiKeySource, ResolvedRuntimeConfig};
    use crate::config::schema::Message;
    use std::collections::HashMap;

    fn args() -> Args {
        Args {
            config: None,
            model: None,
            list: false,
            list_adapters: false,
            message: Vec::new(),
            prompt: Vec::new(),
            adapter: None,
            profile: None,
            base_url: None,
            secret: None,
            key: None,
            stream: false,
            no_stream: false,
            no_proxy: false,
            version: false,
            init: None,
            init_path: None,
            convert: None,
            api_mode: None,
            reasoning: None,
            dry_run: false,
            doctor_config: false,
        }
    }

    fn resolved() -> ResolvedRuntimeConfig {
        ResolvedRuntimeConfig {
            active_profile: "default".to_string(),
            adapter: "openai".to_string(),
            model: "gpt-5".to_string(),
            base_url: Some("https://api.openai.com/v1".to_string()),
            api_key: "secret".to_string(),
            api_key_source: ApiKeySource::Config,
            stream: false,
            context: vec![Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            system: None,
            timeout_seconds: Some(60),
            reasoning_effort: None,
            api_mode: OpenAiApiMode::Auto,
            api_mode_enforced: false,
            effective_model: "openai::gpt-5".to_string(),
            no_proxy: false,
            reasoning_setting: None,
            capture_usage: true,
            capture_reasoning_content: true,
            normalize_reasoning_content: true,
            extra_body: HashMap::new(),
        }
    }

    #[test]
    fn explicit_openai_endpoint_requires_enforcement() {
        let mut cfg = resolved();
        cfg.api_mode = OpenAiApiMode::Responses;
        cfg.api_mode_enforced = false;
        let report = validate_resolved_config(&cfg, &args());
        assert!(report.has_errors());
        assert!(report
            .diagnostics
            .iter()
            .any(|d| d.code == "api_mode_not_enforced"));
    }

    #[test]
    fn enforced_openai_endpoint_has_no_error() {
        let mut cfg = resolved();
        cfg.api_mode = OpenAiApiMode::ChatCompletions;
        cfg.api_mode_enforced = true;
        let report = validate_resolved_config(&cfg, &args());
        assert!(!report.has_errors());
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn extra_body_no_longer_emits_legacy_fallback_warning() {
        let mut cfg = resolved();
        cfg.adapter = "anthropic".to_string();
        cfg.effective_model = "anthropic::claude-sonnet-4-5".to_string();
        cfg.extra_body
            .insert("enable_thinking".to_string(), serde_json::Value::Bool(true));

        let report = validate_resolved_config(&cfg, &args());
        assert!(!report
            .diagnostics
            .iter()
            .any(|d| d.code == "extra_body_legacy_fallback_for_chat"));
    }

    #[test]
    fn top_k_emits_explicit_warning() {
        let mut cfg = resolved();
        cfg.top_k = Some(40);

        let report = validate_resolved_config(&cfg, &args());
        assert!(report
            .diagnostics
            .iter()
            .any(|d| d.code == "top_k_not_applied"));
    }

    #[test]
    fn new_genai_07_adapter_is_accepted() {
        let mut cfg = resolved();
        cfg.adapter = "open_router".to_string();
        cfg.effective_model = "open_router::openai/gpt-4.1".to_string();

        let report = validate_resolved_config(&cfg, &args());
        assert!(!report.has_errors());
    }
}
