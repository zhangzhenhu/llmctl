use crate::config::{
    BuiltinAdapterInfo, ConfigDiagnostic, DiagnosticSeverity, ResolvedRuntimeConfig,
    ValidationReport,
};
use crate::provider::ChatResponse;
use colored::*;
use std::collections::BTreeMap;

pub fn format_chat_response(response: &ChatResponse) {
    if let Some(ref reasoning) = response.reasoning_content {
        println!("{}:", "Thinking".cyan());
        println!("{}", "─".repeat(50).dimmed());
        println!("{}", reasoning);
        println!("{}", "─".repeat(50).dimmed());
    }

    if let Some(ref content) = response.content {
        println!("{}:", "Response".cyan());
        println!("{}", "─".repeat(50).dimmed());
        println!("{}", content);
    }
    println!("{}", "─".repeat(50).dimmed());

    if let (Some(input), Some(output)) = (response.input_tokens, response.output_tokens) {
        println!("{}: Input {}, Output {}", "Token".dimmed(), input, output);
    }
    for line in response_metadata_lines(response) {
        println!("{line}");
    }
    println!("{}: {} ms", "Duration".yellow(), response.duration_ms);
}

pub(crate) fn response_metadata_lines(response: &ChatResponse) -> Vec<String> {
    let mut lines = vec![format!(
        "{}: {}  {}: {}  {}: {}",
        "Profile".blue(),
        response.profile.blue(),
        "Adapter".cyan(),
        response.adapter.cyan(),
        "Model".green(),
        response.provider_model.green()
    )];

    if response.provider_model != response.requested_model {
        lines.push(format!(
            "{}: {}",
            "Requested Model".dimmed(),
            response.requested_model.dimmed()
        ));
    }

    lines
}

pub fn format_model_list(models: &[crate::provider::ModelInfo]) {
    let sorted_names = sorted_unique_model_names(models);

    println!("{}", "─".repeat(50).dimmed());
    println!("{}", "Available Models:".green());
    println!("{}", "─".repeat(50).dimmed());

    for name in sorted_names {
        println!("  - {}", name);
    }

    println!("{}", "─".repeat(50).dimmed());
}

fn sorted_unique_model_names(models: &[crate::provider::ModelInfo]) -> Vec<String> {
    let mut names: Vec<String> = models.iter().map(|m| m.name.clone()).collect();
    names.sort_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()).then(a.cmp(b)));
    names.dedup();
    names
}

pub fn print_adapter_list(adapters: &[BuiltinAdapterInfo]) {
    println!("{}", "Supported Adapters".green());
    println!("{}", "─".repeat(50).dimmed());
    for adapter in adapters {
        println!("name: {}", adapter.name);
        println!(
            "  default_base_url: {}",
            adapter.default_base_url.as_deref().unwrap_or("<none>")
        );
        println!(
            "  api_key_env: {}",
            adapter.api_key_env.as_deref().unwrap_or("<none>")
        );
        println!(
            "  default_model: {}",
            adapter.default_model.as_deref().unwrap_or("<none>")
        );
        if !adapter.aliases.is_empty() {
            println!("  aliases: {}", adapter.aliases.join(", "));
        }
        println!("{}", "─".repeat(50).dimmed());
    }
}

pub fn print_error(error: &str) {
    eprintln!("{}: {}", "Error".red(), error.red());
}

pub fn print_success(message: &str) {
    println!("{}", message.green());
}

pub fn print_info(message: &str) {
    println!("{}", message);
}

pub fn format_dry_run_lines(
    resolved: &ResolvedRuntimeConfig,
    runtime_backend: &str,
    runtime_fallback_reason: Option<&str>,
    extra_body_supported: bool,
) -> Vec<String> {
    let mut lines = vec![
        format!("runtime_backend: {runtime_backend}"),
        format!("active_profile: {}", resolved.active_profile),
        format!("adapter: {}", resolved.adapter),
        format!("model: {}", resolved.model),
        format!("effective_model: {}", resolved.effective_model),
        format!(
            "base_url: {}",
            resolved.base_url.as_deref().unwrap_or("<default>")
        ),
        format!("api_key_source: {}", resolved.api_key_source.as_label()),
        format!("stream: {}", resolved.stream),
        format!(
            "proxy_mode: {}",
            if resolved.no_proxy {
                "disabled"
            } else {
                "inherit_system"
            }
        ),
        format!("api_mode_requested: {:?}", resolved.api_mode),
        format!("api_mode_enforced: {}", resolved.api_mode_enforced),
        format!(
            "api_mode_runtime: {}",
            if resolved.api_mode == crate::config::schema::OpenAiApiMode::Auto {
                "sdk_default"
            } else if resolved.api_mode_enforced {
                "enforced_by_model_namespace"
            } else {
                "requested_but_not_enforced"
            }
        ),
        format!(
            "reasoning_setting: {}",
            resolved.reasoning_setting.as_deref().unwrap_or("<none>")
        ),
        format!(
            "reasoning_effort: {}",
            resolved.reasoning_effort.as_deref().unwrap_or("<none>")
        ),
        format!(
            "top_k: {}",
            resolved
                .top_k
                .map(|value| value.to_string())
                .unwrap_or_else(|| "<none>".to_string())
        ),
        format!("capture_usage: {}", resolved.capture_usage),
        format!(
            "capture_reasoning_content: {}",
            resolved.capture_reasoning_content
        ),
        format!(
            "normalize_reasoning_content: {}",
            resolved.normalize_reasoning_content
        ),
        format!("extra_body_supported: {extra_body_supported}"),
    ];
    if let Some(reason) = runtime_fallback_reason {
        lines.push(format!("runtime_fallback_reason: {reason}"));
    }

    let mut sorted_extra_body = BTreeMap::new();
    for (k, v) in &resolved.extra_body {
        sorted_extra_body.insert(k, v);
    }
    if !sorted_extra_body.is_empty() {
        lines.push("extra_body_keys:".to_string());
        for key in sorted_extra_body.keys() {
            lines.push(format!("  - {}", key));
        }
    }

    lines
}

pub fn print_dry_run(
    resolved: &ResolvedRuntimeConfig,
    report: &ValidationReport,
    runtime_backend: &str,
    runtime_fallback_reason: Option<&str>,
    extra_body_supported: bool,
) {
    println!("{}", "Dry Run Plan".green());
    println!("{}", "─".repeat(50).dimmed());
    for line in format_dry_run_lines(
        resolved,
        runtime_backend,
        runtime_fallback_reason,
        extra_body_supported,
    ) {
        println!("{line}");
    }

    let warnings: Vec<&ConfigDiagnostic> = report
        .diagnostics
        .iter()
        .filter(|d| d.severity == DiagnosticSeverity::Warning)
        .collect();
    if !warnings.is_empty() {
        println!("{}", "─".repeat(50).dimmed());
        println!("{}", "Warnings".yellow());
        for warning in warnings {
            println!("- [{}] {}", warning.code, warning.message);
        }
    }
}

pub fn print_doctor_report(report: &ValidationReport) {
    println!("{}", "Config Diagnostics".green());
    println!("{}", "─".repeat(50).dimmed());

    if report.diagnostics.is_empty() {
        println!("status: ok");
        return;
    }

    let mut has_error = false;
    for diagnostic in &report.diagnostics {
        match diagnostic.severity {
            DiagnosticSeverity::Error => {
                has_error = true;
                println!("error [{}] {}", diagnostic.code, diagnostic.message);
            }
            DiagnosticSeverity::Warning => {
                println!("warning [{}] {}", diagnostic.code, diagnostic.message);
            }
        }
    }

    println!("{}", "─".repeat(50).dimmed());
    if has_error {
        println!("status: failed");
    } else {
        println!("status: passed_with_warnings");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::resolver::ApiKeySource;
    use crate::config::schema::{Message, OpenAiApiMode};
    use crate::config::ValidationReport;
    use crate::provider::ChatResponse;
    use std::collections::HashMap;

    fn resolved_for_test() -> ResolvedRuntimeConfig {
        ResolvedRuntimeConfig {
            active_profile: "dashscope_qwen".to_string(),
            adapter: "openai".to_string(),
            model: "qwen3-max".to_string(),
            base_url: Some("https://dashscope.aliyuncs.com/compatible-mode/v1".to_string()),
            api_key: "sk-very-secret".to_string(),
            api_key_source: ApiKeySource::Env("DASHSCOPE_API_KEY".to_string()),
            stream: true,
            context: vec![Message {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            system: None,
            timeout_seconds: Some(60),
            reasoning_effort: Some("medium".to_string()),
            api_mode: OpenAiApiMode::ChatCompletions,
            api_mode_enforced: true,
            effective_model: "openai::qwen3-max".to_string(),
            no_proxy: false,
            reasoning_setting: Some("high".to_string()),
            capture_usage: true,
            capture_reasoning_content: true,
            normalize_reasoning_content: true,
            extra_body: {
                let mut m = HashMap::new();
                m.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
                m
            },
        }
    }

    #[test]
    fn dry_run_lines_do_not_leak_key_value() {
        let lines = format_dry_run_lines(&resolved_for_test(), "genai", None, true);
        let merged = lines.join("\n");
        assert!(!merged.contains("sk-very-secret"));
        assert!(merged.contains("api_key_source: env:DASHSCOPE_API_KEY"));
    }

    #[test]
    fn dry_run_lists_extra_body_keys_only() {
        let lines = format_dry_run_lines(&resolved_for_test(), "genai", None, true);
        let merged = lines.join("\n");
        assert!(merged.contains("extra_body_keys:"));
        assert!(merged.contains("enable_thinking"));
        assert!(!merged.contains("enable_thinking: true"));
    }

    #[test]
    fn doctor_report_has_stable_empty_status() {
        let report = ValidationReport::default();
        assert!(!report.has_errors());
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn model_list_names_are_sorted_and_deduplicated() {
        let models = vec![
            crate::provider::ModelInfo {
                id: "2".to_string(),
                name: "qwen-plus".to_string(),
                provider: "aliyun".to_string(),
            },
            crate::provider::ModelInfo {
                id: "1".to_string(),
                name: "qwen-max".to_string(),
                provider: "aliyun".to_string(),
            },
            crate::provider::ModelInfo {
                id: "3".to_string(),
                name: "qwen-plus".to_string(),
                provider: "aliyun".to_string(),
            },
        ];

        let names = sorted_unique_model_names(&models);
        assert_eq!(names, vec!["qwen-max".to_string(), "qwen-plus".to_string()]);
    }

    #[test]
    fn response_metadata_keeps_profile_adapter_and_model_on_one_line() {
        let response = ChatResponse {
            profile: "aliyun".to_string(),
            adapter: "aliyun".to_string(),
            requested_model: "aliyun::MiniMax-M2.5".to_string(),
            provider_model: "MiniMax-M2.5".to_string(),
            content: Some("hello".to_string()),
            reasoning_content: None,
            duration_ms: 42,
            input_tokens: Some(10),
            output_tokens: Some(20),
        };

        let lines = response_metadata_lines(&response);
        let rendered = lines.join("\n");

        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("Profile: aliyun"));
        assert!(lines[0].contains("Adapter: aliyun"));
        assert!(lines[0].contains("Model: MiniMax-M2.5"));
        assert!(rendered.contains("Requested Model: aliyun::MiniMax-M2.5"));
    }
}
