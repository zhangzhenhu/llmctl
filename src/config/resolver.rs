//! Resolve v2 config + CLI overrides into a single runtime plan.
//!
//! This module is the boundary between configuration parsing and runtime
//! execution. It keeps merge precedence explicit so `--dry-run` and
//! `--doctor-config` can explain exactly what llmctl will run.

use crate::config::schema::{
    AppConfigV2, Args, Message, OpenAiApiMode, ProviderProfile, RuntimeConfig,
};
use crate::error::LlmProbeError;
use serde_json::Value;
use std::collections::HashMap;

/// Records where the effective API key came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiKeySource {
    Cli,
    Env(String),
    Config,
    EmptyAllowed,
    Missing,
}

impl ApiKeySource {
    pub fn as_label(&self) -> String {
        match self {
            Self::Cli => "cli".to_string(),
            Self::Env(name) => format!("env:{name}"),
            Self::Config => "config".to_string(),
            Self::EmptyAllowed => "empty_allowed".to_string(),
            Self::Missing => "missing".to_string(),
        }
    }
}

/// Resolved runtime plan shared by execution, dry-run and doctor diagnostics.
#[derive(Debug, Clone)]
pub struct ResolvedRuntimeConfig {
    pub active_provider: String,
    pub adapter: String,
    pub provider_for_legacy_backend: String,
    pub model: String,
    pub base_url: Option<String>,
    pub api_key: String,
    pub api_key_source: ApiKeySource,
    pub stream: bool,
    pub context: Vec<Message>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub reasoning: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub reasoning_budget_tokens: Option<u32>,
    pub openai_api: OpenAiApiMode,
    pub openai_api_enforced: bool,
    pub effective_model: String,
    pub reasoning_setting: Option<String>,
    pub capture_usage: bool,
    pub capture_reasoning_content: bool,
    pub normalize_reasoning_content: bool,
    pub extra_body: HashMap<String, serde_json::Value>,
}

impl ResolvedRuntimeConfig {
    /// Convert to legacy runtime config so current llm backend path can keep
    /// running while we migrate request execution to genai.
    pub fn to_legacy_runtime_config(&self) -> RuntimeConfig {
        RuntimeConfig {
            provider: self.provider_for_legacy_backend.clone(),
            base_url: self.base_url.clone().unwrap_or_default(),
            api_key: self.api_key.clone(),
            model: self.model.clone(),
            stream: self.stream,
            context: self.context.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            system: self.system.clone(),
            timeout_seconds: self.timeout_seconds,
            reasoning: self.reasoning,
            reasoning_effort: self.reasoning_effort.clone(),
            reasoning_budget_tokens: self.reasoning_budget_tokens,
            extra_body: self.extra_body.clone(),
        }
    }
}

pub fn resolve_runtime_config(
    mut app: AppConfigV2,
    args: &Args,
) -> Result<ResolvedRuntimeConfig, LlmProbeError> {
    ensure_provider_profiles(&mut app, args);

    let selected_name = select_provider_name(&app, args)?;
    let mut profile = app.providers.get(&selected_name).cloned().ok_or_else(|| {
        LlmProbeError::ConfigError(format!("Provider profile not found: {selected_name}"))
    })?;

    if let Some(provider_arg) = &args.provider {
        profile.adapter = provider_arg.clone();
    }
    apply_cli_provider_defaults(&mut profile, args);

    let adapter = normalize_adapter_name(&profile.adapter);
    let base_url = args
        .url
        .as_ref()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .or_else(|| {
            profile
                .base_url
                .as_ref()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        });
    let model = args
        .model
        .clone()
        .or(profile.model.clone())
        .unwrap_or_default();

    let openai_api = args
        .endpoint
        .or(profile.openai_api)
        .or(app.defaults.openai_api)
        .unwrap_or(OpenAiApiMode::Auto);
    let (effective_model, openai_api_enforced) =
        resolve_effective_model(&adapter, &model, openai_api);

    let stream = if args.no_stream {
        false
    } else if args.stream {
        true
    } else {
        profile.stream.or(app.defaults.stream).unwrap_or(true)
    };

    let (api_key, api_key_source) = resolve_api_key(&adapter, &profile, args);

    let mut context = Vec::new();
    context.extend(app.context.clone());
    context.extend(profile.context.clone());
    for msg in &args.message {
        context.push(Message {
            role: "user".to_string(),
            content: msg.clone(),
        });
    }
    let (context, system) = split_system_messages(context, None);

    let provider_for_legacy_backend = legacy_provider_name(&adapter);
    let requested_reasoning = args
        .reasoning
        .clone()
        .or(profile.reasoning.clone())
        .or(profile.reasoning_effort.clone())
        .or(app.defaults.reasoning.clone());
    let capture_reasoning_default = app.defaults.capture_reasoning_content.unwrap_or(true);
    let (reasoning_setting, reasoning_effort, capture_reasoning_content, extra_body) =
        resolve_reasoning_controls(
            requested_reasoning,
            capture_reasoning_default,
            profile.extra_body,
        )?;

    Ok(ResolvedRuntimeConfig {
        active_provider: selected_name,
        adapter,
        provider_for_legacy_backend,
        model,
        base_url,
        api_key,
        api_key_source,
        stream,
        context,
        max_tokens: profile.max_tokens,
        temperature: profile.temperature,
        top_p: profile.top_p,
        top_k: None,
        system,
        timeout_seconds: profile.timeout_seconds.or(app.defaults.timeout_seconds),
        reasoning: None,
        reasoning_effort,
        reasoning_budget_tokens: None,
        openai_api,
        openai_api_enforced,
        effective_model,
        reasoning_setting,
        capture_usage: app.defaults.capture_usage.unwrap_or(true),
        capture_reasoning_content,
        normalize_reasoning_content: app.defaults.normalize_reasoning_content.unwrap_or(true),
        extra_body,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ReasoningControl {
    Off,
    Auto,
    Effort(String),
}

fn resolve_reasoning_controls(
    requested: Option<String>,
    capture_default: bool,
    mut extra_body: HashMap<String, Value>,
) -> Result<(Option<String>, Option<String>, bool, HashMap<String, Value>), LlmProbeError> {
    let Some(raw) = requested else {
        return Ok((None, None, capture_default, extra_body));
    };

    let control = parse_reasoning_control(&raw)?;
    let setting_label = Some(raw.trim().to_string());

    match control {
        ReasoningControl::Off => {
            if extra_body.contains_key("enable_thinking") {
                extra_body.insert("enable_thinking".to_string(), Value::Bool(false));
            }
            Ok((setting_label, Some("none".to_string()), false, extra_body))
        }
        ReasoningControl::Auto => Ok((setting_label, None, true, extra_body)),
        ReasoningControl::Effort(effort) => {
            if extra_body.contains_key("enable_thinking") {
                extra_body.insert("enable_thinking".to_string(), Value::Bool(true));
            }
            Ok((setting_label, Some(effort), true, extra_body))
        }
    }
}

fn parse_reasoning_control(input: &str) -> Result<ReasoningControl, LlmProbeError> {
    let value = input.trim().to_lowercase();
    if value.is_empty() {
        return Err(LlmProbeError::ConfigError(
            "reasoning cannot be empty".to_string(),
        ));
    }

    match value.as_str() {
        "off" | "false" | "disable" | "disabled" => return Ok(ReasoningControl::Off),
        "auto" => return Ok(ReasoningControl::Auto),
        "none" | "low" | "medium" | "high" | "xhigh" | "max" | "minimal" => {
            return Ok(ReasoningControl::Effort(value));
        }
        _ => {}
    }

    if let Some(raw) = value.strip_prefix("budget:") {
        let budget = raw.parse::<u32>().map_err(|_| {
            LlmProbeError::ConfigError(format!(
                "invalid reasoning value '{input}', budget must be an integer"
            ))
        })?;
        return Ok(ReasoningControl::Effort(budget.to_string()));
    }

    if let Ok(budget) = value.parse::<u32>() {
        return Ok(ReasoningControl::Effort(budget.to_string()));
    }

    Err(LlmProbeError::ConfigError(format!(
        "invalid reasoning value '{input}', expected off|auto|low|medium|high|xhigh|max|budget:<n>"
    )))
}

fn resolve_effective_model(
    adapter: &str,
    model: &str,
    openai_api: OpenAiApiMode,
) -> (String, bool) {
    if model.trim().is_empty() {
        return (String::new(), false);
    }

    if let Some((namespace, _)) = model.split_once("::") {
        // Preserve explicit namespace from user config/CLI.
        // This allows power users to force adapter/protocol manually.
        return (model.to_string(), namespace.starts_with("openai"));
    }

    let namespaced = match (adapter, openai_api) {
        ("openai", OpenAiApiMode::Responses) => format!("openai_resp::{model}"),
        ("openai", OpenAiApiMode::ChatCompletions) => format!("openai::{model}"),
        ("openai", OpenAiApiMode::Auto) => format!("openai::{model}"),
        ("aliyun", _) => format!("aliyun::{model}"),
        ("anthropic", _) => format!("anthropic::{model}"),
        ("gemini", _) => format!("gemini::{model}"),
        ("ollama", _) => format!("ollama::{model}"),
        ("deepseek", _) => format!("deepseek::{model}"),
        ("xai", _) => format!("xai::{model}"),
        ("groq", _) => format!("groq::{model}"),
        ("cohere", _) => format!("cohere::{model}"),
        ("fireworks", _) => format!("fireworks::{model}"),
        ("together", _) => format!("together::{model}"),
        ("zai", _) => format!("zai::{model}"),
        _ => model.to_string(),
    };
    let enforced = adapter == "openai" && openai_api != OpenAiApiMode::Auto;
    (namespaced, enforced)
}

fn ensure_provider_profiles(app: &mut AppConfigV2, args: &Args) {
    if !app.providers.is_empty() {
        return;
    }

    let mut profile = ProviderProfile::default();
    apply_preset_into_profile(&mut profile, resolve_cli_preset(args), true);

    if profile.adapter.is_empty() {
        profile.adapter = args
            .provider
            .clone()
            .unwrap_or_else(|| "openai".to_string());
    }

    app.providers.insert("default".to_string(), profile);

    if app.active_provider.is_none() {
        app.active_provider = Some("default".to_string());
    }
}

#[derive(Clone, Copy)]
struct BuiltinProviderPreset {
    adapter: &'static str,
    base_url: Option<&'static str>,
    api_key_env: Option<&'static str>,
    default_model: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinPresetInfo {
    pub name: String,
    pub aliases: Vec<String>,
    pub adapter: String,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub default_model: Option<String>,
}

#[derive(Clone, Copy)]
struct BuiltinPresetSpec {
    name: &'static str,
    aliases: &'static [&'static str],
    preset: BuiltinProviderPreset,
}

fn builtin_preset_specs() -> &'static [BuiltinPresetSpec] {
    &[
        BuiltinPresetSpec {
            name: "openai",
            aliases: &["openai", "openai-compatible", "openai_compatible"],
            preset: BuiltinProviderPreset {
                adapter: "openai",
                base_url: Some("https://api.openai.com/v1"),
                api_key_env: Some("OPENAI_API_KEY"),
                default_model: Some("gpt-4o"),
            },
        },
        BuiltinPresetSpec {
            name: "aliyun",
            aliases: &["aliyun", "dashscope"],
            preset: BuiltinProviderPreset {
                adapter: "aliyun",
                base_url: Some("https://dashscope.aliyuncs.com/compatible-mode/v1/"),
                api_key_env: Some("ALIYUN_API_KEY"),
                default_model: Some("qwen-max"),
            },
        },
        BuiltinPresetSpec {
            name: "anthropic",
            aliases: &["anthropic", "claude"],
            preset: BuiltinProviderPreset {
                adapter: "anthropic",
                base_url: Some("https://api.anthropic.com"),
                api_key_env: Some("ANTHROPIC_API_KEY"),
                default_model: Some("claude-sonnet-4-5"),
            },
        },
        BuiltinPresetSpec {
            name: "gemini",
            aliases: &["gemini", "google"],
            preset: BuiltinProviderPreset {
                adapter: "gemini",
                base_url: Some("https://generativelanguage.googleapis.com/v1beta"),
                api_key_env: Some("GEMINI_API_KEY"),
                default_model: Some("gemini-2.5-pro"),
            },
        },
        BuiltinPresetSpec {
            name: "ollama",
            aliases: &["ollama"],
            preset: BuiltinProviderPreset {
                adapter: "ollama",
                base_url: Some("http://localhost:11434"),
                api_key_env: None,
                default_model: Some("llama3.1"),
            },
        },
        BuiltinPresetSpec {
            name: "deepseek",
            aliases: &["deepseek"],
            preset: BuiltinProviderPreset {
                adapter: "deepseek",
                base_url: Some("https://api.deepseek.com/v1"),
                api_key_env: Some("DEEPSEEK_API_KEY"),
                default_model: Some("deepseek-chat"),
            },
        },
        BuiltinPresetSpec {
            name: "groq",
            aliases: &["groq"],
            preset: BuiltinProviderPreset {
                adapter: "groq",
                base_url: Some("https://api.groq.com/openai/v1"),
                api_key_env: Some("GROQ_API_KEY"),
                default_model: Some("llama-3.1-70b-versatile"),
            },
        },
        BuiltinPresetSpec {
            name: "mistral",
            aliases: &["mistral"],
            preset: BuiltinProviderPreset {
                adapter: "mistral",
                base_url: Some("https://api.mistral.ai/v1"),
                api_key_env: Some("MISTRAL_API_KEY"),
                default_model: Some("mistral-large-latest"),
            },
        },
    ]
}

pub fn list_builtin_provider_presets() -> Vec<BuiltinPresetInfo> {
    builtin_preset_specs()
        .iter()
        .map(|spec| BuiltinPresetInfo {
            name: spec.name.to_string(),
            aliases: spec.aliases.iter().map(|v| (*v).to_string()).collect(),
            adapter: spec.preset.adapter.to_string(),
            base_url: spec.preset.base_url.map(str::to_string),
            api_key_env: spec.preset.api_key_env.map(str::to_string),
            default_model: spec.preset.default_model.map(str::to_string),
        })
        .collect()
}

fn resolve_cli_preset(args: &Args) -> Option<BuiltinProviderPreset> {
    if let Some(provider) = args.provider.as_deref() {
        return preset_by_alias(provider);
    }
    preset_by_alias("openai")
}

fn apply_cli_provider_defaults(profile: &mut ProviderProfile, args: &Args) {
    let preset = args.provider.as_deref().and_then(preset_by_alias);
    let provider_overridden = args.provider.is_some();
    let model_explicitly_overridden = args.model.is_some();

    apply_preset_into_profile(profile, preset, false);

    // If user explicitly switches provider via CLI (e.g. `-p aliyun`) and does
    // not pass `-m`, we should also switch to that provider's default model.
    // This avoids invalid cross-provider defaults like `aliyun + gpt-4o`.
    if provider_overridden && !model_explicitly_overridden {
        if let Some(preset) = preset {
            profile.model = preset.default_model.map(str::to_string);
        }
    }
}

fn apply_preset_into_profile(
    profile: &mut ProviderProfile,
    preset: Option<BuiltinProviderPreset>,
    fill_model_when_missing: bool,
) {
    let Some(preset) = preset else {
        return;
    };

    if profile.adapter.trim().is_empty() {
        profile.adapter = preset.adapter.to_string();
    }
    if profile.base_url.as_deref().unwrap_or_default().trim().is_empty() {
        profile.base_url = preset.base_url.map(str::to_string);
    }
    if profile.api_key_env.as_deref().unwrap_or_default().trim().is_empty() {
        profile.api_key_env = preset.api_key_env.map(str::to_string);
    }
    if fill_model_when_missing || profile.model.is_none() {
        if profile.model.as_deref().unwrap_or_default().trim().is_empty() {
            profile.model = preset.default_model.map(str::to_string);
        }
    }
}

fn preset_by_alias(alias: &str) -> Option<BuiltinProviderPreset> {
    let alias = alias.trim().to_lowercase();
    builtin_preset_specs()
        .iter()
        .find(|spec| spec.aliases.iter().any(|candidate| *candidate == alias.as_str()))
        .map(|spec| spec.preset)
}

fn select_provider_name(app: &AppConfigV2, args: &Args) -> Result<String, LlmProbeError> {
    if let Some(name) = &args.profile {
        if app.providers.contains_key(name) {
            return Ok(name.clone());
        }
        return Err(LlmProbeError::ConfigError(format!(
            "Provider profile not found: {name}"
        )));
    }

    if let Some(active) = &app.active_provider {
        if app.providers.contains_key(active) {
            return Ok(active.clone());
        }
    }

    app.providers.keys().next().cloned().ok_or_else(|| {
        LlmProbeError::ConfigError("No provider profiles are configured".to_string())
    })
}

fn normalize_adapter_name(raw: &str) -> String {
    match raw.trim().to_lowercase().as_str() {
        "openai-compatible" | "openai_compatible" => "openai".to_string(),
        "dashscope" | "aliyun" => "aliyun".to_string(),
        "openairesp" | "openai_resp" | "openai-resp" => "openai".to_string(),
        "google" => "gemini".to_string(),
        "claude" => "anthropic".to_string(),
        other => other.to_string(),
    }
}

fn legacy_provider_name(adapter: &str) -> String {
    // We no longer remap LegacyReasoningContent to the historical
    // openai-compatible backend. Genai now handles reasoning + extra_body as
    // the primary path, and legacy provider name should stay adapter-aligned.
    adapter.to_string()
}

fn resolve_api_key(
    adapter: &str,
    profile: &ProviderProfile,
    args: &Args,
) -> (String, ApiKeySource) {
    if let Some(value) = args.secret.as_ref().or(args.key.as_ref()) {
        return (value.clone(), ApiKeySource::Cli);
    }

    let configured_env = profile
        .api_key_env
        .clone()
        .or_else(|| default_env_var(adapter));
    if let Some(env_name) = configured_env {
        if let Ok(value) = std::env::var(&env_name) {
            if !value.is_empty() {
                return (value, ApiKeySource::Env(env_name));
            }
        }
    }

    if let Ok(value) = std::env::var("LLM_API_KEY") {
        if !value.is_empty() {
            return (value, ApiKeySource::Env("LLM_API_KEY".to_string()));
        }
    }

    if let Some(value) = &profile.api_key {
        if !value.is_empty() {
            return (value.clone(), ApiKeySource::Config);
        }
    }

    if adapter == "ollama" {
        return (String::new(), ApiKeySource::EmptyAllowed);
    }

    (String::new(), ApiKeySource::Missing)
}

fn default_env_var(adapter: &str) -> Option<String> {
    match adapter {
        "openai" => Some("OPENAI_API_KEY".to_string()),
        "aliyun" => Some("ALIYUN_API_KEY".to_string()),
        "anthropic" => Some("ANTHROPIC_API_KEY".to_string()),
        "gemini" => Some("GEMINI_API_KEY".to_string()),
        "deepseek" => Some("DEEPSEEK_API_KEY".to_string()),
        "xai" => Some("XAI_API_KEY".to_string()),
        "groq" => Some("GROQ_API_KEY".to_string()),
        "mistral" => Some("MISTRAL_API_KEY".to_string()),
        _ => None,
    }
}

fn split_system_messages(
    messages: Vec<Message>,
    existing_system: Option<String>,
) -> (Vec<Message>, Option<String>) {
    let mut system_messages = Vec::new();
    let mut non_system = Vec::new();

    for message in messages {
        if message.role.eq_ignore_ascii_case("system") {
            system_messages.push(message.content);
        } else {
            non_system.push(message);
        }
    }

    if system_messages.is_empty() {
        return (non_system, existing_system);
    }

    let joined = system_messages.join("\n");
    let merged_system = match existing_system {
        Some(existing) if !existing.is_empty() => format!("{existing}\n{joined}"),
        _ => joined,
    };
    (non_system, Some(merged_system))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::{AppConfigV2, DefaultsConfig, ProviderProfile};
    use std::collections::BTreeMap;

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
            allow_sdk_default_api: false,
        }
    }

    #[test]
    fn resolve_uses_cli_overrides_and_profile_defaults() {
        let mut providers = BTreeMap::new();
        providers.insert(
            "openai_main".to_string(),
            ProviderProfile {
                adapter: "openai".to_string(),
                model: Some("gpt-4.1".to_string()),
                stream: Some(false),
                ..ProviderProfile::default()
            },
        );
        let app = AppConfigV2 {
            version: Some(2),
            active_provider: Some("openai_main".to_string()),
            defaults: DefaultsConfig {
                stream: Some(true),
                ..DefaultsConfig::default()
            },
            providers,
            context: vec![Message {
                role: "user".to_string(),
                content: "from-file".to_string(),
            }],
        };

        let mut input = args();
        input.model = Some("gpt-5".to_string());
        input.stream = true;
        input.message = vec!["from-cli".to_string()];
        input.endpoint = Some(OpenAiApiMode::Responses);

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert_eq!(resolved.model, "gpt-5");
        assert_eq!(resolved.effective_model, "openai_resp::gpt-5");
        assert!(resolved.stream);
        assert_eq!(resolved.context.len(), 2);
        assert_eq!(resolved.openai_api, OpenAiApiMode::Responses);
        assert!(resolved.openai_api_enforced);
    }

    #[test]
    fn resolve_supports_profile_switch_via_profile_arg() {
        let mut providers = BTreeMap::new();
        providers.insert(
            "openai_main".to_string(),
            ProviderProfile {
                adapter: "openai".to_string(),
                model: Some("gpt-5".to_string()),
                ..ProviderProfile::default()
            },
        );
        providers.insert(
            "anthropic_main".to_string(),
            ProviderProfile {
                adapter: "anthropic".to_string(),
                model: Some("claude-sonnet".to_string()),
                ..ProviderProfile::default()
            },
        );
        let app = AppConfigV2 {
            version: Some(2),
            active_provider: Some("openai_main".to_string()),
            defaults: DefaultsConfig::default(),
            providers,
            context: Vec::new(),
        };

        let mut input = args();
        input.profile = Some("anthropic_main".to_string());

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert_eq!(resolved.active_provider, "anthropic_main");
        assert_eq!(resolved.adapter, "anthropic");
        assert_eq!(resolved.model, "claude-sonnet");
    }

    #[test]
    fn resolve_keeps_adapter_for_legacy_backend() {
        let mut providers = BTreeMap::new();
        providers.insert(
            "compat".to_string(),
            ProviderProfile {
                adapter: "openai".to_string(),
                ..ProviderProfile::default()
            },
        );
        let app = AppConfigV2 {
            version: Some(2),
            active_provider: Some("compat".to_string()),
            defaults: DefaultsConfig::default(),
            providers,
            context: Vec::new(),
        };

        let resolved = resolve_runtime_config(app, &args()).expect("resolve failed");
        assert_eq!(resolved.provider_for_legacy_backend, "openai");
        assert_eq!(resolved.reasoning, None);
    }

    #[test]
    fn ensure_default_profile_uses_openai_preset_when_no_input() {
        let app = AppConfigV2::default();
        let resolved = resolve_runtime_config(app, &args()).expect("resolve failed");

        assert_eq!(resolved.adapter, "openai");
        assert_eq!(resolved.model, "gpt-4o");
        assert_eq!(resolved.base_url.as_deref(), Some("https://api.openai.com/v1"));
    }

    #[test]
    fn provider_alias_applies_builtin_preset_for_quick_start() {
        let app = AppConfigV2::default();
        let mut input = args();
        input.provider = Some("dashscope".to_string());

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert_eq!(resolved.adapter, "aliyun");
        assert_eq!(resolved.model, "qwen-max");
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("https://dashscope.aliyuncs.com/compatible-mode/v1/")
        );
    }

    #[test]
    fn unknown_provider_without_preset_keeps_raw_adapter() {
        let app = AppConfigV2::default();
        let mut input = args();
        input.provider = Some("my_custom_adapter".to_string());

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert_eq!(resolved.adapter, "my_custom_adapter");
        assert_eq!(resolved.model, "");
    }

    #[test]
    fn provider_arg_overrides_selected_profile_adapter() {
        let mut providers = BTreeMap::new();
        providers.insert(
            "profile_a".to_string(),
            ProviderProfile {
                adapter: "openai".to_string(),
                model: Some("gpt-4o".to_string()),
                ..ProviderProfile::default()
            },
        );
        let app = AppConfigV2 {
            version: Some(2),
            active_provider: Some("profile_a".to_string()),
            defaults: DefaultsConfig::default(),
            providers,
            context: Vec::new(),
        };

        let mut input = args();
        input.provider = Some("dashscope".to_string());

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert_eq!(resolved.active_provider, "profile_a");
        assert_eq!(resolved.adapter, "aliyun");
        assert_eq!(resolved.model, "qwen-max");
        assert_eq!(resolved.base_url.as_deref(), Some("https://dashscope.aliyuncs.com/compatible-mode/v1/"));
    }

    #[test]
    fn builtin_preset_list_contains_openai_and_aliyun() {
        let list = list_builtin_provider_presets();
        assert!(list.iter().any(|p| p.name == "openai"));
        assert!(list.iter().any(|p| p.name == "aliyun"));
        assert!(list
            .iter()
            .any(|p| p.name == "aliyun" && p.aliases.iter().any(|a| a == "dashscope")));
    }

    #[test]
    fn no_stream_overrides_default_stream_true() {
        let app = AppConfigV2::default();
        let mut input = args();
        input.no_stream = true;

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert!(!resolved.stream);
    }

    #[test]
    fn reasoning_off_disables_capture_and_sets_none_effort() {
        let mut app = AppConfigV2::default();
        app.defaults.reasoning = Some("off".to_string());
        let resolved = resolve_runtime_config(app, &args()).expect("resolve failed");
        assert_eq!(resolved.reasoning_effort.as_deref(), Some("none"));
        assert!(!resolved.capture_reasoning_content);
    }

    #[test]
    fn reasoning_budget_parses_to_numeric_effort() {
        let mut app = AppConfigV2::default();
        app.defaults.reasoning = Some("budget:2048".to_string());
        let resolved = resolve_runtime_config(app, &args()).expect("resolve failed");
        assert_eq!(resolved.reasoning_effort.as_deref(), Some("2048"));
        assert!(resolved.capture_reasoning_content);
    }

    #[test]
    fn reasoning_overrides_enable_thinking_when_present() {
        let mut providers = BTreeMap::new();
        let mut profile = ProviderProfile {
            adapter: "aliyun".to_string(),
            model: Some("glm-5".to_string()),
            ..ProviderProfile::default()
        };
        profile
            .extra_body
            .insert("enable_thinking".to_string(), serde_json::json!(true));
        providers.insert("ali".to_string(), profile);

        let app = AppConfigV2 {
            version: Some(2),
            active_provider: Some("ali".to_string()),
            defaults: DefaultsConfig::default(),
            providers,
            context: Vec::new(),
        };

        let mut input = args();
        input.reasoning = Some("off".to_string());

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert_eq!(
            resolved.extra_body.get("enable_thinking"),
            Some(&serde_json::json!(false))
        );
    }

    #[test]
    fn invalid_reasoning_is_rejected() {
        let mut input = args();
        input.reasoning = Some("weird".to_string());
        let err = resolve_runtime_config(AppConfigV2::default(), &input).unwrap_err();
        assert!(err
            .to_string()
            .contains("expected off|auto|low|medium|high|xhigh|max|budget:<n>"));
    }
}
