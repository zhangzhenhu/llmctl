//! Resolve v2 config + CLI overrides into a single runtime plan.
//!
//! This module is the boundary between configuration parsing and runtime
//! execution. It keeps merge precedence explicit so `--dry-run` and
//! `--doctor-config` can explain exactly what llmctl will run.

use crate::config::schema::{AppConfigV2, Args, Message, OpenAiApiMode, ProviderProfile};
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
    pub active_profile: String,
    pub adapter: String,
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
    pub reasoning_effort: Option<String>,
    pub api_mode: OpenAiApiMode,
    pub api_mode_enforced: bool,
    pub effective_model: String,
    pub no_proxy: bool,
    pub reasoning_setting: Option<String>,
    pub capture_usage: bool,
    pub capture_reasoning_content: bool,
    pub normalize_reasoning_content: bool,
    pub extra_body: HashMap<String, serde_json::Value>,
}

#[derive(Clone, Copy)]
struct AdapterDefaults {
    base_url: Option<&'static str>,
    api_key_env: Option<&'static str>,
    default_model: Option<&'static str>,
}

#[derive(Clone, Copy)]
struct BuiltinAdapterSpec {
    name: &'static str,
    aliases: &'static [&'static str],
    defaults: AdapterDefaults,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinAdapterInfo {
    pub name: String,
    pub aliases: Vec<String>,
    pub default_base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub default_model: Option<String>,
}

const BUILTIN_ADAPTER_SPECS: &[BuiltinAdapterSpec] = &[
    BuiltinAdapterSpec {
        name: "openai",
        aliases: &[
            "openai",
            "oi",
            "oai",
            "openai-compatible",
            "openai_compatible",
        ],
        defaults: AdapterDefaults {
            base_url: Some("https://api.openai.com/v1"),
            api_key_env: Some("OPENAI_API_KEY"),
            default_model: Some("gpt-4o"),
        },
    },
    BuiltinAdapterSpec {
        name: "aliyun",
        aliases: &["aliyun", "ali", "dashscope", "ds"],
        defaults: AdapterDefaults {
            base_url: Some("https://dashscope.aliyuncs.com/compatible-mode/v1/"),
            api_key_env: Some("ALIYUN_API_KEY"),
            default_model: Some("qwen-max"),
        },
    },
    BuiltinAdapterSpec {
        name: "anthropic",
        aliases: &["anthropic", "claude", "anth"],
        defaults: AdapterDefaults {
            base_url: Some("https://api.anthropic.com"),
            api_key_env: Some("ANTHROPIC_API_KEY"),
            default_model: Some("claude-sonnet-4-5"),
        },
    },
    BuiltinAdapterSpec {
        name: "gemini",
        aliases: &["gemini", "google", "gmi"],
        defaults: AdapterDefaults {
            base_url: Some("https://generativelanguage.googleapis.com/v1beta"),
            api_key_env: Some("GEMINI_API_KEY"),
            default_model: Some("gemini-2.5-pro"),
        },
    },
    BuiltinAdapterSpec {
        name: "ollama",
        aliases: &["ollama", "ol"],
        defaults: AdapterDefaults {
            base_url: Some("http://localhost:11434"),
            api_key_env: None,
            default_model: Some("llama3.1"),
        },
    },
    BuiltinAdapterSpec {
        name: "deepseek",
        aliases: &["deepseek", "dsk"],
        defaults: AdapterDefaults {
            base_url: Some("https://api.deepseek.com/v1"),
            api_key_env: Some("DEEPSEEK_API_KEY"),
            default_model: Some("deepseek-chat"),
        },
    },
    BuiltinAdapterSpec {
        name: "xai",
        aliases: &["xai", "grok"],
        defaults: AdapterDefaults {
            base_url: Some("https://api.x.ai/v1"),
            api_key_env: Some("XAI_API_KEY"),
            default_model: Some("grok-4"),
        },
    },
    BuiltinAdapterSpec {
        name: "groq",
        aliases: &["groq", "gq"],
        defaults: AdapterDefaults {
            base_url: Some("https://api.groq.com/openai/v1"),
            api_key_env: Some("GROQ_API_KEY"),
            default_model: Some("llama-3.1-70b-versatile"),
        },
    },
    BuiltinAdapterSpec {
        name: "cohere",
        aliases: &["cohere", "co"],
        defaults: AdapterDefaults {
            base_url: Some("https://api.cohere.com/v2"),
            api_key_env: Some("COHERE_API_KEY"),
            default_model: Some("command-r-plus"),
        },
    },
    BuiltinAdapterSpec {
        name: "fireworks",
        aliases: &["fireworks", "fw"],
        defaults: AdapterDefaults {
            base_url: Some("https://api.fireworks.ai/inference/v1"),
            api_key_env: Some("FIREWORKS_API_KEY"),
            default_model: Some("accounts/fireworks/models/llama-v3p1-70b-instruct"),
        },
    },
    BuiltinAdapterSpec {
        name: "together",
        aliases: &["together", "tg"],
        defaults: AdapterDefaults {
            base_url: Some("https://api.together.xyz/v1"),
            api_key_env: Some("TOGETHER_API_KEY"),
            default_model: Some("meta-llama/Llama-3.1-70B-Instruct-Turbo"),
        },
    },
    BuiltinAdapterSpec {
        name: "zai",
        aliases: &["zai", "zhipu", "zhi"],
        defaults: AdapterDefaults {
            base_url: Some("https://api.z.ai/api/paas/v4"),
            api_key_env: Some("ZAI_API_KEY"),
            default_model: Some("glm-4.5"),
        },
    },
    BuiltinAdapterSpec {
        name: "aihubmix",
        aliases: &["aihubmix", "ahm"],
        defaults: AdapterDefaults {
            base_url: Some("https://aihubmix.com/v1/"),
            api_key_env: Some("AIHUBMIX_API_KEY"),
            default_model: None,
        },
    },
    BuiltinAdapterSpec {
        name: "mimo",
        aliases: &["mimo"],
        defaults: AdapterDefaults {
            base_url: Some("https://api.xiaomimimo.com/v1/"),
            api_key_env: Some("MIMO_API_KEY"),
            default_model: None,
        },
    },
    BuiltinAdapterSpec {
        name: "moonshot",
        aliases: &["moonshot"],
        defaults: AdapterDefaults {
            base_url: Some("https://api.moonshot.cn/v1/"),
            api_key_env: Some("MOONSHOT_API_KEY"),
            default_model: None,
        },
    },
    BuiltinAdapterSpec {
        name: "nebius",
        aliases: &["nebius"],
        defaults: AdapterDefaults {
            base_url: Some("https://api.studio.nebius.ai/v1/"),
            api_key_env: Some("NEBIUS_API_KEY"),
            default_model: None,
        },
    },
    BuiltinAdapterSpec {
        name: "ollama_cloud",
        aliases: &["ollama_cloud", "ollama-cloud"],
        defaults: AdapterDefaults {
            base_url: Some("https://ollama.com/"),
            api_key_env: Some("OLLAMA_API_KEY"),
            default_model: None,
        },
    },
    BuiltinAdapterSpec {
        name: "vertex",
        aliases: &["vertex"],
        defaults: AdapterDefaults {
            base_url: None,
            api_key_env: Some("VERTEX_API_KEY"),
            default_model: None,
        },
    },
    BuiltinAdapterSpec {
        name: "github_copilot",
        aliases: &["github_copilot", "github-copilot"],
        defaults: AdapterDefaults {
            base_url: Some("https://models.github.ai/inference/"),
            api_key_env: Some("GITHUB_TOKEN"),
            default_model: None,
        },
    },
    BuiltinAdapterSpec {
        name: "opencode_go",
        aliases: &["opencode_go", "opencode-go"],
        defaults: AdapterDefaults {
            base_url: Some("https://opencode.ai/zen/go/v1/"),
            api_key_env: Some("OPENCODE_GO_API_KEY"),
            default_model: None,
        },
    },
    BuiltinAdapterSpec {
        name: "bedrock_api",
        aliases: &["bedrock_api", "bedrock-api"],
        defaults: AdapterDefaults {
            base_url: None,
            api_key_env: Some("BEDROCK_API_KEY"),
            default_model: None,
        },
    },
    BuiltinAdapterSpec {
        name: "open_router",
        aliases: &["open_router", "open-router", "openrouter"],
        defaults: AdapterDefaults {
            base_url: Some("https://openrouter.ai/api/v1/"),
            api_key_env: Some("OPEN_ROUTER_API_KEY"),
            default_model: None,
        },
    },
    BuiltinAdapterSpec {
        name: "minimax",
        aliases: &["minimax"],
        defaults: AdapterDefaults {
            base_url: Some("https://api.minimax.io/anthropic/v1/"),
            api_key_env: Some("MINIMAX_API_KEY"),
            default_model: None,
        },
    },
    BuiltinAdapterSpec {
        name: "baidu",
        aliases: &["baidu"],
        defaults: AdapterDefaults {
            base_url: Some("https://qianfan.baidubce.com/v2/"),
            api_key_env: Some("BAIDU_API_KEY"),
            default_model: None,
        },
    },
    BuiltinAdapterSpec {
        name: "bigmodel",
        aliases: &["bigmodel"],
        defaults: AdapterDefaults {
            base_url: Some("https://open.bigmodel.cn/api/paas/v4/"),
            api_key_env: Some("BIGMODEL_API_KEY"),
            default_model: None,
        },
    },
];

pub fn list_builtin_adapters() -> Vec<BuiltinAdapterInfo> {
    BUILTIN_ADAPTER_SPECS
        .iter()
        .map(|spec| BuiltinAdapterInfo {
            name: spec.name.to_string(),
            aliases: spec
                .aliases
                .iter()
                .filter(|alias| **alias != spec.name)
                .map(|alias| (*alias).to_string())
                .collect(),
            default_base_url: spec.defaults.base_url.map(str::to_string),
            api_key_env: spec.defaults.api_key_env.map(str::to_string),
            default_model: spec.defaults.default_model.map(str::to_string),
        })
        .collect()
}

pub fn is_builtin_adapter_name(adapter: &str) -> bool {
    adapter_spec(adapter).is_some()
}

pub fn resolve_runtime_config(
    mut app: AppConfigV2,
    args: &Args,
) -> Result<ResolvedRuntimeConfig, LlmProbeError> {
    ensure_profiles(&mut app, args);

    let selected_name = select_profile_name(&app, args)?;
    let mut profile =
        app.profiles.get(&selected_name).cloned().ok_or_else(|| {
            LlmProbeError::ConfigError(format!("Profile not found: {selected_name}"))
        })?;

    apply_cli_adapter_override(&mut profile, args);
    profile.adapter = normalize_adapter_name(&profile.adapter);
    let adapter_name = profile.adapter.clone();
    apply_adapter_defaults_into_profile(&mut profile, adapter_spec(&adapter_name), true);

    let adapter = normalize_adapter_name(&profile.adapter);
    let base_url = args
        .base_url
        .as_ref()
        .map(|v| normalize_base_url(v))
        .filter(|v| !v.is_empty())
        .or_else(|| {
            profile
                .base_url
                .as_ref()
                .map(|v| normalize_base_url(v))
                .filter(|v| !v.is_empty())
        });
    let model = args
        .model
        .clone()
        .or(profile.model.clone())
        .unwrap_or_default();

    let api_mode = args
        .api_mode
        .or(profile.api_mode)
        .or(app.defaults.api_mode)
        .unwrap_or(OpenAiApiMode::Auto);
    let (effective_model, api_mode_enforced) = resolve_effective_model(&adapter, &model, api_mode);

    let stream = if args.no_stream {
        false
    } else if args.stream {
        true
    } else {
        profile.stream.or(app.defaults.stream).unwrap_or(true)
    };
    let no_proxy = if args.no_proxy {
        true
    } else {
        profile.no_proxy.or(app.defaults.no_proxy).unwrap_or(false)
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
    if !args.prompt.is_empty() {
        context.push(Message {
            role: "user".to_string(),
            content: args.prompt.join(" "),
        });
    }
    let (context, system) = split_system_messages(context, None);

    let requested_reasoning = args
        .reasoning
        .clone()
        .or(profile.reasoning.clone())
        .or(profile.reasoning_effort.clone())
        .or(app.defaults.reasoning.clone());
    let capture_reasoning_default = app.defaults.capture_reasoning_content.unwrap_or(true);
    let reasoning_resolution = resolve_reasoning_controls(
        &adapter,
        requested_reasoning,
        capture_reasoning_default,
        profile.extra_body,
    )?;

    Ok(ResolvedRuntimeConfig {
        active_profile: selected_name,
        adapter,
        model,
        base_url,
        api_key,
        api_key_source,
        stream,
        context,
        max_tokens: profile.max_tokens,
        temperature: profile.temperature,
        top_p: profile.top_p,
        top_k: profile.top_k,
        system,
        timeout_seconds: profile.timeout_seconds.or(app.defaults.timeout_seconds),
        reasoning_effort: reasoning_resolution.effort,
        api_mode,
        api_mode_enforced,
        effective_model,
        no_proxy,
        reasoning_setting: reasoning_resolution.setting,
        capture_usage: app.defaults.capture_usage.unwrap_or(true),
        capture_reasoning_content: reasoning_resolution.capture_reasoning_content,
        normalize_reasoning_content: app.defaults.normalize_reasoning_content.unwrap_or(true),
        extra_body: reasoning_resolution.extra_body,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ReasoningControl {
    Off,
    Auto,
    Effort(String),
}

fn resolve_reasoning_controls(
    adapter: &str,
    requested: Option<String>,
    capture_default: bool,
    mut extra_body: HashMap<String, Value>,
) -> Result<ReasoningResolution, LlmProbeError> {
    let Some(raw) = requested else {
        return Ok(ReasoningResolution {
            setting: None,
            effort: None,
            capture_reasoning_content: capture_default,
            extra_body,
        });
    };

    let control = parse_reasoning_control(&raw)?;
    let setting_label = Some(raw.trim().to_string());

    match control {
        ReasoningControl::Off => {
            if adapter == "aliyun" || extra_body.contains_key("enable_thinking") {
                extra_body.insert("enable_thinking".to_string(), Value::Bool(false));
            }
            Ok(ReasoningResolution {
                setting: setting_label,
                effort: Some("none".to_string()),
                capture_reasoning_content: false,
                extra_body,
            })
        }
        ReasoningControl::Auto => Ok(ReasoningResolution {
            setting: setting_label,
            effort: None,
            capture_reasoning_content: true,
            extra_body,
        }),
        ReasoningControl::Effort(effort) => {
            if adapter == "aliyun" || extra_body.contains_key("enable_thinking") {
                extra_body.insert("enable_thinking".to_string(), Value::Bool(true));
            }
            Ok(ReasoningResolution {
                setting: setting_label,
                effort: Some(effort),
                capture_reasoning_content: true,
                extra_body,
            })
        }
    }
}

struct ReasoningResolution {
    setting: Option<String>,
    effort: Option<String>,
    capture_reasoning_content: bool,
    extra_body: HashMap<String, Value>,
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

fn resolve_effective_model(adapter: &str, model: &str, api_mode: OpenAiApiMode) -> (String, bool) {
    if model.trim().is_empty() {
        return (String::new(), false);
    }

    if let Some((namespace, _)) = model.split_once("::") {
        return (model.to_string(), namespace.starts_with("openai"));
    }

    let namespaced = match (adapter, api_mode) {
        ("openai", OpenAiApiMode::Responses) => format!("openai_resp::{model}"),
        ("openai", OpenAiApiMode::ChatCompletions) => format!("openai::{model}"),
        ("openai", OpenAiApiMode::Auto) => format!("openai::{model}"),
        ("aliyun", OpenAiApiMode::Responses) => format!("openai_resp::{model}"),
        ("aliyun", _) => format!("aliyun::{model}"),
        _ if is_builtin_adapter_name(adapter) => format!("{adapter}::{model}"),
        _ => model.to_string(),
    };
    let enforced = matches!(adapter, "openai" | "aliyun") && api_mode != OpenAiApiMode::Auto;
    (namespaced, enforced)
}

fn ensure_profiles(app: &mut AppConfigV2, args: &Args) {
    if !app.profiles.is_empty() {
        return;
    }

    let profile = ProviderProfile {
        adapter: args
            .adapter
            .as_deref()
            .map(normalize_adapter_name)
            .unwrap_or_else(|| "openai".to_string()),
        ..ProviderProfile::default()
    };
    app.profiles.insert("default".to_string(), profile);

    if app.active_profile.is_none() {
        app.active_profile = Some("default".to_string());
    }
}

fn apply_cli_adapter_override(profile: &mut ProviderProfile, args: &Args) {
    if let Some(adapter_arg) = args.adapter.as_deref() {
        profile.adapter = normalize_adapter_name(adapter_arg);
        profile.base_url = None;
        profile.api_key_env = None;
        profile.api_key = None;
        profile.api_mode = None;
        profile.extra_body.clear();
        if args.model.is_none() {
            profile.model = None;
        }
    }
}

fn apply_adapter_defaults_into_profile(
    profile: &mut ProviderProfile,
    spec: Option<&BuiltinAdapterSpec>,
    fill_model_when_missing: bool,
) {
    let Some(spec) = spec else {
        return;
    };

    if profile.adapter.trim().is_empty() {
        profile.adapter = spec.name.to_string();
    }
    if profile
        .base_url
        .as_deref()
        .unwrap_or_default()
        .trim()
        .is_empty()
    {
        profile.base_url = spec.defaults.base_url.map(str::to_string);
    }
    if profile
        .api_key_env
        .as_deref()
        .unwrap_or_default()
        .trim()
        .is_empty()
    {
        profile.api_key_env = spec.defaults.api_key_env.map(str::to_string);
    }
    if (fill_model_when_missing || profile.model.is_none())
        && profile
            .model
            .as_deref()
            .unwrap_or_default()
            .trim()
            .is_empty()
    {
        profile.model = spec.defaults.default_model.map(str::to_string);
    }
}

fn adapter_spec(alias: &str) -> Option<&'static BuiltinAdapterSpec> {
    let alias = alias.trim().to_lowercase();
    BUILTIN_ADAPTER_SPECS
        .iter()
        .find(|spec| spec.aliases.contains(&alias.as_str()) || spec.name == alias)
}

fn select_profile_name(app: &AppConfigV2, args: &Args) -> Result<String, LlmProbeError> {
    if let Some(name) = &args.profile {
        if app.profiles.contains_key(name) {
            return Ok(name.clone());
        }
        return Err(LlmProbeError::ConfigError(format!(
            "Profile not found: {name}"
        )));
    }

    if let Some(active) = &app.active_profile {
        if app.profiles.contains_key(active) {
            return Ok(active.clone());
        }
        return Err(LlmProbeError::ConfigError(format!(
            "active_profile points to a missing profile: {active}"
        )));
    }

    app.profiles
        .keys()
        .next()
        .cloned()
        .ok_or_else(|| LlmProbeError::ConfigError("No profiles are configured".to_string()))
}

fn normalize_base_url(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let Ok(mut parsed) = reqwest::Url::parse(trimmed) else {
        return trimmed.to_string();
    };
    let path = parsed.path().to_string();
    if !path.ends_with('/') {
        parsed.set_path(&format!("{path}/"));
    }
    parsed.to_string()
}

fn normalize_adapter_name(raw: &str) -> String {
    let normalized = raw.trim().to_lowercase();
    if let Some(spec) = adapter_spec(&normalized) {
        return spec.name.to_string();
    }
    match normalized.as_str() {
        "openairesp" | "openai_resp" | "openai-resp" => "openai".to_string(),
        other => other.to_string(),
    }
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
    adapter_spec(adapter)
        .and_then(|spec| spec.defaults.api_key_env)
        .map(str::to_string)
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

    #[test]
    fn resolve_uses_cli_overrides_and_profile_defaults() {
        let mut profiles = BTreeMap::new();
        profiles.insert(
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
            active_profile: Some("openai_main".to_string()),
            defaults: DefaultsConfig {
                stream: Some(true),
                ..DefaultsConfig::default()
            },
            profiles,
            context: vec![Message {
                role: "user".to_string(),
                content: "from-file".to_string(),
            }],
        };

        let mut input = args();
        input.model = Some("gpt-5".to_string());
        input.stream = true;
        input.message = vec!["from-cli".to_string()];
        input.prompt = vec!["tail".to_string(), "prompt".to_string()];
        input.api_mode = Some(OpenAiApiMode::Responses);

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert_eq!(resolved.model, "gpt-5");
        assert_eq!(resolved.effective_model, "openai_resp::gpt-5");
        assert!(resolved.stream);
        assert_eq!(resolved.context.len(), 3);
        assert_eq!(resolved.context[1].content, "from-cli");
        assert_eq!(resolved.context[2].content, "tail prompt");
        assert_eq!(resolved.api_mode, OpenAiApiMode::Responses);
        assert!(resolved.api_mode_enforced);
    }

    #[test]
    fn resolve_supports_profile_switch_via_profile_arg() {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "openai_main".to_string(),
            ProviderProfile {
                adapter: "openai".to_string(),
                model: Some("gpt-5".to_string()),
                ..ProviderProfile::default()
            },
        );
        profiles.insert(
            "anthropic_main".to_string(),
            ProviderProfile {
                adapter: "anthropic".to_string(),
                model: Some("claude-sonnet".to_string()),
                ..ProviderProfile::default()
            },
        );
        let app = AppConfigV2 {
            version: Some(2),
            active_profile: Some("openai_main".to_string()),
            defaults: DefaultsConfig::default(),
            profiles,
            context: Vec::new(),
        };

        let mut input = args();
        input.profile = Some("anthropic_main".to_string());

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert_eq!(resolved.active_profile, "anthropic_main");
        assert_eq!(resolved.adapter, "anthropic");
        assert_eq!(resolved.model, "claude-sonnet");
    }

    #[test]
    fn default_profile_uses_openai_adapter_defaults() {
        let app = AppConfigV2::default();
        let resolved = resolve_runtime_config(app, &args()).expect("resolve failed");

        assert_eq!(resolved.adapter, "openai");
        assert_eq!(resolved.model, "gpt-4o");
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("https://api.openai.com/v1/")
        );
    }

    #[test]
    fn adapter_alias_applies_builtin_defaults_for_quick_start() {
        let app = AppConfigV2::default();
        let mut input = args();
        input.adapter = Some("ds".to_string());

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert_eq!(resolved.adapter, "aliyun");
        assert_eq!(resolved.model, "qwen-max");
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("https://dashscope.aliyuncs.com/compatible-mode/v1/")
        );
    }

    #[test]
    fn unknown_adapter_keeps_raw_name() {
        let app = AppConfigV2::default();
        let mut input = args();
        input.adapter = Some("my_custom_adapter".to_string());

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert_eq!(resolved.adapter, "my_custom_adapter");
        assert_eq!(resolved.model, "");
    }

    #[test]
    fn adapter_arg_overrides_selected_profile_identity() {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "profile_a".to_string(),
            ProviderProfile {
                adapter: "openai".to_string(),
                model: Some("gpt-4o".to_string()),
                base_url: Some("https://api.openai.com/v1".to_string()),
                api_key_env: Some("OPENAI_API_KEY".to_string()),
                ..ProviderProfile::default()
            },
        );
        let app = AppConfigV2 {
            version: Some(2),
            active_profile: Some("profile_a".to_string()),
            defaults: DefaultsConfig::default(),
            profiles,
            context: Vec::new(),
        };

        let mut input = args();
        input.adapter = Some("aliyun".to_string());

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert_eq!(resolved.active_profile, "profile_a");
        assert_eq!(resolved.adapter, "aliyun");
        assert_eq!(resolved.model, "qwen-max");
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("https://dashscope.aliyuncs.com/compatible-mode/v1/")
        );
    }

    #[test]
    fn invalid_active_profile_is_rejected() {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "openai_main".to_string(),
            ProviderProfile {
                adapter: "openai".to_string(),
                model: Some("gpt-4o".to_string()),
                ..ProviderProfile::default()
            },
        );
        let app = AppConfigV2 {
            version: Some(2),
            active_profile: Some("missing_profile".to_string()),
            defaults: DefaultsConfig::default(),
            profiles,
            context: Vec::new(),
        };

        let err = resolve_runtime_config(app, &args()).unwrap_err();
        assert!(err
            .to_string()
            .contains("active_profile points to a missing profile"));
    }

    #[test]
    fn defaults_only_v2_file_keeps_defaults_semantics() {
        let app = AppConfigV2 {
            version: Some(2),
            active_profile: None,
            defaults: DefaultsConfig {
                stream: Some(false),
                no_proxy: Some(true),
                ..DefaultsConfig::default()
            },
            profiles: BTreeMap::new(),
            context: Vec::new(),
        };

        let resolved = resolve_runtime_config(app, &args()).expect("resolve failed");
        assert!(!resolved.stream);
        assert!(resolved.no_proxy);
        assert_eq!(resolved.api_mode, OpenAiApiMode::Auto);
    }

    #[test]
    fn builtin_adapter_list_contains_openai_and_aliyun() {
        let list = list_builtin_adapters();
        assert!(list.iter().any(|p| p.name == "openai"));
        assert!(list.iter().any(|p| p.name == "aliyun"));
        assert!(list.iter().any(|p| p.name == "open_router"));
        assert!(list.iter().any(|p| p.name == "github_copilot"));
        assert!(list.iter().any(|p| p.name == "bedrock_api"));
        assert!(list
            .iter()
            .any(|p| p.name == "aliyun" && p.aliases.iter().any(|a| a == "dashscope")));
    }

    #[test]
    fn new_builtin_adapter_uses_namespaced_effective_model() {
        let mut input = args();
        input.adapter = Some("openrouter".to_string());
        input.model = Some("openai/gpt-4.1-mini".to_string());

        let resolved =
            resolve_runtime_config(AppConfigV2::default(), &input).expect("resolve failed");
        assert_eq!(resolved.adapter, "open_router");
        assert_eq!(resolved.effective_model, "open_router::openai/gpt-4.1-mini");
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("https://openrouter.ai/api/v1/")
        );
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
    fn no_proxy_propagates_to_resolved_runtime() {
        let app = AppConfigV2::default();
        let mut input = args();
        input.no_proxy = true;

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert!(resolved.no_proxy);
    }

    #[test]
    fn base_url_is_normalized_to_directory_semantics() {
        let app = AppConfigV2::default();
        let mut input = args();
        input.base_url = Some("https://fastai.enncloud.cn/v1".to_string());

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("https://fastai.enncloud.cn/v1/")
        );
    }

    #[test]
    fn base_url_with_trailing_slash_is_preserved() {
        let app = AppConfigV2::default();
        let mut input = args();
        input.base_url = Some("https://fastai.enncloud.cn/v1/".to_string());

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("https://fastai.enncloud.cn/v1/")
        );
    }

    #[test]
    fn no_proxy_can_come_from_defaults() {
        let app = AppConfigV2 {
            version: Some(2),
            active_profile: None,
            defaults: DefaultsConfig {
                no_proxy: Some(true),
                ..DefaultsConfig::default()
            },
            profiles: BTreeMap::new(),
            context: Vec::new(),
        };

        let resolved = resolve_runtime_config(app, &args()).expect("resolve failed");
        assert!(resolved.no_proxy);
    }

    #[test]
    fn no_proxy_profile_overrides_defaults() {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "openai_main".to_string(),
            ProviderProfile {
                adapter: "openai".to_string(),
                model: Some("gpt-4o".to_string()),
                no_proxy: Some(false),
                ..ProviderProfile::default()
            },
        );
        let app = AppConfigV2 {
            version: Some(2),
            active_profile: Some("openai_main".to_string()),
            defaults: DefaultsConfig {
                no_proxy: Some(true),
                ..DefaultsConfig::default()
            },
            profiles,
            context: Vec::new(),
        };

        let resolved = resolve_runtime_config(app, &args()).expect("resolve failed");
        assert!(!resolved.no_proxy);
    }

    #[test]
    fn cli_no_proxy_overrides_profile_setting() {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "openai_main".to_string(),
            ProviderProfile {
                adapter: "openai".to_string(),
                model: Some("gpt-4o".to_string()),
                no_proxy: Some(false),
                ..ProviderProfile::default()
            },
        );
        let app = AppConfigV2 {
            version: Some(2),
            active_profile: Some("openai_main".to_string()),
            defaults: DefaultsConfig::default(),
            profiles,
            context: Vec::new(),
        };
        let mut input = args();
        input.no_proxy = true;

        let resolved = resolve_runtime_config(app, &input).expect("resolve failed");
        assert!(resolved.no_proxy);
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
        let mut profiles = BTreeMap::new();
        let mut profile = ProviderProfile {
            adapter: "aliyun".to_string(),
            model: Some("glm-5".to_string()),
            ..ProviderProfile::default()
        };
        profile
            .extra_body
            .insert("enable_thinking".to_string(), serde_json::json!(true));
        profiles.insert("ali".to_string(), profile);

        let app = AppConfigV2 {
            version: Some(2),
            active_profile: Some("ali".to_string()),
            defaults: DefaultsConfig::default(),
            profiles,
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
    fn reasoning_off_injects_enable_thinking_false_for_aliyun() {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "ali".to_string(),
            ProviderProfile {
                adapter: "aliyun".to_string(),
                model: Some("glm-5".to_string()),
                ..ProviderProfile::default()
            },
        );

        let app = AppConfigV2 {
            version: Some(2),
            active_profile: Some("ali".to_string()),
            defaults: DefaultsConfig::default(),
            profiles,
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
