use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiApiMode {
    Auto,
    Responses,
    #[serde(alias = "chat-completions")]
    #[value(alias = "chat_completions", alias = "chat-completions")]
    ChatCompletions,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct DefaultsConfig {
    pub stream: Option<bool>,
    pub no_proxy: Option<bool>,
    pub timeout_seconds: Option<u64>,
    pub capture_usage: Option<bool>,
    pub capture_reasoning_content: Option<bool>,
    pub normalize_reasoning_content: Option<bool>,
    #[serde(alias = "openai_api")]
    pub api_mode: Option<OpenAiApiMode>,
    /// Unified reasoning control:
    /// off | auto | low | medium | high | xhigh | max | budget:<n>
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ProviderProfile {
    pub adapter: String,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub stream: Option<bool>,
    pub no_proxy: Option<bool>,
    pub timeout_seconds: Option<u64>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    /// Unified reasoning control:
    /// off | auto | low | medium | high | xhigh | max | budget:<n>
    pub reasoning: Option<String>,
    /// Deprecated alias kept within the v2 profile schema.
    pub reasoning_effort: Option<String>,
    #[serde(alias = "openai_api")]
    pub api_mode: Option<OpenAiApiMode>,
    #[serde(default)]
    pub extra_body: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub context: Vec<Message>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct AppConfigV2 {
    pub version: Option<u32>,
    #[serde(alias = "active_provider")]
    pub active_profile: Option<String>,
    #[serde(default)]
    pub defaults: DefaultsConfig,
    #[serde(default, alias = "providers")]
    pub profiles: BTreeMap<String, ProviderProfile>,
    #[serde(default)]
    pub context: Vec<Message>,
}

#[derive(Parser, Debug)]
#[command(name = "llmctl")]
#[command(version)]
#[command(disable_version_flag = true)]
#[command(about = "A CLI tool for testing and validating LLM services", long_about = None)]
pub struct Args {
    #[arg(
        short = 'c',
        long,
        value_name = "PATH",
        help = "Config file path (v2 YAML or JSON)"
    )]
    pub config: Option<PathBuf>,

    #[arg(
        short,
        long,
        value_name = "STRING",
        help = "Model name (e.g., gpt-4o, claude-3-opus)"
    )]
    pub model: Option<String>,

    #[arg(short, long, help = "List available models from provider")]
    pub list: bool,

    #[arg(
        long = "list-adapters",
        alias = "list-presets",
        help = "List supported adapters, aliases, and built-in defaults"
    )]
    pub list_adapters: bool,

    #[arg(
        long,
        value_name = "STRING",
        help = "Append user message to context (can be used multiple times)"
    )]
    pub message: Vec<String>,

    #[arg(
        short = 'p',
        long,
        value_name = "STRING",
        alias = "provider",
        help = "Adapter name or alias: openai/oi, aliyun/ali/dashscope, anthropic/claude, gemini/google, ollama, deepseek, xai/grok, groq, cohere, fireworks, together, zai/zhipu"
    )]
    pub adapter: Option<String>,

    #[arg(
        short = 'P',
        long,
        value_name = "NAME",
        help = "Profile name from config v2 (profiles.<name>)"
    )]
    pub profile: Option<String>,

    #[arg(
        short = 'u',
        long = "base-url",
        value_name = "STRING",
        alias = "url",
        help = "API base URL (overrides profile and adapter default)"
    )]
    pub base_url: Option<String>,

    #[arg(
        short,
        long,
        value_name = "STRING",
        help = "API Key (or set LLM_API_KEY env var)"
    )]
    pub secret: Option<String>,

    #[arg(
        short,
        long,
        value_name = "STRING",
        help = "API Key (alias for --secret, or set LLM_API_KEY env var)"
    )]
    pub key: Option<String>,

    #[arg(long, help = "Enable streaming response")]
    pub stream: bool,

    #[arg(
        long = "no-stream",
        help = "Disable streaming response for this run",
        conflicts_with = "stream"
    )]
    pub no_stream: bool,

    #[arg(
        long = "no-proxy",
        help = "Disable all proxies for this run (overrides config and reqwest/system proxy settings)"
    )]
    pub no_proxy: bool,

    #[arg(short, long, help = "Show version information")]
    pub version: bool,

    #[arg(
        short,
        long,
        value_name = "FORMAT",
        help = "Initialize config file: yaml, json, or custom filename (e.g., myconfig.yaml)"
    )]
    pub init: Option<String>,

    #[arg(
        long,
        value_name = "PATH",
        help = "Custom config file path for initialization"
    )]
    pub init_path: Option<PathBuf>,

    #[arg(
        short = 't',
        long,
        num_args = 1..=2,
        value_name = "INPUT",
        help = "Convert v2 config between YAML and JSON (input file, optional output file)"
    )]
    pub convert: Option<Vec<PathBuf>>,

    #[arg(
        long = "api-mode",
        value_enum,
        alias = "endpoint",
        help = "API mode for OpenAI-compatible adapters: auto, responses, chat-completions (alias: chat_completions)"
    )]
    pub api_mode: Option<OpenAiApiMode>,

    #[arg(
        long = "reasoning",
        value_name = "MODE",
        help = "Unified reasoning control: off|auto|low|medium|high|xhigh|max|budget:<n>"
    )]
    pub reasoning: Option<String>,

    #[arg(long, help = "Print resolved runtime plan without sending requests")]
    pub dry_run: bool,

    #[arg(long, help = "Validate configuration and print diagnostics")]
    pub doctor_config: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_v2_multi_provider_yaml() {
        let yaml = r#"
version: 2
active_profile: openai_main
profiles:
  openai_main:
    adapter: openai
    model: gpt-5
    api_key_env: OPENAI_API_KEY
    api_mode: auto
  anthropic_main:
    adapter: anthropic
    model: claude-sonnet-4-5
    api_key_env: ANTHROPIC_API_KEY
defaults:
  stream: true
  no_proxy: true
  capture_usage: true
context:
  - role: user
    content: hello
"#;

        let parsed: AppConfigV2 = serde_yaml::from_str(yaml).expect("failed to parse v2 config");
        assert_eq!(parsed.version, Some(2));
        assert_eq!(parsed.active_profile.as_deref(), Some("openai_main"));
        assert_eq!(parsed.profiles.len(), 2);
        assert_eq!(
            parsed
                .profiles
                .get("openai_main")
                .and_then(|p| p.model.as_deref()),
            Some("gpt-5")
        );
        assert_eq!(parsed.defaults.stream, Some(true));
        assert_eq!(parsed.defaults.no_proxy, Some(true));
        assert_eq!(parsed.context.len(), 1);
    }

    #[test]
    fn parse_convert_accepts_optional_output_path() {
        let args = Args::try_parse_from(["llmctl", "--convert", "in.yaml", "out.json"])
            .expect("convert should parse");
        assert_eq!(
            args.convert,
            Some(vec![PathBuf::from("in.yaml"), PathBuf::from("out.json")])
        );
    }

    #[test]
    fn parses_legacy_v2_alias_keys() {
        let yaml = r#"
version: 2
active_provider: openai_main
providers:
  openai_main:
    adapter: openai
    model: gpt-5
    openai_api: responses
"#;

        let parsed: AppConfigV2 = serde_yaml::from_str(yaml).expect("failed to parse aliased v2");
        assert_eq!(parsed.active_profile.as_deref(), Some("openai_main"));
        assert_eq!(parsed.profiles.len(), 1);
        assert_eq!(
            parsed.profiles.get("openai_main").and_then(|p| p.api_mode),
            Some(OpenAiApiMode::Responses)
        );
    }
}
