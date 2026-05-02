use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileConfig {
    pub provider: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub stream: Option<bool>,
    pub context: Option<Vec<Message>>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    #[serde(alias = "enable_thinking")]
    pub reasoning: Option<bool>,
    pub reasoning_effort: Option<String>,
    #[serde(alias = "thinking_budget_tokens")]
    pub reasoning_budget_tokens: Option<u32>,
    #[serde(default)]
    pub extra_body: HashMap<String, serde_json::Value>,
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
pub struct DefaultsConfig {
    pub stream: Option<bool>,
    pub timeout_seconds: Option<u64>,
    pub capture_usage: Option<bool>,
    pub capture_reasoning_content: Option<bool>,
    pub normalize_reasoning_content: Option<bool>,
    pub openai_api: Option<OpenAiApiMode>,
    /// Unified reasoning control:
    /// off | auto | low | medium | high | xhigh | max | budget:<n>
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderProfile {
    pub adapter: String,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub stream: Option<bool>,
    pub timeout_seconds: Option<u64>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    /// Unified reasoning control:
    /// off | auto | low | medium | high | xhigh | max | budget:<n>
    pub reasoning: Option<String>,
    /// Backward-compatible alias for old configs.
    pub reasoning_effort: Option<String>,
    pub openai_api: Option<OpenAiApiMode>,
    #[serde(default)]
    pub extra_body: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub context: Vec<Message>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppConfigV2 {
    pub version: Option<u32>,
    pub active_provider: Option<String>,
    #[serde(default)]
    pub defaults: DefaultsConfig,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderProfile>,
    #[serde(default)]
    pub context: Vec<Message>,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub provider: String,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
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
    pub extra_body: HashMap<String, serde_json::Value>,
}

impl RuntimeConfig {
    pub fn new() -> Self {
        Self {
            provider: String::new(),
            base_url: String::new(),
            api_key: String::new(),
            model: String::new(),
            stream: false,
            context: Vec::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            system: None,
            timeout_seconds: None,
            reasoning: None,
            reasoning_effort: None,
            reasoning_budget_tokens: None,
            extra_body: HashMap::new(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "llmctl")]
#[command(version = "1.0.1")]
#[command(disable_version_flag = true)]
#[command(about = "A CLI tool for testing and validating LLM services", long_about = None)]
pub struct Args {
    #[arg(
        short = 'c',
        long,
        value_name = "PATH",
        help = "Config file path (YAML or JSON)"
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

    #[arg(long, help = "List built-in provider presets and exit")]
    pub list_presets: bool,

    #[arg(
        long,
        value_name = "STRING",
        help = "Append user message to context (can be used multiple times)"
    )]
    pub message: Vec<String>,

    #[arg(
        short,
        long,
        value_name = "STRING",
        help = "Provider adapter or alias: openai, dashscope/aliyun, anthropic/claude, gemini/google, ollama, deepseek, groq, mistral"
    )]
    pub provider: Option<String>,

    #[arg(
        short = 'P',
        long,
        value_name = "NAME",
        help = "Provider profile name from config v2 (providers.<name>)"
    )]
    pub profile: Option<String>,

    #[arg(
        short,
        long,
        value_name = "STRING",
        help = "API base URL (overrides provider default)"
    )]
    pub url: Option<String>,

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
        value_name = "INPUT",
        help = "Convert config file format (input file, optional output file)"
    )]
    pub convert: Option<Vec<PathBuf>>,

    #[arg(
        long = "endpoint",
        value_enum,
        help = "OpenAI API endpoint mode: auto, responses, chat-completions (alias: chat_completions)"
    )]
    pub endpoint: Option<OpenAiApiMode>,

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

    #[arg(
        long,
        help = "Allow OpenAI API mode to fall back to SDK default when endpoint mode cannot be enforced"
    )]
    pub allow_sdk_default_api: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_v2_multi_provider_yaml() {
        let yaml = r#"
version: 2
active_provider: openai_main
providers:
  openai_main:
    adapter: openai
    model: gpt-5
    api_key_env: OPENAI_API_KEY
    openai_api: auto
  anthropic_main:
    adapter: anthropic
    model: claude-sonnet-4-5
    api_key_env: ANTHROPIC_API_KEY
defaults:
  stream: true
  capture_usage: true
context:
  - role: user
    content: hello
"#;

        let parsed: AppConfigV2 = serde_yaml::from_str(yaml).expect("failed to parse v2 config");
        assert_eq!(parsed.version, Some(2));
        assert_eq!(parsed.active_provider.as_deref(), Some("openai_main"));
        assert_eq!(parsed.providers.len(), 2);
        assert_eq!(
            parsed
                .providers
                .get("openai_main")
                .and_then(|p| p.model.as_deref()),
            Some("gpt-5")
        );
        assert_eq!(parsed.defaults.stream, Some(true));
        assert_eq!(parsed.context.len(), 1);
    }
}
