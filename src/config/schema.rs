use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
    pub reasoning: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub reasoning_budget_tokens: Option<u32>,
    #[serde(default)]
    pub extra_body: HashMap<String, serde_json::Value>,
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
#[command(version = "0.1.4")]
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
        help = "Model provider: openai, google/gemini, anthropic/claude, ollama, deepseek, xai, groq, mistral, openai-compatible"
    )]
    pub provider: Option<String>,

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

    #[arg(long, help = "Enable streaming response")]
    pub stream: bool,

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
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::new()
    }
}
