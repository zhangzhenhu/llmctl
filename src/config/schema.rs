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
    // Builder 配置
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    // Reasoning 配置
    pub reasoning: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub reasoning_budget_tokens: Option<u32>,
    // 自定义 extra_body
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
    // Builder 配置
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    // Reasoning 配置
    pub reasoning: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub reasoning_budget_tokens: Option<u32>,
    // 自定义 extra_body
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
#[command(version = "0.1.0")]
#[command(about = "LLM 服务验证 CLI 工具", long_about = None)]
pub struct Args {
    #[arg(short = 'c', long, value_name = "PATH", help = "指定配置文件路径")]
    pub config: Option<PathBuf>,

    #[arg(short, long, value_name = "STRING", help = "指定模型名称")]
    pub model: Option<String>,

    #[arg(short, long, help = "列出当前服务商支持的所有模型")]
    pub list: bool,

    #[arg(long, value_name = "STRING", help = "追加用户消息(可多次使用)")]
    pub message: Vec<String>,

    #[arg(short, long, value_name = "STRING", help = "指定模型服务商")]
    pub provider: Option<String>,

    #[arg(short, long, value_name = "STRING", help = "指定 API base_url")]
    pub url: Option<String>,

    #[arg(short, long, value_name = "STRING", help = "指定 API Key")]
    pub secret: Option<String>,

    #[arg(long, help = "启用流式返回")]
    pub stream: bool,

    #[arg(
        short,
        long,
        value_name = "FORMAT",
        help = "初始化配置文件格式(yaml/json)"
    )]
    pub init: Option<String>,

    #[arg(long, value_name = "PATH", help = "自定义初始化配置文件路径")]
    pub init_path: Option<PathBuf>,

    #[arg(short = 't', long, value_name = "INPUT", help = "配置文件格式转换")]
    pub convert: Option<Vec<PathBuf>>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::new()
    }
}
