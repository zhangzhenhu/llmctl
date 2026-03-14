// use crate::backends::OpenAI;
// use crate::backends::openai::OpenAI;
use crate::backends::openai_reasoning::OpenAIWithReasoning;
// use crate::builder::{LLMBackend, LLMBuilder};
use crate::builder::{LLMBackend, LLMBuilder};
use crate::config::schema::Message;
use crate::error::LlmProbeError;
use futures::StreamExt;
// use llm::builder::LLMBackend;
use llm::chat::{
    ChatMessage,
    ChatMessageBuilder,
    ChatRole,
    ReasoningEffort,
    // Usage
};
use llm::models::ModelListRequest;
use llm::LLMProvider;
use reqwest;
use serde::{Deserialize, Serialize};
// use std::io::Write;
use std::time::Instant;
// #[derive(Debug, Clone)]
pub struct ChatResponse {
    pub provider: String,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub model: String,
    pub duration_ms: u64,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub provider: String,
}

pub enum LLMBackendEnum {
    Standard(Box<dyn LLMProvider>),
    WithReasoning(Box<dyn OpenAIWithReasoning>),
}
pub struct LLMClient {
    llm: LLMBackendEnum,
    // llm_x: Option<Box<dyn OpenAIWithReasoning>>,
    base_url: String,
    api_key: String,
    provider_name: String,
    is_openai_compatible: bool,
}

fn build_chat_message(msg: Message) -> ChatMessage {
    let role = msg.role.as_str();
    let content = msg.content;

    let chat_role = match role {
        "system" => ChatRole::User,
        "assistant" => ChatRole::Assistant,
        _ => ChatRole::User,
    };

    ChatMessageBuilder::new(chat_role).content(content).build()
}

impl LLMClient {
    // pub async fn chat_completion(
    //     &self,
    //     messages: Vec<Message>,
    //     model: &str,
    // ) -> Result<ChatResponse, LlmProbeError> {
    //     let llm: = match &self.llm {
    //         LLMBackendEnum::Standard(llm) => llm,
    //         LLMBackendEnum::WithReasoning(llm) => llm,
    //     };
    //     // 假设 OpenAIWithReasoning 也实现了 LLMProvider
    //     // 或者提供类似的 chat 方法
    //     return self
    //         .chat_completion_standard(llm.as_ref(), messages, model)
    //         .await;
    // }
    pub async fn chat_completion(
        &self,
        messages: Vec<Message>,
        model: &str,
    ) -> Result<ChatResponse, LlmProbeError> {
        let start = Instant::now();

        let chat_messages: Vec<ChatMessage> =
            messages.into_iter().map(build_chat_message).collect();
        let response = match &self.llm {
            LLMBackendEnum::Standard(llm) => llm.chat(&chat_messages).await,
            LLMBackendEnum::WithReasoning(llm) => llm.chat(&chat_messages).await,
        };
        // match llm.chat(&chat_messages).await {
        match response {
            Ok(response) => {
                // let content = response.text().unwrap_or_default().to_string();
                // let reasoning_content = response.thinking().unwrap_or_default().to_string();

                let (input_tokens, output_tokens) = if let Some(usage) = response.usage() {
                    (Some(usage.prompt_tokens), Some(usage.completion_tokens))
                } else {
                    (None, None)
                };

                let duration_ms = start.elapsed().as_millis() as u64;

                Ok(ChatResponse {
                    provider: self.provider_name.clone(),
                    content: response.text(),
                    reasoning_content: response.thinking(),
                    model: model.to_string(),
                    duration_ms,
                    input_tokens,
                    output_tokens,
                })
            }
            Err(e) => Err(map_llm_error(&e.to_string())),
        }
    }
    pub async fn chat_completion_standard(
        &self,
        llm: &dyn LLMProvider,
        messages: Vec<Message>,
        model: &str,
    ) -> Result<ChatResponse, LlmProbeError> {
        let start = Instant::now();

        let chat_messages: Vec<ChatMessage> =
            messages.into_iter().map(build_chat_message).collect();

        match llm.chat(&chat_messages).await {
            Ok(response) => {
                // let content = response.text().unwrap_or_default().to_string();

                let (input_tokens, output_tokens) = if let Some(usage) = response.usage() {
                    (Some(usage.prompt_tokens), Some(usage.completion_tokens))
                } else {
                    (None, None)
                };

                let duration_ms = start.elapsed().as_millis() as u64;

                Ok(ChatResponse {
                    provider: self.provider_name.clone(),
                    content: response.text(),
                    reasoning_content: response.thinking(),
                    model: model.to_string(),
                    duration_ms,
                    input_tokens,
                    output_tokens,
                })
            }
            Err(e) => Err(map_llm_error(&e.to_string())),
        }
    }
    pub async fn stream_chat(
        &self,
        messages: Vec<Message>,
        model: &str,
    ) -> Result<(), LlmProbeError> {
        match &self.llm {
            LLMBackendEnum::Standard(llm) => {
                return self.stream_chat_old(llm.as_ref(), messages, model).await;
            }
            LLMBackendEnum::WithReasoning(llm) => {
                return self
                    .stream_chat_with_reasoning(llm.as_ref(), messages, model)
                    .await;
            }
        }
        // if self.is_openai_compatible {
        //     return self.stream_chat_with_reasoning(messages, model).await;
        // }
        // return self.stream_chat_old(messages, model).await;
    }

    pub async fn stream_chat_with_reasoning(
        &self,
        llm: &dyn OpenAIWithReasoning,
        messages: Vec<Message>,
        model: &str,
    ) -> Result<(), LlmProbeError> {
        use colored::*;
        use std::io::Write;

        let start = Instant::now();
        let mut content = String::new();
        let mut reasoning_content = String::new();
        let mut in_reasoning = false;

        let chat_messages: Vec<ChatMessage> =
            messages.into_iter().map(build_chat_message).collect();
        let stream = match llm.chat_stream_struct_with_reasoning(&chat_messages).await {
            Ok(s) => s,
            Err(e) => return Err(map_llm_error(&e.to_string())),
        };

        // let stream = match &self.llm {
        //     LLMBackendEnum::WithReasoning(llm) => {
        //         match llm.chat_stream_struct_with_reasoning(&chat_messages).await {
        //             Ok(s) => s,
        //             Err(e) => return Err(map_llm_error(&e.to_string())),
        //         }
        //     }
        //     _ => return Err(LlmProbeError::RuntimeError("llm_x not inited".to_string())),
        //     // else => return Err(LlmProbeError::RuntimeError("llm_x not inited".to_string())),
        // };

        futures::pin_mut!(stream);
        // let mut usage: Option<Usage> = None;
        let mut tokens_input: u32 = 0;
        let mut tokens_output: u32 = 0;
        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    let choice = chunk.choices.first();
                    if let Some(choice) = choice {
                        // 先流式输出思考过程
                        if let Some(text) = &choice.delta.reasoning_content {
                            if !in_reasoning {
                                println!("{}:", "思考过程".cyan());
                                println!("{}", "─".repeat(50).dimmed());
                                in_reasoning = true;
                            }
                            print!("{}", text);
                            std::io::stdout().flush().ok();
                            reasoning_content.push_str(text);
                        }
                        // 再流式输出最终内容
                        if let Some(text) = &choice.delta.content {
                            if in_reasoning {
                                println!("\n{}", "─".repeat(50).dimmed());
                                println!("{}:", "回答".cyan());
                                println!("{}", "─".repeat(50).dimmed());
                                in_reasoning = false;
                            }
                            print!("{}", text);
                            std::io::stdout().flush().ok();
                            content.push_str(text);
                        }
                    }
                    if let Some(s) = chunk.usage {
                        // usage = chunk.usage;
                        tokens_input += s.prompt_tokens;
                        tokens_output += s.completion_tokens;
                    }
                }
                Err(e) => {
                    return Err(map_llm_error(&e.to_string()));
                }
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        // 打印最终汇总信息
        println!("");
        println!("{}", "─".repeat(50).dimmed());
        // match usage {
        // Some(s) => {
        println!(
            "{}: 输入 {}, 输出 {}",
            "Token".dimmed(),
            tokens_input,
            tokens_output
        );
        // }
        // None => {}
        // }

        println!(
            "{}: ({}){}",
            "模型".green(),
            self.provider_name.green(),
            model.green()
        );
        println!("{}: {} ms", "耗时".yellow(), duration_ms);

        Ok(())
    }
    pub async fn stream_chat_old(
        &self,
        llm: &dyn LLMProvider,
        messages: Vec<Message>,
        model: &str,
    ) -> Result<(), LlmProbeError> {
        use colored::*;
        use std::io::Write;

        let start = Instant::now();
        let mut content = String::new();

        let chat_messages: Vec<ChatMessage> =
            messages.into_iter().map(build_chat_message).collect();
        match llm.chat_stream(&chat_messages).await {
            Ok(mut stream) => {
                while let Some(token_result) = stream.next().await {
                    match token_result {
                        Ok(token) => {
                            print!("{}", token);
                            std::io::stdout().flush().ok();
                            content.push_str(&token);
                        }
                        Err(e) => {
                            return Err(map_llm_error(&e.to_string()));
                        }
                    }
                }
            }
            Err(e) => return Err(map_llm_error(&e.to_string())),
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        // 打印最终汇总信息
        println!("");
        println!("{}", "─".repeat(50).dimmed());
        println!(
            "{}: ({}){}",
            "模型".green(),
            self.provider_name.green(),
            model.green()
        );
        println!("{}: {} ms", "耗时".yellow(), duration_ms);
        println!("{}", "─".repeat(50).dimmed());

        Ok(())
    }
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmProbeError> {
        let request: Option<&ModelListRequest> = None;
        // 直接在 match 中调用
        let response = match &self.llm {
            LLMBackendEnum::Standard(llm) => llm.list_models(request).await,
            LLMBackendEnum::WithReasoning(llm_x) => llm_x.list_models(request).await,
        };
        match response {
            Ok(response) => {
                let backend = response.get_backend();
                let provider_name = match backend {
                    llm::builder::LLMBackend::OpenAI => "openai",
                    llm::builder::LLMBackend::Google => "google",
                    llm::builder::LLMBackend::Anthropic => "anthropic",
                    llm::builder::LLMBackend::Ollama => "ollama",
                    llm::builder::LLMBackend::DeepSeek => "deepseek",
                    llm::builder::LLMBackend::XAI => "xai",
                    llm::builder::LLMBackend::Phind => "phind",
                    llm::builder::LLMBackend::Groq => "groq",
                    llm::builder::LLMBackend::Mistral => "mistral",
                    llm::builder::LLMBackend::ElevenLabs => "elevenlabs",
                    _ => "unknown",
                };
                let models: Vec<ModelInfo> = response
                    .get_models()
                    .iter()
                    .map(|id| ModelInfo {
                        id: id.clone(),
                        name: id.clone(),
                        provider: provider_name.to_string(),
                    })
                    .collect();
                return Ok(models);
            }
            Err(e) => {
                eprintln!("llm.list_models failed: {}, falling back to reqwest", e);
            }
        }

        let client = reqwest::Client::new();
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));

        match client.get(&url).bearer_auth(&self.api_key).send().await {
            Ok(resp) => {
                if !resp.status().is_success() {
                    return Err(LlmProbeError::ApiError(format!("HTTP {}", resp.status())));
                }

                #[derive(Deserialize)]
                struct ModelsResponse {
                    data: Vec<ModelData>,
                }

                #[derive(Deserialize)]
                struct ModelData {
                    id: String,
                }

                match resp.json::<ModelsResponse>().await {
                    Ok(models_resp) => {
                        let models: Vec<ModelInfo> = models_resp
                            .data
                            .into_iter()
                            .map(|m| ModelInfo {
                                id: m.id.clone(),
                                name: m.id,
                                provider: self.provider_name.clone(),
                            })
                            .collect();
                        Ok(models)
                    }
                    Err(e) => Err(LlmProbeError::ApiError(format!("解析模型列表失败: {}", e))),
                }
            }
            Err(e) => Err(map_llm_error(&e.to_string())),
        }
    }
}

fn map_llm_error(error: &str) -> LlmProbeError {
    let lower = error.to_lowercase();

    if lower.contains("could not resolve host")
        || lower.contains("connection refused")
        || lower.contains("connection timed out")
        || lower.contains("connect timed out")
        || lower.contains("network unreachable")
        || lower.contains("no route to host")
        || lower.contains("error sending request")
        || lower.contains("error during request")
        || lower.contains("dns")
        || (lower.contains("network") && !lower.contains("api"))
        || (lower.contains("connection") && !lower.contains("api"))
        || lower.contains("timeout")
    {
        let friendly_msg = if lower.contains("could not resolve host") {
            "DNS 解析失败，请检查 API 地址是否正确".to_string()
        } else if lower.contains("connection refused") {
            "连接被拒绝，请检查 API 地址是否正确".to_string()
        } else if lower.contains("timed out") || lower.contains("timeout") {
            "请求超时，请检查网络连接或 API 地址".to_string()
        } else if lower.contains("could not resolve") {
            "无法解析域名，请检查 API 地址是否正确".to_string()
        } else {
            format!("网络错误: {}", error)
        };
        return LlmProbeError::ApiError(friendly_msg);
    }

    if lower.contains("api key") || lower.contains("unauthorized") || lower.contains("401") {
        LlmProbeError::InvalidApiKey
    } else if lower.contains("rate limit") || lower.contains("429") {
        LlmProbeError::RateLimitError
    } else if lower.contains("model") && (lower.contains("not found") || lower.contains("404")) {
        LlmProbeError::ModelNotFound
    } else if lower.contains("500") || lower.contains("502") || lower.contains("503") {
        LlmProbeError::ServerError
    } else {
        LlmProbeError::ApiError(error.to_string())
    }
}

use crate::config::schema::RuntimeConfig;
// use std::collections::HashMap;

pub fn create_llm_backend(
    provider: &str,
    api_key: &str,
    base_url: Option<&str>,
    model: &str,
    config: Option<&RuntimeConfig>,
) -> Result<LLMClient, LlmProbeError> {
    let provider_lower = provider.to_lowercase();

    let default_url = match provider_lower.as_str() {
        "openai" => "https://api.openai.com/v1",
        "gemini" | "google" => "https://generativelanguage.googleapis.com/v1beta",
        "anthropic" | "claude" => "https://api.anthropic.com",
        "ollama" => "http://localhost:11434",
        "deepseek" => "https://api.deepseek.com/v1",
        "xai" => "https://api.x.ai/v1",
        "phind" => "https://api.phind.com",
        "groq" => "https://api.groq.com/openai/v1",
        "mistral" => "https://api.mistral.ai/v1",
        "elevenlabs" => "https://api.elevenlabs.io/v1",
        "openai_compatible" | "openai-compatible" | "aliyun" | "dashscope" => {
            "https://api.openai.com/v1"
        }
        _ => "https://api.openai.com/v1",
    };

    let final_url = base_url.unwrap_or(default_url);

    let backend = match provider_lower.as_str() {
        "openai" => LLMBackend::OpenAI,
        "gemini" | "google" => LLMBackend::Google,
        "anthropic" | "claude" => LLMBackend::Anthropic,
        "ollama" => LLMBackend::Ollama,
        "deepseek" => LLMBackend::DeepSeek,
        "xai" => LLMBackend::XAI,
        "phind" => LLMBackend::Phind,
        "groq" => LLMBackend::Groq,
        "mistral" => LLMBackend::Mistral,
        "elevenlabs" => LLMBackend::ElevenLabs,
        _ => LLMBackend::OpenAI,
        // _ => return Err(LlmProbeError::UnsupportedProvider(provider.to_string())),
    };
    let provider_name = provider_lower.clone();
    let is_openai_compatible = provider_lower == "openai-compatible";
    let mut builder = LLMBuilder::new()
        .backend(backend.clone())
        .api_key(api_key)
        .model(model);

    if let Some(cfg) = config {
        if let Some(timeout) = cfg.timeout_seconds {
            builder = builder.timeout_seconds(timeout);
        }
        if let Some(temp) = cfg.temperature {
            builder = builder.temperature(temp);
        }
        if let Some(tp) = cfg.top_p {
            builder = builder.top_p(tp);
        }
        if let Some(tk) = cfg.top_k {
            builder = builder.top_k(tk);
        }
        if let Some(mt) = cfg.max_tokens {
            builder = builder.max_tokens(mt);
        }
        if let Some(system) = &cfg.system {
            builder = builder.system(system);
        }
        if !cfg.extra_body.is_empty() {
            builder = builder.extra_body(&cfg.extra_body);
        }
        if let Some(v) = cfg.reasoning_budget_tokens {
            builder = builder.reasoning_budget_tokens(v);
        }
        if let Some(v) = cfg.reasoning {
            builder = builder.reasoning(v);
        }
        if let Some(v) = &cfg.reasoning_effort {
            let re = match v.to_lowercase().as_str() {
                "high" => ReasoningEffort::High,
                "low" => ReasoningEffort::Low,
                "medium" => ReasoningEffort::Medium,
                _ => {
                    return Err(LlmProbeError::ConfigError(format!(
                        "错误的值 reasoning_effort: {}",
                        v
                    )))
                }
            };
            builder = builder.reasoning_effort(re);
        }

        // system 通过 extra_body 传递
    } else {
        builder = builder.timeout_seconds(60);
    }

    if !final_url.is_empty() {
        builder = builder.base_url(final_url);
    }
    // println!("is_openai_compatible={}", is_openai_compatible);
    if is_openai_compatible {
        match builder.build_openai_compatible() {
            Ok(llm) => Ok(LLMClient {
                llm: LLMBackendEnum::WithReasoning(llm),
                // llm_x: Some(llm),
                base_url: final_url.to_string(),
                api_key: api_key.to_string(),
                provider_name,
                is_openai_compatible: true,
            }),
            Err(e) => Err(map_llm_error(&e.to_string())),
        }
    } else {
        match builder.build() {
            Ok(llm) => Ok(LLMClient {
                llm: LLMBackendEnum::Standard(llm),
                // llm_x: None,
                base_url: final_url.to_string(),
                api_key: api_key.to_string(),
                provider_name,
                is_openai_compatible: false, // enable_thinking,
            }),
            Err(e) => Err(map_llm_error(&e.to_string())),
        }
    }
}
