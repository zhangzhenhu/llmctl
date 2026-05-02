// use crate::backends::OpenAI;
// use crate::backends::openai::OpenAI;
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
    // Legacy fallback path keeps a single backend shape to avoid maintaining
    // a second custom reasoning stream branch in llmctl.
    Standard(Box<dyn LLMProvider>),
}
pub struct LLMClient {
    llm: LLMBackendEnum,
    base_url: String,
    api_key: String,
    provider_name: String,
}

fn build_chat_message(msg: Message) -> ChatMessage {
    let role = msg.role.as_str();
    let content = msg.content;

    let chat_role = match role {
        "assistant" => ChatRole::Assistant,
        _ => ChatRole::User,
    };

    ChatMessageBuilder::new(chat_role).content(content).build()
}

impl LLMClient {
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
    pub async fn stream_chat(
        &self,
        messages: Vec<Message>,
        model: &str,
    ) -> Result<(), LlmProbeError> {
        match &self.llm {
            LLMBackendEnum::Standard(llm) => self.stream_chat_old(llm.as_ref(), messages, model),
        }
        .await
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

        if content.trim().is_empty() {
            return Err(LlmProbeError::ApiError(
                "Stream finished without assistant content. Retry with --no-stream to diagnose provider behavior.".to_string(),
            ));
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        println!();
        println!("{}", "─".repeat(50).dimmed());
        println!(
            "{}: ({}){}",
            "Model".green(),
            self.provider_name.green(),
            model.green()
        );
        println!("{}: {} ms", "Duration".yellow(), duration_ms);
        println!("{}", "─".repeat(50).dimmed());

        Ok(())
    }
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmProbeError> {
        let request: Option<&ModelListRequest> = None;
        let response = match &self.llm {
            LLMBackendEnum::Standard(llm) => llm.list_models(request).await,
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
                    Err(e) => Err(LlmProbeError::ApiError(format!(
                        "Failed to parse model list: {}",
                        e
                    ))),
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
            "DNS resolution failed, please check if the API address is correct".to_string()
        } else if lower.contains("connection refused") {
            "Connection refused, please check if the API address is correct".to_string()
        } else if lower.contains("timed out") || lower.contains("timeout") {
            "Request timeout, please check network connection or API address".to_string()
        } else if lower.contains("could not resolve") {
            "Unable to resolve domain name, please check if the API address is correct".to_string()
        } else {
            format!("Network error: {}", error)
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

fn parse_legacy_reasoning_effort(value: &str) -> Result<Option<ReasoningEffort>, LlmProbeError> {
    let normalized = value.trim().to_lowercase();

    // The legacy llm backend only supports low/medium/high.
    // We normalize broader unified values into this smaller set, and treat
    // disable/auto-like values as "no explicit effort".
    let mapped = match normalized.as_str() {
        "off" | "none" | "auto" | "false" | "disable" | "disabled" => None,
        "low" | "minimal" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" | "xhigh" | "max" => Some(ReasoningEffort::High),
        _ if normalized.starts_with("budget:") => Some(ReasoningEffort::High),
        _ if normalized.parse::<u32>().is_ok() => Some(ReasoningEffort::High),
        _ => {
            return Err(LlmProbeError::ConfigError(format!(
                "Invalid reasoning_effort value: {}",
                value
            )));
        }
    };

    Ok(mapped)
}

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

    let final_url = base_url
        .filter(|url| !url.trim().is_empty())
        .unwrap_or(default_url);

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
        "openai_compatible" | "openai-compatible" | "aliyun" | "dashscope" => LLMBackend::OpenAI,
        _ => return Err(LlmProbeError::UnsupportedProvider(provider.to_string())),
    };
    let provider_name = provider_lower.clone();
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
            if let Some(re) = parse_legacy_reasoning_effort(v)? {
                builder = builder.reasoning_effort(re);
            }
        }

        // system 通过 extra_body 传递
    } else {
        builder = builder.timeout_seconds(60);
    }

    if !final_url.is_empty() {
        builder = builder.base_url(final_url);
    }
    // Legacy fallback path stays on the standard llm backend only.
    // OpenAI-compatible reasoning streaming is now handled by the genai path.
    match builder.build() {
        Ok(llm) => Ok(LLMClient {
            llm: LLMBackendEnum::Standard(llm),
            base_url: final_url.to_string(),
            api_key: api_key.to_string(),
            provider_name,
        }),
        Err(e) => Err(map_llm_error(&e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_backend_uses_standard_path_for_openai_compatible_alias() {
        let client = create_llm_backend(
            "openai-compatible",
            "test-key",
            Some("https://example.com/v1"),
            "test-model",
            None,
        )
        .expect("backend should build");

        match client.llm {
            LLMBackendEnum::Standard(_) => {}
        }
    }

    #[test]
    fn legacy_reasoning_effort_accepts_off_like_values() {
        assert!(parse_legacy_reasoning_effort("off").unwrap().is_none());
        assert!(parse_legacy_reasoning_effort("none").unwrap().is_none());
        assert!(parse_legacy_reasoning_effort("auto").unwrap().is_none());
    }

    #[test]
    fn legacy_reasoning_effort_maps_extended_levels() {
        assert!(matches!(
            parse_legacy_reasoning_effort("minimal").unwrap(),
            Some(ReasoningEffort::Low)
        ));
        assert!(matches!(
            parse_legacy_reasoning_effort("xhigh").unwrap(),
            Some(ReasoningEffort::High)
        ));
        assert!(matches!(
            parse_legacy_reasoning_effort("budget:2048").unwrap(),
            Some(ReasoningEffort::High)
        ));
    }
}
