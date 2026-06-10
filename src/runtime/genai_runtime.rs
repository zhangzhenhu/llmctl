//! GenAI runtime bridge for llmctl.
//!
//! This module is intentionally thin: it maps resolved llmctl config into
//! `genai::Client` and chat/list calls without introducing a new engine layer.

use crate::config::ResolvedRuntimeConfig;
use crate::error::LlmProbeError;
use crate::http::build_reqwest_client;
use crate::provider::{ChatResponse, ModelInfo};
use futures::StreamExt;
use genai::adapter::AdapterKind;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, ChatStreamEvent, ReasoningEffort};
use genai::resolver::{AuthData, Endpoint, ServiceTargetResolver};
use genai::{Client, ServiceTarget};
use serde::Deserialize;

pub enum ModelListSource {
    ExplicitBaseUrl,
    LiveProviderEndpoint,
    StaticCatalog,
}

impl ModelListSource {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::ExplicitBaseUrl => "provider /models endpoint (explicit base_url)",
            Self::LiveProviderEndpoint => "provider /models endpoint",
            Self::StaticCatalog => "genai static catalog fallback",
        }
    }
}

pub struct ModelListResult {
    pub models: Vec<ModelInfo>,
    pub source: ModelListSource,
}

pub struct GenaiRuntime {
    client: Client,
    http_client: reqwest::Client,
    resolved: ResolvedRuntimeConfig,
}

impl GenaiRuntime {
    pub fn is_stream_enabled(&self) -> bool {
        self.resolved.stream
    }

    #[cfg(test)]
    pub fn unsupported_reason_for_chat(resolved: &ResolvedRuntimeConfig) -> Option<String> {
        if adapter_kind_for(resolved).is_none() {
            return Some(format!(
                "adapter_not_supported_by_genai ({})",
                resolved.adapter
            ));
        }
        None
    }

    #[cfg(test)]
    pub fn unsupported_reason_for_list(resolved: &ResolvedRuntimeConfig) -> Option<String> {
        if adapter_kind_for(resolved).is_none() {
            return Some(format!(
                "adapter_not_supported_by_genai ({})",
                resolved.adapter
            ));
        }
        None
    }

    #[cfg(test)]
    pub fn is_chat_supported(resolved: &ResolvedRuntimeConfig) -> bool {
        Self::unsupported_reason_for_chat(resolved).is_none()
    }

    #[cfg(test)]
    pub fn is_list_supported(resolved: &ResolvedRuntimeConfig) -> bool {
        Self::unsupported_reason_for_list(resolved).is_none()
    }

    pub fn supports_resolved_config(resolved: &ResolvedRuntimeConfig) -> bool {
        adapter_kind_for(resolved).is_some()
    }

    pub fn from_resolved(resolved: ResolvedRuntimeConfig) -> Result<Self, LlmProbeError> {
        let http_client = build_reqwest_client(resolved.timeout_seconds, resolved.no_proxy)?;
        let api_key = resolved.api_key.clone();
        let mut builder = Client::builder()
            .with_reqwest(http_client.clone())
            .with_auth_resolver_fn(move |_| Ok(Some(AuthData::from_single(api_key.clone()))));

        if let Some(base_url) = resolved.base_url.clone() {
            let resolver = ServiceTargetResolver::from_resolver_fn(move |target: ServiceTarget| {
                let ServiceTarget { model, auth, .. } = target;
                Ok(ServiceTarget {
                    endpoint: Endpoint::from_owned(base_url.clone()),
                    auth,
                    model,
                })
            });
            builder = builder.with_service_target_resolver(resolver);
        }

        Ok(Self {
            client: builder.build(),
            http_client,
            resolved,
        })
    }

    pub async fn list_models(&self) -> Result<ModelListResult, LlmProbeError> {
        if self.resolved.base_url.is_some() {
            // Keep explicit base_url listing on a direct /models request path.
            // This deliberately avoids relying on genai's Client::all_model_names
            // resolver semantics for custom endpoints (see rust-genai issue #217).
            return self.list_models_from_base_url().await;
        }

        if let Some(live_models) = self.try_list_models_live().await? {
            return Ok(ModelListResult {
                models: live_models,
                source: ModelListSource::LiveProviderEndpoint,
            });
        }

        let models = self.list_models_via_client().await?;
        Ok(ModelListResult {
            models,
            source: ModelListSource::StaticCatalog,
        })
    }

    async fn list_models_via_client(&self) -> Result<Vec<ModelInfo>, LlmProbeError> {
        let kind = adapter_kind_for(&self.resolved)
            .ok_or_else(|| LlmProbeError::UnsupportedProvider(self.resolved.adapter.clone()))?;
        let model_names = self
            .client
            .all_model_names(kind, ())
            .await
            .map_err(map_genai_error)?;
        let provider = self.resolved.adapter.clone();

        Ok(model_names
            .into_iter()
            .map(|name| ModelInfo {
                id: name.clone(),
                name,
                provider: provider.clone(),
            })
            .collect())
    }

    async fn list_models_from_base_url(&self) -> Result<ModelListResult, LlmProbeError> {
        let base_url = self
            .resolved
            .base_url
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                LlmProbeError::ApiError(
                    "base_url is required for explicit model listing".to_string(),
                )
            })?;
        let url = format!("{}/models", base_url.trim_end_matches('/'));
        let mut request = self.http_client.get(&url);

        if !self.resolved.api_key.is_empty() {
            request = request.bearer_auth(&self.resolved.api_key);
        }

        let response = request.send().await.map_err(|err| {
            LlmProbeError::ApiError(format!("Model list request failed for {url}: {err}"))
        })?;
        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<unavailable>".to_string());
            return Err(LlmProbeError::ApiError(format!(
                "Model list request failed for {url}: HTTP {status}. Response body:\n{body}"
            )));
        }

        let payload = response
            .json::<OpenAiModelsResponse>()
            .await
            .map_err(|err| {
                LlmProbeError::ApiError(format!(
                    "Failed to parse model list response from {url}: {err}"
                ))
            })?;
        let provider = self.resolved.adapter.clone();

        Ok(ModelListResult {
            models: payload
                .data
                .into_iter()
                .map(|item| ModelInfo {
                    id: item.id.clone(),
                    name: item.id,
                    provider: provider.clone(),
                })
                .collect(),
            source: ModelListSource::ExplicitBaseUrl,
        })
    }

    async fn try_list_models_live(&self) -> Result<Option<Vec<ModelInfo>>, LlmProbeError> {
        if !supports_live_models_endpoint(&self.resolved.adapter) {
            return Ok(None);
        }

        let Some(base_url) = models_base_url_for(&self.resolved) else {
            return Ok(None);
        };
        let url = format!("{}/models", base_url.trim_end_matches('/'));
        let mut request = self.http_client.get(url);

        if !self.resolved.api_key.is_empty() {
            request = request.bearer_auth(&self.resolved.api_key);
        }

        let response = match request.send().await {
            Ok(resp) => resp,
            Err(_) => return Ok(None),
        };
        if !response.status().is_success() {
            // Keep compatibility with previous behavior: fallback to SDK static
            // list when provider-side list endpoint is unavailable.
            return Ok(None);
        }

        let payload = match response.json::<OpenAiModelsResponse>().await {
            Ok(payload) => payload,
            Err(_) => return Ok(None),
        };
        let provider = self.resolved.adapter.clone();
        let models = payload
            .data
            .into_iter()
            .map(|item| ModelInfo {
                id: item.id.clone(),
                name: item.id,
                provider: provider.clone(),
            })
            .collect::<Vec<_>>();

        Ok(Some(models))
    }

    pub async fn chat_completion(&self) -> Result<ChatResponse, LlmProbeError> {
        let request = build_chat_request(&self.resolved);
        let options = build_chat_options(&self.resolved);
        let model = self.resolved.effective_model.clone();
        let started_at = std::time::Instant::now();

        let response = self
            .client
            .exec_chat(&model, request, Some(&options))
            .await
            .map_err(map_genai_error)?;

        let content = response.first_text().map(|v| v.to_string());
        let usage = response.usage.clone();
        let input_tokens = usage.prompt_tokens.and_then(to_u32_opt);
        let output_tokens = usage.completion_tokens.and_then(to_u32_opt);
        let provider_model = response.provider_model_iden.model_name.to_string();
        let reasoning_content = if self.resolved.capture_reasoning_content {
            response.reasoning_content
        } else {
            None
        };

        Ok(ChatResponse {
            profile: self.resolved.active_profile.clone(),
            adapter: self.resolved.adapter.clone(),
            requested_model: self.resolved.model.clone(),
            provider_model,
            content,
            reasoning_content,
            duration_ms: started_at.elapsed().as_millis() as u64,
            input_tokens,
            output_tokens,
        })
    }

    pub async fn stream_chat(&self) -> Result<(), LlmProbeError> {
        use colored::*;
        use std::io::Write;

        let request = build_chat_request(&self.resolved);
        let options = build_chat_options(&self.resolved)
            .with_capture_content(true)
            .with_capture_reasoning_content(self.resolved.capture_reasoning_content)
            .with_capture_usage(self.resolved.capture_usage);
        let model = self.resolved.effective_model.clone();
        let started_at = std::time::Instant::now();

        let mut stream_response = self
            .client
            .exec_chat_stream(&model, request, Some(&options))
            .await
            .map_err(map_genai_error)?;

        let mut in_reasoning = false;
        // Some OpenAI-compatible providers do not emit normal text chunks
        // consistently in streaming mode, but still provide final concatenated
        // text in `End.captured_content`. Track whether we have printed any
        // response text and keep an end-of-stream fallback when needed.
        let mut printed_response_content = false;
        let mut fallback_response_text: Option<String> = None;
        let mut usage_prompt_tokens: Option<u32> = None;
        let mut usage_completion_tokens: Option<u32> = None;
        let mut provider_model = self.resolved.effective_model.clone();

        while let Some(event) = stream_response.stream.next().await {
            match event.map_err(map_genai_error)? {
                ChatStreamEvent::Start => {}
                ChatStreamEvent::ReasoningChunk(chunk) => {
                    if !self.resolved.capture_reasoning_content {
                        continue;
                    }
                    if !in_reasoning {
                        println!("{}:", "Thinking".cyan());
                        println!("{}", "─".repeat(50).dimmed());
                        in_reasoning = true;
                    }
                    print!("{}", chunk.content);
                    std::io::stdout().flush().ok();
                }
                ChatStreamEvent::Chunk(chunk) => {
                    if in_reasoning {
                        println!("\n{}", "─".repeat(50).dimmed());
                        println!("{}:", "Response".cyan());
                        println!("{}", "─".repeat(50).dimmed());
                        in_reasoning = false;
                    }
                    print!("{}", chunk.content);
                    std::io::stdout().flush().ok();
                    printed_response_content = true;
                }
                ChatStreamEvent::End(end) => {
                    if let Some(ref usage) = end.captured_usage {
                        usage_prompt_tokens = usage.prompt_tokens.and_then(to_u32_opt);
                        usage_completion_tokens = usage.completion_tokens.and_then(to_u32_opt);
                    }
                    if let Some(captured_provider_model) = end.captured_provider_model_name() {
                        provider_model = captured_provider_model.to_string();
                    }
                    if !printed_response_content {
                        fallback_response_text = end.captured_first_text().map(ToOwned::to_owned);
                    }
                }
                ChatStreamEvent::ThoughtSignatureChunk(_) | ChatStreamEvent::ToolCallChunk(_) => {}
            }
        }

        if !printed_response_content {
            if let Some(text) = fallback_response_text {
                if !text.is_empty() {
                    if in_reasoning {
                        println!("\n{}", "─".repeat(50).dimmed());
                        println!("{}:", "Response".cyan());
                        println!("{}", "─".repeat(50).dimmed());
                    }
                    print!("{text}");
                    std::io::stdout().flush().ok();
                }
            }
        }

        println!();
        println!("{}", "─".repeat(50).dimmed());
        if let (Some(input), Some(output)) = (usage_prompt_tokens, usage_completion_tokens) {
            println!("{}: Input {}, Output {}", "Token".dimmed(), input, output);
        }
        let response = ChatResponse {
            profile: self.resolved.active_profile.clone(),
            adapter: self.resolved.adapter.clone(),
            requested_model: self.resolved.model.clone(),
            provider_model,
            content: None,
            reasoning_content: None,
            duration_ms: started_at.elapsed().as_millis() as u64,
            input_tokens: usage_prompt_tokens,
            output_tokens: usage_completion_tokens,
        };
        for line in crate::output::formatter::response_metadata_lines(&response) {
            println!("{line}");
        }
        println!(
            "{}: {} ms",
            "Duration".yellow(),
            started_at.elapsed().as_millis()
        );

        Ok(())
    }
}

fn build_chat_request(resolved: &ResolvedRuntimeConfig) -> ChatRequest {
    let mut req = ChatRequest::default();
    if let Some(system) = &resolved.system {
        req = req.with_system(system.clone());
    }

    for message in &resolved.context {
        let msg = match message.role.to_lowercase().as_str() {
            "assistant" => ChatMessage::assistant(message.content.clone()),
            "system" => ChatMessage::system(message.content.clone()),
            _ => ChatMessage::user(message.content.clone()),
        };
        req = req.append_message(msg);
    }
    req
}

fn build_chat_options(resolved: &ResolvedRuntimeConfig) -> ChatOptions {
    // Keep compatibility lightweight and genai-native. The vendored genai
    // patch exposes request `extra_body` passthrough, which lets provider
    // profiles carry small OpenAI-compatible extensions such as
    // Aliyun/DashScope `enable_thinking`.
    let mut options = ChatOptions::default()
        .with_capture_usage(resolved.capture_usage)
        .with_capture_reasoning_content(resolved.capture_reasoning_content)
        .with_normalize_reasoning_content(resolved.normalize_reasoning_content);

    if let Some(temperature) = resolved.temperature {
        options = options.with_temperature(temperature as f64);
    }
    if let Some(max_tokens) = resolved.max_tokens {
        options = options.with_max_tokens(max_tokens);
    }
    if let Some(top_p) = resolved.top_p {
        options = options.with_top_p(top_p as f64);
    }
    if let Some(raw_reasoning_effort) = resolved.reasoning_effort.as_deref() {
        // `--reasoning off` is internally represented as "none" for dry-run
        // visibility. On some OpenAI-compatible providers (observed on Aliyun
        // Responses compatibility), sending `reasoning.effort=none` can cause
        // an empty assistant content response. For `off`, provider-specific
        // disable flags are carried via extra_body instead.
        if !raw_reasoning_effort.eq_ignore_ascii_case("none") {
            if let Some(reasoning_effort) = parse_reasoning_effort(raw_reasoning_effort) {
                options = options.with_reasoning_effort(reasoning_effort);
            }
        }
    }
    if !resolved.extra_body.is_empty() {
        options = options.with_extra_body(serde_json::Value::Object(
            resolved.extra_body.clone().into_iter().collect(),
        ));
    }
    options
}

fn parse_reasoning_effort(value: &str) -> Option<ReasoningEffort> {
    let normalized = value.trim().to_lowercase();
    if let Some(raw_budget) = normalized.strip_prefix("budget:") {
        return raw_budget.parse::<u32>().ok().map(ReasoningEffort::Budget);
    }

    match normalized.as_str() {
        "none" => Some(ReasoningEffort::None),
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::XHigh),
        "max" => Some(ReasoningEffort::Max),
        "minimal" => Some(ReasoningEffort::Minimal),
        _ => value.parse::<u32>().ok().map(ReasoningEffort::Budget),
    }
}

fn adapter_kind_for(resolved: &ResolvedRuntimeConfig) -> Option<AdapterKind> {
    match resolved.adapter.as_str() {
        "openai" => {
            if resolved.effective_model.starts_with("openai_resp::") {
                Some(AdapterKind::OpenAIResp)
            } else {
                Some(AdapterKind::OpenAI)
            }
        }
        "aliyun" => Some(AdapterKind::Aliyun),
        "anthropic" => Some(AdapterKind::Anthropic),
        "gemini" => Some(AdapterKind::Gemini),
        "ollama" => Some(AdapterKind::Ollama),
        "deepseek" => Some(AdapterKind::DeepSeek),
        "xai" => Some(AdapterKind::Xai),
        "groq" => Some(AdapterKind::Groq),
        "cohere" => Some(AdapterKind::Cohere),
        "fireworks" => Some(AdapterKind::Fireworks),
        "together" => Some(AdapterKind::Together),
        "zai" => Some(AdapterKind::Zai),
        _ => None,
    }
}

fn to_u32_opt(value: i32) -> Option<u32> {
    if value < 0 {
        None
    } else {
        Some(value as u32)
    }
}

fn map_genai_error(error: genai::Error) -> LlmProbeError {
    LlmProbeError::ApiError(error.to_string())
}

fn supports_live_models_endpoint(adapter: &str) -> bool {
    matches!(
        adapter,
        "openai"
            | "aliyun"
            | "deepseek"
            | "xai"
            | "groq"
            | "cohere"
            | "fireworks"
            | "together"
            | "zai"
    )
}

fn models_base_url_for(resolved: &ResolvedRuntimeConfig) -> Option<String> {
    if let Some(url) = resolved
        .base_url
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        return Some(url.to_string());
    }

    let default = match resolved.adapter.as_str() {
        "openai" => Some("https://api.openai.com/v1"),
        "aliyun" => Some("https://dashscope.aliyuncs.com/compatible-mode/v1/"),
        "deepseek" => Some("https://api.deepseek.com/v1"),
        "xai" => Some("https://api.x.ai/v1"),
        "groq" => Some("https://api.groq.com/openai/v1"),
        "cohere" => Some("https://api.cohere.com/v2"),
        "fireworks" => Some("https://api.fireworks.ai/inference/v1"),
        "together" => Some("https://api.together.xyz/v1"),
        "zai" => Some("https://api.z.ai/api/paas/v4"),
        _ => None,
    };
    default.map(str::to_string)
}

#[derive(Debug, Deserialize)]
struct OpenAiModelsResponse {
    data: Vec<OpenAiModelItem>,
}

#[derive(Debug, Deserialize)]
struct OpenAiModelItem {
    id: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::resolver::ApiKeySource;
    use crate::config::schema::{Message, OpenAiApiMode};
    use std::collections::HashMap;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn resolved() -> ResolvedRuntimeConfig {
        ResolvedRuntimeConfig {
            active_profile: "default".to_string(),
            adapter: "openai".to_string(),
            model: "gpt-4.1".to_string(),
            base_url: Some("https://api.openai.com/v1".to_string()),
            api_key: "dummy".to_string(),
            api_key_source: ApiKeySource::Config,
            stream: false,
            context: vec![Message {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            max_tokens: Some(256),
            temperature: Some(0.2),
            top_p: Some(0.9),
            top_k: None,
            system: Some("be concise".to_string()),
            timeout_seconds: Some(60),
            reasoning_effort: Some("medium".to_string()),
            api_mode: OpenAiApiMode::Auto,
            api_mode_enforced: false,
            effective_model: "openai::gpt-4.1".to_string(),
            no_proxy: false,
            reasoning_setting: None,
            capture_usage: true,
            capture_reasoning_content: true,
            normalize_reasoning_content: true,
            extra_body: HashMap::new(),
        }
    }

    #[test]
    fn list_support_is_not_blocked_by_extra_body() {
        let mut cfg = resolved();
        cfg.extra_body
            .insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
        assert!(GenaiRuntime::is_list_supported(&cfg));
        assert!(GenaiRuntime::is_chat_supported(&cfg));
    }

    #[test]
    fn extra_body_is_not_a_genai_adapter_support_decision() {
        let mut cfg = resolved();
        cfg.adapter = "anthropic".to_string();
        cfg.effective_model = "anthropic::claude-sonnet-4-5".to_string();
        cfg.extra_body
            .insert("enable_thinking".to_string(), serde_json::Value::Bool(true));

        assert!(GenaiRuntime::is_chat_supported(&cfg));
    }

    #[test]
    fn unknown_adapter_blocks_both_list_and_chat() {
        let mut cfg = resolved();
        cfg.adapter = "unknown_adapter".to_string();
        cfg.effective_model = "unknown_adapter::model-x".to_string();

        assert!(!GenaiRuntime::is_list_supported(&cfg));
        assert!(!GenaiRuntime::is_chat_supported(&cfg));
        assert!(GenaiRuntime::unsupported_reason_for_list(&cfg)
            .unwrap_or_default()
            .contains("adapter_not_supported_by_genai"));
    }

    #[test]
    fn models_base_url_prefers_explicit_base_url() {
        let mut cfg = resolved();
        cfg.adapter = "aliyun".to_string();
        cfg.base_url = Some("https://example.com/v1/".to_string());
        assert_eq!(
            models_base_url_for(&cfg).as_deref(),
            Some("https://example.com/v1/")
        );
    }

    #[test]
    fn models_base_url_uses_builtin_default_for_aliyun() {
        let mut cfg = resolved();
        cfg.adapter = "aliyun".to_string();
        cfg.base_url = None;
        assert_eq!(
            models_base_url_for(&cfg).as_deref(),
            Some("https://dashscope.aliyuncs.com/compatible-mode/v1/")
        );
    }

    #[test]
    fn live_models_support_matrix_includes_aliyun() {
        assert!(supports_live_models_endpoint("aliyun"));
        assert!(!supports_live_models_endpoint("anthropic"));
    }

    #[tokio::test]
    async fn explicit_base_url_model_listing_hits_user_provided_models_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test listener");
        let addr = listener.local_addr().expect("listener addr");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut buf = vec![0_u8; 4096];
            let n = socket.read(&mut buf).await.expect("read request");
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = r#"{"data":[{"id":"custom-model"}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write response");
            request
        });

        let mut cfg = resolved();
        cfg.base_url = Some(format!("http://{addr}/v1/"));
        cfg.api_key = "test-key".to_string();

        let runtime = GenaiRuntime::from_resolved(cfg).expect("runtime");
        let result = runtime.list_models().await.expect("list models");
        let request = server.await.expect("server task");

        assert!(request.starts_with("GET /v1/models HTTP/1.1"));
        assert!(request.contains("Authorization: Bearer test-key"));
        assert_eq!(
            result.source.as_label(),
            "provider /models endpoint (explicit base_url)"
        );
        assert_eq!(result.models.len(), 1);
        assert_eq!(result.models[0].id, "custom-model");
    }
}
