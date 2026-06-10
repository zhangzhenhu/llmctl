#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub profile: String,
    pub adapter: String,
    pub requested_model: String,
    pub provider_model: String,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub duration_ms: u64,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub provider: String,
}
