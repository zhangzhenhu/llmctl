#![allow(dead_code)]
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlmProbeError {
    #[error("Config file not found: {0}")]
    ConfigFileNotFound(String),

    #[error("Invalid config format, please check YAML/JSON syntax")]
    ConfigFormatError,

    #[error("Config error: {0}")]
    ConfigError(String),

    #[error("Missing required field: {0}")]
    MissingRequiredField(String),

    #[error("Unsupported provider: {0}")]
    UnsupportedProvider(String),

    #[error("Invalid API key, please check if the key is correct or expired")]
    InvalidApiKey,

    #[error("API endpoint unreachable, please check network or address")]
    NetworkError,

    #[error("Model not found, use -l to list available models")]
    ModelNotFound,

    #[error("Input file not found, please check the path")]
    InputFileNotFound,

    #[error("File already exists, overwrite? [y/N]")]
    FileExists,

    #[error("Operation cancelled")]
    OperationCancelled,

    #[error("Failed to write file")]
    WriteFileError,

    #[error("{0}")]
    ApiError(String),

    #[error("Request timeout, please check network connection or API address")]
    Timeout,

    #[error("Rate limit exceeded, please retry later")]
    RateLimitError,

    #[error("Server error, please retry later")]
    ServerError,

    #[error("{0}")]
    RuntimeError(String),
}

impl LlmProbeError {
    pub fn user_message(&self) -> String {
        self.to_string()
    }
}
