use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlmProbeError {
    #[error("Config file not found: {0}")]
    ConfigFileNotFound(String),

    #[error("Invalid config format, please check YAML/JSON syntax")]
    ConfigFormatError,

    #[error("Config error: {0}")]
    ConfigError(String),

    #[error("Unsupported provider: {0}")]
    UnsupportedProvider(String),

    #[error("Operation cancelled")]
    OperationCancelled,

    #[error("Failed to write file")]
    WriteFileError,

    #[error("{0}")]
    ApiError(String),

    #[error("{0}")]
    RuntimeError(String),
}

impl LlmProbeError {
    pub fn user_message(&self) -> String {
        self.to_string()
    }
}
