#![allow(dead_code)]
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlmProbeError {
    #[error("配置文件不存在: {0}")]
    ConfigFileNotFound(String),

    #[error("配置文件格式错误，请检查 YAML/JSON 语法是否正确")]
    ConfigFormatError,

    #[error("配置错误: {0}")]
    ConfigError(String),

    #[error("缺少必填字段: {0}")]
    MissingRequiredField(String),

    #[error("不支持的服务商: {0}")]
    UnsupportedProvider(String),

    #[error("API 密钥无效，请检查密钥是否正确或是否过期")]
    InvalidApiKey,

    #[error("API 地址无法访问，请检查网络或地址是否正确")]
    NetworkError,

    #[error("模型名称不存在，可通过 -l 参数查看当前服务商支持的模型列表")]
    ModelNotFound,

    #[error("输入文件不存在，请检查路径是否正确")]
    InputFileNotFound,

    #[error("文件已存在，是否覆盖？[y/N]")]
    FileExists,

    #[error("操作已取消")]
    OperationCancelled,

    #[error("写入文件失败")]
    WriteFileError,

    #[error("{0}")]
    ApiError(String),

    #[error("请求超时，请检查网络连接或 API 地址")]
    Timeout,

    #[error("请求过于频繁，请稍后重试")]
    RateLimitError,

    #[error("服务器内部错误，请稍后重试")]
    ServerError,

    #[error("{0}")]
    RuntimeError(String),
}

impl LlmProbeError {
    pub fn user_message(&self) -> String {
        self.to_string()
    }
}
