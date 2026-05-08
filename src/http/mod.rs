use crate::error::LlmProbeError;

pub fn build_reqwest_client(
    timeout_seconds: Option<u64>,
    no_proxy: bool,
) -> Result<reqwest::Client, LlmProbeError> {
    let mut builder = reqwest::Client::builder();
    builder = builder
        .tcp_nodelay(true)
        .gzip(true)
        .pool_max_idle_per_host(4)
        .http2_keep_alive_interval(Some(std::time::Duration::from_secs(20)))
        .http2_keep_alive_timeout(std::time::Duration::from_secs(10))
        .http2_keep_alive_while_idle(true)
        .http2_adaptive_window(true);

    if let Some(timeout_seconds) = timeout_seconds {
        builder = builder.timeout(std::time::Duration::from_secs(timeout_seconds));
    }
    if no_proxy {
        builder = builder.no_proxy();
    }

    builder
        .build()
        .map_err(|err| LlmProbeError::RuntimeError(format!("Failed to build HTTP client: {err}")))
}
