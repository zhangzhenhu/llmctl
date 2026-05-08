use crate::config::loader::load_app_config;
use crate::error::LlmProbeError;
use crate::utils::prompt::prompt_overwrite;
use std::fs;
use std::path::{Path, PathBuf};

pub fn convert_config(input_path: &Path, output_path: Option<&Path>) -> Result<(), LlmProbeError> {
    let input_ext = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if !matches!(input_ext.to_lowercase().as_str(), "yaml" | "yml" | "json") {
        return Err(LlmProbeError::ConfigFormatError);
    }
    let config = load_app_config(input_path)?;

    let output: PathBuf = if let Some(out_path) = output_path {
        out_path.to_path_buf()
    } else {
        let mut output = input_path.to_path_buf();
        output.set_extension(match input_ext {
            "json" => "yaml",
            "yaml" | "yml" => "json",
            _ => "yaml",
        });
        output
    };

    if output.exists() && !prompt_overwrite(&output) {
        return Err(LlmProbeError::OperationCancelled);
    }

    let output_ext = output.extension().and_then(|e| e.to_str()).unwrap_or("");

    let output_content = match output_ext.to_lowercase().as_str() {
        "json" => {
            serde_json::to_string_pretty(&config).map_err(|_| LlmProbeError::ConfigFormatError)?
        }
        _ => serde_yaml::to_string(&config).map_err(|_| LlmProbeError::ConfigFormatError)?,
    };

    fs::write(&output, output_content).map_err(|_| LlmProbeError::WriteFileError)?;

    println!("Config converted successfully: {}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_path(ext: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let seq = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("llmctl_convert_test_{pid}_{stamp}_{seq}.{ext}"))
    }

    #[test]
    fn converts_v2_yaml_to_json() {
        let input = temp_path("yaml");
        let output = temp_path("json");
        fs::write(
            &input,
            r#"
version: 2
active_profile: openai_main
profiles:
  openai_main:
    adapter: openai
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
"#,
        )
        .expect("write input");

        convert_config(&input, Some(&output)).expect("conversion should succeed");

        let converted = fs::read_to_string(&output).expect("read output");
        let parsed: Value = serde_json::from_str(&converted).expect("valid json");
        assert_eq!(parsed.get("version"), Some(&Value::from(2)));
        assert_eq!(
            parsed.get("active_profile"),
            Some(&Value::from("openai_main"))
        );

        let _ = fs::remove_file(input);
        let _ = fs::remove_file(output);
    }

    #[test]
    fn rejects_legacy_yaml_input() {
        let input = temp_path("yaml");
        let output = temp_path("json");
        fs::write(
            &input,
            r#"
provider: dashscope
base_url: https://dashscope.aliyuncs.com/compatible-mode/v1/
api_key: test-key
model: qwen-plus
"#,
        )
        .expect("write input");

        let err = convert_config(&input, Some(&output)).expect_err("legacy input should fail");
        assert!(err
            .to_string()
            .contains("Legacy v1 flat config is no longer supported"));

        let _ = fs::remove_file(input);
    }
}
