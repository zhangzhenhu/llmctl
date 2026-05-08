use crate::config::schema::AppConfigV2;
use crate::error::LlmProbeError;
use serde_json::Value;
use std::fs;
use std::path::Path;

const V2_MARKERS: [&str; 7] = [
    "version",
    "defaults",
    "profiles",
    "active_profile",
    "providers",
    "active_provider",
    "context",
];
const LEGACY_V1_KEYS: [&str; 15] = [
    "provider",
    "base_url",
    "api_key",
    "model",
    "stream",
    "no_proxy",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "system",
    "timeout_seconds",
    "reasoning",
    "reasoning_effort",
    "extra_body",
];

pub fn load_app_config(path: &Path) -> Result<AppConfigV2, LlmProbeError> {
    let content = fs::read_to_string(path)
        .map_err(|_| LlmProbeError::ConfigFileNotFound(path.display().to_string()))?;
    let value = parse_config_value(path, &content)?;
    validate_top_level_schema(&value)?;

    serde_json::from_value(value)
        .map_err(|e| LlmProbeError::ConfigError(format!("Invalid v2 config schema: {e}")))
}

pub fn search_config_file() -> Option<std::path::PathBuf> {
    let search_paths = [
        std::path::PathBuf::from("./llmctl.yaml"),
        std::path::PathBuf::from("./llmctl.json"),
        std::path::PathBuf::from("./llm.yaml"),
        std::path::PathBuf::from("./llm.json"),
    ];

    for path in &search_paths {
        if path.exists() {
            return Some(path.clone());
        }
    }

    if let Some(config_dir) = dirs::config_dir() {
        let app_config_dir = config_dir.join("llmctl");
        for ext in ["yaml", "json"] {
            let path = app_config_dir.join(format!("config.{}", ext));
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}

fn parse_config_value(path: &Path, content: &str) -> Result<Value, LlmProbeError> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext.to_lowercase().as_str() {
        "yaml" | "yml" => {
            serde_yaml::from_str::<Value>(content).map_err(|_| LlmProbeError::ConfigFormatError)
        }
        "json" => {
            serde_json::from_str::<Value>(content).map_err(|_| LlmProbeError::ConfigFormatError)
        }
        _ => Err(LlmProbeError::ConfigFormatError),
    }
}

fn validate_top_level_schema(value: &Value) -> Result<(), LlmProbeError> {
    let Some(map) = value.as_object() else {
        return Err(LlmProbeError::ConfigError(
            "Invalid v2 config schema: top-level YAML/JSON value must be an object".to_string(),
        ));
    };

    validate_version_field(map)?;

    if is_v2_config_value(map) {
        return Ok(());
    }

    if looks_like_legacy_v1_config(map) {
        return Err(LlmProbeError::ConfigError(
            "Legacy v1 flat config is no longer supported. Rewrite it to the v2 schema using defaults/profiles/active_profile, or run `llmctl --init` to generate a fresh template.".to_string(),
        ));
    }

    Err(LlmProbeError::ConfigError(
        "Invalid v2 config schema: expected at least one of version/defaults/profiles/active_profile".to_string(),
    ))
}

fn validate_version_field(map: &serde_json::Map<String, Value>) -> Result<(), LlmProbeError> {
    let Some(version) = map.get("version") else {
        return Ok(());
    };

    let Some(version_number) = version.as_u64() else {
        return Err(LlmProbeError::ConfigError(
            "Invalid v2 config schema: version must be the integer 2".to_string(),
        ));
    };

    if version_number != 2 {
        return Err(LlmProbeError::ConfigError(format!(
            "Unsupported config version: {version_number}. Only version 2 is supported."
        )));
    }

    Ok(())
}

fn is_v2_config_value(map: &serde_json::Map<String, Value>) -> bool {
    V2_MARKERS.iter().any(|key| map.contains_key(*key))
}

fn looks_like_legacy_v1_config(map: &serde_json::Map<String, Value>) -> bool {
    LEGACY_V1_KEYS.iter().any(|key| map.contains_key(*key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
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
        std::env::temp_dir().join(format!("llmctl_loader_test_{pid}_{stamp}_{seq}.{ext}"))
    }

    #[test]
    fn loads_v2_yaml_config() {
        let path = temp_path("yaml");
        fs::write(
            &path,
            r#"
version: 2
active_profile: openai_main
profiles:
  openai_main:
    adapter: openai
    model: gpt-5
"#,
        )
        .expect("write yaml");

        let config = load_app_config(&path).expect("yaml should load");
        assert_eq!(config.version, Some(2));
        assert_eq!(config.active_profile.as_deref(), Some("openai_main"));
        assert_eq!(
            config
                .profiles
                .get("openai_main")
                .and_then(|profile| profile.model.as_deref()),
            Some("gpt-5")
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn loads_v2_json_config() {
        let path = temp_path("json");
        fs::write(
            &path,
            r#"{
  "version": 2,
  "defaults": {
    "stream": true
  },
  "profiles": {
    "openai_main": {
      "adapter": "openai",
      "model": "gpt-4o"
    }
  }
}"#,
        )
        .expect("write json");

        let config = load_app_config(&path).expect("json should load");
        assert_eq!(config.version, Some(2));
        assert_eq!(config.defaults.stream, Some(true));
        assert_eq!(config.profiles.len(), 1);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn accepts_context_only_v2_config() {
        let path = temp_path("yaml");
        fs::write(
            &path,
            r#"
context:
  - role: system
    content: You are concise.
"#,
        )
        .expect("write yaml");

        let config = load_app_config(&path).expect("context-only config should load");
        assert_eq!(config.context.len(), 1);
        assert_eq!(config.context[0].role, "system");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn rejects_legacy_v1_flat_config() {
        let path = temp_path("yaml");
        fs::write(
            &path,
            r#"
provider: dashscope
base_url: https://dashscope.aliyuncs.com/compatible-mode/v1/
model: qwen-plus
"#,
        )
        .expect("write legacy yaml");

        let err = load_app_config(&path).expect_err("legacy config should be rejected");
        assert!(err
            .to_string()
            .contains("Legacy v1 flat config is no longer supported"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn rejects_unsupported_version() {
        let path = temp_path("yaml");
        fs::write(
            &path,
            r#"
version: 1
profiles:
  openai_main:
    adapter: openai
"#,
        )
        .expect("write yaml");

        let err = load_app_config(&path).expect_err("version 1 should be rejected");
        assert!(err.to_string().contains("Only version 2 is supported"));

        let _ = fs::remove_file(path);
    }
}
