use crate::config::schema::FileConfig;
use crate::error::LlmProbeError;
use std::fs;
use std::path::{Path, PathBuf};

pub fn convert_config(input_path: &Path, output_path: Option<&Path>) -> Result<(), LlmProbeError> {
    let content = fs::read_to_string(input_path).map_err(|_| LlmProbeError::InputFileNotFound)?;

    let input_ext = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let config: FileConfig = match input_ext.to_lowercase().as_str() {
        "yaml" | "yml" => {
            serde_yaml::from_str(&content).map_err(|_| LlmProbeError::ConfigFormatError)?
        }
        "json" => serde_json::from_str(&content).map_err(|_| LlmProbeError::ConfigFormatError)?,
        _ => return Err(LlmProbeError::ConfigFormatError),
    };

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

    if output.exists() {
        print!("文件已存在，是否覆盖？[y/N]: ");
        use std::io::Write;
        std::io::stdout().flush().ok();
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer).ok();
        if !answer.trim().eq_ignore_ascii_case("y") {
            return Err(LlmProbeError::OperationCancelled);
        }
    }

    let output_ext = output.extension().and_then(|e| e.to_str()).unwrap_or("");

    let output_content = match output_ext.to_lowercase().as_str() {
        "json" => {
            serde_json::to_string_pretty(&config).map_err(|_| LlmProbeError::ConfigFormatError)?
        }
        _ => serde_yaml::to_string(&config).map_err(|_| LlmProbeError::ConfigFormatError)?,
    };

    fs::write(&output, output_content).map_err(|_| LlmProbeError::WriteFileError)?;

    println!("配置文件转换成功: {}", output.display());
    Ok(())
}
