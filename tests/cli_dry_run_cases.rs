use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Deserialize)]
struct CaseFile {
    cases: Vec<CliDryRunCase>,
}

#[derive(Debug, Deserialize)]
struct CliDryRunCase {
    name: String,
    config_yaml: String,
    args: Vec<String>,
    expect_exit_code: i32,
    #[serde(default)]
    stdout_contains: Vec<String>,
    #[serde(default)]
    stderr_contains: Vec<String>,
}

#[test]
fn run_all_cli_dry_run_cases() {
    let case_file_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cases")
        .join("cli_dry_run_cases.yaml");
    let case_file_text =
        fs::read_to_string(&case_file_path).expect("failed to read tests/cases yaml file");
    let case_file: CaseFile =
        serde_yaml::from_str(&case_file_text).expect("failed to parse tests/cases yaml file");

    // Run every case and aggregate failures, so one regression won't hide others.
    let mut failures = Vec::new();
    for case in case_file.cases {
        if let Err(msg) = run_one_case(&case) {
            failures.push(msg);
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} case(s) failed:\n\n{}",
            failures.len(),
            failures.join("\n\n")
        );
    }
}

fn run_one_case(case: &CliDryRunCase) -> Result<(), String> {
    let config_path = write_temp_config(case)?;
    let result = (|| -> Result<(), String> {
        let bin = llmctl_bin_path();
        let args = build_args(case, &config_path);
        let output = Command::new(&bin)
            .args(&args)
            .output()
            .map_err(|e| format!("case '{}': failed to execute {:?}: {e}", case.name, bin))?;

        let status_code = output.status.code().unwrap_or(-1);
        let stdout_text = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr_text = String::from_utf8_lossy(&output.stderr).to_string();

        if status_code != case.expect_exit_code {
            return Err(format!(
                "case '{}': exit code mismatch, expected {}, got {}\nargs: {:?}\nstdout:\n{}\nstderr:\n{}",
                case.name, case.expect_exit_code, status_code, args, stdout_text, stderr_text
            ));
        }

        for expected in &case.stdout_contains {
            if !stdout_text.contains(expected) {
                return Err(format!(
                    "case '{}': stdout missing expected text: {:?}\nargs: {:?}\nstdout:\n{}\nstderr:\n{}",
                    case.name, expected, args, stdout_text, stderr_text
                ));
            }
        }

        for expected in &case.stderr_contains {
            if !stderr_text.contains(expected) {
                return Err(format!(
                    "case '{}': stderr missing expected text: {:?}\nargs: {:?}\nstdout:\n{}\nstderr:\n{}",
                    case.name, expected, args, stdout_text, stderr_text
                ));
            }
        }

        Ok(())
    })();

    let _ = fs::remove_file(&config_path);
    result
}

fn build_args(case: &CliDryRunCase, config_path: &Path) -> Vec<String> {
    let config_path_str = config_path.to_string_lossy();
    case.args
        .iter()
        .map(|arg| arg.replace("{config_path}", &config_path_str))
        .collect()
}

fn write_temp_config(case: &CliDryRunCase) -> Result<PathBuf, String> {
    let timestamp_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("case '{}': invalid system time: {e}", case.name))?
        .as_nanos();
    let pid = std::process::id();
    let file_name = format!(
        "llmctl_cli_case_{}_{}_{}.yaml",
        sanitize_name(&case.name),
        pid,
        timestamp_nanos
    );
    let path = env::temp_dir().join(file_name);
    fs::write(&path, case.config_yaml.as_bytes()).map_err(|e| {
        format!(
            "case '{}': failed to write temp config {:?}: {e}",
            case.name, path
        )
    })?;
    Ok(path)
}

fn sanitize_name(input: &str) -> String {
    input
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn llmctl_bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_llmctl"))
}
