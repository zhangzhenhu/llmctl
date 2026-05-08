use std::path::PathBuf;
use std::process::Command;

#[test]
fn version_flags_match_package_version() {
    let expected = format!("llmctl {}\n", env!("CARGO_PKG_VERSION"));
    let bin = llmctl_bin_path();

    for flag in ["-v", "--version"] {
        let output = Command::new(&bin)
            .arg(flag)
            .output()
            .unwrap_or_else(|e| panic!("failed to execute {:?} with {}: {e}", bin, flag));

        assert!(
            output.status.success(),
            "expected {} to succeed, got status {:?}",
            flag,
            output.status.code()
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            expected,
            "unexpected stdout for {}",
            flag
        );
        assert!(
            output.stderr.is_empty(),
            "expected empty stderr for {}, got: {}",
            flag,
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn llmctl_bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_llmctl"))
}
