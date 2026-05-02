# CLI Dry-Run Test Suite

This folder contains centralized, data-driven CLI regression tests.

## Structure

- `cases/cli_dry_run_cases.yaml`: all test case definitions.
- `cli_dry_run_cases.rs`: unified integration test runner.
- `../scripts/run-cli-dry-run-tests.sh`: one-command test entrypoint.

## Case Format

Each case in YAML supports:

- `name`: unique case name.
- `config_yaml`: inline v2 config file content (written to a temp file at runtime).
- `args`: CLI arguments. Use `{config_path}` placeholder for the generated temp config path.
- `expect_exit_code`: expected process exit code.
- `stdout_contains`: required stdout substrings.
- `stderr_contains`: required stderr substrings.

## Run

```bash
cargo test --test cli_dry_run_cases -- --nocapture
```

or:

```bash
./scripts/run-cli-dry-run-tests.sh
```
