#!/usr/bin/env bash
set -euo pipefail

# Run data-driven CLI dry-run regression tests.
cargo test --test cli_dry_run_cases -- --nocapture
