# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.0] - 2026-05-08

### Changed

- Standardized the v2 CLI and config vocabulary around `adapter`, `profile`, `base_url`, and `api_mode`, while keeping legacy aliases as compatibility shims.
- Simplified execution to the genai runtime path and removed the older custom backend/runtime branches from `llmctl`.
- Tightened the documented and generated v2 config shape around `active_profile`, `profiles`, and profile-first overrides.

### Added

- Added `--list-adapters` output with built-in adapter aliases, default endpoints, and quick-start defaults.
- Added layered proxy controls with CLI `--no-proxy`, `defaults.no_proxy`, and `profiles.<name>.no_proxy`.
- Added model-list source reporting so `--list` can distinguish live provider `/models` responses from static fallback catalogs.

### Fixed

- Improved v2 config validation errors for unsupported legacy flat configs, invalid version fields, and missing `active_profile` references.
- Improved OpenAI-compatible streaming tolerance and request passthrough coverage through the vendored `genai` patch set used by `llmctl`.

### Docs

- Refreshed `README.md` and `README_CN.md` to match the current adapter/profile naming, proxy behavior, and runtime architecture notes.

## [2.0.1] - 2026-05-03

### Fixed

- Fixed `--list` with explicit `--url/--base_url` so model listing now stays on the user-provided `/models` endpoint instead of falling back to the SDK default provider endpoint.
- Improved explicit-endpoint model list errors to report the actual requested URL, HTTP status, and provider response body for easier debugging.

## [1.0.1] - 2026-05-02

### Fixed

- Fixed crates.io publish verification failure caused by relying on a patched local `genai` API (`ChatOptions::with_extra_body`).
- Added a compatibility fallback: when `extra_body` is present, chat execution routes to legacy backend instead of genai to keep published crate buildable with upstream `genai`.
- Clarified fallback behavior with inline code comments and dry-run observability.

### Docs

- Synced README/README_CN CLI sections with actual `--help` output.
- Added Cargo installation instructions (`cargo install llmctl`, `cargo install --path .`).

## [1.0.0] - 2026-05-02

### Changed

- Upgraded runtime architecture to a genai-first execution path, with legacy backend fallback only when needed.
- Migrated configuration model to v2 multi-provider profiles (`version: 2`, `active_provider`, `providers.*`, `defaults`).
- Enabled streaming by default; added explicit `--no-stream` override.
- Removed legacy `reasoning_mode` strategy path and unified reasoning parsing behavior around provider response shapes.

### Added

- Added unified reasoning control via CLI/config: `reasoning` / `--reasoning`.
  - Supported values: `off|auto|low|medium|high|xhigh|max|budget:<n>`.
- Added OpenAI endpoint selection via `--endpoint` with `chat-completions` and `chat_completions` compatibility.
- Added built-in provider preset discovery via `--list-presets`.
- Added runtime diagnostics:
  - `--dry-run` for resolved execution plan.
  - `--doctor-config` for config validation diagnostics.
- Added live-model-list attempt (`/models`) with static SDK fallback for compatible providers.

### Fixed

- Improved OpenAI-compatible reasoning compatibility by using genai native `extra_body` passthrough and reasoning content capture.
- Improved stream error handling for Responses API provider-side error payloads.
- Reduced noisy usage-deserialization errors when providers return `usage: null`.
- Sorted and deduplicated model list output.

### Docs

- Updated README/README_CN and init templates to reflect v2 configuration, provider presets, endpoint selection, and unified reasoning controls.
