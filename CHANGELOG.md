# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

