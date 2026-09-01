# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Merged duplicate `liquidai-2.5` preset into `liquidai-lfm25` (same GGUF, kept the 128K/tool-use note).
- Replaced nested `urlopen` in `_download.py` with a `_request_without_range()` helper for clarity.

## [0.1.0] - 2026-08-22

### Added
- Plugin registers the `llama-cpp` provider (`display_name="Llama CPP"`) so a local `llama-server` shows up as a first-class provider in `hermes model` / `/model`.
- Slash command `/llama` and CLI command `hermes llama` for managing the binary and models.
- Subcommands: `check`, `install`, `upgrade`, `uninstall`, `status`, `models`, `pull`, `serve`, `stop`, `help`.
- Cross-platform install: official prebuilt binaries from `ggml-org/llama.cpp` releases (primary) → CMake source build (fallback). No package managers.
- Post-install smoke test (`llama-server --version`) catches incompatible prebuilts (e.g. macOS < 13.3).
- Version metadata (`.version`) with `check` freshness reporting and idempotent `upgrade`.
- Archive caching under `.cache/` so re-install of the same tag is instant.
- Shared-library-aware extraction — the whole extracted tree (binaries + `lib*.dylib/.so/.dll`) installs together.
- Health-polling on `serve` (waits for `GET /health` → 200, up to 60s).
- GGUF download integrity: sha256 verification against upstream `lfs.oid` before `.part` promotion; fails open when no digest is available.
- Download resume: curl `-C -`, urllib `Range` header on `.part` files.
- Model registry with interprocess locking (`.registry.lock`) and PID-unique temp files.
- Corrupt registry preservation (`models.json.corrupt-<pid>`) instead of silent overwrite.
- Archive hardening: tar/zip extraction rejects symlink members and path traversal; extracts per-vetted member.
- Windows support: `msvcrt` locking, `taskkill`/CTRL_BREAK shutdown, ctypes process probes, in-box curl.
- Bundled skill `llama-cpp-local-models` (opt-in, no prompt budget cost).
- CI: GitHub Actions on ubuntu/windows/macos × Python 3.10–3.13, action pinning by SHA, ruff linting.
- 28 unit tests + 1 integration test covering install/uninstall, backend resolution, asset naming, registry concurrency, checksum verification, and the full pull → serve → stop lifecycle.

### Sample models
- LiquidAI: `liquidai` (1.2B, default), `liquidai-350m`, `liquidai-2.6b`, `liquidai-lfm25` (128K-native, reasoning).
- Permissive (Apache-2.0): `qwen2.5-1.5b`, `qwen2.5-coder-1.5b`, `smollm2-1.7b`.
