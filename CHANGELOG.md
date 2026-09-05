# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-09-05

### Fixed
- `_download.py`: stock `urllib` raises `HTTPError` for every non-2xx status, so the old `resp.status == 416` branch was unreachable and a `.part` that had already finished downloading (promotion interrupted) failed the whole download. HTTP 416 is now handled as an exception: `Content-Range` carries the true total, a `.part` already at that size is verified and promoted, anything else is discarded and the download restarts without the Range header.
- `_registry_txn()`: the unlocked fallback now guards only lock acquisition, so a `RuntimeError` raised inside a transaction body propagates (and skips the save) instead of being swallowed into a confusing unlocked re-entry.

### Changed
- Dashboard sync skips the `config.yaml` rewrite when the "Llama CPP" rows are already up to date — plugin registration no longer rewrites the file on every Hermes start.
- Docs: corrected the `/llama reinstall` reference in the bundled skill to `/llama upgrade`; subcommand hints (`args_hint`, CLI help) now include `upgrade`; CONTRIBUTING documents how to run the tests.

## [0.2.0] - 2026-09-04

### Added
- `liquidai-lfm25` preset (LFM2.5-2.6B, 128K-native context, tool-use trained) with agent-floor guidance; `liquidai-2.5` shipped as a deprecated alias for it.
- Dashboard visibility: downloaded models are synced into the config's `custom_providers` entry and `providers` map (scoped to the "Llama CPP" row, matched by URL) so they appear in the web model list.
- Integration tests (mocked pull → serve → stop lifecycle) run in CI, plus cross-platform test hardening (Windows curl stubs, win/mac runners, recursive sha256).

### Changed
- Merged duplicate `liquidai-2.5` preset into `liquidai-lfm25` (same GGUF, kept the 128K/tool-use note).
- Registry entries resolve by key when the `alias` field is absent; legacy absolute model paths migrate to models_dir-relative form on load.
- `_expected_sha256` follows HF tree pagination (`Link: rel="next"` / cursor) so large repos still resolve digests.
- Replaced nested `urlopen` in `_download.py` with a `_request_without_range()` helper for clarity.

### Fixed
- Audit hardening: PID-reuse guards, graceful shutdown (SIGTERM → SIGKILL on POSIX, CTRL_BREAK → `taskkill` on Windows), download resume, ETag-cached freshness probes.
- Mirror endpoint is reflected into the config `providers` map so the dashboard lists the provider.
- `_load_registry` no longer writes during lock-free reads.
- Resolved 5 open CodeQL alerts (`provider.py`, `install.py`, tests).

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
