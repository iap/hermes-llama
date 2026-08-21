# hermes-llama — Research & Cross-check Notes

> **Design note (2026-08-22):** the plugin no longer uses package managers —
> install is self-contained (prebuilt binaries + CMake source build only). The
> package-manager facts below are retained as the source-backed evidence behind
> that decision and for anyone extending the install matrix.

All facts below were verified during this task against primary sources (GitHub
raw content / releases API, Homebrew formula site, Hugging Face API). Confidence
levels follow the review convention: **High** (≥2 independent sources),
**Medium** (one authoritative source), **Low** (weak/unverified), **Conflict**.

## 1. llama.cpp — repo & install

| Fact | Confidence | Source |
|---|---|---|
| Repo is `ggml-org/llama.cpp`; old `ggerganov` org 301-redirects here; default branch `master` | High | https://github.com/ggml-org/llama.cpp (API + redirect probe) |
| macOS/Linux Homebrew formula is literally `llama.cpp`: `brew install llama.cpp` | High | https://formulae.brew.sh/formula/llama.cpp ; https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md |
| Official package routes = conda-forge, Winget, Homebrew, MacPorts, Nix; **no apt/pacman/choco** | High | https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md |
| Windows Winget package id `ggml.llamacpp` (`winget install llama.cpp`) | High | https://github.com/microsoft/winget-pkgs (manifest `ggml.llamacpp`) ; https://winstall.app/apps/ggml.llamacpp |
| Prebuilt binaries published per release; tag `b10516`; asset pattern `llama-<tag>-bin-<os>[-<backend>]-<arch>.{tar.gz,zip}` | High | https://github.com/ggml-org/llama.cpp/releases (live releases API) |
| Build is CMake-only (`cmake -B build && cmake --build build --config Release`) | High | https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md |
| Intel macOS = CPU-only (Accelerate BLAS); Apple Silicon = Metal default | High | docs/build.md (Metal + Accelerate sections), releases asset names |
| **Current macOS prebuilt requires macOS 13.3+** (`minos 13.3`, SDK 15.5); hangs on macOS 12 and older — use brew/conda/source build | High | live probe: `otool -l llama-server` → `LC_BUILD_VERSION minos 13.3` on this macOS 12.7.6 host |
| Uninstall is **not** documented officially (derived per-method commands) | High (of the absence) | searched docs/install.md, docs/build.md, README — no uninstall section |

## 2. llama-server — flags & OpenAI-compatible endpoints

| Fact | Confidence | Source |
|---|---|---|
| `-m/--model`, `--host` (127.0.0.1), `--port` (8080), `-c/--ctx-size`, `-ngl/--n-gpu-layers`, `-np/--parallel`, `-a/--alias`, `--api-key`, `-hf/--hf-repo`, `-hff/--hf-file` — all real, exact spellings | High | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md |
| `GET /health` public; `200 {"status":"ok"}` when loaded, `503` while loading | High | tools/server/README.md |
| `GET /v1/models` → OpenAI shape `{"object":"list","data":[{"id":…,"object":"model",…}]}` | High | tools/server/README.md |
| **Model `id` = the exact `-m` path string**, NOT the filename stem; override via `--alias` | High (corrects an earlier assumption) | tools/server/README.md (documented example id `../models/…gguf`) |
| `POST /v1/chat/completions` OpenAI-compatible | High | tools/server/README.md |

## 3. Liquid AI — models & GGUF

| Fact | Confidence | Source |
|---|---|---|
| HF org `LiquidAI`; family "Liquid Foundation Models" (LFM2 / LFM2.5) | High | https://huggingface.co/LiquidAI |
| Official GGUF repos exist: `LiquidAI/LFM2-350M-GGUF`, `-1.2B-GGUF`, `-2.6B-GGUF`, `LFM2.5-2.6B-GGUF` | High | https://huggingface.co/api/models/<id> (HTTP 200 each) |
| Quant files: `F16/Q4_0/Q4_K_M/Q5_K_M/Q6_K/Q8_0` (+ a `-hip-optimized` AMD build) | High | HF API sibling listing per repo |
| `LFM2-1.2B-Q4_K_M.gguf` = 730,893,248 bytes (0.68 GB); `LFM2-2.6B-Q4_K_M` = 1.46 GB; `LFM2-350M-Q4_K_M` = 0.21 GB; `LFM2.5-2.6B-Q4_K_M` = 1.56 GB | High | HF `resolve` CDN content-length |
| License = **"LFM Open License v1.0"** (custom; non-commercial research + commercial ≤ US$10M revenue / non-profit) | High | https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF/blob/main/LICENSE (raw fetch) |
| Fit for 8 GB RAM / 2-core Intel CPU → **LFM2-1.2B Q4_K_M (0.68 GB)** default | Medium (engineering judgment; sizes are High) | size data above + RAM budget reasoning |

## Cross-check / corrections captured

1. **Correction:** model id is the `-m` path, not the filename stem — the plugin
   therefore always passes an explicit `--alias` for a clean, stable id.
2. **Correction:** `_pick_gguf_file` must skip `-hip-optimized` builds (AMD GPU)
   when choosing a default CPU file.
3. Org/URL must be `ggml-org` (not `ggerganov`); branch `master` (not `main`).
4. Prebuilt tags are build numbers (`b10516`), not semver — asset URL must use the
   live `releases/latest` tag, which the installer resolves dynamically.

## Gaps

- Exact `llama-server --version` output string not captured (version check is
  best-effort, reports the first output line).
- Tarball/zip internal layout not verified by an actual download (the installer
  walks the extracted tree to locate `llama-server`, so layout variance is handled).
- Winget manifest (`b10507`) trails the latest release (`b10516`) at research time.
