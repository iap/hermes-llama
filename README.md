# hermes-llama

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that checks,
installs, and uninstalls **llama.cpp** on macOS / Windows / Linux **without any
package manager**, and runs local **GGUF** models through the **"Llama CPP"**
provider. The sample model is **LiquidAI LFM2**, sized for the host machine.

## Design: no package managers

Installation uses exactly two self-contained paths, so behaviour is identical
across platforms and needs no admin privileges or pre-installed toolchain:

1. **Official prebuilt binaries** from `ggml-org/llama.cpp` releases (primary).
2. **CMake source build** (fallback when no prebuilt matches the host).

No Homebrew, Winget, conda, Nix, or MacPorts. Everything lives under
`$HERMES_HOME/llama-cpp/` (portable, per-user).

## What it does

1. **Manages `llama-server`** (llama.cpp's OpenAI-compatible server):
   `check`, `install`, `upgrade`, `uninstall`.
2. **Registers provider `llama-cpp`** (`display_name = "Llama CPP"`) pointing at
   the local server (`http://127.0.0.1:8080/v1` by default).
3. **Downloads and serves GGUF models** — a LiquidAI preset plus any
   `Org/Repo` from Hugging Face.
4. Models loaded in `llama-server` appear under **"Llama CPP"** in `hermes model`,
   `/model`, and the local model list (e.g. `127.0.0.1:9119/models`).

## Install

```bash
# Install and enable the plugin (slash + CLI commands + the "Llama CPP" provider)
hermes plugins install iap/hermes-llama --enable
```

> `$HERMES_HOME` is `~/.hermes` (POSIX/WSL) or `%LOCALAPPDATA%\hermes` (Windows);
> run `hermes config path` to confirm. The plugin self-registers the "Llama CPP"
> provider at load, so no separate provider install step is needed.

## Usage

| Command | Action |
|---|---|
| `/llama check` | Is `llama-server` installed? Which version / release tag? Up to date? |
| `/llama install` | Install llama.cpp (prebuilt, then source fallback) |
| `/llama upgrade` | Reinstall to the latest release |
| `/llama uninstall` | Remove the plugin-managed install |
| `/llama status` | Endpoint + health of the running server |
| `/llama models` | List LiquidAI presets + downloaded GGUFs |
| `/llama pull liquidai` | Download the sample model (LiquidAI LFM2-1.2B Q4_K_M) |
| `/llama serve liquidai` | Start `llama-server` and wait for `/health` |
| `/llama stop` | Stop the server |

The same subcommands are available in a terminal as `hermes llama <sub> …`.

## Operations & file layout

Everything lives under `$HERMES_HOME/llama-cpp/`:

| Path | Purpose |
|---|---|
| `bin/llama-server` | The installed server binary (prebuilt or source-built) |
| `models/<org>__<repo>/<file>.gguf` | Downloaded GGUF weights (never deleted by uninstall) |
| `models.json` | Registry of pulled models: alias → path, size, sha256 when known |
| `server.log` | llama-server stdout/stderr (append mode; check here first when serve fails) |
| `server.pid` | PID file for the running server (removed on stop) |
| `.version` | Install metadata: tag/method/backend/commit — powers `check` freshness |
| `.cache/source_head.json`, `.cache/tag.json` | Upstream freshness caches (600 s TTL) |
| `.install.lock`, `.registry.lock` | Interprocess locks (install vs model pulls) |

**Ports:** llama-server listens on `127.0.0.1:8080/v1` (configurable). The Hermes
dashboard's model list at `127.0.0.1:9119/models` is a *different* service — it
shows the "Llama CPP" provider entry but does not proxy to the server.

**Exit codes:** `hermes llama …` subcommands print a human-readable result and do
not set a meaningful exit status yet; scripts should parse output, not `$?`.

## Sample models

| Alias | Repo / file | Size | Licence | Note |
|---|---|---|---|---|
| `liquidai` | `LiquidAI/LFM2-1.2B-GGUF` · `LFM2-1.2B-Q4_K_M.gguf` | 0.68 GB | LFM v1.0 | **Default** — best fit for 8 GB RAM, CPU-only |
| `liquidai-350m` | `LiquidAI/LFM2-350M-GGUF` · `LFM2-350M-Q4_K_M.gguf` | 0.21 GB | LFM v1.0 | Ultra-light edge model |
| `liquidai-2.6b` | `LiquidAI/LFM2-2.6B-GGUF` · `LFM2-2.6B-Q4_K_M.gguf` | 1.46 GB | LFM v1.0 | Stronger, slower on a 2-core CPU |
| `liquidai-2.5` | `LiquidAI/LFM2.5-2.6B-GGUF` · `LFM2.5-2.6B-Q4_K_M.gguf` | 1.56 GB | LFM v1.0 | Most-downloaded LiquidAI GGUF |
| `liquidai-lfm25` | `LiquidAI/LFM2.5-2.6B-GGUF` · `LFM2.5-2.6B-Q4_K_M.gguf` | 1.56 GB | LFM v1.0 | **128K-native context, tool-use trained** — the only preset that can clear Hermes' agent floor (see context note below) |
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` · `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1.04 GB | Apache-2.0 | Permissive general instruct; tool-calling via `--jinja` |
| `qwen2.5-coder-1.5b` | `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` · `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | 1.04 GB | Apache-2.0 | Permissive code-focused instruct |
| `smollm2-1.7b` | `bartowski/SmolLM2-1.7B-Instruct-GGUF` · `SmolLM2-1.7B-Instruct-Q4_K_M.gguf` | 0.98 GB | Apache-2.0 | Smallest permissive option; chat-oriented |

> **Licences differ.** LiquidAI models use the **LFM Open License v1.0** —
> non-commercial research use; commercial use limited to non-profits and
> entities under US$10M annual revenue. The Qwen2.5 and SmolLM2 presets are
> **Apache-2.0** and carry no such restriction. Review the `LICENSE` in the
> model repo before commercial use.

> **Tool-calling.** `--jinja` is passed by default, which loads the model's own
> chat template and enables OpenAI-style `tools` / `tool_calls`. At the 1.5B
> scale, `tool_choice: "required"` is far more reliable than `"auto"` — a small
> model given `"auto"` will often answer in prose instead of emitting a call.

> **Context window vs the Hermes agent floor — read before switching models.**
> Hermes Agent refuses any main-model whose context window is below
> **64,000 tokens**, and the window it sees is the server's *allocated*
> `--ctx-size` (default **2048**), not the model's training maximum:
>
> - The dashboard/agent error "context window of 2,048 … below the minimum
>   64,000" means `LLAMA_CPP_CTX_SIZE` is too low, **not** that the model is
>   too small.
> - Raise it via plugin settings: `hermes config set`
>   `plugins.entries.hermes-llama.settings.ctx_size 32768` (then restart the
>   server). RAM is the limit — on an 8 GB host, 32768 with q8_0 KV runs a
>   1.5–2.6 B model at ~8 tok/s; 65536 thrashes swap and drops to ~0.06 tok/s.
> - True ceilings differ per model: Qwen2.5-1.5B GGUFs cap at 32 K,
>   SmolLM2 at 8 K, LFM2.5 at 128 K (32 K practical here).
> - Per-model overrides live in `model_overrides.<provider>.<model>.context_window`
>   in config.yaml; set them to the value you actually run so the agent's
>   planner doesn't overfill the window.
> - **LFM2.5 is a reasoning model**: short replies land in
>   `reasoning_content` first — a small `max_tokens` looks like an empty answer.

## Configuration

Environment variables (override plugin defaults):

| Variable | Default | Meaning |
|---|---|---|
| `LLAMA_CPP_BASE_URL` | `http://127.0.0.1:8080/v1` | Provider base URL |
| `LLAMA_CPP_HOST` | `127.0.0.1` | `llama-server --host` |
| `LLAMA_CPP_PORT` | `8080` | `llama-server --port` |
| `LLAMA_CPP_CTX_SIZE` | `2048` | `--ctx-size` (tokens) |
| `LLAMA_CPP_N_GPU_LAYERS` | `0` | `--n-gpu-layers` (0 = CPU-only) |
| `LLAMA_CPP_PARALLEL` | `1` | `--parallel` slots |
| `LLAMA_CPP_BACKEND` | `auto` | `auto` / `cpu` / `cuda` / `vulkan` / `source` (prebuilt selection) |
| `LLAMA_CPP_VERSION` | *(latest)* | Pin a release tag (e.g. `b10549`) |
| `LLAMA_CPP_INSTALL_DIR` | `$HERMES_HOME/llama-cpp` | Root dir for the plugin-managed install |
| `LLAMA_CPP_MODELS_DIR` | `$HERMES_HOME/llama-cpp/models` | Where GGUF models are stored |
| `LLAMA_CPP_API_KEY` | *(empty)* | Optional `--api-key` / Bearer |
| `LLAMA_CPP_THREADS` | *(physical cores)* | `--threads` / `--threads-batch`; `0` = let llama.cpp choose |
| `LLAMA_CPP_CACHE_TYPE_K` | `q8_0` | `--cache-type-k` (`f16` for max quality, `q8_0` halves cache RAM) |
| `LLAMA_CPP_CACHE_TYPE_V` | `q8_0` | `--cache-type-v` |
| `LLAMA_CPP_JINJA` | `true` | Pass `--jinja` (loads the model's chat template; needed for tool-calling) |
| `LLAMA_CPP_GITHUB_BASE` | `https://github.com` | GitHub download/clone base (for GitHub Enterprise) |
| `LLAMA_CPP_GITHUB_API_BASE` | `https://api.github.com` | GitHub API base (for GitHub Enterprise) |
| `LLAMA_CPP_HF_ENDPOINT` | `https://huggingface.co` | Hugging Face endpoint (for mirrors) |
| `LLAMA_CPP_SMOKE_TIMEOUT` | `8` | Timeout (seconds) for `llama-server --version` smoke test on install. Increase on slow storage. |
| `LLAMA_CPP_HEALTH_TIMEOUT` | `60` | Timeout (seconds) for `/health` polling when serving. |

> The schema keys (`base_url`, `host`, `port`, `ctx_size`, `n_gpu_layers`,
> `parallel`, `install_dir`, `backend`, `version`, `models_dir`, `api_key`,
> `github_base`, `threads`, `cache_type_k`, `cache_type_v`, `jinja`) are
> also settable as Hermes plugin settings (`hermes config` →
> `plugins.entries.hermes-llama.settings.*`, mirroring `config_schema` in
> `plugin.yaml`). The remaining mirror overrides
> (`LLAMA_CPP_GITHUB_API_BASE`, `LLAMA_CPP_HF_ENDPOINT`) are env-only.
> An explicitly-set `LLAMA_CPP_*` environment variable takes precedence over the
> Hermes setting.

## Platform notes

- **macOS Apple Silicon** → Metal GPU by default (`--n-gpu-layers` > 0 to offload).
- **macOS Intel** → CPU-only (Accelerate BLAS). Note: the current macOS
  prebuilt is built with `minos 13.3`, so on macOS 12 or older it won't launch —
  the installer detects this and falls back to a source build.
- **Windows** → `*-bin-win-cpu-x64.zip` (or `-cuda-`/`-vulkan-` when
  `LLAMA_CPP_BACKEND` is set / an NVIDIA GPU is detected).
- **Linux** → `*-bin-ubuntu-{x64,arm64}.tar.gz` (or `-vulkan-`). CUDA has no
  Linux prebuilt, so `backend=cuda` on Linux triggers a source build.
- Source build needs CMake (available via `python3 -m pip install cmake`).

## Layout

```
hermes-llama/
├── plugin.yaml              # general plugin manifest
├── __init__.py              # register(ctx): /llama + hermes llama + provider
├── provider.py              # LlamaCppProfile ("Llama CPP")
├── install.py               # check / install / upgrade / uninstall (no pkg mgrs)
├── models.py                # GGUF pull / serve / stop / status / list
├── skills/llama-cpp-local-models/SKILL.md
├── RESEARCH.md              # source-backed findings + cross-check
└── OPTIMIZATIONS.md         # design decisions + optimization analysis
```

## License

MIT. LiquidAI model weights are governed by their own LFM Open License v1.0.
