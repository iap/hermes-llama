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
# 1. The general plugin (slash + CLI commands)
hermes plugins install iap/hermes-llama --enable

# 2. The provider profile (guaranteed picker auto-injection via standard discovery)
mkdir -p "$HERMES_HOME/plugins/model-providers/llama-cpp"
cp model-provider/llama-cpp/__init__.py model-provider/llama-cpp/plugin.yaml \
   "$HERMES_HOME/plugins/model-providers/llama-cpp/"
```

> `$HERMES_HOME` is `~/.hermes` (POSIX/WSL) or `%LOCALAPPDATA%\hermes` (Windows);
> run `hermes config path` to confirm. Step 2 is optional — the general plugin
> also self-registers the provider at load — but it is the reliable way to
> guarantee "Llama CPP" appears in the picker regardless of load order.

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

## Sample model — LiquidAI

| Alias | Repo / file | Size | Note |
|---|---|---|---|
| `liquidai` | `LiquidAI/LFM2-1.2B-GGUF` · `LFM2-1.2B-Q4_K_M.gguf` | 0.68 GB | **Default** — best fit for 8 GB RAM, CPU-only |
| `liquidai-350m` | `LiquidAI/LFM2-350M-GGUF` · `LFM2-350M-Q4_K_M.gguf` | 0.21 GB | Ultra-light edge model |
| `liquidai-2.6b` | `LiquidAI/LFM2-2.6B-GGUF` · `LFM2-2.6B-Q4_K_M.gguf` | 1.46 GB | Stronger, slower on a 2-core CPU |
| `liquidai-2.5` | `LiquidAI/LFM2.5-2.6B-GGUF` · `LFM2.5-2.6B-Q4_K_M.gguf` | 1.56 GB | Most-downloaded LiquidAI GGUF |

> **License:** LiquidAI models use the **LFM Open License v1.0** — non-commercial
> research use; commercial use limited to non-profits and entities under
> US$10M annual revenue. Review `LICENSE` in the model repo before commercial use.

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
| `LLAMA_CPP_GITHUB_BASE` | `https://github.com` | GitHub download/clone base (for GitHub Enterprise) |
| `LLAMA_CPP_GITHUB_API_BASE` | `https://api.github.com` | GitHub API base (for GitHub Enterprise) |
| `LLAMA_CPP_HF_ENDPOINT` | `https://huggingface.co` | Hugging Face endpoint (for mirrors) |

> The same keys are also settable as Hermes plugin settings
> (`hermes config` → `plugins.entries.hermes-llama.settings.*`, mirroring the
> `config_schema` in `plugin.yaml`). An explicitly-set `LLAMA_CPP_*` environment
> variable takes precedence over the Hermes setting.

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
├── model-provider/llama-cpp/  # standalone provider plugin (optional companion)
├── skills/llama-cpp-local-models/SKILL.md
├── RESEARCH.md              # source-backed findings + cross-check
└── OPTIMIZATIONS.md         # design decisions + optimization analysis
```

## License

MIT. LiquidAI model weights are governed by their own LFM Open License v1.0.
