---
name: llama-cpp-local-models
description: >-
  Use when the user wants to install or remove llama.cpp, run a local GGUF
  model (e.g. LiquidAI LFM2) through the "Llama CPP" provider, download a GGUF
  from Hugging Face, or start/stop a local llama-server. Also use to interpret
  "Llama CPP" model listings and diagnose a local model server that is down.
---

# llama-cpp-local-models

How to run local GGUF models through the `hermes-llama` plugin and the
**"Llama CPP"** provider in Hermes Agent.

## Commands

- `/llama check` — is `llama-server` installed, and which version?
- `/llama install` — install llama.cpp for this OS (official prebuilt binaries
  from `ggml-org/llama.cpp` releases, or a CMake source build — no package
  managers).
- `/llama upgrade` — reinstall to the latest release.
- `/llama uninstall` — remove it.
- `/llama status` — endpoint + health of the running server.
- `/llama models` — list LiquidAI presets and already-downloaded GGUFs.
- `/llama pull liquidai` — download the default sample (LiquidAI LFM2-1.2B Q4_K_M).
- `/llama serve liquidai` — start `llama-server` for a downloaded model.
- `/llama stop` — stop the server.

## How the model appears under "Llama CPP"

1. The plugin registers provider `llama-cpp` (`display_name="Llama CPP"`),
   whose `base_url` is the local `llama-server` (default `http://127.0.0.1:8080/v1`).
2. `llama-server` exposes OpenAI-compatible `GET /v1/models`, which returns the
   loaded model's `id`. The plugin starts the server with an explicit `--alias`,
   so the id is clean and stable.
3. Hermes lists those ids under the "Llama CPP" provider in `hermes model`,
   `/model`, and the local model list (127.0.0.1:9119/models).

## Troubleshooting

- Server not listed / empty model list → run `/llama status`; the model may
  still be loading (health returns 503 until ready).
- Port conflict → set `LLAMA_CPP_PORT` (and `LLAMA_CPP_HOST`).
- Out of memory → use a smaller quant (Q4_K_M) or smaller model (350M/1.2B),
  and lower `LLAMA_CPP_CTX_SIZE`.
- "context window … below the minimum 64,000" → the server's `--ctx-size` is
  below Hermes' agent floor. Raise it via plugin settings
  (`plugins.entries.hermes-llama.settings.ctx_size`, e.g. 32768) and restart.
  RAM, not the model's training max, is usually what limits you: on 8 GB run
  ~32768 for 1.5–2.6 B models; 65536 thrashes swap.
- Empty-looking reply → LFM2.5-class reasoning models emit `reasoning_content`
  before `content`; raise `max_tokens` (≥512) or check the reasoning field.
- Per-model context overrides → `model_overrides.<provider>.<model>.context_window`
  in config.yaml should match the value you actually serve.
- GPU offload → set `LLAMA_CPP_N_GPU_LAYERS` (0 = CPU-only; on Apple Silicon or
  CUDA, a positive number offloads layers).
- Binary precedence → `find_binary()` checks the plugin-managed `bin/` directory
  first, then PATH. If the managed binary is broken, it won't fall back to a
  system copy — run `/llama upgrade` to replace it.
