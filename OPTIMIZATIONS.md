# hermes-llama — Optimization analysis

Request: drop package-manager installation; determine the optimal cross-platform
design and the highest-value efficiency/effectiveness improvements.

## 1. Design decision — drop package managers

Removed: Homebrew, Winget, conda-forge, Nix, MacPorts.

Rationale — package managers are the *least* portable layer of the install path:

| Concern | Package manager | Prebuilt binary |
|---|---|---|
| Requires admin / system install | often (system-wide) | no (user dir) |
| Version pinned to distro/vendor, not upstream | yes | no (exact release tag) |
| Present on the host? | varies (this host had none) | always via HTTPS |
| Identical across macOS/Win/Linux | no (brew/winget/apt differ) | yes (one code path) |
| Offline/portable | no | yes (self-contained under `$HERMES_HOME`) |

**Resulting install path:** official prebuilt binaries from GitHub releases
(primary) → CMake source build (fallback). One code path, zero system deps,
no admin, portable per-user.

## 2. Optimizations implemented

1. **Backend-aware asset selection** — the right prebuilt is picked for the
   host arch/OS: CPU by default; NVIDIA auto-detect on Windows → CUDA;
   explicit `LLAMA_CPP_BACKEND` (`cpu|cuda|vulkan`) overrides. Avoids installing
   a CPU build on a GPU machine and vice versa.

2. **Post-install runtime verification** — after install, `llama-server --version`
   is smoke-tested. Catches the macOS `< 13.3` prebuilt incompatibility (the
   binary hangs on launch) and any corrupt/broken download, so the plugin never
   reports a false "installed".

3. **Version metadata + `upgrade`** — install writes the release tag to
   `.version`; `check` reports `tag`, `latest_tag`, and `up_to_date`; `upgrade`
   is idempotent reinstall-to-latest. Turns a one-shot installer into a managed
   tool with an upgrade path.

4. **Archive caching** — the release tarball/zip is cached under
   `$HERMES_HOME/llama-cpp/.cache/<tag>-<asset>`; re-install of the same tag is
   instant (no re-download).

5. **Shared-library-aware extraction** — the whole extracted tree (binaries +
   `lib*.dylib/.so/.dll`) is installed together, fixing the `@rpath`
   `libllama-server-impl` load failure on macOS (found via live test).

6. **Health-polling on serve** — `serve` waits for `GET /health` to return 200
   (up to 30 s) and reports "ready" vs "still loading", instead of returning
   before the model is loaded.

7. **Stdlib-only, zero runtime deps** — `urllib`, `tarfile`, `zipfile`,
   `subprocess`, `json` only. No pip packages required at runtime, so the plugin
   works everywhere Python runs.

## 3. Considered but deferred (with reason)

| Idea | Status | Why deferred |
|---|---|---|
| Checksum verification of the binary | deferred | Releases publish **no** `.sha256`/checksums assets (verified). Nothing to verify against; would be fake assurance. |
| `llama-server --hf-repo` native download | deferred | Removes the local registry + resume control; `pull` already gives predictable local files. Left as a documented alternative. |
| HF sha256 for model files | deferred | `GET /api/models/<id>` returns `lfs: null` for these repos, so no checksum to compare. Prefer `huggingface-cli download` (does its own integrity check) when available. |
| Parallel/chunked downloads | deferred | urllib streaming is simple and reliable; add `huggingface-cli` if high-bandwidth resume is needed. |
| GPU-layer auto-detection for serve | deferred | Backend is detected at install; `--n-gpu-layers` is a runtime knob the user sets (`LLAMA_CPP_N_GPU_LAYERS`). |
| Uninstall of system-wide installs | removed | By design there are none — the plugin only ever touches `$HERMES_HOME/llama-cpp/`. |

## 4. Remaining gaps / risks

- **macOS < 13.3** has no working prebuilt; the only path is a source build
  (needs CMake + a few minutes). This is inherent to upstream's build matrix,
  not the plugin.
- **Linux CUDA** has no prebuilt asset → source build fallback (correct, but slow).
- No end-to-end `pull → serve` was exercised on this host (it lacks a runnable
  llama-server); that path is implemented + flag-verified but not live-tested here.
