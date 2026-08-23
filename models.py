"""Local GGUF model management + llama-server lifecycle for hermes-llama.

Model downloads use the Hugging Face ``resolve`` CDN directly (no extra
dependency), with ``huggingface-cli`` as an optional fallback. The server is
launched with an explicit ``--alias`` so ``GET /v1/models`` returns a clean,
stable model id under the "Llama CPP" provider.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from . import install

# Hugging Face endpoint — env-overridable for mirrors (LLAMA_CPP_HF_ENDPOINT).
HF_BASE = os.environ.get("LLAMA_CPP_HF_ENDPOINT", "https://huggingface.co").rstrip("/")

# Liquid AI sample models (verified HF repos — see RESEARCH.md).
# License: "LFM Open License v1.0" (non-commercial research; commercial use
# limited to non-profits and entities < $10M annual revenue). Review before use.
LIQUIDAI_PRESETS = {
    "liquidai": {
        "alias": "liquidai-lfm2-1.2b",
        "repo": "LiquidAI/LFM2-1.2B-GGUF",
        "file": "LFM2-1.2B-Q4_K_M.gguf",
        "size_gb": 0.68,
        "note": "Default sample — best fit for 8 GB RAM, CPU-only.",
    },
    "liquidai-350m": {
        "alias": "liquidai-lfm2-350m",
        "repo": "LiquidAI/LFM2-350M-GGUF",
        "file": "LFM2-350M-Q4_K_M.gguf",
        "size_gb": 0.21,
        "note": "Ultra-light edge model.",
    },
    "liquidai-2.6b": {
        "alias": "liquidai-lfm2-2.6b",
        "repo": "LiquidAI/LFM2-2.6B-GGUF",
        "file": "LFM2-2.6B-Q4_K_M.gguf",
        "size_gb": 1.46,
        "note": "Stronger, slower on 2-core CPU.",
    },
    "liquidai-2.5": {
        "alias": "liquidai-lfm2.5-2.6b",
        "repo": "LiquidAI/LFM2.5-2.6B-GGUF",
        "file": "LFM2.5-2.6B-Q4_K_M.gguf",
        "size_gb": 1.56,
        "note": "Most-downloaded LiquidAI GGUF.",
    },
}


def _model_dest(repo: str, local_name: str) -> Path:
    """Model file location, namespaced by repo to avoid basename collisions."""
    return install.models_dir() / repo.replace("/", "__") / local_name


def _registry_path() -> Path:
    return install.install_root() / install.REGISTRY_FILE_NAME


def _server_pid_path() -> Path:
    return install.install_root() / install.SERVER_PID_FILE_NAME


def _load_registry() -> dict:
    path = _registry_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_registry(reg: dict) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def list_models() -> str:
    lines = ["Presets (not yet downloaded):"]
    for name, p in LIQUIDAI_PRESETS.items():
        lines.append(
            f"  {name:<16} {p['repo']} · {p['file']} (~{p['size_gb']} GB) — {p['note']}"
        )
    reg = _load_registry()
    if reg:
        lines.append("\nDownloaded models:")
        for alias, m in reg.items():
            lines.append(f"  {alias:<16} {m.get('path', '?')} ({m.get('size_gb', '?')} GB)")
    else:
        lines.append("\nDownloaded models: (none)")
    return "\n".join(lines)


def _int_env(name: str, default: int) -> int:
    """Parse an integer env var defensively (fall back to default on garbage)."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _settings() -> dict:
    """Resolve runtime settings: env overrides > defaults."""
    return {
        "host": os.environ.get("LLAMA_CPP_HOST", "127.0.0.1"),
        "port": _int_env("LLAMA_CPP_PORT", 8080),
        "ctx_size": _int_env("LLAMA_CPP_CTX_SIZE", 2048),
        "n_gpu_layers": _int_env("LLAMA_CPP_N_GPU_LAYERS", 0),
        "parallel": _int_env("LLAMA_CPP_PARALLEL", 1),
    }


def _resolve_model(spec: str) -> tuple[str, str, str] | None:
    """Return (repo, file, alias) for a preset name or arbitrary repo.

    Accepts either a preset key (e.g. ``liquidai``) or a Hugging Face repo id
    (``Org/Repo``), in which case the first ``*.gguf`` sibling is used.
    """
    preset = LIQUIDAI_PRESETS.get(spec.lower())
    if preset:
        return preset["repo"], preset["file"], preset["alias"]
    if "/" in spec and not spec.lower().endswith(".gguf"):
        repo = spec
        return repo, "", repo.split("/")[-1]
    return None


def _hf_file_url(repo: str, file: str) -> str:
    return f"{HF_BASE}/{repo}/resolve/main/{file}"


def _pick_gguf_file(repo: str) -> str | None:
    """Return the first GGUF filename in a repo, preferring Q4_K_M."""
    try:
        url = f"{HF_BASE}/api/models/{repo}"
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        files = [s.get("rfilename") for s in data.get("siblings", [])]
        ggufs = [f for f in files if f and f.endswith(".gguf")]
        # Prefer generic builds over vendor-optimized ones (e.g. "-hip-optimized").
        generic = [f for f in ggufs if "hip" not in f.lower() and "optimized" not in f.lower()]
        pool = generic or ggufs
        if not pool:
            return None
        for pref in ("Q4_K_M", "Q4_0", "Q5_K_M", "Q8_0"):
            for f in pool:
                if f.endswith(pref + ".gguf"):
                    return f
        return pool[0]
    except Exception:
        return None


def _download_model(url: str, dest: Path) -> None:
    """Download a model file robustly.

    Prefers ``curl`` when available because Python's ``urllib`` can stall for
    minutes before the first byte arrives against Hugging Face's Xet CDN
    (``us.aws.cdn.hf.co``) redirects. ``curl`` starts the transfer immediately.
    Falls back to ``urllib`` when ``curl`` is absent. Raises on failure.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    curl = shutil.which("curl")
    try:
        if curl:
            proc = subprocess.run(
                [curl, "-L", "--fail", "--retry", "3", "--retry-delay", "2",
                 "--retry-all-errors", "-C", "-",
                 "-A", "hermes-llama", "-o", str(tmp), url],
                capture_output=True, text=True, timeout=7200,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"curl download failed: {proc.stderr.strip()[:300]}")
            expected = None
            try:
                head = subprocess.run([curl, "-sI", "-L", url], capture_output=True, text=True, timeout=20).stdout
                for line in head.splitlines():
                    if line.lower().startswith("content-length:"):
                        expected = int(line.split(":", 1)[1].strip())
                        break
            except Exception:
                pass
            if expected is not None and tmp.stat().st_size != expected:
                raise RuntimeError(f"download truncated: expected {expected} bytes, got {tmp.stat().st_size}")
            os.replace(tmp, dest)
            return
        req = urllib.request.Request(url, headers={"User-Agent": "hermes-llama"})
        with urllib.request.urlopen(req, timeout=3600) as resp, open(tmp, "wb") as out:
            if getattr(resp, "status", 200) != 200:
                raise RuntimeError(f"download failed: HTTP {getattr(resp, 'status', '?')}")
            expected_len = None
            try:
                raw = resp.headers.get("Content-Length") if hasattr(resp, "headers") else None
                if raw:
                    expected_len = int(str(raw).strip())
            except Exception:
                pass
            shutil.copyfileobj(resp, out, length=1024 * 1024)
            out.flush()
            if expected_len is not None and tmp.stat().st_size != expected_len:
                raise RuntimeError(f"download truncated: Content-Length {expected_len}, got {tmp.stat().st_size}")
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def pull(spec: str, alias: str | None = None) -> str:
    """Download a GGUF model into the plugin models dir and register it."""
    resolved = _resolve_model(spec)
    if resolved is None:
        return f"Unknown model spec '{spec}'. Use a preset ({', '.join(LIQUIDAI_PRESETS)}) or Org/Repo."
    repo, file, default_alias = resolved
    if not file:
        file = _pick_gguf_file(repo)
        if not file:
            return f"No .gguf file found in {repo}."
    # HF sibling names may include subdirs; keep only the basename locally so the
    # download lands in models_dir (and a hostile listing can't escape it).
    remote_file = file
    local_name = Path(file).name
    alias = alias or default_alias
    dest = _model_dest(repo, local_name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        # Guard: 0-byte/truncated files are not "present" — re-download them.
        try:
            sz = dest.stat().st_size
        except Exception:
            sz = 0
        if sz < 1024 * 1024:  # <1 MiB is definitely truncated
            dest.unlink(missing_ok=True)
        else:
            # Re-register an existing file (repairs a lost/corrupted registry entry).
            reg = _load_registry()
            reg[alias] = {
                "repo": repo, "file": local_name, "path": str(dest),
                "size_gb": round(sz / 1024**3, 2),
            }
            _save_registry(reg)
            return f"Already present: {dest} (registered as '{alias}')."
    url = _hf_file_url(repo, remote_file)
    try:
        _download_model(url, dest)
    except Exception as exc:
        return f"Download failed: {exc}"
    size_gb = round(dest.stat().st_size / 1024**3, 2)
    reg = _load_registry()
    reg[alias] = {"repo": repo, "file": local_name, "path": str(dest), "size_gb": size_gb}
    _save_registry(reg)
    return f"Downloaded {repo}/{remote_file} ({size_gb} GB) → registered as '{alias}'."


def _pid_alive(pid: int) -> bool:
    """Return True if the given PID is a live process.

    ``os.kill(pid, 0)`` is the POSIX existence probe, but on Windows it raises
    ``OSError(22, 'The parameter is incorrect')`` for processes launched with
    ``CREATE_NEW_PROCESS_GROUP`` (our llama-server). Use ``OpenProcess`` +
    ``GetExitCodeProcess`` there, which is the correct liveness check.
    """
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False
    import ctypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Windows BOOL is a 32-bit int, not ctypes.c_bool (1 byte).
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    alive = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel32.CloseHandle(handle)
    return bool(alive) and code.value == STILL_ACTIVE


def _is_llama_server(pid: int) -> bool:
    """Confirm a PID belongs to llama-server (guard against PID reuse).

    On Windows we open the process and read its executable image name (so a
    reused PID is not mistaken for a live server). On POSIX we prefer
    ``/proc/<pid>/comm`` (Linux, no ``ps`` needed) and fall back to ``ps``.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        try:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.QueryFullProcessImageNameW.argtypes = [
                ctypes.c_void_p, ctypes.c_ulong, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong),
            ]
            kernel32.QueryFullProcessImageNameW.restype = ctypes.c_int
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                buf = ctypes.create_unicode_buffer(32768)
                size = ctypes.c_ulong(len(buf))
                if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    return False
                return Path(buf.value).name.lower() == "llama-server.exe"
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        comm = Path(f"/proc/{pid}/comm")
        if comm.is_file():
            return "llama-server" in comm.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — best-effort read
        pass
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return "llama-server" in out
    except Exception:
        return False


def _find_loaded_server() -> int | None:
    pid_path = _server_pid_path()
    if pid_path.is_file():
        try:
            pid = int(pid_path.read_text().strip())
        except Exception:
            pid = None
        if pid is not None and _pid_alive(pid) and _is_llama_server(pid):
            return pid
        pid_path.unlink(missing_ok=True)
    return None


def _wait_healthy(base: str, timeout: float = 60.0) -> bool:
    """Poll ``/health`` until the server reports ready (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001 — best-effort read
            pass
        time.sleep(1)
    return False


def serve(alias: str) -> str:
    """Launch llama-server for a registered model (or preset)."""
    binary = install.find_binary()
    if binary is None:
        return "llama-server not found. Run `/llama install` first."
    if _find_loaded_server() is not None:
        return "A llama-server is already running. Run `/llama stop` first."

    reg = _load_registry()
    model = reg.get(alias)
    if model is None:
        # Allow serving a preset by its key (e.g. `liquidai`) if already pulled.
        resolved = _resolve_model(alias)
        if resolved is None:
            return (
                f"Unknown model '{alias}'. Run `/llama models` to see presets, "
                f"then `/llama pull {alias}`."
            )
        _repo, file, default_alias = resolved
        file = file or _pick_gguf_file(_repo) or ""
        if not file:
            return f"No .gguf file found in {_repo}."
        path = _model_dest(_repo, Path(file).name)
        if path.is_file():
            # Use the preset's canonical alias so /v1/models matches `pull`'s id.
            model = {"path": str(path), "alias": default_alias}
        else:
            return f"Model '{alias}' is not downloaded yet. Run `/llama pull {alias}` first."

    serve_alias = model.get("alias", alias)
    s = _settings()
    # Explicit --alias keeps the /v1/models id clean and stable.
    cmd = [
        str(binary),
        "--model", model["path"],
        "--alias", serve_alias,
        "--host", s["host"],
        "--port", str(s["port"]),
        "--ctx-size", str(s["ctx_size"]),
        "--n-gpu-layers", str(s["n_gpu_layers"]),
        "--parallel", str(s["parallel"]),
    ]
    # Optional server-side auth: when LLAMA_CPP_API_KEY is set (via plugin
    # config `api_key` or env var), the provider sends it as Bearer and the
    # server must require it, so pass --api-key to llama-server.
    api_key = (os.environ.get("LLAMA_CPP_API_KEY") or "").strip()
    if api_key:
        cmd += ["--api-key", api_key]
    install.install_root().mkdir(parents=True, exist_ok=True)
    log_path = install.install_root() / install.SERVER_LOG_FILE_NAME
    log_file = open(log_path, "ab")
    try:
        if sys.platform == "win32":
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            proc = subprocess.Popen(
                cmd, stdout=log_file, stderr=log_file, start_new_session=True
            )
    except OSError as exc:
        return f"Failed to launch llama-server: {exc}"
    finally:
        log_file.close()
    _server_pid_path().write_text(str(proc.pid), encoding="utf-8")
    # If the server exited immediately (bad flags, missing model, port already
    # bound), don't leave a stale pid file or report "still loading" for a dead
    # process.
    try:
        code = proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        code = None
    if code is not None:
        _server_pid_path().unlink(missing_ok=True)
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-300:].strip()
        except Exception:  # noqa: BLE001 — best-effort read
            pass
        return f"llama-server exited immediately (exit {code}). {tail}"
    base = f"http://{s['host']}:{s['port']}"
    ready = _wait_healthy(base, timeout=float(_int_env("LLAMA_CPP_HEALTH_TIMEOUT", 60)))
    state = "ready" if ready else "still loading (watch `/llama status`)"
    return (
        f"Started llama-server (pid {proc.pid}) with model '{serve_alias}' on "
        f"{base}/v1 — {state}. It will appear as provider 'Llama CPP'."
    )


def stop() -> str:
    pid = _find_loaded_server()
    if pid is None:
        return "No llama-server running."
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:  # noqa: BLE001 — PID already gone
                pass
            # Poll briefly before SIGKILL; avoid killing a reused PID group.
            for _ in range(10):
                time.sleep(0.2)
                if not _pid_alive(pid) or not _is_llama_server(pid):
                    break
            else:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except Exception:  # noqa: BLE001 — PID already gone or reused
                    pass
    except Exception as exc:
        return f"Failed to stop pid {pid}: {exc}"
    _server_pid_path().unlink(missing_ok=True)
    return f"Stopped llama-server (pid {pid})."


def status() -> str:
    s = _settings()
    base = f"http://{s['host']}:{s['port']}"
    lines = [f"Endpoint: {base}/v1  (provider 'Llama CPP')"]
    binary = install.find_binary()
    lines.append(f"llama-server: {'installed' if binary else 'NOT installed'} ({binary or '-'})")
    pid = _find_loaded_server()
    lines.append(f"server running: {'yes (pid %s)' % pid if pid else 'no'}")
    if pid:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
                lines.append(f"health: HTTP {resp.status} {resp.read().decode().strip()}")
        except Exception:
            lines.append("health: unreachable (model may still be loading)")
    return "\n".join(lines)
