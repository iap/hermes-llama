"""Local GGUF model management + llama-server lifecycle for hermes-llama.

Model downloads use the Hugging Face ``resolve`` CDN directly (no extra
dependency), with ``huggingface-cli`` as an optional fallback. The server is
launched with an explicit ``--alias`` so ``GET /v1/models`` returns a clean,
stable model id under the "Llama CPP" provider.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import install

# Hugging Face endpoint — env-overridable for mirrors (LLAMA_CPP_HF_ENDPOINT).
# Read at call time so _wire_config-set values apply after import.


def _hf_base() -> str:
    return os.environ.get("LLAMA_CPP_HF_ENDPOINT", "https://huggingface.co").rstrip("/")

# Bundled sample models (verified HF repos — see RESEARCH.md).
#
# Two licence families ship here, deliberately:
#   * LiquidAI LFM2 — "LFM Open License v1.0": non-commercial research; commercial
#     use limited to non-profits and entities < $10M annual revenue.
#   * Qwen2.5 / SmolLM2 — Apache-2.0: permissive, commercial use allowed, and
#     tool-calling capable (pair with llama-server ``--jinja``).
# Sizes are the real GGUF byte sizes from the HF tree API, not estimates.
LIQUIDAI_PRESETS = {
    "liquidai": {
        "alias": "liquidai-lfm2-1.2b",
        "repo": "LiquidAI/LFM2-1.2B-GGUF",
        "file": "LFM2-1.2B-Q4_K_M.gguf",
        "size_gb": 0.68,
        "note": "Default sample — best fit for 8 GB RAM, CPU-only. LFM licence.",
    },
    "liquidai-350m": {
        "alias": "liquidai-lfm2-350m",
        "repo": "LiquidAI/LFM2-350M-GGUF",
        "file": "LFM2-350M-Q4_K_M.gguf",
        "size_gb": 0.21,
        "note": "Ultra-light edge model. LFM licence.",
    },
    "liquidai-2.6b": {
        "alias": "liquidai-lfm2-2.6b",
        "repo": "LiquidAI/LFM2-2.6B-GGUF",
        "file": "LFM2-2.6B-Q4_K_M.gguf",
        "size_gb": 1.46,
        "note": "Stronger, slower on 2-core CPU. LFM licence.",
    },
    "liquidai-2.5": {
        "alias": "liquidai-lfm2.5-2.6b",
        "repo": "LiquidAI/LFM2.5-2.6B-GGUF",
        "file": "LFM2.5-2.6B-Q4_K_M.gguf",
        "size_gb": 1.56,
        "note": "Most-downloaded LiquidAI GGUF. LFM licence.",
    },
    "qwen2.5-1.5b": {
        "alias": "qwen2.5-1.5b-instruct",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_gb": 1.04,
        "note": "Apache-2.0 general instruct. Tool-calling works via --jinja; at 1.5B "
                "prefer tool_choice='required' — 'auto' often answers in prose instead.",
    },
    "qwen2.5-coder-1.5b": {
        "alias": "qwen2.5-coder-1.5b-instruct",
        "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "file": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "size_gb": 1.04,
        "note": "Apache-2.0 code-focused instruct. Same tool-calling caveat as qwen2.5-1.5b.",
    },
    "smollm2-1.7b": {
        "alias": "smollm2-1.7b-instruct",
        "repo": "bartowski/SmolLM2-1.7B-Instruct-GGUF",
        "file": "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        "size_gb": 0.98,
        "note": "Apache-2.0, smallest permissive option here. Chat-oriented — "
                "not recommended for tool-calling.",
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
    """Read the model registry.

    A missing file is normal (nothing pulled yet) and yields ``{}``. A file that
    exists but does not parse is NOT treated as empty: it is preserved beside the
    original as ``models.json.corrupt-<pid>`` before returning ``{}``, so a
    subsequent save cannot silently destroy recoverable entries.
    """
    path = _registry_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — corrupt/unreadable registry
        try:
            path.replace(path.with_suffix(f".json.corrupt-{os.getpid()}"))
        except Exception:  # noqa: BLE001 — best-effort preservation
            pass
        return {}
    return data if isinstance(data, dict) else {}


def _save_registry(reg: dict) -> None:
    """Atomically write the registry via a PID-unique temp file.

    A fixed temp name races when two processes save concurrently (both write the
    same ``.tmp`` then both ``os.replace``); the unique suffix removes that.
    Callers mutating the registry should hold ``_registry_txn()``.
    """
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".json.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


@contextmanager
def _registry_txn():
    """Serialize a registry read-modify-write across processes.

    Yields the current registry dict; on clean exit it is written back inside the
    same lock, so concurrent ``pull`` calls cannot lose each other's entries.
    Falls back to an unlocked transaction when the lock cannot be taken, since
    losing a registry entry is preferable to refusing to record a finished
    multi-GiB download.
    """
    try:
        with install.registry_lock():
            reg = _load_registry()
            yield reg
            _save_registry(reg)
            return
    except RuntimeError:
        pass
    reg = _load_registry()
    yield reg
    _save_registry(reg)


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


def _str_env(name: str, default: str) -> str:
    """Read a string env var, falling back to *default* when unset/blank."""
    return (os.environ.get(name, "") or "").strip() or default


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var ("1/true/yes/on" == True), else *default*."""
    raw = (os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _physical_cores() -> int:
    """Best-effort physical (not logical) core count; 0 means "let llama.cpp decide".

    Hyper-threaded siblings do not help llama.cpp's compute-bound matmuls, so
    the physical count is the better default on CPU-only hosts.
    """
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            return int(out) if out.isdigit() else 0
        if sys.platform.startswith("linux"):
            # Count distinct (physical id, core id) pairs from /proc/cpuinfo.
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
            pairs, phys, core = set(), None, None
            for line in text.splitlines():
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip() and phys is not None and core is not None:
                    pairs.add((phys, core))
                    phys = core = None
            if phys is not None and core is not None:
                pairs.add((phys, core))
            return len(pairs)
    except Exception:  # noqa: BLE001 — detection is best-effort
        return 0
    return 0


def _settings() -> dict:
    """Resolve runtime settings: env overrides > defaults."""
    return {
        "host": os.environ.get("LLAMA_CPP_HOST", "127.0.0.1"),
        "port": _int_env("LLAMA_CPP_PORT", 8080),
        "ctx_size": _int_env("LLAMA_CPP_CTX_SIZE", 2048),
        "n_gpu_layers": _int_env("LLAMA_CPP_N_GPU_LAYERS", 0),
        "parallel": _int_env("LLAMA_CPP_PARALLEL", 1),
        # 0 = omit the flag and let llama.cpp pick.
        "threads": _int_env("LLAMA_CPP_THREADS", _physical_cores()),
        # Quantized KV cache is the single biggest RAM lever on small hosts.
        "cache_type_k": _str_env("LLAMA_CPP_CACHE_TYPE_K", "q8_0"),
        "cache_type_v": _str_env("LLAMA_CPP_CACHE_TYPE_V", "q8_0"),
        # Jinja chat templates are what make tool-calling models usable.
        "jinja": _bool_env("LLAMA_CPP_JINJA", True),
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
    return f"{_hf_base()}/{repo}/resolve/main/{file}"


def _pick_gguf_file(repo: str) -> str | None:
    """Return the first GGUF filename in a repo, preferring Q4_K_M."""
    try:
        url = f"{_hf_base()}/api/models/{repo}"
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


def _expected_sha256(repo: str, remote_file: str) -> str | None:
    """Return the upstream sha256 for a repo file, or None when unavailable.

    Hugging Face stores GGUF weights in LFS, and the *tree* endpoint exposes the
    object digest as ``lfs.oid`` (the model-detail endpoint does not — it reports
    ``lfs: null``, which is why integrity checking was previously deferred).

    Returns None whenever the digest cannot be established (offline, private repo,
    non-LFS file, unexpected payload). Callers must treat None as "cannot verify"
    and proceed, so a metadata outage never blocks a download.
    """
    try:
        url = f"{_hf_base()}/api/models/{repo}/tree/main"
        req = urllib.request.Request(url, headers={"User-Agent": "hermes-llama"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            if getattr(resp, "status", 200) != 200:
                return None
            entries = json.loads(resp.read().decode())
    except Exception:  # noqa: BLE001 — verification is best-effort by design
        return None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("path") != remote_file:
            continue
        lfs = entry.get("lfs")
        if isinstance(lfs, dict):
            oid = lfs.get("oid")
            # HF returns a bare hex digest; some mirrors use the "sha256:<hex>" form.
            if isinstance(oid, str):
                oid = oid.split(":")[-1].strip().lower()
                if len(oid) == 64 and all(c in "0123456789abcdef" for c in oid):
                    return oid
        return None
    return None


def _file_sha256(path: Path) -> str:
    """Streaming sha256 of a file (1 MiB chunks — never loads a GiB model in RAM)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: str | None) -> None:
    """Raise when *path* does not match *expected*. No-op when expected is None."""
    if not expected:
        return
    actual = _file_sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"checksum mismatch: expected sha256 {expected[:16]}…, got {actual[:16]}… "
            "(download corrupted or upstream file changed mid-transfer)"
        )


def _download_model(url: str, dest: Path, *, expected_sha256: str | None = None) -> None:
    """Download a model file robustly.

    Thin wrapper around the shared download helper. Adds sha256 verification
    when an expected digest is supplied.
    """
    from . import _download as _shared_download

    def _verify_sha(path: Path) -> None:
        _verify_sha256(path, expected_sha256)

    _shared_download.download_file(url, dest, timeout=7200, verify=_verify_sha if expected_sha256 else None)


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
            # A large file is not automatically a good file: verify it against
            # upstream before re-registering, so a corrupted or tampered payload
            # is not accepted just because it survived a previous run. Entries
            # pulled before checksums existed get their digest backfilled here.
            expected = _expected_sha256(repo, remote_file)
            if expected:
                try:
                    _verify_sha256(dest, expected)
                except Exception as exc:
                    dest.unlink(missing_ok=True)
                    return (
                        f"Existing file failed verification and was removed: {exc} "
                        f"Run `/llama pull {spec}` again to re-download."
                    )
            entry = {
                "repo": repo, "file": local_name, "path": str(dest),
                "size_gb": round(sz / 1024**3, 2),
            }
            if expected:
                entry["sha256"] = expected
            with _registry_txn() as reg:
                reg[alias] = entry
            state = "verified" if expected else "checksum unavailable upstream"
            return f"Already present: {dest} ({state}, registered as '{alias}')."
    url = _hf_file_url(repo, remote_file)
    # Ask upstream for the expected digest before transferring. None means
    # "cannot verify" (offline / non-LFS / mirror without the field) and the
    # download proceeds unverified rather than failing closed.
    expected = _expected_sha256(repo, remote_file)
    try:
        _download_model(url, dest, expected_sha256=expected)
    except Exception as exc:
        return f"Download failed: {exc}"
    size_gb = round(dest.stat().st_size / 1024**3, 2)
    entry = {"repo": repo, "file": local_name, "path": str(dest), "size_gb": size_gb}
    if expected:
        entry["sha256"] = expected
    with _registry_txn() as reg:
        reg[alias] = entry
    verified = " (sha256 verified)" if expected else " (checksum unavailable upstream)"
    return f"Downloaded {repo}/{remote_file} ({size_gb} GB){verified} → registered as '{alias}'."


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
    # CPU-only tuning. Each flag is additive and independently overridable:
    #   --threads          physical cores beat logical ones for llama.cpp matmuls
    #   --cache-type-k/v   q8_0 KV cache roughly halves cache RAM vs f16
    #   --jinja            enables the model's chat template => tool-calling works
    if int(s["threads"]) > 0:
        cmd += ["--threads", str(s["threads"]), "--threads-batch", str(s["threads"])]
    if s["cache_type_k"]:
        cmd += ["--cache-type-k", str(s["cache_type_k"])]
    if s["cache_type_v"]:
        cmd += ["--cache-type-v", str(s["cache_type_v"])]
    if s["jinja"]:
        cmd += ["--jinja"]
    install.install_root().mkdir(parents=True, exist_ok=True)
    log_path = install.install_root() / install.SERVER_LOG_FILE_NAME
    # Rotate: if server.log exceeds 10 MiB, move it to .log.1 (one-deep
    # history) before opening a fresh log. A long-running host that
    # sleeps/wakes and restarts the server over months would otherwise
    # accumulate an unbounded log.
    if log_path.is_file() and log_path.stat().st_size > 10 * 1024 * 1024:
        try:
            log_path.rename(log_path.with_suffix(".log.1"))
        except Exception:  # noqa: BLE001 — best-effort; fall through to append
            pass
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


def _windows_send_ctrl_break(pid: int) -> bool:
    """Send CTRL_BREAK_EVENT to a Windows process group for graceful shutdown.

    llama-server handles SIGBREAK as a graceful exit. Returns True if the
    event was delivered. Returns False when the call fails (no console,
    process already gone) so the caller can fall back to a force kill.
    """
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # CTRL_BREAK_EVENT = 1; CTRL_C_EVENT cannot be sent to a process
        # group from another process group, but CTRL_BREAK can.
        CTRL_BREAK_EVENT = 1
        return bool(kernel32.GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid))
    except Exception:
        return False


def stop() -> str:
    pid = _find_loaded_server()
    if pid is None:
        return "No llama-server running."
    try:
        if sys.platform == "win32":
            # Graceful first: send Ctrl+Break to the process group. llama-server
            # treats SIGBREAK as a graceful exit. Poll briefly; if the process
            # survives, fall back to a force kill — symmetric with the POSIX
            # SIGTERM → poll → SIGKILL path below.
            _windows_send_ctrl_break(pid)
            for _ in range(15):
                time.sleep(0.2)
                if not _pid_alive(pid) or not _is_llama_server(pid):
                    break
            else:
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
    # Confirm the process is actually gone before unlinking the pid file.
    # If the kill did not take effect, leave the pid file in place so
    # _find_loaded_server() still reports the server as running instead of
    # silently losing track of a live process.
    if _pid_alive(pid) and _is_llama_server(pid):
        return f"Failed to stop llama-server (pid {pid}): process still running."
    _server_pid_path().unlink(missing_ok=True)
    return f"Stopped llama-server (pid {pid})."


def stop_loaded_server() -> bool:
    """Stop any managed llama-server and report whether removal is now safe.

    This is the public server-control entry point for ``install.py`` — it owns
    the whole find -> stop -> re-confirm sequence so callers never need this
    module's private helpers.

    Returns True when no server was loaded, or when a loaded server was stopped
    and the PROCESS is confirmed gone. Returns False when a loaded server could
    not be confirmed stopped, so the caller must abort rather than delete
    ``bin/`` from under a live process.

    Confirmation deliberately probes the process rather than the pid file:
    ``stop()`` now leaves the pid file in place when the kill does not take
    effect, so a still-running server stays discoverable.
    """
    try:
        pid = _find_loaded_server()
        if pid is None:
            return True
        stop()
        return not (_pid_alive(pid) and _is_llama_server(pid))
    except Exception:  # noqa: BLE001 — shutdown could not be confirmed
        return False


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
