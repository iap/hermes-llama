"""Cross-platform check / install / uninstall for llama.cpp — self-contained.

Design decision: **no package managers.** Installation uses exactly two paths
so behaviour is identical on macOS, Windows, and Linux and needs no admin
privileges, no Homebrew/Winget/conda/Nix, and no pre-installed toolchain:

  1. **Prebuilt binaries** from GitHub releases (primary — fastest, portable).
  2. **CMake source build** (fallback — when no prebuilt matches the host,
     e.g. macOS < 13.3, or a backend with no prebuilt asset).

Verified facts (see RESEARCH.md): repo ``ggml-org/llama.cpp``; release tags are
build numbers (e.g. ``b10549``); asset pattern
``llama-<tag>-bin-<os>[-<backend>]-<arch>.{tar.gz,zip}``; the current macOS
prebuilt is compiled with ``minos 13.3`` so it will not run on macOS 12 or older
(detected by the post-install smoke test).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path
from contextlib import contextmanager

REPO = "ggml-org/llama.cpp"
# Endpoints — env-overridable for GitHub Enterprise / HF mirror deployments.
# Read at call time (not import) so _wire_config can set them from Hermes
# settings after this module is imported.


def _github_base() -> str:
    return os.environ.get("LLAMA_CPP_GITHUB_BASE", "https://github.com").rstrip("/")


def _github_api_base() -> str:
    return os.environ.get("LLAMA_CPP_GITHUB_API_BASE", "https://api.github.com").rstrip("/")


def _github_api() -> str:
    """API base for this repo; derived per call so late env overrides apply."""
    return f"{_github_api_base()}/repos/{REPO}"


SERVER_BIN = "llama-server.exe" if sys.platform == "win32" else "llama-server"
BACKENDS = ("cpu", "cuda", "vulkan", "source")

# Layout segments under install_root() — centralized so no path segment is
# repeated as a bare string literal (see CONTRIBUTING.md "no hardcoded paths").
DEFAULT_HERMES_DIR_NAME = ".hermes"
WINDOWS_HERMES_DIR_NAME = "hermes"
INSTALL_DIR_NAME = "llama-cpp"
BIN_DIR_NAME = "bin"
MODELS_DIR_NAME = "models"
SRC_DIR_NAME = "src"
SRC_REPO_DIR_NAME = "llama.cpp"
CACHE_DIR_NAME = ".cache"
VERSION_FILE_NAME = ".version"
SERVER_LOG_FILE_NAME = "server.log"
SERVER_PID_FILE_NAME = "server.pid"
REGISTRY_FILE_NAME = "models.json"


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_macos() -> bool:
    return sys.platform == "darwin"


@contextmanager
def _file_lock(lock_path: Path, timeout: float = 30.0):
    """Interprocess advisory lock on *lock_path*, stdlib-only (fcntl/msvcrt).

    POSIX path polls with LOCK_NB so a hung holder (e.g. stalled cmake) does
    not block uninstall/install forever; Windows already has NBLCK semantics.

    Generic on the path so callers can guard distinct resources (the install
    root, the model registry) without contending on a single global lock.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = None
    try:
        fh = open(lock_path, "a+")
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                # fallback: blocking
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                except OSError as exc:
                    fh.close()
                    raise RuntimeError(f"lock failed on {lock_path.name}: {exc}") from exc
        else:
            try:
                import fcntl
                deadline = time.time() + timeout
                while True:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.time() >= deadline:
                            fh.close()
                            raise RuntimeError(
                                f"{lock_path.name} held — another operation is running "
                                f"(waited {timeout:.0f}s)"
                            ) from None
                        time.sleep(0.1)
            except RuntimeError:
                raise
            except Exception as exc:
                fh.close()
                raise RuntimeError(f"lock failed on {lock_path.name}: {exc}") from exc
        yield
    finally:
        if fh is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    try:
                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    except Exception:  # noqa: BLE001 — best-effort unlock, file closed below anyway
                        pass
                else:
                    try:
                        import fcntl
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except Exception:  # noqa: BLE001 — best-effort unlock, file closed below anyway
                        pass
            finally:
                try:
                    fh.close()
                except Exception:  # noqa: BLE001 — best-effort close
                    pass


@contextmanager
def _install_lock(timeout: float = 30.0):
    """Interprocess lock guarding the install root (bin/, src/, metadata)."""
    with _file_lock(install_root() / ".install.lock", timeout=timeout):
        yield


@contextmanager
def registry_lock(timeout: float = 30.0):
    """Interprocess lock guarding the model registry (models.json).

    Public because ``models.py`` owns the registry but this module owns the
    locking primitive. Separate from the install lock so a long source build
    never blocks a model pull, and vice versa.
    """
    with _file_lock(install_root() / ".registry.lock", timeout=timeout):
        yield


def _unique_names() -> tuple[Path, Path]:
    suf = f".{os.getpid()}.{uuid.uuid4().hex[:8]}"
    return install_root() / f"{BIN_DIR_NAME}.tmp{suf}", install_root() / f"{BIN_DIR_NAME}.bak{suf}"


def _commit_staged_bin(staging: Path, backup: Path, meta: dict) -> dict:
    """Atomically replace ``bin_dir()`` with ``staging``. Call with ``_install_lock`` held.

    ``staging`` must already contain a smoke-tested binary. Handles the
    live-server interlock, backup, swap, verification, metadata, and cleanup.
    Never raises — returns an ``{ok, method, detail}`` dict.
    """
    method = str(meta.get("method") or "prebuilt")
    shutil.rmtree(backup, ignore_errors=True)
    if not _stop_loaded_server():
        shutil.rmtree(staging, ignore_errors=True)
        return {
            "ok": False,
            "method": method,
            "detail": "a running llama-server could not be stopped; stop it manually and retry",
        }
    swapped = False
    if bin_dir().exists():
        try:
            bin_dir().rename(backup)
            swapped = True
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            return {"ok": False, "method": method, "detail": "failed to backup existing bin"}
    try:
        staging.rename(bin_dir())
    except Exception as exc:
        if swapped and backup.exists():
            try:
                if bin_dir().exists():
                    shutil.rmtree(bin_dir(), ignore_errors=True)
                backup.rename(bin_dir())
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(staging, ignore_errors=True)
        return {"ok": False, "method": method, "detail": f"swap failed: {exc}"}
    moved = find_binary()
    if moved is None:
        shutil.rmtree(bin_dir(), ignore_errors=True)
        if backup.exists():
            try:
                backup.rename(bin_dir())
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(staging, ignore_errors=True)
        detail = (
            "build finished but llama-server was not produced"
            if method == "source"
            else "installed but binary not found after swap"
        )
        return {"ok": False, "method": method, "detail": detail}
    try:
        _write_meta(meta)
    except Exception as exc:
        shutil.rmtree(bin_dir(), ignore_errors=True)
        if backup.exists():
            try:
                backup.rename(bin_dir())
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(staging, ignore_errors=True)
        return {"ok": False, "method": method, "detail": f"metadata write failed: {exc}"}
    shutil.rmtree(backup, ignore_errors=True)
    shutil.rmtree(staging, ignore_errors=True)
    if method == "source":
        return {"ok": True, "method": method, "detail": f"Built llama.cpp from source → {moved}"}
    tag = str(meta.get("tag") or "")
    return {"ok": True, "method": method, "detail": f"Installed {tag} prebuilt → {moved}"}


def _arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return machine


def _macos_ver() -> tuple[int, int]:
    """Host macOS version as (major, minor); (0, 0) on parse failure or non-macOS."""
    try:
        parts = platform.mac_ver()[0].split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (0, 0)


# ── paths ────────────────────────────────────────────────────────────────────

def _hermes_home() -> Path:
    override = os.environ.get("HERMES_HOME", "").strip()
    if override:
        return Path(os.path.expandvars(override)).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / WINDOWS_HERMES_DIR_NAME
    return Path.home() / DEFAULT_HERMES_DIR_NAME


def install_root() -> Path:
    override = os.environ.get("LLAMA_CPP_INSTALL_DIR", "").strip()
    if override:
        return Path(os.path.expandvars(override)).expanduser()
    return _hermes_home() / INSTALL_DIR_NAME


def bin_dir() -> Path:
    return install_root() / BIN_DIR_NAME


def models_dir() -> Path:
    override = os.environ.get("LLAMA_CPP_MODELS_DIR", "").strip()
    if override:
        return Path(os.path.expandvars(override)).expanduser()
    return install_root() / MODELS_DIR_NAME


def _meta_path() -> Path:
    return install_root() / VERSION_FILE_NAME


def _cache_dir() -> Path:
    return install_root() / CACHE_DIR_NAME


def _read_meta() -> dict:
    try:
        return json.loads(_meta_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_meta(meta: dict) -> None:
    path = _meta_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _source_commit() -> str | None:
    """Return the current HEAD commit of the source checkout, or None.

    Used by ``check()`` to report source-build freshness honestly. Returns None
    when there is no checkout or git is unavailable, so callers never raise.
    """
    src = install_root() / SRC_DIR_NAME / SRC_REPO_DIR_NAME
    if not (src / ".git").is_dir():
        return None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(src), capture_output=True, text=True, timeout=60
        )
    except Exception:  # noqa: BLE001
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


# ── discovery ────────────────────────────────────────────────────────────────

def find_binary() -> Path | None:
    """Locate llama-server: plugin-managed bin dir first, then PATH."""
    names = ("llama-server.exe",) if _is_windows() else ("llama-server",)
    for name in names:
        p = bin_dir() / name
        if p.is_file():
            return p
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _smoke_test(binary: Path, timeout: float | None = None) -> tuple[bool, str]:
    """Run ``--version``. Returns ``(ok, info)``.

    Catches the subtle failure where a prebuilt was compiled for a newer OS than
    the host (Mach-O ``minos`` exceeds the running macOS): such a binary HANGS on
    launch rather than printing an error.

    Timeout can be overridden via ``LLAMA_CPP_SMOKE_TIMEOUT`` (seconds) for
    slow-storage hosts where first-launch binary loading exceeds the 8s default.
    """
    if timeout is None:
        try:
            timeout = float(os.environ.get("LLAMA_CPP_SMOKE_TIMEOUT", "8.0"))
        except Exception:
            timeout = 8.0
    try:
        proc = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True, timeout=timeout
        )
        out = (proc.stdout or proc.stderr or "").strip()
        if out:
            return True, out.splitlines()[0]
        return False, "ran but produced no output"
    except subprocess.TimeoutExpired:
        return False, "binary hangs on launch — likely built for a newer OS than this host"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def check(*, fetch_latest: bool = True) -> dict:
    """Return install status incl. version, runs flag, and upgrade hint. Never raises."""
    try:
        return _check_impl(fetch_latest=fetch_latest)
    except Exception as exc:  # noqa: BLE001
        return {
            "installed": False, "binary": None, "version": None, "runs": False,
            "tag": None, "latest_tag": None, "up_to_date": None,
            "backend": None, "method": None, "source_commit": None, "detail": str(exc),
        }


def _check_impl(*, fetch_latest: bool = True) -> dict:
    result = {
        "installed": False,
        "binary": None,
        "version": None,
        "runs": False,
        "tag": None,
        "latest_tag": None,
        "up_to_date": None,
        "source_commit": None,
    }
    binary = find_binary()
    latest = _latest_tag() if fetch_latest else None
    # If fetch suppressed, try cache without network
    if not fetch_latest:
        try:
            cache = _cache_dir() / "latest_tag.json"
            if cache.is_file():
                import json as _json, time as _time
                data = _json.loads(cache.read_text())
                if _time.time() - float(data.get("ts", 0)) < _TAG_CACHE_TTL:
                    latest = data.get("tag")
        except Exception:  # noqa: BLE001 — cache miss is fine, network fetch follows
            pass
    result["latest_tag"] = latest
    if binary is None:
        return result
    result["installed"] = True
    result["binary"] = str(binary)
    ok, info = _smoke_test(binary)
    result["runs"] = ok
    result["version"] = info
    meta = _read_meta()
    result["tag"] = meta.get("tag")
    result["backend"] = meta.get("backend")
    method = meta.get("method")
    # Backward-compat: pre-`method` meta files recorded source builds as tag="source".
    if method is None and meta.get("tag") == "source":
        method = "source"
    result["method"] = method
    if method == "source":
        # Honest freshness for source builds: compare the recorded build commit
        # against upstream master. Unknown (no commit recorded) stays None
        # rather than claiming up-to-date; a stale checkout reports False so
        # upgrade() will actually refresh and rebuild it. The remote-head probe
        # is network — honor fetch_latest=False (offline/lazy mode reports
        # unknown instead).
        remote_head = _source_remote_head() if fetch_latest else None
        src_commit = meta.get("commit")
        if isinstance(src_commit, str) and src_commit:
            result["source_commit"] = src_commit[:12]
            if remote_head:
                result["up_to_date"] = src_commit.startswith(remote_head[:12])
        else:
            result["up_to_date"] = None if remote_head is None else False
    else:
        result["up_to_date"] = (result["tag"] == latest) if (result["tag"] and latest) else None
    return result


def _source_remote_head() -> str | None:
    """Return the current master commit of the llama.cpp upstream repo.

    Uses the GitHub API (no local checkout required). Results are cached
    alongside _latest_tag (same TTL) so repeated check/install calls do not
    add an extra network round-trip. When the cache expires, sends the
    stored ETag via If-None-Match — a 304 refreshes the cache timestamp
    without burning a rate-limit token.
    """
    cache = _cache_dir() / "source_head.json"
    try:
        if cache.is_file():
            data = json.loads(cache.read_text())
            if time.time() - float(data.get("ts", 0)) < _TAG_CACHE_TTL:
                sha = data.get("sha")
                if isinstance(sha, str) and sha:
                    return sha
    except Exception:
        pass
    url = _github_api() + "/commits/master"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        # Conditional request: if we have a cached ETag, send it. A 304 means
        # the data hasn't changed — refresh the cache timestamp and reuse.
        try:
            if cache.is_file():
                etag = json.loads(cache.read_text()).get("etag")
                if isinstance(etag, str) and etag:
                    req.add_header("If-None-Match", etag)
        except Exception:  # noqa: BLE001 — cache read is best-effort
            pass
        with urllib.request.urlopen(req, timeout=20) as resp:
            if getattr(resp, "status", 200) == 304:
                # Not modified — refresh the cache timestamp.
                try:
                    data = json.loads(cache.read_text())
                    data["ts"] = time.time()
                    _cache_dir().mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(data))
                except Exception:  # noqa: BLE001 — cache write is best-effort
                    pass
                return json.loads(cache.read_text()).get("sha")
            data = json.loads(resp.read().decode())
            sha = data.get("sha")
            shastr = str(sha) if isinstance(sha, str) and sha else None
            if shastr:
                try:
                    etag = resp.headers.get("ETag") if hasattr(resp, "headers") else None
                    cache_data = {"sha": shastr, "ts": time.time()}
                    if isinstance(etag, str) and etag:
                        cache_data["etag"] = etag
                    _cache_dir().mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(cache_data))
                except Exception:
                    pass
            return shastr
    except Exception:  # noqa: BLE001 — offline / rate-limited: freshness unknown
        return None


# ── backend / asset selection ────────────────────────────────────────────────

def _nvidia_present() -> bool:
    try:
        return subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, timeout=5
        ).returncode == 0
    except Exception:
        return False


def resolve_backend(explicit: str | None = None) -> str:
    """Pick a backend: explicit arg/env > light auto-detect > cpu.

    ``""`` and ``"auto"`` both mean "auto-detect" (NVIDIA on Windows → CUDA,
    otherwise CPU). Unknown values fall back to CPU.
    """
    want = (explicit or os.environ.get("LLAMA_CPP_BACKEND", "") or "").strip().lower()
    if want in ("", "auto"):
        # No CUDA prebuilt exists for Linux/macOS; CUDA auto-detect only helps Windows.
        if _is_windows() and _nvidia_present():
            return "cuda"
        return "cpu"
    return want if want in BACKENDS else "cpu"


def _asset_name(tag: str, backend: str) -> str | None:
    """Return the prebuilt asset name for this host, or None if unavailable.

    Returns ``None`` when the requested backend has no prebuilt asset for this
    host (e.g. CUDA on Linux/macOS), so the caller falls back to a source build
    instead of silently installing a mismatched (CPU) binary.
    """
    arch = _arch()
    if _is_macos():
        # macOS ships a single Metal/CPU prebuilt; no cuda/vulkan asset.
        if backend in ("cuda", "vulkan"):
            return None
        # Prebuilt is compiled with minos 13.3; older hosts must source-build.
        if _macos_ver() < (13, 3):
            return None
        return f"llama-{tag}-bin-macos-{arch}.tar.gz"
    if _is_windows():
        if backend == "cuda":
            if arch == "x64":
                # cudart-* bundles the CUDA runtime (self-contained, no toolkit needed);
                # the bare llama-* CUDA build requires a separately-installed CUDA runtime.
                return "cudart-llama-bin-win-cuda-12.4-x64.zip"
            if arch == "arm64":
                return "cudart-llama-bin-win-cuda-13.4-arm64.zip"
            return None  # no CUDA asset for this arch
        if backend == "vulkan":
            # Only x64 Vulkan assets are published (no win-vulkan-arm64).
            return f"llama-{tag}-bin-win-vulkan-x64.zip" if arch == "x64" else None
        if backend == "cpu" and arch in ("x64", "arm64"):
            return f"llama-{tag}-bin-win-cpu-{arch}.zip"
        return None
    # Linux (or other POSIX).
    if backend == "vulkan":
        return f"llama-{tag}-bin-ubuntu-vulkan-{arch}.tar.gz"
    if backend == "cuda":
        # No Linux CUDA prebuilt -> source build with -DGGML_CUDA=ON.
        return None
    return f"llama-{tag}-bin-ubuntu-{arch}.tar.gz"


# ── download / extract ───────────────────────────────────────────────────────

def _is_build(tag: str) -> bool:
    """True for a llama.cpp build-number tag (e.g. ``b10549``)."""
    return bool(tag) and tag[0] == "b" and tag[1:].isdigit()


_TAG_CACHE_TTL = 600  # seconds; avoid repeated GitHub API hits within a run


def _latest_tag() -> str | None:
    """Resolve the newest build-number tag that ships prebuilt binaries.

    Upstream now publishes a semver "pointer" release (e.g. ``v0.2.0``) whose
    only asset is ``nightly-tag.txt``; the real binaries live on ``bNNNN`` build
    tags. Prefer the latest non-prerelease build tag, falling back to any build
    tag with assets (nightly), so the primary prebuilt path resolves correctly.
    Results are cached briefly to avoid repeated GitHub API hits within a run.
    When the cache expires, sends the stored ETag via If-None-Match — a 304
    refreshes the cache timestamp without burning a rate-limit token.
    """
    cache = _cache_dir() / "latest_tag.json"
    try:
        if cache.is_file():
            data = json.loads(cache.read_text())
            if time.time() - float(data.get("ts", 0)) < _TAG_CACHE_TTL:
                return data.get("tag")
    except Exception:  # noqa: BLE001 — cache miss is fine, network fetch follows
        pass
    # Paginated fetch: per_page=100 and follow Link rel=next (up to 3 pages)
    # so a burst of nightly prereleases cannot push the first stable
    # bNNNN off the first page (previously per_page=30, single page).
    releases_all: list[dict] = []
    url = _github_api() + "/releases?per_page=100"
    last_resp = None
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
            # Conditional request on the first page only: if we have a cached
            # ETag, send it. A 304 means the data hasn't changed — refresh the
            # cache timestamp and reuse.
            if not releases_all and cache.is_file():
                try:
                    etag = json.loads(cache.read_text()).get("etag")
                    if isinstance(etag, str) and etag:
                        req.add_header("If-None-Match", etag)
                except Exception:  # noqa: BLE001 — cache read is best-effort
                    pass
            with urllib.request.urlopen(req, timeout=20) as resp:
                last_resp = resp
                if getattr(resp, "status", 200) == 304 and not releases_all:
                    # Not modified — refresh the cache timestamp.
                    try:
                        data = json.loads(cache.read_text())
                        data["ts"] = time.time()
                        _cache_dir().mkdir(parents=True, exist_ok=True)
                        cache.write_text(json.dumps(data))
                    except Exception:  # noqa: BLE001 — cache write is best-effort
                        pass
                    return json.loads(cache.read_text()).get("tag")
                page = json.loads(resp.read().decode())
                if not isinstance(page, list):
                    break
                releases_all.extend(page)
                link = resp.headers.get("Link", "") if hasattr(resp, "headers") else ""
                nxt = None
                if link:
                    for part in link.split(","):
                        if 'rel="next"' in part:
                            s = part.find("<")
                            e = part.find(">")
                            if s != -1 and e != -1:
                                nxt = part[s + 1:e]
                                break
                if nxt:
                    url = nxt
                    continue
                break
        except Exception:
            if releases_all:
                break
            return None
    releases = releases_all
    if not releases:
        return None

    tag = None
    for rel in releases:
        t = rel.get("tag_name")
        if _is_build(t) and not rel.get("prerelease") and rel.get("assets"):
            tag = t
            break
    if tag is None:
        for rel in releases:
            t = rel.get("tag_name")
            if _is_build(t) and rel.get("assets"):
                tag = t
                break
    if tag:
        try:
            etag = last_resp.headers.get("ETag") if last_resp is not None and hasattr(last_resp, "headers") else None
            cache_data = {"tag": tag, "ts": time.time()}
            if isinstance(etag, str) and etag:
                cache_data["etag"] = etag
            _cache_dir().mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(cache_data))
        except Exception:  # noqa: BLE001 — cache write failure is non-fatal
            pass
    return tag


def _download(url: str, dest: Path) -> None:
    """Download a file — thin wrapper around the shared download helper."""
    from . import _download as _shared_download

    _shared_download.download_file(url, dest, timeout=600)


def _download_cached(tag: str, asset: str) -> Path | None:
    """Download to a cache dir; reuse on re-install of the same tag."""
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{tag}-{asset}"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    url = f"{_github_base()}/{REPO}/releases/download/{tag}/{asset}"
    try:
        _download(url, dest)
        return dest if dest.stat().st_size > 0 else None
    except Exception:
        dest.unlink(missing_ok=True)
        return None


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            try:
                tf.extractall(dest, filter='data')
            except TypeError:
                # Python < 3.12: filter= not supported. Validate and extract manually.
                # Reject all non-regular/non-dir members and validate containment
                # before each extract to prevent symlink/hardlink bypass, traversal,
                # and special file (FIFO/device) attacks.
                for member in tf.getmembers():
                    if not member.isfile() and not member.isdir():
                        raise RuntimeError(f"Blocked tar non-file/non-dir member: {member.name} (type={member.type})")
                    member_path = (dest / member.name).resolve()
                    if not member_path.is_relative_to(dest.resolve()):
                        raise RuntimeError(f"Blocked tar member with traversal: {member.name}")
                    tf.extract(member, dest)
    elif archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                # Reject symlinks BEFORE extracting anything. Validating paths
                # alone is not enough: a symlink member (`x -> /`) passes the
                # containment check because it does not exist yet, and a later
                # regular member (`x/evil`) then writes through it. The tar
                # branch already rejects link members; keep the two symmetric.
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise RuntimeError(f"Blocked zip symlink member: {member.filename}")
                target = (dest / member.filename).resolve()
                try:
                    target.relative_to(dest.resolve())
                except ValueError:
                    raise RuntimeError(f"Blocked zip member with traversal: {member.filename}")
            # Extract per member (not extractall) so the vetted list is what lands.
            for member in zf.infolist():
                zf.extract(member, dest)


def _find_server_dir(root: Path) -> Path | None:
    """Locate the directory containing llama-server under root."""
    for cur, _dirs, files in os.walk(root):
        if "llama-server" in files or "llama-server.exe" in files:
            return Path(cur)
    return None


def _install_extracted(extracted: Path, dest_bin: Path) -> Path | None:
    """Move the extracted llama.cpp tree (binaries + shared libs) into bin_dir.

    Prebuilt archives are a FLAT directory of executables PLUS shared libraries
    (``libllama-server-impl.dylib``, ``libggml*.dylib/.so/.dll``). ``llama-server``
    loads its impl via ``@rpath`` (its own directory), so the whole directory must
    be installed together.
    """
    dest_bin.mkdir(parents=True, exist_ok=True)
    server_dir = _find_server_dir(extracted)
    if server_dir is None:
        return None
    moved: Path | None = None
    for item in server_dir.iterdir():
        if not item.is_file():
            continue
        target = dest_bin / item.name
        shutil.move(str(item), str(target))
        if item.name in ("llama-server", "llama-server.exe"):
            target.chmod(0o755)
            moved = target
    return moved


# ── source build ─────────────────────────────────────────────────────────────

def _build_from_source(backend: str) -> dict:
    cmake = shutil.which("cmake")
    if cmake is None:
        return {
            "ok": False,
            "method": "source",
            "detail": (
                "No prebuilt binary matches this host and CMake is not installed. "
                "Run `python3 -m pip install cmake` (or install CMake), then retry. "
                f"Or: git clone {_github_base()}/{REPO} && cmake -B build "
                "&& cmake --build build --config Release -j"
            ),
        }
    src = install_root() / SRC_DIR_NAME / SRC_REPO_DIR_NAME
    src.parent.mkdir(parents=True, exist_ok=True)
    if not (src / "CMakeLists.txt").is_file():
        # Fresh checkout — clone the full history (not --depth 1) so subsequent
        # rebuilds can `git fetch` + reset instead of re-cloning from scratch.
        proc = subprocess.run(
            ["git", "clone", f"{_github_base()}/{REPO}", str(src)],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            return {"ok": False, "method": "source", "detail": f"git clone failed: {proc.stderr.strip()[:300]}"}
    else:
        # Refresh the existing checkout so a rebuild picks up upstream fixes
        # rather than reusing a stale local master. Reset to FETCH_HEAD instead
        # of origin/HEAD: the latter is a local convenience ref that can be
        # missing (remote never advertised HEAD, manual remote setup) or stale
        # after a default-branch change, and either failure aborts the build.
        for git_args in (
            ["git", "fetch", "origin", "+master:refs/remotes/origin/master"],
            ["git", "reset", "--hard", "origin/master"],
            ["git", "clean", "-fdx", "build"],  # stale CMake cache from another commit
        ):
            proc = subprocess.run(git_args, cwd=str(src), capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                return {
                    "ok": False,
                    "method": "source",
                    "detail": f"git {' '.join(git_args)} failed: {proc.stderr.strip()[:300]}",
                }
    # Record the exact commit we build from so check() can report freshness.
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(src), capture_output=True, text=True, timeout=60)
    src_commit = proc.stdout.strip() if proc.returncode == 0 else None
    build = src / "build"
    cmake_args = [cmake, "-S", str(src), "-B", str(build)]
    # No embedded web UI: it needs node/npm + a HF download and dominates build
    # time (often hangs). The plugin only serves the HTTP API (/v1/...), so skip
    # the UI and the tests/examples we never build.
    cmake_args += [
        "-DLLAMA_BUILD_UI=OFF",
        "-DLLAMA_USE_PREBUILT_UI=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
    ]
    if backend == "cuda":
        cmake_args += ["-DGGML_CUDA=ON"]
    if backend == "vulkan":
        cmake_args += ["-DGGML_VULKAN=ON"]
    if _is_macos() and _arch() == "x64":
        cmake_args += ["-DGGML_METAL=OFF"]
    proc = subprocess.run(cmake_args, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        return {"ok": False, "method": "source", "detail": f"cmake configure failed: {proc.stderr.strip()[:300]}"}
    jobs = str(min(os.cpu_count() or 2, 4))
    proc = subprocess.run(
        [cmake, "--build", str(build), "--config", "Release", "--target", "llama-server", "-j", jobs],
        capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0:
        return {"ok": False, "method": "source", "detail": f"build failed: {proc.stderr.strip()[:300]}"}
    # Locate the built llama-server + its shared libs. Single-config generators
    # (Unix Makefiles / Ninja) emit into ``build/bin/``; multi-config generators
    # (Visual Studio on Windows) emit into ``build/bin/<Config>/`` (e.g. Release).
    server_dir = _find_server_dir(build)
    if server_dir is None:
        return {"ok": False, "method": "source", "detail": "build finished but llama-server was not produced"}
    # Locked, unique staging -> atomic swap with restore on any failure
    with _install_lock():
        staging, backup = _unique_names()
        try:
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True, exist_ok=True)
            for item in server_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, staging / item.name)
            cand = staging / SERVER_BIN
            if not cand.is_file():
                candidates = [p for p in staging.iterdir() if p.name.startswith("llama-server")]
                cand = candidates[0] if candidates else None
            if cand is None or not cand.is_file():
                shutil.rmtree(staging, ignore_errors=True)
                return {"ok": False, "method": "source", "detail": "build finished but llama-server was not produced"}
            ok2, info2 = _smoke_test(cand, timeout=20.0)
            if not ok2:
                shutil.rmtree(staging, ignore_errors=True)
                return {"ok": False, "method": "source", "detail": f"built binary does not run: {info2}"}
            return _commit_staged_bin(
                staging,
                backup,
                {"tag": "source", "method": "source", "backend": backend, "commit": src_commit},
            )
        finally:
            # best-effort cleanup of this transaction's temp dirs
            try:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
            except Exception:  # noqa: BLE001 — restore already failed, best-effort cleanup
                pass


# ── install / uninstall / upgrade ────────────────────────────────────────────

def install(backend: str | None = None, version: str | None = None, force: bool = False) -> dict:
    """Install llama.cpp (prebuilt first, source-build fallback). Never raises.

    ``force=True`` bypasses the "already installed and up to date" fast path
    (used by ``upgrade``).
    """
    try:
        return _install_impl(backend, version, force)
    except Exception as exc:  # noqa: BLE001 — honor the "never raises" contract
        return {"ok": False, "detail": f"Install failed: {exc}"}


def _install_impl(backend: str | None, version: str | None, force: bool) -> dict:
    backend = resolve_backend(backend)
    if backend == "source":
        # Explicit source build — used on hosts whose prebuilt is incompatible
        # (e.g. macOS < 13.3).
        return _build_from_source("cpu")
    # Version pinning: explicit arg > LLAMA_CPP_VERSION > latest release.
    version = version or os.environ.get("LLAMA_CPP_VERSION", "").strip() or None
    if version and not _is_build(version):
        return {"ok": False, "detail": f"Invalid version pin '{version}': expected a build tag like b10549."}
    _existing = check() if not force else None
    target_tag = version or (_existing.get("latest_tag") if _existing else None) or _latest_tag()
    if not target_tag:
        return {"ok": False, "detail": "Could not determine the latest release tag."}

    # Skip only when the installed backend matches AND the tag matches, or the
    # install is a source build *confirmed* current and no explicit release was
    # pinned, unless `force` is set (used by `upgrade`).
    if not force:
        is_source = _existing.get("method") == "source"
        same_backend = _existing.get("backend") == backend
        same_tag = _existing.get("tag") == target_tag
        # Unknown freshness (probe unavailable) must not count as current, and a
        # source build never satisfies an explicit version pin: it carries no
        # release tag, so skipping would silently ignore the requested tag.
        source_current = is_source and not version and _existing.get("up_to_date") is True
        if _existing["installed"] and _existing["runs"] and same_backend and (same_tag or source_current):
            return {
                "ok": True,
                "skipped": True,
                "detail": f"Already installed and working at {target_tag}: {_existing['binary']}",
            }
    tag = target_tag
    asset = _asset_name(tag, backend)
    if asset:
        archive = _download_cached(tag, asset)
        if archive is not None:
            with tempfile.TemporaryDirectory() as tmp:
                _extract(archive, Path(tmp))
                # Locked unique staging -> atomic swap with restore
                with _install_lock():
                    staging, backup = _unique_names()
                    try:
                        shutil.rmtree(staging, ignore_errors=True)
                        staging.mkdir(parents=True, exist_ok=True)
                        moved = _install_extracted(Path(tmp), staging)
                        if not moved:
                            shutil.rmtree(staging, ignore_errors=True)
                            return {"ok": False, "method": "prebuilt", "detail": f"extract failed for {tag}"}
                        ok, info = _smoke_test(moved)
                        if not ok:
                            shutil.rmtree(staging, ignore_errors=True)
                            return {
                                "ok": False,
                                "method": "prebuilt",
                                "detail": (
                                    f"Downloaded {tag}, but the binary does not run here "
                                    f"({info}). Falling back to a source build is recommended. "
                                    "If this is macOS < 13.3, the prebuilt requires a newer OS."
                                ),
                            }
                        return _commit_staged_bin(
                            staging, backup, {"tag": tag, "method": "prebuilt", "backend": backend}
                        )
                    finally:
                        try:
                            if staging.exists():
                                shutil.rmtree(staging, ignore_errors=True)
                        except Exception:  # noqa: BLE001 — restore already failed, best-effort cleanup
                            pass
    # No prebuilt asset for this host/backend (e.g. Linux CUDA), or the download
    # failed → source build. (A prebuilt that downloads but fails the smoke test
    # returns an explicit error above rather than silently building for minutes.)
    return _build_from_source(backend)


def upgrade(backend: str | None = None) -> dict:
    """Reinstall (respects the LLAMA_CPP_VERSION pin; latest release otherwise)."""
    return install(backend=backend, force=True)


def _stop_loaded_server(models_module=None) -> bool:
    """Stop a loaded llama-server and report whether removal is now safe.

    Used by uninstall and the atomic-swap paths so we never remove ``bin/`` out
    from under a live server. Delegates to ``models.stop_loaded_server()``,
    which owns the find -> stop -> re-confirm sequence; this wrapper only
    resolves the sibling module and decides what an unavailable sibling means.

    ``models_module`` is an injection seam for tests; production callers omit
    it. The lazy ``from . import models`` fails when this file is loaded
    standalone (e.g. by the test harness) or when the plugin install is
    corrupted; in both cases there is no functioning sibling to have started a
    managed server, so it counts as "nothing loaded" — uninstall stays usable
    as the recovery path instead of bricking.
    """
    if models_module is None:
        try:
            from . import models as _models
        except ImportError:
            return True
        models_module = _models
    try:
        return bool(models_module.stop_loaded_server())
    except Exception:  # noqa: BLE001 — shutdown could not be confirmed
        return False


def uninstall() -> dict:
    """Remove the plugin-managed llama.cpp install. Never raises.

    Removes ``bin/``, ``src/``, and ``.cache/`` plus version metadata, but NEVER
    the ``models/`` dir or ``models.json`` — downloaded GGUFs are user data and
    survive uninstall.
    """
    try:
        with _install_lock():
            return _uninstall_impl()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"Uninstall failed: {exc}"}


def _uninstall_impl() -> dict:
    # Ownership is decided by layout, not by PATH: the plugin owns the install
    # iff its bin/<SERVER_BIN> exists or its .version metadata exists. A
    # llama-server installed elsewhere on PATH (e.g. a system-wide copy the
    # user also has) must not make us refuse to manage our own install.
    ours = (bin_dir() / SERVER_BIN).is_file() or _meta_path().is_file()
    if ours:
        # Never remove bin/ out from under a live server: abort unless shutdown
        # is confirmed (no loaded server, or stop verified it is gone).
        if not _stop_loaded_server():
            return {
                "ok": False,
                "detail": "A running llama-server could not be stopped; uninstall aborted. Stop it manually and retry.",
            }
        survivors = _remove_plugin_artifacts()
        # Final integrity check: the bin dir must actually be gone.
        if bin_dir().exists():
            survivors.append(str(bin_dir()))
        if survivors:
            return {
                "ok": False,
                "detail": "Uninstall incomplete; could not remove: " + "; ".join(sorted(set(survivors))),
            }
        return {"ok": True, "detail": f"Removed plugin-managed llama.cpp (models kept at {models_dir()})."}
    # Not ours. If a llama-server exists elsewhere on PATH, say so; otherwise
    # there is simply nothing to remove.
    if find_binary() is not None:
        return {
            "ok": False,
            "detail": "llama-server was not installed by hermes-llama; remove it with the tool that installed it.",
        }
    return {"ok": True, "detail": "Nothing installed; cleared plugin artifacts (models kept)."}


def _rmtree_collecting(path, failures: list) -> None:
    """``shutil.rmtree`` that records unremovable paths instead of raising.

    Prefers the ``onexc`` hook (Python 3.12+) and falls back to ``onerror``
    (Python 3.11 and earlier); both share the same callback signature, only the
    keyword name differs. The ``TypeError`` from an unsupported keyword fires
    before any deletion happens, so the retry never double-deletes.
    """

    def _record(_func, path_str, _excinfo):
        failures.append(str(path_str))

    try:
        shutil.rmtree(path, onexc=_record)
    except TypeError:
        shutil.rmtree(path, onerror=_record)


def _remove_plugin_artifacts() -> list[str]:
    """Remove plugin-managed artifacts; never raises.

    Returns a list of survivor paths that could not be removed. The caller
    reports these so uninstall surfaces a real failure instead of silently
    succeeding.
    """
    failures: list[str] = []
    for sub in (BIN_DIR_NAME, SRC_DIR_NAME, CACHE_DIR_NAME):
        target = install_root() / sub
        if target.exists():
            _rmtree_collecting(target, failures)
    # purge leftover atomic-swap temps (do not touch lock file while locked)
    for pat in (f"{BIN_DIR_NAME}.tmp*", f"{BIN_DIR_NAME}.bak*"):
        for p in install_root().glob(pat):
            try:
                if p.is_dir():
                    _rmtree_collecting(p, failures)
                else:
                    p.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001 — restore already failed, best-effort cleanup
                failures.append(str(p))
    for f in (SERVER_PID_FILE_NAME, SERVER_LOG_FILE_NAME, VERSION_FILE_NAME):
        p = install_root() / f
        try:
            p.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            failures.append(str(p))
    return failures
