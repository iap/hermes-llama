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
GITHUB_BASE = os.environ.get("LLAMA_CPP_GITHUB_BASE", "https://github.com").rstrip("/")
GITHUB_API_BASE = os.environ.get("LLAMA_CPP_GITHUB_API_BASE", "https://api.github.com").rstrip("/")
GITHUB_API = f"{GITHUB_API_BASE}/repos/{REPO}"
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
def _install_lock():
    """Interprocess lock for install root, stdlib-only (fcntl/msvcrt)."""
    lock_path = install_root() / ".install.lock"
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
                    raise RuntimeError(f"install lock failed: {exc}") from exc
        else:
            try:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            except Exception as exc:
                fh.close()
                raise RuntimeError(f"install lock failed: {exc}") from exc
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


def _unique_names() -> tuple[Path, Path]:
    suf = f".{os.getpid()}.{uuid.uuid4().hex[:8]}"
    return install_root() / f"{BIN_DIR_NAME}.tmp{suf}", install_root() / f"{BIN_DIR_NAME}.bak{suf}"


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


def _smoke_test(binary: Path, timeout: float = 8.0) -> tuple[bool, str]:
    """Run ``--version``. Returns ``(ok, info)``.

    Catches the subtle failure where a prebuilt was compiled for a newer OS than
    the host (Mach-O ``minos`` exceeds the running macOS): such a binary HANGS on
    launch rather than printing an error.
    """
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
            "backend": None, "method": None, "detail": str(exc),
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
        result["up_to_date"] = True  # source builds are always built from latest master
    else:
        result["up_to_date"] = (result["tag"] == latest) if (result["tag"] and latest) else None
    return result


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
    url = GITHUB_API + "/releases?per_page=100"
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
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
            _cache_dir().mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"tag": tag, "ts": time.time()}))
        except Exception:  # noqa: BLE001 — cache write failure is non-fatal
            pass
    return tag


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-llama"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _download_cached(tag: str, asset: str) -> Path | None:
    """Download to a cache dir; reuse on re-install of the same tag."""
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{tag}-{asset}"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    url = f"{GITHUB_BASE}/{REPO}/releases/download/{tag}/{asset}"
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
                target = (dest / member.filename).resolve()
                try:
                    target.relative_to(dest.resolve())
                except ValueError:
                    raise RuntimeError(f"Blocked zip member with traversal: {member.filename}")
            zf.extractall(dest)


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
                f"Or: git clone {GITHUB_BASE}/{REPO} && cmake -B build "
                "&& cmake --build build --config Release -j"
            ),
        }
    src = install_root() / SRC_DIR_NAME / SRC_REPO_DIR_NAME
    if not (src / "CMakeLists.txt").is_file():
        src.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", f"{GITHUB_BASE}/{REPO}", str(src)],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            return {"ok": False, "method": "source", "detail": f"git clone failed: {proc.stderr.strip()[:300]}"}
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
        # Stop a live server before we replace its bin/ (C1): the running
        # process may still dlopen these dylibs, so stop it first.
        _stop_running_server_if_any()
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
            shutil.rmtree(backup, ignore_errors=True)
            swapped = False
            if bin_dir().exists():
                try:
                    bin_dir().rename(backup)
                    swapped = True
                except Exception:
                    shutil.rmtree(staging, ignore_errors=True)
                    return {"ok": False, "method": "source", "detail": "failed to backup existing bin"}
            try:
                staging.rename(bin_dir())
            except Exception as exc:
                if swapped and backup.exists():
                    try:
                        # restore
                        if bin_dir().exists():
                            shutil.rmtree(bin_dir(), ignore_errors=True)
                        backup.rename(bin_dir())
                    except Exception:  # noqa: BLE001 — restore already failed, best-effort cleanup
                        pass
                shutil.rmtree(staging, ignore_errors=True)
                return {"ok": False, "method": "source", "detail": f"swap failed: {exc}"}
            # metadata must succeed or restore
            moved = find_binary()
            if moved is None:
                # swap succeeded but binary not found -> restore
                shutil.rmtree(bin_dir(), ignore_errors=True)
                if backup.exists():
                    try:
                        backup.rename(bin_dir())
                    except Exception:  # noqa: BLE001 — restore already failed, best-effort cleanup
                        pass
                shutil.rmtree(staging, ignore_errors=True)
                return {"ok": False, "method": "source", "detail": "build finished but llama-server was not produced"}
            try:
                _write_meta({"tag": "source", "method": "source", "backend": backend})
            except Exception as exc:
                # restore prior install
                shutil.rmtree(bin_dir(), ignore_errors=True)
                if backup.exists():
                    try:
                        backup.rename(bin_dir())
                    except Exception:  # noqa: BLE001 — restore already failed, best-effort cleanup
                        pass
                shutil.rmtree(staging, ignore_errors=True)
                return {"ok": False, "method": "source", "detail": f"metadata write failed: {exc}"}
            shutil.rmtree(backup, ignore_errors=True)
            shutil.rmtree(staging, ignore_errors=True)
            return {"ok": True, "method": "source", "detail": f"Built llama.cpp from source → {moved}"}
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

    # Skip only when the installed backend matches AND (the tag matches, or the
    # install is a source build — always built from latest master), unless
    # `force` is set (used by `upgrade`).
    if not force:
        is_source = _existing.get("method") == "source"
        same_backend = _existing.get("backend") == backend
        same_tag = _existing.get("tag") == target_tag
        if _existing["installed"] and _existing["runs"] and same_backend and (same_tag or is_source):
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
                    # Stop a live server before we replace its bin/ (C1).
                    _stop_running_server_if_any()
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
                        shutil.rmtree(backup, ignore_errors=True)
                        swapped = False
                        if bin_dir().exists():
                            try:
                                bin_dir().rename(backup)
                                swapped = True
                            except Exception:
                                shutil.rmtree(staging, ignore_errors=True)
                                return {"ok": False, "method": "prebuilt", "detail": "failed to backup existing bin"}
                        try:
                            staging.rename(bin_dir())
                        except Exception as exc:
                            if swapped and backup.exists():
                                try:
                                    if bin_dir().exists():
                                        shutil.rmtree(bin_dir(), ignore_errors=True)
                                    backup.rename(bin_dir())
                                except Exception:  # noqa: BLE001 — restore already failed, best-effort cleanup
                                    pass
                            shutil.rmtree(staging, ignore_errors=True)
                            return {"ok": False, "method": "prebuilt", "detail": f"swap failed: {exc}"}
                        final = find_binary()
                        if final is None:
                            shutil.rmtree(bin_dir(), ignore_errors=True)
                            if backup.exists():
                                try:
                                    backup.rename(bin_dir())
                                except Exception:  # noqa: BLE001 — restore already failed, best-effort cleanup
                                    pass
                            shutil.rmtree(staging, ignore_errors=True)
                            return {"ok": False, "method": "prebuilt", "detail": "installed but binary not found after swap"}
                        try:
                            _write_meta({"tag": tag, "method": "prebuilt", "backend": backend})
                        except Exception as exc:
                            shutil.rmtree(bin_dir(), ignore_errors=True)
                            if backup.exists():
                                try:
                                    backup.rename(bin_dir())
                                except Exception:  # noqa: BLE001 — restore already failed, best-effort cleanup
                                    pass
                            shutil.rmtree(staging, ignore_errors=True)
                            return {"ok": False, "method": "prebuilt", "detail": f"metadata write failed: {exc}"}
                        shutil.rmtree(backup, ignore_errors=True)
                        shutil.rmtree(staging, ignore_errors=True)
                        return {"ok": True, "method": "prebuilt", "detail": f"Installed {tag} prebuilt → {final}"}
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


def _plugin_owned() -> bool:
    """True when the install root holds plugin-managed artifacts.

    Ownership is derived from the plugin's own layout — the server binary in
    ``bin/`` or the ``.version`` metadata file — and NEVER from ``PATH``. A
    system ``llama-server`` on ``PATH`` must not block plugin cleanup (M5):
    ``find_binary()`` falls back to ``shutil.which``, which would otherwise make
    us refuse and leak every artifact (bin/, src/, .cache/, …).
    """
    if (bin_dir() / SERVER_BIN).is_file():
        return True
    if _meta_path().is_file():
        return True
    return False


def _stop_running_server_if_any() -> None:
    """Stop a live plugin llama-server before we mutate/delete its files (C1).

    Uninstall — or a bin/ swap during install/upgrade — would otherwise orphan a
    running server and erase the only handle to it (``server.pid``), leaving a
    process bound to its port forever. A live server may also still lazily
    ``dlopen`` the dylibs we are about to replace, so we stop it first.

    The sibling ``models`` module is imported lazily to avoid a top-level
    circular import (``install.py`` also loads standalone in the test harness).
    Stopping is best-effort: if ``models`` is unavailable or ``stop()`` raises,
    we proceed with removal regardless.
    """
    try:
        from . import models  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 — models unavailable (standalone test load)
        return
    try:
        models.stop()
    except Exception:  # noqa: BLE001 — best-effort; removal proceeds regardless
        pass


def _uninstall_impl() -> dict:
    # Ownership is derived from the plugin's own layout, never from PATH (M5).
    if not _plugin_owned():
        # No plugin-managed install. Still clear any stray swap temps / pid left
        # by an aborted prior install, but report "nothing installed".
        _remove_plugin_artifacts()
        return {"ok": True, "detail": "Nothing installed; cleared plugin artifacts (models kept)."}
    # Stop a running server before destroying its only handle (bin/ + server.pid)
    # so we never orphan a live process bound to its port (C1).
    _stop_running_server_if_any()
    ok, survivors = _remove_plugin_artifacts()
    if ok:
        detail = f"Removed plugin-managed llama.cpp (models kept at {models_dir()})."
    else:
        detail = "Removed most plugin artifacts; survivors: " + ", ".join(
            str(p) for p in survivors
        )
    return {"ok": ok, "detail": detail}


def _remove_plugin_artifacts() -> tuple[bool, list[Path]]:
    """Delete plugin-managed artifacts (bin/, src/, .cache/, swap temps, meta).

    Returns ``(ok, survivors)`` where ``survivors`` lists paths that could not be
    removed. The caller must treat ``ok is False`` as an incomplete uninstall:
    previously ``shutil.rmtree(..., ignore_errors=True)`` swallowed every failure
    (e.g. ``PermissionError`` on a read-only ``bin/``) and ``_uninstall_impl``
    hard-coded ``ok=True`` (M2).

    ``models/`` + ``models.json`` are intentionally never touched — downloaded
    GGUFs are user data and survive uninstall.
    """
    survivors: list[Path] = []

    def _on_error(_fn: object, path: str, _excinfo: object) -> None:
        # Record every path rmtree could not remove so we can report it.
        survivors.append(Path(path))

    root = install_root()
    for sub in (BIN_DIR_NAME, SRC_DIR_NAME, CACHE_DIR_NAME):
        d = root / sub
        if d.exists():
            shutil.rmtree(d, onexc=_on_error)
    # purge leftover atomic-swap temps (do not touch the lock file while locked)
    for pat in (f"{BIN_DIR_NAME}.tmp*", f"{BIN_DIR_NAME}.bak*"):
        for p in sorted(root.glob(pat)):
            try:
                if p.is_dir():
                    shutil.rmtree(p, onexc=_on_error)
                else:
                    p.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001 — record and continue
                survivors.append(p)
    _meta_path().unlink(missing_ok=True)
    (root / SERVER_PID_FILE_NAME).unlink(missing_ok=True)
    (root / SERVER_LOG_FILE_NAME).unlink(missing_ok=True)
    # Post-condition (M2): assert bin/ is actually gone, since a failed rmtree
    # may have left it (and its dylibs) behind.
    if bin_dir().exists():
        survivors.append(bin_dir())
    return (not survivors, survivors)
