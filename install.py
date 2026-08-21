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
import urllib.request
import zipfile
from pathlib import Path

REPO = "ggml-org/llama.cpp"
GITHUB_API = f"https://api.github.com/repos/{REPO}"
SERVER_BIN = "llama-server.exe" if sys.platform == "win32" else "llama-server"
BACKENDS = ("cpu", "cuda", "vulkan", "source")


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return machine


# ── paths ────────────────────────────────────────────────────────────────────

def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


def install_root() -> Path:
    override = os.environ.get("LLAMA_CPP_INSTALL_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return _hermes_home() / "llama-cpp"


def bin_dir() -> Path:
    return install_root() / "bin"


def models_dir() -> Path:
    override = os.environ.get("LLAMA_CPP_MODELS_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return install_root() / "models"


def _meta_path() -> Path:
    return install_root() / ".version"


def _cache_dir() -> Path:
    return install_root() / ".cache"


def _read_meta() -> dict:
    try:
        return json.loads(_meta_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_meta(meta: dict) -> None:
    _meta_path().parent.mkdir(parents=True, exist_ok=True)
    _meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")


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


def check() -> dict:
    """Return install status incl. version, runs flag, and upgrade hint. Never raises."""
    try:
        return _check_impl()
    except Exception as exc:  # noqa: BLE001
        return {
            "installed": False, "binary": None, "version": None, "runs": False,
            "tag": None, "latest_tag": None, "up_to_date": None, "detail": str(exc),
        }


def _check_impl() -> dict:
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
    latest = _latest_tag()
    result["latest_tag"] = latest
    if binary is None:
        return result
    result["installed"] = True
    result["binary"] = str(binary)
    ok, info = _smoke_test(binary)
    result["runs"] = ok
    result["version"] = info
    tag = _read_meta().get("tag")
    result["tag"] = tag
    result["up_to_date"] = (tag == latest) if (tag and latest) else None
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
        return f"llama-{tag}-bin-macos-{arch}.tar.gz"
    if _is_windows():
        if backend == "cuda" and arch == "x64":
            return f"llama-{tag}-bin-win-cuda-12.4-x64.zip"
        if backend == "cuda" and arch == "arm64":
            return f"llama-{tag}-bin-win-cuda-13.4-arm64.zip"
        if backend == "vulkan" and arch == "x64":
            return f"llama-{tag}-bin-win-vulkan-x64.zip"
        return f"llama-{tag}-bin-win-cpu-{arch}.zip"
    # Linux (or other POSIX).
    if backend == "vulkan":
        return f"llama-{tag}-bin-ubuntu-vulkan-{arch}.tar.gz"
    if backend == "cuda":
        # No Linux CUDA prebuilt -> source build with -DGGML_CUDA=ON.
        return None
    return f"llama-{tag}-bin-ubuntu-{arch}.tar.gz"


# ── download / extract ───────────────────────────────────────────────────────

def _latest_tag() -> str | None:
    try:
        with urllib.request.urlopen(GITHUB_API + "/releases/latest", timeout=20) as resp:
            data = json.loads(resp.read().decode())
        return data.get("tag_name")
    except Exception:
        return None


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
    url = f"https://github.com/{REPO}/releases/download/{tag}/{asset}"
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
            tf.extractall(dest)
    elif archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)


def _install_extracted(extracted: Path, dest_bin: Path) -> Path | None:
    """Move the extracted llama.cpp tree (binaries + shared libs) into bin_dir.

    Prebuilt archives are a FLAT directory of executables PLUS shared libraries
    (``libllama-server-impl.dylib``, ``libggml*.dylib/.so/.dll``). ``llama-server``
    loads its impl via ``@rpath`` (its own directory), so the whole directory must
    be installed together.
    """
    dest_bin.mkdir(parents=True, exist_ok=True)
    server_dir: Path | None = None
    for root, _dirs, files in os.walk(extracted):
        if "llama-server" in files or "llama-server.exe" in files:
            server_dir = Path(root)
            break
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
                f"Or: git clone https://github.com/{REPO} && cmake -B build "
                "&& cmake --build build --config Release -j"
            ),
        }
    src = install_root() / "src" / "llama.cpp"
    if not (src / "CMakeLists.txt").is_file():
        src.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", f"https://github.com/{REPO}", str(src)],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            return {"ok": False, "method": "source", "detail": f"git clone failed: {proc.stderr.strip()[:300]}"}
    build = src / "build"
    cmake_args = [cmake, "-S", str(src), "-B", str(build)]
    if backend == "cuda":
        cmake_args += ["-DGGML_CUDA=ON"]
    if backend == "vulkan":
        cmake_args += ["-DGGML_VULKAN=ON"]
    if _is_macos() and _arch() == "x64":
        cmake_args += ["-DGGML_METAL=OFF"]
    proc = subprocess.run(cmake_args, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        return {"ok": False, "method": "source", "detail": f"cmake configure failed: {proc.stderr.strip()[:300]}"}
    proc = subprocess.run(
        [cmake, "--build", str(build), "--config", "Release", "--target", "llama-server", "-j"],
        capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0:
        return {"ok": False, "method": "source", "detail": f"build failed: {proc.stderr.strip()[:300]}"}
    # Locate the built llama-server + its shared libs. Single-config generators
    # (Unix Makefiles / Ninja) emit into ``build/bin/``; multi-config generators
    # (Visual Studio on Windows) emit into ``build/bin/<Config>/`` (e.g. Release).
    server_dir: Path | None = None
    for root, _dirs, files in os.walk(str(build)):
        if "llama-server" in files or "llama-server.exe" in files:
            server_dir = Path(root)
            break
    if server_dir is None:
        return {"ok": False, "method": "source", "detail": "build finished but llama-server was not produced"}
    # Clear any previous install, then install the whole dir (binary + shared libs).
    shutil.rmtree(bin_dir(), ignore_errors=True)
    bin_dir().mkdir(parents=True, exist_ok=True)
    for item in server_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, bin_dir() / item.name)
    moved = find_binary()
    if moved is None:
        return {"ok": False, "method": "source", "detail": "build finished but llama-server was not produced"}
    ok, info = _smoke_test(moved)
    if not ok:
        return {"ok": False, "method": "source", "detail": f"built binary does not run: {info}"}
    _write_meta({"tag": "source", "backend": backend})
    return {"ok": True, "method": "source", "detail": f"Built llama.cpp from source → {moved}"}


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
    existing = check()
    # Version pinning: explicit arg > LLAMA_CPP_VERSION > latest release.
    version = version or os.environ.get("LLAMA_CPP_VERSION", "").strip() or None
    # Skip only when already working AND nothing is forcing a change.
    if existing["installed"] and existing["runs"] and not force:
        if version:
            if version == existing.get("tag"):
                return {"ok": True, "skipped": True, "detail": f"Already installed at {version}."}
        elif existing.get("up_to_date"):
            return {"ok": True, "skipped": True, "detail": f"Already installed and up to date: {existing['binary']}"}
    tag = version or existing.get("latest_tag") or _latest_tag()
    if not tag:
        return {"ok": False, "detail": "Could not determine the latest release tag."}
    asset = _asset_name(tag, backend)
    if asset:
        archive = _download_cached(tag, asset)
        if archive is not None:
            with tempfile.TemporaryDirectory() as tmp:
                _extract(archive, Path(tmp))
                # Clear any previous (broken) install before installing fresh.
                shutil.rmtree(bin_dir(), ignore_errors=True)
                moved = _install_extracted(Path(tmp), bin_dir())
                if moved:
                    ok, info = _smoke_test(moved)
                    if not ok:
                        shutil.rmtree(bin_dir(), ignore_errors=True)
                        return {
                            "ok": False,
                            "method": "prebuilt",
                            "detail": (
                                f"Downloaded {tag}, but the binary does not run here "
                                f"({info}). Falling back to a source build is recommended. "
                                "If this is macOS < 13.3, the prebuilt requires a newer OS."
                            ),
                        }
                    _write_meta({"tag": tag, "backend": backend})
                    return {"ok": True, "method": "prebuilt", "detail": f"Installed {tag} prebuilt → {moved}"}
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
        return _uninstall_impl()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"Uninstall failed: {exc}"}


def _uninstall_impl() -> dict:
    binary = find_binary()
    if binary is not None and binary.is_relative_to(bin_dir()):
        _remove_plugin_artifacts()
        return {"ok": True, "detail": f"Removed plugin-managed llama.cpp (models kept at {models_dir()})."}
    if binary is not None:
        return {
            "ok": False,
            "detail": "llama-server was not installed by hermes-llama; remove it with the tool that installed it.",
        }
    _remove_plugin_artifacts()
    return {"ok": True, "detail": "Nothing installed; cleared plugin artifacts (models kept)."}


def _remove_plugin_artifacts() -> None:
    for sub in ("bin", "src", ".cache"):
        shutil.rmtree(install_root() / sub, ignore_errors=True)
    _meta_path().unlink(missing_ok=True)
