"""Smoke tests for install.py, imported in isolation.

install.py is deliberately self-contained (Python stdlib only) and must be
importable *without* the Hermes `providers` package. The package `__init__.py`
transitively imports `provider.py`, which does `from providers import ...`
(only available inside Hermes), so we load install.py directly as a standalone
module and exercise only its pure, deterministic helpers.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_INSTALL_PATH = _REPO_ROOT / "install.py"


def _load_install():
    spec = importlib.util.spec_from_file_location("hermes_install", _INSTALL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["hermes_install"] = module
    spec.loader.exec_module(module)
    return module


install = _load_install()


class _Platform:
    """Scoped override of install.py's platform probes (restored on exit)."""

    def __init__(self, *, macos=False, windows=False, arch="x64", nvidia=False, backend_env=None, mac_ver=(14, 0)):
        self.macos = macos
        self.windows = windows
        self.arch = arch
        self.nvidia = nvidia
        self.backend_env = backend_env
        self.mac_ver = mac_ver

    def __enter__(self):
        self._saved = {
            "_is_macos": install._is_macos,
            "_is_windows": install._is_windows,
            "_arch": install._arch,
            "_nvidia_present": install._nvidia_present,
            "_macos_ver": install._macos_ver,
            "env": os.environ.get("LLAMA_CPP_BACKEND"),
        }
        install._is_macos = lambda: self.macos
        install._is_windows = lambda: self.windows
        install._arch = lambda: self.arch
        install._nvidia_present = lambda: self.nvidia
        install._macos_ver = lambda: self.mac_ver
        if self.backend_env is None:
            os.environ.pop("LLAMA_CPP_BACKEND", None)
        else:
            os.environ["LLAMA_CPP_BACKEND"] = self.backend_env
        return self

    def __exit__(self, *exc):
        install._is_macos = self._saved["_is_macos"]
        install._is_windows = self._saved["_is_windows"]
        install._arch = self._saved["_arch"]
        install._nvidia_present = self._saved["_nvidia_present"]
        install._macos_ver = self._saved["_macos_ver"]
        if self._saved["env"] is None:
            os.environ.pop("LLAMA_CPP_BACKEND", None)
        else:
            os.environ["LLAMA_CPP_BACKEND"] = self._saved["env"]


def test_asset_name():
    tag = "b10549"
    with _Platform(macos=True, arch="arm64"):
        assert install._asset_name(tag, "cpu") == "llama-b10549-bin-macos-arm64.tar.gz"
        assert install._asset_name(tag, "cuda") is None  # no macOS cuda prebuilt
        assert install._asset_name(tag, "vulkan") is None  # no macOS vulkan prebuilt
    with _Platform(macos=True, arch="arm64", mac_ver=(12, 7)):
        assert install._asset_name(tag, "cpu") is None  # prebuilt needs macOS 13.3+ -> source
    with _Platform(windows=True, arch="x64"):
        assert install._asset_name(tag, "cpu") == "llama-b10549-bin-win-cpu-x64.zip"
        assert install._asset_name(tag, "cuda") == "cudart-llama-bin-win-cuda-12.4-x64.zip"
        assert install._asset_name(tag, "vulkan") == "llama-b10549-bin-win-vulkan-x64.zip"
    with _Platform(arch="x64"):  # Linux / other POSIX
        assert install._asset_name(tag, "cpu") == "llama-b10549-bin-ubuntu-x64.tar.gz"
        assert install._asset_name(tag, "vulkan") == "llama-b10549-bin-ubuntu-vulkan-x64.tar.gz"
        assert install._asset_name(tag, "cuda") is None  # no Linux cuda prebuilt -> source
    with _Platform(arch="arm64"):
        assert install._asset_name(tag, "vulkan") == "llama-b10549-bin-ubuntu-vulkan-arm64.tar.gz"


def test_resolve_backend():
    with _Platform():  # no env, not Windows -> cpu
        assert install.resolve_backend() == "cpu"
        assert install.resolve_backend(None) == "cpu"
        assert install.resolve_backend("auto") == "cpu"  # auto-detect, non-Windows -> cpu
        assert install.resolve_backend("cuda") == "cuda"
        assert install.resolve_backend("source") == "source"
        assert install.resolve_backend("vulkan") == "vulkan"
        assert install.resolve_backend("bogus") == "cpu"  # unknown -> cpu
    with _Platform(backend_env="CUDA"):
        assert install.resolve_backend() == "cuda"
    with _Platform(backend_env="vulkan"):
        assert install.resolve_backend() == "vulkan"
    with _Platform(windows=True, nvidia=False):
        assert install.resolve_backend() == "cpu"
        assert install.resolve_backend("auto") == "cpu"
    with _Platform(windows=True, nvidia=True):
        assert install.resolve_backend() == "cuda"
        assert install.resolve_backend("auto") == "cuda"


def test_install_skips_only_on_matching_tag():
    """install() skips only when the installed tag matches the target.

    Regression for the upgrade no-op bug: previously install() short-circuited
    on ANY working install, so `upgrade` never applied a newer release.
    """
    saved = (install.check, install._download_cached)
    try:
        install.check = lambda: {
            "installed": True, "binary": "b", "version": "v", "runs": True,
            "tag": "b10549", "latest_tag": "b10549", "up_to_date": True,
            "backend": "cpu", "method": "prebuilt",
        }

        def _fail_download(*_a, **_k):
            raise AssertionError("must not re-download when the tag already matches")

        install._download_cached = _fail_download
        with _Platform(arch="x64"):  # Linux: _asset_name returns a real asset name
            r = install.install(backend="cpu")
        assert r.get("skipped") is True, r
    finally:
        install.check, install._download_cached = saved


def test_install_upgrades_when_tag_differs():
    """install() proceeds to download when the installed tag != target.

    Regression for the upgrade no-op bug: a newer release (or the version pin)
    must not be skipped just because the binary already runs.
    """
    saved = (install.check, install._download_cached, install._build_from_source,
             install._asset_name, install._macos_ver, install._arch)
    calls = {}
    try:
        install.check = lambda: {
            "installed": True, "binary": "b", "version": "v", "runs": True,
            "tag": "b10540", "latest_tag": "b10549", "up_to_date": False,
            "backend": "cpu", "method": "prebuilt",
        }
        # Force asset selection to succeed even on hosts where the CPU prebuilt
        # is unavailable (e.g. macOS < 13.3) so the test probes the upgrade
        # logic, not host-specific asset availability.
        install._asset_name = lambda tag, backend: f"llama-{tag}-bin-macos-x64.tar.gz"
        install._macos_ver = lambda: (14, 0)
        install._arch = lambda: "x64"

        def _record(tag, asset):
            calls["tag"] = tag
            return None  # simulate a failed download -> source-build path

        install._download_cached = _record
        install._build_from_source = lambda backend: {"ok": False, "method": "source", "detail": "stub"}
        r = install.install(backend="cpu")
        assert calls.get("tag") == "b10549", calls
        assert r.get("skipped") is not True, r
    finally:
        install.check, install._download_cached, install._build_from_source = saved[:3]
        install._asset_name, install._macos_ver, install._arch = saved[3:]


def test_install_skips_source_build():
    """A source-built install is always current (built from latest master)."""
    saved = (install.check, install._download_cached, install._build_from_source)
    try:
        install.check = lambda: {
            "installed": True, "binary": "b", "version": "v", "runs": True,
            "tag": "source", "latest_tag": "b10549", "up_to_date": None,
            "backend": "cpu", "method": "source",
        }

        def _fail_download(*_a, **_k):
            raise AssertionError("source builds must not re-download")

        install._download_cached = _fail_download
        install._build_from_source = _fail_download
        r = install.install(backend="cpu")
        assert r.get("skipped") is True, r
    finally:
        install.check, install._download_cached, install._build_from_source = saved


def test_install_reinstalls_when_backend_differs():
    """A CPU install must not skip a request for a different backend."""
    saved = (install.check, install._download_cached, install._build_from_source)
    calls = {}
    try:
        install.check = lambda: {
            "installed": True, "binary": "b", "version": "v", "runs": True,
            "tag": "b10549", "latest_tag": "b10549", "up_to_date": True,
            "backend": "cpu", "method": "prebuilt",
        }

        def _record(tag, asset):
            calls["tag"] = tag
            return None

        install._download_cached = _record
        install._build_from_source = lambda backend: {"ok": False, "method": "source", "detail": "stub"}
        with _Platform(arch="x64"):
            r = install.install(backend="vulkan")
        assert r.get("skipped") is not True, r
        assert calls.get("tag") == "b10549", calls
    finally:
        install.check, install._download_cached, install._build_from_source = saved


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
