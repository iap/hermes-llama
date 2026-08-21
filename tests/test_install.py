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

    def __init__(self, *, macos=False, windows=False, arch="x64", nvidia=False, backend_env=None):
        self.macos = macos
        self.windows = windows
        self.arch = arch
        self.nvidia = nvidia
        self.backend_env = backend_env

    def __enter__(self):
        self._saved = {
            "_is_macos": install._is_macos,
            "_is_windows": install._is_windows,
            "_arch": install._arch,
            "_nvidia_present": install._nvidia_present,
            "env": os.environ.get("LLAMA_CPP_BACKEND"),
        }
        install._is_macos = lambda: self.macos
        install._is_windows = lambda: self.windows
        install._arch = lambda: self.arch
        install._nvidia_present = lambda: self.nvidia
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
    with _Platform(windows=True, arch="x64"):
        assert install._asset_name(tag, "cpu") == "llama-b10549-bin-win-cpu-x64.zip"
        assert install._asset_name(tag, "cuda") == "llama-b10549-bin-win-cuda-12.4-x64.zip"
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
