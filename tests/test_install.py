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
import shutil
import sys
import tempfile
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
    """A source build confirmed current is skipped (no pin, up_to_date=True)."""
    saved = (install.check, install._download_cached, install._build_from_source)
    try:
        install.check = lambda: {
            "installed": True, "binary": "b", "version": "v", "runs": True,
            "tag": "source", "latest_tag": "b10549", "up_to_date": True,
            "backend": "cpu", "method": "source",
        }

        def _fail_download(*_a, **_k):
            raise AssertionError("a current source build must not re-download or rebuild")

        install._download_cached = _fail_download
        install._build_from_source = _fail_download
        r = install.install(backend="cpu")
        assert r.get("skipped") is True, r
    finally:
        install.check, install._download_cached, install._build_from_source = saved


def test_install_rebuilds_source_when_freshness_unknown():
    """Unknown source freshness must not count as current.

    Regression: `up_to_date is not False` treated an unavailable probe as
    up-to-date and skipped the install, keeping a possibly stale binary.
    """
    saved = (install.check, install._download_cached, install._build_from_source,
             install._asset_name)
    calls = {}
    try:
        install.check = lambda: {
            "installed": True, "binary": "b", "version": "v", "runs": True,
            "tag": "source", "latest_tag": "b10549", "up_to_date": None,
            "backend": "cpu", "method": "source",
        }
        install._asset_name = lambda tag, backend: None  # no prebuilt -> source path

        def _record_build(backend):
            calls["built"] = backend
            return {"ok": True, "method": "source", "detail": "rebuilt"}

        install._build_from_source = _record_build
        r = install.install(backend="cpu")
        assert r.get("skipped") is not True, r
        assert calls.get("built") == "cpu", calls
    finally:
        install.check, install._download_cached, install._build_from_source = saved[:3]
        install._asset_name = saved[3]


def test_install_source_does_not_satisfy_version_pin():
    """An explicit release pin is never satisfied by a source build."""
    saved = (install.check, install._download_cached, install._build_from_source,
             install._asset_name, install._macos_ver, install._arch)
    calls = {}
    try:
        install.check = lambda: {
            "installed": True, "binary": "b", "version": "v", "runs": True,
            "tag": "source", "latest_tag": "b10549", "up_to_date": True,
            "backend": "cpu", "method": "source",
        }
        install._asset_name = lambda tag, backend: f"llama-{tag}-bin-ubuntu-x64.tar.gz"
        install._macos_ver = lambda: (14, 0)
        install._arch = lambda: "x64"

        def _record(tag, asset):
            calls["tag"] = tag
            return None  # simulate download failure -> source fallback

        install._download_cached = _record
        install._build_from_source = lambda backend: {"ok": False, "method": "source", "detail": "stub"}
        r = install.install(backend="cpu", version="b10540")
        assert r.get("skipped") is not True, r
        assert calls.get("tag") == "b10540", calls
    finally:
        install.check, install._download_cached, install._build_from_source = saved[:3]
        install._asset_name, install._macos_ver, install._arch = saved[3:]


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


def test_uninstall_removes_artifacts_on_py311():
    """_remove_plugin_artifacts() works on Python < 3.12 (onerror fallback).

    Regression for the rmtree(onexc=...) TypeError: on 3.10/3.11 uninstall
    raised before deleting anything, leaving bin/, src/, .cache/ behind. We
    cannot force the interpreter version here, but we can assert the cleanup
    contract holds on the running interpreter and that no TypeError escapes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "llama-cpp"
        (root / "bin").mkdir(parents=True)
        (root / "src" / "llama.cpp").mkdir(parents=True)
        (root / ".cache").mkdir(parents=True)
        (root / "models").mkdir(parents=True)
        (root / "bin" / "llama-server").write_text("#!/bin/sh\n")
        (root / "src" / "llama.cpp" / "CMakeLists.txt").write_text("x\n")
        (root / ".version").write_text('{"tag": "b1234"}')
        (root / "models" / "sample.gguf").write_text("gguf-bytes")
        saved = os.environ.get("LLAMA_CPP_INSTALL_DIR")
        os.environ["LLAMA_CPP_INSTALL_DIR"] = str(root)
        try:
            survivors = install._remove_plugin_artifacts()
            assert survivors == [], survivors
            remaining = sorted(p.name for p in root.iterdir())
            assert remaining == ["models"], f"only models/ should survive: {remaining}"
            assert (root / "models" / "sample.gguf").is_file(), "user GGUFs must survive"
        finally:
            if saved is None:
                os.environ.pop("LLAMA_CPP_INSTALL_DIR", None)
            else:
                os.environ["LLAMA_CPP_INSTALL_DIR"] = saved


def test_stop_loaded_server_confirms_shutdown():
    """_stop_loaded_server() only reports True when the PROCESS is gone.

    Regression for the ignored stop() failure: a failed stop must yield False so
    callers abort instead of deleting bin/ under a live server. Confirmation must
    not use _find_loaded_server(), because stop() unlinks the pid file even when
    the kill failed. The ImportError path (standalone module load) counts as
    nothing-loaded so uninstall stays usable as the recovery tool.
    """
    saved = install._stop_loaded_server

    class _StubModels:
        def __init__(self, pid, *, alive_after, is_server_after=True, stop_raises=False):
            self._pid = pid
            self._alive_after = alive_after
            self._is_server_after = is_server_after
            self._stop_raises = stop_raises
            self.stop_calls = 0

        def _find_loaded_server(self):
            return self._pid

        def stop(self):
            self.stop_calls += 1
            if self._stop_raises:
                raise RuntimeError("kill failed")
            return "Stopped."

        # Process-level probes: unaffected by stop() unlinking the pid file.
        def _pid_alive(self, pid):
            assert pid == self._pid
            return self._alive_after

        def _is_llama_server(self, pid):
            return self._is_server_after

    try:
        # No loaded server -> True without calling stop().
        stub = _StubModels(None, alive_after=False)
        assert install._stop_loaded_server(models_module=stub) is True
        assert stub.stop_calls == 0

        # Loaded -> stopped -> process gone: True.
        stub = _StubModels(1234, alive_after=False)
        assert install._stop_loaded_server(models_module=stub) is True
        assert stub.stop_calls == 1

        # Loaded -> stop() ran but process STILL ALIVE: False (must abort).
        # This is the case a _find_loaded_server()-based check would miss.
        stub = _StubModels(1234, alive_after=True)
        assert install._stop_loaded_server(models_module=stub) is False

        # Still alive but PID reused by another program -> not our server: True.
        stub = _StubModels(1234, alive_after=True, is_server_after=False)
        assert install._stop_loaded_server(models_module=stub) is True

        # Loaded -> stop raises: False.
        stub = _StubModels(1234, alive_after=True, stop_raises=True)
        assert install._stop_loaded_server(models_module=stub) is False
    finally:
        install._stop_loaded_server = saved


def test_uninstall_aborts_when_stop_fails():
    """uninstall() refuses to delete artifacts when the server won't die."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "llama-cpp"
        (root / "bin").mkdir(parents=True)
        (root / "bin" / "llama-server").write_text("#!/bin/sh\n")
        (root / ".version").write_text('{"tag": "b1234"}')
        saved_env = os.environ.get("LLAMA_CPP_INSTALL_DIR")
        os.environ["LLAMA_CPP_INSTALL_DIR"] = str(root)

        class _FailingModels:
            @staticmethod
            def _find_loaded_server():
                return 4321

            @staticmethod
            def stop():
                return "Failed to stop pid 4321: simulated"

            @staticmethod
            def _pid_alive(pid):
                return True  # process survives the stop attempt

            @staticmethod
            def _is_llama_server(pid):
                return True

        saved_stop = install._stop_loaded_server
        install._stop_loaded_server = lambda models_module=None: saved_stop(_FailingModels)
        try:
            r = install.uninstall()
            assert r["ok"] is False, r
            assert "could not be stopped" in r["detail"], r
            assert (root / "bin" / "llama-server").is_file(), "bin/ must survive an aborted uninstall"
            assert (root / ".version").is_file(), "metadata must survive an aborted uninstall"
        finally:
            install._stop_loaded_server = saved_stop
            if saved_env is None:
                os.environ.pop("LLAMA_CPP_INSTALL_DIR", None)
            else:
                os.environ["LLAMA_CPP_INSTALL_DIR"] = saved_env


def test_check_reports_stale_source_build():
    """A source build older than upstream master reports up_to_date=False."""
    meta = {"tag": "source", "method": "source", "backend": "cpu", "commit": "a" * 40}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            import json as _json

            return _json.dumps({"sha": "b" * 40}).encode()

        headers = {}

    seen_urls = []

    import urllib.request as _ur

    def _capture(req, timeout=20):
        seen_urls.append(req.full_url if hasattr(req, "full_url") else str(req))
        return _Resp()

    saved_urlopen, saved_meta = _ur.urlopen, install._read_meta
    saved_latest = install._latest_tag, install._smoke_test, install.find_binary
    saved_dir = os.environ.get("LLAMA_CPP_INSTALL_DIR")
    try:
        tmp = tempfile.mkdtemp(prefix="llama-src-fresh-")
        os.environ["LLAMA_CPP_INSTALL_DIR"] = tmp
        install._read_meta = lambda: dict(meta)
        install._latest_tag = lambda: None
        install._smoke_test = lambda binary, timeout=30.0: (True, "llama-server version 1")
        install.find_binary = lambda: "/fake/bin/llama-server"
        _ur.urlopen = _capture
        # fetch_latest=True: the freshness probe is network-backed, mirroring
        # what install()'s skip check uses. fetch_latest=False is the offline
        # mode and must stay probe-free (reports unknown).
        r = install.check()
        assert r["up_to_date"] is False, r
        assert r["source_commit"] == "a" * 12, r

        # Same commit as remote head -> current.
        meta["commit"] = "b" * 40
        r = install.check()
        assert r["up_to_date"] is True, r

        # Regression: the commits URL must contain /repos/{REPO} exactly once.
        # A doubled path (…/repos/X/Y/repos/X/Y/…) 404s and silently degrades
        # freshness to unknown.
        for u in seen_urls:
            assert "/repos/" in u, u
            assert u.count("/repos/") == 1, f"doubled repos path: {u}"
    finally:
        _ur.urlopen = saved_urlopen
        install._read_meta = saved_meta
        install._latest_tag, install._smoke_test, install.find_binary = saved_latest
        if saved_dir is None:
            os.environ.pop("LLAMA_CPP_INSTALL_DIR", None)
        else:
            os.environ["LLAMA_CPP_INSTALL_DIR"] = saved_dir
        shutil.rmtree(tmp, ignore_errors=True)


def test_wire_config_derives_base_url():
    """_wire_config derives LLAMA_CPP_BASE_URL from host/port when no base_url."""
    from pathlib import Path as _Path
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    import types as _types
    # stub the Hermes-only `providers` package so __init__.py can be imported
    if "providers" not in _sys.modules:
        _stub_providers = _types.ModuleType("providers")
        _stub_providers.register_provider = lambda p: None  # noqa: ARG001
        _sys.modules["providers"] = _stub_providers
    if "providers.base" not in _sys.modules:
        _stub_base = _types.ModuleType("providers.base")
        class _PP:  # minimal ProviderProfile
            def __init__(self, **kw): self.__dict__.update(kw)
        _stub_base.ProviderProfile = _PP
        _sys.modules["providers.base"] = _stub_base
    _p = _Path(__file__).resolve().parents[1] / "__init__.py"
    _spec = _ilu.spec_from_file_location("_hmaudit_init", _p)
    assert _spec is not None and _spec.loader is not None
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["_hmaudit_init"] = _mod
    _spec.loader.exec_module(_mod)
    class _Ctx:
        def get_config(self, k):
            return {"host": "192.168.1.2", "port": 9999}.get(k)
    saved = {n: _os.environ.get(n) for n in ("LLAMA_CPP_BASE_URL","LLAMA_CPP_HOST","LLAMA_CPP_PORT")}
    for n in saved: _os.environ.pop(n, None)
    try:
        _mod._wire_config(_Ctx())
        assert _os.environ.get("LLAMA_CPP_BASE_URL") == "http://192.168.1.2:9999/v1"
        assert _os.environ.get("LLAMA_CPP_HOST") == "192.168.1.2"
        assert _os.environ.get("LLAMA_CPP_PORT") == "9999"
        # explicit base_url wins
        _os.environ["LLAMA_CPP_BASE_URL"] = "http://explicit:8000/v1"
        _os.environ.pop("LLAMA_CPP_HOST", None)
        class _Ctx2:
            def get_config(self, k): return {"host":"1.1.1.1","port":1234}.get(k)
        _mod._wire_config(_Ctx2())
        assert _os.environ["LLAMA_CPP_BASE_URL"] == "http://explicit:8000/v1"
    finally:
        for n, v in saved.items():
            _os.environ.pop(n, None)
            if v is not None: _os.environ[n] = v

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
