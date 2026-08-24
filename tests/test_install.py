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


def _load_models():
    """Load models.py with a synthetic package so ``from . import install`` works.

    models.py is not stdlib-only in the import sense — it does a relative
    import of its sibling. We build a throwaway package namespace rather than
    installing the plugin, keeping the suite dependency-free.
    """
    import types

    pkg_name = "hermes_llama_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(_REPO_ROOT)]
    sys.modules[pkg_name] = pkg
    sys.modules[f"{pkg_name}.install"] = install

    spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.models", _REPO_ROOT / "models.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{pkg_name}.models"] = module
    spec.loader.exec_module(module)

    # Also load _download.py and attach it to the models module
    spec = importlib.util.spec_from_file_location(
        f"{pkg_name}._download", _REPO_ROOT / "_download.py"
    )
    assert spec is not None and spec.loader is not None
    download_mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{pkg_name}._download"] = download_mod
    spec.loader.exec_module(download_mod)
    module._download = download_mod

    return module


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


def test_models_stop_loaded_server_confirms_shutdown():
    """models.stop_loaded_server() only reports True when the PROCESS is gone.

    Regression for the ignored stop() failure: a failed stop must yield False so
    callers abort instead of deleting bin/ under a live server. Confirmation must
    not use _find_loaded_server(), because stop() unlinks the pid file even when
    the kill failed.
    """
    models = _load_models()
    saved = (models._find_loaded_server, models.stop, models._pid_alive,
             models._is_llama_server)
    calls = {"stop": 0}

    def _wire(pid, *, alive_after, is_server_after=True, stop_raises=False):
        calls["stop"] = 0

        def _stop():
            calls["stop"] += 1
            if stop_raises:
                raise RuntimeError("kill failed")
            return "Stopped."

        models._find_loaded_server = lambda: pid
        models.stop = _stop
        # Process-level probes: unaffected by stop() unlinking the pid file.
        models._pid_alive = lambda p: alive_after
        models._is_llama_server = lambda p: is_server_after

    try:
        # No loaded server -> True without calling stop().
        _wire(None, alive_after=False)
        assert models.stop_loaded_server() is True
        assert calls["stop"] == 0

        # Loaded -> stopped -> process gone: True.
        _wire(1234, alive_after=False)
        assert models.stop_loaded_server() is True
        assert calls["stop"] == 1

        # Loaded -> stop() ran but process STILL ALIVE: False (must abort).
        # This is the case a _find_loaded_server()-based check would miss.
        _wire(1234, alive_after=True)
        assert models.stop_loaded_server() is False

        # Still alive but PID reused by another program -> not our server: True.
        _wire(1234, alive_after=True, is_server_after=False)
        assert models.stop_loaded_server() is True

        # Loaded -> stop raises: False.
        _wire(1234, alive_after=True, stop_raises=True)
        assert models.stop_loaded_server() is False
    finally:
        (models._find_loaded_server, models.stop, models._pid_alive,
         models._is_llama_server) = saved


def test_install_stop_loaded_server_delegates():
    """install._stop_loaded_server() delegates and never touches private members.

    It must forward to the sibling's public stop_loaded_server(), propagate the
    verdict, treat a raising sibling as unsafe, and treat an unavailable sibling
    (standalone load / corrupt install) as nothing-loaded so uninstall stays
    usable as the recovery path.
    """
    class _Stub:
        def __init__(self, verdict=None, raises=False):
            self.verdict = verdict
            self.raises = raises
            self.calls = 0

        def stop_loaded_server(self):
            self.calls += 1
            if self.raises:
                raise RuntimeError("boom")
            return self.verdict

        def __getattr__(self, name):  # any private reach-in would explode here
            raise AssertionError(f"install must not touch models.{name}")

    stub = _Stub(verdict=True)
    assert install._stop_loaded_server(models_module=stub) is True
    assert stub.calls == 1

    stub = _Stub(verdict=False)
    assert install._stop_loaded_server(models_module=stub) is False

    # Truthiness is normalised to a real bool.
    assert install._stop_loaded_server(models_module=_Stub(verdict=1)) is True

    # A raising sibling means shutdown was not confirmed -> unsafe.
    assert install._stop_loaded_server(models_module=_Stub(raises=True)) is False

    # No sibling importable (standalone load): nothing loaded -> safe.
    assert install._stop_loaded_server() is True


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

def test_presets_are_wellformed():
    """Every preset has the keys pull()/serve() read, and unique aliases.

    Guards the class of bug where a new catalog entry is missing a field or
    silently collides with an existing alias (which would make /v1/models
    ambiguous).
    """
    models = _load_models()
    presets = models.MODEL_PRESETS
    assert presets, "preset catalog is empty"
    aliases = []
    for key, p in presets.items():
        for field in ("alias", "repo", "file", "size_gb", "note"):
            assert field in p, f"preset '{key}' missing '{field}'"
        assert p["file"].endswith(".gguf"), f"preset '{key}' file is not a .gguf"
        assert "/" in p["repo"], f"preset '{key}' repo is not Org/Repo"
        assert isinstance(p["size_gb"], (int, float)) and p["size_gb"] > 0
        aliases.append(p["alias"])
    assert len(aliases) == len(set(aliases)), f"duplicate aliases: {aliases}"
    # At least one permissively-licensed option must ship.
    assert any("Qwen/" in p["repo"] or "SmolLM2" in p["repo"] for p in presets.values())


def test_settings_cpu_tuning_defaults_and_overrides():
    """_settings() exposes the CPU-tuning knobs and honours env overrides."""
    models = _load_models()
    keys = ("LLAMA_CPP_THREADS", "LLAMA_CPP_CACHE_TYPE_K",
            "LLAMA_CPP_CACHE_TYPE_V", "LLAMA_CPP_JINJA")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        s = models._settings()
        assert s["cache_type_k"] == "q8_0", s
        assert s["cache_type_v"] == "q8_0", s
        assert s["jinja"] is True, s
        assert isinstance(s["threads"], int) and s["threads"] >= 0, s

        os.environ["LLAMA_CPP_CACHE_TYPE_K"] = "f16"
        os.environ["LLAMA_CPP_JINJA"] = "false"
        os.environ["LLAMA_CPP_THREADS"] = "3"
        s = models._settings()
        assert s["cache_type_k"] == "f16", s
        assert s["jinja"] is False, s
        assert s["threads"] == 3, s

        # Garbage int must not raise — falls back to the detected default.
        os.environ["LLAMA_CPP_THREADS"] = "not-a-number"
        assert isinstance(models._settings()["threads"], int)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_register_wires_bundled_skill():
    """register() registers the bundled SKILL.md, and survives older cores.

    The skill file shipped in the repo for a while with zero references — this
    guards against it going orphaned again. Registration must also degrade
    quietly on a Hermes build whose PluginContext has no register_skill.
    """
    import types

    pkg_name = "hermes_llama_reg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(_REPO_ROOT)]
    sys.modules[pkg_name] = pkg
    sys.modules[f"{pkg_name}.install"] = install
    sys.modules[f"{pkg_name}.models"] = _load_models()
    # provider.py imports the Hermes-only `providers` package; stub it so the
    # package __init__ imports cleanly outside a Hermes runtime.
    providers_stub = types.ModuleType("providers")
    providers_stub.register_provider = lambda profile: None
    base_stub = types.ModuleType("providers.base")

    class _Profile:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
            self.default_headers = {}

    base_stub.ProviderProfile = _Profile
    saved_mods = {k: sys.modules.get(k) for k in ("providers", "providers.base")}
    sys.modules["providers"] = providers_stub
    sys.modules["providers.base"] = base_stub
    try:
        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.__init__", _REPO_ROOT / "__init__.py",
            submodule_search_locations=[str(_REPO_ROOT)],
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg_name
        sys.modules[f"{pkg_name}.entry"] = mod
        spec.loader.exec_module(mod)

        class _Ctx:
            def __init__(self, with_skill=True):
                self.skills = []
                self.commands = []
                self._with_skill = with_skill

            def get_config(self, key):
                return None

            def register_command(self, name, **kw):
                self.commands.append(name)

            def register_cli_command(self, name, **kw):
                self.commands.append(f"cli:{name}")

            def register_skill(self, name, path=None, description=""):
                if not self._with_skill:
                    raise AttributeError("register_skill")
                self.skills.append((name, Path(path), description))

        ctx = _Ctx()
        mod.register(ctx)
        assert len(ctx.skills) == 1, ctx.skills
        name, path, desc = ctx.skills[0]
        # Namespace comes from plugin.yaml's `name`, so the resolvable id is
        # hermes-llama:llama-cpp-local-models.
        assert name == "llama-cpp-local-models", name
        assert path.is_file() and path.name == "SKILL.md", path
        assert desc.strip(), "skill registered without a description"

        # A core without register_skill must not break plugin load.
        ctx2 = _Ctx(with_skill=False)
        mod.register(ctx2)
        assert ctx2.skills == []
        assert "llama" in ctx2.commands
    finally:
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_sha256_helpers_and_verification():
    """_file_sha256 / _verify_sha256 accept a match and REJECT a mismatch.

    A verifier that never rejects is worse than none, so the negative case is
    the load-bearing assertion here.
    """
    import hashlib as _hl

    models = _load_models()
    tmpdir = tempfile.mkdtemp(prefix="llama-sha-")
    try:
        f = Path(tmpdir) / "blob.bin"
        payload = b"hermes-llama sha256 probe\n" * 4096
        f.write_bytes(payload)
        want = _hl.sha256(payload).hexdigest()

        # Streaming hash must equal the one-shot hash (chunk boundaries handled).
        assert models._file_sha256(f) == want, "streaming digest disagrees with hashlib"

        # Match -> silent pass.
        models._verify_sha256(f, want)

        # None/empty -> no-op (cannot verify is not a failure).
        models._verify_sha256(f, None)
        models._verify_sha256(f, "")

        # Mismatch -> must raise, and the message must not leak a full digest.
        try:
            models._verify_sha256(f, "0" * 64)
        except RuntimeError as exc:
            assert "checksum mismatch" in str(exc), exc
        else:
            raise AssertionError("mismatched checksum did NOT raise")

        # A single flipped byte must be caught.
        corrupted = bytearray(payload)
        corrupted[len(corrupted) // 2] ^= 0xFF
        f.write_bytes(bytes(corrupted))
        try:
            models._verify_sha256(f, want)
        except RuntimeError:
            pass
        else:
            raise AssertionError("single-bit corruption was NOT detected")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_expected_sha256_parses_tree_api():
    """_expected_sha256 extracts lfs.oid, and returns None on every unusable shape."""
    models = _load_models()
    import urllib.request as _ur

    class _Resp:
        status = 200

        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            import json as _json

            return _json.dumps(self._payload).encode()

    saved = _ur.urlopen
    try:
        good = "a" * 64
        # Happy path: matching entry with an lfs.oid.
        _ur.urlopen = lambda req, timeout=30: _Resp(
            [{"path": "other.gguf", "lfs": {"oid": "b" * 64}},
             {"path": "m.gguf", "lfs": {"oid": good}}]
        )
        assert models._expected_sha256("Org/Repo", "m.gguf") == good

        # "sha256:<hex>" prefixed form (some mirrors).
        _ur.urlopen = lambda req, timeout=30: _Resp(
            [{"path": "m.gguf", "lfs": {"oid": f"sha256:{good}"}}]
        )
        assert models._expected_sha256("Org/Repo", "m.gguf") == good

        # lfs: null (the model-detail shape that made this "impossible" before).
        _ur.urlopen = lambda req, timeout=30: _Resp([{"path": "m.gguf", "lfs": None}])
        assert models._expected_sha256("Org/Repo", "m.gguf") is None

        # File absent from the tree.
        _ur.urlopen = lambda req, timeout=30: _Resp([{"path": "z.gguf", "lfs": {"oid": good}}])
        assert models._expected_sha256("Org/Repo", "m.gguf") is None

        # Garbage digest (not 64 hex chars) must be rejected, not returned.
        _ur.urlopen = lambda req, timeout=30: _Resp([{"path": "m.gguf", "lfs": {"oid": "nope"}}])
        assert models._expected_sha256("Org/Repo", "m.gguf") is None

        # Non-list payload.
        _ur.urlopen = lambda req, timeout=30: _Resp({"error": "nope"})
        assert models._expected_sha256("Org/Repo", "m.gguf") is None

        # Network failure -> None (never raises into the caller).
        def _boom(req, timeout=30):
            raise OSError("offline")

        _ur.urlopen = _boom
        assert models._expected_sha256("Org/Repo", "m.gguf") is None
    finally:
        _ur.urlopen = saved


def test_download_model_rejects_bad_checksum():
    """_download_model must NOT promote a .part whose digest is wrong.

    Exercises the urllib branch end to end: dest must not exist afterwards and
    the staging file must be cleaned up.
    """
    models = _load_models()
    import urllib.request as _ur

    payload = b"pretend gguf bytes " * 1000

    class _Resp:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            nonlocal payload
            data, payload = payload, b""
            return data

    tmpdir = tempfile.mkdtemp(prefix="llama-dl-")
    saved_urlopen, saved_which = _ur.urlopen, models._download.shutil.which
    try:
        # Force the urllib branch (no curl) so the test is deterministic.
        models._download.shutil.which = lambda name: None
        _ur.urlopen = lambda req, timeout=3600: _Resp()
        dest = Path(tmpdir) / "model.gguf"

        try:
            models._download_model("https://example.invalid/m.gguf", dest,
                                   expected_sha256="f" * 64)
        except RuntimeError as exc:
            assert "checksum mismatch" in str(exc), exc
        else:
            raise AssertionError("bad checksum did NOT abort the download")

        assert not dest.exists(), "corrupt payload was promoted to dest"
        assert not dest.with_suffix(dest.suffix + ".part").exists(), ".part left behind"
    finally:
        _ur.urlopen = saved_urlopen
        models._download.shutil.which = saved_which
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_download_model_resumes_partial():
    """_download_model resumes from a partial .part file via Range header."""
    models = _load_models()
    import urllib.request as _ur

    # The "remaining" bytes that the server should return
    remaining = b"remaining bytes " * 100
    full_payload = b"already have " * 50 + remaining

    class _Resp:
        status = 206  # Partial Content
        headers = {"Content-Length": str(len(remaining))}

        def __init__(self):
            self._data = remaining
            self._sent = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            if self._sent:
                return b""
            self._sent = True
            return self._data

    tmpdir = tempfile.mkdtemp(prefix="llama-resume-")
    saved_urlopen, saved_which = _ur.urlopen, models._download.shutil.which
    try:
        models._download.shutil.which = lambda name: None

        # Track the Range header sent
        captured_headers = {}

        def mock_urlopen(req, timeout=3600):
            captured_headers.update(dict(req.headers))
            return _Resp()

        _ur.urlopen = mock_urlopen
        dest = Path(tmpdir) / "model.gguf"

        # Create a partial file to trigger resume
        partial = dest.with_suffix(dest.suffix + ".part")
        partial.write_bytes(b"already have " * 50)

        models._download_model("https://example.invalid/m.gguf", dest)

        # Verify Range header was sent
        assert "Range" in captured_headers, f"Range header not sent: {captured_headers}"
        assert captured_headers["Range"] == f"bytes={len(b'already have ' * 50)}-", \
            f"wrong Range: {captured_headers['Range']}"

        # Verify the final file has both parts
        assert dest.exists(), "dest not created"
        content = dest.read_bytes()
        assert content == full_payload, f"content mismatch: got {len(content)} bytes"
    finally:
        _ur.urlopen = saved_urlopen
        models._download.shutil.which = saved_which
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_extract_rejects_zip_symlink():
    """The zip branch refuses symlink members, matching the tar branch.

    Scope note, established by experiment: stdlib ``zipfile.extractall`` does
    NOT honour the symlink bit — it writes such a member as a regular file
    containing the target path — so this is defence-in-depth parity with the tar
    branch (which already rejects link members), not a patched live exploit.
    The guard also means only vetted members are ever written.
    """
    import stat as _stat
    import zipfile as _zip

    tmpdir = tempfile.mkdtemp(prefix="llama-zipsym-")
    try:
        archive = Path(tmpdir) / "evil.zip"
        outside = Path(tmpdir) / "outside"
        outside.mkdir()
        dest = Path(tmpdir) / "dest"
        dest.mkdir()

        with _zip.ZipFile(archive, "w") as zf:
            info = _zip.ZipInfo("link")
            info.create_system = 3  # unix
            info.external_attr = (_stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, str(outside))
            zf.writestr("link/pwned.txt", "payload")

        try:
            install._extract(archive, dest)
        except RuntimeError as exc:
            assert "symlink" in str(exc).lower(), exc
        else:
            raise AssertionError("zip symlink member was NOT rejected")

        # Nothing may land outside the extraction root, and the rejection must
        # happen before any member is written inside it either.
        assert not (outside / "pwned.txt").exists(), "payload escaped the extract root"
        assert not (dest / "link").exists(), "member written despite rejection"

        # A benign zip must still extract normally (no over-blocking).
        good = Path(tmpdir) / "good.zip"
        with _zip.ZipFile(good, "w") as zf:
            zf.writestr("build/bin/llama-server", "#!/bin/sh\n")
        dest2 = Path(tmpdir) / "dest2"
        dest2.mkdir()
        install._extract(good, dest2)
        assert (dest2 / "build" / "bin" / "llama-server").is_file()

        # Path traversal (../) must still be blocked — the original guard.
        trav = Path(tmpdir) / "trav.zip"
        with _zip.ZipFile(trav, "w") as zf:
            zf.writestr("../escaped.txt", "nope")
        dest3 = Path(tmpdir) / "dest3"
        dest3.mkdir()
        try:
            install._extract(trav, dest3)
        except RuntimeError as exc:
            assert "traversal" in str(exc).lower(), exc
        else:
            raise AssertionError("zip traversal member was NOT rejected")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_registry_load_preserves_corrupt_file():
    """A corrupt registry is set aside, not silently overwritten with {}."""
    models = _load_models()
    tmpdir = tempfile.mkdtemp(prefix="llama-reg-")
    saved = os.environ.get("LLAMA_CPP_INSTALL_DIR")
    try:
        os.environ["LLAMA_CPP_INSTALL_DIR"] = tmpdir
        path = models._registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json at all", encoding="utf-8")

        assert models._load_registry() == {}, "corrupt registry should read as empty"
        corrupt = list(path.parent.glob("models.json.corrupt-*"))
        assert corrupt, "corrupt registry was discarded instead of preserved"
        assert "not json" in corrupt[0].read_text(encoding="utf-8")

        # A non-dict payload is also rejected (list would break reg[alias]).
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert models._load_registry() == {}

        # Missing file is normal, not corruption.
        path.unlink(missing_ok=True)
        before = len(list(path.parent.glob("models.json.corrupt-*")))
        assert models._load_registry() == {}
        assert len(list(path.parent.glob("models.json.corrupt-*"))) == before
    finally:
        if saved is None:
            os.environ.pop("LLAMA_CPP_INSTALL_DIR", None)
        else:
            os.environ["LLAMA_CPP_INSTALL_DIR"] = saved
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_registry_txn_serializes_concurrent_writers():
    """Interleaved read-modify-write must not lose entries.

    Simulates the real race: two writers each open a transaction and add a
    different alias. Without locking the second save clobbers the first.
    """
    models = _load_models()
    tmpdir = tempfile.mkdtemp(prefix="llama-regtxn-")
    saved = os.environ.get("LLAMA_CPP_INSTALL_DIR")
    try:
        os.environ["LLAMA_CPP_INSTALL_DIR"] = tmpdir

        # Sequential transactions accumulate.
        for i in range(5):
            with models._registry_txn() as reg:
                reg[f"model-{i}"] = {"path": f"/x/{i}", "size_gb": 0.1}
        assert len(models._load_registry()) == 5, models._load_registry()

        # Threads hitting the same transaction must all survive.
        import threading

        errors = []

        def add(n):
            try:
                with models._registry_txn() as reg:
                    reg[f"threaded-{n}"] = {"path": f"/t/{n}", "size_gb": 0.2}
            except Exception as exc:  # noqa: BLE001 — surfaced via errors list
                errors.append(exc)

        threads = [threading.Thread(target=add, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        final = models._load_registry()
        assert len(final) == 9, f"expected 9 entries, got {len(final)}: {sorted(final)}"

        # Unique temp names: no stale fixed-name tmp left behind.
        assert not list(Path(tmpdir).glob("models.json.tmp")), "fixed tmp name still used"
    finally:
        if saved is None:
            os.environ.pop("LLAMA_CPP_INSTALL_DIR", None)
        else:
            os.environ["LLAMA_CPP_INSTALL_DIR"] = saved
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_download_curl_branch_no_second_head():
    """The curl branch must succeed without a second HEAD request.

    This branch was previously untested AND carried its own copy of the
    Xet-stub defect: it fetched Content-Length via `curl -sI -L`, which returns
    the redirect-stub size for Xet assets, failing every download. The stub is
    installed via shutil.which so the test is deterministic; urllib is forced
    OFF to prove we are really exercising the curl path.
    """
    models = _load_models()

    payload = b"curl-branch body " * 200

    tmpdir = tempfile.mkdtemp(prefix="llama-curlbr-")
    fake_bin = Path(tmpdir) / "bin"
    fake_bin.mkdir()
    curl_stub = fake_bin / "curl"
    # A stand-in curl: records argv, writes a deterministic payload to whatever
    # follows -o, exits 0. Robust against any flag ordering.
    curl_stub.write_text(
        "#!/bin/sh\n"
        'echo "$@" >> "$CURL_LOG"\n'
        "prev=\n"
        'for a in "$@"; do\n'
        '  if [ "$prev" = "-o" ]; then printf "%s" "' + payload.decode("latin-1").replace('"', '\\"') + '" > "$a"; fi\n'
        '  prev=$a\n'
        "done\n"
        "exit 0\n",
        encoding="latin-1",
    )
    curl_stub.chmod(0o755)
    log = Path(tmpdir) / "argv.log"

    saved_which = models._download.shutil.which
    saved_env = os.environ.get("CURL_LOG")
    try:
        os.environ["CURL_LOG"] = str(log)
        models._download.shutil.which = lambda name: str(curl_stub) if name == "curl" else None
        dest = Path(tmpdir) / "m.gguf"

        models._download_model("https://example.invalid/m.gguf", dest)

        assert dest.is_file() and dest.stat().st_size == len(payload), (
            dest.stat().st_size if dest.exists() else "dest missing")
        calls = log.read_text().splitlines()
        # Exactly one invocation: no separate HEAD round-trip.
        assert len(calls) == 1, f"expected 1 curl call, got {len(calls)}: {calls}"
        assert not any("-sI" in c or "--head" in c for c in calls), calls
    finally:
        models._download.shutil.which = saved_which
        if saved_env is None:
            os.environ.pop("CURL_LOG", None)
        else:
            os.environ["CURL_LOG"] = saved_env
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_download_curl_branch_zero_byte_fails():
    """A zero-byte curl success must be rejected, not promoted."""
    models = _load_models()
    tmpdir = tempfile.mkdtemp(prefix="llama-curlz-")
    fake_bin = Path(tmpdir) / "bin"
    fake_bin.mkdir()
    curl_stub = fake_bin / "curl"
    curl_stub.write_text(
        "#!/bin/sh\n"
        "# create an EMPTY file at whatever follows -o, then exit 0\n"
        "prev=\n"
        'for a in "$@"; do\n'
        '  if [ "$prev" = "-o" ]; then : > "$a"; fi\n'
        '  prev=$a\n'
        "done\n"
        "exit 0\n"
    )
    curl_stub.chmod(0o755)
    saved_which = models._download.shutil.which
    try:
        models._download.shutil.which = lambda name: str(curl_stub) if name == "curl" else None
        dest = Path(tmpdir) / "m.gguf"
        try:
            models._download_model("https://example.invalid/m.gguf", dest)
        except RuntimeError as exc:
            assert "0 bytes" in str(exc), exc
        else:
            raise AssertionError("zero-byte download was accepted")
        assert not dest.exists(), "empty file promoted to dest"
    finally:
        models._download.shutil.which = saved_which
        shutil.rmtree(tmpdir, ignore_errors=True)


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
