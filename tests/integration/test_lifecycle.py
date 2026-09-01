"""Integration tests for the full pull → serve → stop lifecycle.

Uses mocked process management to exercise the real lifecycle code paths
without requiring an actual llama.cpp build or GPU.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_models():
    """Load models.py with a synthetic package so ``from . import install`` works."""
    import types

    # Load install.py standalone first
    spec = importlib.util.spec_from_file_location("hermes_install", _REPO_ROOT / "install.py")
    assert spec is not None and spec.loader is not None
    install_mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_install"] = install_mod
    spec.loader.exec_module(install_mod)

    # Build a throwaway package for models.py
    pkg_name = "hermes_llama_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(_REPO_ROOT)]
    sys.modules[pkg_name] = pkg
    sys.modules[f"{pkg_name}.install"] = install_mod

    spec = importlib.util.spec_from_file_location(f"{pkg_name}.models", _REPO_ROOT / "models.py")
    assert spec is not None and spec.loader is not None
    models_mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{pkg_name}.models"] = models_mod
    spec.loader.exec_module(models_mod)
    return models_mod


def _make_mock(models, tmpdir):
    """Set up mock binary, process, and patches for a lifecycle test."""
    mock_bin = tmpdir / "bin" / "llama-server"
    mock_bin.parent.mkdir(parents=True, exist_ok=True)
    mock_bin.write_text("#!/bin/sh\necho mock\n")

    mock_state = {"alive": True, "pid": 99999}

    models.install.find_binary = lambda: mock_bin
    models._pick_gguf_file = lambda repo: "test.gguf"
    models._wait_healthy = lambda base, timeout=60: True

    class MockProc:
        pid = mock_state["pid"]
        def __init__(self, *args, **kwargs):
            pass
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="llama-server", timeout=timeout or 1)

    original_popen = models.subprocess.Popen
    models.subprocess.Popen = MockProc

    original_pid_alive = models._pid_alive
    models._pid_alive = lambda pid: mock_state["alive"] if pid == mock_state["pid"] else False

    original_is_llama = models._is_llama_server
    models._is_llama_server = lambda pid: pid == mock_state["pid"]

    def restore():
        models.subprocess.Popen = original_popen
        models._pid_alive = original_pid_alive
        models._is_llama_server = original_is_llama

    return mock_state, restore


def test_lifecycle_pull_serve_stop():
    """Full lifecycle: pull a model, serve it, verify running, stop it."""
    models = _load_models()
    tmpdir = Path(tempfile.mkdtemp(prefix="llama-lifecycle-"))
    try:
        os.environ["LLAMA_CPP_INSTALL_DIR"] = str(tmpdir)
        os.environ["LLAMA_CPP_MODELS_DIR"] = str(tmpdir / "models")
        os.environ["LLAMA_CPP_PORT"] = "18080"

        models_dir = tmpdir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        repo_dir = models_dir / "Org__Repo"
        repo_dir.mkdir(parents=True, exist_ok=True)
        gguf_file = repo_dir / "test.gguf"
        gguf_file.write_bytes(b"mock gguf content " * 60000)

        mock_state, restore = _make_mock(models, tmpdir)

        original_run = models.subprocess.run
        original_killpg = None
        if sys.platform == "win32":
            def mock_run(cmd, **kwargs):
                if "taskkill" in str(cmd):
                    mock_state["alive"] = False
                class MockResult:
                    returncode = 0
                    stdout = ""
                    stderr = ""
                return MockResult()
            models.subprocess.run = mock_run
        else:
            original_killpg = models.os.killpg
            def mock_killpg(pid, sig):
                if pid == mock_state["pid"]:
                    mock_state["alive"] = False
            models.os.killpg = mock_killpg

        try:
            result = models.pull("Org/Repo", "test-model")
            assert "registered" in result or "Downloaded" in result or "Already present" in result, result

            reg = models._load_registry()
            assert "test-model" in reg, f"model not in registry: {reg}"

            result = models.serve("test-model")
            assert "Started" in result or "ready" in result, result

            status = models.status()
            assert "yes" in status, f"server not running: {status}"

            result = models.stop()
            assert "Stopped" in result, f"stop failed: {result}"

            status = models.status()
            assert "no" in status, f"server still running: {status}"
        finally:
            restore()
            if sys.platform == "win32":
                models.subprocess.run = original_run
            else:
                models.os.killpg = original_killpg
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_serve_preset_by_key():
    """Serving a preset by its key (e.g. ``liquidai``) works if already pulled."""
    models = _load_models()
    tmpdir = Path(tempfile.mkdtemp(prefix="llama-preset-serve-"))
    try:
        os.environ["LLAMA_CPP_INSTALL_DIR"] = str(tmpdir)
        os.environ["LLAMA_CPP_MODELS_DIR"] = str(tmpdir / "models")
        os.environ["LLAMA_CPP_PORT"] = "18081"

        # Create the GGUF at the location the liquidai preset resolves to
        models_dir = tmpdir / "models"
        repo_dir = models_dir / "LiquidAI__LFM2-1.2B-GGUF"
        repo_dir.mkdir(parents=True, exist_ok=True)
        gguf_file = repo_dir / "LFM2-1.2B-Q4_K_M.gguf"
        gguf_file.write_bytes(b"mock gguf content " * 60000)

        mock_state, restore = _make_mock(models, tmpdir)

        try:
            # Serve by preset key — should resolve via _resolve_model
            result = models.serve("liquidai")
            assert "Started" in result or "ready" in result, result

            # Verify the registry was NOT modified (preset serve doesn't register)
            reg = models._load_registry()
            assert "liquidai-lfm2-1.2b" not in reg, "preset serve should not create registry entry"
        finally:
            restore()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_serve_stale_registry_entry():
    """Serving a model whose file was deleted returns a clear error."""
    models = _load_models()
    tmpdir = Path(tempfile.mkdtemp(prefix="llama-stale-"))
    try:
        os.environ["LLAMA_CPP_INSTALL_DIR"] = str(tmpdir)
        os.environ["LLAMA_CPP_MODELS_DIR"] = str(tmpdir / "models")
        os.environ["LLAMA_CPP_PORT"] = "18082"

        mock_bin = tmpdir / "bin" / "llama-server"
        mock_bin.parent.mkdir(parents=True, exist_ok=True)
        mock_bin.write_text("#!/bin/sh\necho mock\n")
        models.install.find_binary = lambda: mock_bin

        # Register a model whose file does NOT exist
        reg = {
            "ghost": {
                "repo": "Org/Repo",
                "file": "ghost.gguf",
                "path": "Org__Repo/ghost.gguf",
                "size_gb": 1.0,
            }
        }
        models._save_registry(reg)

        result = models.serve("ghost")
        assert "not found" in result or "not downloaded" in result.lower(), result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_serve_unknown_model():
    """Serving an unknown model spec returns a clear error."""
    models = _load_models()
    tmpdir = Path(tempfile.mkdtemp(prefix="llama-unknown-"))
    try:
        os.environ["LLAMA_CPP_INSTALL_DIR"] = str(tmpdir)
        os.environ["LLAMA_CPP_MODELS_DIR"] = str(tmpdir / "models")
        os.environ["LLAMA_CPP_PORT"] = "18083"

        mock_bin = tmpdir / "bin" / "llama-server"
        mock_bin.parent.mkdir(parents=True, exist_ok=True)
        mock_bin.write_text("#!/bin/sh\necho mock\n")
        models.install.find_binary = lambda: mock_bin

        result = models.serve("nonexistent-model-xyz")
        assert "Unknown" in result or "not downloaded" in result.lower(), result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_stop_no_server_running():
    """Stop when no server is running returns a clear message."""
    models = _load_models()
    tmpdir = Path(tempfile.mkdtemp(prefix="llama-nostop-"))
    try:
        os.environ["LLAMA_CPP_INSTALL_DIR"] = str(tmpdir)
        os.environ["LLAMA_CPP_MODELS_DIR"] = str(tmpdir / "models")

        result = models.stop()
        assert "No llama-server running" in result, result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_status_no_binary():
    """Status when no binary is installed reports NOT installed."""
    models = _load_models()
    tmpdir = Path(tempfile.mkdtemp(prefix="llama-nobin-"))
    try:
        os.environ["LLAMA_CPP_INSTALL_DIR"] = str(tmpdir)
        os.environ["LLAMA_CPP_MODELS_DIR"] = str(tmpdir / "models")

        status = models.status()
        assert "NOT installed" in status, status
        assert "no" in status, status
    finally:
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
