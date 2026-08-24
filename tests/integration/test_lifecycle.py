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


def test_lifecycle_pull_serve_stop():
    """Full lifecycle: pull a model, serve it, verify running, stop it."""
    models = _load_models()
    tmpdir = Path(tempfile.mkdtemp(prefix="llama-lifecycle-"))
    try:
        # Override install dir to our temp dir
        os.environ["LLAMA_CPP_INSTALL_DIR"] = str(tmpdir)
        os.environ["LLAMA_CPP_MODELS_DIR"] = str(tmpdir / "models")
        os.environ["LLAMA_CPP_PORT"] = "18080"  # non-default port to avoid conflicts

        # Create a fake GGUF file in the expected location
        models_dir = tmpdir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        # _model_dest creates: models_dir / repo.replace("/", "__") / local_name
        repo_dir = models_dir / "Org__Repo"
        repo_dir.mkdir(parents=True, exist_ok=True)
        gguf_file = repo_dir / "test.gguf"
        gguf_file.write_bytes(b"mock gguf content " * 60000)  # > 1 MiB to avoid re-download guard

        # Create a mock binary
        mock_bin = tmpdir / "bin" / "llama-server"
        mock_bin.parent.mkdir(parents=True, exist_ok=True)
        mock_bin.write_text("#!/bin/sh\necho mock\n")

        # Track mock process state
        mock_state = {"alive": True, "pid": 99999}

        # Patch find_binary to return our mock
        models.install.find_binary = lambda: mock_bin
        # Patch _pick_gguf_file to return our test file
        models._pick_gguf_file = lambda repo: "test.gguf"
        # Patch _wait_healthy to immediately return True (server is "ready")
        models._wait_healthy = lambda base, timeout=60: True

        # Patch subprocess.Popen to return a mock process
        class MockProc:
            pid = mock_state["pid"]
            def __init__(self, *args, **kwargs):
                pass
            def wait(self, timeout=None):
                # Simulate a running server — always timeout
                raise subprocess.TimeoutExpired(cmd="llama-server", timeout=timeout or 1)

        original_popen = models.subprocess.Popen
        models.subprocess.Popen = MockProc

        # Patch _pid_alive to track mock state
        original_pid_alive = models._pid_alive
        models._pid_alive = lambda pid: mock_state["alive"] if pid == mock_state["pid"] else False

        # Patch _is_llama_server to always return True for our mock
        original_is_llama = models._is_llama_server
        models._is_llama_server = lambda pid: pid == mock_state["pid"]

        # Patch the kill mechanism to update mock state
        original_killpg = None
        if sys.platform == "win32":
            # On Windows, stop() uses subprocess.run(["taskkill", ...])
            original_run = models.subprocess.run
            def mock_run(cmd, **kwargs):
                if "taskkill" in str(cmd):
                    mock_state["alive"] = False
                # Return a mock result
                class MockResult:
                    returncode = 0
                    stdout = ""
                    stderr = ""
                return MockResult()
            models.subprocess.run = mock_run
        else:
            # On POSIX, stop() uses os.killpg()
            original_killpg = models.os.killpg
            def mock_killpg(pid, sig):
                if pid == mock_state["pid"]:
                    mock_state["alive"] = False
            models.os.killpg = mock_killpg

        try:
            # Pull the model (will find the existing file and register it)
            result = models.pull("Org/Repo", "test-model")
            assert "registered" in result or "Downloaded" in result or "Already present" in result, result

            # Verify registry has the entry
            reg = models._load_registry()
            assert "test-model" in reg, f"model not in registry: {reg}"

            # Serve the model
            result = models.serve("test-model")
            assert "Started" in result or "ready" in result, result

            # Verify server is running
            status = models.status()
            assert "yes" in status, f"server not running: {status}"

            # Stop the server (kill will set mock_state["alive"] = False)
            result = models.stop()
            assert "Stopped" in result, f"stop failed: {result}"

            # Verify server is stopped
            status = models.status()
            assert "no" in status, f"server still running: {status}"
        finally:
            models.subprocess.Popen = original_popen
            models._pid_alive = original_pid_alive
            models._is_llama_server = original_is_llama
            if sys.platform == "win32":
                models.subprocess.run = original_run
            else:
                models.os.killpg = original_killpg
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
