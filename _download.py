"""Shared download helper for hermes-llama — curl + urllib with .part staging."""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Callable


def _parse_expected_len(resp, resume_offset: int) -> int | None:
    """Parse expected download length from response headers."""
    try:
        raw = resp.headers.get("X-Linked-Size") if hasattr(resp, "headers") else None
        if raw:
            return int(str(raw).strip())
        raw = resp.headers.get("Content-Length") if hasattr(resp, "headers") else None
        if raw:
            content_len = int(str(raw).strip())
            # For a 206 response, Content-Length is the range size,
            # not the full file size. Add the resume offset.
            return content_len + resume_offset if resume_offset > 0 else content_len
    except Exception:  # noqa: BLE001 — header parsing is best-effort
        pass
    return None


def _stream_response(resp, tmp: Path, resume_offset: int) -> None:
    """Stream response body to tmp file."""
    with open(tmp, "ab" if resume_offset > 0 else "wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
        out.flush()


def download_file(
    url: str,
    dest: Path,
    *,
    timeout: int = 600,
    verify: Callable[[Path], None] | None = None,
) -> None:
    """Download a file robustly.

    Prefers ``curl`` when available because Python's ``urllib`` can stall for
    minutes before the first byte arrives against Hugging Face's Xet CDN
    redirects. ``curl`` starts the transfer immediately. Falls back to
    ``urllib`` when ``curl`` is absent. Raises on failure.

    The download is staged to a ``.part`` file and promoted to *dest* only
    on success, so an interrupted download never leaves a partial file under
    its final name. When *verify* is supplied, the staged file is passed to
    it before promotion — used for checksum verification.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    curl = shutil.which("curl")
    try:
        if curl:
            proc = subprocess.run(
                [curl, "-L", "--fail", "--retry", "3", "--retry-delay", "2",
                 "--retry-all-errors", "-C", "-",
                 "-A", "hermes-llama", "-o", str(tmp), url],
                capture_output=True, text=True, timeout=timeout,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"curl download failed: {proc.stderr.strip()[:300]} (exit {proc.returncode})"
                )
            if tmp.stat().st_size == 0:
                raise RuntimeError("download truncated: curl wrote 0 bytes")
            if verify is not None:
                verify(tmp)
            os.replace(tmp, dest)
            return
        req = urllib.request.Request(url, headers={"User-Agent": "hermes-llama"})
        # Resume: if a partial download exists, send a Range header to continue
        # from where we left off. The server must support HTTP 206 (partial
        # content); if it returns 200 instead, the server ignored Range and we
        # restart from scratch.
        resume_offset = 0
        if tmp.is_file() and tmp.stat().st_size > 0:
            resume_offset = tmp.stat().st_size
            req.add_header("Range", f"bytes={resume_offset}-")

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) == 416:
                # Range not satisfiable — file already complete or server
                # doesn't have the requested range. Restart from scratch.
                resume_offset = 0
                tmp.unlink(missing_ok=True)
                req.headers.pop("Range", None)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if getattr(resp, "status", 200) != 200:
                        raise RuntimeError(f"download failed: HTTP {getattr(resp, 'status', '?')}")
                    expected_len = _parse_expected_len(resp, resume_offset)
                    _stream_response(resp, tmp, resume_offset)
            else:
                if getattr(resp, "status", 200) not in (200, 206):
                    raise RuntimeError(f"download failed: HTTP {getattr(resp, 'status', '?')}")
                # If the server returned 200 when we asked for a Range, it ignored
                # the header — restart from scratch.
                if resume_offset > 0 and getattr(resp, "status", 200) == 200:
                    resume_offset = 0
                    tmp.unlink(missing_ok=True)
                expected_len = _parse_expected_len(resp, resume_offset)
                _stream_response(resp, tmp, resume_offset)
            if expected_len is not None and tmp.stat().st_size != expected_len:
                raise RuntimeError(
                    f"download truncated: expected {expected_len} bytes, got {tmp.stat().st_size}"
                )
        if verify is not None:
            verify(tmp)
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
