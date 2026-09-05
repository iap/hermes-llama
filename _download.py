"""Shared download helper for hermes-llama — curl + urllib with .part staging."""

from __future__ import annotations

import os
import shutil
import sys
import subprocess
import urllib.error
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


def _parse_content_range_total(exc) -> int | None:
    """Total transfer size from an HTTP 416's ``Content-Range: bytes */N``."""
    try:
        raw = exc.headers.get("Content-Range") if getattr(exc, "headers", None) else None
        if raw and "/" in raw:
            total = raw.rsplit("/", 1)[1].strip()
            if total.isdigit():
                return int(total)
    except Exception:  # noqa: BLE001 — header parsing is best-effort
        pass
    return None


def _stream_response(resp, tmp: Path, resume_offset: int) -> None:
    """Stream response body to tmp file."""
    with open(tmp, "ab" if resume_offset > 0 else "wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
        out.flush()


def _request_without_range(req):
    """Return a fresh request with the Range header removed."""
    new_req = urllib.request.Request(req.full_url, headers=dict(req.headers))
    new_req.headers.pop("Range", None)
    return new_req


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

    Resume semantics (urllib branch): a ``Range`` request answered with 206
    continues the ``.part``; answered with 200 (server ignored Range) it
    restarts from scratch; answered with 416 the ``.part`` has reached (or
    passed) EOF — it is promoted only when a *verify* callback can prove its
    content identity, otherwise it is discarded and the download restarts.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    curl = shutil.which("curl")
    try:
        if curl:
            argv = [curl, "-L", "--fail", "--retry", "3", "--retry-delay", "2",
                    "--retry-all-errors", "-C", "-",
                    "-A", "hermes-llama", "-o", str(tmp), url]
            # .bat/.cmd shims cannot be exec'd directly by CreateProcess on
            # Windows — they need a cmd.exe wrapper (same as typing `curl` in
            # a shell). Real-world curl.bat shims make this path real.
            if sys.platform == "win32" and curl.lower().endswith((".bat", ".cmd")):
                argv = ["cmd", "/c", *argv]
            proc = subprocess.run(
                argv,
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

        expected_len = None
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 416:
                tmp.unlink(missing_ok=True)
                raise
            # Stock urlopen raises HTTPError for every non-2xx status, so a 416
            # never reaches a `resp.status == 416` check on a response object —
            # the "Range Not Satisfiable" recovery has to happen here. The
            # Content-Range header carries the true total: a .part already that
            # size is a finished transfer whose promotion was interrupted
            # (crash/kill between download end and os.replace) — keep it and
            # let the verify + promote steps below run. Anything else is an
            # unusable .part: discard it and restart from scratch.
            total = _parse_content_range_total(exc)
            if (
                verify is not None
                and total is not None
                and resume_offset > 0
                and resume_offset == total
            ):
                # A finished transfer whose promotion was interrupted
                # (crash/kill between download end and os.replace). Keep the
                # .part and let the verify + promote steps below run — the
                # caller's verify is what proves the bytes are the ones asked
                # for. Without a verify callback there is no content identity:
                # a same-size stale .part (upstream file replaced since the
                # interrupted run) must not be trusted on byte count alone.
                expected_len = total  # nothing left to stream
                resp = None
            else:
                tmp.unlink(missing_ok=True)
                resume_offset = 0
                resp = urllib.request.urlopen(_request_without_range(req), timeout=timeout)
        if resp is not None:
            with resp:
                status = getattr(resp, "status", 200)
                if status not in (200, 206):
                    raise RuntimeError(f"download failed: HTTP {status}")
                # A 200 to a Range request means the server ignored the header
                # and is resending the whole body — drop the stale .part.
                if resume_offset > 0 and status == 200:
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
