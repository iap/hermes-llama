"""Llama CPP provider profile for Hermes Agent.

Registers the ``llama-cpp`` provider so a local ``llama-server`` (llama.cpp's
OpenAI-compatible server) is selectable as **"Llama CPP"** in the model picker.
Because the registry auto-extends ``CANONICAL_PROVIDERS`` in
``hermes_cli/models.py``, this profile surfaces in ``hermes model`` / ``/model``
with no other changes.

The provider is intentionally OpenAI-compatible: llama-server exposes
``/v1/models``, ``/v1/chat/completions``, and ``/health`` on the configured
host/port (default ``http://127.0.0.1:8080/v1``). No API key is required unless
the user launched llama-server with ``--api-key``.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


def _env_base_url() -> str:
    """Resolve the llama-server base URL: env override > config default."""
    return os.environ.get("LLAMA_CPP_BASE_URL", "").strip() or DEFAULT_BASE_URL


class LlamaCppProfile(ProviderProfile):
    """Local llama.cpp llama-server (OpenAI-compatible, no key by default)."""

    def __init__(self) -> None:
        super().__init__(
            name="llama-cpp",
            aliases=("llamacpp", "llama"),
            display_name="Llama CPP",
            description=(
                "Local llama.cpp server — run GGUF models on-device "
                "(OpenAI-compatible)"
            ),
            env_vars=("LLAMA_CPP_API_KEY", "LLAMA_CPP_BASE_URL"),
            base_url=_env_base_url(),
            auth_type="api_key",  # key is optional; empty -> no Authorization header
            supports_health_check=True,
            fallback_models=(),
        )

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """List models currently loaded in llama-server via ``GET /v1/models``.

        llama-server returns an OpenAI-shaped payload::

            {"object": "list", "data": [{"id": "<model-alias>", "object": "model"}]}

        The ``id`` is the GGUF filename stem unless the server was started with
        ``--alias``. We return those ids directly so they appear under the
        "Llama CPP" provider. Returns ``None`` (with a logged reason) when the
        server is down so callers fall back to the static list.
        """
        effective_base = (base_url or self.base_url or DEFAULT_BASE_URL).rstrip("/")
        # llama-server serves OpenAI-compat under /v1; /v1/models is the catalog.
        url = f"{effective_base}/models"
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        for k, v in self.default_headers.items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            items = data if isinstance(data, list) else data.get("data", [])
            ids = [m["id"] for m in items if isinstance(m, dict) and m.get("id")]
            return ids or None
        except Exception as exc:  # server not running / wrong port
            logger.debug("fetch_models(llama-cpp): %s", exc)
            return None


_profile = LlamaCppProfile()


def register() -> None:
    """Idempotent provider registration (last-writer-wins in the registry)."""
    register_provider(_profile)


# Self-register on import, mirroring every bundled model-provider plugin's
# ``__init__.py`` contract. Also re-entrant: the general plugin's ``register(ctx)``
# calls :func:`register` explicitly so the profile is available even if provider
# discovery ran before plugin load.
register()
