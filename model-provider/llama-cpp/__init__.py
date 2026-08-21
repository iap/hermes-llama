"""Llama CPP model-provider plugin (self-contained).

Drop this directory into ``$HERMES_HOME/plugins/model-providers/llama-cpp/``
(or the bundled ``plugins/model-providers/`` dir) so the "Llama CPP" provider
auto-injects into ``CANONICAL_PROVIDERS`` via standard discovery — independent
of the general ``hermes-llama`` plugin load timing.

The profile points at the local ``llama-server`` (llama.cpp's OpenAI-compatible
server). No API key is required unless the server was started with ``--api-key``.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


class LlamaCppProfile(ProviderProfile):
    def __init__(self) -> None:
        super().__init__(
            name="llama-cpp",
            aliases=("llamacpp", "llama"),
            display_name="Llama CPP",
            description="Local llama.cpp server — run GGUF models on-device (OpenAI-compatible)",
            env_vars=("LLAMA_CPP_API_KEY", "LLAMA_CPP_BASE_URL"),
            base_url=os.environ.get("LLAMA_CPP_BASE_URL", "").strip() or DEFAULT_BASE_URL,
            auth_type="api_key",
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
        effective_base = (base_url or self.base_url or DEFAULT_BASE_URL).rstrip("/")
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
        except Exception as exc:
            logger.debug("fetch_models(llama-cpp): %s", exc)
            return None


register_provider(LlamaCppProfile())
