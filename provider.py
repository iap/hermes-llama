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

        The ``id`` defaults to the full ``-m`` model path unless the server was
        started with ``--alias`` (which this plugin always does). We return those
        ids directly so they appear under the "Llama CPP" provider. Returns
        ``None`` (with a logged reason) when the server is down so callers fall
        back to the static list.
        """
        effective_base = (base_url or self.base_url or DEFAULT_BASE_URL).rstrip("/")
        # Normalize: users may set base without /v1; llama-server serves under /v1
        if not effective_base.endswith("/v1"):
            effective_base = effective_base.rstrip("/") + "/v1"
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


# ── Dashboard visibility (config-scoped, plugin-owned) ──────────────────────
#
# The dashboard's model list is built from config.yaml — specifically the
# ``custom_providers`` list (web_server._models_from_custom_endpoint_entry) —
# NOT from provider profiles. So downloaded models stay invisible there until
# an entry exists. We keep that entry in sync from here, scoped strictly to
# our own data:
#
#   * we only ever add/update the entry named "Llama CPP" whose base_url is
#     OUR resolved endpoint (matched by URL, not by name);
#   * we never touch sibling entries belonging to other plugins or the user;
#   * models listed are exactly what OUR registry reports as downloaded;
#   * a failed write is swallowed — dashboard visibility is best-effort and
#     must never break pull/serve.
#
# This survives core config migrations: even if _config_version bumps or
# core rewrites other sections, it only re-reads whatever custom_providers
# contains and we reconcile just our own row.


def _sync_dashboard_entry() -> None:
    """Best-effort sync of downloaded models into config.yaml for the dashboard.

    Reads the live registry (source of truth for what is downloaded), then
    upserts a single ``custom_providers`` entry keyed by our base_url. Sibling
    entries are left untouched. Any failure is non-fatal.
    """
    try:
        from hermes_cli.config import load_config, save_config  # noqa: PLC0415
        from pathlib import Path

        # Resolve the registry file directly from the environment rather than
        # importing sibling modules — PluginManager can load provider.py and
        # models.py under fresh per-call module namespaces, so a relative import
        # here may bind to an empty instance. Reading the JSON on disk is
        # robust to any module-wiring order and has no dependency on import
        # mechanics.
        root = os.environ.get("LLAMA_CPP_INSTALL_DIR", "").strip()
        if not root:
            # install_root() default when env is unset
            root = str(
                Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
                / "llama-cpp"
            )
        reg_path = Path(root) / "models.json"
        if reg_path.is_file():
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            if isinstance(reg, dict):
                # Each registry row may or may not carry an "alias" field.
                # Prefer the explicit alias, then the preset-style key, then
                # a human form of the repo id (Org/Repo → "Repo").
                models = []
                for key, e in reg.items():
                    if not isinstance(e, dict):
                        continue
                    alias = e.get("alias") or key
                    if alias:
                        models.append(alias)
            else:
                models = []
        else:
            models = []

        cfg = load_config()

        providers = cfg.get("custom_providers")
        if not isinstance(providers, list):
            providers = []

        base = _env_base_url().rstrip("/")

        ours = None
        for entry in providers:
            if isinstance(entry, dict) and str(entry.get("base_url", "")).rstrip("/") == base:
                ours = entry
                break

        # The dashboard reads endpoints from cfg["providers"] (a dict keyed by
        # id), NOT from custom_providers — mirror our entry there too so the
        # "switch model" list sees it. Scoped the same way: URL-matched, ours
        # only, siblings untouched. (Read-only search: runs before any mutation
        # so the snapshot below sees the original state.)
        pmap = cfg.get("providers")
        if not isinstance(pmap, dict):
            pmap = {}
        prow = None
        matched_pid = None
        for pid, row in pmap.items():
            if isinstance(row, dict) and str(row.get("base_url", "")).rstrip("/") == base:
                prow = row
                matched_pid = pid
                break

        # Snapshot our rows before mutating: register() runs on every Hermes
        # start, and an unconditional save_config() would turn every load into
        # a config.yaml rewrite (mtime churn, clobber races with concurrent
        # hermes processes) even when nothing moved.
        before = json.dumps([providers, prow], sort_keys=True, default=str)

        if ours is None:
            ours = {"name": "Llama CPP", "base_url": base}
            providers.append(ours)

        ours["api_mode"] = "chat_completions"
        ours["model"] = models[0] if models else ""
        # dict form = per-model metadata; keys become the visible model ids.
        ours["models"] = {m: {} for m in models} if models else {}
        ours["discover_models"] = True  # /v1/models is live on our server

        if prow is None:
            pid_ours = "llama-cpp"
            prow = {"name": "Llama CPP"}
        else:
            pid_ours = matched_pid
        pmap[pid_ours] = prow
        prow.update({
            "name": "Llama CPP",
            "base_url": base,
            "transport": "openai_chat",
            "api_mode": "chat_completions",
            "model": models[0] if models else "",
            "models": models,
            "discover_models": True,
        })
        cfg["providers"] = pmap

        cfg["custom_providers"] = providers
        after = json.dumps([providers, prow], sort_keys=True, default=str)
        if after != before:
            save_config(cfg)
    except Exception:  # noqa: BLE001 — visibility is best-effort
        logger.debug("dashboard entry sync skipped", exc_info=True)


def register() -> None:

    """(Re)register a freshly-built profile (last-writer-wins in the registry).

    A new profile is built on every call so ``base_url`` is read from the current
    environment — which lets settings applied by ``_wire_config`` (which sets
    ``LLAMA_CPP_BASE_URL`` before this runs) take effect.
    """
    register_provider(LlamaCppProfile())
    # Keep the dashboard's custom_providers row in sync with downloaded models
    # (best-effort; scoped to our own entry only).
    _sync_dashboard_entry()
