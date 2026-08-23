"""hermes-llama plugin — check/install/uninstall llama.cpp, run local GGUF models.

Registers:
- Provider profile ``llama-cpp`` (display name "Llama CPP") so a local
  llama-server shows up as a first-class provider in ``hermes model`` / ``/model``.
- Slash command ``/llama`` and CLI command ``hermes llama`` for managing
  the binary and models.

Subcommands: check, install, uninstall, status, models, pull, serve, stop, help.
"""

from __future__ import annotations

import os
from typing import Any

from . import install, models, provider

HELP = (
    "/llama <subcommand> — manage llama.cpp and local GGUF models\n"
    "  check     — is llama-server installed? which version?\n"
    "  install   — install llama.cpp (prebuilt / source) for this OS\n"
    "  upgrade   — reinstall to the latest release\n"
    "  uninstall — remove llama.cpp\n"
    "  status    — server endpoint + health (provider 'Llama CPP')\n"
    "  models    — list presets + downloaded GGUF models\n"
    "  pull <m>  — download a model: `pull liquidai` or `pull Org/Repo`\n"
    "  serve <m> — start llama-server for a downloaded model\n"
    "  stop      — stop the running llama-server\n"
    "  help      — this text"
)


def _fmt_dict(d: dict) -> str:
    if d.get("skipped"):
        return d.get("detail", "Skipped.")
    status = "OK" if d.get("ok") else "FAILED"
    method = f" [{d.get('method')}]" if d.get("method") else ""
    detail = d.get("detail", "")
    return f"{status}{method} {detail}".strip()


def _dispatch(cmd: str, argv: list[str]) -> str:
    cmd = (cmd or "help").strip().lower()
    if cmd == "help":
        return HELP
    if cmd == "check":
        r = install.check()
        if not r["installed"]:
            return "llama-server is NOT installed. Run `/llama install`."
        if r.get("runs"):
            line = f"llama-server installed at {r['binary']} — version: {r.get('version') or 'unknown'}"
        else:
            line = f"llama-server installed at {r['binary']} — DOES NOT RUN: {r.get('version') or 'unknown reason'}"
        if r.get("up_to_date") is True:
            if r.get("method") == "source":
                line += "; up to date (source build)"
            else:
                line += f"; up to date (release {r.get('latest_tag')})"
        elif r.get("up_to_date") is False:
            line += f"; update available ({r.get('tag')} → {r.get('latest_tag')})"
        elif r.get("latest_tag"):
            line += f"; latest release: {r.get('latest_tag')}"
        return line
    if cmd == "install":
        return _fmt_dict(install.install())
    if cmd == "upgrade":
        return _fmt_dict(install.upgrade())
    if cmd == "uninstall":
        return _fmt_dict(install.uninstall())
    if cmd == "status":
        return models.status()
    if cmd in ("models", "list"):
        return models.list_models()
    if cmd == "pull":
        if not argv:
            return "Usage: /llama pull <preset|Org/Repo> [alias]"
        return models.pull(argv[0], argv[1] if len(argv) > 1 else None)
    if cmd == "serve":
        if not argv:
            return "Usage: /llama serve <alias>"
        return models.serve(argv[0])
    if cmd == "stop":
        return models.stop()
    return f"Unknown subcommand '{cmd}'.\n\n{HELP}"


def _handle_slash(raw_args: str) -> str:
    parts = (raw_args or "").split()
    if not parts:
        return HELP
    return _dispatch(parts[0], parts[1:])


def _setup_cli(subparser: Any) -> None:
    subparser.add_argument("sub", nargs="?", default="help", help="check|install|uninstall|status|models|pull|serve|stop|help")
    subparser.add_argument("args", nargs="*", help="arguments for the subcommand")


def _handle_cli(args: Any) -> None:
    print(_dispatch(args.sub, list(args.args)))


# config_schema key -> LLAMA_CPP_* env var (the stdlib modules read env vars).
_CONFIG_ENV = {
    "base_url": "LLAMA_CPP_BASE_URL",
    "host": "LLAMA_CPP_HOST",
    "port": "LLAMA_CPP_PORT",
    "ctx_size": "LLAMA_CPP_CTX_SIZE",
    "n_gpu_layers": "LLAMA_CPP_N_GPU_LAYERS",
    "parallel": "LLAMA_CPP_PARALLEL",
    "backend": "LLAMA_CPP_BACKEND",
    "version": "LLAMA_CPP_VERSION",
    "install_dir": "LLAMA_CPP_INSTALL_DIR",
    "models_dir": "LLAMA_CPP_MODELS_DIR",
    "api_key": "LLAMA_CPP_API_KEY",
}


def _wire_config(ctx: Any) -> None:
    """Apply Hermes settings (config_schema) to the LLAMA_CPP_* env vars.

    Precedence: an explicitly-set env var wins; otherwise a value in Hermes
    settings (``plugins.entries.<id>.settings.<key>``) is applied. Schema
    defaults are never written into the environment, so ``backend=auto``
    detection and the modules' own defaults stay intact.

    When ``host`` or ``port`` are set and no explicit ``LLAMA_CPP_BASE_URL`` is
    configured, the provider base URL is derived as ``http://{host}:{port}/v1``
    so the server and provider agree on the endpoint.
    """
    if not hasattr(ctx, "get_config"):
        return
    for key, env_name in _CONFIG_ENV.items():
        if os.environ.get(env_name):
            continue
        try:
            val = ctx.get_config(key)
        except Exception:
            val = None
        if val is not None and str(val).strip() != "":
            os.environ[env_name] = str(val).strip()
    # Derive provider base URL from host+port when not explicitly set.
    if not os.environ.get("LLAMA_CPP_BASE_URL"):
        host = (os.environ.get("LLAMA_CPP_HOST") or "").strip()
        port = (os.environ.get("LLAMA_CPP_PORT") or "").strip()
        if host or port:
            base_host = host or "127.0.0.1"
            base_port = port or "8080"
            os.environ["LLAMA_CPP_BASE_URL"] = f"http://{base_host}:{base_port}/v1"


def register(ctx: Any) -> None:
    # 0. Wire Hermes settings (config_schema) into the LLAMA_CPP_* env vars the
    #    stdlib modules read. Env vars already set by the user win.
    _wire_config(ctx)

    # 1. Ensure the "Llama CPP" provider profile is registered (idempotent).
    provider.register()

    # 2. In-session slash command.
    ctx.register_command(
        "llama",
        handler=_handle_slash,
        description=(
            "Manage llama.cpp: check/install/uninstall, pull GGUF models "
            "(LiquidAI sample), and serve/stop a local 'Llama CPP' server."
        ),
        args_hint="<check|install|uninstall|status|models|pull|serve|stop|help>",
    )

    # 3. Terminal command `hermes llama ...`.
    try:
        ctx.register_cli_command(
            "llama",
            help="Manage llama.cpp and local GGUF models (provider 'Llama CPP')",
            description="Check/install/uninstall llama.cpp; pull and serve local GGUF models.",
            setup_fn=_setup_cli,
            handler_fn=_handle_cli,
        )
    except Exception:
        # CLI command registration is optional; slash command remains available.
        pass
