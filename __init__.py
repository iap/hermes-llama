"""hermes-llama plugin — check/install/uninstall llama.cpp, run local GGUF models.

Registers:
- Provider profile ``llama-cpp`` (display name "Llama CPP") so a local
  llama-server shows up as a first-class provider in ``hermes model`` / ``/model``.
- Slash command ``/llama`` and CLI command ``hermes llama`` for managing
  the binary and models.

Subcommands: check, install, uninstall, status, models, pull, serve, stop, help.
"""

from __future__ import annotations

from typing import Any, Callable

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
        if r["installed"]:
            return f"llama-server installed at {r['binary']} (version: {r.get('version', 'unknown')}, method: {r.get('method', 'unknown')})"
        return "llama-server is NOT installed. Run `/llama install`."
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


def register(ctx: Any) -> None:
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
