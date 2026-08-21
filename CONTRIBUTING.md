# Contributing

Thank you for your interest in contributing! Here are some guidelines to help you get started.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub.

## Branch Naming

Use lowercase prefixes with a short description separated by a slash:

| Prefix | Use for |
|---|---|
| `feat/` | New features or capabilities |
| `fix/` | Bug fixes |
| `docs/` | Documentation-only changes |
| `chore/` | Maintenance (deps, tooling, config) |
| `refactor/` | Code changes that neither fix a bug nor add a feature |
| `test/` | Adding or updating tests |

Example: `feat/user-auth`, `fix/login-crash`, `docs/api-endpoints`.

## Pull Requests

1. Fork the repository.
2. Create a branch using the naming convention above (`git checkout -b feat/my-feature`).
3. Make your changes and commit with a descriptive message.
4. Push to your branch (`git push origin feat/my-feature`).
5. Open a pull request.

### Open PRs Early

Prefer opening a pull request as soon as you have something reviewable, even if the work isn't finished. This gives maintainers visibility into what's happening and allows early feedback on direction. Keep PRs small and focused — a series of small, merged PRs is easier to review than one large one.

## Code Style

Please follow the existing code style in the project. If a linter or formatter is configured, make sure your code passes before submitting.

### Code conventions

- **Prefer no hardcoded paths.** Never hardcode absolute filesystem paths or
  machine-specific locations. Derive every local path from a configurable root
  (`HERMES_HOME`, `LLAMA_CPP_INSTALL_DIR`, `LLAMA_CPP_MODELS_DIR`), with a
  sensible default only as a fallback. Fixed subdirectory names (`bin/`,
  `models/`, `src/`) are internal layout, not absolute paths.
- **Prefer env/config over literals.** Runtime settings (host, port, backend,
  context size, base URL) come from `LLAMA_CPP_*` env vars or the `config_schema`
  in `plugin.yaml`, never hardcoded.
- **Keep endpoints as constants.** External service URLs (GitHub, Hugging Face)
  are named module-level constants (`REPO`, `GITHUB_API`, …) so they stay obvious
  and replaceable (e.g. for mirrors).

## Questions?

If you have any questions, feel free to open an issue and we'll be happy to help.
