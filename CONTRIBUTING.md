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

## Running the tests

The tests are dependency-free standalone scripts — they are **not** run with
pytest (the plugin modules are loaded through custom import machinery, so
pytest collection fails). Run them exactly the way CI does:

```bash
python tests/test_install.py                 # unit tests
python tests/integration/test_lifecycle.py   # pull → serve → stop lifecycle (mocked)
python tests/validate_manifests.py           # plugin.yaml sanity
ruff check .                                 # lint (version pinned in CI)
```

Each script prints `PASS`/`FAIL` lines and exits non-zero on any failure.

## Pinning on git CI

GitHub Actions referenced in `.github/workflows/*.yml` MUST be pinned to prevent
supply-chain attacks where a compromised upstream action could execute arbitrary
code in CI. Pin by **immutable commit SHA** — never by mutable branch (`@main`)
or bare major tag (`@v3`).

**Convention:**

- Pin to a **full 40-char commit SHA** with a trailing comment naming the version:
  `uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1`
- For actions that publish proper semver tags (`@v3`, `@v7.0.1`), pinning to the
  tag is acceptable as a fallback — but SHA pinning is preferred.
- **Never** pin to `@main`, `@master`, or bare `@v3` without a SHA.

**Verify before submitting:**

```bash
cd .github/workflows && grep -rn "uses:" . | grep -vE "@[0-9a-f]{40}|# v[0-9]"
```

This flags any action referenced by mutable tag or branch. Fix by resolving the
tag to its commit SHA via the action's GitHub release page.

## Questions?

If you have any questions, feel free to open an issue and we'll be happy to help.
