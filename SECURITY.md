# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| 0.2.x (latest) | :white_check_mark: |

The plugin tracks a single latest release. Older versions receive no backports — upgrade to latest for security fixes.

## Supply-Chain Integrity

- GGUF downloads are verified against upstream HuggingFace `lfs.oid` digests before `.part` promotion. Fails open when no digest is available.
- Prebuilt binary archives are verified against their own sha256 at install time; a digest change on the same release tag triggers a **TAMPER WARNING**.
- CI workflow actions are pinned by full 40-char commit SHA (see CONTRIBUTING.md).

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly by
opening a [private issue](https://github.com/iap/.github/issues/new) or
contacting the maintainer at <6572003+iap@users.noreply.github.com>. We will respond as soon as possible and work with you to address it.

Please do not publicly disclose the vulnerability until it has been resolved.
