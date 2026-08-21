"""Validate that every plugin manifest parses as YAML.

Checks the root plugin.yaml and every model-provider/*/plugin.yaml. Uses PyYAML
(the only non-stdlib dependency, installed in CI only).
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def manifests() -> list[Path]:
    root = REPO_ROOT / "plugin.yaml"
    providers = sorted((REPO_ROOT / "model-provider").glob("*/plugin.yaml"))
    return [root, *providers]


def main() -> int:
    files = manifests()
    if not files:
        print("ERROR: no plugin.yaml manifests found")
        return 1
    for path in files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            print(f"FAIL {path}: expected a mapping, got {type(data).__name__}")
            return 1
        print(f"OK {path}  (name={data.get('name')!r})")
    print(f"\nValidated {len(files)} manifest(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
