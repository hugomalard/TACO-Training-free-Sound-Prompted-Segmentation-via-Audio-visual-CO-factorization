"""Resolve repository and dataset paths."""

from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_PATHS = PACKAGE_ROOT / "configs" / "paths.yaml"
EXPECTED_RESULTS = Path(__file__).resolve().parent / "expected_results.yaml"


def load_paths(path=None):
    path = Path(path) if path is not None else DEFAULT_PATHS
    with open(path, "r") as handle:
        raw = yaml.safe_load(handle)
    resolved = {}
    for key, value in raw.items():
        if key == "openclip" or not isinstance(value, str):
            resolved[key] = value
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        resolved[key] = str(candidate)
    return resolved


def ensure_repo_on_path():
    import sys

    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
