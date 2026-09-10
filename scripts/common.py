import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/experiment.json").read_text())


def relative_result_paths(value):
    """Remove the local checkout prefix from paths in a JSON report."""
    if isinstance(value, dict):
        return {key: relative_result_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relative_result_paths(item) for item in value]
    if isinstance(value, str):
        return value.replace(ROOT.as_posix() + "/", "")
    return value


def save_json(path, value):
    path = Path(path)
    # Result reports are portable; executable input manifests keep their paths.
    if path.resolve().is_relative_to(ROOT / "results"):
        value = relative_result_paths(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
