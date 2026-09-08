"""Download a pinned public dataset; never overwrite different existing data."""
import urllib.request
from scripts.common import ROOT, config, save_json, sha256


def main():
    cfg = config()
    url = ("https://huggingface.co/datasets/spectrallabs/credit-scoring-training-dataset/resolve/"
           + cfg["dataset_revision"] + "/" + cfg["dataset_file"])
    destination = ROOT / "data/raw/spectral.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(".partial")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    manifest = {"url": url, "revision": cfg["dataset_revision"],
                "sha256": sha256(destination), "bytes": destination.stat().st_size}
    manifest_path = ROOT / "data/manifest.json"
    if manifest_path.exists():
        import json
        previous = json.loads(manifest_path.read_text())
        if previous["sha256"] != manifest["sha256"]:
            raise ValueError("Dataset checksum differs from the recorded manifest")
    save_json(manifest_path, manifest)
    print(manifest)


if __name__ == "__main__":
    main()
