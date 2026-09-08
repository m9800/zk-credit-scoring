"""Integration checks against actual Halo2/KZG proofs, including standalone verification."""
import os
import shutil
import subprocess
import pytest
from scripts.common import ROOT
from scripts.zk import BINARY


@pytest.fixture(scope="module")
def smoke_proof():
    from scripts.smoke_model import main as make_model
    from scripts.zk import smoke
    make_model()
    smoke()
    return ROOT / "artifacts/smoke"


def verify(smoke, proof_dir):
    return subprocess.run([str(BINARY), "verify", str(smoke / "public_model.msgpack"), str(proof_dir), "0"],
                          env={**os.environ, "RAYON_NUM_THREADS": "4"}, capture_output=True, text=True)


def test_standalone_verifier_accepts_without_private_input(smoke_proof):
    result = verify(smoke_proof, smoke_proof / "proof")
    assert result.returncode == 0, result.stderr
    assert '"verified":true' in result.stdout


@pytest.mark.parametrize("mutation", ["score", "weight", "proof"])
def test_standalone_verifier_rejects_tampering(smoke_proof, tmp_path, mutation):
    for filename in ("params.bin", "vk.bin", "proof_0.bin", "public_0.bin"):
        shutil.copyfile(smoke_proof / "proof" / filename, tmp_path / filename)
    path = tmp_path / ("proof_0.bin" if mutation == "proof" else "public_0.bin")
    content = bytearray(path.read_bytes())
    index = len(content) - 32 if mutation == "score" else 0
    content[index] ^= 1
    path.write_bytes(content)
    assert verify(smoke_proof, tmp_path).returncode != 0
