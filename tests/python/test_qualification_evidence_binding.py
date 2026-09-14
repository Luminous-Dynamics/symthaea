import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


train = load("integration_train_manifest")
load("qualification_evidence_id")
binding = load("qualification_evidence_binding")

HELLO = b"hello\n"
HELLO_SHA256 = "sha256:5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
HELLO_GIT_BLOB = "git-blob-sha1:ce013625030ba8dba906f756967f9e9ca394464a"


def test_raw_sha256_binding_uses_exact_bytes():
    result = binding.verify_bytes(HELLO_SHA256, HELLO)
    assert result["content_id"] == HELLO_SHA256
    assert result["byte_length"] == len(HELLO)
    assert result["verification_method"] == "RawSha256V1"


def test_git_blob_binding_uses_git_blob_framing_not_raw_sha1():
    result = binding.verify_bytes(HELLO_GIT_BLOB, HELLO)
    assert result["content_id"] == HELLO_GIT_BLOB
    assert result["verification_method"] == "GitBlobSha1V1"


def test_digest_mismatch_fails_closed():
    wrong = "sha256:" + "0" * 64
    with pytest.raises(train.TrainManifestError, match="digest mismatch"):
        binding.verify_bytes(wrong, HELLO)


def test_label_like_reference_is_not_a_content_descriptor():
    with pytest.raises(train.TrainManifestError, match="expected sha256"):
        binding.verify_bytes("artifact:receipt", HELLO)


def test_binding_explicitly_does_not_establish_acquisition_or_provenance():
    result = binding.verify_bytes(HELLO_SHA256, HELLO)
    assert "does not establish trustworthy acquisition of the verified bytes" in result["non_claims"]
    assert "does not establish provider/run provenance or producer authenticity" in result["non_claims"]


def test_same_verified_content_has_same_binding_without_file_locator():
    first = binding.verify_bytes(HELLO_SHA256, HELLO)
    second = binding.verify_bytes(HELLO_SHA256, bytes(HELLO))
    assert first == second
    assert "path" not in first


def test_file_verification_matches_byte_verification(tmp_path: Path):
    path = tmp_path / "receipt.json"
    path.write_bytes(HELLO)
    assert binding.verify_file(HELLO_SHA256, path) == binding.verify_bytes(HELLO_SHA256, HELLO)


def test_oversize_bytes_fail_before_hashing():
    data = b"x" * (binding.MAX_EVIDENCE_BYTES + 1)
    with pytest.raises(train.TrainManifestError, match="exceeds"):
        binding.verify_bytes("sha256:" + "0" * 64, data)
