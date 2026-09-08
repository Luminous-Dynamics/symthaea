import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "integration_train_manifest.py"
SPEC = importlib.util.spec_from_file_location("integration_train_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
integration_train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(integration_train)


def sha(char):
    return char * 40


def valid_manifest():
    return {
        "schema": integration_train.SCHEMA,
        "base_subject": sha("a"),
        "ordered_tranches": [
            {
                "commit_sha": sha("b"),
                "predecessor_sha": sha("a"),
                "theorem_id": "RES-1",
                "claim": "separate topology from committed flow",
                "changed_files": [
                    "crates/domains/a/Cargo.toml",
                    "crates/domains/a/src/lib.rs",
                ],
                "evidence_refs": ["pr:695"],
                "non_claims": ["not qualified"],
                "origin_pr": 695,
            },
            {
                "commit_sha": sha("c"),
                "predecessor_sha": sha("b"),
                "theorem_id": "RES-2",
                "claim": "make temporal capacity semantics explicit",
                "changed_files": [
                    "crates/domains/b/Cargo.toml",
                    "crates/domains/b/src/lib.rs",
                ],
                "evidence_refs": ["pr:699"],
                "non_claims": ["not qualified"],
                "origin_pr": 699,
            },
        ],
        "cumulative_tip_sha": sha("c"),
    }


def test_train_identity_is_deterministic_and_semantic():
    manifest = valid_manifest()
    first = integration_train.compute_train_id(manifest)
    reordered_keys = json.loads(json.dumps(manifest, sort_keys=True))
    assert integration_train.compute_train_id(reordered_keys) == first

    changed = valid_manifest()
    changed["ordered_tranches"][1]["claim"] = "different claim"
    assert integration_train.compute_train_id(changed) != first


def test_declared_identity_must_match():
    manifest = valid_manifest()
    manifest["train_id"] = "sha256:" + "0" * 64
    with pytest.raises(integration_train.TrainManifestError, match="train_id"):
        integration_train.normalize_manifest(manifest)


def test_chain_must_be_linear_and_unique():
    manifest = valid_manifest()
    manifest["ordered_tranches"][1]["predecessor_sha"] = sha("a")
    with pytest.raises(integration_train.TrainManifestError, match="predecessor_sha"):
        integration_train.normalize_manifest(manifest)

    manifest = valid_manifest()
    manifest["ordered_tranches"][1]["theorem_id"] = "RES-1"
    with pytest.raises(integration_train.TrainManifestError, match="duplicate theorem"):
        integration_train.normalize_manifest(manifest)


def test_set_like_fields_are_canonical_and_paths_are_safe():
    manifest = valid_manifest()
    manifest["ordered_tranches"][0]["evidence_refs"] = ["z", "a"]
    with pytest.raises(
        integration_train.TrainManifestError, match="lexicographically sorted"
    ):
        integration_train.normalize_manifest(manifest)

    manifest = valid_manifest()
    manifest["ordered_tranches"][0]["changed_files"] = ["../outside"]
    with pytest.raises(
        integration_train.TrainManifestError, match="non-canonical repository path"
    ):
        integration_train.normalize_manifest(manifest)


def run(repo, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_optional_git_verification_checks_parent_and_changed_files(tmp_path):
    run(tmp_path, "init")
    run(tmp_path, "config", "user.email", "test@example.com")
    run(tmp_path, "config", "user.name", "Test")

    base_file = tmp_path / "base.txt"
    base_file.write_text("base\n")
    run(tmp_path, "add", "base.txt")
    run(tmp_path, "commit", "-m", "base")
    base = run(tmp_path, "rev-parse", "HEAD")

    first_file = tmp_path / "first.txt"
    first_file.write_text("first\n")
    run(tmp_path, "add", "first.txt")
    run(tmp_path, "commit", "-m", "first")
    first = run(tmp_path, "rev-parse", "HEAD")

    second_file = tmp_path / "second.txt"
    second_file.write_text("second\n")
    run(tmp_path, "add", "second.txt")
    run(tmp_path, "commit", "-m", "second")
    second = run(tmp_path, "rev-parse", "HEAD")

    manifest = {
        "schema": integration_train.SCHEMA,
        "base_subject": base,
        "ordered_tranches": [
            {
                "commit_sha": first,
                "predecessor_sha": base,
                "theorem_id": "T1",
                "claim": "first",
                "changed_files": ["first.txt"],
                "evidence_refs": [],
                "non_claims": [],
            },
            {
                "commit_sha": second,
                "predecessor_sha": first,
                "theorem_id": "T2",
                "claim": "second",
                "changed_files": ["second.txt"],
                "evidence_refs": [],
                "non_claims": [],
            },
        ],
        "cumulative_tip_sha": second,
    }

    integration_train.validate_git_chain(manifest, tmp_path)

    manifest["ordered_tranches"][1]["changed_files"] = ["wrong.txt"]
    with pytest.raises(integration_train.TrainManifestError, match="changed_files mismatch"):
        integration_train.validate_git_chain(manifest, tmp_path)


def test_golden_identity_and_unicode_canonicalization():
    manifest = valid_manifest()
    assert integration_train.compute_train_id(manifest) == (
        "sha256:65617f0ccdb60817f69d6bfb4212435e83f5b94f82cac2b291979f9757891979"
    )

    manifest = valid_manifest()
    manifest["ordered_tranches"][0]["claim"] = "Cafe\u0301"
    with pytest.raises(integration_train.TrainManifestError, match="NFC"):
        integration_train.normalize_manifest(manifest)


def test_duplicate_json_keys_and_alternate_paths_fail_closed(tmp_path):
    manifest_path = tmp_path / "duplicate.json"
    manifest_path.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(
        integration_train.TrainManifestError, match="duplicate JSON object key"
    ):
        integration_train.load_manifest(manifest_path)

    manifest = valid_manifest()
    manifest["ordered_tranches"][0]["changed_files"] = ["a//b"]
    with pytest.raises(
        integration_train.TrainManifestError, match="non-canonical repository path"
    ):
        integration_train.normalize_manifest(manifest)
