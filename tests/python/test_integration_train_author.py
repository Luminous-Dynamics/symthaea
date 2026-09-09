import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
MANIFEST_SCRIPT = SCRIPTS / "integration_train_manifest.py"
AUTHOR_SCRIPT = SCRIPTS / "integration_train_author.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


integration_train = load_module("integration_train_manifest", MANIFEST_SCRIPT)

# The authoring module imports integration_train_manifest by module name.
import sys
sys.modules["integration_train_manifest"] = integration_train
integration_author = load_module("integration_train_author", AUTHOR_SCRIPT)


def run(repo, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_linear_repo(tmp_path):
    run(tmp_path, "init")
    run(tmp_path, "config", "user.email", "test@example.com")
    run(tmp_path, "config", "user.name", "Test")

    (tmp_path / "base.txt").write_text("base\n")
    run(tmp_path, "add", "base.txt")
    run(tmp_path, "commit", "-m", "base")
    base = run(tmp_path, "rev-parse", "HEAD")

    (tmp_path / "first.txt").write_text("first\n")
    run(tmp_path, "add", "first.txt")
    run(tmp_path, "commit", "-m", "first")
    first = run(tmp_path, "rev-parse", "HEAD")

    (tmp_path / "second.txt").write_text("second\n")
    run(tmp_path, "add", "second.txt")
    run(tmp_path, "commit", "-m", "second")
    second = run(tmp_path, "rev-parse", "HEAD")
    return base, first, second


def valid_author_spec(base, first, second):
    return {
        "schema": integration_author.AUTHOR_SCHEMA,
        "base_subject": base,
        "ordered_tranches": [
            {
                "commit_sha": first,
                "theorem_id": "T1",
                "claim": "first theorem",
                "evidence_refs": ["issue:987"],
                "non_claims": ["hosted qualification not established"],
                "origin_pr": 1,
            },
            {
                "commit_sha": second,
                "theorem_id": "T2",
                "claim": "second theorem",
                "evidence_refs": ["issue:987"],
                "non_claims": ["hosted qualification not established"],
                "origin_pr": 2,
            },
        ],
    }


def test_author_spec_forbids_git_owned_facts():
    spec = {
        "schema": integration_author.AUTHOR_SCHEMA,
        "base_subject": "a" * 40,
        "ordered_tranches": [
            {
                "commit_sha": "b" * 40,
                "theorem_id": "T1",
                "claim": "semantic claim",
                "evidence_refs": [],
                "non_claims": [],
                "predecessor_sha": "a" * 40,
            }
        ],
    }
    with pytest.raises(
        integration_train.TrainManifestError, match="unknown fields: predecessor_sha"
    ):
        integration_author.normalize_author_spec(spec)

    del spec["ordered_tranches"][0]["predecessor_sha"]
    spec["ordered_tranches"][0]["changed_files"] = ["invented.txt"]
    with pytest.raises(
        integration_train.TrainManifestError, match="unknown fields: changed_files"
    ):
        integration_author.normalize_author_spec(spec)


def test_derive_manifest_observes_parent_and_changed_files(tmp_path):
    base, first, second = make_linear_repo(tmp_path)
    finalized = integration_author.derive_manifest(
        valid_author_spec(base, first, second), tmp_path
    )

    assert finalized["schema"] == integration_train.SCHEMA
    assert finalized["base_subject"] == base
    assert finalized["cumulative_tip_sha"] == second
    assert finalized["ordered_tranches"][0]["predecessor_sha"] == base
    assert finalized["ordered_tranches"][0]["changed_files"] == ["first.txt"]
    assert finalized["ordered_tranches"][1]["predecessor_sha"] == first
    assert finalized["ordered_tranches"][1]["changed_files"] == ["second.txt"]
    assert finalized["train_id"] == integration_train.compute_train_id(finalized)


def test_derive_manifest_rejects_noncontiguous_order(tmp_path):
    base, first, second = make_linear_repo(tmp_path)
    spec = valid_author_spec(base, second, first)
    with pytest.raises(integration_train.TrainManifestError, match="expected parent"):
        integration_author.derive_manifest(spec, tmp_path)


def test_derive_manifest_rejects_base_that_is_not_parent(tmp_path):
    base, first, second = make_linear_repo(tmp_path)
    spec = valid_author_spec(first, first, second)
    with pytest.raises(integration_train.TrainManifestError, match="expected parent"):
        integration_author.derive_manifest(spec, tmp_path)


def test_derive_manifest_rejects_merge_commit(tmp_path):
    base, first, _second = make_linear_repo(tmp_path)
    main_branch = run(tmp_path, "branch", "--show-current")
    run(tmp_path, "checkout", "-b", "side", base)
    (tmp_path / "side.txt").write_text("side\n")
    run(tmp_path, "add", "side.txt")
    run(tmp_path, "commit", "-m", "side")
    run(tmp_path, "checkout", main_branch)
    run(tmp_path, "merge", "--no-ff", "side", "-m", "merge")
    merge = run(tmp_path, "rev-parse", "HEAD")

    spec = {
        "schema": integration_author.AUTHOR_SCHEMA,
        "base_subject": first,
        "ordered_tranches": [
            {
                "commit_sha": merge,
                "theorem_id": "MERGE",
                "claim": "merge theorem",
                "evidence_refs": [],
                "non_claims": [],
            }
        ],
    }
    with pytest.raises(
        integration_train.TrainManifestError, match="expected exactly one parent"
    ):
        integration_author.derive_manifest(spec, tmp_path)


def test_semantic_change_changes_final_train_identity(tmp_path):
    base, first, second = make_linear_repo(tmp_path)
    spec = valid_author_spec(base, first, second)
    first_id = integration_author.derive_manifest(spec, tmp_path)["train_id"]
    spec["ordered_tranches"][1]["claim"] = "different semantic theorem"
    second_id = integration_author.derive_manifest(spec, tmp_path)["train_id"]
    assert first_id != second_id


def test_author_cli_prints_finalized_manifest(tmp_path, capsys):
    base, first, second = make_linear_repo(tmp_path)
    author_path = tmp_path / "author.json"
    author_path.write_text(json.dumps(valid_author_spec(base, first, second)))

    assert integration_author.main(
        [
            str(author_path),
            "--repo",
            str(tmp_path),
            "--print-normalized",
        ]
    ) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["schema"] == integration_train.SCHEMA
    assert rendered["ordered_tranches"][0]["changed_files"] == ["first.txt"]
    assert rendered["ordered_tranches"][1]["predecessor_sha"] == first
    assert rendered["train_id"].startswith("sha256:")


def test_duplicate_author_json_keys_fail_closed(tmp_path):
    author_path = tmp_path / "duplicate.json"
    author_path.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(
        integration_train.TrainManifestError, match="duplicate JSON object key"
    ):
        integration_author.load_author_spec(author_path)
