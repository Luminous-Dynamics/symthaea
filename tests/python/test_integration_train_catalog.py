import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
MANIFEST_SCRIPT = SCRIPTS / "integration_train_manifest.py"
CATALOG_SCRIPT = SCRIPTS / "integration_train_catalog.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


integration_train = load_module("integration_train_manifest", MANIFEST_SCRIPT)
sys.modules["integration_train_manifest"] = integration_train
integration_catalog = load_module("integration_train_catalog", CATALOG_SCRIPT)


def sha(char):
    return char * 40


def ident(char):
    return "sha256:" + char * 64


def valid_catalog():
    return {
        "schema": integration_catalog.SCHEMA,
        "program_id": "resources.planning-research.v1",
        "trains": [
            {"name": "B1", "train_id": ident("1"), "manifest_branch": "evidence/b1", "manifest_path": "docs/b1.json", "base_subject": sha("b"), "cumulative_tip_sha": sha("d"), "role": "bundle", "status": "SourceManifestOnly"},
            {"name": "R1", "train_id": ident("2"), "manifest_branch": "evidence/r1", "manifest_path": "docs/r1.json", "base_subject": sha("a"), "cumulative_tip_sha": sha("b"), "role": "base", "status": "SourceManifestOnly"},
            {"name": "R2", "train_id": ident("3"), "manifest_branch": "evidence/r2", "manifest_path": "docs/r2.json", "base_subject": sha("b"), "cumulative_tip_sha": sha("c"), "role": "single", "status": "SourceManifestOnly"},
        ],
        "edges": [
            {"from": "R1", "to": "B1", "relation": "parallel_successor", "boundary_sha": sha("b")},
            {"from": "R1", "to": "R2", "relation": "linear_successor", "boundary_sha": sha("b")},
        ],
        "non_claims": ["does not establish hosted qualification", "does not merge pull requests"],
    }


def test_catalog_identity_is_deterministic_and_semantic():
    catalog = valid_catalog()
    first = integration_catalog.compute_catalog_id(catalog)
    reordered = json.loads(json.dumps(catalog, sort_keys=True))
    assert integration_catalog.compute_catalog_id(reordered) == first
    changed = valid_catalog()
    changed["trains"][2]["role"] = "different"
    assert integration_catalog.compute_catalog_id(changed) != first


def test_declared_catalog_identity_must_match():
    catalog = valid_catalog()
    catalog["catalog_id"] = ident("0")
    with pytest.raises(integration_train.TrainManifestError, match="catalog_id"):
        integration_catalog.normalize_catalog(catalog)


def test_boundary_and_unknown_train_fail_closed():
    catalog = valid_catalog()
    catalog["edges"][1]["boundary_sha"] = sha("f")
    with pytest.raises(integration_train.TrainManifestError, match="cumulative_tip_sha"):
        integration_catalog.normalize_catalog(catalog)
    catalog = valid_catalog()
    catalog["edges"][1]["to"] = "missing"
    with pytest.raises(integration_train.TrainManifestError, match="unknown train"):
        integration_catalog.normalize_catalog(catalog)


def test_duplicate_train_identity_and_location_fail_closed():
    catalog = valid_catalog()
    catalog["trains"][2]["train_id"] = catalog["trains"][1]["train_id"]
    with pytest.raises(integration_train.TrainManifestError, match="duplicate train identity"):
        integration_catalog.normalize_catalog(catalog)
    catalog = valid_catalog()
    catalog["trains"][2]["manifest_branch"] = catalog["trains"][1]["manifest_branch"]
    catalog["trains"][2]["manifest_path"] = catalog["trains"][1]["manifest_path"]
    with pytest.raises(integration_train.TrainManifestError, match="duplicate manifest"):
        integration_catalog.normalize_catalog(catalog)


def test_fork_siblings_cannot_be_directly_linearized_in_v1():
    catalog = valid_catalog()
    catalog["edges"].append({"from": "B1", "to": "R2", "relation": "linear_successor", "boundary_sha": sha("d")})
    catalog["trains"][2]["base_subject"] = sha("d")
    catalog["edges"] = sorted(catalog["edges"], key=lambda edge: (edge["from"], edge["to"], edge["relation"], edge["boundary_sha"]))
    with pytest.raises(integration_train.TrainManifestError, match="sibling successors"):
        integration_catalog.normalize_catalog(catalog)


def test_multiple_linear_successors_fail_closed():
    catalog = valid_catalog()
    catalog["trains"].append({"name": "R3", "train_id": ident("4"), "manifest_branch": "evidence/r3", "manifest_path": "docs/r3.json", "base_subject": sha("b"), "cumulative_tip_sha": sha("e"), "role": "other", "status": "SourceManifestOnly"})
    catalog["trains"] = sorted(catalog["trains"], key=lambda item: item["name"])
    catalog["edges"].append({"from": "R1", "to": "R3", "relation": "linear_successor", "boundary_sha": sha("b")})
    catalog["edges"] = sorted(catalog["edges"], key=lambda edge: (edge["from"], edge["to"], edge["relation"], edge["boundary_sha"]))
    with pytest.raises(integration_train.TrainManifestError, match="multiple linear successors"):
        integration_catalog.normalize_catalog(catalog)


def run(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def test_git_binding_resolves_manifest_branch_and_revalidates_chain(tmp_path):
    run(tmp_path, "init")
    run(tmp_path, "config", "user.email", "test@example.com")
    run(tmp_path, "config", "user.name", "Test")
    (tmp_path / "base.txt").write_text("base\n")
    run(tmp_path, "add", "base.txt")
    run(tmp_path, "commit", "-m", "base")
    base = run(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "work.txt").write_text("work\n")
    run(tmp_path, "add", "work.txt")
    run(tmp_path, "commit", "-m", "work")
    tip = run(tmp_path, "rev-parse", "HEAD")
    finalized = integration_train.normalize_manifest({"schema": integration_train.SCHEMA, "base_subject": base, "ordered_tranches": [{"commit_sha": tip, "predecessor_sha": base, "theorem_id": "T1", "claim": "work", "changed_files": ["work.txt"], "evidence_refs": [], "non_claims": ["not qualified"]}], "cumulative_tip_sha": tip}, verify_declared_id=False)
    run(tmp_path, "branch", "code-tip", tip)
    run(tmp_path, "checkout", "-b", "evidence/r1", base)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "r1.json").write_text(json.dumps(finalized))
    run(tmp_path, "add", "docs/r1.json")
    run(tmp_path, "commit", "-m", "manifest")
    catalog = {"schema": integration_catalog.SCHEMA, "program_id": "p", "trains": [{"name": "R1", "train_id": finalized["train_id"], "manifest_branch": "evidence/r1", "manifest_path": "docs/r1.json", "base_subject": base, "cumulative_tip_sha": tip, "role": "base", "status": "SourceManifestOnly"}], "edges": [], "non_claims": ["not qualified"]}
    integration_catalog.validate_git_bindings(catalog, tmp_path)
    catalog["trains"][0]["cumulative_tip_sha"] = base
    with pytest.raises(integration_train.TrainManifestError, match="cumulative_tip_sha"):
        integration_catalog.validate_git_bindings(catalog, tmp_path)


def test_duplicate_json_keys_and_noncanonical_locations_fail_closed(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(integration_train.TrainManifestError, match="duplicate JSON object key"):
        integration_catalog.load_catalog(path)
    catalog = valid_catalog()
    catalog["trains"][0]["manifest_path"] = "../bad.json"
    with pytest.raises(integration_train.TrainManifestError, match="non-canonical JSON"):
        integration_catalog.normalize_catalog(catalog)
