#!/usr/bin/env python3
"""Trusted-CPU independent SE-001Q reobservation helper.

This helper is trusted harness code. It treats the unmerged EV2.4 experiment and
classifier as data only, proves that their frozen gate semantics equal this
helper's hard-coded contract, captures every gate, and emits provider-specific
evidence. It never grants qualification or repair authority.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

SCHEMA = "symthaea.se001q.trusted-cpu-observation.v2"
OBS_SCHEMA = "symthaea.se001q.trusted-cpu-gate-observation.v1"
CLASS_SCHEMA = "symthaea.se001q.trusted-cpu-classification.v1"
MANIFEST_SCHEMA = "symthaea.se001q.trusted-cpu-manifest.v1"
PROVIDER = "trusted-cpu-v1"
TARGET_SHA = "47de7f2a306cffb66b5505786220590aa5f42e90"
TARGET_TREE = "7abdf2ede579b2b729b057ca64470d11eb551031"
TOOLCHAIN = "1.96.0"

INPUTS = [
    "Cargo.lock",
    "Cargo.toml",
    "crates/domains/symthaea-se-model/Cargo.toml",
    "crates/domains/symthaea-se-model/src/lib.rs",
]

GATES = [
    (
        "focused-format",
        [
            "rustfmt",
            "--edition",
            "2024",
            "--check",
            "crates/domains/symthaea-se-model/src/lib.rs",
        ],
    ),
    (
        "locked-metadata",
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
    ),
    (
        "focused-all-targets",
        ["cargo", "check", "-p", "symthaea-se-model", "--locked", "--all-targets"],
    ),
    (
        "focused-tests",
        ["cargo", "test", "-p", "symthaea-se-model", "--locked"],
    ),
    (
        "focused-clippy",
        [
            "cargo",
            "clippy",
            "-p",
            "symthaea-se-model",
            "--locked",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
    ),
]

EXPECTED_FORMAT_RULE = {
    "rule_id": "rustfmt-diff-marker-v1",
    "gate_id": "focused-format",
    "when": {"exit_code": "nonzero", "combined_output_contains": "Diff in "},
    "classification": {
        "state": "CLASSIFIED_FAIL",
        "failure_class": "FORMAT_DIVERGENCE",
    },
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> str:
    data = canonical_bytes(value) + b"\n"
    path.write_bytes(data)
    return sha256_bytes(data)


def run_checked(argv: list[str], cwd: Path) -> str:
    completed = subprocess.run(argv, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout.decode("utf-8", errors="replace").strip()


def git_identity(path: Path) -> tuple[str, str]:
    return (
        run_checked(["git", "rev-parse", "HEAD"], path),
        run_checked(["git", "rev-parse", "HEAD^{tree}"], path),
    )


def require_clean_git(path: Path) -> None:
    subprocess.run(["git", "diff", "--exit-code"], cwd=path, check=True)
    subprocess.run(["git", "diff", "--cached", "--exit-code"], cwd=path, check=True)
    status = run_checked(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching"],
        path,
    )
    if status:
        raise RuntimeError(f"git tree not pristine: {path}: {status}")


def input_digests(source: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for rel in INPUTS:
        path = source / rel
        if not path.is_file():
            raise RuntimeError(f"declared input missing: {rel}")
        result[rel] = sha256_file(path)
    return result


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_reference_contract(experiment_path: Path, classifier_path: Path) -> dict[str, str]:
    experiment = load_json(experiment_path)
    classifier = load_json(classifier_path)
    if not isinstance(experiment, dict) or not isinstance(classifier, dict):
        raise RuntimeError("reference contract roots must be objects")

    if experiment.get("schema") != "symthaea.qualification-experiment.v1":
        raise RuntimeError("unexpected experiment schema")
    if experiment.get("experiment_id") != "SE-001Q":
        raise RuntimeError("unexpected experiment id")
    if experiment.get("mode") != "evidence-replay":
        raise RuntimeError("unexpected experiment mode")
    if experiment.get("qualification_claim") != "NONE":
        raise RuntimeError("reference experiment grants qualification")
    if experiment.get("repair_authority_claim") != "NONE":
        raise RuntimeError("reference experiment grants repair authority")
    if experiment.get("subject_sha") != TARGET_SHA:
        raise RuntimeError("reference experiment subject mismatch")
    if experiment.get("toolchain") != TOOLCHAIN:
        raise RuntimeError("reference experiment toolchain mismatch")
    if experiment.get("working_directory") != ".":
        raise RuntimeError("reference experiment working directory changed")
    if experiment.get("inputs") != INPUTS:
        raise RuntimeError("reference experiment input set changed")

    observed_gates = []
    for gate in experiment.get("gates", []):
        if not isinstance(gate, dict):
            raise RuntimeError("malformed reference gate")
        observed_gates.append((gate.get("id"), gate.get("argv")))
    if observed_gates != GATES:
        raise RuntimeError("trusted recovery gate vector differs from authenticated EV2.4 experiment")

    controls = experiment.get("negative_controls")
    expected_control = [
        {
            "gate_id": "focused-format",
            "expected_state": "CLASSIFIED_FAIL",
            "expected_failure_class": "FORMAT_DIVERGENCE",
            "meaning": "The frozen SE-001 subject must continue to demonstrate the historical formatting defect under the pinned experiment.",
        }
    ]
    if controls != expected_control:
        raise RuntimeError("reference negative control changed")

    if classifier.get("schema") != "symthaea.qualification-classifier-contract.v1":
        raise RuntimeError("unexpected classifier schema")
    if classifier.get("classifier_id") != "SE-001Q-classifier-v1":
        raise RuntimeError("unexpected classifier id")
    if classifier.get("rules") != [EXPECTED_FORMAT_RULE]:
        raise RuntimeError("reference classifier rule set changed")
    if classifier.get("default_nonzero") != {"state": "FAIL_UNCLASSIFIED", "failure_class": None}:
        raise RuntimeError("reference classifier default changed")

    return {
        "experiment_sha256": sha256_file(experiment_path),
        "classifier_sha256": sha256_file(classifier_path),
    }


def tool_versions(source: Path) -> dict[str, str]:
    versions = {
        "rustc": run_checked(["rustc", "--version"], source),
        "cargo": run_checked(["cargo", "--version"], source),
        "rustfmt": run_checked(["rustfmt", "--version"], source),
        "clippy": run_checked(["cargo", "clippy", "--version"], source),
        "python": sys.version.splitlines()[0],
    }
    if f"rustc {TOOLCHAIN}" not in versions["rustc"]:
        raise RuntimeError(f"unexpected Rust compiler: {versions['rustc']}")
    return versions


def classify(gate_id: str, exit_code: int, stdout: bytes, stderr: bytes) -> tuple[str, str | None, str | None]:
    if exit_code == 0:
        return "PASS", None, None
    if gate_id == "focused-format" and b"Diff in " in stdout + stderr:
        return "CLASSIFIED_FAIL", "FORMAT_DIVERGENCE", "rustfmt-diff-marker-v1"
    return "FAIL_UNCLASSIFIED", None, None


def capture_gate(
    gate_id: str,
    argv: list[str],
    source: Path,
    evidence: Path,
    inputs_before: dict[str, str],
    versions: dict[str, str],
) -> dict[str, object]:
    gate_dir = evidence / gate_id
    gate_dir.mkdir(parents=True, exist_ok=False)

    completed = subprocess.run(argv, cwd=source, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = completed.stdout
    stderr = completed.stderr
    state, failure_class, rule_id = classify(gate_id, completed.returncode, stdout, stderr)

    command = {"argv": argv, "cwd": ".", "gate_id": gate_id}
    (gate_dir / "stdout.log").write_bytes(stdout)
    (gate_dir / "stderr.log").write_bytes(stderr)
    (gate_dir / "exit-code.txt").write_text(f"{completed.returncode}\n", encoding="utf-8")
    write_json(gate_dir / "command.json", command)

    observation_identity = {
        "schema": OBS_SCHEMA,
        "provider": PROVIDER,
        "subject_sha": TARGET_SHA,
        "subject_tree": TARGET_TREE,
        "gate_id": gate_id,
        "argv": argv,
        "cwd": ".",
        "exit_code": completed.returncode,
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "inputs_before": inputs_before,
        "tool_versions": versions,
    }
    observation_id = f"sha256:{sha256_bytes(canonical_bytes(observation_identity))}"
    observation = dict(observation_identity)
    observation["observation_id"] = observation_id
    write_json(gate_dir / "observation.json", observation)

    classification_identity = {
        "schema": CLASS_SCHEMA,
        "observation_id": observation_id,
        "state": state,
        "failure_class": failure_class,
        "rule_id": rule_id,
        "repair_authority_claim": "NONE",
    }
    classification_id = f"sha256:{sha256_bytes(canonical_bytes(classification_identity))}"
    classification = dict(classification_identity)
    classification["classification_id"] = classification_id
    write_json(gate_dir / "classification.json", classification)

    return {
        "gate_id": gate_id,
        "observation_id": observation_id,
        "classification_id": classification_id,
        "exit_code": completed.returncode,
        "state": state,
        "failure_class": failure_class,
    }


def make_manifest(evidence: Path) -> dict[str, object]:
    entries = []
    for path in sorted(p for p in evidence.rglob("*") if p.is_file() and p.name != "manifest.json"):
        rel = path.relative_to(evidence).as_posix()
        entries.append({"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    identity = {"schema": MANIFEST_SCHEMA, "entries": entries}
    manifest_id = f"sha256:{sha256_bytes(canonical_bytes(identity))}"
    result = dict(identity)
    result["manifest_id"] = manifest_id
    write_json(evidence / "manifest.json", result)
    return result


def deterministic_zip(evidence: Path, bundle: Path) -> None:
    files = sorted(p for p in evidence.rglob("*") if p.is_file())
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            rel = path.relative_to(evidence).as_posix()
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def emit_bundle(bundle: Path) -> None:
    data = bundle.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    chunk_size = 4096
    chunks = [encoded[i : i + chunk_size] for i in range(0, len(encoded), chunk_size)]
    print(f"evidence_bundle_sha256={sha256_bytes(data)}")
    print(f"evidence_bundle_bytes={len(data)}")
    print(f"evidence_bundle_base64_chunks={len(chunks)}")
    for index, chunk in enumerate(chunks):
        print(f"evidence_bundle_base64_{index:06d}={chunk}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    harness = args.harness.resolve()
    recipe = args.recipe.resolve()
    evidence = args.evidence.resolve()
    bundle = args.bundle.resolve()

    if evidence.exists() and any(evidence.iterdir()):
        raise RuntimeError("evidence directory must be empty")
    evidence.mkdir(parents=True, exist_ok=True)
    bundle.parent.mkdir(parents=True, exist_ok=True)

    source_sha, source_tree = git_identity(source)
    harness_sha, harness_tree = git_identity(harness)
    recipe_sha, recipe_tree = git_identity(recipe)
    if source_sha != TARGET_SHA or source_tree != TARGET_TREE:
        raise RuntimeError("frozen source identity mismatch")
    if harness_sha != os.environ.get("GITHUB_SHA"):
        raise RuntimeError("trusted harness does not equal GITHUB_SHA")
    if recipe_sha != os.environ.get("HOSTED_VERIFIER_COMMIT"):
        raise RuntimeError("reference recipe commit mismatch")
    if recipe_tree != os.environ.get("HOSTED_VERIFIER_TREE"):
        raise RuntimeError("reference recipe tree mismatch")

    reference = verify_reference_contract(
        recipe / ".github/qualification/se001q-experiment-v1.json",
        recipe / ".github/qualification/se001q-classifier-v1.json",
    )
    versions = tool_versions(source)
    inputs_before = input_digests(source)

    write_json(evidence / "tool-versions.json", versions)
    write_json(evidence / "inputs-before.json", inputs_before)
    write_json(evidence / "reference-contract.json", reference)

    outcomes = []
    for gate_id, argv in GATES:
        outcomes.append(capture_gate(gate_id, argv, source, evidence, inputs_before, versions))

    inputs_after = input_digests(source)
    write_json(evidence / "inputs-after.json", inputs_after)
    if inputs_after != inputs_before:
        raise RuntimeError("declared frozen inputs changed during replay")

    require_clean_git(source)
    require_clean_git(harness)
    require_clean_git(recipe)

    format_outcome = next(item for item in outcomes if item["gate_id"] == "focused-format")
    if format_outcome["state"] != "CLASSIFIED_FAIL" or format_outcome["failure_class"] != "FORMAT_DIVERGENCE":
        raise RuntimeError("frozen formatting negative control did not reproduce")

    workflow_blob = run_checked(
        ["git", "rev-parse", f"{harness_sha}:.github/workflows/self-hosted-se001q-evidence-recovery.yml"],
        harness,
    )
    helper_blob = run_checked(
        ["git", "rev-parse", f"{harness_sha}:nix/ci/se001q-trusted-replay.py"],
        harness,
    )

    summary = {
        "schema": SCHEMA,
        "result": "EVIDENCE_CAPTURE_COMPLETE",
        "qualification_claim": "NONE",
        "repair_authority_claim": "NONE",
        "provider": PROVIDER,
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "runner_name": os.environ.get("RUNNER_NAME", "unknown"),
        "runner_os": os.environ.get("RUNNER_OS", "unknown"),
        "runner_arch": os.environ.get("RUNNER_ARCH", "unknown"),
        "harness_commit": harness_sha,
        "harness_tree": harness_tree,
        "recovery_workflow_blob": workflow_blob,
        "trusted_replay_helper_blob": helper_blob,
        "target_commit": source_sha,
        "target_tree": source_tree,
        "target_base": os.environ.get("TARGET_BASE", ""),
        "hosted_verifier_commit": recipe_sha,
        "hosted_verifier_tree": recipe_tree,
        "hosted_workflow_blob": os.environ.get("HOSTED_WORKFLOW_BLOB", ""),
        "hosted_experiment_blob": os.environ.get("HOSTED_EXPERIMENT_BLOB", ""),
        "hosted_classifier_blob": os.environ.get("HOSTED_CLASSIFIER_BLOB", ""),
        "hosted_capture_blob": os.environ.get("HOSTED_CAPTURE_BLOB", ""),
        "hosted_verify_blob": os.environ.get("HOSTED_VERIFY_BLOB", ""),
        "hosted_attest_blob": os.environ.get("HOSTED_ATTEST_BLOB", ""),
        "reference_recipe_semantics_match": "PASS",
        "source_immutability_checked": "PASS",
        "harness_immutability_checked": "PASS",
        "reference_recipe_immutability_checked": "PASS",
        "hosted_recipe_executed": False,
        "outcomes": outcomes,
        "evidence_scope": "independent-provider-observation-only",
    }
    write_json(evidence / "summary.json", summary)
    manifest = make_manifest(evidence)
    deterministic_zip(evidence, bundle)

    print(canonical_bytes(summary).decode())
    print(f"evidence_manifest_id={manifest['manifest_id']}")
    emit_bundle(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
