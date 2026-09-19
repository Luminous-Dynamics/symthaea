#!/usr/bin/env python3
"""Independently verify SE-001Q replay evidence and fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

OBSERVATION_SCHEMA = "symthaea.qualification-observation.v2"
CLASSIFICATION_SCHEMA = "symthaea.qualification-classification.v1"
SUMMARY_SCHEMA = "symthaea.qualification-replay-summary.v2"
MANIFEST_SCHEMA = "symthaea.qualification-evidence-manifest.v1"
CAPTURE_CONTRACT = "symthaea.se001q.capture.v2.1"
COMBINED_OUTPUT_DOMAIN = b"symthaea.qualification.combined-output.v1"
SUPPORTED_PREDICATES = {"exit_code", "combined_output_contains"}
GATE_FILES = {
    "command.json",
    "stdout.log",
    "stderr.log",
    "exit-code.txt",
    "observation.json",
    "classification.json",
}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def framed_digest(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return "sha256:" + h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_text(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=path, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def reject_authority(value: Any, where: str) -> None:
    if isinstance(value, dict):
        need("repair_authority" not in value, f"{where}: repair_authority is forbidden")
        if "repair_authority_claim" in value:
            need(value["repair_authority_claim"] == "NONE", f"{where}: non-NONE repair authority claim")
        for key, child in value.items():
            reject_authority(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_authority(child, f"{where}[{index}]")


def input_digests(subject: Path, paths: list[str]) -> dict[str, str]:
    out = {}
    for relative in paths:
        path = subject / relative
        need(path.is_file(), f"missing subject input: {relative}")
        out[relative] = sha256_file(path)
    return out


def match_rule(rule: dict[str, Any], exit_code: int, text: str) -> tuple[bool, list[dict[str, Any]]]:
    when = rule.get("when", {})
    unknown = set(when) - SUPPORTED_PREDICATES
    need(not unknown, f"{rule.get('rule_id')}: unsupported predicates {sorted(unknown)}")
    need(bool(when), f"{rule.get('rule_id')}: empty classifier predicate")
    predicates = []
    if "exit_code" in when:
        expected = when["exit_code"]
        satisfied = exit_code != 0 if expected == "nonzero" else exit_code == 0 if expected == "zero" else exit_code == int(expected)
        predicates.append({"kind": "exit_code", "expected": expected, "actual": exit_code, "satisfied": satisfied})
    if "combined_output_contains" in when:
        needle = str(when["combined_output_contains"])
        need(bool(needle), f"{rule.get('rule_id')}: empty output predicate")
        predicates.append({"kind": "combined_output_contains", "needle": needle, "satisfied": needle in text})
    return all(item["satisfied"] for item in predicates), predicates


def expected_classification(gate_id: str, exit_code: int, stdout: bytes, stderr: bytes, classifier: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    combined = stdout.decode(errors="replace") + "\n" + stderr.decode(errors="replace")
    combined_sha = framed_digest(COMBINED_OUTPUT_DOMAIN, stdout, stderr)
    if exit_code == 0:
        result = {"state": "PASS", "failure_class": None}
        predicates = [{"kind": "exit_code", "expected": "zero", "actual": 0, "satisfied": True}]
        return result, {"exit_code": 0, "combined_output_sha256": combined_sha, "matched_rule_id": None, "predicates": predicates}
    result = dict(classifier["default_nonzero"])
    matched_rule = None
    predicates: list[dict[str, Any]] = []
    for rule in classifier.get("rules", []):
        if rule["gate_id"] != gate_id:
            continue
        matched, tested = match_rule(rule, exit_code, combined)
        if matched:
            matched_rule = rule["rule_id"]
            predicates = tested
            result = dict(rule["classification"])
            break
    if matched_rule is None:
        predicates = [{"kind": "no_classifier_rule_matched", "satisfied": True}]
    return result, {"exit_code": exit_code, "combined_output_sha256": combined_sha, "matched_rule_id": matched_rule, "predicates": predicates}


def main() -> int:
    p = argparse.ArgumentParser()
    for name in ("evidence", "subject", "verifier", "experiment", "classifier", "workflow", "capture-runner"):
        p.add_argument(f"--{name}", required=True, type=Path)
    a = p.parse_args()
    evidence, subject, verifier = a.evidence.resolve(), a.subject.resolve(), a.verifier.resolve()
    experiment_path, classifier_path = a.experiment.resolve(), a.classifier.resolve()
    workflow_path, runner_path = a.workflow.resolve(), getattr(a, "capture_runner").resolve()
    need(evidence.is_dir(), "evidence directory missing")

    experiment, classifier = load(experiment_path), load(classifier_path)
    reject_authority(experiment, "$.experiment")
    reject_authority(classifier, "$.classifier")
    need(experiment.get("qualification_claim") == "NONE", "experiment qualification claim must be NONE")
    need(experiment.get("repair_authority_claim") == "NONE", "experiment repair authority claim must be NONE")
    need(classifier.get("schema") == "symthaea.qualification-classifier-contract.v1", "unexpected classifier schema")

    gates = experiment.get("gates", [])
    need(bool(gates), "experiment has no gates")
    gate_ids = [gate["id"] for gate in gates]
    need(len(gate_ids) == len(set(gate_ids)), "duplicate gate id")
    rule_ids = [rule["rule_id"] for rule in classifier.get("rules", [])]
    need(len(rule_ids) == len(set(rule_ids)), "duplicate classifier rule id")
    for rule in classifier.get("rules", []):
        need(rule["gate_id"] in gate_ids, f"{rule['rule_id']}: undeclared gate")
        unknown = set(rule.get("when", {})) - SUPPORTED_PREDICATES
        need(not unknown, f"{rule['rule_id']}: unsupported predicates {sorted(unknown)}")
        need("repair_authority" not in rule.get("classification", {}), f"{rule['rule_id']}: classifier grants authority")

    subject_sha, verifier_sha = git_text(subject, "rev-parse", "HEAD"), git_text(verifier, "rev-parse", "HEAD")
    need(subject_sha == experiment["subject_sha"], "frozen subject SHA mismatch")
    need(not git_text(subject, "status", "--porcelain"), "subject is dirty")
    exp_sha, cls_sha = sha256_file(experiment_path), sha256_file(classifier_path)
    workflow_sha, runner_sha = sha256_file(workflow_path), sha256_file(runner_path)
    common_inputs = input_digests(subject, list(experiment["inputs"]))

    actual_dirs = {path.name for path in evidence.iterdir() if path.is_dir()}
    need(actual_dirs == set(gate_ids), f"evidence gate directories mismatch: {sorted(actual_dirs ^ set(gate_ids))}")
    summary_items, by_gate = [], {}

    for gate in gates:
        gate_id, gate_dir = gate["id"], evidence / gate["id"]
        actual_files = {path.name for path in gate_dir.iterdir() if path.is_file()}
        need(actual_files == GATE_FILES, f"{gate_id}: gate file set mismatch")
        need(load(gate_dir / "command.json") == {"argv": list(gate["argv"]), "cwd": "."}, f"{gate_id}: command binding mismatch")
        try:
            exit_code = int((gate_dir / "exit-code.txt").read_text().strip())
        except ValueError as exc:
            raise RuntimeError(f"{gate_id}: invalid exit code") from exc
        stdout, stderr = (gate_dir / "stdout.log").read_bytes(), (gate_dir / "stderr.log").read_bytes()
        observation, classification = load(gate_dir / "observation.json"), load(gate_dir / "classification.json")
        reject_authority(observation, f"$.{gate_id}.observation")
        reject_authority(classification, f"$.{gate_id}.classification")
        need(observation.get("schema") == OBSERVATION_SCHEMA, f"{gate_id}: observation schema")
        identity = observation["identity"]
        need(identity.get("domain") == OBSERVATION_SCHEMA and identity.get("capture_contract") == CAPTURE_CONTRACT, f"{gate_id}: observation contract")
        need(identity.get("subject") == {"sha": subject_sha}, f"{gate_id}: subject binding")
        exp = identity["experiment"]
        need(exp.get("qualification_claim") == "NONE" and exp.get("mode") == "evidence-replay", f"{gate_id}: experiment claim")
        need(exp.get("verifier_sha") == verifier_sha, f"{gate_id}: verifier binding")
        need(exp.get("workflow_sha256") == workflow_sha and exp.get("experiment_contract_sha256") == exp_sha and exp.get("capture_runner_sha256") == runner_sha, f"{gate_id}: verifier artifact digest binding")
        execution, result = identity["execution"], identity["result"]
        need(execution.get("gate_id") == gate_id and execution.get("argv") == list(gate["argv"]) and execution.get("cwd") == ".", f"{gate_id}: execution binding")
        need(isinstance(execution.get("toolchain"), dict) and isinstance(execution.get("platform"), dict), f"{gate_id}: execution identity incomplete")
        need(identity.get("inputs") == common_inputs, f"{gate_id}: subject input digest mismatch")
        need(result.get("exit_code") == exit_code and result.get("stdout_sha256") == sha256_bytes(stdout) and result.get("stderr_sha256") == sha256_bytes(stderr), f"{gate_id}: result binding mismatch")
        obs_id = sha256_bytes(canonical(identity))
        need(observation.get("observation_id") == obs_id, f"{gate_id}: observation id mismatch")

        expected_result, expected_basis = expected_classification(gate_id, exit_code, stdout, stderr, classifier)
        need(classification.get("schema") == CLASSIFICATION_SCHEMA, f"{gate_id}: classification schema")
        need(classification.get("observation_id") == obs_id and classification.get("classifier_contract_sha256") == cls_sha and classification.get("gate_id") == gate_id, f"{gate_id}: classification binding")
        need(classification.get("basis") == expected_basis and classification.get("result") == expected_result, f"{gate_id}: classifier replay mismatch")
        cls_body = {key: value for key, value in classification.items() if key != "classification_id"}
        cls_id = sha256_bytes(canonical(cls_body))
        need(classification.get("classification_id") == cls_id, f"{gate_id}: classification id mismatch")
        by_gate[gate_id] = (observation, classification)
        summary_items.append({"gate_id": gate_id, "observation_id": obs_id, "classification_id": cls_id, "exit_code": exit_code, "state": expected_result["state"], "failure_class": expected_result.get("failure_class")})

    summary = load(evidence / "summary.json")
    reject_authority(summary, "$.summary")
    need(summary.get("schema") == SUMMARY_SCHEMA and summary.get("capture_contract") == CAPTURE_CONTRACT, "summary contract")
    need(summary.get("subject_sha") == subject_sha and summary.get("verifier_sha") == verifier_sha, "summary identity")
    need(summary.get("experiment_contract_sha256") == exp_sha and summary.get("classifier_contract_sha256") == cls_sha and summary.get("workflow_sha256") == workflow_sha and summary.get("capture_runner_sha256") == runner_sha, "summary digest binding")
    need(summary.get("observations") == summary_items, "summary observation projection mismatch")
    need(summary.get("qualification_claim") == "NONE" and summary.get("repair_authority_claim") == "NONE", "summary authority claim")
    need(summary.get("evidence_complete") is True, "summary does not claim complete evidence")
    need(summary.get("immutability") == {"head_unchanged": True, "working_tree_clean": True, "input_identity_preserved": True}, "immutability proof failed")
    expected_fc = all(item["exit_code"] == 0 or item["state"] == "CLASSIFIED_FAIL" for item in summary_items)
    need(summary.get("failure_classification_complete") == expected_fc, "classification-completeness mismatch")

    expected_controls = []
    for control in experiment.get("negative_controls", []):
        observation, classification = by_gate[control["gate_id"]]
        actual = classification["result"]
        expected_controls.append({**control, "observation_id": observation["observation_id"], "classification_id": classification["classification_id"], "actual_state": actual["state"], "actual_failure_class": actual.get("failure_class"), "satisfied": actual["state"] == control["expected_state"] and actual.get("failure_class") == control.get("expected_failure_class")})
    need(summary.get("negative_controls") == expected_controls, "negative-control projection mismatch")
    need(all(item["satisfied"] for item in expected_controls), "verifier-health negative control failed")
    summary_body = {key: value for key, value in summary.items() if key != "summary_id"}
    need(summary.get("summary_id") == sha256_bytes(canonical(summary_body)), "summary id mismatch")

    manifest = load(evidence / "manifest.json")
    reject_authority(manifest, "$.manifest")
    need(manifest.get("schema") == MANIFEST_SCHEMA and manifest.get("capture_contract") == CAPTURE_CONTRACT, "manifest contract")
    need(manifest.get("subject_sha") == subject_sha and manifest.get("verifier_sha") == verifier_sha, "manifest identity")
    need(manifest.get("workflow_sha256") == workflow_sha and manifest.get("experiment_contract_sha256") == exp_sha and manifest.get("classifier_contract_sha256") == cls_sha and manifest.get("capture_runner_sha256") == runner_sha, "manifest digest binding")
    files = sorted(path.relative_to(evidence).as_posix() for path in evidence.rglob("*") if path.is_file() and path.name != "manifest.json")
    need(manifest.get("files") == [{"path": path, "sha256": sha256_file(evidence / path)} for path in files], "manifest file set or digest mismatch")
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_id"}
    need(manifest.get("manifest_id") == sha256_bytes(canonical(manifest_body)), "manifest id mismatch")

    print(f"SE-001Q evidence independently verified: subject={subject_sha} verifier={verifier_sha} gates={len(gates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
