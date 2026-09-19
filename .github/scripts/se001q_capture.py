#!/usr/bin/env python3
"""Durably capture and classify SE-001Q evidence without authorizing mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


OBSERVATION_SCHEMA = "symthaea.qualification-observation.v2"
CLASSIFICATION_SCHEMA = "symthaea.qualification-classification.v1"
SUMMARY_SCHEMA = "symthaea.qualification-replay-summary.v2"
MANIFEST_SCHEMA = "symthaea.qualification-evidence-manifest.v1"
CAPTURE_CONTRACT = "symthaea.se001q.capture.v2.1"
COMBINED_OUTPUT_DOMAIN = b"symthaea.qualification.combined-output.v1"


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def framed_digest(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return "sha256:" + hasher.hexdigest()


def command_output(argv: list[str], cwd: Path) -> bytes:
    result = subprocess.run(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout


def git_text(subject: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=subject,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def toolchain_identity(subject: Path) -> dict[str, str]:
    tools = {
        "rustc": ["rustc", "--version", "--verbose"],
        "cargo": ["cargo", "--version"],
        "rustfmt": ["rustfmt", "--version"],
        "clippy": ["cargo", "clippy", "--version"],
    }
    return {
        name: command_output(argv, subject).decode("utf-8", errors="replace").strip()
        for name, argv in tools.items()
    }


def input_digests(subject: Path, paths: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        path = subject / relative
        if not path.is_file():
            raise RuntimeError(f"required input does not exist: {relative}")
        result[relative] = sha256_file(path)
    return result


def capture_gate(
    *,
    gate: dict[str, Any],
    subject: Path,
    evidence_root: Path,
    context: dict[str, Any],
    common_inputs: dict[str, str],
) -> dict[str, Any]:
    gate_dir = evidence_root / gate["id"]
    gate_dir.mkdir(parents=True, exist_ok=False)

    argv = list(gate["argv"])
    write_json(gate_dir / "command.json", {"argv": argv, "cwd": "."})

    completed = subprocess.run(
        argv,
        cwd=subject,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout = completed.stdout
    stderr = completed.stderr
    (gate_dir / "stdout.log").write_bytes(stdout)
    (gate_dir / "stderr.log").write_bytes(stderr)
    (gate_dir / "exit-code.txt").write_text(
        f"{completed.returncode}\n", encoding="utf-8"
    )

    identity: dict[str, Any] = {
        "domain": OBSERVATION_SCHEMA,
        "capture_contract": CAPTURE_CONTRACT,
        "subject": {"sha": context["subject_sha"]},
        "experiment": {
            "qualification_claim": "NONE",
            "mode": "evidence-replay",
            "verifier_sha": context["verifier_sha"],
            "workflow_sha256": context["workflow_sha256"],
            "experiment_contract_sha256": context["experiment_contract_sha256"],
            "capture_runner_sha256": context["capture_runner_sha256"],
        },
        "execution": {
            "gate_id": gate["id"],
            "argv": argv,
            "cwd": ".",
            "toolchain": context["toolchain"],
            "platform": context["platform"],
        },
        "inputs": common_inputs,
        "result": {
            "exit_code": completed.returncode,
            "stdout_sha256": sha256_bytes(stdout),
            "stderr_sha256": sha256_bytes(stderr),
        },
    }
    observation: dict[str, Any] = {
        "schema": OBSERVATION_SCHEMA,
        "identity": identity,
        "observation_id": sha256_bytes(canonical_bytes(identity)),
        "provenance": {
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "github_workflow": os.environ.get("GITHUB_WORKFLOW"),
        },
    }
    write_json(gate_dir / "observation.json", observation)
    return observation


def rule_matches(
    rule: dict[str, Any],
    *,
    exit_code: int,
    combined_text: str,
) -> tuple[bool, list[dict[str, Any]]]:
    predicates: list[dict[str, Any]] = []
    when = rule.get("when", {})

    if "exit_code" in when:
        expected = when["exit_code"]
        if expected == "nonzero":
            satisfied = exit_code != 0
        elif expected == "zero":
            satisfied = exit_code == 0
        else:
            satisfied = exit_code == int(expected)
        predicates.append(
            {
                "kind": "exit_code",
                "expected": expected,
                "actual": exit_code,
                "satisfied": satisfied,
            }
        )

    if "combined_output_contains" in when:
        needle = str(when["combined_output_contains"])
        satisfied = needle in combined_text
        predicates.append(
            {
                "kind": "combined_output_contains",
                "needle": needle,
                "satisfied": satisfied,
            }
        )

    return all(predicate["satisfied"] for predicate in predicates), predicates


def classify_gate(
    *,
    observation: dict[str, Any],
    gate_dir: Path,
    classifier: dict[str, Any],
    classifier_sha256: str,
) -> dict[str, Any]:
    identity = observation["identity"]
    gate_id = identity["execution"]["gate_id"]
    exit_code = int(identity["result"]["exit_code"])
    stdout = (gate_dir / "stdout.log").read_bytes()
    stderr = (gate_dir / "stderr.log").read_bytes()
    combined_text = (
        stdout.decode("utf-8", errors="replace")
        + "\n"
        + stderr.decode("utf-8", errors="replace")
    )
    combined_sha256 = framed_digest(COMBINED_OUTPUT_DOMAIN, stdout, stderr)

    matched_rule_id: str | None = None
    matched_predicates: list[dict[str, Any]] = []
    if exit_code == 0:
        result = {"state": "PASS", "failure_class": None}
        matched_predicates = [
            {
                "kind": "exit_code",
                "expected": "zero",
                "actual": exit_code,
                "satisfied": True,
            }
        ]
    else:
        result = dict(classifier["default_nonzero"])
        for rule in classifier.get("rules", []):
            if rule["gate_id"] != gate_id:
                continue
            matched, predicates = rule_matches(
                rule,
                exit_code=exit_code,
                combined_text=combined_text,
            )
            if matched:
                matched_rule_id = rule["rule_id"]
                matched_predicates = predicates
                result = dict(rule["classification"])
                break
        if matched_rule_id is None:
            matched_predicates = [
                {
                    "kind": "no_classifier_rule_matched",
                    "satisfied": True,
                }
            ]

    body: dict[str, Any] = {
        "schema": CLASSIFICATION_SCHEMA,
        "observation_id": observation["observation_id"],
        "classifier_contract_sha256": classifier_sha256,
        "gate_id": gate_id,
        "basis": {
            "exit_code": exit_code,
            "combined_output_sha256": combined_sha256,
            "matched_rule_id": matched_rule_id,
            "predicates": matched_predicates,
        },
        "result": result,
    }
    classification = {
        **body,
        "classification_id": sha256_bytes(canonical_bytes(body)),
    }
    write_json(gate_dir / "classification.json", classification)
    return classification


def ensure_complete(
    evidence_root: Path,
    observations: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
) -> None:
    classifications_by_gate = {
        classification["gate_id"]: classification for classification in classifications
    }
    for observation in observations:
        identity = observation["identity"]
        gate = identity["execution"]["gate_id"]
        gate_dir = evidence_root / gate
        required = (
            "command.json",
            "stdout.log",
            "stderr.log",
            "exit-code.txt",
            "observation.json",
            "classification.json",
        )
        missing = [name for name in required if not (gate_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"{gate}: incomplete durable evidence: {missing}")

        if sha256_file(gate_dir / "stdout.log") != identity["result"]["stdout_sha256"]:
            raise RuntimeError(f"{gate}: stdout digest mismatch")
        if sha256_file(gate_dir / "stderr.log") != identity["result"]["stderr_sha256"]:
            raise RuntimeError(f"{gate}: stderr digest mismatch")
        if observation["observation_id"] != sha256_bytes(canonical_bytes(identity)):
            raise RuntimeError(f"{gate}: observation identity mismatch")

        classification = classifications_by_gate[gate]
        classification_body = {
            key: value
            for key, value in classification.items()
            if key != "classification_id"
        }
        expected_classification_id = sha256_bytes(canonical_bytes(classification_body))
        if classification["classification_id"] != expected_classification_id:
            raise RuntimeError(f"{gate}: classification identity mismatch")
        if classification["observation_id"] != observation["observation_id"]:
            raise RuntimeError(f"{gate}: classification is bound to wrong observation")


def evaluate_negative_controls(
    experiment: dict[str, Any],
    observations: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observations_by_gate = {
        observation["identity"]["execution"]["gate_id"]: observation
        for observation in observations
    }
    classifications_by_gate = {
        classification["gate_id"]: classification for classification in classifications
    }
    results: list[dict[str, Any]] = []

    for control in experiment.get("negative_controls", []):
        gate_id = control["gate_id"]
        observation = observations_by_gate[gate_id]
        classification = classifications_by_gate[gate_id]
        expected_state = control["expected_state"]
        expected_failure_class = control.get("expected_failure_class")
        actual = classification["result"]
        satisfied = (
            actual["state"] == expected_state
            and actual.get("failure_class") == expected_failure_class
        )
        results.append(
            {
                **control,
                "observation_id": observation["observation_id"],
                "classification_id": classification["classification_id"],
                "actual_state": actual["state"],
                "actual_failure_class": actual.get("failure_class"),
                "satisfied": satisfied,
            }
        )
    return results


def write_manifest(
    *,
    output: Path,
    context: dict[str, Any],
    subject_sha: str,
) -> dict[str, Any]:
    files = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative == "manifest.json":
            continue
        files.append({"path": relative, "sha256": sha256_file(path)})

    body: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "capture_contract": CAPTURE_CONTRACT,
        "subject_sha": subject_sha,
        "verifier_sha": context["verifier_sha"],
        "workflow_sha256": context["workflow_sha256"],
        "experiment_contract_sha256": context["experiment_contract_sha256"],
        "classifier_contract_sha256": context["classifier_contract_sha256"],
        "capture_runner_sha256": context["capture_runner_sha256"],
        "files": files,
    }
    manifest = {**body, "manifest_id": sha256_bytes(canonical_bytes(body))}
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, type=Path)
    parser.add_argument("--verifier", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--classifier", required=True, type=Path)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    subject = args.subject.resolve()
    verifier = args.verifier.resolve()
    experiment_path = args.experiment.resolve()
    classifier_path = args.classifier.resolve()
    workflow_path = args.workflow.resolve()
    output = args.output.resolve()
    runner_path = Path(__file__).resolve()

    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.mkdir(parents=True)

    experiment = load_json(experiment_path)
    classifier = load_json(classifier_path)
    expected_subject = experiment["subject_sha"]
    actual_subject = git_text(subject, "rev-parse", "HEAD")
    if actual_subject != expected_subject:
        raise RuntimeError(
            f"subject mismatch: expected {expected_subject}, got {actual_subject}"
        )

    pre_status = git_text(subject, "status", "--porcelain")
    if pre_status:
        raise RuntimeError(f"subject is dirty before replay:\n{pre_status}")

    verifier_sha = git_text(verifier, "rev-parse", "HEAD")
    toolchain = toolchain_identity(subject)
    expected_toolchain = experiment["toolchain"]
    if expected_toolchain not in toolchain["rustc"]:
        raise RuntimeError(
            f"unexpected rustc identity: expected {expected_toolchain}, "
            f"got {toolchain['rustc']!r}"
        )

    context = {
        "subject_sha": actual_subject,
        "verifier_sha": verifier_sha,
        "workflow_sha256": sha256_file(workflow_path),
        "experiment_contract_sha256": sha256_file(experiment_path),
        "classifier_contract_sha256": sha256_file(classifier_path),
        "capture_runner_sha256": sha256_file(runner_path),
        "toolchain": toolchain,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "runner_os": os.environ.get("RUNNER_OS"),
            "image_os": os.environ.get("ImageOS"),
            "image_version": os.environ.get("ImageVersion"),
        },
    }
    common_inputs = input_digests(subject, experiment["inputs"])

    observations: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    for gate in experiment["gates"]:
        observation = capture_gate(
            gate=gate,
            subject=subject,
            evidence_root=output,
            context=context,
            common_inputs=common_inputs,
        )
        observations.append(observation)
        classifications.append(
            classify_gate(
                observation=observation,
                gate_dir=output / gate["id"],
                classifier=classifier,
                classifier_sha256=context["classifier_contract_sha256"],
            )
        )

    ensure_complete(output, observations, classifications)

    post_head = git_text(subject, "rev-parse", "HEAD")
    post_status = git_text(subject, "status", "--porcelain")
    post_inputs = input_digests(subject, experiment["inputs"])
    input_identity_preserved = post_inputs == common_inputs
    immutable = (
        post_head == actual_subject
        and not post_status
        and input_identity_preserved
    )

    negative_controls = evaluate_negative_controls(
        experiment,
        observations,
        classifications,
    )

    observation_by_gate = {
        observation["identity"]["execution"]["gate_id"]: observation
        for observation in observations
    }
    summary_observations = []
    for classification in classifications:
        gate_id = classification["gate_id"]
        observation = observation_by_gate[gate_id]
        summary_observations.append(
            {
                "gate_id": gate_id,
                "observation_id": observation["observation_id"],
                "classification_id": classification["classification_id"],
                "exit_code": observation["identity"]["result"]["exit_code"],
                "state": classification["result"]["state"],
                "failure_class": classification["result"]["failure_class"],
            }
        )

    failure_classification_complete = all(
        item["exit_code"] == 0 or item["state"] == "CLASSIFIED_FAIL"
        for item in summary_observations
    )
    summary_body: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "capture_contract": CAPTURE_CONTRACT,
        "subject_sha": actual_subject,
        "verifier_sha": verifier_sha,
        "experiment_contract_sha256": context["experiment_contract_sha256"],
        "classifier_contract_sha256": context["classifier_contract_sha256"],
        "workflow_sha256": context["workflow_sha256"],
        "capture_runner_sha256": context["capture_runner_sha256"],
        "observations": summary_observations,
        "negative_controls": negative_controls,
        "immutability": {
            "head_unchanged": post_head == actual_subject,
            "working_tree_clean": not post_status,
            "input_identity_preserved": input_identity_preserved,
        },
        "evidence_complete": True,
        "failure_classification_complete": failure_classification_complete,
        "qualification_claim": "NONE",
        "repair_authority_claim": "NONE",
    }
    summary = {
        **summary_body,
        "summary_id": sha256_bytes(canonical_bytes(summary_body)),
    }
    write_json(output / "summary.json", summary)
    manifest = write_manifest(
        output=output,
        context=context,
        subject_sha=actual_subject,
    )

    negative_controls_ok = all(item["satisfied"] for item in negative_controls)
    if not immutable:
        print("SE001Q_EVIDENCE_REPLAY=FAIL subject mutated", file=sys.stderr)
        return 2
    if not negative_controls_ok:
        print(
            "SE001Q_EVIDENCE_REPLAY=FAIL typed negative control invalidated",
            file=sys.stderr,
        )
        return 3

    print(
        "SE001Q_EVIDENCE_REPLAY=PASS "
        f"summary_id={summary['summary_id']} manifest_id={manifest['manifest_id']}"
    )
    for item in summary["observations"]:
        print(
            f"{item['gate_id']}: exit={item['exit_code']} "
            f"state={item['state']} "
            f"failure_class={item['failure_class']} "
            f"observation_id={item['observation_id']} "
            f"classification_id={item['classification_id']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
