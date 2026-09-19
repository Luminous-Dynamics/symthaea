#!/usr/bin/env python3
"""Durably capture SE-001Q gate observations without mutating the frozen subject."""

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


SCHEMA = "symthaea.qualification-observation.v1"
SUMMARY_SCHEMA = "symthaea.qualification-replay-summary.v1"
CAPTURE_CONTRACT = "symthaea.se001q.capture.v2"


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def command_output(argv: list[str], cwd: Path) -> bytes:
    result = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
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


def classify(gate: dict[str, Any], exit_code: int) -> dict[str, Any]:
    if exit_code == 0:
        return {
            "state": "PASS",
            "failure_class": None,
            "repair_authority": "NONE",
        }
    policy = gate["failure_policy"]
    return {
        "state": policy["state"],
        "failure_class": policy["failure_class"],
        "repair_authority": policy["repair_authority"],
    }


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
    (gate_dir / "exit-code.txt").write_text(f"{completed.returncode}\n", encoding="utf-8")

    observation: dict[str, Any] = {
        "schema": SCHEMA,
        "capture_contract": CAPTURE_CONTRACT,
        "subject": {"sha": context["subject_sha"]},
        "qualification": {
            "claim": "NONE",
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
        "classification": classify(gate, completed.returncode),
        "provenance": {
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "github_workflow": os.environ.get("GITHUB_WORKFLOW"),
        },
    }
    observation["observation_id"] = sha256_bytes(canonical_bytes(observation))
    write_json(gate_dir / "observation.json", observation)
    return observation


def ensure_complete(evidence_root: Path, observations: list[dict[str, Any]]) -> None:
    for observation in observations:
        gate = observation["execution"]["gate_id"]
        gate_dir = evidence_root / gate
        required = ("command.json", "stdout.log", "stderr.log", "exit-code.txt", "observation.json")
        missing = [name for name in required if not (gate_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"{gate}: incomplete durable evidence: {missing}")
        if sha256_file(gate_dir / "stdout.log") != observation["result"]["stdout_sha256"]:
            raise RuntimeError(f"{gate}: stdout digest mismatch")
        if sha256_file(gate_dir / "stderr.log") != observation["result"]["stderr_sha256"]:
            raise RuntimeError(f"{gate}: stderr digest mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, type=Path)
    parser.add_argument("--verifier", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    subject = args.subject.resolve()
    verifier = args.verifier.resolve()
    experiment_path = args.experiment.resolve()
    workflow_path = args.workflow.resolve()
    output = args.output.resolve()
    runner_path = Path(__file__).resolve()

    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.mkdir(parents=True)

    experiment = load_json(experiment_path)
    expected_subject = experiment["subject_sha"]
    actual_subject = git_text(subject, "rev-parse", "HEAD")
    if actual_subject != expected_subject:
        raise RuntimeError(f"subject mismatch: expected {expected_subject}, got {actual_subject}")

    pre_status = git_text(subject, "status", "--porcelain")
    if pre_status:
        raise RuntimeError(f"subject is dirty before replay:\n{pre_status}")

    verifier_sha = git_text(verifier, "rev-parse", "HEAD")
    context = {
        "subject_sha": actual_subject,
        "verifier_sha": verifier_sha,
        "workflow_sha256": sha256_file(workflow_path),
        "experiment_contract_sha256": sha256_file(experiment_path),
        "capture_runner_sha256": sha256_file(runner_path),
        "toolchain": toolchain_identity(subject),
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
    for gate in experiment["gates"]:
        observations.append(
            capture_gate(
                gate=gate,
                subject=subject,
                evidence_root=output,
                context=context,
                common_inputs=common_inputs,
            )
        )

    ensure_complete(output, observations)

    post_head = git_text(subject, "rev-parse", "HEAD")
    post_status = git_text(subject, "status", "--porcelain")
    immutable = post_head == actual_subject and not post_status

    by_gate = {o["execution"]["gate_id"]: o for o in observations}
    negative_controls: list[dict[str, Any]] = []
    for control in experiment.get("negative_controls", []):
        observation = by_gate[control["gate_id"]]
        negative_controls.append(
            {
                **control,
                "observation_id": observation["observation_id"],
                "satisfied": observation["result"]["exit_code"] != 0,
            }
        )

    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "capture_contract": CAPTURE_CONTRACT,
        "subject_sha": actual_subject,
        "verifier_sha": verifier_sha,
        "experiment_contract_sha256": context["experiment_contract_sha256"],
        "workflow_sha256": context["workflow_sha256"],
        "capture_runner_sha256": context["capture_runner_sha256"],
        "observations": [
            {
                "gate_id": observation["execution"]["gate_id"],
                "observation_id": observation["observation_id"],
                "exit_code": observation["result"]["exit_code"],
                "state": observation["classification"]["state"],
                "failure_class": observation["classification"]["failure_class"],
                "repair_authority": observation["classification"]["repair_authority"],
            }
            for observation in observations
        ],
        "negative_controls": negative_controls,
        "immutability": {
            "head_unchanged": post_head == actual_subject,
            "working_tree_clean": not post_status,
        },
        "evidence_complete": True,
        "qualification_claim": "NONE",
    }
    summary["summary_id"] = sha256_bytes(canonical_bytes(summary))
    write_json(output / "summary.json", summary)

    negative_controls_ok = all(item["satisfied"] for item in negative_controls)
    if not immutable:
        print("SE001Q_EVIDENCE_REPLAY=FAIL subject mutated", file=sys.stderr)
        return 2
    if not negative_controls_ok:
        print("SE001Q_EVIDENCE_REPLAY=FAIL negative control invalidated", file=sys.stderr)
        return 3

    print(f"SE001Q_EVIDENCE_REPLAY=PASS summary_id={summary['summary_id']}")
    for observation in summary["observations"]:
        print(
            f"{observation['gate_id']}: exit={observation['exit_code']} "
            f"state={observation['state']} "
            f"observation_id={observation['observation_id']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
