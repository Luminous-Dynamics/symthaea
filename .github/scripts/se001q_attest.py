#!/usr/bin/env python3
"""Run the independent SE-001Q verifier and durably attest its result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ATTESTATION_SCHEMA = "symthaea.qualification-verification-attestation.v1"
ATTESTATION_MANIFEST_SCHEMA = "symthaea.qualification-verification-manifest.v1"


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_sha(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {"sha": result.stdout.strip(), "error": None}
    except Exception as exc:  # Durable infrastructure rejection evidence.
        return {"sha": None, "error": f"{type(exc).__name__}: {exc}"}


def json_reference(path: Path, identity_key: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "sha256": None,
            "identity": None,
            "parse_error": None,
        }
    digest = sha256_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        identity = value.get(identity_key) if isinstance(value, dict) else None
        return {
            "present": True,
            "sha256": digest,
            "identity": identity,
            "parse_error": None,
        }
    except Exception as exc:  # Preserve malformed evidence as a referenced byte object.
        return {
            "present": True,
            "sha256": digest,
            "identity": None,
            "parse_error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "evidence",
        "subject",
        "verifier",
        "experiment",
        "classifier",
        "workflow",
        "capture-runner",
        "verifier-script",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()

    evidence = args.evidence.resolve()
    subject = args.subject.resolve()
    verifier = args.verifier.resolve()
    experiment = args.experiment.resolve()
    classifier = args.classifier.resolve()
    workflow = args.workflow.resolve()
    capture_runner = getattr(args, "capture_runner").resolve()
    verifier_script = getattr(args, "verifier_script").resolve()
    output = args.output.resolve()
    attester_script = Path(__file__).resolve()

    if output.exists():
        raise RuntimeError(f"attestation output already exists: {output}")
    output.mkdir(parents=True)

    logical_argv = [
        "python3",
        ".github/scripts/se001q_verify.py",
        "--evidence", "evidence",
        "--subject", "subject",
        "--verifier", "verifier",
        "--experiment", ".github/qualification/se001q-experiment-v1.json",
        "--classifier", ".github/qualification/se001q-classifier-v1.json",
        "--workflow", ".github/workflows/systems-engineering-se001q-evidence-v2.yml",
        "--capture-runner", ".github/scripts/se001q_capture.py",
    ]
    exec_argv = [
        sys.executable,
        str(verifier_script),
        "--evidence", str(evidence),
        "--subject", str(subject),
        "--verifier", str(verifier),
        "--experiment", str(experiment),
        "--classifier", str(classifier),
        "--workflow", str(workflow),
        "--capture-runner", str(capture_runner),
    ]

    completed = subprocess.run(
        exec_argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (output / "verification-stdout.log").write_bytes(completed.stdout)
    (output / "verification-stderr.log").write_bytes(completed.stderr)
    (output / "verification-exit-code.txt").write_text(
        f"{completed.returncode}\n", encoding="utf-8"
    )

    manifest_ref = json_reference(evidence / "manifest.json", "manifest_id")
    summary_ref = json_reference(evidence / "summary.json", "summary_id")
    subject_identity = git_sha(subject)
    verifier_identity = git_sha(verifier)

    reference_complete = (
        manifest_ref["present"]
        and manifest_ref["parse_error"] is None
        and manifest_ref["identity"] is not None
        and summary_ref["present"]
        and summary_ref["parse_error"] is None
        and summary_ref["identity"] is not None
        and subject_identity["sha"] is not None
        and verifier_identity["sha"] is not None
    )

    body: dict[str, Any] = {
        "schema": ATTESTATION_SCHEMA,
        "subject": subject_identity,
        "verifier": verifier_identity,
        "qualification_claim": "NONE",
        "repair_authority_claim": "NONE",
        "verification": {
            "logical_argv": logical_argv,
            "verifier_script_sha256": sha256_file(verifier_script),
            "attester_script_sha256": sha256_file(attester_script),
            "experiment_contract_sha256": sha256_file(experiment),
            "classifier_contract_sha256": sha256_file(classifier),
            "workflow_sha256": sha256_file(workflow),
            "capture_runner_sha256": sha256_file(capture_runner),
            "result": {
                "exit_code": completed.returncode,
                "stdout_sha256": sha256_bytes(completed.stdout),
                "stderr_sha256": sha256_bytes(completed.stderr),
                "verified": completed.returncode == 0,
            },
        },
        "evidence": {
            "manifest": manifest_ref,
            "summary": summary_ref,
            "reference_complete": reference_complete,
        },
        "provenance": {
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "github_workflow": os.environ.get("GITHUB_WORKFLOW"),
        },
    }
    attestation = {
        **body,
        "attestation_id": sha256_bytes(canonical(body)),
    }
    write_json(output / "verification.json", attestation)

    files = []
    for path in sorted(p for p in output.iterdir() if p.is_file()):
        if path.name == "manifest.json":
            continue
        files.append({"path": path.name, "sha256": sha256_file(path)})
    manifest_body = {
        "schema": ATTESTATION_MANIFEST_SCHEMA,
        "subject_sha": subject_identity["sha"],
        "verifier_sha": verifier_identity["sha"],
        "evidence_manifest_id": manifest_ref["identity"],
        "verification_attestation_id": attestation["attestation_id"],
        "qualification_claim": "NONE",
        "repair_authority_claim": "NONE",
        "files": files,
    }
    verification_manifest = {
        **manifest_body,
        "manifest_id": sha256_bytes(canonical(manifest_body)),
    }
    write_json(output / "manifest.json", verification_manifest)

    print(
        f"SE-001Q independent verification attested: exit={completed.returncode} "
        f"references_complete={reference_complete} "
        f"attestation_id={attestation['attestation_id']} "
        f"manifest_id={verification_manifest['manifest_id']}"
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
