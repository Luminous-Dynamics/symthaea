#!/usr/bin/env python3
"""Audit a VLEB evaluator-produced witness receipt against out-of-band pins.

This validates identity, replay bindings, schema and artifact-byte integrity. It
cannot authenticate the producer without verifying the attested bundle, nor can
it prove semantic correctness merely from a receipt declaration. Confirmatory
use must be nested under the separately pinned run-manifest/Sigstore gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:
    raise SystemExit("jsonschema==4.26.0 is required; refusing to skip schema validation") from exc

HEX = re.compile(r"^[a-f0-9]{64}$")
HEX40 = re.compile(r"^[a-f0-9]{40}$")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError("artifact path must be a non-empty POSIX relative path")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or str(rel) != relative:
        raise ValueError("absolute, traversal, or ambiguous artifact path rejected")
    root_resolved = root.resolve(strict=True)
    current = root_resolved
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink artifact path rejected")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
        raise ValueError("artifact must be a regular file inside the evidence root")
    return resolved


def _add_pin_issue(issues: list[dict[str, str]], receipt: dict[str, Any], path: tuple[str, ...], expected: Any, code: str) -> None:
    observed: Any = receipt
    try:
        for key in path:
            observed = observed[key]
    except (KeyError, TypeError):
        observed = None
    if observed != expected:
        issues.append({"code": code, "message": f"{'.'.join(path)} differs from evaluator-pinned value"})


def validate_receipt(
    receipt_path: Path,
    schema_path: Path,
    artifact_root: Path,
    *,
    expected_receipt_sha256: str,
    expected_schema_sha256: str,
    expected_repository_id: str,
    expected_workflow_path: str,
    expected_workflow_sha: str,
    expected_run_id: str,
    expected_run_attempt: int,
    expected_task_id: str,
    expected_claim_id: str,
    expected_source_sha256: str,
    expected_verifier_kind: str,
    expected_verifier_id: str,
    expected_verifier_sha256: str,
    expected_runtime_image_digest: str,
    expected_toolchain_sha256: str,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    for label, value in (("receipt", expected_receipt_sha256), ("schema", expected_schema_sha256),
                         ("source", expected_source_sha256), ("verifier", expected_verifier_sha256),
                         ("toolchain", expected_toolchain_sha256)):
        if not isinstance(value, str) or not HEX.fullmatch(value):
            issues.append({"code": f"invalid_external_{label}_pin", "message": f"external {label} pin must be a lowercase SHA-256"})
    if not isinstance(expected_workflow_sha, str) or not HEX40.fullmatch(expected_workflow_sha):
        issues.append({"code": "invalid_external_workflow_sha_pin", "message": "external workflow SHA pin must be a lowercase 40-character Git revision"})
    if not isinstance(expected_run_attempt, int) or isinstance(expected_run_attempt, bool) or expected_run_attempt < 1:
        issues.append({"code": "invalid_external_run_attempt_pin", "message": "external run attempt pin must be a positive integer"})
    if issues:
        return {"schema_version": "vleb-witness-receipt-audit-v1", "eligible": False, "verdict": "fail", "issues": issues}

    try:
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError("receipt must be a regular non-symlink file")
        receipt_bytes = receipt_path.read_bytes()
        observed_receipt_sha = digest(receipt_bytes)
    except (OSError, ValueError) as exc:
        return {"schema_version": "vleb-witness-receipt-audit-v1", "eligible": False, "verdict": "fail", "issues": [{"code": "receipt_unreadable", "message": str(exc)}]}
    if observed_receipt_sha != expected_receipt_sha256:
        issues.append({"code": "receipt_digest_mismatch", "message": "receipt bytes differ from the evaluator-pinned receipt digest"})

    try:
        schema_bytes = schema_path.read_bytes()
        observed_schema_sha = digest(schema_bytes)
        if observed_schema_sha != expected_schema_sha256:
            issues.append({"code": "schema_digest_mismatch", "message": "schema bytes differ from the evaluator-pinned schema digest"})
        schema = json.loads(schema_bytes.decode("utf-8"))
        Draft202012Validator.check_schema(schema)
        receipt = json.loads(receipt_bytes.decode("utf-8"))
        if not isinstance(receipt, dict):
            raise ValueError("receipt JSON root must be an object")
    except Exception as exc:
        issues.append({"code": "receipt_or_schema_invalid", "message": f"cannot parse or validate receipt/schema: {exc}"})
        return {"schema_version": "vleb-witness-receipt-audit-v1", "eligible": False, "verdict": "fail", "observed_receipt_sha256": observed_receipt_sha, "issues": issues}

    errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(receipt), key=lambda e: list(map(str, e.absolute_path)))
    for error in errors:
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        issues.append({"code": "receipt_schema_invalid", "message": f"{path}: {error.message}"})

    if not errors:
        pins = [
            (("producer", "repository_id"), expected_repository_id, "repository_id_mismatch"),
            (("producer", "workflow_path"), expected_workflow_path, "workflow_path_mismatch"),
            (("producer", "workflow_sha"), expected_workflow_sha, "workflow_sha_mismatch"),
            (("producer", "run_id"), expected_run_id, "run_id_mismatch"),
            (("producer", "run_attempt"), expected_run_attempt, "run_attempt_mismatch"),
            (("task_id",), expected_task_id, "task_id_mismatch"),
            (("claim_id",), expected_claim_id, "claim_id_mismatch"),
            (("source", "sha256"), expected_source_sha256, "source_digest_mismatch"),
            (("verifier", "kind"), expected_verifier_kind, "verifier_kind_mismatch"),
            (("verifier", "id"), expected_verifier_id, "verifier_id_mismatch"),
            (("verifier", "sha256"), expected_verifier_sha256, "verifier_digest_mismatch"),
            (("runtime", "image_digest"), expected_runtime_image_digest, "runtime_image_mismatch"),
            (("runtime", "toolchain_sha256"), expected_toolchain_sha256, "toolchain_digest_mismatch"),
        ]
        for path, expected, code in pins:
            _add_pin_issue(issues, receipt, path, expected, code)

        if receipt["execution"]["status"] != "completed" and receipt["claim_result"] != "inconclusive":
            issues.append({"code": "incomplete_execution_claimed_as_decisive", "message": "timed-out, cancelled or failed-to-spawn execution must be inconclusive"})

        try:
            source_path = safe_file(artifact_root, receipt["source"]["path"])
            source_raw = source_path.read_bytes()
            if digest(source_raw) != receipt["source"]["sha256"]:
                issues.append({"code": "source_artifact_digest_mismatch", "message": "source bytes differ from the receipt's source digest"})
            lines = source_raw.splitlines()
            for index, span in enumerate(receipt["source"]["spans"]):
                first, last = span["line_start"], span["line_end"]
                start_col, end_col = span["column_start"], span["column_end"]
                label = f"source.spans[{index}]"
                if first < 1 or last < first or last > len(lines) or start_col > len(lines[first - 1]) or end_col > len(lines[last - 1]):
                    issues.append({"code": "source_span_out_of_range", "message": f"{label} is outside the source snapshot"})
                    continue
                if first == last:
                    if end_col < start_col:
                        issues.append({"code": "source_span_invalid", "message": f"{label} ends before it starts"})
                        continue
                    excerpt = lines[first - 1][start_col - 1:end_col]
                else:
                    chunks = [lines[first - 1][start_col - 1:]]
                    chunks.extend(lines[first:last - 1])
                    chunks.append(lines[last - 1][:end_col])
                    excerpt = b"\n".join(chunks)
                if digest(excerpt) != span["excerpt_sha256"]:
                    issues.append({"code": "source_excerpt_digest_mismatch", "message": f"{label} bytes differ from the declared excerpt digest"})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append({"code": "source_path_invalid", "message": str(exc)})

        seen_paths: set[str] = set()
        for group in ("inputs", "outputs"):
            for index, artifact in enumerate(receipt[group]):
                label = f"{group}[{index}]"
                relative = artifact["path"]
                if relative in seen_paths:
                    issues.append({"code": "duplicate_artifact_path", "message": f"{label} reuses another input/output path"})
                    continue
                seen_paths.add(relative)
                try:
                    actual_path = safe_file(artifact_root, relative)
                    observed = digest(actual_path.read_bytes())
                    if observed != artifact["sha256"]:
                        issues.append({"code": "artifact_digest_mismatch", "message": f"{label} bytes differ from declared SHA-256"})
                except (OSError, ValueError) as exc:
                    issues.append({"code": "artifact_path_invalid", "message": f"{label}: {exc}"})

    eligible = not issues
    return {
        "schema_version": "vleb-witness-receipt-audit-v1",
        "eligible": eligible,
        "verdict": "pass" if eligible else "fail",
        "claim_result": receipt.get("claim_result") if isinstance(receipt, dict) else None,
        "receipt_id": receipt.get("receipt_id") if isinstance(receipt, dict) else None,
        "task_id": receipt.get("task_id") if isinstance(receipt, dict) else None,
        "claim_id": receipt.get("claim_id") if isinstance(receipt, dict) else None,
        "observed_receipt_sha256": observed_receipt_sha,
        "observed_schema_sha256": observed_schema_sha,
        "issues": issues,
        "assurance_boundary": "Receipt audit checks pinned identity, replay bindings and artifact integrity; producer authentication and semantic truth require the outer attestation gate and trusted verifier execution policy.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--expected-schema-sha256", required=True)
    parser.add_argument("--expected-repository-id", required=True)
    parser.add_argument("--expected-workflow-path", required=True)
    parser.add_argument("--expected-workflow-sha", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-run-attempt", type=int, required=True)
    parser.add_argument("--expected-task-id", required=True)
    parser.add_argument("--expected-claim-id", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-verifier-kind", required=True)
    parser.add_argument("--expected-verifier-id", required=True)
    parser.add_argument("--expected-verifier-sha256", required=True)
    parser.add_argument("--expected-runtime-image-digest", required=True)
    parser.add_argument("--expected-toolchain-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = validate_receipt(
        args.receipt, args.schema, args.artifact_root,
        expected_receipt_sha256=args.expected_receipt_sha256,
        expected_schema_sha256=args.expected_schema_sha256,
        expected_repository_id=args.expected_repository_id,
        expected_workflow_path=args.expected_workflow_path,
        expected_workflow_sha=args.expected_workflow_sha,
        expected_run_id=args.expected_run_id,
        expected_run_attempt=args.expected_run_attempt,
        expected_task_id=args.expected_task_id,
        expected_claim_id=args.expected_claim_id,
        expected_source_sha256=args.expected_source_sha256,
        expected_verifier_kind=args.expected_verifier_kind,
        expected_verifier_id=args.expected_verifier_id,
        expected_verifier_sha256=args.expected_verifier_sha256,
        expected_runtime_image_digest=args.expected_runtime_image_digest,
        expected_toolchain_sha256=args.expected_toolchain_sha256,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result.get("eligible") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
