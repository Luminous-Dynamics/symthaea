#!/usr/bin/env python3
"""Audit VLEB legacy claim evidence hashes and trust bindings.

This checks evidence integrity and identity pins. It does not independently
re-run a compiler or transaction scenario; the trusted witness/verifier and
run-provenance policies remain separate requirements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:
    raise SystemExit("jsonschema==4.26.0 is required; refusing to skip schema validation") from exc

HEX = re.compile(r"^[a-f0-9]{64}$")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def parse_unique_json(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_unique_json_object)
WITNESS_KINDS = {
    "source_index",
    "static_analysis",
    "compiler",
    "independent_execution",
    "independent_expert_review",
}


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
        raise ValueError("artifact must be a regular file under the declared root")
    return resolved


def validate_claim(
    claim_path: Path,
    schema_path: Path,
    source_root: Path,
    evidence_root: Path,
    *,
    expected_claim_sha256: str,
    expected_auditor_sha256: str,
    expected_schema_sha256: str,
    expected_source_sha256: str,
    trusted_verifier_pins: dict[str, str],
    check_running_auditor: bool = True,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []

    pins = {
        "claim": expected_claim_sha256,
        "auditor": expected_auditor_sha256,
        "schema": expected_schema_sha256,
        "source": expected_source_sha256,
        **{f"verifier:{kind}": value for kind, value in trusted_verifier_pins.items()},
    }
    for label, value in pins.items():
        if not HEX.fullmatch(value):
            issues.append({"code": "invalid_trusted_pin", "message": f"{label} must be a lowercase SHA-256 digest"})
    if set(trusted_verifier_pins) - WITNESS_KINDS:
        issues.append({"code": "unknown_trusted_verifier_kind", "message": "trusted verifier map contains an unknown witness kind"})
    if issues:
        return {"schema_version": "vleb-legacy-claim-audit-v1", "evidence_chain_valid": False, "verdict": "fail", "issues": issues}

    try:
        if check_running_auditor:
            observed_auditor_sha = digest(Path(__file__).read_bytes())
            if observed_auditor_sha != expected_auditor_sha256:
                issues.append({"code": "auditor_pin_mismatch", "message": "running claim auditor differs from evaluator-pinned source"})
        raw_schema = schema_path.read_bytes()
        observed_schema_sha = digest(raw_schema)
        if observed_schema_sha != expected_schema_sha256:
            issues.append({"code": "schema_pin_mismatch", "message": "claim schema differs from evaluator-pinned schema"})
        raw_claim = claim_path.read_bytes()
        observed_claim_sha = digest(raw_claim)
        if observed_claim_sha != expected_claim_sha256:
            issues.append({"code": "claim_pin_mismatch", "message": "claim bytes differ from caller-supplied trusted digest"})
    except OSError as exc:
        return {
            "schema_version": "vleb-legacy-claim-audit-v1",
            "evidence_chain_valid": False,
            "verdict": "unresolved",
            "issues": [{"code": "trusted_input_unavailable", "message": str(exc)}],
        }

    if issues:
        return {
            "schema_version": "vleb-legacy-claim-audit-v1",
            "evidence_chain_valid": False,
            "verdict": "fail",
            "observed_claim_sha256": observed_claim_sha,
            "observed_schema_sha256": observed_schema_sha,
            "issues": issues,
        }

    try:
        schema = parse_unique_json(raw_schema)
        claim = parse_unique_json(raw_claim)
    except (UnicodeError, ValueError) as exc:
        return {
            "schema_version": "vleb-legacy-claim-audit-v1",
            "evidence_chain_valid": False,
            "verdict": "fail",
            "observed_claim_sha256": observed_claim_sha,
            "issues": [{"code": "invalid_json", "message": str(exc)}],
        }
    if not isinstance(claim, dict):
        return {
            "schema_version": "vleb-legacy-claim-audit-v1",
            "evidence_chain_valid": False,
            "verdict": "fail",
            "observed_claim_sha256": observed_claim_sha,
            "issues": [{"code": "claim_root_invalid", "message": "claim root must be a JSON object"}],
        }

    try:
        Draft202012Validator.check_schema(schema)
        schema_errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(claim))
    except Exception as exc:
        return {
            "schema_version": "vleb-legacy-claim-audit-v1",
            "evidence_chain_valid": False,
            "verdict": "fail",
            "observed_claim_sha256": observed_claim_sha,
            "issues": [{"code": "schema_validator_error", "message": str(exc)}],
        }
    for error in schema_errors:
        location = ".".join(map(str, error.absolute_path)) or "$"
        issues.append({"code": "claim_schema_invalid", "message": f"{location}: {error.message}"})

    source_record: dict[str, Any] = claim.get("source", {})
    source_path: Path | None = None
    source_raw: bytes | None = None
    try:
        source_path = safe_file(source_root, source_record.get("path"))
        source_raw = source_path.read_bytes()
        observed_source_sha = digest(source_raw)
        if observed_source_sha != expected_source_sha256 or observed_source_sha != source_record.get("sha256"):
            issues.append({"code": "source_digest_mismatch", "message": "source bytes do not match both the evaluator pin and claim manifest"})
    except (OSError, ValueError) as exc:
        observed_source_sha = None
        issues.append({"code": "source_path_invalid", "message": str(exc)})

    if source_raw is not None:
        # The initial pilot treats each span as byte-oriented fixed-format text.
        # Newline boundaries are normalized to LF only inside multi-line excerpts;
        # the full source digest above still binds exact original bytes.
        lines = source_raw.splitlines()
        for index, span in enumerate(source_record.get("spans", [])):
            label = f"source.spans[{index}]"
            first = span.get("line_start")
            last = span.get("line_end")
            start_col = span.get("column_start")
            end_col = span.get("column_end")
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in (first, last, start_col, end_col)):
                issues.append({"code": "source_span_invalid", "message": f"{label} line/column fields must be integers"})
                continue
            if first < 1 or last < first or last > len(lines) or start_col < 1 or end_col < 1 or start_col > 72 or end_col > 72:
                issues.append({"code": "source_span_out_of_range", "message": f"{label} is outside the source file or fixed-format source columns"})
                continue
            first_line, last_line = lines[first - 1], lines[last - 1]
            if start_col > len(first_line) or end_col > len(last_line):
                issues.append({"code": "source_span_out_of_range", "message": f"{label} exceeds a physical source record"})
                continue
            if first == last:
                if end_col < start_col:
                    issues.append({"code": "source_span_invalid", "message": f"{label} ends before it starts"})
                    continue
                excerpt = first_line[start_col - 1:end_col]
            else:
                chunks = [first_line[start_col - 1:]]
                chunks.extend(lines[first:last - 1])
                chunks.append(last_line[:end_col])
                excerpt = b"\n".join(chunks)
            if digest(excerpt) != span.get("excerpt_sha256"):
                issues.append({"code": "source_excerpt_digest_mismatch", "message": f"{label} excerpt bytes do not match the declared digest"})

    witnesses = claim.get("witnesses", [])
    witness_results: list[str] = []
    execution_results: list[str] = []
    for index, witness in enumerate(witnesses):
        label = f"witnesses[{index}]"
        kind = witness.get("kind")
        if not isinstance(kind, str) or kind not in WITNESS_KINDS:
            issues.append({"code": "witness_kind_invalid", "message": f"{label} has an invalid witness kind"})
        else:
            expected_verifier_sha = trusted_verifier_pins.get(kind)
            if expected_verifier_sha is None:
                issues.append({"code": "trusted_verifier_missing", "message": f"{label} kind {kind!r} has no out-of-band verifier pin"})
            elif witness.get("verifier_sha256") != expected_verifier_sha:
                issues.append({"code": "witness_verifier_pin_mismatch", "message": f"{label} verifier digest differs from evaluator allowlist"})
        try:
            artifact = safe_file(evidence_root, witness.get("artifact_path"))
            observed = digest(artifact.read_bytes())
            if observed != witness.get("artifact_sha256"):
                issues.append({"code": "witness_artifact_digest_mismatch", "message": f"{label} artifact bytes differ from claim evidence"})
        except (OSError, ValueError) as exc:
            issues.append({"code": "witness_artifact_invalid", "message": f"{label}: {exc}"})
        try:
            execution_receipt = safe_file(evidence_root, witness.get("execution_receipt_path"))
            receipt_observed = digest(execution_receipt.read_bytes())
            if receipt_observed != witness.get("execution_receipt_sha256"):
                issues.append({"code": "execution_receipt_digest_mismatch", "message": f"{label} execution receipt bytes differ from claim evidence"})
        except (OSError, ValueError) as exc:
            issues.append({"code": "execution_receipt_invalid", "message": f"{label}: {exc}"})
        result = witness.get("observed_result")
        witness_results.append(result)
        if kind == "independent_execution":
            execution_results.append(result)

    verdict = claim.get("verdict")
    scope = claim.get("claim_scope")
    if "supports" in witness_results and "contradicts" in witness_results:
        issues.append({"code": "witnesses_conflict", "message": "supporting and contradicting witnesses conflict; claim needs adjudication"})
    if verdict == "supported" and "supports" not in witness_results:
        issues.append({"code": "supported_verdict_lacks_supporting_witness", "message": "no witness supports the claim"})
    if verdict == "contradicted" and "contradicts" not in witness_results:
        issues.append({"code": "contradicted_verdict_lacks_contradicting_witness", "message": "no witness contradicts the claim"})
    if scope == "runtime_behavior" and verdict == "supported" and "supports" not in execution_results:
        issues.append({"code": "runtime_claim_requires_execution", "message": "runtime behavior cannot be supported by source indexing or static evidence alone"})
    if scope == "runtime_behavior" and verdict == "contradicted" and "contradicts" not in execution_results:
        issues.append({"code": "runtime_contradiction_requires_execution", "message": "runtime contradiction needs a failing independent execution witness"})

    valid = not issues
    return {
        "schema_version": "vleb-legacy-claim-audit-v1",
        "evidence_chain_valid": valid,
        "verdict": "pass" if valid else "fail",
        "claim_id": claim.get("claim_id"),
        "task_id": claim.get("task_id"),
        "claim_verdict": verdict,
        "claim_scope": scope,
        "claim_sha256": observed_claim_sha,
        "source_sha256": observed_source_sha,
        "schema_sha256": observed_schema_sha,
        "evidence_chain_note": "Pass validates pins, source spans, witness bytes and verifier identity; it does not itself reproduce the underlying compiler or transaction behavior.",
        "issues": issues,
    }


def parse_verifier_pin(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected KIND=SHA256")
    kind, sha = value.split("=", 1)
    if kind not in WITNESS_KINDS or not HEX.fullmatch(sha):
        raise argparse.ArgumentTypeError("expected a known witness kind and lowercase SHA-256 digest")
    return kind, sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path(__file__).with_name("legacy-claim-evidence.schema.json"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--expected-claim-sha256", required=True)
    parser.add_argument("--expected-auditor-sha256", required=True)
    parser.add_argument("--expected-schema-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--trusted-verifier", action="append", type=parse_verifier_pin, default=[])
    args = parser.parse_args(argv)

    trusted: dict[str, str] = {}
    duplicate_kind = False
    for kind, sha in args.trusted_verifier:
        if kind in trusted:
            duplicate_kind = True
        trusted[kind] = sha
    if duplicate_kind:
        print(json.dumps({
            "schema_version": "vleb-legacy-claim-audit-v1",
            "evidence_chain_valid": False,
            "verdict": "fail",
            "issues": [{"code": "duplicate_verifier_pin", "message": "only one trusted digest may be supplied per witness kind"}],
        }, indent=2))
        return 1

    result = validate_claim(
        args.claim,
        args.schema,
        args.source_root,
        args.evidence_root,
        expected_claim_sha256=args.expected_claim_sha256,
        expected_auditor_sha256=args.expected_auditor_sha256,
        expected_schema_sha256=args.expected_schema_sha256,
        expected_source_sha256=args.expected_source_sha256,
        trusted_verifier_pins=trusted,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("evidence_chain_valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
