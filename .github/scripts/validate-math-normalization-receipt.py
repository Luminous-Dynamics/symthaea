#!/usr/bin/env python3
"""Semantic validator for MATH-REP-001A normalization receipts.

Dependency-free by design. JSON Schema closes the wire shape; this validator
adds cross-field semantic rules that must remain true even if a future schema
implementation is permissive.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ALLOWED_TOP_LEVEL = {
    "version",
    "receipt_id",
    "normalizer_id",
    "authority",
    "source_digest",
    "source_fragment",
    "source_provenance_refs",
    "source_preserved",
    "numeric_domain",
    "outcome",
    "normal_form_digest",
    "canonical_serialization",
    "transformations",
    "validation_state",
    "side_conditions",
    "validation_evidence_refs",
    "rejection_reason",
}

ALLOWED_TRANSFORMS = {
    "AlphaNormalizeVariables",
    "ReduceRationalCoefficients",
    "FlattenAssociativeAdd",
    "FlattenAssociativeMul",
    "SortCommutativeTerms",
    "CollectLikeTerms",
    "ExpandNonnegativeIntegerPower",
    "DistributeMultiplication",
    "NormalizeSigns",
    "CancelNonzeroFactor",
}

ALLOWED_VALIDATION = {
    "NotApplicable",
    "StructuralRewriteChecked",
    "ExactPolynomialCanonicalization",
    "SolverCrossChecked",
    "FormalEquivalenceReceipt",
}

ALLOWED_OUTCOMES = {"Normalized", "Unsupported", "Rejected"}
ALLOWED_DOMAINS = {"Int", "Nat", "Real"}
ALLOWED_FRAGMENTS = {
    "ExactPolynomialTerm",
    "ExactPolynomialEquality",
    "UnsupportedFragment",
}
ALLOWED_REJECTIONS = {
    "VariableDenominator",
    "NonPolynomialDivision",
    "NegativeExponent",
    "FractionalExponent",
    "Transcendental",
    "OrderSensitiveRewrite",
    "MissingSideCondition",
    "DomainAmbiguity",
    "UnsupportedOperator",
    "ResourceLimit",
    "OtherUnsupported",
}
ALLOWED_SIDE_KINDS = {
    "NonZero",
    "NonNegative",
    "Positive",
    "Negative",
    "DomainConstraint",
    "Other",
}
ALLOWED_SIDE_STATUS = {"Assumed", "Derived", "FormallyVerified"}

FORBIDDEN_AUTHORITY_FIELDS = {
    "truth_value",
    "formal_authority",
    "theorem_authority",
    "epistemic_confidence",
    "proof_valid",
}


class ValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def require_sha256(value: object, field: str) -> None:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{field} must be lowercase SHA-256 hex")


def require_string_list(value: object, field: str, *, nonempty: bool = False) -> list[str]:
    require(isinstance(value, list), f"{field} must be an array")
    require(all(isinstance(item, str) and item for item in value),
            f"{field} entries must be non-empty strings")
    require(len(value) == len(set(value)), f"{field} entries must be unique")
    if nonempty:
        require(bool(value), f"{field} must not be empty")
    return value


def validate_side_condition(condition: object, index: int) -> None:
    require(isinstance(condition, dict), f"side_conditions[{index}] must be an object")
    allowed = {"kind", "expression_digest", "status", "evidence_refs"}
    unknown = set(condition) - allowed
    require(not unknown, f"side_conditions[{index}] has unknown fields: {sorted(unknown)}")
    require(condition.get("kind") in ALLOWED_SIDE_KINDS,
            f"side_conditions[{index}].kind invalid")
    require_sha256(condition.get("expression_digest"),
                   f"side_conditions[{index}].expression_digest")
    status = condition.get("status")
    require(status in ALLOWED_SIDE_STATUS, f"side_conditions[{index}].status invalid")
    evidence = require_string_list(condition.get("evidence_refs", []),
                                   f"side_conditions[{index}].evidence_refs")
    if status == "FormallyVerified":
        require(evidence, f"side_conditions[{index}] formally verified but has no evidence refs")


def validate_receipt(receipt: object) -> None:
    require(isinstance(receipt, dict), "receipt must be an object")

    unknown = set(receipt) - ALLOWED_TOP_LEVEL
    require(not unknown, f"unknown top-level fields: {sorted(unknown)}")
    forbidden = set(receipt) & FORBIDDEN_AUTHORITY_FIELDS
    require(not forbidden, f"normalization receipt may not carry theorem/truth authority: {sorted(forbidden)}")

    required = {
        "version", "receipt_id", "normalizer_id", "authority", "source_digest",
        "source_fragment", "source_provenance_refs", "source_preserved",
        "numeric_domain", "outcome", "validation_state", "side_conditions",
        "validation_evidence_refs",
    }
    missing = required - set(receipt)
    require(not missing, f"missing fields: {sorted(missing)}")

    require(receipt["version"] == "math-normalization-receipt-v1", "unexpected version")
    require(isinstance(receipt["receipt_id"], str) and receipt["receipt_id"], "receipt_id required")
    require(isinstance(receipt["normalizer_id"], str) and receipt["normalizer_id"], "normalizer_id required")
    require(receipt["authority"] == "RetrievalOnly", "authority must remain RetrievalOnly")
    require(receipt["source_preserved"] is True, "source_preserved must be true")
    require_sha256(receipt["source_digest"], "source_digest")
    require(receipt["source_fragment"] in ALLOWED_FRAGMENTS, "invalid source_fragment")
    require(receipt["numeric_domain"] in ALLOWED_DOMAINS, "invalid numeric_domain")
    require(receipt["outcome"] in ALLOWED_OUTCOMES, "invalid outcome")
    require(receipt["validation_state"] in ALLOWED_VALIDATION, "invalid validation_state")
    require_string_list(receipt["source_provenance_refs"], "source_provenance_refs", nonempty=True)
    validation_refs = require_string_list(receipt["validation_evidence_refs"],
                                          "validation_evidence_refs")

    side_conditions = receipt["side_conditions"]
    require(isinstance(side_conditions, list), "side_conditions must be an array")
    for index, condition in enumerate(side_conditions):
        validate_side_condition(condition, index)

    outcome = receipt["outcome"]
    if outcome == "Normalized":
        require(receipt["source_fragment"] != "UnsupportedFragment",
                "UnsupportedFragment cannot have Normalized outcome")
        for field in ("normal_form_digest", "canonical_serialization", "transformations"):
            require(field in receipt, f"Normalized receipt requires {field}")
        require_sha256(receipt["normal_form_digest"], "normal_form_digest")
        require(isinstance(receipt["canonical_serialization"], str)
                and receipt["canonical_serialization"],
                "canonical_serialization must be non-empty")
        transforms = receipt["transformations"]
        require(isinstance(transforms, list), "transformations must be an array")
        require(len(transforms) == len(set(transforms)), "transformations must be unique")
        require(all(item in ALLOWED_TRANSFORMS for item in transforms),
                "unknown transformation")
        require("rejection_reason" not in receipt,
                "Normalized receipt cannot have rejection_reason")
        require(receipt["validation_state"] != "NotApplicable",
                "Normalized receipt requires an applicable validation state")
    else:
        require(receipt["validation_state"] == "NotApplicable",
                "Unsupported/Rejected receipt validation_state must be NotApplicable")
        require(receipt.get("rejection_reason") in ALLOWED_REJECTIONS,
                "Unsupported/Rejected receipt requires valid rejection_reason")
        for field in ("normal_form_digest", "canonical_serialization", "transformations"):
            require(field not in receipt,
                    f"Unsupported/Rejected receipt must not carry {field}")

    if receipt["validation_state"] in {"SolverCrossChecked", "FormalEquivalenceReceipt"}:
        require(validation_refs,
                f"{receipt['validation_state']} requires validation_evidence_refs")

    if "CancelNonzeroFactor" in receipt.get("transformations", []):
        require(any(item.get("kind") == "NonZero" for item in side_conditions),
                "CancelNonzeroFactor requires an explicit NonZero side condition")

    for condition in side_conditions:
        if condition["status"] == "FormallyVerified":
            require(condition.get("evidence_refs"),
                    "FormallyVerified side condition requires evidence refs")


def _sha(ch: str) -> str:
    return ch * 64


def _valid_receipt() -> dict:
    return {
        "version": "math-normalization-receipt-v1",
        "receipt_id": "receipt-1",
        "normalizer_id": "exact-polynomial-normalizer-v1",
        "authority": "RetrievalOnly",
        "source_digest": _sha("a"),
        "source_fragment": "ExactPolynomialEquality",
        "source_provenance_refs": ["research-graph:event:1"],
        "source_preserved": True,
        "numeric_domain": "Real",
        "outcome": "Normalized",
        "normal_form_digest": _sha("b"),
        "canonical_serialization": "domain=Real;poly=x^2-1=0",
        "transformations": ["AlphaNormalizeVariables", "SortCommutativeTerms"],
        "validation_state": "ExactPolynomialCanonicalization",
        "side_conditions": [],
        "validation_evidence_refs": [],
    }


def self_test() -> None:
    valid = _valid_receipt()
    validate_receipt(valid)

    cases: list[tuple[str, dict]] = []

    case = dict(valid)
    case["truth_value"] = "True"
    cases.append(("truth authority smuggling", case))

    case = dict(valid)
    case["source_preserved"] = False
    cases.append(("source deletion", case))

    case = dict(valid)
    case["transformations"] = ["CancelNonzeroFactor"]
    cases.append(("unsafe cancellation", case))

    case = dict(valid)
    case["validation_state"] = "FormalEquivalenceReceipt"
    case["validation_evidence_refs"] = []
    cases.append(("formal receipt without evidence", case))

    case = dict(valid)
    case["outcome"] = "Unsupported"
    case["validation_state"] = "NotApplicable"
    case["rejection_reason"] = "VariableDenominator"
    cases.append(("unsupported carrying normal form", case))

    for label, bad in cases:
        try:
            validate_receipt(bad)
        except ValidationError:
            continue
        raise AssertionError(f"self-test failed to reject: {label}")

    safe_cancel = _valid_receipt()
    safe_cancel["transformations"] = ["CancelNonzeroFactor"]
    safe_cancel["side_conditions"] = [{
        "kind": "NonZero",
        "expression_digest": _sha("c"),
        "status": "Assumed",
    }]
    validate_receipt(safe_cancel)

    unsupported = {
        "version": "math-normalization-receipt-v1",
        "receipt_id": "unsupported-1",
        "normalizer_id": "exact-polynomial-normalizer-v1",
        "authority": "RetrievalOnly",
        "source_digest": _sha("d"),
        "source_fragment": "UnsupportedFragment",
        "source_provenance_refs": ["research-graph:event:2"],
        "source_preserved": True,
        "numeric_domain": "Real",
        "outcome": "Unsupported",
        "validation_state": "NotApplicable",
        "side_conditions": [],
        "validation_evidence_refs": [],
        "rejection_reason": "VariableDenominator",
    }
    validate_receipt(unsupported)


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        print("math-normalization-receipt semantic self-tests: PASS")
        return 0

    if len(argv) != 2:
        print(f"usage: {argv[0]} RECEIPT.json | --self-test", file=sys.stderr)
        return 2

    path = Path(argv[1])
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        validate_receipt(receipt)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
