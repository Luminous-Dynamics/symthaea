#!/usr/bin/env python3
"""Independent structured-claim comparison oracle for SCIP realization research.

This oracle deliberately does not parse natural language and cannot establish
surface semantic fidelity. It compares a normalized claim inventory against a
normalized grounded-source claim inventory and reports dimension-specific
Supported / Contradicted / NotApplicable results.

A later text->claim extractor must be independently qualified before this
comparison can contribute to any positive surface-fidelity capability.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

INPUT_SCHEMA = "symthaea.scip-semantic-claim-comparison-input/v1"
RESULT_SCHEMA = "symthaea.scip-semantic-claim-comparison-result/v1"
AUTHORITY = "measurement-only"
DIMENSIONS = (
    "entity-reference",
    "relation-direction",
    "numeric-value-and-unit",
    "polarity-and-negation",
    "quantifier-and-cardinality",
    "temporal-scope",
    "epistemic-modality",
    "attribution-and-source",
    "causal-strength",
    "unsupported-additions",
    "required-detail-coverage",
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
F64 = re.compile(r"^f64:[0-9a-f]{16}$")
DIRECTIONS = {"forward", "reverse"}
POLARITIES = {"positive", "negative"}
EPISTEMIC = {"certain", "probable", "uncertain", "unknown", "out_of_domain"}
CAUSAL = {"none", "associated", "contributory", "causal"}
QUANTIFIERS = {"exact", "at_least", "at_most", "all", "some", "none", "unspecified"}

SOURCE_FIELDS = {
    "claim_id",
    "subject_ref",
    "relation",
    "object_ref",
    "direction",
    "polarity",
    "numeric",
    "quantifier",
    "temporal_scope",
    "epistemic_modality",
    "attribution",
    "causal_strength",
}
SURFACE_FIELDS = {
    "surface_claim_id",
    "source_claim_id",
    "subject_ref",
    "relation",
    "object_ref",
    "direction",
    "polarity",
    "numeric",
    "quantifier",
    "temporal_scope",
    "epistemic_modality",
    "attribution",
    "causal_strength",
}
TOP_FIELDS = {
    "schema",
    "contract_digest",
    "surface_digest",
    "source_claims",
    "surface_claims",
}


class OracleError(ValueError):
    pass


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise OracleError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"), object_pairs_hook=strict_object)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise OracleError(f"invalid input JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise OracleError("top-level input must be an object")
    return value


def require_exact_fields(value: dict[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise OracleError(f"{where} fields mismatch; missing={missing} extra={extra}")


def require_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or any(ord(ch) < 0x20 for ch in value):
        raise OracleError(f"{where} must be non-empty printable text")
    return value


def optional_text(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return require_text(value, where)


def require_digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise OracleError(f"{where} must be 64 lowercase hex characters")
    return value


def validate_numeric(value: Any, where: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise OracleError(f"{where} must be null or an object")
    require_exact_fields(value, {"value", "unit"}, where)
    encoded = value["value"]
    if not isinstance(encoded, str) or not F64.fullmatch(encoded):
        raise OracleError(f"{where}.value must use canonical f64:<16 lowercase hex> encoding")
    if encoded == "f64:8000000000000000":
        raise OracleError(f"{where}.value must canonicalize -0.0 to +0.0")
    unit = require_text(value["unit"], f"{where}.unit")
    return {"value": encoded, "unit": unit}


def validate_quantifier(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OracleError(f"{where} must be an object")
    require_exact_fields(value, {"kind", "value"}, where)
    kind = value["kind"]
    if kind not in QUANTIFIERS:
        raise OracleError(f"{where}.kind is unsupported")
    qvalue = value["value"]
    if kind in {"exact", "at_least", "at_most"}:
        if isinstance(qvalue, bool) or not isinstance(qvalue, int) or qvalue < 0:
            raise OracleError(f"{where}.value must be a non-negative integer for {kind}")
    elif qvalue is not None:
        raise OracleError(f"{where}.value must be null for {kind}")
    return {"kind": kind, "value": qvalue}


def validate_common(claim: dict[str, Any], where: str) -> dict[str, Any]:
    subject_ref = require_text(claim["subject_ref"], f"{where}.subject_ref")
    relation = require_text(claim["relation"], f"{where}.relation")
    object_ref = require_text(claim["object_ref"], f"{where}.object_ref")
    direction = claim["direction"]
    polarity = claim["polarity"]
    epistemic = claim["epistemic_modality"]
    causal = claim["causal_strength"]
    if direction not in DIRECTIONS:
        raise OracleError(f"{where}.direction is unsupported")
    if polarity not in POLARITIES:
        raise OracleError(f"{where}.polarity is unsupported")
    if epistemic not in EPISTEMIC:
        raise OracleError(f"{where}.epistemic_modality is unsupported")
    if causal not in CAUSAL:
        raise OracleError(f"{where}.causal_strength is unsupported")
    return {
        "subject_ref": subject_ref,
        "relation": relation,
        "object_ref": object_ref,
        "direction": direction,
        "polarity": polarity,
        "numeric": validate_numeric(claim["numeric"], f"{where}.numeric"),
        "quantifier": validate_quantifier(claim["quantifier"], f"{where}.quantifier"),
        "temporal_scope": optional_text(claim["temporal_scope"], f"{where}.temporal_scope"),
        "epistemic_modality": epistemic,
        "attribution": optional_text(claim["attribution"], f"{where}.attribution"),
        "causal_strength": causal,
    }


def validate_source_claim(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OracleError(f"source_claims[{index}] must be an object")
    require_exact_fields(value, SOURCE_FIELDS, f"source_claims[{index}]")
    out = {"claim_id": require_text(value["claim_id"], f"source_claims[{index}].claim_id")}
    out.update(validate_common(value, f"source_claims[{index}]"))
    return out


def validate_surface_claim(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OracleError(f"surface_claims[{index}] must be an object")
    require_exact_fields(value, SURFACE_FIELDS, f"surface_claims[{index}]")
    source_claim_id = value["source_claim_id"]
    if source_claim_id is not None:
        source_claim_id = require_text(
            source_claim_id, f"surface_claims[{index}].source_claim_id"
        )
    out = {
        "surface_claim_id": require_text(
            value["surface_claim_id"], f"surface_claims[{index}].surface_claim_id"
        ),
        "source_claim_id": source_claim_id,
    }
    out.update(validate_common(value, f"surface_claims[{index}]"))
    return out


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def verdict(mismatches: int, applicable: bool = True) -> dict[str, Any]:
    return {
        "verdict": "contradicted" if mismatches else ("supported" if applicable else "not_applicable"),
        "mismatch_count": mismatches,
    }


def compare_document(document: dict[str, Any]) -> dict[str, Any]:
    require_exact_fields(document, TOP_FIELDS, "top-level")
    if document["schema"] != INPUT_SCHEMA:
        raise OracleError("unsupported input schema")

    contract_digest = require_digest(document["contract_digest"], "contract_digest")
    surface_digest = require_digest(document["surface_digest"], "surface_digest")

    if not isinstance(document["source_claims"], list) or not document["source_claims"]:
        raise OracleError("source_claims must be a non-empty list")
    if not isinstance(document["surface_claims"], list):
        raise OracleError("surface_claims must be a list")

    source_claims = [
        validate_source_claim(value, i) for i, value in enumerate(document["source_claims"])
    ]
    surface_claims = [
        validate_surface_claim(value, i) for i, value in enumerate(document["surface_claims"])
    ]

    source_ids: set[str] = set()
    for claim in source_claims:
        if claim["claim_id"] in source_ids:
            raise OracleError("duplicate source claim_id")
        source_ids.add(claim["claim_id"])

    surface_ids: set[str] = set()
    mapped: dict[str, dict[str, Any]] = {}
    extras: list[dict[str, Any]] = []
    for claim in surface_claims:
        sid = claim["surface_claim_id"]
        if sid in surface_ids:
            raise OracleError("duplicate surface_claim_id")
        surface_ids.add(sid)
        source_id = claim["source_claim_id"]
        if source_id is None or source_id not in source_ids:
            extras.append(claim)
            continue
        if source_id in mapped:
            raise OracleError("multiple surface claims map to one source claim")
        mapped[source_id] = claim

    canonical_source = sorted(source_claims, key=lambda item: item["claim_id"])
    canonical_surface = sorted(surface_claims, key=lambda item: item["surface_claim_id"])

    mismatch = {dimension: 0 for dimension in DIMENSIONS}
    numeric_applicable = False
    temporal_applicable = False
    attribution_applicable = False

    for source in canonical_source:
        surface = mapped.get(source["claim_id"])
        if surface is None:
            mismatch["required-detail-coverage"] += 1
            continue

        if (
            source["subject_ref"] != surface["subject_ref"]
            or source["object_ref"] != surface["object_ref"]
        ):
            mismatch["entity-reference"] += 1
        if (
            source["relation"] != surface["relation"]
            or source["direction"] != surface["direction"]
        ):
            mismatch["relation-direction"] += 1

        if source["numeric"] is not None or surface["numeric"] is not None:
            numeric_applicable = True
            if source["numeric"] != surface["numeric"]:
                mismatch["numeric-value-and-unit"] += 1

        if source["polarity"] != surface["polarity"]:
            mismatch["polarity-and-negation"] += 1
        if source["quantifier"] != surface["quantifier"]:
            mismatch["quantifier-and-cardinality"] += 1

        if source["temporal_scope"] is not None or surface["temporal_scope"] is not None:
            temporal_applicable = True
            if source["temporal_scope"] != surface["temporal_scope"]:
                mismatch["temporal-scope"] += 1

        if source["epistemic_modality"] != surface["epistemic_modality"]:
            mismatch["epistemic-modality"] += 1

        if source["attribution"] is not None or surface["attribution"] is not None:
            attribution_applicable = True
            if source["attribution"] != surface["attribution"]:
                mismatch["attribution-and-source"] += 1

        if source["causal_strength"] != surface["causal_strength"]:
            mismatch["causal-strength"] += 1

    mismatch["unsupported-additions"] = len(extras)

    dimension_rows = []
    for dimension in DIMENSIONS:
        applicable = True
        if dimension == "numeric-value-and-unit":
            applicable = numeric_applicable
        elif dimension == "temporal-scope":
            applicable = temporal_applicable
        elif dimension == "attribution-and-source":
            applicable = attribution_applicable
        row = {"dimension": dimension, **verdict(mismatch[dimension], applicable)}
        dimension_rows.append(row)

    inventory_equivalent = all(row["verdict"] != "contradicted" for row in dimension_rows)
    result_preimage = {
        "schema": RESULT_SCHEMA,
        "authority": AUTHORITY,
        "contract_digest": contract_digest,
        "surface_digest": surface_digest,
        "source_inventory_sha256": sha256_hex(canonical_source),
        "surface_inventory_sha256": sha256_hex(canonical_surface),
        "source_claim_count": len(canonical_source),
        "surface_claim_count": len(canonical_surface),
        "dimensions": dimension_rows,
        "claim_inventory_equivalent": inventory_equivalent,
        "surface_fidelity_established": False,
        "text_to_claim_extraction_qualified": False,
    }
    result = copy.deepcopy(result_preimage)
    result["comparison_sha256"] = sha256_hex(result_preimage)
    return result


def self_test() -> None:
    source = {
        "claim_id": "claim-1",
        "subject_ref": "sensor:S17",
        "relation": "reports-temperature",
        "object_ref": "reactor:R1",
        "direction": "forward",
        "polarity": "positive",
        "numeric": {"value": "f64:4073b00000000000", "unit": "K"},
        "quantifier": {"kind": "exact", "value": 1},
        "temporal_scope": "window:2026-09-20T10:00Z/2026-09-20T10:05Z",
        "epistemic_modality": "probable",
        "attribution": "observation:17",
        "causal_strength": "none",
    }
    surface = dict(source)
    surface.pop("claim_id")
    surface["surface_claim_id"] = "surface-1"
    surface["source_claim_id"] = "claim-1"
    document = {
        "schema": INPUT_SCHEMA,
        "contract_digest": "1" * 64,
        "surface_digest": "2" * 64,
        "source_claims": [source],
        "surface_claims": [surface],
    }
    result = compare_document(document)
    assert result["claim_inventory_equivalent"] is True
    assert result["surface_fidelity_established"] is False
    assert result["text_to_claim_extraction_qualified"] is False

    mutated = copy.deepcopy(document)
    mutated["surface_claims"][0]["polarity"] = "negative"
    result = compare_document(mutated)
    row = next(row for row in result["dimensions"] if row["dimension"] == "polarity-and-negation")
    assert row["verdict"] == "contradicted"
    assert result["claim_inventory_equivalent"] is False

    print("PASS_SELF_TEST")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if not args.input:
            parser.error("input JSON is required unless --self-test is used")
        result = compare_document(load_json(Path(args.input)))
        sys.stdout.buffer.write(canonical_bytes(result) + b"\n")
        return 0
    except OracleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
