#!/usr/bin/env python3
"""External adversarial harness for scip_semantic_claim_oracle.py."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "scripts" / "scip_semantic_claim_oracle.py"
FIXTURE = ROOT / "tests" / "fixtures" / "scip" / "semantic_claim_inventory_v1.json"


def run_document(document: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "input.json"
        path.write_text(json.dumps(document, ensure_ascii=False), "utf-8")
        proc = subprocess.run(
            [sys.executable, str(ORACLE), str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(f"oracle rejected valid mutation: {proc.stderr}")
        return json.loads(proc.stdout)


def run_invalid_document(document: dict) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "input.json"
        path.write_text(json.dumps(document), "utf-8")
        proc = subprocess.run(
            [sys.executable, str(ORACLE), str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            raise AssertionError("oracle accepted invalid input")


def run_invalid_raw(raw: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "input.json"
        path.write_text(raw, "utf-8")
        proc = subprocess.run(
            [sys.executable, str(ORACLE), str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            raise AssertionError("oracle accepted invalid raw input")


def verdict(result: dict, dimension: str) -> str:
    for row in result["dimensions"]:
        if row["dimension"] == dimension:
            return row["verdict"]
    raise AssertionError(f"missing dimension {dimension}")


def main() -> int:
    fixture = json.loads(FIXTURE.read_text("utf-8"))
    positive = run_document(fixture)
    assert positive["authority"] == "measurement-only"
    assert positive["claim_inventory_equivalent"] is True
    assert positive["surface_fidelity_established"] is False
    assert positive["text_to_claim_extraction_qualified"] is False
    positive_digest = positive["comparison_sha256"]

    reversed_fixture = copy.deepcopy(fixture)
    reversed_fixture["source_claims"].reverse()
    reversed_fixture["surface_claims"].reverse()
    assert run_document(reversed_fixture)["comparison_sha256"] == positive_digest

    mutations = []

    def mutate(dimension, fn):
        doc = copy.deepcopy(fixture)
        fn(doc)
        mutations.append((dimension, doc))

    mutate("entity-reference", lambda d: d["surface_claims"][0].__setitem__("subject_ref", "sensor:S18"))
    mutate("relation-direction", lambda d: d["surface_claims"][0].__setitem__("direction", "reverse"))
    mutate(
        "numeric-value-and-unit",
        lambda d: d["surface_claims"][0]["numeric"].__setitem__("value", "f64:4073c00000000000"),
    )
    mutate("polarity-and-negation", lambda d: d["surface_claims"][0].__setitem__("polarity", "negative"))
    mutate(
        "quantifier-and-cardinality",
        lambda d: d["surface_claims"][0].__setitem__("quantifier", {"kind": "exact", "value": 2}),
    )
    mutate(
        "temporal-scope",
        lambda d: d["surface_claims"][0].__setitem__(
            "temporal_scope", "window:2026-09-20T11:00Z/2026-09-20T11:05Z"
        ),
    )
    mutate(
        "epistemic-modality",
        lambda d: d["surface_claims"][0].__setitem__("epistemic_modality", "certain"),
    )
    mutate(
        "attribution-and-source",
        lambda d: d["surface_claims"][0].__setitem__("attribution", "observation:999"),
    )
    mutate(
        "causal-strength",
        lambda d: d["surface_claims"][0].__setitem__("causal_strength", "causal"),
    )

    def add_unsupported(d):
        extra = copy.deepcopy(d["surface_claims"][0])
        extra["surface_claim_id"] = "surface-extra"
        extra["source_claim_id"] = None
        d["surface_claims"].append(extra)

    mutate("unsupported-additions", add_unsupported)
    mutate("required-detail-coverage", lambda d: d["surface_claims"].pop(0))

    for dimension, document in mutations:
        result = run_document(document)
        assert verdict(result, dimension) == "contradicted", dimension
        assert result["claim_inventory_equivalent"] is False
        assert result["surface_fidelity_established"] is False

    invalid = copy.deepcopy(fixture)
    invalid["unexpected_authority"] = True
    run_invalid_document(invalid)

    duplicate_mapping = copy.deepcopy(fixture)
    duplicate = copy.deepcopy(duplicate_mapping["surface_claims"][0])
    duplicate["surface_claim_id"] = "surface-duplicate"
    duplicate_mapping["surface_claims"].append(duplicate)
    run_invalid_document(duplicate_mapping)

    negative_zero = copy.deepcopy(fixture)
    negative_zero["surface_claims"][0]["numeric"]["value"] = "f64:8000000000000000"
    run_invalid_document(negative_zero)

    raw = FIXTURE.read_text("utf-8")
    run_invalid_raw(raw.replace('"schema"', '"schema": "shadow", "schema"', 1))

    print(f"PASS_ADVERSARIAL comparison_sha256={positive_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
