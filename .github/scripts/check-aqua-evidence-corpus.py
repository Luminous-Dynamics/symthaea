#!/usr/bin/env python3
"""Validate the versioned Luminous Aqua evidence corpus.

This is a CI-only documentation/data validator. It does not calculate product performance,
qualify instruments, or authorize standards/compliance claims.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "docs/research/aqua-evidence-corpus-v0.1.schema.json"
CORPUS_PATH = ROOT / "docs/research/aqua-evidence-corpus-v0.1.json"

ALLOWED_SOURCE_TYPES = {
    "manufacturer_declaration": {"manufacturer_datasheet"},
    "independent_laboratory_measurement": {"independent_market_study"},
    "standard_or_programme_criterion": {
        "official_standard_or_programme",
        "industry_association_notice",
    },
    "installation_design_reference": {"official_standard_or_programme"},
    "peer_reviewed_study_result": {"peer_reviewed_article"},
    "independent_market_sample": {"independent_market_study"},
}


def fail(message: str) -> None:
    print(f"AQUA-EVIDENCE-CORPUS: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot load {path.relative_to(ROOT)}: {exc}")
    if not isinstance(value, dict):
        fail(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def pointer(path_parts: object) -> str:
    return "/" + "/".join(str(part) for part in path_parts)


def main() -> None:
    schema = load_json(SCHEMA_PATH)
    corpus = load_json(CORPUS_PATH)

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(corpus),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        for error in errors[:50]:
            fail(f"{pointer(error.absolute_path)}: {error.message}")
        fail(f"{len(errors)} schema errors; first 50 shown")

    entry_ids: set[str] = set()
    claim_ids: set[str] = set()
    claim_count = 0

    for entry_index, entry in enumerate(corpus["entries"]):
        entry_id = entry["entry_id"]
        if entry_id in entry_ids:
            fail(f"/entries/{entry_index}/entry_id duplicates {entry_id}")
        entry_ids.add(entry_id)

        subject = entry["subject"]
        if subject["type"] == "product":
            for field in ("manufacturer", "sku", "market"):
                if not subject[field]:
                    fail(
                        f"/entries/{entry_index}/subject/{field}: "
                        "product identity must be exact"
                    )

        for claim_index, claim in enumerate(entry["claims"]):
            claim_count += 1
            claim_id = claim["claim_id"]
            location = f"/entries/{entry_index}/claims/{claim_index}"
            if claim_id in claim_ids:
                fail(f"{location}/claim_id duplicates {claim_id}")
            claim_ids.add(claim_id)

            source_type = claim["source"]["source_type"]
            allowed = ALLOWED_SOURCE_TYPES[claim["evidence_class"]]
            if source_type not in allowed:
                fail(
                    f"{location}: evidence class {claim['evidence_class']} "
                    f"cannot be supported by source type {source_type}"
                )

            if claim["result_class"] == "derived" and "derivation" not in claim:
                fail(f"{location}: derived result is missing its derivation record")

            if (
                claim["parameter"] == "flow_rate"
                and claim["conditions"]["pressure_kpa"] is None
                and "pressure" not in " ".join(claim["conditions"]["unknown_conditions"]).lower()
                and claim["conditions"]["pressure_range_kpa"] is None
                and not claim["conditions"]["pressure_points_kpa"]
            ):
                fail(
                    f"{location}: flow-rate claim must state its pressure condition "
                    "or explicitly list pressure as unknown"
                )

            if claim["evidence_class"] == "manufacturer_declaration":
                if subject["type"] != "product":
                    fail(
                        f"{location}: manufacturer declaration must attach to an exact product"
                    )

            if claim["evidence_class"] == "peer_reviewed_study_result":
                if subject["type"] != "research_study":
                    fail(
                        f"{location}: peer-reviewed result must attach to a research-study entry"
                    )

    if not entry_ids or not claim_ids:
        fail("corpus contains no entries or claims")

    print(
        "AQUA-EVIDENCE-CORPUS: PASS "
        f"schema={corpus['schema_version']} "
        f"snapshot={corpus['snapshot_date']} "
        f"entries={len(entry_ids)} claims={claim_count}"
    )


if __name__ == "__main__":
    main()
