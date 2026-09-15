#!/usr/bin/env python3
"""Static verifier for SPINE-000D causal-ablation preregistration."""

import json
from pathlib import Path

CONTRACT = Path("docs/research/SPINE_000D_CAUSAL_ABLATION_CONTRACT.md")
SCHEMA = Path("docs/research/SPINE_000D_CAMPAIGN_MANIFEST.schema.json")


def fail(msg: str) -> None:
    raise SystemExit(f"SPINE-000D CONTRACT FAIL: {msg}")


def main() -> int:
    contract = CONTRACT.read_text(encoding="utf-8")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail("unexpected JSON Schema dialect")
    if schema.get("additionalProperties") is not False:
        fail("campaign manifest must fail closed on unknown top-level fields")

    required = set(schema.get("required", []))
    for field in (
        "subject",
        "target_subsystem",
        "intervention_arms",
        "primary_observables",
        "safety_hard_gates",
        "multiplicity_policy",
        "confidence_interval_policy",
        "interaction_followup_rule",
        "stopping_rule",
        "classification_scope_fields",
    ):
        if field not in required:
            fail(f"manifest no longer requires {field}")

    arm_items = schema["properties"]["intervention_arms"]["items"]["enum"]
    for arm in ("FULL", "OUTPUT_SHAM", "DISABLED", "STATE_FROZEN"):
        if arm not in arm_items:
            fail(f"missing intervention arm {arm}")

    scope_const = schema["properties"]["classification_scope_fields"].get("const")
    if scope_const != [
        "subject",
        "workload",
        "observable",
        "intervention",
        "comparison",
        "evidence_lineage",
    ]:
        fail("classification scope tuple changed")

    for phrase in (
        "No global load-bearing label",
        "Null claims require equivalence evidence",
        "Difference is not benefit",
        "Redundancy and synergy require interaction follow-up",
        "Legacy inline paths can mask manager ablations",
        "Safety and authority gates remain separate",
        "LOAD_BEARING_POSITIVE",
        "EQUIVALENT_WITHIN_BOUND",
        "MASKED_BY_OVERLAP",
        "INTERACTION_SUSPECTED",
    ):
        if phrase not in contract:
            fail(f"contract missing required concept: {phrase}")

    print("SPINE-000D contract/schema verifier: PASS")
    print("authority=measurement-only")
    print("causal_results_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
