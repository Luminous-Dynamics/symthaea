#!/usr/bin/env python3
"""Verify BIO-MILLENNIUM-001C source-bound criterion facets."""
from __future__ import annotations
import json
from pathlib import Path

MANIFEST = Path("docs/research/biology/millennium_problems_source_manifest_v1.json")
CONTRACT = Path("docs/research/biology/millennium_problems_source_criterion_contract_v1.json")

def fail(message: str) -> None:
    raise SystemExit(f"BIO-MILLENNIUM-001C FAIL: {message}")

def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    if contract.get("schema") != "symthaea.bio-millennium.source-criterion-contract.v1":
        fail("unexpected contract schema")
    if contract.get("contract_generation") != 1:
        fail("unexpected contract generation")

    source = contract["source_binding"]
    if source["source_is_living"] is not True:
        fail("living-source status must remain explicit")
    if source["source_snapshot_vendored"] is not False:
        fail("source snapshot vendoring must remain false")
    if source["retrieved_date"] != "2026-09-27":
        fail("retrieval date drifted")

    authority = contract["authority"]
    if authority["official_source_is_external"] is not True:
        fail("official source must remain external")
    if authority["local_paraphrase_is_official_source"] is not False:
        fail("local paraphrase cannot become official source")
    if authority["scientific_result_claimed"] or authority["challenge_solved_claimed"]:
        fail("criterion contract cannot claim scientific results")
    if authority["experimental_authority_granted"]:
        fail("criterion contract cannot grant experimental authority")

    expected = [(x["id"], x["ordinal"], x["title"]) for x in manifest["challenges"]]
    actual = [(x["id"], x["ordinal"], x["title"]) for x in contract["criteria"]]
    if expected != actual:
        fail("criterion IDs/order/titles diverge from source manifest")

    if len(contract["criteria"]) != 12:
        fail("criterion count is not 12")

    for criterion in contract["criteria"]:
        if not criterion.get("facets"):
            fail(f"{criterion['id']} has no criterion facets")
        if not criterion.get("evidence_modes"):
            fail(f"{criterion['id']} has no declared evidence mode")
        if criterion["id"] in {"MPB-2026-09-23-03", "MPB-2026-09-23-08", "MPB-2026-09-23-09"}:
            if criterion.get("prospective_constraints") is not True:
                fail(f"{criterion['id']} must preserve its prospective nature")
        if criterion["id"] == "MPB-2026-09-23-06":
            if "IndependentReplication" not in criterion["evidence_modes"]:
                fail("limb-regeneration profile must preserve replication evidence")

    rules = set(contract["global_source_rules"])
    required = {
        "A later source revision requires a new contract generation and must not silently rescore prior results.",
        "A computational prediction cannot satisfy an experimental criterion unless a future source revision explicitly changes that criterion.",
        "Partial-success language from the source must remain distinct from completion."
    }
    if not required.issubset(rules):
        fail("missing source-custody rule(s)")

    print("BIO-MILLENNIUM-001C PASS: source-bound criterion facets are internally consistent")
    print("source_is_living=true")
    print("local_paraphrase_is_official_source=false")
    print("experimental_authority_granted=false")
    print("criteria=12")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
