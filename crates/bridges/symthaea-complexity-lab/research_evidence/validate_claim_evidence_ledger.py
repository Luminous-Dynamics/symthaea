#!/usr/bin/env python3
"""Validate the versioned Symthaea complexity claim-to-theorem evidence ledger."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCHEMA = "symthaea.complexity-claim-evidence-ledger/v1"
STATUSES = {
    "lean_formalization_scope_documented",
    "manuscript_only",
    "experiment_only",
    "independently_reproduced",
    "externally_reviewed",
}
SOURCE_KINDS = {
    "repository_revision",
    "lean_scope_note",
    "lean_source",
    "comparator_manifest",
    "manuscript_metadata",
    "manuscript_pdf",
    "unrefereed_preprint",
}


class LedgerError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LedgerError(message)


def validate_ledger(data: dict) -> dict:
    require(isinstance(data, dict), "ledger root must be an object")
    require(data.get("schema") == SCHEMA, "unsupported schema version")
    require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", data.get("audit_date", "")) is not None,
            "audit_date must be YYYY-MM-DD")
    sources = data.get("sources")
    claims = data.get("claims")
    require(isinstance(sources, list) and sources, "sources must be a nonempty list")
    require(isinstance(claims, list) and claims, "claims must be a nonempty list")

    source_by_id = {}
    for source in sources:
        require(isinstance(source, dict), "source entry must be an object")
        source_id = source.get("source_id")
        require(isinstance(source_id, str) and source_id, "source_id missing")
        require(source_id not in source_by_id, f"duplicate source_id: {source_id}")
        require(source.get("kind") in SOURCE_KINDS, f"unknown source kind: {source_id}")
        url = source.get("immutable_url")
        require(isinstance(url, str) and url.startswith("https://"),
                f"source URL missing/invalid: {source_id}")
        sha = source.get("blob_sha")
        if sha is not None:
            require(re.fullmatch(r"[0-9a-f]{40}", sha) is not None,
                    f"invalid git blob SHA: {source_id}")
        revision = source.get("revision")
        if revision is not None:
            require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
                    f"invalid git revision: {source_id}")
        require(sha is not None or revision is not None or source.get("kind") == "unrefereed_preprint",
                f"source is not immutable/pinned: {source_id}")
        source_by_id[source_id] = source

    claim_ids = set()
    for claim in claims:
        require(isinstance(claim, dict), "claim entry must be an object")
        claim_id = claim.get("claim_id")
        require(isinstance(claim_id, str) and claim_id, "claim_id missing")
        require(claim_id not in claim_ids, f"duplicate claim_id: {claim_id}")
        claim_ids.add(claim_id)

        status = claim.get("evidence_status")
        require(status in STATUSES, f"unknown evidence_status: {claim_id}")
        require(isinstance(claim.get("bound"), str) and claim["bound"],
                f"bound expression missing: {claim_id}")
        require(isinstance(claim.get("input_family"), str) and claim["input_family"],
                f"input family missing: {claim_id}")
        require(isinstance(claim.get("output_contract"), list) and claim["output_contract"],
                f"output contract missing: {claim_id}")
        require(isinstance(claim.get("not_established_here"), list),
                f"claim boundary missing: {claim_id}")

        all_refs = []
        for key in ("formal_source_ids", "comparator_manifest_source_ids",
                    "manuscript_source_ids"):
            refs = claim.get(key, [])
            require(isinstance(refs, list), f"{key} must be a list: {claim_id}")
            for ref in refs:
                require(ref in source_by_id, f"unknown source reference {ref}: {claim_id}")
                all_refs.append(ref)

        if status == "lean_formalization_scope_documented":
            require(isinstance(claim.get("formal_theorem"), str) and claim["formal_theorem"],
                    f"formal theorem missing: {claim_id}")
            formal_refs = claim.get("formal_source_ids", [])
            comparator_refs = claim.get("comparator_manifest_source_ids", [])
            require(formal_refs, f"formal source missing: {claim_id}")
            require(any(source_by_id[ref]["kind"] == "lean_source" for ref in formal_refs),
                    f"Lean source artifact missing: {claim_id}")
            require(any(source_by_id[ref]["kind"] == "lean_scope_note" for ref in formal_refs),
                    f"Lean scope note missing: {claim_id}")
            require(comparator_refs and any(
                source_by_id[ref]["kind"] == "comparator_manifest" for ref in comparator_refs
            ), f"comparator manifest missing: {claim_id}")
            axioms = claim.get("permitted_axioms")
            require(isinstance(axioms, list) and axioms,
                    f"permitted axioms must be explicit: {claim_id}")
            require(claim.get("reproduction_status") == "not_independently_rebuilt_by_symthaea"
                    or claim.get("reproduction_status") == "independently_rebuilt_by_symthaea",
                    f"formal claim must state local rebuild status: {claim_id}")
        elif status == "manuscript_only":
            require(claim.get("formal_theorem") is None,
                    f"manuscript-only claim must not carry a Lean theorem: {claim_id}")
            require(not claim.get("formal_source_ids"),
                    f"manuscript-only claim must not cite formal theorem evidence as its proof: {claim_id}")
            require(claim.get("manuscript_source_ids"),
                    f"manuscript-only claim must cite a manuscript: {claim_id}")
            require(claim.get("peer_review_status") is not None
                    or claim.get("external_review_status") is not None,
                    f"manuscript review status is not explicit: {claim_id}")
        elif status in {"independently_reproduced", "externally_reviewed"}:
            require(claim.get("reproduction_status") or claim.get("external_review_status"),
                    f"independent evidence status needs a description: {claim_id}")

        require(claim.get("last_audited") == data["audit_date"],
                f"claim audit date differs from ledger date: {claim_id}")

    return {
        "schema": SCHEMA,
        "sources": len(sources),
        "claims": len(claims),
        "status_counts": {
            status: sum(1 for claim in claims if claim["evidence_status"] == status)
            for status in sorted(STATUSES)
        },
        "all_passed": True,
    }


def classify_claim(claim: dict, requested_evidence: str) -> bool:
    """Return whether a requested wording/evidence tier is supported by this record."""
    actual = claim["evidence_status"]
    if requested_evidence == "lean_formalized":
        return actual == "lean_formalization_scope_documented"
    if requested_evidence == "manuscript_stated":
        return actual in {"lean_formalization_scope_documented", "manuscript_only"}
    if requested_evidence == "independently_reproduced":
        return actual == "independently_reproduced"
    if requested_evidence == "externally_reviewed":
        return actual == "externally_reviewed"
    return False


def mutation_tests(data: dict) -> dict:
    import copy

    tests = []

    bad = copy.deepcopy(data)
    bad["claims"][1]["evidence_status"] = "lean_formalization_scope_documented"
    tests.append(("manuscript exponent promoted to Lean", bad))

    bad = copy.deepcopy(data)
    bad["claims"][0]["formal_theorem"] = None
    tests.append(("formal theorem removed", bad))

    bad = copy.deepcopy(data)
    bad["claims"][0]["permitted_axioms"] = []
    tests.append(("axiom manifest omitted", bad))

    bad = copy.deepcopy(data)
    bad["claims"][0]["formal_source_ids"] = ["weighted-counting-preprint"]
    tests.append(("preprint relabelled as Lean source", bad))

    bad = copy.deepcopy(data)
    bad["claims"][2]["output_contract"] = []
    tests.append(("output contract omitted", bad))

    bad = copy.deepcopy(data)
    bad["claims"][3]["peer_review_status"] = None
    bad["claims"][3]["external_review_status"] = None
    tests.append(("preprint review status omitted", bad))

    rejected = 0
    for name, candidate in tests:
        try:
            validate_ledger(candidate)
        except (LedgerError, TypeError, KeyError):
            rejected += 1
        else:
            raise AssertionError(f"mutation accepted: {name}")

    return {
        "mutation_tests": len(tests),
        "mutations_rejected": rejected,
        "all_passed": rejected == len(tests),
    }


def main() -> int:
    ledger_path = Path(__file__).with_name("CLAIM_EVIDENCE_LEDGER.json")
    data = json.loads(ledger_path.read_text(encoding="utf-8"))
    result = validate_ledger(data)
    result.update(mutation_tests(data))
    formal = next(c for c in data["claims"] if c["claim_id"] == "ETH-CLAIM-001")
    general = next(c for c in data["claims"] if c["claim_id"] == "ETH-CLAIM-002")
    result["claim_promotion_controls"] = {
        "formal_claim_accepted_as_lean": classify_claim(formal, "lean_formalized"),
        "manuscript_claim_rejected_as_lean": not classify_claim(general, "lean_formalized"),
        "manuscript_claim_accepted_as_manuscript_stated": classify_claim(general, "manuscript_stated"),
    }
    result["all_passed"] = result["all_passed"] and result["mutations_rejected"] == result["mutation_tests"]
    output = Path(__file__).with_name("claim_evidence_ledger_results.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
