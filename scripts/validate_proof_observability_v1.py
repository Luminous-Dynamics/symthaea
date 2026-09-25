#!/usr/bin/env python3
"""Validate Proof Observability V1 and hostile semantic mutations."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs/formal/proof_observability_v1.json"
CAPSULE_PATH = ROOT / "docs/formal/proof_review_capsule_schema_v1.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_NONCLAIMS = {
    "proof observability != theorem truth",
    "no detected vacuity != complete specification",
    "used-by-proof != semantically intended",
    "coverage report != source refinement",
    "Exact coverage != compiler/native-binary correctness",
    "Unknown observability != evidence of defect",
}

REQUIRED_FIELDS = {
    "id", "capsule_id", "proof_goal", "subject_sha256", "checker_receipt",
    "proof_checked", "assumptions_in_scope", "assumptions_used",
    "critical_premises", "unused_critical_premises", "axioms_used",
    "dependency_closure", "dependency_digest", "represented_regions",
    "required_regions", "unconstrained_regions", "vacuity", "coverage",
    "previous", "semantic_changes", "semantic_review_required", "state",
}


def die(message: str) -> None:
    raise ValueError(message)


def report_digest(report: dict) -> str:
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def expect_invalid(report: dict, coverage_classes: set[str], vacuity_states: set[str], report_states: set[str]) -> None:
    try:
        validate_report(report, coverage_classes, vacuity_states, report_states)
    except ValueError:
        return
    die("negative control unexpectedly validated")


def validate_report(report: dict, coverage_classes: set[str], vacuity_states: set[str], report_states: set[str]) -> None:
    missing = REQUIRED_FIELDS - set(report)
    if missing:
        die(f"report {report.get('id', '<unknown>')} missing fields: {sorted(missing)}")

    if not report["proof_goal"] or not report["checker_receipt"]:
        die("proof goal and checker receipt must be non-empty")
    if not HEX64.fullmatch(report["subject_sha256"]):
        die("subject_sha256 must be 64 lowercase hex characters")
    if not HEX64.fullmatch(report["dependency_digest"]):
        die("dependency_digest must be 64 lowercase hex characters")

    array_fields = (
        "assumptions_in_scope", "assumptions_used", "critical_premises",
        "unused_critical_premises", "axioms_used", "dependency_closure",
        "represented_regions", "required_regions", "unconstrained_regions",
        "semantic_changes",
    )
    for field in array_fields:
        if not isinstance(report[field], list):
            die(f"{field} must be an array")

    in_scope = set(report["assumptions_in_scope"])
    used = set(report["assumptions_used"])
    critical = set(report["critical_premises"])
    unused_critical = set(report["unused_critical_premises"])
    if not used <= in_scope:
        die("used assumptions must be in scope")
    if not critical <= in_scope:
        die("critical premises must be in scope")
    if not unused_critical <= critical:
        die("unused critical premises must be critical premises")
    if unused_critical & used:
        die("unused critical premise cannot also be marked used")

    vacuity = report["vacuity"]
    if vacuity.get("state") not in vacuity_states:
        die("unknown vacuity state")
    if not isinstance(vacuity.get("evidence"), list):
        die("vacuity evidence must be an array")
    if vacuity["state"] == "Clear" and not vacuity["evidence"]:
        die("Clear vacuity state requires retained evidence")

    coverage = report["coverage"]
    coverage_class = coverage.get("class")
    if coverage_class not in coverage_classes:
        die("unknown coverage class")
    if not coverage.get("details"):
        die("coverage details must be explicit")
    if coverage_class == "Bounded" and coverage.get("bound") in (None, ""):
        die("Bounded coverage requires an explicit bound")
    if coverage_class != "Bounded" and coverage.get("bound") not in (None, ""):
        die("only Bounded coverage may carry a bound")

    represented = set(report["represented_regions"])
    required = set(report["required_regions"])
    unconstrained = set(report["unconstrained_regions"])
    if unconstrained - required:
        die("unconstrained regions must be part of required regions")
    if coverage_class == "Exact":
        if unconstrained:
            die("Exact coverage cannot have unconstrained required regions")
        if not required <= represented:
            die("Exact coverage requires every required region to be represented")
    if coverage_class == "Partial" and not unconstrained:
        die("Partial coverage must name at least one unconstrained required region")

    state = report["state"]
    if state not in report_states:
        die("unknown report state")
    review_required = report["semantic_review_required"]
    if not isinstance(review_required, bool):
        die("semantic_review_required must be boolean")
    if review_required and state == "reviewed-current":
        die("review-required report cannot remain reviewed-current")
    if report["semantic_changes"] and not review_required:
        die("semantic changes require semantic review")
    if unused_critical and not review_required:
        die("unused critical premise requires semantic review")
    if vacuity["state"] == "Detected" and state == "reviewed-current":
        die("detected vacuity cannot remain reviewed-current")
    if report["proof_checked"] is False and state == "reviewed-current":
        die("unchecked proof cannot be reviewed-current")

    previous = report["previous"]
    for field in ("report_sha256", "dependency_digest", "coverage_class"):
        if field not in previous:
            die(f"previous missing {field}")
    previous_report = previous["report_sha256"]
    previous_dep = previous["dependency_digest"]
    previous_coverage = previous["coverage_class"]
    if previous_report is not None and not HEX64.fullmatch(previous_report):
        die("previous report digest must be 64 lowercase hex or null")
    if previous_dep is not None and not HEX64.fullmatch(previous_dep):
        die("previous dependency digest must be 64 lowercase hex or null")
    if previous_coverage is not None and previous_coverage not in coverage_classes:
        die("previous coverage class invalid")

    dependency_changed = previous_dep is not None and previous_dep != report["dependency_digest"]
    coverage_changed = previous_coverage is not None and previous_coverage != coverage_class
    if (dependency_changed or coverage_changed) and not review_required:
        die("dependency/coverage semantic delta requires review")


def main() -> int:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    capsule_contract = json.loads(CAPSULE_PATH.read_text(encoding="utf-8"))

    if contract.get("schema") != "symthaea.formal-verification.proof-observability.v1":
        die("wrong observability schema")
    if contract.get("authority") != "ReviewMetadataOnly":
        die("observability authority escalated")
    if capsule_contract.get("schema") != "symthaea.formal-verification.proof-review-capsule.v1":
        die("parent proof capsule contract missing or wrong")
    if contract.get("parent_contract") != capsule_contract.get("schema"):
        die("observability parent contract drifted")
    if set(contract.get("required_nonclaims", [])) != REQUIRED_NONCLAIMS:
        die("required nonclaims changed")

    coverage_classes = set(contract.get("coverage_classes", []))
    vacuity_states = set(contract.get("vacuity_states", []))
    report_states = set(contract.get("report_states", []))
    if coverage_classes != {"Exact", "Partial", "Bounded", "Unknown"}:
        die("coverage class census changed")
    if vacuity_states != {"Clear", "Detected", "Unknown"}:
        die("vacuity state census changed")
    if report_states != {"reviewed-current", "review-required", "blocked"}:
        die("report state census changed")

    reports = contract.get("synthetic_reports", [])
    if len(reports) != 2:
        die("expected exactly two synthetic observability reports")
    for report in reports:
        validate_report(report, coverage_classes, vacuity_states, report_states)
        print(f"OBSERVABILITY {report['id']} sha256={report_digest(report)}")

    exact = next(r for r in reports if r["id"] == "SYN-OBS-EXACT-001")
    partial = next(r for r in reports if r["id"] == "SYN-OBS-PARTIAL-001")

    # Vacuous proof cannot remain current for the unconditional claim.
    vacuous = copy.deepcopy(partial)
    vacuous["vacuity"] = {"state": "Detected", "evidence": ["contradiction-witness"]}
    vacuous["state"] = "reviewed-current"
    expect_invalid(vacuous, coverage_classes, vacuity_states, report_states)

    # Exact coverage cannot hide a required unconstrained region.
    hidden_region = copy.deepcopy(exact)
    hidden_region["unconstrained_regions"] = ["synthetic/induction/step"]
    expect_invalid(hidden_region, coverage_classes, vacuity_states, report_states)

    # A newly unused critical premise requires semantic review.
    unused_critical = copy.deepcopy(exact)
    unused_critical["assumptions_used"] = ["P 0"]
    unused_critical["unused_critical_premises"] = ["forall n, P n -> P (n + 1)"]
    unused_critical["semantic_changes"] = []
    unused_critical["semantic_review_required"] = False
    unused_critical["state"] = "reviewed-current"
    expect_invalid(unused_critical, coverage_classes, vacuity_states, report_states)

    # Exact -> Partial is a semantic review event.
    narrowed = copy.deepcopy(partial)
    narrowed["previous"]["coverage_class"] = "Exact"
    narrowed["semantic_review_required"] = False
    narrowed["state"] = "reviewed-current"
    expect_invalid(narrowed, coverage_classes, vacuity_states, report_states)

    # Bounded evidence must name its bound.
    unbounded_bounded = copy.deepcopy(partial)
    unbounded_bounded["coverage"] = {"class": "Bounded", "bound": None, "details": "missing bound"}
    expect_invalid(unbounded_bounded, coverage_classes, vacuity_states, report_states)

    # Dependency identity drift cannot silently remain current.
    stale_dependencies = copy.deepcopy(partial)
    stale_dependencies["previous"]["dependency_digest"] = "6" * 64
    stale_dependencies["semantic_review_required"] = False
    stale_dependencies["state"] = "reviewed-current"
    expect_invalid(stale_dependencies, coverage_classes, vacuity_states, report_states)

    # Clear vacuity status cannot be asserted without retained analysis evidence.
    fake_clear = copy.deepcopy(partial)
    fake_clear["vacuity"] = {"state": "Clear", "evidence": []}
    expect_invalid(fake_clear, coverage_classes, vacuity_states, report_states)

    # Review-critical semantic changes must alter report identity.
    for label, mutate in (
        ("used-assumption", lambda r: r["assumptions_used"].append("synthetic-extra")),
        ("dependency", lambda r: r["dependency_closure"].append("new-lemma")),
        ("unconstrained-region", lambda r: r["unconstrained_regions"].append("src/new.rs:edge")),
        ("vacuity", lambda r: r["vacuity"].update({"state": "Unknown", "evidence": []})),
        ("coverage", lambda r: r["coverage"].update({"class": "Unknown", "details": "reclassified"})),
    ):
        candidate = copy.deepcopy(exact)
        before = report_digest(candidate)
        mutate(candidate)
        after = report_digest(candidate)
        if before == after:
            die(f"{label} semantic mutation did not alter observability identity")

    print("PROOF_OBSERVABILITY_V1_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"PROOF_OBSERVABILITY_V1_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
