#!/usr/bin/env python3
"""Independent oracle for MAT-ENG-001A.

Stdlib-only. Derives dispositions from raw synthetic inputs, then compares
against frozen expected outputs. It intentionally imports no Symthaea code.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

CORPUS = Path("docs/engineering/data/mat_eng_001a_reference_v1.json")
EXPECTED_SHA256 = "f252c72ebf89be0115738708423e9565de7be80b9680f2bf4128a59a8e35c134"
EXPECTED_SCHEMA = "mat-eng-001a-reference-v1"
EXPECTED_IDS = [f"M{i:02d}" for i in range(1, 33)]


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def derive(case: dict) -> dict:
    r = case["raw"]
    k = case["kind"]
    out: dict[str, str] = {}

    if k == "demand":
        out["demand"] = "Represented" if r.get("conditioned") and r.get("hard_constraints") else "Incomplete"

    elif k == "bottleneck":
        state = r.get("state")
        if state == "MaterialVariableNotBottlenecking":
            out.update(bottleneck="NotJustified", research="NoStrongDiscoveryDemand")
        elif state == "EvidenceInsufficient":
            out["bottleneck"] = "Unresolved"
        elif r.get("material_improved") and r.get("new_bottleneck"):
            out["bottleneck"] = "Migrated"
        else:
            raise AssertionError(f"unhandled bottleneck raw: {r}")

    elif k == "research":
        if r.get("existing_material_satisfies") is True:
            out["research"] = "ExistingMaterialSufficient"
        elif set(r.get("alternatives", [])) >= {"new_material", "process_innovation"}:
            out["research"] = "AlternativesPreserved"
        elif "material_gain" in r and "architecture_gain" in r:
            out["research"] = "MaterialCampaignNotForced" if r["architecture_gain"] > r["material_gain"] else "AlternativesPreserved"
        elif r.get("target_gain") and r.get("hard_constraint_violation"):
            out["projection"] = "Ineligible"
        elif r.get("relation") == "Incomparable":
            out["research"] = "ParetoIncomparable"
        else:
            raise AssertionError(f"unhandled research raw: {r}")

    elif k == "identity":
        if r.get("same_generation") and r.get("old_min") != r.get("new_min"):
            out["currentness"] = "NewGenerationRequired"
        elif "left" in r and "right" in r and r["left"] != r["right"]:
            out["scientific"] = "DistinctEvidenceIdentity"
        else:
            raise AssertionError(f"unhandled identity raw: {r}")

    elif k == "authority":
        src, tgt = r.get("source"), r.get("target")
        if src in {"AdvisoryCandidate", "SurrogatePrediction", "SyntheticBridgePass"} and tgt in {"MeasuredProperty", "ProductQualified"}:
            out["authority"] = "RejectPromotion"
        elif r.get("synthetic_all_pass") is True:
            out["authority"] = "SoftwareSemanticsOnly"
        else:
            raise AssertionError(f"unhandled authority raw: {r}")

    elif k == "projection":
        if r.get("source_condition") == "0K-perfect-crystal" and r.get("target_condition") == "300K-process-conditioned":
            out["projection"] = "EvidenceLimited"
        elif r.get("source_form") == "perfect_crystal" and r.get("target_form") == "porous_article":
            out["projection"] = "RejectStrict"
        elif r.get("coupon_process") == r.get("article_process") and r.get("coupon_orientation") == r.get("article_orientation") and r.get("condition") == "match" and r.get("measured") is True:
            out["projection"] = "EligibleBounded"
        elif r.get("coupon_process") != r.get("article_process"):
            out["projection"] = "ApplicabilityMismatch"
        else:
            raise AssertionError(f"unhandled projection raw: {r}")

    elif k == "process":
        if r.get("state") == "RouteProposed":
            out["process"] = "NotPhysicalEvidence"
        elif r.get("sample") == "Produced" and r.get("target_phase") == "Unestablished":
            out["projection"] = "Blocked"
        elif r.get("target_phase") == "Established" and r.get("required_property") == "Unmeasured":
            out["projection"] = "Blocked"
        elif r.get("properties_met") and r.get("required_form") not in r.get("produced_forms", []):
            out.update(process="FormBlocked", projection="Blocked")
        elif r.get("route") == "rapid_quench" and r.get("compatible") is False:
            out["process"] = "BlockedUnderProfile"
        else:
            raise AssertionError(f"unhandled process raw: {r}")

    elif k == "robustness":
        lower = float(r["mean"]) - float(r["sigma"])
        out["robustness"] = "Blocked" if lower < float(r["hard_min"]) else "EligibleBounded"

    elif k == "currentness":
        if r.get("evidence_lot") != r.get("article_lot") and not r.get("transfer_theorem"):
            out["currentness"] = "ReviewRequired"
        elif r.get("bound_temp_c") != r.get("current_temp_c"):
            out["currentness"] = "ReviewRequired"
        else:
            raise AssertionError(f"unhandled currentness raw: {r}")

    elif k == "lifecycle":
        if r.get("initial_pass") is True and r.get("end_of_life_pass") is False:
            out["projection"] = "LifecycleBlocked"
        else:
            raise AssertionError(f"unhandled lifecycle raw: {r}")

    elif k == "diagnosis":
        causes = r.get("plausible_causes", [])
        if len(set(causes)) > 1:
            out["projection"] = "Ambiguous"
        else:
            raise AssertionError(f"unhandled diagnosis raw: {r}")

    elif k == "feedback":
        if r.get("synthesis") == "Failed":
            out.update(feedback="RecordNegativeMemory", authority="NoUniversalUnsynthesizableClaim")
        elif r.get("surrogate") == "pass" and r.get("physical") == "fail":
            out["feedback"] = "RetainContradictionRecalibrate"
        else:
            raise AssertionError(f"unhandled feedback raw: {r}")

    else:
        raise AssertionError(f"unknown kind: {k}")

    return out


def validate_structure(doc: dict, raw_bytes: bytes) -> None:
    require(hashlib.sha256(raw_bytes).hexdigest() == EXPECTED_SHA256, "canonical corpus digest drift")
    require(doc.get("schema") == EXPECTED_SCHEMA, "schema drift")
    ids = [c.get("id") for c in doc.get("cases", [])]
    require(ids == EXPECTED_IDS, f"case ordering/identity drift: {ids}")
    require(len(doc["cases"]) == 32, "case count drift")


def validate_canonical(doc: dict) -> list[dict]:
    derived = []
    for case in doc["cases"]:
        got = derive(case)
        exp = case["expected"]
        require(got == exp, f"{case['id']} derived {got} != expected {exp}")
        derived.append({"id": case["id"], "derived": got})
    return derived


def mutation_suite(doc: dict) -> list[str]:
    passed: list[str] = []

    def changes(case_id: str, mutator, label: str) -> None:
        c = copy.deepcopy(next(c for c in doc["cases"] if c["id"] == case_id))
        baseline = derive(c)
        mutator(c["raw"])
        try:
            changed = derive(c)
        except (AssertionError, KeyError, TypeError, ValueError):
            passed.append(label)
            return
        require(changed != baseline, f"mutation did not change disposition: {label}")
        passed.append(label)

    changes("M01", lambda r: r.__setitem__("conditioned", False), "conditioned-demand-removed")
    changes("M05", lambda r: r.__setitem__("existing_material_satisfies", False), "existing-material-support-removed")
    changes("M09", lambda r: r.__setitem__("source", "CouponMeasured"), "advisory-authority-relabel")
    changes("M10", lambda r: r.__setitem__("source", "CouponMeasured"), "surrogate-authority-relabel")
    changes("M11", lambda r: r.__setitem__("target_condition", "0K-perfect-crystal"), "condition-identity-hidden")
    changes("M12", lambda r: r.__setitem__("target_form", "perfect_crystal"), "form-identity-hidden")
    changes("M14", lambda r: r.__setitem__("state", "Measured"), "route-promoted-without-proof")
    changes("M15", lambda r: r.__setitem__("target_phase", "Established"), "phase-promoted-without-required-property")
    changes("M17", lambda r: r.__setitem__("article_process", "P2"), "coupon-process-mismatch")
    changes("M17", lambda r: r.__setitem__("article_orientation", "Y"), "coupon-orientation-mismatch")
    changes("M19", lambda r: r.__setitem__("sigma", 0), "uncertainty-zeroed")
    changes("M20", lambda r: r.__setitem__("produced_forms", ["powder", "2mm_sheet"]), "required-form-blocker-deleted")
    changes("M22", lambda r: r.__setitem__("article_lot", "A"), "lot-identity-hidden")
    changes("M23", lambda r: r.__setitem__("current_temp_c", 80), "operating-profile-drift-hidden")
    changes("M24", lambda r: r.__setitem__("end_of_life_pass", True), "lifecycle-failure-deleted")
    changes("M31", lambda r: r.__setitem__("target", "SyntheticReport"), "synthetic-authority-target-changed")

    d = copy.deepcopy(doc)
    d["schema"] = "mat-eng-001a-reference-v2"
    try:
        validate_structure(d, json.dumps(d, separators=(",", ":")).encode())
    except AssertionError:
        passed.append("schema-drift")
    else:
        raise AssertionError("schema drift accepted")

    d = copy.deepcopy(doc)
    d["cases"] = d["cases"][::-1]
    try:
        validate_structure(d, json.dumps(d, separators=(",", ":")).encode())
    except AssertionError:
        passed.append("case-order-drift")
    else:
        raise AssertionError("case order drift accepted")

    d = copy.deepcopy(doc)
    d["cases"] = d["cases"][:-1]
    try:
        validate_structure(d, json.dumps(d, separators=(",", ":")).encode())
    except AssertionError:
        passed.append("missing-case")
    else:
        raise AssertionError("missing case accepted")

    raw = CORPUS.read_bytes() + b"\n"
    try:
        validate_structure(doc, raw)
    except AssertionError:
        passed.append("source-byte-drift")
    else:
        raise AssertionError("source byte drift accepted")

    require(len(passed) >= 20, "mutation suite unexpectedly small")
    return passed


def main() -> int:
    raw = CORPUS.read_bytes()
    doc = json.loads(raw)
    validate_structure(doc, raw)
    derived = validate_canonical(doc)
    mutations = mutation_suite(doc)
    result = {
        "schema": "mat-eng-001a1-qualification-result-v1",
        "source_schema": EXPECTED_SCHEMA,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "case_count": len(derived),
        "mutation_count": len(mutations),
        "cases": derived,
        "mutations_passed": mutations,
        "disposition": "LocalDevelopmentPass",
        "claim_ceiling": "independent synthetic software-semantic re-derivation only",
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"MAT-ENG-001A1 FAIL: {exc}", file=sys.stderr)
        raise
