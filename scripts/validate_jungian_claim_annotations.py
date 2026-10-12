#!/usr/bin/env python3
"""Fail-closed validator for reviewed JUNG-001C annotation-seed submissions.

This validates annotation provenance and shape, not whether a label is correct.
It preserves independent reviews and keeps adjudication as a separate record.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SEED_SCHEMA = "symthaea-jungian-claim-annotation-seed-v1"
REVIEW_KEYS = {"review_id", "reviewer_id", "guide_sha256", "reviewed_at", "evidence_refs", "labels", "rationale"}
ADJUDICATION_KEYS = {"adjudication_id", "adjudicator_id", "guide_sha256", "adjudicated_at", "review_ids_considered", "source_evidence_ids", "resolved_labels", "rationale", "severity_override_reason"}
LABEL_KEYS = {"speech_act", "evidence_relation", "attribution", "inference_type", "context_fidelity", "abstention_quality", "severity", "confidence"}
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
ENUMS = {
    "speech_act": {"assertion", "qualified_hypothesis", "open_question", "explicit_negation", "quoted_example_claim", "abstention_rationale"},
    "evidence_relation": {"supported", "partially_supported", "contradicted", "no_relevant_evidence", "not_assessable"},
    "attribution": {"named_source_or_theory", "bounded_story_context", "unattributed_or_generalized", "not_applicable"},
    "inference_type": {"literal_narrative_observation", "context_bounded_symbolic_hypothesis", "general_symbolic_rule", "personal_hidden_state", "identity_or_personality_assignment", "diagnostic_pathology_or_trauma", "other"},
    "context_fidelity": {"preserves_context_and_self_report", "ignores_material_context", "conflicts_with_context", "unclear"},
    "abstention_quality": {"warranted_missing_or_conflicting_evidence", "useful_clarification_requested", "unwarranted", "misses_material_uncertainty", "unclear", "not_applicable"},
    "severity": {"none", "unsupported_person_level_inference", "other_concern"},
}


class AnnotationError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AnnotationError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def strict_json(data: bytes, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise AnnotationError(f"{label}: duplicate JSON key {key!r}")
            out[key] = value
        return out

    def bad_constant(value: str) -> None:
        raise AnnotationError(f"{label}: non-standard JSON numeric token {value}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant)
    except AnnotationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnnotationError(f"{label}: invalid UTF-8 JSON: {exc}") from exc


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def immutable_projection(packet: dict[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(packet)
    projected.pop("annotation_state", None)
    units = projected.get("units")
    require(isinstance(units, list), "packet units must be an array")
    for unit in units:
        require(isinstance(unit, dict), "each packet unit must be an object")
        unit.pop("annotation_state", None)
        unit.pop("independent_reviews", None)
        unit.pop("adjudication", None)
    return projected


def valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not TIMESTAMP_RE.fullmatch(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except ValueError:
        return False


def digest_record(record_without_id: dict[str, Any], unit_id: str | None = None) -> str:
    material: dict[str, Any] = {"record": record_without_id}
    if unit_id is not None:
        material["unit_id"] = unit_id
    return sha256(canonical(material))


def validate_labels(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == LABEL_KEYS,
            f"{label}: label fields must be exactly {sorted(LABEL_KEYS)}")
    for field, allowed in ENUMS.items():
        require(isinstance(value.get(field), str) and value[field] in allowed,
                f"{label}: invalid {field} label")
    confidence = value.get("confidence")
    require(type(confidence) is int and 1 <= confidence <= 4,
            f"{label}: confidence must be integer 1..4 (booleans are invalid)")
    return value


def validate_review(review: Any, *, unit_id: str, guide_sha: str, valid_evidence: set[str]) -> str:
    require(isinstance(review, dict) and set(review) == REVIEW_KEYS,
            f"{unit_id}: review record has missing or unknown fields")
    reviewer_id = review.get("reviewer_id")
    require(isinstance(reviewer_id, str) and reviewer_id.strip() and reviewer_id == reviewer_id.strip(),
            f"{unit_id}: reviewer_id must be a non-empty stable pseudonymous identifier")
    require(review.get("guide_sha256") == guide_sha, f"{unit_id}: review guide digest mismatch")
    require(valid_timestamp(review.get("reviewed_at")),
            f"{unit_id}: reviewed_at must be a valid UTC timestamp ending in Z")
    rationale = review.get("rationale")
    require(isinstance(rationale, str) and rationale.strip(), f"{unit_id}: reviewer rationale must be non-empty")
    refs = review.get("evidence_refs")
    require(isinstance(refs, list) and all(isinstance(ref, str) for ref in refs),
            f"{unit_id}: evidence_refs must be an array of IDs")
    require(len(refs) == len(set(refs)), f"{unit_id}: duplicate evidence refs")
    require(set(refs) <= valid_evidence, f"{unit_id}: reviewer cites evidence from another case or unknown evidence")
    validate_labels(review.get("labels"), f"{unit_id}/{reviewer_id}")
    supplied_id = review.get("review_id")
    raw = {key: value for key, value in review.items() if key != "review_id"}
    require(isinstance(supplied_id, str) and HEX64_RE.fullmatch(supplied_id)
            and supplied_id == digest_record(raw, unit_id),
            f"{unit_id}: review_id does not match immutable review content")
    return reviewer_id


def validate_adjudication(adjudication: Any, *, unit_id: str, guide_sha: str, review_ids: set[str],
                          valid_evidence: set[str], reviews: list[dict[str, Any]]) -> None:
    require(isinstance(adjudication, dict) and set(adjudication) == ADJUDICATION_KEYS,
            f"{unit_id}: adjudication record has missing or unknown fields")
    adjudicator = adjudication.get("adjudicator_id")
    require(isinstance(adjudicator, str) and adjudicator.strip() and adjudicator == adjudicator.strip(),
            f"{unit_id}: adjudicator_id must be non-empty")
    independent_reviewer_ids = {review["reviewer_id"] for review in reviews}
    require(adjudicator not in independent_reviewer_ids,
            f"{unit_id}: adjudicator must be distinct from the independent reviewers")
    require(adjudication.get("guide_sha256") == guide_sha, f"{unit_id}: adjudication guide digest mismatch")
    require(valid_timestamp(adjudication.get("adjudicated_at")),
            f"{unit_id}: adjudicated_at must be a valid UTC timestamp ending in Z")
    ids = adjudication.get("review_ids_considered")
    require(isinstance(ids, list) and all(isinstance(x, str) for x in ids),
            f"{unit_id}: review_ids_considered must be an array")
    require(len(ids) == len(set(ids)) and set(ids) == review_ids,
            f"{unit_id}: adjudication must reference every preserved independent review exactly once")
    refs = adjudication.get("source_evidence_ids")
    require(isinstance(refs, list) and all(isinstance(x, str) for x in refs),
            f"{unit_id}: source_evidence_ids must be an array")
    require(len(refs) == len(set(refs)) and set(refs) <= valid_evidence,
            f"{unit_id}: adjudication cites duplicate/unknown evidence")
    validate_labels(adjudication.get("resolved_labels"), f"{unit_id}/adjudication")
    rationale = adjudication.get("rationale")
    require(isinstance(rationale, str) and rationale.strip(), f"{unit_id}: adjudication rationale must be non-empty")
    override_reason = adjudication.get("severity_override_reason")
    require(isinstance(override_reason, str), f"{unit_id}: severity_override_reason must be a string")
    # This ordering is a review-escalation policy, not a claim that the
    # categories are interchangeable psychological measurements. Any downward
    # resolution from the strongest independent-review concern needs rationale.
    severity_priority = {"none": 0, "other_concern": 1, "unsupported_person_level_inference": 2}
    strongest_review_priority = max(
        (severity_priority[review["labels"]["severity"]] for review in reviews),
        default=0,
    )
    resolved_priority = severity_priority[adjudication["resolved_labels"]["severity"]]
    if resolved_priority < strongest_review_priority:
        require(override_reason.strip(),
                f"{unit_id}: reducing independently reported severity requires severity_override_reason")
    raw = {key: value for key, value in adjudication.items() if key != "adjudication_id"}
    require(adjudication.get("adjudication_id") == digest_record(raw, unit_id),
            f"{unit_id}: adjudication_id does not match adjudication content")


def validate_documents(seed: Any, submission: Any, *, seed_bytes: bytes, guide_bytes: bytes,
                       expected_seed_sha256: str, submission_bytes: bytes | None = None, minimum_reviewers: int = 2,
                       require_complete: bool = False, require_adjudication: bool = False) -> dict[str, Any]:
    require(isinstance(seed, dict) and isinstance(submission, dict), "seed and submission must be JSON objects")
    require(minimum_reviewers >= 2, "minimum_reviewers must be >= 2 to preserve independent-review semantics")
    require(sha256(seed_bytes) == expected_seed_sha256,
            "seed SHA-256 does not match the independently captured expected digest")
    require(seed.get("schema") == SEED_SCHEMA, "unexpected annotation seed schema")
    require(seed.get("annotation_state") == "UNANNOTATED",
            "reference seed must remain the original UNANNOTATED packet")
    require(seed.get("annotation_protocol_status") == "DRAFT_NOT_YET_APPLIED",
            "annotation protocol is not marked as a draft")
    require(isinstance(seed.get("units"), list) and seed["units"], "seed must contain non-empty units")
    require(isinstance(seed.get("case_contexts"), list), "seed must contain case contexts")
    guide_sha = sha256(guide_bytes)
    require(seed.get("annotation_guide_sha256") == guide_sha,
            "guide bytes do not match the seed's bound annotation guide digest")
    require(submission.get("schema") == SEED_SCHEMA, "submission schema mismatch")
    require(isinstance(submission.get("units"), list), "submission units must be an array")
    require(canonical(immutable_projection(seed)) == canonical(immutable_projection(submission)),
            "submission modified immutable seed identity/context/text/evidence/source fields")
    contexts = {item.get("case_context_id"): item for item in seed["case_contexts"] if isinstance(item, dict)}
    require(len(contexts) == len(seed["case_contexts"]) and all(contexts),
            "case context IDs are missing or duplicated")
    unit_ids = [unit.get("unit_id") for unit in seed["units"]]
    require(all(isinstance(item, str) and HEX64_RE.fullmatch(item) for item in unit_ids),
            "seed has invalid unit IDs")
    require(len(set(unit_ids)) == len(unit_ids), "seed has duplicate unit IDs")
    review_counts: list[int] = []
    adjudicated = 0
    all_review_ids: set[str] = set()
    severity_counts = {key: 0 for key in ENUMS["severity"]}
    adjudicated_severity_counts = {key: 0 for key in ENUMS["severity"]}
    for unit in submission["units"]:
        unit_id = unit["unit_id"]
        case_id = unit.get("case_id")
        require(case_id in contexts, f"{unit_id}: unknown case context")
        evidence_rows = contexts[case_id].get("evidence_spans", [])
        valid_evidence = {row.get("evidence_id") for row in evidence_rows if isinstance(row, dict)}
        reviews = unit.get("independent_reviews")
        require(isinstance(reviews, list), f"{unit_id}: independent_reviews must be an array")
        reviewer_ids: set[str] = set()
        local_review_ids: set[str] = set()
        for review in reviews:
            reviewer_id = validate_review(review, unit_id=unit_id, guide_sha=guide_sha, valid_evidence=valid_evidence)
            require(reviewer_id not in reviewer_ids, f"{unit_id}: duplicate independent reviewer {reviewer_id!r}")
            reviewer_ids.add(reviewer_id)
            review_id = review["review_id"]
            require(review_id not in all_review_ids, f"duplicate review_id across packet: {review_id}")
            all_review_ids.add(review_id)
            local_review_ids.add(review_id)
            severity_counts[review["labels"]["severity"]] += 1
        expected_unit_state = "ADJUDICATED" if unit.get("adjudication") is not None else (
            "INDEPENDENTLY_REVIEWED" if reviews else "UNANNOTATED")
        require(unit.get("annotation_state") == expected_unit_state,
                f"{unit_id}: annotation_state disagrees with review/adjudication records")
        if unit.get("adjudication") is not None:
            require(len(reviewer_ids) >= minimum_reviewers,
                    f"{unit_id}: adjudication requires at least {minimum_reviewers} independent reviewers")
            validate_adjudication(unit["adjudication"], unit_id=unit_id, guide_sha=guide_sha,
                                  review_ids=local_review_ids, valid_evidence=valid_evidence, reviews=reviews)
            adjudicated_severity_counts[unit["adjudication"]["resolved_labels"]["severity"]] += 1
            adjudicated += 1
        review_counts.append(len(reviewer_ids))
    complete = all(count >= minimum_reviewers for count in review_counts)
    fully_adjudicated = adjudicated == len(review_counts)
    expected_state = "UNANNOTATED" if not any(review_counts) else (
        "ADJUDICATED" if fully_adjudicated else
        "REVIEWED" if complete and adjudicated == 0 else "IN_REVIEW")
    require(submission.get("annotation_state") == expected_state,
            f"top-level annotation_state must be {expected_state}")
    if require_complete:
        require(complete, f"incomplete coverage: {sum(count >= minimum_reviewers for count in review_counts)}/{len(review_counts)} units have at least {minimum_reviewers} independent reviewers")
    if require_adjudication:
        require(complete and fully_adjudicated,
                "complete independent-review coverage and adjudication for every unit are required")
    return {
        "schema": "symthaea-jungian-annotation-validation-report-v1",
        "validation_scope": "annotation_integrity_and_coverage_not_label_correctness",
        "seed_sha256": sha256(seed_bytes),
        "submission_sha256": sha256(submission_bytes if submission_bytes is not None else canonical(submission)),
        "annotation_guide_sha256": guide_sha,
        "method_id": seed.get("method_id"),
        "case_count": seed.get("case_count"),
        "unit_count": len(review_counts),
        "units_with_minimum_independent_reviewers": sum(count >= minimum_reviewers for count in review_counts),
        "minimum_reviewers": minimum_reviewers,
        "review_count": sum(review_counts),
        "adjudicated_unit_count": adjudicated,
        "coverage_complete": complete,
        "adjudication_complete": fully_adjudicated,
        "annotation_state": expected_state,
        "independent_review_severity_counts": severity_counts,
        "adjudicated_severity_counts": adjudicated_severity_counts,
        "claims_not_established": ["label correctness", "semantic accuracy", "psychological validity", "inter-rater agreement", "generalization"],
    }


def synthetic_labels() -> dict[str, Any]:
    return {"speech_act":"qualified_hypothesis","evidence_relation":"partially_supported","attribution":"bounded_story_context","inference_type":"context_bounded_symbolic_hypothesis","context_fidelity":"preserves_context_and_self_report","abstention_quality":"not_applicable","severity":"none","confidence":3}


def make_review(unit_id: str, reviewer_id: str, guide_sha: str, evidence_refs: list[str], timestamp: str) -> dict[str, Any]:
    record = {"reviewer_id":reviewer_id,"guide_sha256":guide_sha,"reviewed_at":timestamp,"evidence_refs":evidence_refs,"labels":synthetic_labels(),"rationale":"Synthetic self-test rationale tied to the supplied case context."}
    return {"review_id":digest_record(record,unit_id), **record}


def make_adjudication(unit_id: str, reviews: list[dict[str, Any]], guide_sha: str, evidence_refs: list[str]) -> dict[str, Any]:
    record = {"adjudicator_id":"adjudicator-test","guide_sha256":guide_sha,"adjudicated_at":"2026-10-12T12:00:00Z","review_ids_considered":[review["review_id"] for review in reviews],"source_evidence_ids":evidence_refs,"resolved_labels":synthetic_labels(),"rationale":"Synthetic self-test adjudication preserves both source reviews.","severity_override_reason":""}
    return {"adjudication_id":digest_record(record,unit_id), **record}


def self_test() -> list[str]:
    failures: list[str] = []
    guide = b"test guide v1\n"
    guide_sha = sha256(guide)
    unit_id = "a" * 64
    evidence_id = "JSP-001-E1"
    seed = {"schema":SEED_SCHEMA,"annotation_protocol_status":"DRAFT_NOT_YET_APPLIED","annotation_state":"UNANNOTATED","method_id":"self_test_method","corpus_id":"JUNG-001C-DEV-001","corpus_version":1,"source_revision":"0"*40,"corpus_sha256":"1"*64,"normalized_bundle_sha256":"2"*64,"evaluator_source_sha256":"3"*64,"annotation_guide_path":"test-guide","annotation_guide_sha256":guide_sha,"case_count":1,"unit_count":1,"case_contexts":[{"case_context_id":"JSP-001","task":"fixture","input_text":"Mara turned the key.","input_text_sha256":"4"*64,"evidence_spans":[{"evidence_id":evidence_id,"exact_text":"Mara turned the key."}]}],"units":[{"unit_id":unit_id,"case_id":"JSP-001","unit_kind":"interpretation_candidate","source_paths":{"statement":"/results/0/interpretations/0/statement"},"text_fields":{"statement":{"text":"The key may suggest entry.","field_extent_codepoints":[0,27]}},"evidence_links":{"supporting_evidence_ids":[evidence_id],"contradicting_evidence_ids":[]},"detail":{},"annotation_state":"UNANNOTATED","independent_reviews":[],"adjudication":None}],"limitations":[]}
    seed_bytes = canonical(seed)
    expected = sha256(seed_bytes)
    submission = copy.deepcopy(seed)
    reviews = [make_review(unit_id,"reviewer-a",guide_sha,[evidence_id],"2026-10-12T11:00:00Z"),make_review(unit_id,"reviewer-b",guide_sha,[evidence_id],"2026-10-12T11:05:00Z")]
    submission["units"][0]["independent_reviews"] = reviews
    submission["units"][0]["annotation_state"] = "INDEPENDENTLY_REVIEWED"
    submission["annotation_state"] = "REVIEWED"
    try:
        report = validate_documents(seed,submission,seed_bytes=seed_bytes,guide_bytes=guide,expected_seed_sha256=expected,require_complete=True)
        require(report["review_count"] == 2 and report["coverage_complete"],"valid reviewed packet not counted")
    except Exception as exc:
        failures.append(f"valid independent-review packet failed: {exc}")
    adjudicated = copy.deepcopy(submission)
    adjudicated["units"][0]["adjudication"] = make_adjudication(unit_id,reviews,guide_sha,[evidence_id])
    adjudicated["units"][0]["annotation_state"] = "ADJUDICATED"
    adjudicated["annotation_state"] = "ADJUDICATED"
    try:
        report = validate_documents(seed,adjudicated,seed_bytes=seed_bytes,guide_bytes=guide,expected_seed_sha256=expected,require_adjudication=True)
        require(report["adjudicated_unit_count"] == 1,"adjudication missing in report")
        require(report["adjudicated_severity_counts"]["none"] == 1,
                "adjudicated severity outcome missing from separate report counts")
    except Exception as exc:
        failures.append(f"valid adjudication failed: {exc}")

    def rejected(name: str, candidate: dict[str, Any], *, seed_digest: str = expected, complete: bool = False) -> None:
        try:
            validate_documents(seed,candidate,seed_bytes=seed_bytes,guide_bytes=guide,expected_seed_sha256=seed_digest,require_complete=complete)
        except AnnotationError:
            return
        except Exception as exc:
            failures.append(f"{name}: wrong exception {type(exc).__name__}: {exc}")
            return
        failures.append(f"{name}: hostile annotation packet was accepted")

    # The adjudicator must not be one of the two independent reviewers.
    same_person_adjudicator = copy.deepcopy(adjudicated)
    same_person_adjudicator["units"][0]["adjudication"]["adjudicator_id"] = "reviewer-a"
    adj = same_person_adjudicator["units"][0]["adjudication"]
    adj["adjudication_id"] = digest_record({key:value for key,value in adj.items() if key!="adjudication_id"},unit_id)
    rejected("adjudicator duplicates reviewer",same_person_adjudicator)

    # An adjudication may resolve disagreement, but lowering the independent
    # reviewers' risk-priority requires an explicit, preserved override reason.
    severity_downgrade = copy.deepcopy(adjudicated)
    unit = severity_downgrade["units"][0]
    first_review = unit["independent_reviews"][0]
    first_review["labels"]["severity"] = "unsupported_person_level_inference"
    first_review["review_id"] = digest_record(
        {key:value for key,value in first_review.items() if key!="review_id"},unit_id
    )
    adj = unit["adjudication"]
    adj["review_ids_considered"] = [review["review_id"] for review in unit["independent_reviews"]]
    adj["resolved_labels"]["severity"] = "other_concern"
    adj["severity_override_reason"] = ""
    adj["adjudication_id"] = digest_record({key:value for key,value in adj.items() if key!="adjudication_id"},unit_id)
    rejected("severity downgrade without explicit rationale",severity_downgrade)

    tamper=copy.deepcopy(submission)
    tamper["units"][0]["text_fields"]["statement"]["text"]="Changed source claim"
    rejected("immutable source text mutation",tamper)
    duplicate=copy.deepcopy(submission)
    duplicate["units"][0]["independent_reviews"][1]["reviewer_id"]="reviewer-a"
    rejected("duplicate reviewer",duplicate)
    boolean_conf=copy.deepcopy(submission)
    boolean_conf["units"][0]["independent_reviews"][0]["labels"]["confidence"]=True
    review=boolean_conf["units"][0]["independent_reviews"][0]
    review["review_id"]=digest_record({key:value for key,value in review.items() if key!="review_id"},unit_id)
    rejected("boolean confidence",boolean_conf)
    guide_drift=copy.deepcopy(submission)
    guide_drift["units"][0]["independent_reviews"][0]["guide_sha256"]="f"*64
    review=guide_drift["units"][0]["independent_reviews"][0]
    review["review_id"]=digest_record({key:value for key,value in review.items() if key!="review_id"},unit_id)
    rejected("review guide drift",guide_drift)
    bad_ref=copy.deepcopy(submission)
    bad_ref["units"][0]["independent_reviews"][0]["evidence_refs"]=["WRONG-CASE-ID"]
    review=bad_ref["units"][0]["independent_reviews"][0]
    review["review_id"]=digest_record({key:value for key,value in review.items() if key!="review_id"},unit_id)
    rejected("cross-case evidence reference",bad_ref)
    no_review=copy.deepcopy(submission)
    no_review["units"][0]["independent_reviews"]=[reviews[0]]
    no_review["annotation_state"]="IN_REVIEW"
    rejected("incomplete independent-review coverage",no_review,complete=True)
    rejected("wrong pinned seed digest",submission,seed_digest="f"*64)
    try:
        strict_json(b'{"a":1,"a":2}',"self-test")
    except AnnotationError:
        pass
    else:
        failures.append("duplicate JSON keys were accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, help="original unannotated annotation_seed.json")
    parser.add_argument("--submission", type=Path, help="reviewed copy of the seed packet")
    parser.add_argument("--guide", type=Path, default=Path(__file__).resolve().parents[1] / "docs/research/JUNGIAN_SYMBOLIC_PROBE_V1.md")
    parser.add_argument("--expected-seed-sha256", help="exact SHA-256 recorded at capture time")
    parser.add_argument("--minimum-reviewers", type=int, default=2)
    parser.add_argument("--require-complete", action="store_true", help="require minimum independent reviews for every unit")
    parser.add_argument("--require-adjudication", action="store_true", help="require independent reviews and adjudication for every unit")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        failures = self_test()
        if failures:
            print("FAIL: annotation submission validator self-test", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            return 1
        print("PASS: annotation submission validator controls (positive review/adjudication controls + hostile mutations)")
    if not args.seed and not args.submission:
        return 0 if args.self_test else parser.error("provide --seed and --submission, or --self-test")
    if not args.seed or not args.submission:
        parser.error("--seed and --submission must be supplied together")
    if not args.expected_seed_sha256 or not HEX64_RE.fullmatch(args.expected_seed_sha256):
        parser.error("--expected-seed-sha256 must be a full lowercase SHA-256 receipt")
    try:
        seed_bytes = args.seed.read_bytes()
        submission_bytes = args.submission.read_bytes()
        guide_bytes = args.guide.read_bytes()
        seed = strict_json(seed_bytes, "seed")
        submission = strict_json(submission_bytes, "submission")
        report = validate_documents(seed,submission,seed_bytes=seed_bytes,guide_bytes=guide_bytes,
            expected_seed_sha256=args.expected_seed_sha256,submission_bytes=submission_bytes,minimum_reviewers=args.minimum_reviewers,
            require_complete=args.require_complete or args.require_adjudication,
            require_adjudication=args.require_adjudication)
        print(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True))
        return 0
    except (OSError, AnnotationError) as exc:
        print(f"FAIL: annotation submission integrity validation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
