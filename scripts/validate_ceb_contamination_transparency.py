#!/usr/bin/env python3
"""Validate CEB per-task contamination transparency cards offline.

The validator records assessed surfaces and evidence scope. It never asserts
that a task is globally contamination-free or absent from model training data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


SURFACES = {
    "source_git_history", "issue_pr", "forks", "dataset_mirrors", "benchmark_docs",
    "release_notes", "package_source_archives", "code_search", "web_search", "connected_tools",
}
TIERS = {"T1_exact", "T2_syntactic", "T3_semantic", "T4_task_level"}
MATCH_OUTCOMES = {"match_found"}


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {path}: {error}") from error


def canonical_record_digest(card: dict) -> str:
    payload = copy.deepcopy(card)
    payload.pop("card_record_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def card_semantic_errors(card: dict, required_surfaces: set[str]) -> list[str]:
    errors: list[str] = []
    tier_rows = card["tier_assessments"]
    tier_ids = [row["tier"] for row in tier_rows]
    if len(tier_ids) != len(set(tier_ids)) or set(tier_ids) != TIERS:
        errors.append("card must contain exactly one assessment for each T1–T4 leakage tier")

    surfaces = card["search_surfaces"]
    observed_surfaces = [row["surface"] for row in surfaces]
    if len(observed_surfaces) != len(set(observed_surfaces)):
        errors.append("search surfaces must be unique")
    if set(observed_surfaces) != required_surfaces:
        errors.append(
            f"search surface coverage mismatch; missing={sorted(required_surfaces-set(observed_surfaces))}, "
            f"extra={sorted(set(observed_surfaces)-required_surfaces)}"
        )

    for row in surfaces:
        if row["status"] == "searched":
            if not row["query_sha256"] or not row["result_sha256"] or not row["searched_on"]:
                errors.append(f"{row['surface']}: searched surface requires query/result digests and date")
        elif row["status"] == "not_available":
            if not row["coverage_and_limits"].strip():
                errors.append(f"{row['surface']}: unavailable surface requires a limitation reason")
        elif row["status"] == "not_searched" and card["assessment_status"] == "searched_no_match_found":
            errors.append(f"{row['surface']}: incomplete surface cannot support searched_no_match_found")

    outcomes = [row["outcome"] for row in tier_rows]
    has_match = any(outcome in MATCH_OUTCOMES for outcome in outcomes)
    has_inconclusive = any(outcome == "inconclusive" for outcome in outcomes)
    if card["assessment_status"] == "searched_no_match_found":
        if any(outcome != "no_match_found" for outcome in outcomes):
            errors.append("searched_no_match_found requires no_match_found at all four tiers")
        if any(row["status"] not in ("searched", "not_available") for row in surfaces):
            errors.append("searched_no_match_found requires all surfaces searched or explicitly unavailable")
    if card["assessment_status"] == "not_assessed":
        if any(outcome != "not_assessed" for outcome in outcomes):
            errors.append("not_assessed card cannot contain assessed tier outcomes")
    if card["assessment_status"] in ("suspected_leakage", "confirmed_leakage") and not (has_match or has_inconclusive):
        errors.append("suspected/confirmed leakage requires matching or inconclusive evidence")
    if has_match and card["assessment_status"] in ("not_assessed", "searched_no_match_found"):
        errors.append("positive T1–T4 match cannot be labeled not_assessed or no_match_found")
    t1 = next(row for row in tier_rows if row["tier"] == "T1_exact")
    if t1["outcome"] == "match_found" and card["assessment_status"] != "confirmed_leakage":
        errors.append("T1 exact solution/answer match must be labeled confirmed_leakage")

    exposure = card["training_exposure"]
    if exposure["status"] == "known_exposed" and card["assessment_status"] not in ("suspected_leakage", "confirmed_leakage"):
        errors.append("known training exposure cannot be labeled no_match_found")
    if exposure["status"] == "independently_tested_no_evidence" and not exposure["evidence_sha256"]:
        errors.append("independently_tested_no_evidence requires a scoped evidence digest")
    if "clean" in exposure["status"].lower() or "absent" in exposure["status"].lower():
        errors.append("training exposure status must not make a global clean/absence claim")

    access = card["agent_access"]
    if access["answer_key_in_agent_artifacts"] or access["candidate_workspace_has_hidden_tests"]:
        errors.append("answer key and hidden tests must remain outside agent-accessible artifacts")
    if not card["answer_key_custody"]["outside_agent_workspace"]:
        errors.append("answer key must remain in an external custody domain")

    if card["reviewer_disposition"] == "eligible_for_holdout_review":
        if card["source_visibility"]["fix_visibility"] not in ("private", "not_published"):
            errors.append("public or unknown fix visibility cannot be marked eligible for holdout review")
        if card["assessment_status"] != "searched_no_match_found":
            errors.append("holdout review requires completed search with no matches; this is not a clean guarantee")
        if access["egress_policy"] in ("unqualified", "not_run_blocked"):
            errors.append("holdout review requires a declared agent egress policy")
        if access["trajectory_logging"] != "enabled":
            errors.append("holdout review requires trajectory logging to be enabled")

    if card["reviewer_disposition"] == "development_only":
        if card["source_visibility"]["fix_visibility"] == "public" and "development_only_public_exposure" not in card["mitigations"]:
            errors.append("public fix exposure requires an explicit development-only mitigation")

    if card["reviewer_disposition"] in ("blocked", "reject") and not card["residual_uncertainty"]:
        errors.append("blocked/rejected card must preserve its residual-risk explanation")

    expected_digest = canonical_record_digest(card)
    if card["card_record_sha256"] != expected_digest:
        errors.append("card_record_sha256 does not match canonical card content")
    return errors


def validate_registry(registry: dict, registry_schema: dict, card_schema: dict) -> list[str]:
    registry_validator = Draft202012Validator(registry_schema, format_checker=FormatChecker())
    schema_issues = sorted(registry_validator.iter_errors(registry), key=lambda issue: list(issue.absolute_path))
    if schema_issues:
        return [
            f"registry schema {'/'.join(map(str, issue.absolute_path)) or '<root>'}: {issue.message}"
            for issue in schema_issues
        ]

    required_surfaces = set(registry["required_surfaces"])
    if required_surfaces != SURFACES:
        return ["required surface taxonomy must contain the ten registered leakage search surfaces"]

    card_validator = Draft202012Validator(card_schema, format_checker=FormatChecker())
    errors: list[str] = []
    card_ids: set[str] = set()
    task_ids: set[str] = set()
    for card in registry["cards"]:
        issues = sorted(card_validator.iter_errors(card), key=lambda issue: list(issue.absolute_path))
        if issues:
            errors.extend(
                f"{card.get('card_id','<card>')} schema {'/'.join(map(str, issue.absolute_path))}: {issue.message}"
                for issue in issues
            )
            continue
        if card["card_id"] in card_ids:
            errors.append(f"duplicate contamination card ID: {card['card_id']}")
        if card["task_id"] in task_ids:
            errors.append(f"duplicate contamination card task ID: {card['task_id']}")
        card_ids.add(card["card_id"])
        task_ids.add(card["task_id"])
        errors.extend(
            f"{card['card_id']}: {error}"
            for error in card_semantic_errors(card, required_surfaces)
        )

    if registry["actual_card_count"] != len(registry["cards"]):
        errors.append("actual_card_count must equal the number of card records")
    if registry["status"] == "design_only_no_task_cards" and registry["actual_card_count"] != 0:
        errors.append("design-only registry must contain zero task cards")
    if registry["actual_card_count"] == 0 and registry["status"] != "design_only_no_task_cards":
        errors.append("empty card registry must remain explicitly design-only")
    return errors


def make_fixture() -> dict:
    digest = "a" * 64
    surfaces = []
    for surface in sorted(SURFACES):
        surfaces.append({
            "surface": surface, "status": "searched", "method": "reviewer-run source query and audit",
            "query_sha256": digest, "result_sha256": digest, "searched_on": "2026-11-05",
            "reviewer_id": "leakage-reviewer-01",
            "coverage_and_limits": "Search covers the recorded public/private source endpoint and indexed results available on the assessment date; unindexed copies may remain.",
        })
    card = {
        "schema_version": 0, "card_id": "ceb-leak-fixture-001", "version": "0.1.0",
        "task_id": "ceb-rust-fixture-001", "assessment_status": "searched_no_match_found",
        "created_on": "2026-11-05", "assessed_on": "2026-11-05",
        "source_visibility": {
            "repository_visibility": "private", "issue_visibility": "private",
            "fix_visibility": "not_published", "benchmark_publication_visibility": "not_published",
        },
        "artifact_digests": {
            "repository_snapshot_sha256": digest, "task_prompt_sha256": digest,
            "acceptance_contract_sha256": digest, "answer_key_sha256": digest,
            "environment_sha256": digest,
        },
        "lifecycle_dates": {
            "issue_opened_on": "2026-01-01", "fix_published_on": None,
            "task_published_on": None, "snapshot_frozen_on": "2026-11-05",
        },
        "tier_assessments": [
            {"tier": tier, "outcome": "no_match_found", "method": "exact or semantic search appropriate to tier",
             "evidence_sha256": digest, "reviewer_id": "leakage-reviewer-01",
             "notes": "No match found in the searched evidence; this is not proof of global absence."}
            for tier in ("T1_exact", "T2_syntactic", "T3_semantic", "T4_task_level")
        ],
        "search_surfaces": surfaces,
        "training_exposure": {
            "status": "unknown", "method": "no direct access to all model training corpora",
            "evidence_sha256": None, "reviewer_id": "leakage-reviewer-01",
            "residual_limit": "Training corpus inclusion is unknown and cannot be ruled out by repository/web searches.",
        },
        "agent_access": {
            "egress_policy": "disabled", "trajectory_logging": "enabled",
            "answer_key_in_agent_artifacts": False, "candidate_workspace_has_hidden_tests": False,
            "access_policy_sha256": digest,
        },
        "answer_key_custody": {
            "custodian_id": "custodian-01", "custody_domain_id": "holdout-vault-01",
            "outside_agent_workspace": True, "custody_record_sha256": digest,
        },
        "mitigations": ["immutable_pre_fix_snapshot", "network_disabled", "answer_key_external_custody", "private_or_fresh_task", "trajectory_audit"],
        "residual_uncertainty": ["No indexed match found does not establish absence from model training or unindexed copies."],
        "reviewer_disposition": "eligible_for_holdout_review", "reviewer_id": "leakage-reviewer-01",
        "search_protocol_id": "CEB-LEAKAGE-SEARCH-001", "search_record_sha256": digest,
        "card_record_sha256": "",
    }
    card["card_record_sha256"] = canonical_record_digest(card)
    return card


def self_test(registry: dict, registry_schema: dict, card_schema: dict) -> None:
    if validate_registry(registry, registry_schema, card_schema):
        raise SystemExit("valid design-only transparency registry unexpectedly failed")

    fixture = make_fixture()
    validator = Draft202012Validator(card_schema, format_checker=FormatChecker())
    issues = list(validator.iter_errors(fixture))
    if issues:
        raise SystemExit("valid synthetic transparency card failed schema: " + "; ".join(issue.message for issue in issues))
    base_errors = card_semantic_errors(fixture, set(SURFACES))
    if base_errors:
        raise SystemExit("valid synthetic transparency card failed semantics: " + "; ".join(base_errors))

    def must_reject(mutant: dict, label: str) -> None:
        # Recompute the card digest to ensure each control reaches semantic rules.
        mutant["card_record_sha256"] = canonical_record_digest(mutant)
        errors = card_semantic_errors(mutant, set(SURFACES))
        if not errors:
            raise SystemExit(f"contamination negative control unexpectedly passed: {label}")

    mutant = copy.deepcopy(fixture)
    mutant["assessment_status"] = "searched_no_match_found"
    mutant["search_surfaces"][0]["status"] = "not_searched"
    must_reject(mutant, "no-match claim with incomplete search surfaces")

    mutant = copy.deepcopy(fixture)
    mutant["source_visibility"]["fix_visibility"] = "public"
    mutant["reviewer_disposition"] = "eligible_for_holdout_review"
    must_reject(mutant, "public fix represented as eligible sealed holdout")

    mutant = copy.deepcopy(fixture)
    mutant["search_surfaces"][0]["result_sha256"] = None
    must_reject(mutant, "missing search-result digest")

    mutant = copy.deepcopy(fixture)
    mutant["tier_assessments"][0]["outcome"] = "match_found"
    must_reject(mutant, "T1 exact match mislabeled no-match")

    mutant = copy.deepcopy(fixture)
    mutant["training_exposure"]["status"] = "independently_tested_no_evidence"
    mutant["training_exposure"]["evidence_sha256"] = None
    must_reject(mutant, "training exposure negative finding without scoped evidence")

    mutant = copy.deepcopy(fixture)
    mutant["agent_access"]["answer_key_in_agent_artifacts"] = True
    must_reject(mutant, "answer key exposed to candidate")

    mutant = copy.deepcopy(fixture)
    mutant["agent_access"]["egress_policy"] = "unqualified"
    mutant["reviewer_disposition"] = "eligible_for_holdout_review"
    must_reject(mutant, "holdout eligibility without a qualified egress declaration")

    mutant = copy.deepcopy(fixture)
    mutant["assessment_status"] = "verified_absent"
    must_reject(mutant, "unsupported global absence status")

    print("CEB_CONTAMINATION_CARD_SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    base = args.repo_root.resolve() / "benchmarks" / "ceb"
    try:
        registry = load(base / "contamination-transparency-registry.v0.json")
        registry_schema = load(base / "contamination-transparency-registry.schema.json")
        card_schema = load(base / "contamination-transparency-card.schema.json")
        Draft202012Validator.check_schema(registry_schema)
        Draft202012Validator.check_schema(card_schema)
    except Exception as error:
        print(f"CEB_CONTAMINATION_ERROR: {error}", file=sys.stderr)
        return 2

    errors = validate_registry(registry, registry_schema, card_schema)
    if errors:
        for error in errors:
            print(f"CEB_CONTAMINATION_FAIL: {error}", file=sys.stderr)
        return 1
    if args.self_test:
        self_test(registry, registry_schema, card_schema)
    print(
        "CEB_CONTAMINATION_PASS: "
        f"cards={registry['actual_card_count']} status={registry['status']} "
        f"tiers={len(registry['tier_taxonomy'])} surfaces={len(registry['required_surfaces'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
