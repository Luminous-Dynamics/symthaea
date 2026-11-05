#!/usr/bin/env python3
"""Validate CEB Rust pilot's balanced sampling slot plan offline."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

FAMILIES = [
    "rust-types-traits",
    "multi-crate-api-regression",
    "feature-target-configuration",
    "issue-reproduction-debugging",
    "security-correctness",
    "performance-correctness",
]
STRATA = ("low", "medium", "high")


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {path}: {error}") from error


def semantic_errors(plan: dict) -> list[str]:
    errors: list[str] = []
    slots = plan["slot_assignments"]
    repo_ids = [row["repository_slot_id"] for row in plan["repository_slots"]]
    if len(repo_ids) != len(set(repo_ids)):
        errors.append("duplicate repository placeholder")
    if len(set(plan["family_ids"])) != 6 or set(plan["family_ids"]) != set(FAMILIES):
        errors.append("fault-family registry must contain the six pilot families exactly once")
    if len({row["slot_id"] for row in slots}) != 48:
        errors.append("slot IDs must be unique")

    repo_counts = Counter(row["repository_slot_id"] for row in slots)
    family_counts = Counter(row["family_id"] for row in slots)
    partition_repo = Counter((row["repository_slot_id"], row["partition"]) for row in slots)
    partition_family = Counter((row["family_id"], row["partition"]) for row in slots)
    slot_kinds = Counter(row["slot_kind"] for row in slots)
    strata_counts = Counter(row["planned_difficulty_stratum"] for row in slots)

    if set(repo_counts) != set(repo_ids) or any(repo_counts[key] != 8 for key in repo_ids):
        errors.append("each of six repository placeholders must receive exactly eight slots")
    if set(family_counts) != set(FAMILIES) or any(family_counts[key] != 8 for key in FAMILIES):
        errors.append("each of six fault families must receive exactly eight slots")
    if slot_kinds != {"base": 36, "extra": 12}:
        errors.append("sampling design must contain 36 base cells and 12 extra slots")
    if Counter(row["partition"] for row in slots) != {"development": 32, "holdout": 16}:
        errors.append("sampling design must contain 32 development and 16 holdout slots")
    if any(not 2 <= partition_repo[(repo, "holdout")] <= 3 for repo in repo_ids):
        errors.append("each repository must receive two or three holdout slots")
    if any(not 2 <= partition_family[(family, "holdout")] <= 3 for family in FAMILIES):
        errors.append("each fault family must receive two or three holdout slots")
    if strata_counts != {"low": 16, "medium": 16, "high": 16}:
        errors.append("planned difficulty strata must contain 16 slots each")

    # Verify the explicit 36-cell + 12-extra bipartite construction and deterministic
    # holdout/stratum pattern. This keeps repository and family identity orthogonal.
    for i, repo_id in enumerate(repo_ids):
        repo_slots = [row for row in slots if row["repository_slot_id"] == repo_id]
        base = [row for row in repo_slots if row["slot_kind"] == "base"]
        extra = [row for row in repo_slots if row["slot_kind"] == "extra"]
        if {row["family_id"] for row in base} != set(FAMILIES) or len(base) != 6:
            errors.append(f"{repo_id}: needs exactly one base slot for every fault family")
        expected_extra_families = {FAMILIES[i], FAMILIES[(i + 1) % 6]}
        if {row["family_id"] for row in extra} != expected_extra_families or len(extra) != 2:
            errors.append(f"{repo_id}: extra slots must target its own and next cyclic family")
        for row in base:
            j = FAMILIES.index(row["family_id"])
            expected_holdout = j in ((i + 2) % 6, (i + 3) % 6)
            expected_stratum = STRATA[(i + j) % 3]
            if row["partition"] != ("holdout" if expected_holdout else "development"):
                errors.append(f"{row['slot_id']}: base slot partition deviates from preregistered balance")
            if row["planned_difficulty_stratum"] != expected_stratum:
                errors.append(f"{row['slot_id']}: planned difficulty stratum drifted")
        for row in extra:
            j = FAMILIES.index(row["family_id"])
            expected_holdout = (j == i and i < 4)
            expected_stratum = STRATA[(i + j + 1) % 3]
            if row["partition"] != ("holdout" if expected_holdout else "development"):
                errors.append(f"{row['slot_id']}: extra slot partition deviates from preregistered balance")
            if row["planned_difficulty_stratum"] != expected_stratum:
                errors.append(f"{row['slot_id']}: extra slot difficulty stratum drifted")

    if plan["actual_task_count"] != 0:
        errors.append("design-only V0 must not claim actual task assignments")
    if plan["status"] != "design_only_no_actual_task_assignments":
        errors.append("plan must remain design-only until repository/task identities are reviewed")
    if any(row["actual_repository_full_name"] is not None or row["actual_source_revision"] is not None for row in plan["repository_slots"]):
        errors.append("repository placeholders cannot be reported as real repositories")
    if any(row["task_id"] is not None or row["task_manifest_sha256"] is not None or row["actual_repository_full_name"] is not None or row["actual_difficulty_label"] is not None or row["difficulty_record_sha256"] is not None for row in slots):
        errors.append("slot plan cannot claim task identity, manifest, actual difficulty or repository assignment")
    if plan["difficulty_assessment_policy"]["actual_label_status"] == "":
        errors.append("difficulty-label status must remain explicit")
    if not plan["balancing_policy"]["no_post_score_exclusion"]:
        errors.append("post-score task exclusion is prohibited")
    return errors


def validate(plan: dict, schema: dict) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = sorted(validator.iter_errors(plan), key=lambda error: list(error.absolute_path))
    if schema_errors:
        return [f"schema {'/'.join(map(str, error.absolute_path))}: {error.message}" for error in schema_errors]
    return semantic_errors(plan)


def self_test(plan: dict, schema: dict) -> None:
    if validate(plan, schema):
        raise SystemExit("valid balanced sampling plan unexpectedly failed")

    def must_reject(mutant: dict, label: str) -> None:
        if not validate(mutant, schema):
            raise SystemExit(f"balanced sampling negative control unexpectedly passed: {label}")

    mutant = copy.deepcopy(plan)
    mutant["slot_assignments"][0]["family_id"] = mutant["slot_assignments"][1]["family_id"]
    must_reject(mutant, "repository/family cell confounding")

    mutant = copy.deepcopy(plan)
    mutant["slot_assignments"][1]["slot_id"] = mutant["slot_assignments"][0]["slot_id"]
    must_reject(mutant, "duplicate slot identity")

    mutant = copy.deepcopy(plan)
    holdout = next(row for row in mutant["slot_assignments"] if row["partition"] == "holdout")
    holdout["partition"] = "development"
    must_reject(mutant, "holdout denominator silently reduced")

    mutant = copy.deepcopy(plan)
    mutant["repository_slots"][0]["actual_repository_full_name"] = "example/public-repo"
    must_reject(mutant, "placeholder promoted to actual repository")

    mutant = copy.deepcopy(plan)
    mutant["slot_assignments"][0]["actual_difficulty_label"] = "high"
    mutant["slot_assignments"][0]["difficulty_record_sha256"] = "a" * 64
    must_reject(mutant, "difficulty label added post hoc to design-only slot")

    mutant = copy.deepcopy(plan)
    mutant["balancing_policy"]["no_post_score_exclusion"] = False
    must_reject(mutant, "post-score task exclusion enabled")

    mutant = copy.deepcopy(plan)
    mutant["actual_task_count"] = 1
    must_reject(mutant, "design-only plan claims an actual task")
    print("CEB_BALANCED_SAMPLING_SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    base = args.repo_root.resolve() / "benchmarks" / "ceb"
    try:
        plan = load(base / "rust-balanced-sampling.v0.json")
        schema = load(base / "rust-balanced-sampling.schema.json")
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        print(f"CEB_BALANCED_SAMPLING_ERROR: {error}", file=sys.stderr)
        return 2
    errors = validate(plan, schema)
    if errors:
        for error in errors:
            print(f"CEB_BALANCED_SAMPLING_FAIL: {error}", file=sys.stderr)
        return 1
    if args.self_test:
        self_test(plan, schema)
    print(
        "CEB_BALANCED_SAMPLING_PASS: "
        f"slots={len(plan['slot_assignments'])} "
        f"repositories={plan['repository_slot_count']} "
        f"families={plan['family_slot_count']} "
        f"development={plan['development_slot_count']} "
        f"holdout={plan['heldout_slot_count']} "
        f"actual_tasks={plan['actual_task_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
