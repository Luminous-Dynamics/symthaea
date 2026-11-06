#!/usr/bin/env python3
"""Validate CEB's public Rust candidate/oracle crosswalk without executing fixtures.

This is a static authoring-contract check. It reads JSON, README files, Cargo
manifest presence and tracked source-file inventories. It never invokes Cargo,
a compiler, a build script, a procedural macro, a test, a grader or candidate
code.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

CROSSWALK = Path(
    "docs/research/CEB_RUST_CANDIDATE_ORACLE_TRACEABILITY_V0_2026-11-06.json"
)

# Pinned one-to-one mapping from the reviewed #10617 plan. The exact source
# file Git blob IDs are also recorded inside the crosswalk.
EXPECTED_FAMILY_BY_CANDIDATE = {
    "api-migration": "multi-crate-api-regression",
    "trait-async": "rust-types-traits",
    "issue-reproduction": "issue-reproduction-debugging",
    "security-boundary": "security-correctness",
    "feature-target-matrix": "feature-target-configuration",
    "bounded-topk": "performance-correctness",
}
EXPECTED_PLAN_BLOB = "7a17223e98c3c2a87c333535620d3937db5e80f0"
EXPECTED_PREREG_BLOB = "d355ed00accfe4a02995894463ccd6ac714d760b"
EXPECTED_SOURCE_BRANCH = "research/ceb-balanced-task-sampling-v0-2026-11-05"
EXPECTED_SOURCE_PATHS = [
    "benchmarks/ceb/rust-balanced-sampling.v0.json",
    "benchmarks/ceb/rust-pilot-preregistration.v0.json",
]
DISPOSITION_BY_KIND = {
    "valid_alternative": "accepted_candidate",
    "negative_near_miss": "rejected_candidate",
    "required_composition": "accepted_candidate",
    "unresolved_control": "unresolved_not_scored",
}
DESIGN_ONLY_CLAIM_FIELDS = (
    "task_ids_assigned",
    "repository_slots_assigned",
    "repository_identities_assigned",
    "development_or_holdout_partition_assigned",
    "difficulty_labels_assigned",
    "reference_patches_created",
    "graders_implemented",
    "candidate_execution_performed",
    "task_admission_claimed",
    "sealed_holdout_claimed",
    "performance_result_claimed",
    "fixtures_compiled",
    "fixture_tests_run",
)


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def semantic_errors(payload: dict) -> list[str]:
    errors: list[str] = []

    if payload.get("format_id") != "CEB-RUST-CANDIDATE-ORACLE-TRACEABILITY":
        errors.append("unexpected format_id")
    if payload.get("status") != "public_development_design_only":
        errors.append("crosswalk must remain public_development_design_only")
    if payload.get("official_benchmark_name") != "Computational Engineering Bench":
        errors.append("official benchmark name must remain Computational Engineering Bench")
    if payload.get("canonical_sampling_sources") != EXPECTED_SOURCE_PATHS:
        errors.append("canonical sampling source paths do not match the reviewed #10617 contract")
    if payload.get("sampling_source_branch") != EXPECTED_SOURCE_BRANCH:
        errors.append("sampling source branch does not match the reviewed #10617 branch")
    if payload.get("canonical_sampling_plan_blob_sha1") != EXPECTED_PLAN_BLOB:
        errors.append("canonical sampling plan blob pin changed; re-review the #10617 source")
    if payload.get("canonical_preregistration_blob_sha1") != EXPECTED_PREREG_BLOB:
        errors.append("canonical preregistration blob pin changed; re-review the #10617 source")

    claim = payload.get("claim_boundary")
    if not isinstance(claim, dict):
        errors.append("claim_boundary must be an object")
    else:
        for field in DESIGN_ONLY_CLAIM_FIELDS:
            if claim.get(field) is not False:
                errors.append(f"claim_boundary.{field} must remain false")
        if claim.get("fixtures_created") is not True:
            errors.append("claim_boundary.fixtures_created must be true")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        errors.append("candidates must be an array")
        candidates = []

    candidate_keys = [row.get("candidate_key") for row in candidates if isinstance(row, dict)]
    if len(candidate_keys) != len(candidates) or len(candidate_keys) != len(set(candidate_keys)):
        errors.append("candidate_key values must be present and unique")
    if set(candidate_keys) != set(EXPECTED_FAMILY_BY_CANDIDATE):
        errors.append("candidate set does not match the six reviewed Rust candidate themes")

    canonical_families = payload.get("canonical_family_ids")
    expected_families = set(EXPECTED_FAMILY_BY_CANDIDATE.values())
    if not isinstance(canonical_families, list) or set(canonical_families) != expected_families:
        errors.append("canonical_family_ids do not match the pinned #10617 family set")

    fixtures = payload.get("public_development_fixtures")
    if not isinstance(fixtures, list):
        errors.append("public_development_fixtures must be an array")
        fixtures = []
    fixture_keys = [row.get("candidate_key") for row in fixtures if isinstance(row, dict)]
    if len(fixture_keys) != len(fixtures) or len(fixture_keys) != len(set(fixture_keys)):
        errors.append("fixture candidate_key values must be present and unique")
    if set(fixture_keys) != set(candidate_keys):
        errors.append("fixture inventory must cover exactly the candidate set")

    fixture_by_key = {
        row["candidate_key"]: row
        for row in fixtures
        if isinstance(row, dict) and isinstance(row.get("candidate_key"), str)
    }

    all_requirement_ids: set[str] = set()
    all_case_ids: set[str] = set()
    family_ids: list[str] = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        key = candidate.get("candidate_key")
        expected_family = EXPECTED_FAMILY_BY_CANDIDATE.get(key)
        family = candidate.get("family_id")
        family_ids.append(family)
        if expected_family is None or family != expected_family:
            errors.append(f"{key}: family_id does not match its pinned #10617 family")

        requirements = candidate.get("requirements")
        if not isinstance(requirements, list) or not requirements:
            errors.append(f"{key}: requirements must be a non-empty array")
            requirements = []
        local_requirement_ids: set[str] = set()
        for requirement in requirements:
            if not isinstance(requirement, dict):
                errors.append(f"{key}: malformed requirement row")
                continue
            requirement_id = requirement.get("id")
            statement = requirement.get("statement")
            if not isinstance(requirement_id, str) or not requirement_id.strip():
                errors.append(f"{key}: requirement ID is missing")
                continue
            if requirement_id in all_requirement_ids:
                errors.append(f"duplicate requirement ID: {requirement_id}")
            all_requirement_ids.add(requirement_id)
            local_requirement_ids.add(requirement_id)
            if not isinstance(statement, str) or len(statement.strip()) < 20:
                errors.append(f"{key}/{requirement_id}: requirement statement is too short")

        oracle_cases = candidate.get("oracle_cases")
        if not isinstance(oracle_cases, list) or not oracle_cases:
            errors.append(f"{key}: oracle_cases must be a non-empty array")
            oracle_cases = []
        kind_counts = {kind: 0 for kind in DISPOSITION_BY_KIND}
        for case in oracle_cases:
            if not isinstance(case, dict):
                errors.append(f"{key}: malformed oracle case")
                continue
            case_id = case.get("id")
            if not isinstance(case_id, str) or not case_id.strip():
                errors.append(f"{key}: oracle case ID is missing")
            elif case_id in all_case_ids:
                errors.append(f"duplicate oracle case ID: {case_id}")
            else:
                all_case_ids.add(case_id)

            kind = case.get("kind")
            if kind not in DISPOSITION_BY_KIND:
                errors.append(f"{key}/{case_id}: unsupported oracle case kind {kind!r}")
            else:
                kind_counts[kind] += 1
                expected_disposition = DISPOSITION_BY_KIND[kind]
                if case.get("expected_verifier_disposition") != expected_disposition:
                    errors.append(
                        f"{key}/{case_id}: {kind} must map to {expected_disposition}"
                    )

            references = case.get("requirement_ids")
            if not isinstance(references, list) or not references:
                errors.append(f"{key}/{case_id}: requirement_ids must be non-empty")
            elif any(reference not in local_requirement_ids for reference in references):
                errors.append(f"{key}/{case_id}: dangling or cross-candidate requirement reference")

            for field in ("scenario", "why_spec_derived"):
                value = case.get(field)
                if not isinstance(value, str) or len(value.strip()) < 20:
                    errors.append(f"{key}/{case_id}: {field} must explain the specification link")

        if kind_counts["valid_alternative"] < 2:
            errors.append(f"{key}: at least two positive valid alternatives are required")
        if kind_counts["negative_near_miss"] < 2:
            errors.append(f"{key}: at least two negative near-misses are required")
        if kind_counts["required_composition"] != 1:
            errors.append(f"{key}: exactly one required-composition case is required")

        unresolved = candidate.get("unresolved_before_fixture")
        if not isinstance(unresolved, str) or len(unresolved.strip()) < 20:
            errors.append(f"{key}: unresolved_before_fixture must state remaining prerequisites")

        source_fixture = candidate.get("source_fixture")
        fixture = fixture_by_key.get(key)
        if not isinstance(source_fixture, dict) or not isinstance(fixture, dict):
            errors.append(f"{key}: source_fixture and inventory row must both exist")
            continue
        if source_fixture.get("path") != fixture.get("path"):
            errors.append(f"{key}: candidate and fixture inventory paths differ")
        if source_fixture.get("file_count") != fixture.get("file_count"):
            errors.append(f"{key}: candidate and fixture inventory file counts differ")
        if source_fixture.get("has_cargo_lock") is not False:
            errors.append(f"{key}: candidate fixture must not claim a Cargo.lock")
        if source_fixture.get("reference_patch_present") is not False:
            errors.append(f"{key}: reference patch must remain absent")
        if source_fixture.get("executable_oracle_present") is not False:
            errors.append(f"{key}: executable oracle must remain absent")
        if source_fixture.get("task_manifest_present") is not False:
            errors.append(f"{key}: task manifest must remain absent")

    if len(set(family_ids)) != len(EXPECTED_FAMILY_BY_CANDIDATE):
        errors.append("candidate family IDs must be one-to-one")

    for fixture in fixtures:
        if not isinstance(fixture, dict):
            continue
        if fixture.get("status") not in {
            "source_only_not_compiled_or_executed",
            "public_source_only_not_compiled_or_executed",
        }:
            errors.append(f"{fixture.get('candidate_key')}: fixture status is not source-only")
        if fixture.get("admission_status") != "not_admitted":
            errors.append(f"{fixture.get('candidate_key')}: fixture must remain not_admitted")
        if fixture.get("holdout_eligible") is not False:
            errors.append(f"{fixture.get('candidate_key')}: public fixture cannot be holdout eligible")
        if not isinstance(fixture.get("file_count"), int) or fixture["file_count"] < 1:
            errors.append(f"{fixture.get('candidate_key')}: file_count must be positive")

    return errors


def fixture_errors(payload: dict, repo_root: Path) -> list[str]:
    errors: list[str] = []
    fixtures = payload.get("public_development_fixtures", [])
    seen_paths: set[str] = set()
    total_declared = 0

    root = repo_root.resolve()
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            continue
        candidate_key = fixture.get("candidate_key", "<unknown>")
        relative = fixture.get("path")
        if not isinstance(relative, str) or not relative:
            errors.append(f"{candidate_key}: fixture path is missing")
            continue
        if relative in seen_paths:
            errors.append(f"{candidate_key}: duplicate fixture path")
        seen_paths.add(relative)

        candidate_path = Path(relative)
        if candidate_path.is_absolute() or ".." in candidate_path.parts:
            errors.append(f"{candidate_key}: fixture path must be repository-relative and traversal-free")
            continue
        absolute = (root / candidate_path).resolve()
        if not absolute.is_relative_to(root):
            errors.append(f"{candidate_key}: fixture path escapes repository root")
            continue
        if not absolute.is_dir():
            errors.append(f"{candidate_key}: fixture directory does not exist: {relative}")
            continue

        file_paths: list[Path] = []
        symlink_paths: list[Path] = []
        for path in absolute.rglob("*"):
            if path.is_symlink():
                symlink_paths.append(path)
            elif path.is_file():
                file_paths.append(path)
        if symlink_paths:
            errors.append(f"{candidate_key}: fixture contains symlink(s): {[p.name for p in symlink_paths]}")
        actual_count = len(file_paths)
        declared_count = fixture.get("file_count")
        total_declared += declared_count if isinstance(declared_count, int) else 0
        if actual_count != declared_count:
            errors.append(
                f"{candidate_key}: declared file_count={declared_count}, actual={actual_count}"
            )

        readme = absolute / "README.md"
        if not readme.is_file():
            errors.append(f"{candidate_key}: README.md missing")
        else:
            try:
                readme_text = readme.read_text(encoding="utf-8").lower()
            except (OSError, UnicodeError) as exc:
                errors.append(f"{candidate_key}: cannot read README.md: {exc}")
                readme_text = ""
            for phrase in ("not an admitted ceb task", "not a sealed holdout", "not been compiled"):
                if phrase not in readme_text:
                    errors.append(f"{candidate_key}: README.md must state {phrase!r}")

        if not (absolute / "Cargo.toml").is_file():
            errors.append(f"{candidate_key}: root Cargo.toml missing")
        manifests = list(absolute.rglob("Cargo.toml"))
        if len(manifests) < 3:
            errors.append(f"{candidate_key}: expected a multi-crate fixture with at least 3 Cargo manifests")
        if list(absolute.rglob("Cargo.lock")):
            errors.append(f"{candidate_key}: Cargo.lock must remain absent before environment pinning")
        if any(path.name == "target" and path.is_dir() for path in absolute.rglob("target")):
            errors.append(f"{candidate_key}: compiled target directory must not be present")

    if len(seen_paths) != len(fixtures):
        errors.append("fixture inventory path set is incomplete")
    return errors


def self_test() -> list[str]:
    errors: list[str] = []
    minimal = {
        "format_id": "CEB-RUST-CANDIDATE-ORACLE-TRACEABILITY",
        "status": "public_development_design_only",
        "official_benchmark_name": "Computational Engineering Bench",
        "canonical_sampling_sources": EXPECTED_SOURCE_PATHS.copy(),
        "sampling_source_branch": EXPECTED_SOURCE_BRANCH,
        "canonical_sampling_plan_blob_sha1": EXPECTED_PLAN_BLOB,
        "canonical_preregistration_blob_sha1": EXPECTED_PREREG_BLOB,
        "canonical_family_ids": list(EXPECTED_FAMILY_BY_CANDIDATE.values()),
        "claim_boundary": {
            **{field: False for field in DESIGN_ONLY_CLAIM_FIELDS},
            "fixtures_created": True,
        },
        "candidates": [],
        "public_development_fixtures": [],
    }

    # A deliberately tiny valid one-candidate shape proves the local checks
    # reject the mutations below; full-catalog validation follows afterward.
    one = {
        "candidate_key": "api-migration",
        "family_id": EXPECTED_FAMILY_BY_CANDIDATE["api-migration"],
        "requirements": [{"id": "API-01", "statement": "A normative requirement with a sufficiently clear observable behavior."}],
        "oracle_cases": [
            {"id": "P01", "kind": "valid_alternative", "requirement_ids": ["API-01"], "expected_verifier_disposition": "accepted_candidate", "scenario": "A valid alternate implementation preserves the required contract.", "why_spec_derived": "The specification constrains behavior, not source layout."},
            {"id": "P02", "kind": "valid_alternative", "requirement_ids": ["API-01"], "expected_verifier_disposition": "accepted_candidate", "scenario": "Another valid implementation preserves the required contract.", "why_spec_derived": "The specification permits multiple semantically equivalent designs."},
            {"id": "N01", "kind": "negative_near_miss", "requirement_ids": ["API-01"], "expected_verifier_disposition": "rejected_candidate", "scenario": "A patch violates the required observable behavior.", "why_spec_derived": "The stated requirement explicitly excludes this behavior."},
            {"id": "N02", "kind": "negative_near_miss", "requirement_ids": ["API-01"], "expected_verifier_disposition": "rejected_candidate", "scenario": "A second patch violates the required observable behavior.", "why_spec_derived": "The stated requirement explicitly excludes this behavior."},
            {"id": "C01", "kind": "required_composition", "requirement_ids": ["API-01"], "expected_verifier_disposition": "accepted_candidate", "scenario": "A required interaction preserves the contract.", "why_spec_derived": "The specification requires the interaction."},
        ],
        "unresolved_before_fixture": "Freeze the exact public interface and environment assumptions before fixture admission.",
        "source_fixture": {
            "path": "fixture/",
            "file_count": 3,
            "has_cargo_lock": False,
            "reference_patch_present": False,
            "executable_oracle_present": False,
            "task_manifest_present": False,
        },
    }
    fixture = {
        "candidate_key": "api-migration",
        "path": "fixture/",
        "status": "source_only_not_compiled_or_executed",
        "file_count": 3,
        "admission_status": "not_admitted",
        "holdout_eligible": False,
        "purpose": "Synthetic fixture for structural self-test.",
    }
    minimal["candidates"] = [one]
    minimal["public_development_fixtures"] = [fixture]

    if semantic_errors(minimal):
        errors.append("self-test baseline unexpectedly failed")
    for label, mutation in [
        ("duplicate requirement ID", lambda p: p["candidates"][0]["requirements"].append(copy.deepcopy(p["candidates"][0]["requirements"][0]))),
        ("dangling requirement reference", lambda p: p["candidates"][0]["oracle_cases"][0]["requirement_ids"].append("MISSING")),
        ("wrong disposition polarity", lambda p: p["candidates"][0]["oracle_cases"][2].update({"expected_verifier_disposition": "accepted_candidate"})),
        ("sealed public fixture", lambda p: p["public_development_fixtures"][0].update({"holdout_eligible": True})),
        ("wrong canonical family", lambda p: p["candidates"][0].update({"family_id": "wrong-family"})),
    ]:
        mutant = copy.deepcopy(minimal)
        mutation(mutant)
        if not semantic_errors(mutant):
            errors.append(f"negative self-test unexpectedly passed: {label}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        failures = self_test()
        if failures:
            for failure in failures:
                print(f"SELF_TEST_FAIL: {failure}", file=sys.stderr)
            return 1
        print("CEB_RUST_CANDIDATE_TRACEABILITY_SELF_TEST_PASS: 5 negative controls")

    crosswalk_path = args.repo_root / CROSSWALK
    try:
        payload = load_json(crosswalk_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    failures = semantic_errors(payload) + fixture_errors(payload, args.repo_root)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    candidates = payload["candidates"]
    requirements = sum(len(candidate["requirements"]) for candidate in candidates)
    oracle_cases = sum(len(candidate["oracle_cases"]) for candidate in candidates)
    fixtures = payload["public_development_fixtures"]
    files = sum(fixture["file_count"] for fixture in fixtures)
    print(
        "CEB_RUST_CANDIDATE_TRACEABILITY_PASS: "
        f"candidates={len(candidates)} requirements={requirements} "
        f"oracle_cases={oracle_cases} fixtures={len(fixtures)} declared_files={files}"
    )
    print(
        "CEB_RUST_CANDIDATE_CLAIM_BOUNDARY: "
        "static inventory only; no candidate code, Cargo command, tests, graders or benchmark tasks executed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
