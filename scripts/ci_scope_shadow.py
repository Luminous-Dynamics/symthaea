#!/usr/bin/env python3
"""Shadow-only CI scope advisor for Symthaea.

Predictions from this tool are observational only. They never authorize skipping,
gating, cancelling, or qualifying a workflow. Full CI remains authoritative while
shadow false-negative risk is measured.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "symthaea-ci-scope-shadow-v1"
ACTUAL_SCHEMA = "symthaea-ci-actual-jobs-v1"
EVAL_SCHEMA = "symthaea-ci-scope-shadow-evaluation-v1"

ALWAYS_ON = (
    "Governance Check (Class A/B Changes)",
    "Format Check",
    "Workspace Target Integrity",
    "Orphan Module Check",
    "Secrets Scan",
)
MUSE_PREFIXES = (
    "crates/domains/symthaea-muse/",
    "crates/domains/symthaea-muse-ui/",
    "crates/domains/symthaea-muse-protocol/",
    "crates/domains/symthaea-music-theory/",
)
CI_INFRA_EXACT = {
    "scripts/classify_ci_runner_state.py",
    "scripts/test_classify_ci_runner_state.py",
    "scripts/ci_scope_shadow.py",
    "scripts/test_ci_scope_shadow.py",
    ".github/workflows/ci-runner-state-classifier.yml",
    ".github/workflows/ci-scope-shadow.yml",
}
CI_INFRA_PREFIXES = (
    "tests/fixtures/ci_runner_state/",
    "tests/fixtures/ci_scope_shadow/",
)
CROSS_CUTTING_EXACT = {
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain",
    "rust-toolchain.toml",
    "build.rs",
    ".github/workflows/ci.yml",
}
CROSS_CUTTING_PREFIXES = (".cargo/", "crates/core/", "src/", "xtask/")
FAILURES = {"failure", "timed_out", "action_required", "startup_failure"}
INCONCLUSIVE = {None, "cancelled", "skipped", "neutral"}


class ShadowInputError(ValueError):
    pass


def normalize_paths(lines: list[str]) -> list[str]:
    values: list[str] = []
    for raw in lines:
        path = raw.strip().replace("\\", "/")
        if not path:
            continue
        while path.startswith("./"):
            path = path[2:]
        if path.startswith("/") or path == ".." or path.startswith("../") or "//" in path:
            raise ShadowInputError(f"invalid repository-relative path: {raw!r}")
        values.append(path)
    values = sorted(set(values))
    if not values:
        raise ShadowInputError("changed-file set must be non-empty")
    return values


def is_docs(path: str) -> bool:
    return path.startswith("docs/") or path.endswith(".md") or path == "LICENSE"


def is_ci_infra(path: str) -> bool:
    return path in CI_INFRA_EXACT or any(path.startswith(p) for p in CI_INFRA_PREFIXES)


def is_muse(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in MUSE_PREFIXES)


def is_cross_cutting(path: str) -> bool:
    if path in CROSS_CUTTING_EXACT or any(path.startswith(p) for p in CROSS_CUTTING_PREFIXES):
        return True
    if path.startswith("scripts/") and not is_ci_infra(path):
        return True
    if path.startswith(".github/") and not is_ci_infra(path):
        return True
    return False


def path_digest(paths: list[str]) -> str:
    return hashlib.sha256("".join(f"{p}\n" for p in paths).encode()).hexdigest()


def recommend(lines: list[str]) -> dict[str, Any]:
    paths = normalize_paths(lines)
    substantive = [path for path in paths if not is_docs(path)]
    reasons: list[str] = []
    surfaces: set[str] = set()

    if not substantive:
        klass = "docs_only"
        full = False
        patterns = list(ALWAYS_ON)
        surfaces.add("documentation")
        reasons.append("all changed files are documentation/licensing surfaces")
    elif all(is_ci_infra(path) for path in substantive):
        klass = "ci_evidence_infra_only"
        full = False
        patterns = list(ALWAYS_ON) + ["CI Runner State Classifier", "CI Scope Shadow Advisor"]
        surfaces.add("ci-evidence-infrastructure")
        reasons.append("all substantive files are explicitly modeled CI evidence/tooling surfaces")
    elif all(is_muse(path) for path in substantive):
        klass = "muse_isolated"
        full = False
        patterns = list(ALWAYS_ON) + ["Muse (tests, studio, wasm UI)"]
        surfaces.add("muse")
        reasons.append("all substantive files are within the explicitly modeled Muse surface")
    else:
        klass = "full_matrix_conservative"
        full = True
        patterns = ["*"]
        if any(is_cross_cutting(path) for path in substantive):
            surfaces.add("cross-cutting")
            reasons.append("at least one shared/core/build/CI-control path is cross-cutting")
        domain_roots = sorted({
            "/".join(path.split("/")[:3])
            for path in substantive
            if path.startswith("crates/domains/") and len(path.split("/")) >= 3
        })
        surfaces.update(domain_roots)
        unknown = [path for path in substantive if not is_cross_cutting(path) and not is_muse(path)]
        if unknown:
            reasons.append("one or more substantive paths are not yet modeled; unknown escalates to full matrix")
        if len(domain_roots) > 1:
            reasons.append("multiple domain roots changed; v1 does not assume isolation across them")

    return {
        "schema": SCHEMA,
        "mode": "shadow-only",
        "enforcement_allowed": False,
        "changed_file_count": len(paths),
        "documentation_file_count": len(paths) - len(substantive),
        "changed_files_sha256": path_digest(paths),
        "changed_files": paths,
        "surfaces": sorted(surfaces),
        "recommendation_class": klass,
        "full_matrix_recommended": full,
        "predicted_job_patterns": patterns,
        "reasons": reasons,
        "claim_boundary": (
            "Observational scope prediction only. It MUST NOT skip, gate, cancel, "
            "or qualify jobs until shadow false-negative risk is accepted."
        ),
    }


def validate_actual(actual: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(actual, dict) or actual.get("schema") != ACTUAL_SCHEMA:
        raise ShadowInputError(f"actual jobs must use schema {ACTUAL_SCHEMA!r}")
    for field in ("run_id", "run_attempt", "head_sha", "complete", "jobs"):
        if field not in actual:
            raise ShadowInputError(f"actual.{field} is required")
    if isinstance(actual["run_attempt"], bool) or not isinstance(actual["run_attempt"], int) or actual["run_attempt"] <= 0:
        raise ShadowInputError("actual.run_attempt must be a positive integer")
    if not isinstance(actual["head_sha"], str) or not actual["head_sha"]:
        raise ShadowInputError("actual.head_sha must be a non-empty string")
    if not isinstance(actual["complete"], bool):
        raise ShadowInputError("actual.complete must be boolean")
    jobs = actual["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ShadowInputError("actual.jobs must be a non-empty array")
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or not isinstance(job.get("name"), str) or not job["name"]:
            raise ShadowInputError(f"actual.jobs[{index}].name must be non-empty")
    return actual, jobs


def evaluate(prediction: dict[str, Any], actual_raw: Any) -> dict[str, Any]:
    if prediction.get("schema") != SCHEMA:
        raise ShadowInputError("prediction has unsupported schema")
    patterns = prediction.get("predicted_job_patterns")
    if not isinstance(patterns, list) or not patterns or not all(isinstance(p, str) for p in patterns):
        raise ShadowInputError("prediction.predicted_job_patterns must be a non-empty string array")
    actual, jobs = validate_actual(actual_raw)

    misses: list[dict[str, Any]] = []
    inconclusive: list[dict[str, Any]] = []
    failure_count = 0
    for job in jobs:
        name = job["name"]
        conclusion = job.get("conclusion")
        if conclusion in FAILURES:
            failure_count += 1
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
                misses.append({"name": name, "conclusion": conclusion})
        elif conclusion in INCONCLUSIVE:
            inconclusive.append({"name": name, "conclusion": conclusion})

    observation_complete_without_miss = (
        actual["complete"] and not misses and not inconclusive
    )
    return {
        "schema": EVAL_SCHEMA,
        "prediction_changed_files_sha256": prediction.get("changed_files_sha256"),
        "recommendation_class": prediction.get("recommendation_class"),
        "actual_run": {
            "run_id": actual["run_id"],
            "run_attempt": actual["run_attempt"],
            "head_sha": actual["head_sha"],
            "complete": actual["complete"],
        },
        "observed_failure_count": failure_count,
        "candidate_false_negative_count": len(misses),
        "candidate_false_negatives": misses,
        "inconclusive_job_count": len(inconclusive),
        "inconclusive_jobs": inconclusive,
        "observation_complete_without_candidate_miss": observation_complete_without_miss,
        "claim_boundary": (
            "Even a complete shadow observation with zero candidate misses is not "
            "an enforcement verdict; policy activation requires accumulated evidence "
            "and an explicitly accepted false-negative bound."
        ),
    }


def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ShadowInputError(f"cannot read {path}: {exc}") from exc


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ShadowInputError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ShadowInputError(f"invalid JSON in {path}: {exc}") from exc


def emit(value: dict[str, Any], pretty: bool) -> None:
    json.dump(value, sys.stdout, indent=2 if pretty else None, sort_keys=True)
    sys.stdout.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Symthaea shadow-only CI scope advisor")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("recommend")
    rec.add_argument("changed_files", type=Path)
    rec.add_argument("--pretty", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("prediction", type=Path)
    ev.add_argument("actual_jobs", type=Path)
    ev.add_argument("--pretty", action="store_true")
    ev.add_argument("--assert-no-candidate-false-negatives", action="store_true")
    args = parser.parse_args()

    try:
        result = (
            recommend(read_lines(args.changed_files))
            if args.command == "recommend"
            else evaluate(read_json(args.prediction), read_json(args.actual_jobs))
        )
    except ShadowInputError as exc:
        print(f"ci-scope-shadow input error: {exc}", file=sys.stderr)
        return 2

    emit(result, args.pretty)
    if args.command == "evaluate" and args.assert_no_candidate_false_negatives:
        return 1 if result["candidate_false_negative_count"] else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
