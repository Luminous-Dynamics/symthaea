#!/usr/bin/env python3
"""Fail-closed static contract for SYM-FV-007B freshness/blocking algebra."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEAN = ROOT / "formal/lean/evidence/EvidenceFreshness.lean"
MANIFEST = ROOT / "docs/formal/sym-fv-007b-evidence-freshness-v1.json"

PARENT_HEAD = "f4b7fbc16aa132d60860f52a49843e9a6ccc8b7e"
PARENT_BLOB = "4684326db9d980647914bca180c88cb7fad4ec7e"
THEOREM_BLOB = "21e8a2633be61b96d1c66c7b589fdd179ce2758e"
THEOREMS = [
    "stale_reference_not_admissible",
    "fail_result_not_admissible",
    "blocked_result_not_admissible",
    "environment_failure_not_admissible",
    "advancing_live_generation_invalidates_prior_reference",
    "composite_admission_implies_root_current_and_pass",
    "composite_admission_implies_dependency_current_and_pass",
    "stale_dependency_blocks_composite",
    "nonpass_dependency_blocks_composite",
    "adding_dependency_requires_that_dependency_admissible",
    "adding_admissible_dependency_preserves_composite",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def git_blob(path: str) -> str:
    return subprocess.check_output(["git", "rev-parse", f"HEAD:{path}"], text=True).strip()


def validate(source: str, manifest: dict, *, check_blobs: bool = True) -> None:
    require(manifest["schema"] == "symthaea.formal.sym-fv-007b-evidence-freshness.v1", "schema drift")
    require(manifest["tracking_issue"] == 5972, "tracking issue drift")
    require(manifest["parent"]["head"] == PARENT_HEAD, "parent head drift")
    require(manifest["parent"]["inherited_theorem_blob"] == PARENT_BLOB, "parent theorem blob drift")
    require(manifest["authority"] == "EvidenceOnly", "authority escalation")
    require(manifest["evidence_class"] == "AbstractFormalTheorem", "evidence class drift")
    require(manifest["qualification_results"] == ["Pass", "Fail", "Blocked", "EnvironmentFailure"], "result census drift")
    require(manifest["admission_rule"] == "ExactGenerationAndPassOnly", "admission rule drift")
    require(manifest["theorem_blob"] == THEOREM_BLOB, "child theorem blob drift")
    require(manifest["theorems"] == THEOREMS, "theorem census drift")
    require("DagAcyclicity" in manifest["nonclaims"], "DAG nonclaim required")
    require("DistributedCurrentness" in manifest["nonclaims"], "distributed-currentness nonclaim required")
    require("RuntimeAuthorization" in manifest["nonclaims"], "runtime-authority nonclaim required")

    if check_blobs:
        require(git_blob("formal/lean/evidence/EvidenceComposition.lean") == PARENT_BLOB, "observed parent theorem blob drift")
        require(git_blob("formal/lean/evidence/EvidenceFreshness.lean") == THEOREM_BLOB, "observed child theorem blob drift")

    # Exact semantic roots.
    require("ref.observedGeneration = ref.liveGeneration" in source, "currentness equality removed")
    require("IsCurrent ref ∧ ref.result = QualificationResult.pass" in source, "Pass-only admission weakened")
    require("Admissible root ∧ DependenciesAdmissible dependencies" in source, "dependency admission removed")
    require("∀ dependency, dependency ∈ dependencies → Admissible dependency" in source, "dependency universal admission removed")

    for ctor in ["pass", "fail", "blocked", "environmentFailure"]:
        require(re.search(rf"\|\s+{re.escape(ctor)}\b", source) is not None, f"missing qualification constructor: {ctor}")

    for theorem in THEOREMS:
        require(re.search(rf"\btheorem\s+{re.escape(theorem)}\b", source) is not None, f"missing theorem: {theorem}")
        require(f"#print axioms {theorem}" in source, f"missing axiom probe: {theorem}")

    require(source.count("#print axioms ") == len(THEOREMS), "axiom-probe census drift")

    forbidden = [r"\bsorry\b", r"\badmit\b", r"^\s*axiom\b", r"^\s*opaque\b"]
    for pattern in forbidden:
        require(re.search(pattern, source, flags=re.MULTILINE) is None, f"forbidden proof escape: {pattern}")


def expect_reject(name: str, source: str, manifest: dict) -> None:
    try:
        validate(source, manifest, check_blobs=False)
    except Exception:
        return
    raise AssertionError(f"hostile mutant unexpectedly accepted: {name}")


def self_test(source: str, manifest: dict) -> None:
    expect_reject(
        "remove-generation-currentness",
        source.replace("ref.observedGeneration = ref.liveGeneration", "True", 1),
        manifest,
    )
    expect_reject(
        "blocked-as-pass",
        source.replace("| blocked", "| pass", 1),
        manifest,
    )
    expect_reject(
        "environment-failure-as-pass",
        source.replace("| environmentFailure", "| pass", 1),
        manifest,
    )
    expect_reject(
        "ignore-dependencies",
        source.replace("Admissible root ∧ DependenciesAdmissible dependencies", "Admissible root", 1),
        manifest,
    )
    expect_reject(
        "stale-auto-refresh",
        source.replace("observedGeneration : Nat", "observedGeneration : Nat\n  autoRefresh : Bool", 1).replace(
            "ref.observedGeneration = ref.liveGeneration",
            "ref.autoRefresh = true ∨ ref.observedGeneration = ref.liveGeneration",
            1,
        ),
        manifest,
    )
    expect_reject(
        "theorem-drop",
        source.replace("theorem stale_dependency_blocks_composite", "theorem stale_dependency_blocks_composite_REMOVED", 1),
        manifest,
    )
    expect_reject(
        "proof-hole",
        source.replace("exact hStale h.1", "sorry", 1),
        manifest,
    )

    mutant = json.loads(json.dumps(manifest))
    mutant["authority"] = "RuntimeAuthority"
    expect_reject("authority-escalation", source, mutant)

    mutant = json.loads(json.dumps(manifest))
    mutant["parent"]["head"] = "0" * 40
    expect_reject("parent-drift", source, mutant)


def main() -> int:
    source = LEAN.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate(source, manifest)
    self_test(source, manifest)
    print("SYM-FV-007B static contract: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SYM-FV-007B static contract: FAIL: {exc}", file=sys.stderr)
        raise
