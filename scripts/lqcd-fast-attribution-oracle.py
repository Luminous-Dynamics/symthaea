#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

SCHEMA = "symthaea.lqcd.fast-lane.attribution-result.v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")

def canonical_bytes(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def classify_path(path: str) -> str:
    if path.startswith("crates/domains/symthaea-particle-physics/"):
        return "scientific_subject"
    if path in {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml"}:
        return "scientific_subject"
    if path.startswith(".cargo/"):
        return "scientific_subject"
    if path == "scripts/check-class-a-changes.sh":
        return "governance"
    if path in {
        ".github/workflows/lqcd-fast.yml",
        "scripts/qualify-lqcd-fast.py",
        "scripts/build-lqcd-fast-diagnostic.py",
    }:
        return "verifier_or_diagnostic"
    return "other"

def validate_input(data: dict) -> None:
    assert data["schema_version"] == "symthaea.lqcd.fast-lane.attribution-input.v1"
    assert data["source"] == "github_compare_commits"
    assert data["transitions"]
    for t in data["transitions"]:
        assert HEX40.match(t["base_sha"])
        assert HEX40.match(t["head_sha"])
        assert t["base_sha"] != t["head_sha"]
        assert isinstance(t["ahead_by"], int) and t["ahead_by"] >= 1
        seen = set()
        for f in t["files"]:
            p = f["path"]
            assert isinstance(p, str) and p and p not in seen
            seen.add(p)
            assert f["status"] in {"added", "modified", "removed", "renamed"}
            assert isinstance(f["additions"], int) and f["additions"] >= 0
            assert isinstance(f["deletions"], int) and f["deletions"] >= 0

def classify_transition(t: dict) -> dict:
    buckets = {"scientific_subject": [], "verifier_or_diagnostic": [], "governance": [], "other": []}
    for f in t["files"]:
        buckets[classify_path(f["path"])].append(f["path"])

    subject_delta = bool(buckets["scientific_subject"])
    other_delta = bool(buckets["other"])
    canonical_qualifier_changed = "scripts/qualify-lqcd-fast.py" in buckets["verifier_or_diagnostic"]
    workflow_changed = ".github/workflows/lqcd-fast.yml" in buckets["verifier_or_diagnostic"]
    diagnostic_changed = "scripts/build-lqcd-fast-diagnostic.py" in buckets["verifier_or_diagnostic"]

    if other_delta:
        structural = "UnclassifiedDelta"
    elif subject_delta:
        structural = "SubjectInputDeltaPresent"
    elif canonical_qualifier_changed:
        structural = "VerifierProfileDeltaWithoutSubjectDelta"
    elif workflow_changed or diagnostic_changed:
        structural = "VerifierOperationalDeltaWithoutSubjectDelta"
    else:
        structural = "NoRelevantDelta"

    return {
        "name": t["name"],
        "base_sha": t["base_sha"],
        "head_sha": t["head_sha"],
        "changed_file_count": len(t["files"]),
        "changed_paths": [f["path"] for f in t["files"]],
        "classification": structural,
        "buckets": buckets,
        "scientific_subject_input_delta": subject_delta,
        "canonical_qualifier_changed": canonical_qualifier_changed,
        "workflow_changed": workflow_changed,
        "diagnostic_changed": diagnostic_changed,
        "candidate_caused_rust_lint_established": False,
        "candidate_rust_repair_authorized": False,
        "reason": (
            "No package/Cargo/lock/toolchain subject input changed; a head-level Clippy failure "
            "cannot by itself establish candidate-caused Rust regression."
            if not subject_delta and not other_delta
            else
            "Structural delta alone is insufficient to identify or attribute a specific lint."
        ),
    }

def self_test() -> None:
    def c(paths):
        t = {
            "name": "synthetic",
            "base_sha": "0"*40, "head_sha": "1"*40, "ahead_by": 1,
            "files": [{"path": p, "status": "modified", "additions": 1, "deletions": 1} for p in paths],
        }
        return classify_transition(t)

    assert c(["crates/domains/symthaea-particle-physics/src/lib.rs"])["scientific_subject_input_delta"]
    assert c(["Cargo.lock"])["classification"] == "SubjectInputDeltaPresent"
    assert c(["rust-toolchain.toml"])["classification"] == "SubjectInputDeltaPresent"
    assert c([".cargo/config.toml"])["classification"] == "SubjectInputDeltaPresent"
    assert c(["scripts/qualify-lqcd-fast.py"])["classification"] == "VerifierProfileDeltaWithoutSubjectDelta"
    assert c([".github/workflows/lqcd-fast.yml"])["classification"] == "VerifierOperationalDeltaWithoutSubjectDelta"
    assert c(["docs/unrelated.md"])["classification"] == "UnclassifiedDelta"
    assert not c(["docs/unrelated.md"])["candidate_rust_repair_authorized"]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    self_test()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    validate_input(data)
    transitions = [classify_transition(t) for t in data["transitions"]]

    zero_subject = all(not t["scientific_subject_input_delta"] for t in transitions)
    no_other = all(not t["buckets"]["other"] for t in transitions)
    result = {
        "schema_version": SCHEMA,
        "input_sha256": sha256_bytes(raw),
        "transition_count": len(transitions),
        "transitions": transitions,
        "global": {
            "all_transitions_zero_scientific_subject_input_delta": zero_subject,
            "all_changed_paths_classified": no_other,
            "candidate_caused_rust_lint_established": False,
            "candidate_rust_repair_authorized": False,
            "base_execution_required_for_inherited_failure_claim": True,
            "typed_lint_identity_required_before_any_repair": True,
            "structural_theorem": (
                "The observed fast-lane/diagnostic lineage changed verifier or diagnostic bytes only; "
                "it did not change package/Cargo/lock/toolchain inputs. Therefore the known head-level "
                "Clippy failure is a conformance observation, not evidence of a candidate-caused Rust regression."
                if zero_subject and no_other else
                "Structural attribution inconclusive."
            ),
        },
        "synthetic_controls_passed": True,
    }
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.output.write_text(payload)
    print(f"input_sha256={result['input_sha256']}")
    print(f"result_sha256={sha256_bytes(payload.encode())}")
    print(f"transition_count={len(transitions)}")
    for t in transitions:
        print(
            f"{t['name']}: classification={t['classification']} "
            f"scientific_subject_input_delta={str(t['scientific_subject_input_delta']).lower()} "
            f"candidate_rust_repair_authorized=false"
        )
    print(f"all_transitions_zero_scientific_subject_input_delta={str(zero_subject).lower()}")
    print(f"all_changed_paths_classified={str(no_other).lower()}")
    print("candidate_caused_rust_lint_established=false")
    print("candidate_rust_repair_authorized=false")
    print("base_execution_required_for_inherited_failure_claim=true")
    print("synthetic_controls_passed=true")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
