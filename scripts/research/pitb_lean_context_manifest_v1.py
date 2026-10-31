#!/usr/bin/env python3
"""Derive PITB support-lattice graphs from Lean-exported local-context evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from pitb_validate_lean_context_evidence_v2 import (
    EvidenceError,
    synthetic_fixture,
    verify_evidence,
)


class ManifestError(ValueError):
    """Raised when a Lean context cannot be bound to a support-lattice role plan."""


ROLE_KEYS = {
    "schema_version",
    "study_id",
    "target_id",
    "expected_local_context_sha256",
    "expected_target_statement_sha256",
    "toolchain_sha256",
    "mutable_fvar_ids",
    "fixed_fvar_ids",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ManifestError(f"{label} must be a lowercase SHA-256 digest")
    if any(char not in "0123456789abcdef" for char in value):
        raise ManifestError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_sorted_ids(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or value != sorted(value)
        or len(value) != len(set(value))
    ):
        raise ManifestError(f"{label} must be sorted unique non-empty IDs")
    return value


def validate_role_plan(role_plan: Any) -> dict:
    if not isinstance(role_plan, dict) or set(role_plan) != ROLE_KEYS:
        actual = set(role_plan) if isinstance(role_plan, dict) else set()
        raise ManifestError(
            f"role plan keys mismatch: missing={sorted(ROLE_KEYS - actual)}, "
            f"extra={sorted(actual - ROLE_KEYS)}"
        )
    if role_plan["schema_version"] != "pitb.lean-context-role-plan.v1":
        raise ManifestError("unsupported role plan schema version")
    for field in ("study_id", "target_id"):
        if not isinstance(role_plan[field], str) or not role_plan[field].strip():
            raise ManifestError(f"{field} must be a non-empty string")
    for field in (
        "expected_local_context_sha256",
        "expected_target_statement_sha256",
        "toolchain_sha256",
    ):
        require_digest(role_plan[field], field)
    mutable = require_sorted_ids(role_plan["mutable_fvar_ids"], "mutable_fvar_ids")
    fixed = require_sorted_ids(role_plan["fixed_fvar_ids"], "fixed_fvar_ids")
    if set(mutable) & set(fixed):
        raise ManifestError("mutable and fixed FVarId assignments overlap")
    if not mutable:
        raise ManifestError("at least one mutable declaration is required")
    return role_plan


def build_manifest(evidence: dict, role_plan: dict, source_evidence_sha256: str) -> dict:
    verify_evidence(evidence)
    plan = validate_role_plan(role_plan)
    build_metadata = evidence["build_metadata"]
    if plan["toolchain_sha256"] != build_metadata["lean_toolchain_file_sha256"]:
        raise ManifestError("role plan toolchain digest does not match Lean build evidence")
    declarations = evidence["local_context"]["declarations"]
    order = [row["fvar_id"] for row in declarations]
    all_ids = set(order)
    mutable = plan["mutable_fvar_ids"]
    fixed = plan["fixed_fvar_ids"]
    if set(mutable) | set(fixed) != all_ids:
        missing = sorted(all_ids - (set(mutable) | set(fixed)))
        unknown = sorted((set(mutable) | set(fixed)) - all_ids)
        raise ManifestError(
            f"role plan must assign every local declaration exactly once: "
            f"unassigned={missing}, unknown={unknown}"
        )
    context_digest = digest(canonical(evidence["local_context"]))
    target_digest = digest(evidence["target"]["type"].encode("utf-8"))
    if context_digest != plan["expected_local_context_sha256"]:
        raise ManifestError("role plan local-context digest does not match Lean evidence")
    if target_digest != plan["expected_target_statement_sha256"]:
        raise ManifestError("role plan target-statement digest does not match Lean evidence")

    local_instances = evidence["local_context"]["local_instances"]
    local_instance_ids = [row["fvar_id"] for row in local_instances]
    typed_graph: dict[str, list[str]] = {}
    value_provenance_graph: dict[str, list[str]] = {}
    for row in declarations:
        fvar_id = row["fvar_id"]
        typed = set(row["type_fvar_ids"]) | set(row["transparent_value_fvar_ids"])
        provenance = typed | set(row["nondep_value_fvar_ids"])
        typed_graph[fvar_id] = sorted(typed)
        value_provenance_graph[fvar_id] = sorted(provenance)

    # Every dependency is earlier in the exact Lean telescope; the independent
    # evidence verifier already checked this. Keep the original order as data.
    return {
        "schema_version": "pitb.lean-context-manifest.v1",
        "study_id": plan["study_id"],
        "target_id": plan["target_id"],
        "source_evidence_sha256": require_digest(source_evidence_sha256, "source_evidence_sha256"),
        "role_plan_sha256": digest(canonical(plan)),
        "local_context_sha256": context_digest,
        "target_statement_sha256": target_digest,
        "toolchain_sha256": plan["toolchain_sha256"],
        "lean_toolchain": build_metadata["lean_toolchain"],
        "mutable_declarations": sorted(mutable),
        "fixed_declarations": sorted(fixed),
        "declaration_order": order,
        "local_context_declarations": declarations,
        "local_instances": local_instances,
        "local_instance_fvar_ids": local_instance_ids,
        "declaration_dependencies": typed_graph,
        "assigned_value_provenance_dependencies": value_provenance_graph,
        "target_typed_dependency_closure": sorted(evidence["target"]["typed_dependency_closure_ids"]),
        "proof_term_direct_support": sorted(evidence["proof_support"]["direct_free_variable_ids"]),
        "proof_term_typed_dependency_closure": sorted(evidence["proof_support"]["typed_dependency_closure_ids"]),
        "proof_term_assigned_value_provenance_closure": sorted(
            evidence["proof_support"]["assigned_value_provenance_closure_ids"]
        ),
        "interpretation_boundary": (
            "declaration_dependencies contains only type and visible transparent-let prerequisites. "
            "assigned_value_provenance_dependencies separately records hidden nondependent have values "
            "and is not used for context validity. Both are operational evidence, not logical necessity."
        ),
    }


def context_is_well_formed(selected: set[str], fixed: set[str], graph: dict[str, list[str]]) -> bool:
    present = selected | fixed
    if not present.issubset(graph):
        return False
    return all(dependency in present for name in present for dependency in graph[name])



def project_context(manifest: dict, selected: set[str]) -> dict:
    mutable = set(manifest["mutable_declarations"])
    fixed = set(manifest["fixed_declarations"])
    if not selected.issubset(mutable):
        raise ManifestError("selected context contains an unknown or non-mutable FVarId")
    if not context_is_well_formed(selected, fixed, manifest["declaration_dependencies"]):
        raise ManifestError("selected context is not dependency-closed with its fixed declarations")
    present = selected | fixed
    declarations = [
        row for row in manifest["local_context_declarations"]
        if row["fvar_id"] in present
    ]
    local_instances = [
        row for row in manifest["local_instances"]
        if row["fvar_id"] in present
    ]
    projected_ids = {row["fvar_id"] for row in declarations}
    if projected_ids != present:
        raise ManifestError("context projection failed to retain every selected/fixed declaration")
    if any(row["fvar_id"] not in projected_ids for row in local_instances):
        raise ManifestError("context projection retained a local instance without its declaration")
    payload = {
        "declaration_order": [row["fvar_id"] for row in declarations],
        "declarations": declarations,
        "local_instances": local_instances,
    }
    return {
        **payload,
        "local_instance_fvar_ids": [row["fvar_id"] for row in local_instances],
        "local_context_sha256": digest(canonical(payload)),
    }

def self_test() -> None:
    evidence = synthetic_fixture()
    verify_evidence(evidence)
    declarations = evidence["local_context"]["declarations"]
    ids = [row["fvar_id"] for row in declarations]
    context_digest = digest(canonical(evidence["local_context"]))
    target_digest = digest(evidence["target"]["type"].encode("utf-8"))
    plan = {
        "schema_version": "pitb.lean-context-role-plan.v1",
        "study_id": "adapter-self-test",
        "target_id": "opaque-have-calibration",
        "expected_local_context_sha256": context_digest,
        "expected_target_statement_sha256": target_digest,
        "toolchain_sha256": evidence["build_metadata"]["lean_toolchain_file_sha256"],
        "mutable_fvar_ids": ["fvar-have", "fvar-inst"],
        "fixed_fvar_ids": ["fvar-h", "fvar-let"],
    }
    manifest = build_manifest(evidence, plan, "b" * 64)
    assert manifest["declaration_order"] == ids
    assert manifest["declaration_dependencies"]["fvar-have"] == []
    assert manifest["assigned_value_provenance_dependencies"]["fvar-have"] == ["fvar-let"]
    assert manifest["declaration_dependencies"]["fvar-let"] == ["fvar-h"]
    assert manifest["local_instance_fvar_ids"] == ["fvar-inst"]
    full_context = project_context(manifest, {"fvar-have", "fvar-inst"})
    instance_ablated_context = project_context(manifest, {"fvar-have"})
    assert len(full_context["local_instances"]) == 1
    assert len(instance_ablated_context["local_instances"]) == 0
    assert instance_ablated_context["local_instance_fvar_ids"] == []
    assert full_context["local_context_sha256"] == digest(canonical({
        "declaration_order": full_context["declaration_order"],
        "declarations": full_context["declarations"],
        "local_instances": full_context["local_instances"],
    }))
    assert full_context["local_context_sha256"] != instance_ablated_context["local_context_sha256"]
    try:
        project_context(manifest, {"fvar-unknown"})
    except ManifestError:
        pass
    else:
        raise AssertionError("context projection accepted an unknown FVarId")
    assert context_is_well_formed({"fvar-have"}, {"fvar-h", "fvar-let"}, manifest["declaration_dependencies"])
    assert context_is_well_formed(set(), {"fvar-h", "fvar-let"}, manifest["declaration_dependencies"])

    fixed_dependency_graph = {"base": [], "fixed-child": ["base"]}
    assert context_is_well_formed({"base"}, {"fixed-child"}, fixed_dependency_graph)
    assert not context_is_well_formed(set(), {"fixed-child"}, fixed_dependency_graph)

    def rejected(candidate_evidence: dict, candidate_plan: dict, label: str) -> None:
        try:
            build_manifest(candidate_evidence, candidate_plan, "b" * 64)
        except (EvidenceError, ManifestError):
            return
        raise AssertionError(f"negative control accepted: {label}")

    import copy

    bad = copy.deepcopy(plan)
    bad["expected_local_context_sha256"] = "0" * 64
    rejected(evidence, bad, "role plan bound to a different local context")

    bad = copy.deepcopy(plan)
    bad["expected_target_statement_sha256"] = "0" * 64
    rejected(evidence, bad, "role plan bound to a different target")

    bad = copy.deepcopy(plan)
    bad["toolchain_sha256"] = "0" * 64
    rejected(evidence, bad, "role plan bound to a different Lean toolchain")

    bad = copy.deepcopy(plan)
    bad["fixed_fvar_ids"] = ["fvar-h"]
    rejected(evidence, bad, "unassigned local declaration")

    bad = copy.deepcopy(plan)
    bad["fixed_fvar_ids"] = ["fvar-h", "fvar-let"]
    bad["mutable_fvar_ids"] = ["fvar-have", "fvar-let"]
    rejected(evidence, bad, "overlapping fixed and mutable assignment")

    bad = copy.deepcopy(plan)
    bad["mutable_fvar_ids"] = ["fvar-unknown"]
    rejected(evidence, bad, "unknown FVarId")

    bad = copy.deepcopy(plan)
    bad["mutable_fvar_ids"] = ["fvar-have", "fvar-h"]
    bad["fixed_fvar_ids"] = ["fvar-let"]
    rejected(evidence, bad, "unsorted FVarId role manifest")

    print(
        "PITB_LEAN_CONTEXT_ADAPTER_SELF_TEST_PASS: "
        "typed-versus-provenance edges, fixed-dependency closure, "
        "complete FVar assignment and digest binding controls"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="?", type=Path)
    parser.add_argument("role_plan", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            if args.evidence or args.role_plan or args.output:
                raise ManifestError("--self-test cannot be combined with file paths")
            self_test()
            return 0
        if not args.evidence or not args.role_plan or not args.output:
            raise ManifestError("evidence, role_plan, and output paths are required")
        evidence_bytes = args.evidence.read_bytes()
        role_bytes = args.role_plan.read_bytes()
        evidence = json.loads(evidence_bytes)
        role_plan = json.loads(role_bytes)
        manifest = build_manifest(evidence, role_plan, digest(evidence_bytes))
        args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        print(f"PITB_LEAN_CONTEXT_MANIFEST_PASS: output={args.output}")
        print(f"local_context_sha256={manifest['local_context_sha256']}")
        print(f"target_statement_sha256={manifest['target_statement_sha256']}")
        return 0
    except (EvidenceError, ManifestError, OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError) as exc:
        print(f"PITB Lean context manifest rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
