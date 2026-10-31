#!/usr/bin/env python3
"""Independent structural verifier for PITB Lean local-context evidence v2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    """Raised when local-context evidence violates its typed contract."""


DECLARATION_KEYS = {
    "fvar_id", "index", "user_name", "type", "type_fvar_ids",
    "type_global_constants", "is_let_declaration", "is_nondep",
    "is_implementation_detail", "transparent_value",
    "transparent_value_fvar_ids", "transparent_value_global_constants",
    "nondep_value_provenance", "nondep_value_fvar_ids",
    "nondep_value_global_constants", "binder_info", "is_aux_declaration",
}
TOP_KEYS = {
    "schema_version", "label", "build_metadata", "target", "local_context",
    "proof_support", "interpretation_boundary",
}
BUILD_METADATA_KEYS = {
    "lean_toolchain", "lean_toolchain_file_sha256", "lakefile_sha256",
    "root_module_path", "root_module_sha256",
    "context_exporter_path", "context_exporter_source_sha256",
}
TARGET_KEYS = {
    "type", "free_variable_ids", "typed_dependency_closure_ids",
    "typed_dependency_closure", "global_constants",
}
CONTEXT_KEYS = {"declaration_order", "declarations", "local_instances"}
LOCAL_INSTANCE_KEYS = {"class_name", "fvar_id", "type"}
SUPPORT_KEYS = {
    "proof_term", "direct_free_variable_ids", "direct_declarations",
    "typed_dependency_closure_ids", "typed_dependency_closure",
    "typed_dependency_global_constants", "assigned_value_provenance_closure_ids",
    "assigned_value_provenance_closure", "assigned_value_provenance_global_constants",
    "opaque_value_provenance", "proof_term_global_constants",
}


def _keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise EvidenceError(
            f"{label}: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{label} must be a non-empty string")
    return value


def _ids(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(value) != len(set(value))
    ):
        raise EvidenceError(f"{label} must be an array of unique non-empty strings")
    return value


def _constants(value: Any, label: str) -> list[str]:
    return _ids(value, label)


def _same_set(left: list[str], right: set[str], label: str) -> None:
    if set(left) != right or len(left) != len(right):
        raise EvidenceError(f"{label} does not equal the independently reconstructed set")


def _closure(
    roots: list[str],
    declarations: dict[str, dict[str, Any]],
    include_nondep_values: bool,
    label: str,
) -> set[str]:
    visited: set[str] = set()
    pending = list(roots)
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        row = declarations.get(current)
        if row is None:
            raise EvidenceError(f"{label}: unknown free-variable ID {current}")
        visited.add(current)
        pending.extend(row["type_fvar_ids"])
        pending.extend(row["transparent_value_fvar_ids"])
        if include_nondep_values:
            pending.extend(row["nondep_value_fvar_ids"])
    return visited


def _rows_match(
    rows: Any,
    expected_ids: set[str],
    declarations: dict[str, dict[str, Any]],
    label: str,
) -> None:
    if not isinstance(rows, list):
        raise EvidenceError(f"{label} must be an array")
    actual_ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise EvidenceError(f"{label}[{index}] must be an object")
        fvar_id = _string(row.get("fvar_id"), f"{label}[{index}].fvar_id")
        actual_ids.append(fvar_id)
        if declarations.get(fvar_id) != row:
            raise EvidenceError(f"{label}[{index}] is not the exact captured declaration row")
    _same_set(actual_ids, expected_ids, label)


def _global_constant_union(
    ids: set[str],
    declarations: dict[str, dict[str, Any]],
    include_nondep_values: bool,
) -> set[str]:
    result: set[str] = set()
    for fvar_id in ids:
        row = declarations[fvar_id]
        result.update(row["type_global_constants"])
        result.update(row["transparent_value_global_constants"])
        if include_nondep_values:
            result.update(row["nondep_value_global_constants"])
    return result


def verify_evidence(evidence: Any) -> dict[str, int]:
    _keys(evidence, TOP_KEYS, "evidence")
    if evidence["schema_version"] != "pitb.local-context-evidence.v2":
        raise EvidenceError("unsupported schema_version")
    _string(evidence["label"], "label")
    _string(evidence["interpretation_boundary"], "interpretation_boundary")

    build_metadata = evidence["build_metadata"]
    _keys(build_metadata, BUILD_METADATA_KEYS, "build_metadata")
    _string(build_metadata["lean_toolchain"], "build_metadata.lean_toolchain")
    for field in (
        "lean_toolchain_file_sha256", "lakefile_sha256", "root_module_sha256",
        "context_exporter_source_sha256",
    ):
        digest = _string(build_metadata[field], f"build_metadata.{field}")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise EvidenceError(f"build_metadata.{field} must be lowercase SHA-256 hex")
    if build_metadata["root_module_path"] != "PITBCounterfactualPilot.lean":
        raise EvidenceError("unexpected root module path")
    if build_metadata["context_exporter_path"] != "PITBCounterfactualPilot/ContextEvidenceV2.lean":
        raise EvidenceError("unexpected context exporter path")

    target = evidence["target"]
    _keys(target, TARGET_KEYS, "target")
    _string(target["type"], "target.type")
    target_roots = _ids(target["free_variable_ids"], "target.free_variable_ids")
    target_closure_ids = _ids(
        target["typed_dependency_closure_ids"], "target.typed_dependency_closure_ids"
    )
    _constants(target["global_constants"], "target.global_constants")

    context = evidence["local_context"]
    _keys(context, CONTEXT_KEYS, "local_context")
    order = _ids(context["declaration_order"], "local_context.declaration_order")
    rows = context["declarations"]
    if not isinstance(rows, list):
        raise EvidenceError("local_context.declarations must be an array")
    declarations: dict[str, dict[str, Any]] = {}
    previous_index = -1
    actual_order: list[str] = []
    for position, row in enumerate(rows):
        _keys(row, DECLARATION_KEYS, f"declarations[{position}]")
        fvar_id = _string(row["fvar_id"], f"declarations[{position}].fvar_id")
        if fvar_id in declarations:
            raise EvidenceError(f"duplicate FVarId {fvar_id}")
        index = row["index"]
        if type(index) is not int or index <= previous_index:
            raise EvidenceError("local declaration indices must be strictly increasing")
        previous_index = index
        _string(row["user_name"], f"declarations[{position}].user_name")
        _string(row["type"], f"declarations[{position}].type")
        if row["binder_info"] not in {
            "Lean.BinderInfo.default", "Lean.BinderInfo.implicit",
            "Lean.BinderInfo.strictImplicit", "Lean.BinderInfo.instImplicit",
        }:
            raise EvidenceError(f"declarations[{position}].binder_info is unsupported")
        if type(row["is_aux_declaration"]) is not bool:
            raise EvidenceError(f"declarations[{position}].is_aux_declaration must be boolean")
        for field in (
            "type_fvar_ids", "transparent_value_fvar_ids", "nondep_value_fvar_ids"
        ):
            _ids(row[field], f"declarations[{position}].{field}")
        for field in (
            "type_global_constants", "transparent_value_global_constants",
            "nondep_value_global_constants"
        ):
            _constants(row[field], f"declarations[{position}].{field}")
        for field in ("is_let_declaration", "is_nondep", "is_implementation_detail"):
            if type(row[field]) is not bool:
                raise EvidenceError(f"declarations[{position}].{field} must be boolean")
        transparent_value = row["transparent_value"]
        nondep_value = row["nondep_value_provenance"]
        if transparent_value is not None:
            _string(transparent_value, f"declarations[{position}].transparent_value")
        if nondep_value is not None:
            _string(nondep_value, f"declarations[{position}].nondep_value_provenance")

        is_nondep = row["is_nondep"]
        is_let = row["is_let_declaration"]
        if is_nondep and not is_let:
            raise EvidenceError("a nondependent declaration must be a let/have declaration")
        if is_nondep:
            if transparent_value is not None or row["transparent_value_fvar_ids"]:
                raise EvidenceError("nondependent values cannot be typed as transparent let values")
            if nondep_value is None:
                raise EvidenceError("nondependent declaration is missing assigned-value provenance")
        else:
            if nondep_value is not None or row["nondep_value_fvar_ids"] or row["nondep_value_global_constants"]:
                raise EvidenceError("ordinary declarations must not carry nondependent value provenance")
        if is_let != (transparent_value is not None or nondep_value is not None):
            raise EvidenceError("is_let_declaration disagrees with the visible/opaque value fields")

        declarations[fvar_id] = row
        actual_order.append(fvar_id)

    if order != actual_order:
        raise EvidenceError("declaration_order does not exactly match declarations array order")
    local_instances = context["local_instances"]
    if not isinstance(local_instances, list):
        raise EvidenceError("local_context.local_instances must be an array")
    seen_local_instances: set[str] = set()
    for index, instance in enumerate(local_instances):
        _keys(instance, LOCAL_INSTANCE_KEYS, f"local_instances[{index}]")
        class_name = _string(instance["class_name"], f"local_instances[{index}].class_name")
        fvar_id = _string(instance["fvar_id"], f"local_instances[{index}].fvar_id")
        type_text = _string(instance["type"], f"local_instances[{index}].type")
        if fvar_id in seen_local_instances:
            raise EvidenceError(f"duplicate local instance FVarId {fvar_id}")
        seen_local_instances.add(fvar_id)
        declaration = declarations.get(fvar_id)
        if declaration is None:
            raise EvidenceError(f"local instance references unknown FVarId {fvar_id}")
        if type_text != declaration["type"]:
            raise EvidenceError(f"local instance type disagrees with its declaration: {class_name}")
    positions = {fvar_id: index for index, fvar_id in enumerate(order)}
    for fvar_id, row in declarations.items():
        for field in (
            "type_fvar_ids", "transparent_value_fvar_ids", "nondep_value_fvar_ids"
        ):
            for dependency in row[field]:
                if dependency not in positions:
                    raise EvidenceError(f"{fvar_id}.{field} references unknown ID {dependency}")
                if positions[dependency] >= positions[fvar_id]:
                    raise EvidenceError(
                        f"{fvar_id}.{field} violates ordered telescope: {dependency} is not earlier"
                    )

    if not set(target_roots).issubset(declarations):
        raise EvidenceError("target references a free variable outside the captured context")
    computed_target_closure = _closure(target_roots, declarations, False, "target closure")
    _same_set(target_closure_ids, computed_target_closure, "target typed closure IDs")
    _rows_match(
        target["typed_dependency_closure"], computed_target_closure,
        declarations, "target typed dependency closure"
    )

    support = evidence["proof_support"]
    _keys(support, SUPPORT_KEYS, "proof_support")
    _string(support["proof_term"], "proof_support.proof_term")
    direct_ids = _ids(support["direct_free_variable_ids"], "direct_free_variable_ids")
    typed_ids = _ids(support["typed_dependency_closure_ids"], "typed_dependency_closure_ids")
    assigned_ids = _ids(
        support["assigned_value_provenance_closure_ids"],
        "assigned_value_provenance_closure_ids",
    )
    if not set(direct_ids).issubset(declarations):
        raise EvidenceError("direct proof-term support references an unknown local declaration")

    computed_typed = _closure(direct_ids, declarations, False, "typed dependency closure")
    computed_assigned = _closure(direct_ids, declarations, True, "assigned-value provenance")
    _same_set(typed_ids, computed_typed, "typed dependency closure IDs")
    _same_set(assigned_ids, computed_assigned, "assigned-value provenance closure IDs")
    if not computed_typed.issubset(computed_assigned):
        raise EvidenceError("assigned-value provenance closure must contain typed closure")
    _rows_match(support["direct_declarations"], set(direct_ids), declarations, "direct declarations")
    _rows_match(support["typed_dependency_closure"], computed_typed, declarations, "typed dependency closure")
    _rows_match(
        support["assigned_value_provenance_closure"], computed_assigned,
        declarations, "assigned-value provenance closure"
    )

    expected_typed_constants = _global_constant_union(computed_typed, declarations, False)
    expected_assigned_constants = _global_constant_union(computed_assigned, declarations, True)
    _same_set(
        _constants(support["typed_dependency_global_constants"], "typed dependency constants"),
        expected_typed_constants,
        "typed dependency global constants",
    )
    _same_set(
        _constants(support["assigned_value_provenance_global_constants"], "assigned-value constants"),
        expected_assigned_constants,
        "assigned-value provenance global constants",
    )
    _constants(support["proof_term_global_constants"], "proof-term global constants")

    expected_opaque = {
        fvar_id for fvar_id in computed_typed if declarations[fvar_id]["is_nondep"]
    }
    opaque_rows = support["opaque_value_provenance"]
    if not isinstance(opaque_rows, list):
        raise EvidenceError("opaque_value_provenance must be an array")
    observed_opaque: set[str] = set()
    for index, opaque in enumerate(opaque_rows):
        expected_keys = {
            "declaration_fvar_id", "declaration_user_name",
            "assigned_value_fvar_ids", "assigned_value_global_constants",
        }
        _keys(opaque, expected_keys, f"opaque_value_provenance[{index}]")
        fvar_id = _string(opaque["declaration_fvar_id"], "opaque declaration ID")
        if fvar_id in observed_opaque:
            raise EvidenceError("duplicate opaque value provenance row")
        observed_opaque.add(fvar_id)
        source = declarations.get(fvar_id)
        if source is None or not source["is_nondep"]:
            raise EvidenceError("opaque provenance row does not identify a captured nondependent declaration")
        if opaque["declaration_user_name"] != source["user_name"]:
            raise EvidenceError("opaque provenance user name does not match source declaration")
        _same_set(
            _ids(opaque["assigned_value_fvar_ids"], "opaque assigned value IDs"),
            set(source["nondep_value_fvar_ids"]),
            "opaque assigned value IDs",
        )
        _same_set(
            _constants(opaque["assigned_value_global_constants"], "opaque assigned value constants"),
            set(source["nondep_value_global_constants"]),
            "opaque assigned value constants",
        )
    if observed_opaque != expected_opaque:
        raise EvidenceError("opaque provenance rows do not exactly cover nondependent declarations in typed closure")

    return {
        "declaration_count": len(declarations),
        "target_typed_dependency_count": len(computed_target_closure),
        "direct_support_count": len(direct_ids),
        "typed_dependency_count": len(computed_typed),
        "assigned_value_provenance_count": len(computed_assigned),
        "opaque_value_provenance_count": len(observed_opaque),
    }


def synthetic_fixture() -> dict:
    common_constants = ["Eq", "Nat"]
    rows = [
        {
            "fvar_id": "fvar-inst", "index": 0, "user_name": "localDecidable",
            "type": "Decidable (0 = 0)", "type_fvar_ids": [],
            "type_global_constants": ["Decidable", "Eq", "Nat"],
            "binder_info": "Lean.BinderInfo.instImplicit", "is_aux_declaration": False,
            "is_let_declaration": False, "is_nondep": False,
            "is_implementation_detail": False, "transparent_value": None,
            "transparent_value_fvar_ids": [], "transparent_value_global_constants": [],
            "nondep_value_provenance": None, "nondep_value_fvar_ids": [],
            "nondep_value_global_constants": [],
        },
        {
            "fvar_id": "fvar-h", "index": 1, "user_name": "h", "type": "0 = 0",
            "type_fvar_ids": [], "type_global_constants": common_constants,
            "binder_info": "Lean.BinderInfo.default", "is_aux_declaration": False,
            "is_let_declaration": False, "is_nondep": False,
            "is_implementation_detail": False, "transparent_value": None,
            "transparent_value_fvar_ids": [], "transparent_value_global_constants": [],
            "nondep_value_provenance": None, "nondep_value_fvar_ids": [],
            "nondep_value_global_constants": [],
        },
        {
            "fvar_id": "fvar-let", "index": 2, "user_name": "transparentValue", "type": "0 = 0",
            "type_fvar_ids": [], "type_global_constants": common_constants,
            "binder_info": "Lean.BinderInfo.default", "is_aux_declaration": False,
            "is_let_declaration": True, "is_nondep": False,
            "is_implementation_detail": False, "transparent_value": "h",
            "transparent_value_fvar_ids": ["fvar-h"], "transparent_value_global_constants": [],
            "nondep_value_provenance": None, "nondep_value_fvar_ids": [],
            "nondep_value_global_constants": [],
        },
        {
            "fvar_id": "fvar-have", "index": 3, "user_name": "opaqueValue", "type": "0 = 0",
            "type_fvar_ids": [], "type_global_constants": common_constants,
            "binder_info": "Lean.BinderInfo.default", "is_aux_declaration": False,
            "is_let_declaration": True, "is_nondep": True,
            "is_implementation_detail": False, "transparent_value": None,
            "transparent_value_fvar_ids": [], "transparent_value_global_constants": [],
            "nondep_value_provenance": "transparentValue",
            "nondep_value_fvar_ids": ["fvar-let"], "nondep_value_global_constants": [],
        },
    ]
    typed = [rows[3]]
    assigned = rows[1:4]
    return {
        "schema_version": "pitb.local-context-evidence.v2",
        "label": "transparent-let-have-localinstance-separation",
        "build_metadata": {
            "lean_toolchain": "leanprover/lean4:v4.34.1",
            "lean_toolchain_file_sha256": "a" * 64,
            "lakefile_sha256": "b" * 64,
            "root_module_path": "PITBCounterfactualPilot.lean",
            "root_module_sha256": "c" * 64,
            "context_exporter_path": "PITBCounterfactualPilot/ContextEvidenceV2.lean",
            "context_exporter_source_sha256": "d" * 64,
        },
        "target": {
            "type": "0 = 0",
            "free_variable_ids": [],
            "typed_dependency_closure_ids": [],
            "typed_dependency_closure": [],
            "global_constants": common_constants,
        },
        "local_context": {
            "declaration_order": ["fvar-inst", "fvar-h", "fvar-let", "fvar-have"],
            "declarations": rows,
            "local_instances": [{
                "class_name": "Decidable",
                "fvar_id": "fvar-inst",
                "type": "Decidable (0 = 0)",
            }],
        },
        "proof_support": {
            "proof_term": "opaqueValue",
            "direct_free_variable_ids": ["fvar-have"],
            "direct_declarations": [rows[3]],
            "typed_dependency_closure_ids": ["fvar-have"],
            "typed_dependency_closure": typed,
            "typed_dependency_global_constants": common_constants,
            "assigned_value_provenance_closure_ids": ["fvar-h", "fvar-let", "fvar-have"],
            "assigned_value_provenance_closure": assigned,
            "assigned_value_provenance_global_constants": common_constants,
            "opaque_value_provenance": [{
                "declaration_fvar_id": "fvar-have",
                "declaration_user_name": "opaqueValue",
                "assigned_value_fvar_ids": ["fvar-let"],
                "assigned_value_global_constants": [],
            }],
            "proof_term_global_constants": [],
        },
        "interpretation_boundary": (
            "Typed context dependencies and assigned-value provenance are distinct; "
            "neither closure establishes logical necessity."
        ),
    }


def self_test() -> None:
    fixture = synthetic_fixture()
    report = verify_evidence(fixture)
    assert report["typed_dependency_count"] == 1
    assert report["assigned_value_provenance_count"] == 3
    assert report["opaque_value_provenance_count"] == 1
    assert fixture["local_context"]["local_instances"] == [{
        "class_name": "Decidable", "fvar_id": "fvar-inst", "type": "Decidable (0 = 0)"
    }]

    def rejected(candidate: dict, label: str) -> None:
        try:
            verify_evidence(candidate)
        except EvidenceError:
            return
        raise AssertionError(f"negative control accepted: {label}")

    import copy

    bad = copy.deepcopy(fixture)
    bad["proof_support"]["typed_dependency_closure_ids"] = ["fvar-have", "fvar-let"]
    bad["proof_support"]["typed_dependency_closure"] = [bad["local_context"]["declarations"][2], bad["local_context"]["declarations"][3]]
    rejected(bad, "opaque value incorrectly counted as typing prerequisite")

    bad = copy.deepcopy(fixture)
    bad["proof_support"]["assigned_value_provenance_closure_ids"].remove("fvar-h")
    bad["proof_support"]["assigned_value_provenance_closure"].remove(bad["local_context"]["declarations"][0])
    rejected(bad, "truncated assigned-value provenance")

    bad = copy.deepcopy(fixture)
    bad["local_context"]["declarations"][1]["transparent_value_fvar_ids"] = ["unknown"]
    rejected(bad, "unknown local dependency")

    bad = copy.deepcopy(fixture)
    bad["local_context"]["declarations"][0]["type_fvar_ids"] = ["fvar-have"]
    rejected(bad, "forward dependency violates telescope order")

    bad = copy.deepcopy(fixture)
    bad["local_context"]["declarations"][3]["is_nondep"] = False
    rejected(bad, "nondependent metadata mismatch")

    bad = copy.deepcopy(fixture)
    bad["local_context"]["declaration_order"][0] = "not-fvar-inst"
    rejected(bad, "declaration order disagrees with rows")

    bad = copy.deepcopy(fixture)
    bad["local_context"]["local_instances"][0]["fvar_id"] = "missing-instance"
    rejected(bad, "local instance outside captured context")

    bad = copy.deepcopy(fixture)
    bad["local_context"]["local_instances"].append(bad["local_context"]["local_instances"][0])
    rejected(bad, "duplicate local instance registration")

    bad = copy.deepcopy(fixture)
    bad["local_context"]["local_instances"][0]["type"] = "Nat"
    rejected(bad, "local instance type disagrees with declaration")

    bad = copy.deepcopy(fixture)
    bad["proof_support"]["typed_dependency_global_constants"].append("Forged.Constant")
    rejected(bad, "forged global constant closure")

    print(
        "PITB_CONTEXT_EVIDENCE_V2_VERIFIER_SELF_TEST_PASS: "
        "transparent/opaque separation, closure reconstruction, order and metadata negative controls"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            if args.evidence:
                raise EvidenceError("--self-test cannot be combined with an evidence file")
            self_test()
            return 0
        if not args.evidence:
            raise EvidenceError("an evidence JSON path is required")
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        report = verify_evidence(evidence)
        print("PITB_CONTEXT_EVIDENCE_VERIFY_PASS")
        print(json.dumps(report, sort_keys=True))
        return 0
    except (EvidenceError, OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError) as exc:
        print(f"PITB context evidence rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
