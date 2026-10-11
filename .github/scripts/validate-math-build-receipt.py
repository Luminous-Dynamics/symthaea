#!/usr/bin/env python3
"""Validate canonical MATH-BUILD-002 execution receipts.

This validator is intentionally stdlib-only so receipt validation cannot mutate
or depend on the Cargo graph it is meant to qualify.

Authority rule:

    valid receipt != successful receipt != stronger authority

The receipt hash is SHA-256 over canonical JSON with `receipt_sha256` omitted.
Canonical JSON is UTF-8, sorted-key, compact JSON. The schema forbids floats,
so Python's number formatting cannot create cross-runtime float ambiguity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "symthaea.math.build.receipt/v1"
RECEIPT_DOMAIN = "symthaea.math.build"

ROLE_GENERATION = "artifact_generation"
ROLE_REPLAY = "artifact_replay"
ROLE_EXACT_HEAD = "exact_head_build_qualification"
ROLES = {ROLE_GENERATION, ROLE_REPLAY, ROLE_EXACT_HEAD}

OUTCOMES = {"pass", "fail"}
GATE_RESULTS = {"pass", "fail", "not_run"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GIT_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

REQUIRED_EXACT_HEAD_GATES = {
    "cargo_metadata_locked",
    "cargo_fmt",
    "cargo_test_locked_offline",
    "cargo_clippy_locked_offline",
}

TOP_KEYS = {
    "schema_version",
    "receipt_domain",
    "receipt_role",
    "outcome",
    "payload",
    "authority",
    "provenance",
    "receipt_sha256",
}

AUTHORITY_KEYS = {
    "artifact_contained_in_source_subject",
    "subject_build_qualified",
    "formal_authority",
    "mathematical_authority",
}

PROVENANCE_KEYS = {
    "provider",
    "run_id",
    "started_at",
    "finished_at",
}

SUBJECT_KEYS = {"commit_sha", "tree_sha"}
RECIPE_KEYS = {
    "recipe_semantics_sha256",
    "action_pin_set_sha256",
    "rust_toolchain_identity",
    "nix_environment_identity",
    "target_identity",
}
GATE_KEYS = {"name", "result", "command_sha256", "stdout_sha256", "stderr_sha256"}
SCOPE_KEYS = {"package", "profile", "whole_workspace_qualified"}

GENERATION_KEYS = {
    "source_subject",
    "qualification_subject",
    "qualification_overlay_digest",
    "qualification_overlay_paths",
    "recipe",
    "lock_before_sha256",
    "lock_candidate_sha256",
    "canonical_metadata_graph_sha256",
    "crate_source_set_sha256",
    "output_artifact_manifest_sha256",
    "gates",
}

REPLAY_KEYS = {
    "generation_receipt_sha256",
    "old_subject",
    "repaired_subject",
    "generated_artifact_sha256",
    "replayed_artifact_sha256",
    "source_delta_digest",
    "changed_paths",
}

EXACT_HEAD_KEYS = {
    "replay_receipt_sha256",
    "subject",
    "contained_lock_sha256",
    "recipe",
    "scope",
    "gates",
}


class ReceiptError(ValueError):
    pass


def fail(message: str) -> None:
    raise ReceiptError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def require_dict(value: Any, path: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{path} must be an object")
    return value


def require_list(value: Any, path: str) -> list[Any]:
    require(isinstance(value, list), f"{path} must be an array")
    return value


def require_str(value: Any, path: str, *, nonempty: bool = True) -> str:
    require(isinstance(value, str), f"{path} must be a string")
    if nonempty:
        require(bool(value), f"{path} must not be empty")
    return value


def require_bool(value: Any, path: str) -> bool:
    require(type(value) is bool, f"{path} must be a boolean")
    return value


def require_exact_keys(value: dict[str, Any], keys: set[str], path: str) -> None:
    observed = set(value)
    missing = sorted(keys - observed)
    unknown = sorted(observed - keys)
    require(not missing and not unknown, f"{path} keys mismatch: missing={missing} unknown={unknown}")


def require_sha256(value: Any, path: str) -> str:
    text = require_str(value, path)
    require(HEX64.fullmatch(text) is not None, f"{path} must be lowercase 64-hex SHA-256")
    return text


def require_git_oid(value: Any, path: str) -> str:
    text = require_str(value, path)
    require(GIT_OID.fullmatch(text) is not None, f"{path} must be lowercase 40- or 64-hex Git object ID")
    return text


def reject_floats(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        fail(f"{path}: floats are forbidden in canonical receipt v1")
    if isinstance(value, dict):
        for key, child in value.items():
            require(isinstance(key, str), f"{path}: object keys must be strings")
            reject_floats(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_floats(child, f"{path}[{index}]")


def canonical_bytes(receipt: dict[str, Any]) -> bytes:
    reject_floats(receipt)
    return json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def receipt_preimage(receipt: dict[str, Any]) -> bytes:
    unsigned = copy.deepcopy(receipt)
    unsigned.pop("receipt_sha256", None)
    return canonical_bytes(unsigned)


def compute_receipt_sha256(receipt: dict[str, Any]) -> str:
    return hashlib.sha256(receipt_preimage(receipt)).hexdigest()


def validate_subject(value: Any, path: str) -> None:
    obj = require_dict(value, path)
    require_exact_keys(obj, SUBJECT_KEYS, path)
    require_git_oid(obj["commit_sha"], f"{path}.commit_sha")
    require_git_oid(obj["tree_sha"], f"{path}.tree_sha")


def validate_recipe(value: Any, path: str) -> None:
    obj = require_dict(value, path)
    require_exact_keys(obj, RECIPE_KEYS, path)
    require_sha256(obj["recipe_semantics_sha256"], f"{path}.recipe_semantics_sha256")
    require_sha256(obj["action_pin_set_sha256"], f"{path}.action_pin_set_sha256")
    require_str(obj["rust_toolchain_identity"], f"{path}.rust_toolchain_identity")
    require_str(obj["nix_environment_identity"], f"{path}.nix_environment_identity")
    require_str(obj["target_identity"], f"{path}.target_identity")


def validate_string_list(value: Any, path: str, *, nonempty: bool = True) -> list[str]:
    items = require_list(value, path)
    if nonempty:
        require(bool(items), f"{path} must not be empty")
    result: list[str] = []
    for index, item in enumerate(items):
        result.append(require_str(item, f"{path}[{index}]"))
    require(result == sorted(result), f"{path} must be lexicographically sorted")
    require(len(result) == len(set(result)), f"{path} must not contain duplicates")
    return result


def validate_gates(value: Any, path: str) -> list[dict[str, Any]]:
    gates = require_list(value, path)
    require(bool(gates), f"{path} must not be empty")
    names: list[str] = []
    for index, raw_gate in enumerate(gates):
        gate_path = f"{path}[{index}]"
        gate = require_dict(raw_gate, gate_path)
        require_exact_keys(gate, GATE_KEYS, gate_path)
        name = require_str(gate["name"], f"{gate_path}.name")
        names.append(name)
        require(gate["result"] in GATE_RESULTS, f"{gate_path}.result invalid")
        require_sha256(gate["command_sha256"], f"{gate_path}.command_sha256")
        require_sha256(gate["stdout_sha256"], f"{gate_path}.stdout_sha256")
        require_sha256(gate["stderr_sha256"], f"{gate_path}.stderr_sha256")
    require(names == sorted(names), f"{path} must be sorted by gate name")
    require(len(names) == len(set(names)), f"{path} contains duplicate gate names")
    return gates


def validate_authority(value: Any, path: str) -> dict[str, bool]:
    authority = require_dict(value, path)
    require_exact_keys(authority, AUTHORITY_KEYS, path)
    for key in sorted(AUTHORITY_KEYS):
        require_bool(authority[key], f"{path}.{key}")
    require(authority["formal_authority"] is False, f"{path}.formal_authority must be false for math-build receipts")
    require(authority["mathematical_authority"] is False, f"{path}.mathematical_authority must be false for math-build receipts")
    return authority


def validate_provenance(value: Any, path: str) -> None:
    provenance = require_dict(value, path)
    require_exact_keys(provenance, PROVENANCE_KEYS, path)
    for key in sorted(PROVENANCE_KEYS):
        require_str(provenance[key], f"{path}.{key}", nonempty=False)


def validate_outcome_and_gates(outcome: str, gates: list[dict[str, Any]], path: str) -> None:
    results = [gate["result"] for gate in gates]
    if outcome == "pass":
        require(all(result == "pass" for result in results), f"{path}: PASS receipt requires every gate to pass")
    else:
        require(any(result == "fail" for result in results), f"{path}: FAIL receipt requires at least one failing gate")


def validate_generation(payload: dict[str, Any], outcome: str, authority: dict[str, bool]) -> None:
    require_exact_keys(payload, GENERATION_KEYS, "$.payload")
    validate_subject(payload["source_subject"], "$.payload.source_subject")
    validate_subject(payload["qualification_subject"], "$.payload.qualification_subject")
    require_sha256(payload["qualification_overlay_digest"], "$.payload.qualification_overlay_digest")
    validate_string_list(payload["qualification_overlay_paths"], "$.payload.qualification_overlay_paths")
    validate_recipe(payload["recipe"], "$.payload.recipe")
    for key in (
        "lock_before_sha256",
        "lock_candidate_sha256",
        "canonical_metadata_graph_sha256",
        "crate_source_set_sha256",
        "output_artifact_manifest_sha256",
    ):
        require_sha256(payload[key], f"$.payload.{key}")
    gates = validate_gates(payload["gates"], "$.payload.gates")
    validate_outcome_and_gates(outcome, gates, "artifact_generation")
    require(authority["artifact_contained_in_source_subject"] is False, "generation receipt cannot claim artifact containment")
    require(authority["subject_build_qualified"] is False, "generation receipt cannot claim build qualification")


def validate_replay(payload: dict[str, Any], outcome: str, authority: dict[str, bool]) -> None:
    require_exact_keys(payload, REPLAY_KEYS, "$.payload")
    require_sha256(payload["generation_receipt_sha256"], "$.payload.generation_receipt_sha256")
    validate_subject(payload["old_subject"], "$.payload.old_subject")
    validate_subject(payload["repaired_subject"], "$.payload.repaired_subject")
    generated = require_sha256(payload["generated_artifact_sha256"], "$.payload.generated_artifact_sha256")
    replayed = require_sha256(payload["replayed_artifact_sha256"], "$.payload.replayed_artifact_sha256")
    require_sha256(payload["source_delta_digest"], "$.payload.source_delta_digest")
    validate_string_list(payload["changed_paths"], "$.payload.changed_paths")
    same_bytes = generated == replayed
    if outcome == "pass":
        require(same_bytes, "replay PASS requires generated and replayed artifact digests to match")
        require(authority["artifact_contained_in_source_subject"] is True, "replay PASS must claim artifact containment")
    else:
        require(authority["artifact_contained_in_source_subject"] is False, "replay FAIL cannot claim artifact containment")
    require(authority["subject_build_qualified"] is False, "replay receipt cannot claim build qualification")


def validate_exact_head(payload: dict[str, Any], outcome: str, authority: dict[str, bool]) -> None:
    require_exact_keys(payload, EXACT_HEAD_KEYS, "$.payload")
    require_sha256(payload["replay_receipt_sha256"], "$.payload.replay_receipt_sha256")
    validate_subject(payload["subject"], "$.payload.subject")
    require_sha256(payload["contained_lock_sha256"], "$.payload.contained_lock_sha256")
    validate_recipe(payload["recipe"], "$.payload.recipe")

    scope = require_dict(payload["scope"], "$.payload.scope")
    require_exact_keys(scope, SCOPE_KEYS, "$.payload.scope")
    require_str(scope["package"], "$.payload.scope.package")
    require_str(scope["profile"], "$.payload.scope.profile")
    require_bool(scope["whole_workspace_qualified"], "$.payload.scope.whole_workspace_qualified")
    require(scope["whole_workspace_qualified"] is False, "MATH-BUILD-002 v1 is package/profile scoped, not whole-workspace authority")

    gates = validate_gates(payload["gates"], "$.payload.gates")
    validate_outcome_and_gates(outcome, gates, "exact_head_build_qualification")
    gate_names = {gate["name"] for gate in gates}
    if outcome == "pass":
        missing = sorted(REQUIRED_EXACT_HEAD_GATES - gate_names)
        require(not missing, f"exact-head PASS missing required gates: {missing}")
        require(authority["artifact_contained_in_source_subject"] is True, "exact-head PASS requires contained artifact")
        require(authority["subject_build_qualified"] is True, "exact-head PASS must claim package-scoped build qualification")
    else:
        require(authority["subject_build_qualified"] is False, "exact-head FAIL cannot claim build qualification")


def validate_receipt(receipt: Any, *, verify_digest: bool = True) -> None:
    obj = require_dict(receipt, "$")
    require_exact_keys(obj, TOP_KEYS, "$")
    reject_floats(obj)

    require(obj["schema_version"] == SCHEMA_VERSION, f"unsupported schema_version: {obj['schema_version']!r}")
    require(obj["receipt_domain"] == RECEIPT_DOMAIN, f"unexpected receipt_domain: {obj['receipt_domain']!r}")
    role = require_str(obj["receipt_role"], "$.receipt_role")
    require(role in ROLES, f"unsupported receipt_role: {role!r}")
    outcome = require_str(obj["outcome"], "$.outcome")
    require(outcome in OUTCOMES, f"invalid outcome: {outcome!r}")

    authority = validate_authority(obj["authority"], "$.authority")
    validate_provenance(obj["provenance"], "$.provenance")
    payload = require_dict(obj["payload"], "$.payload")

    if role == ROLE_GENERATION:
        validate_generation(payload, outcome, authority)
    elif role == ROLE_REPLAY:
        validate_replay(payload, outcome, authority)
    else:
        validate_exact_head(payload, outcome, authority)

    claimed_digest = require_sha256(obj["receipt_sha256"], "$.receipt_sha256")
    if verify_digest:
        observed_digest = compute_receipt_sha256(obj)
        require(claimed_digest == observed_digest, f"receipt_sha256 mismatch: claimed={claimed_digest} observed={observed_digest}")


def zero_sha(byte: str = "0") -> str:
    return byte * 64


def oid(byte: str = "a") -> str:
    return byte * 40


def gate(name: str, result: str = "pass") -> dict[str, str]:
    return {
        "name": name,
        "result": result,
        "command_sha256": zero_sha("1"),
        "stdout_sha256": zero_sha("2"),
        "stderr_sha256": zero_sha("3"),
    }


def recipe() -> dict[str, str]:
    return {
        "recipe_semantics_sha256": zero_sha("4"),
        "action_pin_set_sha256": zero_sha("5"),
        "rust_toolchain_identity": "rustc 1.96.0",
        "nix_environment_identity": "flake-lock:example",
        "target_identity": "x86_64-unknown-linux-gnu",
    }


def authority(*, contained: bool = False, qualified: bool = False) -> dict[str, bool]:
    return {
        "artifact_contained_in_source_subject": contained,
        "subject_build_qualified": qualified,
        "formal_authority": False,
        "mathematical_authority": False,
    }


def envelope(role: str, payload: dict[str, Any], auth: dict[str, bool], outcome: str = "pass") -> dict[str, Any]:
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_domain": RECEIPT_DOMAIN,
        "receipt_role": role,
        "outcome": outcome,
        "payload": payload,
        "authority": auth,
        "provenance": {
            "provider": "self-test",
            "run_id": "1",
            "started_at": "",
            "finished_at": "",
        },
        "receipt_sha256": zero_sha(),
    }
    receipt["receipt_sha256"] = compute_receipt_sha256(receipt)
    return receipt


def generation_fixture() -> dict[str, Any]:
    payload = {
        "source_subject": {"commit_sha": oid("a"), "tree_sha": oid("b")},
        "qualification_subject": {"commit_sha": oid("c"), "tree_sha": oid("d")},
        "qualification_overlay_digest": zero_sha("6"),
        "qualification_overlay_paths": [".github/workflows/math-lock-qualification.yml"],
        "recipe": recipe(),
        "lock_before_sha256": zero_sha("7"),
        "lock_candidate_sha256": zero_sha("8"),
        "canonical_metadata_graph_sha256": zero_sha("9"),
        "crate_source_set_sha256": "a" * 64,
        "output_artifact_manifest_sha256": "b" * 64,
        "gates": [gate("lock_delta"), gate("nix_clippy"), gate("rust_test")],
    }
    payload["gates"] = sorted(payload["gates"], key=lambda item: item["name"])
    return envelope(ROLE_GENERATION, payload, authority())


def replay_fixture(generation_digest: str) -> dict[str, Any]:
    artifact = zero_sha("8")
    payload = {
        "generation_receipt_sha256": generation_digest,
        "old_subject": {"commit_sha": oid("a"), "tree_sha": oid("b")},
        "repaired_subject": {"commit_sha": oid("e"), "tree_sha": oid("f")},
        "generated_artifact_sha256": artifact,
        "replayed_artifact_sha256": artifact,
        "source_delta_digest": "c" * 64,
        "changed_paths": ["Cargo.lock"],
    }
    return envelope(ROLE_REPLAY, payload, authority(contained=True))


def exact_fixture(replay_digest: str) -> dict[str, Any]:
    gates = [gate(name) for name in sorted(REQUIRED_EXACT_HEAD_GATES)]
    payload = {
        "replay_receipt_sha256": replay_digest,
        "subject": {"commit_sha": oid("e"), "tree_sha": oid("f")},
        "contained_lock_sha256": zero_sha("8"),
        "recipe": recipe(),
        "scope": {
            "package": "symthaea-math-research",
            "profile": "math-build-v1",
            "whole_workspace_qualified": False,
        },
        "gates": gates,
    }
    return envelope(ROLE_EXACT_HEAD, payload, authority(contained=True, qualified=True))


def expect_rejected(receipt: dict[str, Any], label: str) -> None:
    receipt["receipt_sha256"] = compute_receipt_sha256(receipt)
    try:
        validate_receipt(receipt)
    except ReceiptError:
        return
    fail(f"self-test expected rejection: {label}")


def self_test() -> None:
    generation = generation_fixture()
    validate_receipt(generation)
    replay = replay_fixture(generation["receipt_sha256"])
    validate_receipt(replay)
    exact = exact_fixture(replay["receipt_sha256"])
    validate_receipt(exact)

    bad = copy.deepcopy(generation)
    bad["authority"]["artifact_contained_in_source_subject"] = True
    expect_rejected(bad, "generation claims containment")

    bad = copy.deepcopy(replay)
    bad["payload"]["replayed_artifact_sha256"] = "d" * 64
    expect_rejected(bad, "replay pass digest mismatch")

    bad = copy.deepcopy(exact)
    bad["payload"]["gates"] = bad["payload"]["gates"][1:]
    expect_rejected(bad, "exact-head pass missing required gate")

    bad = copy.deepcopy(exact)
    bad["authority"]["formal_authority"] = True
    expect_rejected(bad, "build receipt claims formal authority")

    bad = copy.deepcopy(exact)
    bad["unexpected"] = True
    expect_rejected(bad, "unknown top-level field")

    bad = copy.deepcopy(exact)
    bad["payload"]["gates"][0]["runtime_seconds"] = 1.25
    bad["receipt_sha256"] = zero_sha()
    expect_rejected(bad, "float/unknown gate field")

    left = generation_fixture()
    right = json.loads(json.dumps(left, sort_keys=False))
    right = {key: right[key] for key in reversed(list(right.keys()))}
    require(receipt_preimage(left) == receipt_preimage(right), "canonical preimage changed under object-key reorder")

    print("math_build_receipt_self_test=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--print-canonical", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.receipt is None:
        parser.error("receipt path is required unless --self-test is used")

    try:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        validate_receipt(receipt)
    except (OSError, json.JSONDecodeError, ReceiptError) as error:
        print(f"math-build-receipt: FAIL: {error}", file=sys.stderr)
        return 1

    if args.print_canonical:
        sys.stdout.buffer.write(canonical_bytes(receipt))
        sys.stdout.write("\n")
    print(f"math_build_receipt=PASS role={receipt['receipt_role']} outcome={receipt['outcome']}")
    print(f"receipt_sha256={receipt['receipt_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
