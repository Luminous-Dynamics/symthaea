#!/usr/bin/env python3
"""Fail-closed structural validation for target-preserving PITB repair trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCHEMA = "pitb.typed-proof-repair-operation.v0"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ROOT_KEYS = {
    "schema_version", "episode_id", "operation_id", "sequence_index",
    "target_theorem_id", "target_statement_sha256_before", "target_statement_sha256_after",
    "blueprint_sha256_before", "blueprint_sha256_after", "operation",
    "pre_state_node_ids", "post_state_node_ids", "affected_node_ids", "declared_support",
    "decision", "lean_check", "operation_receipt_sha256", "run_receipt_sha256",
}
OPTIONAL_KEYS = {"rejection_code", "review_receipt_sha256"}
OPERATION_KEYS = {
    "kind", "node_id", "source_node_id", "destination_node_id",
    "replacement_node_id", "rationale_code", "rationale",
}
KINDS = {
    "add_dependency", "remove_dependency", "replace_dependency", "refine_obligation",
    "split_obligation", "merge_obligations", "strengthen_precondition", "weaken_precondition",
}
REJECTION_CODES = {
    "precondition_failed", "target_mutation_attempt", "ill_typed_edit",
    "cycle_or_scope_violation", "unsupported_node", "budget_exhausted",
    "review_required", "other",
}


class ContractError(ValueError):
    pass


def digest(value, label):
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ContractError(f"{label} must be lowercase SHA-256")
    return value


def nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty string")
    return value


def sorted_ids(value, label):
    if (not isinstance(value, list) or
            any(not isinstance(x, str) or not x.strip() for x in value) or
            value != sorted(value) or len(value) != len(set(value))):
        raise ContractError(f"{label} must be sorted unique non-empty IDs")
    return value


def validate_row(row, line):
    label = f"line {line}"
    if not isinstance(row, dict) or not ROOT_KEYS <= set(row) or set(row) - ROOT_KEYS - OPTIONAL_KEYS:
        raise ContractError(f"{label}: missing or unsupported operation fields")
    if row["schema_version"] != SCHEMA:
        raise ContractError(f"{label}: unsupported schema version")
    for key in ("episode_id", "operation_id", "target_theorem_id"):
        nonempty(row[key], f"{label}.{key}")
    if isinstance(row["sequence_index"], bool) or not isinstance(row["sequence_index"], int) or row["sequence_index"] < 0:
        raise ContractError(f"{label}.sequence_index must be a non-negative integer")
    for key in (
        "target_statement_sha256_before", "target_statement_sha256_after",
        "blueprint_sha256_before", "blueprint_sha256_after",
        "operation_receipt_sha256", "run_receipt_sha256",
    ):
        digest(row[key], f"{label}.{key}")
    if "review_receipt_sha256" in row:
        digest(row["review_receipt_sha256"], f"{label}.review_receipt_sha256")
    op = row["operation"]
    if not isinstance(op, dict) or not {"kind", "node_id"} <= set(op) or set(op) - OPERATION_KEYS:
        raise ContractError(f"{label}.operation has missing or unsupported fields")
    if op["kind"] not in KINDS:
        raise ContractError(f"{label}.operation.kind is unsupported")
    nonempty(op["node_id"], f"{label}.operation.node_id")
    for k in OPERATION_KEYS - {"kind", "node_id"}:
        if k in op:
            nonempty(op[k], f"{label}.operation.{k}")
    pre = sorted_ids(row["pre_state_node_ids"], f"{label}.pre_state_node_ids")
    post = sorted_ids(row["post_state_node_ids"], f"{label}.post_state_node_ids")
    affected = sorted_ids(row["affected_node_ids"], f"{label}.affected_node_ids")
    sorted_ids(row["declared_support"], f"{label}.declared_support")
    target = row["target_theorem_id"]
    if target not in pre or target not in post:
        raise ContractError(f"{label}: target theorem node must exist before and after the operation")
    if target in affected:
        raise ContractError(f"{label}: target theorem node may never be edited")
    if op["node_id"] not in pre:
        raise ContractError(f"{label}: operation node does not exist in pre-state")
    if not set(affected) <= (set(pre) | set(post)):
        raise ContractError(f"{label}: affected node is absent from both states")
    if row["target_statement_sha256_before"] != row["target_statement_sha256_after"]:
        raise ContractError(f"{label}: target statement digest changed")
    decision, check = row["decision"], row["lean_check"]
    if decision not in ("applied", "rejected", "abstained") or check not in ("passed", "failed", "not_run"):
        raise ContractError(f"{label}: unsupported decision/check state")
    before, after = row["blueprint_sha256_before"], row["blueprint_sha256_after"]
    if decision == "applied":
        if check != "passed":
            raise ContractError(f"{label}: applied operation must pass Lean")
        if before == after:
            raise ContractError(f"{label}: applied operation did not change blueprint digest")
        if not affected:
            raise ContractError(f"{label}: applied operation must name affected nodes")
        if "rejection_code" in row:
            raise ContractError(f"{label}: applied operation must not carry a rejection code")
    elif decision == "rejected":
        if check not in ("failed", "not_run"):
            raise ContractError(f"{label}: rejected operation cannot have Lean check passed")
        if before != after or pre != post:
            raise ContractError(f"{label}: rejected operation mutated blueprint state")
        if row.get("rejection_code") not in REJECTION_CODES:
            raise ContractError(f"{label}: rejected operation requires a registered rejection_code")
        if affected:
            raise ContractError(f"{label}: rejected operation cannot claim affected nodes")
    else:
        if check != "not_run":
            raise ContractError(f"{label}: abstained operation must not run Lean")
        if before != after or pre != post or affected:
            raise ContractError(f"{label}: abstention must not mutate or claim affected nodes")
        if "rejection_code" in row:
            raise ContractError(f"{label}: abstention must not carry a rejection code")
    return row


def validate_episodes(rows):
    grouped = defaultdict(list)
    for index, row in enumerate(rows, 1):
        validate_row(row, index)
        grouped[row["episode_id"]].append(row)
    reports = []
    for episode_id, episode in sorted(grouped.items()):
        episode.sort(key=lambda r: r["sequence_index"])
        indices = [r["sequence_index"] for r in episode]
        if indices != list(range(len(episode))):
            raise ContractError(f"{episode_id}: sequence indices must be exactly 0..n-1")
        target_ids = {r["target_theorem_id"] for r in episode}
        target_digests = {r["target_statement_sha256_before"] for r in episode} | {
            r["target_statement_sha256_after"] for r in episode
        }
        if len(target_ids) != 1 or len(target_digests) != 1:
            raise ContractError(f"{episode_id}: target theorem identity changed during repair")
        operation_ids = [r["operation_id"] for r in episode]
        operation_receipts = [r["operation_receipt_sha256"] for r in episode]
        if len(operation_ids) != len(set(operation_ids)) or len(operation_receipts) != len(set(operation_receipts)):
            raise ContractError(f"{episode_id}: duplicate operation ID or operation receipt")
        for prev, nxt in zip(episode, episode[1:]):
            if prev["blueprint_sha256_after"] != nxt["blueprint_sha256_before"]:
                raise ContractError(f"{episode_id}: blueprint digest chain is discontinuous")
            if prev["post_state_node_ids"] != nxt["pre_state_node_ids"]:
                raise ContractError(f"{episode_id}: node state chain is discontinuous")
        reports.append({
            "episode_id": episode_id,
            "target_theorem_id": next(iter(target_ids)),
            "operation_count": len(episode),
            "applied_count": sum(r["decision"] == "applied" for r in episode),
            "rejected_count": sum(r["decision"] == "rejected" for r in episode),
            "abstained_count": sum(r["decision"] == "abstained" for r in episode),
            "target_preserved": True,
            "blueprint_chain_contiguous": True,
            "lean_checked_applied_operations": True,
            "theorem_success_status": "not_evaluated_by_trajectory_integrity_validator",
        })
    return {
        "schema_version": "pitb.typed-proof-repair-trajectory-report.v0",
        "episode_count": len(reports),
        "operation_count": len(rows),
        "episodes": reports,
        "interpretation_boundary": "Trajectory integrity only. Does not authenticate receipt bytes, validate Lean semantics, or establish final exact-target theorem success.",
    }


def synthetic_episode():
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()
    common = {
        "schema_version": SCHEMA, "episode_id": "self-test-episode",
        "target_theorem_id": "target", "target_statement_sha256_before": h("target"),
        "target_statement_sha256_after": h("target"),
        "pre_state_node_ids": ["lemma-a", "target"],
        "post_state_node_ids": ["lemma-a", "target"],
        "declared_support": ["lemma-a"], "run_receipt_sha256": h("run"),
    }
    rejected = dict(common, operation_id="reject-0", sequence_index=0,
        blueprint_sha256_before=h("graph-0"), blueprint_sha256_after=h("graph-0"),
        operation={"kind":"replace_dependency","node_id":"lemma-a"},
        affected_node_ids=[], decision="rejected", lean_check="failed",
        rejection_code="wrong_direction", operation_receipt_sha256=h("op-0"))
    applied = dict(common, operation_id="apply-1", sequence_index=1,
        blueprint_sha256_before=h("graph-0"), blueprint_sha256_after=h("graph-1"),
        operation={"kind":"add_dependency","node_id":"lemma-a","source_node_id":"lemma-a","destination_node_id":"lemma-b"},
        pre_state_node_ids=["lemma-a","target"],
        post_state_node_ids=["lemma-a","lemma-b","target"],
        affected_node_ids=["lemma-a","lemma-b"], decision="applied", lean_check="passed",
        operation_receipt_sha256=h("op-1"))
    return [rejected, applied]


def self_test():
    rows = synthetic_episode()
    report = validate_episodes(rows)
    assert report["episode_count"] == 1 and report["operation_count"] == 2
    assert report["episodes"][0]["applied_count"] == 1
    assert report["episodes"][0]["rejected_count"] == 1

    mutations = []
    changed_target = [dict(x) for x in rows]
    changed_target[1] = dict(changed_target[1], target_statement_sha256_after=hashlib.sha256(b"changed-target").hexdigest())
    mutations.append(("target mutation", changed_target))
    rejected_mutation = [dict(x) for x in rows]
    rejected_mutation[0] = dict(rejected_mutation[0], blueprint_sha256_after=hashlib.sha256(b"mutated").hexdigest())
    mutations.append(("rejected edit mutation", rejected_mutation))
    target_affected = [dict(x) for x in rows]
    target_affected[1] = dict(target_affected[1], affected_node_ids=["target"])
    mutations.append(("target in affected nodes", target_affected))
    broken_chain = [dict(x) for x in rows]
    broken_chain[1] = dict(broken_chain[1], blueprint_sha256_before=hashlib.sha256(b"wrong-start").hexdigest())
    mutations.append(("broken digest chain", broken_chain))
    bad_lean = [dict(x) for x in rows]
    bad_lean[1] = dict(bad_lean[1], lean_check="failed")
    mutations.append(("applied operation without Lean pass", bad_lean))
    bad_node = [dict(x) for x in rows]
    bad_node[1] = dict(bad_node[1], operation={"kind":"add_dependency","node_id":"missing"})
    mutations.append(("unknown operation node", bad_node))
    for label, mutated in mutations:
        try:
            validate_episodes(mutated)
        except ContractError:
            continue
        raise AssertionError(f"negative control accepted: {label}")
    print("PITB_TYPED_REPAIR_SELF_TEST_PASS: accepted/rejected trajectory, target immutability, state continuity, and 6 negative controls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            if args.input or args.report:
                raise ContractError("--self-test cannot be combined with file arguments")
            self_test()
            return 0
        if not args.input or not args.report:
            raise ContractError("--input and --report are required")
        rows = []
        for n, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                rows.append(json.loads(line))
        if not rows:
            raise ContractError("input JSONL must contain at least one operation")
        report = validate_episodes(rows)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"PITB_TYPED_REPAIR_TRAJECTORY: episodes={report['episode_count']} operations={report['operation_count']} target_preservation=true")
        return 0
    except (ContractError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"PITB typed repair trajectory rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
