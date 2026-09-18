#!/usr/bin/env python3
"""Validate Symthaea mathematical result/search memory v1 records.

This validator intentionally uses only the Python standard library.  It validates
semantic invariants that a JSON Schema alone cannot express and, crucially, never
promotes search-quality signals (Phi, similarity, recurrence) into mathematical
truth authority.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCHEMA_VERSION = "symthaea.math-memory.v1"
RESULT_KINDS = {
    "NumericalComputation",
    "SymbolicIdentity",
    "ExhaustiveFiniteResult",
    "SmtResult",
    "FormalLemma",
    "FormalTheorem",
    "Counterexample",
    "EquivalentFormulation",
    "Bound",
    "SpecialCase",
}
EVIDENCE_STATES = {
    "Unqualified",
    "NumericallyCorroborated",
    "ExhaustiveFinite",
    "SymbolicallyChecked",
    "SmtChecked",
    "KernelAccepted",
    "ComparatorAccepted",
    "IndependentlyChecked",
}
SEARCH_OUTCOMES = {
    "Succeeded",
    "Refuted",
    "CounterexampleFound",
    "Timeout",
    "ResourceExhausted",
    "Unsupported",
    "NoProgress",
    "Abandoned",
    "Superseded",
    "RepresentationMismatch",
    "SolverUnknown",
}
NON_TRUTH_OUTCOMES = {
    "Timeout",
    "ResourceExhausted",
    "Unsupported",
    "NoProgress",
    "Abandoned",
    "Superseded",
    "RepresentationMismatch",
    "SolverUnknown",
}

ROOT_KEYS = {
    "schema_version",
    "research_graph_ref",
    "result_memory",
    "search_memory",
}
RESULT_KEYS = {
    "episode_id",
    "claim_id",
    "result_kind",
    "evidence_state",
    "evidence_refs",
    "method",
    "representation",
    "dependency_claim_ids",
    "timestamp",
    "retrieval_embedding_ref",
    "phi_search_score",
}
SEARCH_KEYS = {
    "episode_id",
    "research_program_id",
    "problem_ref",
    "subgoal_ref",
    "strategy",
    "representation",
    "tactic_family",
    "parameters_ref",
    "budget_ref",
    "outcome",
    "failure_class",
    "cost",
    "useful_artifact_refs",
    "source_provenance_refs",
    "transfer_context",
    "timestamp",
    "retrieval_embedding_ref",
    "phi_search_score",
}
COST_KEYS = {"wall_ms", "cpu_ms", "proof_calls", "solver_calls", "search_nodes"}


class ValidationError(ValueError):
    pass


def _closed(obj: dict, allowed: set[str], where: str) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        raise ValidationError(f"{where}: unknown fields: {', '.join(extra)}")


def _required(obj: dict, fields: tuple[str, ...], where: str) -> None:
    missing = [field for field in fields if field not in obj]
    if missing:
        raise ValidationError(f"{where}: missing fields: {', '.join(missing)}")


def _nonempty_string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{where}: expected non-empty string")
    return value


def _string_list(value: object, where: str) -> list[str]:
    if not isinstance(value, list):
        raise ValidationError(f"{where}: expected list")
    out: list[str] = []
    for index, item in enumerate(value):
        out.append(_nonempty_string(item, f"{where}[{index}]"))
    if len(out) != len(set(out)):
        raise ValidationError(f"{where}: duplicate references are not allowed")
    return out


def _score(value: object, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{where}: expected number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValidationError(f"{where}: expected finite value in [0, 1]")
    return result


def _nonnegative_int(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{where}: expected non-negative integer")
    return value


def validate_result_episode(ep: object, index: int) -> str:
    where = f"result_memory[{index}]"
    if not isinstance(ep, dict):
        raise ValidationError(f"{where}: expected object")
    _closed(ep, RESULT_KEYS, where)
    _required(
        ep,
        (
            "episode_id",
            "claim_id",
            "result_kind",
            "evidence_state",
            "evidence_refs",
            "method",
            "representation",
            "dependency_claim_ids",
            "timestamp",
            "retrieval_embedding_ref",
            "phi_search_score",
        ),
        where,
    )
    episode_id = _nonempty_string(ep["episode_id"], f"{where}.episode_id")
    _nonempty_string(ep["claim_id"], f"{where}.claim_id")
    kind = _nonempty_string(ep["result_kind"], f"{where}.result_kind")
    if kind not in RESULT_KINDS:
        raise ValidationError(f"{where}.result_kind: unsupported value {kind!r}")
    state = _nonempty_string(ep["evidence_state"], f"{where}.evidence_state")
    if state not in EVIDENCE_STATES:
        raise ValidationError(f"{where}.evidence_state: unsupported value {state!r}")
    refs = _string_list(ep["evidence_refs"], f"{where}.evidence_refs")
    _nonempty_string(ep["method"], f"{where}.method")
    _nonempty_string(ep["representation"], f"{where}.representation")
    _string_list(ep["dependency_claim_ids"], f"{where}.dependency_claim_ids")
    _nonnegative_int(ep["timestamp"], f"{where}.timestamp")
    _nonempty_string(ep["retrieval_embedding_ref"], f"{where}.retrieval_embedding_ref")
    _score(ep["phi_search_score"], f"{where}.phi_search_score")

    # Formal-looking result classes need evidence references, but this validator
    # does not claim those references are authentic verifier receipts.  That is
    # checked by the dedicated MATH-EVID/MATH-VERIFY authority plane.
    if kind in {"FormalLemma", "FormalTheorem"} and not refs:
        raise ValidationError(f"{where}: formal result kind requires evidence_refs")
    if state in {"KernelAccepted", "ComparatorAccepted", "IndependentlyChecked"} and not refs:
        raise ValidationError(f"{where}: formal evidence state requires evidence_refs")

    return episode_id


def validate_search_episode(ep: object, index: int) -> str:
    where = f"search_memory[{index}]"
    if not isinstance(ep, dict):
        raise ValidationError(f"{where}: expected object")
    _closed(ep, SEARCH_KEYS, where)
    _required(
        ep,
        (
            "episode_id",
            "research_program_id",
            "problem_ref",
            "strategy",
            "representation",
            "tactic_family",
            "parameters_ref",
            "budget_ref",
            "outcome",
            "failure_class",
            "cost",
            "useful_artifact_refs",
            "source_provenance_refs",
            "transfer_context",
            "timestamp",
            "retrieval_embedding_ref",
            "phi_search_score",
        ),
        where,
    )
    episode_id = _nonempty_string(ep["episode_id"], f"{where}.episode_id")
    _nonempty_string(ep["research_program_id"], f"{where}.research_program_id")
    _nonempty_string(ep["problem_ref"], f"{where}.problem_ref")
    if "subgoal_ref" in ep and ep["subgoal_ref"] is not None:
        _nonempty_string(ep["subgoal_ref"], f"{where}.subgoal_ref")
    _nonempty_string(ep["strategy"], f"{where}.strategy")
    _nonempty_string(ep["representation"], f"{where}.representation")
    _nonempty_string(ep["tactic_family"], f"{where}.tactic_family")
    _nonempty_string(ep["parameters_ref"], f"{where}.parameters_ref")
    _nonempty_string(ep["budget_ref"], f"{where}.budget_ref")
    outcome = _nonempty_string(ep["outcome"], f"{where}.outcome")
    if outcome not in SEARCH_OUTCOMES:
        raise ValidationError(f"{where}.outcome: unsupported value {outcome!r}")

    failure_class = ep["failure_class"]
    if failure_class is not None:
        _nonempty_string(failure_class, f"{where}.failure_class")
    if outcome in NON_TRUTH_OUTCOMES and failure_class is None:
        raise ValidationError(f"{where}: {outcome} requires failure_class")

    cost = ep["cost"]
    if not isinstance(cost, dict):
        raise ValidationError(f"{where}.cost: expected object")
    _closed(cost, COST_KEYS, f"{where}.cost")
    _required(cost, tuple(sorted(COST_KEYS)), f"{where}.cost")
    for key in COST_KEYS:
        _nonnegative_int(cost[key], f"{where}.cost.{key}")

    artifacts = _string_list(ep["useful_artifact_refs"], f"{where}.useful_artifact_refs")
    provenance = _string_list(ep["source_provenance_refs"], f"{where}.source_provenance_refs")
    if not provenance:
        raise ValidationError(f"{where}: source_provenance_refs must not be empty")
    if outcome == "CounterexampleFound" and not artifacts:
        raise ValidationError(f"{where}: CounterexampleFound requires a useful artifact reference")

    if not isinstance(ep["transfer_context"], dict):
        raise ValidationError(f"{where}.transfer_context: expected object")
    _nonnegative_int(ep["timestamp"], f"{where}.timestamp")
    _nonempty_string(ep["retrieval_embedding_ref"], f"{where}.retrieval_embedding_ref")
    _score(ep["phi_search_score"], f"{where}.phi_search_score")
    return episode_id


def validate_document(doc: object) -> None:
    if not isinstance(doc, dict):
        raise ValidationError("root: expected object")
    _closed(doc, ROOT_KEYS, "root")
    _required(doc, ("schema_version", "research_graph_ref", "result_memory", "search_memory"), "root")
    if doc["schema_version"] != SCHEMA_VERSION:
        raise ValidationError(f"root.schema_version: expected {SCHEMA_VERSION!r}")
    _nonempty_string(doc["research_graph_ref"], "root.research_graph_ref")
    if not isinstance(doc["result_memory"], list):
        raise ValidationError("root.result_memory: expected list")
    if not isinstance(doc["search_memory"], list):
        raise ValidationError("root.search_memory: expected list")

    ids: set[str] = set()
    for index, ep in enumerate(doc["result_memory"]):
        episode_id = validate_result_episode(ep, index)
        if episode_id in ids:
            raise ValidationError(f"duplicate episode_id across memories: {episode_id}")
        ids.add(episode_id)
    for index, ep in enumerate(doc["search_memory"]):
        episode_id = validate_search_episode(ep, index)
        if episode_id in ids:
            raise ValidationError(f"duplicate episode_id across memories: {episode_id}")
        ids.add(episode_id)


def _fixture() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "research_graph_ref": "sha256:research-graph",
        "result_memory": [
            {
                "episode_id": "result-1",
                "claim_id": "claim-1",
                "result_kind": "FormalLemma",
                "evidence_state": "KernelAccepted",
                "evidence_refs": ["receipt:lean-1"],
                "method": "Lean",
                "representation": "Lean4",
                "dependency_claim_ids": [],
                "timestamp": 1,
                "retrieval_embedding_ref": "hdc:result-1",
                "phi_search_score": 0.1,
            }
        ],
        "search_memory": [
            {
                "episode_id": "search-1",
                "research_program_id": "program-1",
                "problem_ref": "claim-1",
                "subgoal_ref": None,
                "strategy": "try induction",
                "representation": "recursive structure",
                "tactic_family": "induction",
                "parameters_ref": "sha256:params",
                "budget_ref": "sha256:budget",
                "outcome": "Timeout",
                "failure_class": "TimeBudgetExceeded",
                "cost": {
                    "wall_ms": 1000,
                    "cpu_ms": 900,
                    "proof_calls": 3,
                    "solver_calls": 0,
                    "search_nodes": 12,
                },
                "useful_artifact_refs": [],
                "source_provenance_refs": ["branch:abandoned-1"],
                "transfer_context": {"family": "induction"},
                "timestamp": 2,
                "retrieval_embedding_ref": "hdc:search-1",
                "phi_search_score": 0.9,
            }
        ],
    }


def self_test() -> None:
    valid = _fixture()
    validate_document(valid)

    attacks: list[tuple[str, callable]] = []

    def add(name: str, mutate) -> None:
        attacks.append((name, mutate))

    add("search cannot claim formal authority", lambda d: d["search_memory"][0].__setitem__("formal_authority", "KernelAccepted"))
    add("timeout needs failure class", lambda d: d["search_memory"][0].__setitem__("failure_class", None))
    add("formal result needs receipt", lambda d: d["result_memory"][0].__setitem__("evidence_refs", []))
    add("phi must be bounded", lambda d: d["search_memory"][0].__setitem__("phi_search_score", 99.0))
    add("episode ids globally unique", lambda d: d["search_memory"][0].__setitem__("episode_id", "result-1"))
    add("counterexample requires artifact", lambda d: (d["search_memory"][0].__setitem__("outcome", "CounterexampleFound"), d["search_memory"][0].__setitem__("failure_class", None)))

    for name, mutate in attacks:
        candidate = json.loads(json.dumps(valid))
        mutate(candidate)
        try:
            validate_document(candidate)
        except ValidationError:
            continue
        raise AssertionError(f"self-test attack unexpectedly passed: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("math-search-memory validator self-test: PASS")
        return 0
    if args.path is None:
        parser.error("path is required unless --self-test is used")
    try:
        validate_document(json.loads(args.path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
