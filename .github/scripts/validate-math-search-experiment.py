#!/usr/bin/env python3
"""Validate frozen Symthaea mathematical-search experiment manifests.

Stdlib-only on purpose: preregistration must not introduce a new build dependency.
The manifest establishes experiment identity and equal-budget/contamination
constraints.  It never establishes mathematical truth or HDC superiority.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCHEMA_VERSION = "symthaea.math-search-experiment.v1"
STAGES = {
    "Q0StructuralRetrieval",
    "Q1BlindSolvedTransfer",
    "Q2CrossDomainTransfer",
    "Q3FormalConjecturesRediscovery",
    "Q4ResearchHeldOut",
}
ARMS = {"A", "B", "C", "D"}
PRIMARY_ENDPOINTS = {
    "FormallySolvedRate",
    "VerifiedUsefulLemmaRate",
    "ValidCounterexamples",
    "ProofCallsPerSolved",
    "SearchNodesPerSolved",
    "NormalizedComputePerSolved",
    "TimeToFirstUsefulLemma",
    "CrossDomainTransferRate",
    "RepeatedFailureRate",
    "FalsePruningRate",
}
REQUIRED_NEGATIVE_CONTROLS = {
    "RandomRetrieval",
    "LexicalRetrieval",
    "ShuffledHdcVectors",
    "PermutedChallengeAssociations",
    "MajorityStrategy",
}
REQUIRED_MANIPULATION_CHECKS = {
    "RetrievalDiffersFromBaseline",
    "StructuralNeighborShift",
    "StrategyDistributionShift",
    "SearchTrajectoryShift",
}
ROOT_KEYS = {
    "schema_version",
    "experiment_id",
    "stage",
    "frozen_before_evaluation",
    "challenge_set_sha256",
    "corpus_snapshot_sha256",
    "knowledge_boundary_sha256",
    "statistical_plan_sha256",
    "budget_contract_sha256",
    "toolchain_manifest_sha256",
    "human_intervention_policy_sha256",
    "encoder_sha256",
    "seeds",
    "primary_endpoints",
    "secondary_endpoints",
    "negative_controls",
    "manipulation_checks",
    "arms",
}
ARM_KEYS = {
    "arm_id",
    "retrieval_kind",
    "budget_contract_sha256",
    "challenge_set_sha256",
    "corpus_snapshot_sha256",
    "knowledge_boundary_sha256",
    "toolchain_manifest_sha256",
    "encoder_sha256",
    "negative_search_memory",
    "evolutionary_search",
}


class ValidationError(ValueError):
    pass


def _closed(obj: dict, allowed: set[str], where: str) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        raise ValidationError(f"{where}: unknown fields: {', '.join(extra)}")


def _req(obj: dict, fields: set[str], where: str) -> None:
    missing = sorted(fields - set(obj))
    if missing:
        raise ValidationError(f"{where}: missing fields: {', '.join(missing)}")


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{where}: expected non-empty string")
    return value


def _sha(value: object, where: str) -> str:
    text = _text(value, where)
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValidationError(f"{where}: expected sha256:<64 lowercase hex>")
    digest = text[7:]
    if any(ch not in "0123456789abcdef" for ch in digest):
        raise ValidationError(f"{where}: invalid SHA-256 digest")
    return text


def _unique_strings(value: object, where: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{where}: expected non-empty list")
    out = [_text(item, f"{where}[{i}]") for i, item in enumerate(value)]
    if len(out) != len(set(out)):
        raise ValidationError(f"{where}: duplicates are not allowed")
    return out


def validate_manifest(doc: object) -> None:
    if not isinstance(doc, dict):
        raise ValidationError("root: expected object")
    _closed(doc, ROOT_KEYS, "root")
    _req(doc, ROOT_KEYS, "root")
    if doc["schema_version"] != SCHEMA_VERSION:
        raise ValidationError(f"root.schema_version: expected {SCHEMA_VERSION}")
    _text(doc["experiment_id"], "root.experiment_id")
    if doc["stage"] not in STAGES:
        raise ValidationError("root.stage: unsupported stage")
    if doc["frozen_before_evaluation"] is not True:
        raise ValidationError("root.frozen_before_evaluation: must be true")

    shared_sha_fields = (
        "challenge_set_sha256",
        "corpus_snapshot_sha256",
        "knowledge_boundary_sha256",
        "statistical_plan_sha256",
        "budget_contract_sha256",
        "toolchain_manifest_sha256",
        "human_intervention_policy_sha256",
        "encoder_sha256",
    )
    for field in shared_sha_fields:
        _sha(doc[field], f"root.{field}")

    seeds = doc["seeds"]
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ValidationError("root.seeds: require at least three preregistered seeds")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValidationError("root.seeds: seeds must be non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValidationError("root.seeds: duplicate seeds are not allowed")

    primary = set(_unique_strings(doc["primary_endpoints"], "root.primary_endpoints"))
    if not primary <= PRIMARY_ENDPOINTS:
        raise ValidationError(f"root.primary_endpoints: unsupported endpoint(s): {sorted(primary - PRIMARY_ENDPOINTS)}")
    secondary = _unique_strings(doc["secondary_endpoints"], "root.secondary_endpoints")
    if any(name in primary for name in secondary):
        raise ValidationError("root.secondary_endpoints: primary endpoints must not be duplicated")

    controls = set(_unique_strings(doc["negative_controls"], "root.negative_controls"))
    missing_controls = REQUIRED_NEGATIVE_CONTROLS - controls
    if missing_controls:
        raise ValidationError(f"root.negative_controls: missing {sorted(missing_controls)}")

    manipulation = set(_unique_strings(doc["manipulation_checks"], "root.manipulation_checks"))
    missing_checks = REQUIRED_MANIPULATION_CHECKS - manipulation
    if missing_checks:
        raise ValidationError(f"root.manipulation_checks: missing {sorted(missing_checks)}")

    arms = doc["arms"]
    if not isinstance(arms, list) or len(arms) < 3:
        raise ValidationError("root.arms: require at least A/B/C")
    seen: set[str] = set()
    for index, arm in enumerate(arms):
        where = f"root.arms[{index}]"
        if not isinstance(arm, dict):
            raise ValidationError(f"{where}: expected object")
        _closed(arm, ARM_KEYS, where)
        _req(arm, ARM_KEYS, where)
        arm_id = arm["arm_id"]
        if arm_id not in ARMS or arm_id in seen:
            raise ValidationError(f"{where}.arm_id: invalid or duplicate arm")
        seen.add(arm_id)
        _text(arm["retrieval_kind"], f"{where}.retrieval_kind")
        for field in (
            "budget_contract_sha256",
            "challenge_set_sha256",
            "corpus_snapshot_sha256",
            "knowledge_boundary_sha256",
            "toolchain_manifest_sha256",
            "encoder_sha256",
        ):
            _sha(arm[field], f"{where}.{field}")
        if arm["budget_contract_sha256"] != doc["budget_contract_sha256"]:
            raise ValidationError(f"{where}: budget contract differs from experiment")
        if arm["challenge_set_sha256"] != doc["challenge_set_sha256"]:
            raise ValidationError(f"{where}: challenge set differs from experiment")
        if arm["corpus_snapshot_sha256"] != doc["corpus_snapshot_sha256"]:
            raise ValidationError(f"{where}: corpus snapshot differs from experiment")
        if arm["knowledge_boundary_sha256"] != doc["knowledge_boundary_sha256"]:
            raise ValidationError(f"{where}: knowledge boundary differs from experiment")
        if arm["toolchain_manifest_sha256"] != doc["toolchain_manifest_sha256"]:
            raise ValidationError(f"{where}: toolchain manifest differs from experiment")
        if arm["encoder_sha256"] != doc["encoder_sha256"]:
            raise ValidationError(f"{where}: encoder identity differs from experiment")
        if not isinstance(arm["negative_search_memory"], bool):
            raise ValidationError(f"{where}.negative_search_memory: expected boolean")
        if arm["evolutionary_search"] is not False:
            raise ValidationError(f"{where}.evolutionary_search: v1 causal HDC experiment forbids evolutionary search")

    if not {"A", "B", "C"} <= seen:
        raise ValidationError("root.arms: baseline A, conventional retrieval B, and HDC C are mandatory")

    by_id = {arm["arm_id"]: arm for arm in arms}
    if by_id["A"]["negative_search_memory"]:
        raise ValidationError("arm A: prover-only baseline cannot use negative-search memory")
    if by_id["B"]["negative_search_memory"]:
        raise ValidationError("arm B: conventional retrieval baseline cannot use negative-search memory in v1")
    if by_id["C"]["negative_search_memory"]:
        raise ValidationError("arm C: HDC-only arm cannot use negative-search memory; reserve that intervention for D")
    if "D" in by_id and not by_id["D"]["negative_search_memory"]:
        raise ValidationError("arm D: must enable negative-search memory")


def _digest(label: str) -> str:
    # Fixtures only; still syntactically valid SHA-256 identities.
    return "sha256:" + (label.encode().hex() + "0" * 64)[:64]


def fixture() -> dict:
    common = {
        "budget_contract_sha256": _digest("budget"),
        "challenge_set_sha256": _digest("challenges"),
        "corpus_snapshot_sha256": _digest("corpus"),
        "knowledge_boundary_sha256": _digest("knowledge"),
        "toolchain_manifest_sha256": _digest("toolchain"),
        "encoder_sha256": _digest("encoder"),
    }
    def arm(arm_id: str, retrieval: str, negative: bool) -> dict:
        return {
            "arm_id": arm_id,
            "retrieval_kind": retrieval,
            **common,
            "negative_search_memory": negative,
            "evolutionary_search": False,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "MATH-EXP-001-fixture",
        "stage": "Q1BlindSolvedTransfer",
        "frozen_before_evaluation": True,
        **common,
        "statistical_plan_sha256": _digest("statistics"),
        "human_intervention_policy_sha256": _digest("human"),
        "seeds": [11, 23, 47],
        "primary_endpoints": ["FormallySolvedRate", "ProofCallsPerSolved", "RepeatedFailureRate"],
        "secondary_endpoints": ["Phi", "ProofLength", "StrategyDiversity"],
        "negative_controls": sorted(REQUIRED_NEGATIVE_CONTROLS),
        "manipulation_checks": sorted(REQUIRED_MANIPULATION_CHECKS),
        "arms": [arm("A", "None", False), arm("B", "Conventional", False), arm("C", "StructuralHDC", False), arm("D", "StructuralHDC", True)],
    }


def self_test() -> None:
    valid = fixture()
    validate_manifest(valid)
    attacks = []
    def add(name, mutate): attacks.append((name, mutate))
    add("post-hoc manifest", lambda d: d.__setitem__("frozen_before_evaluation", False))
    add("unequal budget", lambda d: d["arms"][2].__setitem__("budget_contract_sha256", _digest("bigger")))
    add("different corpus", lambda d: d["arms"][2].__setitem__("corpus_snapshot_sha256", _digest("leaky")))
    add("evolution confound", lambda d: d["arms"][2].__setitem__("evolutionary_search", True))
    add("negative memory confound in C", lambda d: d["arms"][2].__setitem__("negative_search_memory", True))
    add("missing shuffled-HDC control", lambda d: d["negative_controls"].remove("ShuffledHdcVectors"))
    add("encoder drift", lambda d: d["arms"][2].__setitem__("encoder_sha256", _digest("changed")))
    add("duplicate seed", lambda d: d.__setitem__("seeds", [11, 11, 47]))
    for name, mutate in attacks:
        candidate = json.loads(json.dumps(valid))
        mutate(candidate)
        try:
            validate_manifest(candidate)
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
        print("math-search experiment manifest self-test: PASS")
        return 0
    if args.path is None:
        parser.error("path is required unless --self-test is used")
    try:
        validate_manifest(json.loads(args.path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
