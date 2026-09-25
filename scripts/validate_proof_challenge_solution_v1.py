#!/usr/bin/env python3
"""Validate Proof Challenge / Solution V1 and hostile trust-topology mutations."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs/formal/proof_challenge_solution_v1.json"
OBS_PATH = ROOT / "docs/formal/proof_observability_v1.json"
EVIDENCE_PATH = ROOT / "docs/formal/formal_verification_evidence_classes_v1.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_NONCLAIMS = {
    "kernel replay != theorem-intent correctness",
    "challenge/solution agreement != source refinement",
    "independent certificate check != runtime authority",
    "multiple checkers != elimination of shared assumptions",
}

CHALLENGE_FIELDS = {
    "id", "obligation_id", "human_claim", "formal_statement", "statement_sha256",
    "permitted_axioms", "trust_roots", "subject_sha256", "evidence_class",
    "coverage_digest", "required_negative_controls", "claim_ceiling", "checker_profile",
}

SOLUTION_FIELDS = {
    "id", "challenge_sha256", "presented_human_claim", "producer_toolchain",
    "artifact_digests", "checker_receipts", "producer_warnings", "result",
}


def die(message: str) -> None:
    raise ValueError(message)


def digest_json(obj: dict) -> str:
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def statement_digest(statement: str) -> str:
    return hashlib.sha256(statement.encode("utf-8")).hexdigest()


def validate_challenge(challenge: dict, profiles: set[str], evidence_ids: set[str]) -> str:
    missing = CHALLENGE_FIELDS - set(challenge)
    if missing:
        die(f"challenge missing fields: {sorted(missing)}")
    if not challenge["human_claim"] or not challenge["formal_statement"]:
        die("challenge claim/statement must be non-empty")
    if challenge["statement_sha256"] != statement_digest(challenge["formal_statement"]):
        die("challenge statement digest mismatch")
    if not HEX64.fullmatch(challenge["subject_sha256"]):
        die("challenge subject digest invalid")
    if not HEX64.fullmatch(challenge["coverage_digest"]):
        die("challenge coverage digest invalid")
    for field in ("permitted_axioms", "trust_roots", "required_negative_controls", "claim_ceiling"):
        if not isinstance(challenge[field], list):
            die(f"challenge {field} must be an array")
    if not challenge["claim_ceiling"]:
        die("challenge claim ceiling must be explicit")
    if challenge["evidence_class"] not in evidence_ids:
        die("challenge evidence class unknown")
    if challenge["checker_profile"] not in profiles:
        die("challenge checker profile unknown")
    return digest_json(challenge)


def validate_solution(solution: dict, challenge: dict, challenge_digest: str, profiles: set[str]) -> None:
    missing = SOLUTION_FIELDS - set(solution)
    if missing:
        die(f"solution missing fields: {sorted(missing)}")
    if solution["challenge_sha256"] != challenge_digest:
        die("solution does not bind exact challenge digest")
    if solution["presented_human_claim"] != challenge["human_claim"]:
        die("solution presentation claim diverges from trusted challenge")
    if not solution["producer_toolchain"]:
        die("solution producer toolchain must be named")
    if solution["result"] not in {"PASS", "FAIL"}:
        die("solution result must be PASS or FAIL")
    for field in ("artifact_digests", "checker_receipts", "producer_warnings"):
        if not isinstance(solution[field], list):
            die(f"solution {field} must be an array")
    if not solution["artifact_digests"]:
        die("solution must bind at least one proof/certificate artifact")
    for digest in solution["artifact_digests"]:
        if not HEX64.fullmatch(digest):
            die("solution artifact digest invalid")

    pass_profiles: set[str] = set()
    for receipt in solution["checker_receipts"]:
        profile = receipt.get("profile")
        if profile not in profiles:
            die("checker receipt profile unknown")
        if not receipt.get("checker_identity"):
            die("checker identity must be explicit")
        if receipt.get("problem_sha256") != challenge_digest:
            die("checker receipt is bound to a different challenge/problem")
        result = receipt.get("result")
        if result not in {"PASS", "FAIL"}:
            die("checker receipt result must be PASS or FAIL")
        if result == "PASS":
            pass_profiles.add(profile)
        elif solution["result"] == "PASS":
            die("candidate PASS conflicts with an admitted checker FAIL")

    required = challenge["checker_profile"]
    if solution["result"] == "PASS":
        if required == "ProducerOnly":
            return
        if required not in pass_profiles:
            die(f"required checker profile {required} did not PASS")


def expect_invalid(challenge: dict, solution: dict, profiles: set[str], evidence_ids: set[str]) -> None:
    try:
        digest = validate_challenge(challenge, profiles, evidence_ids)
        validate_solution(solution, challenge, digest, profiles)
    except ValueError:
        return
    die("negative control unexpectedly validated")


def main() -> int:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    observability = json.loads(OBS_PATH.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    if contract.get("schema") != "symthaea.formal-verification.proof-challenge-solution.v1":
        die("wrong challenge/solution schema")
    if contract.get("authority") != "TrustTopologyMetadataOnly":
        die("challenge/solution authority escalated")
    if observability.get("schema") != "symthaea.formal-verification.proof-observability.v1":
        die("parent observability contract missing or wrong")
    if contract.get("parent_contract") != observability.get("schema"):
        die("parent contract drifted")
    if set(contract.get("required_nonclaims", [])) != REQUIRED_NONCLAIMS:
        die("required nonclaims changed")

    profiles = set(contract.get("checker_profiles", []))
    if profiles != {"ProducerOnly", "ChallengeComparator", "KernelReplay", "IndependentCertificate"}:
        die("checker profile census changed")
    evidence_ids = {entry["id"] for entry in evidence.get("classes", [])}

    challenge = contract["synthetic_challenge"]
    solution = contract["synthetic_solution"]
    challenge_digest = validate_challenge(challenge, profiles, evidence_ids)
    if contract.get("synthetic_challenge_sha256") != challenge_digest:
        die("declared synthetic challenge digest mismatch")
    validate_solution(solution, challenge, challenge_digest, profiles)

    # Candidate cannot mutate the trusted theorem statement under a stale digest.
    statement_mutation = copy.deepcopy(challenge)
    statement_mutation["formal_statement"] += " and True"
    expect_invalid(statement_mutation, copy.deepcopy(solution), profiles, evidence_ids)

    # Candidate cannot inject a new axiom without producing a new challenge identity.
    axiom_injection = copy.deepcopy(challenge)
    axiom_injection["permitted_axioms"].append("Classical.choice")
    expect_invalid(axiom_injection, copy.deepcopy(solution), profiles, evidence_ids)

    # Solution must bind the exact challenge.
    wrong_challenge = copy.deepcopy(solution)
    wrong_challenge["challenge_sha256"] = "4" * 64
    expect_invalid(copy.deepcopy(challenge), wrong_challenge, profiles, evidence_ids)

    # Checker receipt must be for the same problem/challenge.
    wrong_problem = copy.deepcopy(solution)
    wrong_problem["checker_receipts"][0]["problem_sha256"] = "5" * 64
    expect_invalid(copy.deepcopy(challenge), wrong_problem, profiles, evidence_ids)

    # Producer PASS cannot override an admitted checker FAIL.
    checker_fail = copy.deepcopy(solution)
    checker_fail["checker_receipts"][1]["result"] = "FAIL"
    expect_invalid(copy.deepcopy(challenge), checker_fail, profiles, evidence_ids)

    # Required checker profile must actually appear and pass.
    missing_kernel = copy.deepcopy(solution)
    missing_kernel["checker_receipts"] = [missing_kernel["checker_receipts"][0]]
    expect_invalid(copy.deepcopy(challenge), missing_kernel, profiles, evidence_ids)

    # Solution presentation cannot widen or rewrite the trusted human claim.
    widened_claim = copy.deepcopy(solution)
    widened_claim["presented_human_claim"] = "All implementations satisfy P for all inputs."
    expect_invalid(copy.deepcopy(challenge), widened_claim, profiles, evidence_ids)

    # Artifact identity is mandatory and content-addressed.
    bad_artifact = copy.deepcopy(solution)
    bad_artifact["artifact_digests"][0] = "not-a-digest"
    expect_invalid(copy.deepcopy(challenge), bad_artifact, profiles, evidence_ids)

    print(f"CHALLENGE sha256={challenge_digest}")
    print("PROOF_CHALLENGE_SOLUTION_V1_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"PROOF_CHALLENGE_SOLUTION_V1_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
