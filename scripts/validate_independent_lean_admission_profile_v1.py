#!/usr/bin/env python3
"""Validate the immutable candidate profile for independent Lean admission.

This validates supply-chain/profile semantics only. It deliberately does not
claim comparator, nanoda, export, sandbox, or theorem execution succeeded.
"""

from __future__ import annotations

import copy
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "docs/formal/independent_lean_admission_profile_v1.json"

HEX40 = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_TOOLS = {
    "landrun": ("zouuup/landrun", "5ed4a3db3a4ad930d577215c6b9abaa19df7f99f"),
    "lean4export": ("leanprover/lean4export", "076e8e57707e813375e8f9da8bf989799ace9680"),
    "comparator": ("leanprover/comparator", "d03acab154d269c06e60e4de7e4cc85deebff94b"),
    "nanoda": ("robsimmons/nanoda_lib", "68d5ca9db226849b41a6fff59d796ff19d0a8840"),
}
EXPECTED_MAPPING = {
    "all_required_checkers_accept_and_axiom_policy_accepts": "Pass",
    "checker_disagreement": "Blocked",
    "checker_or_sandbox_unavailable": "EnvironmentFailure",
    "malformed_or_rejected_proof_subject": "Fail",
}
EXPECTED_DIAGNOSTICS = {
    "OfficialAcceptsExternalRejects",
    "OfficialRejectsExternalAccepts",
    "ExternalCheckerUnavailable",
    "ExportIncompatible",
    "SandboxUnavailable",
    "AxiomPolicyRejected",
    "StatementBindingMismatch",
}


class ProfileError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileError(message)


def validate(profile: dict) -> None:
    require(profile.get("schema") == "symthaea.formal.independent-lean-admission-profile.v1", "schema drift")
    require(profile.get("tracking_issue") == 6014, "tracking issue drift")
    require(profile.get("authority") == "EvidenceOnly", "authority escalation")
    require(profile.get("status") == "CandidateProfileNeedsQualification", "candidate profile may not self-promote")

    lean = profile.get("lean")
    require(isinstance(lean, dict), "lean profile missing")
    require(lean.get("repository") == "leanprover/lean4", "Lean repository drift")
    require(lean.get("toolchain") == "v4.34.1", "Lean candidate toolchain drift")
    require(lean.get("release_date") == "2026-09-24", "Lean release identity drift")

    tools = profile.get("tools")
    require(isinstance(tools, dict) and set(tools) == set(EXPECTED_TOOLS), "tool census drift")
    for name, (repository, commit) in EXPECTED_TOOLS.items():
        entry = tools[name]
        require(entry.get("repository") == repository, f"{name}: repository drift")
        require(entry.get("commit") == commit, f"{name}: immutable pin drift")
        require(HEX40.fullmatch(entry.get("commit", "")) is not None, f"{name}: pin must be full 40-hex commit")
        for moving in ("main", "master", "latest", "stable", "HEAD"):
            require(entry.get("commit") != moving, f"{name}: moving reference forbidden")

    require(
        tools["lean4export"].get("build_rule")
        == "build against the exact workspace lean-toolchain; moving/default toolchain forbidden",
        "lean4export exact-toolchain build rule required",
    )

    upstream = profile.get("upstream_reference")
    require(isinstance(upstream, dict), "upstream reference missing")
    require(upstream.get("lean_version_in_reference_profile") == "v4.34.0", "upstream reference version drift")
    require("v4.34.1" in upstream.get("candidate_difference", ""), "candidate compatibility delta must remain explicit")

    compatibility = profile.get("compatibility")
    require(isinstance(compatibility, dict) and compatibility, "compatibility block required")
    require(all(value == "NeedsQualification" for value in compatibility.values()),
            "profile may not predeclare compatibility without execution")

    require(profile.get("required_checkers") == ["OfficialLeanKernel", "Nanoda"], "required checker census drift")
    paranoid = profile.get("optional_later_profile")
    require(isinstance(paranoid, dict) and paranoid.get("status") == "NotAdmittedByThisProfile",
            "paranoid profile may not be silently admitted")

    binding = profile.get("statement_binding")
    require(isinstance(binding, dict) and binding, "statement binding contract required")
    require(all(value is True for value in binding.values()), "all v1 statement-binding invariants are required")

    axioms = profile.get("axiom_policy")
    require(isinstance(axioms, dict), "axiom policy missing")
    require(axioms.get("mode") == "ConstitutionalAllowlist", "axiom policy mode drift")
    require(axioms.get("sorry_ax_forbidden") is True, "sorryAx must remain forbidden")
    require(axioms.get("undeclared_axioms_forbidden") is True, "undeclared axioms must remain forbidden")
    require(axioms.get("policy_identity_required_in_receipt") is True, "axiom-policy identity must be receipt-bound")

    require(profile.get("qualification_result_mapping") == EXPECTED_MAPPING, "canonical qualification mapping drift")
    require(set(profile.get("diagnostic_classes", [])) == EXPECTED_DIAGNOSTICS, "diagnostic class census drift")

    ceiling = profile.get("claim_ceiling")
    require(isinstance(ceiling, list) and len(ceiling) >= 5, "claim ceiling incomplete")
    joined = "\n".join(ceiling).lower()
    for phrase in ("does not strengthen theorem statement semantics", "does not establish rust source refinement", "does not grant runtime authority"):
        require(phrase in joined, f"claim ceiling missing: {phrase}")

    nonclaims = set(profile.get("nonclaims", []))
    for required in {
        "ComparatorPipelineQualified",
        "Lean4341CompatibilityEstablished",
        "NanodaAgreementExecuted",
        "TheoremTruthEstablishedByThisManifest",
        "SourceRefinementEstablished",
        "RuntimeAuthority",
    }:
        require(required in nonclaims, f"required nonclaim missing: {required}")


def expect_reject(name: str, profile: dict) -> None:
    try:
        validate(profile)
    except ProfileError:
        return
    raise AssertionError(f"hostile mutant unexpectedly accepted: {name}")


def self_test(profile: dict) -> None:
    mutant = copy.deepcopy(profile)
    mutant["tools"]["comparator"]["commit"] = "main"
    expect_reject("moving-comparator-ref", mutant)

    mutant = copy.deepcopy(profile)
    mutant["tools"]["nanoda"]["commit"] = "0" * 40
    expect_reject("nanoda-pin-substitution", mutant)

    mutant = copy.deepcopy(profile)
    mutant["status"] = "Qualified"
    expect_reject("profile-self-promotion", mutant)

    mutant = copy.deepcopy(profile)
    mutant["compatibility"]["lean_4_34_1_with_pinned_comparator"] = "Established"
    expect_reject("unexecuted-compatibility-promotion", mutant)

    mutant = copy.deepcopy(profile)
    mutant["required_checkers"] = ["OfficialLeanKernel"]
    expect_reject("external-checker-drop", mutant)

    mutant = copy.deepcopy(profile)
    mutant["qualification_result_mapping"]["checker_disagreement"] = "Pass"
    expect_reject("checker-disagreement-promoted", mutant)

    mutant = copy.deepcopy(profile)
    mutant["qualification_result_mapping"]["checker_or_sandbox_unavailable"] = "Pass"
    expect_reject("environment-failure-promoted", mutant)

    mutant = copy.deepcopy(profile)
    mutant["axiom_policy"]["sorry_ax_forbidden"] = False
    expect_reject("sorryax-permitted", mutant)

    mutant = copy.deepcopy(profile)
    mutant["statement_binding"]["proof_receipt_may_not_survive_statement_digest_change"] = False
    expect_reject("stale-statement-receipt", mutant)


def main() -> int:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    validate(profile)
    self_test(profile)
    print("independent_lean_admission_profile_v1=PASS")
    print("profile_status=CandidateProfileNeedsQualification")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"independent_lean_admission_profile_v1=FAIL: {exc}", file=sys.stderr)
        raise
