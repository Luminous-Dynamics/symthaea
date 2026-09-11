#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ORACLE = pathlib.Path(__file__).with_name("continuity-actuation-enforcement-invalidation-oracle.py")
SCHEMA = "symthaea-continuity-actuation-enforcement-invalidation-oracle-v1"

HARD_FIELDS = (
    "backend_id", "backend_implementation_digest", "backend_generation",
    "enforcement_profile_id", "boundary_implementation_digest", "one_use_mechanism_digest",
    "enforcement_profile_generation", "authentication_profile_id", "authentication_implementation_digest",
    "authentication_root_id", "authentication_root_epoch", "harness_implementation_digest",
    "scenario_suite_manifest_digest", "environment_manifest_digest", "topology_dependency_manifest_digest",
    "hardware_firmware_manifest_digest", "toolchain_realization_digest", "obligation_schema_digest",
    "evidence_basis_mapping_digest", "evidence_manifest_digest", "verifier_semantics_digest",
    "verifier_adoption_id",
)


def hx(n: int) -> str:
    return (bytes([n]) * 32).hex()


def base_request():
    hard = {}
    for i, field in enumerate(HARD_FIELDS, start=1):
        if field in {"backend_generation", "enforcement_profile_generation", "authentication_root_epoch"}:
            hard[field] = i
        else:
            hard[field] = hx(i)
    return {
        "schema": SCHEMA,
        "baseline": dict(hard),
        "current": dict(hard),
        "fresh": {
            "verifier_challenge_current": True,
            "currentness_reestablished": True,
            "trusted_time_epoch_advanced_normally": False,
            "periodic_confirmation_due": False,
            "successor_readmitted_exact_campaign": False,
            "root_rotation_portable": False,
            "root_rotation_admission_id": None,
        },
        "attempt": {
            "owner_authority_current": True,
            "distributed_context_unchanged": True,
            "subject_target_unchanged": True,
            "active_lkg_unchanged": True,
            "execution_attempt_unchanged": True,
            "actuation_fence_unchanged": True,
            "deny_or_emergency_stop": False,
            "permit_unconsumed": True,
            "resource_available": True,
        },
        "diagnostic": {
            "human_label_changed": False,
            "log_storage_location_changed": False,
            "report_format_changed": False,
            "comments_changed": False,
            "packaging_filename_changed": False,
        },
    }


def execute(request):
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "request.json"
        path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(ORACLE), str(path)], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if result.returncode != 0:
            raise AssertionError(f"oracle denied valid request: stdout={result.stdout!r} stderr={result.stderr!r}")
        return json.loads(result.stdout)


def require(request, classification, reason=None):
    result = execute(request)
    if result["classification"] != classification:
        raise AssertionError(f"expected {classification}, got {result!r}")
    if reason is not None and reason not in result["reasons"]:
        raise AssertionError(f"expected reason {reason!r}, got {result!r}")
    return result


def main() -> int:
    cases = 0

    require(base_request(), "CAMPAIGN_REUSABLE"); cases += 1

    req = base_request(); req["current"]["backend_implementation_digest"] = hx(90)
    req["fresh"]["verifier_challenge_current"] = True
    require(req, "FULL_CAMPAIGN_REQUALIFICATION", "hard_identity_changed:backend_implementation_digest"); cases += 1

    req = base_request(); req["current"]["toolchain_realization_digest"] = hx(91)
    req["fresh"]["currentness_reestablished"] = True
    require(req, "FULL_CAMPAIGN_REQUALIFICATION", "hard_identity_changed:toolchain_realization_digest"); cases += 1

    req = base_request(); req["current"]["hardware_firmware_manifest_digest"] = hx(92)
    req["fresh"]["periodic_confirmation_due"] = True
    require(req, "FULL_CAMPAIGN_REQUALIFICATION", "hard_identity_changed:hardware_firmware_manifest_digest"); cases += 1

    req = base_request(); req["fresh"]["verifier_challenge_current"] = False
    require(req, "FRESH_CURRENTNESS_READMISSION", "verifier_challenge_not_current"); cases += 1

    req = base_request(); req["attempt"]["owner_authority_current"] = False
    require(req, "ATTEMPT_LOCAL_REQUALIFICATION", "owner_authority_revoked_or_superseded"); cases += 1

    req = base_request(); req["attempt"]["resource_available"] = False
    require(req, "ATTEMPT_LOCAL_REQUALIFICATION", "resource_temporarily_unavailable"); cases += 1

    req = base_request(); req["diagnostic"]["human_label_changed"] = True
    result = require(req, "CAMPAIGN_REUSABLE")
    assert result["diagnostic_changes"] == ["human_label_changed"]; cases += 1

    req = base_request(); req["current"]["authentication_root_id"] = hx(93); req["current"]["authentication_root_epoch"] += 1
    require(req, "FULL_CAMPAIGN_REQUALIFICATION", "authentication_root_changed_without_portable_successor_admission"); cases += 1

    req = base_request(); req["current"]["authentication_root_id"] = hx(93); req["current"]["authentication_root_epoch"] += 1
    req["fresh"]["root_rotation_admission_id"] = hx(94)
    req["fresh"]["root_rotation_portable"] = True
    req["fresh"]["successor_readmitted_exact_campaign"] = True
    require(req, "FRESH_CURRENTNESS_READMISSION", "authenticated_root_rotation_requires_fresh_readmission"); cases += 1

    req = base_request(); req["current"]["verifier_adoption_id"] = hx(95)
    require(req, "FULL_CAMPAIGN_REQUALIFICATION", "verifier_adoption_superseded_without_exact_campaign_readmission"); cases += 1

    req = base_request(); req["current"]["verifier_adoption_id"] = hx(95)
    req["fresh"]["successor_readmitted_exact_campaign"] = True
    require(req, "FRESH_CURRENTNESS_READMISSION", "verifier_adoption_advanced_with_explicit_exact_campaign_readmission"); cases += 1

    req = base_request(); req["fresh"]["verifier_challenge_current"] = False; req["attempt"]["owner_authority_current"] = False
    require(req, "FRESH_CURRENTNESS_READMISSION", "verifier_challenge_not_current"); cases += 1

    req = base_request(); req["current"]["verifier_semantics_digest"] = hx(96); req["attempt"]["deny_or_emergency_stop"] = True
    require(req, "FULL_CAMPAIGN_REQUALIFICATION", "hard_identity_changed:verifier_semantics_digest"); cases += 1

    print(json.dumps({"status": "PASS", "cases": cases}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
