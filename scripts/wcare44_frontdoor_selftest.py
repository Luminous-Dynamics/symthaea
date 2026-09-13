#!/usr/bin/env python3
"""Adversarial contract tests for the strict WCARE-44 Stage-A front door."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import wcare44_candidate_selftest as base

ROOT = Path(__file__).resolve().parent.parent
FRONTDOOR = ROOT / "scripts/wcare44_candidate_qualify.py"


def rewrite_json(path: Path, mutate) -> None:
    value = json.loads(path.read_text())
    mutate(value)
    base.write(path, value)


def strict_fixture(tmp: Path) -> tuple[dict, list[Path], Path]:
    f = base.fixture(tmp)
    observations = base.all_builder_observations(tmp, f)
    for path in observations:
        rewrite_json(path, lambda value: value.__setitem__("wcare42_verifier_sha256", base.hx("1")))
    temporal = base.temporal_observation(tmp / "temporal.json", f)
    rewrite_json(temporal, lambda value: value.__setitem__("wcare43_verifier_sha256", base.hx("3")))
    return f, observations, temporal


def argv(f: dict, temporal: Path, observations: list[Path]) -> list[str]:
    cmd = base.argv(f, temporal, observations)
    cmd[1] = str(FRONTDOOR)
    return cmd


def execute(cmd: list[str]) -> tuple[int, dict]:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    try:
        value = json.loads(result.stdout)
    except Exception as exc:
        raise AssertionError(
            f"invalid frontdoor JSON: rc={result.returncode} out={result.stdout!r} err={result.stderr!r}"
        ) from exc
    return result.returncode, value


def expect_invalid(cmd: list[str], detail_fragment: str) -> None:
    rc, value = execute(cmd)
    assert rc == 4, value
    assert value["disposition"] == "CANDIDATE_INVALID", value
    assert value["builder_authentication_candidate_status"] == "INVALID", value
    assert value["temporal_candidate_status"] == "INVALID", value
    assert detail_fragment in value["detail"], value
    assert value["builder_authentication_established"] is False, value
    assert value["preregistration_temporal_precedence_established"] is False, value
    assert value["authenticated_preregistered_replication_established"] is False, value
    assert value["runtime_authority_granted"] is False, value


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="wcare44-frontdoor-") as raw_tmp:
        root = Path(raw_tmp)

        valid = root / "valid"
        valid.mkdir()
        f, observations, temporal = strict_fixture(valid)
        rc, value = execute(argv(f, temporal, observations))
        assert rc == 0 and value["disposition"] == "CANDIDATE_MEASURED", value
        assert value["candidate_authenticated_preregistered_replication"] is True, value
        assert value["builder_authentication_established"] is False, value
        assert value["preregistration_temporal_precedence_established"] is False, value
        assert value["authenticated_preregistered_replication_established"] is False, value

        missing = root / "missing"
        missing.mkdir()
        f, observations, temporal = strict_fixture(missing)
        rewrite_json(observations[0], lambda value: value.pop("wcare42_result_sha256"))
        expect_invalid(argv(f, temporal, observations), "builder_observation_missing_fields:wcare42_result_sha256")

        unknown = root / "unknown"
        unknown.mkdir()
        f, observations, temporal = strict_fixture(unknown)
        rewrite_json(observations[0], lambda value: value.__setitem__("unexpected", "field"))
        expect_invalid(argv(f, temporal, observations), "builder_observation_unknown_fields:unexpected")

        wrong_builder = root / "wrong-builder"
        wrong_builder.mkdir()
        f, observations, temporal = strict_fixture(wrong_builder)
        rewrite_json(observations[0], lambda value: value.__setitem__("wcare42_verifier_sha256", base.hx("6")))
        expect_invalid(argv(f, temporal, observations), "builder_observation_verifier_sha256_not_preregistered")

        wrong_temporal = root / "wrong-temporal"
        wrong_temporal.mkdir()
        f, observations, temporal = strict_fixture(wrong_temporal)
        rewrite_json(temporal, lambda value: value.__setitem__("wcare43_verifier_sha256", base.hx("9")))
        expect_invalid(argv(f, temporal, observations), "temporal_observation_verifier_sha256_not_preregistered")

        wrong_w40_frontdoor = root / "wrong-w40-frontdoor"
        wrong_w40_frontdoor.mkdir()
        f, observations, temporal = strict_fixture(wrong_w40_frontdoor)
        rewrite_json(f["auth"], lambda value: value.__setitem__("wcare40_frontdoor_sha256", base.hx("f")))
        expect_invalid(
            argv(f, temporal, observations),
            "wcare41_wcare40_frontdoor_sha256_does_not_match_wcare40_result",
        )

        wrong_w40_core = root / "wrong-w40-core"
        wrong_w40_core.mkdir()
        f, observations, temporal = strict_fixture(wrong_w40_core)
        rewrite_json(f["auth"], lambda value: value.__setitem__("wcare40_core_verifier_sha256", base.hx("f")))
        expect_invalid(
            argv(f, temporal, observations),
            "wcare41_wcare40_core_verifier_sha256_does_not_match_wcare40_result",
        )

        malformed_bool = root / "malformed-bool"
        malformed_bool.mkdir()
        f, observations, temporal = strict_fixture(malformed_bool)
        rewrite_json(observations[0], lambda value: value.__setitem__("verifier_execution_qualified", "true"))
        expect_invalid(argv(f, temporal, observations), "builder_observation_invalid_boolean:verifier_execution_qualified")

        malformed_time = root / "malformed-time"
        malformed_time.mkdir()
        f, observations, temporal = strict_fixture(malformed_time)
        rewrite_json(f["auth"], lambda value: value.__setitem__("evaluation_utc", "2026-09-14 00:00:00"))
        expect_invalid(argv(f, temporal, observations), "wcare41_authentication_plan_invalid_evaluation_utc")

        malformed_backend = root / "malformed-backend"
        malformed_backend.mkdir()
        f, observations, temporal = strict_fixture(malformed_backend)
        rewrite_json(f["auth"], lambda value: value["builder_verifier"].pop("policy_sha256"))
        expect_invalid(argv(f, temporal, observations), "builder_verifier_missing_fields:policy_sha256")

        malformed_temporal = root / "malformed-temporal"
        malformed_temporal.mkdir()
        f, observations, temporal = strict_fixture(malformed_temporal)
        rewrite_json(temporal, lambda value: value.pop("wcare43_result_sha256"))
        expect_invalid(argv(f, temporal, observations), "temporal_observation_missing_fields:wcare43_result_sha256")

        malformed_notes = root / "malformed-notes"
        malformed_notes.mkdir()
        f, observations, temporal = strict_fixture(malformed_notes)
        rewrite_json(observations[0], lambda value: value.__setitem__("notes", 42))
        expect_invalid(argv(f, temporal, observations), "builder_observation_invalid_string:notes")

    print(json.dumps({
        "authority": "MeasurementOnly",
        "classification": "PASS_WCARE44_FRONTDOOR_SELFTEST",
        "valid_candidate_path_preserved": True,
        "missing_builder_field_rejected": True,
        "unknown_builder_field_rejected": True,
        "wrong_builder_verifier_identity_rejected": True,
        "wrong_temporal_verifier_identity_rejected": True,
        "wrong_wcare40_frontdoor_identity_rejected": True,
        "wrong_wcare40_core_identity_rejected": True,
        "malformed_builder_boolean_rejected": True,
        "malformed_auth_plan_time_rejected": True,
        "malformed_backend_contract_rejected": True,
        "missing_temporal_field_rejected": True,
        "malformed_optional_field_rejected": True,
        "final_promotion_remains_blocked": True,
        "runtime_authority_granted": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
