#!/usr/bin/env python3
"""CI-only SAWEM profile schema and synthetic-fixture checks; not a market verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def fail(message: str) -> None:
    raise SystemExit(f"SAWEM profile conformance: FAIL: {message}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    schema_path = root / "docs/research/sawem/v1/sawem-bess-underwriting-profile-v1.schema.json"
    fixture_path = root / "docs/research/sawem/v1/sawem-bess-underwriting-profile-v1.synthetic.json"

    try:
        schema_bytes = schema_path.read_bytes()
        fixture_bytes = fixture_path.read_bytes()
        schema = json.loads(schema_bytes)
        fixture = json.loads(fixture_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot load schema or fixture: {exc}")

    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail("schema must declare JSON Schema Draft 2020-12")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        fail(f"schema meta-validation failed: {exc}")

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(fixture), key=lambda error: list(error.absolute_path))
    if errors:
        first = errors[0]
        fail(f"synthetic fixture violates schema at {list(first.absolute_path)}: {first.message}")

    # Status is intentionally non-authoritative in the first public profile.
    if fixture["profile_state"] != "research_profile_not_authoritative":
        fail("synthetic sample must remain research_profile_not_authoritative")
    market = fixture["market_rule_profile"]
    operations = fixture["operational_status"]
    if market["publication_state"] != "published":
        fail("the source profile should preserve the Code's published state")
    if market["effective_status"] != "unknown" or market["effective_date"] is not None:
        fail("published Code text must not imply that every rule is effective")
    if operations["participant_admission_state"] != "unknown":
        fail("secondary pre-launch reporting must not imply participant admission")
    if operations["commercial_trading_state"] != "unknown":
        fail("secondary pre-launch reporting must not imply commercial trading")
    if operations["market_operations_state"] != "pre_launch_testing_reported":
        fail("pre-launch status must remain explicitly reported, not promoted to live")
    for source in (market["source"], operations["status_source"]):
        if source["raw_artifact_sha256"] is not None or source["source_freeze_state"] != "not_frozen":
            fail("this initial profile must not claim source-byte freeze before exact source ingestion")

    unit = fixture["trading_unit"]
    if unit["min_stable_charging_mw"] > unit["max_continuous_charging_mw"]:
        fail("minimum stable charging point exceeds maximum continuous charging rating")
    if unit["min_stored_energy_mwh"] >= unit["max_stored_energy_mwh"]:
        fail("minimum stored energy must be below maximum stored energy")

    constraints = fixture["energy_constraints"]
    declared = constraints["daily_declared_energy_mwh"]
    if len(declared) != 7:
        fail("the energy-constrained Trading Unit fixture must cover seven declared trading days")
    if any(value > constraints["daily_limit_mwh"] for value in declared):
        fail("a daily declaration exceeds its daily MWh limit")
    if sum(declared[1:]) > constraints["next_six_days_limit_mwh"]:
        fail("the next-six-days declaration exceeds its aggregate limit")
    if sum(declared) > constraints["seven_day_total_limit_mwh"]:
        fail("the seven-day declaration exceeds its aggregate limit")

    period = fixture["trading_period"]
    if period["end_at"] <= period["start_at"]:
        fail("Trading Period end must follow its start")
    stages = period["schedule_stages"]
    expected_stages = [
        "day_ahead_unconstrained",
        "day_ahead_network_constrained",
        "intraday_network_constrained",
    ]
    actual_stages = [stage["stage"] for stage in stages]
    if actual_stages != expected_stages:
        fail(f"schedule stages are missing, duplicated or misordered: {actual_stages}")
    unconstrained, constrained, intraday = stages
    if unconstrained["scheduled_discharge_mw"] < constrained["scheduled_discharge_mw"]:
        fail("network-constrained schedule unexpectedly exceeds the unconstrained schedule")
    for stage in (constrained, intraday):
        if stage["scheduled_discharge_mw"] > stage["network_export_limit_mw"]:
            fail(f"{stage['stage']} exceeds its declared network export limit")

    reserve_products = [item["product"] for item in period["reserve_commitments"]]
    if len(reserve_products) != len(set(reserve_products)):
        fail("duplicate reserve products are not allowed in this fixture")
    if set(reserve_products) != {
        "regulating_reserve",
        "instantaneous_reserve",
        "ten_minute_reserve",
    }:
        fail("the synthetic reserve panel must exercise all three declared reserve products")
    if any(item["simultaneous_energy_double_counted"] is not False for item in period["reserve_commitments"]):
        fail("reserve commitments must not be marked as double-counted energy")
    if any(item["capability_class"] != "synthetic" for item in period["reserve_commitments"]):
        fail("synthetic reserve commitments cannot be promoted to certified capability")

    dispatch = period["dispatch_instruction"]["instructed_discharge_mw"]
    metered = period["metered_outcome"]["metered_discharge_mw"]
    if period["dispatch_instruction"]["instruction_source_state"] != "synthetic":
        fail("fixture dispatch instruction must remain synthetic")
    if period["metered_outcome"]["measurement_class"] != "synthetic":
        fail("fixture meter result must not be represented as authenticated measurement")
    if period["metered_outcome"]["meter_evidence_ref"] is not None:
        fail("synthetic metered result cannot carry a purported real evidence reference")
    if dispatch <= 0 or metered < 0:
        fail("dispatch and metered values must be nonnegative with positive instruction")

    settlement = fixture["settlement_exposure"]
    if settlement["required_credit_cover_minor_units"] != (
        settlement["actual_exposure_minor_units"]
        + settlement["undefined_potential_exposure_minor_units"]
    ):
        fail("required credit cover must reconcile actual plus undefined-potential exposure")
    if not settlement["balancing_payment_components"]:
        fail("fixture must preserve balancing payment components separately")

    source_commit = os.environ.get("SAWEM_SOURCE_COMMIT")
    if not source_commit:
        fail("SAWEM_SOURCE_COMMIT is required for a reproducible receipt")

    receipt = {
        "receipt_version": "1",
        "source_commit": source_commit,
        "scope": "schema_meta_and_synthetic_profile_invariants",
        "validator": "python-jsonschema",
        "validator_version": version("jsonschema"),
        "schema_sha256": sha256(schema_bytes),
        "fixture_sha256": sha256(fixture_bytes),
        "profile_state": fixture["profile_state"],
        "market_code_publication_state": market["publication_state"],
        "market_code_effective_status": market["effective_status"],
        "participant_admission_state": operations["participant_admission_state"],
        "commercial_trading_state": operations["commercial_trading_state"],
        "daily_declaration_count": len(declared),
        "reserve_product_count": len(reserve_products),
        "schedule_stage_count": len(stages),
        "credit_cover_reconciled": True,
        "result": "pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
