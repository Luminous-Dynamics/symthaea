#!/usr/bin/env python3
"""CI-only SAWEM profile schema and synthetic-fixture checks; not a market verifier."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from importlib.metadata import version
from datetime import datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


def fail(message: str) -> None:
    raise SystemExit(f"SAWEM profile conformance: FAIL: {message}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()



def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate_comparator_invariants(manifest: dict[str, Any]) -> None:
    frontier = manifest["decision_information"]
    decision_time = parse_utc(frontier["decision_time_utc"])
    forecast_cutoff = parse_utc(frontier["forecast_vintage_cutoff_utc"])
    outcome_end = parse_utc(frontier["actual_outcome_window_end_utc"])
    if forecast_cutoff > decision_time:
        raise ValueError("forecast vintage cutoff is after the decision information frontier")
    if decision_time >= outcome_end:
        raise ValueError("actual outcome window must extend beyond the decision timestamp")
    if frontier["no_lookahead_enforced"] is not True:
        raise ValueError("the comparator protocol must enforce no-lookahead")

    scenario = manifest["scenario"]
    if parse_utc(scenario["horizon_end_utc"]) <= parse_utc(scenario["horizon_start_utc"]):
        raise ValueError("scenario horizon end must follow its start")
    if scenario["interval_basis"] == "synthetic_not_market_default" and scenario["interval_minutes"] <= 0:
        raise ValueError("synthetic interval duration must be positive")

    for package in manifest["input_packages"]:
        if parse_utc(package["source_vintage_utc"]) > forecast_cutoff:
            raise ValueError(f"input package {package['package_id']} leaks a future source vintage")
        if package["data_class"] == "synthetic":
            if package["content_sha256"] is not None:
                raise ValueError("synthetic package must not invent a frozen content digest")
            if package["licence_state"] != "not_applicable_synthetic":
                raise ValueError("synthetic package licence state is inconsistent")

    asset = manifest["asset_model"]
    if asset["min_soc_fraction"] >= asset["max_soc_fraction"]:
        raise ValueError("asset SOC interval is empty or inverted")
    if asset["charge_power_mw"] <= 0 or asset["discharge_power_mw"] <= 0:
        raise ValueError("charge and discharge power must be positive")

    engines = manifest["engine_roster"]
    engine_ids = [engine["engine_id"] for engine in engines]
    if len(engine_ids) != len(set(engine_ids)):
        raise ValueError("engine roster contains duplicate engine identifiers")
    if len(engines) < 2:
        raise ValueError("at least two engine/comparator entries are required")

    policy = manifest["policy_baseline"]
    if policy["policy_class"] == "perfect_information_oracle":
        if policy["implementable"] is not False:
            raise ValueError("perfect-information oracle must be non-implementable")
        if policy["forecast_policy"] != "realized_future_not_allowed":
            raise ValueError("oracle must be isolated from implementable forecast policies")
        if policy["oracle_use"] not in {"lower_bound_only", "upper_bound_only"}:
            raise ValueError("oracle must only be reported as a mathematical bound")
    elif policy["implementable"] is False:
        raise ValueError("only the explicitly designated perfect-information oracle may be non-implementable")

    matching = manifest["matching_contract"]
    required_dimensions = {
        "market_rule_version", "information_frontier", "price_vintage", "scenario_seed",
        "topology", "asset_constraints", "warranty", "time_resolution", "objective_function",
        "risk_measure", "settlement_rules", "participant_eligibility"
    }
    observed = {item["dimension"] for item in matching["dimensions"]}
    if observed != required_dimensions:
        raise ValueError("matching contract must enumerate every required comparability dimension")
    if matching["oracle_separation_required"] is not True:
        raise ValueError("perfect-information oracle must remain a separate comparison class")

    if manifest["run_state"] == "not_run":
        if manifest.get("execution_receipt") is not None:
            raise ValueError("not_run manifest must not include an execution receipt")
        if any(engine["execution_state"] != "not_run" for engine in engines):
            raise ValueError("not_run manifest contains an engine claimed as executed")
        if any(metric["comparison_status"] != "not_run" or metric["value"] is not None
               for metric in manifest["reported_metrics"]):
            raise ValueError("not_run manifest must have null metrics marked not_run")
    elif manifest["run_state"] == "completed":
        if manifest.get("execution_receipt") is None:
            raise ValueError("completed manifest must include an execution receipt")
        if any(engine["execution_state"] != "executed" for engine in engines):
            raise ValueError("completed comparison contains an engine without execution evidence")

    if manifest["manifest_state"] == "synthetic_protocol_example":
        if manifest["run_state"] != "not_run":
            raise ValueError("synthetic protocol example must not claim an executed run")
        if manifest["execution_receipt"] is not None:
            raise ValueError("synthetic protocol example cannot contain an execution receipt")


def assert_comparator_negative_controls(manifest: dict[str, Any]) -> int:
    controls = [
        ("future-vintage", lambda value: value["input_packages"][0].update(
            source_vintage_utc="2026-10-02T00:00:00Z")),
        ("oracle-marked-implementable", lambda value: value["policy_baseline"].update(
            policy_class="perfect_information_oracle", implementable=True)),
        ("duplicate-engine", lambda value: value["engine_roster"].append(
            copy.deepcopy(value["engine_roster"][0]))),
        ("not-run-with-result", lambda value: value["reported_metrics"][0].update(
            value=1.0)),
        ("missing-comparability-dimension", lambda value: value["matching_contract"]["dimensions"].pop()),
        ("oracle-not-separated", lambda value: value["matching_contract"].update(
            oracle_separation_required=False)),
    ]
    for name, mutate in controls:
        changed = copy.deepcopy(manifest)
        mutate(changed)
        try:
            validate_comparator_invariants(changed)
        except (ValueError, KeyError, TypeError):
            continue
        raise ValueError(f"negative control did not fail closed: {name}")
    validate_comparator_invariants(manifest)
    return len(controls)

EXPECTED_CASE_IDS = {
    "BASE-001",
    "PVHYBRID-001",
    "SOC-FLOOR-001",
    "SOC-CEILING-001",
    "POWER-HEADROOM-001",
    "MODE-TRANSITION-001",
    "DAILY-LIMIT-001",
    "ROLLING-WEEK-001",
    "NETWORK-CHARGE-001",
    "NETWORK-DISCHARGE-001",
    "INTRADAY-GATE-001",
    "INSTRUCTION-METER-001",
    "CUMULATIVE-BALANCING-001",
    "SETTLEMENT-RERUN-001",
    "CREDIT-COVER-001",
    "WARRANTY-001",
    "PRICE-SPIKE-001",
    "FORECAST-SHIFT-001",
    "AVAILABILITY-001",
    "RULE-SUPERSESSION-001",
    "RESERVE-UP-CHARGE-REDUCTION-001",
    "RESERVE-DOWN-DISCHARGE-REDUCTION-001",
    "RESERVE-DEADBAND-001",
    "RESERVE-PRODUCT-DURATION-001",
    "RESERVE-UNKNOWN-SUSTAIN-001",
    "RESERVE-TRANSITION-001",
    "RESERVE-PAYMENT-QUALIFICATION-001",
    "RESERVE-LOST-OPPORTUNITY-001",
    "DSU-RESERVE-PRICE-CURVE-001",
    "DSU-RESERVE-CAPACITY-QUALIFICATION-001",
    "RESERVE-CAPACITY-PAYMENT-001",
    "CONSTRAINED-SCHEDULE-COMPENSATION-001",
    "CREDIT-COVER-LIQUIDITY-TIMELINE-001",
    "CREDIT-COVER-FUNDING-OPTION-ECONOMICS-001",
    "CREDIT-COVER-LIQUIDITY-STRESS-001",
    "CREDIT-COVER-LIQUIDITY-WATERFALL-001",
}


def validate_case_corpus(
    corpus: dict[str, Any],
    kernel_source: str,
    expected_benchmark_id: str,
) -> tuple[int, int, int]:
    if corpus.get("corpus_state") != "draft_case_register":
        raise ValueError("corpus must remain a draft case register until independent adjudication")
    if corpus.get("benchmark_id") != expected_benchmark_id:
        raise ValueError("benchmark corpus identifier does not match comparator manifest")
    cases = corpus.get("cases")
    if not isinstance(cases, list):
        raise ValueError("benchmark cases must be an array")
    identifiers = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(identifiers) != len(cases) or len(set(identifiers)) != len(identifiers):
        raise ValueError("benchmark corpus contains malformed or duplicate case identifiers")
    if set(identifiers) != EXPECTED_CASE_IDS:
        raise ValueError("benchmark corpus case census drifted from the frozen v1 case register")

    kernel_tested_cases = 0
    not_run_cases = 0
    referenced_tests: set[str] = set()
    for case in cases:
        implementation_state = case["implementation_state"]
        if case["oracle_state"] != "draft_expected_disposition":
            raise ValueError("no case may claim independent adjudication in this initial corpus")
        if implementation_state == "kernel_unit_tested":
            if case["expected_disposition"] == "not_run" or not case["test_names"]:
                raise ValueError(f"kernel-tested case lacks an expected disposition or test mapping: {case['case_id']}")
            for test_name in case["test_names"]:
                pattern = r"\#\[test\]\s+fn\s+" + re.escape(test_name) + r"\s*\("
                if re.search(pattern, kernel_source) is None:
                    raise ValueError(f"Rust test mapping is missing from kernel: {case['case_id']} -> {test_name}")
                referenced_tests.add(test_name)
            kernel_tested_cases += 1
        elif implementation_state == "declared_not_implemented":
            if case["expected_disposition"] != "not_run" or case["test_names"]:
                raise ValueError(f"not-implemented case has an outcome/test claim: {case['case_id']}")
            not_run_cases += 1
        else:
            raise ValueError(f"unexpected implementation state: {implementation_state}")
    return kernel_tested_cases, not_run_cases, len(referenced_tests)


def assert_case_corpus_negative_controls(
    corpus: dict[str, Any],
    kernel_source: str,
    expected_benchmark_id: str,
) -> int:
    duplicate = copy.deepcopy(corpus)
    duplicate["cases"].append(copy.deepcopy(duplicate["cases"][0]))
    try:
        validate_case_corpus(duplicate, kernel_source, expected_benchmark_id)
    except (ValueError, KeyError, TypeError):
        pass
    else:
        raise ValueError("duplicate benchmark case identifier did not fail closed")

    fabricated = copy.deepcopy(corpus)
    not_run_case = next(
        item for item in fabricated["cases"]
        if item["implementation_state"] == "declared_not_implemented"
    )
    not_run_case["implementation_state"] = "kernel_unit_tested"
    not_run_case["expected_disposition"] = "feasible"
    not_run_case["test_names"] = []
    try:
        validate_case_corpus(fabricated, kernel_source, expected_benchmark_id)
    except (ValueError, KeyError, TypeError):
        pass
    else:
        raise ValueError("unimplemented benchmark case was promoted without a test")
    validate_case_corpus(corpus, kernel_source, expected_benchmark_id)
    return 2


def validate_operational_status(operations: dict[str, Any]) -> None:
    """Keep market-wide test access distinct from production admission/trading."""
    if operations["market_operations_state"] != "pre_launch_testing_reported":
        raise ValueError("pre-launch status must remain reported, not promoted to live")
    if operations["participant_admission_state"] != "unknown":
        raise ValueError("pre-launch invitations do not prove this participant is admitted")
    if operations["commercial_trading_state"] != "unknown":
        raise ValueError("pre-launch simulation does not prove commercial trading is live")

    source = operations["status_source"]
    if source["source_class"] != "secondary_report":
        raise ValueError("pre-launch status must retain its secondary-report source class")
    if source["raw_artifact_sha256"] is not None or source["source_freeze_state"] != "not_frozen":
        raise ValueError("secondary status report cannot claim a frozen source artifact")
    if parse_utc(source["retrieved_at"]) < parse_utc(operations["status_as_of"]):
        raise ValueError("market status cannot be retrieved before its stated as-of time")
    note = operations["qualification_note"].lower()
    if "does not establish" not in note or "commercial" not in note:
        raise ValueError("pre-launch status must retain an explicit commercial-access qualification")


def assert_operational_status_negative_controls(
    operations: dict[str, Any],
) -> int:
    mutations = [
        ("participant-admission-promoted", "participant_admission_state", "admitted"),
        ("commercial-trading-promoted", "commercial_trading_state", "live"),
    ]
    for name, field, value in mutations:
        changed = copy.deepcopy(operations)
        changed[field] = value
        try:
            validate_operational_status(changed)
        except (ValueError, KeyError, TypeError):
            continue
        raise ValueError(f"operational-status negative control did not fail closed: {name}")
    validate_operational_status(operations)
    return len(mutations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    schema_path = root / "docs/research/sawem/v1/sawem-bess-underwriting-profile-v1.schema.json"
    fixture_path = root / "docs/research/sawem/v1/sawem-bess-underwriting-profile-v1.synthetic.json"
    comparator_schema_path = root / "docs/research/sawem/v1/sawem-bess-comparator-manifest-v1.schema.json"
    comparator_fixture_path = root / "docs/research/sawem/v1/sawem-bess-comparator-manifest-v1.synthetic.json"
    benchmark_schema_path = root / "docs/research/sawem/v1/sawem-bess-benchmark-corpus-v1.schema.json"
    benchmark_corpus_path = root / "docs/research/sawem/v1/sawem-bess-benchmark-corpus-v1.json"
    kernel_path = root / "tools/sawem-market-reference/src/lib.rs"
    credit_cover_kernel_path = root / "tools/sawem-market-reference/src/credit_cover.rs"
    liquidity_options_kernel_path = root / "tools/sawem-market-reference/src/liquidity_options.rs"
    liquidity_stress_kernel_path = root / "tools/sawem-market-reference/src/liquidity_stress.rs"
    liquidity_waterfall_kernel_path = root / "tools/sawem-market-reference/src/liquidity_waterfall.rs"

    try:
        schema_bytes = schema_path.read_bytes()
        fixture_bytes = fixture_path.read_bytes()
        comparator_schema_bytes = comparator_schema_path.read_bytes()
        comparator_fixture_bytes = comparator_fixture_path.read_bytes()
        benchmark_schema_bytes = benchmark_schema_path.read_bytes()
        benchmark_corpus_bytes = benchmark_corpus_path.read_bytes()
        kernel_bytes = kernel_path.read_bytes()
        credit_cover_kernel_bytes = credit_cover_kernel_path.read_bytes()
        liquidity_options_kernel_bytes = liquidity_options_kernel_path.read_bytes()
        liquidity_stress_kernel_bytes = liquidity_stress_kernel_path.read_bytes()
        liquidity_waterfall_kernel_bytes = liquidity_waterfall_kernel_path.read_bytes()
        schema = json.loads(schema_bytes)
        fixture = json.loads(fixture_bytes)
        comparator_schema = json.loads(comparator_schema_bytes)
        comparator_fixture = json.loads(comparator_fixture_bytes)
        benchmark_schema = json.loads(benchmark_schema_bytes)
        benchmark_corpus = json.loads(benchmark_corpus_bytes)
        kernel_source = (
            kernel_bytes
            + b"\n"
            + credit_cover_kernel_bytes
            + b"\n"
            + liquidity_options_kernel_bytes
            + b"\n"
            + liquidity_stress_kernel_bytes
            + b"\n"
            + liquidity_waterfall_kernel_bytes
        ).decode("utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot load schema or fixture: {exc}")

    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail("schema must declare JSON Schema Draft 2020-12")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        fail(f"schema meta-validation failed: {exc}")

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(fixture), key=lambda error: list(error.absolute_path))
    if errors:
        first = errors[0]
        fail(f"synthetic fixture violates schema at {list(first.absolute_path)}: {first.message}")

    if comparator_schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail("comparator schema must declare JSON Schema Draft 2020-12")
    try:
        Draft202012Validator.check_schema(comparator_schema)
    except Exception as exc:
        fail(f"comparator schema meta-validation failed: {exc}")
    comparator_validator = Draft202012Validator(
        comparator_schema, format_checker=FormatChecker()
    )
    comparator_errors = sorted(
        comparator_validator.iter_errors(comparator_fixture),
        key=lambda error: list(error.absolute_path),
    )
    if comparator_errors:
        first = comparator_errors[0]
        fail(
            "synthetic comparator manifest violates schema at "
            f"{list(first.absolute_path)}: {first.message}"
        )
    try:
        negative_control_count = assert_comparator_negative_controls(comparator_fixture)
    except (ValueError, KeyError, TypeError) as exc:
        fail(f"comparator invariants or negative controls failed: {exc}")

    if benchmark_schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail("benchmark corpus schema must declare JSON Schema Draft 2020-12")
    try:
        Draft202012Validator.check_schema(benchmark_schema)
    except Exception as exc:
        fail(f"benchmark corpus schema meta-validation failed: {exc}")
    benchmark_validator = Draft202012Validator(
        benchmark_schema, format_checker=FormatChecker()
    )
    benchmark_errors = sorted(
        benchmark_validator.iter_errors(benchmark_corpus),
        key=lambda error: list(error.absolute_path),
    )
    if benchmark_errors:
        first = benchmark_errors[0]
        fail(
            "benchmark corpus violates schema at "
            f"{list(first.absolute_path)}: {first.message}"
        )
    try:
        (
            benchmark_kernel_tested_case_count,
            benchmark_not_run_case_count,
            benchmark_unique_test_count,
        ) = validate_case_corpus(
            benchmark_corpus,
            kernel_source,
            comparator_fixture["benchmark_id"],
        )
        benchmark_negative_control_count = assert_case_corpus_negative_controls(
            benchmark_corpus,
            kernel_source,
            comparator_fixture["benchmark_id"],
        )
    except (ValueError, KeyError, TypeError) as exc:
        fail(f"benchmark corpus traceability/negative controls failed: {exc}")

    # Market phase, individual admission and commercial trading are separate.
    if fixture["profile_state"] != "research_profile_not_authoritative":
        fail("synthetic sample must remain research_profile_not_authoritative")
    market = fixture["market_rule_profile"]
    operations = fixture["operational_status"]
    if market["publication_state"] != "published":
        fail("the source profile should preserve the Code's published state")
    if market["effective_status"] != "unknown" or market["effective_date"] is not None:
        fail("published Code text must not imply that every rule is effective")
    try:
        validate_operational_status(operations)
        market_status_negative_control_count = assert_operational_status_negative_controls(operations)
    except (ValueError, KeyError, TypeError) as exc:
        fail(f"market operational status / negative controls failed: {exc}")
    for source in (market["source"], operations["status_source"]):
        if source["raw_artifact_sha256"] is not None or source["source_freeze_state"] != "not_frozen":
            fail("this initial profile must not claim source-byte freeze before exact source ingestion")

    unit = fixture["trading_unit"]
    if unit["min_stable_charging_mw"] > unit["max_continuous_charging_mw"]:
        fail("minimum stable charging point exceeds maximum continuous charging rating")
    if unit["min_stored_energy_mwh"] >= unit["max_stored_energy_mwh"]:
        fail("minimum stored energy must be below maximum stored energy")
    transitions = unit["mode_transition_seconds"]
    required_transition_keys = {
        "standstill_to_charging",
        "standstill_to_discharging",
        "charging_to_standstill",
        "discharging_to_standstill",
        "charging_to_discharging",
        "discharging_to_charging",
    }
    if set(transitions) != required_transition_keys:
        fail("all six directed transitions between standstill, charging and discharging must be present")

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
    for stage in stages:
        if stage["scheduled_discharge_mw"] > stage["network_export_limit_mw"]:
            fail(f"{stage['stage']} exceeds its declared network export limit")

    reserve_products = [item["product"] for item in period["reserve_commitments"]]
    if len(reserve_products) != len(set(reserve_products)):
        fail("duplicate reserve products are not allowed in this fixture")
    reserve_keys = [(item["product"], item["direction"]) for item in period["reserve_commitments"]]
    if len(reserve_keys) != len(set(reserve_keys)):
        fail("duplicate reserve product/direction pairs are not allowed")
    for reserve in period["reserve_commitments"]:
        product = reserve["product"]
        direction = reserve["direction"]
        if product in {"instantaneous_reserve", "ten_minute_reserve"} and direction != "up":
            fail(f"{product} direction is not supported by the declared SAWEM Rev 2.3 profile")
        start = reserve["start_response_seconds"]
        full = reserve["full_activation_seconds"]
        obligation_sustain = reserve["sustain_duration_seconds"]
        proven_sustain = reserve["proven_sustain_seconds"]
        if product == "regulating_reserve":
            if start is None or start > 10:
                fail("regulating reserve must begin response within 10 seconds")
            if full is None or full > 600:
                fail("regulating reserve must fully activate within 10 minutes")
        elif product == "instantaneous_reserve":
            if full is None or full > 10:
                fail("instantaneous reserve must fully activate within 10 seconds")
            if obligation_sustain is None or obligation_sustain < 600:
                fail("instantaneous reserve must carry at least 10 minutes of sustain obligation")
        elif product == "ten_minute_reserve":
            if full is None or full > 600:
                fail("ten-minute reserve must fully respond within 10 minutes")
            # The Code definition does not set a sustain duration. Do not invent one.
        if obligation_sustain is not None and (
            proven_sustain is None or proven_sustain < obligation_sustain
        ):
            fail("asset-proven reserve sustain is missing or shorter than the operational obligation")
        if reserve["capability_class"] != "synthetic":
            fail("synthetic reference must not promote reserve capability to a certified class")
        if reserve["response_evidence_class"] not in {"synthetic", "unknown"}:
            fail("synthetic reserve response evidence cannot be promoted")
        if reserve["sustain_evidence_class"] not in {"synthetic", "unknown"}:
            fail("synthetic reserve sustain evidence cannot be promoted")
        if reserve["response_evidence_ref"] is not None or reserve["sustain_evidence_ref"] is not None:
            fail("synthetic reserve entries cannot carry purported live evidence references")
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
    committed_reserve_mw = sum(item["committed_mw"] for item in period["reserve_commitments"])
    for stage in stages:
        if stage["scheduled_discharge_mw"] + committed_reserve_mw > unit["max_continuous_discharging_mw"]:
            fail(f"{stage['stage']} double-counts discharge power and reserve headroom")

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
        "comparator_schema_sha256": sha256(comparator_schema_bytes),
        "comparator_fixture_sha256": sha256(comparator_fixture_bytes),
        "benchmark_corpus_schema_sha256": sha256(benchmark_schema_bytes),
        "benchmark_corpus_sha256": sha256(benchmark_corpus_bytes),
        "benchmark_case_count": len(benchmark_corpus["cases"]),
        "benchmark_kernel_tested_case_count": benchmark_kernel_tested_case_count,
        "benchmark_not_run_case_count": benchmark_not_run_case_count,
        "benchmark_unique_rust_test_count": benchmark_unique_test_count,
        "benchmark_negative_control_count": benchmark_negative_control_count,
        "profile_state": fixture["profile_state"],
        "comparator_manifest_state": comparator_fixture["manifest_state"],
        "comparator_run_state": comparator_fixture["run_state"],
        "comparator_engine_count": len(comparator_fixture["engine_roster"]),
        "comparator_negative_control_count": negative_control_count,
        "market_status_negative_control_count": market_status_negative_control_count,
        "comparator_reported_metric_count": len(comparator_fixture["reported_metrics"]),
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
