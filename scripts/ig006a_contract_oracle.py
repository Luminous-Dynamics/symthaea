#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from typing import Any

SCHEMA = "symthaea-institutional-lab-receipt-v1"
AUTHORITY = "MeasurementOnly"

PROFILE_KEYS = (
    "scenario",
    "mechanism",
    "population",
    "adversary",
    "metric_schema",
    "constitutional_policy",
    "rng_profile",
)
PROFILE_FIELDS = {"id", "revision"}
IDENTITY_FIELDS = {"schema", *PROFILE_KEYS, "seed"}
METRIC_DEFINITION_FIELDS = {
    "id",
    "revision",
    "direction",
    "unit",
    "description",
    "lower_bound",
    "upper_bound",
}
METRIC_SCHEMA_FIELDS = {"id", "revision", "metrics"}
CONSTITUTIONAL_RESULT_FIELDS = {"id", "status", "witness", "reason"}
RECEIPT_FIELDS = {
    "schema",
    "authority",
    "experiment_identity",
    "completion",
    "metric_schema",
    "metric_values",
    "constitutional_results",
    "constitutional_valid",
    "mechanism_trace",
    "analysis_witnesses",
    "warnings",
    "non_claims",
}

DIRECTIONS = {"HigherBetter", "LowerBetter", "DescriptiveOnly"}
COMPLETION = {
    "CompleteWithinDeclaredFiniteDomain",
    "BudgetExhausted",
    "InvalidScenario",
    "NumericFailure",
    "UnsupportedAnalysis",
}
CONSTITUTIONAL_STATUSES = {"Satisfied", "Violated", "NotEvaluated"}
FORBIDDEN_SCORE_FIELDS = {
    "governance_score",
    "alignment_score",
    "flourishing_score",
    "safe",
}


class ContractError(ValueError):
    pass


def require_exact_keys(obj: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(obj)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def require_nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    return value


def require_revision(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def require_seed(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > (2**64 - 1):
        raise ContractError("seed must be an unsigned 64-bit integer")
    return value


def require_finite_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ContractError(f"{label} must be finite")
    return value


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def validate_profile_ref(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    require_exact_keys(value, PROFILE_FIELDS, label)
    return {
        "id": require_nonempty_text(value["id"], f"{label}.id"),
        "revision": require_revision(value["revision"], f"{label}.revision"),
    }


def validate_experiment_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("experiment_identity must be an object")
    require_exact_keys(value, IDENTITY_FIELDS, "experiment_identity")
    if value["schema"] != SCHEMA:
        raise ContractError(f"experiment_identity.schema must equal {SCHEMA}")
    result: dict[str, Any] = {"schema": SCHEMA}
    for key in PROFILE_KEYS:
        result[key] = validate_profile_ref(value[key], f"experiment_identity.{key}")
    result["seed"] = require_seed(value["seed"])
    return result


def validate_metric_definition(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("metric definition must be an object")
    require_exact_keys(value, METRIC_DEFINITION_FIELDS, "metric definition")
    metric_id = require_nonempty_text(value["id"], "metric.id")
    revision = require_revision(value["revision"], "metric.revision")
    direction = value["direction"]
    if direction not in DIRECTIONS:
        raise ContractError(f"invalid metric direction: {direction!r}")
    unit = require_nonempty_text(value["unit"], "metric.unit")
    description = require_nonempty_text(value["description"], "metric.description")
    lower = value["lower_bound"]
    upper = value["upper_bound"]
    if lower is not None:
        lower = require_finite_number(lower, f"{metric_id}.lower_bound")
    if upper is not None:
        upper = require_finite_number(upper, f"{metric_id}.upper_bound")
    if lower is not None and upper is not None and lower > upper:
        raise ContractError(f"{metric_id} lower_bound exceeds upper_bound")
    return {
        "id": metric_id,
        "revision": revision,
        "direction": direction,
        "unit": unit,
        "description": description,
        "lower_bound": lower,
        "upper_bound": upper,
    }


def validate_metric_schema(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("metric_schema must be an object")
    require_exact_keys(value, METRIC_SCHEMA_FIELDS, "metric_schema")
    schema_id = require_nonempty_text(value["id"], "metric_schema.id")
    revision = require_revision(value["revision"], "metric_schema.revision")
    metrics = value["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ContractError("metric_schema.metrics must be a non-empty array")
    validated = [validate_metric_definition(metric) for metric in metrics]
    ids = [metric["id"] for metric in validated]
    if len(ids) != len(set(ids)):
        raise ContractError("duplicate metric ids")
    validated.sort(key=lambda metric: metric["id"])
    return {"id": schema_id, "revision": revision, "metrics": validated}


def validate_metric_values(schema: dict[str, Any], values: Any) -> dict[str, float]:
    if not isinstance(values, dict):
        raise ContractError("metric_values must be an object")
    schema_by_id = {metric["id"]: metric for metric in schema["metrics"]}
    if set(values) != set(schema_by_id):
        missing = sorted(set(schema_by_id) - set(values))
        extra = sorted(set(values) - set(schema_by_id))
        raise ContractError(f"metric_values keys mismatch: missing={missing}, extra={extra}")
    result: dict[str, float] = {}
    for metric_id in sorted(values):
        metric = schema_by_id[metric_id]
        number = require_finite_number(values[metric_id], f"metric_values.{metric_id}")
        lower = metric["lower_bound"]
        upper = metric["upper_bound"]
        if lower is not None and number < lower:
            raise ContractError(f"{metric_id} is below schema lower_bound")
        if upper is not None and number > upper:
            raise ContractError(f"{metric_id} is above schema upper_bound")
        result[metric_id] = number
    return result


def validate_constitutional_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("constitutional result must be an object")
    require_exact_keys(value, CONSTITUTIONAL_RESULT_FIELDS, "constitutional result")
    invariant_id = require_nonempty_text(value["id"], "constitutional_result.id")
    status = value["status"]
    if status not in CONSTITUTIONAL_STATUSES:
        raise ContractError(f"invalid constitutional status: {status!r}")
    witness = value["witness"]
    reason = value["reason"]

    if witness is not None and (not isinstance(witness, str) or not witness):
        raise ContractError("constitutional witness must be null or non-empty string")
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise ContractError("constitutional reason must be null or non-empty string")

    if status == "Satisfied":
        if witness is not None or reason is not None:
            raise ContractError("Satisfied requires witness=null and reason=null")
    elif status == "Violated":
        if witness is None or reason is not None:
            raise ContractError("Violated requires witness and reason=null")
    elif status == "NotEvaluated":
        if reason is None or witness is not None:
            raise ContractError("NotEvaluated requires reason and witness=null")

    return {
        "id": invariant_id,
        "status": status,
        "witness": witness,
        "reason": reason,
    }


def validate_constitutional_results(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(value, list) or not value:
        raise ContractError("constitutional_results must be a non-empty array")
    validated = [validate_constitutional_result(item) for item in value]
    ids = [item["id"] for item in validated]
    if len(ids) != len(set(ids)):
        raise ContractError("duplicate constitutional invariant ids")
    validated.sort(key=lambda item: item["id"])
    valid = all(item["status"] == "Satisfied" for item in validated)
    return validated, valid


def validate_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("receipt must be an object")
    require_exact_keys(value, RECEIPT_FIELDS, "receipt")
    if FORBIDDEN_SCORE_FIELDS.intersection(value):
        raise ContractError("forbidden aggregate score/safety field")
    if value["schema"] != SCHEMA:
        raise ContractError(f"receipt.schema must equal {SCHEMA}")
    if value["authority"] != AUTHORITY:
        raise ContractError(f"receipt.authority must equal {AUTHORITY}")

    identity = validate_experiment_identity(value["experiment_identity"])
    completion = value["completion"]
    if completion not in COMPLETION:
        raise ContractError(f"invalid completion state: {completion!r}")

    metric_schema = validate_metric_schema(value["metric_schema"])
    if (
        identity["metric_schema"]["id"] != metric_schema["id"]
        or identity["metric_schema"]["revision"] != metric_schema["revision"]
    ):
        raise ContractError("experiment identity metric-schema reference does not match receipt schema")
    metric_values = validate_metric_values(metric_schema, value["metric_values"])

    constitutional_results, constitutional_valid = validate_constitutional_results(
        value["constitutional_results"]
    )
    if value["constitutional_valid"] is not constitutional_valid:
        raise ContractError("constitutional_valid does not match invariant results")

    mechanism_trace = require_nonempty_text(value["mechanism_trace"], "mechanism_trace")
    if not isinstance(value["analysis_witnesses"], list):
        raise ContractError("analysis_witnesses must be an array")
    analysis_witnesses = []
    for index, witness in enumerate(value["analysis_witnesses"]):
        analysis_witnesses.append(require_nonempty_text(witness, f"analysis_witnesses[{index}]"))

    def validate_text_array(name: str) -> list[str]:
        raw = value[name]
        if not isinstance(raw, list):
            raise ContractError(f"{name} must be an array")
        return [require_nonempty_text(item, f"{name}[]") for item in raw]

    warnings = validate_text_array("warnings")
    non_claims = validate_text_array("non_claims")

    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "experiment_identity": identity,
        "completion": completion,
        "metric_schema": metric_schema,
        "metric_values": metric_values,
        "constitutional_results": constitutional_results,
        "constitutional_valid": constitutional_valid,
        "mechanism_trace": mechanism_trace,
        "analysis_witnesses": analysis_witnesses,
        "warnings": warnings,
        "non_claims": non_claims,
    }


def canonical_receipt(value: Any) -> tuple[dict[str, Any], bytes, str]:
    validated = validate_receipt(value)
    encoded = canonical_bytes(validated)
    return validated, encoded, hashlib.sha256(encoded).hexdigest()


def profile(profile_id: str, revision: int) -> dict[str, Any]:
    return {"id": profile_id, "revision": revision}


def fixture() -> dict[str, Any]:
    metric_schema = {
        "id": "institutional-core-metrics",
        "revision": 1,
        "metrics": [
            {
                "id": "welfare_total",
                "revision": 1,
                "direction": "HigherBetter",
                "unit": "utility",
                "description": "Declared scenario utility sum; not a constitutional score.",
                "lower_bound": None,
                "upper_bound": None,
            },
            {
                "id": "capture_concentration",
                "revision": 1,
                "direction": "LowerBetter",
                "unit": "share",
                "description": "Declared concentration diagnostic on [0,1].",
                "lower_bound": 0.0,
                "upper_bound": 1.0,
            },
            {
                "id": "verification_events",
                "revision": 1,
                "direction": "DescriptiveOnly",
                "unit": "count",
                "description": "Evidence verification operations performed.",
                "lower_bound": 0.0,
                "upper_bound": None,
            },
        ],
    }
    identity = {
        "schema": SCHEMA,
        "scenario": profile("transparent-pd-fixture", 1),
        "mechanism": profile("fixture-mechanism", 1),
        "population": profile("best-response-population", 1),
        "adversary": profile("benign", 1),
        "metric_schema": profile(metric_schema["id"], metric_schema["revision"]),
        "constitutional_policy": profile("minimal-rights-fixture", 1),
        "rng_profile": profile("deterministic-no-rng", 1),
        "seed": 7,
    }
    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "experiment_identity": identity,
        "completion": "CompleteWithinDeclaredFiniteDomain",
        "metric_schema": metric_schema,
        "metric_values": {
            "welfare_total": 2.0,
            "capture_concentration": 0.5,
            "verification_events": 0.0,
        },
        "constitutional_results": [
            {"id": "no-unauthorized-action", "status": "Satisfied", "witness": None, "reason": None},
            {"id": "meaningful-exit", "status": "Satisfied", "witness": None, "reason": None},
        ],
        "constitutional_valid": True,
        "mechanism_trace": "fixture-trace-v1",
        "analysis_witnesses": ["pd-dd-zero-unilateral-regret", "pd-grand-coalition-cc-witness"],
        "warnings": [],
        "non_claims": [
            "fixture_only",
            "no_human_behavioral_validity",
            "no_mycelix_safety_claim",
        ],
    }


def expect_error(label: str, mutator) -> None:
    candidate = fixture()
    mutator(candidate)
    try:
        canonical_receipt(candidate)
    except ContractError:
        return
    raise AssertionError(f"{label}: expected ContractError")


def self_test() -> dict[str, Any]:
    base = fixture()
    validated_a, encoded_a, digest_a = canonical_receipt(base)
    validated_b, encoded_b, digest_b = canonical_receipt(copy.deepcopy(base))
    assert encoded_a == encoded_b
    assert digest_a == digest_b
    assert validated_a == validated_b

    mutations = {
        "scenario_revision": lambda x: x["experiment_identity"]["scenario"].__setitem__("revision", 2),
        "mechanism_revision": lambda x: x["experiment_identity"]["mechanism"].__setitem__("revision", 2),
        "population_revision": lambda x: x["experiment_identity"]["population"].__setitem__("revision", 2),
        "adversary_revision": lambda x: x["experiment_identity"]["adversary"].__setitem__("revision", 2),
        "constitutional_revision": lambda x: x["experiment_identity"]["constitutional_policy"].__setitem__("revision", 2),
        "rng_revision": lambda x: x["experiment_identity"]["rng_profile"].__setitem__("revision", 2),
        "seed": lambda x: x["experiment_identity"].__setitem__("seed", 8),
    }
    identity_digests = {}
    for label, mutate in mutations.items():
        candidate = copy.deepcopy(base)
        mutate(candidate)
        identity_digests[label] = digest(validate_experiment_identity(candidate["experiment_identity"]))
        assert identity_digests[label] != digest(validate_experiment_identity(base["experiment_identity"]))

    metric_changed = copy.deepcopy(base)
    metric_changed["metric_schema"]["revision"] = 2
    metric_changed["experiment_identity"]["metric_schema"]["revision"] = 2
    _, _, metric_changed_digest = canonical_receipt(metric_changed)
    assert metric_changed_digest != digest_a

    descriptive = next(
        m for m in validated_a["metric_schema"]["metrics"] if m["id"] == "verification_events"
    )
    assert descriptive["direction"] == "DescriptiveOnly"

    violated = copy.deepcopy(base)
    violated["metric_values"]["welfare_total"] = 1_000_000.0
    violated["constitutional_results"][0] = {
        "id": "no-unauthorized-action",
        "status": "Violated",
        "witness": "unauthorized-transition-17",
        "reason": None,
    }
    violated["constitutional_valid"] = False
    validated_violated, _, _ = canonical_receipt(violated)
    assert validated_violated["constitutional_valid"] is False
    assert validated_violated["metric_values"]["welfare_total"] == 1_000_000.0

    not_evaluated = copy.deepcopy(base)
    not_evaluated["constitutional_results"][0] = {
        "id": "no-unauthorized-action",
        "status": "NotEvaluated",
        "witness": None,
        "reason": "adapter lacks authority trace",
    }
    not_evaluated["constitutional_valid"] = False
    assert canonical_receipt(not_evaluated)[0]["constitutional_valid"] is False

    for state in sorted(COMPLETION - {"CompleteWithinDeclaredFiniteDomain"}):
        candidate = copy.deepcopy(base)
        candidate["completion"] = state
        validated, _, _ = canonical_receipt(candidate)
        assert validated["completion"] == state

    expect_error(
        "duplicate metric ids",
        lambda x: x["metric_schema"]["metrics"].append(copy.deepcopy(x["metric_schema"]["metrics"][0])),
    )
    expect_error(
        "unknown metric value",
        lambda x: x["metric_values"].__setitem__("undeclared_metric", 1.0),
    )
    expect_error(
        "missing metric value",
        lambda x: x["metric_values"].pop("verification_events"),
    )
    expect_error(
        "nonfinite metric value",
        lambda x: x["metric_values"].__setitem__("welfare_total", float("inf")),
    )
    expect_error(
        "violated requires witness",
        lambda x: (
            x["constitutional_results"].__setitem__(
                0,
                {
                    "id": "no-unauthorized-action",
                    "status": "Violated",
                    "witness": None,
                    "reason": None,
                },
            ),
            x.__setitem__("constitutional_valid", False),
        ),
    )
    expect_error(
        "not evaluated requires reason",
        lambda x: (
            x["constitutional_results"].__setitem__(
                0,
                {
                    "id": "no-unauthorized-action",
                    "status": "NotEvaluated",
                    "witness": None,
                    "reason": None,
                },
            ),
            x.__setitem__("constitutional_valid", False),
        ),
    )
    expect_error(
        "constitutional validity cannot lie",
        lambda x: (
            x["constitutional_results"].__setitem__(
                0,
                {
                    "id": "no-unauthorized-action",
                    "status": "Violated",
                    "witness": "x",
                    "reason": None,
                },
            ),
            x.__setitem__("constitutional_valid", True),
        ),
    )
    expect_error(
        "unknown receipt key",
        lambda x: x.__setitem__("governance_score", 0.99),
    )
    expect_error(
        "unknown identity key",
        lambda x: x["experiment_identity"].__setitem__("display_name", "ignored?"),
    )
    expect_error(
        "metric schema ref mismatch",
        lambda x: x["experiment_identity"]["metric_schema"].__setitem__("revision", 99),
    )

    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "fixture_receipt_sha256": digest_a,
        "fixture_experiment_identity_sha256": digest(
            validate_experiment_identity(base["experiment_identity"])
        ),
        "metric_schema_sha256": digest(validate_metric_schema(base["metric_schema"])),
        "identity_mutation_sha256": identity_digests,
        "constitutional_violation_not_compensated": True,
        "descriptive_metric_direction_preserved": True,
        "completion_states_checked": sorted(COMPLETION),
        "negative_controls_passed": 10,
        "all_self_tests_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="execute contract regression corpus")
    parser.add_argument("--fixture", action="store_true", help="emit canonical fixture receipt")
    args = parser.parse_args()

    if not args.self_test and not args.fixture:
        parser.error("choose --self-test or --fixture")

    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True, separators=(",", ":"), allow_nan=False))
    else:
        _, encoded, receipt_digest = canonical_receipt(fixture())
        print(
            json.dumps(
                {
                    "receipt_json": encoded.decode("utf-8"),
                    "receipt_sha256": receipt_digest,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
