#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from typing import Any

import ig006a_contract_oracle as v1

SCHEMA = "symthaea-institutional-lab-receipt-v2"
AUTHORITY = v1.AUTHORITY
PROFILE_FIELDS = {"id", "revision", "content_sha256"}
IDENTITY_FIELDS = {"schema", *v1.PROFILE_KEYS, "seed"}


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise v1.ContractError(f"{label} must be a 64-character lowercase SHA-256 hex string")
    if any(ch not in "0123456789abcdef" for ch in value):
        raise v1.ContractError(f"{label} must be a 64-character lowercase SHA-256 hex string")
    return value


def validate_profile_ref(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise v1.ContractError(f"{label} must be an object")
    v1.require_exact_keys(value, PROFILE_FIELDS, label)
    return {
        "id": v1.require_nonempty_text(value["id"], f"{label}.id"),
        "revision": v1.require_revision(value["revision"], f"{label}.revision"),
        "content_sha256": require_sha256(value["content_sha256"], f"{label}.content_sha256"),
    }


def validate_experiment_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise v1.ContractError("experiment_identity must be an object")
    v1.require_exact_keys(value, IDENTITY_FIELDS, "experiment_identity")
    if value["schema"] != SCHEMA:
        raise v1.ContractError(f"experiment_identity.schema must equal {SCHEMA}")
    result: dict[str, Any] = {"schema": SCHEMA}
    for key in v1.PROFILE_KEYS:
        result[key] = validate_profile_ref(value[key], f"experiment_identity.{key}")
    result["seed"] = v1.require_seed(value["seed"])
    return result


def validate_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise v1.ContractError("receipt must be an object")
    v1.require_exact_keys(value, v1.RECEIPT_FIELDS, "receipt")
    if v1.FORBIDDEN_SCORE_FIELDS.intersection(value):
        raise v1.ContractError("forbidden aggregate score/safety field")
    if value["schema"] != SCHEMA:
        raise v1.ContractError(f"receipt.schema must equal {SCHEMA}")
    if value["authority"] != AUTHORITY:
        raise v1.ContractError(f"receipt.authority must equal {AUTHORITY}")

    identity = validate_experiment_identity(value["experiment_identity"])
    completion = value["completion"]
    if completion not in v1.COMPLETION:
        raise v1.ContractError(f"invalid completion state: {completion!r}")

    metric_schema = v1.validate_metric_schema(value["metric_schema"])
    metric_schema_sha256 = v1.digest(metric_schema)
    metric_ref = identity["metric_schema"]
    if (
        metric_ref["id"] != metric_schema["id"]
        or metric_ref["revision"] != metric_schema["revision"]
        or metric_ref["content_sha256"] != metric_schema_sha256
    ):
        raise v1.ContractError(
            "experiment identity metric-schema reference does not match full receipt schema"
        )
    metric_values = v1.validate_metric_values(metric_schema, value["metric_values"])

    constitutional_results, constitutional_valid = v1.validate_constitutional_results(
        value["constitutional_results"]
    )
    if value["constitutional_valid"] is not constitutional_valid:
        raise v1.ContractError("constitutional_valid does not match invariant results")

    mechanism_trace = v1.require_nonempty_text(value["mechanism_trace"], "mechanism_trace")
    if not isinstance(value["analysis_witnesses"], list):
        raise v1.ContractError("analysis_witnesses must be an array")
    analysis_witnesses = [
        v1.require_nonempty_text(witness, f"analysis_witnesses[{index}]")
        for index, witness in enumerate(value["analysis_witnesses"])
    ]

    def validate_text_array(name: str) -> list[str]:
        raw = value[name]
        if not isinstance(raw, list):
            raise v1.ContractError(f"{name} must be an array")
        return [v1.require_nonempty_text(item, f"{name}[]") for item in raw]

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
        "warnings": validate_text_array("warnings"),
        "non_claims": validate_text_array("non_claims"),
    }


def canonical_receipt(value: Any) -> tuple[dict[str, Any], bytes, str]:
    validated = validate_receipt(value)
    encoded = v1.canonical_bytes(validated)
    return validated, encoded, hashlib.sha256(encoded).hexdigest()


def fixture_profile_ref(profile_id: str, revision: int) -> dict[str, Any]:
    commitment = hashlib.sha256(
        f"ig006a0h-fixture-profile-v2:{profile_id}:{revision}".encode("utf-8")
    ).hexdigest()
    return {
        "id": profile_id,
        "revision": revision,
        "content_sha256": commitment,
    }


def fixture() -> dict[str, Any]:
    base = copy.deepcopy(v1.fixture())
    base["schema"] = SCHEMA
    old_identity = base["experiment_identity"]
    identity: dict[str, Any] = {"schema": SCHEMA}
    for key in v1.PROFILE_KEYS:
        profile_id = old_identity[key]["id"]
        revision = old_identity[key]["revision"]
        identity[key] = fixture_profile_ref(profile_id, revision)
    identity["metric_schema"]["content_sha256"] = v1.digest(
        v1.validate_metric_schema(base["metric_schema"])
    )
    identity["seed"] = old_identity["seed"]
    base["experiment_identity"] = identity
    return base


def expect_error(label: str, mutator) -> None:
    candidate = fixture()
    mutator(candidate)
    try:
        canonical_receipt(candidate)
    except v1.ContractError:
        return
    raise AssertionError(f"{label}: expected ContractError")


def self_test() -> dict[str, Any]:
    v1_result = v1.self_test()
    assert v1_result["all_self_tests_passed"] is True

    base = fixture()
    validated_a, encoded_a, digest_a = canonical_receipt(base)
    validated_b, encoded_b, digest_b = canonical_receipt(copy.deepcopy(base))
    assert encoded_a == encoded_b
    assert digest_a == digest_b
    assert validated_a == validated_b

    base_identity_digest = v1.digest(
        validate_experiment_identity(base["experiment_identity"])
    )

    mechanism_changed = copy.deepcopy(base)
    mechanism_changed["experiment_identity"]["mechanism"]["content_sha256"] = hashlib.sha256(
        b"different mechanism bytes under same id/revision"
    ).hexdigest()
    mechanism_identity_digest = v1.digest(
        validate_experiment_identity(mechanism_changed["experiment_identity"])
    )
    assert mechanism_identity_digest != base_identity_digest

    scenario_changed = copy.deepcopy(base)
    scenario_changed["experiment_identity"]["scenario"]["content_sha256"] = hashlib.sha256(
        b"different scenario bytes under same id/revision"
    ).hexdigest()
    scenario_identity_digest = v1.digest(
        validate_experiment_identity(scenario_changed["experiment_identity"])
    )
    assert scenario_identity_digest != base_identity_digest

    expect_error(
        "same-revision metric schema substitution",
        lambda x: x["metric_schema"]["metrics"][0].__setitem__(
            "description", "mutated semantics under same reference"
        ),
    )

    metric_rebound = copy.deepcopy(base)
    metric_rebound["metric_schema"]["metrics"][0]["description"] = (
        "intentionally revised fixture semantics"
    )
    metric_rebound["experiment_identity"]["metric_schema"]["content_sha256"] = v1.digest(
        v1.validate_metric_schema(metric_rebound["metric_schema"])
    )
    _, _, metric_rebound_digest = canonical_receipt(metric_rebound)
    metric_rebound_identity_digest = v1.digest(
        validate_experiment_identity(metric_rebound["experiment_identity"])
    )
    assert metric_rebound_digest != digest_a
    assert metric_rebound_identity_digest != base_identity_digest

    expect_error(
        "malformed content commitment",
        lambda x: x["experiment_identity"]["mechanism"].__setitem__(
            "content_sha256", "not-a-valid-sha256"
        ),
    )
    expect_error(
        "uppercase content commitment",
        lambda x: x["experiment_identity"]["mechanism"].__setitem__(
            "content_sha256", "A" * 64
        ),
    )

    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "v1_prerequisite_self_test_passed": True,
        "fixture_receipt_sha256": digest_a,
        "fixture_experiment_identity_sha256": base_identity_digest,
        "metric_schema_sha256": v1.digest(v1.validate_metric_schema(base["metric_schema"])),
        "same_revision_mechanism_content_changes_identity": True,
        "same_revision_scenario_content_changes_identity": True,
        "metric_schema_content_is_self_bound": True,
        "profile_commitment_format_fail_closed": True,
        "mechanism_changed_identity_sha256": mechanism_identity_digest,
        "scenario_changed_identity_sha256": scenario_identity_digest,
        "metric_rebound_identity_sha256": metric_rebound_identity_digest,
        "metric_rebound_receipt_sha256": metric_rebound_digest,
        "all_self_tests_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fixture", action="store_true")
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
