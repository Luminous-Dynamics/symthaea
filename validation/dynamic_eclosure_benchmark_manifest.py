#!/usr/bin/env python3
"""Deterministic manifests and comparison contracts for dynamic e-closure tests.

The module avoids wall-clock timestamps and Python's process-randomized hash().
Seed derivation is stable across platforms and Python versions. A comparison
contract makes the common data-generating process, filtration, null/alternative,
arrival, reporting, and capacity semantics explicit before comparing methods.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Iterable, Mapping


SCHEMA = "symthaea-dynamic-eclosure-benchmark-manifest-v1"
SEED_DOMAIN = "symthaea-dynamic-eclosure-seed-v1"
GUARANTEE_CLASSES = {
    "stopped_fdr",
    "supfdr",
    "component_crossing",
    "operational_decision_quality",
}
CONTRACT_KEYS = (
    "data_generator_sha256",
    "filtration_contract_sha256",
    "arrival_policy_sha256",
    "null_spec_sha256",
    "alternative_spec_sha256",
    "reporting_schedule_sha256",
    "capacity_utility_sha256",
)


def canonical_json(payload: Any) -> str:
    """Canonical JSON used for source, scenario, and receipt digests."""
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"payload is not canonical-JSON serializable: {error}") from error


def canonical_sha256(payload: Any) -> str:
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return value


def derive_replicate_seed(
    master_seed: int,
    scenario_id: str,
    replicate_index: int,
) -> int:
    """Stable unsigned 64-bit seed, independent of process hash randomization."""
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a nonempty string")
    if isinstance(replicate_index, bool) or not isinstance(replicate_index, int) or replicate_index < 0:
        raise ValueError("replicate_index must be a nonnegative integer")
    material = (
        f"{SEED_DOMAIN}\0{master_seed}\0{scenario_id}\0{replicate_index}"
    ).encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], "big", signed=False)


def validate_comparison_contract(contract: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(contract, Mapping):
        raise TypeError("comparison_contract must be a mapping")
    normalized = {}
    missing = [key for key in CONTRACT_KEYS if key not in contract]
    if missing:
        raise ValueError(f"comparison_contract missing fields: {missing}")
    for key in CONTRACT_KEYS:
        normalized[key] = _validate_sha256(contract[key], key)
    if set(contract).difference(CONTRACT_KEYS):
        raise ValueError(
            f"comparison_contract has unknown fields: {sorted(set(contract).difference(CONTRACT_KEYS))}"
        )
    return normalized


def build_benchmark_manifest(
    *,
    method_id: str,
    guarantee_class: str,
    alpha_numerator: int,
    alpha_denominator: int,
    source_sha256: str,
    generator_sha256: str,
    comparison_contract: Mapping[str, Any],
    scenario_id: str,
    master_seed: int,
    replicate_index: int,
    samples: int,
    horizon: int,
    method_assumptions: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    metrics: Iterable[str] = (),
    solver_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one immutable benchmark-run manifest with a reproducible seed."""
    if not isinstance(method_id, str) or not method_id.strip():
        raise ValueError("method_id must be a nonempty string")
    if guarantee_class not in GUARANTEE_CLASSES:
        raise ValueError(f"guarantee_class must be one of {sorted(GUARANTEE_CLASSES)}")
    for value, name in (
        (alpha_numerator, "alpha_numerator"),
        (alpha_denominator, "alpha_denominator"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if alpha_numerator >= alpha_denominator:
        raise ValueError("alpha must lie strictly between zero and one")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a nonempty string")

    source_digest = _validate_sha256(source_sha256, "source_sha256")
    generator_digest = _validate_sha256(generator_sha256, "generator_sha256")
    contract = validate_comparison_contract(comparison_contract)
    seed = derive_replicate_seed(master_seed, scenario_id, replicate_index)

    assumptions = dict(method_assumptions or {})
    env = dict(environment or {})
    solver = dict(solver_config or {})
    metric_names = tuple(metrics)
    if any(not isinstance(name, str) or not name for name in metric_names):
        raise ValueError("metric names must be nonempty strings")
    if len(set(metric_names)) != len(metric_names):
        raise ValueError("metric names must be unique")

    manifest = {
        "schema": SCHEMA,
        "method_id": method_id,
        "guarantee_class": guarantee_class,
        "alpha": [alpha_numerator, alpha_denominator],
        "source_sha256": source_digest,
        "generator_sha256": generator_digest,
        "comparison_contract": contract,
        "scenario_id": scenario_id,
        "seed": {
            "derivation": SEED_DOMAIN,
            "master_seed": master_seed,
            "replicate_index": replicate_index,
            "derived_seed_u64": seed,
        },
        "samples": samples,
        "horizon": horizon,
        "method_assumptions": assumptions,
        "environment": env,
        "metrics": list(metric_names),
        "solver_config": solver,
    }
    # Fail before hashing if any caller-provided nested metadata is invalid JSON.
    canonical_json(manifest)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_benchmark_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    payload = dict(manifest)
    digest = payload.pop("manifest_sha256", None)
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported benchmark manifest schema")
    required_fields = {
        "schema", "method_id", "guarantee_class", "alpha", "source_sha256",
        "generator_sha256", "comparison_contract", "scenario_id", "seed",
        "samples", "horizon", "method_assumptions", "environment", "metrics",
        "solver_config",
    }
    if set(payload) != required_fields:
        raise ValueError(
            f"manifest fields differ from schema: missing={sorted(required_fields-set(payload))}, "
            f"unknown={sorted(set(payload)-required_fields)}"
        )
    if not isinstance(payload["method_id"], str) or not payload["method_id"].strip():
        raise ValueError("method_id must be a nonempty string")
    if payload["guarantee_class"] not in GUARANTEE_CLASSES:
        raise ValueError(f"unknown guarantee class {payload['guarantee_class']!r}")
    alpha = payload["alpha"]
    if (
        not isinstance(alpha, list)
        or len(alpha) != 2
        or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in alpha)
        or alpha[0] >= alpha[1]
    ):
        raise ValueError("manifest alpha must be a positive rational strictly between zero and one")
    _validate_sha256(payload["source_sha256"], "source_sha256")
    _validate_sha256(payload["generator_sha256"], "generator_sha256")
    validate_comparison_contract(payload["comparison_contract"])
    if not isinstance(payload["scenario_id"], str) or not payload["scenario_id"]:
        raise ValueError("scenario_id must be a nonempty string")
    seed_record = payload["seed"]
    if not isinstance(seed_record, Mapping):
        raise TypeError("seed record must be a mapping")
    if set(seed_record) != {"derivation", "master_seed", "replicate_index", "derived_seed_u64"}:
        raise ValueError("seed record fields do not match schema")
    expected_seed = derive_replicate_seed(
        seed_record["master_seed"],
        payload["scenario_id"],
        seed_record["replicate_index"],
    )
    if seed_record.get("derivation") != SEED_DOMAIN:
        raise ValueError("unknown seed derivation scheme")
    if (
        isinstance(seed_record.get("derived_seed_u64"), bool)
        or not isinstance(seed_record.get("derived_seed_u64"), int)
        or seed_record.get("derived_seed_u64") != expected_seed
    ):
        raise ValueError("derived seed does not match the deterministic seed contract")
    for field in ("samples", "horizon"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("method_assumptions", "environment", "solver_config"):
        if not isinstance(payload[field], Mapping):
            raise TypeError(f"{field} must be a mapping")
    metrics = payload["metrics"]
    if (
        not isinstance(metrics, list)
        or any(not isinstance(name, str) or not name for name in metrics)
        or len(set(metrics)) != len(metrics)
    ):
        raise ValueError("metrics must be a list of unique nonempty strings")
    # Fail closed on nonfinite or unserializable nested metadata.
    canonical_json(payload)
    _validate_sha256(digest, "manifest_sha256")
    calculated = canonical_sha256(payload)
    if digest != calculated:
        raise ValueError("manifest_sha256 mismatch")
    return {
        "valid": True,
        "manifest_sha256": calculated,
        "derived_seed_u64": expected_seed,
        "guarantee_class": payload["guarantee_class"],
    }


def verify_comparable_manifests(manifests: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Require a matched scenario/guarantee contract before comparing methods."""
    rows = list(manifests)
    if len(rows) < 2:
        raise ValueError("at least two manifests are required for comparison")
    for row in rows:
        validate_benchmark_manifest(row)

    compared_fields = (
        "guarantee_class",
        "alpha",
        "generator_sha256",
        "comparison_contract",
        "scenario_id",
        "seed",
        "samples",
        "horizon",
    )
    reference = rows[0]
    mismatches = []
    for index, candidate in enumerate(rows[1:], start=1):
        for field in compared_fields:
            if candidate[field] != reference[field]:
                mismatches.append((index, field, reference[field], candidate[field]))
    method_ids = [row["method_id"] for row in rows]
    if len(set(method_ids)) != len(method_ids):
        mismatches.append(("duplicate_method_id", tuple(method_ids)))
    return {
        "comparable": not mismatches,
        "mismatches": mismatches,
        "methods": method_ids,
        "guarantee_class": reference["guarantee_class"],
        "alpha": reference["alpha"],
        "scenario_id": reference["scenario_id"],
        "derived_seed_u64": reference["seed"]["derived_seed_u64"],
        "matched_fields": list(compared_fields),
        "scope": "metadata contract verification; does not prove stochastic validity or benchmark implementation correctness",
    }


if __name__ == "__main__":
    raise SystemExit("Import from benchmark runners and manifest tests.")
