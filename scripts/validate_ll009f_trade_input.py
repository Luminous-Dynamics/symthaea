#!/usr/bin/env python3
"""Validate a common-accounting LL-009F lunar transport trade-study input.

LL-009F admits or rejects comparable candidate inputs. It does not rank candidates
or choose a transport architecture. A promoted input must be bound to one valid
LL-009E envelope and one canonical accounting contract shared by every candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

TRADE_SCHEMA = "ll009f.trade-input.v1"
RECEIPT_SCHEMA = "ll009f.trade-input-receipt.v1"
ENVELOPE_INDEX_SCHEMA = "ll009e.contract-bound-capsule-index.v1"
ENVELOPE_RECEIPT_SCHEMA = "ll009e.contract-bound-capsule-receipt.v1"

ALLOWED_COMPONENT_STATUS = {"included", "not_applicable", "unresolved"}
ALLOWED_EVIDENCE_CLASSES = {
    "synthetic",
    "authoritative_source",
    "derived_real_data",
    "simulation",
    "measured",
    "qualified",
}
PROMOTION = "pareto_ready_for_real_site_comparison"
RESEARCH = "research_trade_input"

BASELINE_REQUIRED_COMPONENTS = {
    "decommission_disposal",
    "earth_imported_construction",
    "failure_recovery",
    "feeder_last_mile",
    "local_construction",
    "maintenance_spares",
    "operations",
    "power_generation_storage",
    "stationkeeping_propellant",
    "transport_hardware",
}
BASELINE_REQUIRED_METRICS = {
    "availability_fraction",
    "energy_kwh_per_delivered_kg",
    "expected_lost_cargo_kg_per_year",
    "imported_mass_kg",
    "latency_hours",
    "local_mass_kg",
    "maintenance_import_mass_kg_per_year",
    "peak_power_kw",
    "propellant_reaction_mass_kg_per_year",
    "reliability_success_probability",
    "reusable_hardware_inventory_kg",
    "throughput_capacity_kg_per_year",
}


class TradeInputError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TradeInputError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TradeInputError(f"expected JSON object: {path}")
    return value


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def tree_manifest(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude or set()
    records: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in excluded:
            continue
        records.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def aggregate_digest(records: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        h.update(record["path"].encode("utf-8"))
        h.update(b"\0")
        h.update(record["sha256"].encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def validate_ll009e_envelope(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise TradeInputError(f"LL-009E envelope root not found: {root}")
    index_path = root / "ll009e_index.json"
    receipt_path = root / "ll009e_receipt.json"
    if not index_path.is_file() or not receipt_path.is_file():
        raise TradeInputError("LL-009F requires an LL-009E envelope with index and receipt")

    index = load_json(index_path)
    receipt = load_json(receipt_path)
    if index.get("schema_version") != ENVELOPE_INDEX_SCHEMA:
        raise TradeInputError("unsupported/missing LL-009E index schema")
    if receipt.get("schema_version") != ENVELOPE_RECEIPT_SCHEMA:
        raise TradeInputError("unsupported/missing LL-009E receipt schema")
    if receipt.get("status") != "pass":
        raise TradeInputError("LL-009E receipt status is not pass")

    index_hash = sha256_file(index_path)
    if receipt.get("ll009e_index_sha256") != index_hash:
        raise TradeInputError("LL-009E receipt/index hash mismatch")

    pre_receipt = tree_manifest(root, exclude={"ll009e_receipt.json"})
    aggregate = aggregate_digest(pre_receipt)
    if receipt.get("content_aggregate_sha256") != aggregate:
        raise TradeInputError("LL-009E aggregate digest mismatch")

    for field in ("bundle_id", "study_id", "frame_contract_id", "epoch_contract_id"):
        if not nonempty(index.get(field)):
            raise TradeInputError(f"LL-009E index missing {field}")
    if receipt.get("bundle_id") != index["bundle_id"]:
        raise TradeInputError("LL-009E receipt bundle_id mismatch")
    if receipt.get("study_id") != index["study_id"]:
        raise TradeInputError("LL-009E receipt study_id mismatch")

    return {
        "bundle_id": index["bundle_id"],
        "study_id": index["study_id"],
        "frame_contract_id": index["frame_contract_id"],
        "epoch_contract_id": index["epoch_contract_id"],
        "index_sha256": index_hash,
        "receipt_sha256": sha256_file(receipt_path),
        "aggregate_sha256": aggregate,
    }


def validate_accounting_contract(contract: dict[str, Any]) -> str:
    required_strings = (
        "name",
        "corridor_ref",
        "cargo_class",
        "demand_scenario_ref",
        "earth_import_boundary",
        "local_material_boundary",
        "power_boundary",
        "maintenance_boundary",
        "feeder_last_mile_boundary",
        "failure_recovery_boundary",
        "decommission_boundary",
        "currency_basis",
    )
    for field in required_strings:
        if not nonempty(contract.get(field)):
            raise TradeInputError(f"accounting contract missing {field}")

    horizon = contract.get("study_horizon_days")
    demand = contract.get("throughput_demand_kg_per_year")
    if not finite_number(horizon) or horizon <= 0:
        raise TradeInputError("accounting study_horizon_days must be positive")
    if not finite_number(demand) or demand < 0:
        raise TradeInputError("accounting throughput_demand_kg_per_year must be non-negative")

    reliability = contract.get("reliability_horizon_days")
    if not finite_number(reliability) or reliability <= 0:
        raise TradeInputError("accounting reliability_horizon_days must be positive")

    return f"accounting-{sha256_bytes(canonical_json_bytes(contract))}"


def require_unique_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise TradeInputError(f"{label} must be a non-empty list")
    if any(not nonempty(item) for item in value):
        raise TradeInputError(f"{label} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise TradeInputError(f"{label} contains duplicates")
    return list(value)


def validate_component(
    candidate_id: str,
    component_name: str,
    value: Any,
    promoted: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TradeInputError(f"{candidate_id}.{component_name} must be an object")
    status = value.get("status")
    if status not in ALLOWED_COMPONENT_STATUS:
        raise TradeInputError(
            f"{candidate_id}.{component_name}: invalid status {status!r}"
        )
    if status == "included":
        if not nonempty(value.get("evidence_ref")):
            raise TradeInputError(
                f"{candidate_id}.{component_name}: included requires evidence_ref"
            )
    elif status == "not_applicable":
        if not nonempty(value.get("reason")):
            raise TradeInputError(
                f"{candidate_id}.{component_name}: not_applicable requires reason"
            )
    elif promoted:
        raise TradeInputError(
            f"{candidate_id}.{component_name}: unresolved is forbidden for promoted comparison"
        )
    return {
        "status": status,
        "evidence_ref": value.get("evidence_ref"),
        "reason": value.get("reason"),
    }


def validate_metric(
    candidate_id: str,
    metric_name: str,
    value: Any,
    promoted: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TradeInputError(f"{candidate_id}.{metric_name} must be an object")
    for field in ("low", "central", "high"):
        if not finite_number(value.get(field)):
            raise TradeInputError(
                f"{candidate_id}.{metric_name}: {field} must be a finite number"
            )
    low, central, high = value["low"], value["central"], value["high"]
    if not low <= central <= high:
        raise TradeInputError(
            f"{candidate_id}.{metric_name}: uncertainty must satisfy low <= central <= high"
        )
    if not nonempty(value.get("unit")):
        raise TradeInputError(f"{candidate_id}.{metric_name}: missing unit")
    evidence_class = value.get("evidence_class")
    if evidence_class not in ALLOWED_EVIDENCE_CLASSES:
        raise TradeInputError(
            f"{candidate_id}.{metric_name}: invalid evidence_class {evidence_class!r}"
        )
    if not nonempty(value.get("source_ref")):
        raise TradeInputError(f"{candidate_id}.{metric_name}: missing source_ref")
    if promoted and evidence_class == "synthetic":
        raise TradeInputError(
            f"{candidate_id}.{metric_name}: synthetic evidence forbidden for promoted comparison"
        )
    return {
        "low": low,
        "central": central,
        "high": high,
        "unit": value["unit"],
        "evidence_class": evidence_class,
        "source_ref": value["source_ref"],
    }


def validate_trade_input(
    manifest_path: Path,
    envelope_root: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != TRADE_SCHEMA:
        raise TradeInputError(f"trade manifest schema_version must be {TRADE_SCHEMA}")

    promotion = manifest.get("requested_promotion")
    if promotion not in {RESEARCH, PROMOTION}:
        raise TradeInputError(f"invalid requested_promotion: {promotion!r}")
    promoted = promotion == PROMOTION

    envelope = validate_ll009e_envelope(envelope_root)
    for field in ("study_id", "frame_contract_id", "epoch_contract_id"):
        if manifest.get(field) != envelope[field]:
            raise TradeInputError(
                f"trade manifest {field} does not match LL-009E envelope"
            )

    accounting = manifest.get("accounting_contract")
    if not isinstance(accounting, dict):
        raise TradeInputError("trade manifest missing accounting_contract object")
    accounting_id = validate_accounting_contract(accounting)
    if manifest.get("accounting_contract_id") != accounting_id:
        raise TradeInputError(
            f"accounting_contract_id mismatch expected={accounting_id} "
            f"actual={manifest.get('accounting_contract_id')}"
        )

    required_components = require_unique_string_list(
        manifest.get("required_components"), "required_components"
    )
    required_metrics = require_unique_string_list(
        manifest.get("required_metrics"), "required_metrics"
    )
    missing_baseline_components = sorted(
        BASELINE_REQUIRED_COMPONENTS.difference(required_components)
    )
    if missing_baseline_components:
        raise TradeInputError(
            "required_components omits baseline accounting components: "
            + ", ".join(missing_baseline_components)
        )
    missing_baseline_metrics = sorted(
        BASELINE_REQUIRED_METRICS.difference(required_metrics)
    )
    if missing_baseline_metrics:
        raise TradeInputError(
            "required_metrics omits baseline comparison metrics: "
            + ", ".join(missing_baseline_metrics)
        )

    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise TradeInputError("trade study requires at least two candidates")

    normalized_candidates: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TradeInputError("candidate entries must be objects")
        candidate_id = candidate.get("candidate_id")
        architecture = candidate.get("architecture")
        if not nonempty(candidate_id) or not nonempty(architecture):
            raise TradeInputError("candidate requires candidate_id and architecture")
        if candidate_id in candidate_ids:
            raise TradeInputError(f"duplicate candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        if candidate.get("accounting_contract_id") != accounting_id:
            raise TradeInputError(
                f"{candidate_id}: accounting_contract_id mismatch"
            )

        components = candidate.get("components")
        if not isinstance(components, dict):
            raise TradeInputError(f"{candidate_id}: components must be an object")
        if set(components) != set(required_components):
            missing = sorted(set(required_components) - set(components))
            extra = sorted(set(components) - set(required_components))
            raise TradeInputError(
                f"{candidate_id}: component key mismatch missing={missing} extra={extra}"
            )
        normalized_components = {
            name: validate_component(candidate_id, name, components[name], promoted)
            for name in sorted(required_components)
        }

        metrics = candidate.get("metrics")
        if not isinstance(metrics, dict):
            raise TradeInputError(f"{candidate_id}: metrics must be an object")
        if set(metrics) != set(required_metrics):
            missing = sorted(set(required_metrics) - set(metrics))
            extra = sorted(set(metrics) - set(required_metrics))
            raise TradeInputError(
                f"{candidate_id}: metric key mismatch missing={missing} extra={extra}"
            )
        normalized_metrics = {
            name: validate_metric(candidate_id, name, metrics[name], promoted)
            for name in sorted(required_metrics)
        }

        normalized_candidates.append(
            {
                "candidate_id": candidate_id,
                "architecture": architecture,
                "accounting_contract_id": accounting_id,
                "components": normalized_components,
                "metrics": normalized_metrics,
            }
        )

    normalized = {
        "schema_version": "ll009f.normalized-trade-input.v1",
        "requested_promotion": promotion,
        "study_id": envelope["study_id"],
        "frame_contract_id": envelope["frame_contract_id"],
        "epoch_contract_id": envelope["epoch_contract_id"],
        "accounting_contract_id": accounting_id,
        "accounting_contract": accounting,
        "required_components": sorted(required_components),
        "required_metrics": sorted(required_metrics),
        "candidates": sorted(normalized_candidates, key=lambda item: item["candidate_id"]),
        "ll009e": envelope,
        "non_claim": (
            "LL-009F admission establishes comparable research input boundaries only; "
            "it does not rank candidates or select infrastructure."
        ),
    }
    normalized_payload = canonical_json_bytes(normalized)
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "pass",
        "requested_promotion": promotion,
        "study_id": envelope["study_id"],
        "accounting_contract_id": accounting_id,
        "candidate_count": len(normalized_candidates),
        "normalized_trade_input_sha256": sha256_bytes(normalized_payload),
        "source_manifest_sha256": sha256_file(manifest_path),
        "ll009e_index_sha256": envelope["index_sha256"],
        "ll009e_receipt_sha256": envelope["receipt_sha256"],
        "normalized_trade_input": normalized,
        "non_claim": (
            "A passing LL-009F receipt is an admission/comparability result, not "
            "a Pareto result, engineering selection, or launch authorization."
        ),
    }


def make_synthetic_envelope(root: Path) -> dict[str, str]:
    root.mkdir(parents=True)
    study_id = "study-" + "1" * 64
    frame_id = "frame-" + "2" * 64
    epoch_id = "epoch-" + "3" * 64
    (root / "contracts").mkdir()
    (root / "contracts" / "dummy.json").write_text('{"ok":true}\n', encoding="utf-8")
    index = {
        "schema_version": ENVELOPE_INDEX_SCHEMA,
        "bundle_id": "synthetic-bundle",
        "study_id": study_id,
        "frame_contract_id": frame_id,
        "epoch_contract_id": epoch_id,
        "requested_promotion": "research_bundle",
        "non_claims": ["synthetic"],
    }
    index_path = root / "ll009e_index.json"
    index_path.write_bytes(canonical_json_bytes(index))
    pre_receipt = tree_manifest(root)
    receipt = {
        "schema_version": ENVELOPE_RECEIPT_SCHEMA,
        "status": "pass",
        "bundle_id": index["bundle_id"],
        "study_id": study_id,
        "ll009e_index_sha256": sha256_file(index_path),
        "content_aggregate_sha256": aggregate_digest(pre_receipt),
        "file_count_excluding_receipt": len(pre_receipt),
        "non_claim": "synthetic",
    }
    (root / "ll009e_receipt.json").write_bytes(canonical_json_bytes(receipt))
    return {
        "study_id": study_id,
        "frame_contract_id": frame_id,
        "epoch_contract_id": epoch_id,
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ll009f-selftest-") as tmp:
        root = Path(tmp)
        envelope_root = root / "envelope"
        ids = make_synthetic_envelope(envelope_root)

        accounting = {
            "name": "synthetic-common-boundary",
            "corridor_ref": "corridor:synthetic",
            "cargo_class": "bulk",
            "demand_scenario_ref": "demand:synthetic",
            "study_horizon_days": 3650.0,
            "throughput_demand_kg_per_year": 1_000_000.0,
            "reliability_horizon_days": 365.0,
            "earth_import_boundary": "all imported construction/spares",
            "local_material_boundary": "all lunar-derived construction mass",
            "power_boundary": "generation plus storage",
            "maintenance_boundary": "planned/unplanned maintenance and spares",
            "feeder_last_mile_boundary": "source to destination inclusive",
            "failure_recovery_boundary": "lost cargo and hardware replacement",
            "decommission_boundary": "safe disposal/removal",
            "currency_basis": "none",
        }
        accounting_id = f"accounting-{sha256_bytes(canonical_json_bytes(accounting))}"
        components = [
            "decommission_disposal",
            "earth_imported_construction",
            "failure_recovery",
            "feeder_last_mile",
            "local_construction",
            "maintenance_spares",
            "operations",
            "power_generation_storage",
            "stationkeeping_propellant",
            "transport_hardware",
        ]
        metrics = sorted(BASELINE_REQUIRED_METRICS)

        def candidate(cid: str, architecture: str) -> dict[str, Any]:
            comp = {
                name: {
                    "status": "included",
                    "evidence_ref": f"evidence:{cid}:{name}",
                }
                for name in components
            }
            comp["stationkeeping_propellant"] = {
                "status": "not_applicable",
                "reason": "synthetic surface-only candidate",
            }
            met = {}
            for i, name in enumerate(metrics):
                central = 0.0 if name == "local_mass_kg" else float(i + 1)
                met[name] = {
                    "low": central,
                    "central": central,
                    "high": central,
                    "unit": "dimensionless" if "fraction" in name or "probability" in name else "synthetic-unit",
                    "evidence_class": "simulation",
                    "source_ref": f"metric:{cid}:{name}",
                }
            return {
                "candidate_id": cid,
                "architecture": architecture,
                "accounting_contract_id": accounting_id,
                "components": comp,
                "metrics": met,
            }

        manifest = {
            "schema_version": TRADE_SCHEMA,
            "requested_promotion": PROMOTION,
            **ids,
            "accounting_contract_id": accounting_id,
            "accounting_contract": accounting,
            "required_components": components,
            "required_metrics": metrics,
            "candidates": [
                candidate("rover-v1", "rover"),
                candidate("ballistic-v1", "ballistic_surface_freight"),
            ],
        }
        manifest_path = root / "trade.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        receipt = validate_trade_input(manifest_path, envelope_root)
        assert receipt["status"] == "pass"
        assert receipt["candidate_count"] == 2

        unresolved = json.loads(json.dumps(manifest))
        unresolved["candidates"][0]["components"]["maintenance_spares"] = {
            "status": "unresolved"
        }
        unresolved_path = root / "unresolved.json"
        unresolved_path.write_bytes(canonical_json_bytes(unresolved))
        try:
            validate_trade_input(unresolved_path, envelope_root)
        except TradeInputError:
            pass
        else:
            raise AssertionError("promoted unresolved component must fail")

        wrong_accounting = json.loads(json.dumps(manifest))
        wrong_accounting["candidates"][1]["accounting_contract_id"] = "accounting-" + "f" * 64
        wrong_path = root / "wrong-accounting.json"
        wrong_path.write_bytes(canonical_json_bytes(wrong_accounting))
        try:
            validate_trade_input(wrong_path, envelope_root)
        except TradeInputError:
            pass
        else:
            raise AssertionError("candidate accounting mismatch must fail")

        missing_component = json.loads(json.dumps(manifest))
        del missing_component["candidates"][0]["components"]["operations"]
        missing_path = root / "missing-component.json"
        missing_path.write_bytes(canonical_json_bytes(missing_component))
        try:
            validate_trade_input(missing_path, envelope_root)
        except TradeInputError:
            pass
        else:
            raise AssertionError("missing common component must fail")

        # Zero is a legitimate numeric value, not missing data.
        zero_metric = manifest["candidates"][0]["metrics"]["local_mass_kg"]
        assert zero_metric["central"] == 0.0

        index_path = envelope_root / "ll009e_index.json"
        original = index_path.read_bytes()
        index_path.write_text('{"tampered":true}\n', encoding="utf-8")
        try:
            validate_trade_input(manifest_path, envelope_root)
        except TradeInputError:
            pass
        else:
            raise AssertionError("tampered LL-009E envelope must fail")
        index_path.write_bytes(original)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ll009e-envelope", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("LL-009F common-accounting gate self-test: PASS")
        return 0
    if args.manifest is None or args.ll009e_envelope is None:
        raise TradeInputError(
            "--manifest and --ll009e-envelope are required unless --self-test is used"
        )
    receipt = validate_trade_input(args.manifest, args.ll009e_envelope)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TradeInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
