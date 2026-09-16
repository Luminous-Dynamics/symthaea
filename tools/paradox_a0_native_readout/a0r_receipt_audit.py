#!/usr/bin/env python3
"""Static, single-receipt, and receipt-set audit for PARADOX A0-R.

Presence of this file or a static PASS does not create scientific result
authority. Confirmatory execution remains forbidden by the frozen contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTRACT_PATH = HERE / "a0r_receipt_contract.json"
LATENT_PATH = HERE / "a0r_latent_contract.json"
STATS_PATH = HERE / "a0r_statistics.json"

EXPECTED_SCHEMA = "PARADOX-A0R-MEASUREMENT-RECEIPT-CONTRACT-V1"
EXPECTED_LATENT_SCHEMA = "PARADOX-A0R-LATENT-READOUT-CONTRACT-V2"
EXPECTED_STATS_SCHEMA = "PARADOX-A0R-STATISTICS-V1"
EXPECTED_PRODUCTION = "eb73527d05a913e79d1f05135ad6b06c1da8e2ee"
EXPECTED_G2B = "09d83a1d1fddbbbd30e4eba7cc95946c8eab871f"
EXPECTED_THEOREM = "A0R-CFC-INFALLIBLE-RESULT-PATH-V1"
EXPECTED_GENESIS_SHA256 = "0c481e5561fba5797438043e11b9c88b40eee2e8c6b1a4cacc9bbbf0e5eb19d4"


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def require(self, cond: bool, message: str) -> None:
        if not cond:
            self.errors.append(message)

    def equal(self, actual: Any, expected: Any, label: str) -> None:
        if actual != expected:
            self.errors.append(f"{label}: expected {expected!r}, got {actual!r}")

    def extend(self, prefix: str, errors: list[str]) -> None:
        self.errors.extend(f"{prefix}: {err}" for err in errors)

    def finish(self, label: str) -> None:
        if self.errors:
            print(f"{label}: FAIL ({len(self.errors)} error(s))", file=sys.stderr)
            for err in self.errors:
                print(f" - {err}", file=sys.stderr)
            raise SystemExit(1)
        print(f"{label}: PASS")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: top-level JSON value must be an object")
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def feature_set_sha256(features: list[str]) -> str:
    return sha256_bytes("\n".join(features).encode("utf-8"))


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def get_path(obj: dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(dotted)
        cur = cur[key]
    return cur


def static_audit() -> dict[str, Any]:
    a = Audit()
    contract = load_json(CONTRACT_PATH)
    latent = load_json(LATENT_PATH)
    stats = load_json(STATS_PATH)

    a.equal(contract.get("schema_version"), EXPECTED_SCHEMA, "contract schema")
    a.equal(contract.get("production_subject_sha"), EXPECTED_PRODUCTION, "production subject")
    a.equal(contract.get("g2b_subject_sha"), EXPECTED_G2B, "G2b subject")
    a.equal(contract.get("latent_contract_schema"), EXPECTED_LATENT_SCHEMA, "latent schema binding")
    a.equal(contract.get("statistics_contract_schema"), EXPECTED_STATS_SCHEMA, "statistics schema binding")
    a.equal(contract.get("confirmatory_execution_allowed"), False, "confirmatory execution gate")
    a.equal(contract.get("production_cognition_mutation_allowed"), False, "production mutation gate")

    a.equal(latent.get("schema_version"), EXPECTED_LATENT_SCHEMA, "loaded latent schema")
    a.equal(stats.get("schema_version"), EXPECTED_STATS_SCHEMA, "loaded statistics schema")
    for name, doc in (("latent", latent), ("statistics", stats)):
        a.equal(doc.get("production_subject_sha"), EXPECTED_PRODUCTION, f"{name} production subject")
        a.equal(doc.get("g2b_subject_sha"), EXPECTED_G2B, f"{name} G2b subject")
        a.equal(doc.get("confirmatory_execution_allowed"), False, f"{name} confirmatory gate")

    state_scope = latent.get("state_scope", {})
    a.equal(
        state_scope.get("unit"),
        "one independent production-service instance per development fixture",
        "latent state-scope unit",
    )
    a.equal(state_scope.get("cross_fixture_state"), "forbidden", "latent cross-fixture state")
    primary = latent.get("channels", {}).get("cfc_recurrent_state", {})
    a.equal(primary.get("role"), "primary_latent", "primary latent role")
    a.equal(primary.get("receipt"), "CycleResult.output", "primary latent receipt")

    inferential = stats.get("inferential_unit", {})
    a.equal(inferential.get("unit"), "independent base semantic fixture", "inferential unit")
    a.equal(inferential.get("metamorphic_variants_count_as_independent"), False, "metamorphic n rule")
    a.equal(inferential.get("multiple_cycles_count_as_independent"), False, "cycle n rule")

    model = contract.get("model_instance_scope", {})
    phrase = model.get("genesis_phrase")
    a.require(isinstance(phrase, str) and bool(phrase), "genesis phrase must be non-empty")
    if isinstance(phrase, str):
        a.equal(sha256_bytes(phrase.encode("utf-8")), EXPECTED_GENESIS_SHA256, "genesis phrase digest")
    a.equal(model.get("genesis_phrase_sha256"), EXPECTED_GENESIS_SHA256, "declared genesis digest")
    a.equal(model.get("same_genesis_for_all_base_fixtures"), True, "same-genesis fixture rule")
    a.equal(model.get("same_genesis_for_all_metamorphic_variants"), True, "same-genesis transform rule")
    a.equal(model.get("target_conditioned_genesis_forbidden"), True, "target-conditioned genesis rule")
    a.equal(model.get("cross_genesis_claim_authorized"), False, "cross-genesis claim gate")

    state = contract.get("service_state_contract", {})
    for key in (
        "fresh_service_per_base_fixture",
        "fresh_service_per_metamorphic_variant",
        "fresh_service_per_technical_replicate",
    ):
        a.equal(state.get(key), True, key)
    a.equal(state.get("cross_fixture_service_reuse"), "forbidden", "cross-fixture service reuse")
    a.equal(state.get("cross_variant_service_reuse"), "forbidden", "cross-variant service reuse")

    expected_cfg = contract.get("runner_config_v1", {}).get("required_exact_values", {})
    required_cfg = {
        "temporal_backend": "CfC",
        "genesis_phrase": "PARADOX-A0R-DEV-V1-GENESIS-2026-09-16",
        "cfc_config.num_neurons": 256,
        "cfc_config.input_dim": 256,
        "cfc_config.delta_t": 0.02,
        "cfc_config.prediction_horizons": [0.02, 0.1, 0.2],
        "async_training": False,
        "enable_online_learning": False,
        "episodic_replay_training": False,
        "memory_graduation": False,
        "enable_recurrent_dim_masking": False,
        "enable_spectral_entropy_masking": False,
        "effective_dim_fraction_override": None,
        "attention_budget_override_us": 60000000,
        "timezone_offset_hours": 0.0,
    }
    a.equal(expected_cfg, required_cfg, "runner_config_v1 exact values")

    theorem = contract.get("source_theorem", {})
    a.equal(theorem.get("id"), EXPECTED_THEOREM, "source theorem id")
    bindings = theorem.get("source_bindings")
    a.require(isinstance(bindings, list) and len(bindings) >= 5, "source theorem must bind >=5 files")
    seen: set[str] = set()
    if isinstance(bindings, list):
        a.equal(
            theorem.get("source_binding_digest_sha256"),
            canonical_json_sha256(bindings),
            "source-binding digest",
        )
        for idx, binding in enumerate(bindings):
            if not isinstance(binding, dict):
                a.errors.append(f"source binding {idx} is not an object")
                continue
            rel = binding.get("path")
            expected_blob = binding.get("blob_sha")
            required = binding.get("required_substrings", [])
            a.require(isinstance(rel, str) and bool(rel), f"source binding {idx} path")
            if not isinstance(rel, str):
                continue
            a.require(rel not in seen, f"duplicate source binding path: {rel}")
            seen.add(rel)
            path = ROOT / rel
            a.require(path.is_file(), f"bound source file missing: {rel}")
            if not path.is_file():
                continue
            data = path.read_bytes()
            a.equal(git_blob_sha1(data), expected_blob, f"{rel} blob SHA")
            text = data.decode("utf-8")
            a.require(isinstance(required, list) and bool(required), f"{rel} required_substrings")
            if isinstance(required, list):
                for needle in required:
                    a.require(
                        isinstance(needle, str) and needle in text,
                        f"{rel} missing frozen theorem text: {needle!r}",
                    )

    repeat = contract.get("technical_repeatability_gate", {})
    a.equal(repeat.get("replicates_per_measurement"), 2, "technical replicate count")
    a.equal(repeat.get("fresh_service_each_replicate"), True, "fresh technical-replicate service")
    a.equal(repeat.get("replicates_increase_statistical_n"), False, "technical replicate n rule")
    a.require("pairwise_exclusion" in repeat, "pairwise exclusion rule absent")

    fields = contract.get("required_receipt_fields", [])
    a.require(isinstance(fields, list), "required_receipt_fields must be a list")
    if isinstance(fields, list):
        a.equal(len(fields), len(set(fields)), "required_receipt_fields uniqueness")
        for field in (
            "receipt_id",
            "runner_config_effective",
            "enabled_features",
            "service_instance_id",
            "cfc_vector_f32le_sha256",
            "technical_repeatability_match",
            "probe_artifact_sha256",
            "sealed_prediction_sha256",
            "seal_stage",
            "label_join_stage",
            "measurement_validity",
            "invalidity_reasons",
        ):
            a.require(field in fields, f"required receipt field absent: {field}")

    set_rules = contract.get("receipt_set_rules", {})
    a.equal(set_rules.get("replicate_indices_exactly"), [0, 1], "receipt-set replicate indices")
    a.equal(set_rules.get("receipt_id_unique_globally"), True, "receipt-id uniqueness")
    a.equal(set_rules.get("service_instance_id_unique_globally"), True, "service-id uniqueness")
    a.equal(set_rules.get("peer_binding_symmetric"), True, "peer symmetry")

    rules = contract.get("validity_rules", {})
    a.equal(rules.get("eligible_status"), "ELIGIBLE_A0R_MEASUREMENT", "eligible status")
    a.equal(rules.get("invalid_measurements_enter_probe_fit"), False, "invalid probe-fit gate")
    a.equal(rules.get("invalid_measurements_enter_scientific_denominator"), False, "invalid denominator gate")
    a.require(
        "record but do not automatically invalidate" in str(rules.get("all_zero_vector_rule", "")),
        "all-zero rule must not infer generic fallback",
    )

    seal = contract.get("seal_rules", {})
    a.equal(seal.get("seal_stage"), "before_heldout_label_join", "seal stage")
    a.equal(seal.get("heldout_label_available_to_probe"), False, "heldout-label probe gate")
    a.equal(seal.get("heldout_condition_id_available_to_probe"), False, "condition probe gate")
    a.equal(seal.get("heldout_oracle_available_to_probe"), False, "oracle probe gate")
    a.equal(seal.get("label_join_is_analysis_only"), True, "analysis-only label join")

    try:
        changed = subprocess.run(
            ["git", "diff", "--name-only", EXPECTED_PRODUCTION, "--", *sorted(seen)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        a.equal(changed, "", "production-source diff from frozen subject")
    except (OSError, subprocess.CalledProcessError) as exc:
        a.errors.append(f"git production-source audit unavailable: {exc}")

    a.finish("PARADOX-A0R-RECEIPT-CONTRACT-STATIC")
    return contract


def validate_sha256_field(a: Audit, receipt: dict[str, Any], field: str) -> None:
    value = receipt.get(field)
    a.require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        f"{field} must be lowercase SHA-256 hex",
    )


def receipt_errors(contract: dict[str, Any], receipt: dict[str, Any]) -> list[str]:
    a = Audit()

    for field in contract["required_receipt_fields"]:
        a.require(field in receipt, f"missing required receipt field: {field}")

    a.equal(receipt.get("receipt_schema_version"), EXPECTED_SCHEMA, "receipt schema")
    a.equal(receipt.get("production_subject_sha"), EXPECTED_PRODUCTION, "receipt production subject")
    a.equal(receipt.get("g2b_subject_sha"), EXPECTED_G2B, "receipt G2b subject")
    a.equal(receipt.get("latent_contract_schema"), EXPECTED_LATENT_SCHEMA, "receipt latent schema")
    a.equal(receipt.get("statistics_contract_schema"), EXPECTED_STATS_SCHEMA, "receipt statistics schema")
    a.equal(receipt.get("source_theorem_id"), EXPECTED_THEOREM, "receipt source theorem")
    a.equal(receipt.get("genesis_phrase_sha256"), EXPECTED_GENESIS_SHA256, "receipt genesis digest")

    for field in (
        "canonical_config_sha256",
        "enabled_feature_set_sha256",
        "executable_sha256",
        "source_binding_digest",
        "cfc_vector_f32le_sha256",
        "perception_hdv_sha256",
        "perception_chunk32_f32le_sha256",
        "probe_artifact_sha256",
        "preprocessing_artifact_sha256",
        "sealed_feature_bundle_sha256",
        "sealed_prediction_sha256",
    ):
        validate_sha256_field(a, receipt, field)

    a.equal(
        receipt.get("source_binding_digest"),
        contract["source_theorem"]["source_binding_digest_sha256"],
        "receipt source-binding digest",
    )

    effective = receipt.get("runner_config_effective")
    a.require(isinstance(effective, dict), "runner_config_effective must be an object")
    if isinstance(effective, dict):
        for dotted, expected in contract["runner_config_v1"]["required_exact_values"].items():
            try:
                actual = get_path(effective, dotted)
            except KeyError:
                a.errors.append(f"runner_config_effective missing {dotted}")
                continue
            a.equal(actual, expected, f"runner config {dotted}")
        a.equal(
            receipt.get("canonical_config_sha256"),
            canonical_json_sha256(effective),
            "canonical config digest",
        )

    features = receipt.get("enabled_features")
    a.require(isinstance(features, list), "enabled_features must be an array")
    if isinstance(features, list):
        a.require(all(isinstance(x, str) and x for x in features), "enabled_features must be non-empty strings")
        if all(isinstance(x, str) for x in features):
            a.equal(features, sorted(set(features)), "enabled_features sorted unique")
            a.equal(
                receipt.get("enabled_feature_set_sha256"),
                feature_set_sha256(features),
                "enabled feature-set digest",
            )

    a.equal(receipt.get("cfc_vector_length"), 256, "CfC vector length")
    nonfinite = receipt.get("cfc_vector_nonfinite_count")
    a.require(isinstance(nonfinite, int) and not isinstance(nonfinite, bool), "nonfinite count must be int")
    if isinstance(nonfinite, int):
        a.equal(nonfinite, 0, "CfC nonfinite count")

    a.equal(receipt.get("recurrent_masking_enabled"), False, "recurrent masking")
    a.equal(receipt.get("spectral_entropy_masking_enabled"), False, "spectral masking")
    a.equal(receipt.get("effective_dim_fraction_override"), None, "effective dimension override")
    a.equal(receipt.get("observed_recurrent_mask_event_count"), 0, "recurrent mask event count")

    rep_idx = receipt.get("technical_replicate_index")
    a.require(rep_idx in (0, 1), "technical_replicate_index must be 0 or 1")
    peer = receipt.get("technical_repeatability_peer_receipt")
    a.require(isinstance(peer, str) and bool(peer), "technical repeatability peer must be identified")

    for field in ("receipt_id", "service_instance_id", "run_id", "base_fixture_id", "transform_id"):
        value = receipt.get(field)
        a.require(isinstance(value, str) and bool(value), f"{field} must be a non-empty string")

    a.equal(receipt.get("seal_stage"), "before_heldout_label_join", "receipt seal stage")
    a.require(
        receipt.get("label_join_stage") in ("after_prediction_seal", "not_joined_yet"),
        "label_join_stage must be after_prediction_seal or not_joined_yet",
    )

    status = receipt.get("measurement_validity")
    reasons = receipt.get("invalidity_reasons")
    a.require(isinstance(reasons, list), "invalidity_reasons must be a list")
    if status == "ELIGIBLE_A0R_MEASUREMENT":
        a.equal(reasons, [], "eligible receipt invalidity reasons")
    else:
        a.require(
            status in contract["validity_rules"]["invalid_statuses"],
            f"unknown measurement validity status: {status!r}",
        )
        a.require(isinstance(reasons, list) and len(reasons) > 0, "invalid receipt must report reason(s)")

    cycle = receipt.get("measurement_cycle_index")
    a.require(isinstance(cycle, int) and not isinstance(cycle, bool) and cycle >= 0, "measurement cycle index")
    a.require(isinstance(receipt.get("cfc_vector_all_zero"), bool), "cfc_vector_all_zero must be bool")
    a.require(
        isinstance(receipt.get("technical_repeatability_match"), bool),
        "technical_repeatability_match must be bool",
    )
    return a.errors


def receipt_audit(contract: dict[str, Any], receipt_path: Path) -> dict[str, Any]:
    receipt = load_json(receipt_path)
    errors = receipt_errors(contract, receipt)
    if errors:
        a = Audit()
        a.extend(receipt_path.name, errors)
        a.finish(f"PARADOX-A0R-RECEIPT:{receipt_path.name}")
    print(f"PARADOX-A0R-RECEIPT:{receipt_path.name}: PASS")
    return receipt


def receipt_set_audit(contract: dict[str, Any], directory: Path) -> None:
    a = Audit()
    files = sorted(p for p in directory.glob("*.json") if p.is_file())
    a.require(bool(files), f"no JSON receipts found in {directory}")
    receipts: list[dict[str, Any]] = []

    for path in files:
        receipt = load_json(path)
        errs = receipt_errors(contract, receipt)
        if errs:
            a.extend(path.name, errs)
        receipt["_audit_filename"] = path.name
        receipts.append(receipt)

    receipt_ids = [r.get("receipt_id") for r in receipts]
    service_ids = [r.get("service_instance_id") for r in receipts]
    if all(isinstance(x, str) for x in receipt_ids):
        a.equal(len(receipt_ids), len(set(receipt_ids)), "receipt_id global uniqueness")
    if all(isinstance(x, str) for x in service_ids):
        a.equal(len(service_ids), len(set(service_ids)), "service_instance_id global uniqueness")

    by_id = {r.get("receipt_id"): r for r in receipts if isinstance(r.get("receipt_id"), str)}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    key_fields = contract["receipt_set_rules"]["group_key"]
    for r in receipts:
        key = tuple(r.get(k) for k in key_fields)
        groups[key].append(r)

    same_fields = contract["receipt_set_rules"]["same_within_pair"]
    bit_fields = contract["receipt_set_rules"]["bit_identical_within_pair"]

    for key, pair in groups.items():
        label = f"group {key!r}"
        a.equal(len(pair), 2, f"{label} replicate count")
        if len(pair) != 2:
            continue
        pair = sorted(pair, key=lambda r: r.get("technical_replicate_index", -1))
        a.equal(
            [r.get("technical_replicate_index") for r in pair],
            [0, 1],
            f"{label} replicate indices",
        )
        r0, r1 = pair
        a.equal(
            r0.get("technical_repeatability_peer_receipt"),
            r1.get("receipt_id"),
            f"{label} rep0 peer binding",
        )
        a.equal(
            r1.get("technical_repeatability_peer_receipt"),
            r0.get("receipt_id"),
            f"{label} rep1 peer binding",
        )
        for field in same_fields:
            a.equal(r0.get(field), r1.get(field), f"{label} same {field}")

        actual_match = all(r0.get(field) == r1.get(field) for field in bit_fields)
        a.equal(r0.get("technical_repeatability_match"), actual_match, f"{label} rep0 repeatability flag")
        a.equal(r1.get("technical_repeatability_match"), actual_match, f"{label} rep1 repeatability flag")
        if not actual_match:
            a.require(
                r0.get("measurement_validity") != "ELIGIBLE_A0R_MEASUREMENT"
                and r1.get("measurement_validity") != "ELIGIBLE_A0R_MEASUREMENT",
                f"{label} nondeterministic pair must be excluded in full",
            )

        eligible = [
            r.get("measurement_validity") == "ELIGIBLE_A0R_MEASUREMENT"
            for r in pair
        ]
        a.require(
            eligible[0] == eligible[1],
            f"{label} pairwise exclusion violated: one replicate eligible and the other invalid",
        )

        for r in pair:
            peer_id = r.get("technical_repeatability_peer_receipt")
            a.require(peer_id in by_id, f"{label} peer receipt not present in set: {peer_id!r}")

    a.finish("PARADOX-A0R-RECEIPT-SET")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--receipt", type=Path, help="validate one concrete A0-R receipt")
    group.add_argument(
        "--receipt-dir",
        type=Path,
        help="validate all JSON receipts in a directory plus cross-receipt pairing/uniqueness invariants",
    )
    args = parser.parse_args()

    contract = static_audit()
    if args.receipt is not None:
        receipt_audit(contract, args.receipt)
    elif args.receipt_dir is not None:
        receipt_set_audit(contract, args.receipt_dir)


if __name__ == "__main__":
    main()
