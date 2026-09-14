#!/usr/bin/env python3
"""Validate and freeze the EHK beta=6.0 historical-fidelity ledger (LQCD-021K)."""

import hashlib
import json
from pathlib import Path

ORACLE_ID = "lqcd_historical_fidelity_validator_v1"
LEDGER_REL = Path("docs/release/evidence/LQCD_021K_EHK_BETA6_HISTORICAL_FIDELITY.json")

ALLOWED_CLASSES = {
    "ExactlySpecified",
    "DerivedFromSpecifiedQuantity",
    "ApproximatelySpecified",
    "HistoricallyUnderdetermined",
    "DeclaredReproductionConvention",
    "IndependentlyReconstructed",
    "Unavailable",
    "NotApplicable",
}

EXPECTED = {
    "action": "ExactlySpecified",
    "beta": "ExactlySpecified",
    "lattice_geometry": "ExactlySpecified",
    "configuration_count": "ExactlySpecified",
    "or_to_heatbath_ratio": "ApproximatelySpecified",
    "ape_epsilon_times_n": "ApproximatelySpecified",
    "burn_in_cycles": "HistoricallyUnderdetermined",
    "measurement_stride": "HistoricallyUnderdetermined",
    "rng_algorithm": "HistoricallyUnderdetermined",
    "rng_seed_values": "HistoricallyUnderdetermined",
    "off_axis_transporter_path": "HistoricallyUnderdetermined",
    "su3_smearing_projection_algorithm": "HistoricallyUnderdetermined",
    "ape_alpha_exact": "HistoricallyUnderdetermined",
    "ape_iteration_count_exact": "HistoricallyUnderdetermined",
    "best_few_fit_weighting": "HistoricallyUnderdetermined",
    "published_scale_joint_covariance": "Unavailable",
    "symthaea_ape_alpha": "DeclaredReproductionConvention",
    "symthaea_ape_iterations": "DeclaredReproductionConvention",
    "symthaea_ape_projection": "DeclaredReproductionConvention",
    "symthaea_off_axis_transporter": "DeclaredReproductionConvention",
    "symthaea_target_isolation": "DeclaredReproductionConvention",
    "derived_ape_epsilon_proxy": "DerivedFromSpecifiedQuantity",
}

DIRECT_BENCHMARK_FIELDS = (
    "benchmark_a_sqrt_sigma",
    "benchmark_r0_over_a",
    "benchmark_r4_over_a",
    "benchmark_r6_over_a",
)

EXACT_HISTORICAL_IMPLEMENTATION_FIELDS = (
    "burn_in_cycles",
    "measurement_stride",
    "rng_algorithm",
    "rng_seed_values",
    "independent_chain_count",
    "checkpoint_restart_semantics",
    "off_axis_transporter_path",
    "su3_smearing_projection_algorithm",
    "ape_alpha_exact",
    "ape_iteration_count_exact",
    "final_plateau_windows",
    "final_r_fit_ranges",
    "best_few_fit_weighting",
    "binary_gauge_configuration_format",
)


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def fail(message):
    raise AssertionError(message)


def validate_entry(entry):
    required = {"field_id", "semantic_scope", "fidelity_class"}
    if not required.issubset(entry):
        fail("missing required entry field")
    klass = entry["fidelity_class"]
    if klass not in ALLOWED_CLASSES:
        fail("unknown fidelity class: " + str(klass))

    scope = entry["semantic_scope"]
    if scope == "historical":
        if "source_locator" not in entry:
            fail(entry["field_id"] + ": historical entry missing source_locator")
        if klass == "DeclaredReproductionConvention":
            fail(entry["field_id"] + ": historical entry cannot declare Symthaea convention")
    elif scope == "symthaea_convention":
        if klass != "DeclaredReproductionConvention":
            fail(entry["field_id"] + ": Symthaea convention must use declared-convention class")
        if "symthaea_subject" not in entry:
            fail(entry["field_id"] + ": convention missing Symthaea subject")
    elif scope == "derived":
        if klass != "DerivedFromSpecifiedQuantity":
            fail(entry["field_id"] + ": derived scope/class mismatch")
        if "derivation_subject" not in entry:
            fail(entry["field_id"] + ": derived entry missing derivation subject")
    else:
        fail(entry["field_id"] + ": unknown semantic scope")

    if klass in {"ExactlySpecified", "ApproximatelySpecified"} and "normalized_value" not in entry:
        fail(entry["field_id"] + ": sourced value missing normalized_value")
    if klass == "HistoricallyUnderdetermined" and "normalized_value" in entry:
        fail(entry["field_id"] + ": underdetermined historical field invented a value")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    ledger_path = repo_root / LEDGER_REL
    raw = ledger_path.read_bytes()
    ledger = json.loads(raw)

    if ledger.get("schema") != "lqcd_historical_fidelity_ledger_v1":
        fail("wrong ledger schema")
    if ledger.get("campaign") != "EHK_beta6_static_potential_reproduction":
        fail("wrong campaign")
    if ledger.get("historical_source", {}).get("source_id") != "arxiv:hep-lat/9711003":
        fail("wrong historical source")

    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        fail("entries missing")
    ids = [entry.get("field_id") for entry in entries]
    if len(ids) != len(set(ids)):
        fail("duplicate field_id")
    by_id = {entry["field_id"]: entry for entry in entries}

    for entry in entries:
        validate_entry(entry)

    for field_id, klass in EXPECTED.items():
        if field_id not in by_id:
            fail("missing expected field: " + field_id)
        if by_id[field_id]["fidelity_class"] != klass:
            fail(field_id + ": unexpected fidelity class")

    for field_id in DIRECT_BENCHMARK_FIELDS:
        if by_id[field_id]["fidelity_class"] != "ExactlySpecified":
            fail(field_id + ": benchmark coordinate not exactly sourced")

    exact_historical_implementation_allowed = all(
        by_id[field_id]["fidelity_class"]
        in {"ExactlySpecified", "IndependentlyReconstructed"}
        for field_id in EXACT_HISTORICAL_IMPLEMENTATION_FIELDS
    )
    if exact_historical_implementation_allowed:
        fail("negative control failed: ledger incorrectly permits exact historical identity")

    declared_convention_fields = {
        entry["field_id"]
        for entry in entries
        if entry["fidelity_class"] == "DeclaredReproductionConvention"
    }
    required_conventions = {
        "symthaea_ape_alpha",
        "symthaea_ape_iterations",
        "symthaea_ape_projection",
        "symthaea_off_axis_transporter",
        "symthaea_target_isolation",
        "symthaea_schedule_selection",
        "symthaea_rng_namespace",
        "symthaea_gauge_field_encoding",
    }
    if not required_conventions.issubset(declared_convention_fields):
        fail("required declared reproduction conventions incomplete")

    authoritative = {
        "schema": ledger["schema"],
        "campaign": ledger["campaign"],
        "historical_source": ledger["historical_source"],
        "entries": sorted(entries, key=lambda entry: entry["field_id"]),
    }
    authoritative_digest = hashlib.sha256(canonical_bytes(authoritative)).hexdigest()
    file_digest = hashlib.sha256(raw).hexdigest()

    result = {
        "oracle_id": ORACLE_ID,
        "schema": ledger["schema"],
        "entry_count": len(entries),
        "class_counts": {
            klass: sum(entry["fidelity_class"] == klass for entry in entries)
            for klass in sorted(ALLOWED_CLASSES)
        },
        "ledger_file_sha256": file_digest,
        "authoritative_ledger_sha256": authoritative_digest,
        "direct_benchmark_coordinates_exactly_sourced": True,
        "exact_historical_implementation_claim_allowed": False,
        "future_claim_template": (
            "numerical reproduction of the published EHK beta=6.0 16^3x32 "
            "benchmark under declared Symthaea conventions for historically "
            "underdetermined implementation details"
        ),
        "future_claim_template_requires_numerical_result": True,
    }
    text = json.dumps(result, sort_keys=True, separators=(",", ":"))
    print("ok")
    print("result_sha256=" + hashlib.sha256(text.encode()).hexdigest())
    print(text)


if __name__ == "__main__":
    main()
