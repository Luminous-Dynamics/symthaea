#!/usr/bin/env python3
"""Static audit for the PARADOX A0-R inferential contract.

Development-only. This validates preregistered statistical structure and does
not execute cognition, fit probes, inspect held-out labels, or score behavior.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED_SCHEMA = "PARADOX-A0R-STATISTICS-V1"
EXPECTED_LATENT_SCHEMA = "PARADOX-A0R-LATENT-READOUT-CONTRACT-V2"
EXPECTED_PRODUCTION = "eb73527d05a913e79d1f05135ad6b06c1da8e2ee"
EXPECTED_G2B = "09d83a1d1fddbbbd30e4eba7cc95946c8eab871f"
EXPECTED_GROUP_KEYS = {
    "base_fixture_id",
    "semantic_fixture_family",
    "proposition_identity",
    "context_identity",
    "source_identity",
}
EXPECTED_CONTROLS = {"perception_hdv_binary", "perception_chunk32"}


def fail(message: str) -> None:
    raise SystemExit(f"A0-R statistics audit failed: {message}")


def load_json(name: str) -> dict:
    path = Path(__file__).with_name(name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot load {name}: {exc}")
    if not isinstance(value, dict):
        fail(f"{name} root must be an object")
    return value


def main() -> int:
    stats = load_json("a0r_statistics.json")
    latent = load_json("a0r_latent_contract.json")

    if stats.get("schema_version") != EXPECTED_SCHEMA:
        fail("unexpected statistics schema")
    if stats.get("latent_contract_schema") != EXPECTED_LATENT_SCHEMA:
        fail("latent contract schema binding drift")
    if latent.get("schema_version") != EXPECTED_LATENT_SCHEMA:
        fail("loaded latent contract does not match statistics binding")
    for data, label in ((stats, "statistics"), (latent, "latent")):
        if data.get("production_subject_sha") != EXPECTED_PRODUCTION:
            fail(f"{label}: production subject drift")
        if data.get("g2b_subject_sha") != EXPECTED_G2B:
            fail(f"{label}: G2b subject drift")
        if data.get("confirmatory_execution_allowed") is not False:
            fail(f"{label}: confirmatory execution must remain disabled")

    unit = stats.get("inferential_unit")
    if not isinstance(unit, dict):
        fail("inferential_unit missing")
    if unit.get("unit") != "independent base semantic fixture":
        fail("base semantic fixture must remain the inferential unit")
    if unit.get("metamorphic_variants_count_as_independent") is not False:
        fail("metamorphic variants must not inflate n")
    if unit.get("multiple_cycles_count_as_independent") is not False:
        fail("cycle-level pseudo-replication must remain forbidden")
    if "single preregistered final" not in unit.get("primary_cycle", ""):
        fail("primary endpoint must remain one final relational-state cycle")
    if "never increases n" not in unit.get("temporal_negative_control", ""):
        fail("temporal control must remain paired, not independent")
    cluster_rule = unit.get("cluster_rule", "")
    for word in ("variants", "cycles", "channels", "controls", "same split"):
        if word not in cluster_rule:
            fail(f"cluster rule missing {word}")

    mins = stats.get("minimum_independent_groups")
    if not isinstance(mins, dict):
        fail("minimum_independent_groups missing")
    if mins.get("frontier_atom_positive") != 64 or mins.get("frontier_atom_negative") != 64:
        fail("frontier minimum must remain 64 independent groups per class")
    if mins.get("positive_control_positive") != 32 or mins.get("positive_control_negative") != 32:
        fail("positive-control minimum must remain 32 groups per class")
    if "NotEstablished" not in mins.get("shortfall_rule", ""):
        fail("sample shortfall must fail closed")

    split = stats.get("splitting")
    if not isinstance(split, dict):
        fail("splitting missing")
    if split.get("outer_development_holdout_fraction") != 0.25:
        fail("outer development holdout drift")
    if split.get("nested_cv_folds") != 5:
        fail("nested CV fold count drift")
    if set(split.get("group_keys", [])) != EXPECTED_GROUP_KEYS:
        fail("group key set drift")
    if split.get("same_partitions_for_all_channels") is not True:
        fail("channels must use paired partitions")
    if split.get("same_partitions_for_all_probe_families") is not True:
        fail("probe families must use paired partitions")

    prep = stats.get("preprocessing")
    if not isinstance(prep, dict):
        fail("preprocessing missing")
    if "training-fold" not in prep.get("primary_linear_probe", ""):
        fail("normalization must be training-fold-only")
    if prep.get("learned_feature_reduction") != "forbidden in V1":
        fail("learned feature reduction must remain forbidden")
    if "no synthetic oversampling" not in prep.get("class_rebalancing", ""):
        fail("synthetic oversampling prohibition missing")

    probe = stats.get("linear_probe")
    if not isinstance(probe, dict):
        fail("linear_probe missing")
    if probe.get("family") != "L2 logistic regression":
        fail("probe family drift")
    grid = probe.get("regularization_grid")
    if grid != [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        fail("regularization grid drift")
    if probe.get("tie_break") != "choose stronger regularization":
        fail("regularization tie-break drift")
    if "freeze coefficients" not in probe.get("finalization", ""):
        fail("final probe parameters must freeze before held-out evaluation")

    resampling = stats.get("resampling")
    if not isinstance(resampling, dict):
        fail("resampling missing")
    if resampling.get("bootstrap_replicates") != 10000:
        fail("bootstrap replicate count drift")
    if resampling.get("bootstrap_unit") != "base fixture cluster":
        fail("bootstrap must cluster on base fixture")
    if "paired" not in resampling.get("bootstrap_pairing", ""):
        fail("channel comparison bootstrap must remain paired")
    if resampling.get("chance_permutations") != 10000:
        fail("chance permutation count drift")
    if "grouped label-permutation" not in resampling.get("chance_test", ""):
        fail("chance test must remain grouped label permutation")
    if "better control is selected on development-training" not in resampling.get("surface_control_selection", ""):
        fail("surface control selection must not use held-out labels")

    multiplicity = stats.get("multiplicity")
    if not isinstance(multiplicity, dict):
        fail("multiplicity missing")
    if multiplicity.get("frontier_atom_count") != 6:
        fail("frontier multiplicity count drift")
    if multiplicity.get("familywise_alpha") != 0.05:
        fail("familywise alpha drift")
    if multiplicity.get("method") != "Holm step-down":
        fail("multiplicity method drift")
    if multiplicity.get("positive_control_in_family") is not False:
        fail("positive control must remain outside frontier multiplicity family")
    if "Holm-adjusted" not in multiplicity.get("rule", ""):
        fail("adjusted p-value acceptance gate missing")

    metrics = stats.get("metrics")
    if not isinstance(metrics, dict):
        fail("metrics missing")
    if metrics.get("primary") != "balanced_accuracy":
        fail("primary metric drift")
    if metrics.get("no_omnibus") is not True:
        fail("omnibus score must remain forbidden")
    if metrics.get("no_best_seed_reporting") is not True:
        fail("best-seed reporting must remain forbidden")
    if metrics.get("report_all_atoms") is not True or metrics.get("report_all_channels") is not True:
        fail("complete reporting requirements missing")

    paired = stats.get("paired_surface_comparison")
    if not isinstance(paired, dict):
        fail("paired_surface_comparison missing")
    if set(paired.get("controls", [])) != EXPECTED_CONTROLS:
        fail("perceptual-control set drift")
    if paired.get("primary") != "cfc_recurrent_state":
        fail("primary channel drift")
    for field in ("same_base_fixtures", "same_outer_holdout", "same_label_manifest"):
        if paired.get(field) is not True:
            fail(f"paired surface comparison requires {field}")
    if "NotEstablished" not in paired.get("claim_rule", ""):
        fail("surface-abstraction claim must fail closed")

    stopping = stats.get("stopping")
    if not isinstance(stopping, dict):
        fail("stopping missing")
    for field in ("adaptive_stopping", "interim_peeking", "post_hoc_sample_extension_after_seeing_target_results"):
        if stopping.get(field) is not False:
            fail(f"{field} must remain disabled")
    if "before any held-out development labels are opened" not in stopping.get("allowed_extension", ""):
        fail("sample extension amendment boundary missing")

    print(
        "PARADOX A0-R statistics audit PASS "
        f"frontier_atoms={multiplicity['frontier_atom_count']} "
        f"bootstrap={resampling['bootstrap_replicates']} "
        f"permutations={resampling['chance_permutations']} "
        f"production={EXPECTED_PRODUCTION} g2b={EXPECTED_G2B}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
