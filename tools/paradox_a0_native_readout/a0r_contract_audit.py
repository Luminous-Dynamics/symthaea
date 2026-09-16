#!/usr/bin/env python3
"""Static audit for the PARADOX A0-R latent-readout contract.

Development-only. This validates the frozen measurement contract; it does not
execute Symthaea cognition, fit a probe, inspect confirmatory fixtures, or score
behavior.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EXPECTED_SCHEMA = "PARADOX-A0R-LATENT-READOUT-CONTRACT-V2"
EXPECTED_PRODUCTION = "eb73527d05a913e79d1f05135ad6b06c1da8e2ee"
EXPECTED_G2B = "09d83a1d1fddbbbd30e4eba7cc95946c8eab871f"
SHA40 = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_CHANNELS = {
    "cfc_recurrent_state": "primary_latent",
    "perception_hdv_binary": "surface_control",
    "perception_chunk32": "surface_control",
    "typed_runtime_telemetry": "structured_control",
}
EXPECTED_TARGET_ATOMS = {
    "conflict_positive_control": "detect_conflict",
    "visible_context_binding": "bind_visible_context",
    "context_conditioned_polarity": "select_context_conditioned_polarity",
    "opposed_evidence_retention": "preserve_opposed_evidence",
    "independent_provenance": "recognize_independent_provenance",
    "self_causal_relation": "detect_self_causal_relation",
    "representation_inadequacy": "detect_representation_inadequacy",
}
EXPECTED_FRONTIER_ATOMS = set(EXPECTED_TARGET_ATOMS.values()) - {"detect_conflict"}
REQUIRED_FORBIDDEN_INPUTS = {
    "raw_fixture_bytes",
    "canonical_fixture_bytes",
    "raw_input_text",
    "input_byte_length",
    "token_count",
    "condition_id",
    "condition_name",
    "trial_index",
    "confirmatory_seed",
    "expected_response",
    "oracle_output",
    "score",
    "fixture_hash_as_feature",
    "source_file_path_as_feature",
}
REQUIRED_METAMORPHICS = {
    "paraphrase_invariance",
    "evidence_order_invariance",
    "source_rename_invariance",
    "context_rename_invariance",
    "polarity_swap_invariance",
    "polarity_swap_equivariance",
    "self_reference_sham_specificity",
    "uncertainty_vs_model_inadequacy_specificity",
}
EXPECTED_TELEMETRY_FIELDS = {
    "epistemic_conflict_count",
    "feedback_conflict_ratio",
    "epistemic_reasoning_accelerated",
    "metacognitive_anomaly",
    "predictive_self_safety",
    "predictive_behavioral_error",
    "self_model_accuracy",
}


def fail(message: str) -> None:
    raise SystemExit(f"A0-R contract audit failed: {message}")


def load_contract() -> dict:
    path = Path(__file__).with_name("a0r_latent_contract.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot load contract: {exc}")
    if not isinstance(value, dict):
        fail("contract root must be an object")
    return value


def require_source_bindings(channel_id: str, channel: dict) -> None:
    bindings = channel.get("source_bindings")
    if not isinstance(bindings, list) or not bindings:
        fail(f"{channel_id}: source_bindings must be non-empty")
    for binding in bindings:
        if not isinstance(binding, dict):
            fail(f"{channel_id}: malformed source binding")
        path = binding.get("path")
        sha = binding.get("blob_sha")
        fact = binding.get("fact")
        if not isinstance(path, str) or not path:
            fail(f"{channel_id}: binding path missing")
        if not isinstance(sha, str) or not SHA40.fullmatch(sha):
            fail(f"{channel_id}: invalid binding blob SHA")
        if not isinstance(fact, str) or not fact:
            fail(f"{channel_id}: binding fact missing")


def main() -> int:
    data = load_contract()

    if data.get("schema_version") != EXPECTED_SCHEMA:
        fail("unexpected schema version")
    if data.get("production_subject_sha") != EXPECTED_PRODUCTION:
        fail("production subject drift")
    if data.get("g2b_subject_sha") != EXPECTED_G2B:
        fail("G2b subject drift")
    if data.get("confirmatory_execution_allowed") is not False:
        fail("contract must not authorize confirmatory execution")
    if data.get("response_mapping_authorized") is not False:
        fail("contract must not authorize behavioral response mapping")
    if data.get("no_omnibus_score") is not True:
        fail("per-atom reporting/no-omnibus rule must remain enabled")

    state = data.get("state_scope")
    if not isinstance(state, dict):
        fail("state_scope must be an object")
    if state.get("cross_fixture_state") != "forbidden":
        fail("cross-fixture state must be forbidden")
    paired_rule = state.get("paired_transform_rule", "")
    if "rerun production" not in paired_rule or "never transform a latent vector" not in paired_rule:
        fail("metamorphic pairs must be regenerated through production")
    if "held-out label" not in state.get("receipt_requirement", ""):
        fail("receipt rule must seal predictions before held-out labels")

    channels = data.get("channels")
    if not isinstance(channels, dict) or set(channels) != set(EXPECTED_CHANNELS):
        fail("channel set drift")
    for cid, expected_role in EXPECTED_CHANNELS.items():
        channel = channels[cid]
        if channel.get("role") != expected_role:
            fail(f"{cid}: unexpected role")
        require_source_bindings(cid, channel)

    primary = channels["cfc_recurrent_state"]
    if primary.get("receipt") != "CycleResult.output":
        fail("primary latent must remain CycleResult.output")
    if primary.get("eligible_for_abstraction_claim") is not True:
        fail("primary latent must be abstraction-eligible")
    primary_paths = {b["path"] for b in primary["source_bindings"]}
    required_primary_paths = {
        "src/cognitive_loop/phase_results.rs",
        "src/cognitive_loop/cycle_phase_dynamics/mod.rs",
        "src/cognitive_loop/cycle_phase_dynamics/planning.rs",
        "src/cognitive_loop/cycle_phase_output/mod.rs",
    }
    if not required_primary_paths.issubset(primary_paths):
        fail("primary CfC provenance chain is incomplete")

    wisdom = channels["perception_hdv_binary"]
    if wisdom.get("receipt") != "CycleResult.wisdom_hv":
        fail("binary perception control receipt drift")
    if wisdom.get("eligible_for_abstraction_claim") is not False:
        fail("wisdom_hv must remain a surface control")
    if "cached binary perception HDC encoding" not in wisdom.get("semantics", ""):
        fail("wisdom_hv legacy-name correction missing")

    thought = channels["perception_chunk32"]
    if thought.get("receipt") != "CycleResult.thought_vector":
        fail("32D perception control receipt drift")
    if thought.get("eligible_for_abstraction_claim") is not False:
        fail("thought_vector must remain a surface control")
    if "chunk-average" not in thought.get("semantics", ""):
        fail("thought_vector perceptual construction must remain explicit")

    telemetry = channels["typed_runtime_telemetry"]
    if set(telemetry.get("allowed_fields", [])) != EXPECTED_TELEMETRY_FIELDS:
        fail("typed telemetry whitelist drift")
    if telemetry.get("eligible_for_abstraction_claim") is not False:
        fail("typed telemetry cannot establish latent abstraction")

    forbidden = set(data.get("forbidden_probe_inputs", []))
    missing_forbidden = REQUIRED_FORBIDDEN_INPUTS - forbidden
    if missing_forbidden:
        fail(f"missing forbidden inputs {sorted(missing_forbidden)}")

    label_authority = data.get("label_authority")
    if not isinstance(label_authority, dict):
        fail("label_authority must be an object")
    if "development-only" not in label_authority.get("source", ""):
        fail("labels must come from a development-only semantic manifest")
    if "never probe features" not in label_authority.get("feature_boundary", ""):
        fail("label metadata must be excluded from probe features")
    if "predictions are sealed" not in label_authority.get("training_rule", ""):
        fail("held-out development labels must follow sealed predictions")
    if "confirmatory labels" not in label_authority.get("future_confirmatory_rule", ""):
        fail("future confirmatory label boundary missing")
    if "forbidden as probe features" not in label_authority.get("condition_identifier_rule", ""):
        fail("condition identifier feature firewall missing")

    targets = data.get("targets")
    if not isinstance(targets, list):
        fail("targets must be a list")
    if len(targets) != len(EXPECTED_TARGET_ATOMS):
        fail("target count drift")
    target_by_id: dict[str, dict] = {}
    seen_atoms: set[str] = set()
    for target in targets:
        if not isinstance(target, dict):
            fail("malformed target")
        tid = target.get("id")
        if tid not in EXPECTED_TARGET_ATOMS or tid in target_by_id:
            fail(f"unexpected or duplicate target {tid}")
        atom = target.get("capability_atom")
        if atom != EXPECTED_TARGET_ATOMS[tid]:
            fail(f"{tid}: capability atom drift")
        if atom in seen_atoms:
            fail(f"{tid}: each atom must have its own probe")
        seen_atoms.add(atom)
        if target.get("role") not in {"positive_control", "frontier"}:
            fail(f"{tid}: invalid role")
        if not isinstance(target.get("label_contract"), str) or not target["label_contract"]:
            fail(f"{tid}: label contract missing")
        target_by_id[tid] = target
    if set(target_by_id) != set(EXPECTED_TARGET_ATOMS):
        fail("target set drift")
    frontier_atoms = {
        t["capability_atom"] for t in targets if t.get("role") == "frontier"
    }
    if frontier_atoms != EXPECTED_FRONTIER_ATOMS:
        fail("frontier capability-atom set drift")
    if target_by_id["conflict_positive_control"].get("role") != "positive_control":
        fail("detect_conflict must remain a positive control")

    # Critical anti-shortcut pairings are part of the construct, not prose advice.
    if "same final evidence item" not in target_by_id["opposed_evidence_retention"]["label_contract"]:
        fail("opposed-evidence target must match the final perceptual item")
    if "duplicate-source" not in target_by_id["independent_provenance"]["label_contract"]:
        fail("independent-provenance target must include pseudo-disagreement control")
    if "causal edge is severed" not in target_by_id["self_causal_relation"]["label_contract"]:
        fail("self-causal target must include a sham causal-edge control")
    if "matched uncertainty" not in target_by_id["representation_inadequacy"]["label_contract"]:
        fail("representation-inadequacy target must control generic uncertainty")

    probe = data.get("probe_family")
    if not isinstance(probe, dict):
        fail("probe_family must be an object")
    if "linear" not in probe.get("primary", "").lower():
        fail("primary probe must remain linear")
    if "one binary probe per capability atom" not in probe.get("primary", ""):
        fail("probe granularity must remain atom-level")
    forbidden_probe_types = " ".join(probe.get("forbidden", [])).lower()
    for phrase in ("theorem", "condition-specific", "fixture-id", "post-confirmatory"):
        if phrase not in forbidden_probe_types:
            fail(f"missing probe-family prohibition: {phrase}")

    split = data.get("development_split")
    if not isinstance(split, dict):
        fail("development_split must be an object")
    grouping = set(split.get("grouping_keys", []))
    if grouping != {
        "semantic_fixture_family",
        "proposition_identity",
        "context_identity",
        "source_identity",
    }:
        fail("development grouping keys drift")

    controls = data.get("metamorphic_controls")
    if not isinstance(controls, list):
        fail("metamorphic_controls must be a list")
    control_ids = {c.get("id") for c in controls if isinstance(c, dict)}
    if control_ids != REQUIRED_METAMORPHICS or len(controls) != len(REQUIRED_METAMORPHICS):
        fail("metamorphic control set drift")
    for control in controls:
        if control.get("kind") not in {"invariance", "equivariance", "specificity"}:
            fail(f"{control.get('id')}: invalid metamorphic kind")
        applies = set(control.get("applies_to", []))
        if not applies or not applies.issubset(EXPECTED_TARGET_ATOMS):
            fail(f"{control.get('id')}: invalid target coverage")
    polarity_eq = next(c for c in controls if c["id"] == "polarity_swap_equivariance")
    if polarity_eq.get("applies_to") != ["context_conditioned_polarity"]:
        fail("directional polarity equivariance must apply only to the directional target")

    negatives = set(data.get("negative_controls", []))
    for required in ("perception_hdv_binary", "perception_chunk32", "label_permutation"):
        if required not in negatives:
            fail(f"missing negative control {required}")

    acceptance = data.get("acceptance")
    if not isinstance(acceptance, dict):
        fail("acceptance must be an object")
    if acceptance.get("unit") != "each capability atom independently":
        fail("acceptance unit must remain atom-level")
    if acceptance.get("primary_channel") != "cfc_recurrent_state":
        fail("primary acceptance channel drift")
    for key in (
        "chance_requirement",
        "surface_abstraction_requirement",
        "metamorphic_requirement",
        "label_permutation_requirement",
        "failure_rule",
        "promotion_rule",
    ):
        if not isinstance(acceptance.get(key), str) or not acceptance[key]:
            fail(f"acceptance.{key} missing")
    if "0.50" not in acceptance["chance_requirement"]:
        fail("binary chance threshold drift")
    if "0.95" not in acceptance["metamorphic_requirement"] or "0.90" not in acceptance["metamorphic_requirement"]:
        fail("metamorphic thresholds drift")
    if "do not average atoms" not in acceptance["failure_rule"]:
        fail("failure rule must forbid grouped/omnibus rescue")

    print(
        "PARADOX A0-R latent contract audit PASS "
        f"channels={len(channels)} atoms={len(targets)} controls={len(controls)} "
        f"production={EXPECTED_PRODUCTION} g2b={EXPECTED_G2B}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
