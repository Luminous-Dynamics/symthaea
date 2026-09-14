// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

const SOURCE: &str = include_str!("../src/lib.rs");

#[test]
fn terminal_transition_requires_the_exact_authorization_execution_and_state_binding_chain() {
    for needle in [
        "execution_permit.authorization_id() != authorization.id()",
        "execution_permit.context_id() != authorization.context().id()",
        "state_binding.execution_permit_id() != execution_permit.id()",
        "authorization.context().handoff_id() != handoff.id()",
        "state_binding.handoff_plan_digest() != handoff.plan_digest()",
    ] {
        assert!(SOURCE.contains(needle), "missing authority-chain ratchet: {needle}");
    }
}

#[test]
fn terminal_record_is_finalized_and_generation_adjacent() {
    assert!(SOURCE.contains("terminal_stage: UpgradeStage::Finalized"));
    assert!(SOURCE.contains("upgrade_state_generation()\n        .checked_add(1)"));
    assert!(SOURCE.contains("predecessor_upgrade_state_digest: state_binding.upgrade_state_digest()"));
    assert!(SOURCE.contains("finalization_sequence: state_binding.handoff_sequence()"));
}

#[test]
fn finalization_binds_successor_artifacts_and_fresh_execution_evidence() {
    for needle in [
        "successor_source_tree_digest",
        "successor_executable_digest",
        "successor_durable_state_digest",
        "successor_replay_contract_digest",
        "fresh_checkpoint_digest: execution_permit.fresh_checkpoint_digest()",
        "fresh_transparency_log_digest: execution_permit.fresh_transparency_log_digest()",
        "fresh_clock_envelope_id: execution_permit.fresh_clock_envelope_id().to_hex()",
        "fresh_operational_basis_id: execution_permit.fresh_operational_basis_id().to_hex()",
        "hardware_refresh_set_digest: execution_permit.hardware_refresh_set_digest()",
    ] {
        assert!(SOURCE.contains(needle), "missing finalization evidence ratchet: {needle}");
    }
}

#[test]
fn terminal_transition_is_deterministic_and_contains_no_legacy_scalar_mutation_path() {
    assert!(SOURCE.contains("FINALIZATION_RECORD_DOMAIN"));
    assert!(SOURCE.contains("FINALIZED_UPGRADE_DOMAIN"));
    assert!(!SOURCE.contains("now_unix_s"));
    assert!(!SOURCE.contains("observed_at_unix_ms"));
    assert!(!SOURCE.contains("authorized_at_unix_ms"));
    assert!(!SOURCE.contains("UpgradeHandoffTracker"));
    assert!(!SOURCE.contains("AuthorizedUpgradeHandoff"));
    assert!(!SOURCE.contains("FabricationUpgradeState::successor"));
    assert!(!SOURCE.contains("saturating_"));
}

#[test]
fn portable_record_is_evidence_but_live_terminal_authority_is_opaque() {
    assert!(SOURCE.contains("pub struct ClockGovernedUpgradeFinalizationRecordV1"));
    assert!(SOURCE.contains("Serialize, Deserialize"));
    let prefix = SOURCE
        .split("pub struct ClockGovernedFinalizedUpgradeV1")
        .next()
        .expect("opaque terminal declaration prefix");
    let derive_window = prefix.rsplit("#[derive(").next().expect("derive window");
    assert!(!derive_window.contains("Deserialize"));
}
