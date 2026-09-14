// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn no_rollback_authority_consumes_only_observed_head() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains(
        "observed: &QuorumObservedUpgradeOperationalHeadV1"
    ));
    assert!(!source.contains(
        "use symthaea_fabrication_kernel::upgrade_operational_state"
    ));
    assert!(!source.contains("AutomaticRollbackTrigger"));
    assert!(!source.contains("Option<&"));
}

#[test]
fn durable_rollback_fails_closed_and_preserves_digest() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("observed.automatic_rollback_digest()"));
    assert!(source.contains("DurableRollbackObserved"));
    assert!(source.contains("rollback_digest"));
}

#[test]
fn negative_capability_commits_observation_scope() {
    let source = include_str!("../src/lib.rs");
    for required in [
        "observed_head_id",
        "state_digest",
        "state_generation",
        "handoff_digest",
        "publication_digest",
        "publication_entry_sequence",
        "transparency_log_digest",
        "transparency_log_size",
        "transparency_root_digest",
        "checkpoint_digest",
        "witness_registry_id",
        "witness_registry_sequence",
        "exact_verifier_set_digest",
        "trust_snapshot_digest",
        "containment_state_digest",
        "compromise_tracker_digest",
        "clock_envelope_id",
        "operational_basis_id",
    ] {
        assert!(source.contains(required), "missing commitment field {required}");
    }
}

#[test]
fn no_scalar_time_or_serde_live_construction() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("evaluation_time_unix_s:"));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct WitnessedNoRollbackUpgradeHeadV1"
    ));
}

#[test]
fn theorem_name_remains_witness_scoped() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("WitnessedNoRollbackUpgradeHeadV1"));
    assert!(!source.contains("GlobalNoRollback"));
}
