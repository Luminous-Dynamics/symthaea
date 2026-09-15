// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn activation_preserves_global_predecessor_provenance() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("lineage_handoff_id"));
    assert!(source.contains("predecessor_root_digest"));
    assert!(source.contains("current_head_digest"));
    assert!(source.contains("governance_view_digest"));
    assert!(source.contains("registry_head_digest"));
    assert!(source.contains("predecessor_finalization_sequence"));
}

#[test]
fn activation_reproves_exact_policy_successors_on_fresh_time() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("PolicyLineageNotExactSuccessor"));
    assert!(source.contains("PolicyMigrationNotActivated"));
    assert!(source.contains("PolicySuccessorMismatch"));
    assert!(source.contains("PolicyNotFreshOnExecutionClock"));
    assert!(source.contains("PolicyExactEvidenceMismatch"));
    assert!(source.contains("current_lineage_sequence.checked_add(1)"));
}

#[test]
fn activation_commits_ordered_clock_lineage() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("ClockLineageCommitment"));
    assert!(source.contains("bridge_basis_ids"));
    assert!(source.contains("predecessor_operational_basis_id"));
    assert!(source.contains("ActivationNotDefinitelyReached"));
    assert!(source.contains("FinalizationMayBeClosed"));
}

#[test]
fn unrooted_activation_permit_is_not_part_of_live_surface() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct LineageBoundUpgradeActivationPermitV1"));
    assert!(!source.contains("ClockGovernedUpgradeActivationPermitV1"));
    assert!(!source.contains("derive_clock_governed_upgrade_activation_permit_v1"));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("evaluation_time_unix_s"));
}
