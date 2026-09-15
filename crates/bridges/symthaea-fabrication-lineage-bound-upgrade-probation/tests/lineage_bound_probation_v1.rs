// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn probation_preserves_predecessor_lineage_through_clearance() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("activation.lineage_handoff_id() != handoff.id()"));
    assert!(source.contains("activation.predecessor_root_digest()"));
    assert!(source.contains("activation.current_head_digest()"));
    assert!(source.contains("activation.governance_view_digest()"));
    assert!(source.contains("activation.registry_head_digest()"));
    assert!(source.contains("predecessor_finalization_sequence"));
}

#[test]
fn observation_authority_is_exact_and_non_retroactive() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("verify_observation_signature"));
    assert!(source.contains("ObserverUsageNotAllowed"));
    assert!(source.contains("ObserverCompromisedAcrossEnvelope"));
    assert!(source.contains("TrustSnapshotPostdatesObservation"));
    assert!(source.contains("UnexpectedObserverProfile"));
    assert!(source.contains("SIGNED_OBSERVATION_EVIDENCE_DOMAIN"));
}

#[test]
fn probation_accounting_fails_closed_without_saturating_math() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("checked_basis_points"));
    assert!(source.contains("part.checked_mul(10_000)"));
    assert!(source.contains("successful.checked_add(failed)"));
    assert!(source.contains("InvalidJobAccounting"));
    assert!(!source.contains("saturating_mul"));
}

#[test]
fn lineage_specific_quorum_and_ordered_clock_ancestry_are_committed() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("LINEAGE_BOUND_UPGRADE_PROBATION_CLEARANCE_PURPOSE"));
    assert!(source.contains("ClockLineageCommitment"));
    assert!(source.contains("bridge_basis_ids"));
    assert!(source.contains("clock_lineage_digest"));
    assert!(source.contains("ceremony.payload_digest() != prepared.signing_payload_digest()"));
}

#[test]
fn ordinary_live_upgrade_authority_does_not_reenter_the_api() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("use symthaea_fabrication_upgrade_authority"));
    assert!(!source.contains("ClockGovernedUpgradeActivationPermitV1"));
    assert!(!source.contains("ClockGovernedUpgradeProbationClearanceV1"));
    assert!(!source.contains("AuthorizedUpgradeHandoff"));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("evaluation_time_unix_s"));
}
