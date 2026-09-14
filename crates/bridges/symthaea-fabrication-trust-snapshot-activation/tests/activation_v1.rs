// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn activation_consumes_opaque_rotation_and_recursive_clock_lineage() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("rotation: &ClockGovernedTrustRotationV1"));
    assert!(source.contains("authorization_basis: &OperationalClockBasisV1"));
    assert!(source.contains("clock_bridge: &[OperationalClockBasisV1]"));
    assert!(source.contains("current_basis: &OperationalClockBasisV1"));
    assert!(source.contains("predecessor_operational_basis_id"));
}

#[test]
fn activation_is_certain_across_fresh_interval() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains(
        "rotation.activates_at_unix_ms() > current_clock.lower_unix_ms()"
    ));
    assert!(source.contains("ActivationNotYetCertain"));
    assert!(source.contains("require_valid_across_seconds_window"));
    assert!(source.contains("SnapshotNotValidAcrossCurrentEnvelope"));
}

#[test]
fn activation_rebinds_original_authorization_clock() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("authorization_basis.id() != rotation.operational_basis_id()"));
    assert!(source.contains("authorization_clock.id() != rotation.clock_envelope_id()"));
    assert!(source.contains("AuthorizationBasisMismatch"));
    assert!(source.contains("AuthorizationEnvelopeMismatch"));
}

#[test]
fn no_scalar_time_or_serde_live_construction() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("evaluation_time_unix_s:"));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct ClockGovernedTrustSnapshotActivationPermitV1"
    ));
}

#[test]
fn activation_commits_exact_clock_lineage() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("CLOCK_LINEAGE_DOMAIN"));
    assert!(source.contains("clock_lineage_digest"));
    assert!(source.contains("clock_hop_count"));
    assert!(source.contains("MAX_TRUST_SNAPSHOT_ACTIVATION_CLOCK_HOPS"));
}
