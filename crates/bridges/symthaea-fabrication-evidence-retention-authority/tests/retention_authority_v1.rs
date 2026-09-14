// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_fabrication_kernel::trust::KeyUsage;

#[test]
fn live_retention_authority_excludes_scalar_and_legacy_authorization() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("evaluated_at_unix_s:"));
    assert!(!source.contains("AuthorizedEvidenceRetentionPolicy"));
    assert!(!source.contains("VerifiedThresholdCeremony"));
    assert!(source.contains("ClockGovernedThresholdCeremonyV1"));
    assert!(source.contains("KeyUsage::EvidenceRetention"));
    assert!(source.contains("derive_clock_governance_evaluation_envelope_v1"));
}

#[test]
fn policy_effectivity_uses_trusted_lower_bound() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("policy.effective_at_unix_s.checked_mul(1_000)"));
    assert!(source.contains("effective_ms <= clock.lower_unix_ms()"));
    assert!(source.contains("PolicyMayNotBeEffective"));
    assert!(source.contains("PolicyEffectiveTimeOverflow"));
}

#[test]
fn policy_sequence_and_full_context_are_committed() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("policy_sequence: u64"));
    assert!(source.contains("containment_generation: u64"));
    assert!(source.contains("compromise_tracker_digest"));
    assert!(source.contains("operational_basis_id"));
    assert!(source.contains("CLOCK_GOVERNED_EVIDENCE_RETENTION_PURPOSE"));
}

#[test]
fn live_authority_has_no_serde_construction_path() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct PreparedClockGovernedEvidenceRetentionPolicyV1"
    ));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct ClockGovernedEvidenceRetentionPolicyV1"
    ));
}

#[test]
fn dedicated_retention_usage_remains_available() {
    assert_eq!(KeyUsage::EvidenceRetention, KeyUsage::EvidenceRetention);
}
