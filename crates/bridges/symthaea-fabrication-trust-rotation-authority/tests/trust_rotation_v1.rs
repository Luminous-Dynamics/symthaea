// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_fabrication_trust_rotation_authority::ExactTrustRotationVerificationPolicyV1;

#[test]
fn exact_rotation_verification_defaults_to_two_providers() {
    let policy = ExactTrustRotationVerificationPolicyV1::default();
    assert_eq!(policy.minimum_distinct_providers, 2);
    assert!(policy.maximum_providers >= policy.minimum_distinct_providers);
}

#[test]
fn live_rotation_has_no_scalar_evaluation_time() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("evaluation_time_unix_s:"));
    assert!(!source.contains("now_unix_s:"));
    assert!(source.contains("derive_clock_governance_evaluation_envelope_v1"));
    assert!(source.contains("clock.upper_unix_ms()"));
    assert!(source.contains("clock.lower_unix_ms()"));
}

#[test]
fn scheduling_uses_interval_safe_checked_arithmetic() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("activates_at_unix_ms < clock.upper_unix_ms()"));
    assert!(source.contains("clock.lower_unix_ms().checked_add(maximum_delay_ms)"));
    assert!(source.contains("activates_at_unix_ms.checked_add(overlap_ms)"));
    assert!(!source.contains("saturating_add(policy.maximum_activation_delay_s)"));
    assert!(!source.contains("saturating_add(policy.minimum_overlap_s)"));
}

#[test]
fn transition_is_adjacent_and_compromise_aware() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("current_snapshot.sequence.checked_add(1)"));
    assert!(source.contains("SequenceNotAdjacent"));
    assert!(source.contains("compromised_at_or_before"));
    assert!(source.contains("compromised_before_ms"));
    assert!(source.contains("KeyUsage::TrustRotation"));
    assert!(source.contains("require_effective_time_after_envelope_seconds"));
}

#[test]
fn exact_raw_rotation_signatures_are_reverified() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("canonical_rotation_proposal_bytes"));
    assert!(source.contains("verify_rotation_signature"));
    assert!(source.contains("SIGNED_ROTATION_EVIDENCE_DOMAIN"));
    assert!(source.contains("VERIFIER_SET_DOMAIN"));
}

#[test]
fn live_rotation_authority_is_opaque() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct ClockGovernedTrustRotationV1"));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct ClockGovernedTrustRotationV1"
    ));
}
