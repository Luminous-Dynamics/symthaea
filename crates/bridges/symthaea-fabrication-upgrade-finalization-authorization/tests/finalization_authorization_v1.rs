// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn prepare_then_authorize_is_explicitly_two_stage() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("prepare_upgrade_finalization_authorization_v1("));
    assert!(source.contains("authorize_clock_governed_upgrade_finalization_v1("));
    assert!(source.contains("PreparedUpgradeFinalizationAuthorizationV1"));
    assert!(source.contains("AuthorizedClockGovernedUpgradeFinalizationV1"));
}

#[test]
fn prepared_payload_binds_context_policy_and_threshold_policy() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("context_id: context.id().to_hex()"));
    assert!(source.contains("context_policy_digest: context.context_policy_digest().to_hex()"));
    assert!(source.contains("threshold_policy_digest: threshold_policy_digest.to_hex()"));
    assert!(source.contains("threshold_floor_digest: threshold_floor_digest.to_hex()"));
    assert!(source.contains("governance_checkpoint_digest: context.governance_checkpoint_digest().to_hex()"));
}

#[test]
fn finalization_threshold_policy_has_a_hard_security_floor() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("minimum_distinct_signers: 2"));
    assert!(source.contains("require_algorithm_diversity: true"));
    assert!(source.contains("threshold_policy.key_usage != KeyUsage::ThresholdCeremony"));
    assert!(source.contains("ThresholdPolicyBelowFloor"));
}

#[test]
fn threshold_policy_digest_matches_interval_bridge_canonical_domain() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("symthaea.fabrication.clock-governed-threshold-policy.v1\\0"));
    for required in [
        "minimum_distinct_signers",
        "maximum_approvals",
        "require_algorithm_diversity",
        "required_algorithms",
        "allowed_key_ids",
        "key_usage",
    ] {
        assert!(source.contains(required), "missing threshold commitment field {required}");
    }
}

#[test]
fn ceremony_must_sign_exact_prepared_payload_under_exact_live_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("ceremony.purpose() != CLOCK_GOVERNED_UPGRADE_FINALIZATION_PURPOSE"));
    assert!(source.contains("ceremony.payload_digest() != prepared.id().as_digest()"));
    assert!(source.contains("ceremony.policy_digest() != prepared.threshold_policy_digest"));
    assert!(source.contains("ceremony.trust_snapshot_digest() != prepared.context.current_trust_snapshot_digest()"));
    assert!(source.contains("ceremony.compromise_tracker_digest() != prepared.context.current_compromise_tracker_digest()"));
    assert!(source.contains("ceremony.clock_envelope_id() != prepared.context.current_clock_envelope_id()"));
}

#[test]
fn ceremony_signer_floor_is_rechecked_not_only_assumed_from_policy() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("ceremony.signers().len() < prepared.minimum_distinct_signers"));
    assert!(source.contains("InsufficientCeremonySigners"));
    assert!(source.contains("prepared.require_algorithm_diversity && algorithms.len() < 2"));
    assert!(source.contains("MissingCeremonyAlgorithmDiversity"));
}

#[test]
fn authorized_result_remains_non_executable_and_opaque() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct AuthorizedClockGovernedUpgradeFinalizationV1"
    ));
    assert!(!source.contains("execute_upgrade_finalization"));
    assert!(!source.contains("retire_predecessor"));
    assert!(!source.contains("FinalizedUpgrade"));
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("authorized_at_unix_ms:"));
}
