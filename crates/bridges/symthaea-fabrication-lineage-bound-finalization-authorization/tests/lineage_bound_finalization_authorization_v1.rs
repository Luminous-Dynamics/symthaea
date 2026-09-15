// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn authorization_has_a_distinct_lineage_specific_ceremony_purpose() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("lineage-bound-upgrade-finalization-v1"));
    assert!(source.contains("ceremony.purpose() != LINEAGE_BOUND_UPGRADE_FINALIZATION_PURPOSE"));
    assert!(!source.contains("clock-governed-upgrade-finalization-v1"));
}

#[test]
fn prepared_payload_commits_exact_lineage_and_current_view() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("predecessor_root_digest: context.predecessor_root_digest().to_hex()"));
    assert!(source.contains("predecessor_current_head_digest"));
    assert!(source.contains("predecessor_finalization_sequence"));
    assert!(source.contains("upgrade_cycle_sequence"));
    assert!(source.contains("operational_lineage_digest"));
    assert!(source.contains("hardware_authority_set_digest"));
    assert!(source.contains("current_transparency_log_digest"));
    assert!(source.contains("current_checkpoint_digest"));
}

#[test]
fn threshold_policy_and_security_floor_are_exact() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("minimum_distinct_signers: 2"));
    assert!(source.contains("require_algorithm_diversity: true"));
    assert!(source.contains("KeyUsage::ThresholdCeremony"));
    assert!(source.contains("ThresholdPolicyBelowFloor"));
    assert!(source.contains("symthaea.fabrication.clock-governed-threshold-policy.v1\\0"));
}

#[test]
fn ceremony_must_match_payload_policy_trust_compromise_and_clock() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("ceremony.payload_digest() != prepared.signing_payload_digest()"));
    assert!(source.contains("ceremony.policy_digest() != prepared.threshold_policy_digest"));
    assert!(source.contains("ceremony.trust_snapshot_digest() != prepared.context.current_trust_snapshot_digest()"));
    assert!(source.contains("ceremony.compromise_tracker_digest() != prepared.context.current_compromise_tracker_digest()"));
    assert!(source.contains("ceremony.clock_envelope_id() != prepared.context.current_clock_envelope_id()"));
}

#[test]
fn authorization_is_opaque_and_non_executable() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct AuthorizedLineageBoundFinalizationV1"));
    assert!(!source.contains("Serialize, Deserialize"));
    assert!(!source.contains("finalize_clock_governed_upgrade"));
    assert!(!source.contains("ExecutionPermit"));
    assert!(!source.contains("predecessor retirement"));
}

#[test]
fn ordinary_finalization_context_does_not_reenter() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("QualifiedUpgradeFinalizationContextV1"));
    assert!(!source.contains("AuthorizedClockGovernedUpgradeFinalizationV1"));
    assert!(!source.contains("symthaea_fabrication_upgrade_finalization_context"));
    assert!(!source.contains("symthaea_fabrication_upgrade_finalization_authorization"));
    assert!(!source.contains("now_unix_s"));
}
