// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_fabrication_upgrade_operational_head::ExactUpgradeOperationalHeadVerificationPolicyV1;

#[test]
fn exact_evidence_requires_multiple_verification_providers_by_default() {
    let policy = ExactUpgradeOperationalHeadVerificationPolicyV1::default();
    assert_eq!(policy.minimum_distinct_providers, 2);
    assert!(policy.maximum_providers >= policy.minimum_distinct_providers);
}

#[test]
fn source_ratchets_keep_live_head_free_of_scalar_now() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("evaluation_time_unix_s:"));
    assert!(source.contains("derive_clock_governance_evaluation_envelope_v1"));
    assert!(source.contains("clock.lower_unix_ms()"));
    assert!(source.contains("clock.upper_unix_ms()"));
}

#[test]
fn source_ratchets_require_latest_handoff_scoped_publication() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("UPGRADE_OPERATIONAL_HEAD_LOG_KIND_PREFIX"));
    assert!(source.contains("matching.last()"));
    assert!(source.contains("PublicationNotLatestInCheckpointView"));
    assert!(source.contains("PublicationBeforeStateCommit"));
    assert!(source.contains("PublicationMayBeFuture"));
}

#[test]
fn source_ratchets_bind_governed_witness_identity_and_exact_raw_signatures() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("witness_registry.profile"));
    assert!(source.contains("WitnessOrganizationMismatch"));
    assert!(source.contains("WitnessFailureDomainMismatch"));
    assert!(source.contains("verify_checkpoint_signature"));
    assert!(source.contains("verify_witness_signature"));
    assert!(source.contains("CHECKPOINT_SIGNATURE_DOMAIN"));
    assert!(source.contains("WITNESS_SIGNATURE_DOMAIN"));
}

#[test]
fn source_ratchets_close_retroactive_trust_and_checkpoint_time_gaps() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("TrustSnapshotPostdatesCheckpoint"));
    assert!(source.contains("CheckpointPredatesLog"));
    assert!(source.contains("SignerInvalidAtEvidenceTime"));
    assert!(source.contains("require_valid_across_optional_seconds_window"));
    assert!(source.contains("require_effective_time_after_envelope_seconds"));
}

#[test]
fn live_head_is_opaque_and_commits_rollback_state() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("automatic_rollback_digest: Option<Sha256Digest>"));
    assert!(source.contains("pub struct QuorumObservedUpgradeOperationalHeadV1"));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct QuorumObservedUpgradeOperationalHeadV1"
    ));
}
