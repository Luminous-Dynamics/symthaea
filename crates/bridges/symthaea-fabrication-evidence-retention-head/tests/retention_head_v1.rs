// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn retention_currentness_consumes_opaque_authority_not_raw_policy() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("retention: &ClockGovernedEvidenceRetentionPolicyV1"));
    assert!(source.contains("build_evidence_retention_head_publication_v1("));
    assert!(!source.contains("policy: &EvidenceRetentionPolicy"));
    assert!(source.contains("retention.id().to_hex()"));
}

#[test]
fn exact_registry_head_is_explicitly_rebound_to_composite_view() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("registry_head: &QuorumObservedWitnessRegistryHeadV1"));
    assert!(source.contains("governance_view.registry_head_id() != registry_head.id()"));
    assert!(source.contains("governance_view.registry_digest() != registry_head.registry_digest()"));
    assert!(source.contains("governance_view.registry_sequence() != registry_head.sequence()"));
    assert!(source.contains("governance_view.checkpoint_digest() != registry_head.checkpoint_digest()"));
    assert!(source.contains("RegistryHeadMismatch"));
}

#[test]
fn retention_must_share_highest_containment_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("current_containment_authority.id() != governance_view.containment_authority_id()"));
    assert!(source.contains("retention.containment_state_digest() != current_containment_authority.state_digest()"));
    assert!(source.contains("retention.containment_generation() != current_containment_authority.generation()"));
    assert!(source.contains("retention.compromise_tracker_digest()"));
    assert!(source.contains("RetentionContainmentMismatch"));
}

#[test]
fn retention_must_share_exact_trust_snapshot_and_fresh_interval() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("trust_snapshot_digest != retention.trust_snapshot_digest()"));
    assert!(source.contains("trust_snapshot_digest != registry_head.trust_snapshot_digest()"));
    assert!(source.contains("trust_snapshot.sequence != registry_head.trust_snapshot_sequence()"));
    assert!(source.contains("TrustSnapshotNotValidAcrossObservationEnvelope"));
    assert!(source.contains("require_valid_across_seconds_window"));
}

#[test]
fn retention_threshold_ceremony_is_rebound_exactly() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("ceremony.id() != retention.threshold_ceremony_id()"));
    assert!(source.contains("CLOCK_GOVERNED_EVIDENCE_RETENTION_PURPOSE"));
    assert!(source.contains("ceremony.payload_digest() != retention.prepared_id().as_digest()"));
    assert!(source.contains("ceremony.compromise_tracker_digest() != retention.compromise_tracker_digest()"));
    assert!(source.contains("ceremony.clock_envelope_id() != retention.clock_envelope_id()"));
}

#[test]
fn currentness_uses_same_authenticated_log_and_strict_sequence() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("transparency_log_digest != governance_view.transparency_log_digest()"));
    assert!(source.contains("transparency_log_digest != registry_head.transparency_log_digest()"));
    assert!(source.contains("EVIDENCE_RETENTION_HEAD_LOG_KIND_PREFIX"));
    assert!(source.contains("RetentionSequenceRegressed"));
    assert!(source.contains("DuplicateRetentionSequence"));
    assert!(source.contains("HigherRetentionSequencePublished"));
    assert!(source.contains("suffix != sequence.to_string()"));
}

#[test]
fn publication_is_definitely_post_authorization_and_pre_observation() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("publication_recorded_at_ms < authorization_clock.upper_unix_ms()"));
    assert!(source.contains("PublicationBeforeAuthorization"));
    assert!(source.contains("publication_recorded_at_ms > observation_clock.lower_unix_ms()"));
    assert!(source.contains("PublicationMayBeFuture"));
}

#[test]
fn current_head_commits_direct_checkpoint_and_registry_provenance() {
    let source = include_str!("../src/lib.rs");
    for required in [
        "governance_view_id",
        "registry_head_id",
        "governance_checkpoint_digest",
        "retention_authority_id",
        "policy_digest",
        "transparency_log_digest",
        "trust_snapshot_digest",
        "containment_authority_id",
        "compromise_tracker_digest",
        "clock_lineage_digest",
    ] {
        assert!(source.contains(required), "missing retention-head commitment {required}");
    }
}

#[test]
fn live_retention_head_is_opaque_and_scalar_time_free() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct CurrentEvidenceRetentionHeadV1"
    ));
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("evaluation_time_unix_s:"));
    assert!(!source.contains("AuthorizedEvidenceRetentionPolicy"));
}
