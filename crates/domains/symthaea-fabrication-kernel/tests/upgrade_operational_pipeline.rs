// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::clock::VerifiedClockWindow;
use symthaea_fabrication_kernel::clock_continuity::{
    ClockContinuityPolicy, verify_clock_continuity,
};
use symthaea_fabrication_kernel::crypto_digest::sha256;
use symthaea_fabrication_kernel::evidence_retention::{
    EVIDENCE_RETENTION_POLICY_SCHEMA, EvidenceClass, EvidenceRetentionPolicy,
    EvidenceRetentionRule, digest_evidence_retention_policy,
};
use symthaea_fabrication_kernel::hardware_reauthorization_tracker::{
    HardwareReauthorizationTracker, digest_hardware_reauthorization_tracker,
};
use symthaea_fabrication_kernel::key_continuity::{KeyContinuityPolicy, verify_key_continuity};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
};
use symthaea_fabrication_kernel::upgrade_operational_bundle::{
    UpgradeOperationalBundleLimits, build_upgrade_operational_evidence_bundle,
    decode_upgrade_operational_evidence_bundle, encode_upgrade_operational_evidence_bundle,
};
use symthaea_fabrication_kernel::upgrade_operational_state::{
    FabricationUpgradeOperationalState, UpgradeOperationalEvidenceDigests,
};
use symthaea_fabrication_kernel::upgrade_probation_tracker::{
    UpgradeProbationTracker, digest_upgrade_probation_tracker,
};
use symthaea_fabrication_kernel::upgrade_state::{
    FabricationUpgradeState, UpgradeEvidenceDigests, digest_upgrade_state,
};

fn trust_snapshot(sequence: u64) -> TrustSnapshot {
    let usages = [KeyUsage::HardwareReauthorization, KeyUsage::UpgradeHandoff]
        .into_iter()
        .collect::<BTreeSet<_>>();
    TrustSnapshot::new(
        sequence,
        1,
        10_000,
        vec![
            KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "bridge-ed25519".into(),
                not_before_unix_s: 1,
                not_after_unix_s: Some(10_000),
                status: KeyLifecycleStatus::Active,
                usages: usages.clone(),
            },
            KeyTrustRecord {
                algorithm: SignatureAlgorithm::MlDsa65,
                key_id: "bridge-mldsa65".into(),
                not_before_unix_s: 1,
                not_after_unix_s: Some(10_000),
                status: KeyLifecycleStatus::Active,
                usages,
            },
        ],
    )
    .unwrap()
}

fn retention_policy() -> EvidenceRetentionPolicy {
    let classes = [
        EvidenceClass::SafetyCritical,
        EvidenceClass::Audit,
        EvidenceClass::Governance,
        EvidenceClass::MachineTelemetry,
        EvidenceClass::BuildArtifact,
        EvidenceClass::Diagnostic,
        EvidenceClass::Temporary,
    ];
    EvidenceRetentionPolicy {
        schema_version: EVIDENCE_RETENTION_POLICY_SCHEMA.into(),
        sequence: 1,
        effective_at_unix_s: 1,
        rules: classes
            .into_iter()
            .map(|class| EvidenceRetentionRule {
                class,
                minimum_hot_duration_s: 10,
                minimum_total_retention_s: 100,
                compaction_permitted: true,
                deletion_permitted: class == EvidenceClass::Temporary,
            })
            .collect(),
    }
}

#[test]
fn operational_upgrade_bundle_round_trips_exactly() {
    let previous_trust = trust_snapshot(1);
    let successor_trust = trust_snapshot(2);
    let key_continuity = verify_key_continuity(
        &previous_trust,
        &successor_trust,
        100,
        &KeyContinuityPolicy {
            required_usages: [KeyUsage::HardwareReauthorization].into_iter().collect(),
            minimum_bridge_keys_per_usage: 1,
            minimum_successor_keys_per_usage: 2,
            minimum_overlap_s: 100,
            require_successor_algorithm_diversity: true,
        },
    )
    .unwrap();

    let previous_clock = VerifiedClockWindow {
        lower_unix_ms: 1_000,
        upper_unix_ms: 1_100,
        consensus_unix_ms: 1_050,
        epoch: 1,
        source_ids: vec!["clock-a".into(), "clock-b".into()],
        algorithms: vec![SignatureAlgorithm::Ed25519, SignatureAlgorithm::MlDsa65],
        trust_snapshot_digest: sha256(b"trust-1"),
        evidence_digest: sha256(b"clock-1"),
    };
    let successor_clock = VerifiedClockWindow {
        lower_unix_ms: 1_090,
        upper_unix_ms: 1_200,
        consensus_unix_ms: 1_150,
        epoch: 2,
        source_ids: vec!["clock-b".into(), "clock-c".into()],
        algorithms: vec![SignatureAlgorithm::Ed25519, SignatureAlgorithm::MlDsa65],
        trust_snapshot_digest: sha256(b"trust-2"),
        evidence_digest: sha256(b"clock-2"),
    };
    let clock_continuity = verify_clock_continuity(
        &previous_clock,
        &successor_clock,
        &ClockContinuityPolicy::default(),
    )
    .unwrap();

    let handoff_digest = sha256(b"handoff");
    let upgrade_state = FabricationUpgradeState::genesis(
        1_200,
        UpgradeEvidenceDigests {
            handoff_digest,
            upgrade_tracker_digest: sha256(b"upgrade-tracker"),
            policy_migration_set_digest: sha256(b"migration"),
            clock_tracker_digest: sha256(b"clock-tracker"),
            authority_epoch_tracker_digest: sha256(b"epoch-tracker"),
            recovery_tracker_digest: sha256(b"recovery-tracker"),
            evidence_compaction_tracker_digest: sha256(b"compaction-tracker"),
        },
    )
    .unwrap();
    let probation_tracker = UpgradeProbationTracker::default();
    let hardware_tracker = HardwareReauthorizationTracker::default();
    let retention_policy = retention_policy();
    let operational_state = FabricationUpgradeOperationalState::genesis(
        1_300,
        handoff_digest,
        UpgradeOperationalEvidenceDigests {
            upgrade_state_digest: digest_upgrade_state(&upgrade_state).unwrap(),
            probation_tracker_digest: digest_upgrade_probation_tracker(&probation_tracker).unwrap(),
            hardware_reauthorization_tracker_digest: digest_hardware_reauthorization_tracker(
                &hardware_tracker,
            )
            .unwrap(),
            retention_policy_digest: digest_evidence_retention_policy(&retention_policy).unwrap(),
            key_continuity_digest: key_continuity.evidence_digest,
            clock_continuity_digest: clock_continuity.continuity_digest,
            probation_clearance_digest: None,
            automatic_rollback_digest: None,
            probation_sequence: None,
            reauthorized_machine_count: 0,
            retention_policy_sequence: retention_policy.sequence,
            key_snapshot_sequence: key_continuity.successor_snapshot_sequence,
            clock_epoch: clock_continuity.successor_epoch,
        },
    )
    .unwrap();

    let bundle = build_upgrade_operational_evidence_bundle(
        sha256(b"source-tree"),
        handoff_digest,
        upgrade_state,
        None,
        probation_tracker,
        hardware_tracker,
        retention_policy,
        key_continuity,
        clock_continuity,
        None,
        operational_state,
    )
    .unwrap();
    let limits = UpgradeOperationalBundleLimits::default();
    let encoded = encode_upgrade_operational_evidence_bundle(&bundle, &limits).unwrap();
    let decoded = decode_upgrade_operational_evidence_bundle(&encoded, &limits).unwrap();
    assert_eq!(decoded, bundle);
}
