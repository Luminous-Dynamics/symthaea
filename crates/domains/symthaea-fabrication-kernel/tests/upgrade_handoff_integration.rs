// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use symthaea_fabrication_kernel::*;

struct TestSigner {
    algorithm: SignatureAlgorithm,
    key_id: &'static str,
}

impl ThresholdApprovalSigner for TestSigner {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        self.key_id
    }
    fn sign_threshold_approval(&self, _message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(vec![1, 2, 3])
    }
}

struct AcceptVerifier;
impl ThresholdApprovalVerifier for AcceptVerifier {
    fn verify_threshold_approval(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        _message: &[u8],
        _signature: &[u8],
    ) -> Result<bool, String> {
        Ok(true)
    }
}
impl ClockObservationVerifier for AcceptVerifier {
    fn verify_clock_observation(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        _message: &[u8],
        _signature: &[u8],
    ) -> Result<bool, String> {
        Ok(true)
    }
}

fn trust_snapshot() -> TrustSnapshot {
    let usages = BTreeSet::from([
        KeyUsage::ClockAuthority,
        KeyUsage::PolicyMigration,
        KeyUsage::UpgradeHandoff,
        KeyUsage::RecoveryAuthorization,
    ]);
    TrustSnapshot::new(
        4,
        1,
        1_000,
        vec![
            KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "authority-a".into(),
                not_before_unix_s: 1,
                not_after_unix_s: None,
                status: KeyLifecycleStatus::Active,
                usages: usages.clone(),
            },
            KeyTrustRecord {
                algorithm: SignatureAlgorithm::MlDsa65,
                key_id: "authority-b".into(),
                not_before_unix_s: 1,
                not_after_unix_s: None,
                status: KeyLifecycleStatus::Active,
                usages,
            },
        ],
    )
    .unwrap()
}

fn ceremony(
    purpose: &str,
    payload: Sha256Digest,
    usage: KeyUsage,
    snapshot: &TrustSnapshot,
) -> VerifiedThresholdCeremony {
    let a = TestSigner {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "authority-a",
    };
    let b = TestSigner {
        algorithm: SignatureAlgorithm::MlDsa65,
        key_id: "authority-b",
    };
    let approvals = vec![
        sign_threshold_approval(purpose, payload, 100, 300, &a).unwrap(),
        sign_threshold_approval(purpose, payload, 100, 300, &b).unwrap(),
    ];
    let policy = ThresholdCeremonyPolicy {
        minimum_distinct_signers: 2,
        maximum_approvals: 4,
        require_algorithm_diversity: true,
        required_algorithms: BTreeSet::new(),
        allowed_key_ids: None,
        key_usage: usage,
    };
    verify_threshold_ceremony(
        purpose,
        payload,
        &approvals,
        &policy,
        snapshot,
        150,
        &AcceptVerifier,
    )
    .unwrap()
}

#[test]
fn secure_upgrade_bundle_round_trips_with_exact_authority_evidence() {
    let snapshot = trust_snapshot();
    let observations = [
        ClockObservation {
            schema_version: CLOCK_OBSERVATION_SCHEMA.into(),
            source_id: "clock-a".into(),
            observed_unix_ms: 150_000,
            uncertainty_ms: 100,
            epoch: 8,
            signature: DetachedSignature {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "authority-a".into(),
                signature: vec![1],
            },
        },
        ClockObservation {
            schema_version: CLOCK_OBSERVATION_SCHEMA.into(),
            source_id: "clock-b".into(),
            observed_unix_ms: 150_050,
            uncertainty_ms: 100,
            epoch: 8,
            signature: DetachedSignature {
                algorithm: SignatureAlgorithm::MlDsa65,
                key_id: "authority-b".into(),
                signature: vec![1],
            },
        },
    ];
    let clock = verify_clock_quorum(
        &observations,
        &ClockQuorumPolicy::default(),
        &snapshot,
        150,
        &AcceptVerifier,
    )
    .unwrap();

    let predecessor_policy = PolicyBinding::new(
        "upgrade-authority",
        "1",
        sha256(b"policy-1"),
        vec![PolicyInvariantBinding {
            name: "fail-closed".into(),
            digest: sha256(b"v1"),
        }],
    )
    .unwrap();
    let successor_policy = PolicyBinding::new(
        "upgrade-authority",
        "2",
        sha256(b"policy-2"),
        vec![PolicyInvariantBinding {
            name: "fail-closed".into(),
            digest: sha256(b"v2"),
        }],
    )
    .unwrap();
    let migration_plan = PolicyMigrationPlan {
        schema_version: "symthaea.fabrication.policy-migration.v1".into(),
        predecessor: predecessor_policy,
        successor: successor_policy,
        activates_at_unix_s: 200,
        rollback_deadline_unix_s: 500,
        rationale: "bind quorum time and explicit migration evidence".into(),
        migrations: vec![PolicyInvariantMigration {
            name: "fail-closed".into(),
            predecessor_digest: sha256(b"v1"),
            successor_digest: Some(sha256(b"v2")),
            disposition: PolicyInvariantDisposition::Strengthened,
        }],
    };
    let migration_policy = PolicyMigrationPolicy::default();
    let migration_digest =
        digest_policy_migration_plan(&migration_plan, &migration_policy, 150).unwrap();
    let migration_ceremony = ceremony(
        "policy-migration",
        migration_digest,
        KeyUsage::PolicyMigration,
        &snapshot,
    );
    let migration =
        authorize_policy_migration(migration_plan, &migration_policy, 150, &migration_ceremony)
            .unwrap();

    let recovery_key_set = RecoveryKeySet::new(
        "offline-recovery",
        2,
        1,
        1_000,
        2,
        2,
        BTreeSet::from([RecoveryScope::RestoreTrustSnapshot]),
        vec![
            RecoveryParticipant {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "authority-a".into(),
                custodian_id: "custodian-a".into(),
                region: "af-south".into(),
            },
            RecoveryParticipant {
                algorithm: SignatureAlgorithm::MlDsa65,
                key_id: "authority-b".into(),
                custodian_id: "custodian-b".into(),
                region: "eu-west".into(),
            },
        ],
    )
    .unwrap();

    let mut journal = EvidenceJournal::default();
    journal.append(1, "upgrade-prepared", sha256(b"p")).unwrap();
    journal
        .append(2, "policy-migrated", migration.plan_digest)
        .unwrap();
    journal
        .append(3, "clock-verified", clock.evidence_digest)
        .unwrap();
    let compaction_policy = EvidenceCompactionPolicy {
        minimum_retained_tail: 2,
        maximum_retained_tail: 8,
    };
    let compacted = compact_evidence(&journal, 2, None, &compaction_policy).unwrap();
    let compacted_digest = digest_compacted_evidence(&compacted, &compaction_policy).unwrap();
    let mut compaction_tracker = EvidenceCompactionTracker::default();
    compaction_tracker
        .accept(&compacted, &compaction_policy)
        .unwrap();

    let predecessor_epoch = AuthorityEpochVector::new(4, 3, 7, 5, 2, 30, 8, 9).unwrap();
    let successor_epoch = AuthorityEpochVector::new(4, 3, 8, 5, 3, 31, 8, 9).unwrap();
    let mut clock_tracker = ClockEpochTracker::default();
    clock_tracker.accept(&clock).unwrap();
    let mut authority_epoch_tracker = AuthorityEpochTracker::default();
    authority_epoch_tracker
        .accept(successor_epoch.clone())
        .unwrap();
    let recovery_tracker = RecoveryActivationTracker::default();
    let endpoint = |version: &str, state: &[u8], epoch: AuthorityEpochVector| UpgradeEndpoint {
        schema_version: "symthaea.fabrication.upgrade-endpoint.v1".into(),
        software_version: version.into(),
        source_tree_digest: sha256(format!("source-{version}").as_bytes()),
        executable_digest: sha256(format!("executable-{version}").as_bytes()),
        durable_state_digest: sha256(state),
        replay_contract_digest: sha256(format!("replay-{version}").as_bytes()),
        authority_epoch: epoch,
    };
    let predecessor = endpoint("0.17.0", b"state-17", predecessor_epoch);
    let successor = endpoint("0.18.0", b"state-18", successor_epoch.clone());
    let handoff_plan = UpgradeHandoffPlan {
        schema_version: "symthaea.fabrication.upgrade-handoff.v1".into(),
        predecessor,
        successor,
        prepared_at_unix_ms: 150_000,
        activates_at_unix_ms: 200_000,
        finalization_deadline_unix_ms: 400_000,
        rollback_target_digest: sha256(b"state-17"),
        policy_migration_digests: vec![migration.plan_digest],
        clock_evidence_digest: clock.evidence_digest,
        evidence_checkpoint_digest: compacted_digest,
        recovery_key_set_digest: digest_recovery_key_set(&recovery_key_set).unwrap(),
        reason: "upgrade with explicit authority transfer".into(),
    };
    let handoff_policy = UpgradeHandoffPolicy::default();
    let handoff_digest = digest_upgrade_handoff_plan(
        &handoff_plan,
        &handoff_policy,
        &clock,
        std::slice::from_ref(&migration),
    )
    .unwrap();
    let handoff_ceremony = ceremony(
        "upgrade-handoff",
        handoff_digest,
        KeyUsage::UpgradeHandoff,
        &snapshot,
    );
    let handoff = authorize_upgrade_handoff(
        handoff_plan,
        &handoff_policy,
        &clock,
        std::slice::from_ref(&migration),
        &handoff_ceremony,
    )
    .unwrap();

    let mut upgrade_tracker = UpgradeHandoffTracker::new(&handoff);
    upgrade_tracker
        .append(
            &handoff,
            UpgradeStage::Prepared,
            150_000,
            handoff.plan.predecessor.durable_state_digest,
            sha256(b"prepared-evidence"),
        )
        .unwrap();
    let upgrade_tracker_digest = digest_upgrade_tracker(&upgrade_tracker, &handoff).unwrap();

    let mut policy_tracker = PolicyMigrationTracker::default();
    policy_tracker.accept(&migration, 150).unwrap();
    let migration_set_digest =
        digest_policy_migration_set(std::slice::from_ref(&migration)).unwrap();
    let state = FabricationUpgradeState::genesis(
        150_000,
        UpgradeEvidenceDigests {
            handoff_digest: handoff.plan_digest,
            upgrade_tracker_digest,
            policy_migration_set_digest: migration_set_digest,
            clock_tracker_digest: digest_clock_epoch_tracker(&clock_tracker).unwrap(),
            authority_epoch_tracker_digest: digest_authority_epoch_tracker(
                &authority_epoch_tracker,
            )
            .unwrap(),
            recovery_tracker_digest: digest_recovery_activation_tracker(&recovery_tracker).unwrap(),
            evidence_compaction_tracker_digest: digest_evidence_compaction_tracker(
                &compaction_tracker,
            )
            .unwrap(),
        },
    )
    .unwrap();

    let bundle = build_upgrade_evidence_bundle(
        sha256(b"source-tree-18"),
        handoff,
        vec![migration],
        policy_tracker,
        150,
        clock,
        clock_tracker,
        successor_epoch,
        authority_epoch_tracker,
        recovery_key_set,
        recovery_tracker,
        compacted,
        compaction_policy,
        compaction_tracker,
        upgrade_tracker,
        state,
    )
    .unwrap();
    let limits = UpgradeBundleLimits::default();
    let bytes = encode_upgrade_evidence_bundle(&bundle, &limits).unwrap();
    let decoded = decode_upgrade_evidence_bundle(&bytes, &limits).unwrap();
    assert_eq!(
        decoded.replay_contract_digest,
        bundle.replay_contract_digest
    );
}
