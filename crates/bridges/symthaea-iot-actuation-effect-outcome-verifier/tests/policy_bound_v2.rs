// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use std::error::Error as StdError;
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signer, SigningKey};
use symthaea_authority::{Digest32, Operation, PrincipalId, ResourceRef};
use symthaea_iot_actuation_effect_attempt_journal::{
    DurableEffectAttemptJournalCheckpointV1, DurableEffectAttemptJournalHeadV1,
    IndependentEffectAttemptHeadAnchor, RollbackProtectedEffectAttemptJournal,
};
use symthaea_iot_actuation_effect_dispatch::{
    PhysicalEffectAttemptCorrelation, RollbackProtectedPhysicalEffectAttemptJournal,
};
use symthaea_iot_actuation_effect_outcome_verifier::{
    CurrentPhysicalEffectOutcomeGuardV2, EFFECT_OUTCOME_ED25519_ALGORITHM,
    EFFECT_OUTCOME_EVIDENCE_SCHEMA_VERSION, EFFECT_OUTCOME_EVIDENCE_V2_SCHEMA_VERSION,
    EFFECT_OUTCOME_POLICY_SCHEMA_VERSION, EFFECT_OUTCOME_TRUST_SCHEMA_VERSION,
    EffectOutcomeClaimKindV1, EffectOutcomeClaimV1, EffectOutcomeEvidenceProvenanceModeV2,
    EffectOutcomePolicyV1, EffectOutcomePolicyV2, EffectOutcomeTrustRegistry,
    EffectOutcomeTrustSnapshotV1, EffectOutcomeVerifierKeyStatus, EffectOutcomeVerifierKeyV1,
    GuardPhysicalEffectOutcomeStateV2, MAX_EFFECT_OUTCOME_EVIDENCE_LIFETIME_MS,
    PhysicalEffectOutcomeEvidenceBodyV1, PhysicalEffectOutcomeEvidenceBodyV2,
    PhysicalEffectOutcomeEvidenceV2, PolicyBoundOutcomeError,
};
use symthaea_iot_actuation_effect_reconciliation_challenge::{
    EffectReconciliationChallengeV1, issue_effect_reconciliation_challenge,
};
use symthaea_iot_effect_outcome_policy_bound_protocol::PolicyBoundEffectReconciliationChallengeV2;

const REFERENCE_VALUES: Digest32 = Digest32([0x44; 32]);
const OUTCOME_PROFILE: Digest32 = Digest32([0x55; 32]);
const APPRAISAL_POLICY: Digest32 = Digest32([0x66; 32]);
const VERIFIER_ID: &str = "verifier:outcome-a";
const KEY_ID: &str = "key:outcome-a-v1";

fn temp_root(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "symthaea-policy-bound-v2-{label}-{}-{nanos}",
        std::process::id()
    ))
}

#[derive(Debug)]
struct TestAnchorError;

impl fmt::Display for TestAnchorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("test anchor error")
    }
}

impl StdError for TestAnchorError {}

#[derive(Clone)]
struct TestAnchor {
    head: Arc<Mutex<DurableEffectAttemptJournalHeadV1>>,
}

impl TestAnchor {
    fn new(head: DurableEffectAttemptJournalHeadV1) -> Self {
        Self {
            head: Arc::new(Mutex::new(head)),
        }
    }
}

impl IndependentEffectAttemptHeadAnchor for TestAnchor {
    type Error = TestAnchorError;

    fn current_head(&mut self) -> Result<DurableEffectAttemptJournalHeadV1, Self::Error> {
        Ok(*self.head.lock().unwrap())
    }

    fn compare_and_swap(
        &mut self,
        expected: DurableEffectAttemptJournalHeadV1,
        next: DurableEffectAttemptJournalHeadV1,
    ) -> Result<DurableEffectAttemptJournalHeadV1, Self::Error> {
        let mut head = self.head.lock().unwrap();
        if *head != expected || next.generation() != expected.generation().saturating_add(1) {
            return Err(TestAnchorError);
        }
        *head = next;
        Ok(next)
    }
}

fn fresh_challenge() -> EffectReconciliationChallengeV1 {
    let root = temp_root("challenge");
    let device = ResourceRef("iot:valve:72".into());
    let genesis = DurableEffectAttemptJournalCheckpointV1::genesis(&device)
        .unwrap()
        .head()
        .unwrap();
    let anchor = TestAnchor::new(genesis);
    let mut journal = RollbackProtectedEffectAttemptJournal::open(&root, &device, anchor).unwrap();
    let correlation = PhysicalEffectAttemptCorrelation::qualification_fixture(device, 1);
    journal.persist_prepared_anchored(&correlation).unwrap();
    let challenge = issue_effect_reconciliation_challenge(&mut journal).unwrap();
    drop(journal);
    fs::remove_dir_all(root).unwrap();
    challenge
}

fn base_policy(generation: u64) -> EffectOutcomePolicyV1 {
    EffectOutcomePolicyV1 {
        schema_version: EFFECT_OUTCOME_POLICY_SCHEMA_VERSION,
        generation,
        device: ResourceRef("iot:valve:72".into()),
        operation: Operation("qualification.effect".into()),
        allowed_verifier_ids: BTreeSet::from([VERIFIER_ID.to_owned()]),
        allowed_claim_kinds: BTreeSet::from([
            EffectOutcomeClaimKindV1::ExecutionAndPostcondition,
            EffectOutcomeClaimKindV1::NonExecution,
        ]),
        accepted_reference_values: BTreeSet::from([REFERENCE_VALUES]),
        exact_outcome_profile_digest: OUTCOME_PROFILE,
        exact_appraisal_policy_digest: APPRAISAL_POLICY,
        max_evidence_lifetime_ms: MAX_EFFECT_OUTCOME_EVIDENCE_LIFETIME_MS,
    }
}

fn semantic_body() -> PhysicalEffectOutcomeEvidenceBodyV1 {
    PhysicalEffectOutcomeEvidenceBodyV1 {
        schema_version: EFFECT_OUTCOME_EVIDENCE_SCHEMA_VERSION,
        device: ResourceRef("iot:valve:72".into()),
        operation: Operation("qualification.effect".into()),
        executor: PrincipalId("gateway:a".into()),
        challenge_digest: Digest32([0x11; 32]),
        command_digest: Digest32([0x22; 32]),
        sequence: 7,
        outcome_profile_digest: OUTCOME_PROFILE,
        reference_values_digest: REFERENCE_VALUES,
        appraisal_policy_digest: APPRAISAL_POLICY,
        verifier_id: VERIFIER_ID.into(),
        key_id: KEY_ID.into(),
        algorithm: EFFECT_OUTCOME_ED25519_ALGORITHM.into(),
        claim: EffectOutcomeClaimV1::ExecutionAndPostcondition {
            execution_record_digest: Digest32([0x77; 32]),
            effect_recorded_at_unix_ms: 1_100,
            postcondition_evidence_digest: Digest32([0x88; 32]),
            postcondition_observed_at_unix_ms: 1_300,
        },
        evidence_issued_at_unix_ms: 1_400,
        evidence_expires_at_unix_ms: 1_800,
    }
}

fn signing_key() -> SigningKey {
    SigningKey::from_bytes(&[0x72; 32])
}

fn policy_bound_challenge(
    policy: &EffectOutcomePolicyV2,
) -> PolicyBoundEffectReconciliationChallengeV2 {
    PolicyBoundEffectReconciliationChallengeV2::new(
        fresh_challenge(),
        policy.expected_identity().unwrap(),
    )
    .unwrap()
}

fn snapshot(
    challenge: &PolicyBoundEffectReconciliationChallengeV2,
    signing: &SigningKey,
) -> EffectOutcomeTrustSnapshotV1 {
    EffectOutcomeTrustSnapshotV1 {
        schema_version: EFFECT_OUTCOME_TRUST_SCHEMA_VERSION,
        sequence: 1,
        issued_at_unix_ms: challenge.issued_at_unix_ms(),
        expires_at_unix_ms: challenge.expires_at_unix_ms(),
        previous_snapshot_digest: None,
        keys: vec![EffectOutcomeVerifierKeyV1 {
            verifier_id: VERIFIER_ID.into(),
            key_id: KEY_ID.into(),
            algorithm: EFFECT_OUTCOME_ED25519_ALGORITHM.into(),
            public_key: signing.verifying_key().to_bytes(),
            status: EffectOutcomeVerifierKeyStatus::Active,
            not_before_unix_ms: challenge.issued_at_unix_ms().saturating_sub(1_000),
            not_after_unix_ms: challenge.expires_at_unix_ms(),
            max_evidence_lifetime_ms: MAX_EFFECT_OUTCOME_EVIDENCE_LIFETIME_MS,
        }],
    }
}

fn signed_policy_bound_evidence(
    challenge: &PolicyBoundEffectReconciliationChallengeV2,
    policy: &EffectOutcomePolicyV2,
    signing: &SigningKey,
    signed_policy_generation: u64,
    signed_policy_digest: Digest32,
) -> PhysicalEffectOutcomeEvidenceV2 {
    let issued = challenge.issued_at_unix_ms();
    let semantic = PhysicalEffectOutcomeEvidenceBodyV1 {
        schema_version: EFFECT_OUTCOME_EVIDENCE_SCHEMA_VERSION,
        device: challenge.device().clone(),
        operation: challenge.operation().clone(),
        executor: challenge.executor().clone(),
        challenge_digest: challenge.digest().unwrap(),
        command_digest: challenge.command_digest(),
        sequence: challenge.sequence(),
        outcome_profile_digest: policy.base.exact_outcome_profile_digest,
        reference_values_digest: REFERENCE_VALUES,
        appraisal_policy_digest: policy.base.exact_appraisal_policy_digest,
        verifier_id: VERIFIER_ID.into(),
        key_id: KEY_ID.into(),
        algorithm: EFFECT_OUTCOME_ED25519_ALGORITHM.into(),
        claim: EffectOutcomeClaimV1::ExecutionAndPostcondition {
            execution_record_digest: Digest32([0x77; 32]),
            effect_recorded_at_unix_ms: challenge.attempt_common_fenced_at_unix_ms(),
            postcondition_evidence_digest: Digest32([0x88; 32]),
            postcondition_observed_at_unix_ms: issued,
        },
        evidence_issued_at_unix_ms: issued,
        evidence_expires_at_unix_ms: challenge.expires_at_unix_ms(),
    };
    let body = PhysicalEffectOutcomeEvidenceBodyV2 {
        schema_version: EFFECT_OUTCOME_EVIDENCE_V2_SCHEMA_VERSION,
        semantic,
        outcome_policy_generation: signed_policy_generation,
        outcome_policy_digest: signed_policy_digest,
    };
    let signature = signing.sign(&body.signature_message().unwrap()).to_bytes();
    PhysicalEffectOutcomeEvidenceV2 { body, signature }
}

fn historical_guard(
    policy: &EffectOutcomePolicyV2,
    snapshot: EffectOutcomeTrustSnapshotV1,
) -> GuardPhysicalEffectOutcomeStateV2 {
    let registry = EffectOutcomeTrustRegistry::genesis(snapshot).unwrap();
    let head = registry.head();
    GuardPhysicalEffectOutcomeStateV2::new(
        policy.clone(),
        policy.digest().unwrap(),
        registry,
        head,
    )
    .unwrap()
}

fn current_guard(
    policy: &EffectOutcomePolicyV2,
    registry: EffectOutcomeTrustRegistry,
) -> CurrentPhysicalEffectOutcomeGuardV2 {
    let head = registry.head();
    CurrentPhysicalEffectOutcomeGuardV2::new(
        policy.clone(),
        policy.digest().unwrap(),
        registry,
        head,
    )
    .unwrap()
}

#[test]
fn v2_policy_mode_is_authoritative_identity_not_terminal_switch() {
    let base = base_policy(7);
    let base_digest = base.digest().unwrap();
    let strict = EffectOutcomePolicyV2::strict(base.clone()).unwrap();
    let compatibility = EffectOutcomePolicyV2::compatibility(base.clone()).unwrap();

    assert_eq!(base.digest().unwrap(), base_digest);
    assert_eq!(strict.base_digest().unwrap(), base_digest);
    assert_eq!(compatibility.base_digest().unwrap(), base_digest);
    assert_ne!(strict.digest().unwrap(), compatibility.digest().unwrap());
    assert!(!strict.allows_v1_fresh_reauthorization());
    assert!(compatibility.allows_v1_fresh_reauthorization());
}

#[test]
fn v2_signed_body_changes_when_only_policy_generation_changes() {
    let semantic = semantic_body();
    let policy1 = EffectOutcomePolicyV2::strict(base_policy(1)).unwrap();
    let policy3 = EffectOutcomePolicyV2::strict(base_policy(3)).unwrap();

    let body1 = PhysicalEffectOutcomeEvidenceBodyV2 {
        schema_version: EFFECT_OUTCOME_EVIDENCE_V2_SCHEMA_VERSION,
        semantic: semantic.clone(),
        outcome_policy_generation: policy1.generation(),
        outcome_policy_digest: policy1.digest().unwrap(),
    };
    let body3 = PhysicalEffectOutcomeEvidenceBodyV2 {
        schema_version: EFFECT_OUTCOME_EVIDENCE_V2_SCHEMA_VERSION,
        semantic: semantic.clone(),
        outcome_policy_generation: policy3.generation(),
        outcome_policy_digest: policy3.digest().unwrap(),
    };

    assert_eq!(
        semantic.canonical_bytes().unwrap(),
        semantic_body().canonical_bytes().unwrap()
    );
    assert_ne!(
        body1.canonical_bytes().unwrap(),
        body3.canonical_bytes().unwrap()
    );
    assert_ne!(body1.digest().unwrap(), body3.digest().unwrap());
    assert_ne!(
        body1.signature_message().unwrap(),
        semantic.signature_message().unwrap()
    );
}

#[test]
fn v2_body_rejects_zero_signed_policy_identity() {
    let body = PhysicalEffectOutcomeEvidenceBodyV2 {
        schema_version: EFFECT_OUTCOME_EVIDENCE_V2_SCHEMA_VERSION,
        semantic: semantic_body(),
        outcome_policy_generation: 0,
        outcome_policy_digest: Digest32([0; 32]),
    };
    assert!(body.validate_structure().is_err());
}

#[test]
fn real_policy_bound_v2_signature_and_current_fence_succeed() {
    let policy = EffectOutcomePolicyV2::strict(base_policy(3)).unwrap();
    let challenge = policy_bound_challenge(&policy);
    let signing = signing_key();
    let snapshot = snapshot(&challenge, &signing);
    let issued = challenge.issued_at_unix_ms();
    let evidence = signed_policy_bound_evidence(
        &challenge,
        &policy,
        &signing,
        policy.generation(),
        policy.digest().unwrap(),
    );

    let proof = historical_guard(&policy, snapshot.clone())
        .verify_policy_bound_evidence(evidence, &challenge)
        .unwrap();
    assert_eq!(proof.policy_generation(), policy.generation());
    assert_eq!(proof.policy_digest(), policy.digest().unwrap());
    assert_eq!(
        proof.provenance_mode(),
        EffectOutcomeEvidenceProvenanceModeV2::SignedPolicyIdentityRequired
    );
    assert_eq!(proof.challenge_digest(), challenge.digest().unwrap());

    let registry = EffectOutcomeTrustRegistry::genesis(snapshot).unwrap();
    let current = current_guard(&policy, registry);
    let fence = current.fence_current(&proof).unwrap();
    assert!(fence.fenced_at_unix_ms() >= issued);
    assert_eq!(fence.proof().evidence_digest(), proof.evidence_digest());
}

#[test]
fn signed_v2_policy_identity_mismatch_is_rejected_even_with_valid_signature() {
    let policy = EffectOutcomePolicyV2::strict(base_policy(3)).unwrap();
    let other = EffectOutcomePolicyV2::strict(base_policy(4)).unwrap();
    let challenge = policy_bound_challenge(&policy);
    let signing = signing_key();
    let evidence = signed_policy_bound_evidence(
        &challenge,
        &policy,
        &signing,
        other.generation(),
        other.digest().unwrap(),
    );
    let result = historical_guard(&policy, snapshot(&challenge, &signing))
        .verify_policy_bound_evidence(evidence, &challenge);
    assert!(matches!(
        result,
        Err(PolicyBoundOutcomeError::SignedPolicyIdentityMismatch)
    ));
}

#[test]
fn trust_successor_invalidates_historical_v2_proof_before_current_fence() {
    let policy = EffectOutcomePolicyV2::strict(base_policy(3)).unwrap();
    let challenge = policy_bound_challenge(&policy);
    let signing = signing_key();
    let first_snapshot = snapshot(&challenge, &signing);
    let evidence = signed_policy_bound_evidence(
        &challenge,
        &policy,
        &signing,
        policy.generation(),
        policy.digest().unwrap(),
    );
    let proof = historical_guard(&policy, first_snapshot.clone())
        .verify_policy_bound_evidence(evidence, &challenge)
        .unwrap();

    let first_registry = EffectOutcomeTrustRegistry::genesis(first_snapshot.clone()).unwrap();
    let successor_snapshot = EffectOutcomeTrustSnapshotV1 {
        schema_version: EFFECT_OUTCOME_TRUST_SCHEMA_VERSION,
        sequence: 2,
        issued_at_unix_ms: first_snapshot.issued_at_unix_ms + 1,
        expires_at_unix_ms: first_snapshot.expires_at_unix_ms,
        previous_snapshot_digest: Some(first_snapshot.digest().unwrap()),
        keys: first_snapshot.keys.clone(),
    };
    let successor = first_registry.successor(successor_snapshot).unwrap();
    let current = current_guard(&policy, successor);
    let result = current.fence_current(&proof);
    assert!(matches!(
        result,
        Err(PolicyBoundOutcomeError::CurrentProofTrustHeadMismatch)
    ));
}
