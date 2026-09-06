use std::collections::BTreeSet;
use std::error::Error as StdError;
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signer, SigningKey};
use symthaea_authority::{Digest32, Operation, ResourceRef};
use symthaea_iot_actuation_effect_attempt_journal::{
    DurableEffectAttemptJournalCheckpointV1, DurableEffectAttemptJournalHeadV1,
    IndependentEffectAttemptHeadAnchor, RollbackProtectedEffectAttemptJournal,
};
use symthaea_iot_actuation_effect_dispatch::{
    PhysicalEffectAttemptCorrelation, RollbackProtectedPhysicalEffectAttemptJournal,
};
use symthaea_iot_actuation_effect_reconciliation_challenge::{
    EffectReconciliationChallengeV1, issue_effect_reconciliation_challenge,
};

use crate::*;

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
        "symthaea-effect-outcome-{label}-{}-{nanos}",
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

fn signing_key() -> SigningKey {
    SigningKey::from_bytes(&[0x72; 32])
}

fn policy() -> EffectOutcomePolicyV1 {
    EffectOutcomePolicyV1 {
        schema_version: EFFECT_OUTCOME_POLICY_SCHEMA_VERSION,
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

fn snapshot(
    challenge: &EffectReconciliationChallengeV1,
    signing: &SigningKey,
    status: EffectOutcomeVerifierKeyStatus,
    key_not_after_unix_ms: u64,
    trust_expires_at_unix_ms: u64,
) -> EffectOutcomeTrustSnapshotV1 {
    EffectOutcomeTrustSnapshotV1 {
        schema_version: EFFECT_OUTCOME_TRUST_SCHEMA_VERSION,
        sequence: 1,
        issued_at_unix_ms: challenge.issued_at_unix_ms(),
        expires_at_unix_ms: trust_expires_at_unix_ms,
        previous_snapshot_digest: None,
        keys: vec![EffectOutcomeVerifierKeyV1 {
            verifier_id: VERIFIER_ID.into(),
            key_id: KEY_ID.into(),
            algorithm: EFFECT_OUTCOME_ED25519_ALGORITHM.into(),
            public_key: signing.verifying_key().to_bytes(),
            status,
            not_before_unix_ms: challenge.issued_at_unix_ms().saturating_sub(1_000),
            not_after_unix_ms: key_not_after_unix_ms,
            max_evidence_lifetime_ms: MAX_EFFECT_OUTCOME_EVIDENCE_LIFETIME_MS,
        }],
    }
}

fn signed_evidence(
    challenge: &EffectReconciliationChallengeV1,
    signing: &SigningKey,
    claim: EffectOutcomeClaimV1,
    issued_at_unix_ms: u64,
    expires_at_unix_ms: u64,
    outcome_profile_digest: Digest32,
) -> PhysicalEffectOutcomeEvidenceV1 {
    let body = PhysicalEffectOutcomeEvidenceBodyV1 {
        schema_version: EFFECT_OUTCOME_EVIDENCE_SCHEMA_VERSION,
        device: challenge.device().clone(),
        operation: challenge.operation().clone(),
        executor: challenge.executor().clone(),
        challenge_digest: challenge.digest().unwrap(),
        command_digest: challenge.command_digest(),
        sequence: challenge.sequence(),
        outcome_profile_digest,
        reference_values_digest: REFERENCE_VALUES,
        appraisal_policy_digest: APPRAISAL_POLICY,
        verifier_id: VERIFIER_ID.into(),
        key_id: KEY_ID.into(),
        algorithm: EFFECT_OUTCOME_ED25519_ALGORITHM.into(),
        claim,
        evidence_issued_at_unix_ms: issued_at_unix_ms,
        evidence_expires_at_unix_ms: expires_at_unix_ms,
    };
    let signature = signing.sign(&body.signature_message().unwrap()).to_bytes();
    PhysicalEffectOutcomeEvidenceV1 { body, signature }
}

fn guard(
    policy: &EffectOutcomePolicyV1,
    snapshot: EffectOutcomeTrustSnapshotV1,
) -> GuardPhysicalEffectOutcomeState {
    let registry = EffectOutcomeTrustRegistry::genesis(snapshot).unwrap();
    let head = registry.head();
    GuardPhysicalEffectOutcomeState::new(policy.clone(), policy.digest().unwrap(), registry, head)
        .unwrap()
}

fn current_guard(
    policy: &EffectOutcomePolicyV1,
    snapshot: EffectOutcomeTrustSnapshotV1,
) -> CurrentPhysicalEffectOutcomeGuard {
    let registry = EffectOutcomeTrustRegistry::genesis(snapshot).unwrap();
    let head = registry.head();
    CurrentPhysicalEffectOutcomeGuard::new(
        policy.clone(),
        policy.digest().unwrap(),
        registry,
        head,
    )
    .unwrap()
}

#[test]
fn execution_and_postcondition_requires_exact_execution_window_and_fresh_observation() {
    let challenge = fresh_challenge();
    let signing = signing_key();
    let policy = policy();
    let issued = challenge.issued_at_unix_ms() + 10;
    let evidence_expires = issued + 1_000;
    let snapshot = snapshot(
        &challenge,
        &signing,
        EffectOutcomeVerifierKeyStatus::Active,
        challenge.expires_at_unix_ms(),
        challenge.expires_at_unix_ms(),
    );
    let claim = EffectOutcomeClaimV1::ExecutionAndPostcondition {
        execution_record_digest: Digest32([0x71; 32]),
        effect_recorded_at_unix_ms: challenge.attempt_common_fenced_at_unix_ms(),
        postcondition_evidence_digest: Digest32([0x72; 32]),
        postcondition_observed_at_unix_ms: issued,
    };
    let evidence = signed_evidence(
        &challenge,
        &signing,
        claim,
        issued,
        evidence_expires,
        OUTCOME_PROFILE,
    );
    let proof = guard(&policy, snapshot.clone())
        .verify_evidence_at(evidence, &challenge, issued)
        .unwrap();
    assert_eq!(proof.challenge_digest(), challenge.digest().unwrap());
    assert_eq!(proof.challenge_journal_generation(), challenge.journal_generation());
    assert_eq!(proof.challenge_journal_digest(), challenge.journal_digest());

    let current = current_guard(&policy, snapshot);
    let fence = current.fence_current_at(&proof, issued + 1).unwrap();
    assert_eq!(fence.evidence_expires_at_unix_ms(), evidence_expires);
    assert_eq!(fence.valid_until_unix_ms(), evidence_expires);
}

#[test]
fn complete_window_non_execution_proof_succeeds() {
    let challenge = fresh_challenge();
    let signing = signing_key();
    let policy = policy();
    let issued = challenge.issued_at_unix_ms() + 10;
    let snapshot = snapshot(
        &challenge,
        &signing,
        EffectOutcomeVerifierKeyStatus::Active,
        challenge.expires_at_unix_ms(),
        challenge.expires_at_unix_ms(),
    );
    let claim = EffectOutcomeClaimV1::NonExecution {
        non_execution_proof_digest: Digest32([0x81; 32]),
        execution_log_head_digest: Digest32([0x82; 32]),
        coverage_from_unix_ms: challenge.attempt_common_fenced_at_unix_ms(),
        coverage_through_unix_ms: challenge.attempt_wall_valid_until_unix_ms(),
    };
    let proof = guard(&policy, snapshot)
        .verify_evidence_at(
            signed_evidence(
                &challenge,
                &signing,
                claim,
                issued,
                issued + 1_000,
                OUTCOME_PROFILE,
            ),
            &challenge,
            issued,
        )
        .unwrap();
    assert_eq!(proof.evidence().body.claim.kind(), EffectOutcomeClaimKindV1::NonExecution);
}

#[test]
fn partial_window_non_execution_is_rejected() {
    let challenge = fresh_challenge();
    let signing = signing_key();
    let policy = policy();
    let issued = challenge.issued_at_unix_ms() + 10;
    let snapshot = snapshot(
        &challenge,
        &signing,
        EffectOutcomeVerifierKeyStatus::Active,
        challenge.expires_at_unix_ms(),
        challenge.expires_at_unix_ms(),
    );
    let claim = EffectOutcomeClaimV1::NonExecution {
        non_execution_proof_digest: Digest32([0x81; 32]),
        execution_log_head_digest: Digest32([0x82; 32]),
        coverage_from_unix_ms: challenge.attempt_common_fenced_at_unix_ms() + 1,
        coverage_through_unix_ms: challenge.attempt_wall_valid_until_unix_ms(),
    };
    let result = guard(&policy, snapshot).verify_evidence_at(
        signed_evidence(
            &challenge,
            &signing,
            claim,
            issued,
            issued + 1_000,
            OUTCOME_PROFILE,
        ),
        &challenge,
        issued,
    );
    assert!(matches!(result, Err(EffectOutcomeError::NonExecutionCoverageIncomplete)));
}

#[test]
fn execution_record_at_exclusive_actuation_deadline_is_rejected() {
    let challenge = fresh_challenge();
    let signing = signing_key();
    let policy = policy();
    let issued = challenge.issued_at_unix_ms() + 10;
    let snapshot = snapshot(
        &challenge,
        &signing,
        EffectOutcomeVerifierKeyStatus::Active,
        challenge.expires_at_unix_ms(),
        challenge.expires_at_unix_ms(),
    );
    let claim = EffectOutcomeClaimV1::ExecutionAndPostcondition {
        execution_record_digest: Digest32([0x71; 32]),
        effect_recorded_at_unix_ms: challenge.attempt_wall_valid_until_unix_ms(),
        postcondition_evidence_digest: Digest32([0x72; 32]),
        postcondition_observed_at_unix_ms: issued,
    };
    let result = guard(&policy, snapshot).verify_evidence_at(
        signed_evidence(
            &challenge,
            &signing,
            claim,
            issued,
            issued + 1_000,
            OUTCOME_PROFILE,
        ),
        &challenge,
        issued,
    );
    assert!(matches!(
        result,
        Err(EffectOutcomeError::ExecutionRecordOutsideActuationWindow)
    ));
}

#[test]
fn wrong_profile_and_bad_signature_fail_closed() {
    let challenge = fresh_challenge();
    let signing = signing_key();
    let policy = policy();
    let issued = challenge.issued_at_unix_ms() + 10;
    let snapshot = snapshot(
        &challenge,
        &signing,
        EffectOutcomeVerifierKeyStatus::Active,
        challenge.expires_at_unix_ms(),
        challenge.expires_at_unix_ms(),
    );
    let claim = EffectOutcomeClaimV1::ExecutionAndPostcondition {
        execution_record_digest: Digest32([0x71; 32]),
        effect_recorded_at_unix_ms: challenge.attempt_common_fenced_at_unix_ms(),
        postcondition_evidence_digest: Digest32([0x72; 32]),
        postcondition_observed_at_unix_ms: issued,
    };

    let wrong_profile = guard(&policy, snapshot.clone()).verify_evidence_at(
        signed_evidence(
            &challenge,
            &signing,
            claim,
            issued,
            issued + 1_000,
            Digest32([0x99; 32]),
        ),
        &challenge,
        issued,
    );
    assert!(matches!(
        wrong_profile,
        Err(EffectOutcomeError::EvidenceOutcomeProfileMismatch)
    ));

    let mut bad_signature = signed_evidence(
        &challenge,
        &signing,
        claim,
        issued,
        issued + 1_000,
        OUTCOME_PROFILE,
    );
    bad_signature.signature[0] ^= 0x01;
    let result = guard(&policy, snapshot).verify_evidence_at(bad_signature, &challenge, issued);
    assert!(matches!(result, Err(EffectOutcomeError::InvalidEvidenceSignature)));
}

#[test]
fn successor_trust_generation_invalidates_historical_proof() {
    let challenge = fresh_challenge();
    let signing = signing_key();
    let policy = policy();
    let issued = challenge.issued_at_unix_ms() + 10;
    let first_snapshot = snapshot(
        &challenge,
        &signing,
        EffectOutcomeVerifierKeyStatus::Active,
        challenge.expires_at_unix_ms(),
        challenge.expires_at_unix_ms(),
    );
    let first_registry = EffectOutcomeTrustRegistry::genesis(first_snapshot.clone()).unwrap();
    let first_head = first_registry.head();
    let historical_guard = GuardPhysicalEffectOutcomeState::new(
        policy.clone(),
        policy.digest().unwrap(),
        first_registry,
        first_head,
    )
    .unwrap();
    let claim = EffectOutcomeClaimV1::ExecutionAndPostcondition {
        execution_record_digest: Digest32([0x71; 32]),
        effect_recorded_at_unix_ms: challenge.attempt_common_fenced_at_unix_ms(),
        postcondition_evidence_digest: Digest32([0x72; 32]),
        postcondition_observed_at_unix_ms: issued,
    };
    let proof = historical_guard
        .verify_evidence_at(
            signed_evidence(
                &challenge,
                &signing,
                claim,
                issued,
                issued + 1_000,
                OUTCOME_PROFILE,
            ),
            &challenge,
            issued,
        )
        .unwrap();

    let first_again = EffectOutcomeTrustRegistry::genesis(first_snapshot.clone()).unwrap();
    let mut retired = first_snapshot.keys[0].clone();
    retired.status = EffectOutcomeVerifierKeyStatus::Retired;
    let successor = first_again
        .successor(EffectOutcomeTrustSnapshotV1 {
            schema_version: EFFECT_OUTCOME_TRUST_SCHEMA_VERSION,
            sequence: 2,
            issued_at_unix_ms: first_snapshot.issued_at_unix_ms,
            expires_at_unix_ms: first_snapshot.expires_at_unix_ms,
            previous_snapshot_digest: Some(first_again.head().digest),
            keys: vec![retired],
        })
        .unwrap();
    let successor_head = successor.head();
    let current = CurrentPhysicalEffectOutcomeGuard::new(
        policy.clone(),
        policy.digest().unwrap(),
        successor,
        successor_head,
    )
    .unwrap();
    let result = current.fence_current_at(&proof, issued + 1);
    assert!(matches!(
        result,
        Err(EffectOutcomeError::CurrentProofTrustHeadMismatch)
    ));
}

#[test]
fn each_natural_expiry_boundary_kills_current_use() {
    let signing = signing_key();
    let policy = policy();

    // Evidence expiry.
    let challenge = fresh_challenge();
    let issued = challenge.issued_at_unix_ms() + 10;
    let snapshot = snapshot(
        &challenge,
        &signing,
        EffectOutcomeVerifierKeyStatus::Active,
        challenge.expires_at_unix_ms() + 1_000,
        challenge.expires_at_unix_ms() + 1_000,
    );
    let claim = EffectOutcomeClaimV1::ExecutionAndPostcondition {
        execution_record_digest: Digest32([0x71; 32]),
        effect_recorded_at_unix_ms: challenge.attempt_common_fenced_at_unix_ms(),
        postcondition_evidence_digest: Digest32([0x72; 32]),
        postcondition_observed_at_unix_ms: issued,
    };
    let evidence_expiry = issued + 200;
    let proof = guard(&policy, snapshot.clone())
        .verify_evidence_at(
            signed_evidence(
                &challenge,
                &signing,
                claim,
                issued,
                evidence_expiry,
                OUTCOME_PROFILE,
            ),
            &challenge,
            issued,
        )
        .unwrap();
    let current = current_guard(&policy, snapshot);
    assert!(matches!(
        current.fence_current_at(&proof, evidence_expiry),
        Err(EffectOutcomeError::CurrentProofWindowElapsed)
    ));

    // Exact key expiry.
    let challenge = fresh_challenge();
    let issued = challenge.issued_at_unix_ms() + 10;
    let key_expiry = issued + 200;
    let snapshot = snapshot(
        &challenge,
        &signing,
        EffectOutcomeVerifierKeyStatus::Active,
        key_expiry,
        challenge.expires_at_unix_ms() + 1_000,
    );
    let proof = guard(&policy, snapshot.clone())
        .verify_evidence_at(
            signed_evidence(
                &challenge,
                &signing,
                claim,
                issued,
                issued + 1_000,
                OUTCOME_PROFILE,
            ),
            &challenge,
            issued,
        )
        .unwrap();
    let current = current_guard(&policy, snapshot);
    assert!(matches!(
        current.fence_current_at(&proof, key_expiry),
        Err(EffectOutcomeError::VerifierKeyNotActive)
    ));

    // Trust-snapshot expiry.
    let challenge = fresh_challenge();
    let issued = challenge.issued_at_unix_ms() + 10;
    let trust_expiry = issued + 200;
    let snapshot = snapshot(
        &challenge,
        &signing,
        EffectOutcomeVerifierKeyStatus::Active,
        challenge.expires_at_unix_ms() + 1_000,
        trust_expiry,
    );
    let proof = guard(&policy, snapshot.clone())
        .verify_evidence_at(
            signed_evidence(
                &challenge,
                &signing,
                claim,
                issued,
                issued + 1_000,
                OUTCOME_PROFILE,
            ),
            &challenge,
            issued,
        )
        .unwrap();
    let current = current_guard(&policy, snapshot);
    assert!(matches!(
        current.fence_current_at(&proof, trust_expiry),
        Err(EffectOutcomeError::TrustSnapshotNotFresh)
    ));

    // Challenge expiry, with every other natural boundary later.
    let challenge = fresh_challenge();
    let issued = challenge.issued_at_unix_ms() + 10;
    let snapshot = snapshot(
        &challenge,
        &signing,
        EffectOutcomeVerifierKeyStatus::Active,
        challenge.expires_at_unix_ms() + 1_000,
        challenge.expires_at_unix_ms() + 1_000,
    );
    let proof = guard(&policy, snapshot.clone())
        .verify_evidence_at(
            signed_evidence(
                &challenge,
                &signing,
                claim,
                issued,
                challenge.expires_at_unix_ms(),
                OUTCOME_PROFILE,
            ),
            &challenge,
            issued,
        )
        .unwrap();
    let current = current_guard(&policy, snapshot);
    assert!(matches!(
        current.fence_current_at(&proof, challenge.expires_at_unix_ms()),
        Err(EffectOutcomeError::CurrentProofWindowElapsed)
    ));
}
