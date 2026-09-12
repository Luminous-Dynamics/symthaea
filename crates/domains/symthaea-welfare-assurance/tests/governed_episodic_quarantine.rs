// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use chrono::{TimeZone, Utc};
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::intervention_interlock::{
    ExplicitConsentState, InterventionEvidence, InterventionRequest, WelfareConstraintLevel,
};
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
};
use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};
use symthaea_psych_bench::moral_patient::{MoralPatientEvidenceProfile, PrecautionPolicy};
use symthaea_welfare_assurance::authority_evidence_binding::build_evidence_bound_authority_statement;
use symthaea_welfare_assurance::evidence_context::bind_welfare_evidence_to_request;
use symthaea_welfare_assurance::execution_adapter::{
    ExecutionJournalPersistence, JournaledExecutionOutcome,
};
use symthaea_welfare_assurance::execution_recovery::InterventionExecutionJournal;
use symthaea_welfare_assurance::memory_identity::episode_content_id;
use symthaea_welfare_assurance::memory_quarantine::{
    EpisodicQuarantineEscrow, EpisodicQuarantineEscrowPersistence,
    digest_episodic_quarantine_escrow, episodic_instance_target_id,
    execute_governed_episodic_quarantine,
};
use symthaea_welfare_assurance::replay_recovery::{
    DurableAuthorityReplayFence, DurableAuthorityReplaySnapshot, ReplayFencePersistence,
    authorize_evidence_bound_intervention_durable,
};
use symthaea_welfare_authority::{
    WelfareAuthorityPolicyManifest, WelfareAuthorityRole, WelfareAuthoritySigner,
    WelfareAuthoritySignerBinding, WelfareAuthoritySignatureVerifier, WelfareAuthorityTracker,
    sign_welfare_authority,
};
use symthaea_welfare_consent::{
    SubjectConsentDecision, SubjectConsentLedger, SubjectConsentPolicy,
    SubjectConsentSignatureVerifier, SubjectConsentSigner, SubjectConsentStatement,
    SubjectIdentityBinding, SubjectIdentityRegistry, SubjectIdentityStatus, bind_latest_live,
    sign_subject_consent, verify_subject_consent,
};

const SUBJECT_ID: &str = "symthaea:self";
const STORE_TARGET_ID: &str = "symthaea:self:episodic-memory";
const BASE_NONCE: &str = "memory-quarantine-issuance-1";

struct SubjectSigner;
struct SubjectVerifier;

impl SubjectConsentSigner for SubjectSigner {
    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::Ed25519
    }

    fn key_id(&self) -> &str {
        "subject-key"
    }

    fn sign_subject_consent(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(signature(b"subject", self.key_id(), message))
    }
}

impl SubjectConsentSignatureVerifier for SubjectVerifier {
    fn verify_subject_consent(
        &self,
        _algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature_bytes: &[u8],
    ) -> Result<bool, String> {
        Ok(signature(b"subject", key_id, message) == signature_bytes)
    }
}

struct AuthoritySigner(&'static str);
struct AuthorityVerifier;

impl WelfareAuthoritySigner for AuthoritySigner {
    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::Ed25519
    }

    fn key_id(&self) -> &str {
        self.0
    }

    fn sign_welfare_authority(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(signature(b"authority", self.key_id(), message))
    }
}

impl WelfareAuthoritySignatureVerifier for AuthorityVerifier {
    fn verify_welfare_authority(
        &self,
        _algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature_bytes: &[u8],
    ) -> Result<bool, String> {
        Ok(signature(b"authority", key_id, message) == signature_bytes)
    }
}

fn signature(domain: &[u8], key_id: &str, message: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.welfare.governed-quarantine-test.v1\0");
    hasher.update(domain);
    hasher.update(key_id.as_bytes());
    hasher.update(message);
    hasher.finalize().0.to_vec()
}

fn request(target_id: String) -> InterventionRequest {
    InterventionRequest {
        action: SubjectAffectingAction::MemoryModification,
        target_id,
        rationale: "consensual reversible quarantine of one exact episodic occurrence".into(),
        welfare_constraint: WelfareConstraintLevel::Baseline,
        emergency: false,
        less_restrictive_unavailable: false,
        post_hoc_review_required: false,
        evaluated_at: Utc.timestamp_opt(120, 0).single().unwrap(),
        evidence: InterventionEvidence {
            authority_ref: None,
            consent_state: ExplicitConsentState::Unknown,
            consent_ref: None,
            welfare_review_ref: Some("welfare-review:episodic-quarantine:1".into()),
            independent_review_ref: Some("independent-review:episodic-quarantine:1".into()),
            independent_safety_evidence: Vec::new(),
            welfare_report_ids: Vec::new(),
        },
    }
}

fn subject_registry() -> SubjectIdentityRegistry {
    let mut registry = SubjectIdentityRegistry::default();
    registry
        .register(SubjectIdentityBinding {
            subject_id: SUBJECT_ID.into(),
            identity_epoch: 1,
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "subject-key".into(),
            not_before_unix_s: 50,
            not_after_unix_s: Some(500),
            status: SubjectIdentityStatus::Active,
        })
        .unwrap();
    registry
}

fn consent_ledger(
    request: &InterventionRequest,
    registry: &SubjectIdentityRegistry,
) -> SubjectConsentLedger {
    let statement = SubjectConsentStatement::for_request(
        "memory-quarantine-consent-1",
        SUBJECT_ID,
        1,
        1,
        SubjectConsentDecision::Grant,
        100,
        200,
        request,
    )
    .unwrap();
    let signed = sign_subject_consent(statement, &SubjectSigner).unwrap();
    let verified = verify_subject_consent(
        &signed,
        registry,
        SubjectConsentPolicy::default(),
        120,
        &SubjectVerifier,
    )
    .unwrap();
    let mut ledger = SubjectConsentLedger::default();
    ledger.ingest(verified).unwrap();
    ledger
}

fn operator_key(key_id: &str) -> KeyTrustRecord {
    KeyTrustRecord {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: key_id.into(),
        not_before_unix_s: 50,
        not_after_unix_s: Some(500),
        status: KeyLifecycleStatus::Active,
        usages: BTreeSet::from([KeyUsage::OperatorCommand]),
    }
}

fn trust_snapshot() -> TrustSnapshot {
    TrustSnapshot::new(
        7,
        80,
        400,
        vec![operator_key("operator"), operator_key("reviewer")],
    )
    .unwrap()
}

fn authority_manifest() -> WelfareAuthorityPolicyManifest {
    WelfareAuthorityPolicyManifest::new(
        "welfare-quarantine-policy-v1",
        [
            WelfareAuthoritySignerBinding::new(
                SignatureAlgorithm::Ed25519,
                "operator",
                WelfareAuthorityRole::Operator,
            ),
            WelfareAuthoritySignerBinding::new(
                SignatureAlgorithm::Ed25519,
                "reviewer",
                WelfareAuthorityRole::IndependentReviewer,
            ),
        ],
    )
    .unwrap()
}

fn duplicate_episode() -> Episode {
    Episode::new(
        ContinuousHV::from_values(vec![1.0, 2.0, 3.0]),
        ContinuousHV::from_values(vec![4.0, 5.0, 6.0]),
        0.82,
        42,
    )
}

#[derive(Default)]
struct ReplayPersistence;

impl ReplayFencePersistence for ReplayPersistence {
    type Error = std::io::Error;

    fn persist_replay_snapshot(
        &mut self,
        snapshot: &DurableAuthorityReplaySnapshot,
        _digest: Sha256Digest,
    ) -> Result<String, Self::Error> {
        Ok(format!("test:replay-generation:{}", snapshot.generation))
    }
}

#[derive(Default)]
struct ExecutionPersistence;

impl ExecutionJournalPersistence for ExecutionPersistence {
    type Error = std::io::Error;

    fn persist_execution_journal(
        &mut self,
        events: &[symthaea_welfare_assurance::execution_recovery::ExecutionJournalEnvelope],
        _head_hash: Sha256Digest,
    ) -> Result<String, Self::Error> {
        Ok(format!("test:execution-events:{}", events.len()))
    }
}

#[derive(Default)]
struct EscrowPersistence {
    saved: Option<(EpisodicQuarantineEscrow, Sha256Digest)>,
}

impl EpisodicQuarantineEscrowPersistence for EscrowPersistence {
    type Error = std::io::Error;

    fn persist_episodic_quarantine_escrow(
        &mut self,
        escrow: &EpisodicQuarantineEscrow,
        escrow_digest: Sha256Digest,
    ) -> Result<String, Self::Error> {
        self.saved = Some((escrow.clone(), escrow_digest));
        Ok("test:episodic-quarantine-escrow:1".into())
    }
}

#[test]
fn full_chain_quarantines_only_the_authorized_duplicate_occurrence() {
    let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
    let duplicate = duplicate_episode();
    let first_id = memory
        .store_if_significant_with_id(duplicate.clone())
        .expect("first duplicate should store");
    let second_id = memory
        .store_if_significant_with_id(duplicate)
        .expect("second duplicate should store");
    assert_ne!(first_id, second_id);

    let initial_instances = memory.get_top_episode_instances(10);
    let first_episode = initial_instances
        .iter()
        .find(|(id, _)| *id == first_id)
        .map(|(_, episode)| episode.clone())
        .unwrap();
    let second_episode = initial_instances
        .iter()
        .find(|(id, _)| *id == second_id)
        .map(|(_, episode)| episode.clone())
        .unwrap();
    assert_eq!(
        episode_content_id(&first_episode).unwrap(),
        episode_content_id(&second_episode).unwrap()
    );

    let target_id = episodic_instance_target_id(STORE_TARGET_ID, first_id).unwrap();
    let profile = MoralPatientEvidenceProfile::default();
    let precaution_policy = PrecautionPolicy::default();
    let (request, context) =
        bind_welfare_evidence_to_request(&request(target_id.clone()), &profile, &precaution_policy)
            .unwrap();
    assert_eq!(context.constraint(), WelfareConstraintLevel::Baseline);

    let registry = subject_registry();
    let ledger = consent_ledger(&request, &registry);
    let consent_bound = bind_latest_live(&ledger, &registry, SUBJECT_ID, &request, 120).unwrap();
    let trust = trust_snapshot();
    let manifest = authority_manifest();
    let statement = build_evidence_bound_authority_statement(
        BASE_NONCE,
        &profile,
        &precaution_policy,
        &manifest,
        &trust,
        1,
        1,
        100,
        200,
        &consent_bound,
    )
    .unwrap();
    let operator = AuthoritySigner("operator");
    let reviewer = AuthoritySigner("reviewer");
    let signed = sign_welfare_authority(statement, &[&operator, &reviewer]).unwrap();

    let mut authority_tracker = WelfareAuthorityTracker::default();
    let mut replay_fence = DurableAuthorityReplayFence::new().unwrap();
    let mut replay_persistence = ReplayPersistence;
    let permit = authorize_evidence_bound_intervention_durable(
        BASE_NONCE,
        &profile,
        &precaution_policy,
        SUBJECT_ID,
        &ledger,
        &registry,
        &signed,
        &manifest,
        &trust,
        &AuthorityVerifier,
        &mut authority_tracker,
        &mut replay_fence,
        &mut replay_persistence,
        &request,
        120,
    )
    .unwrap();

    let mut journal = InterventionExecutionJournal::new();
    let mut execution_persistence = ExecutionPersistence;
    let mut escrow_persistence = EscrowPersistence::default();
    let outcome = execute_governed_episodic_quarantine(
        permit,
        "exec:episodic-quarantine:1",
        STORE_TARGET_ID,
        first_id,
        &mut memory,
        &profile,
        &precaution_policy,
        &ledger,
        &registry,
        &manifest,
        &trust,
        120,
        &mut journal,
        &mut execution_persistence,
        &mut escrow_persistence,
    )
    .unwrap();

    let (escrow, escrow_digest) = escrow_persistence.saved.as_ref().expect("escrow persisted");
    assert_eq!(escrow.target_id, target_id);
    assert_eq!(escrow.instance_id, first_id);
    assert_eq!(escrow.episode.instance_id, Some(first_id));
    assert_eq!(escrow.content_id, episode_content_id(&first_episode).unwrap());
    assert_eq!(
        digest_episodic_quarantine_escrow(escrow).unwrap(),
        *escrow_digest
    );

    match outcome {
        JournaledExecutionOutcome::Completed {
            output,
            prepared_persistence_ref,
            completion_persistence_ref,
            ..
        } => {
            assert_eq!(output.target_id, target_id);
            assert_eq!(output.instance_id, first_id);
            assert_eq!(output.before_active_count, 2);
            assert_eq!(output.after_active_count, 1);
            assert_eq!(output.escrow_digest, *escrow_digest);
            assert_eq!(
                output.escrow_persistence_ref,
                "test:episodic-quarantine-escrow:1"
            );
            assert_eq!(prepared_persistence_ref, "test:execution-events:1");
            assert_eq!(completion_persistence_ref, "test:execution-events:2");
        }
        other => panic!("expected fully persisted quarantine, got {other:?}"),
    }

    assert_eq!(memory.len(), 1);
    assert_eq!(memory.quarantined_len(), 1);
    assert!(memory.quarantined_instance(first_id).is_some());
    assert!(memory.quarantined_instance(second_id).is_none());
    let active = memory.get_top_episode_instances(10);
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].0, second_id);
    assert_eq!(
        episode_content_id(&active[0].1).unwrap(),
        episode_content_id(&first_episode).unwrap()
    );
    assert_eq!(journal.recovery_report().completed, 1);
    assert!(journal.recovery_report().in_doubt.is_empty());
}
