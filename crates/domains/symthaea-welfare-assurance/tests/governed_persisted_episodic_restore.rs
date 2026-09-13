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
use symthaea_memory::episodic_replay::{
    Episode, EpisodeInstanceId, EpisodicMemory, EpisodicReplayConfig,
};
use symthaea_psych_bench::moral_patient::{MoralPatientEvidenceProfile, PrecautionPolicy};
use symthaea_welfare_assurance::authority_evidence_binding::build_evidence_bound_authority_statement;
use symthaea_welfare_assurance::evidence_context::bind_welfare_evidence_to_request;
use symthaea_welfare_assurance::execution_adapter::{
    ExecutionJournalPersistence, JournaledExecutionOutcome,
};
use symthaea_welfare_assurance::execution_recovery::{
    ExecutionJournalEnvelope, InterventionExecutionJournal,
};
use symthaea_welfare_assurance::memory_identity::{EpisodeContentId, episode_content_id};
use symthaea_welfare_assurance::memory_quarantine::{
    EPISODIC_QUARANTINE_ESCROW_SCHEMA, EpisodicQuarantineEscrow,
    digest_episodic_quarantine_escrow, episodic_instance_target_id,
};
use symthaea_welfare_assurance::persisted_memory_restore::{
    EpisodicQuarantineEscrowLookup, PersistedEpisodicEscrowRow,
    execute_governed_persisted_episodic_restore,
};
use symthaea_welfare_assurance::quarantine_ledger_persistence::QuarantineLedgerPersistence;
use symthaea_welfare_assurance::quarantine_state_ledger::{
    EpisodicQuarantineStateLedger, QuarantineLedgerEnvelope,
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
const BASE_NONCE: &str = "persisted-memory-restore-issuance-1";
const EXECUTION_ID: &str = "exec:persisted-episodic-restore:1";

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
    hasher.update(b"symthaea.welfare.governed-persisted-restore-test.v1\0");
    hasher.update(domain);
    hasher.update(key_id.as_bytes());
    hasher.update(message);
    hasher.finalize().0.to_vec()
}

fn request(target_id: String) -> InterventionRequest {
    InterventionRequest {
        action: SubjectAffectingAction::MemoryModification,
        target_id,
        rationale: "consensual exact restoration of one restart-withheld episodic occurrence".into(),
        welfare_constraint: WelfareConstraintLevel::Baseline,
        emergency: false,
        less_restrictive_unavailable: false,
        post_hoc_review_required: false,
        evaluated_at: Utc.timestamp_opt(120, 0).single().unwrap(),
        evidence: InterventionEvidence {
            authority_ref: None,
            consent_state: ExplicitConsentState::Unknown,
            consent_ref: None,
            welfare_review_ref: Some("welfare-review:persisted-episodic-restore:1".into()),
            independent_review_ref: Some("independent-review:persisted-episodic-restore:1".into()),
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
        "persisted-memory-restore-consent-1",
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
        "welfare-persisted-restore-policy-v1",
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
struct ExecutionPersistence {
    calls: usize,
}

impl ExecutionJournalPersistence for ExecutionPersistence {
    type Error = std::io::Error;

    fn persist_execution_journal(
        &mut self,
        events: &[ExecutionJournalEnvelope],
        _head_hash: Sha256Digest,
    ) -> Result<String, Self::Error> {
        self.calls += 1;
        Ok(format!("test:execution-events:{}", events.len()))
    }
}

#[derive(Default)]
struct LedgerPersistence {
    calls: usize,
    fail_on_call: Option<usize>,
    snapshots: Vec<(Vec<QuarantineLedgerEnvelope>, Sha256Digest)>,
}

impl QuarantineLedgerPersistence for LedgerPersistence {
    type Error = std::io::Error;

    fn persist_quarantine_ledger(
        &mut self,
        events: &[QuarantineLedgerEnvelope],
        head_hash: Sha256Digest,
    ) -> Result<String, Self::Error> {
        self.calls += 1;
        if self.fail_on_call == Some(self.calls) {
            return Err(std::io::Error::other(
                "injected quarantine-ledger persistence failure",
            ));
        }
        self.snapshots.push((events.to_vec(), head_hash));
        Ok(format!("test:quarantine-ledger:{}", self.calls))
    }
}

#[derive(Clone)]
struct EscrowLookup {
    row: PersistedEpisodicEscrowRow,
}

impl EpisodicQuarantineEscrowLookup for EscrowLookup {
    type Error = std::io::Error;

    fn load_episodic_quarantine_escrow(
        &self,
        instance_id: EpisodeInstanceId,
    ) -> Result<Option<PersistedEpisodicEscrowRow>, Self::Error> {
        Ok((self.row.escrow.instance_id == instance_id).then(|| self.row.clone()))
    }
}

struct PreparedPersistedRestore {
    memory: EpisodicMemory,
    ledger: EpisodicQuarantineStateLedger,
    lookup: EscrowLookup,
    first_id: EpisodeInstanceId,
    second_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
    target_id: String,
}

fn prepared_persisted_restore_state() -> PreparedPersistedRestore {
    let mut source_memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
    let duplicate = duplicate_episode();
    let content_id = episode_content_id(&duplicate).unwrap();
    let first_id = source_memory
        .store_if_significant_with_id(duplicate.clone())
        .unwrap();
    let second_id = source_memory.store_if_significant_with_id(duplicate).unwrap();
    assert_ne!(first_id, second_id);

    let instances = source_memory.get_top_episode_instances(10);
    let first_episode = instances
        .iter()
        .find(|(id, _)| *id == first_id)
        .unwrap()
        .1
        .clone();
    let second_episode = instances
        .iter()
        .find(|(id, _)| *id == second_id)
        .unwrap()
        .1
        .clone();
    assert_eq!(episode_content_id(&first_episode).unwrap(), content_id);
    assert_eq!(episode_content_id(&second_episode).unwrap(), content_id);

    // This is the post-restart shape: only B is active and there is no live quarantine payload.
    let memory = EpisodicMemory::from_validated_persisted_active_state(
        EpisodicReplayConfig::broad_capture(),
        120,
        vec![second_episode],
    )
    .unwrap();
    assert_eq!(memory.len(), 1);
    assert_eq!(memory.quarantined_len(), 0);

    let target_id = episodic_instance_target_id(STORE_TARGET_ID, first_id).unwrap();
    let escrow = EpisodicQuarantineEscrow {
        schema_version: EPISODIC_QUARANTINE_ESCROW_SCHEMA.into(),
        target_id: target_id.clone(),
        instance_id: first_id,
        content_id,
        captured_at_unix_s: 90,
        pre_active_state_digest: Sha256Digest([8; 32]),
        episode: first_episode,
    };
    let escrow_digest = digest_episodic_quarantine_escrow(&escrow).unwrap();
    let escrow_ref = format!("test:persisted-quarantine-escrow:{first_id}");
    let lookup = EscrowLookup {
        row: PersistedEpisodicEscrowRow {
            escrow,
            stored_digest: escrow_digest,
            persistence_ref: escrow_ref.clone(),
        },
    };

    let mut ledger = EpisodicQuarantineStateLedger::new();
    ledger
        .append_quarantined(
            &target_id,
            first_id,
            content_id,
            91,
            escrow_digest,
            escrow_ref,
        )
        .unwrap();

    PreparedPersistedRestore {
        memory,
        ledger,
        lookup,
        first_id,
        second_id,
        content_id,
        target_id,
    }
}

#[allow(clippy::type_complexity)]
fn authorized_restore_permit(
    target_id: &str,
) -> (
    MoralPatientEvidenceProfile,
    PrecautionPolicy,
    SubjectIdentityRegistry,
    SubjectConsentLedger,
    TrustSnapshot,
    WelfareAuthorityPolicyManifest,
    symthaea_welfare_assurance::replay_recovery::DurableEvidenceBoundInterventionPermit,
) {
    let profile = MoralPatientEvidenceProfile::default();
    let precaution_policy = PrecautionPolicy::default();
    let (request, context) = bind_welfare_evidence_to_request(
        &request(target_id.to_string()),
        &profile,
        &precaution_policy,
    )
    .unwrap();
    assert_eq!(context.constraint(), WelfareConstraintLevel::Baseline);

    let registry = subject_registry();
    let consent_ledger = consent_ledger(&request, &registry);
    let consent_bound =
        bind_latest_live(&consent_ledger, &registry, SUBJECT_ID, &request, 120).unwrap();
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
        &consent_ledger,
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

    (
        profile,
        precaution_policy,
        registry,
        consent_ledger,
        trust,
        manifest,
        permit,
    )
}

fn assert_only_second_active(state: &PreparedPersistedRestore) {
    assert_eq!(state.memory.len(), 1);
    assert_eq!(state.memory.quarantined_len(), 0);
    assert!(state.memory.quarantined_instance(state.first_id).is_none());
    let active = state.memory.get_top_episode_instances(10);
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].0, state.second_id);
    assert_eq!(episode_content_id(&active[0].1).unwrap(), state.content_id);
}

#[test]
fn full_chain_persisted_restore_preserves_exact_identity_and_closes_ledger() {
    let mut state = prepared_persisted_restore_state();
    assert_only_second_active(&state);
    let (profile, precaution_policy, registry, consent_ledger, trust, manifest, permit) =
        authorized_restore_permit(&state.target_id);

    let mut execution_journal = InterventionExecutionJournal::new();
    let mut execution_persistence = ExecutionPersistence::default();
    let mut ledger_persistence = LedgerPersistence::default();
    let outcome = execute_governed_persisted_episodic_restore(
        permit,
        EXECUTION_ID,
        STORE_TARGET_ID,
        state.first_id,
        &mut state.memory,
        &mut state.ledger,
        &state.lookup,
        &profile,
        &precaution_policy,
        &consent_ledger,
        &registry,
        &manifest,
        &trust,
        120,
        &mut execution_journal,
        &mut execution_persistence,
        &mut ledger_persistence,
    )
    .unwrap();

    match outcome {
        JournaledExecutionOutcome::Completed {
            output,
            prepared_persistence_ref,
            completion_persistence_ref,
            ..
        } => {
            assert_eq!(output.target_id, state.target_id);
            assert_eq!(output.instance_id, state.first_id);
            assert_eq!(output.content_id, state.content_id);
            assert_eq!(output.before_active_count, 1);
            assert_eq!(output.after_active_count, 2);
            assert_eq!(output.before_quarantined_count, 0);
            assert_eq!(output.after_quarantined_count, 0);
            assert_eq!(output.ledger_generation, 3);
            assert_eq!(prepared_persistence_ref, "test:execution-events:1");
            assert_eq!(completion_persistence_ref, "test:execution-events:2");
        }
        other => panic!("expected fully persisted post-restart restore, got {other:?}"),
    }

    assert_eq!(execution_persistence.calls, 2);
    assert_eq!(ledger_persistence.calls, 2);
    assert_eq!(state.memory.len(), 2);
    assert_eq!(state.memory.quarantined_len(), 0);
    assert!(state.ledger.unresolved_state(state.first_id).is_none());
    let active = state.memory.get_top_episode_instances(10);
    assert!(active.iter().any(|(id, episode)| {
        *id == state.first_id && episode_content_id(episode).unwrap() == state.content_id
    }));
    assert!(active.iter().any(|(id, episode)| {
        *id == state.second_id && episode_content_id(episode).unwrap() == state.content_id
    }));

    let (events, trusted_head) = ledger_persistence.snapshots.last().unwrap();
    let recovered =
        EpisodicQuarantineStateLedger::recover_anchored(events, *trusted_head).unwrap();
    assert_eq!(recovered.generation(), 3);
    assert_eq!(recovered.unresolved_count(), 0);
    assert_eq!(execution_journal.recovery_report().completed, 1);
    assert!(execution_journal.recovery_report().in_doubt.is_empty());
}

#[test]
fn final_ledger_persistence_failure_leaves_restart_occurrence_inactive_and_prepare_pending() {
    let mut state = prepared_persisted_restore_state();
    let (profile, precaution_policy, registry, consent_ledger, trust, manifest, permit) =
        authorized_restore_permit(&state.target_id);

    let mut execution_journal = InterventionExecutionJournal::new();
    let mut execution_persistence = ExecutionPersistence::default();
    let mut ledger_persistence = LedgerPersistence {
        fail_on_call: Some(2),
        ..Default::default()
    };
    let outcome = execute_governed_persisted_episodic_restore(
        permit,
        EXECUTION_ID,
        STORE_TARGET_ID,
        state.first_id,
        &mut state.memory,
        &mut state.ledger,
        &state.lookup,
        &profile,
        &precaution_policy,
        &consent_ledger,
        &registry,
        &manifest,
        &trust,
        120,
        &mut execution_journal,
        &mut execution_persistence,
        &mut ledger_persistence,
    )
    .unwrap();

    assert!(matches!(
        outcome,
        JournaledExecutionOutcome::ExecutorInDoubt { .. }
    ));
    assert_eq!(ledger_persistence.calls, 2);
    assert_eq!(ledger_persistence.snapshots.len(), 1);
    assert_only_second_active(&state);

    let unresolved = state.ledger.unresolved_state(state.first_id).unwrap();
    assert_eq!(unresolved.content_id, state.content_id);
    let pending = unresolved
        .restore_pending
        .as_ref()
        .expect("RestorePrepared remains pending");
    assert_eq!(pending.execution_id, EXECUTION_ID);
    assert_eq!(state.ledger.generation(), 2);

    let (events, trusted_head) = ledger_persistence.snapshots.last().unwrap();
    let recovered =
        EpisodicQuarantineStateLedger::recover_anchored(events, *trusted_head).unwrap();
    assert_eq!(
        recovered
            .unresolved_state(state.first_id)
            .unwrap()
            .restore_pending
            .as_ref()
            .unwrap()
            .execution_id,
        EXECUTION_ID
    );
    assert_eq!(execution_journal.recovery_report().completed, 0);
    assert_eq!(execution_journal.recovery_report().in_doubt.len(), 1);
}

#[test]
fn tampered_persisted_escrow_is_rejected_before_execution_prepare_or_live_mutation() {
    let mut state = prepared_persisted_restore_state();
    state.lookup.row.stored_digest = Sha256Digest([0xAA; 32]);
    let (profile, precaution_policy, registry, consent_ledger, trust, manifest, permit) =
        authorized_restore_permit(&state.target_id);

    let mut execution_journal = InterventionExecutionJournal::new();
    let mut execution_persistence = ExecutionPersistence::default();
    let mut ledger_persistence = LedgerPersistence::default();
    let outcome = execute_governed_persisted_episodic_restore(
        permit,
        EXECUTION_ID,
        STORE_TARGET_ID,
        state.first_id,
        &mut state.memory,
        &mut state.ledger,
        &state.lookup,
        &profile,
        &precaution_policy,
        &consent_ledger,
        &registry,
        &manifest,
        &trust,
        120,
        &mut execution_journal,
        &mut execution_persistence,
        &mut ledger_persistence,
    )
    .unwrap();

    assert!(matches!(
        outcome,
        JournaledExecutionOutcome::PreflightRejected { .. }
    ));
    assert_eq!(execution_persistence.calls, 0);
    assert_eq!(ledger_persistence.calls, 0);
    assert_eq!(state.ledger.generation(), 1);
    assert_only_second_active(&state);
    assert_eq!(execution_journal.recovery_report().completed, 0);
    assert!(execution_journal.recovery_report().in_doubt.is_empty());
}
