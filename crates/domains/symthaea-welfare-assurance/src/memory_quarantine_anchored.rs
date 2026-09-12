// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash-conservative exact-instance episodic quarantine.
//!
//! This is the hardened successor to the first governed quarantine adapter. It closes the crash
//! window between durable escrow and canonical mutation by persisting an exact write-ahead
//! quarantine intent before removing the occurrence from active memory. Recovery policy for a
//! pending intent is deliberately conservative: the UUID remains inactive until reconciled.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use serde::{Deserialize, Serialize};
use symthaea_core::intervention_interlock::ExplicitConsentState;
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::trust::TrustSnapshot;
use symthaea_memory::episodic_replay::{
    Episode, EpisodeInstanceId, EpisodicMemory, EpisodicQuarantineError,
};
use symthaea_psych_bench::moral_patient::{MoralPatientEvidenceProfile, PrecautionPolicy};
use symthaea_welfare_authority::WelfareAuthorityPolicyManifest;
use symthaea_welfare_consent::{SubjectConsentLedger, SubjectIdentityRegistry};
use thiserror::Error;

use crate::AssuredInterventionPermit;
use crate::execution_adapter::{
    ExecutionJournalPersistence, ExecutionObservationError, JournaledExecutionGateError,
    JournaledExecutionOutcome, ReceiptedExecution, ReceiptedInterventionExecutor,
    execute_durable_intervention_journaled,
};
use crate::execution_recovery::InterventionExecutionJournal;
use crate::memory_identity::{EpisodeContentId, EpisodeContentIdError, episode_content_id};
use crate::memory_intervention::digest_episodic_memory;
use crate::memory_quarantine::{
    EPISODIC_QUARANTINE_ESCROW_SCHEMA, EpisodicQuarantineEscrow,
    EpisodicQuarantineEscrowPersistence, digest_episodic_quarantine_escrow,
    episodic_instance_target_id,
};
use crate::quarantine_intent_ledger::{
    EpisodicQuarantineIntentLedger, QuarantineIntentLedgerError,
};
use crate::quarantine_intent_persistence::QuarantineIntentLedgerPersistence;
use crate::quarantine_ledger_persistence::QuarantineLedgerPersistence;
use crate::quarantine_state_ledger::{EpisodicQuarantineStateLedger, QuarantineLedgerError};
use crate::replay_recovery::DurableEvidenceBoundInterventionPermit;

const RESULT_DOMAIN: &[u8] = b"symthaea.welfare.anchored-episodic-quarantine-result.v1\0";
const MAX_REF_BYTES: usize = 2048;

/// Evidence returned only after the exact occurrence is inactive and both the quarantine-state
/// ledger and the write-ahead intent ledger have durably committed that fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchoredEpisodicQuarantineReceipt {
    pub target_id: String,
    pub execution_id: String,
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub before_active_count: usize,
    pub after_active_count: usize,
    pub before_active_digest: Sha256Digest,
    pub after_active_digest: Sha256Digest,
    pub escrow_digest: Sha256Digest,
    pub escrow_persistence_ref: String,
    pub intent_prepared_head: Sha256Digest,
    pub intent_prepared_persistence_ref: String,
    pub quarantine_state_head: Sha256Digest,
    pub quarantine_state_persistence_ref: String,
    pub intent_committed_head: Sha256Digest,
    pub intent_committed_persistence_ref: String,
}

struct AnchoredQuarantineExecutor<'a, E, I, S>
where
    E: EpisodicQuarantineEscrowPersistence,
    I: QuarantineIntentLedgerPersistence,
    S: QuarantineLedgerPersistence,
{
    expected_target_id: String,
    execution_id: String,
    instance_id: EpisodeInstanceId,
    memory: &'a mut EpisodicMemory,
    escrow_persistence: &'a mut E,
    intent_ledger: &'a mut EpisodicQuarantineIntentLedger,
    intent_persistence: &'a mut I,
    quarantine_ledger: &'a mut EpisodicQuarantineStateLedger,
    quarantine_persistence: &'a mut S,
    completed_at_unix_s: u64,
}

impl<E, I, S> AnchoredQuarantineExecutor<'_, E, I, S>
where
    E: EpisodicQuarantineEscrowPersistence,
    I: QuarantineIntentLedgerPersistence,
    S: QuarantineLedgerPersistence,
{
    fn validate_live_state(
        &self,
        permit: &AssuredInterventionPermit,
    ) -> Result<(Episode, EpisodeContentId), AnchoredQuarantineInterventionError> {
        if permit.action() != SubjectAffectingAction::MemoryModification {
            return Err(AnchoredQuarantineInterventionError::WrongAction {
                actual: permit.action(),
            });
        }
        if permit.target_id() != self.expected_target_id {
            return Err(AnchoredQuarantineInterventionError::WrongTarget {
                expected: self.expected_target_id.clone(),
                actual: permit.target_id().to_string(),
            });
        }
        if permit.is_emergency() {
            return Err(AnchoredQuarantineInterventionError::EmergencyQuarantineNotTyped);
        }
        if permit.explicit_consent_state() != ExplicitConsentState::Granted {
            return Err(AnchoredQuarantineInterventionError::ExplicitConsentRequired);
        }
        if !permit.has_welfare_review_reference() {
            return Err(AnchoredQuarantineInterventionError::WelfareReviewRequired);
        }
        if !permit.has_independent_review_reference() {
            return Err(AnchoredQuarantineInterventionError::IndependentReviewRequired);
        }
        if self.intent_ledger.pending_state(self.instance_id).is_some() {
            return Err(AnchoredQuarantineInterventionError::IntentAlreadyPending(
                self.instance_id,
            ));
        }
        if self
            .quarantine_ledger
            .unresolved_state(self.instance_id)
            .is_some()
        {
            return Err(AnchoredQuarantineInterventionError::LedgerAlreadyQuarantined(
                self.instance_id,
            ));
        }
        let episode = active_episode(self.memory, self.instance_id)?;
        if episode.instance_id != Some(self.instance_id) {
            return Err(AnchoredQuarantineInterventionError::InstanceIdentityMismatch);
        }
        let content_id = episode_content_id(&episode)?;
        Ok((episode, content_id))
    }
}

impl<E, I, S> ReceiptedInterventionExecutor for AnchoredQuarantineExecutor<'_, E, I, S>
where
    E: EpisodicQuarantineEscrowPersistence,
    I: QuarantineIntentLedgerPersistence,
    S: QuarantineLedgerPersistence,
{
    type Output = AnchoredEpisodicQuarantineReceipt;
    type Error = AnchoredQuarantineExecutionError<E::Error, I::Error, S::Error>;

    fn preflight(&self, permit: &AssuredInterventionPermit) -> Result<(), Self::Error> {
        self.validate_live_state(permit)?;
        Ok(())
    }

    fn execute_receipted(
        &mut self,
        permit: &AssuredInterventionPermit,
    ) -> Result<ReceiptedExecution<Self::Output>, Self::Error> {
        let (episode, content_id) = self.validate_live_state(permit)?;
        let before_active_count = self.memory.len();
        let before_active_digest = digest_episodic_memory(self.memory)
            .map_err(|error| AnchoredQuarantineInterventionError::MemoryState(error.to_string()))?;

        let escrow = EpisodicQuarantineEscrow {
            schema_version: EPISODIC_QUARANTINE_ESCROW_SCHEMA.into(),
            target_id: self.expected_target_id.clone(),
            instance_id: self.instance_id,
            content_id,
            captured_at_unix_s: self.completed_at_unix_s,
            pre_active_state_digest: before_active_digest,
            episode,
        };
        let escrow_digest = digest_episodic_quarantine_escrow(&escrow)?;
        let escrow_persistence_ref = self
            .escrow_persistence
            .persist_episodic_quarantine_escrow(&escrow, escrow_digest)
            .map_err(AnchoredQuarantineExecutionError::EscrowPersistence)?;
        validate_ref(&escrow_persistence_ref)?;

        // Write-ahead quarantine intent: once this is durable, restart logic must keep the UUID
        // inactive even if the process dies before the canonical mutation below.
        let intent_before_prepare = self.intent_ledger.clone();
        let intent_prepared_head = self.intent_ledger.append_prepared(
            &self.execution_id,
            &self.expected_target_id,
            self.instance_id,
            content_id,
            self.completed_at_unix_s,
            before_active_digest,
            escrow_digest,
            &escrow_persistence_ref,
        )?;
        let intent_prepared_persistence_ref = match self
            .intent_persistence
            .persist_quarantine_intent_ledger(
                self.intent_ledger.events(),
                intent_prepared_head,
            ) {
            Ok(reference) => reference,
            Err(error) => {
                *self.intent_ledger = intent_before_prepare;
                return Err(AnchoredQuarantineExecutionError::IntentPreparedPersistence(
                    error,
                ));
            }
        };
        if let Err(error) = validate_ref(&intent_prepared_persistence_ref) {
            *self.intent_ledger = intent_before_prepare;
            return Err(error.into());
        }

        // A pending intent is now durable. From this point onward, failure remains conservative:
        // the occurrence stays or becomes inactive and recovery requires reconciliation.
        let immediate_digest = digest_episodic_memory(self.memory)
            .map_err(|error| AnchoredQuarantineInterventionError::MemoryState(error.to_string()))?;
        if immediate_digest != before_active_digest {
            return Err(AnchoredQuarantineInterventionError::PreStateChanged.into());
        }
        let immediate = active_episode(self.memory, self.instance_id)?;
        if episode_content_id(&immediate)? != content_id {
            return Err(AnchoredQuarantineInterventionError::ContentIdentityMismatch.into());
        }

        let quarantined = self
            .memory
            .quarantine_instance(self.instance_id)
            .map_err(AnchoredQuarantineInterventionError::MemoryMechanism)?;
        if quarantined.instance_id != self.instance_id
            || quarantined.episode.instance_id != Some(self.instance_id)
            || episode_content_id(&quarantined.episode)? != content_id
        {
            return Err(AnchoredQuarantineInterventionError::QuarantinePostconditionMismatch.into());
        }

        let after_active_count = self.memory.len();
        if after_active_count.checked_add(1) != Some(before_active_count) {
            return Err(AnchoredQuarantineInterventionError::ActiveCountPostcondition {
                before: before_active_count,
                after: after_active_count,
            }
            .into());
        }
        let after_active_digest = digest_episodic_memory(self.memory)
            .map_err(|error| AnchoredQuarantineInterventionError::MemoryState(error.to_string()))?;
        let canonical_quarantined = self
            .memory
            .quarantined_instance(self.instance_id)
            .ok_or(AnchoredQuarantineInterventionError::QuarantinePostconditionMissing)?;
        if episode_content_id(&canonical_quarantined.episode)? != content_id {
            return Err(AnchoredQuarantineInterventionError::QuarantinePostconditionMismatch.into());
        }

        // Commit the durable lifecycle ledger. If persistence fails, the UUID remains quarantined
        // in memory and the already-durable write-ahead intent remains pending. That is safer than
        // compensating back to active and removes any blind-retry interpretation.
        let state_before_append = self.quarantine_ledger.clone();
        let quarantine_state_head = self.quarantine_ledger.append_quarantined(
            &self.expected_target_id,
            self.instance_id,
            content_id,
            self.completed_at_unix_s,
            escrow_digest,
            &escrow_persistence_ref,
        )?;
        let quarantine_state_persistence_ref = match self
            .quarantine_persistence
            .persist_quarantine_ledger(self.quarantine_ledger.events(), quarantine_state_head)
        {
            Ok(reference) => reference,
            Err(error) => {
                *self.quarantine_ledger = state_before_append;
                return Err(AnchoredQuarantineExecutionError::QuarantineStatePersistence(
                    error,
                ));
            }
        };
        if let Err(error) = validate_ref(&quarantine_state_persistence_ref) {
            *self.quarantine_ledger = state_before_append;
            return Err(error.into());
        }

        // The state ledger is now durably authoritative. Clear the write-ahead pending intent only
        // by committing an event that binds the exact durable state-ledger head.
        let intent_prepared_state = self.intent_ledger.clone();
        let intent_committed_head = self.intent_ledger.append_committed(
            &self.execution_id,
            &self.expected_target_id,
            self.instance_id,
            content_id,
            self.completed_at_unix_s,
            quarantine_state_head,
        )?;
        let intent_committed_persistence_ref = match self
            .intent_persistence
            .persist_quarantine_intent_ledger(
                self.intent_ledger.events(),
                intent_committed_head,
            ) {
            Ok(reference) => reference,
            Err(error) => {
                // Durable storage still contains Prepared, so mirror that conservative state in
                // memory. Canonical memory and the state ledger remain quarantined.
                *self.intent_ledger = intent_prepared_state;
                return Err(AnchoredQuarantineExecutionError::IntentCommittedPersistence(
                    error,
                ));
            }
        };
        if let Err(error) = validate_ref(&intent_committed_persistence_ref) {
            *self.intent_ledger = intent_prepared_state;
            return Err(error.into());
        }

        let receipt = AnchoredEpisodicQuarantineReceipt {
            target_id: permit.target_id().to_string(),
            execution_id: self.execution_id.clone(),
            instance_id: self.instance_id,
            content_id,
            before_active_count,
            after_active_count,
            before_active_digest,
            after_active_digest,
            escrow_digest,
            escrow_persistence_ref,
            intent_prepared_head,
            intent_prepared_persistence_ref,
            quarantine_state_head,
            quarantine_state_persistence_ref,
            intent_committed_head,
            intent_committed_persistence_ref,
        };
        let result_digest = digest_result(&receipt, permit.rationale())?;
        let evidence_ref = format!(
            "symthaea-memory:anchored-episodic-quarantine:v1:sha256:{}",
            hex_digest(result_digest)
        );
        ReceiptedExecution::new(
            receipt,
            self.completed_at_unix_s,
            result_digest,
            evidence_ref,
        )
        .map_err(AnchoredQuarantineExecutionError::Observation)
    }
}

/// Govern one exact occurrence through write-ahead intent, canonical quarantine, durable
/// quarantine-state commitment, and terminal intent commitment.
#[allow(clippy::too_many_arguments)]
pub fn execute_governed_episodic_quarantine_anchored<P, E, I, S>(
    permit: DurableEvidenceBoundInterventionPermit,
    execution_id: impl Into<String>,
    store_target_id: &str,
    instance_id: EpisodeInstanceId,
    memory: &mut EpisodicMemory,
    current_profile: &MoralPatientEvidenceProfile,
    current_precaution_policy: &PrecautionPolicy,
    consent_ledger: &SubjectConsentLedger,
    subject_registry: &SubjectIdentityRegistry,
    current_authority_manifest: &WelfareAuthorityPolicyManifest,
    current_trust_snapshot: &TrustSnapshot,
    unix_s: u64,
    execution_journal: &mut InterventionExecutionJournal,
    execution_persistence: &mut P,
    escrow_persistence: &mut E,
    intent_ledger: &mut EpisodicQuarantineIntentLedger,
    intent_persistence: &mut I,
    quarantine_ledger: &mut EpisodicQuarantineStateLedger,
    quarantine_persistence: &mut S,
) -> Result<
    JournaledExecutionOutcome<
        AnchoredEpisodicQuarantineReceipt,
        AnchoredQuarantineExecutionError<E::Error, I::Error, S::Error>,
        P::Error,
    >,
    GovernedAnchoredQuarantineError<P::Error>,
>
where
    P: ExecutionJournalPersistence,
    E: EpisodicQuarantineEscrowPersistence,
    I: QuarantineIntentLedgerPersistence,
    S: QuarantineLedgerPersistence,
{
    let execution_id = execution_id.into();
    validate_execution_id(&execution_id).map_err(GovernedAnchoredQuarantineError::Configuration)?;
    let expected_target_id = episodic_instance_target_id(store_target_id, instance_id)
        .map_err(|error| {
            GovernedAnchoredQuarantineError::Configuration(
                AnchoredQuarantineInterventionError::TargetConstruction(error.to_string()),
            )
        })?;
    if permit.action() != SubjectAffectingAction::MemoryModification {
        return Err(GovernedAnchoredQuarantineError::Configuration(
            AnchoredQuarantineInterventionError::WrongAction {
                actual: permit.action(),
            },
        ));
    }
    if permit.target_id() != expected_target_id {
        return Err(GovernedAnchoredQuarantineError::Configuration(
            AnchoredQuarantineInterventionError::WrongTarget {
                expected: expected_target_id,
                actual: permit.target_id().to_string(),
            },
        ));
    }

    let mut executor = AnchoredQuarantineExecutor {
        expected_target_id: permit.target_id().to_string(),
        execution_id: execution_id.clone(),
        instance_id,
        memory,
        escrow_persistence,
        intent_ledger,
        intent_persistence,
        quarantine_ledger,
        quarantine_persistence,
        completed_at_unix_s: unix_s,
    };

    execute_durable_intervention_journaled(
        permit,
        execution_id,
        current_profile,
        current_precaution_policy,
        consent_ledger,
        subject_registry,
        current_authority_manifest,
        current_trust_snapshot,
        unix_s,
        execution_journal,
        execution_persistence,
        &mut executor,
    )
    .map_err(GovernedAnchoredQuarantineError::Gate)
}

fn active_episode(
    memory: &EpisodicMemory,
    instance_id: EpisodeInstanceId,
) -> Result<Episode, AnchoredQuarantineInterventionError> {
    memory
        .get_top_episode_instances(memory.len())
        .into_iter()
        .find_map(|(id, episode)| (id == instance_id).then_some(episode))
        .ok_or_else(|| {
            if memory.quarantined_instance(instance_id).is_some() {
                AnchoredQuarantineInterventionError::AlreadyQuarantined(instance_id)
            } else {
                AnchoredQuarantineInterventionError::InstanceNotActive(instance_id)
            }
        })
}

fn digest_result(
    receipt: &AnchoredEpisodicQuarantineReceipt,
    rationale: &str,
) -> Result<Sha256Digest, AnchoredQuarantineInterventionError> {
    let encoded = bincode::serialize(receipt)
        .map_err(|error| AnchoredQuarantineInterventionError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(RESULT_DOMAIN);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    hash_text(&mut hasher, rationale);
    Ok(hasher.finalize())
}

fn hash_text(hasher: &mut Sha256, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn validate_execution_id(value: &str) -> Result<(), AnchoredQuarantineInterventionError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(char::is_control)
    {
        Err(AnchoredQuarantineInterventionError::InvalidExecutionId)
    } else {
        Ok(())
    }
}

fn validate_ref(value: &str) -> Result<(), AnchoredQuarantineInterventionError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_REF_BYTES
        || value.chars().any(char::is_control)
    {
        Err(AnchoredQuarantineInterventionError::InvalidPersistenceReference)
    } else {
        Ok(())
    }
}

fn hex_digest(digest: Sha256Digest) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest.0 {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[derive(Debug, Error)]
pub enum AnchoredQuarantineInterventionError {
    #[error("anchored episodic quarantine requires MemoryModification authority; actual={actual:?}")]
    WrongAction { actual: SubjectAffectingAction },
    #[error("anchored episodic quarantine target mismatch: expected={expected:?}, actual={actual:?}")]
    WrongTarget { expected: String, actual: String },
    #[error("could not construct exact episodic quarantine target: {0}")]
    TargetConstruction(String),
    #[error("anchored episodic quarantine execution id is invalid")]
    InvalidExecutionId,
    #[error("emergency episodic quarantine is not typed in v1")]
    EmergencyQuarantineNotTyped,
    #[error("episodic quarantine requires explicit granted subject consent")]
    ExplicitConsentRequired,
    #[error("episodic quarantine requires a welfare-review reference")]
    WelfareReviewRequired,
    #[error("episodic quarantine requires an independent-review reference")]
    IndependentReviewRequired,
    #[error("quarantine intent is already pending for occurrence {0}")]
    IntentAlreadyPending(EpisodeInstanceId),
    #[error("quarantine lifecycle ledger already marks occurrence {0} as quarantined")]
    LedgerAlreadyQuarantined(EpisodeInstanceId),
    #[error("episodic occurrence is already quarantined: {0}")]
    AlreadyQuarantined(EpisodeInstanceId),
    #[error("episodic occurrence is not active: {0}")]
    InstanceNotActive(EpisodeInstanceId),
    #[error("episodic occurrence identity disagrees with its storage key")]
    InstanceIdentityMismatch,
    #[error("episodic content identity changed during quarantine")]
    ContentIdentityMismatch,
    #[error("episodic active state changed after escrow/intent capture")]
    PreStateChanged,
    #[error("canonical quarantine postcondition is missing")]
    QuarantinePostconditionMissing,
    #[error("canonical quarantine UUID/content postcondition mismatch")]
    QuarantinePostconditionMismatch,
    #[error("canonical active count postcondition failed: before={before}, after={after}")]
    ActiveCountPostcondition { before: usize, after: usize },
    #[error("canonical episodic-memory mechanism failed: {0}")]
    MemoryMechanism(#[source] EpisodicQuarantineError),
    #[error("episodic-memory state digest failed: {0}")]
    MemoryState(String),
    #[error("persistence returned an invalid durable reference")]
    InvalidPersistenceReference,
    #[error("anchored quarantine receipt encoding failed: {0}")]
    Encoding(String),
    #[error(transparent)]
    ContentIdentity(#[from] EpisodeContentIdError),
    #[error(transparent)]
    IntentLedger(#[from] QuarantineIntentLedgerError),
    #[error(transparent)]
    QuarantineLedger(#[from] QuarantineLedgerError),
    #[error("escrow validation failed: {0}")]
    Escrow(String),
}

impl From<crate::memory_quarantine::EpisodicQuarantineInterventionError>
    for AnchoredQuarantineInterventionError
{
    fn from(value: crate::memory_quarantine::EpisodicQuarantineInterventionError) -> Self {
        Self::Escrow(value.to_string())
    }
}

#[derive(Debug, Error)]
pub enum AnchoredQuarantineExecutionError<E, I, S>
where
    E: StdError + Send + Sync + 'static,
    I: StdError + Send + Sync + 'static,
    S: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Intervention(#[from] AnchoredQuarantineInterventionError),
    #[error("could not persist episodic quarantine escrow: {0}")]
    EscrowPersistence(#[source] E),
    #[error("could not durably persist write-ahead quarantine intent: {0}")]
    IntentPreparedPersistence(#[source] I),
    #[error("could not durably persist quarantine lifecycle state: {0}")]
    QuarantineStatePersistence(#[source] S),
    #[error("could not durably commit quarantine intent: {0}")]
    IntentCommittedPersistence(#[source] I),
    #[error(transparent)]
    Observation(#[from] ExecutionObservationError),
}

#[derive(Debug, Error)]
pub enum GovernedAnchoredQuarantineError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("anchored governed episodic quarantine is misconfigured: {0}")]
    Configuration(#[source] AnchoredQuarantineInterventionError),
    #[error(transparent)]
    Gate(#[from] JournaledExecutionGateError<E>),
}
