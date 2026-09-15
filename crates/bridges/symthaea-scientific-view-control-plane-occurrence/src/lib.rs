//! Observed SCI-014 control-plane occurrence CAS and reconciliation mechanics.
//!
//! This crate deliberately stops before a qualified historical-occurrence theorem:
//!
//! ```text
//! candidate transition
//!     != observed store commit
//!     != qualified historical occurrence
//!     != current control-plane head
//! ```
//!
//! A public backend trait may provide storage observations, but trait conformance
//! alone does not qualify a production persistence source. Consequently this crate
//! returns `Observed...` outcomes only. A later concrete/owner-qualified adapter
//! may upgrade exact observations into a historical occurrence theorem.
//!
//! Architecture: SCI-014R2 / #3439.

#![forbid(unsafe_code)]

pub mod wire;

use sha2::{Digest, Sha256};
use std::error::Error as StdError;
use symthaea_scientific_view_control_plane::CandidateScientificViewControlPlaneTransitionV1;
use symthaea_scientific_view_profile::Commitment32;
use thiserror::Error;

pub const COMMITMENT_ALGORITHM_V1: &str = "sha256";

const MAX_STABLE_ID_BYTES: usize = 192;
const MAX_STORE_REFERENCE_BYTES: usize = 2048;
const STORE_BINDING_DOMAIN: &[u8] =
    b"symthaea.science.view.control-plane.occurrence-store-binding.v1";
const COMMIT_OPERATION_DOMAIN: &[u8] =
    b"symthaea.science.view.control-plane.commit-operation.v1";
const OCCURRENCE_DOMAIN: &[u8] = b"symthaea.science.view.control-plane.occurrence.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlPlaneOccurrenceStoreBindingV1 {
    deployment_id: String,
    view_namespace: String,
    store_namespace: String,
    provisioning_epoch: u64,
    persistence_profile_commitment: Commitment32,
    commitment: Commitment32,
}

impl ControlPlaneOccurrenceStoreBindingV1 {
    pub fn new(
        deployment_id: impl Into<String>,
        view_namespace: impl Into<String>,
        store_namespace: impl Into<String>,
        provisioning_epoch: u64,
        persistence_profile_commitment: Commitment32,
    ) -> Result<Self, ControlPlaneOccurrenceError> {
        let deployment_id = deployment_id.into();
        let view_namespace = view_namespace.into();
        let store_namespace = store_namespace.into();
        validate_stable_id("deployment_id", &deployment_id)?;
        validate_stable_id("view_namespace", &view_namespace)?;
        validate_stable_id("store_namespace", &store_namespace)?;
        if provisioning_epoch == 0 {
            return Err(ControlPlaneOccurrenceError::ProvisioningEpochZero);
        }
        require_nonzero(
            "persistence_profile_commitment",
            persistence_profile_commitment,
        )?;

        let commitment = hash_with(STORE_BINDING_DOMAIN, |hasher| {
            put_text(hasher, &deployment_id);
            put_text(hasher, &view_namespace);
            put_text(hasher, &store_namespace);
            put_u64(hasher, provisioning_epoch);
            put_commitment(hasher, persistence_profile_commitment);
        });

        Ok(Self {
            deployment_id,
            view_namespace,
            store_namespace,
            provisioning_epoch,
            persistence_profile_commitment,
            commitment,
        })
    }

    pub fn deployment_id(&self) -> &str { &self.deployment_id }
    pub fn view_namespace(&self) -> &str { &self.view_namespace }
    pub fn store_namespace(&self) -> &str { &self.store_namespace }
    pub const fn provisioning_epoch(&self) -> u64 { self.provisioning_epoch }
    pub const fn persistence_profile_commitment(&self) -> Commitment32 { self.persistence_profile_commitment }
    pub const fn commitment(&self) -> Commitment32 { self.commitment }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ControlPlaneOccurrenceHeadV1 {
    sequence: u64,
    occurrence_commitment: Commitment32,
}

impl ControlPlaneOccurrenceHeadV1 {
    pub const fn sequence(self) -> u64 { self.sequence }
    pub const fn occurrence_commitment(self) -> Commitment32 { self.occurrence_commitment }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ControlPlaneCommitOperationIdV1(Commitment32);

impl ControlPlaneCommitOperationIdV1 {
    pub const fn commitment(self) -> Commitment32 { self.0 }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposedControlPlaneOccurrenceV1 {
    store_binding_commitment: Commitment32,
    sequence: u64,
    predecessor_occurrence: Option<Commitment32>,
    candidate_transition_commitment: Commitment32,
    operation_id: ControlPlaneCommitOperationIdV1,
    commitment: Commitment32,
}

impl ProposedControlPlaneOccurrenceV1 {
    pub const fn store_binding_commitment(&self) -> Commitment32 { self.store_binding_commitment }
    pub const fn sequence(&self) -> u64 { self.sequence }
    pub const fn predecessor_occurrence(&self) -> Option<Commitment32> { self.predecessor_occurrence }
    pub const fn candidate_transition_commitment(&self) -> Commitment32 { self.candidate_transition_commitment }
    pub const fn operation_id(&self) -> ControlPlaneCommitOperationIdV1 { self.operation_id }
    pub const fn commitment(&self) -> Commitment32 { self.commitment }
    pub const fn head(&self) -> ControlPlaneOccurrenceHeadV1 {
        ControlPlaneOccurrenceHeadV1 {
            sequence: self.sequence,
            occurrence_commitment: self.commitment,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ControlPlaneCommitPlanV1 {
    store_binding: ControlPlaneOccurrenceStoreBindingV1,
    expected_frontier: Option<ControlPlaneOccurrenceHeadV1>,
    occurrence: ProposedControlPlaneOccurrenceV1,
}

impl ControlPlaneCommitPlanV1 {
    pub fn prepare(
        store_binding: ControlPlaneOccurrenceStoreBindingV1,
        candidate: &CandidateScientificViewControlPlaneTransitionV1,
        expected_predecessor: Option<&ProposedControlPlaneOccurrenceV1>,
    ) -> Result<Self, ControlPlaneOccurrenceError> {
        if store_binding.deployment_id() != candidate.deployment_id()
            || store_binding.view_namespace() != candidate.view_namespace()
        {
            return Err(ControlPlaneOccurrenceError::CandidateScopeMismatch);
        }

        let (expected_frontier, predecessor_occurrence) = match expected_predecessor {
            None => {
                if candidate.sequence() != 1 || candidate.predecessor_transition().is_some() {
                    return Err(ControlPlaneOccurrenceError::UnexpectedGenesisCandidate);
                }
                (None, None)
            }
            Some(previous) => {
                if previous.store_binding_commitment() != store_binding.commitment() {
                    return Err(ControlPlaneOccurrenceError::StoreBindingMismatch);
                }
                let expected_sequence = previous
                    .sequence()
                    .checked_add(1)
                    .ok_or(ControlPlaneOccurrenceError::SequenceOverflow)?;
                if candidate.sequence() != expected_sequence {
                    return Err(ControlPlaneOccurrenceError::SequenceMismatch);
                }
                if candidate.predecessor_transition()
                    != Some(previous.candidate_transition_commitment())
                {
                    return Err(ControlPlaneOccurrenceError::CandidatePredecessorMismatch);
                }
                (Some(previous.head()), Some(previous.commitment()))
            }
        };

        let operation_id = ControlPlaneCommitOperationIdV1(hash_with(
            COMMIT_OPERATION_DOMAIN,
            |hasher| {
                put_text(hasher, store_binding.deployment_id());
                put_text(hasher, store_binding.view_namespace());
                put_commitment(hasher, store_binding.commitment());
                put_optional_commitment(hasher, predecessor_occurrence);
                put_commitment(hasher, candidate.commitment());
                put_commitment(hasher, store_binding.persistence_profile_commitment());
            },
        ));

        let occurrence_commitment = hash_with(OCCURRENCE_DOMAIN, |hasher| {
            put_commitment(hasher, store_binding.commitment());
            put_u64(hasher, candidate.sequence());
            put_optional_commitment(hasher, predecessor_occurrence);
            put_commitment(hasher, candidate.commitment());
            put_commitment(hasher, operation_id.commitment());
        });

        let store_binding_commitment = store_binding.commitment();

        Ok(Self {
            store_binding,
            expected_frontier,
            occurrence: ProposedControlPlaneOccurrenceV1 {
                store_binding_commitment,
                sequence: candidate.sequence(),
                predecessor_occurrence,
                candidate_transition_commitment: candidate.commitment(),
                operation_id,
                commitment: occurrence_commitment,
            },
        })
    }

    pub fn store_binding(&self) -> &ControlPlaneOccurrenceStoreBindingV1 { &self.store_binding }
    pub const fn expected_frontier(&self) -> Option<ControlPlaneOccurrenceHeadV1> { self.expected_frontier }
    pub const fn occurrence(&self) -> &ProposedControlPlaneOccurrenceV1 { &self.occurrence }
    pub const fn operation_id(&self) -> ControlPlaneCommitOperationIdV1 { self.occurrence.operation_id }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawControlPlaneOccurrenceRecordV1 {
    occurrence: ProposedControlPlaneOccurrenceV1,
    store_reference: String,
}

impl RawControlPlaneOccurrenceRecordV1 {
    pub fn new_unqualified(
        occurrence: ProposedControlPlaneOccurrenceV1,
        store_reference: impl Into<String>,
    ) -> Result<Self, ControlPlaneOccurrenceError> {
        let store_reference = store_reference.into();
        validate_store_reference(&store_reference)?;
        Ok(Self { occurrence, store_reference })
    }
    pub const fn occurrence(&self) -> &ProposedControlPlaneOccurrenceV1 { &self.occurrence }
    pub fn store_reference(&self) -> &str { &self.store_reference }
    pub const fn head(&self) -> ControlPlaneOccurrenceHeadV1 { self.occurrence.head() }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlPlaneStoreCasResultV1 {
    Applied { store_reference: String },
    Conflict { actual_frontier: Option<ControlPlaneOccurrenceHeadV1> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlPlaneStoreOperationResolutionV1 {
    Found(RawControlPlaneOccurrenceRecordV1),
    ProvenAbsent { current_frontier: Option<ControlPlaneOccurrenceHeadV1> },
}

pub trait ControlPlaneOccurrenceStoreV1 {
    type Error: StdError + Send + Sync + 'static;

    fn load_frontier(
        &self,
        binding: &ControlPlaneOccurrenceStoreBindingV1,
    ) -> Result<Option<RawControlPlaneOccurrenceRecordV1>, Self::Error>;

    fn compare_and_swap(
        &mut self,
        binding: &ControlPlaneOccurrenceStoreBindingV1,
        expected: Option<ControlPlaneOccurrenceHeadV1>,
        proposed: &ProposedControlPlaneOccurrenceV1,
    ) -> Result<ControlPlaneStoreCasResultV1, Self::Error>;

    fn resolve_operation(
        &self,
        binding: &ControlPlaneOccurrenceStoreBindingV1,
        operation_id: ControlPlaneCommitOperationIdV1,
    ) -> Result<ControlPlaneStoreOperationResolutionV1, Self::Error>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedExactControlPlaneCommitV1 {
    occurrence: ProposedControlPlaneOccurrenceV1,
    store_reference: String,
}

impl ObservedExactControlPlaneCommitV1 {
    pub const fn occurrence(&self) -> &ProposedControlPlaneOccurrenceV1 { &self.occurrence }
    pub fn store_reference(&self) -> &str { &self.store_reference }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservedNonCommitReasonV1 {
    PreflightFrontierMismatch,
    CasConflict,
    ReconciledAbsent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedControlPlaneNonCommitV1 {
    operation_id: ControlPlaneCommitOperationIdV1,
    reason: ObservedNonCommitReasonV1,
    actual_frontier: Option<ControlPlaneOccurrenceHeadV1>,
}

impl ObservedControlPlaneNonCommitV1 {
    pub const fn operation_id(&self) -> ControlPlaneCommitOperationIdV1 { self.operation_id }
    pub const fn reason(&self) -> ObservedNonCommitReasonV1 { self.reason }
    pub const fn actual_frontier(&self) -> Option<ControlPlaneOccurrenceHeadV1> { self.actual_frontier }
}

#[derive(Debug, Error)]
pub enum ObservedControlPlaneCommitAmbiguityReasonV1<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("control-plane occurrence CAS acknowledgement failed: {0}")]
    CompareAndSwap(#[source] E),
    #[error("control-plane occurrence CAS returned an invalid store reference")]
    InvalidStoreReference,
    #[error("control-plane occurrence post-CAS readback failed: {0}")]
    Readback(#[source] E),
    #[error("control-plane occurrence disappeared during post-CAS readback")]
    MissingReadback,
    #[error("control-plane occurrence post-CAS readback differed from the proposal")]
    MismatchedReadback,
    #[error("control-plane occurrence conflict follow-up resolution failed: {0}")]
    ConflictResolution(#[source] E),
    #[error("control-plane occurrence conflict follow-up found a mismatched operation record")]
    ConflictResolutionMismatch,
    #[error("control-plane occurrence conflict and operation-resolution frontier disagree")]
    ConflictResolutionFrontierMismatch,
    #[error("control-plane occurrence reconciliation failed: {0}")]
    Reconciliation(#[source] E),
    #[error("control-plane occurrence reconciliation found a mismatched operation record")]
    ReconciliationMismatch,
}

#[derive(Debug)]
pub struct ObservedControlPlaneCommitAmbiguityV1<E>
where
    E: StdError + Send + Sync + 'static,
{
    plan: ControlPlaneCommitPlanV1,
    reason: ObservedControlPlaneCommitAmbiguityReasonV1<E>,
}

impl<E> ObservedControlPlaneCommitAmbiguityV1<E>
where
    E: StdError + Send + Sync + 'static,
{
    pub const fn operation_id(&self) -> ControlPlaneCommitOperationIdV1 { self.plan.operation_id() }
    pub fn reason(&self) -> &ObservedControlPlaneCommitAmbiguityReasonV1<E> { &self.reason }
}

#[derive(Debug)]
pub enum ObservedControlPlaneCommitOutcomeV1<E>
where
    E: StdError + Send + Sync + 'static,
{
    CommittedObserved(ObservedExactControlPlaneCommitV1),
    ProvenNotCommitted(ObservedControlPlaneNonCommitV1),
    OutcomeUnknown(ObservedControlPlaneCommitAmbiguityV1<E>),
}

impl<E> ObservedControlPlaneCommitOutcomeV1<E>
where
    E: StdError + Send + Sync + 'static,
{
    pub fn requires_reconciliation(&self) -> bool { matches!(self, Self::OutcomeUnknown(_)) }
}

fn observed_exact(record: RawControlPlaneOccurrenceRecordV1) -> ObservedExactControlPlaneCommitV1 {
    ObservedExactControlPlaneCommitV1 {
        occurrence: record.occurrence,
        store_reference: record.store_reference,
    }
}

pub fn attempt_observed_control_plane_commit<S>(
    store: &mut S,
    plan: ControlPlaneCommitPlanV1,
) -> Result<ObservedControlPlaneCommitOutcomeV1<S::Error>, ControlPlaneCommitProtocolError<S::Error>>
where
    S: ControlPlaneOccurrenceStoreV1,
{
    let current = store
        .load_frontier(plan.store_binding())
        .map_err(ControlPlaneCommitProtocolError::PreflightLoad)?;
    let loaded_frontier = current.as_ref().map(RawControlPlaneOccurrenceRecordV1::head);

    let resolved = store
        .resolve_operation(plan.store_binding(), plan.operation_id())
        .map_err(ControlPlaneCommitProtocolError::PreflightResolve)?;

    match resolved {
        ControlPlaneStoreOperationResolutionV1::Found(record) => {
            if record.occurrence() != plan.occurrence() {
                return Err(ControlPlaneCommitProtocolError::PreflightOperationMismatch);
            }
            return Ok(ObservedControlPlaneCommitOutcomeV1::CommittedObserved(
                observed_exact(record),
            ));
        }
        ControlPlaneStoreOperationResolutionV1::ProvenAbsent { current_frontier } => {
            if current_frontier != loaded_frontier {
                return Err(ControlPlaneCommitProtocolError::PreflightObservationChanged {
                    loaded_frontier,
                    resolved_frontier: current_frontier,
                });
            }
            if current
                .as_ref()
                .is_some_and(|record| record.occurrence() == plan.occurrence())
            {
                return Err(ControlPlaneCommitProtocolError::PreflightOperationIndexMissingCurrentOccurrence);
            }
            if loaded_frontier != plan.expected_frontier() {
                return Ok(ObservedControlPlaneCommitOutcomeV1::ProvenNotCommitted(
                    ObservedControlPlaneNonCommitV1 {
                        operation_id: plan.operation_id(),
                        reason: ObservedNonCommitReasonV1::PreflightFrontierMismatch,
                        actual_frontier: loaded_frontier,
                    },
                ));
            }
        }
    }

    let ack = match store.compare_and_swap(
        plan.store_binding(),
        plan.expected_frontier(),
        plan.occurrence(),
    ) {
        Ok(result) => result,
        Err(error) => {
            return Ok(ObservedControlPlaneCommitOutcomeV1::OutcomeUnknown(
                ObservedControlPlaneCommitAmbiguityV1 {
                    plan,
                    reason: ObservedControlPlaneCommitAmbiguityReasonV1::CompareAndSwap(error),
                },
            ));
        }
    };

    match ack {
        ControlPlaneStoreCasResultV1::Conflict { actual_frontier } => {
            match store.resolve_operation(plan.store_binding(), plan.operation_id()) {
                Ok(ControlPlaneStoreOperationResolutionV1::Found(record)) => {
                    if record.occurrence() == plan.occurrence() {
                        Ok(ObservedControlPlaneCommitOutcomeV1::CommittedObserved(
                            observed_exact(record),
                        ))
                    } else {
                        Ok(ObservedControlPlaneCommitOutcomeV1::OutcomeUnknown(
                            ObservedControlPlaneCommitAmbiguityV1 {
                                plan,
                                reason: ObservedControlPlaneCommitAmbiguityReasonV1::ConflictResolutionMismatch,
                            },
                        ))
                    }
                }
                Ok(ControlPlaneStoreOperationResolutionV1::ProvenAbsent { current_frontier }) => {
                    if current_frontier != actual_frontier {
                        return Ok(ObservedControlPlaneCommitOutcomeV1::OutcomeUnknown(
                            ObservedControlPlaneCommitAmbiguityV1 {
                                plan,
                                reason: ObservedControlPlaneCommitAmbiguityReasonV1::ConflictResolutionFrontierMismatch,
                            },
                        ));
                    }
                    Ok(ObservedControlPlaneCommitOutcomeV1::ProvenNotCommitted(
                        ObservedControlPlaneNonCommitV1 {
                            operation_id: plan.operation_id(),
                            reason: ObservedNonCommitReasonV1::CasConflict,
                            actual_frontier: current_frontier,
                        },
                    ))
                }
                Err(error) => Ok(ObservedControlPlaneCommitOutcomeV1::OutcomeUnknown(
                    ObservedControlPlaneCommitAmbiguityV1 {
                        plan,
                        reason: ObservedControlPlaneCommitAmbiguityReasonV1::ConflictResolution(error),
                    },
                )),
            }
        }
        ControlPlaneStoreCasResultV1::Applied { store_reference } => {
            if validate_store_reference(&store_reference).is_err() {
                return Ok(ObservedControlPlaneCommitOutcomeV1::OutcomeUnknown(
                    ObservedControlPlaneCommitAmbiguityV1 {
                        plan,
                        reason: ObservedControlPlaneCommitAmbiguityReasonV1::InvalidStoreReference,
                    },
                ));
            }

            let readback = match store.load_frontier(plan.store_binding()) {
                Ok(Some(record)) => record,
                Ok(None) => {
                    return Ok(ObservedControlPlaneCommitOutcomeV1::OutcomeUnknown(
                        ObservedControlPlaneCommitAmbiguityV1 {
                            plan,
                            reason: ObservedControlPlaneCommitAmbiguityReasonV1::MissingReadback,
                        },
                    ));
                }
                Err(error) => {
                    return Ok(ObservedControlPlaneCommitOutcomeV1::OutcomeUnknown(
                        ObservedControlPlaneCommitAmbiguityV1 {
                            plan,
                            reason: ObservedControlPlaneCommitAmbiguityReasonV1::Readback(error),
                        },
                    ));
                }
            };

            if readback.occurrence() != plan.occurrence()
                || readback.store_reference() != store_reference
            {
                return Ok(ObservedControlPlaneCommitOutcomeV1::OutcomeUnknown(
                    ObservedControlPlaneCommitAmbiguityV1 {
                        plan,
                        reason: ObservedControlPlaneCommitAmbiguityReasonV1::MismatchedReadback,
                    },
                ));
            }

            Ok(ObservedControlPlaneCommitOutcomeV1::CommittedObserved(
                observed_exact(readback),
            ))
        }
    }
}

pub fn reconcile_observed_control_plane_commit<S>(
    store: &S,
    ambiguity: ObservedControlPlaneCommitAmbiguityV1<S::Error>,
) -> ObservedControlPlaneCommitOutcomeV1<S::Error>
where
    S: ControlPlaneOccurrenceStoreV1,
{
    let ObservedControlPlaneCommitAmbiguityV1 { plan, .. } = ambiguity;
    match store.resolve_operation(plan.store_binding(), plan.operation_id()) {
        Ok(ControlPlaneStoreOperationResolutionV1::Found(record)) => {
            if record.occurrence() != plan.occurrence() {
                return ObservedControlPlaneCommitOutcomeV1::OutcomeUnknown(
                    ObservedControlPlaneCommitAmbiguityV1 {
                        plan,
                        reason: ObservedControlPlaneCommitAmbiguityReasonV1::ReconciliationMismatch,
                    },
                );
            }
            ObservedControlPlaneCommitOutcomeV1::CommittedObserved(observed_exact(record))
        }
        Ok(ControlPlaneStoreOperationResolutionV1::ProvenAbsent { current_frontier }) => {
            ObservedControlPlaneCommitOutcomeV1::ProvenNotCommitted(
                ObservedControlPlaneNonCommitV1 {
                    operation_id: plan.operation_id(),
                    reason: ObservedNonCommitReasonV1::ReconciledAbsent,
                    actual_frontier: current_frontier,
                },
            )
        }
        Err(error) => ObservedControlPlaneCommitOutcomeV1::OutcomeUnknown(
            ObservedControlPlaneCommitAmbiguityV1 {
                plan,
                reason: ObservedControlPlaneCommitAmbiguityReasonV1::Reconciliation(error),
            },
        ),
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ControlPlaneOccurrenceError {
    #[error("invalid stable identifier in field {field}")]
    InvalidStableId { field: &'static str },
    #[error("zero commitment in field {field}")]
    ZeroCommitment { field: &'static str },
    #[error("store provisioning epoch must be nonzero")]
    ProvisioningEpochZero,
    #[error("candidate deployment/view scope does not match occurrence-store binding")]
    CandidateScopeMismatch,
    #[error("candidate has unexpected genesis shape")]
    UnexpectedGenesisCandidate,
    #[error("occurrence store binding does not match predecessor occurrence")]
    StoreBindingMismatch,
    #[error("occurrence sequence overflow")]
    SequenceOverflow,
    #[error("candidate transition sequence does not follow predecessor occurrence")]
    SequenceMismatch,
    #[error("candidate transition predecessor does not match predecessor occurrence candidate")]
    CandidatePredecessorMismatch,
    #[error("invalid store evidence reference")]
    InvalidStoreReference,
}

#[derive(Debug, Error)]
pub enum ControlPlaneCommitProtocolError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("control-plane occurrence preflight frontier load failed: {0}")]
    PreflightLoad(#[source] E),
    #[error("control-plane occurrence preflight operation resolution failed: {0}")]
    PreflightResolve(#[source] E),
    #[error("control-plane occurrence preflight operation id resolved to a different occurrence")]
    PreflightOperationMismatch,
    #[error("control-plane occurrence frontier contains the exact occurrence but operation index reports it absent")]
    PreflightOperationIndexMissingCurrentOccurrence,
    #[error("control-plane occurrence preflight observations changed between frontier load and operation resolution: loaded={loaded_frontier:?}, resolved={resolved_frontier:?}")]
    PreflightObservationChanged {
        loaded_frontier: Option<ControlPlaneOccurrenceHeadV1>,
        resolved_frontier: Option<ControlPlaneOccurrenceHeadV1>,
    },
}

fn validate_stable_id(field: &'static str, value: &str) -> Result<(), ControlPlaneOccurrenceError> {
    if value.is_empty()
        || value.len() > MAX_STABLE_ID_BYTES
        || !value.is_ascii()
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'_' | b'-' | b'/' | b':')
        })
    {
        return Err(ControlPlaneOccurrenceError::InvalidStableId { field });
    }
    Ok(())
}

fn validate_store_reference(value: &str) -> Result<(), ControlPlaneOccurrenceError> {
    if value.is_empty()
        || value != value.trim()
        || value.len() > MAX_STORE_REFERENCE_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(ControlPlaneOccurrenceError::InvalidStoreReference);
    }
    Ok(())
}

fn require_nonzero(field: &'static str, commitment: Commitment32) -> Result<(), ControlPlaneOccurrenceError> {
    if commitment.is_zero() {
        return Err(ControlPlaneOccurrenceError::ZeroCommitment { field });
    }
    Ok(())
}

fn hash_with(domain: &[u8], write_fields: impl FnOnce(&mut Sha256)) -> Commitment32 {
    let mut hasher = Sha256::new();
    put_bytes(&mut hasher, domain);
    write_fields(&mut hasher);
    let digest: [u8; 32] = hasher.finalize().into();
    Commitment32::from_bytes(digest)
}

fn put_bytes(hasher: &mut Sha256, value: &[u8]) {
    put_u64(hasher, value.len() as u64);
    hasher.update(value);
}
fn put_text(hasher: &mut Sha256, value: &str) { put_bytes(hasher, value.as_bytes()); }
fn put_u64(hasher: &mut Sha256, value: u64) { hasher.update(value.to_be_bytes()); }
fn put_u8(hasher: &mut Sha256, value: u8) { hasher.update([value]); }
fn put_commitment(hasher: &mut Sha256, value: Commitment32) { hasher.update(value.as_bytes()); }
fn put_optional_commitment(hasher: &mut Sha256, value: Option<Commitment32>) {
    match value {
        None => put_u8(hasher, 0),
        Some(value) => {
            put_u8(hasher, 1);
            put_commitment(hasher, value);
        }
    }
}
