// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Authorized mutation boundary for epistemic-support state.
//!
//! EKM-024 decides whether a bounded belief revision is eligible under policy and
//! EKM-025 records the exact decision inputs. This module is the first layer in
//! the belief-revision series that may actually change epistemic-support state.
//!
//! The mutation remains isolated from the legacy `EnhancedKnowledgeGraph` and
//! `TemporalFact::confidence`. It operates only on an explicit
//! [`EpistemicSupportStore`] and preserves exact pre/post receipts.
//!
//! Authorization records in this module are typed approvals, not cryptographic
//! authentication. Signature/identity verification belongs to a higher authority
//! layer.

use super::belief_revision_gate::{
    BeliefRevisionFailure, BeliefRevisionGate, EpistemicRevisionProposal,
};
use super::belief_revision_receipt::{
    BeliefRevisionReceipt, BeliefRevisionReceiptId, RevisionEvidenceSnapshot,
};
use super::claim_evidence::{ClaimId, EpistemicLedger, EvidenceId};
use super::knowledge_weight_routing::{BoundedWeight, KnowledgeWeightError};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BeliefMutationReceiptId(pub u64);

/// Explicit epistemic-support state for one claim.
///
/// A state exists only after an explicit baseline is registered. Missing state is
/// therefore distinct from support=0.0.
#[derive(Debug, Clone, PartialEq)]
pub struct EpistemicSupportState {
    claim_id: ClaimId,
    support: BoundedWeight,
    revision: u64,
    initialized_at_cycle: u64,
    last_updated_cycle: u64,
    last_mutation_id: Option<BeliefMutationReceiptId>,
}

impl EpistemicSupportState {
    pub fn claim_id(&self) -> ClaimId {
        self.claim_id
    }

    pub fn support(&self) -> BoundedWeight {
        self.support
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn initialized_at_cycle(&self) -> u64 {
        self.initialized_at_cycle
    }

    pub fn last_updated_cycle(&self) -> u64 {
        self.last_updated_cycle
    }

    pub fn last_mutation_id(&self) -> Option<BeliefMutationReceiptId> {
        self.last_mutation_id
    }
}

#[derive(Debug, Clone, PartialEq)]
struct AuthorizationBinding {
    receipt_id: BeliefRevisionReceiptId,
    claim_id: ClaimId,
    proposed_delta: f32,
    expected_state_revision: u64,
    expected_support: BoundedWeight,
    approved_at_cycle: u64,
    authority_label: String,
}

/// Immutable receipt for one applied epistemic-support mutation.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefMutationReceipt {
    id: BeliefMutationReceiptId,
    source_revision_receipt_id: BeliefRevisionReceiptId,
    claim_id: ClaimId,
    proposed_delta: f32,
    support_before: BoundedWeight,
    support_after: BoundedWeight,
    state_revision_before: u64,
    state_revision_after: u64,
    authorization_id: String,
    authority_label: String,
    authorized_at_cycle: u64,
    applied_at_cycle: u64,
}

impl BeliefMutationReceipt {
    pub fn id(&self) -> BeliefMutationReceiptId {
        self.id
    }

    pub fn source_revision_receipt_id(&self) -> BeliefRevisionReceiptId {
        self.source_revision_receipt_id
    }

    pub fn claim_id(&self) -> ClaimId {
        self.claim_id
    }

    pub fn proposed_delta(&self) -> f32 {
        self.proposed_delta
    }

    pub fn support_before(&self) -> BoundedWeight {
        self.support_before
    }

    pub fn support_after(&self) -> BoundedWeight {
        self.support_after
    }

    pub fn state_revision_before(&self) -> u64 {
        self.state_revision_before
    }

    pub fn state_revision_after(&self) -> u64 {
        self.state_revision_after
    }

    pub fn authorization_id(&self) -> &str {
        &self.authorization_id
    }

    pub fn authority_label(&self) -> &str {
        &self.authority_label
    }

    pub fn authorized_at_cycle(&self) -> u64 {
        self.authorized_at_cycle
    }

    pub fn applied_at_cycle(&self) -> u64 {
        self.applied_at_cycle
    }

    /// Generate a non-executing exact rollback plan.
    ///
    /// Rollback itself is intentionally a separate future authority boundary.
    pub fn rollback_plan(
        &self,
        generated_at_cycle: u64,
    ) -> Result<BeliefMutationRollbackPlan, BeliefMutationError> {
        if generated_at_cycle < self.applied_at_cycle {
            return Err(BeliefMutationError::RollbackPlanPredatesMutation {
                generated_at_cycle,
                applied_at_cycle: self.applied_at_cycle,
            });
        }
        Ok(BeliefMutationRollbackPlan {
            source_mutation_id: self.id,
            claim_id: self.claim_id,
            restore_support: self.support_before,
            expected_current_support: self.support_after,
            expected_current_revision: self.state_revision_after,
            generated_at_cycle,
        })
    }
}

/// Exact, non-executing reversal description for one mutation.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefMutationRollbackPlan {
    source_mutation_id: BeliefMutationReceiptId,
    claim_id: ClaimId,
    restore_support: BoundedWeight,
    expected_current_support: BoundedWeight,
    expected_current_revision: u64,
    generated_at_cycle: u64,
}

impl BeliefMutationRollbackPlan {
    pub fn source_mutation_id(&self) -> BeliefMutationReceiptId {
        self.source_mutation_id
    }

    pub fn claim_id(&self) -> ClaimId {
        self.claim_id
    }

    pub fn restore_support(&self) -> BoundedWeight {
        self.restore_support
    }

    pub fn expected_current_support(&self) -> BoundedWeight {
        self.expected_current_support
    }

    pub fn expected_current_revision(&self) -> u64 {
        self.expected_current_revision
    }

    pub fn generated_at_cycle(&self) -> u64 {
        self.generated_at_cycle
    }
}

/// Isolated epistemic-support state plus append-only mutation history.
#[derive(Debug, Clone)]
pub struct EpistemicSupportStore {
    states: HashMap<ClaimId, EpistemicSupportState>,
    history: Vec<BeliefMutationReceipt>,
    authorization_bindings: HashMap<String, AuthorizationBinding>,
    next_mutation_id: u64,
}

impl Default for EpistemicSupportStore {
    fn default() -> Self {
        Self::new()
    }
}

impl EpistemicSupportStore {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            history: Vec::new(),
            authorization_bindings: HashMap::new(),
            next_mutation_id: 1,
        }
    }

    pub fn register_claim(
        &mut self,
        ledger: &EpistemicLedger,
        claim_id: ClaimId,
        support: BoundedWeight,
        initialized_at_cycle: u64,
    ) -> Result<(), BeliefMutationError> {
        let claim = ledger
            .claim(claim_id)
            .ok_or(BeliefMutationError::UnknownClaim(claim_id))?;
        if initialized_at_cycle < claim.created_at_cycle {
            return Err(BeliefMutationError::InitializationPredatesClaim {
                initialized_at_cycle,
                claim_created_at_cycle: claim.created_at_cycle,
            });
        }
        if self.states.contains_key(&claim_id) {
            return Err(BeliefMutationError::StateAlreadyRegistered(claim_id));
        }
        self.states.insert(
            claim_id,
            EpistemicSupportState {
                claim_id,
                support,
                revision: 0,
                initialized_at_cycle,
                last_updated_cycle: initialized_at_cycle,
                last_mutation_id: None,
            },
        );
        Ok(())
    }

    pub fn state(&self, claim_id: ClaimId) -> Option<&EpistemicSupportState> {
        self.states.get(&claim_id)
    }

    pub fn history(&self) -> &[BeliefMutationReceipt] {
        &self.history
    }

    pub fn mutation(&self, id: BeliefMutationReceiptId) -> Option<&BeliefMutationReceipt> {
        self.history.iter().find(|receipt| receipt.id == id)
    }

    pub fn mutation_for_revision_receipt(
        &self,
        receipt_id: BeliefRevisionReceiptId,
    ) -> Option<&BeliefMutationReceipt> {
        self.history
            .iter()
            .find(|receipt| receipt.source_revision_receipt_id == receipt_id)
    }

    pub fn consumed_authorization_count(&self) -> usize {
        self.authorization_bindings.len()
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    fn authorization_binding(&self, authorization_id: &str) -> Option<&AuthorizationBinding> {
        self.authorization_bindings.get(authorization_id)
    }

    fn record_authorization_binding(
        &mut self,
        authorization_id: &str,
        binding: AuthorizationBinding,
    ) -> Result<(), BeliefMutationError> {
        if let Some(existing) = self.authorization_bindings.get(authorization_id) {
            if existing != &binding {
                return Err(BeliefMutationError::AuthorizationReplayMismatch {
                    authorization_id: authorization_id.to_string(),
                });
            }
            return Ok(());
        }
        self.authorization_bindings
            .insert(authorization_id.to_string(), binding);
        Ok(())
    }

    fn commit(
        &mut self,
        receipt: BeliefMutationReceipt,
        binding: AuthorizationBinding,
    ) -> Result<(), BeliefMutationError> {
        let state = self
            .states
            .get_mut(&receipt.claim_id)
            .ok_or(BeliefMutationError::StateNotRegistered(receipt.claim_id))?;
        if state.revision != receipt.state_revision_before
            || state.support != receipt.support_before
        {
            return Err(BeliefMutationError::StateChangedBeforeCommit {
                expected_revision: receipt.state_revision_before,
                actual_revision: state.revision,
            });
        }
        if let Some(existing) = self.authorization_bindings.get(&receipt.authorization_id) {
            if existing != &binding {
                return Err(BeliefMutationError::AuthorizationReplayMismatch {
                    authorization_id: receipt.authorization_id.clone(),
                });
            }
        }

        state.support = receipt.support_after;
        state.revision = receipt.state_revision_after;
        state.last_updated_cycle = receipt.applied_at_cycle;
        state.last_mutation_id = Some(receipt.id);
        self.next_mutation_id = self.next_mutation_id.max(receipt.id.0 + 1);
        self.authorization_bindings
            .insert(receipt.authorization_id.clone(), binding);
        self.history.push(receipt);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefMutationAuthorizationDecision {
    Approved,
    Denied,
}

/// Typed authorization bound to one eligible revision receipt and one exact
/// pre-mutation state version/value.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefMutationAuthorization {
    authorization_id: String,
    authority_label: String,
    decision: BeliefMutationAuthorizationDecision,
    source_revision_receipt_id: BeliefRevisionReceiptId,
    claim_id: ClaimId,
    proposed_delta: f32,
    expected_state_revision: u64,
    expected_support: BoundedWeight,
    approved_at_cycle: u64,
}

impl BeliefMutationAuthorization {
    pub fn new(
        authorization_id: impl Into<String>,
        authority_label: impl Into<String>,
        decision: BeliefMutationAuthorizationDecision,
        approved_at_cycle: u64,
        receipt: &BeliefRevisionReceipt,
        state: &EpistemicSupportState,
    ) -> Result<Self, BeliefMutationError> {
        let authorization_id = authorization_id.into();
        if authorization_id.trim().is_empty() {
            return Err(BeliefMutationError::EmptyAuthorizationId);
        }
        let authority_label = authority_label.into();
        if authority_label.trim().is_empty() {
            return Err(BeliefMutationError::EmptyAuthorityLabel);
        }
        if state.claim_id != receipt.claim_id() {
            return Err(BeliefMutationError::AuthorizationStateClaimMismatch {
                state_claim: state.claim_id,
                receipt_claim: receipt.claim_id(),
            });
        }
        if approved_at_cycle < receipt.evaluated_at_cycle() {
            return Err(BeliefMutationError::AuthorizationPredatesRevisionDecision {
                approved_at_cycle,
                evaluated_at_cycle: receipt.evaluated_at_cycle(),
            });
        }
        Ok(Self {
            authorization_id,
            authority_label,
            decision,
            source_revision_receipt_id: receipt.id(),
            claim_id: receipt.claim_id(),
            proposed_delta: receipt.proposed_delta(),
            expected_state_revision: state.revision,
            expected_support: state.support,
            approved_at_cycle,
        })
    }

    pub fn authorization_id(&self) -> &str {
        &self.authorization_id
    }

    pub fn authority_label(&self) -> &str {
        &self.authority_label
    }

    pub fn decision(&self) -> BeliefMutationAuthorizationDecision {
        self.decision
    }

    pub fn source_revision_receipt_id(&self) -> BeliefRevisionReceiptId {
        self.source_revision_receipt_id
    }

    pub fn approved_at_cycle(&self) -> u64 {
        self.approved_at_cycle
    }

    fn binding(&self) -> AuthorizationBinding {
        AuthorizationBinding {
            receipt_id: self.source_revision_receipt_id,
            claim_id: self.claim_id,
            proposed_delta: self.proposed_delta,
            expected_state_revision: self.expected_state_revision,
            expected_support: self.expected_support,
            approved_at_cycle: self.approved_at_cycle,
            authority_label: self.authority_label.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefMutationOutcome {
    Applied(BeliefMutationReceipt),
    AlreadyApplied(BeliefMutationReceipt),
}

impl BeliefMutationOutcome {
    pub fn receipt(&self) -> &BeliefMutationReceipt {
        match self {
            Self::Applied(receipt) | Self::AlreadyApplied(receipt) => receipt,
        }
    }

    pub fn applied_new_revision(&self) -> bool {
        matches!(self, Self::Applied(_))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefMutationError {
    EmptyAuthorizationId,
    EmptyAuthorityLabel,
    UnknownClaim(ClaimId),
    StateAlreadyRegistered(ClaimId),
    StateNotRegistered(ClaimId),
    InitializationPredatesClaim {
        initialized_at_cycle: u64,
        claim_created_at_cycle: u64,
    },
    RevisionReceiptRejected(BeliefRevisionReceiptId),
    RevisionReceiptHasDuplicateEvidence(Vec<EvidenceId>),
    AuthorizationDenied,
    AuthorizationReceiptMismatch,
    AuthorizationStateClaimMismatch {
        state_claim: ClaimId,
        receipt_claim: ClaimId,
    },
    AuthorizationPredatesRevisionDecision {
        approved_at_cycle: u64,
        evaluated_at_cycle: u64,
    },
    MutationPredatesAuthorization {
        mutation_cycle: u64,
        approved_at_cycle: u64,
    },
    MutationPredatesReceiptEvaluation {
        mutation_cycle: u64,
        evaluated_at_cycle: u64,
    },
    AuthorizationReplayMismatch {
        authorization_id: String,
    },
    StaleStateRevision {
        expected: u64,
        actual: u64,
    },
    StaleStateSupport {
        expected: f32,
        actual: f32,
    },
    ReceiptPredatesCurrentState {
        evaluated_at_cycle: u64,
        state_last_updated_cycle: u64,
    },
    EvidencePostdatesRevisionDecision {
        latest_evidence_cycle: u64,
        evaluated_at_cycle: u64,
    },
    MissingBasisEvidence(EvidenceId),
    BasisEvidenceChanged(EvidenceId),
    RevisionDecisionChanged(Vec<BeliefRevisionFailure>),
    Weight(KnowledgeWeightError),
    StateChangedBeforeCommit {
        expected_revision: u64,
        actual_revision: u64,
    },
    RollbackPlanPredatesMutation {
        generated_at_cycle: u64,
        applied_at_cycle: u64,
    },
}

impl fmt::Display for BeliefMutationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyAuthorizationId => {
                write!(f, "belief mutation authorization id cannot be empty")
            }
            Self::EmptyAuthorityLabel => write!(f, "belief mutation authority label cannot be empty"),
            Self::UnknownClaim(id) => write!(f, "unknown claim {}", id.0),
            Self::StateAlreadyRegistered(id) => write!(
                f,
                "epistemic support state already registered for claim {}",
                id.0
            ),
            Self::StateNotRegistered(id) => write!(
                f,
                "epistemic support state is not registered for claim {}",
                id.0
            ),
            Self::InitializationPredatesClaim {
                initialized_at_cycle,
                claim_created_at_cycle,
            } => write!(
                f,
                "state initialization cycle {initialized_at_cycle} predates claim creation cycle {claim_created_at_cycle}"
            ),
            Self::RevisionReceiptRejected(id) => {
                write!(f, "belief revision receipt {} was not eligible", id.0)
            }
            Self::RevisionReceiptHasDuplicateEvidence(ids) => write!(
                f,
                "belief revision receipt contains duplicate basis evidence attempts: {ids:?}"
            ),
            Self::AuthorizationDenied => write!(f, "belief mutation authorization was denied"),
            Self::AuthorizationReceiptMismatch => write!(
                f,
                "belief mutation authorization is bound to a different revision receipt or delta"
            ),
            Self::AuthorizationStateClaimMismatch {
                state_claim,
                receipt_claim,
            } => write!(
                f,
                "authorization state claim {} does not match receipt claim {}",
                state_claim.0, receipt_claim.0
            ),
            Self::AuthorizationPredatesRevisionDecision {
                approved_at_cycle,
                evaluated_at_cycle,
            } => write!(
                f,
                "authorization cycle {approved_at_cycle} predates revision decision cycle {evaluated_at_cycle}"
            ),
            Self::MutationPredatesAuthorization {
                mutation_cycle,
                approved_at_cycle,
            } => write!(
                f,
                "mutation cycle {mutation_cycle} predates authorization cycle {approved_at_cycle}"
            ),
            Self::MutationPredatesReceiptEvaluation {
                mutation_cycle,
                evaluated_at_cycle,
            } => write!(
                f,
                "mutation cycle {mutation_cycle} predates revision evaluation cycle {evaluated_at_cycle}"
            ),
            Self::AuthorizationReplayMismatch { authorization_id } => write!(
                f,
                "authorization '{authorization_id}' was previously consumed for a different belief mutation"
            ),
            Self::StaleStateRevision { expected, actual } => write!(
                f,
                "authorized state revision {expected} is stale; current revision is {actual}"
            ),
            Self::StaleStateSupport { expected, actual } => write!(
                f,
                "authorized support {expected} is stale; current support is {actual}"
            ),
            Self::ReceiptPredatesCurrentState {
                evaluated_at_cycle,
                state_last_updated_cycle,
            } => write!(
                f,
                "revision receipt evaluated at cycle {evaluated_at_cycle} predates current state update at cycle {state_last_updated_cycle}"
            ),
            Self::EvidencePostdatesRevisionDecision {
                latest_evidence_cycle,
                evaluated_at_cycle,
            } => write!(
                f,
                "claim evidence observed at cycle {latest_evidence_cycle} postdates revision decision cycle {evaluated_at_cycle}"
            ),
            Self::MissingBasisEvidence(id) => {
                write!(f, "basis evidence {} no longer exists", id.0)
            }
            Self::BasisEvidenceChanged(id) => write!(
                f,
                "basis evidence {} no longer matches the revision receipt snapshot",
                id.0
            ),
            Self::RevisionDecisionChanged(failures) => write!(
                f,
                "belief revision is no longer eligible under its frozen policy: {failures:?}"
            ),
            Self::Weight(error) => write!(f, "belief weight update failed: {error:?}"),
            Self::StateChangedBeforeCommit {
                expected_revision,
                actual_revision,
            } => write!(
                f,
                "belief state changed before commit: expected revision {expected_revision}, actual {actual_revision}"
            ),
            Self::RollbackPlanPredatesMutation {
                generated_at_cycle,
                applied_at_cycle,
            } => write!(
                f,
                "rollback plan cycle {generated_at_cycle} predates mutation cycle {applied_at_cycle}"
            ),
        }
    }
}

impl Error for BeliefMutationError {}

impl From<KnowledgeWeightError> for BeliefMutationError {
    fn from(value: KnowledgeWeightError) -> Self {
        Self::Weight(value)
    }
}

/// Replay-safe authorization firewall around the isolated epistemic-support store.
#[derive(Debug, Clone, Default)]
pub struct BeliefMutationFirewall {
    consumed_authorizations: HashMap<String, AuthorizationBinding>,
}

impl BeliefMutationFirewall {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(
        &mut self,
        ledger: &EpistemicLedger,
        store: &mut EpistemicSupportStore,
        revision_receipt: &BeliefRevisionReceipt,
        authorization: &BeliefMutationAuthorization,
        mutation_cycle: u64,
    ) -> Result<BeliefMutationOutcome, BeliefMutationError> {
        if !revision_receipt.eligible() {
            return Err(BeliefMutationError::RevisionReceiptRejected(
                revision_receipt.id(),
            ));
        }
        if !revision_receipt.duplicate_basis_evidence_ids().is_empty() {
            return Err(BeliefMutationError::RevisionReceiptHasDuplicateEvidence(
                revision_receipt.duplicate_basis_evidence_ids().to_vec(),
            ));
        }
        if authorization.decision != BeliefMutationAuthorizationDecision::Approved {
            return Err(BeliefMutationError::AuthorizationDenied);
        }
        if authorization.source_revision_receipt_id != revision_receipt.id()
            || authorization.claim_id != revision_receipt.claim_id()
            || authorization.proposed_delta != revision_receipt.proposed_delta()
        {
            return Err(BeliefMutationError::AuthorizationReceiptMismatch);
        }
        if mutation_cycle < authorization.approved_at_cycle {
            return Err(BeliefMutationError::MutationPredatesAuthorization {
                mutation_cycle,
                approved_at_cycle: authorization.approved_at_cycle,
            });
        }
        if mutation_cycle < revision_receipt.evaluated_at_cycle() {
            return Err(BeliefMutationError::MutationPredatesReceiptEvaluation {
                mutation_cycle,
                evaluated_at_cycle: revision_receipt.evaluated_at_cycle(),
            });
        }

        let binding = authorization.binding();
        if let Some(previous) = self
            .consumed_authorizations
            .get(&authorization.authorization_id)
        {
            if previous != &binding {
                return Err(BeliefMutationError::AuthorizationReplayMismatch {
                    authorization_id: authorization.authorization_id.clone(),
                });
            }
        }
        if let Some(previous) = store.authorization_binding(&authorization.authorization_id) {
            if previous != &binding {
                return Err(BeliefMutationError::AuthorizationReplayMismatch {
                    authorization_id: authorization.authorization_id.clone(),
                });
            }
        }

        // A revision receipt is single-use. Rebuilding the firewall cannot cause
        // a second application while the store/history is retained.
        if let Some(existing) = store.mutation_for_revision_receipt(revision_receipt.id()) {
            let existing = existing.clone();
            store.record_authorization_binding(
                &authorization.authorization_id,
                binding.clone(),
            )?;
            self.consumed_authorizations
                .insert(authorization.authorization_id.clone(), binding);
            return Ok(BeliefMutationOutcome::AlreadyApplied(existing));
        }

        let state = store
            .state(revision_receipt.claim_id())
            .ok_or(BeliefMutationError::StateNotRegistered(
                revision_receipt.claim_id(),
            ))?
            .clone();

        if state.revision != authorization.expected_state_revision {
            return Err(BeliefMutationError::StaleStateRevision {
                expected: authorization.expected_state_revision,
                actual: state.revision,
            });
        }
        if state.support != authorization.expected_support {
            return Err(BeliefMutationError::StaleStateSupport {
                expected: authorization.expected_support.get(),
                actual: state.support.get(),
            });
        }
        if revision_receipt.evaluated_at_cycle() < state.last_updated_cycle {
            return Err(BeliefMutationError::ReceiptPredatesCurrentState {
                evaluated_at_cycle: revision_receipt.evaluated_at_cycle(),
                state_last_updated_cycle: state.last_updated_cycle,
            });
        }

        if let Some(latest_evidence_cycle) = ledger
            .evidence_for_claim(revision_receipt.claim_id())
            .into_iter()
            .map(|record| record.observed_at_cycle)
            .max()
        {
            if latest_evidence_cycle > revision_receipt.evaluated_at_cycle() {
                return Err(BeliefMutationError::EvidencePostdatesRevisionDecision {
                    latest_evidence_cycle,
                    evaluated_at_cycle: revision_receipt.evaluated_at_cycle(),
                });
            }
        }

        verify_basis_snapshots(ledger, revision_receipt)?;
        revalidate_revision_decision(ledger, revision_receipt)?;

        // Deliberately fail instead of silently clamping: the approved delta must
        // describe the exact state transition that is actually committed.
        let support_after =
            BoundedWeight::new(state.support.get() + revision_receipt.proposed_delta())?;
        let receipt = BeliefMutationReceipt {
            id: BeliefMutationReceiptId(store.next_mutation_id),
            source_revision_receipt_id: revision_receipt.id(),
            claim_id: revision_receipt.claim_id(),
            proposed_delta: revision_receipt.proposed_delta(),
            support_before: state.support,
            support_after,
            state_revision_before: state.revision,
            state_revision_after: state.revision + 1,
            authorization_id: authorization.authorization_id.clone(),
            authority_label: authorization.authority_label.clone(),
            authorized_at_cycle: authorization.approved_at_cycle,
            applied_at_cycle: mutation_cycle,
        };

        store.commit(receipt.clone(), binding.clone())?;
        self.consumed_authorizations
            .insert(authorization.authorization_id.clone(), binding);
        Ok(BeliefMutationOutcome::Applied(receipt))
    }
}

fn verify_basis_snapshots(
    ledger: &EpistemicLedger,
    receipt: &BeliefRevisionReceipt,
) -> Result<(), BeliefMutationError> {
    for reference in receipt.basis() {
        let expected = reference
            .snapshot
            .as_ref()
            .ok_or(BeliefMutationError::MissingBasisEvidence(
                reference.requested_id,
            ))?;
        let live = ledger
            .evidence(reference.requested_id)
            .ok_or(BeliefMutationError::MissingBasisEvidence(
                reference.requested_id,
            ))?;
        if !snapshot_matches_live(expected, live) {
            return Err(BeliefMutationError::BasisEvidenceChanged(
                reference.requested_id,
            ));
        }
    }
    Ok(())
}

fn snapshot_matches_live(
    snapshot: &RevisionEvidenceSnapshot,
    live: &super::claim_evidence::EvidenceRecord,
) -> bool {
    snapshot.evidence_id == live.id
        && snapshot.claim_id == live.claim_id
        && snapshot.kind == live.kind
        && snapshot.polarity == live.polarity
        && snapshot.provenance_id == live.provenance_id
        && snapshot.observed_at_cycle == live.observed_at_cycle
        && snapshot.context == live.context
        && snapshot.method == live.method
}

fn revalidate_revision_decision(
    ledger: &EpistemicLedger,
    receipt: &BeliefRevisionReceipt,
) -> Result<(), BeliefMutationError> {
    let basis_ids = receipt
        .basis()
        .iter()
        .map(|reference| reference.requested_id)
        .collect::<Vec<_>>();
    let proposal = EpistemicRevisionProposal::new(
        receipt.claim_id(),
        receipt.proposed_delta(),
        basis_ids,
        receipt.rationale(),
    )?;
    let current = BeliefRevisionGate::evaluate(
        ledger,
        &proposal,
        receipt.policy(),
        receipt.calibration(),
        receipt.uncertainty(),
    );
    if !current.eligible() || &current != receipt.decision() {
        return Err(BeliefMutationError::RevisionDecisionChanged(
            current.failures().to_vec(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefRevisionHistory, BeliefRevisionPolicy, ClaimKind, EvidenceKind,
        EvidencePolarity,
    };

    fn fixture(
        polarity: EvidencePolarity,
        delta: f32,
    ) -> (
        EpistemicLedger,
        BeliefRevisionReceipt,
        EpistemicSupportStore,
    ) {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("lab", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let evidence = ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                polarity,
                provenance,
                2,
                Some("fixture".into()),
                Some("protocol-v1".into()),
            )
            .unwrap();
        let proposal = EpistemicRevisionProposal::new(
            claim,
            delta,
            vec![evidence],
            "bounded revision",
        )
        .unwrap();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
        let mut history = BeliefRevisionHistory::new();
        let receipt_id = history
            .evaluate_and_record(&ledger, &proposal, &policy, None, None, 3)
            .unwrap();
        let receipt = history.get(receipt_id).unwrap().clone();
        let mut store = EpistemicSupportStore::new();
        store
            .register_claim(&ledger, claim, BoundedWeight::new(0.50).unwrap(), 3)
            .unwrap();
        (ledger, receipt, store)
    }

    fn authorization(
        id: &str,
        receipt: &BeliefRevisionReceipt,
        store: &EpistemicSupportStore,
    ) -> BeliefMutationAuthorization {
        let state = store.state(receipt.claim_id()).unwrap();
        BeliefMutationAuthorization::new(
            id,
            "test-authority",
            BeliefMutationAuthorizationDecision::Approved,
            4,
            receipt,
            state,
        )
        .unwrap()
    }

    #[test]
    fn default_and_new_share_mutation_id_semantics() {
        assert_eq!(EpistemicSupportStore::new().next_mutation_id, 1);
        assert_eq!(EpistemicSupportStore::default().next_mutation_id, 1);
    }

    #[test]
    fn eligible_revision_applies_exact_bounded_delta_once() {
        let (ledger, receipt, mut store) = fixture(EvidencePolarity::Supports, 0.10);
        assert!(receipt.eligible());
        let authorization = authorization("belief-auth-1", &receipt, &store);
        let mut firewall = BeliefMutationFirewall::new();

        let first = firewall
            .apply(&ledger, &mut store, &receipt, &authorization, 5)
            .unwrap();
        assert!(first.applied_new_revision());
        assert!((first.receipt().support_before().get() - 0.50).abs() < 1e-6);
        assert!((first.receipt().support_after().get() - 0.60).abs() < 1e-6);
        assert_eq!(store.state(receipt.claim_id()).unwrap().revision(), 1);

        let replay = firewall
            .apply(&ledger, &mut store, &receipt, &authorization, 6)
            .unwrap();
        assert!(!replay.applied_new_revision());
        assert_eq!(replay.receipt().id(), first.receipt().id());
        assert_eq!(store.history().len(), 1);
    }

    #[test]
    fn rebuilt_firewall_does_not_reapply_consumed_revision_receipt() {
        let (ledger, receipt, mut store) = fixture(EvidencePolarity::Supports, 0.10);
        let authorization = authorization("belief-auth-1", &receipt, &store);
        let mut first = BeliefMutationFirewall::new();
        let applied = first
            .apply(&ledger, &mut store, &receipt, &authorization, 5)
            .unwrap();

        let mut rebuilt = BeliefMutationFirewall::new();
        let replay = rebuilt
            .apply(&ledger, &mut store, &receipt, &authorization, 6)
            .unwrap();
        assert!(!replay.applied_new_revision());
        assert_eq!(replay.receipt().id(), applied.receipt().id());
        assert_eq!(store.history().len(), 1);
        assert_eq!(store.consumed_authorization_count(), 1);
    }

    #[test]
    fn idempotent_ack_authorization_binding_survives_firewall_rebuild() {
        let (ledger, receipt, mut store) = fixture(EvidencePolarity::Supports, 0.10);
        let first_auth = authorization("belief-auth-1", &receipt, &store);
        let mut first = BeliefMutationFirewall::new();
        first
            .apply(&ledger, &mut store, &receipt, &first_auth, 5)
            .unwrap();

        // A second authorization ID may acknowledge the already-applied receipt;
        // that ID is still consumed in the retained store.
        let current_state = store.state(receipt.claim_id()).unwrap();
        let ack_auth = BeliefMutationAuthorization::new(
            "belief-auth-ack",
            "test-authority",
            BeliefMutationAuthorizationDecision::Approved,
            6,
            &receipt,
            current_state,
        )
        .unwrap();
        let mut second = BeliefMutationFirewall::new();
        let replay = second
            .apply(&ledger, &mut store, &receipt, &ack_auth, 6)
            .unwrap();
        assert!(!replay.applied_new_revision());
        assert_eq!(store.consumed_authorization_count(), 2);

        let mut third = BeliefMutationFirewall::new();
        let replay_again = third
            .apply(&ledger, &mut store, &receipt, &ack_auth, 7)
            .unwrap();
        assert!(!replay_again.applied_new_revision());
        assert_eq!(store.history().len(), 1);
    }

    #[test]
    fn rejected_revision_receipt_never_mutates_state() {
        let (ledger, eligible, mut store) = fixture(EvidencePolarity::Supports, 0.10);
        let claim = eligible.claim_id();
        let support_evidence = ledger.claim(claim).unwrap().evidence_ids[0];
        let wrong_direction = EpistemicRevisionProposal::new(
            claim,
            -0.10,
            vec![support_evidence],
            "wrong direction",
        )
        .unwrap();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
        let mut history = BeliefRevisionHistory::new();
        let id = history
            .evaluate_and_record(&ledger, &wrong_direction, &policy, None, None, 3)
            .unwrap();
        let rejected = history.get(id).unwrap().clone();
        assert!(!rejected.eligible());
        let authorization = authorization("belief-auth-rejected", &rejected, &store);
        let before = store.state(claim).unwrap().clone();
        let mut firewall = BeliefMutationFirewall::new();
        assert!(matches!(
            firewall.apply(&ledger, &mut store, &rejected, &authorization, 5),
            Err(BeliefMutationError::RevisionReceiptRejected(_))
        ));
        assert_eq!(store.state(claim).unwrap(), &before);
    }

    #[test]
    fn duplicate_basis_receipt_is_not_mutation_eligible() {
        let (ledger, base_receipt, mut store) = fixture(EvidencePolarity::Supports, 0.10);
        let claim = base_receipt.claim_id();
        let support = ledger.claim(claim).unwrap().evidence_ids[0];
        let proposal = EpistemicRevisionProposal::new(
            claim,
            0.10,
            vec![support, support],
            "duplicate evidence",
        )
        .unwrap();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
        let mut history = BeliefRevisionHistory::new();
        let id = history
            .evaluate_and_record(&ledger, &proposal, &policy, None, None, 3)
            .unwrap();
        let receipt = history.get(id).unwrap().clone();
        assert!(receipt.eligible());
        assert_eq!(receipt.duplicate_basis_evidence_ids(), &[support]);
        let auth = authorization("belief-auth-dup", &receipt, &store);
        let mut firewall = BeliefMutationFirewall::new();
        assert!(matches!(
            firewall.apply(&ledger, &mut store, &receipt, &auth, 5),
            Err(BeliefMutationError::RevisionReceiptHasDuplicateEvidence(_))
        ));
        assert!(store.history().is_empty());
    }

    #[test]
    fn any_postdecision_evidence_makes_receipt_stale() {
        let (mut ledger, receipt, mut store) = fixture(EvidencePolarity::Supports, 0.10);
        let claim = receipt.claim_id();
        let auth = authorization("belief-auth-stale", &receipt, &store);
        let provenance = ledger
            .add_provenance("later-source", None, None, 4, vec![])
            .unwrap();
        ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                provenance,
                4,
                Some("later result".into()),
                Some("protocol-v2".into()),
            )
            .unwrap();

        let mut firewall = BeliefMutationFirewall::new();
        assert_eq!(
            firewall
                .apply(&ledger, &mut store, &receipt, &auth, 5)
                .unwrap_err(),
            BeliefMutationError::EvidencePostdatesRevisionDecision {
                latest_evidence_cycle: 4,
                evaluated_at_cycle: 3,
            }
        );
        assert!((store.state(claim).unwrap().support().get() - 0.50).abs() < 1e-6);
    }

    #[test]
    fn out_of_range_transition_fails_instead_of_clamping() {
        let (ledger, receipt, _) = fixture(EvidencePolarity::Supports, 0.10);
        let claim = receipt.claim_id();
        let mut near_one = EpistemicSupportStore::new();
        near_one
            .register_claim(&ledger, claim, BoundedWeight::new(0.95).unwrap(), 3)
            .unwrap();
        let auth = authorization("belief-auth-boundary", &receipt, &near_one);
        let mut firewall = BeliefMutationFirewall::new();
        assert!(matches!(
            firewall.apply(&ledger, &mut near_one, &receipt, &auth, 5),
            Err(BeliefMutationError::Weight(_))
        ));
        assert!((near_one.state(claim).unwrap().support().get() - 0.95).abs() < 1e-6);
        assert!(near_one.history().is_empty());
    }

    #[test]
    fn mutation_receipt_generates_exact_nonexecuting_rollback_plan() {
        let (ledger, receipt, mut store) = fixture(EvidencePolarity::Supports, 0.10);
        let authorization = authorization("belief-auth-rollback", &receipt, &store);
        let mut firewall = BeliefMutationFirewall::new();
        let outcome = firewall
            .apply(&ledger, &mut store, &receipt, &authorization, 5)
            .unwrap();
        let plan = outcome.receipt().rollback_plan(6).unwrap();
        assert_eq!(plan.source_mutation_id(), outcome.receipt().id());
        assert!((plan.restore_support().get() - 0.50).abs() < 1e-6);
        assert!((plan.expected_current_support().get() - 0.60).abs() < 1e-6);
        assert_eq!(plan.expected_current_revision(), 1);
        assert!((store.state(receipt.claim_id()).unwrap().support().get() - 0.60).abs() < 1e-6);
    }
}
