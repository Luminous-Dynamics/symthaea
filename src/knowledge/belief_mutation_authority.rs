// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Single public authority surface for epistemic-support mutation.
//!
//! EKM-028 established the safe transaction sequence, but the raw EKM-026
//! firewall and EKM-028 transaction coordinator were still publicly reachable.
//! This module closes that authority surface: public callers prepare a sealed
//! revision through [`BeliefMutationAuthority::prepare`] and may apply it only
//! through [`BeliefMutationAuthority::apply`]. The raw writer remains owned by
//! this facade.
//!
//! Authorization remains a typed approval record rather than cryptographic
//! authentication. Restart persistence and signer verification are separate
//! boundaries.

use super::belief_mutation_decision_guard::{
    BeliefMutationDecisionGuard, BeliefMutationDecisionGuardError,
};
use super::belief_mutation_firewall::{
    BeliefMutationAuthorization, BeliefMutationFirewall, EpistemicSupportStore,
};
use super::belief_mutation_transaction::{
    BeliefMutationTransactionCoordinator, BeliefMutationTransactionError,
    BeliefMutationTransactionOutcome, BeliefRevisionEvidenceSeal,
};
use super::belief_revision_gate::{
    BeliefRevisionPolicy, CalibrationSnapshot, EpistemicRevisionProposal,
};
use super::belief_revision_receipt::{
    BeliefRevisionHistory, BeliefRevisionReceipt, BeliefRevisionReceiptId,
};
use super::claim_evidence::{ClaimId, EpistemicLedger};
use super::epistemic_vector::ClaimUncertaintyAssessment;
use std::error::Error;
use std::fmt;

/// Exact decision+seal pair produced by the authority preparation boundary.
///
/// Fields are private so callers cannot assemble a prepared mutation from an
/// arbitrary receipt and arbitrary seal. The pair is cloneable for transport or
/// review, but every application revalidates the live ledger and state.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedBeliefMutation {
    receipt: BeliefRevisionReceipt,
    seal: BeliefRevisionEvidenceSeal,
}

impl PreparedBeliefMutation {
    pub fn receipt(&self) -> &BeliefRevisionReceipt {
        &self.receipt
    }

    pub fn seal(&self) -> &BeliefRevisionEvidenceSeal {
        &self.seal
    }

    pub fn receipt_id(&self) -> BeliefRevisionReceiptId {
        self.receipt.id()
    }

    pub fn claim_id(&self) -> ClaimId {
        self.receipt.claim_id()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefMutationAuthorityError {
    Preparation(BeliefMutationDecisionGuardError),
    PreparedReceiptMissing(BeliefRevisionReceiptId),
    PreparedPairMismatch,
    ReplayAuthorizationMismatch {
        expected: String,
        actual: String,
    },
    Transaction(BeliefMutationTransactionError),
}

impl fmt::Display for BeliefMutationAuthorityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preparation(error) => write!(f, "belief mutation preparation failed: {error}"),
            Self::PreparedReceiptMissing(id) => write!(
                f,
                "belief mutation receipt {} is missing immediately after preparation",
                id.0
            ),
            Self::PreparedPairMismatch => write!(
                f,
                "prepared belief mutation receipt and evidence seal are not bound to the same decision"
            ),
            Self::ReplayAuthorizationMismatch { expected, actual } => write!(
                f,
                "belief mutation replay must reuse original authorization id '{expected}', got '{actual}'"
            ),
            Self::Transaction(error) => write!(f, "belief mutation transaction failed: {error}"),
        }
    }
}

impl Error for BeliefMutationAuthorityError {}

impl From<BeliefMutationDecisionGuardError> for BeliefMutationAuthorityError {
    fn from(value: BeliefMutationDecisionGuardError) -> Self {
        Self::Preparation(value)
    }
}

impl From<BeliefMutationTransactionError> for BeliefMutationAuthorityError {
    fn from(value: BeliefMutationTransactionError) -> Self {
        Self::Transaction(value)
    }
}

/// Sole public mutation facade for the EKM epistemic-support path.
///
/// The raw firewall is intentionally owned internally. Callers may inspect the
/// prepared receipt/seal and construct a separately authorized approval, but
/// cannot invoke the lower-level writer through the public knowledge API.
#[derive(Debug, Clone, Default)]
pub struct BeliefMutationAuthority {
    firewall: BeliefMutationFirewall,
}

impl BeliefMutationAuthority {
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluate a revision and seal its complete claim/evidence/provenance state
    /// in one immutable-ledger operation.
    pub fn prepare(
        &self,
        ledger: &EpistemicLedger,
        history: &mut BeliefRevisionHistory,
        proposal: &EpistemicRevisionProposal,
        policy: &BeliefRevisionPolicy,
        calibration: Option<CalibrationSnapshot>,
        uncertainty: Option<&ClaimUncertaintyAssessment>,
        evaluated_at_cycle: u64,
    ) -> Result<PreparedBeliefMutation, BeliefMutationAuthorityError> {
        let seal = BeliefMutationDecisionGuard::evaluate_and_seal(
            ledger,
            history,
            proposal,
            policy,
            calibration,
            uncertainty,
            evaluated_at_cycle,
        )?;
        let receipt_id = seal.source_revision_receipt_id();
        let receipt = history
            .get(receipt_id)
            .cloned()
            .ok_or(BeliefMutationAuthorityError::PreparedReceiptMissing(
                receipt_id,
            ))?;
        if receipt.id() != seal.source_revision_receipt_id()
            || receipt.claim_id() != seal.claim().claim_id
        {
            return Err(BeliefMutationAuthorityError::PreparedPairMismatch);
        }
        Ok(PreparedBeliefMutation { receipt, seal })
    }

    /// Apply an already prepared revision through the sealed EKM-028 transaction.
    ///
    /// This is the only public method in the knowledge surface that reaches the
    /// EKM-026 epistemic-support writer.
    pub fn apply(
        &mut self,
        ledger: &EpistemicLedger,
        store: &mut EpistemicSupportStore,
        prepared: &PreparedBeliefMutation,
        authorization: &BeliefMutationAuthorization,
        mutation_cycle: u64,
    ) -> Result<BeliefMutationTransactionOutcome, BeliefMutationAuthorityError> {
        if prepared.receipt.id() != prepared.seal.source_revision_receipt_id()
            || prepared.receipt.claim_id() != prepared.seal.claim().claim_id
        {
            return Err(BeliefMutationAuthorityError::PreparedPairMismatch);
        }

        // A revision receipt has one authoritative authorization identity. The
        // lower-level firewall can acknowledge alternate IDs on an idempotent
        // replay, but the public facade forbids that so restart persistence can
        // reconstruct every consumed public authorization from mutation history.
        if let Some(existing) = store.mutation_for_revision_receipt(prepared.receipt.id()) {
            if existing.authorization_id() != authorization.authorization_id() {
                return Err(BeliefMutationAuthorityError::ReplayAuthorizationMismatch {
                    expected: existing.authorization_id().to_string(),
                    actual: authorization.authorization_id().to_string(),
                });
            }
        }

        BeliefMutationTransactionCoordinator::apply(
            &prepared.seal,
            ledger,
            store,
            &prepared.receipt,
            authorization,
            &mut self.firewall,
            mutation_cycle,
        )
        .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationAuthorizationDecision, BoundedWeight, ClaimKind, EvidenceKind,
        EvidencePolarity,
    };

    fn fixture() -> (
        EpistemicLedger,
        EpistemicRevisionProposal,
        BeliefRevisionPolicy,
        ClaimId,
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
                EvidencePolarity::Supports,
                provenance,
                2,
                Some("fixture".into()),
                Some("protocol-v1".into()),
            )
            .unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "support")
            .unwrap();
        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
        (ledger, proposal, policy, claim)
    }

    #[test]
    fn public_facade_prepares_authorizes_applies_and_verifies() {
        let (ledger, proposal, policy, claim) = fixture();
        let mut history = BeliefRevisionHistory::new();
        let mut authority = BeliefMutationAuthority::new();
        let prepared = authority
            .prepare(&ledger, &mut history, &proposal, &policy, None, None, 3)
            .unwrap();
        assert_eq!(prepared.receipt_id(), prepared.seal().source_revision_receipt_id());
        assert_eq!(prepared.claim_id(), claim);

        let mut store = EpistemicSupportStore::new();
        store
            .register_claim(&ledger, claim, BoundedWeight::new(0.50).unwrap(), 3)
            .unwrap();
        let authorization = BeliefMutationAuthorization::new(
            "auth-1",
            "test-authority",
            BeliefMutationAuthorizationDecision::Approved,
            4,
            prepared.receipt(),
            store.state(claim).unwrap(),
        )
        .unwrap();

        let outcome = authority
            .apply(&ledger, &mut store, &prepared, &authorization, 5)
            .unwrap();
        assert!(outcome.mutation().applied_new_revision());
        assert!(outcome.verified(), "{:?}", outcome.verification().failures());
        assert!((store.state(claim).unwrap().support().get() - 0.60).abs() < 1e-6);
    }

    #[test]
    fn rejected_revision_cannot_be_prepared_for_mutation() {
        let (ledger, _, policy, claim) = fixture();
        let evidence = ledger.claim(claim).unwrap().evidence_ids[0];
        let rejected = EpistemicRevisionProposal::new(
            claim,
            -0.10,
            vec![evidence],
            "wrong-direction revision",
        )
        .unwrap();
        let mut history = BeliefRevisionHistory::new();
        let authority = BeliefMutationAuthority::new();

        assert!(authority
            .prepare(&ledger, &mut history, &rejected, &policy, None, None, 3)
            .is_err());
        assert_eq!(history.len(), 1);
        assert!(!history.receipts()[0].eligible());
    }

    #[test]
    fn replay_through_rebuilt_facade_is_idempotent_when_store_survives() {
        let (ledger, proposal, policy, claim) = fixture();
        let mut history = BeliefRevisionHistory::new();
        let authority = BeliefMutationAuthority::new();
        let prepared = authority
            .prepare(&ledger, &mut history, &proposal, &policy, None, None, 3)
            .unwrap();
        let mut store = EpistemicSupportStore::new();
        store
            .register_claim(&ledger, claim, BoundedWeight::new(0.50).unwrap(), 3)
            .unwrap();
        let authorization = BeliefMutationAuthorization::new(
            "auth-replay",
            "test-authority",
            BeliefMutationAuthorizationDecision::Approved,
            4,
            prepared.receipt(),
            store.state(claim).unwrap(),
        )
        .unwrap();

        let mut first = BeliefMutationAuthority::new();
        let first_outcome = first
            .apply(&ledger, &mut store, &prepared, &authorization, 5)
            .unwrap();
        assert!(first_outcome.mutation().applied_new_revision());

        let mut rebuilt = BeliefMutationAuthority::new();
        let replay = rebuilt
            .apply(&ledger, &mut store, &prepared, &authorization, 6)
            .unwrap();
        assert!(!replay.mutation().applied_new_revision());
        assert!(replay.verified(), "{:?}", replay.verification().failures());
        assert_eq!(store.history().len(), 1);
    }

    #[test]
    fn replay_cannot_consume_a_second_authorization_identity() {
        let (ledger, proposal, policy, claim) = fixture();
        let mut history = BeliefRevisionHistory::new();
        let authority = BeliefMutationAuthority::new();
        let prepared = authority
            .prepare(&ledger, &mut history, &proposal, &policy, None, None, 3)
            .unwrap();
        let mut store = EpistemicSupportStore::new();
        store
            .register_claim(&ledger, claim, BoundedWeight::new(0.50).unwrap(), 3)
            .unwrap();
        let original = BeliefMutationAuthorization::new(
            "auth-original",
            "test-authority",
            BeliefMutationAuthorizationDecision::Approved,
            4,
            prepared.receipt(),
            store.state(claim).unwrap(),
        )
        .unwrap();
        let mut first = BeliefMutationAuthority::new();
        first
            .apply(&ledger, &mut store, &prepared, &original, 5)
            .unwrap();

        let alternate = BeliefMutationAuthorization::new(
            "auth-alternate",
            "test-authority",
            BeliefMutationAuthorizationDecision::Approved,
            6,
            prepared.receipt(),
            store.state(claim).unwrap(),
        )
        .unwrap();
        let mut rebuilt = BeliefMutationAuthority::new();
        assert_eq!(
            rebuilt
                .apply(&ledger, &mut store, &prepared, &alternate, 7)
                .unwrap_err(),
            BeliefMutationAuthorityError::ReplayAuthorizationMismatch {
                expected: "auth-original".into(),
                actual: "auth-alternate".into(),
            }
        );
        assert_eq!(store.history().len(), 1);
        assert_eq!(store.consumed_authorization_count(), 1);
    }
}
