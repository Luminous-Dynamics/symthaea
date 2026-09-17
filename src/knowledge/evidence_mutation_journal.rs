// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Restart-stable replay journal for the evidence mutation firewall.
//!
//! `EvidenceMutationFirewall` already prevents duplicate draft insertion by
//! scanning the ledger, but its authorization replay map is in-memory. This
//! module adds a deterministic export/import snapshot so a higher persistence
//! layer can preserve authorization bindings across restart.
//!
//! This module does not write files. A snapshot is only a data object suitable
//! for persistence by another layer.

use super::claim_evidence::{EpistemicLedger, EvidenceId};
use super::evidence_mutation_firewall::{
    EvidenceDraftIdentity, EvidenceIngestionOutcome, EvidenceIngestionReceipt,
    EvidenceMutationAuthorization, EvidenceMutationError, EvidenceMutationFirewall,
};
use super::receipt_admission::AdmissibleEvidenceDraft;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationJournalEntry {
    authorization_id: String,
    authority_label: String,
    draft_identity: EvidenceDraftIdentity,
    evidence_id: EvidenceId,
    authorized_at_cycle: u64,
    first_ingested_at_cycle: u64,
}

impl MutationJournalEntry {
    pub fn authorization_id(&self) -> &str {
        &self.authorization_id
    }

    pub fn authority_label(&self) -> &str {
        &self.authority_label
    }

    pub fn draft_identity(&self) -> &EvidenceDraftIdentity {
        &self.draft_identity
    }

    pub fn evidence_id(&self) -> EvidenceId {
        self.evidence_id
    }

    pub fn authorized_at_cycle(&self) -> u64 {
        self.authorized_at_cycle
    }

    pub fn first_ingested_at_cycle(&self) -> u64 {
        self.first_ingested_at_cycle
    }
}

/// Deterministically ordered snapshot for external persistence.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EvidenceMutationJournalSnapshot {
    entries: Vec<MutationJournalEntry>,
}

impl EvidenceMutationJournalSnapshot {
    pub fn entries(&self) -> &[MutationJournalEntry] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceMutationJournalError {
    EmptyAuthorizationId,
    EmptyAuthorityLabel,
    AuthorizationBindingConflict {
        authorization_id: String,
    },
    DraftEvidenceConflict {
        previous: EvidenceId,
        incoming: EvidenceId,
    },
    JournalLedgerMissingEvidence(EvidenceId),
    JournalLedgerMismatch(EvidenceId),
    Mutation(EvidenceMutationError),
}

impl fmt::Display for EvidenceMutationJournalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyAuthorizationId => write!(f, "journal authorization id cannot be empty"),
            Self::EmptyAuthorityLabel => write!(f, "journal authority label cannot be empty"),
            Self::AuthorizationBindingConflict { authorization_id } => write!(
                f,
                "authorization '{authorization_id}' is bound to inconsistent mutation records"
            ),
            Self::DraftEvidenceConflict { previous, incoming } => write!(
                f,
                "one evidence draft is bound to conflicting evidence ids {} and {}",
                previous.0, incoming.0
            ),
            Self::JournalLedgerMissingEvidence(id) => write!(
                f,
                "mutation journal references missing ledger evidence {}",
                id.0
            ),
            Self::JournalLedgerMismatch(id) => write!(
                f,
                "mutation journal evidence {} does not match its recorded draft identity",
                id.0
            ),
            Self::Mutation(error) => write!(f, "evidence mutation failed: {error}"),
        }
    }
}

impl Error for EvidenceMutationJournalError {}

impl From<EvidenceMutationError> for EvidenceMutationJournalError {
    fn from(value: EvidenceMutationError) -> Self {
        Self::Mutation(value)
    }
}

/// In-memory journal with deterministic snapshot import/export.
#[derive(Debug, Clone, Default)]
pub struct EvidenceMutationJournal {
    by_authorization: HashMap<String, MutationJournalEntry>,
    by_draft: HashMap<EvidenceDraftIdentity, EvidenceId>,
}

impl EvidenceMutationJournal {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_snapshot(
        snapshot: EvidenceMutationJournalSnapshot,
    ) -> Result<Self, EvidenceMutationJournalError> {
        let mut journal = Self::new();
        for entry in snapshot.entries {
            journal.record_entry(entry)?;
        }
        Ok(journal)
    }

    pub fn snapshot(&self) -> EvidenceMutationJournalSnapshot {
        let mut entries = self
            .by_authorization
            .values()
            .cloned()
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.authorization_id.cmp(&right.authorization_id));
        EvidenceMutationJournalSnapshot { entries }
    }

    pub fn len(&self) -> usize {
        self.by_authorization.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_authorization.is_empty()
    }

    pub fn authorization_entry(&self, authorization_id: &str) -> Option<&MutationJournalEntry> {
        self.by_authorization.get(authorization_id)
    }

    pub fn evidence_for_draft(&self, draft: &EvidenceDraftIdentity) -> Option<EvidenceId> {
        self.by_draft.get(draft).copied()
    }

    fn record_receipt(
        &mut self,
        receipt: &EvidenceIngestionReceipt,
    ) -> Result<(), EvidenceMutationJournalError> {
        self.record_entry(MutationJournalEntry {
            authorization_id: receipt.authorization_id().to_string(),
            authority_label: receipt.authority_label().to_string(),
            draft_identity: receipt.draft_identity().clone(),
            evidence_id: receipt.evidence_id(),
            authorized_at_cycle: receipt.authorized_at_cycle(),
            first_ingested_at_cycle: receipt.ingested_at_cycle(),
        })
    }

    fn record_entry(
        &mut self,
        entry: MutationJournalEntry,
    ) -> Result<(), EvidenceMutationJournalError> {
        if entry.authorization_id.trim().is_empty() {
            return Err(EvidenceMutationJournalError::EmptyAuthorizationId);
        }
        if entry.authority_label.trim().is_empty() {
            return Err(EvidenceMutationJournalError::EmptyAuthorityLabel);
        }

        if let Some(existing) = self.by_authorization.get(&entry.authorization_id) {
            if existing != &entry {
                return Err(EvidenceMutationJournalError::AuthorizationBindingConflict {
                    authorization_id: entry.authorization_id,
                });
            }
            return Ok(());
        }

        if let Some(previous) = self.by_draft.get(&entry.draft_identity).copied() {
            if previous != entry.evidence_id {
                return Err(EvidenceMutationJournalError::DraftEvidenceConflict {
                    previous,
                    incoming: entry.evidence_id,
                });
            }
        }

        self.by_draft
            .insert(entry.draft_identity.clone(), entry.evidence_id);
        self.by_authorization
            .insert(entry.authorization_id.clone(), entry);
        Ok(())
    }
}

/// Firewall plus importable/exportable replay journal.
#[derive(Debug, Clone, Default)]
pub struct JournaledEvidenceMutationFirewall {
    firewall: EvidenceMutationFirewall,
    journal: EvidenceMutationJournal,
}

impl JournaledEvidenceMutationFirewall {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_snapshot(
        snapshot: EvidenceMutationJournalSnapshot,
    ) -> Result<Self, EvidenceMutationJournalError> {
        Ok(Self {
            firewall: EvidenceMutationFirewall::new(),
            journal: EvidenceMutationJournal::from_snapshot(snapshot)?,
        })
    }

    pub fn journal(&self) -> &EvidenceMutationJournal {
        &self.journal
    }

    pub fn snapshot(&self) -> EvidenceMutationJournalSnapshot {
        self.journal.snapshot()
    }

    pub fn ingest(
        &mut self,
        ledger: &mut EpistemicLedger,
        draft: &AdmissibleEvidenceDraft,
        authorization: &EvidenceMutationAuthorization,
        ingestion_cycle: u64,
    ) -> Result<EvidenceIngestionOutcome, EvidenceMutationJournalError> {
        let identity = EvidenceDraftIdentity::from(draft);

        if let Some(previous) = self
            .journal
            .authorization_entry(authorization.authorization_id())
        {
            if previous.draft_identity() != &identity {
                return Err(EvidenceMutationJournalError::AuthorizationBindingConflict {
                    authorization_id: authorization.authorization_id().to_string(),
                });
            }
        }

        // A restored journal is evidence about prior mutation state. If it says a
        // draft was already inserted, the ledger must still contain that exact
        // record. Do not silently recreate missing or divergent state.
        if let Some(expected_id) = self.journal.evidence_for_draft(&identity) {
            verify_ledger_binding(ledger, expected_id, &identity)?;
        }

        let outcome = self
            .firewall
            .ingest(ledger, draft, authorization, ingestion_cycle)?;

        if let Some(expected_id) = self.journal.evidence_for_draft(&identity) {
            if expected_id != outcome.receipt().evidence_id() {
                return Err(EvidenceMutationJournalError::DraftEvidenceConflict {
                    previous: expected_id,
                    incoming: outcome.receipt().evidence_id(),
                });
            }
        }

        self.journal.record_receipt(outcome.receipt())?;
        Ok(outcome)
    }
}

fn verify_ledger_binding(
    ledger: &EpistemicLedger,
    evidence_id: EvidenceId,
    identity: &EvidenceDraftIdentity,
) -> Result<(), EvidenceMutationJournalError> {
    let record = ledger
        .evidence(evidence_id)
        .ok_or(EvidenceMutationJournalError::JournalLedgerMissingEvidence(
            evidence_id,
        ))?;

    let expected_context = Some(format!("inquiry-result: {}", identity.result_summary));
    let expected_method = Some(format!(
        "preregistered-decision[{}]: {}",
        identity.decision_rule_label, identity.decision_criterion
    ));

    if record.claim_id != identity.claim_id
        || record.kind != identity.kind
        || record.polarity != identity.polarity
        || record.provenance_id != identity.provenance_id
        || record.observed_at_cycle != identity.observed_at_cycle
        || record.context != expected_context
        || record.method != expected_method
    {
        return Err(EvidenceMutationJournalError::JournalLedgerMismatch(
            evidence_id,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        ClaimKind, DecisionInterpretation, EvidenceKind, IgnoranceFrontier,
        InquiryContractBuilder, InquiryPreregistration, InquiryRequest, InquiryResultReceipt,
        MutationAuthorizationDecision, PreregisteredDecisionRule, ReceiptAdmissionGate,
        ReceiptAdmissionPolicy,
    };

    fn rules() -> Vec<PreregisteredDecisionRule> {
        vec![
            PreregisteredDecisionRule::new(
                "supports",
                "measure > upper",
                DecisionInterpretation::SupportsClaim,
            ),
            PreregisteredDecisionRule::new(
                "contradicts",
                "measure < lower",
                DecisionInterpretation::ContradictsClaim,
            ),
            PreregisteredDecisionRule::new(
                "inconclusive",
                "otherwise",
                DecisionInterpretation::Inconclusive,
            ),
        ]
    }

    fn admitted_draft() -> (EpistemicLedger, AdmissibleEvidenceDraft) {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("experiment", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let frontier = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();
        let plan = InquiryContractBuilder::from_frontier(&ledger, &frontier).unwrap();
        let contract = plan
            .contracts
            .into_iter()
            .find(|contract| contract.request == InquiryRequest::SeekDiscriminatingEvidence)
            .unwrap();
        let preregistration = InquiryPreregistration::new(
            &contract,
            "bounded comparison",
            "prediction error delta",
            rules(),
            "fixed sample budget",
            vec![],
            vec![],
            5,
        )
        .unwrap();
        let receipt = InquiryResultReceipt::new(
            &ledger,
            &contract,
            &preregistration,
            EvidenceKind::Measurement,
            provenance,
            6,
            "observed result",
            "supports",
        )
        .unwrap();
        let policy = ReceiptAdmissionPolicy::new(vec![EvidenceKind::Measurement], true, true, false);
        let admission = ReceiptAdmissionGate::evaluate(&ledger, &receipt, &policy);
        (ledger, admission.draft().unwrap().clone())
    }

    fn authorization(
        id: &str,
        draft: &AdmissibleEvidenceDraft,
    ) -> EvidenceMutationAuthorization {
        EvidenceMutationAuthorization::new(
            id,
            "test-authority",
            MutationAuthorizationDecision::Approved,
            7,
            draft,
        )
        .unwrap()
    }

    #[test]
    fn snapshot_round_trip_preserves_authorization_binding() {
        let (mut ledger, draft) = admitted_draft();
        let auth = authorization("auth-1", &draft);
        let mut firewall = JournaledEvidenceMutationFirewall::new();
        firewall.ingest(&mut ledger, &draft, &auth, 8).unwrap();

        let snapshot = firewall.snapshot();
        assert_eq!(snapshot.entries().len(), 1);
        let restored = JournaledEvidenceMutationFirewall::from_snapshot(snapshot).unwrap();
        let entry = restored.journal().authorization_entry("auth-1").unwrap();
        assert_eq!(entry.draft_identity(), &EvidenceDraftIdentity::from(&draft));
    }

    #[test]
    fn restored_journal_and_ledger_allow_idempotent_replay() {
        let (mut ledger, draft) = admitted_draft();
        let auth = authorization("auth-1", &draft);
        let mut first = JournaledEvidenceMutationFirewall::new();
        let inserted = first.ingest(&mut ledger, &draft, &auth, 8).unwrap();
        let count = ledger.evidence_count();
        let snapshot = first.snapshot();

        let mut restored = JournaledEvidenceMutationFirewall::from_snapshot(snapshot).unwrap();
        let replay = restored.ingest(&mut ledger, &draft, &auth, 9).unwrap();
        assert!(!replay.inserted_new_record());
        assert_eq!(replay.receipt().evidence_id(), inserted.receipt().evidence_id());
        assert_eq!(ledger.evidence_count(), count);
    }

    #[test]
    fn restored_journal_fails_closed_when_ledger_record_is_missing() {
        let (mut ledger, draft) = admitted_draft();
        let auth = authorization("auth-1", &draft);
        let mut first = JournaledEvidenceMutationFirewall::new();
        first.ingest(&mut ledger, &draft, &auth, 8).unwrap();
        let snapshot = first.snapshot();

        // Recreate only claim/provenance IDs, not the evidence record.
        let (mut ledger_without_evidence, equivalent_draft) = admitted_draft();
        assert_eq!(
            EvidenceDraftIdentity::from(&draft),
            EvidenceDraftIdentity::from(&equivalent_draft)
        );
        let mut restored = JournaledEvidenceMutationFirewall::from_snapshot(snapshot).unwrap();
        assert!(matches!(
            restored.ingest(
                &mut ledger_without_evidence,
                &equivalent_draft,
                &auth,
                9,
            ),
            Err(EvidenceMutationJournalError::JournalLedgerMissingEvidence(_))
        ));
        assert_eq!(ledger_without_evidence.evidence_count(), 0);
    }

    #[test]
    fn conflicting_authorization_binding_is_rejected_during_import() {
        let (mut ledger, draft) = admitted_draft();
        let auth = authorization("auth-1", &draft);
        let mut firewall = JournaledEvidenceMutationFirewall::new();
        firewall.ingest(&mut ledger, &draft, &auth, 8).unwrap();
        let mut snapshot = firewall.snapshot();
        let mut conflicting = snapshot.entries()[0].clone();
        conflicting.draft_identity.result_summary.push_str(" changed");
        snapshot.entries.push(conflicting);

        assert!(matches!(
            JournaledEvidenceMutationFirewall::from_snapshot(snapshot),
            Err(EvidenceMutationJournalError::AuthorizationBindingConflict { .. })
        ));
    }
}
