// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Replay-safe mutation boundary from admitted evidence drafts into the epistemic ledger.
//!
//! This is the first EKM layer in this series that is allowed to call
//! `EpistemicLedger::add_evidence`. The mutation is deliberately narrow: inserting
//! an evidence record does not change confidence, uncertainty, causal status,
//! action selection, or any external system.
//!
//! An [`EvidenceMutationAuthorization`] is a typed approval record, **not** a
//! cryptographic authentication mechanism. Authentication/signature verification
//! belongs to a higher authority layer. This module only binds a supplied approval
//! to the exact draft that may be inserted.

use super::claim_evidence::{
    ClaimId, EpistemicLedger, EvidenceId, EvidenceKind, EvidencePolarity, LedgerError,
    ProvenanceId,
};
use super::receipt_admission::AdmissibleEvidenceDraft;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// Exact semantic identity of an admitted evidence draft.
///
/// This is intentionally the full typed payload rather than a non-cryptographic
/// hash. Equality therefore cannot collide because two different draft payloads
/// happened to produce the same small digest.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvidenceDraftIdentity {
    pub claim_id: ClaimId,
    pub kind: EvidenceKind,
    pub polarity: EvidencePolarity,
    pub provenance_id: ProvenanceId,
    pub observed_at_cycle: u64,
    pub result_summary: String,
    pub decision_rule_label: String,
    pub decision_criterion: String,
}

impl From<&AdmissibleEvidenceDraft> for EvidenceDraftIdentity {
    fn from(draft: &AdmissibleEvidenceDraft) -> Self {
        Self {
            claim_id: draft.claim_id(),
            kind: draft.kind(),
            polarity: draft.polarity(),
            provenance_id: draft.provenance_id(),
            observed_at_cycle: draft.observed_at_cycle(),
            result_summary: draft.result_summary().to_string(),
            decision_rule_label: draft.decision_rule_label().to_string(),
            decision_criterion: draft.decision_criterion().to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationAuthorizationDecision {
    Approved,
    Denied,
}

/// Explicit approval/denial bound to one exact evidence draft.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceMutationAuthorization {
    authorization_id: String,
    authority_label: String,
    decision: MutationAuthorizationDecision,
    approved_at_cycle: u64,
    draft_identity: EvidenceDraftIdentity,
}

impl EvidenceMutationAuthorization {
    pub fn new(
        authorization_id: impl Into<String>,
        authority_label: impl Into<String>,
        decision: MutationAuthorizationDecision,
        approved_at_cycle: u64,
        draft: &AdmissibleEvidenceDraft,
    ) -> Result<Self, EvidenceMutationError> {
        let authorization_id = authorization_id.into();
        if authorization_id.trim().is_empty() {
            return Err(EvidenceMutationError::EmptyAuthorizationId);
        }
        let authority_label = authority_label.into();
        if authority_label.trim().is_empty() {
            return Err(EvidenceMutationError::EmptyAuthorityLabel);
        }
        if approved_at_cycle < draft.observed_at_cycle() {
            return Err(EvidenceMutationError::AuthorizationPredatesObservation {
                approved_at_cycle,
                observed_at_cycle: draft.observed_at_cycle(),
            });
        }

        Ok(Self {
            authorization_id,
            authority_label,
            decision,
            approved_at_cycle,
            draft_identity: EvidenceDraftIdentity::from(draft),
        })
    }

    pub fn authorization_id(&self) -> &str {
        &self.authorization_id
    }

    pub fn authority_label(&self) -> &str {
        &self.authority_label
    }

    pub fn decision(&self) -> MutationAuthorizationDecision {
        self.decision
    }

    pub fn approved_at_cycle(&self) -> u64 {
        self.approved_at_cycle
    }

    pub fn draft_identity(&self) -> &EvidenceDraftIdentity {
        &self.draft_identity
    }
}

/// Receipt for the exact ledger record satisfying one admitted draft.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceIngestionReceipt {
    evidence_id: EvidenceId,
    draft_identity: EvidenceDraftIdentity,
    authorization_id: String,
    authority_label: String,
    authorized_at_cycle: u64,
    ingested_at_cycle: u64,
    /// False when an exact record was already present in the ledger and the
    /// firewall therefore performed no duplicate mutation.
    inserted_by_firewall: bool,
}

impl EvidenceIngestionReceipt {
    pub fn evidence_id(&self) -> EvidenceId {
        self.evidence_id
    }

    pub fn draft_identity(&self) -> &EvidenceDraftIdentity {
        &self.draft_identity
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

    pub fn ingested_at_cycle(&self) -> u64 {
        self.ingested_at_cycle
    }

    pub fn inserted_by_firewall(&self) -> bool {
        self.inserted_by_firewall
    }
}

/// Idempotent outcome of an ingestion request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceIngestionOutcome {
    Inserted(EvidenceIngestionReceipt),
    AlreadySatisfied(EvidenceIngestionReceipt),
}

impl EvidenceIngestionOutcome {
    pub fn receipt(&self) -> &EvidenceIngestionReceipt {
        match self {
            Self::Inserted(receipt) | Self::AlreadySatisfied(receipt) => receipt,
        }
    }

    pub fn inserted_new_record(&self) -> bool {
        matches!(self, Self::Inserted(_))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceMutationError {
    EmptyAuthorizationId,
    EmptyAuthorityLabel,
    AuthorizationDenied,
    AuthorizationDraftMismatch,
    AuthorizationPredatesObservation {
        approved_at_cycle: u64,
        observed_at_cycle: u64,
    },
    IngestionPredatesAuthorization {
        ingestion_cycle: u64,
        approved_at_cycle: u64,
    },
    AuthorizationReplayMismatch {
        authorization_id: String,
    },
    AmbiguousExistingEvidence {
        evidence_ids: Vec<EvidenceId>,
    },
    Ledger(LedgerError),
    PostIngestionMismatch(EvidenceId),
}

impl fmt::Display for EvidenceMutationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyAuthorizationId => write!(f, "authorization id cannot be empty"),
            Self::EmptyAuthorityLabel => write!(f, "authority label cannot be empty"),
            Self::AuthorizationDenied => write!(f, "evidence mutation authorization was denied"),
            Self::AuthorizationDraftMismatch => {
                write!(f, "authorization is bound to a different evidence draft")
            }
            Self::AuthorizationPredatesObservation {
                approved_at_cycle,
                observed_at_cycle,
            } => write!(
                f,
                "authorization cycle {approved_at_cycle} predates observation cycle {observed_at_cycle}"
            ),
            Self::IngestionPredatesAuthorization {
                ingestion_cycle,
                approved_at_cycle,
            } => write!(
                f,
                "ingestion cycle {ingestion_cycle} predates authorization cycle {approved_at_cycle}"
            ),
            Self::AuthorizationReplayMismatch { authorization_id } => write!(
                f,
                "authorization '{authorization_id}' was already consumed for a different draft"
            ),
            Self::AmbiguousExistingEvidence { evidence_ids } => write!(
                f,
                "multiple existing ledger records exactly match this evidence draft: {evidence_ids:?}"
            ),
            Self::Ledger(error) => write!(f, "ledger mutation failed: {error}"),
            Self::PostIngestionMismatch(id) => write!(
                f,
                "inserted evidence {} does not exactly match the authorized draft",
                id.0
            ),
        }
    }
}

impl Error for EvidenceMutationError {}

impl From<LedgerError> for EvidenceMutationError {
    fn from(value: LedgerError) -> Self {
        Self::Ledger(value)
    }
}

/// Stateful replay/idempotency guard around the narrow evidence mutation.
///
/// The guard survives repeated calls while this value is retained. It also scans
/// the ledger for an exact deterministic record before inserting, so a rebuilt
/// firewall does not duplicate an already persisted exact evidence record.
#[derive(Debug, Default, Clone)]
pub struct EvidenceMutationFirewall {
    ingested: HashMap<EvidenceDraftIdentity, EvidenceIngestionReceipt>,
    consumed_authorizations: HashMap<String, EvidenceDraftIdentity>,
}

impl EvidenceMutationFirewall {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ingest(
        &mut self,
        ledger: &mut EpistemicLedger,
        draft: &AdmissibleEvidenceDraft,
        authorization: &EvidenceMutationAuthorization,
        ingestion_cycle: u64,
    ) -> Result<EvidenceIngestionOutcome, EvidenceMutationError> {
        let identity = EvidenceDraftIdentity::from(draft);

        if authorization.draft_identity != identity {
            return Err(EvidenceMutationError::AuthorizationDraftMismatch);
        }
        if authorization.decision != MutationAuthorizationDecision::Approved {
            return Err(EvidenceMutationError::AuthorizationDenied);
        }
        if ingestion_cycle < authorization.approved_at_cycle {
            return Err(EvidenceMutationError::IngestionPredatesAuthorization {
                ingestion_cycle,
                approved_at_cycle: authorization.approved_at_cycle,
            });
        }

        if let Some(previous_identity) = self
            .consumed_authorizations
            .get(&authorization.authorization_id)
        {
            if previous_identity != &identity {
                return Err(EvidenceMutationError::AuthorizationReplayMismatch {
                    authorization_id: authorization.authorization_id.clone(),
                });
            }
        }

        if let Some(existing) = self.ingested.get(&identity).cloned() {
            self.consumed_authorizations
                .entry(authorization.authorization_id.clone())
                .or_insert(identity);
            return Ok(EvidenceIngestionOutcome::AlreadySatisfied(existing));
        }

        // Revalidate ledger referential integrity at the mutation boundary.
        if ledger.claim(identity.claim_id).is_none() {
            return Err(LedgerError::UnknownClaim(identity.claim_id).into());
        }
        if ledger.provenance(identity.provenance_id).is_none() {
            return Err(LedgerError::UnknownProvenance(identity.provenance_id).into());
        }

        let context = deterministic_context(&identity);
        let method = deterministic_method(&identity);
        let exact_existing = ledger
            .evidence_for_claim(identity.claim_id)
            .into_iter()
            .filter(|record| {
                record.kind == identity.kind
                    && record.polarity == identity.polarity
                    && record.provenance_id == identity.provenance_id
                    && record.observed_at_cycle == identity.observed_at_cycle
                    && record.context == context
                    && record.method == method
            })
            .map(|record| record.id)
            .collect::<Vec<_>>();

        let receipt = match exact_existing.as_slice() {
            [] => {
                let evidence_id = ledger.add_evidence(
                    identity.claim_id,
                    identity.kind,
                    identity.polarity,
                    identity.provenance_id,
                    identity.observed_at_cycle,
                    context.clone(),
                    method.clone(),
                )?;
                let inserted = ledger
                    .evidence(evidence_id)
                    .ok_or(EvidenceMutationError::PostIngestionMismatch(evidence_id))?;
                if inserted.claim_id != identity.claim_id
                    || inserted.kind != identity.kind
                    || inserted.polarity != identity.polarity
                    || inserted.provenance_id != identity.provenance_id
                    || inserted.observed_at_cycle != identity.observed_at_cycle
                    || inserted.context != context
                    || inserted.method != method
                {
                    return Err(EvidenceMutationError::PostIngestionMismatch(evidence_id));
                }

                EvidenceIngestionReceipt {
                    evidence_id,
                    draft_identity: identity.clone(),
                    authorization_id: authorization.authorization_id.clone(),
                    authority_label: authorization.authority_label.clone(),
                    authorized_at_cycle: authorization.approved_at_cycle,
                    ingested_at_cycle: ingestion_cycle,
                    inserted_by_firewall: true,
                }
            }
            [evidence_id] => EvidenceIngestionReceipt {
                evidence_id: *evidence_id,
                draft_identity: identity.clone(),
                authorization_id: authorization.authorization_id.clone(),
                authority_label: authorization.authority_label.clone(),
                authorized_at_cycle: authorization.approved_at_cycle,
                ingested_at_cycle: ingestion_cycle,
                inserted_by_firewall: false,
            },
            many => {
                return Err(EvidenceMutationError::AmbiguousExistingEvidence {
                    evidence_ids: many.to_vec(),
                })
            }
        };

        self.consumed_authorizations
            .insert(authorization.authorization_id.clone(), identity.clone());
        self.ingested.insert(identity, receipt.clone());

        if receipt.inserted_by_firewall {
            Ok(EvidenceIngestionOutcome::Inserted(receipt))
        } else {
            Ok(EvidenceIngestionOutcome::AlreadySatisfied(receipt))
        }
    }

    pub fn ingested_draft_count(&self) -> usize {
        self.ingested.len()
    }

    pub fn consumed_authorization_count(&self) -> usize {
        self.consumed_authorizations.len()
    }
}

fn deterministic_context(identity: &EvidenceDraftIdentity) -> Option<String> {
    Some(format!("inquiry-result: {}", identity.result_summary))
}

fn deterministic_method(identity: &EvidenceDraftIdentity) -> Option<String> {
    Some(format!(
        "preregistered-decision[{}]: {}",
        identity.decision_rule_label, identity.decision_criterion
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        ClaimKind, DecisionInterpretation, EvidencePolarity, IgnoranceFrontier,
        InquiryContractBuilder, InquiryPreregistration, InquiryRequest, InquiryResultReceipt,
        PreregisteredDecisionRule, ReceiptAdmissionGate, ReceiptAdmissionPolicy,
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

    fn admitted_draft(
        rule_label: &str,
        evidence_kind: EvidenceKind,
    ) -> (EpistemicLedger, AdmissibleEvidenceDraft) {
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
            evidence_kind,
            provenance,
            6,
            "observed result",
            rule_label,
        )
        .unwrap();
        let policy = ReceiptAdmissionPolicy::new(vec![evidence_kind], true, true, true);
        let admission = ReceiptAdmissionGate::evaluate(&ledger, &receipt, &policy);
        (ledger, admission.draft().unwrap().clone())
    }

    #[test]
    fn approved_draft_is_inserted_exactly_once() {
        let (mut ledger, draft) = admitted_draft("supports", EvidenceKind::Measurement);
        let before = ledger.evidence_count();
        let authorization = EvidenceMutationAuthorization::new(
            "auth-1",
            "test-authority",
            MutationAuthorizationDecision::Approved,
            7,
            &draft,
        )
        .unwrap();
        let mut firewall = EvidenceMutationFirewall::new();

        let first = firewall
            .ingest(&mut ledger, &draft, &authorization, 8)
            .unwrap();
        assert!(first.inserted_new_record());
        assert_eq!(ledger.evidence_count(), before + 1);

        let second = firewall
            .ingest(&mut ledger, &draft, &authorization, 9)
            .unwrap();
        assert!(!second.inserted_new_record());
        assert_eq!(second.receipt().evidence_id(), first.receipt().evidence_id());
        assert_eq!(ledger.evidence_count(), before + 1);
    }

    #[test]
    fn rebuilt_firewall_detects_exact_persisted_record() {
        let (mut ledger, draft) = admitted_draft("supports", EvidenceKind::Measurement);
        let authorization = EvidenceMutationAuthorization::new(
            "auth-1",
            "test-authority",
            MutationAuthorizationDecision::Approved,
            7,
            &draft,
        )
        .unwrap();
        let mut first_firewall = EvidenceMutationFirewall::new();
        let inserted = first_firewall
            .ingest(&mut ledger, &draft, &authorization, 8)
            .unwrap();
        let evidence_count = ledger.evidence_count();

        let restart_authorization = EvidenceMutationAuthorization::new(
            "auth-after-restart",
            "test-authority",
            MutationAuthorizationDecision::Approved,
            9,
            &draft,
        )
        .unwrap();
        let mut rebuilt = EvidenceMutationFirewall::new();
        let replay = rebuilt
            .ingest(&mut ledger, &draft, &restart_authorization, 10)
            .unwrap();
        assert!(!replay.inserted_new_record());
        assert_eq!(replay.receipt().evidence_id(), inserted.receipt().evidence_id());
        assert_eq!(ledger.evidence_count(), evidence_count);
    }

    #[test]
    fn authorization_cannot_be_replayed_for_different_draft() {
        let (mut ledger, support) = admitted_draft("supports", EvidenceKind::Measurement);
        let (_, contradiction) = admitted_draft("contradicts", EvidenceKind::Measurement);
        let authorization = EvidenceMutationAuthorization::new(
            "auth-shared",
            "test-authority",
            MutationAuthorizationDecision::Approved,
            7,
            &support,
        )
        .unwrap();
        let mut firewall = EvidenceMutationFirewall::new();
        firewall
            .ingest(&mut ledger, &support, &authorization, 8)
            .unwrap();

        // The exact authorization/draft binding fails before any second mutation.
        assert_eq!(
            firewall
                .ingest(&mut ledger, &contradiction, &authorization, 9)
                .unwrap_err(),
            EvidenceMutationError::AuthorizationDraftMismatch
        );
    }

    #[test]
    fn denied_authorization_never_mutates_ledger() {
        let (mut ledger, draft) = admitted_draft("supports", EvidenceKind::Measurement);
        let before = ledger.evidence_count();
        let authorization = EvidenceMutationAuthorization::new(
            "auth-denied",
            "test-authority",
            MutationAuthorizationDecision::Denied,
            7,
            &draft,
        )
        .unwrap();
        let mut firewall = EvidenceMutationFirewall::new();
        assert_eq!(
            firewall
                .ingest(&mut ledger, &draft, &authorization, 8)
                .unwrap_err(),
            EvidenceMutationError::AuthorizationDenied
        );
        assert_eq!(ledger.evidence_count(), before);
    }

    #[test]
    fn report_evidence_does_not_gain_interventional_status_during_ingestion() {
        let (mut ledger, draft) = admitted_draft("supports", EvidenceKind::Report);
        let claim_id = draft.claim_id();
        let authorization = EvidenceMutationAuthorization::new(
            "auth-report",
            "test-authority",
            MutationAuthorizationDecision::Approved,
            7,
            &draft,
        )
        .unwrap();
        let mut firewall = EvidenceMutationFirewall::new();
        let outcome = firewall
            .ingest(&mut ledger, &draft, &authorization, 8)
            .unwrap();

        let record = ledger.evidence(outcome.receipt().evidence_id()).unwrap();
        assert_eq!(record.kind, EvidenceKind::Report);
        assert_eq!(record.polarity, EvidencePolarity::Supports);
        assert_eq!(ledger.interventional_support_count(claim_id), 0);
    }

    #[test]
    fn temporal_ordering_fails_closed() {
        let (_, draft) = admitted_draft("supports", EvidenceKind::Measurement);
        assert!(matches!(
            EvidenceMutationAuthorization::new(
                "too-early",
                "test-authority",
                MutationAuthorizationDecision::Approved,
                draft.observed_at_cycle() - 1,
                &draft,
            ),
            Err(EvidenceMutationError::AuthorizationPredatesObservation { .. })
        ));
    }
}
