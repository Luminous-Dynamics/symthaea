// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Contestability, correction, and redress for Wisdom & Care decisions.
//!
//! Affected people may challenge Symthaea's factual model, normative reasoning,
//! consent interpretation, or consequential decision without first proving the
//! challenge. Supporting evidence is optional at intake. Evidence is required to
//! close a challenge as corrected/upheld; deferred challenges remain active.
//!
//! WCARE-15 is advisory/shadow-only: it recommends reassessment or pause but does
//! not itself mint, revoke, or bypass executable authority.

use std::collections::{BTreeMap, BTreeSet};

use crate::consent::{ConsentLedger, ConsentScopeId, ConsentState};
use crate::evidence_ledger::{
    DecisionId, DeliberationEvidenceLedger, EvidenceRelation, FactClaimId, NormativeClaimId,
};
use crate::perspective::{PerspectiveGraph, StakeholderId};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChallengeId(String);

impl ChallengeId {
    pub fn new(value: impl Into<String>) -> Result<Self, ContestabilityError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ContestabilityError::EmptyChallengeId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChallengeTarget {
    Fact(FactClaimId),
    NormativeClaim(NormativeClaimId),
    Decision(DecisionId),
    ConsentScope {
        stakeholder: StakeholderId,
        scope: ConsentScopeId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChallengeKind {
    FactualCorrection,
    NormativeDispute,
    ConsentDispute,
    MissingContext,
    HarmReport,
    AuthorityDispute,
    ExplanationRequest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChallengeStatus {
    Open,
    Acknowledged,
    Resolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionKind {
    Corrected,
    Upheld,
    Deferred,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetSnapshot {
    ExistingFact,
    ExistingNormativeClaim,
    ExistingDecision,
    Consent {
        record_present: bool,
        state: Option<ConsentState>,
        recorded_revision: Option<u64>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChallengeResolution {
    pub kind: ResolutionKind,
    pub explanation: String,
    pub evidence: BTreeSet<FactClaimId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChallengeRecord {
    pub id: ChallengeId,
    pub challenger: StakeholderId,
    pub target: ChallengeTarget,
    pub kind: ChallengeKind,
    pub summary: String,
    pub supporting_facts: BTreeSet<FactClaimId>,
    pub target_snapshot: TargetSnapshot,
    pub status: ChallengeStatus,
    pub resolution: Option<ChallengeResolution>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContestabilityDisposition {
    ProvideExplanation,
    ReassessBeforeConsequentialAction,
    PauseConsequentialActionPendingReview,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContestabilityAssessment {
    pub open_challenges: Vec<ChallengeId>,
    pub disposition: Option<ContestabilityDisposition>,
    pub shadow_only: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ContestabilityLedger {
    challenges: BTreeMap<ChallengeId, ChallengeRecord>,
}

impl ContestabilityLedger {
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open_challenge(
        &mut self,
        id: ChallengeId,
        challenger: StakeholderId,
        target: ChallengeTarget,
        kind: ChallengeKind,
        summary: impl Into<String>,
        supporting_facts: impl IntoIterator<Item = FactClaimId>,
        evidence: &DeliberationEvidenceLedger,
        consent: &ConsentLedger,
        perspectives: &PerspectiveGraph,
    ) -> Result<(), ContestabilityError> {
        if self.challenges.contains_key(&id) {
            return Err(ContestabilityError::DuplicateChallenge(id));
        }
        if perspectives.find(&challenger).is_none() {
            return Err(ContestabilityError::UnknownChallenger(challenger));
        }

        let summary = summary.into();
        if summary.trim().is_empty() {
            return Err(ContestabilityError::EmptySummary);
        }
        let supporting_facts: BTreeSet<_> = supporting_facts.into_iter().collect();
        validate_fact_refs(&supporting_facts, evidence)?;
        let target_snapshot = snapshot_target(&target, evidence, consent, perspectives)?;

        self.challenges.insert(
            id.clone(),
            ChallengeRecord {
                id,
                challenger,
                target,
                kind,
                summary,
                supporting_facts,
                target_snapshot,
                status: ChallengeStatus::Open,
                resolution: None,
            },
        );
        Ok(())
    }

    pub fn acknowledge(&mut self, id: &ChallengeId) -> Result<(), ContestabilityError> {
        let challenge = self
            .challenges
            .get_mut(id)
            .ok_or_else(|| ContestabilityError::MissingChallenge(id.clone()))?;
        if challenge.status == ChallengeStatus::Resolved {
            return Err(ContestabilityError::ChallengeAlreadyResolved(id.clone()));
        }
        challenge.status = ChallengeStatus::Acknowledged;
        Ok(())
    }

    pub fn resolve(
        &mut self,
        id: &ChallengeId,
        kind: ResolutionKind,
        explanation: impl Into<String>,
        resolution_evidence: impl IntoIterator<Item = FactClaimId>,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), ContestabilityError> {
        let explanation = explanation.into();
        if explanation.trim().is_empty() {
            return Err(ContestabilityError::ResolutionExplanationRequired);
        }
        let resolution_evidence: BTreeSet<_> = resolution_evidence.into_iter().collect();
        validate_fact_refs(&resolution_evidence, evidence)?;
        if kind != ResolutionKind::Deferred {
            if resolution_evidence.is_empty() {
                return Err(ContestabilityError::ResolutionEvidenceRequired);
            }
            for fact in &resolution_evidence {
                require_qualified_fact(fact, evidence)?;
            }
        }

        let challenge = self
            .challenges
            .get_mut(id)
            .ok_or_else(|| ContestabilityError::MissingChallenge(id.clone()))?;
        if challenge.status == ChallengeStatus::Resolved {
            return Err(ContestabilityError::ChallengeAlreadyResolved(id.clone()));
        }
        challenge.status = if kind == ResolutionKind::Deferred {
            ChallengeStatus::Acknowledged
        } else {
            ChallengeStatus::Resolved
        };
        challenge.resolution = Some(ChallengeResolution {
            kind,
            explanation,
            evidence: resolution_evidence,
        });
        Ok(())
    }

    pub fn get(&self, id: &ChallengeId) -> Option<&ChallengeRecord> {
        self.challenges.get(id)
    }

    pub fn assess(&self) -> ContestabilityAssessment {
        let active: Vec<_> = self
            .challenges
            .values()
            .filter(|challenge| challenge.status != ChallengeStatus::Resolved)
            .collect();
        ContestabilityAssessment {
            open_challenges: active.iter().map(|challenge| challenge.id.clone()).collect(),
            disposition: active
                .iter()
                .map(|challenge| disposition_for(challenge.kind))
                .max(),
            shadow_only: true,
        }
    }
}

fn disposition_for(kind: ChallengeKind) -> ContestabilityDisposition {
    match kind {
        ChallengeKind::ExplanationRequest => ContestabilityDisposition::ProvideExplanation,
        ChallengeKind::NormativeDispute | ChallengeKind::MissingContext => {
            ContestabilityDisposition::ReassessBeforeConsequentialAction
        }
        ChallengeKind::FactualCorrection
        | ChallengeKind::ConsentDispute
        | ChallengeKind::HarmReport
        | ChallengeKind::AuthorityDispute => {
            ContestabilityDisposition::PauseConsequentialActionPendingReview
        }
    }
}

fn snapshot_target(
    target: &ChallengeTarget,
    evidence: &DeliberationEvidenceLedger,
    consent: &ConsentLedger,
    perspectives: &PerspectiveGraph,
) -> Result<TargetSnapshot, ContestabilityError> {
    match target {
        ChallengeTarget::Fact(id) => evidence
            .facts()
            .get(id)
            .map(|_| TargetSnapshot::ExistingFact)
            .ok_or_else(|| ContestabilityError::MissingFact(id.clone())),
        ChallengeTarget::NormativeClaim(id) => evidence
            .normative()
            .get(id)
            .map(|_| TargetSnapshot::ExistingNormativeClaim)
            .ok_or_else(|| ContestabilityError::MissingNormativeClaim(id.clone())),
        ChallengeTarget::Decision(id) => evidence
            .decisions()
            .get(id)
            .map(|_| TargetSnapshot::ExistingDecision)
            .ok_or_else(|| ContestabilityError::MissingDecision(id.clone())),
        ChallengeTarget::ConsentScope { stakeholder, scope } => {
            if perspectives.find(stakeholder).is_none() {
                return Err(ContestabilityError::UnknownConsentStakeholder(
                    stakeholder.clone(),
                ));
            }
            let record = consent.latest(stakeholder, scope);
            Ok(TargetSnapshot::Consent {
                record_present: record.is_some(),
                state: record.map(|record| record.state),
                recorded_revision: record.map(|record| record.recorded_revision),
            })
        }
    }
}

fn validate_fact_refs(
    ids: &BTreeSet<FactClaimId>,
    evidence: &DeliberationEvidenceLedger,
) -> Result<(), ContestabilityError> {
    for id in ids {
        if evidence.facts().get(id).is_none() {
            return Err(ContestabilityError::MissingFact(id.clone()));
        }
    }
    Ok(())
}

fn require_qualified_fact(
    id: &FactClaimId,
    evidence: &DeliberationEvidenceLedger,
) -> Result<(), ContestabilityError> {
    let claim = evidence
        .facts()
        .get(id)
        .ok_or_else(|| ContestabilityError::MissingFact(id.clone()))?;
    if claim.superseded_by.is_some() {
        return Err(ContestabilityError::ResolutionFactSuperseded(id.clone()));
    }
    let mut supported = false;
    for observation in &claim.evidence {
        match observation.relation {
            EvidenceRelation::Supports => supported = true,
            EvidenceRelation::Contradicts => {
                return Err(ContestabilityError::ResolutionFactContradicted(id.clone()));
            }
        }
    }
    if !supported {
        return Err(ContestabilityError::ResolutionFactUnsupported(id.clone()));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContestabilityError {
    EmptyChallengeId,
    DuplicateChallenge(ChallengeId),
    MissingChallenge(ChallengeId),
    ChallengeAlreadyResolved(ChallengeId),
    UnknownChallenger(StakeholderId),
    UnknownConsentStakeholder(StakeholderId),
    EmptySummary,
    ResolutionExplanationRequired,
    ResolutionEvidenceRequired,
    MissingFact(FactClaimId),
    MissingNormativeClaim(NormativeClaimId),
    MissingDecision(DecisionId),
    ResolutionFactUnsupported(FactClaimId),
    ResolutionFactContradicted(FactClaimId),
    ResolutionFactSuperseded(FactClaimId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consent::{ConsentProvenance, ConsentRecord};
    use crate::evidence_ledger::EvidenceObservation;
    use crate::perspective::{PerspectiveCoverage, StakeholderPerspective};

    fn fact(value: &str) -> FactClaimId {
        FactClaimId::new(value).unwrap()
    }

    fn person(value: &str) -> StakeholderId {
        StakeholderId::new(value).unwrap()
    }

    fn context() -> (DeliberationEvidenceLedger, ConsentLedger, PerspectiveGraph) {
        let mut evidence = DeliberationEvidenceLedger::new();
        evidence
            .add_fact(fact("user-statement"), "person disputes the claim", 0.9)
            .unwrap();
        evidence
            .add_fact(fact("resolution"), "correction independently verified", 0.9)
            .unwrap();
        let normative = NormativeClaimId::new("n1").unwrap();
        evidence
            .add_normative_claim(normative.clone(), "respect the person's correction", 0.8)
            .unwrap();
        evidence
            .register_decision(
                DecisionId::new("d1").unwrap(),
                [fact("user-statement")],
                [normative],
            )
            .unwrap();

        let graph = PerspectiveGraph::try_new(vec![StakeholderPerspective::new(
            person("person-a"),
            true,
            PerspectiveCoverage::Adequate,
        )])
        .unwrap();

        let mut consent = ConsentLedger::new();
        consent
            .record(
                ConsentRecord::new(
                    person("person-a"),
                    ConsentScopeId::new("option-a").unwrap(),
                    ConsentState::Affirmed,
                    ConsentProvenance::DirectStatement,
                    [fact("user-statement")],
                    [],
                    [],
                    0.0,
                    1,
                    None,
                )
                .unwrap(),
                &evidence,
            )
            .unwrap();
        (evidence, consent, graph)
    }

    #[test]
    fn challenge_intake_does_not_require_formal_proof() {
        let (evidence, consent, graph) = context();
        let mut ledger = ContestabilityLedger::new();
        for (id, kind) in [
            ("harm", ChallengeKind::HarmReport),
            ("correction", ChallengeKind::FactualCorrection),
            ("authority", ChallengeKind::AuthorityDispute),
        ] {
            ledger
                .open_challenge(
                    ChallengeId::new(id).unwrap(),
                    person("person-a"),
                    ChallengeTarget::Decision(DecisionId::new("d1").unwrap()),
                    kind,
                    "please review this",
                    [],
                    &evidence,
                    &consent,
                    &graph,
                )
                .unwrap();
        }
        assert_eq!(ledger.assess().open_challenges.len(), 3);
        assert_eq!(
            ledger.assess().disposition,
            Some(ContestabilityDisposition::PauseConsequentialActionPendingReview)
        );
    }

    #[test]
    fn consent_dispute_preserves_opening_snapshot() {
        let (evidence, consent, graph) = context();
        let mut ledger = ContestabilityLedger::new();
        let id = ChallengeId::new("c1").unwrap();
        ledger
            .open_challenge(
                id.clone(),
                person("person-a"),
                ChallengeTarget::ConsentScope {
                    stakeholder: person("person-a"),
                    scope: ConsentScopeId::new("option-a").unwrap(),
                },
                ChallengeKind::ConsentDispute,
                "that affirmation no longer represents my wishes",
                [],
                &evidence,
                &consent,
                &graph,
            )
            .unwrap();
        assert_eq!(
            ledger.get(&id).unwrap().target_snapshot,
            TargetSnapshot::Consent {
                record_present: true,
                state: Some(ConsentState::Affirmed),
                recorded_revision: Some(1),
            }
        );
    }

    #[test]
    fn explanation_request_recommends_explanation_not_pause() {
        let (evidence, consent, graph) = context();
        let mut ledger = ContestabilityLedger::new();
        ledger
            .open_challenge(
                ChallengeId::new("c1").unwrap(),
                person("person-a"),
                ChallengeTarget::Decision(DecisionId::new("d1").unwrap()),
                ChallengeKind::ExplanationRequest,
                "please explain the basis",
                [],
                &evidence,
                &consent,
                &graph,
            )
            .unwrap();
        assert_eq!(
            ledger.assess().disposition,
            Some(ContestabilityDisposition::ProvideExplanation)
        );
    }

    #[test]
    fn unsupported_resolution_cannot_close_challenge() {
        let (evidence, consent, graph) = context();
        let mut ledger = ContestabilityLedger::new();
        let id = ChallengeId::new("c1").unwrap();
        ledger
            .open_challenge(
                id.clone(),
                person("person-a"),
                ChallengeTarget::Decision(DecisionId::new("d1").unwrap()),
                ChallengeKind::AuthorityDispute,
                "challenge",
                [],
                &evidence,
                &consent,
                &graph,
            )
            .unwrap();
        assert_eq!(
            ledger.resolve(
                &id,
                ResolutionKind::Upheld,
                "review completed",
                [fact("resolution")],
                &evidence,
            ),
            Err(ContestabilityError::ResolutionFactUnsupported(fact("resolution")))
        );
        assert_ne!(ledger.get(&id).unwrap().status, ChallengeStatus::Resolved);
    }

    #[test]
    fn evidence_bound_resolution_can_close_challenge() {
        let (mut evidence, consent, graph) = context();
        evidence
            .add_fact_evidence(
                &fact("resolution"),
                EvidenceObservation::new("independent-review", EvidenceRelation::Supports, 0.9)
                    .unwrap(),
            )
            .unwrap();
        let mut ledger = ContestabilityLedger::new();
        let id = ChallengeId::new("c1").unwrap();
        ledger
            .open_challenge(
                id.clone(),
                person("person-a"),
                ChallengeTarget::Decision(DecisionId::new("d1").unwrap()),
                ChallengeKind::AuthorityDispute,
                "challenge",
                [],
                &evidence,
                &consent,
                &graph,
            )
            .unwrap();
        ledger
            .resolve(
                &id,
                ResolutionKind::Corrected,
                "decision corrected after independent review",
                [fact("resolution")],
                &evidence,
            )
            .unwrap();
        assert_eq!(ledger.get(&id).unwrap().status, ChallengeStatus::Resolved);
        assert!(ledger.assess().open_challenges.is_empty());
    }

    #[test]
    fn deferred_resolution_keeps_challenge_active() {
        let (evidence, consent, graph) = context();
        let mut ledger = ContestabilityLedger::new();
        let id = ChallengeId::new("c1").unwrap();
        ledger
            .open_challenge(
                id.clone(),
                person("person-a"),
                ChallengeTarget::Decision(DecisionId::new("d1").unwrap()),
                ChallengeKind::MissingContext,
                "important context is missing",
                [],
                &evidence,
                &consent,
                &graph,
            )
            .unwrap();
        ledger
            .resolve(
                &id,
                ResolutionKind::Deferred,
                "awaiting additional context",
                [],
                &evidence,
            )
            .unwrap();
        assert_eq!(ledger.get(&id).unwrap().status, ChallengeStatus::Acknowledged);
        assert_eq!(ledger.assess().open_challenges, vec![id]);
    }
}
