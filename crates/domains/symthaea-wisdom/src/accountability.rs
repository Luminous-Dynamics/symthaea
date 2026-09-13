// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consequence, responsibility, and repair accounting for Wisdom & Care.
//!
//! Consequential decisions retain the premises and relational-authority ceiling
//! they relied on. Predictions, observed outcomes, discrepancies, harms, and
//! repair obligations remain explicit. A repair can be completed only by the
//! exact factual completion claim bound to that repair when it was created.

use std::collections::{BTreeSet, HashSet};

use crate::authority_envelope::{
    AuthorityRestrictionReason, RelationalAuthorityAssessment,
};
use crate::evidence_ledger::{
    DecisionId, DeliberationEvidenceLedger, EvidenceRelation, FactClaimId, NormativeClaimId,
};
use crate::ontology::ActionAuthority;
use crate::perspective::StakeholderId;

macro_rules! accountability_id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, AccountabilityError> {
                let value = value.into();
                if value.trim().is_empty() {
                    return Err(AccountabilityError::EmptyIdentifier);
                }
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
    };
}

accountability_id_type!(AccountabilityCaseId);
accountability_id_type!(RepairActionId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConsequenceDomain {
    Agency,
    Wellbeing,
    Safety,
    Truthfulness,
    RelationshipIndependence,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsequenceDirection {
    Worse,
    Unchanged,
    Better,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PredictedConsequence {
    pub stakeholder: StakeholderId,
    pub domain: ConsequenceDomain,
    pub direction: ConsequenceDirection,
    pub confidence: f32,
}

impl PredictedConsequence {
    pub fn new(
        stakeholder: StakeholderId,
        domain: ConsequenceDomain,
        direction: ConsequenceDirection,
        confidence: f32,
    ) -> Result<Self, AccountabilityError> {
        validate_metric(confidence)?;
        Ok(Self {
            stakeholder,
            domain,
            direction,
            confidence,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservedConsequence {
    pub stakeholder: StakeholderId,
    pub domain: ConsequenceDomain,
    pub direction: ConsequenceDirection,
    pub confidence: f32,
    pub harm_detected: bool,
    pub fact_evidence: BTreeSet<FactClaimId>,
}

impl ObservedConsequence {
    pub fn new(
        stakeholder: StakeholderId,
        domain: ConsequenceDomain,
        direction: ConsequenceDirection,
        confidence: f32,
        harm_detected: bool,
        fact_evidence: impl IntoIterator<Item = FactClaimId>,
    ) -> Result<Self, AccountabilityError> {
        validate_metric(confidence)?;
        let fact_evidence: BTreeSet<_> = fact_evidence.into_iter().collect();
        if fact_evidence.is_empty() {
            return Err(AccountabilityError::OutcomeEvidenceRequired);
        }
        Ok(Self {
            stakeholder,
            domain,
            direction,
            confidence,
            harm_detected,
            fact_evidence,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorityBasisSnapshot {
    pub requested: ActionAuthority,
    pub ceiling: ActionAuthority,
    pub restriction_reasons: BTreeSet<AuthorityRestrictionReason>,
    pub human_review_recommended: bool,
}

impl From<&RelationalAuthorityAssessment> for AuthorityBasisSnapshot {
    fn from(value: &RelationalAuthorityAssessment) -> Self {
        Self {
            requested: value.requested(),
            ceiling: value.ceiling(),
            restriction_reasons: value.reasons().clone(),
            human_review_recommended: value.human_review_recommended(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccountabilityTrigger {
    PremiseReviewRequired,
    MissingOutcome {
        stakeholder: StakeholderId,
        domain: ConsequenceDomain,
    },
    PredictionMismatch {
        stakeholder: StakeholderId,
        domain: ConsequenceDomain,
    },
    ObservedHarm {
        stakeholder: StakeholderId,
        domain: ConsequenceDomain,
    },
    AgencyWorsened(StakeholderId),
    RelationshipIndependenceWorsened(StakeholderId),
    AuthorityCeilingExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RepairKind {
    RetractOrCorrectAdvice,
    InformAffectedParty,
    RestoreChoice,
    ReassessPremises,
    EscalateIndependentReview,
    MitigateHarm,
    ReduceDependency,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairStatus {
    Open,
    InProgress,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairCompletionReceipt {
    pub completion_claim: FactClaimId,
    pub supporting_sources: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepairAction {
    id: RepairActionId,
    kind: RepairKind,
    stakeholder: Option<StakeholderId>,
    completion_claim: FactClaimId,
    status: RepairStatus,
    completion_receipt: Option<RepairCompletionReceipt>,
}

impl RepairAction {
    pub fn new(
        id: RepairActionId,
        kind: RepairKind,
        stakeholder: Option<StakeholderId>,
        completion_claim: FactClaimId,
    ) -> Self {
        Self {
            id,
            kind,
            stakeholder,
            completion_claim,
            status: RepairStatus::Open,
            completion_receipt: None,
        }
    }

    pub fn id(&self) -> &RepairActionId {
        &self.id
    }

    pub fn kind(&self) -> RepairKind {
        self.kind
    }

    pub fn stakeholder(&self) -> Option<&StakeholderId> {
        self.stakeholder.as_ref()
    }

    pub fn completion_claim(&self) -> &FactClaimId {
        &self.completion_claim
    }

    pub fn status(&self) -> RepairStatus {
        self.status
    }

    pub fn completion_receipt(&self) -> Option<&RepairCompletionReceipt> {
        self.completion_receipt.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AccountabilityAssessment {
    pub triggers: Vec<AccountabilityTrigger>,
    pub suggested_repairs: BTreeSet<RepairKind>,
    pub open_repairs: Vec<RepairActionId>,
}

impl AccountabilityAssessment {
    pub fn requires_followup(&self) -> bool {
        !self.triggers.is_empty() || !self.open_repairs.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct AccountabilityCase {
    pub id: AccountabilityCaseId,
    pub decision: DecisionId,
    pub affected_stakeholders: HashSet<StakeholderId>,
    pub premise_snapshot_facts: BTreeSet<FactClaimId>,
    pub premise_snapshot_normative: BTreeSet<NormativeClaimId>,
    pub authority: AuthorityBasisSnapshot,
    pub reversible: bool,
    predictions: Vec<PredictedConsequence>,
    observations: Vec<ObservedConsequence>,
    repairs: Vec<RepairAction>,
    executed_authority: Option<ActionAuthority>,
}

impl AccountabilityCase {
    pub fn try_new(
        id: AccountabilityCaseId,
        decision: DecisionId,
        affected_stakeholders: impl IntoIterator<Item = StakeholderId>,
        authority: &RelationalAuthorityAssessment,
        reversible: bool,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<Self, AccountabilityError> {
        let decision_record = evidence
            .decisions()
            .get(&decision)
            .ok_or_else(|| AccountabilityError::MissingDecision(decision.clone()))?;
        let affected_stakeholders: HashSet<_> = affected_stakeholders.into_iter().collect();
        if affected_stakeholders.is_empty() {
            return Err(AccountabilityError::AffectedStakeholderRequired);
        }
        if authority.ceiling() > authority.requested() {
            return Err(AccountabilityError::InvalidAuthoritySnapshot);
        }
        Ok(Self {
            id,
            decision,
            affected_stakeholders,
            premise_snapshot_facts: decision_record.fact_premises.clone(),
            premise_snapshot_normative: decision_record.normative_premises.clone(),
            authority: AuthorityBasisSnapshot::from(authority),
            reversible,
            predictions: Vec::new(),
            observations: Vec::new(),
            repairs: Vec::new(),
            executed_authority: None,
        })
    }

    pub fn add_prediction(
        &mut self,
        prediction: PredictedConsequence,
    ) -> Result<(), AccountabilityError> {
        self.require_affected(&prediction.stakeholder)?;
        self.predictions.push(prediction);
        Ok(())
    }

    pub fn record_observation(
        &mut self,
        observation: ObservedConsequence,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), AccountabilityError> {
        self.require_affected(&observation.stakeholder)?;
        validate_fact_refs(&observation.fact_evidence, evidence)?;
        self.observations.push(observation);
        Ok(())
    }

    pub fn record_execution(
        &mut self,
        actual_authority: ActionAuthority,
    ) -> Result<(), AccountabilityError> {
        if self.executed_authority.is_some() {
            return Err(AccountabilityError::ExecutionAlreadyRecorded);
        }
        self.executed_authority = Some(actual_authority);
        Ok(())
    }

    pub fn add_repair(
        &mut self,
        repair: RepairAction,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), AccountabilityError> {
        if self.repairs.iter().any(|existing| existing.id == repair.id) {
            return Err(AccountabilityError::DuplicateRepair(repair.id));
        }
        if let Some(stakeholder) = &repair.stakeholder {
            self.require_affected(stakeholder)?;
        }
        if evidence.facts().get(&repair.completion_claim).is_none() {
            return Err(AccountabilityError::MissingFact(
                repair.completion_claim.clone(),
            ));
        }
        self.repairs.push(repair);
        Ok(())
    }

    pub fn start_repair(&mut self, id: &RepairActionId) -> Result<(), AccountabilityError> {
        let repair = self
            .repairs
            .iter_mut()
            .find(|repair| &repair.id == id)
            .ok_or_else(|| AccountabilityError::MissingRepair(id.clone()))?;
        if repair.status == RepairStatus::Completed {
            return Err(AccountabilityError::RepairAlreadyCompleted(id.clone()));
        }
        repair.status = RepairStatus::InProgress;
        Ok(())
    }

    pub fn complete_repair(
        &mut self,
        id: &RepairActionId,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), AccountabilityError> {
        let repair = self
            .repairs
            .iter_mut()
            .find(|repair| &repair.id == id)
            .ok_or_else(|| AccountabilityError::MissingRepair(id.clone()))?;
        if repair.status == RepairStatus::Completed {
            return Err(AccountabilityError::RepairAlreadyCompleted(id.clone()));
        }

        let claim = evidence
            .facts()
            .get(&repair.completion_claim)
            .ok_or_else(|| AccountabilityError::MissingFact(repair.completion_claim.clone()))?;
        if claim.superseded_by.is_some() {
            return Err(AccountabilityError::RepairCompletionClaimSuperseded(
                repair.completion_claim.clone(),
            ));
        }

        let mut supporting_sources = BTreeSet::new();
        let mut contradicted = false;
        for observation in &claim.evidence {
            match observation.relation {
                EvidenceRelation::Supports => {
                    supporting_sources.insert(observation.source_ref.clone());
                }
                EvidenceRelation::Contradicts => contradicted = true,
            }
        }
        if contradicted {
            return Err(AccountabilityError::RepairCompletionClaimContradicted(
                repair.completion_claim.clone(),
            ));
        }
        if supporting_sources.is_empty() {
            return Err(AccountabilityError::RepairEvidenceRequired);
        }

        repair.status = RepairStatus::Completed;
        repair.completion_receipt = Some(RepairCompletionReceipt {
            completion_claim: repair.completion_claim.clone(),
            supporting_sources: supporting_sources.into_iter().collect(),
        });
        Ok(())
    }

    pub fn repairs(&self) -> &[RepairAction] {
        &self.repairs
    }

    pub fn assess(
        &self,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<AccountabilityAssessment, AccountabilityError> {
        let decision_record = evidence
            .decisions()
            .get(&self.decision)
            .ok_or_else(|| AccountabilityError::MissingDecision(self.decision.clone()))?;

        let mut triggers = Vec::new();
        if decision_record.requires_review() {
            push_unique(&mut triggers, AccountabilityTrigger::PremiseReviewRequired);
        }
        if self
            .executed_authority
            .is_some_and(|actual| actual > self.authority.ceiling)
        {
            push_unique(&mut triggers, AccountabilityTrigger::AuthorityCeilingExceeded);
        }

        for prediction in &self.predictions {
            let observation = self.observations.iter().rev().find(|observation| {
                observation.stakeholder == prediction.stakeholder
                    && observation.domain == prediction.domain
            });
            let Some(observation) = observation else {
                push_unique(
                    &mut triggers,
                    AccountabilityTrigger::MissingOutcome {
                        stakeholder: prediction.stakeholder.clone(),
                        domain: prediction.domain,
                    },
                );
                continue;
            };
            if prediction.direction != ConsequenceDirection::Unknown
                && observation.direction != ConsequenceDirection::Unknown
                && prediction.direction != observation.direction
            {
                push_unique(
                    &mut triggers,
                    AccountabilityTrigger::PredictionMismatch {
                        stakeholder: prediction.stakeholder.clone(),
                        domain: prediction.domain,
                    },
                );
            }
        }

        for observation in &self.observations {
            if observation.harm_detected {
                push_unique(
                    &mut triggers,
                    AccountabilityTrigger::ObservedHarm {
                        stakeholder: observation.stakeholder.clone(),
                        domain: observation.domain,
                    },
                );
            }
            if observation.domain == ConsequenceDomain::Agency
                && observation.direction == ConsequenceDirection::Worse
            {
                push_unique(
                    &mut triggers,
                    AccountabilityTrigger::AgencyWorsened(observation.stakeholder.clone()),
                );
            }
            if observation.domain == ConsequenceDomain::RelationshipIndependence
                && observation.direction == ConsequenceDirection::Worse
            {
                push_unique(
                    &mut triggers,
                    AccountabilityTrigger::RelationshipIndependenceWorsened(
                        observation.stakeholder.clone(),
                    ),
                );
            }
        }

        let suggested_repairs = suggested_repairs(&triggers, self.reversible);
        let open_repairs = self
            .repairs
            .iter()
            .filter(|repair| repair.status != RepairStatus::Completed)
            .map(|repair| repair.id.clone())
            .collect();

        Ok(AccountabilityAssessment {
            triggers,
            suggested_repairs,
            open_repairs,
        })
    }

    fn require_affected(&self, stakeholder: &StakeholderId) -> Result<(), AccountabilityError> {
        if self.affected_stakeholders.contains(stakeholder) {
            Ok(())
        } else {
            Err(AccountabilityError::UnknownAffectedStakeholder(
                stakeholder.clone(),
            ))
        }
    }
}

fn suggested_repairs(
    triggers: &[AccountabilityTrigger],
    reversible: bool,
) -> BTreeSet<RepairKind> {
    let mut repairs = BTreeSet::new();
    for trigger in triggers {
        match trigger {
            AccountabilityTrigger::PremiseReviewRequired => {
                repairs.insert(RepairKind::ReassessPremises);
                repairs.insert(RepairKind::RetractOrCorrectAdvice);
            }
            AccountabilityTrigger::MissingOutcome { .. } => {}
            AccountabilityTrigger::PredictionMismatch { .. } => {
                repairs.insert(RepairKind::ReassessPremises);
            }
            AccountabilityTrigger::ObservedHarm { .. } => {
                repairs.insert(RepairKind::MitigateHarm);
                repairs.insert(RepairKind::InformAffectedParty);
            }
            AccountabilityTrigger::AgencyWorsened(_) => {
                repairs.insert(RepairKind::InformAffectedParty);
                if reversible {
                    repairs.insert(RepairKind::RestoreChoice);
                }
            }
            AccountabilityTrigger::RelationshipIndependenceWorsened(_) => {
                repairs.insert(RepairKind::ReduceDependency);
            }
            AccountabilityTrigger::AuthorityCeilingExceeded => {
                repairs.insert(RepairKind::EscalateIndependentReview);
                repairs.insert(RepairKind::InformAffectedParty);
            }
        }
    }
    repairs
}

fn push_unique(triggers: &mut Vec<AccountabilityTrigger>, trigger: AccountabilityTrigger) {
    if !triggers.contains(&trigger) {
        triggers.push(trigger);
    }
}

fn validate_fact_refs(
    ids: &BTreeSet<FactClaimId>,
    evidence: &DeliberationEvidenceLedger,
) -> Result<(), AccountabilityError> {
    for id in ids {
        if evidence.facts().get(id).is_none() {
            return Err(AccountabilityError::MissingFact(id.clone()));
        }
    }
    Ok(())
}

fn validate_metric(value: f32) -> Result<(), AccountabilityError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(AccountabilityError::InvalidMetric(value))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AccountabilityError {
    EmptyIdentifier,
    InvalidMetric(f32),
    MissingDecision(DecisionId),
    MissingFact(FactClaimId),
    AffectedStakeholderRequired,
    UnknownAffectedStakeholder(StakeholderId),
    OutcomeEvidenceRequired,
    InvalidAuthoritySnapshot,
    ExecutionAlreadyRecorded,
    DuplicateRepair(RepairActionId),
    MissingRepair(RepairActionId),
    RepairAlreadyCompleted(RepairActionId),
    RepairEvidenceRequired,
    RepairCompletionClaimContradicted(FactClaimId),
    RepairCompletionClaimSuperseded(FactClaimId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_ledger::EvidenceObservation;

    fn fact(value: &str) -> FactClaimId {
        FactClaimId::new(value).unwrap()
    }

    fn norm(value: &str) -> NormativeClaimId {
        NormativeClaimId::new(value).unwrap()
    }

    fn decision() -> DecisionId {
        DecisionId::new("decision-a").unwrap()
    }

    fn person() -> StakeholderId {
        StakeholderId::new("person-a").unwrap()
    }

    fn evidence() -> DeliberationEvidenceLedger {
        let mut evidence = DeliberationEvidenceLedger::new();
        evidence
            .add_fact(fact("premise"), "premise is true", 0.8)
            .unwrap();
        evidence
            .add_fact(fact("outcome"), "outcome was observed", 0.9)
            .unwrap();
        evidence
            .add_fact(fact("repair-done"), "affected person was informed", 0.9)
            .unwrap();
        evidence
            .add_fact(fact("unrelated"), "an unrelated fact is true", 0.9)
            .unwrap();
        evidence
            .add_normative_claim(norm("value"), "action is justified", 0.7)
            .unwrap();
        evidence
            .register_decision(decision(), [fact("premise")], [norm("value")])
            .unwrap();
        evidence
    }

    fn authority() -> RelationalAuthorityAssessment {
        RelationalAuthorityAssessment::for_test(
            ActionAuthority::ActReversible,
            ActionAuthority::Recommend,
            BTreeSet::new(),
            false,
        )
    }

    fn case(evidence: &DeliberationEvidenceLedger) -> AccountabilityCase {
        AccountabilityCase::try_new(
            AccountabilityCaseId::new("case-a").unwrap(),
            decision(),
            [person()],
            &authority(),
            true,
            evidence,
        )
        .unwrap()
    }

    #[test]
    fn opaque_authority_api_is_consumed_via_accessors() {
        let evidence = evidence();
        let case = case(&evidence);
        assert_eq!(case.authority.requested, ActionAuthority::ActReversible);
        assert_eq!(case.authority.ceiling, ActionAuthority::Recommend);
    }

    #[test]
    fn harmful_prediction_miss_creates_explicit_repair_pressure() {
        let evidence = evidence();
        let mut case = case(&evidence);
        case.add_prediction(
            PredictedConsequence::new(
                person(),
                ConsequenceDomain::Agency,
                ConsequenceDirection::Better,
                0.8,
            )
            .unwrap(),
        )
        .unwrap();
        case.record_observation(
            ObservedConsequence::new(
                person(),
                ConsequenceDomain::Agency,
                ConsequenceDirection::Worse,
                0.9,
                true,
                [fact("outcome")],
            )
            .unwrap(),
            &evidence,
        )
        .unwrap();
        let assessment = case.assess(&evidence).unwrap();
        assert!(assessment.suggested_repairs.contains(&RepairKind::MitigateHarm));
        assert!(assessment.suggested_repairs.contains(&RepairKind::RestoreChoice));
        assert!(assessment
            .suggested_repairs
            .contains(&RepairKind::InformAffectedParty));
    }

    #[test]
    fn superseded_premise_reopens_accountability() {
        let mut evidence = evidence();
        let case = case(&evidence);
        let replacement = fact("replacement");
        evidence
            .add_fact(replacement.clone(), "new premise", 0.95)
            .unwrap();
        evidence
            .supersede_fact(&fact("premise"), &replacement)
            .unwrap();
        assert!(case
            .assess(&evidence)
            .unwrap()
            .triggers
            .contains(&AccountabilityTrigger::PremiseReviewRequired));
    }

    #[test]
    fn executing_above_recorded_ceiling_is_visible() {
        let evidence = evidence();
        let mut case = case(&evidence);
        case.record_execution(ActionAuthority::ActReversible).unwrap();
        assert!(case
            .assess(&evidence)
            .unwrap()
            .triggers
            .contains(&AccountabilityTrigger::AuthorityCeilingExceeded));
    }

    #[test]
    fn repair_requires_its_exact_completion_claim_to_be_supported() {
        let evidence = evidence();
        let mut case = case(&evidence);
        let id = RepairActionId::new("repair-a").unwrap();
        case.add_repair(
            RepairAction::new(
                id.clone(),
                RepairKind::InformAffectedParty,
                Some(person()),
                fact("repair-done"),
            ),
            &evidence,
        )
        .unwrap();
        assert_eq!(
            case.complete_repair(&id, &evidence),
            Err(AccountabilityError::RepairEvidenceRequired)
        );
        assert_eq!(case.repairs()[0].status(), RepairStatus::Open);
    }

    #[test]
    fn unrelated_supported_fact_cannot_complete_repair() {
        let mut evidence = evidence();
        evidence
            .add_fact_evidence(
                &fact("unrelated"),
                EvidenceObservation::new("receipt-x", EvidenceRelation::Supports, 1.0).unwrap(),
            )
            .unwrap();
        let mut case = case(&evidence);
        let id = RepairActionId::new("repair-a").unwrap();
        case.add_repair(
            RepairAction::new(
                id.clone(),
                RepairKind::InformAffectedParty,
                Some(person()),
                fact("repair-done"),
            ),
            &evidence,
        )
        .unwrap();
        assert_eq!(
            case.complete_repair(&id, &evidence),
            Err(AccountabilityError::RepairEvidenceRequired)
        );
    }

    #[test]
    fn supported_exact_claim_mints_completion_receipt() {
        let mut evidence = evidence();
        evidence
            .add_fact_evidence(
                &fact("repair-done"),
                EvidenceObservation::new("person-confirmation", EvidenceRelation::Supports, 0.95)
                    .unwrap(),
            )
            .unwrap();
        let mut case = case(&evidence);
        let id = RepairActionId::new("repair-a").unwrap();
        case.add_repair(
            RepairAction::new(
                id.clone(),
                RepairKind::InformAffectedParty,
                Some(person()),
                fact("repair-done"),
            ),
            &evidence,
        )
        .unwrap();
        case.complete_repair(&id, &evidence).unwrap();
        let repair = &case.repairs()[0];
        assert_eq!(repair.status(), RepairStatus::Completed);
        let receipt = repair.completion_receipt().unwrap();
        assert_eq!(receipt.completion_claim, fact("repair-done"));
        assert_eq!(receipt.supporting_sources, vec!["person-confirmation".to_string()]);
    }

    #[test]
    fn contradicted_completion_claim_fails_closed() {
        let mut evidence = evidence();
        evidence
            .add_fact_evidence(
                &fact("repair-done"),
                EvidenceObservation::new("receipt", EvidenceRelation::Supports, 0.9).unwrap(),
            )
            .unwrap();
        evidence
            .add_fact_evidence(
                &fact("repair-done"),
                EvidenceObservation::new("counter", EvidenceRelation::Contradicts, 0.9).unwrap(),
            )
            .unwrap();
        let mut case = case(&evidence);
        let id = RepairActionId::new("repair-a").unwrap();
        case.add_repair(
            RepairAction::new(
                id.clone(),
                RepairKind::InformAffectedParty,
                Some(person()),
                fact("repair-done"),
            ),
            &evidence,
        )
        .unwrap();
        assert_eq!(
            case.complete_repair(&id, &evidence),
            Err(AccountabilityError::RepairCompletionClaimContradicted(
                fact("repair-done")
            ))
        );
    }

    #[test]
    fn completion_claim_supersession_fails_closed() {
        let mut evidence = evidence();
        let replacement = fact("repair-done-v2");
        evidence
            .add_fact(replacement.clone(), "new repair receipt claim", 0.9)
            .unwrap();
        evidence
            .supersede_fact(&fact("repair-done"), &replacement)
            .unwrap();
        let mut case = case(&evidence);
        let id = RepairActionId::new("repair-a").unwrap();
        case.add_repair(
            RepairAction::new(
                id.clone(),
                RepairKind::InformAffectedParty,
                Some(person()),
                fact("repair-done"),
            ),
            &evidence,
        )
        .unwrap();
        assert_eq!(
            case.complete_repair(&id, &evidence),
            Err(AccountabilityError::RepairCompletionClaimSuperseded(
                fact("repair-done")
            ))
        );
    }
}
