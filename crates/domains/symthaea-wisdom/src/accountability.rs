// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consequence, responsibility, and repair accounting for Wisdom & Care.
//!
//! A consequential decision remains bound to the premises and authority ceiling
//! it relied on. Predictions, observed outcomes, discrepancies, harms, and repair
//! work stay explicit. Repair completion requires evidence; it is not a reward
//! update and it does not erase the original decision history.

use std::collections::{BTreeSet, HashSet};

use crate::authority_envelope::{
    AuthorityRestrictionReason, RelationalAuthorityAssessment,
};
use crate::evidence_ledger::{
    DecisionId, DeliberationEvidenceLedger, FactClaimId, NormativeClaimId,
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
}

impl From<&RelationalAuthorityAssessment> for AuthorityBasisSnapshot {
    fn from(value: &RelationalAuthorityAssessment) -> Self {
        Self {
            requested: value.requested,
            ceiling: value.ceiling,
            restriction_reasons: value.reasons.clone(),
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

#[derive(Debug, Clone, PartialEq)]
pub struct RepairAction {
    id: RepairActionId,
    kind: RepairKind,
    stakeholder: Option<StakeholderId>,
    status: RepairStatus,
    completion_evidence: BTreeSet<FactClaimId>,
}

impl RepairAction {
    pub fn new(
        id: RepairActionId,
        kind: RepairKind,
        stakeholder: Option<StakeholderId>,
    ) -> Self {
        Self {
            id,
            kind,
            stakeholder,
            status: RepairStatus::Open,
            completion_evidence: BTreeSet::new(),
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

    pub fn status(&self) -> RepairStatus {
        self.status
    }

    pub fn completion_evidence(&self) -> &BTreeSet<FactClaimId> {
        &self.completion_evidence
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
        if authority.ceiling > authority.requested {
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

    pub fn add_repair(&mut self, repair: RepairAction) -> Result<(), AccountabilityError> {
        if self.repairs.iter().any(|existing| existing.id == repair.id) {
            return Err(AccountabilityError::DuplicateRepair(repair.id));
        }
        if let Some(stakeholder) = &repair.stakeholder {
            self.require_affected(stakeholder)?;
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
        completion_evidence: impl IntoIterator<Item = FactClaimId>,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), AccountabilityError> {
        let completion_evidence: BTreeSet<_> = completion_evidence.into_iter().collect();
        if completion_evidence.is_empty() {
            return Err(AccountabilityError::RepairEvidenceRequired);
        }
        validate_fact_refs(&completion_evidence, evidence)?;
        let repair = self
            .repairs
            .iter_mut()
            .find(|repair| &repair.id == id)
            .ok_or_else(|| AccountabilityError::MissingRepair(id.clone()))?;
        if repair.status == RepairStatus::Completed {
            return Err(AccountabilityError::RepairAlreadyCompleted(id.clone()));
        }
        repair.status = RepairStatus::Completed;
        repair.completion_evidence = completion_evidence;
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
            let observation = self
                .observations
                .iter()
                .rev()
                .find(|observation| {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authority_envelope::RelationalAuthorityAssessment;
    use crate::evidence_ledger::{EvidenceObservation, EvidenceRelation};

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
            .add_fact(fact("repair"), "repair was completed", 0.9)
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
        RelationalAuthorityAssessment {
            requested: ActionAuthority::ActReversible,
            ceiling: ActionAuthority::Recommend,
            reasons: BTreeSet::new(),
            human_review_recommended: false,
        }
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
    fn accountability_case_requires_real_decision() {
        let evidence = evidence();
        let missing = DecisionId::new("missing").unwrap();
        assert!(matches!(
            AccountabilityCase::try_new(
                AccountabilityCaseId::new("case-a").unwrap(),
                missing,
                [person()],
                &authority(),
                true,
                &evidence,
            ),
            Err(AccountabilityError::MissingDecision(_))
        ));
    }

    #[test]
    fn missing_predicted_outcome_remains_visible() {
        let evidence = evidence();
        let mut case = case(&evidence);
        case.add_prediction(
            PredictedConsequence::new(
                person(),
                ConsequenceDomain::Wellbeing,
                ConsequenceDirection::Better,
                0.7,
            )
            .unwrap(),
        )
        .unwrap();
        let assessment = case.assess(&evidence).unwrap();
        assert!(assessment.triggers.iter().any(|trigger| matches!(
            trigger,
            AccountabilityTrigger::MissingOutcome { .. }
        )));
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
        assert!(assessment
            .suggested_repairs
            .contains(&RepairKind::ReassessPremises));
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

        let assessment = case.assess(&evidence).unwrap();
        assert!(assessment
            .triggers
            .contains(&AccountabilityTrigger::PremiseReviewRequired));
        assert!(assessment
            .suggested_repairs
            .contains(&RepairKind::RetractOrCorrectAdvice));
        assert!(assessment
            .suggested_repairs
            .contains(&RepairKind::ReassessPremises));
    }

    #[test]
    fn executing_above_recorded_ceiling_is_visible() {
        let evidence = evidence();
        let mut case = case(&evidence);
        case.record_execution(ActionAuthority::ActReversible).unwrap();
        let assessment = case.assess(&evidence).unwrap();
        assert!(assessment
            .triggers
            .contains(&AccountabilityTrigger::AuthorityCeilingExceeded));
        assert!(assessment
            .suggested_repairs
            .contains(&RepairKind::EscalateIndependentReview));
    }

    #[test]
    fn relationship_independence_loss_suggests_dependency_repair() {
        let evidence = evidence();
        let mut case = case(&evidence);
        case.record_observation(
            ObservedConsequence::new(
                person(),
                ConsequenceDomain::RelationshipIndependence,
                ConsequenceDirection::Worse,
                0.8,
                false,
                [fact("outcome")],
            )
            .unwrap(),
            &evidence,
        )
        .unwrap();
        let assessment = case.assess(&evidence).unwrap();
        assert!(assessment
            .suggested_repairs
            .contains(&RepairKind::ReduceDependency));
    }

    #[test]
    fn repair_cannot_be_marked_complete_without_evidence() {
        let evidence = evidence();
        let mut case = case(&evidence);
        let repair_id = RepairActionId::new("repair-a").unwrap();
        case.add_repair(RepairAction::new(
            repair_id.clone(),
            RepairKind::MitigateHarm,
            Some(person()),
        ))
        .unwrap();
        assert_eq!(
            case.complete_repair(&repair_id, [], &evidence),
            Err(AccountabilityError::RepairEvidenceRequired)
        );
        assert_eq!(case.repairs()[0].status(), RepairStatus::Open);
    }

    #[test]
    fn repair_completion_preserves_evidence_receipt() {
        let evidence = evidence();
        let mut case = case(&evidence);
        let repair_id = RepairActionId::new("repair-a").unwrap();
        case.add_repair(RepairAction::new(
            repair_id.clone(),
            RepairKind::InformAffectedParty,
            Some(person()),
        ))
        .unwrap();
        case.complete_repair(&repair_id, [fact("repair")], &evidence)
            .unwrap();
        assert_eq!(case.repairs()[0].status(), RepairStatus::Completed);
        assert!(case.repairs()[0]
            .completion_evidence()
            .contains(&fact("repair")));
    }

    #[test]
    fn contradictory_evidence_reopens_premise_review() {
        let mut evidence = evidence();
        let case = case(&evidence);
        evidence
            .add_fact_evidence(
                &fact("premise"),
                EvidenceObservation::new(
                    "counter-source",
                    EvidenceRelation::Contradicts,
                    0.9,
                )
                .unwrap(),
            )
            .unwrap();
        assert!(case
            .assess(&evidence)
            .unwrap()
            .triggers
            .contains(&AccountabilityTrigger::PremiseReviewRequired));
    }
}
