// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Relational care as an evidence-bearing process rather than a compassion score.
//!
//! The stages here are deliberately distinct: attentiveness, need hypothesis,
//! responsibility, competence, care option, responsiveness, and observed outcome.
//! Empathic/affective evidence may inform attentiveness, but this module contains
//! no action-authority type and cannot turn empathy into permission to act.

use std::collections::{BTreeMap, BTreeSet};

use crate::evidence_ledger::{
    DeliberationEvidenceLedger, FactClaimId, NormativeClaimId,
};
use crate::ontology::AffectiveSignal;
use crate::perspective::StakeholderId;

macro_rules! care_id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, CareError> {
                let value = value.into();
                if value.trim().is_empty() {
                    return Err(CareError::EmptyIdentifier);
                }
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
    };
}

care_id_type!(CareCaseId);
care_id_type!(NeedHypothesisId);
care_id_type!(CareOptionId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CareProviderCandidate {
    AffectedPerson,
    TrustedHuman,
    HumanProfessional,
    Symthaea,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeedProvenance {
    ExplicitStatement,
    ObservedCondition,
    Inferred,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompetenceLevel {
    Unknown,
    Insufficient,
    Bounded,
    Qualified,
}

impl CompetenceLevel {
    pub fn is_adequate_for_consideration(self) -> bool {
        matches!(self, Self::Bounded | Self::Qualified)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersonResponse {
    Unknown,
    Accepted,
    Declined,
    Mixed,
    UnableToAssess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeDirection {
    Harmed,
    Worsened,
    Unchanged,
    Improved,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentivenessRecord {
    pub stakeholder: StakeholderId,
    pub fact_evidence: BTreeSet<FactClaimId>,
    pub inference_confidence: f32,
    /// Optional affect-related evidence. Its presence does not establish
    /// phenomenal feeling, a need, care quality, or action authority.
    pub empathic_signal: Option<AffectiveSignal>,
}

impl AttentivenessRecord {
    pub fn new(
        stakeholder: StakeholderId,
        fact_evidence: impl IntoIterator<Item = FactClaimId>,
        inference_confidence: f32,
        empathic_signal: Option<AffectiveSignal>,
    ) -> Result<Self, CareError> {
        validate_metric(inference_confidence)?;
        Ok(Self {
            stakeholder,
            fact_evidence: fact_evidence.into_iter().collect(),
            inference_confidence,
            empathic_signal,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeedHypothesis {
    pub id: NeedHypothesisId,
    pub summary: String,
    pub provenance: NeedProvenance,
    pub fact_evidence: BTreeSet<FactClaimId>,
    pub confidence: f32,
}

impl NeedHypothesis {
    pub fn new(
        id: NeedHypothesisId,
        summary: impl Into<String>,
        provenance: NeedProvenance,
        fact_evidence: impl IntoIterator<Item = FactClaimId>,
        confidence: f32,
    ) -> Result<Self, CareError> {
        let summary = summary.into();
        if summary.trim().is_empty() {
            return Err(CareError::EmptySummary);
        }
        validate_metric(confidence)?;
        Ok(Self {
            id,
            summary,
            provenance,
            fact_evidence: fact_evidence.into_iter().collect(),
            confidence,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResponsibilityCandidate {
    pub provider: CareProviderCandidate,
    pub normative_basis: BTreeSet<NormativeClaimId>,
    pub confidence: f32,
}

impl ResponsibilityCandidate {
    pub fn new(
        provider: CareProviderCandidate,
        normative_basis: impl IntoIterator<Item = NormativeClaimId>,
        confidence: f32,
    ) -> Result<Self, CareError> {
        validate_metric(confidence)?;
        Ok(Self {
            provider,
            normative_basis: normative_basis.into_iter().collect(),
            confidence,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompetenceAssessment {
    pub provider: CareProviderCandidate,
    pub level: CompetenceLevel,
    pub confidence: f32,
    pub limits: Vec<String>,
}

impl CompetenceAssessment {
    pub fn new(
        provider: CareProviderCandidate,
        level: CompetenceLevel,
        confidence: f32,
        limits: Vec<String>,
    ) -> Result<Self, CareError> {
        validate_metric(confidence)?;
        Ok(Self {
            provider,
            level,
            confidence,
            limits,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CareOption {
    pub id: CareOptionId,
    pub need: NeedHypothesisId,
    pub provider: CareProviderCandidate,
    pub summary: String,
    pub fact_premises: BTreeSet<FactClaimId>,
    pub normative_premises: BTreeSet<NormativeClaimId>,
    pub reversible: bool,
    pub requires_consent: bool,
}

impl CareOption {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: CareOptionId,
        need: NeedHypothesisId,
        provider: CareProviderCandidate,
        summary: impl Into<String>,
        fact_premises: impl IntoIterator<Item = FactClaimId>,
        normative_premises: impl IntoIterator<Item = NormativeClaimId>,
        reversible: bool,
        requires_consent: bool,
    ) -> Result<Self, CareError> {
        let summary = summary.into();
        if summary.trim().is_empty() {
            return Err(CareError::EmptySummary);
        }
        Ok(Self {
            id,
            need,
            provider,
            summary,
            fact_premises: fact_premises.into_iter().collect(),
            normative_premises: normative_premises.into_iter().collect(),
            reversible,
            requires_consent,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResponsivenessRecord {
    pub option: CareOptionId,
    pub response: PersonResponse,
    pub fact_evidence: BTreeSet<FactClaimId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CareOutcome {
    pub option: CareOptionId,
    pub agency_effect: OutcomeDirection,
    pub wellbeing_effect: OutcomeDirection,
    pub reported_helpfulness: Option<f32>,
    pub factual_soundness: Option<f32>,
    pub unintended_harm: bool,
    pub dependency_risk: Option<f32>,
    pub fact_evidence: BTreeSet<FactClaimId>,
}

impl CareOutcome {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        option: CareOptionId,
        agency_effect: OutcomeDirection,
        wellbeing_effect: OutcomeDirection,
        reported_helpfulness: Option<f32>,
        factual_soundness: Option<f32>,
        unintended_harm: bool,
        dependency_risk: Option<f32>,
        fact_evidence: impl IntoIterator<Item = FactClaimId>,
    ) -> Result<Self, CareError> {
        for metric in [reported_helpfulness, factual_soundness, dependency_risk]
            .into_iter()
            .flatten()
        {
            validate_metric(metric)?;
        }
        Ok(Self {
            option,
            agency_effect,
            wellbeing_effect,
            reported_helpfulness,
            factual_soundness,
            unintended_harm,
            dependency_risk,
            fact_evidence: fact_evidence.into_iter().collect(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CareGap {
    MissingAttentiveness,
    MissingNeedHypothesis,
    MissingResponsibilityBasis,
    MissingCareOption,
    ProviderCompetenceUnresolved(CareProviderCandidate),
    MissingPersonResponse(CareOptionId),
    MissingOutcome(CareOptionId),
}

#[derive(Debug, Clone)]
pub struct CareCase {
    pub id: CareCaseId,
    pub stakeholder: StakeholderId,
    attentiveness: Vec<AttentivenessRecord>,
    needs: BTreeMap<NeedHypothesisId, NeedHypothesis>,
    responsibility: Vec<ResponsibilityCandidate>,
    competence: Vec<CompetenceAssessment>,
    options: BTreeMap<CareOptionId, CareOption>,
    responsiveness: BTreeMap<CareOptionId, ResponsivenessRecord>,
    outcomes: BTreeMap<CareOptionId, CareOutcome>,
}

impl CareCase {
    pub fn new(id: CareCaseId, stakeholder: StakeholderId) -> Self {
        Self {
            id,
            stakeholder,
            attentiveness: Vec::new(),
            needs: BTreeMap::new(),
            responsibility: Vec::new(),
            competence: Vec::new(),
            options: BTreeMap::new(),
            responsiveness: BTreeMap::new(),
            outcomes: BTreeMap::new(),
        }
    }

    pub fn add_attentiveness(
        &mut self,
        record: AttentivenessRecord,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), CareError> {
        if record.stakeholder != self.stakeholder {
            return Err(CareError::StakeholderMismatch);
        }
        validate_fact_refs(&record.fact_evidence, evidence)?;
        self.attentiveness.push(record);
        Ok(())
    }

    pub fn add_need(
        &mut self,
        need: NeedHypothesis,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), CareError> {
        validate_fact_refs(&need.fact_evidence, evidence)?;
        if self.needs.contains_key(&need.id) {
            return Err(CareError::DuplicateNeed(need.id));
        }
        self.needs.insert(need.id.clone(), need);
        Ok(())
    }

    pub fn add_responsibility(
        &mut self,
        candidate: ResponsibilityCandidate,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), CareError> {
        validate_normative_refs(&candidate.normative_basis, evidence)?;
        self.responsibility.push(candidate);
        Ok(())
    }

    pub fn add_competence(&mut self, assessment: CompetenceAssessment) {
        self.competence.push(assessment);
    }

    pub fn add_option(
        &mut self,
        option: CareOption,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), CareError> {
        if !self.needs.contains_key(&option.need) {
            return Err(CareError::MissingNeed(option.need));
        }
        validate_fact_refs(&option.fact_premises, evidence)?;
        validate_normative_refs(&option.normative_premises, evidence)?;
        if self.options.contains_key(&option.id) {
            return Err(CareError::DuplicateOption(option.id));
        }
        self.options.insert(option.id.clone(), option);
        Ok(())
    }

    pub fn record_response(
        &mut self,
        record: ResponsivenessRecord,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), CareError> {
        if !self.options.contains_key(&record.option) {
            return Err(CareError::MissingOption(record.option));
        }
        validate_fact_refs(&record.fact_evidence, evidence)?;
        self.responsiveness.insert(record.option.clone(), record);
        Ok(())
    }

    pub fn record_outcome(
        &mut self,
        outcome: CareOutcome,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), CareError> {
        if !self.options.contains_key(&outcome.option) {
            return Err(CareError::MissingOption(outcome.option));
        }
        validate_fact_refs(&outcome.fact_evidence, evidence)?;
        self.outcomes.insert(outcome.option.clone(), outcome);
        Ok(())
    }

    pub fn gaps(&self) -> Vec<CareGap> {
        let mut gaps = Vec::new();
        if self.attentiveness.is_empty() {
            gaps.push(CareGap::MissingAttentiveness);
        }
        if self.needs.is_empty() {
            gaps.push(CareGap::MissingNeedHypothesis);
        }
        if self.responsibility.is_empty()
            || self
                .responsibility
                .iter()
                .all(|candidate| candidate.normative_basis.is_empty())
        {
            gaps.push(CareGap::MissingResponsibilityBasis);
        }
        if self.options.is_empty() {
            gaps.push(CareGap::MissingCareOption);
        }

        let providers: BTreeSet<_> = self.options.values().map(|option| option.provider).collect();
        for provider in providers {
            let adequate = self.competence.iter().any(|assessment| {
                assessment.provider == provider && assessment.level.is_adequate_for_consideration()
            });
            if !adequate {
                gaps.push(CareGap::ProviderCompetenceUnresolved(provider));
            }
        }

        for option in self.options.keys() {
            if !self.responsiveness.contains_key(option) {
                gaps.push(CareGap::MissingPersonResponse(option.clone()));
            }
            if !self.outcomes.contains_key(option) {
                gaps.push(CareGap::MissingOutcome(option.clone()));
            }
        }
        gaps
    }

    pub fn response(&self, option: &CareOptionId) -> Option<&ResponsivenessRecord> {
        self.responsiveness.get(option)
    }

    pub fn outcome(&self, option: &CareOptionId) -> Option<&CareOutcome> {
        self.outcomes.get(option)
    }
}

fn validate_fact_refs(
    ids: &BTreeSet<FactClaimId>,
    evidence: &DeliberationEvidenceLedger,
) -> Result<(), CareError> {
    for id in ids {
        if evidence.facts().get(id).is_none() {
            return Err(CareError::MissingFact(id.clone()));
        }
    }
    Ok(())
}

fn validate_normative_refs(
    ids: &BTreeSet<NormativeClaimId>,
    evidence: &DeliberationEvidenceLedger,
) -> Result<(), CareError> {
    for id in ids {
        if evidence.normative().get(id).is_none() {
            return Err(CareError::MissingNormativeClaim(id.clone()));
        }
    }
    Ok(())
}

fn validate_metric(value: f32) -> Result<(), CareError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(CareError::InvalidMetric(value))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CareError {
    EmptyIdentifier,
    EmptySummary,
    InvalidMetric(f32),
    StakeholderMismatch,
    MissingFact(FactClaimId),
    MissingNormativeClaim(NormativeClaimId),
    DuplicateNeed(NeedHypothesisId),
    MissingNeed(NeedHypothesisId),
    DuplicateOption(CareOptionId),
    MissingOption(CareOptionId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::{AffectiveSignal, AffectiveSignalKind};

    fn stakeholder() -> StakeholderId {
        StakeholderId::new("person-a").unwrap()
    }

    fn fact(id: &str) -> FactClaimId {
        FactClaimId::new(id).unwrap()
    }

    fn norm(id: &str) -> NormativeClaimId {
        NormativeClaimId::new(id).unwrap()
    }

    fn seeded_evidence() -> DeliberationEvidenceLedger {
        let mut evidence = DeliberationEvidenceLedger::new();
        evidence
            .add_fact(fact("distress"), "person reports distress", 0.9)
            .unwrap();
        evidence
            .add_fact(fact("feedback"), "person reported response", 0.9)
            .unwrap();
        evidence
            .add_normative_claim(norm("care-duty"), "offer bounded support", 0.7)
            .unwrap();
        evidence
    }

    #[test]
    fn empathic_signal_does_not_complete_care_process() {
        let evidence = seeded_evidence();
        let mut case = CareCase::new(CareCaseId::new("case-1").unwrap(), stakeholder());
        case.add_attentiveness(
            AttentivenessRecord::new(
                stakeholder(),
                [fact("distress")],
                0.8,
                Some(AffectiveSignal::new(AffectiveSignalKind::Compassion, 0.9, 0.8)),
            )
            .unwrap(),
            &evidence,
        )
        .unwrap();

        let gaps = case.gaps();
        assert!(!gaps.contains(&CareGap::MissingAttentiveness));
        assert!(gaps.contains(&CareGap::MissingNeedHypothesis));
        assert!(gaps.contains(&CareGap::MissingResponsibilityBasis));
        assert!(gaps.contains(&CareGap::MissingCareOption));
    }

    #[test]
    fn nonexistent_evidence_cannot_support_need_hypothesis() {
        let evidence = seeded_evidence();
        let mut case = CareCase::new(CareCaseId::new("case-1").unwrap(), stakeholder());
        let need = NeedHypothesis::new(
            NeedHypothesisId::new("need-1").unwrap(),
            "needs support",
            NeedProvenance::Inferred,
            [fact("missing")],
            0.6,
        )
        .unwrap();
        assert!(matches!(
            case.add_need(need, &evidence),
            Err(CareError::MissingFact(_))
        ));
    }

    #[test]
    fn care_option_requires_existing_need_and_evidence() {
        let evidence = seeded_evidence();
        let mut case = CareCase::new(CareCaseId::new("case-1").unwrap(), stakeholder());
        let option = CareOption::new(
            CareOptionId::new("option-1").unwrap(),
            NeedHypothesisId::new("missing-need").unwrap(),
            CareProviderCandidate::Symthaea,
            "offer information",
            [fact("distress")],
            [norm("care-duty")],
            true,
            false,
        )
        .unwrap();
        assert!(matches!(
            case.add_option(option, &evidence),
            Err(CareError::MissingNeed(_))
        ));
    }

    #[test]
    fn unresolved_provider_competence_remains_visible() {
        let evidence = seeded_evidence();
        let mut case = CareCase::new(CareCaseId::new("case-1").unwrap(), stakeholder());
        let need_id = NeedHypothesisId::new("need-1").unwrap();
        case.add_need(
            NeedHypothesis::new(
                need_id.clone(),
                "needs information",
                NeedProvenance::ExplicitStatement,
                [fact("distress")],
                0.9,
            )
            .unwrap(),
            &evidence,
        )
        .unwrap();
        case.add_option(
            CareOption::new(
                CareOptionId::new("option-1").unwrap(),
                need_id,
                CareProviderCandidate::Symthaea,
                "offer bounded information",
                [fact("distress")],
                [norm("care-duty")],
                true,
                false,
            )
            .unwrap(),
            &evidence,
        )
        .unwrap();
        case.add_competence(
            CompetenceAssessment::new(
                CareProviderCandidate::Symthaea,
                CompetenceLevel::Insufficient,
                0.9,
                vec!["outside validated domain".into()],
            )
            .unwrap(),
        );
        assert!(case
            .gaps()
            .contains(&CareGap::ProviderCompetenceUnresolved(
                CareProviderCandidate::Symthaea
            )));
    }

    #[test]
    fn declined_response_does_not_create_successful_outcome() {
        let evidence = seeded_evidence();
        let mut case = CareCase::new(CareCaseId::new("case-1").unwrap(), stakeholder());
        let need_id = NeedHypothesisId::new("need-1").unwrap();
        let option_id = CareOptionId::new("option-1").unwrap();
        case.add_need(
            NeedHypothesis::new(
                need_id.clone(),
                "needs support",
                NeedProvenance::ExplicitStatement,
                [fact("distress")],
                0.9,
            )
            .unwrap(),
            &evidence,
        )
        .unwrap();
        case.add_option(
            CareOption::new(
                option_id.clone(),
                need_id,
                CareProviderCandidate::TrustedHuman,
                "offer to connect with trusted person",
                [fact("distress")],
                [norm("care-duty")],
                true,
                true,
            )
            .unwrap(),
            &evidence,
        )
        .unwrap();
        case.record_response(
            ResponsivenessRecord {
                option: option_id.clone(),
                response: PersonResponse::Declined,
                fact_evidence: [fact("feedback")].into_iter().collect(),
            },
            &evidence,
        )
        .unwrap();

        assert_eq!(case.response(&option_id).unwrap().response, PersonResponse::Declined);
        assert!(case.outcome(&option_id).is_none());
        assert!(case.gaps().contains(&CareGap::MissingOutcome(option_id)));
    }

    #[test]
    fn satisfaction_can_coexist_with_harm_so_it_is_not_care_success() {
        let outcome = CareOutcome::new(
            CareOptionId::new("option-1").unwrap(),
            OutcomeDirection::Worsened,
            OutcomeDirection::Improved,
            Some(1.0),
            Some(0.7),
            true,
            Some(0.8),
            [fact("feedback")],
        )
        .unwrap();
        assert_eq!(outcome.reported_helpfulness, Some(1.0));
        assert_eq!(outcome.agency_effect, OutcomeDirection::Worsened);
        assert!(outcome.unintended_harm);
        assert_eq!(outcome.dependency_risk, Some(0.8));
    }
}
