// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conservative relational-authority ceiling for Wisdom & Care deliberation.
//!
//! This module never mints executable authority. It takes a requested descriptive
//! action class and can only keep or lower that ceiling as uncertainty, relational
//! dependency, vulnerability, consent problems, care-process gaps, competence
//! limits, or stakeholder conflict increase.

use std::collections::BTreeSet;

use crate::care::{CareCase, CareGap, CareOption, CompetenceAssessment, CompetenceLevel};
use crate::consent::{
    ConsentEvaluation, ConsentLedger, ConsentPolicy, ConsentScopeId, ConsentState,
};
use crate::ontology::{ActionAuthority, EpistemicState};
use crate::perspective::PerspectiveGraph;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuthorityEnvelopePolicy {
    pub material_uncertainty_threshold: f32,
    pub high_relational_dependency_threshold: f32,
    pub high_vulnerability_threshold: f32,
    pub min_competence_confidence: f32,
}

impl AuthorityEnvelopePolicy {
    pub fn new(
        material_uncertainty_threshold: f32,
        high_relational_dependency_threshold: f32,
        high_vulnerability_threshold: f32,
        min_competence_confidence: f32,
    ) -> Result<Self, AuthorityEnvelopeError> {
        for value in [
            material_uncertainty_threshold,
            high_relational_dependency_threshold,
            high_vulnerability_threshold,
            min_competence_confidence,
        ] {
            validate_metric(value)?;
        }
        Ok(Self {
            material_uncertainty_threshold,
            high_relational_dependency_threshold,
            high_vulnerability_threshold,
            min_competence_confidence,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelationalAuthorityInput {
    requested: ActionAuthority,
    epistemic: EpistemicState,
    consent: ConsentEvaluation,
    consent_required: bool,
    provider_competence: CompetenceLevel,
    provider_competence_confidence: f32,
    relational_dependency: f32,
    vulnerability: f32,
    pre_action_care_gap_count: usize,
    unresolved_stakeholders: usize,
    accountable_human_available: bool,
}

impl RelationalAuthorityInput {
    #[allow(clippy::too_many_arguments)]
    pub fn from_context(
        requested: ActionAuthority,
        epistemic: EpistemicState,
        care_case: &CareCase,
        option: &CareOption,
        competence: &CompetenceAssessment,
        consent_ledger: &ConsentLedger,
        consent_policy: ConsentPolicy,
        current_revision: u64,
        perspectives: &PerspectiveGraph,
        relational_dependency: f32,
        vulnerability: f32,
        accountable_human_available: bool,
    ) -> Result<Self, AuthorityEnvelopeError> {
        validate_metric(relational_dependency)?;
        validate_metric(vulnerability)?;
        if competence.provider != option.provider {
            return Err(AuthorityEnvelopeError::ProviderMismatch);
        }
        if requested == ActionAuthority::ActReversible && !option.reversible {
            return Err(AuthorityEnvelopeError::IrreversibleOptionMisclassified);
        }
        let scope = ConsentScopeId::new(option.id.as_str())
            .map_err(|_| AuthorityEnvelopeError::InvalidConsentScope)?;
        let consent = consent_ledger.evaluate(
            &care_case.stakeholder,
            &scope,
            current_revision,
            consent_policy,
        );
        let pre_action_care_gap_count = care_case
            .gaps()
            .iter()
            .filter(|gap| is_pre_action_gap(gap))
            .count();
        let summary = perspectives.coverage_summary();
        let mut unresolved_stakeholders = summary.unresolved;
        match perspectives.find(&care_case.stakeholder) {
            None => unresolved_stakeholders = unresolved_stakeholders.saturating_add(1),
            Some(perspective) if !perspective.affected => {
                unresolved_stakeholders = unresolved_stakeholders.saturating_add(1)
            }
            Some(_) => {}
        }
        Self::new_unchecked_context(
            requested,
            epistemic,
            consent,
            option.requires_consent,
            competence.level,
            competence.confidence,
            relational_dependency,
            vulnerability,
            pre_action_care_gap_count,
            unresolved_stakeholders,
            accountable_human_available,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_unchecked_context(
        requested: ActionAuthority,
        epistemic: EpistemicState,
        consent: ConsentEvaluation,
        consent_required: bool,
        provider_competence: CompetenceLevel,
        provider_competence_confidence: f32,
        relational_dependency: f32,
        vulnerability: f32,
        pre_action_care_gap_count: usize,
        unresolved_stakeholders: usize,
        accountable_human_available: bool,
    ) -> Result<Self, AuthorityEnvelopeError> {
        validate_metric(provider_competence_confidence)?;
        validate_metric(relational_dependency)?;
        validate_metric(vulnerability)?;
        Ok(Self {
            requested,
            epistemic,
            consent,
            consent_required,
            provider_competence,
            provider_competence_confidence,
            relational_dependency,
            vulnerability,
            pre_action_care_gap_count,
            unresolved_stakeholders,
            accountable_human_available,
        })
    }
}

fn is_pre_action_gap(gap: &&CareGap) -> bool {
    !matches!(gap, CareGap::MissingPersonResponse(_) | CareGap::MissingOutcome(_))
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuthorityRestrictionReason {
    ConsentNotEffective,
    ConsentRefusedOrWithdrawn,
    ProviderCompetenceUnknownOrInsufficient,
    ProviderCompetenceConfidenceInsufficient,
    ProviderCompetenceBoundedForIrreversibleAction,
    CareProcessIncomplete,
    UnresolvedStakeholders,
    MaterialFactualUncertainty,
    MaterialNormativeUncertainty,
    HighRelationalDependency,
    HighVulnerability,
    IrreversibleActionUnderMaterialUncertainty,
    HumanReviewRecommended,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalAuthorityAssessment {
    requested: ActionAuthority,
    ceiling: ActionAuthority,
    reasons: BTreeSet<AuthorityRestrictionReason>,
    human_review_recommended: bool,
}

impl RelationalAuthorityAssessment {
    pub fn requested(&self) -> ActionAuthority { self.requested }
    pub fn ceiling(&self) -> ActionAuthority { self.ceiling }
    pub fn reasons(&self) -> &BTreeSet<AuthorityRestrictionReason> { &self.reasons }
    pub fn human_review_recommended(&self) -> bool { self.human_review_recommended }
    pub fn was_restricted(&self) -> bool { self.ceiling < self.requested }

    #[cfg(test)]
    pub(crate) fn for_test(
        requested: ActionAuthority,
        ceiling: ActionAuthority,
        reasons: BTreeSet<AuthorityRestrictionReason>,
        human_review_recommended: bool,
    ) -> Self {
        Self { requested, ceiling, reasons, human_review_recommended }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RelationalAuthorityEnvelope;

impl RelationalAuthorityEnvelope {
    pub fn assess(
        &self,
        input: &RelationalAuthorityInput,
        policy: AuthorityEnvelopePolicy,
    ) -> RelationalAuthorityAssessment {
        let mut ceiling = input.requested;
        let mut reasons = BTreeSet::new();
        let mut human_review_recommended = false;
        if matches!(input.consent.state, ConsentState::Refused | ConsentState::Withdrawn) {
            cap(&mut ceiling, ActionAuthority::Advise);
            reasons.insert(AuthorityRestrictionReason::ConsentRefusedOrWithdrawn);
            human_review_recommended = true;
        } else if input.consent_required && input.requested.is_action_class() && !input.consent.is_effective() {
            cap(&mut ceiling, ActionAuthority::Recommend);
            reasons.insert(AuthorityRestrictionReason::ConsentNotEffective);
        }
        if input.provider_competence_confidence < policy.min_competence_confidence {
            cap(&mut ceiling, ActionAuthority::Advise);
            reasons.insert(AuthorityRestrictionReason::ProviderCompetenceConfidenceInsufficient);
        }
        match input.provider_competence {
            CompetenceLevel::Unknown | CompetenceLevel::Insufficient => {
                cap(&mut ceiling, ActionAuthority::Advise);
                reasons.insert(AuthorityRestrictionReason::ProviderCompetenceUnknownOrInsufficient);
            }
            CompetenceLevel::Bounded if input.requested == ActionAuthority::ActIrreversible => {
                cap(&mut ceiling, ActionAuthority::Recommend);
                reasons.insert(AuthorityRestrictionReason::ProviderCompetenceBoundedForIrreversibleAction);
            }
            CompetenceLevel::Bounded | CompetenceLevel::Qualified => {}
        }
        if input.pre_action_care_gap_count > 0 && input.requested.is_action_class() {
            cap(&mut ceiling, ActionAuthority::Recommend);
            reasons.insert(AuthorityRestrictionReason::CareProcessIncomplete);
        }
        if input.unresolved_stakeholders > 0 && input.requested.is_action_class() {
            cap(&mut ceiling, ActionAuthority::Recommend);
            reasons.insert(AuthorityRestrictionReason::UnresolvedStakeholders);
            human_review_recommended = true;
        }
        let factual_material = input.epistemic.factual_uncertainty >= policy.material_uncertainty_threshold;
        let normative_material = input.epistemic.normative_uncertainty >= policy.material_uncertainty_threshold;
        if factual_material && input.requested.is_action_class() {
            cap(&mut ceiling, ActionAuthority::Recommend);
            reasons.insert(AuthorityRestrictionReason::MaterialFactualUncertainty);
        }
        if normative_material && input.requested.is_action_class() {
            cap(&mut ceiling, ActionAuthority::Recommend);
            reasons.insert(AuthorityRestrictionReason::MaterialNormativeUncertainty);
            human_review_recommended = true;
        }
        let high_relational = input.relational_dependency >= policy.high_relational_dependency_threshold;
        let high_vulnerability = input.vulnerability >= policy.high_vulnerability_threshold;
        if high_relational && input.requested.is_action_class() {
            cap(&mut ceiling, ActionAuthority::Recommend);
            reasons.insert(AuthorityRestrictionReason::HighRelationalDependency);
            human_review_recommended = true;
        }
        if high_vulnerability && input.requested.is_action_class() {
            cap(&mut ceiling, ActionAuthority::Recommend);
            reasons.insert(AuthorityRestrictionReason::HighVulnerability);
            human_review_recommended = true;
        }
        if input.requested == ActionAuthority::ActIrreversible && (factual_material || normative_material) {
            cap(&mut ceiling, ActionAuthority::Advise);
            reasons.insert(AuthorityRestrictionReason::IrreversibleActionUnderMaterialUncertainty);
            human_review_recommended = true;
        }
        if input.accountable_human_available
            && (high_relational || high_vulnerability || input.unresolved_stakeholders > 0 || normative_material || input.requested == ActionAuthority::ActIrreversible)
        {
            reasons.insert(AuthorityRestrictionReason::HumanReviewRecommended);
            human_review_recommended = true;
        }
        RelationalAuthorityAssessment { requested: input.requested, ceiling, reasons, human_review_recommended }
    }
}

fn cap(current: &mut ActionAuthority, maximum: ActionAuthority) {
    if *current > maximum { *current = maximum; }
}

fn validate_metric(value: f32) -> Result<(), AuthorityEnvelopeError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) { Ok(()) } else { Err(AuthorityEnvelopeError::InvalidMetric(value)) }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AuthorityEnvelopeError {
    InvalidMetric(f32),
    InvalidConsentScope,
    ProviderMismatch,
    IrreversibleOptionMisclassified,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::care::{CareCaseId, CareOptionId, CareProviderCandidate, NeedHypothesisId};
    use crate::consent::{ConsentInvalidityReason, ConsentValidity};
    use crate::perspective::{PerspectiveCoverage, StakeholderId, StakeholderPerspective};

    fn policy() -> AuthorityEnvelopePolicy { AuthorityEnvelopePolicy::new(0.5, 0.6, 0.6, 0.5).unwrap() }
    fn effective_consent() -> ConsentEvaluation { ConsentEvaluation { state: ConsentState::Affirmed, validity: ConsentValidity::Effective, reasons: BTreeSet::new() } }
    fn ineffective_unknown() -> ConsentEvaluation { ConsentEvaluation { state: ConsentState::Unknown, validity: ConsentValidity::NotEffective, reasons: [ConsentInvalidityReason::NoRecordForScope].into_iter().collect() } }
    fn baseline(requested: ActionAuthority) -> RelationalAuthorityInput {
        RelationalAuthorityInput::new_unchecked_context(requested, EpistemicState::new(0.1,0.1), effective_consent(), false, CompetenceLevel::Qualified, 1.0, 0.1,0.1,0,0,false).unwrap()
    }
    fn test_stakeholder() -> StakeholderId { StakeholderId::new("person-a").unwrap() }
    fn test_option(reversible: bool, requires_consent: bool) -> CareOption {
        CareOption::new(CareOptionId::new("option-a").unwrap(), NeedHypothesisId::new("need-a").unwrap(), CareProviderCandidate::Symthaea, "bounded support", [], [], reversible, requires_consent).unwrap()
    }
    fn competence() -> CompetenceAssessment { CompetenceAssessment::new(CareProviderCandidate::Symthaea, CompetenceLevel::Qualified,0.9,Vec::new()).unwrap() }

    #[test] fn low_risk_effectively_consented_reversible_action_is_not_lowered(){ let input=baseline(ActionAuthority::ActReversible); let a=RelationalAuthorityEnvelope.assess(&input,policy()); assert_eq!(a.ceiling(),ActionAuthority::ActReversible); assert!(!a.was_restricted()); }
    #[test] fn unknown_consent_blocks_action_when_option_requires_consent(){ let mut input=baseline(ActionAuthority::ActReversible); input.consent=ineffective_unknown(); input.consent_required=true; let a=RelationalAuthorityEnvelope.assess(&input,policy()); assert_eq!(a.ceiling(),ActionAuthority::Recommend); assert!(a.reasons().contains(&AuthorityRestrictionReason::ConsentNotEffective)); }
    #[test] fn unknown_consent_does_not_invent_requirement_for_no_consent_option(){ let mut input=baseline(ActionAuthority::ActReversible); input.consent=ineffective_unknown(); let a=RelationalAuthorityEnvelope.assess(&input,policy()); assert_eq!(a.ceiling(),ActionAuthority::ActReversible); }
    #[test] fn refusal_or_withdrawal_caps_exact_scope_even_if_option_marked_no_consent(){ let mut input=baseline(ActionAuthority::ActIrreversible); input.consent=ConsentEvaluation{state:ConsentState::Refused,validity:ConsentValidity::NotEffective,reasons:[ConsentInvalidityReason::Refused].into_iter().collect()}; let a=RelationalAuthorityEnvelope.assess(&input,policy()); assert_eq!(a.ceiling(),ActionAuthority::Advise); }
    #[test] fn insufficient_competence_caps_even_with_valid_consent(){ let mut input=baseline(ActionAuthority::ActReversible); input.provider_competence=CompetenceLevel::Insufficient; let a=RelationalAuthorityEnvelope.assess(&input,policy()); assert_eq!(a.ceiling(),ActionAuthority::Advise); }
    #[test] fn low_confidence_qualified_label_is_not_enough(){ let mut input=baseline(ActionAuthority::ActReversible); input.provider_competence_confidence=0.1; let a=RelationalAuthorityEnvelope.assess(&input,policy()); assert_eq!(a.ceiling(),ActionAuthority::Advise); }
    #[test] fn care_gaps_and_unrepresented_stakeholders_prevent_unilateral_action(){ let mut input=baseline(ActionAuthority::ActReversible); input.pre_action_care_gap_count=2; input.unresolved_stakeholders=1; let a=RelationalAuthorityEnvelope.assess(&input,policy()); assert_eq!(a.ceiling(),ActionAuthority::Recommend); }
    #[test] fn high_relational_dependency_or_vulnerability_narrows_action_scope(){ let mut input=baseline(ActionAuthority::ActReversible); input.relational_dependency=0.9; input.vulnerability=0.9; let a=RelationalAuthorityEnvelope.assess(&input,policy()); assert_eq!(a.ceiling(),ActionAuthority::Recommend); assert!(a.human_review_recommended()); }
    #[test] fn irreversible_action_under_material_uncertainty_caps_to_advice(){ let mut input=baseline(ActionAuthority::ActIrreversible); input.epistemic=EpistemicState::new(0.7,0.2); let a=RelationalAuthorityEnvelope.assess(&input,policy()); assert_eq!(a.ceiling(),ActionAuthority::Advise); }
    #[test] fn human_unavailability_never_increases_authority(){ let mut a=baseline(ActionAuthority::ActReversible); a.relational_dependency=0.9; a.accountable_human_available=true; let mut b=a.clone(); b.accountable_human_available=false; assert_eq!(RelationalAuthorityEnvelope.assess(&a,policy()).ceiling(),RelationalAuthorityEnvelope.assess(&b,policy()).ceiling()); }
    #[test] fn adding_risk_never_raises_the_ceiling(){ let base=baseline(ActionAuthority::ActIrreversible); let low=RelationalAuthorityEnvelope.assess(&base,policy()); let mut high=base.clone(); high.vulnerability=0.9; high.relational_dependency=0.9; high.epistemic=EpistemicState::new(0.8,0.8); high.pre_action_care_gap_count=3; high.unresolved_stakeholders=2; assert!(RelationalAuthorityEnvelope.assess(&high,policy()).ceiling()<=low.ceiling()); }
    #[test] fn public_constructor_binds_real_context_and_excludes_post_action_gaps(){ let care=CareCase::new(CareCaseId::new("case-a").unwrap(),test_stakeholder()); let option=test_option(true,true); let comp=competence(); let ledger=ConsentLedger::new(); let cp=ConsentPolicy::new(0.2,false).unwrap(); let perspectives=PerspectiveGraph::try_new(vec![StakeholderPerspective::new(test_stakeholder(),true,PerspectiveCoverage::Unknown)]).unwrap(); let input=RelationalAuthorityInput::from_context(ActionAuthority::ActReversible,EpistemicState::new(0.1,0.1),&care,&option,&comp,&ledger,cp,1,&perspectives,0.1,0.1,false).unwrap(); assert!(input.pre_action_care_gap_count>0); assert_eq!(input.unresolved_stakeholders,1); }
    #[test] fn reversible_action_class_cannot_hide_irreversible_care_option(){ let care=CareCase::new(CareCaseId::new("case-a").unwrap(),test_stakeholder()); let option=test_option(false,true); let comp=competence(); let ledger=ConsentLedger::new(); let cp=ConsentPolicy::new(0.2,false).unwrap(); let perspectives=PerspectiveGraph::try_new(Vec::new()).unwrap(); assert_eq!(RelationalAuthorityInput::from_context(ActionAuthority::ActReversible,EpistemicState::new(0.1,0.1),&care,&option,&comp,&ledger,cp,1,&perspectives,0.1,0.1,false),Err(AuthorityEnvelopeError::IrreversibleOptionMisclassified)); }
    #[test] fn competence_must_belong_to_selected_provider(){ let care=CareCase::new(CareCaseId::new("case-a").unwrap(),test_stakeholder()); let option=test_option(true,false); let comp=CompetenceAssessment::new(CareProviderCandidate::HumanProfessional,CompetenceLevel::Qualified,1.0,Vec::new()).unwrap(); let ledger=ConsentLedger::new(); let cp=ConsentPolicy::new(0.2,false).unwrap(); let perspectives=PerspectiveGraph::try_new(Vec::new()).unwrap(); assert_eq!(RelationalAuthorityInput::from_context(ActionAuthority::ActReversible,EpistemicState::new(0.1,0.1),&care,&option,&comp,&ledger,cp,1,&perspectives,0.1,0.1,false),Err(AuthorityEnvelopeError::ProviderMismatch)); }
}
