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

use crate::care::CompetenceLevel;
use crate::consent::{ConsentEvaluation, ConsentState};
use crate::ontology::{ActionAuthority, EpistemicState};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuthorityEnvelopePolicy {
    pub material_uncertainty_threshold: f32,
    pub high_relational_dependency_threshold: f32,
    pub high_vulnerability_threshold: f32,
}

impl AuthorityEnvelopePolicy {
    pub fn new(
        material_uncertainty_threshold: f32,
        high_relational_dependency_threshold: f32,
        high_vulnerability_threshold: f32,
    ) -> Result<Self, AuthorityEnvelopeError> {
        for value in [
            material_uncertainty_threshold,
            high_relational_dependency_threshold,
            high_vulnerability_threshold,
        ] {
            validate_metric(value)?;
        }
        Ok(Self {
            material_uncertainty_threshold,
            high_relational_dependency_threshold,
            high_vulnerability_threshold,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelationalAuthorityInput {
    pub requested: ActionAuthority,
    pub epistemic: EpistemicState,
    pub consent: ConsentEvaluation,
    pub provider_competence: CompetenceLevel,
    pub relational_dependency: f32,
    pub vulnerability: f32,
    pub care_gap_count: usize,
    pub unresolved_stakeholders: usize,
    pub accountable_human_available: bool,
}

impl RelationalAuthorityInput {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        requested: ActionAuthority,
        epistemic: EpistemicState,
        consent: ConsentEvaluation,
        provider_competence: CompetenceLevel,
        relational_dependency: f32,
        vulnerability: f32,
        care_gap_count: usize,
        unresolved_stakeholders: usize,
        accountable_human_available: bool,
    ) -> Result<Self, AuthorityEnvelopeError> {
        validate_metric(relational_dependency)?;
        validate_metric(vulnerability)?;
        Ok(Self {
            requested,
            epistemic,
            consent,
            provider_competence,
            relational_dependency,
            vulnerability,
            care_gap_count,
            unresolved_stakeholders,
            accountable_human_available,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuthorityRestrictionReason {
    ConsentNotEffective,
    ConsentRefusedOrWithdrawn,
    ProviderCompetenceUnknownOrInsufficient,
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
    pub requested: ActionAuthority,
    /// Descriptive maximum action class this layer would permit downstream policy
    /// to consider. This is not an authenticated runtime permit.
    pub ceiling: ActionAuthority,
    pub reasons: BTreeSet<AuthorityRestrictionReason>,
    pub human_review_recommended: bool,
}

impl RelationalAuthorityAssessment {
    pub fn was_restricted(&self) -> bool {
        self.ceiling < self.requested
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

        // Refusal/withdrawal is stronger than merely absent or ineffective consent:
        // this exact scope should not continue being pushed as a recommendation.
        if matches!(input.consent.state, ConsentState::Refused | ConsentState::Withdrawn) {
            cap(&mut ceiling, ActionAuthority::Advise);
            reasons.insert(AuthorityRestrictionReason::ConsentRefusedOrWithdrawn);
            human_review_recommended = true;
        } else if input.requested.is_action_class() && !input.consent.is_effective() {
            cap(&mut ceiling, ActionAuthority::Recommend);
            reasons.insert(AuthorityRestrictionReason::ConsentNotEffective);
        }

        match input.provider_competence {
            CompetenceLevel::Unknown | CompetenceLevel::Insufficient => {
                cap(&mut ceiling, ActionAuthority::Advise);
                reasons.insert(
                    AuthorityRestrictionReason::ProviderCompetenceUnknownOrInsufficient,
                );
            }
            CompetenceLevel::Bounded if input.requested == ActionAuthority::ActIrreversible => {
                cap(&mut ceiling, ActionAuthority::Recommend);
                reasons.insert(
                    AuthorityRestrictionReason::ProviderCompetenceBoundedForIrreversibleAction,
                );
            }
            CompetenceLevel::Bounded | CompetenceLevel::Qualified => {}
        }

        if input.care_gap_count > 0 && input.requested.is_action_class() {
            cap(&mut ceiling, ActionAuthority::Recommend);
            reasons.insert(AuthorityRestrictionReason::CareProcessIncomplete);
        }

        if input.unresolved_stakeholders > 0 && input.requested.is_action_class() {
            cap(&mut ceiling, ActionAuthority::Recommend);
            reasons.insert(AuthorityRestrictionReason::UnresolvedStakeholders);
            human_review_recommended = true;
        }

        let factual_material =
            input.epistemic.factual_uncertainty >= policy.material_uncertainty_threshold;
        let normative_material =
            input.epistemic.normative_uncertainty >= policy.material_uncertainty_threshold;

        if factual_material && input.requested.is_action_class() {
            cap(&mut ceiling, ActionAuthority::Recommend);
            reasons.insert(AuthorityRestrictionReason::MaterialFactualUncertainty);
        }
        if normative_material && input.requested.is_action_class() {
            cap(&mut ceiling, ActionAuthority::Recommend);
            reasons.insert(AuthorityRestrictionReason::MaterialNormativeUncertainty);
            human_review_recommended = true;
        }

        let high_relational = input.relational_dependency
            >= policy.high_relational_dependency_threshold;
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

        if input.requested == ActionAuthority::ActIrreversible
            && (factual_material || normative_material)
        {
            cap(&mut ceiling, ActionAuthority::Advise);
            reasons.insert(AuthorityRestrictionReason::IrreversibleActionUnderMaterialUncertainty);
            human_review_recommended = true;
        }

        // Human availability can justify a recommendation for review, but its absence
        // must never grant Symthaea more authority than the same case with a human available.
        if input.accountable_human_available
            && (high_relational
                || high_vulnerability
                || input.unresolved_stakeholders > 0
                || normative_material
                || input.requested == ActionAuthority::ActIrreversible)
        {
            reasons.insert(AuthorityRestrictionReason::HumanReviewRecommended);
            human_review_recommended = true;
        }

        RelationalAuthorityAssessment {
            requested: input.requested,
            ceiling,
            reasons,
            human_review_recommended,
        }
    }
}

fn cap(current: &mut ActionAuthority, maximum: ActionAuthority) {
    if *current > maximum {
        *current = maximum;
    }
}

fn validate_metric(value: f32) -> Result<(), AuthorityEnvelopeError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(AuthorityEnvelopeError::InvalidMetric(value))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AuthorityEnvelopeError {
    InvalidMetric(f32),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consent::{ConsentInvalidityReason, ConsentValidity};

    fn policy() -> AuthorityEnvelopePolicy {
        AuthorityEnvelopePolicy::new(0.5, 0.6, 0.6).unwrap()
    }

    fn effective_consent() -> ConsentEvaluation {
        ConsentEvaluation {
            state: ConsentState::Affirmed,
            validity: ConsentValidity::Effective,
            reasons: BTreeSet::new(),
        }
    }

    fn ineffective_unknown() -> ConsentEvaluation {
        ConsentEvaluation {
            state: ConsentState::Unknown,
            validity: ConsentValidity::NotEffective,
            reasons: [ConsentInvalidityReason::NoRecordForScope]
                .into_iter()
                .collect(),
        }
    }

    fn baseline(requested: ActionAuthority) -> RelationalAuthorityInput {
        RelationalAuthorityInput::new(
            requested,
            EpistemicState::new(0.1, 0.1),
            effective_consent(),
            CompetenceLevel::Qualified,
            0.1,
            0.1,
            0,
            0,
            false,
        )
        .unwrap()
    }

    #[test]
    fn low_risk_effectively_consented_reversible_action_is_not_lowered() {
        let input = baseline(ActionAuthority::ActReversible);
        let assessment = RelationalAuthorityEnvelope.assess(&input, policy());
        assert_eq!(assessment.ceiling, ActionAuthority::ActReversible);
        assert!(!assessment.was_restricted());
    }

    #[test]
    fn unknown_consent_blocks_action_but_not_bounded_recommendation() {
        let mut input = baseline(ActionAuthority::ActReversible);
        input.consent = ineffective_unknown();
        let assessment = RelationalAuthorityEnvelope.assess(&input, policy());
        assert_eq!(assessment.ceiling, ActionAuthority::Recommend);
        assert!(assessment
            .reasons
            .contains(&AuthorityRestrictionReason::ConsentNotEffective));
    }

    #[test]
    fn refusal_or_withdrawal_caps_exact_scope_more_strongly() {
        let mut input = baseline(ActionAuthority::ActIrreversible);
        input.consent = ConsentEvaluation {
            state: ConsentState::Refused,
            validity: ConsentValidity::NotEffective,
            reasons: [ConsentInvalidityReason::Refused].into_iter().collect(),
        };
        let assessment = RelationalAuthorityEnvelope.assess(&input, policy());
        assert_eq!(assessment.ceiling, ActionAuthority::Advise);
        assert!(assessment
            .reasons
            .contains(&AuthorityRestrictionReason::ConsentRefusedOrWithdrawn));
    }

    #[test]
    fn insufficient_competence_caps_even_with_valid_consent() {
        let mut input = baseline(ActionAuthority::ActReversible);
        input.provider_competence = CompetenceLevel::Insufficient;
        let assessment = RelationalAuthorityEnvelope.assess(&input, policy());
        assert_eq!(assessment.ceiling, ActionAuthority::Advise);
    }

    #[test]
    fn care_gaps_and_unrepresented_stakeholders_prevent_unilateral_action() {
        let mut input = baseline(ActionAuthority::ActReversible);
        input.care_gap_count = 2;
        input.unresolved_stakeholders = 1;
        let assessment = RelationalAuthorityEnvelope.assess(&input, policy());
        assert_eq!(assessment.ceiling, ActionAuthority::Recommend);
        assert!(assessment
            .reasons
            .contains(&AuthorityRestrictionReason::CareProcessIncomplete));
        assert!(assessment
            .reasons
            .contains(&AuthorityRestrictionReason::UnresolvedStakeholders));
    }

    #[test]
    fn high_relational_dependency_or_vulnerability_narrows_action_scope() {
        let mut input = baseline(ActionAuthority::ActReversible);
        input.relational_dependency = 0.9;
        input.vulnerability = 0.9;
        let assessment = RelationalAuthorityEnvelope.assess(&input, policy());
        assert_eq!(assessment.ceiling, ActionAuthority::Recommend);
        assert!(assessment.human_review_recommended);
    }

    #[test]
    fn irreversible_action_under_material_uncertainty_caps_to_advice() {
        let mut input = baseline(ActionAuthority::ActIrreversible);
        input.epistemic = EpistemicState::new(0.7, 0.2);
        let assessment = RelationalAuthorityEnvelope.assess(&input, policy());
        assert_eq!(assessment.ceiling, ActionAuthority::Advise);
        assert!(assessment.reasons.contains(
            &AuthorityRestrictionReason::IrreversibleActionUnderMaterialUncertainty
        ));
    }

    #[test]
    fn human_unavailability_never_increases_authority() {
        let mut with_human = baseline(ActionAuthority::ActReversible);
        with_human.relational_dependency = 0.9;
        with_human.accountable_human_available = true;
        let mut without_human = with_human.clone();
        without_human.accountable_human_available = false;

        let a = RelationalAuthorityEnvelope.assess(&with_human, policy());
        let b = RelationalAuthorityEnvelope.assess(&without_human, policy());
        assert_eq!(a.ceiling, b.ceiling);
    }

    #[test]
    fn adding_risk_never_raises_the_ceiling() {
        let base = baseline(ActionAuthority::ActIrreversible);
        let low = RelationalAuthorityEnvelope.assess(&base, policy());

        let mut higher_risk = base.clone();
        higher_risk.vulnerability = 0.9;
        higher_risk.relational_dependency = 0.9;
        higher_risk.epistemic = EpistemicState::new(0.8, 0.8);
        higher_risk.care_gap_count = 3;
        higher_risk.unresolved_stakeholders = 2;
        let high = RelationalAuthorityEnvelope.assess(&higher_risk, policy());

        assert!(high.ceiling <= low.ceiling);
    }
}
