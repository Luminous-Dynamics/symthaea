// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Practical wisdom as inspectable deliberation rather than a scalar score.
//!
//! WCARE-01 models a conservative first layer of situated wisdom. It does not
//! decide what is morally right. It identifies when the current decision
//! context calls for more evidence, more perspectives, reversibility, deferral,
//! or abstention.

use crate::ontology::{ActionAuthority, EpistemicState};

/// Features of a consequential decision that matter for practical wisdom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeliberationContext {
    /// Separate factual and normative uncertainty.
    pub epistemics: EpistemicState,
    /// Number of affected stakeholders currently represented in deliberation.
    pub represented_stakeholders: usize,
    /// Number of known affected stakeholders lacking an adequate perspective.
    pub unrepresented_stakeholders: usize,
    /// Number of materially conflicting normative/value claims.
    pub value_conflicts: usize,
    /// Whether consequences across time have been considered explicitly.
    pub temporal_consequences_considered: bool,
    /// Whether the candidate action is difficult or impossible to reverse.
    pub irreversible: bool,
    /// Whether a materially safer/reversible alternative is known.
    pub reversible_alternative_available: bool,
    /// Maximum action class under consideration.
    pub authority: ActionAuthority,
}

impl Default for DeliberationContext {
    fn default() -> Self {
        Self {
            epistemics: EpistemicState::new(0.0, 0.0),
            represented_stakeholders: 1,
            unrepresented_stakeholders: 0,
            value_conflicts: 0,
            temporal_consequences_considered: true,
            irreversible: false,
            reversible_alternative_available: false,
            authority: ActionAuthority::ObserveOnly,
        }
    }
}

/// A reason that a deliberation should change before action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WisdomReason {
    HighFactualUncertainty,
    HighNormativeUncertainty,
    MissingStakeholderPerspectives,
    UnresolvedValueConflict,
    TemporalConsequencesMissing,
    IrreversibleUnderMaterialUncertainty,
    ReversibleAlternativeAvailable,
    ActionAuthorityUnderHighUncertainty,
}

/// What the practical-wisdom layer recommends doing next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WisdomDisposition {
    /// Existing information is sufficient for this bounded layer to raise no
    /// additional procedural objection. This is not a moral approval.
    ProceedWithCurrentDeliberation,
    /// Obtain more factual evidence before relying on the decision.
    GatherInformation,
    /// Seek missing or conflicting stakeholder/value perspectives.
    SeekPerspectives,
    /// Prefer an available action that preserves future options.
    PreferReversibleAction,
    /// Narrow authority or defer the decision to an accountable human process.
    DeferAuthority,
}

/// Inspectable result from the practical-wisdom kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PracticalWisdomAssessment {
    pub reasons: Vec<WisdomReason>,
    pub dispositions: Vec<WisdomDisposition>,
}

impl PracticalWisdomAssessment {
    pub fn has_reason(&self, reason: WisdomReason) -> bool {
        self.reasons.contains(&reason)
    }

    pub fn has_disposition(&self, disposition: WisdomDisposition) -> bool {
        self.dispositions.contains(&disposition)
    }
}

/// Conservative, deterministic first-pass practical-wisdom evaluator.
#[derive(Debug, Default, Clone, Copy)]
pub struct PracticalWisdomKernel;

impl PracticalWisdomKernel {
    pub fn assess(&self, context: DeliberationContext) -> PracticalWisdomAssessment {
        let mut reasons = Vec::new();
        let mut dispositions = Vec::new();

        if context.epistemics.factual_uncertainty >= 0.7 {
            push_unique(&mut reasons, WisdomReason::HighFactualUncertainty);
            push_unique(&mut dispositions, WisdomDisposition::GatherInformation);
        }

        if context.epistemics.normative_uncertainty >= 0.7 {
            push_unique(&mut reasons, WisdomReason::HighNormativeUncertainty);
            push_unique(&mut dispositions, WisdomDisposition::SeekPerspectives);
        }

        if context.unrepresented_stakeholders > 0 {
            push_unique(
                &mut reasons,
                WisdomReason::MissingStakeholderPerspectives,
            );
            push_unique(&mut dispositions, WisdomDisposition::SeekPerspectives);
        }

        if context.value_conflicts > 0 {
            push_unique(&mut reasons, WisdomReason::UnresolvedValueConflict);
            push_unique(&mut dispositions, WisdomDisposition::SeekPerspectives);
        }

        if context.authority.is_action_class() && !context.temporal_consequences_considered {
            push_unique(&mut reasons, WisdomReason::TemporalConsequencesMissing);
            push_unique(&mut dispositions, WisdomDisposition::GatherInformation);
        }

        let material_uncertainty = context.epistemics.max_uncertainty() >= 0.5;

        if context.irreversible && material_uncertainty {
            push_unique(
                &mut reasons,
                WisdomReason::IrreversibleUnderMaterialUncertainty,
            );
            push_unique(&mut dispositions, WisdomDisposition::DeferAuthority);
        }

        if context.irreversible && context.reversible_alternative_available {
            push_unique(&mut reasons, WisdomReason::ReversibleAlternativeAvailable);
            push_unique(
                &mut dispositions,
                WisdomDisposition::PreferReversibleAction,
            );
        }

        if context.authority.is_action_class() && context.epistemics.max_uncertainty() >= 0.8 {
            push_unique(
                &mut reasons,
                WisdomReason::ActionAuthorityUnderHighUncertainty,
            );
            push_unique(&mut dispositions, WisdomDisposition::DeferAuthority);
        }

        if dispositions.is_empty() {
            dispositions.push(WisdomDisposition::ProceedWithCurrentDeliberation);
        }

        PracticalWisdomAssessment {
            reasons,
            dispositions,
        }
    }
}

fn push_unique<T: PartialEq>(items: &mut Vec<T>, item: T) {
    if !items.contains(&item) {
        items.push(item);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_uncertainty_observation_has_no_extra_procedural_block() {
        let result = PracticalWisdomKernel.assess(DeliberationContext::default());
        assert!(result.reasons.is_empty());
        assert_eq!(
            result.dispositions,
            vec![WisdomDisposition::ProceedWithCurrentDeliberation]
        );
    }

    #[test]
    fn factual_and_normative_uncertainty_produce_different_next_steps() {
        let factual = PracticalWisdomKernel.assess(DeliberationContext {
            epistemics: EpistemicState::new(0.9, 0.1),
            ..Default::default()
        });
        assert!(factual.has_disposition(WisdomDisposition::GatherInformation));
        assert!(!factual.has_disposition(WisdomDisposition::SeekPerspectives));

        let normative = PracticalWisdomKernel.assess(DeliberationContext {
            epistemics: EpistemicState::new(0.1, 0.9),
            ..Default::default()
        });
        assert!(normative.has_disposition(WisdomDisposition::SeekPerspectives));
    }

    #[test]
    fn missing_stakeholder_blocks_premature_proceed() {
        let result = PracticalWisdomKernel.assess(DeliberationContext {
            unrepresented_stakeholders: 1,
            ..Default::default()
        });
        assert!(result.has_reason(WisdomReason::MissingStakeholderPerspectives));
        assert!(result.has_disposition(WisdomDisposition::SeekPerspectives));
        assert!(!result.has_disposition(WisdomDisposition::ProceedWithCurrentDeliberation));
    }

    #[test]
    fn irreversible_action_under_uncertainty_defers_authority() {
        let result = PracticalWisdomKernel.assess(DeliberationContext {
            epistemics: EpistemicState::new(0.55, 0.2),
            irreversible: true,
            authority: ActionAuthority::ActIrreversible,
            ..Default::default()
        });
        assert!(result.has_reason(WisdomReason::IrreversibleUnderMaterialUncertainty));
        assert!(result.has_disposition(WisdomDisposition::DeferAuthority));
    }

    #[test]
    fn reversible_alternative_is_preferred_for_irreversible_candidate() {
        let result = PracticalWisdomKernel.assess(DeliberationContext {
            irreversible: true,
            reversible_alternative_available: true,
            authority: ActionAuthority::ActIrreversible,
            ..Default::default()
        });
        assert!(result.has_disposition(WisdomDisposition::PreferReversibleAction));
    }

    #[test]
    fn action_without_temporal_reasoning_requests_more_information() {
        let result = PracticalWisdomKernel.assess(DeliberationContext {
            temporal_consequences_considered: false,
            authority: ActionAuthority::ActReversible,
            ..Default::default()
        });
        assert!(result.has_reason(WisdomReason::TemporalConsequencesMissing));
        assert!(result.has_disposition(WisdomDisposition::GatherInformation));
    }

    #[test]
    fn high_uncertainty_action_defers_even_when_reversible() {
        let result = PracticalWisdomKernel.assess(DeliberationContext {
            epistemics: EpistemicState::new(0.85, 0.1),
            authority: ActionAuthority::ActReversible,
            ..Default::default()
        });
        assert!(result.has_reason(WisdomReason::ActionAuthorityUnderHighUncertainty));
        assert!(result.has_disposition(WisdomDisposition::DeferAuthority));
    }
}
