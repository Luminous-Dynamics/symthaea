// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed linguistic formulation between semantic role structure and phonological planning.
//!
//! This layer deliberately stops before lexical realization. It determines an admissible
//! constituent order and clause-shaping policy from already-grounded roles, but never
//! invents words, inflection, agreement, or omitted content.

use serde::{Deserialize, Serialize};

use crate::speech_plan::{ClauseMode, EpistemicDelivery, SpeechPlan, SpeechPlanRole};

pub const LINGUISTIC_FRAME_VERSION: &str = "broca-linguistic-frame-v1";

/// How much linguistic content is actually available for formulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinguisticBindingStatus {
    /// Role/filler structure only; lexical wording is unavailable.
    RoleStructureOnly,
    /// Explicit lexical bindings are available upstream, but inflectional realization may remain external.
    LexicallyBound,
}

/// Conservative clause-formulation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FormulationStrategy {
    Declarative,
    Interrogative,
    Directive,
    Reflective,
    Relational,
    Abstain,
}

/// Typed constituent slot in a linearized clause skeleton.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstituentSlot {
    pub role: String,
    pub prime: String,
    /// Zero-based linearization position.
    pub position: usize,
    /// Whether this role is structurally expected for the selected strategy.
    pub required: bool,
    /// Whether this constituent is the focus target from the speech plan.
    pub is_focus: bool,
}

/// Linguistic formulation frame produced without lexical hallucination.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinguisticFrame {
    pub version: String,
    pub binding_status: LinguisticBindingStatus,
    pub strategy: FormulationStrategy,
    pub epistemic_delivery: EpistemicDelivery,
    pub focus_role: Option<String>,
    pub constituents: Vec<ConstituentSlot>,
}

impl LinguisticFrame {
    /// Build a conservative linearization skeleton from the existing speech plan.
    ///
    /// Only roles already present in the plan can enter the frame. Missing lexical
    /// material remains missing; the frame contains no generated words.
    pub fn from_speech_plan(plan: &SpeechPlan) -> Self {
        let strategy = strategy_for(plan.clause_mode);
        let required_roles = required_roles(strategy);

        let mut selected = ordered_roles(plan, strategy);
        if selected.is_empty() && !matches!(strategy, FormulationStrategy::Abstain) {
            selected = plan.roles.clone();
        }

        let constituents = selected
            .iter()
            .enumerate()
            .map(|(position, role)| ConstituentSlot {
                role: role.role.clone(),
                prime: role.prime.clone(),
                position,
                required: required_roles.iter().any(|required| *required == role.role),
                is_focus: plan.focus_role.as_deref() == Some(role.role.as_str()),
            })
            .collect();

        Self {
            version: LINGUISTIC_FRAME_VERSION.to_string(),
            binding_status: LinguisticBindingStatus::RoleStructureOnly,
            strategy,
            epistemic_delivery: plan.epistemic_delivery,
            focus_role: plan.focus_role.clone(),
            constituents,
        }
    }

    /// Bind explicit lexical provenance after the lexicalization layer has produced it.
    ///
    /// The lexical text itself stays outside this contract; only its provenance is attached.
    pub fn bind_lexical_provenance(
        &mut self,
        provenance: impl Into<String>,
    ) -> Result<(), LinguisticFrameError> {
        let provenance = provenance.into();
        if provenance.trim().is_empty() {
            return Err(LinguisticFrameError::EmptyLexicalProvenance);
        }

        self.binding_status = LinguisticBindingStatus::LexicallyBound;
        Ok(())
    }

    /// Validate persisted/deserialized cross-field invariants.
    pub fn validate(&self) -> Result<(), LinguisticFrameError> {
        if self
            .constituents
            .iter()
            .enumerate()
            .any(|(index, slot)| slot.position != index)
        {
            return Err(LinguisticFrameError::NonContiguousPositions);
        }

        if self
            .constituents
            .iter()
            .filter(|slot| slot.is_focus)
            .count()
            > 1
        {
            return Err(LinguisticFrameError::MultipleFocusRoles);
        }

        if let Some(focus) = self.focus_role.as_deref() {
            if !self.constituents.iter().any(|slot| slot.role == focus && slot.is_focus) {
                return Err(LinguisticFrameError::FocusRoleNotRepresented);
            }
        }

        let required_roles = required_roles(self.strategy);
        for required in required_roles {
            if !self.constituents.iter().any(|slot| slot.role == *required && slot.required) {
                return Err(LinguisticFrameError::RequiredRoleMissing((*required).to_string()));
            }
        }

        Ok(())
    }

    /// True only when the linearization skeleton is internally valid and non-abstaining.
    pub fn ready_for_phonology(&self) -> bool {
        !matches!(self.strategy, FormulationStrategy::Abstain)
            && !self.constituents.is_empty()
            && self.validate().is_ok()
    }

    /// Stable evidence surface.
    pub fn grounding_surface(&self) -> String {
        let constituents = self
            .constituents
            .iter()
            .map(|slot| {
                format!(
                    "{}:{}@{}:required={}:focus={}",
                    slot.role, slot.prime, slot.position, slot.required, slot.is_focus
                )
            })
            .collect::<Vec<_>>()
            .join("|");

        format!(
            "{};binding={:?};strategy={:?};epistemic={:?};focus={};constituents={}",
            self.version,
            self.binding_status,
            self.strategy,
            self.epistemic_delivery,
            self.focus_role.as_deref().unwrap_or("NONE"),
            constituents,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinguisticFrameError {
    EmptyLexicalProvenance,
    NonContiguousPositions,
    MultipleFocusRoles,
    FocusRoleNotRepresented,
    RequiredRoleMissing(String),
}

impl std::fmt::Display for LinguisticFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyLexicalProvenance => {
                write!(f, "lexical binding requires non-empty provenance")
            }
            Self::NonContiguousPositions => {
                write!(f, "constituent positions must be contiguous from zero")
            }
            Self::MultipleFocusRoles => {
                write!(f, "a linguistic frame may carry at most one focus role")
            }
            Self::FocusRoleNotRepresented => {
                write!(f, "focus role must be represented by a focused constituent")
            }
            Self::RequiredRoleMissing(role) => {
                write!(f, "required role {role} is missing from the formulation frame")
            }
        }
    }
}

impl std::error::Error for LinguisticFrameError {}

fn strategy_for(mode: ClauseMode) -> FormulationStrategy {
    match mode {
        ClauseMode::Statement => FormulationStrategy::Declarative,
        ClauseMode::Question => FormulationStrategy::Interrogative,
        ClauseMode::Directive => FormulationStrategy::Directive,
        ClauseMode::Reflective => FormulationStrategy::Reflective,
        ClauseMode::Relational => FormulationStrategy::Relational,
        ClauseMode::Abstention => FormulationStrategy::Abstain,
    }
}

fn required_roles(strategy: FormulationStrategy) -> &'static [&'static str] {
    match strategy {
        FormulationStrategy::Directive => &["AGENT", "ACTION"],
        FormulationStrategy::Interrogative
        | FormulationStrategy::Declarative
        | FormulationStrategy::Reflective
        | FormulationStrategy::Relational => &["AGENT", "ACTION"],
        FormulationStrategy::Abstain => &[],
    }
}

/// Conservative ordering only; this does not add or remove semantic content.
fn ordered_roles(plan: &SpeechPlan, strategy: FormulationStrategy) -> Vec<SpeechPlanRole> {
    let mut roles = plan.roles.clone();

    let role_priority = match strategy {
        FormulationStrategy::Interrogative => {
            vec!["ACTION", "AGENT", "PATIENT", "PREDICATE", "EVALUATOR", "TIME", "REASON"]
        }
        FormulationStrategy::Directive | FormulationStrategy::Declarative => {
            vec!["AGENT", "ACTION", "PATIENT", "PREDICATE", "EVALUATOR", "LOCATION", "TIME", "REASON"]
        }
        FormulationStrategy::Reflective => {
            vec!["AGENT", "PREDICATE", "ACTION", "EVALUATOR", "PATIENT", "TIME", "REASON"]
        }
        FormulationStrategy::Relational => {
            vec!["AGENT", "ACTION", "PATIENT", "EVALUATOR", "PREDICATE", "LOCATION", "TIME", "REASON"]
        }
        FormulationStrategy::Abstain => Vec::new(),
    };

    roles.sort_by_key(|role| {
        role_priority
            .iter()
            .position(|candidate| *candidate == role.role)
            .unwrap_or(role_priority.len())
    });
    roles
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StructuredDecoder, ThoughtChannels};
    use symthaea_core::genesis::GenesisSeed;

    fn plan_for(intent: usize, epistemic: f32) -> SpeechPlan {
        let genesis = GenesisSeed::from_phrase("linguistic-frame-test");
        let decoder = StructuredDecoder::new(&genesis);
        let mut channels = ThoughtChannels::with_intent(intent);
        channels.set_epistemic(epistemic);
        let readout = decoder.decode(&channels);
        SpeechPlan::from_readout(&channels, &readout)
    }

    #[test]
    fn statement_frame_linearizes_existing_roles_only() {
        let frame = LinguisticFrame::from_speech_plan(&plan_for(4, 0.0));

        assert_eq!(frame.strategy, FormulationStrategy::Declarative);
        assert!(frame.ready_for_phonology());
        assert_eq!(frame.constituents.first().map(|slot| slot.role.as_str()), Some("AGENT"));
        assert_eq!(frame.constituents.get(1).map(|slot| slot.role.as_str()), Some("ACTION"));
    }

    #[test]
    fn abstention_does_not_become_a_realizable_clause() {
        let frame = LinguisticFrame::from_speech_plan(&plan_for(7, 4.0));

        assert_eq!(frame.strategy, FormulationStrategy::Abstain);
        assert!(!frame.ready_for_phonology());
    }

    #[test]
    fn focus_must_be_represented() {
        let mut plan = plan_for(4, 0.0);
        plan.focus_role = Some("PATIENT".to_string());
        let frame = LinguisticFrame::from_speech_plan(&plan);

        assert!(frame.constituents.iter().any(|slot| slot.role == "PATIENT" && slot.is_focus));
        assert!(frame.validate().is_ok());
    }

    #[test]
    fn lexical_provenance_is_explicit() {
        let mut frame = LinguisticFrame::from_speech_plan(&plan_for(4, 0.0));
        frame.bind_lexical_provenance("lexicalizer:v1").unwrap();
        assert_eq!(frame.binding_status, LinguisticBindingStatus::LexicallyBound);
    }

    #[test]
    fn grounding_is_deterministic() {
        let frame = LinguisticFrame::from_speech_plan(&plan_for(2, 0.0));
        assert_eq!(frame.grounding_surface(), frame.grounding_surface());
    }
}
