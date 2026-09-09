// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed, measurement-only outcomes for exploratory interventions.
//!
//! Existing surprise-driven exploration treats a reduction in subsequent surprise as a
//! successful exploration signal. This module deliberately keeps that signal distinct
//! from other outcomes such as knowledge gain, novelty, diversity retention, and option
//! creation. It defines evidence; it does not alter exploration policy or action selection.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};

use super::generativity::{
    GenerativityEstimate, GenerativityEvidence, GenerativityValidationError,
};

/// Semantic version for the typed exploration-outcome evidence contract.
pub const EXPLORATION_OUTCOME_SCHEMA_VERSION: &str = "exploration-outcome-v1";

/// Observed surprise before and after an exploratory intervention.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurpriseTransition {
    /// Surprise that motivated or immediately preceded the intervention.
    pub before: f64,
    /// Surprise observed after the intervention.
    pub after: f64,
}

impl SurpriseTransition {
    /// Construct a checked surprise transition.
    pub fn new(before: f64, after: f64) -> Result<Self, ExplorationOutcomeValidationError> {
        let transition = Self { before, after };
        transition.validate()?;
        Ok(transition)
    }

    /// Signed change (`after - before`). Negative means surprise decreased.
    pub fn delta(&self) -> f64 {
        self.after - self.before
    }

    /// Whether the post-intervention surprise is lower than the pre-intervention value.
    pub fn reduced(&self) -> bool {
        self.after < self.before
    }

    pub fn validate(&self) -> Result<(), ExplorationOutcomeValidationError> {
        validate_surprise("before", self.before)?;
        validate_surprise("after", self.after)?;
        Ok(())
    }
}

/// Evidence-bearing outcome of one exploratory intervention.
///
/// There is intentionally no overall `successful` field or canonical score. A reduction
/// in surprise may coexist with poor novelty/option creation, while a temporarily higher
/// surprise may accompany genuine discovery. Consumers must keep those claims separate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExplorationOutcomeAssessment {
    /// Schema identifier for persisted evidence.
    pub schema: String,
    /// Stable identity of the exploration event.
    pub exploration_id: String,
    /// Stable identity of the state/action/project being explored.
    pub subject_id: String,
    /// Context in which the outcome is claimed to hold.
    pub context: String,
    /// Optional observed surprise transition.
    pub surprise: Option<SurpriseTransition>,
    /// Reliable knowledge gained from the exploration.
    pub knowledge_gain: GenerativityEstimate,
    /// Degree to which the exploration exposed a meaningfully different possibility.
    pub novelty: GenerativityEstimate,
    /// Degree to which viable alternative strategies remained available afterward.
    pub diversity_retention: GenerativityEstimate,
    /// Degree to which new viable future choices were created or preserved.
    pub option_creation: GenerativityEstimate,
    /// Evidence supporting the outcome claims.
    pub evidence: Vec<GenerativityEvidence>,
    /// Assumptions materially affecting interpretation.
    pub assumptions: Vec<String>,
    /// Important unknowns that remain unresolved.
    pub unresolved_uncertainties: Vec<String>,
}

impl ExplorationOutcomeAssessment {
    pub fn new(
        exploration_id: impl Into<String>,
        subject_id: impl Into<String>,
        context: impl Into<String>,
        knowledge_gain: GenerativityEstimate,
        novelty: GenerativityEstimate,
        diversity_retention: GenerativityEstimate,
        option_creation: GenerativityEstimate,
    ) -> Self {
        Self {
            schema: EXPLORATION_OUTCOME_SCHEMA_VERSION.to_string(),
            exploration_id: exploration_id.into(),
            subject_id: subject_id.into(),
            context: context.into(),
            surprise: None,
            knowledge_gain,
            novelty,
            diversity_retention,
            option_creation,
            evidence: Vec::new(),
            assumptions: Vec::new(),
            unresolved_uncertainties: Vec::new(),
        }
    }

    /// Return the descriptive surprise-reduction signal when a transition was observed.
    pub fn surprise_reduced(&self) -> Option<bool> {
        self.surprise.map(|transition| transition.reduced())
    }

    /// Validate evidence-contract invariants without making a policy decision.
    pub fn validate(&self) -> Result<(), ExplorationOutcomeValidationError> {
        if self.schema != EXPLORATION_OUTCOME_SCHEMA_VERSION {
            return Err(ExplorationOutcomeValidationError::UnsupportedSchema(
                self.schema.clone(),
            ));
        }
        validate_non_empty("exploration_id", &self.exploration_id)?;
        validate_non_empty("subject_id", &self.subject_id)?;
        validate_non_empty("context", &self.context)?;

        if let Some(surprise) = self.surprise {
            surprise.validate()?;
        }

        self.knowledge_gain.validate()?;
        self.novelty.validate()?;
        self.diversity_retention.validate()?;
        self.option_creation.validate()?;

        for evidence in &self.evidence {
            validate_non_empty("evidence_id", &evidence.evidence_id)?;
            validate_non_empty("evidence.kind", &evidence.kind)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExplorationOutcomeValidationError {
    EmptyField(&'static str),
    NonFiniteSurprise { field: &'static str, value: f64 },
    NegativeSurprise { field: &'static str, value: f64 },
    UnsupportedSchema(String),
    Generativity(GenerativityValidationError),
}

impl From<GenerativityValidationError> for ExplorationOutcomeValidationError {
    fn from(value: GenerativityValidationError) -> Self {
        Self::Generativity(value)
    }
}

impl std::fmt::Display for ExplorationOutcomeValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "required field is empty: {field}"),
            Self::NonFiniteSurprise { field, value } => {
                write!(f, "surprise {field} must be finite, got {value}")
            }
            Self::NegativeSurprise { field, value } => {
                write!(f, "surprise {field} must be non-negative, got {value}")
            }
            Self::UnsupportedSchema(schema) => {
                write!(f, "unsupported exploration-outcome schema: {schema}")
            }
            Self::Generativity(err) => write!(f, "invalid generativity estimate: {err}"),
        }
    }
}

impl std::error::Error for ExplorationOutcomeValidationError {}

fn validate_non_empty(
    field: &'static str,
    value: &str,
) -> Result<(), ExplorationOutcomeValidationError> {
    if value.trim().is_empty() {
        return Err(ExplorationOutcomeValidationError::EmptyField(field));
    }
    Ok(())
}

fn validate_surprise(
    field: &'static str,
    value: f64,
) -> Result<(), ExplorationOutcomeValidationError> {
    if !value.is_finite() {
        return Err(ExplorationOutcomeValidationError::NonFiniteSurprise { field, value });
    }
    if value < 0.0 {
        return Err(ExplorationOutcomeValidationError::NegativeSurprise { field, value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate(value: f64) -> GenerativityEstimate {
        GenerativityEstimate::new(value, 0.8).expect("valid estimate")
    }

    fn outcome() -> ExplorationOutcomeAssessment {
        ExplorationOutcomeAssessment::new(
            "explore:17",
            "state:controller-a",
            "controller search episode",
            estimate(0.7),
            estimate(0.8),
            estimate(0.9),
            estimate(0.75),
        )
    }

    #[test]
    fn surprise_reduction_is_descriptive_not_overall_success() {
        let mut assessment = outcome();
        assessment.surprise = Some(SurpriseTransition::new(0.8, 0.2).unwrap());
        assessment.knowledge_gain = estimate(0.05);
        assessment.option_creation = estimate(0.05);

        assert_eq!(assessment.surprise_reduced(), Some(true));
        assert_eq!(assessment.knowledge_gain.value, 0.05);
        assert_eq!(assessment.option_creation.value, 0.05);
        assert!(assessment.validate().is_ok());
    }

    #[test]
    fn discovery_can_raise_surprise_and_still_be_valid_evidence() {
        let mut assessment = outcome();
        assessment.surprise = Some(SurpriseTransition::new(0.2, 0.9).unwrap());
        assessment.knowledge_gain = estimate(0.95);
        assessment.novelty = estimate(0.95);

        assert_eq!(assessment.surprise_reduced(), Some(false));
        assert_eq!(assessment.knowledge_gain.value, 0.95);
        assert!(assessment.validate().is_ok());
    }

    #[test]
    fn transition_delta_is_signed() {
        let reduced = SurpriseTransition::new(0.9, 0.3).unwrap();
        let increased = SurpriseTransition::new(0.2, 0.7).unwrap();
        assert!((reduced.delta() + 0.6).abs() < 1e-12);
        assert!((increased.delta() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn rejects_negative_surprise() {
        let err = SurpriseTransition::new(-0.1, 0.2).unwrap_err();
        assert!(matches!(
            err,
            ExplorationOutcomeValidationError::NegativeSurprise {
                field: "before",
                ..
            }
        ));
    }

    #[test]
    fn rejects_non_finite_surprise() {
        let err = SurpriseTransition::new(0.1, f64::INFINITY).unwrap_err();
        assert!(matches!(
            err,
            ExplorationOutcomeValidationError::NonFiniteSurprise {
                field: "after",
                ..
            }
        ));
    }

    #[test]
    fn rejects_invalid_estimate_from_deserialized_state() {
        let mut assessment = outcome();
        assessment.novelty = GenerativityEstimate {
            value: 1.2,
            confidence: 0.9,
        };
        assert!(matches!(
            assessment.validate(),
            Err(ExplorationOutcomeValidationError::Generativity(
                GenerativityValidationError::OutOfRange { .. }
            ))
        ));
    }

    #[test]
    fn rejects_unknown_schema() {
        let mut assessment = outcome();
        assessment.schema = "exploration-outcome-v2".into();
        assert!(matches!(
            assessment.validate(),
            Err(ExplorationOutcomeValidationError::UnsupportedSchema(_))
        ));
    }
}
