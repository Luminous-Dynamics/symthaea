// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bearing generativity evaluation primitives.
//!
//! Generativity is intentionally represented as a vector rather than a single score.
//! The model describes consequences of actions, projects, resources, or policies across
//! multiple dimensions without asserting one universal moral ordering over them.
//!
//! This module is measurement-only. It does not alter EFE, MAGI action selection,
//! execution authority, governance weight, reputation, or financial allocation.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Semantic version for the generativity evidence contract.
pub const GENERATIVITY_SCHEMA_VERSION: &str = "generativity-vector-v1";

/// A bounded estimate for one generativity dimension.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GenerativityEstimate {
    /// Estimated magnitude of the dimension, normalized to `[0.0, 1.0]`.
    pub value: f64,
    /// Confidence in the estimate, normalized to `[0.0, 1.0]`.
    pub confidence: f64,
}

impl GenerativityEstimate {
    /// Construct a checked estimate.
    pub fn new(value: f64, confidence: f64) -> Result<Self, GenerativityValidationError> {
        let estimate = Self { value, confidence };
        estimate.validate()?;
        Ok(estimate)
    }

    /// Validate numeric bounds and finiteness.
    pub fn validate(&self) -> Result<(), GenerativityValidationError> {
        validate_unit_interval("value", self.value)?;
        validate_unit_interval("confidence", self.confidence)?;
        Ok(())
    }
}

/// Multi-dimensional evidence model for generative consequences.
///
/// Positive dimensions describe productive capacity or optionality created.
/// Risk dimensions describe capacities constrained, depleted, concentrated, or
/// made difficult to recover. Consumers should preserve the vector and apply an
/// explicit local policy when comparing alternatives; no canonical scalar score
/// is provided by this type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerativityVector {
    pub immediate_utility: GenerativityEstimate,
    pub epistemic_gain: GenerativityEstimate,
    pub option_value: GenerativityEstimate,
    pub diversity: GenerativityEstimate,
    pub capability_gain: GenerativityEstimate,
    pub diffusion: GenerativityEstimate,
    pub commons_gain: GenerativityEstimate,
    pub regeneration: GenerativityEstimate,
    pub dependency_risk: GenerativityEstimate,
    pub concentration_risk: GenerativityEstimate,
    pub irreversibility_risk: GenerativityEstimate,
}

impl GenerativityVector {
    /// Validate every estimate in the vector.
    pub fn validate(&self) -> Result<(), GenerativityValidationError> {
        for estimate in [
            self.immediate_utility,
            self.epistemic_gain,
            self.option_value,
            self.diversity,
            self.capability_gain,
            self.diffusion,
            self.commons_gain,
            self.regeneration,
            self.dependency_risk,
            self.concentration_risk,
            self.irreversibility_risk,
        ] {
            estimate.validate()?;
        }
        Ok(())
    }
}

/// Provenance for a generativity assessment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerativityEvidence {
    pub evidence_id: String,
    /// Examples: `measurement`, `simulation`, `attestation`, `peer_review`, `domain_model`.
    pub kind: String,
    /// Optional content-addressed or externally resolvable reference.
    pub reference: Option<String>,
    /// What the evidence supports or fails to establish.
    pub note: Option<String>,
}

/// A complete, non-authoritative assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerativityAssessment {
    pub schema: String,
    pub subject_id: String,
    pub context: String,
    pub vector: GenerativityVector,
    pub evidence: Vec<GenerativityEvidence>,
    pub assumptions: Vec<String>,
    pub unresolved_uncertainties: Vec<String>,
}

impl GenerativityAssessment {
    pub fn new(
        subject_id: impl Into<String>,
        context: impl Into<String>,
        vector: GenerativityVector,
    ) -> Self {
        Self {
            schema: GENERATIVITY_SCHEMA_VERSION.to_string(),
            subject_id: subject_id.into(),
            context: context.into(),
            vector,
            evidence: Vec::new(),
            assumptions: Vec::new(),
            unresolved_uncertainties: Vec::new(),
        }
    }

    /// Validate invariants without asserting that the assessment is authoritative.
    pub fn validate(&self) -> Result<(), GenerativityValidationError> {
        if self.schema != GENERATIVITY_SCHEMA_VERSION {
            return Err(GenerativityValidationError::UnsupportedSchema(self.schema.clone()));
        }
        if self.subject_id.trim().is_empty() {
            return Err(GenerativityValidationError::EmptyField("subject_id"));
        }
        if self.context.trim().is_empty() {
            return Err(GenerativityValidationError::EmptyField("context"));
        }
        self.vector.validate()?;
        for evidence in &self.evidence {
            if evidence.evidence_id.trim().is_empty() {
                return Err(GenerativityValidationError::EmptyField("evidence_id"));
            }
            if evidence.kind.trim().is_empty() {
                return Err(GenerativityValidationError::EmptyField("evidence.kind"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenerativityValidationError {
    EmptyField(&'static str),
    NonFinite { field: &'static str, value: f64 },
    OutOfRange { field: &'static str, value: f64 },
    UnsupportedSchema(String),
}

impl std::fmt::Display for GenerativityValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "required field is empty: {field}"),
            Self::NonFinite { field, value } => write!(f, "{field} must be finite, got {value}"),
            Self::OutOfRange { field, value } => write!(f, "{field} must be within [0, 1], got {value}"),
            Self::UnsupportedSchema(schema) => write!(f, "unsupported generativity schema: {schema}"),
        }
    }
}

impl std::error::Error for GenerativityValidationError {}

fn validate_unit_interval(
    field: &'static str,
    value: f64,
) -> Result<(), GenerativityValidationError> {
    if !value.is_finite() {
        return Err(GenerativityValidationError::NonFinite { field, value });
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(GenerativityValidationError::OutOfRange { field, value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate(value: f64) -> GenerativityEstimate {
        GenerativityEstimate::new(value, 0.8).expect("valid estimate")
    }

    fn sample_vector() -> GenerativityVector {
        GenerativityVector {
            immediate_utility: estimate(0.7),
            epistemic_gain: estimate(0.8),
            option_value: estimate(0.9),
            diversity: estimate(0.6),
            capability_gain: estimate(0.7),
            diffusion: estimate(0.8),
            commons_gain: estimate(0.5),
            regeneration: estimate(0.6),
            dependency_risk: estimate(0.2),
            concentration_risk: estimate(0.1),
            irreversibility_risk: estimate(0.15),
        }
    }

    #[test]
    fn accepts_valid_assessment() {
        let mut assessment = GenerativityAssessment::new(
            "proposal:42",
            "municipal water reuse pilot",
            sample_vector(),
        );
        assessment.evidence.push(GenerativityEvidence {
            evidence_id: "measurement:1".into(),
            kind: "measurement".into(),
            reference: Some("blake3:abc".into()),
            note: Some("measured water recovery rate".into()),
        });
        assert_eq!(assessment.validate(), Ok(()));
    }

    #[test]
    fn rejects_out_of_range_estimate() {
        let err = GenerativityEstimate::new(1.01, 0.9).unwrap_err();
        assert!(matches!(
            err,
            GenerativityValidationError::OutOfRange { field: "value", .. }
        ));
    }

    #[test]
    fn rejects_non_finite_confidence() {
        let err = GenerativityEstimate::new(0.5, f64::NAN).unwrap_err();
        assert!(matches!(
            err,
            GenerativityValidationError::NonFinite {
                field: "confidence",
                ..
            }
        ));
    }

    #[test]
    fn rejects_unknown_schema() {
        let mut assessment = GenerativityAssessment::new("x", "y", sample_vector());
        assessment.schema = "future-schema".into();
        assert!(matches!(
            assessment.validate(),
            Err(GenerativityValidationError::UnsupportedSchema(_))
        ));
    }

    #[test]
    fn rejects_empty_evidence_identity() {
        let mut assessment = GenerativityAssessment::new("x", "y", sample_vector());
        assessment.evidence.push(GenerativityEvidence {
            evidence_id: "   ".into(),
            kind: "simulation".into(),
            reference: None,
            note: None,
        });
        assert_eq!(
            assessment.validate(),
            Err(GenerativityValidationError::EmptyField("evidence_id"))
        );
    }
}
