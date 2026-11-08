// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Typed epistemic provenance for visual cognition.
//!
//! The core invariant is intentionally asymmetric:
//!
//! ```text
//! observation evidence may originate only at an observation boundary
//! prediction / memory / simulation / counterfactual state may never upgrade itself to observed
//! ```
//!
//! This module does not grant action authority. It only preserves what kind of visual
//! claim a datum represents and the evidence it is allowed to cite.

use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

/// Epistemic origin of a visual claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualOrigin {
    /// Direct sensor evidence from a concrete observation boundary.
    Observed,
    /// Recalled evidence whose original observation lineage is retained.
    Remembered,
    /// A belief inferred from one or more evidence-bearing claims.
    Inferred,
    /// A forecast about a future visual state.
    Predicted,
    /// A model-generated visual state used for internal simulation.
    Simulated,
    /// A hypothetical visual state under an explicit intervention or assumption.
    Counterfactual,
}

impl VisualOrigin {
    /// Returns true only for direct sensor-originated claims.
    pub const fn is_observed(self) -> bool {
        matches!(self, Self::Observed)
    }

    /// Returns true for origins that are generated rather than directly sensed.
    pub const fn is_generative(self) -> bool {
        matches!(self, Self::Predicted | Self::Simulated | Self::Counterfactual)
    }
}

/// Stable identity of one concrete visual observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VisualObservationRef {
    pub source_id: u64,
    pub frame_id: u64,
    pub captured_at_us: u64,
}

impl VisualObservationRef {
    pub const fn new(source_id: u64, frame_id: u64, captured_at_us: u64) -> Self {
        Self {
            source_id,
            frame_id,
            captured_at_us,
        }
    }
}

/// A compact, typed provenance envelope for visual claims.
///
/// Fields are private so production code cannot construct contradictory provenance directly.
/// Deserialization is also validated before a `VisualEvidence` value can exist.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VisualEvidence {
    origin: VisualOrigin,
    observation: Option<VisualObservationRef>,
    parent_observations: Vec<VisualObservationRef>,
    confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct VisualEvidenceWire {
    origin: VisualOrigin,
    observation: Option<VisualObservationRef>,
    parent_observations: Vec<VisualObservationRef>,
    confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisualEvidenceError {
    InvalidConfidence,
    ObservedRequiresObservation,
    ObservedCannotCarryParentLineage,
    NonObservedCannotCarryDirectObservation,
    RememberedRequiresObservationLineage,
    InferredRequiresObservationLineage,
    DuplicateParentObservation,
}

impl fmt::Display for VisualEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidConfidence => "visual confidence must be finite and within [0, 1]",
            Self::ObservedRequiresObservation => {
                "observed visual evidence requires a direct observation reference"
            }
            Self::ObservedCannotCarryParentLineage => {
                "observed visual evidence cannot carry derived parent-observation lineage"
            }
            Self::NonObservedCannotCarryDirectObservation => {
                "non-observed visual evidence cannot carry the direct-observation slot"
            }
            Self::RememberedRequiresObservationLineage => {
                "remembered visual evidence requires original observation lineage"
            }
            Self::InferredRequiresObservationLineage => {
                "inferred visual evidence requires supporting observation lineage"
            }
            Self::DuplicateParentObservation => {
                "visual evidence cannot count the same parent observation more than once"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for VisualEvidenceError {}

impl<'de> Deserialize<'de> for VisualEvidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = VisualEvidenceWire::deserialize(deserializer)?;
        let evidence = Self {
            origin: wire.origin,
            observation: wire.observation,
            parent_observations: wire.parent_observations,
            confidence: wire.confidence,
        };
        evidence.validate().map_err(serde::de::Error::custom)?;
        Ok(evidence)
    }
}

impl VisualEvidence {
    /// Construct direct observation evidence.
    ///
    /// This is the only constructor that can produce `VisualOrigin::Observed`.
    pub fn observed(
        observation: VisualObservationRef,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::validate_confidence(confidence)?;
        Ok(Self {
            origin: VisualOrigin::Observed,
            observation: Some(observation),
            parent_observations: Vec::new(),
            confidence,
        })
    }

    /// Construct remembered evidence from one or more original observations.
    pub fn remembered(
        parent_observations: Vec<VisualObservationRef>,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::from_derived(VisualOrigin::Remembered, parent_observations, confidence)
    }

    /// Construct an inference grounded in one or more observations.
    pub fn inferred(
        parent_observations: Vec<VisualObservationRef>,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::from_derived(VisualOrigin::Inferred, parent_observations, confidence)
    }

    /// Construct a prediction. Parent observation lineage is optional because some
    /// forecasts may be generated from a purely simulated initial state.
    pub fn predicted(
        parent_observations: Vec<VisualObservationRef>,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::from_generated(VisualOrigin::Predicted, parent_observations, confidence)
    }

    /// Construct internal simulation evidence.
    pub fn simulated(
        parent_observations: Vec<VisualObservationRef>,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::from_generated(VisualOrigin::Simulated, parent_observations, confidence)
    }

    /// Construct an explicit counterfactual visual claim.
    pub fn counterfactual(
        parent_observations: Vec<VisualObservationRef>,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::from_generated(VisualOrigin::Counterfactual, parent_observations, confidence)
    }

    pub const fn origin(&self) -> VisualOrigin {
        self.origin
    }

    pub const fn observation(&self) -> Option<VisualObservationRef> {
        self.observation
    }

    pub fn parent_observations(&self) -> &[VisualObservationRef] {
        &self.parent_observations
    }

    pub const fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Revalidate an envelope after any in-memory trust-boundary crossing.
    ///
    /// Wire deserialization already performs this validation automatically.
    pub fn validate(&self) -> Result<(), VisualEvidenceError> {
        Self::validate_confidence(self.confidence)?;
        Self::validate_unique_parents(&self.parent_observations)?;

        match self.origin {
            VisualOrigin::Observed => {
                if self.observation.is_none() {
                    return Err(VisualEvidenceError::ObservedRequiresObservation);
                }
                if !self.parent_observations.is_empty() {
                    return Err(VisualEvidenceError::ObservedCannotCarryParentLineage);
                }
            }
            VisualOrigin::Remembered => {
                if self.observation.is_some() {
                    return Err(VisualEvidenceError::NonObservedCannotCarryDirectObservation);
                }
                if self.parent_observations.is_empty() {
                    return Err(VisualEvidenceError::RememberedRequiresObservationLineage);
                }
            }
            VisualOrigin::Inferred => {
                if self.observation.is_some() {
                    return Err(VisualEvidenceError::NonObservedCannotCarryDirectObservation);
                }
                if self.parent_observations.is_empty() {
                    return Err(VisualEvidenceError::InferredRequiresObservationLineage);
                }
            }
            VisualOrigin::Predicted | VisualOrigin::Simulated | VisualOrigin::Counterfactual => {
                if self.observation.is_some() {
                    return Err(VisualEvidenceError::NonObservedCannotCarryDirectObservation);
                }
            }
        }

        Ok(())
    }

    fn from_derived(
        origin: VisualOrigin,
        parent_observations: Vec<VisualObservationRef>,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::validate_confidence(confidence)?;
        Self::validate_unique_parents(&parent_observations)?;
        if parent_observations.is_empty() {
            return Err(match origin {
                VisualOrigin::Remembered => VisualEvidenceError::RememberedRequiresObservationLineage,
                VisualOrigin::Inferred => VisualEvidenceError::InferredRequiresObservationLineage,
                _ => unreachable!("derived constructor accepts only remembered/inferred origins"),
            });
        }
        Ok(Self {
            origin,
            observation: None,
            parent_observations,
            confidence,
        })
    }

    fn from_generated(
        origin: VisualOrigin,
        parent_observations: Vec<VisualObservationRef>,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::validate_confidence(confidence)?;
        Self::validate_unique_parents(&parent_observations)?;
        debug_assert!(origin.is_generative());
        Ok(Self {
            origin,
            observation: None,
            parent_observations,
            confidence,
        })
    }

    fn validate_confidence(confidence: f32) -> Result<(), VisualEvidenceError> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(VisualEvidenceError::InvalidConfidence);
        }
        Ok(())
    }

    fn validate_unique_parents(
        parent_observations: &[VisualObservationRef],
    ) -> Result<(), VisualEvidenceError> {
        for (idx, item) in parent_observations.iter().enumerate() {
            if parent_observations[..idx].contains(item) {
                return Err(VisualEvidenceError::DuplicateParentObservation);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(frame_id: u64) -> VisualObservationRef {
        VisualObservationRef::new(7, frame_id, 1_000_000 + frame_id)
    }

    #[test]
    fn observed_requires_direct_observation_identity() {
        let evidence = VisualEvidence::observed(obs(42), 0.9).unwrap();
        assert_eq!(evidence.origin(), VisualOrigin::Observed);
        assert_eq!(evidence.observation(), Some(obs(42)));
        assert!(evidence.parent_observations().is_empty());
        assert!(evidence.validate().is_ok());
    }

    #[test]
    fn remembered_and_inferred_require_observation_lineage() {
        assert_eq!(
            VisualEvidence::remembered(Vec::new(), 0.8),
            Err(VisualEvidenceError::RememberedRequiresObservationLineage)
        );
        assert_eq!(
            VisualEvidence::inferred(Vec::new(), 0.8),
            Err(VisualEvidenceError::InferredRequiresObservationLineage)
        );
    }

    #[test]
    fn generated_origins_never_claim_direct_observation() {
        for evidence in [
            VisualEvidence::predicted(vec![obs(1)], 0.7).unwrap(),
            VisualEvidence::simulated(vec![obs(1)], 0.7).unwrap(),
            VisualEvidence::counterfactual(vec![obs(1)], 0.7).unwrap(),
        ] {
            assert!(!evidence.origin().is_observed());
            assert!(evidence.observation().is_none());
            assert!(evidence.validate().is_ok());
        }
    }

    #[test]
    fn invalid_confidence_fails_closed() {
        assert_eq!(
            VisualEvidence::observed(obs(1), f32::NAN),
            Err(VisualEvidenceError::InvalidConfidence)
        );
        assert_eq!(
            VisualEvidence::simulated(Vec::new(), 1.1),
            Err(VisualEvidenceError::InvalidConfidence)
        );
    }

    #[test]
    fn duplicate_parent_observations_are_rejected() {
        assert_eq!(
            VisualEvidence::predicted(vec![obs(3), obs(3)], 0.6),
            Err(VisualEvidenceError::DuplicateParentObservation)
        );
    }

    #[test]
    fn serde_roundtrip_preserves_origin_and_lineage() {
        let evidence = VisualEvidence::counterfactual(vec![obs(2), obs(3)], 0.55).unwrap();
        let json = serde_json::to_string(&evidence).unwrap();
        let decoded: VisualEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, evidence);
        assert!(decoded.validate().is_ok());
    }

    #[test]
    fn tampered_wire_state_cannot_deserialize_as_observed() {
        let json = r#"{
            "origin":"observed",
            "observation":null,
            "parent_observations":[],
            "confidence":0.99
        }"#;
        assert!(serde_json::from_str::<VisualEvidence>(json).is_err());
    }

    #[test]
    fn tampered_simulation_cannot_carry_direct_observation() {
        let json = r#"{
            "origin":"simulated",
            "observation":{"source_id":1,"frame_id":9,"captured_at_us":10},
            "parent_observations":[],
            "confidence":0.5
        }"#;
        assert!(serde_json::from_str::<VisualEvidence>(json).is_err());
    }
}
