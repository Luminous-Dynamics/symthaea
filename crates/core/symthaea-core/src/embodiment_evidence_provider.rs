// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Additive evidence-provider seam for legacy and v2 embodiments.
//!
//! The legacy [`crate::embodiment::EmbodimentBridge`] API exposes several bare
//! numerical getters whose defaults are compatibility conveniences rather than
//! observed evidence. This module provides a parallel, opt-in contract whose
//! defaults are explicitly [`EvidenceAvailability::Unavailable`].
//!
//! New claim-bearing, safety, trust, and HIL consumers should depend on
//! [`EmbodimentEvidenceProviderV1`] (or a later version), not infer evidence
//! quality from the old numerical getters.

use crate::embodiment::EmbodimentBridge;
use crate::embodiment_evidence::{
    EmbodimentEvidenceKind, EvidenceAvailability, EvidenceMetadataV1, EvidenceSourceClass,
    EvidenceValidationError, NormalizedScalarEvidenceV1, NormalizedVectorEvidenceV1,
};

/// Parallel evidence API for embodiment health, quality, and model metrics.
///
/// Implementing this trait with no overrides is intentionally meaningful: every
/// evidence channel reports `Unavailable`. A platform must opt in to each stronger
/// proposition with a validated evidence record.
pub trait EmbodimentEvidenceProviderV1: Send + Sync {
    /// Current per-actuator health evidence.
    fn actuator_health_evidence(
        &self,
    ) -> Result<NormalizedVectorEvidenceV1, EvidenceValidationError> {
        NormalizedVectorEvidenceV1::unavailable(EmbodimentEvidenceKind::ActuatorHealth)
    }

    /// Current sensorimotor-accuracy evidence.
    fn sensorimotor_accuracy_evidence(
        &self,
    ) -> Result<NormalizedScalarEvidenceV1, EvidenceValidationError> {
        NormalizedScalarEvidenceV1::unavailable(EmbodimentEvidenceKind::SensorimotorAccuracy)
    }

    /// Current physical-reliability evidence.
    fn physical_reliability_evidence(
        &self,
    ) -> Result<NormalizedScalarEvidenceV1, EvidenceValidationError> {
        NormalizedScalarEvidenceV1::unavailable(EmbodimentEvidenceKind::PhysicalReliability)
    }

    /// Predicted-vs-observed residual evidence, when a real predictor exists.
    fn predictive_residual_evidence(
        &self,
    ) -> Result<NormalizedScalarEvidenceV1, EvidenceValidationError> {
        NormalizedScalarEvidenceV1::unavailable(EmbodimentEvidenceKind::PredictiveResidual)
    }

    /// Temporal/inter-frame representation novelty evidence.
    fn temporal_state_novelty_evidence(
        &self,
    ) -> Result<NormalizedScalarEvidenceV1, EvidenceValidationError> {
        NormalizedScalarEvidenceV1::unavailable(EmbodimentEvidenceKind::TemporalStateNovelty)
    }

    /// Observation-confidence/quality evidence.
    fn observation_confidence_evidence(
        &self,
    ) -> Result<NormalizedScalarEvidenceV1, EvidenceValidationError> {
        NormalizedScalarEvidenceV1::unavailable(EmbodimentEvidenceKind::ObservationConfidence)
    }
}

/// Marker for an embodiment that implements both the legacy control bridge and
/// the explicit v1 evidence-provider contract.
///
/// This gives downstream claim-bearing code a concise bound while the legacy
/// fleet migrates incrementally.
pub trait EvidenceBackedEmbodimentV1: EmbodimentBridge + EmbodimentEvidenceProviderV1 {}

impl<T> EvidenceBackedEmbodimentV1 for T where
    T: EmbodimentBridge + EmbodimentEvidenceProviderV1 + ?Sized
{
}

/// Compatibility adapter that labels legacy health/accuracy/reliability getters
/// as assumptions instead of observations.
///
/// This adapter exists only to preserve numerical continuity while making the
/// proposition explicit. It deliberately does **not** map legacy telemetry
/// `prediction_error` or `observation_confidence` into predictive/quality evidence,
/// because many existing platforms still derive those fields from temporal HDC
/// change rather than a qualified predictor/observation-quality estimator.
pub struct LegacyCompatibilityEvidenceAdapter<'a> {
    bridge: &'a dyn EmbodimentBridge,
    actuator_profile_id: String,
}

impl LegacyCompatibilityEvidenceAdapter<'_> {
    /// Create a compatibility adapter.
    ///
    /// `actuator_profile_id` identifies the positional meaning of the legacy
    /// actuator-health vector. It must be explicit because an unlabelled vector
    /// cannot establish which physical actuator each element describes.
    pub fn new(
        bridge: &dyn EmbodimentBridge,
        actuator_profile_id: impl Into<String>,
    ) -> Result<LegacyCompatibilityEvidenceAdapter<'_>, EvidenceValidationError> {
        let actuator_profile_id = actuator_profile_id.into();
        let trimmed = actuator_profile_id.trim();
        if trimmed.is_empty()
            || trimmed.len() != actuator_profile_id.len()
            || actuator_profile_id.chars().any(char::is_control)
        {
            return Err(EvidenceValidationError::InvalidIdentifier(
                "actuator_profile_id",
            ));
        }
        Ok(LegacyCompatibilityEvidenceAdapter {
            bridge,
            actuator_profile_id,
        })
    }

    fn compatibility_metadata(&self) -> Result<EvidenceMetadataV1, EvidenceValidationError> {
        EvidenceMetadataV1::available(EvidenceSourceClass::CompatibilityAssumption, None)
    }
}

impl EmbodimentEvidenceProviderV1 for LegacyCompatibilityEvidenceAdapter<'_> {
    fn actuator_health_evidence(
        &self,
    ) -> Result<NormalizedVectorEvidenceV1, EvidenceValidationError> {
        let values = self.bridge.actuator_health();
        if values.is_empty() {
            return NormalizedVectorEvidenceV1::unavailable(EmbodimentEvidenceKind::ActuatorHealth);
        }
        let mut metadata = self.compatibility_metadata()?;
        metadata.profile_id = Some(self.actuator_profile_id.clone());
        NormalizedVectorEvidenceV1::new(
            EmbodimentEvidenceKind::ActuatorHealth,
            Some(values),
            metadata,
        )
    }

    fn sensorimotor_accuracy_evidence(
        &self,
    ) -> Result<NormalizedScalarEvidenceV1, EvidenceValidationError> {
        NormalizedScalarEvidenceV1::new(
            EmbodimentEvidenceKind::SensorimotorAccuracy,
            Some(self.bridge.sensorimotor_accuracy()),
            self.compatibility_metadata()?,
        )
    }

    fn physical_reliability_evidence(
        &self,
    ) -> Result<NormalizedScalarEvidenceV1, EvidenceValidationError> {
        NormalizedScalarEvidenceV1::new(
            EmbodimentEvidenceKind::PhysicalReliability,
            Some(self.bridge.physical_reliability() as f32),
            self.compatibility_metadata()?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment::{
        AgentIdentity, EmbodimentPlatform, EmbodimentResult, EmbodimentTelemetry, MotorSafetyLevel,
        GROUNDING_SENSORIMOTOR,
    };
    use crate::hdc::{ContinuousHV, HDC_DIMENSION};

    #[derive(Debug, Clone)]
    struct EmptyEvidenceProvider;

    impl EmbodimentEvidenceProviderV1 for EmptyEvidenceProvider {}

    #[derive(Debug, Clone)]
    struct MockBridge {
        health: Vec<f32>,
        accuracy: f32,
        reliability: f64,
    }

    impl Default for MockBridge {
        fn default() -> Self {
            Self {
                health: vec![1.0, 1.0],
                accuracy: 1.0,
                reliability: 1.0,
            }
        }
    }

    impl EmbodimentBridge for MockBridge {
        fn step(&mut self, _thought_hv: &ContinuousHV, _dt: f32, _phi: f64) -> EmbodimentResult {
            EmbodimentResult {
                num_actuators: self.health.len(),
                control_effort: 0.0,
                success: true,
                prediction_error: 0.0,
                safety_level: MotorSafetyLevel::Red,
                epistemic_grounding: GROUNDING_SENSORIMOTOR,
                observation_confidence: 1.0,
            }
        }

        fn encode_perception(&mut self) -> ContinuousHV {
            ContinuousHV::zero(HDC_DIMENSION)
        }

        fn reset(&mut self) {}

        fn safety_level(&self) -> MotorSafetyLevel {
            MotorSafetyLevel::Red
        }

        fn set_safety_override(&mut self, _level: MotorSafetyLevel) {}

        fn clear_safety_override(&mut self) {}

        fn platform(&self) -> EmbodimentPlatform {
            EmbodimentPlatform::None
        }

        fn num_actuators(&self) -> usize {
            self.health.len()
        }

        fn actuator_health(&self) -> Vec<f32> {
            self.health.clone()
        }

        fn sensorimotor_accuracy(&self) -> f32 {
            self.accuracy
        }

        fn total_steps(&self) -> usize {
            0
        }

        fn telemetry(&self) -> EmbodimentTelemetry {
            EmbodimentTelemetry {
                prediction_error: 0.0,
                observation_confidence: 1.0,
                safety_level: MotorSafetyLevel::Red,
                ..EmbodimentTelemetry::default()
            }
        }

        fn agent_identity(&self) -> AgentIdentity {
            AgentIdentity::new("mock")
        }

        fn physical_reliability(&self) -> f64 {
            self.reliability
        }
    }

    #[test]
    fn empty_provider_defaults_to_unavailable_not_perfect() {
        let provider = EmptyEvidenceProvider;
        for scalar in [
            provider.sensorimotor_accuracy_evidence().unwrap(),
            provider.physical_reliability_evidence().unwrap(),
            provider.predictive_residual_evidence().unwrap(),
            provider.temporal_state_novelty_evidence().unwrap(),
            provider.observation_confidence_evidence().unwrap(),
        ] {
            assert_eq!(scalar.value, None);
            assert_eq!(scalar.metadata.availability, EvidenceAvailability::Unavailable);
            assert_eq!(scalar.metadata.source, None);
        }
        let health = provider.actuator_health_evidence().unwrap();
        assert_eq!(health.values, None);
        assert_eq!(health.metadata.availability, EvidenceAvailability::Unavailable);
    }

    #[test]
    fn legacy_perfect_defaults_are_labelled_compatibility_assumptions() {
        let bridge = MockBridge::default();
        let adapter = LegacyCompatibilityEvidenceAdapter::new(&bridge, "mock.actuators.v1")
            .unwrap();

        let health = adapter.actuator_health_evidence().unwrap();
        assert_eq!(health.values.as_deref(), Some(&[1.0, 1.0][..]));
        assert_eq!(
            health.metadata.source,
            Some(EvidenceSourceClass::CompatibilityAssumption)
        );
        assert_eq!(health.metadata.profile_id.as_deref(), Some("mock.actuators.v1"));

        let accuracy = adapter.sensorimotor_accuracy_evidence().unwrap();
        let reliability = adapter.physical_reliability_evidence().unwrap();
        assert_eq!(accuracy.value, Some(1.0));
        assert_eq!(reliability.value, Some(1.0));
        assert_eq!(
            accuracy.metadata.source,
            Some(EvidenceSourceClass::CompatibilityAssumption)
        );
        assert_eq!(
            reliability.metadata.source,
            Some(EvidenceSourceClass::CompatibilityAssumption)
        );
    }

    #[test]
    fn legacy_prediction_and_confidence_are_not_promoted_to_evidence() {
        let bridge = MockBridge::default();
        let adapter = LegacyCompatibilityEvidenceAdapter::new(&bridge, "mock.actuators.v1")
            .unwrap();

        assert_eq!(
            adapter.predictive_residual_evidence().unwrap().metadata.availability,
            EvidenceAvailability::Unavailable
        );
        assert_eq!(
            adapter.observation_confidence_evidence().unwrap().metadata.availability,
            EvidenceAvailability::Unavailable
        );
    }

    #[test]
    fn malformed_legacy_numbers_fail_validation_instead_of_becoming_evidence() {
        let bridge = MockBridge {
            health: vec![1.2],
            accuracy: f32::NAN,
            reliability: 2.0,
        };
        let adapter = LegacyCompatibilityEvidenceAdapter::new(&bridge, "mock.actuators.v1")
            .unwrap();

        assert!(matches!(
            adapter.actuator_health_evidence(),
            Err(EvidenceValidationError::OutOfRangeValue { .. })
        ));
        assert!(matches!(
            adapter.sensorimotor_accuracy_evidence(),
            Err(EvidenceValidationError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            adapter.physical_reliability_evidence(),
            Err(EvidenceValidationError::OutOfRangeValue { .. })
        ));
    }

    #[test]
    fn empty_legacy_health_is_unavailable_not_an_empty_available_vector() {
        let bridge = MockBridge {
            health: Vec::new(),
            ..MockBridge::default()
        };
        let adapter = LegacyCompatibilityEvidenceAdapter::new(&bridge, "mock.actuators.v1")
            .unwrap();
        let health = adapter.actuator_health_evidence().unwrap();
        assert_eq!(health.metadata.availability, EvidenceAvailability::Unavailable);
        assert_eq!(health.values, None);
    }

    #[test]
    fn compatibility_adapter_requires_named_actuator_profile() {
        let bridge = MockBridge::default();
        assert!(matches!(
            LegacyCompatibilityEvidenceAdapter::new(&bridge, "  "),
            Err(EvidenceValidationError::InvalidIdentifier(
                "actuator_profile_id"
            ))
        ));
    }
}
