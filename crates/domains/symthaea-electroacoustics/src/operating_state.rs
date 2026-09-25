// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical operating-state coordinates for nonlinear electro-acoustic models.
//!
//! EAC-004 surfaces use typed axes, but callers should not independently invent
//! coordinate ordering or silently omit thermal/current/velocity state. This
//! module provides one explicit physical operating point and maps it into each
//! parameter's declared axis order.

use crate::nonlinear::{
    NonlinearModelError, StateAxisKind, StateDependentParameter, StateEvaluation,
};
use crate::nonlinear_evidence::{
    EvidenceBoundStateEvaluation, EvidenceBoundStateParameter, NonlinearEvidenceError,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Physical operating point used to query state-dependent transducer parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TransducerOperatingState {
    pub displacement_m: f64,
    pub current_a: f64,
    /// Optional because many state surfaces do not depend on frequency. A
    /// frequency-dependent surface fails closed if this is absent.
    pub frequency_hz: Option<f64>,
    pub velocity_m_per_s: f64,
    pub coil_temperature_c: f64,
}

impl TransducerOperatingState {
    pub fn new(
        displacement_m: f64,
        current_a: f64,
        velocity_m_per_s: f64,
        coil_temperature_c: f64,
    ) -> Self {
        Self {
            displacement_m,
            current_a,
            frequency_hz: None,
            velocity_m_per_s,
            coil_temperature_c,
        }
    }

    pub fn with_frequency(mut self, frequency_hz: f64) -> Self {
        self.frequency_hz = Some(frequency_hz);
        self
    }

    pub fn validate(&self) -> Result<(), OperatingStateError> {
        for (field, value) in [
            ("displacement_m", self.displacement_m),
            ("current_a", self.current_a),
            ("velocity_m_per_s", self.velocity_m_per_s),
            ("coil_temperature_c", self.coil_temperature_c),
        ] {
            if !value.is_finite() {
                return Err(OperatingStateError::NonFiniteState { field, value });
            }
        }
        if let Some(frequency_hz) = self.frequency_hz {
            if !frequency_hz.is_finite() || frequency_hz < 0.0 {
                return Err(OperatingStateError::InvalidFrequency(frequency_hz));
            }
        }
        Ok(())
    }

    pub fn coordinate(&self, axis: StateAxisKind) -> Result<f64, OperatingStateError> {
        self.validate()?;
        match axis {
            StateAxisKind::Displacement => Ok(self.displacement_m),
            StateAxisKind::Current => Ok(self.current_a),
            StateAxisKind::Frequency => self
                .frequency_hz
                .ok_or(OperatingStateError::MissingCoordinate(StateAxisKind::Frequency)),
            StateAxisKind::Velocity => Ok(self.velocity_m_per_s),
            StateAxisKind::Temperature => Ok(self.coil_temperature_c),
        }
    }

    /// Map the operating point into the exact axis order declared by a nonlinear
    /// parameter. Axis order therefore remains model data, not caller convention.
    pub fn coordinates_for(
        &self,
        parameter: &StateDependentParameter,
    ) -> Result<Vec<f64>, OperatingStateError> {
        self.validate()?;
        parameter.validate()?;
        parameter
            .axes
            .iter()
            .map(|axis| self.coordinate(axis.kind))
            .collect()
    }

    pub fn evaluate(
        &self,
        parameter: &StateDependentParameter,
    ) -> Result<StateEvaluation, OperatingStateError> {
        let coordinates = self.coordinates_for(parameter)?;
        Ok(parameter.evaluate(&coordinates)?)
    }

    pub fn evaluate_with_evidence(
        &self,
        parameter: &EvidenceBoundStateParameter,
    ) -> Result<EvidenceBoundStateEvaluation, OperatingStateError> {
        parameter.validate()?;
        let coordinates = self.coordinates_for(&parameter.parameter)?;
        Ok(parameter.evaluate(&coordinates)?)
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum OperatingStateError {
    #[error("operating-state field {field} is non-finite: {value}")]
    NonFiniteState { field: &'static str, value: f64 },
    #[error("operating-state frequency must be finite and nonnegative, got {0}")]
    InvalidFrequency(f64),
    #[error("operating state does not provide required coordinate {0:?}")]
    MissingCoordinate(StateAxisKind),
    #[error(transparent)]
    Model(#[from] NonlinearModelError),
    #[error(transparent)]
    Evidence(#[from] NonlinearEvidenceError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear::{
        ExtrapolationPolicy, InterpolationPolicy, StateAxis, StateAxisUnit, StateResponseKind,
        StateSample,
    };
    use crate::nonlinear_evidence::{
        EvaluationAuthority, StateSampleEvidence, UncertaintyBasis,
    };
    use crate::{ParameterSource, ParameterSourceKind, PhysicalUnit, UncertaintyInterval};

    fn source(name: &str) -> ParameterSource {
        ParameterSource::new(ParameterSourceKind::Datasheet, name)
    }

    fn bl_curve() -> StateDependentParameter {
        StateDependentParameter {
            id: "bl-x".into(),
            response: StateResponseKind::ForceFactor,
            response_unit: PhysicalUnit::TeslaMeter,
            axes: vec![StateAxis::new(
                StateAxisKind::Displacement,
                StateAxisUnit::Meter,
                -0.01,
                0.01,
            )],
            samples: vec![
                StateSample::new(vec![-0.01], 3.5),
                StateSample::new(vec![0.0], 5.0),
                StateSample::new(vec![0.01], 4.0),
            ],
            source: source("bl dataset"),
            interpolation: InterpolationPolicy::Linear1D,
            extrapolation: ExtrapolationPolicy::Reject,
        }
    }

    #[test]
    fn operating_state_queries_parameter_axis_order() {
        let state = TransducerOperatingState::new(0.005, 1.0, 0.2, 35.0);
        let result = state.evaluate(&bl_curve()).unwrap();
        assert!((result.value - 4.5).abs() < 1e-12);
    }

    #[test]
    fn frequency_dependent_surface_requires_frequency_coordinate() {
        let parameter = StateDependentParameter {
            id: "le-f".into(),
            response: StateResponseKind::Inductance,
            response_unit: PhysicalUnit::Henry,
            axes: vec![StateAxis::new(
                StateAxisKind::Frequency,
                StateAxisUnit::Hertz,
                20.0,
                20_000.0,
            )],
            samples: vec![
                StateSample::new(vec![20.0], 0.001),
                StateSample::new(vec![20_000.0], 0.0003),
            ],
            source: source("Le(f)"),
            interpolation: InterpolationPolicy::Linear1D,
            extrapolation: ExtrapolationPolicy::Reject,
        };
        let state = TransducerOperatingState::new(0.0, 0.0, 0.0, 20.0);
        assert_eq!(
            state.evaluate(&parameter),
            Err(OperatingStateError::MissingCoordinate(StateAxisKind::Frequency))
        );
    }

    #[test]
    fn operating_state_does_not_bypass_parameter_envelope() {
        let state = TransducerOperatingState::new(0.02, 0.0, 0.0, 20.0);
        assert!(matches!(
            state.evaluate(&bl_curve()),
            Err(OperatingStateError::Model(
                NonlinearModelError::ExtrapolationRejected { .. }
            ))
        ));
    }

    #[test]
    fn evidence_bound_evaluation_preserves_derived_authority() {
        let parameter = bl_curve();
        let bound = EvidenceBoundStateParameter {
            parameter,
            sample_evidence: vec![
                StateSampleEvidence::new(0, source("neg"))
                    .with_uncertainty(UncertaintyInterval::new(3.4, 3.6).unwrap()),
                StateSampleEvidence::new(1, source("zero"))
                    .with_uncertainty(UncertaintyInterval::new(4.9, 5.1).unwrap()),
                StateSampleEvidence::new(2, source("pos"))
                    .with_uncertainty(UncertaintyInterval::new(3.9, 4.1).unwrap()),
            ],
        };
        let state = TransducerOperatingState::new(0.005, 0.0, 0.0, 20.0);
        let result = state.evaluate_with_evidence(&bound).unwrap();
        assert_eq!(result.authority, EvaluationAuthority::DerivedInterpolation);
        assert_eq!(result.uncertainty_basis, UncertaintyBasis::ConservativeEndpointHull);
    }

    #[test]
    fn non_finite_operating_state_fails_closed() {
        let state = TransducerOperatingState::new(f64::NAN, 0.0, 0.0, 20.0);
        assert!(matches!(
            state.validate(),
            Err(OperatingStateError::NonFiniteState {
                field: "displacement_m",
                ..
            })
        ));
    }
}
