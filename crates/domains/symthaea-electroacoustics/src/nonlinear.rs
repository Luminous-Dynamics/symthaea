// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Explicit state-dependent transducer parameter representation.
//!
//! EAC-004 does not claim a complete nonlinear loudspeaker solver. It provides
//! bounded, provenance-carrying curves/tables for quantities such as `Bl(x,i)`,
//! `Le(x,i,f)`, `Cms(x)` / `Kms(x)`, `Rms(v)`, and `Re(T)`.
//!
//! Extrapolation is rejected. Asymmetric source data remains asymmetric; this
//! module never mirrors a curve around zero unless the source itself contains
//! those samples.

use crate::{ParameterSource, PhysicalUnit, ValidationError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateAxisKind {
    Displacement,
    Current,
    Frequency,
    Velocity,
    Temperature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateAxisUnit {
    Meter,
    Ampere,
    Hertz,
    MeterPerSecond,
    Celsius,
}

impl StateAxisKind {
    fn expected_unit(self) -> StateAxisUnit {
        match self {
            Self::Displacement => StateAxisUnit::Meter,
            Self::Current => StateAxisUnit::Ampere,
            Self::Frequency => StateAxisUnit::Hertz,
            Self::Velocity => StateAxisUnit::MeterPerSecond,
            Self::Temperature => StateAxisUnit::Celsius,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateAxis {
    pub kind: StateAxisKind,
    pub unit: StateAxisUnit,
    pub minimum: f64,
    pub maximum: f64,
}

impl StateAxis {
    pub fn new(kind: StateAxisKind, unit: StateAxisUnit, minimum: f64, maximum: f64) -> Self {
        Self {
            kind,
            unit,
            minimum,
            maximum,
        }
    }

    fn validate(&self) -> Result<(), NonlinearModelError> {
        if self.unit != self.kind.expected_unit() {
            return Err(NonlinearModelError::WrongAxisUnit {
                axis: self.kind,
                expected: self.kind.expected_unit(),
                actual: self.unit,
            });
        }
        if !self.minimum.is_finite()
            || !self.maximum.is_finite()
            || self.minimum >= self.maximum
        {
            return Err(NonlinearModelError::InvalidAxisEnvelope {
                axis: self.kind,
                minimum: self.minimum,
                maximum: self.maximum,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateResponseKind {
    ForceFactor,
    Inductance,
    Compliance,
    Stiffness,
    MechanicalResistance,
    VoiceCoilResistance,
}

impl StateResponseKind {
    fn expected_unit(self) -> PhysicalUnit {
        match self {
            Self::ForceFactor => PhysicalUnit::TeslaMeter,
            Self::Inductance => PhysicalUnit::Henry,
            Self::Compliance => PhysicalUnit::MeterPerNewton,
            Self::Stiffness => PhysicalUnit::NewtonPerMeter,
            Self::MechanicalResistance => PhysicalUnit::NewtonSecondPerMeter,
            Self::VoiceCoilResistance => PhysicalUnit::Ohm,
        }
    }

    fn validate_value(self, value: f64) -> Result<(), NonlinearModelError> {
        if !value.is_finite() {
            return Err(NonlinearModelError::NonFiniteResponse(value));
        }
        let valid = match self {
            // Sign is retained because measured motor-force asymmetry and even a
            // zero crossing must not be erased by representation policy.
            Self::ForceFactor => true,
            Self::Inductance | Self::MechanicalResistance => value >= 0.0,
            Self::Compliance | Self::Stiffness | Self::VoiceCoilResistance => value > 0.0,
        };
        if !valid {
            return Err(NonlinearModelError::InvalidResponseValue {
                response: self,
                value,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpolationPolicy {
    /// Only explicitly supplied sample coordinates are admissible.
    ExactOnly,
    /// Piecewise linear interpolation over exactly one state axis.
    Linear1D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtrapolationPolicy {
    /// EAC-004 V1 never invents behavior outside the admitted envelope.
    Reject,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateSample {
    pub coordinates: Vec<f64>,
    pub value: f64,
}

impl StateSample {
    pub fn new(coordinates: Vec<f64>, value: f64) -> Self {
        Self { coordinates, value }
    }
}

/// Generic state-dependent scalar parameter.
///
/// Multi-axis tables are supported as exact samples. V1 interpolation is only
/// admitted for 1D curves so higher-dimensional behavior cannot be fabricated
/// by an implicit interpolation scheme.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateDependentParameter {
    pub id: String,
    pub response: StateResponseKind,
    pub response_unit: PhysicalUnit,
    pub axes: Vec<StateAxis>,
    pub samples: Vec<StateSample>,
    pub source: ParameterSource,
    pub interpolation: InterpolationPolicy,
    pub extrapolation: ExtrapolationPolicy,
}

impl StateDependentParameter {
    pub fn validate(&self) -> Result<(), NonlinearModelError> {
        if self.id.trim().is_empty() {
            return Err(NonlinearModelError::EmptyIdentifier("state_parameter.id"));
        }
        self.source.validate()?;
        if self.response_unit != self.response.expected_unit() {
            return Err(NonlinearModelError::WrongResponseUnit {
                response: self.response,
                expected: self.response.expected_unit(),
                actual: self.response_unit,
            });
        }
        if self.axes.is_empty() {
            return Err(NonlinearModelError::NoAxes);
        }
        if self.samples.is_empty() {
            return Err(NonlinearModelError::NoSamples);
        }
        if self.extrapolation != ExtrapolationPolicy::Reject {
            return Err(NonlinearModelError::UnsupportedExtrapolation);
        }

        let mut axis_kinds = BTreeSet::new();
        for axis in &self.axes {
            axis.validate()?;
            if !axis_kinds.insert(axis.kind) {
                return Err(NonlinearModelError::DuplicateAxis(axis.kind));
            }
        }

        let mut coordinate_keys = BTreeSet::new();
        for sample in &self.samples {
            if sample.coordinates.len() != self.axes.len() {
                return Err(NonlinearModelError::CoordinateDimension {
                    expected: self.axes.len(),
                    actual: sample.coordinates.len(),
                });
            }
            for (axis, coordinate) in self.axes.iter().zip(&sample.coordinates) {
                if !coordinate.is_finite() {
                    return Err(NonlinearModelError::NonFiniteCoordinate {
                        axis: axis.kind,
                        value: *coordinate,
                    });
                }
                if *coordinate < axis.minimum || *coordinate > axis.maximum {
                    return Err(NonlinearModelError::CoordinateOutsideEnvelope {
                        axis: axis.kind,
                        value: *coordinate,
                        minimum: axis.minimum,
                        maximum: axis.maximum,
                    });
                }
            }
            self.response.validate_value(sample.value)?;
            let key: Vec<_> = sample
                .coordinates
                .iter()
                .map(|value| canonical_bits(*value))
                .collect();
            if !coordinate_keys.insert(key) {
                return Err(NonlinearModelError::DuplicateCoordinate);
            }
        }

        if self.interpolation == InterpolationPolicy::Linear1D {
            if self.axes.len() != 1 {
                return Err(NonlinearModelError::LinearInterpolationRequiresOneAxis(
                    self.axes.len(),
                ));
            }
            if self.samples.len() < 2 {
                return Err(NonlinearModelError::LinearInterpolationRequiresTwoSamples);
            }
            for pair in self.samples.windows(2) {
                if pair[0].coordinates[0] >= pair[1].coordinates[0] {
                    return Err(NonlinearModelError::LinearSamplesNotStrictlyIncreasing);
                }
            }
            let axis = &self.axes[0];
            if self.samples.first().unwrap().coordinates[0] != axis.minimum
                || self.samples.last().unwrap().coordinates[0] != axis.maximum
            {
                return Err(NonlinearModelError::LinearSamplesDoNotSpanEnvelope);
            }
        }

        Ok(())
    }

    pub fn evaluate(&self, coordinates: &[f64]) -> Result<StateEvaluation, NonlinearModelError> {
        self.validate()?;
        validate_query_coordinates(&self.axes, coordinates)?;

        match self.interpolation {
            InterpolationPolicy::ExactOnly => {
                let target: Vec<_> = coordinates.iter().map(|v| canonical_bits(*v)).collect();
                let sample = self
                    .samples
                    .iter()
                    .find(|sample| {
                        sample
                            .coordinates
                            .iter()
                            .map(|v| canonical_bits(*v))
                            .eq(target.iter().copied())
                    })
                    .ok_or(NonlinearModelError::NoExactSample)?;
                Ok(StateEvaluation {
                    parameter_id: self.id.clone(),
                    value: sample.value,
                    unit: self.response_unit,
                    disposition: EvaluationDisposition::ExactSample,
                })
            }
            InterpolationPolicy::Linear1D => {
                let x = coordinates[0];
                for (index, sample) in self.samples.iter().enumerate() {
                    if canonical_bits(sample.coordinates[0]) == canonical_bits(x) {
                        return Ok(StateEvaluation {
                            parameter_id: self.id.clone(),
                            value: sample.value,
                            unit: self.response_unit,
                            disposition: EvaluationDisposition::ExactSample,
                        });
                    }
                    if index + 1 < self.samples.len() {
                        let upper = &self.samples[index + 1];
                        let x0 = sample.coordinates[0];
                        let x1 = upper.coordinates[0];
                        if x > x0 && x < x1 {
                            let fraction = (x - x0) / (x1 - x0);
                            let value = sample.value + fraction * (upper.value - sample.value);
                            self.response.validate_value(value)?;
                            return Ok(StateEvaluation {
                                parameter_id: self.id.clone(),
                                value,
                                unit: self.response_unit,
                                disposition: EvaluationDisposition::LinearInterpolated {
                                    lower_sample: index,
                                    upper_sample: index + 1,
                                },
                            });
                        }
                    }
                }
                Err(NonlinearModelError::NoInterpolationBracket)
            }
        }
    }
}

fn validate_query_coordinates(
    axes: &[StateAxis],
    coordinates: &[f64],
) -> Result<(), NonlinearModelError> {
    if coordinates.len() != axes.len() {
        return Err(NonlinearModelError::CoordinateDimension {
            expected: axes.len(),
            actual: coordinates.len(),
        });
    }
    for (axis, value) in axes.iter().zip(coordinates) {
        if !value.is_finite() {
            return Err(NonlinearModelError::NonFiniteCoordinate {
                axis: axis.kind,
                value: *value,
            });
        }
        if *value < axis.minimum || *value > axis.maximum {
            return Err(NonlinearModelError::ExtrapolationRejected {
                axis: axis.kind,
                value: *value,
                minimum: axis.minimum,
                maximum: axis.maximum,
            });
        }
    }
    Ok(())
}

fn canonical_bits(value: f64) -> u64 {
    if value == 0.0 { 0 } else { value.to_bits() }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateEvaluation {
    pub parameter_id: String,
    pub value: f64,
    pub unit: PhysicalUnit,
    pub disposition: EvaluationDisposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationDisposition {
    ExactSample,
    LinearInterpolated {
        lower_sample: usize,
        upper_sample: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NonlinearSuspensionModel {
    Compliance(StateDependentParameter),
    Stiffness(StateDependentParameter),
}

/// EAC-004 nonlinear/state-dependent additions for one EAC-001 transducer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearTransducerState {
    pub transducer_id: String,
    pub force_factor: Option<StateDependentParameter>,
    pub inductance: Option<StateDependentParameter>,
    pub suspension: Option<NonlinearSuspensionModel>,
    pub mechanical_resistance: Option<StateDependentParameter>,
    pub voice_coil_resistance: Option<StateDependentParameter>,
}

impl NonlinearTransducerState {
    pub fn validate(&self) -> Result<(), NonlinearModelError> {
        if self.transducer_id.trim().is_empty() {
            return Err(NonlinearModelError::EmptyIdentifier(
                "nonlinear_transducer.transducer_id",
            ));
        }
        if let Some(parameter) = &self.force_factor {
            validate_field(
                parameter,
                StateResponseKind::ForceFactor,
                &[StateAxisKind::Displacement, StateAxisKind::Current],
            )?;
            require_axis(parameter, StateAxisKind::Displacement)?;
        }
        if let Some(parameter) = &self.inductance {
            validate_field(
                parameter,
                StateResponseKind::Inductance,
                &[
                    StateAxisKind::Displacement,
                    StateAxisKind::Current,
                    StateAxisKind::Frequency,
                ],
            )?;
        }
        if let Some(suspension) = &self.suspension {
            let parameter = match suspension {
                NonlinearSuspensionModel::Compliance(parameter) => {
                    if parameter.response != StateResponseKind::Compliance {
                        return Err(NonlinearModelError::UnexpectedResponseKind {
                            expected: StateResponseKind::Compliance,
                            actual: parameter.response,
                        });
                    }
                    parameter
                }
                NonlinearSuspensionModel::Stiffness(parameter) => {
                    if parameter.response != StateResponseKind::Stiffness {
                        return Err(NonlinearModelError::UnexpectedResponseKind {
                            expected: StateResponseKind::Stiffness,
                            actual: parameter.response,
                        });
                    }
                    parameter
                }
            };
            parameter.validate()?;
            require_exact_axes(parameter, &[StateAxisKind::Displacement])?;
        }
        if let Some(parameter) = &self.mechanical_resistance {
            validate_field(
                parameter,
                StateResponseKind::MechanicalResistance,
                &[StateAxisKind::Velocity],
            )?;
            require_exact_axes(parameter, &[StateAxisKind::Velocity])?;
        }
        if let Some(parameter) = &self.voice_coil_resistance {
            validate_field(
                parameter,
                StateResponseKind::VoiceCoilResistance,
                &[StateAxisKind::Temperature],
            )?;
            require_exact_axes(parameter, &[StateAxisKind::Temperature])?;
        }
        Ok(())
    }
}

fn validate_field(
    parameter: &StateDependentParameter,
    expected: StateResponseKind,
    allowed_axes: &[StateAxisKind],
) -> Result<(), NonlinearModelError> {
    parameter.validate()?;
    if parameter.response != expected {
        return Err(NonlinearModelError::UnexpectedResponseKind {
            expected,
            actual: parameter.response,
        });
    }
    for axis in &parameter.axes {
        if !allowed_axes.contains(&axis.kind) {
            return Err(NonlinearModelError::AxisNotAllowed {
                response: expected,
                axis: axis.kind,
            });
        }
    }
    Ok(())
}

fn require_axis(
    parameter: &StateDependentParameter,
    required: StateAxisKind,
) -> Result<(), NonlinearModelError> {
    if parameter.axes.iter().any(|axis| axis.kind == required) {
        Ok(())
    } else {
        Err(NonlinearModelError::RequiredAxisMissing {
            response: parameter.response,
            axis: required,
        })
    }
}

fn require_exact_axes(
    parameter: &StateDependentParameter,
    required: &[StateAxisKind],
) -> Result<(), NonlinearModelError> {
    let actual: Vec<_> = parameter.axes.iter().map(|axis| axis.kind).collect();
    if actual == required {
        Ok(())
    } else {
        Err(NonlinearModelError::ExactAxesRequired {
            response: parameter.response,
            required: required.to_vec(),
            actual,
        })
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum NonlinearModelError {
    #[error(transparent)]
    ParameterSource(#[from] ValidationError),
    #[error("identifier {0} cannot be empty")]
    EmptyIdentifier(&'static str),
    #[error("axis {axis:?} has wrong unit: expected {expected:?}, got {actual:?}")]
    WrongAxisUnit {
        axis: StateAxisKind,
        expected: StateAxisUnit,
        actual: StateAxisUnit,
    },
    #[error("axis {axis:?} has invalid envelope [{minimum}, {maximum}]")]
    InvalidAxisEnvelope {
        axis: StateAxisKind,
        minimum: f64,
        maximum: f64,
    },
    #[error("duplicate state axis {0:?}")]
    DuplicateAxis(StateAxisKind),
    #[error("state-dependent parameter requires at least one axis")]
    NoAxes,
    #[error("state-dependent parameter requires at least one sample")]
    NoSamples,
    #[error("response {response:?} has wrong unit: expected {expected:?}, got {actual:?}")]
    WrongResponseUnit {
        response: StateResponseKind,
        expected: PhysicalUnit,
        actual: PhysicalUnit,
    },
    #[error("sample coordinate dimension mismatch: expected {expected}, got {actual}")]
    CoordinateDimension { expected: usize, actual: usize },
    #[error("coordinate on {axis:?} is non-finite: {value}")]
    NonFiniteCoordinate { axis: StateAxisKind, value: f64 },
    #[error("coordinate {value} on {axis:?} lies outside [{minimum}, {maximum}]")]
    CoordinateOutsideEnvelope {
        axis: StateAxisKind,
        value: f64,
        minimum: f64,
        maximum: f64,
    },
    #[error("duplicate state coordinate")]
    DuplicateCoordinate,
    #[error("response value is non-finite: {0}")]
    NonFiniteResponse(f64),
    #[error("response {response:?} rejects value {value}")]
    InvalidResponseValue {
        response: StateResponseKind,
        value: f64,
    },
    #[error("EAC-004 V1 supports only reject-extrapolation")]
    UnsupportedExtrapolation,
    #[error("linear interpolation requires exactly one axis, got {0}")]
    LinearInterpolationRequiresOneAxis(usize),
    #[error("linear interpolation requires at least two samples")]
    LinearInterpolationRequiresTwoSamples,
    #[error("linear interpolation samples must be strictly increasing")]
    LinearSamplesNotStrictlyIncreasing,
    #[error("linear interpolation samples must span the declared axis envelope")]
    LinearSamplesDoNotSpanEnvelope,
    #[error("no exact sample exists at the requested coordinates")]
    NoExactSample,
    #[error("no interpolation bracket exists inside the admitted envelope")]
    NoInterpolationBracket,
    #[error("extrapolation rejected for {axis:?}: {value} outside [{minimum}, {maximum}]")]
    ExtrapolationRejected {
        axis: StateAxisKind,
        value: f64,
        minimum: f64,
        maximum: f64,
    },
    #[error("expected response {expected:?}, got {actual:?}")]
    UnexpectedResponseKind {
        expected: StateResponseKind,
        actual: StateResponseKind,
    },
    #[error("axis {axis:?} is not allowed for response {response:?}")]
    AxisNotAllowed {
        response: StateResponseKind,
        axis: StateAxisKind,
    },
    #[error("required axis {axis:?} is missing for response {response:?}")]
    RequiredAxisMissing {
        response: StateResponseKind,
        axis: StateAxisKind,
    },
    #[error("response {response:?} requires axes {required:?}, got {actual:?}")]
    ExactAxesRequired {
        response: StateResponseKind,
        required: Vec<StateAxisKind>,
        actual: Vec<StateAxisKind>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParameterSourceKind;

    fn source(name: &str) -> ParameterSource {
        ParameterSource::new(ParameterSourceKind::Datasheet, name)
    }

    fn bl_curve() -> StateDependentParameter {
        StateDependentParameter {
            id: "bl-x-asymmetric".into(),
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
            source: source("asymmetric fixture"),
            interpolation: InterpolationPolicy::Linear1D,
            extrapolation: ExtrapolationPolicy::Reject,
        }
    }

    #[test]
    fn asymmetric_force_factor_is_not_mirrored() {
        let curve = bl_curve();
        let negative = curve.evaluate(&[-0.005]).unwrap().value;
        let positive = curve.evaluate(&[0.005]).unwrap().value;
        assert!((negative - 4.25).abs() < 1e-12);
        assert!((positive - 4.5).abs() < 1e-12);
        assert_ne!(negative, positive);
    }

    #[test]
    fn extrapolation_is_rejected() {
        assert!(matches!(
            bl_curve().evaluate(&[0.02]),
            Err(NonlinearModelError::ExtrapolationRejected { .. })
        ));
    }

    #[test]
    fn duplicate_coordinates_fail_closed() {
        let mut curve = bl_curve();
        curve.samples.push(StateSample::new(vec![0.0], 4.9));
        assert_eq!(curve.validate(), Err(NonlinearModelError::DuplicateCoordinate));
    }

    #[test]
    fn wrong_axis_unit_is_rejected() {
        let mut curve = bl_curve();
        curve.axes[0].unit = StateAxisUnit::Ampere;
        assert!(matches!(
            curve.validate(),
            Err(NonlinearModelError::WrongAxisUnit { .. })
        ));
    }

    #[test]
    fn wrong_response_unit_is_rejected() {
        let mut curve = bl_curve();
        curve.response_unit = PhysicalUnit::Henry;
        assert!(matches!(
            curve.validate(),
            Err(NonlinearModelError::WrongResponseUnit { .. })
        ));
    }

    #[test]
    fn exact_two_dimensional_table_requires_exact_sample() {
        let table = StateDependentParameter {
            id: "le-x-i".into(),
            response: StateResponseKind::Inductance,
            response_unit: PhysicalUnit::Henry,
            axes: vec![
                StateAxis::new(
                    StateAxisKind::Displacement,
                    StateAxisUnit::Meter,
                    -0.01,
                    0.01,
                ),
                StateAxis::new(StateAxisKind::Current, StateAxisUnit::Ampere, 0.0, 5.0),
            ],
            samples: vec![
                StateSample::new(vec![-0.01, 0.0], 0.0008),
                StateSample::new(vec![0.0, 2.5], 0.0006),
                StateSample::new(vec![0.01, 5.0], 0.0004),
            ],
            source: source("le fixture"),
            interpolation: InterpolationPolicy::ExactOnly,
            extrapolation: ExtrapolationPolicy::Reject,
        };
        assert!(table.evaluate(&[0.0, 2.5]).is_ok());
        assert_eq!(
            table.evaluate(&[0.0, 2.0]),
            Err(NonlinearModelError::NoExactSample)
        );
    }

    #[test]
    fn linear_interpolation_rejects_multiple_axes() {
        let mut table = bl_curve();
        table.axes.push(StateAxis::new(
            StateAxisKind::Current,
            StateAxisUnit::Ampere,
            0.0,
            5.0,
        ));
        for sample in &mut table.samples {
            sample.coordinates.push(0.0);
        }
        assert!(matches!(
            table.validate(),
            Err(NonlinearModelError::LinearInterpolationRequiresOneAxis(2))
        ));
    }

    #[test]
    fn measured_source_requires_evidence_reference() {
        let mut curve = bl_curve();
        curve.source = ParameterSource::new(ParameterSourceKind::Measured, "bench measurement");
        assert!(matches!(
            curve.validate(),
            Err(NonlinearModelError::ParameterSource(
                ValidationError::MissingEvidenceReference(ParameterSourceKind::Measured)
            ))
        ));
    }

    #[test]
    fn nonlinear_transducer_enforces_semantic_axes() {
        let mut resistance = bl_curve();
        resistance.id = "re-t".into();
        resistance.response = StateResponseKind::VoiceCoilResistance;
        resistance.response_unit = PhysicalUnit::Ohm;
        resistance.axes = vec![StateAxis::new(
            StateAxisKind::Temperature,
            StateAxisUnit::Celsius,
            20.0,
            120.0,
        )];
        resistance.samples = vec![
            StateSample::new(vec![20.0], 6.0),
            StateSample::new(vec![120.0], 8.4),
        ];

        let state = NonlinearTransducerState {
            transducer_id: "driver-1".into(),
            force_factor: Some(bl_curve()),
            inductance: None,
            suspension: None,
            mechanical_resistance: None,
            voice_coil_resistance: Some(resistance),
        };
        assert!(state.validate().is_ok());
    }
}
