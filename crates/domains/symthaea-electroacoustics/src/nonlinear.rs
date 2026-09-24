// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Explicit large-signal/nonlinear transducer field representation.
//!
//! EAC-004 is representation-first. It does not claim a complete nonlinear
//! loudspeaker simulator. The purpose of this module is to preserve measured,
//! fitted, or numerically-derived state dependence without collapsing it into
//! one nominal small-signal scalar.
//!
//! First-tranche interpolation is multilinear *inside* the frozen sampled
//! envelope. Extrapolation is rejected.

use crate::{ParameterSource, ValidationError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FieldAxisKind {
    DisplacementMeter,
    CurrentAmpere,
    FrequencyHertz,
    VelocityMeterPerSecond,
    TemperatureCelsius,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FieldOutputKind {
    ForceFactorTeslaMeter,
    InductanceHenry,
    ComplianceMeterPerNewton,
    StiffnessNewtonPerMeter,
    MechanicalResistanceNewtonSecondPerMeter,
    ResistanceOhm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationPolicy {
    Multilinear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtrapolationPolicy {
    Reject,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampledAxis {
    pub kind: FieldAxisKind,
    pub points: Vec<f64>,
}

impl SampledAxis {
    fn validate(&self) -> Result<(), NonlinearModelError> {
        if self.points.len() < 2 {
            return Err(NonlinearModelError::AxisNeedsAtLeastTwoPoints(self.kind));
        }
        if self.points.iter().any(|value| !value.is_finite()) {
            return Err(NonlinearModelError::NonFiniteAxisPoint(self.kind));
        }
        if self.points.windows(2).any(|window| window[0] >= window[1]) {
            return Err(NonlinearModelError::AxisNotStrictlyIncreasing(self.kind));
        }
        Ok(())
    }
}

/// Flattened sampled scalar field with one to three axes. Values are row-major
/// with the final axis varying fastest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampledScalarField {
    pub id: String,
    pub axes: Vec<SampledAxis>,
    pub values: Vec<f64>,
    pub output: FieldOutputKind,
    pub source: ParameterSource,
    pub interpolation: InterpolationPolicy,
    pub extrapolation: ExtrapolationPolicy,
}

impl SampledScalarField {
    pub fn validate(&self) -> Result<(), NonlinearModelError> {
        if self.id.trim().is_empty() {
            return Err(NonlinearModelError::EmptyIdentifier("field.id"));
        }
        if !(1..=3).contains(&self.axes.len()) {
            return Err(NonlinearModelError::UnsupportedAxisCount(self.axes.len()));
        }
        let mut seen = BTreeSet::new();
        let mut expected_values = 1usize;
        for axis in &self.axes {
            axis.validate()?;
            if !seen.insert(axis.kind) {
                return Err(NonlinearModelError::DuplicateAxis(axis.kind));
            }
            expected_values = expected_values
                .checked_mul(axis.points.len())
                .ok_or(NonlinearModelError::FieldSizeOverflow)?;
        }
        if self.values.len() != expected_values {
            return Err(NonlinearModelError::WrongValueCount {
                expected: expected_values,
                actual: self.values.len(),
            });
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(NonlinearModelError::NonFiniteFieldValue);
        }
        self.source.validate()?;
        Ok(())
    }

    /// Evaluate inside the sampled envelope with multilinear interpolation.
    /// Extrapolation is rejected rather than silently extending the model.
    pub fn evaluate(&self, coordinates: &[f64]) -> Result<f64, NonlinearModelError> {
        self.validate()?;
        if coordinates.len() != self.axes.len() {
            return Err(NonlinearModelError::WrongCoordinateCount {
                expected: self.axes.len(),
                actual: coordinates.len(),
            });
        }

        let mut brackets = Vec::with_capacity(self.axes.len());
        for (axis, coordinate) in self.axes.iter().zip(coordinates.iter().copied()) {
            if !coordinate.is_finite() {
                return Err(NonlinearModelError::NonFiniteCoordinate(axis.kind));
            }
            brackets.push(bracket(axis, coordinate)?);
        }

        let strides = row_major_strides(&self.axes)?;
        let mut result = 0.0;
        let corner_count = 1usize << self.axes.len();
        for corner in 0..corner_count {
            let mut index = 0usize;
            let mut weight = 1.0;
            for dimension in 0..self.axes.len() {
                let (lower_index, t) = brackets[dimension];
                let use_upper = (corner >> dimension) & 1 == 1;
                let point_index = lower_index + usize::from(use_upper);
                index += point_index * strides[dimension];
                weight *= if use_upper { t } else { 1.0 - t };
            }
            result += weight * self.values[index];
        }
        if !result.is_finite() {
            return Err(NonlinearModelError::NonFiniteInterpolatedValue);
        }
        Ok(result)
    }

    fn validate_profile(
        &self,
        expected_axes: &[FieldAxisKind],
        expected_output: FieldOutputKind,
        value_constraint: ValueConstraint,
    ) -> Result<(), NonlinearModelError> {
        self.validate()?;
        let actual_axes: Vec<_> = self.axes.iter().map(|axis| axis.kind).collect();
        if actual_axes != expected_axes {
            return Err(NonlinearModelError::WrongFieldAxes {
                expected: expected_axes.to_vec(),
                actual: actual_axes,
            });
        }
        if self.output != expected_output {
            return Err(NonlinearModelError::WrongFieldOutput {
                expected: expected_output,
                actual: self.output,
            });
        }
        for value in &self.values {
            match value_constraint {
                ValueConstraint::AnyFinite => {}
                ValueConstraint::Nonnegative if *value < 0.0 => {
                    return Err(NonlinearModelError::FieldValueViolatesConstraint {
                        output: self.output,
                        value: *value,
                    });
                }
                ValueConstraint::Positive if *value <= 0.0 => {
                    return Err(NonlinearModelError::FieldValueViolatesConstraint {
                        output: self.output,
                        value: *value,
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum ValueConstraint {
    AnyFinite,
    Nonnegative,
    Positive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NonlinearSuspensionField {
    ComplianceVsDisplacement(SampledScalarField),
    StiffnessVsDisplacement(SampledScalarField),
}

impl NonlinearSuspensionField {
    fn validate(&self) -> Result<(), NonlinearModelError> {
        match self {
            Self::ComplianceVsDisplacement(field) => field.validate_profile(
                &[FieldAxisKind::DisplacementMeter],
                FieldOutputKind::ComplianceMeterPerNewton,
                ValueConstraint::Positive,
            ),
            Self::StiffnessVsDisplacement(field) => field.validate_profile(
                &[FieldAxisKind::DisplacementMeter],
                FieldOutputKind::StiffnessNewtonPerMeter,
                ValueConstraint::Positive,
            ),
        }
    }

    pub fn compliance_m_per_n(&self, displacement_m: f64) -> Result<f64, NonlinearModelError> {
        match self {
            Self::ComplianceVsDisplacement(field) => field.evaluate(&[displacement_m]),
            Self::StiffnessVsDisplacement(field) => {
                let stiffness = field.evaluate(&[displacement_m])?;
                if stiffness <= 0.0 {
                    return Err(NonlinearModelError::CannotInvertNonpositiveStiffness(stiffness));
                }
                Ok(1.0 / stiffness)
            }
        }
    }

    pub fn stiffness_n_per_m(&self, displacement_m: f64) -> Result<f64, NonlinearModelError> {
        match self {
            Self::ComplianceVsDisplacement(field) => {
                let compliance = field.evaluate(&[displacement_m])?;
                if compliance <= 0.0 {
                    return Err(NonlinearModelError::CannotInvertNonpositiveCompliance(compliance));
                }
                Ok(1.0 / compliance)
            }
            Self::StiffnessVsDisplacement(field) => field.evaluate(&[displacement_m]),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct NonlinearTransducerFields {
    /// `Bl(x, i)`.
    pub force_factor_bl_x_i: Option<SampledScalarField>,
    /// `Le(x, i, f)`.
    pub inductance_le_x_i_f: Option<SampledScalarField>,
    /// `Cms(x)` or `Kms(x)` as one canonical source family.
    pub suspension: Option<NonlinearSuspensionField>,
    /// `Rms(v)`.
    pub mechanical_resistance_rms_v: Option<SampledScalarField>,
    /// `Re(T)`.
    pub voice_coil_resistance_re_t: Option<SampledScalarField>,
}

impl NonlinearTransducerFields {
    pub fn validate(&self) -> Result<(), NonlinearModelError> {
        if let Some(field) = &self.force_factor_bl_x_i {
            field.validate_profile(
                &[
                    FieldAxisKind::DisplacementMeter,
                    FieldAxisKind::CurrentAmpere,
                ],
                FieldOutputKind::ForceFactorTeslaMeter,
                ValueConstraint::AnyFinite,
            )?;
        }
        if let Some(field) = &self.inductance_le_x_i_f {
            field.validate_profile(
                &[
                    FieldAxisKind::DisplacementMeter,
                    FieldAxisKind::CurrentAmpere,
                    FieldAxisKind::FrequencyHertz,
                ],
                FieldOutputKind::InductanceHenry,
                ValueConstraint::Nonnegative,
            )?;
        }
        if let Some(field) = &self.suspension {
            field.validate()?;
        }
        if let Some(field) = &self.mechanical_resistance_rms_v {
            field.validate_profile(
                &[FieldAxisKind::VelocityMeterPerSecond],
                FieldOutputKind::MechanicalResistanceNewtonSecondPerMeter,
                ValueConstraint::Nonnegative,
            )?;
        }
        if let Some(field) = &self.voice_coil_resistance_re_t {
            field.validate_profile(
                &[FieldAxisKind::TemperatureCelsius],
                FieldOutputKind::ResistanceOhm,
                ValueConstraint::Positive,
            )?;
        }
        Ok(())
    }

    pub fn force_factor_tesla_meter(
        &self,
        displacement_m: f64,
        current_a: f64,
    ) -> Result<Option<f64>, NonlinearModelError> {
        self.force_factor_bl_x_i
            .as_ref()
            .map(|field| field.evaluate(&[displacement_m, current_a]))
            .transpose()
    }

    pub fn inductance_henry(
        &self,
        displacement_m: f64,
        current_a: f64,
        frequency_hz: f64,
    ) -> Result<Option<f64>, NonlinearModelError> {
        self.inductance_le_x_i_f
            .as_ref()
            .map(|field| field.evaluate(&[displacement_m, current_a, frequency_hz]))
            .transpose()
    }

    pub fn mechanical_resistance_n_s_per_m(
        &self,
        velocity_m_per_s: f64,
    ) -> Result<Option<f64>, NonlinearModelError> {
        self.mechanical_resistance_rms_v
            .as_ref()
            .map(|field| field.evaluate(&[velocity_m_per_s]))
            .transpose()
    }

    pub fn voice_coil_resistance_ohm(
        &self,
        temperature_c: f64,
    ) -> Result<Option<f64>, NonlinearModelError> {
        self.voice_coil_resistance_re_t
            .as_ref()
            .map(|field| field.evaluate(&[temperature_c]))
            .transpose()
    }
}

fn bracket(axis: &SampledAxis, coordinate: f64) -> Result<(usize, f64), NonlinearModelError> {
    let first = axis.points[0];
    let last = axis.points[axis.points.len() - 1];
    if coordinate < first || coordinate > last {
        return Err(NonlinearModelError::OutsideSampledEnvelope {
            axis: axis.kind,
            coordinate,
            lower: first,
            upper: last,
        });
    }
    if coordinate == last {
        return Ok((axis.points.len() - 2, 1.0));
    }
    for (index, pair) in axis.points.windows(2).enumerate() {
        if coordinate >= pair[0] && coordinate <= pair[1] {
            let t = (coordinate - pair[0]) / (pair[1] - pair[0]);
            return Ok((index, t));
        }
    }
    Err(NonlinearModelError::OutsideSampledEnvelope {
        axis: axis.kind,
        coordinate,
        lower: first,
        upper: last,
    })
}

fn row_major_strides(axes: &[SampledAxis]) -> Result<Vec<usize>, NonlinearModelError> {
    let mut strides = vec![1usize; axes.len()];
    for index in (0..axes.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1]
            .checked_mul(axes[index + 1].points.len())
            .ok_or(NonlinearModelError::FieldSizeOverflow)?;
    }
    Ok(strides)
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum NonlinearModelError {
    #[error(transparent)]
    InvalidParameterSource(#[from] ValidationError),
    #[error("identifier {0} cannot be empty")]
    EmptyIdentifier(&'static str),
    #[error("nonlinear sampled field supports one to three axes, got {0}")]
    UnsupportedAxisCount(usize),
    #[error("axis {0:?} requires at least two sampled points")]
    AxisNeedsAtLeastTwoPoints(FieldAxisKind),
    #[error("axis {0:?} contains a non-finite point")]
    NonFiniteAxisPoint(FieldAxisKind),
    #[error("axis {0:?} must be strictly increasing")]
    AxisNotStrictlyIncreasing(FieldAxisKind),
    #[error("axis {0:?} occurs more than once")]
    DuplicateAxis(FieldAxisKind),
    #[error("sampled-field size overflow")]
    FieldSizeOverflow,
    #[error("sampled field has {actual} values, expected {expected}")]
    WrongValueCount { expected: usize, actual: usize },
    #[error("sampled field contains a non-finite value")]
    NonFiniteFieldValue,
    #[error("coordinate count {actual} does not match field dimension {expected}")]
    WrongCoordinateCount { expected: usize, actual: usize },
    #[error("coordinate on axis {0:?} must be finite")]
    NonFiniteCoordinate(FieldAxisKind),
    #[error("coordinate {coordinate} on {axis:?} is outside sampled envelope [{lower}, {upper}]")]
    OutsideSampledEnvelope {
        axis: FieldAxisKind,
        coordinate: f64,
        lower: f64,
        upper: f64,
    },
    #[error("interpolated field value became non-finite")]
    NonFiniteInterpolatedValue,
    #[error("wrong nonlinear field axes: expected {expected:?}, got {actual:?}")]
    WrongFieldAxes {
        expected: Vec<FieldAxisKind>,
        actual: Vec<FieldAxisKind>,
    },
    #[error("wrong nonlinear field output: expected {expected:?}, got {actual:?}")]
    WrongFieldOutput {
        expected: FieldOutputKind,
        actual: FieldOutputKind,
    },
    #[error("field output {output:?} contains invalid constrained value {value}")]
    FieldValueViolatesConstraint { output: FieldOutputKind, value: f64 },
    #[error("cannot invert non-positive stiffness {0}")]
    CannotInvertNonpositiveStiffness(f64),
    #[error("cannot invert non-positive compliance {0}")]
    CannotInvertNonpositiveCompliance(f64),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParameterSourceKind;

    fn measured_source(name: &str) -> ParameterSource {
        ParameterSource::new(ParameterSourceKind::Measured, name)
            .with_evidence_ref(format!("measurement:{name}"))
    }

    fn fitted_source(name: &str) -> ParameterSource {
        ParameterSource::new(ParameterSourceKind::FittedFromMeasurement, name)
            .with_evidence_ref(format!("measurement:{name}"))
            .with_method("surface-fit-v1")
    }

    fn axis(kind: FieldAxisKind, points: &[f64]) -> SampledAxis {
        SampledAxis {
            kind,
            points: points.to_vec(),
        }
    }

    #[test]
    fn asymmetric_bl_surface_is_not_forced_symmetric() {
        let field = SampledScalarField {
            id: "bl-x-i".into(),
            axes: vec![
                axis(FieldAxisKind::DisplacementMeter, &[-0.006, 0.0, 0.005]),
                axis(FieldAxisKind::CurrentAmpere, &[0.0, 2.0]),
            ],
            values: vec![4.0, 3.8, 6.5, 6.1, 5.1, 4.6],
            output: FieldOutputKind::ForceFactorTeslaMeter,
            source: measured_source("klippel-bl-run-1"),
            interpolation: InterpolationPolicy::Multilinear,
            extrapolation: ExtrapolationPolicy::Reject,
        };
        let model = NonlinearTransducerFields {
            force_factor_bl_x_i: Some(field),
            ..Default::default()
        };
        model.validate().unwrap();
        let negative = model.force_factor_tesla_meter(-0.003, 1.0).unwrap().unwrap();
        let positive = model.force_factor_tesla_meter(0.0025, 1.0).unwrap().unwrap();
        assert_ne!(negative, positive);
    }

    #[test]
    fn extrapolation_is_rejected() {
        let field = SampledScalarField {
            id: "re-t".into(),
            axes: vec![axis(FieldAxisKind::TemperatureCelsius, &[20.0, 100.0])],
            values: vec![6.0, 7.9],
            output: FieldOutputKind::ResistanceOhm,
            source: measured_source("thermal-run"),
            interpolation: InterpolationPolicy::Multilinear,
            extrapolation: ExtrapolationPolicy::Reject,
        };
        assert!(matches!(
            field.evaluate(&[150.0]),
            Err(NonlinearModelError::OutsideSampledEnvelope { .. })
        ));
    }

    #[test]
    fn trilinear_inductance_interpolates_inside_frozen_grid() {
        let field = SampledScalarField {
            id: "le-x-i-f".into(),
            axes: vec![
                axis(FieldAxisKind::DisplacementMeter, &[-0.001, 0.001]),
                axis(FieldAxisKind::CurrentAmpere, &[0.0, 2.0]),
                axis(FieldAxisKind::FrequencyHertz, &[100.0, 1000.0]),
            ],
            values: vec![0.00060, 0.00050, 0.00055, 0.00045, 0.00058, 0.00048, 0.00053, 0.00043],
            output: FieldOutputKind::InductanceHenry,
            source: fitted_source("impedance-surface-fit"),
            interpolation: InterpolationPolicy::Multilinear,
            extrapolation: ExtrapolationPolicy::Reject,
        };
        field
            .validate_profile(
                &[
                    FieldAxisKind::DisplacementMeter,
                    FieldAxisKind::CurrentAmpere,
                    FieldAxisKind::FrequencyHertz,
                ],
                FieldOutputKind::InductanceHenry,
                ValueConstraint::Nonnegative,
            )
            .unwrap();
        let value = field.evaluate(&[0.0, 1.0, 550.0]).unwrap();
        let expected = field.values.iter().sum::<f64>() / field.values.len() as f64;
        assert!((value - expected).abs() < 1e-15);
    }

    #[test]
    fn wrong_axis_order_is_rejected_by_domain_profile() {
        let field = SampledScalarField {
            id: "bl-wrong-order".into(),
            axes: vec![
                axis(FieldAxisKind::CurrentAmpere, &[0.0, 2.0]),
                axis(FieldAxisKind::DisplacementMeter, &[-0.005, 0.005]),
            ],
            values: vec![6.0, 5.0, 5.8, 4.8],
            output: FieldOutputKind::ForceFactorTeslaMeter,
            source: measured_source("bl-run"),
            interpolation: InterpolationPolicy::Multilinear,
            extrapolation: ExtrapolationPolicy::Reject,
        };
        let model = NonlinearTransducerFields {
            force_factor_bl_x_i: Some(field),
            ..Default::default()
        };
        assert!(matches!(
            model.validate(),
            Err(NonlinearModelError::WrongFieldAxes { .. })
        ));
    }

    #[test]
    fn duplicate_axis_is_rejected() {
        let field = SampledScalarField {
            id: "duplicate-axis".into(),
            axes: vec![
                axis(FieldAxisKind::CurrentAmpere, &[0.0, 1.0]),
                axis(FieldAxisKind::CurrentAmpere, &[0.0, 2.0]),
            ],
            values: vec![1.0; 4],
            output: FieldOutputKind::ForceFactorTeslaMeter,
            source: measured_source("bad-grid"),
            interpolation: InterpolationPolicy::Multilinear,
            extrapolation: ExtrapolationPolicy::Reject,
        };
        assert_eq!(
            field.validate(),
            Err(NonlinearModelError::DuplicateAxis(FieldAxisKind::CurrentAmpere))
        );
    }

    #[test]
    fn measured_surface_requires_measurement_evidence() {
        let field = SampledScalarField {
            id: "re-t".into(),
            axes: vec![axis(FieldAxisKind::TemperatureCelsius, &[20.0, 100.0])],
            values: vec![6.0, 7.9],
            output: FieldOutputKind::ResistanceOhm,
            source: ParameterSource::new(ParameterSourceKind::Measured, "missing-run"),
            interpolation: InterpolationPolicy::Multilinear,
            extrapolation: ExtrapolationPolicy::Reject,
        };
        assert!(matches!(
            field.validate(),
            Err(NonlinearModelError::InvalidParameterSource(
                ValidationError::MissingEvidenceReference(ParameterSourceKind::Measured)
            ))
        ));
    }

    #[test]
    fn nonlinear_suspension_keeps_one_canonical_source_family() {
        let field = SampledScalarField {
            id: "kms-x".into(),
            axes: vec![axis(FieldAxisKind::DisplacementMeter, &[-0.005, 0.005])],
            values: vec![1800.0, 2200.0],
            output: FieldOutputKind::StiffnessNewtonPerMeter,
            source: fitted_source("kms-fit"),
            interpolation: InterpolationPolicy::Multilinear,
            extrapolation: ExtrapolationPolicy::Reject,
        };
        let suspension = NonlinearSuspensionField::StiffnessVsDisplacement(field);
        suspension.validate().unwrap();
        let compliance = suspension.compliance_m_per_n(0.0).unwrap();
        assert!((compliance - 1.0 / 2000.0).abs() < 1e-15);
    }

    #[test]
    fn serialization_preserves_source_and_envelope() {
        let model = NonlinearTransducerFields {
            voice_coil_resistance_re_t: Some(SampledScalarField {
                id: "re-t".into(),
                axes: vec![axis(FieldAxisKind::TemperatureCelsius, &[20.0, 100.0])],
                values: vec![6.0, 7.9],
                output: FieldOutputKind::ResistanceOhm,
                source: measured_source("thermal-run"),
                interpolation: InterpolationPolicy::Multilinear,
                extrapolation: ExtrapolationPolicy::Reject,
            }),
            ..Default::default()
        };
        let json = serde_json::to_string(&model).unwrap();
        let decoded: NonlinearTransducerFields = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, model);
        decoded.validate().unwrap();
    }
}
