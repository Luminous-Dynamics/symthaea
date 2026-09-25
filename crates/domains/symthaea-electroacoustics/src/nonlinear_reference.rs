// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Diagnostic comparison between EAC-001 nominal parameters and EAC-004 state surfaces.
//!
//! A nonlinear surface is not automatically continuous with the nominal
//! small-signal model merely because both describe the same driver. This module
//! evaluates declared nonlinear surfaces at an explicit equilibrium/reference
//! state and reports discrepancies without turning them into a hidden pass/fail
//! policy.

use crate::nonlinear::{
    EvaluationDisposition, NonlinearModelError, NonlinearSuspensionModel,
    NonlinearTransducerState, StateAxisKind, StateDependentParameter, StateResponseKind,
};
use crate::operating_state::{OperatingStateError, TransducerOperatingState};
use crate::{
    ParameterSource, PhysicalUnit, SuspensionParameter, TransducerModel, ValidationError,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Explicit context under which EAC-001 nominal values are compared to EAC-004.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NominalReferenceContext {
    pub operating_state: TransducerOperatingState,
    /// Required only when a nonlinear inductance surface depends on frequency.
    /// EAC-001 does not itself encode the reference frequency of `Le`, so this
    /// value is an explicit caller commitment rather than an inferred fact.
    pub inductance_reference_frequency_hz: Option<f64>,
    /// Audit-readable source/assumption explaining the reference conditions.
    pub provenance: String,
}

impl NominalReferenceContext {
    /// Construct the ordinary zero-bias equilibrium at the EAC-001 reference
    /// temperature. Frequency remains unspecified until explicitly supplied.
    pub fn equilibrium(
        transducer: &TransducerModel,
        provenance: impl Into<String>,
    ) -> Result<Self, NominalReferenceError> {
        transducer.validate()?;
        Ok(Self {
            operating_state: TransducerOperatingState::new(
                0.0,
                0.0,
                0.0,
                transducer.electrical.reference_temperature.value,
            ),
            inductance_reference_frequency_hz: None,
            provenance: provenance.into(),
        })
    }

    pub fn with_inductance_reference_frequency(mut self, frequency_hz: f64) -> Self {
        self.inductance_reference_frequency_hz = Some(frequency_hz);
        self.operating_state = self.operating_state.with_frequency(frequency_hz);
        self
    }

    fn validate(&self, transducer: &TransducerModel) -> Result<(), NominalReferenceError> {
        self.operating_state.validate()?;
        if self.provenance.trim().is_empty() {
            return Err(NominalReferenceError::MissingContextProvenance);
        }
        for (field, value) in [
            ("displacement_m", self.operating_state.displacement_m),
            ("current_a", self.operating_state.current_a),
            ("velocity_m_per_s", self.operating_state.velocity_m_per_s),
        ] {
            if value != 0.0 {
                return Err(NominalReferenceError::NotEquilibriumReference { field, value });
            }
        }
        let expected_temperature = transducer.electrical.reference_temperature.value;
        if self.operating_state.coil_temperature_c != expected_temperature {
            return Err(NominalReferenceError::ReferenceTemperatureMismatch {
                expected_c: expected_temperature,
                actual_c: self.operating_state.coil_temperature_c,
            });
        }
        if let Some(reference_frequency) = self.inductance_reference_frequency_hz {
            if !reference_frequency.is_finite() || reference_frequency < 0.0 {
                return Err(NominalReferenceError::InvalidInductanceReferenceFrequency(
                    reference_frequency,
                ));
            }
            if self.operating_state.frequency_hz != Some(reference_frequency) {
                return Err(NominalReferenceError::InductanceFrequencyStateMismatch);
            }
        }
        Ok(())
    }
}

/// How the nominal scalar used in a comparison was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NominalTransform {
    Direct,
    /// EAC-001 stored compliance but the nonlinear surface is stiffness, or vice versa.
    ReciprocalSuspension,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NominalComparisonObservation {
    pub response: StateResponseKind,
    pub nominal_value: f64,
    pub nonlinear_value: f64,
    pub unit: PhysicalUnit,
    pub absolute_delta: f64,
    pub relative_delta: Option<f64>,
    pub evaluation_disposition: EvaluationDisposition,
    pub nominal_source: ParameterSource,
    pub nominal_transform: NominalTransform,
    pub reference_coordinates: Vec<f64>,
    pub context_provenance: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NominalComparisonUnavailableReason {
    NominalValueUnknown,
    InductanceReferenceFrequencyUnknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NominalComparisonUnavailable {
    pub response: StateResponseKind,
    pub reason: NominalComparisonUnavailableReason,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NominalComparisonResult {
    Compared(NominalComparisonObservation),
    Unavailable(NominalComparisonUnavailable),
}

/// Compare every supplied nonlinear family to EAC-001 at an explicit nominal
/// reference condition. The result is diagnostic evidence only; no tolerance or
/// product admission threshold is applied here.
pub fn compare_to_nominal_reference(
    transducer: &TransducerModel,
    nonlinear: &NonlinearTransducerState,
    context: &NominalReferenceContext,
) -> Result<Vec<NominalComparisonResult>, NominalReferenceError> {
    transducer.validate()?;
    nonlinear.validate()?;
    if nonlinear.transducer_id != transducer.id {
        return Err(NominalReferenceError::TransducerIdMismatch {
            nominal: transducer.id.clone(),
            nonlinear: nonlinear.transducer_id.clone(),
        });
    }
    context.validate(transducer)?;

    let mut results = Vec::new();

    if let Some(parameter) = &nonlinear.force_factor {
        results.push(NominalComparisonResult::Compared(compare_parameter(
            parameter,
            transducer.motor.force_factor.value,
            transducer.motor.force_factor.source.clone(),
            NominalTransform::Direct,
            context,
        )?));
    }

    if let Some(parameter) = &nonlinear.inductance {
        match transducer.electrical.voice_coil_inductance.as_ref() {
            None => results.push(NominalComparisonResult::Unavailable(
                NominalComparisonUnavailable {
                    response: StateResponseKind::Inductance,
                    reason: NominalComparisonUnavailableReason::NominalValueUnknown,
                },
            )),
            Some(nominal) => {
                let frequency_dependent = parameter
                    .axes
                    .iter()
                    .any(|axis| axis.kind == StateAxisKind::Frequency);
                if frequency_dependent && context.inductance_reference_frequency_hz.is_none() {
                    results.push(NominalComparisonResult::Unavailable(
                        NominalComparisonUnavailable {
                            response: StateResponseKind::Inductance,
                            reason: NominalComparisonUnavailableReason::InductanceReferenceFrequencyUnknown,
                        },
                    ));
                } else {
                    results.push(NominalComparisonResult::Compared(compare_parameter(
                        parameter,
                        nominal.value,
                        nominal.source.clone(),
                        NominalTransform::Direct,
                        context,
                    )?));
                }
            }
        }
    }

    if let Some(suspension) = &nonlinear.suspension {
        let (parameter, nominal_value, nominal_source, transform) = match suspension {
            NonlinearSuspensionModel::Compliance(parameter) => match &transducer.mechanical.suspension {
                SuspensionParameter::Compliance(nominal) => (
                    parameter,
                    nominal.value,
                    nominal.source.clone(),
                    NominalTransform::Direct,
                ),
                SuspensionParameter::Stiffness(nominal) => (
                    parameter,
                    1.0 / nominal.value,
                    nominal.source.clone(),
                    NominalTransform::ReciprocalSuspension,
                ),
            },
            NonlinearSuspensionModel::Stiffness(parameter) => match &transducer.mechanical.suspension {
                SuspensionParameter::Stiffness(nominal) => (
                    parameter,
                    nominal.value,
                    nominal.source.clone(),
                    NominalTransform::Direct,
                ),
                SuspensionParameter::Compliance(nominal) => (
                    parameter,
                    1.0 / nominal.value,
                    nominal.source.clone(),
                    NominalTransform::ReciprocalSuspension,
                ),
            },
        };
        results.push(NominalComparisonResult::Compared(compare_parameter(
            parameter,
            nominal_value,
            nominal_source,
            transform,
            context,
        )?));
    }

    if let Some(parameter) = &nonlinear.mechanical_resistance {
        results.push(NominalComparisonResult::Compared(compare_parameter(
            parameter,
            transducer.mechanical.mechanical_resistance.value,
            transducer.mechanical.mechanical_resistance.source.clone(),
            NominalTransform::Direct,
            context,
        )?));
    }

    if let Some(parameter) = &nonlinear.voice_coil_resistance {
        results.push(NominalComparisonResult::Compared(compare_parameter(
            parameter,
            transducer.electrical.voice_coil_resistance.value,
            transducer.electrical.voice_coil_resistance.source.clone(),
            NominalTransform::Direct,
            context,
        )?));
    }

    Ok(results)
}

fn compare_parameter(
    parameter: &StateDependentParameter,
    nominal_value: f64,
    nominal_source: ParameterSource,
    nominal_transform: NominalTransform,
    context: &NominalReferenceContext,
) -> Result<NominalComparisonObservation, NominalReferenceError> {
    let coordinates = context.operating_state.coordinates_for(parameter)?;
    let evaluation = parameter.evaluate(&coordinates)?;
    let absolute_delta = evaluation.value - nominal_value;
    let relative_delta = if nominal_value.abs() > f64::EPSILON {
        Some(absolute_delta / nominal_value)
    } else {
        None
    };
    Ok(NominalComparisonObservation {
        response: parameter.response,
        nominal_value,
        nonlinear_value: evaluation.value,
        unit: parameter.response_unit,
        absolute_delta,
        relative_delta,
        evaluation_disposition: evaluation.disposition,
        nominal_source,
        nominal_transform,
        reference_coordinates: coordinates,
        context_provenance: context.provenance.clone(),
    })
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum NominalReferenceError {
    #[error(transparent)]
    InvalidTransducer(#[from] ValidationError),
    #[error(transparent)]
    InvalidNonlinearModel(#[from] NonlinearModelError),
    #[error(transparent)]
    OperatingState(#[from] OperatingStateError),
    #[error("nominal/nonlinear transducer ids differ: {nominal:?} != {nonlinear:?}")]
    TransducerIdMismatch { nominal: String, nonlinear: String },
    #[error("nominal reference context requires provenance")]
    MissingContextProvenance,
    #[error("nominal comparison requires equilibrium {field}=0, got {value}")]
    NotEquilibriumReference { field: &'static str, value: f64 },
    #[error("reference temperature mismatch: expected {expected_c} C, got {actual_c} C")]
    ReferenceTemperatureMismatch { expected_c: f64, actual_c: f64 },
    #[error("invalid declared inductance reference frequency {0}")]
    InvalidInductanceReferenceFrequency(f64),
    #[error("declared inductance reference frequency does not match operating-state frequency")]
    InductanceFrequencyStateMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear::{
        ExtrapolationPolicy, InterpolationPolicy, StateAxis, StateAxisUnit, StateSample,
    };
    use crate::{
        AcousticParameters, ElectricalParameters, MechanicalParameters, MotorParameters,
        ParameterSourceKind, ScalarParameter, ThermalParameters,
    };

    fn src(name: &str) -> ParameterSource {
        ParameterSource::new(ParameterSourceKind::Datasheet, name)
    }

    fn q(value: f64, unit: PhysicalUnit, name: &str) -> ScalarParameter {
        ScalarParameter::new(value, unit, src(name))
    }

    fn driver() -> TransducerModel {
        TransducerModel {
            id: "driver-1".into(),
            electrical: ElectricalParameters {
                voice_coil_resistance: q(6.0, PhysicalUnit::Ohm, "Re"),
                voice_coil_inductance: Some(q(0.0005, PhysicalUnit::Henry, "Le")),
                reference_temperature: q(20.0, PhysicalUnit::Celsius, "Tref"),
                max_voltage: None,
                max_current: None,
                max_power: None,
            },
            motor: MotorParameters {
                force_factor: q(5.0, PhysicalUnit::TeslaMeter, "Bl"),
                max_linear_excursion: None,
                max_current: None,
            },
            mechanical: MechanicalParameters {
                moving_mass: q(0.02, PhysicalUnit::Kilogram, "Mms"),
                suspension: SuspensionParameter::Compliance(q(
                    0.0005,
                    PhysicalUnit::MeterPerNewton,
                    "Cms",
                )),
                mechanical_resistance: q(
                    2.0,
                    PhysicalUnit::NewtonSecondPerMeter,
                    "Rms",
                ),
                max_mechanical_excursion: None,
            },
            acoustic: AcousticParameters {
                effective_piston_area: q(0.01, PhysicalUnit::SquareMeter, "Sd"),
                radiation_model_id: "unused".into(),
            },
            thermal: ThermalParameters {
                resistance_temperature_coefficient: None,
                max_coil_temperature: None,
            },
        }
    }

    fn curve(
        id: &str,
        response: StateResponseKind,
        unit: PhysicalUnit,
        axis_kind: StateAxisKind,
        axis_unit: StateAxisUnit,
        lower: f64,
        center: f64,
        upper: f64,
    ) -> StateDependentParameter {
        StateDependentParameter {
            id: id.into(),
            response,
            response_unit: unit,
            axes: vec![StateAxis::new(axis_kind, axis_unit, -1.0, 1.0)],
            samples: vec![
                StateSample::new(vec![-1.0], lower),
                StateSample::new(vec![0.0], center),
                StateSample::new(vec![1.0], upper),
            ],
            source: src(id),
            interpolation: InterpolationPolicy::Linear1D,
            extrapolation: ExtrapolationPolicy::Reject,
        }
    }

    #[test]
    fn matching_bl_reference_reports_zero_discrepancy_without_pass_label() {
        let driver = driver();
        let state = NonlinearTransducerState {
            transducer_id: driver.id.clone(),
            force_factor: Some(curve(
                "bl",
                StateResponseKind::ForceFactor,
                PhysicalUnit::TeslaMeter,
                StateAxisKind::Displacement,
                StateAxisUnit::Meter,
                4.0,
                5.0,
                4.5,
            )),
            inductance: None,
            suspension: None,
            mechanical_resistance: None,
            voice_coil_resistance: None,
        };
        let context = NominalReferenceContext::equilibrium(&driver, "fixture reference").unwrap();
        let results = compare_to_nominal_reference(&driver, &state, &context).unwrap();
        let NominalComparisonResult::Compared(obs) = &results[0] else { panic!() };
        assert_eq!(obs.absolute_delta, 0.0);
        assert_eq!(obs.nominal_transform, NominalTransform::Direct);
    }

    #[test]
    fn disagreement_is_reported_not_silently_rejected_or_corrected() {
        let driver = driver();
        let state = NonlinearTransducerState {
            transducer_id: driver.id.clone(),
            force_factor: Some(curve(
                "bl",
                StateResponseKind::ForceFactor,
                PhysicalUnit::TeslaMeter,
                StateAxisKind::Displacement,
                StateAxisUnit::Meter,
                4.0,
                4.8,
                4.5,
            )),
            inductance: None,
            suspension: None,
            mechanical_resistance: None,
            voice_coil_resistance: None,
        };
        let context = NominalReferenceContext::equilibrium(&driver, "fixture reference").unwrap();
        let results = compare_to_nominal_reference(&driver, &state, &context).unwrap();
        let NominalComparisonResult::Compared(obs) = &results[0] else { panic!() };
        assert!((obs.absolute_delta + 0.2).abs() < 1e-12);
    }

    #[test]
    fn frequency_dependent_inductance_needs_explicit_reference_frequency() {
        let driver = driver();
        let inductance = StateDependentParameter {
            id: "le-f".into(),
            response: StateResponseKind::Inductance,
            response_unit: PhysicalUnit::Henry,
            axes: vec![StateAxis::new(
                StateAxisKind::Frequency,
                StateAxisUnit::Hertz,
                100.0,
                1000.0,
            )],
            samples: vec![
                StateSample::new(vec![100.0], 0.0006),
                StateSample::new(vec![1000.0], 0.0004),
            ],
            source: src("Le(f)"),
            interpolation: InterpolationPolicy::Linear1D,
            extrapolation: ExtrapolationPolicy::Reject,
        };
        let state = NonlinearTransducerState {
            transducer_id: driver.id.clone(),
            force_factor: None,
            inductance: Some(inductance),
            suspension: None,
            mechanical_resistance: None,
            voice_coil_resistance: None,
        };
        let context = NominalReferenceContext::equilibrium(&driver, "frequency unknown").unwrap();
        let results = compare_to_nominal_reference(&driver, &state, &context).unwrap();
        assert_eq!(
            results,
            vec![NominalComparisonResult::Unavailable(
                NominalComparisonUnavailable {
                    response: StateResponseKind::Inductance,
                    reason: NominalComparisonUnavailableReason::InductanceReferenceFrequencyUnknown,
                }
            )]
        );
    }

    #[test]
    fn reciprocal_suspension_comparison_is_explicit() {
        let driver = driver();
        let stiffness = curve(
            "kms",
            StateResponseKind::Stiffness,
            PhysicalUnit::NewtonPerMeter,
            StateAxisKind::Displacement,
            StateAxisUnit::Meter,
            1800.0,
            2000.0,
            2200.0,
        );
        let state = NonlinearTransducerState {
            transducer_id: driver.id.clone(),
            force_factor: None,
            inductance: None,
            suspension: Some(NonlinearSuspensionModel::Stiffness(stiffness)),
            mechanical_resistance: None,
            voice_coil_resistance: None,
        };
        let context = NominalReferenceContext::equilibrium(&driver, "fixture reference").unwrap();
        let results = compare_to_nominal_reference(&driver, &state, &context).unwrap();
        let NominalComparisonResult::Compared(obs) = &results[0] else { panic!() };
        assert_eq!(obs.nominal_transform, NominalTransform::ReciprocalSuspension);
        assert!((obs.nominal_value - 2000.0).abs() < 1e-12);
    }

    #[test]
    fn reference_temperature_must_match_eac001_reference() {
        let driver = driver();
        let mut context = NominalReferenceContext::equilibrium(&driver, "fixture reference").unwrap();
        context.operating_state.coil_temperature_c = 25.0;
        let state = NonlinearTransducerState {
            transducer_id: driver.id.clone(),
            force_factor: None,
            inductance: None,
            suspension: None,
            mechanical_resistance: None,
            voice_coil_resistance: None,
        };
        assert!(matches!(
            compare_to_nominal_reference(&driver, &state, &context),
            Err(NominalReferenceError::ReferenceTemperatureMismatch { .. })
        ));
    }
}
