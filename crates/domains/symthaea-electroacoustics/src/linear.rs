// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Small-signal moving-coil electro-mechanical reference model.
//!
//! This module is deliberately a transparent analytical baseline. It does not
//! model enclosure loading, diaphragm breakup, acoustic radiation, thermal drift,
//! suspension/motor nonlinearities, source impedance, or room response.
//!
//! The governing small-signal equations are:
//!
//! `Z_m = R_ms + j(omega M_ms - 1/(omega C_ms))`
//!
//! `Z_in = R_e + j omega L_e + (Bl)^2 / Z_m`
//!
//! `I = V / Z_in`, `v = Bl I / Z_m`, `x = v / (j omega)`.
//!
//! Results from this module are `AnalyticalPrediction`, never physical
//! measurement evidence.

use crate::{
    ParameterSource, ScalarParameter, SuspensionParameter, TransducerModel, ValidationError,
};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use thiserror::Error;

const MODEL_ID: &str = "moving-coil-lumped-small-signal-v1";
const SINGULAR_EPSILON: f64 = 1.0e-30;

/// Authority attached to EAC-002 results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionAuthority {
    AnalyticalPrediction,
}

/// One source parameter consumed by an analytical derivation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputSourceRef {
    pub field: String,
    pub source: ParameterSource,
}

/// Inspectable derivation metadata carried by every EAC-002 prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivationRecord {
    pub authority: PredictionAuthority,
    pub analytical_model_id: String,
    pub subject_model_id: String,
    pub equation_id: String,
    pub input_sources: Vec<InputSourceRef>,
    pub assumptions: Vec<String>,
}

/// Minimal complex quantity used for analytical phasor results.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ComplexValue {
    pub re: f64,
    pub im: f64,
}

impl ComplexValue {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn magnitude(self) -> f64 {
        self.re.hypot(self.im)
    }

    pub fn phase_radians(self) -> f64 {
        self.im.atan2(self.re)
    }

    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }

    fn scale(self, scale: f64) -> Self {
        Self::new(self.re * scale, self.im * scale)
    }

    fn checked_div(self, rhs: Self, quantity: &'static str) -> Result<Self, LinearModelError> {
        let denominator = rhs.re * rhs.re + rhs.im * rhs.im;
        if !denominator.is_finite() || denominator <= SINGULAR_EPSILON {
            return Err(LinearModelError::SingularComplexQuantity { quantity });
        }
        Ok(Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denominator,
            (self.im * rhs.re - self.re * rhs.im) / denominator,
        ))
    }
}

/// One scalar result with explicit analytical derivation metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarPrediction {
    pub value: f64,
    pub unit: String,
    pub derivation: DerivationRecord,
}

/// Derived free-air quality factors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityFactorPrediction {
    pub resonant_frequency_hz: f64,
    pub q_mechanical: f64,
    pub q_electrical: f64,
    pub q_total: f64,
    pub derivation: DerivationRecord,
}

/// Small-signal sinusoidal steady-state response at one frequency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrequencyPointPrediction {
    pub frequency_hz: f64,
    pub terminal_voltage_rms: f64,
    pub input_impedance_ohm: ComplexValue,
    pub current_a_rms: ComplexValue,
    pub velocity_m_per_s_rms: ComplexValue,
    pub displacement_m_rms: ComplexValue,
    pub derivation: DerivationRecord,
}

/// Borrowed analytical view over one validated EAC-001 transducer model.
pub struct LinearReferenceModel<'a> {
    transducer: &'a TransducerModel,
}

impl<'a> LinearReferenceModel<'a> {
    /// Validate the EAC-001 subject before any analytical prediction is admitted.
    pub fn new(transducer: &'a TransducerModel) -> Result<Self, LinearModelError> {
        transducer.validate()?;
        Ok(Self { transducer })
    }

    /// Free-air small-signal mechanical resonance.
    pub fn free_air_resonance(&self) -> Result<ScalarPrediction, LinearModelError> {
        let mass = self.transducer.mechanical.moving_mass.value;
        let compliance = self.transducer.mechanical.suspension.compliance_m_per_n()?;
        let omega_0 = 1.0 / (mass * compliance).sqrt();
        let frequency_hz = omega_0 / (2.0 * PI);

        ensure_finite_positive(frequency_hz, "free_air_resonance_hz")?;

        Ok(ScalarPrediction {
            value: frequency_hz,
            unit: "Hz".into(),
            derivation: self.derivation(
                "fs-v1",
                vec![
                    source_ref("mechanical.moving_mass", &self.transducer.mechanical.moving_mass),
                    suspension_source_ref(&self.transducer.mechanical.suspension),
                ],
                vec![
                    "linear small-signal moving-coil model".into(),
                    "free-air mechanical resonance; no enclosure/acoustic loading".into(),
                    "mass and compliance are constant over the evaluated amplitude".into(),
                ],
            ),
        })
    }

    /// Classical free-air Qms/Qes/Qts views.
    ///
    /// A zero mechanical resistance is a valid idealized EAC-001 parameter, but
    /// it produces infinite Qms. This method fails closed rather than serializing
    /// infinity as an ordinary finite engineering result.
    pub fn quality_factors(&self) -> Result<QualityFactorPrediction, LinearModelError> {
        let mass = self.transducer.mechanical.moving_mass.value;
        let compliance = self.transducer.mechanical.suspension.compliance_m_per_n()?;
        let rms = self.transducer.mechanical.mechanical_resistance.value;
        let re = self.transducer.electrical.voice_coil_resistance.value;
        let bl = self.transducer.motor.force_factor.value;

        if rms <= 0.0 {
            return Err(LinearModelError::UndefinedFiniteQualityFactor);
        }

        let omega_0 = 1.0 / (mass * compliance).sqrt();
        let fs = omega_0 / (2.0 * PI);
        let q_mechanical = omega_0 * mass / rms;
        let q_electrical = omega_0 * mass * re / (bl * bl);
        let q_total = 1.0 / (1.0 / q_mechanical + 1.0 / q_electrical);

        for (name, value) in [
            ("resonant_frequency_hz", fs),
            ("q_mechanical", q_mechanical),
            ("q_electrical", q_electrical),
            ("q_total", q_total),
        ] {
            ensure_finite_positive(value, name)?;
        }

        Ok(QualityFactorPrediction {
            resonant_frequency_hz: fs,
            q_mechanical,
            q_electrical,
            q_total,
            derivation: self.derivation(
                "free-air-q-v1",
                vec![
                    source_ref("electrical.voice_coil_resistance", &self.transducer.electrical.voice_coil_resistance),
                    source_ref("motor.force_factor", &self.transducer.motor.force_factor),
                    source_ref("mechanical.moving_mass", &self.transducer.mechanical.moving_mass),
                    suspension_source_ref(&self.transducer.mechanical.suspension),
                    source_ref("mechanical.mechanical_resistance", &self.transducer.mechanical.mechanical_resistance),
                ],
                vec![
                    "linear small-signal moving-coil model".into(),
                    "voice-coil inductance neglected in classical Qes derivation".into(),
                    "no acoustic/enclosure load added to mechanical impedance".into(),
                ],
            ),
        })
    }

    /// Terminal electrical and moving-system response to a real RMS sinusoidal
    /// voltage at one positive frequency.
    ///
    /// `L_e` is required explicitly. Callers that intentionally want the
    /// zero-inductance approximation must provide `Some(0 H)` with provenance;
    /// `None` is unknown and is never silently interpreted as zero.
    pub fn response_at(
        &self,
        frequency_hz: f64,
        terminal_voltage_rms: f64,
    ) -> Result<FrequencyPointPrediction, LinearModelError> {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(LinearModelError::InvalidFrequency(frequency_hz));
        }
        if !terminal_voltage_rms.is_finite() || terminal_voltage_rms < 0.0 {
            return Err(LinearModelError::InvalidTerminalVoltage(terminal_voltage_rms));
        }

        let inductance = self
            .transducer
            .electrical
            .voice_coil_inductance
            .as_ref()
            .ok_or(LinearModelError::MissingRequiredParameter(
                "electrical.voice_coil_inductance",
            ))?;

        let re = self.transducer.electrical.voice_coil_resistance.value;
        let le = inductance.value;
        let bl = self.transducer.motor.force_factor.value;
        let mass = self.transducer.mechanical.moving_mass.value;
        let compliance = self.transducer.mechanical.suspension.compliance_m_per_n()?;
        let rms = self.transducer.mechanical.mechanical_resistance.value;
        let omega = 2.0 * PI * frequency_hz;

        let mechanical_impedance = ComplexValue::new(
            rms,
            omega * mass - 1.0 / (omega * compliance),
        );
        let motional_impedance = ComplexValue::new(bl * bl, 0.0)
            .checked_div(mechanical_impedance, "mechanical_impedance")?;
        let electrical_impedance = ComplexValue::new(re, omega * le);
        let input_impedance = electrical_impedance.add(motional_impedance);
        let current = ComplexValue::new(terminal_voltage_rms, 0.0)
            .checked_div(input_impedance, "input_impedance")?;
        let velocity = current
            .scale(bl)
            .checked_div(mechanical_impedance, "mechanical_impedance")?;
        let displacement = velocity.checked_div(
            ComplexValue::new(0.0, omega),
            "angular_frequency",
        )?;

        for (name, value) in [
            ("input_impedance_magnitude", input_impedance.magnitude()),
            ("current_magnitude", current.magnitude()),
            ("velocity_magnitude", velocity.magnitude()),
            ("displacement_magnitude", displacement.magnitude()),
        ] {
            ensure_finite_nonnegative(value, name)?;
        }

        Ok(FrequencyPointPrediction {
            frequency_hz,
            terminal_voltage_rms,
            input_impedance_ohm: input_impedance,
            current_a_rms: current,
            velocity_m_per_s_rms: velocity,
            displacement_m_rms: displacement,
            derivation: self.derivation(
                "sinusoidal-terminal-response-v1",
                vec![
                    source_ref("electrical.voice_coil_resistance", &self.transducer.electrical.voice_coil_resistance),
                    source_ref("electrical.voice_coil_inductance", inductance),
                    source_ref("motor.force_factor", &self.transducer.motor.force_factor),
                    source_ref("mechanical.moving_mass", &self.transducer.mechanical.moving_mass),
                    suspension_source_ref(&self.transducer.mechanical.suspension),
                    source_ref("mechanical.mechanical_resistance", &self.transducer.mechanical.mechanical_resistance),
                ],
                vec![
                    "linear small-signal moving-coil model".into(),
                    "sinusoidal steady state".into(),
                    "voltage is specified directly at driver terminals".into(),
                    "force factor, inductance, compliance, and resistance are state-invariant".into(),
                    "no enclosure, radiation, diaphragm breakup, thermal, or room effects".into(),
                ],
            ),
        })
    }

    fn derivation(
        &self,
        equation_id: &str,
        input_sources: Vec<InputSourceRef>,
        assumptions: Vec<String>,
    ) -> DerivationRecord {
        DerivationRecord {
            authority: PredictionAuthority::AnalyticalPrediction,
            analytical_model_id: MODEL_ID.into(),
            subject_model_id: self.transducer.id.clone(),
            equation_id: equation_id.into(),
            input_sources,
            assumptions,
        }
    }
}

fn source_ref(field: &str, parameter: &ScalarParameter) -> InputSourceRef {
    InputSourceRef {
        field: field.into(),
        source: parameter.source.clone(),
    }
}

fn suspension_source_ref(suspension: &SuspensionParameter) -> InputSourceRef {
    match suspension {
        SuspensionParameter::Compliance(parameter) => {
            source_ref("mechanical.suspension.compliance", parameter)
        }
        SuspensionParameter::Stiffness(parameter) => {
            source_ref("mechanical.suspension.stiffness", parameter)
        }
    }
}

fn ensure_finite_positive(value: f64, quantity: &'static str) -> Result<(), LinearModelError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(LinearModelError::InvalidDerivedValue { quantity, value });
    }
    Ok(())
}

fn ensure_finite_nonnegative(value: f64, quantity: &'static str) -> Result<(), LinearModelError> {
    if !value.is_finite() || value < 0.0 {
        return Err(LinearModelError::InvalidDerivedValue { quantity, value });
    }
    Ok(())
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum LinearModelError {
    #[error(transparent)]
    InvalidTransducer(#[from] ValidationError),
    #[error("required parameter {0} is unknown")]
    MissingRequiredParameter(&'static str),
    #[error("frequency must be finite and positive, got {0}")]
    InvalidFrequency(f64),
    #[error("terminal RMS voltage must be finite and nonnegative, got {0}")]
    InvalidTerminalVoltage(f64),
    #[error("finite Q factors are undefined for a zero-loss mechanical model")]
    UndefinedFiniteQualityFactor,
    #[error("cannot divide by singular {quantity}")]
    SingularComplexQuantity { quantity: &'static str },
    #[error("derived {quantity} is invalid: {value}")]
    InvalidDerivedValue { quantity: &'static str, value: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AcousticParameters, ElectricalParameters, MechanicalParameters, MotorParameters,
        ParameterSourceKind, PhysicalUnit, ThermalParameters,
    };

    fn source(name: &str) -> ParameterSource {
        ParameterSource::new(ParameterSourceKind::Datasheet, name)
    }

    fn q(value: f64, unit: PhysicalUnit, name: &str) -> ScalarParameter {
        ScalarParameter::new(value, unit, source(name))
    }

    fn fixture() -> TransducerModel {
        TransducerModel {
            id: "linear-fixture-001".into(),
            electrical: ElectricalParameters {
                voice_coil_resistance: q(6.0, PhysicalUnit::Ohm, "fixture-re"),
                voice_coil_inductance: Some(q(0.0, PhysicalUnit::Henry, "explicit-zero-le")),
                reference_temperature: q(20.0, PhysicalUnit::Celsius, "fixture-temp"),
                max_voltage: None,
                max_current: None,
                max_power: None,
            },
            motor: MotorParameters {
                force_factor: q(5.0, PhysicalUnit::TeslaMeter, "fixture-bl"),
                max_linear_excursion: None,
                max_current: None,
            },
            mechanical: MechanicalParameters {
                moving_mass: q(0.02, PhysicalUnit::Kilogram, "fixture-mms"),
                suspension: SuspensionParameter::Compliance(q(
                    0.0005,
                    PhysicalUnit::MeterPerNewton,
                    "fixture-cms",
                )),
                mechanical_resistance: q(
                    2.0,
                    PhysicalUnit::NewtonSecondPerMeter,
                    "fixture-rms",
                ),
                max_mechanical_excursion: None,
            },
            acoustic: AcousticParameters {
                effective_piston_area: q(0.01, PhysicalUnit::SquareMeter, "fixture-sd"),
                radiation_model_id: "unused-by-eac-002".into(),
            },
            thermal: ThermalParameters {
                resistance_temperature_coefficient: None,
                max_coil_temperature: None,
            },
        }
    }

    #[test]
    fn free_air_resonance_matches_closed_form() {
        let fixture = fixture();
        let model = LinearReferenceModel::new(&fixture).unwrap();
        let fs = model.free_air_resonance().unwrap();
        assert!((fs.value - 50.329_212_104_487_034).abs() < 1e-12);
        assert_eq!(fs.unit, "Hz");
        assert_eq!(
            fs.derivation.authority,
            PredictionAuthority::AnalyticalPrediction
        );
        assert_eq!(fs.derivation.input_sources.len(), 2);
    }

    #[test]
    fn free_air_quality_factors_match_closed_form() {
        let fixture = fixture();
        let model = LinearReferenceModel::new(&fixture).unwrap();
        let q = model.quality_factors().unwrap();
        assert!((q.q_mechanical - 3.162_277_660_168_379).abs() < 1e-12);
        assert!((q.q_electrical - 1.517_893_276_880_822_2).abs() < 1e-12);
        assert!((q.q_total - 1.025_603_565_460_014_9).abs() < 1e-12);
    }

    #[test]
    fn resonance_impedance_contains_motional_resistance() {
        let fixture = fixture();
        let model = LinearReferenceModel::new(&fixture).unwrap();
        let fs = model.free_air_resonance().unwrap().value;
        let point = model.response_at(fs, 1.0).unwrap();

        // At exact free-air resonance with Le explicitly set to zero:
        // Zm = Rms and Zin = Re + Bl^2/Rms = 6 + 25/2 = 18.5 ohm.
        assert!((point.input_impedance_ohm.re - 18.5).abs() < 1e-10);
        assert!(point.input_impedance_ohm.im.abs() < 1e-10);
        assert!((point.current_a_rms.magnitude() - 1.0 / 18.5).abs() < 1e-12);
    }

    #[test]
    fn unknown_inductance_does_not_silently_become_zero() {
        let mut fixture = fixture();
        fixture.electrical.voice_coil_inductance = None;
        let model = LinearReferenceModel::new(&fixture).unwrap();
        assert_eq!(
            model.response_at(100.0, 1.0),
            Err(LinearModelError::MissingRequiredParameter(
                "electrical.voice_coil_inductance"
            ))
        );
    }

    #[test]
    fn zero_frequency_is_rejected() {
        let fixture = fixture();
        let model = LinearReferenceModel::new(&fixture).unwrap();
        assert_eq!(
            model.response_at(0.0, 1.0),
            Err(LinearModelError::InvalidFrequency(0.0))
        );
    }

    #[test]
    fn zero_mechanical_loss_keeps_resonance_but_not_finite_q() {
        let mut fixture = fixture();
        fixture.mechanical.mechanical_resistance.value = 0.0;
        let model = LinearReferenceModel::new(&fixture).unwrap();
        assert!(model.free_air_resonance().is_ok());
        assert_eq!(
            model.quality_factors(),
            Err(LinearModelError::UndefinedFiniteQualityFactor)
        );
    }

    #[test]
    fn singular_lossless_resonance_fails_response_closed() {
        let mut fixture = fixture();
        fixture.mechanical.mechanical_resistance.value = 0.0;
        let model = LinearReferenceModel::new(&fixture).unwrap();
        let fs = model.free_air_resonance().unwrap().value;
        assert!(matches!(
            model.response_at(fs, 1.0),
            Err(LinearModelError::SingularComplexQuantity {
                quantity: "mechanical_impedance"
            })
        ));
    }
}
