// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reduced-order enclosure/load reference models.
//!
//! EAC-003A begins with an intentionally narrow sealed-box model. The enclosure
//! is assumed rigid and lossless; leakage, panel flexure, stuffing losses,
//! diffraction, radiation loading, and distributed cavity modes are not hidden
//! inside fitted constants.

use crate::linear::{LinearModelError, LinearReferenceModel, PredictionAuthority};
use crate::{
    ParameterSource, PhysicalUnit, ScalarParameter, SuspensionParameter, TransducerModel,
    UncertaintyInterval, ValidationError,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MODEL_ID: &str = "sealed-rigid-lossless-lumped-v1";

/// Explicit SI unit vocabulary for EAC-003 enclosure/environment quantities.
///
/// This is deliberately separate from the transducer-focused EAC-001 unit enum
/// until a broader shared engineering quantity system is promoted. The important
/// invariant is that the unit is data, never implied only by a Rust field name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnclosureUnit {
    CubicMeter,
    KilogramPerCubicMeter,
    MeterPerSecond,
}

/// Positive scalar with explicit unit, source provenance and optional absolute
/// uncertainty interval in the same declared unit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourcedPositiveScalar {
    pub value: f64,
    pub unit: EnclosureUnit,
    pub source: ParameterSource,
    pub uncertainty: Option<UncertaintyInterval>,
}

impl SourcedPositiveScalar {
    pub fn new(value: f64, unit: EnclosureUnit, source: ParameterSource) -> Self {
        Self {
            value,
            unit,
            source,
            uncertainty: None,
        }
    }

    pub fn with_uncertainty(mut self, uncertainty: UncertaintyInterval) -> Self {
        self.uncertainty = Some(uncertainty);
        self
    }

    fn validate(
        &self,
        field: &'static str,
        expected_unit: EnclosureUnit,
    ) -> Result<(), EnclosureModelError> {
        if !self.value.is_finite() || self.value <= 0.0 {
            return Err(EnclosureModelError::InvalidPositiveScalar {
                field,
                value: self.value,
            });
        }
        if self.unit != expected_unit {
            return Err(EnclosureModelError::WrongUnit {
                field,
                expected: expected_unit,
                actual: self.unit,
            });
        }
        self.source.validate()?;
        if let Some(interval) = self.uncertainty {
            interval.validate()?;
            if !interval.contains(self.value) {
                return Err(EnclosureModelError::UncertaintyDoesNotContainValue {
                    field,
                    value: self.value,
                    lower: interval.lower,
                    upper: interval.upper,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AirState {
    pub density: SourcedPositiveScalar,
    pub sound_speed: SourcedPositiveScalar,
}

impl AirState {
    pub fn validate(&self) -> Result<(), EnclosureModelError> {
        self.density.validate(
            "air.density",
            EnclosureUnit::KilogramPerCubicMeter,
        )?;
        self.sound_speed
            .validate("air.sound_speed", EnclosureUnit::MeterPerSecond)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SealedBoundaryModel {
    IdealRigidLossless,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SealedEnclosure {
    pub id: String,
    pub net_internal_volume: SourcedPositiveScalar,
    pub boundary_model: SealedBoundaryModel,
}

impl SealedEnclosure {
    pub fn validate(&self) -> Result<(), EnclosureModelError> {
        if self.id.trim().is_empty() {
            return Err(EnclosureModelError::EmptyIdentifier("sealed_enclosure.id"));
        }
        self.net_internal_volume.validate(
            "sealed_enclosure.net_internal_volume",
            EnclosureUnit::CubicMeter,
        )?;
        Ok(())
    }
}

/// Unit carried by a serialized derivation input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnclosureInputUnit {
    Physical(PhysicalUnit),
    Enclosure(EnclosureUnit),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnclosureInputSnapshot {
    pub field: String,
    pub value: f64,
    pub unit: EnclosureInputUnit,
    pub source: ParameterSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnclosureDerivationRecord {
    pub authority: PredictionAuthority,
    pub analytical_model_id: String,
    pub transducer_model_id: String,
    pub enclosure_id: String,
    pub boundary_model: SealedBoundaryModel,
    pub equation_id: String,
    pub inputs: Vec<EnclosureInputSnapshot>,
    pub assumptions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SealedAlignmentPrediction {
    pub equivalent_compliance_volume_vas_m3: f64,
    pub box_mechanical_compliance_m_per_n: f64,
    pub combined_mechanical_compliance_m_per_n: f64,
    pub compliance_ratio_alpha: f64,
    pub free_air_resonance_hz: f64,
    pub sealed_resonance_hz: f64,
    pub free_air_q_total: f64,
    pub sealed_q_total_ideal: f64,
    pub derivation: EnclosureDerivationRecord,
}

pub struct SealedReferenceModel<'a> {
    transducer: &'a TransducerModel,
    enclosure: &'a SealedEnclosure,
    air: &'a AirState,
}

impl<'a> SealedReferenceModel<'a> {
    pub fn new(
        transducer: &'a TransducerModel,
        enclosure: &'a SealedEnclosure,
        air: &'a AirState,
    ) -> Result<Self, EnclosureModelError> {
        transducer.validate()?;
        enclosure.validate()?;
        air.validate()?;
        Ok(Self {
            transducer,
            enclosure,
            air,
        })
    }

    /// Classical rigid/lossless sealed-box reference:
    ///
    /// `Cmb = Vb / (rho c^2 Sd^2)`
    ///
    /// `Ctotal = 1 / (1/Cms + 1/Cmb)`
    ///
    /// `Vas = rho c^2 Sd^2 Cms`, `alpha = Vas / Vb`.
    pub fn ideal_alignment(&self) -> Result<SealedAlignmentPrediction, EnclosureModelError> {
        let cms = self.transducer.mechanical.suspension.compliance_m_per_n()?;
        let sd = self.transducer.acoustic.effective_piston_area.value;
        let volume = self.enclosure.net_internal_volume.value;
        let rho = self.air.density.value;
        let c = self.air.sound_speed.value;

        let vas = rho * c * c * sd * sd * cms;
        let box_compliance = volume / (rho * c * c * sd * sd);
        let combined_compliance = 1.0 / (1.0 / cms + 1.0 / box_compliance);
        let alpha = vas / volume;

        for (quantity, value) in [
            ("equivalent_compliance_volume_vas_m3", vas),
            ("box_mechanical_compliance_m_per_n", box_compliance),
            ("combined_mechanical_compliance_m_per_n", combined_compliance),
            ("compliance_ratio_alpha", alpha),
        ] {
            ensure_finite_positive(value, quantity)?;
        }

        let linear = LinearReferenceModel::new(self.transducer)?;
        let fs = linear.free_air_resonance()?.value;
        let q = linear.quality_factors()?;
        let multiplier = (1.0 + alpha).sqrt();
        let sealed_resonance = fs * multiplier;
        let sealed_q = q.q_total * multiplier;

        ensure_finite_positive(sealed_resonance, "sealed_resonance_hz")?;
        ensure_finite_positive(sealed_q, "sealed_q_total_ideal")?;

        Ok(SealedAlignmentPrediction {
            equivalent_compliance_volume_vas_m3: vas,
            box_mechanical_compliance_m_per_n: box_compliance,
            combined_mechanical_compliance_m_per_n: combined_compliance,
            compliance_ratio_alpha: alpha,
            free_air_resonance_hz: fs,
            sealed_resonance_hz: sealed_resonance,
            free_air_q_total: q.q_total,
            sealed_q_total_ideal: sealed_q,
            derivation: self.derivation(),
        })
    }

    fn derivation(&self) -> EnclosureDerivationRecord {
        EnclosureDerivationRecord {
            authority: PredictionAuthority::AnalyticalPrediction,
            analytical_model_id: MODEL_ID.into(),
            transducer_model_id: self.transducer.id.clone(),
            enclosure_id: self.enclosure.id.clone(),
            boundary_model: self.enclosure.boundary_model,
            equation_id: "sealed-compliance-alignment-v1".into(),
            inputs: vec![
                driver_snapshot(
                    "mechanical.suspension",
                    suspension_parameter(&self.transducer.mechanical.suspension),
                ),
                driver_snapshot(
                    "acoustic.effective_piston_area",
                    &self.transducer.acoustic.effective_piston_area,
                ),
                driver_snapshot(
                    "electrical.voice_coil_resistance",
                    &self.transducer.electrical.voice_coil_resistance,
                ),
                driver_snapshot("motor.force_factor", &self.transducer.motor.force_factor),
                driver_snapshot(
                    "mechanical.moving_mass",
                    &self.transducer.mechanical.moving_mass,
                ),
                driver_snapshot(
                    "mechanical.mechanical_resistance",
                    &self.transducer.mechanical.mechanical_resistance,
                ),
                enclosure_snapshot(
                    "sealed_enclosure.net_internal_volume",
                    &self.enclosure.net_internal_volume,
                ),
                enclosure_snapshot("air.density", &self.air.density),
                enclosure_snapshot("air.sound_speed", &self.air.sound_speed),
            ],
            assumptions: vec![
                "rigid enclosure walls".into(),
                "perfectly sealed enclosure; no leakage loss".into(),
                "no stuffing/porous-loss model".into(),
                "uniform lumped cavity pressure; no distributed cavity modes".into(),
                "driver small-signal parameters are state-invariant".into(),
                "box air adds stiffness but no additional moving mass or dissipative damping".into(),
                "classical Q scaling neglects voice-coil inductance".into(),
                "no baffle diffraction, radiation loading, or room interaction".into(),
            ],
        }
    }
}

fn suspension_parameter(suspension: &SuspensionParameter) -> &ScalarParameter {
    match suspension {
        SuspensionParameter::Compliance(parameter) | SuspensionParameter::Stiffness(parameter) => {
            parameter
        }
    }
}

fn driver_snapshot(field: &str, parameter: &ScalarParameter) -> EnclosureInputSnapshot {
    EnclosureInputSnapshot {
        field: field.into(),
        value: parameter.value,
        unit: EnclosureInputUnit::Physical(parameter.unit),
        source: parameter.source.clone(),
    }
}

fn enclosure_snapshot(field: &str, parameter: &SourcedPositiveScalar) -> EnclosureInputSnapshot {
    EnclosureInputSnapshot {
        field: field.into(),
        value: parameter.value,
        unit: EnclosureInputUnit::Enclosure(parameter.unit),
        source: parameter.source.clone(),
    }
}

fn ensure_finite_positive(value: f64, quantity: &'static str) -> Result<(), EnclosureModelError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(EnclosureModelError::InvalidDerivedValue { quantity, value });
    }
    Ok(())
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum EnclosureModelError {
    #[error(transparent)]
    InvalidParameter(#[from] ValidationError),
    #[error(transparent)]
    LinearModel(#[from] LinearModelError),
    #[error("identifier {0} cannot be empty")]
    EmptyIdentifier(&'static str),
    #[error("{field} must be finite and positive, got {value}")]
    InvalidPositiveScalar { field: &'static str, value: f64 },
    #[error("{field} has wrong unit: expected {expected:?}, got {actual:?}")]
    WrongUnit {
        field: &'static str,
        expected: EnclosureUnit,
        actual: EnclosureUnit,
    },
    #[error("uncertainty for {field} [{lower}, {upper}] does not contain {value}")]
    UncertaintyDoesNotContainValue {
        field: &'static str,
        value: f64,
        lower: f64,
        upper: f64,
    },
    #[error("derived {quantity} is invalid: {value}")]
    InvalidDerivedValue { quantity: &'static str, value: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AcousticParameters, ElectricalParameters, MechanicalParameters, MotorParameters,
        ParameterSourceKind, ThermalParameters,
    };

    fn source(name: &str) -> ParameterSource {
        ParameterSource::new(ParameterSourceKind::Datasheet, name)
    }

    fn q(value: f64, unit: PhysicalUnit, name: &str) -> ScalarParameter {
        ScalarParameter::new(value, unit, source(name))
    }

    fn driver() -> TransducerModel {
        TransducerModel {
            id: "sealed-fixture-driver".into(),
            electrical: ElectricalParameters {
                voice_coil_resistance: q(6.0, PhysicalUnit::Ohm, "fixture-re"),
                voice_coil_inductance: Some(q(0.0, PhysicalUnit::Henry, "fixture-le")),
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
                radiation_model_id: "unused-by-sealed-reference".into(),
            },
            thermal: ThermalParameters {
                resistance_temperature_coefficient: None,
                max_coil_temperature: None,
            },
        }
    }

    fn enclosure() -> SealedEnclosure {
        SealedEnclosure {
            id: "sealed-box-10l".into(),
            net_internal_volume: SourcedPositiveScalar::new(
                0.01,
                EnclosureUnit::CubicMeter,
                source("cad-net-volume"),
            ),
            boundary_model: SealedBoundaryModel::IdealRigidLossless,
        }
    }

    fn air() -> AirState {
        AirState {
            density: SourcedPositiveScalar::new(
                1.2041,
                EnclosureUnit::KilogramPerCubicMeter,
                source("air-state-density"),
            ),
            sound_speed: SourcedPositiveScalar::new(
                343.2,
                EnclosureUnit::MeterPerSecond,
                source("air-state-speed"),
            ),
        }
    }

    #[test]
    fn sealed_alignment_matches_closed_form_fixture() {
        let driver = driver();
        let enclosure = enclosure();
        let air = air();
        let result = SealedReferenceModel::new(&driver, &enclosure, &air)
            .unwrap()
            .ideal_alignment()
            .unwrap();

        assert!(
            (result.equivalent_compliance_volume_vas_m3 - 0.007_091_320_579_2).abs() < 1e-14
        );
        assert!((result.compliance_ratio_alpha - 0.709_132_057_92).abs() < 1e-12);
        assert!((result.sealed_resonance_hz - 65.797_280_169_760_63).abs() < 1e-11);
        assert!((result.sealed_q_total_ideal - 1.340_810_283_291_952_6).abs() < 1e-12);
        assert!(
            result.combined_mechanical_compliance_m_per_n
                < driver.mechanical.suspension.compliance_m_per_n().unwrap()
        );
        assert_eq!(
            result.derivation.inputs[6].unit,
            EnclosureInputUnit::Enclosure(EnclosureUnit::CubicMeter)
        );
    }

    #[test]
    fn wrong_volume_unit_fails_closed() {
        let driver = driver();
        let mut enclosure = enclosure();
        let air = air();
        enclosure.net_internal_volume.unit = EnclosureUnit::MeterPerSecond;
        assert!(matches!(
            SealedReferenceModel::new(&driver, &enclosure, &air),
            Err(EnclosureModelError::WrongUnit {
                field: "sealed_enclosure.net_internal_volume",
                ..
            })
        ));
    }

    #[test]
    fn wrong_air_unit_fails_closed() {
        let driver = driver();
        let enclosure = enclosure();
        let mut air = air();
        air.density.unit = EnclosureUnit::CubicMeter;
        assert!(matches!(
            SealedReferenceModel::new(&driver, &enclosure, &air),
            Err(EnclosureModelError::WrongUnit {
                field: "air.density",
                ..
            })
        ));
    }

    #[test]
    fn smaller_box_increases_resonance_and_ideal_q() {
        let driver = driver();
        let air = air();
        let large = SealedEnclosure {
            net_internal_volume: SourcedPositiveScalar::new(
                0.02,
                EnclosureUnit::CubicMeter,
                source("large-volume"),
            ),
            ..enclosure()
        };
        let small = SealedEnclosure {
            net_internal_volume: SourcedPositiveScalar::new(
                0.005,
                EnclosureUnit::CubicMeter,
                source("small-volume"),
            ),
            ..enclosure()
        };
        let large_result = SealedReferenceModel::new(&driver, &large, &air)
            .unwrap()
            .ideal_alignment()
            .unwrap();
        let small_result = SealedReferenceModel::new(&driver, &small, &air)
            .unwrap()
            .ideal_alignment()
            .unwrap();
        assert!(small_result.sealed_resonance_hz > large_result.sealed_resonance_hz);
        assert!(small_result.sealed_q_total_ideal > large_result.sealed_q_total_ideal);
    }

    #[test]
    fn very_large_box_approaches_free_air_limit() {
        let driver = driver();
        let air = air();
        let huge = SealedEnclosure {
            id: "sealed-box-huge".into(),
            net_internal_volume: SourcedPositiveScalar::new(
                1_000.0,
                EnclosureUnit::CubicMeter,
                source("huge-volume"),
            ),
            boundary_model: SealedBoundaryModel::IdealRigidLossless,
        };
        let result = SealedReferenceModel::new(&driver, &huge, &air)
            .unwrap()
            .ideal_alignment()
            .unwrap();
        assert!((result.sealed_resonance_hz / result.free_air_resonance_hz - 1.0).abs() < 1e-5);
        assert!((result.sealed_q_total_ideal / result.free_air_q_total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn zero_box_volume_fails_closed() {
        let driver = driver();
        let mut enclosure = enclosure();
        let air = air();
        enclosure.net_internal_volume.value = 0.0;
        assert!(matches!(
            SealedReferenceModel::new(&driver, &enclosure, &air),
            Err(EnclosureModelError::InvalidPositiveScalar {
                field: "sealed_enclosure.net_internal_volume",
                ..
            })
        ));
    }

    #[test]
    fn uncertainty_must_contain_nominal_value() {
        let driver = driver();
        let mut enclosure = enclosure();
        let air = air();
        enclosure.net_internal_volume.uncertainty =
            Some(UncertaintyInterval::new(0.02, 0.03).unwrap());
        assert!(matches!(
            SealedReferenceModel::new(&driver, &enclosure, &air),
            Err(EnclosureModelError::UncertaintyDoesNotContainValue { .. })
        ));
    }

    #[test]
    fn empty_environment_provenance_fails_closed() {
        let driver = driver();
        let enclosure = enclosure();
        let mut air = air();
        air.sound_speed.source = ParameterSource::new(ParameterSourceKind::Datasheet, "");
        assert!(matches!(
            SealedReferenceModel::new(&driver, &enclosure, &air),
            Err(EnclosureModelError::InvalidParameter(
                ValidationError::MissingProvenance
            ))
        ));
    }
}
