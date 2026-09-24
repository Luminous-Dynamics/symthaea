// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reduced-order enclosure/load reference models.
//!
//! EAC-003A begins with an intentionally narrow sealed-box model. The enclosure
//! is assumed rigid and lossless; leakage, panel flexure, stuffing losses,
//! diffraction, radiation loading, and distributed cavity modes are not hidden
//! inside fitted constants. Later EAC-003 tranches may add those effects as
//! separately identified approximation families.

use crate::linear::{LinearModelError, LinearReferenceModel, PredictionAuthority};
use crate::{
    ParameterSource, ScalarParameter, SuspensionParameter, TransducerModel, ValidationError,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MODEL_ID: &str = "sealed-rigid-lossless-lumped-v1";

/// One positive scalar whose unit is encoded by its containing field name.
///
/// This is used only for enclosure/environment quantities whose SI units are not
/// yet part of EAC-001's transducer-specific `PhysicalUnit` enum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourcedPositiveScalar {
    pub value: f64,
    pub source: ParameterSource,
}

impl SourcedPositiveScalar {
    pub fn new(value: f64, source: ParameterSource) -> Self {
        Self { value, source }
    }

    fn validate(&self, field: &'static str) -> Result<(), EnclosureModelError> {
        if !self.value.is_finite() || self.value <= 0.0 {
            return Err(EnclosureModelError::InvalidPositiveScalar {
                field,
                value: self.value,
            });
        }
        self.source.validate()?;
        Ok(())
    }
}

/// Air properties used by the reduced-order pneumatic spring model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AirState {
    pub density_kg_per_m3: SourcedPositiveScalar,
    pub sound_speed_m_per_s: SourcedPositiveScalar,
}

impl AirState {
    pub fn validate(&self) -> Result<(), EnclosureModelError> {
        self.density_kg_per_m3.validate("air.density_kg_per_m3")?;
        self.sound_speed_m_per_s
            .validate("air.sound_speed_m_per_s")?;
        Ok(())
    }
}

/// Explicit approximation family for this first sealed-box tranche.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SealedBoundaryModel {
    /// Perfectly rigid enclosure with no leakage or additional dissipative loss.
    IdealRigidLossless,
}

/// Canonical reduced-order sealed enclosure input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SealedEnclosure {
    pub id: String,
    pub net_internal_volume_m3: SourcedPositiveScalar,
    pub boundary_model: SealedBoundaryModel,
}

impl SealedEnclosure {
    pub fn validate(&self) -> Result<(), EnclosureModelError> {
        if self.id.trim().is_empty() {
            return Err(EnclosureModelError::EmptyIdentifier("sealed_enclosure.id"));
        }
        self.net_internal_volume_m3
            .validate("sealed_enclosure.net_internal_volume_m3")?;
        Ok(())
    }
}

/// Exact scalar snapshot used by an enclosure derivation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnclosureInputSnapshot {
    pub field: String,
    pub value: f64,
    pub unit: String,
    pub source: ParameterSource,
}

/// Inspectable analytical receipt for EAC-003A.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnclosureDerivationRecord {
    pub authority: PredictionAuthority,
    pub analytical_model_id: String,
    pub transducer_model_id: String,
    pub enclosure_id: String,
    pub equation_id: String,
    pub inputs: Vec<EnclosureInputSnapshot>,
    pub assumptions: Vec<String>,
}

/// Ideal rigid sealed-box reduced-order result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SealedAlignmentPrediction {
    /// Equivalent acoustic compliance volume of the free-air driver.
    pub equivalent_compliance_volume_vas_m3: f64,
    /// Mechanical compliance contributed by the trapped enclosure air when
    /// referred through the driver's effective piston area.
    pub box_mechanical_compliance_m_per_n: f64,
    /// Parallel combination of suspension and box-air mechanical compliance.
    pub combined_mechanical_compliance_m_per_n: f64,
    /// Classical `alpha = Vas / Vb` ratio.
    pub compliance_ratio_alpha: f64,
    pub free_air_resonance_hz: f64,
    pub sealed_resonance_hz: f64,
    pub free_air_q_total: f64,
    /// Ideal rigid/lossless sealed-system Q under unchanged driver damping.
    pub sealed_q_total_ideal: f64,
    pub derivation: EnclosureDerivationRecord,
}

/// Borrowed ideal sealed-box analytical view.
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

    /// Compute the classical rigid/lossless sealed-box alignment.
    ///
    /// The pneumatic box stiffness is referred to the driver's mechanical side:
    ///
    /// `C_mb = V_b / (rho c^2 S_d^2)`
    ///
    /// and combines with suspension compliance as parallel springs:
    ///
    /// `C_total = 1 / (1/C_ms + 1/C_mb)`.
    pub fn ideal_alignment(&self) -> Result<SealedAlignmentPrediction, EnclosureModelError> {
        let cms = self.transducer.mechanical.suspension.compliance_m_per_n()?;
        let sd = self.transducer.acoustic.effective_piston_area.value;
        let volume = self.enclosure.net_internal_volume_m3.value;
        let rho = self.air.density_kg_per_m3.value;
        let c = self.air.sound_speed_m_per_s.value;

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
        let resonance_multiplier = (1.0 + alpha).sqrt();
        let sealed_resonance = fs * resonance_multiplier;
        let sealed_q = q.q_total * resonance_multiplier;

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
                EnclosureInputSnapshot {
                    field: "sealed_enclosure.net_internal_volume".into(),
                    value: self.enclosure.net_internal_volume_m3.value,
                    unit: "m^3".into(),
                    source: self.enclosure.net_internal_volume_m3.source.clone(),
                },
                EnclosureInputSnapshot {
                    field: "air.density".into(),
                    value: self.air.density_kg_per_m3.value,
                    unit: "kg/m^3".into(),
                    source: self.air.density_kg_per_m3.source.clone(),
                },
                EnclosureInputSnapshot {
                    field: "air.sound_speed".into(),
                    value: self.air.sound_speed_m_per_s.value,
                    unit: "m/s".into(),
                    source: self.air.sound_speed_m_per_s.source.clone(),
                },
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
        unit: format!("{:?}", parameter.unit),
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
    InvalidTransducer(#[from] ValidationError),
    #[error(transparent)]
    LinearModel(#[from] LinearModelError),
    #[error("identifier {0} cannot be empty")]
    EmptyIdentifier(&'static str),
    #[error("{field} must be finite and positive, got {value}")]
    InvalidPositiveScalar { field: &'static str, value: f64 },
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
            net_internal_volume_m3: SourcedPositiveScalar::new(0.01, source("cad-net-volume")),
            boundary_model: SealedBoundaryModel::IdealRigidLossless,
        }
    }

    fn air() -> AirState {
        AirState {
            density_kg_per_m3: SourcedPositiveScalar::new(1.2041, source("air-state")),
            sound_speed_m_per_s: SourcedPositiveScalar::new(343.2, source("air-state")),
        }
    }

    #[test]
    fn sealed_alignment_matches_closed_form_fixture() {
        let driver = driver();
        let enclosure = enclosure();
        let air = air();
        let model = SealedReferenceModel::new(&driver, &enclosure, &air).unwrap();
        let result = model.ideal_alignment().unwrap();

        assert!((result.equivalent_compliance_volume_vas_m3 - 0.007_091_320_579_2).abs() < 1e-14);
        assert!((result.compliance_ratio_alpha - 0.709_132_057_92).abs() < 1e-12);
        assert!((result.sealed_resonance_hz - 65.797_280_169_760_63).abs() < 1e-11);
        assert!((result.sealed_q_total_ideal - 1.340_810_283_291_952_6).abs() < 1e-12);
        assert_eq!(result.derivation.authority, PredictionAuthority::AnalyticalPrediction);
        assert_eq!(result.derivation.inputs.len(), 9);
    }

    #[test]
    fn smaller_box_increases_resonance_and_ideal_q() {
        let driver = driver();
        let air = air();
        let large = SealedEnclosure {
            net_internal_volume_m3: SourcedPositiveScalar::new(0.02, source("large-volume")),
            ..enclosure()
        };
        let small = SealedEnclosure {
            net_internal_volume_m3: SourcedPositiveScalar::new(0.005, source("small-volume")),
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
    fn zero_box_volume_fails_closed() {
        let driver = driver();
        let mut enclosure = enclosure();
        let air = air();
        enclosure.net_internal_volume_m3.value = 0.0;
        assert!(matches!(
            SealedReferenceModel::new(&driver, &enclosure, &air),
            Err(EnclosureModelError::InvalidPositiveScalar {
                field: "sealed_enclosure.net_internal_volume_m3",
                ..
            })
        ));
    }

    #[test]
    fn invalid_air_state_fails_closed() {
        let driver = driver();
        let enclosure = enclosure();
        let mut air = air();
        air.density_kg_per_m3.value = f64::NAN;
        assert!(matches!(
            SealedReferenceModel::new(&driver, &enclosure, &air),
            Err(EnclosureModelError::InvalidPositiveScalar {
                field: "air.density_kg_per_m3",
                ..
            })
        ));
    }
}
