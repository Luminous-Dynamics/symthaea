// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-oriented electro-acoustic transducer models for Symthaea.
//!
//! This crate deliberately starts below the simulation layer. It defines typed
//! transducer parameters, explicit units, provenance, uncertainty and validation.
//! Numerical solvers remain external authorities behind `symthaea-sim-bridge`,
//! while calibrated physical observations belong to the FIELD architecture.
//!
//! # Authority boundary
//!
//! A parameter value is not promoted merely because it is present in a model.
//! `ParameterSourceKind` preserves whether it was assumed, copied from a
//! datasheet, analytically/numerically derived, directly measured, or fitted
//! from measurement. Unknown optional quantities remain `None` rather than
//! silently becoming zero.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Explicit unit carried by every scalar engineering parameter in EAC-001.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhysicalUnit {
    Ohm,
    Henry,
    Volt,
    Ampere,
    Watt,
    Celsius,
    PerCelsius,
    TeslaMeter,
    Kilogram,
    MeterPerNewton,
    NewtonPerMeter,
    NewtonSecondPerMeter,
    Meter,
    SquareMeter,
}

/// Authority/source class for a model parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParameterSourceKind {
    /// Engineering assumption or placeholder supplied explicitly by an author.
    Assumed,
    /// Published component/manufacturer data.
    Datasheet,
    /// Derived from inspectable analytical equations.
    AnalyticalDerived,
    /// Derived from a numerical solver result.
    NumericalDerived,
    /// Directly obtained from calibrated physical measurement.
    Measured,
    /// Estimated/fitted from one or more physical measurements.
    FittedFromMeasurement,
}

/// Provenance attached to one engineering parameter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterSource {
    pub kind: ParameterSourceKind,
    /// Human/audit-readable source identity. Must never be empty.
    pub provenance: String,
    /// Addressable source evidence where available (receipt, measurement run,
    /// solver artifact, datasheet digest, etc.).
    pub evidence_ref: Option<String>,
    /// Method identity for derived/fitted values.
    pub method: Option<String>,
}

impl ParameterSource {
    pub fn new(kind: ParameterSourceKind, provenance: impl Into<String>) -> Self {
        Self {
            kind,
            provenance: provenance.into(),
            evidence_ref: None,
            method: None,
        }
    }

    pub fn with_evidence_ref(mut self, evidence_ref: impl Into<String>) -> Self {
        self.evidence_ref = Some(evidence_ref.into());
        self
    }

    pub fn with_method(mut self, method: impl Into<String>) -> Self {
        self.method = Some(method.into());
        self
    }

    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.provenance.trim().is_empty() {
            return Err(ValidationError::MissingProvenance);
        }

        if matches!(
            self.kind,
            ParameterSourceKind::Measured | ParameterSourceKind::FittedFromMeasurement
        ) && self
            .evidence_ref
            .as_ref()
            .is_none_or(|value| value.trim().is_empty())
        {
            return Err(ValidationError::MissingEvidenceReference(self.kind));
        }

        if matches!(
            self.kind,
            ParameterSourceKind::AnalyticalDerived
                | ParameterSourceKind::NumericalDerived
                | ParameterSourceKind::FittedFromMeasurement
        ) && self
            .method
            .as_ref()
            .is_none_or(|value| value.trim().is_empty())
        {
            return Err(ValidationError::MissingMethod(self.kind));
        }

        Ok(())
    }
}

/// Optional absolute uncertainty interval in the same unit as the parameter.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyInterval {
    pub lower: f64,
    pub upper: f64,
}

impl UncertaintyInterval {
    pub fn new(lower: f64, upper: f64) -> Result<Self, ValidationError> {
        let interval = Self { lower, upper };
        interval.validate()?;
        Ok(interval)
    }

    pub fn validate(&self) -> Result<(), ValidationError> {
        if !self.lower.is_finite() || !self.upper.is_finite() || self.lower > self.upper {
            return Err(ValidationError::InvalidUncertaintyInterval {
                lower: self.lower,
                upper: self.upper,
            });
        }
        Ok(())
    }

    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }
}

/// One scalar parameter with explicit unit, source and optional uncertainty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarParameter {
    pub value: f64,
    pub unit: PhysicalUnit,
    pub source: ParameterSource,
    pub uncertainty: Option<UncertaintyInterval>,
}

impl ScalarParameter {
    pub fn new(value: f64, unit: PhysicalUnit, source: ParameterSource) -> Self {
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

    fn validate_common(&self) -> Result<(), ValidationError> {
        if !self.value.is_finite() {
            return Err(ValidationError::NonFiniteValue);
        }
        self.source.validate()?;
        if let Some(interval) = self.uncertainty {
            interval.validate()?;
            if !interval.contains(self.value) {
                return Err(ValidationError::UncertaintyDoesNotContainValue {
                    value: self.value,
                    lower: interval.lower,
                    upper: interval.upper,
                });
            }
        }
        Ok(())
    }

    fn require_unit(&self, expected: PhysicalUnit) -> Result<(), ValidationError> {
        if self.unit != expected {
            return Err(ValidationError::WrongUnit {
                expected,
                actual: self.unit,
            });
        }
        Ok(())
    }

    fn require_positive(&self, field: &'static str) -> Result<(), ValidationError> {
        if self.value <= 0.0 {
            return Err(ValidationError::MustBePositive {
                field,
                value: self.value,
            });
        }
        Ok(())
    }

    fn require_nonnegative(&self, field: &'static str) -> Result<(), ValidationError> {
        if self.value < 0.0 {
            return Err(ValidationError::MustBeNonnegative {
                field,
                value: self.value,
            });
        }
        Ok(())
    }
}

/// Electrical small-signal/reference parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectricalParameters {
    pub voice_coil_resistance: ScalarParameter,
    pub voice_coil_inductance: Option<ScalarParameter>,
    pub reference_temperature: ScalarParameter,
    pub max_voltage: Option<ScalarParameter>,
    pub max_current: Option<ScalarParameter>,
    pub max_power: Option<ScalarParameter>,
}

/// Moving-coil motor parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotorParameters {
    pub force_factor: ScalarParameter,
    pub max_linear_excursion: Option<ScalarParameter>,
    pub max_current: Option<ScalarParameter>,
}

/// Canonical suspension source: compliance OR stiffness, never two independent
/// source-of-truth values that can disagree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SuspensionParameter {
    Compliance(ScalarParameter),
    Stiffness(ScalarParameter),
}

impl SuspensionParameter {
    pub fn compliance_m_per_n(&self) -> Result<f64, ValidationError> {
        match self {
            Self::Compliance(value) => {
                value.validate_common()?;
                value.require_unit(PhysicalUnit::MeterPerNewton)?;
                value.require_positive("suspension_compliance")?;
                Ok(value.value)
            }
            Self::Stiffness(value) => {
                value.validate_common()?;
                value.require_unit(PhysicalUnit::NewtonPerMeter)?;
                value.require_positive("suspension_stiffness")?;
                Ok(1.0 / value.value)
            }
        }
    }

    pub fn stiffness_n_per_m(&self) -> Result<f64, ValidationError> {
        match self {
            Self::Compliance(value) => {
                value.validate_common()?;
                value.require_unit(PhysicalUnit::MeterPerNewton)?;
                value.require_positive("suspension_compliance")?;
                Ok(1.0 / value.value)
            }
            Self::Stiffness(value) => {
                value.validate_common()?;
                value.require_unit(PhysicalUnit::NewtonPerMeter)?;
                value.require_positive("suspension_stiffness")?;
                Ok(value.value)
            }
        }
    }

    fn validate(&self) -> Result<(), ValidationError> {
        let _ = self.compliance_m_per_n()?;
        Ok(())
    }
}

/// Mechanical moving-system parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MechanicalParameters {
    pub moving_mass: ScalarParameter,
    pub suspension: SuspensionParameter,
    pub mechanical_resistance: ScalarParameter,
    pub max_mechanical_excursion: Option<ScalarParameter>,
}

/// Acoustic coupling parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticParameters {
    pub effective_piston_area: ScalarParameter,
    /// Stable identity of the radiation/load approximation or model family.
    pub radiation_model_id: String,
}

/// Reference thermal parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermalParameters {
    pub resistance_temperature_coefficient: Option<ScalarParameter>,
    pub max_coil_temperature: Option<ScalarParameter>,
}

/// Canonical EAC-001 transducer parameter model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransducerModel {
    pub id: String,
    pub electrical: ElectricalParameters,
    pub motor: MotorParameters,
    pub mechanical: MechanicalParameters,
    pub acoustic: AcousticParameters,
    pub thermal: ThermalParameters,
}

impl TransducerModel {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.id.trim().is_empty() {
            return Err(ValidationError::EmptyIdentifier("transducer.id"));
        }

        validate_positive_unit(
            &self.electrical.voice_coil_resistance,
            PhysicalUnit::Ohm,
            "voice_coil_resistance",
        )?;
        validate_optional_nonnegative_unit(
            self.electrical.voice_coil_inductance.as_ref(),
            PhysicalUnit::Henry,
            "voice_coil_inductance",
        )?;
        validate_any_unit(
            &self.electrical.reference_temperature,
            PhysicalUnit::Celsius,
        )?;
        validate_optional_positive_unit(
            self.electrical.max_voltage.as_ref(),
            PhysicalUnit::Volt,
            "max_voltage",
        )?;
        validate_optional_positive_unit(
            self.electrical.max_current.as_ref(),
            PhysicalUnit::Ampere,
            "electrical_max_current",
        )?;
        validate_optional_positive_unit(
            self.electrical.max_power.as_ref(),
            PhysicalUnit::Watt,
            "max_power",
        )?;

        validate_positive_unit(
            &self.motor.force_factor,
            PhysicalUnit::TeslaMeter,
            "force_factor",
        )?;
        validate_optional_positive_unit(
            self.motor.max_linear_excursion.as_ref(),
            PhysicalUnit::Meter,
            "max_linear_excursion",
        )?;
        validate_optional_positive_unit(
            self.motor.max_current.as_ref(),
            PhysicalUnit::Ampere,
            "motor_max_current",
        )?;

        validate_positive_unit(
            &self.mechanical.moving_mass,
            PhysicalUnit::Kilogram,
            "moving_mass",
        )?;
        self.mechanical.suspension.validate()?;
        validate_nonnegative_unit(
            &self.mechanical.mechanical_resistance,
            PhysicalUnit::NewtonSecondPerMeter,
            "mechanical_resistance",
        )?;
        validate_optional_positive_unit(
            self.mechanical.max_mechanical_excursion.as_ref(),
            PhysicalUnit::Meter,
            "max_mechanical_excursion",
        )?;

        validate_positive_unit(
            &self.acoustic.effective_piston_area,
            PhysicalUnit::SquareMeter,
            "effective_piston_area",
        )?;
        if self.acoustic.radiation_model_id.trim().is_empty() {
            return Err(ValidationError::EmptyIdentifier(
                "acoustic.radiation_model_id",
            ));
        }

        validate_optional_any_unit(
            self.thermal.resistance_temperature_coefficient.as_ref(),
            PhysicalUnit::PerCelsius,
        )?;
        validate_optional_any_unit(
            self.thermal.max_coil_temperature.as_ref(),
            PhysicalUnit::Celsius,
        )?;

        if let Some(max_temp) = &self.thermal.max_coil_temperature {
            if max_temp.value <= self.electrical.reference_temperature.value {
                return Err(ValidationError::ThermalLimitNotAboveReference {
                    reference_c: self.electrical.reference_temperature.value,
                    maximum_c: max_temp.value,
                });
            }
        }

        Ok(())
    }
}

fn validate_any_unit(value: &ScalarParameter, unit: PhysicalUnit) -> Result<(), ValidationError> {
    value.validate_common()?;
    value.require_unit(unit)
}

fn validate_positive_unit(
    value: &ScalarParameter,
    unit: PhysicalUnit,
    field: &'static str,
) -> Result<(), ValidationError> {
    validate_any_unit(value, unit)?;
    value.require_positive(field)
}

fn validate_nonnegative_unit(
    value: &ScalarParameter,
    unit: PhysicalUnit,
    field: &'static str,
) -> Result<(), ValidationError> {
    validate_any_unit(value, unit)?;
    value.require_nonnegative(field)
}

fn validate_optional_any_unit(
    value: Option<&ScalarParameter>,
    unit: PhysicalUnit,
) -> Result<(), ValidationError> {
    if let Some(value) = value {
        validate_any_unit(value, unit)?;
    }
    Ok(())
}

fn validate_optional_positive_unit(
    value: Option<&ScalarParameter>,
    unit: PhysicalUnit,
    field: &'static str,
) -> Result<(), ValidationError> {
    if let Some(value) = value {
        validate_positive_unit(value, unit, field)?;
    }
    Ok(())
}

fn validate_optional_nonnegative_unit(
    value: Option<&ScalarParameter>,
    unit: PhysicalUnit,
    field: &'static str,
) -> Result<(), ValidationError> {
    if let Some(value) = value {
        validate_nonnegative_unit(value, unit, field)?;
    }
    Ok(())
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ValidationError {
    #[error("parameter value must be finite")]
    NonFiniteValue,
    #[error("parameter provenance cannot be empty")]
    MissingProvenance,
    #[error("{0:?} parameters require an evidence reference")]
    MissingEvidenceReference(ParameterSourceKind),
    #[error("{0:?} parameters require a method identity")]
    MissingMethod(ParameterSourceKind),
    #[error("wrong physical unit: expected {expected:?}, got {actual:?}")]
    WrongUnit {
        expected: PhysicalUnit,
        actual: PhysicalUnit,
    },
    #[error("{field} must be positive, got {value}")]
    MustBePositive { field: &'static str, value: f64 },
    #[error("{field} must be nonnegative, got {value}")]
    MustBeNonnegative { field: &'static str, value: f64 },
    #[error("invalid uncertainty interval [{lower}, {upper}]")]
    InvalidUncertaintyInterval { lower: f64, upper: f64 },
    #[error("uncertainty interval [{lower}, {upper}] does not contain value {value}")]
    UncertaintyDoesNotContainValue {
        value: f64,
        lower: f64,
        upper: f64,
    },
    #[error("identifier {0} cannot be empty")]
    EmptyIdentifier(&'static str),
    #[error(
        "maximum coil temperature {maximum_c} C must exceed reference temperature {reference_c} C"
    )]
    ThermalLimitNotAboveReference { reference_c: f64, maximum_c: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn datasheet(name: &str) -> ParameterSource {
        ParameterSource::new(ParameterSourceKind::Datasheet, name)
    }

    fn measured(name: &str) -> ParameterSource {
        ParameterSource::new(ParameterSourceKind::Measured, name)
            .with_evidence_ref(format!("measurement:{name}"))
    }

    fn q(value: f64, unit: PhysicalUnit, source: ParameterSource) -> ScalarParameter {
        ScalarParameter::new(value, unit, source)
    }

    fn valid_model() -> TransducerModel {
        TransducerModel {
            id: "driver-reference-001".into(),
            electrical: ElectricalParameters {
                voice_coil_resistance: q(5.8, PhysicalUnit::Ohm, measured("dc-resistance")),
                voice_coil_inductance: Some(q(
                    0.00035,
                    PhysicalUnit::Henry,
                    datasheet("manufacturer-sheet-r1"),
                )),
                reference_temperature: q(
                    20.0,
                    PhysicalUnit::Celsius,
                    datasheet("manufacturer-sheet-r1"),
                ),
                max_voltage: None,
                max_current: Some(q(
                    4.0,
                    PhysicalUnit::Ampere,
                    datasheet("manufacturer-sheet-r1"),
                )),
                max_power: Some(q(
                    80.0,
                    PhysicalUnit::Watt,
                    datasheet("manufacturer-sheet-r1"),
                )),
            },
            motor: MotorParameters {
                force_factor: q(
                    6.7,
                    PhysicalUnit::TeslaMeter,
                    measured("small-signal-fit"),
                ),
                max_linear_excursion: Some(q(
                    0.006,
                    PhysicalUnit::Meter,
                    datasheet("manufacturer-sheet-r1"),
                )),
                max_current: None,
            },
            mechanical: MechanicalParameters {
                moving_mass: q(
                    0.014,
                    PhysicalUnit::Kilogram,
                    measured("moving-mass"),
                ),
                suspension: SuspensionParameter::Compliance(q(
                    0.0007,
                    PhysicalUnit::MeterPerNewton,
                    measured("compliance-fit"),
                )),
                mechanical_resistance: q(
                    1.2,
                    PhysicalUnit::NewtonSecondPerMeter,
                    measured("mechanical-resistance-fit"),
                ),
                max_mechanical_excursion: Some(q(
                    0.009,
                    PhysicalUnit::Meter,
                    datasheet("manufacturer-sheet-r1"),
                )),
            },
            acoustic: AcousticParameters {
                effective_piston_area: q(
                    0.013,
                    PhysicalUnit::SquareMeter,
                    measured("cone-area"),
                ),
                radiation_model_id: "piston-in-infinite-baffle-v1".into(),
            },
            thermal: ThermalParameters {
                resistance_temperature_coefficient: Some(q(
                    0.00393,
                    PhysicalUnit::PerCelsius,
                    datasheet("copper-reference"),
                )),
                max_coil_temperature: Some(q(
                    180.0,
                    PhysicalUnit::Celsius,
                    datasheet("manufacturer-sheet-r1"),
                )),
            },
        }
    }

    #[test]
    fn valid_model_passes() {
        valid_model().validate().unwrap();
    }

    #[test]
    fn wrong_unit_fails_closed() {
        let mut model = valid_model();
        model.electrical.voice_coil_resistance.unit = PhysicalUnit::Henry;
        assert!(matches!(
            model.validate(),
            Err(ValidationError::WrongUnit {
                expected: PhysicalUnit::Ohm,
                actual: PhysicalUnit::Henry,
            })
        ));
    }

    #[test]
    fn measured_requires_evidence_reference() {
        let source = ParameterSource::new(ParameterSourceKind::Measured, "bench");
        assert!(matches!(
            source.validate(),
            Err(ValidationError::MissingEvidenceReference(
                ParameterSourceKind::Measured
            ))
        ));
    }

    #[test]
    fn fitted_value_requires_measurement_and_method() {
        let source = ParameterSource::new(ParameterSourceKind::FittedFromMeasurement, "fit")
            .with_evidence_ref("measurement:run-1");
        assert!(matches!(
            source.validate(),
            Err(ValidationError::MissingMethod(
                ParameterSourceKind::FittedFromMeasurement
            ))
        ));

        source.with_method("least-squares-v1").validate().unwrap();
    }

    #[test]
    fn suspension_conversion_is_explicit_and_reciprocal() {
        let compliance = SuspensionParameter::Compliance(q(
            0.0005,
            PhysicalUnit::MeterPerNewton,
            datasheet("sheet"),
        ));
        assert!((compliance.stiffness_n_per_m().unwrap() - 2000.0).abs() < 1e-12);

        let stiffness = SuspensionParameter::Stiffness(q(
            2000.0,
            PhysicalUnit::NewtonPerMeter,
            datasheet("sheet"),
        ));
        assert!((stiffness.compliance_m_per_n().unwrap() - 0.0005).abs() < 1e-12);
    }

    #[test]
    fn zero_suspension_is_rejected() {
        let suspension = SuspensionParameter::Compliance(q(
            0.0,
            PhysicalUnit::MeterPerNewton,
            datasheet("sheet"),
        ));
        assert!(matches!(
            suspension.compliance_m_per_n(),
            Err(ValidationError::MustBePositive { .. })
        ));
    }

    #[test]
    fn uncertainty_must_contain_nominal_value() {
        let value = q(5.8, PhysicalUnit::Ohm, datasheet("sheet"))
            .with_uncertainty(UncertaintyInterval::new(5.0, 5.5).unwrap());
        assert!(matches!(
            value.validate_common(),
            Err(ValidationError::UncertaintyDoesNotContainValue { .. })
        ));
    }

    #[test]
    fn optional_unknowns_survive_serialization_as_none() {
        let model = valid_model();
        let encoded = serde_json::to_string(&model).unwrap();
        let decoded: TransducerModel = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.electrical.max_voltage, None);
        assert_eq!(decoded.motor.max_current, None);
        assert_eq!(decoded, model);
    }

    #[test]
    fn max_temperature_must_exceed_reference_temperature() {
        let mut model = valid_model();
        model.thermal.max_coil_temperature = Some(q(
            15.0,
            PhysicalUnit::Celsius,
            datasheet("bad-sheet"),
        ));
        assert!(matches!(
            model.validate(),
            Err(ValidationError::ThermalLimitNotAboveReference { .. })
        ));
    }

    #[test]
    fn non_finite_parameter_is_rejected() {
        let mut model = valid_model();
        model.motor.force_factor.value = f64::NAN;
        assert_eq!(model.validate(), Err(ValidationError::NonFiniteValue));
    }
}
