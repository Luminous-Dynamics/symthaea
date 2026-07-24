// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Traceable model-parameter calibration and uncertainty ledger.
//!
//! Numeric constants without provenance are useful for exploratory simulation
//! but must not silently become evidence about a real airframe. This module
//! records source class, uncertainty, validity bounds, and evidence identifiers,
//! then applies a named subset to the reduced-order rotor model.

use serde::{Deserialize, Serialize};

use crate::rotor_dynamics::RotorDynamicsConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterSourceClass {
    Measured,
    ManufacturerData,
    PeerReviewedLiterature,
    DerivedFromTraceableInputs,
    Assumed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibratedParameter {
    pub parameter_id: String,
    pub value: f64,
    pub unit: String,
    pub standard_uncertainty: f64,
    pub valid_min: f64,
    pub valid_max: f64,
    pub source_class: ParameterSourceClass,
    pub evidence_id: Option<String>,
}

impl CalibratedParameter {
    fn validate(&self) -> bool {
        !self.parameter_id.trim().is_empty()
            && !self.unit.trim().is_empty()
            && self.value.is_finite()
            && self.standard_uncertainty.is_finite()
            && self.standard_uncertainty >= 0.0
            && self.valid_min.is_finite()
            && self.valid_max.is_finite()
            && self.valid_min <= self.valid_max
            && self.value >= self.valid_min
            && self.value <= self.valid_max
            && match self.source_class {
                ParameterSourceClass::Assumed => true,
                _ => self
                    .evidence_id
                    .as_ref()
                    .is_some_and(|value| !value.trim().is_empty()),
            }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightModelCalibration {
    pub schema_version: String,
    pub airframe_id: String,
    pub calibration_id: String,
    pub parameters: Vec<CalibratedParameter>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationReadiness {
    ResearchOnly,
    TraceableWithAssumptions,
    FullyTraceable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationAssessment {
    pub readiness: CalibrationReadiness,
    pub missing_required_parameters: Vec<String>,
    pub assumed_parameters: Vec<String>,
    pub invalid_parameters: Vec<String>,
    pub duplicate_parameters: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationError {
    InvalidIdentity,
    InvalidParameter,
    DuplicateParameter,
    MissingRequiredParameter,
    InvalidRotorConfiguration,
    SerializationFailed,
}

pub const REQUIRED_ROTOR_PARAMETERS: &[&str] = &[
    "rotor.max_main_rpm",
    "rotor.max_tail_rpm",
    "rotor.main_time_constant_s",
    "rotor.tail_time_constant_s",
    "rotor.thrust_coefficient",
    "rotor.torque_reaction_coefficient",
    "rotor.tail_thrust_coefficient",
    "rotor.tail_moment_arm_m",
    "rotor.main_inertia_kg_m2",
    "rotor.radius_m",
];

impl FlightModelCalibration {
    pub fn validate(&self) -> Result<(), CalibrationError> {
        if self.schema_version.trim().is_empty()
            || self.airframe_id.trim().is_empty()
            || self.calibration_id.trim().is_empty()
        {
            return Err(CalibrationError::InvalidIdentity);
        }
        if self
            .parameters
            .iter()
            .any(|parameter| !parameter.validate())
        {
            return Err(CalibrationError::InvalidParameter);
        }
        for (index, parameter) in self.parameters.iter().enumerate() {
            if self.parameters[..index]
                .iter()
                .any(|previous| previous.parameter_id == parameter.parameter_id)
            {
                return Err(CalibrationError::DuplicateParameter);
            }
        }
        Ok(())
    }

    pub fn parameter(&self, parameter_id: &str) -> Option<&CalibratedParameter> {
        self.parameters
            .iter()
            .find(|parameter| parameter.parameter_id == parameter_id)
    }

    pub fn assess(&self, required: &[&str]) -> CalibrationAssessment {
        let mut invalid_parameters = Vec::new();
        let mut duplicate_parameters = Vec::new();
        for (index, parameter) in self.parameters.iter().enumerate() {
            if !parameter.validate() {
                invalid_parameters.push(parameter.parameter_id.clone());
            }
            if self.parameters[..index]
                .iter()
                .any(|previous| previous.parameter_id == parameter.parameter_id)
            {
                duplicate_parameters.push(parameter.parameter_id.clone());
            }
        }
        let missing_required_parameters = required
            .iter()
            .filter(|required_id| self.parameter(required_id).is_none())
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>();
        let assumed_parameters = self
            .parameters
            .iter()
            .filter(|parameter| parameter.source_class == ParameterSourceClass::Assumed)
            .map(|parameter| parameter.parameter_id.clone())
            .collect::<Vec<_>>();
        let readiness = if !invalid_parameters.is_empty()
            || !duplicate_parameters.is_empty()
            || !missing_required_parameters.is_empty()
        {
            CalibrationReadiness::ResearchOnly
        } else if !assumed_parameters.is_empty() {
            CalibrationReadiness::TraceableWithAssumptions
        } else {
            CalibrationReadiness::FullyTraceable
        };
        CalibrationAssessment {
            readiness,
            missing_required_parameters,
            assumed_parameters,
            invalid_parameters,
            duplicate_parameters,
        }
    }

    pub fn apply_to_rotor_config(
        &self,
        mut config: RotorDynamicsConfig,
    ) -> Result<RotorDynamicsConfig, CalibrationError> {
        self.validate()?;
        let assessment = self.assess(REQUIRED_ROTOR_PARAMETERS);
        if !assessment.missing_required_parameters.is_empty() {
            return Err(CalibrationError::MissingRequiredParameter);
        }
        config.max_main_rpm = self.value("rotor.max_main_rpm")?;
        config.max_tail_rpm = self.value("rotor.max_tail_rpm")?;
        config.main_rotor_tau = self.value("rotor.main_time_constant_s")?;
        config.tail_rotor_tau = self.value("rotor.tail_time_constant_s")?;
        config.thrust_coefficient = self.value("rotor.thrust_coefficient")?;
        config.torque_reaction_coefficient = self.value("rotor.torque_reaction_coefficient")?;
        config.tail_thrust_coefficient = self.value("rotor.tail_thrust_coefficient")?;
        config.tail_moment_arm = self.value("rotor.tail_moment_arm_m")?;
        config.main_rotor_inertia_kg_m2 = self.value("rotor.main_inertia_kg_m2")?;
        config.rotor_radius_m = self.value("rotor.radius_m")?;
        if !config.validate() {
            return Err(CalibrationError::InvalidRotorConfiguration);
        }
        Ok(config)
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, CalibrationError> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical
            .parameters
            .sort_by(|left, right| left.parameter_id.cmp(&right.parameter_id));
        serde_json::to_vec(&canonical).map_err(|_| CalibrationError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, CalibrationError> {
        let bytes = self.canonical_json()?;
        let mut hash = 0xcbf29ce484222325u64;
        for byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }

    fn value(&self, parameter_id: &str) -> Result<f64, CalibrationError> {
        self.parameter(parameter_id)
            .map(|parameter| parameter.value)
            .ok_or(CalibrationError::MissingRequiredParameter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameter(id: &str, value: f64, unit: &str) -> CalibratedParameter {
        CalibratedParameter {
            parameter_id: id.to_string(),
            value,
            unit: unit.to_string(),
            standard_uncertainty: value.abs() * 0.01,
            valid_min: 0.0,
            valid_max: value.abs().max(1.0) * 10.0,
            source_class: ParameterSourceClass::Measured,
            evidence_id: Some(format!("evidence:{id}")),
        }
    }

    fn calibration() -> FlightModelCalibration {
        let defaults = RotorDynamicsConfig::default();
        FlightModelCalibration {
            schema_version: "symthaea-helicopter-calibration-v1".to_string(),
            airframe_id: "research-airframe-001".to_string(),
            calibration_id: "rotor-bench-2026-07".to_string(),
            parameters: vec![
                parameter("rotor.max_main_rpm", defaults.max_main_rpm, "rpm"),
                parameter("rotor.max_tail_rpm", defaults.max_tail_rpm, "rpm"),
                parameter("rotor.main_time_constant_s", defaults.main_rotor_tau, "s"),
                parameter("rotor.tail_time_constant_s", defaults.tail_rotor_tau, "s"),
                parameter(
                    "rotor.thrust_coefficient",
                    defaults.thrust_coefficient,
                    "N/rpm2",
                ),
                parameter(
                    "rotor.torque_reaction_coefficient",
                    defaults.torque_reaction_coefficient,
                    "Nm/rpm2",
                ),
                parameter(
                    "rotor.tail_thrust_coefficient",
                    defaults.tail_thrust_coefficient,
                    "N/rpm2",
                ),
                parameter("rotor.tail_moment_arm_m", defaults.tail_moment_arm, "m"),
                parameter(
                    "rotor.main_inertia_kg_m2",
                    defaults.main_rotor_inertia_kg_m2,
                    "kg*m2",
                ),
                parameter("rotor.radius_m", defaults.rotor_radius_m, "m"),
            ],
        }
    }

    #[test]
    fn complete_measured_rotor_set_is_fully_traceable() {
        let calibration = calibration();
        let assessment = calibration.assess(REQUIRED_ROTOR_PARAMETERS);
        assert_eq!(assessment.readiness, CalibrationReadiness::FullyTraceable);
        assert!(
            calibration
                .apply_to_rotor_config(RotorDynamicsConfig::default())
                .is_ok()
        );
    }

    #[test]
    fn assumptions_are_visible_in_readiness() {
        let mut calibration = calibration();
        calibration.parameters[0].source_class = ParameterSourceClass::Assumed;
        calibration.parameters[0].evidence_id = None;
        assert_eq!(
            calibration.assess(REQUIRED_ROTOR_PARAMETERS).readiness,
            CalibrationReadiness::TraceableWithAssumptions
        );
    }

    #[test]
    fn missing_required_parameter_refuses_application() {
        let mut calibration = calibration();
        calibration.parameters.pop();
        assert!(matches!(
            calibration.apply_to_rotor_config(RotorDynamicsConfig::default()),
            Err(CalibrationError::MissingRequiredParameter)
        ));
    }

    #[test]
    fn canonical_digest_does_not_depend_on_declaration_order() {
        let first = calibration();
        let mut second = first.clone();
        second.parameters.reverse();
        assert_eq!(
            first.digest_fnv1a64().unwrap(),
            second.digest_fnv1a64().unwrap()
        );
    }
}
