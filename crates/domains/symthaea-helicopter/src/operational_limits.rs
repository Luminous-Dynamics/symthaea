// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Machine-evaluable operational-limit gate.
//!
//! The gate distinguishes advisory restrictions from prohibited operation and
//! from missing evidence. It does not claim certification; limits must be bound
//! to an approved airframe configuration and supported by external evidence.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationPhase {
    Preflight,
    Takeoff,
    EnRoute,
    Hover,
    Approach,
    Landing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationalLimitSet {
    pub schema_version: String,
    pub limit_set_id: String,
    pub airframe_configuration_id: String,
    pub minimum_temperature_c: f64,
    pub maximum_temperature_c: f64,
    pub restricted_wind_mps: f64,
    pub prohibited_wind_mps: f64,
    pub restricted_crosswind_mps: f64,
    pub prohibited_crosswind_mps: f64,
    pub restricted_density_altitude_m: f64,
    pub prohibited_density_altitude_m: f64,
    pub restricted_gross_mass_kg: f64,
    pub maximum_gross_mass_kg: f64,
    pub minimum_cg_m: f64,
    pub maximum_cg_m: f64,
    pub restricted_visibility_m: f64,
    pub minimum_visibility_m: f64,
    pub maximum_icing_fraction: f64,
    pub maximum_precipitation_rate_mm_h: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationalObservation {
    pub phase: OperationPhase,
    pub airframe_configuration_id: String,
    pub temperature_c: Option<f64>,
    pub steady_wind_mps: Option<f64>,
    pub crosswind_mps: Option<f64>,
    pub density_altitude_m: Option<f64>,
    pub gross_mass_kg: Option<f64>,
    pub cg_m: Option<f64>,
    pub visibility_m: Option<f64>,
    pub icing_fraction: Option<f64>,
    pub precipitation_rate_mm_h: Option<f64>,
    pub weather_evidence_id: Option<String>,
    pub mass_properties_evidence_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationalGateStatus {
    Go,
    Restricted,
    NoGo,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationalLimitSeverity {
    Restriction,
    Prohibited,
    MissingEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationalLimitIssue {
    pub parameter: String,
    pub severity: OperationalLimitSeverity,
    pub observed: Option<f64>,
    pub limit: Option<f64>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationalGateReport {
    pub schema_version: String,
    pub limit_set_id: String,
    pub phase: OperationPhase,
    pub status: OperationalGateStatus,
    pub airframe_configuration_id: String,
    pub issues: Vec<OperationalLimitIssue>,
}

impl OperationalGateReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, OperationalLimitsError> {
        let mut canonical = self.clone();
        canonical.issues.sort_by(|a, b| {
            a.parameter
                .cmp(&b.parameter)
                .then_with(|| format!("{:?}", a.severity).cmp(&format!("{:?}", b.severity)))
        });
        serde_json::to_vec(&canonical).map_err(|_| OperationalLimitsError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, OperationalLimitsError> {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in self.canonical_json()? {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OperationalLimitsError {
    InvalidLimitSet,
    InvalidObservation(String),
    ConfigurationMismatch { expected: String, observed: String },
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct OperationalLimitsGate {
    limits: OperationalLimitSet,
}

impl OperationalLimitsGate {
    pub fn new(limits: OperationalLimitSet) -> Result<Self, OperationalLimitsError> {
        let finite = [
            limits.minimum_temperature_c,
            limits.maximum_temperature_c,
            limits.restricted_wind_mps,
            limits.prohibited_wind_mps,
            limits.restricted_crosswind_mps,
            limits.prohibited_crosswind_mps,
            limits.restricted_density_altitude_m,
            limits.prohibited_density_altitude_m,
            limits.restricted_gross_mass_kg,
            limits.maximum_gross_mass_kg,
            limits.minimum_cg_m,
            limits.maximum_cg_m,
            limits.restricted_visibility_m,
            limits.minimum_visibility_m,
            limits.maximum_icing_fraction,
            limits.maximum_precipitation_rate_mm_h,
        ];
        if limits.schema_version.trim().is_empty()
            || limits.limit_set_id.trim().is_empty()
            || limits.airframe_configuration_id.trim().is_empty()
            || finite.iter().any(|value| !value.is_finite())
            || limits.minimum_temperature_c >= limits.maximum_temperature_c
            || limits.restricted_wind_mps < 0.0
            || limits.restricted_wind_mps > limits.prohibited_wind_mps
            || limits.restricted_crosswind_mps < 0.0
            || limits.restricted_crosswind_mps > limits.prohibited_crosswind_mps
            || limits.restricted_density_altitude_m > limits.prohibited_density_altitude_m
            || limits.restricted_gross_mass_kg <= 0.0
            || limits.restricted_gross_mass_kg > limits.maximum_gross_mass_kg
            || limits.minimum_cg_m >= limits.maximum_cg_m
            || limits.restricted_visibility_m < limits.minimum_visibility_m
            || limits.minimum_visibility_m < 0.0
            || !(0.0..=1.0).contains(&limits.maximum_icing_fraction)
            || limits.maximum_precipitation_rate_mm_h < 0.0
        {
            return Err(OperationalLimitsError::InvalidLimitSet);
        }
        Ok(Self { limits })
    }

    pub fn evaluate(
        &self,
        observation: &OperationalObservation,
    ) -> Result<OperationalGateReport, OperationalLimitsError> {
        if observation.airframe_configuration_id != self.limits.airframe_configuration_id {
            return Err(OperationalLimitsError::ConfigurationMismatch {
                expected: self.limits.airframe_configuration_id.clone(),
                observed: observation.airframe_configuration_id.clone(),
            });
        }
        validate_observation(observation)?;

        let mut issues = Vec::new();
        require_evidence(
            "weather_evidence",
            observation.weather_evidence_id.as_deref(),
            &mut issues,
        );
        require_evidence(
            "mass_properties_evidence",
            observation.mass_properties_evidence_id.as_deref(),
            &mut issues,
        );

        check_range(
            "temperature_c",
            observation.temperature_c,
            self.limits.minimum_temperature_c,
            self.limits.maximum_temperature_c,
            &mut issues,
        );
        check_upper_pair(
            "steady_wind_mps",
            observation.steady_wind_mps,
            self.limits.restricted_wind_mps,
            self.limits.prohibited_wind_mps,
            &mut issues,
        );
        check_upper_pair(
            "crosswind_mps",
            observation.crosswind_mps,
            self.limits.restricted_crosswind_mps,
            self.limits.prohibited_crosswind_mps,
            &mut issues,
        );
        check_upper_pair(
            "density_altitude_m",
            observation.density_altitude_m,
            self.limits.restricted_density_altitude_m,
            self.limits.prohibited_density_altitude_m,
            &mut issues,
        );
        check_upper_pair(
            "gross_mass_kg",
            observation.gross_mass_kg,
            self.limits.restricted_gross_mass_kg,
            self.limits.maximum_gross_mass_kg,
            &mut issues,
        );
        check_range(
            "cg_m",
            observation.cg_m,
            self.limits.minimum_cg_m,
            self.limits.maximum_cg_m,
            &mut issues,
        );
        check_lower_pair(
            "visibility_m",
            observation.visibility_m,
            self.limits.restricted_visibility_m,
            self.limits.minimum_visibility_m,
            &mut issues,
        );
        check_hard_upper(
            "icing_fraction",
            observation.icing_fraction,
            self.limits.maximum_icing_fraction,
            &mut issues,
        );
        check_hard_upper(
            "precipitation_rate_mm_h",
            observation.precipitation_rate_mm_h,
            self.limits.maximum_precipitation_rate_mm_h,
            &mut issues,
        );

        let no_go = issues
            .iter()
            .any(|issue| issue.severity == OperationalLimitSeverity::Prohibited);
        let incomplete = issues
            .iter()
            .any(|issue| issue.severity == OperationalLimitSeverity::MissingEvidence);
        let restricted = issues
            .iter()
            .any(|issue| issue.severity == OperationalLimitSeverity::Restriction);
        let status = if no_go {
            OperationalGateStatus::NoGo
        } else if incomplete {
            OperationalGateStatus::Incomplete
        } else if restricted {
            OperationalGateStatus::Restricted
        } else {
            OperationalGateStatus::Go
        };

        Ok(OperationalGateReport {
            schema_version: self.limits.schema_version.clone(),
            limit_set_id: self.limits.limit_set_id.clone(),
            phase: observation.phase,
            status,
            airframe_configuration_id: observation.airframe_configuration_id.clone(),
            issues,
        })
    }
}

fn validate_observation(
    observation: &OperationalObservation,
) -> Result<(), OperationalLimitsError> {
    let fields = [
        ("temperature_c", observation.temperature_c),
        ("steady_wind_mps", observation.steady_wind_mps),
        ("crosswind_mps", observation.crosswind_mps),
        ("density_altitude_m", observation.density_altitude_m),
        ("gross_mass_kg", observation.gross_mass_kg),
        ("cg_m", observation.cg_m),
        ("visibility_m", observation.visibility_m),
        ("icing_fraction", observation.icing_fraction),
        (
            "precipitation_rate_mm_h",
            observation.precipitation_rate_mm_h,
        ),
    ];
    for (name, value) in fields {
        if value.is_some_and(|value| !value.is_finite()) {
            return Err(OperationalLimitsError::InvalidObservation(name.into()));
        }
    }
    if observation.steady_wind_mps.is_some_and(|value| value < 0.0)
        || observation.crosswind_mps.is_some_and(|value| value < 0.0)
        || observation.gross_mass_kg.is_some_and(|value| value <= 0.0)
        || observation.visibility_m.is_some_and(|value| value < 0.0)
        || observation
            .icing_fraction
            .is_some_and(|value| !(0.0..=1.0).contains(&value))
        || observation
            .precipitation_rate_mm_h
            .is_some_and(|value| value < 0.0)
    {
        return Err(OperationalLimitsError::InvalidObservation(
            "bounded value".into(),
        ));
    }
    Ok(())
}

fn require_evidence(name: &str, value: Option<&str>, issues: &mut Vec<OperationalLimitIssue>) {
    if value.is_none_or(|value| value.trim().is_empty()) {
        issues.push(OperationalLimitIssue {
            parameter: name.into(),
            severity: OperationalLimitSeverity::MissingEvidence,
            observed: None,
            limit: None,
            detail: "required evidence identifier is missing".into(),
        });
    }
}

fn check_upper_pair(
    name: &str,
    observed: Option<f64>,
    restricted: f64,
    prohibited: f64,
    issues: &mut Vec<OperationalLimitIssue>,
) {
    let Some(value) = observed else {
        missing_value(name, issues);
        return;
    };
    if value > prohibited {
        limit_issue(
            name,
            OperationalLimitSeverity::Prohibited,
            value,
            prohibited,
            issues,
        );
    } else if value > restricted {
        limit_issue(
            name,
            OperationalLimitSeverity::Restriction,
            value,
            restricted,
            issues,
        );
    }
}

fn check_lower_pair(
    name: &str,
    observed: Option<f64>,
    restricted: f64,
    prohibited: f64,
    issues: &mut Vec<OperationalLimitIssue>,
) {
    let Some(value) = observed else {
        missing_value(name, issues);
        return;
    };
    if value < prohibited {
        limit_issue(
            name,
            OperationalLimitSeverity::Prohibited,
            value,
            prohibited,
            issues,
        );
    } else if value < restricted {
        limit_issue(
            name,
            OperationalLimitSeverity::Restriction,
            value,
            restricted,
            issues,
        );
    }
}

fn check_range(
    name: &str,
    observed: Option<f64>,
    minimum: f64,
    maximum: f64,
    issues: &mut Vec<OperationalLimitIssue>,
) {
    let Some(value) = observed else {
        missing_value(name, issues);
        return;
    };
    if value < minimum || value > maximum {
        issues.push(OperationalLimitIssue {
            parameter: name.into(),
            severity: OperationalLimitSeverity::Prohibited,
            observed: Some(value),
            limit: Some(if value < minimum { minimum } else { maximum }),
            detail: format!("outside inclusive range [{minimum}, {maximum}]"),
        });
    }
}

fn check_hard_upper(
    name: &str,
    observed: Option<f64>,
    maximum: f64,
    issues: &mut Vec<OperationalLimitIssue>,
) {
    let Some(value) = observed else {
        missing_value(name, issues);
        return;
    };
    if value > maximum {
        limit_issue(
            name,
            OperationalLimitSeverity::Prohibited,
            value,
            maximum,
            issues,
        );
    }
}

fn missing_value(name: &str, issues: &mut Vec<OperationalLimitIssue>) {
    issues.push(OperationalLimitIssue {
        parameter: name.into(),
        severity: OperationalLimitSeverity::MissingEvidence,
        observed: None,
        limit: None,
        detail: "required observation is missing".into(),
    });
}

fn limit_issue(
    name: &str,
    severity: OperationalLimitSeverity,
    observed: f64,
    limit: f64,
    issues: &mut Vec<OperationalLimitIssue>,
) {
    issues.push(OperationalLimitIssue {
        parameter: name.into(),
        severity,
        observed: Some(observed),
        limit: Some(limit),
        detail: "operational limit exceeded".into(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> OperationalLimitSet {
        OperationalLimitSet {
            schema_version: "symthaea.helicopter.operational-limits.v1".into(),
            limit_set_id: "limits-a".into(),
            airframe_configuration_id: "airframe-a".into(),
            minimum_temperature_c: -20.0,
            maximum_temperature_c: 45.0,
            restricted_wind_mps: 15.0,
            prohibited_wind_mps: 22.0,
            restricted_crosswind_mps: 10.0,
            prohibited_crosswind_mps: 15.0,
            restricted_density_altitude_m: 2_000.0,
            prohibited_density_altitude_m: 3_000.0,
            restricted_gross_mass_kg: 480.0,
            maximum_gross_mass_kg: 500.0,
            minimum_cg_m: -0.2,
            maximum_cg_m: 0.2,
            restricted_visibility_m: 3_000.0,
            minimum_visibility_m: 1_500.0,
            maximum_icing_fraction: 0.05,
            maximum_precipitation_rate_mm_h: 20.0,
        }
    }

    fn observation() -> OperationalObservation {
        OperationalObservation {
            phase: OperationPhase::Preflight,
            airframe_configuration_id: "airframe-a".into(),
            temperature_c: Some(20.0),
            steady_wind_mps: Some(5.0),
            crosswind_mps: Some(3.0),
            density_altitude_m: Some(500.0),
            gross_mass_kg: Some(450.0),
            cg_m: Some(0.0),
            visibility_m: Some(10_000.0),
            icing_fraction: Some(0.0),
            precipitation_rate_mm_h: Some(0.0),
            weather_evidence_id: Some("weather:1".into()),
            mass_properties_evidence_id: Some("mass:1".into()),
        }
    }

    #[test]
    fn nominal_observation_is_go() {
        let gate = OperationalLimitsGate::new(limits()).unwrap();
        assert_eq!(
            gate.evaluate(&observation()).unwrap().status,
            OperationalGateStatus::Go
        );
    }

    #[test]
    fn soft_wind_limit_restricts() {
        let gate = OperationalLimitsGate::new(limits()).unwrap();
        let mut observed = observation();
        observed.steady_wind_mps = Some(18.0);
        assert_eq!(
            gate.evaluate(&observed).unwrap().status,
            OperationalGateStatus::Restricted
        );
    }

    #[test]
    fn hard_mass_limit_is_no_go() {
        let gate = OperationalLimitsGate::new(limits()).unwrap();
        let mut observed = observation();
        observed.gross_mass_kg = Some(510.0);
        assert_eq!(
            gate.evaluate(&observed).unwrap().status,
            OperationalGateStatus::NoGo
        );
    }

    #[test]
    fn missing_weather_is_incomplete() {
        let gate = OperationalLimitsGate::new(limits()).unwrap();
        let mut observed = observation();
        observed.weather_evidence_id = None;
        assert_eq!(
            gate.evaluate(&observed).unwrap().status,
            OperationalGateStatus::Incomplete
        );
    }

    #[test]
    fn mismatched_airframe_is_rejected() {
        let gate = OperationalLimitsGate::new(limits()).unwrap();
        let mut observed = observation();
        observed.airframe_configuration_id = "other".into();
        assert!(matches!(
            gate.evaluate(&observed),
            Err(OperationalLimitsError::ConfigurationMismatch { .. })
        ));
    }
}
