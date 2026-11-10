// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bound, no-extrapolation component efficiency characterization.
//!
//! Curves are deliberately one-dimensional in this first increment. The
//! independent variable must be explicitly named and expressed in the same
//! units as the caller. Values and absolute uncertainty bounds are linearly
//! interpolated; the conservative lower efficiency bound is used for mission
//! energy estimates. Extrapolation is rejected rather than silently guessed.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceClass {
    PhysicalMeasurement,
    ManufacturerReported,
    PublishedResearch,
    Simulation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRecord {
    pub id: String,
    pub uri: String,
    /// Lowercase or uppercase 64-character SHA-256 hexadecimal digest.
    pub sha256: String,
    pub class: EvidenceClass,
    pub description: String,
}

impl EvidenceRecord {
    pub fn validate(&self) -> Result<(), CharacterizationError> {
        if self.id.trim().is_empty()
            || self.uri.trim().is_empty()
            || self.description.trim().is_empty()
        {
            return Err(CharacterizationError::InvalidEvidence(
                "evidence ID, URI, and description must be nonempty".into(),
            ));
        }
        if self.sha256.len() != 64 || !self.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(CharacterizationError::InvalidEvidence(
                "evidence SHA-256 must be exactly 64 hexadecimal characters".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EfficiencyPoint {
    /// Independent variable, in the units declared by the curve.
    pub x: f64,
    /// Nominal measured/reported/simulated efficiency.
    pub efficiency: f64,
    /// Absolute uncertainty bound, not a standard deviation or confidence interval.
    pub absolute_uncertainty: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EfficiencySample {
    pub nominal_efficiency: f64,
    pub absolute_uncertainty: f64,
    pub conservative_lower_bound: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EfficiencyCurve {
    pub id: String,
    pub x_axis: String,
    /// Engineering unit for x (for example "1", "rpm", or "m/s").
    pub x_unit: String,
    pub evidence: EvidenceRecord,
    pub points: Vec<EfficiencyPoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CharacterizationError {
    InvalidEvidence(String),
    InvalidCurve(String),
    OutsideCalibrationDomain {
        curve_id: String,
        observed: f64,
        minimum: f64,
        maximum: f64,
    },
    InvalidOperatingPoint(String),
}

impl fmt::Display for CharacterizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CharacterizationError {}

impl EfficiencyCurve {
    pub fn validate(&self) -> Result<(), CharacterizationError> {
        self.evidence.validate()?;
        if self.id.trim().is_empty()
            || self.x_axis.trim().is_empty()
            || self.x_unit.trim().is_empty()
            || self.points.len() < 2
        {
            return Err(CharacterizationError::InvalidCurve(
                "curve ID/axis must be nonempty and at least two points are required".into(),
            ));
        }
        let mut previous_x = None;
        for point in &self.points {
            if !point.x.is_finite()
                || !point.efficiency.is_finite()
                || !point.absolute_uncertainty.is_finite()
                || !(0.0..=1.0).contains(&point.efficiency)
                || point.efficiency == 0.0
                || point.absolute_uncertainty < 0.0
                || point.absolute_uncertainty >= point.efficiency
            {
                return Err(CharacterizationError::InvalidCurve(format!(
                    "curve {} contains a non-finite or invalid efficiency point",
                    self.id
                )));
            }
            if previous_x.is_some_and(|previous| point.x <= previous) {
                return Err(CharacterizationError::InvalidCurve(format!(
                    "curve {} independent-axis values must be strictly increasing",
                    self.id
                )));
            }
            previous_x = Some(point.x);
        }
        Ok(())
    }

    pub fn sample(&self, x: f64) -> Result<EfficiencySample, CharacterizationError> {
        self.validate()?;
        if !x.is_finite() {
            return Err(CharacterizationError::InvalidOperatingPoint(format!(
                "curve {} received a non-finite operating point",
                self.id
            )));
        }
        let first = self.points.first().expect("validated curve has points");
        let last = self.points.last().expect("validated curve has points");
        if x < first.x || x > last.x {
            return Err(CharacterizationError::OutsideCalibrationDomain {
                curve_id: self.id.clone(),
                observed: x,
                minimum: first.x,
                maximum: last.x,
            });
        }

        if x == last.x {
            return Ok(EfficiencySample {
                nominal_efficiency: last.efficiency,
                absolute_uncertainty: last.absolute_uncertainty,
                conservative_lower_bound: last.efficiency - last.absolute_uncertainty,
            });
        }

        let pair = self
            .points
            .windows(2)
            .find(|pair| x >= pair[0].x && x <= pair[1].x)
            .ok_or_else(|| CharacterizationError::InvalidCurve(
                "validated axis did not contain requested in-domain point".into(),
            ))?;
        let lower = pair[0];
        let upper = pair[1];
        let fraction = (x - lower.x) / (upper.x - lower.x);
        let nominal_efficiency =
            lower.efficiency + fraction * (upper.efficiency - lower.efficiency);
        let absolute_uncertainty = lower.absolute_uncertainty
            + fraction * (upper.absolute_uncertainty - lower.absolute_uncertainty);
        Ok(EfficiencySample {
            nominal_efficiency,
            absolute_uncertainty,
            conservative_lower_bound: nominal_efficiency - absolute_uncertainty,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PropulsionCharacterization {
    /// Efficiency versus mechanical shaft-load fraction.
    pub motor_efficiency_by_load: EfficiencyCurve,
    /// Efficiency versus inverter electrical-throughput fraction.
    pub inverter_efficiency_by_load: EfficiencyCurve,
    /// Efficiency versus propeller advance ratio J = V / (n D).
    pub propeller_efficiency_by_advance_ratio: EfficiencyCurve,
    pub motor_rated_shaft_power_w: f64,
    pub inverter_rated_input_power_w: f64,
    pub propeller_diameter_m: f64,
    /// Fixed operating RPM for this first steady-state model.
    pub propeller_rpm: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropulsionOperatingPoint {
    pub motor_efficiency_used: f64,
    pub inverter_efficiency_used: f64,
    pub propeller_efficiency_used: f64,
    pub motor_load_fraction: f64,
    pub inverter_load_fraction: f64,
    pub propeller_advance_ratio: f64,
}

impl PropulsionCharacterization {
    pub fn validate(&self) -> Result<(), CharacterizationError> {
        self.motor_efficiency_by_load.validate()?;
        self.inverter_efficiency_by_load.validate()?;
        self.propeller_efficiency_by_advance_ratio.validate()?;
        let positive = [
            self.motor_rated_shaft_power_w,
            self.inverter_rated_input_power_w,
            self.propeller_diameter_m,
            self.propeller_rpm,
        ];
        if positive.iter().any(|value| !value.is_finite() || *value <= 0.0) {
            return Err(CharacterizationError::InvalidOperatingPoint(
                "rated powers, propeller diameter, and RPM must be finite and positive".into(),
            ));
        }
        Ok(())
    }

    /// Resolve efficiencies at the supplied steady segment.
    ///
    /// Lower uncertainty bounds are used, so a less certain or less efficient
    /// component cannot make the mission estimate look artificially better.
    pub fn operating_point(
        &self,
        true_airspeed_mps: f64,
        useful_propulsive_power_w: f64,
    ) -> Result<PropulsionOperatingPoint, CharacterizationError> {
        self.validate()?;
        if !true_airspeed_mps.is_finite()
            || true_airspeed_mps <= 0.0
            || !useful_propulsive_power_w.is_finite()
            || useful_propulsive_power_w <= 0.0
        {
            return Err(CharacterizationError::InvalidOperatingPoint(
                "airspeed and useful propulsive power must be finite and positive".into(),
            ));
        }

        let propeller_advance_ratio =
            true_airspeed_mps / (self.propeller_rpm / 60.0 * self.propeller_diameter_m);
        let propeller_efficiency_used = self
            .propeller_efficiency_by_advance_ratio
            .sample(propeller_advance_ratio)?
            .conservative_lower_bound;

        let propeller_shaft_power_w =
            useful_propulsive_power_w / propeller_efficiency_used;
        let motor_load_fraction = propeller_shaft_power_w / self.motor_rated_shaft_power_w;
        let motor_efficiency_used = self
            .motor_efficiency_by_load
            .sample(motor_load_fraction)?
            .conservative_lower_bound;

        let inverter_input_power_w = propeller_shaft_power_w / motor_efficiency_used;
        let inverter_load_fraction =
            inverter_input_power_w / self.inverter_rated_input_power_w;
        let inverter_efficiency_used = self
            .inverter_efficiency_by_load
            .sample(inverter_load_fraction)?
            .conservative_lower_bound;

        Ok(PropulsionOperatingPoint {
            motor_efficiency_used,
            inverter_efficiency_used,
            propeller_efficiency_used,
            motor_load_fraction,
            inverter_load_fraction,
            propeller_advance_ratio,
        })
    }

    pub fn evidence_ids(&self) -> Vec<String> {
        let mut ids = vec![
            self.motor_efficiency_by_load.evidence.id.clone(),
            self.inverter_efficiency_by_load.evidence.id.clone(),
            self.propeller_efficiency_by_advance_ratio.evidence.id.clone(),
        ];
        ids.sort();
        ids.dedup();
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(id: &str) -> EvidenceRecord {
        EvidenceRecord {
            id: id.into(),
            uri: format!("artifact://characterization/{id}"),
            sha256: "a".repeat(64),
            class: EvidenceClass::Simulation,
            description: "Synthetic fixture for unit tests; not physical test evidence".into(),
        }
    }

    fn curve(id: &str, axis: &str, points: Vec<EfficiencyPoint>) -> EfficiencyCurve {
        EfficiencyCurve {
            id: id.into(),
            x_axis: axis.into(),
            x_unit: "dimensionless".into(),
            evidence: evidence(&format!("{id}-dataset")),
            points,
        }
    }

    #[test]
    fn curve_interpolates_nominal_and_uncertainty() {
        let c = curve(
            "motor",
            "load_fraction",
            vec![
                EfficiencyPoint { x: 0.1, efficiency: 0.90, absolute_uncertainty: 0.01 },
                EfficiencyPoint { x: 0.9, efficiency: 0.96, absolute_uncertainty: 0.02 },
            ],
        );
        let sample = c.sample(0.5).unwrap();
        assert!((sample.nominal_efficiency - 0.93).abs() < 1e-12);
        assert!((sample.absolute_uncertainty - 0.015).abs() < 1e-12);
        assert!((sample.conservative_lower_bound - 0.915).abs() < 1e-12);
    }

    #[test]
    fn curve_rejects_extrapolation() {
        let c = curve(
            "motor",
            "load_fraction",
            vec![
                EfficiencyPoint { x: 0.1, efficiency: 0.90, absolute_uncertainty: 0.01 },
                EfficiencyPoint { x: 0.9, efficiency: 0.96, absolute_uncertainty: 0.02 },
            ],
        );
        assert!(matches!(
            c.sample(0.95),
            Err(CharacterizationError::OutsideCalibrationDomain { .. })
        ));
    }

    #[test]
    fn curve_rejects_duplicate_axis_points() {
        let c = curve(
            "motor",
            "load_fraction",
            vec![
                EfficiencyPoint { x: 0.5, efficiency: 0.90, absolute_uncertainty: 0.01 },
                EfficiencyPoint { x: 0.5, efficiency: 0.95, absolute_uncertainty: 0.01 },
            ],
        );
        assert!(matches!(c.validate(), Err(CharacterizationError::InvalidCurve(_))));
    }

    #[test]
    fn evidence_requires_sha256_digest() {
        let mut item = evidence("motor-data");
        item.sha256 = "not-a-digest".into();
        assert!(matches!(item.validate(), Err(CharacterizationError::InvalidEvidence(_))));
    }
}
