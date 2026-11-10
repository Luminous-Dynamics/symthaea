// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bound, no-extrapolation component efficiency characterization.
//!
//! Curves are deliberately one-dimensional in this first increment. The
//! independent variable must be explicitly named and expressed in the same
//! units as the caller. Values and absolute uncertainty bounds are linearly
//! interpolated; the conservative lower efficiency bound is used for mission
//! energy estimates. Extrapolation is rejected rather than silently guessed.

use std::collections::BTreeMap;
use std::fmt;

use sha2::{Digest, Sha256};

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
            || !self.uri.contains("://")
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

    /// Verify that caller-supplied immutable artifact bytes match this record's
    /// SHA-256 digest. This does not fetch the URI or authenticate its publisher.
    pub fn verify_bytes(&self, bytes: &[u8]) -> Result<(), CharacterizationError> {
        self.validate()?;
        let observed_sha256 = format!("{:x}", Sha256::digest(bytes));
        if !observed_sha256.eq_ignore_ascii_case(&self.sha256) {
            return Err(CharacterizationError::EvidenceDigestMismatch {
                evidence_id: self.id.clone(),
                expected_sha256: self.sha256.to_ascii_lowercase(),
                observed_sha256,
            });
        }
        Ok(())
    }
}

fn verify_payload(
    evidence: &EvidenceRecord,
    payloads: &BTreeMap<String, Vec<u8>>,
) -> Result<(), CharacterizationError> {
    let bytes = payloads
        .get(&evidence.id)
        .ok_or_else(|| CharacterizationError::MissingEvidencePayload(evidence.id.clone()))?;
    evidence.verify_bytes(bytes)
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
    InvalidGrid(String),
    MissingEvidencePayload(String),
    EvidenceDigestMismatch {
        evidence_id: String,
        expected_sha256: String,
        observed_sha256: String,
    },
    GridOutsideCalibrationDomain {
        grid_id: String,
        x_observed: f64,
        y_observed: f64,
        x_minimum: f64,
        x_maximum: f64,
        y_minimum: f64,
        y_maximum: f64,
    },
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
        if self.motor_efficiency_by_load.x_axis != "shaft_load_fraction"
            || self.motor_efficiency_by_load.x_unit != "1"
            || self.inverter_efficiency_by_load.x_axis != "input_load_fraction"
            || self.inverter_efficiency_by_load.x_unit != "1"
            || self.propeller_efficiency_by_advance_ratio.x_axis != "advance_ratio"
            || self.propeller_efficiency_by_advance_ratio.x_unit != "1"
        {
            return Err(CharacterizationError::InvalidCurve(
                "propulsion maps must use dimensionless shaft load, input load, and advance-ratio axes".into(),
            ));
        }
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

    /// Verify every curve against the exact caller-supplied artifact bytes.
    /// URI resolution and publisher/signature trust remain external.
    pub fn verify_evidence_payloads(
        &self,
        payloads: &BTreeMap<String, Vec<u8>>,
    ) -> Result<(), CharacterizationError> {
        self.validate()?;
        verify_payload(&self.motor_efficiency_by_load.evidence, payloads)?;
        verify_payload(&self.inverter_efficiency_by_load.evidence, payloads)?;
        verify_payload(&self.propeller_efficiency_by_advance_ratio.evidence, payloads)?;
        Ok(())
    }
}


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarGridPoint {
    pub value: f64,
    /// Absolute uncertainty bound; not a standard deviation.
    pub absolute_uncertainty: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarGridSample {
    pub nominal_value: f64,
    pub absolute_uncertainty: f64,
    pub conservative_lower_bound: f64,
    pub conservative_upper_bound: f64,
}

/// Rectilinear scalar grid stored as rows of x and columns of y.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarGrid2D {
    pub id: String,
    pub x_axis: String,
    pub x_unit: String,
    pub x_values: Vec<f64>,
    pub y_axis: String,
    pub y_unit: String,
    pub y_values: Vec<f64>,
    pub evidence: EvidenceRecord,
    /// values[x_index][y_index]
    pub values: Vec<Vec<ScalarGridPoint>>,
}

impl ScalarGrid2D {
    pub fn validate(&self) -> Result<(), CharacterizationError> {
        self.evidence.validate()?;
        if self.id.trim().is_empty()
            || self.x_axis.trim().is_empty()
            || self.x_unit.trim().is_empty()
            || self.y_axis.trim().is_empty()
            || self.y_unit.trim().is_empty()
            || self.x_values.len() < 2
            || self.y_values.len() < 2
            || self.values.len() != self.x_values.len()
        {
            return Err(CharacterizationError::InvalidGrid(
                "grid metadata, axes, or row count are invalid".into(),
            ));
        }

        for axis in [&self.x_values, &self.y_values] {
            let mut previous = None;
            for value in axis {
                if !value.is_finite() || previous.is_some_and(|prior| *value <= prior) {
                    return Err(CharacterizationError::InvalidGrid(
                        "grid axes must be finite and strictly increasing".into(),
                    ));
                }
                previous = Some(*value);
            }
        }

        for row in &self.values {
            if row.len() != self.y_values.len() {
                return Err(CharacterizationError::InvalidGrid(
                    "grid rows must match the y-axis length".into(),
                ));
            }
            for point in row {
                if !point.value.is_finite()
                    || point.value <= 0.0
                    || !point.absolute_uncertainty.is_finite()
                    || point.absolute_uncertainty < 0.0
                    || point.absolute_uncertainty >= point.value
                {
                    return Err(CharacterizationError::InvalidGrid(
                        "grid values and absolute uncertainty bounds must be finite and positive".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn sample(&self, x: f64, y: f64) -> Result<ScalarGridSample, CharacterizationError> {
        self.validate()?;
        if !x.is_finite() || !y.is_finite() {
            return Err(CharacterizationError::InvalidOperatingPoint(
                "grid operating coordinates must be finite".into(),
            ));
        }
        let x_min = self.x_values[0];
        let x_max = *self.x_values.last().expect("validated x axis");
        let y_min = self.y_values[0];
        let y_max = *self.y_values.last().expect("validated y axis");
        if x < x_min || x > x_max || y < y_min || y > y_max {
            return Err(CharacterizationError::GridOutsideCalibrationDomain {
                grid_id: self.id.clone(),
                x_observed: x,
                y_observed: y,
                x_minimum: x_min,
                x_maximum: x_max,
                y_minimum: y_min,
                y_maximum: y_max,
            });
        }

        let (x0, x1, tx) = bracket(&self.x_values, x);
        let (y0, y1, ty) = bracket(&self.y_values, y);
        let value_at_x0 = interpolate(
            self.values[x0][y0].value,
            self.values[x0][y1].value,
            ty,
        );
        let value_at_x1 = interpolate(
            self.values[x1][y0].value,
            self.values[x1][y1].value,
            ty,
        );
        let uncertainty_at_x0 = interpolate(
            self.values[x0][y0].absolute_uncertainty,
            self.values[x0][y1].absolute_uncertainty,
            ty,
        );
        let uncertainty_at_x1 = interpolate(
            self.values[x1][y0].absolute_uncertainty,
            self.values[x1][y1].absolute_uncertainty,
            ty,
        );
        let nominal_value = interpolate(value_at_x0, value_at_x1, tx);
        let absolute_uncertainty =
            interpolate(uncertainty_at_x0, uncertainty_at_x1, tx);
        Ok(ScalarGridSample {
            nominal_value,
            absolute_uncertainty,
            conservative_lower_bound: nominal_value - absolute_uncertainty,
            conservative_upper_bound: nominal_value + absolute_uncertainty,
        })
    }
}

fn bracket(axis: &[f64], value: f64) -> (usize, usize, f64) {
    if value == axis[axis.len() - 1] {
        let lower = axis.len() - 2;
        return (lower, lower + 1, 1.0);
    }
    let lower = axis
        .windows(2)
        .position(|pair| value >= pair[0] && value <= pair[1])
        .expect("validated in-domain point has a bracket");
    let fraction = (value - axis[lower]) / (axis[lower + 1] - axis[lower]);
    (lower, lower + 1, fraction)
}

fn interpolate(lower: f64, upper: f64, fraction: f64) -> f64 {
    lower + fraction * (upper - lower)
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatteryOperatingPoint {
    /// Conservative low-side voltage used by the mission solver.
    pub open_circuit_voltage_v: f64,
    /// Conservative high-side resistance used by the mission solver.
    pub internal_resistance_ohm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatteryCharacterization {
    /// Pack open-circuit voltage over state of charge and pack temperature.
    pub open_circuit_voltage: ScalarGrid2D,
    /// Pack DC resistance over state of charge and pack temperature.
    pub internal_resistance: ScalarGrid2D,
}

impl BatteryCharacterization {
    pub fn validate(&self) -> Result<(), CharacterizationError> {
        self.open_circuit_voltage.validate()?;
        self.internal_resistance.validate()?;

        let voltage = &self.open_circuit_voltage;
        let resistance = &self.internal_resistance;
        if voltage.x_axis != "state_of_charge"
            || voltage.x_unit != "1"
            || voltage.y_axis != "temperature"
            || voltage.y_unit != "K"
            || resistance.x_axis != voltage.x_axis
            || resistance.x_unit != voltage.x_unit
            || resistance.y_axis != voltage.y_axis
            || resistance.y_unit != voltage.y_unit
            || resistance.x_values != voltage.x_values
            || resistance.y_values != voltage.y_values
        {
            return Err(CharacterizationError::InvalidGrid(
                "battery OCV and resistance grids must share SOC/temperature axes and units".into(),
            ));
        }
        Ok(())
    }

    /// Resolve conservative OCV and resistance at the current SOC/temperature.
    /// OCV uncertainty is subtracted; resistance uncertainty is added.
    pub fn operating_point(
        &self,
        state_of_charge: f64,
        temperature_k: f64,
    ) -> Result<BatteryOperatingPoint, CharacterizationError> {
        self.validate()?;
        let voltage = self.open_circuit_voltage.sample(state_of_charge, temperature_k)?;
        let resistance = self.internal_resistance.sample(state_of_charge, temperature_k)?;
        Ok(BatteryOperatingPoint {
            open_circuit_voltage_v: voltage.conservative_lower_bound,
            internal_resistance_ohm: resistance.conservative_upper_bound,
        })
    }

    pub fn evidence_ids(&self) -> Vec<String> {
        let mut ids = vec![
            self.open_circuit_voltage.evidence.id.clone(),
            self.internal_resistance.evidence.id.clone(),
        ];
        ids.sort();
        ids.dedup();
        ids
    }

    /// Verify both battery grids against the exact caller-supplied artifact
    /// bytes. URI resolution and publisher/signature trust remain external.
    pub fn verify_evidence_payloads(
        &self,
        payloads: &BTreeMap<String, Vec<u8>>,
    ) -> Result<(), CharacterizationError> {
        self.validate()?;
        verify_payload(&self.open_circuit_voltage.evidence, payloads)?;
        verify_payload(&self.internal_resistance.evidence, payloads)?;
        Ok(())
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

    fn battery_grid(
        id: &str,
        evidence_id: &str,
        values: Vec<Vec<f64>>,
        uncertainty: f64,
    ) -> ScalarGrid2D {
        ScalarGrid2D {
            id: id.into(),
            x_axis: "state_of_charge".into(),
            x_unit: "1".into(),
            x_values: vec![0.2, 0.6, 0.95],
            y_axis: "temperature".into(),
            y_unit: "K".into(),
            y_values: vec![273.0, 298.0, 323.0],
            evidence: evidence(evidence_id),
            values: values
                .into_iter()
                .map(|row| row.into_iter()
                    .map(|value| ScalarGridPoint {
                        value,
                        absolute_uncertainty: uncertainty,
                    })
                    .collect())
                .collect(),
        }
    }

    fn battery_characterization() -> BatteryCharacterization {
        BatteryCharacterization {
            open_circuit_voltage: battery_grid(
                "pack-ocv",
                "pack-ocv-data",
                vec![
                    vec![340.0, 350.0, 345.0],
                    vec![365.0, 375.0, 370.0],
                    vec![385.0, 395.0, 390.0],
                ],
                1.5,
            ),
            internal_resistance: battery_grid(
                "pack-resistance",
                "pack-resistance-data",
                vec![
                    vec![0.070, 0.055, 0.070],
                    vec![0.045, 0.030, 0.050],
                    vec![0.050, 0.035, 0.055],
                ],
                0.002,
            ),
        }
    }

    #[test]
    fn battery_grid_bilinearly_interpolates() {
        let grid = battery_grid(
            "ocv",
            "ocv-data",
            vec![
                vec![340.0, 350.0, 345.0],
                vec![365.0, 375.0, 370.0],
                vec![385.0, 395.0, 390.0],
            ],
            1.5,
        );
        let sample = grid.sample(0.6, 298.0).unwrap();
        assert!((sample.nominal_value - 375.0).abs() < 1e-12);
        assert!((sample.conservative_lower_bound - 373.5).abs() < 1e-12);
        assert!((sample.conservative_upper_bound - 376.5).abs() < 1e-12);
    }

    #[test]
    fn battery_grid_rejects_out_of_domain_soc_or_temperature() {
        let grid = battery_grid(
            "ocv",
            "ocv-data",
            vec![
                vec![340.0, 350.0, 345.0],
                vec![365.0, 375.0, 370.0],
                vec![385.0, 395.0, 390.0],
            ],
            1.5,
        );
        assert!(matches!(
            grid.sample(0.1, 298.0),
            Err(CharacterizationError::GridOutsideCalibrationDomain { .. })
        ));
        assert!(matches!(
            grid.sample(0.6, 330.0),
            Err(CharacterizationError::GridOutsideCalibrationDomain { .. })
        ));
    }

    #[test]
    fn battery_uses_low_voltage_and_high_resistance_bounds() {
        let battery = battery_characterization();
        let operating = battery.operating_point(0.6, 298.0).unwrap();
        assert!((operating.open_circuit_voltage_v - 373.5).abs() < 1e-12);
        assert!((operating.internal_resistance_ohm - 0.032).abs() < 1e-12);
        assert_eq!(
            battery.evidence_ids(),
            vec!["pack-ocv-data", "pack-resistance-data"]
        );
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
    fn evidence_record_verifies_exact_sha256_payload() {
        let mut item = evidence("known-payload");
        item.sha256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into();
        item.verify_bytes(b"abc").unwrap();
        assert!(matches!(
            item.verify_bytes(b"abd"),
            Err(CharacterizationError::EvidenceDigestMismatch { .. })
        ));
    }

    #[test]
    fn characterized_maps_require_all_evidence_payloads_and_verify_them() {
        let mut propulsion = PropulsionCharacterization {
            motor_efficiency_by_load: curve(
                "motor",
                "shaft_load_fraction",
                vec![
                    EfficiencyPoint { x: 0.1, efficiency: 0.92, absolute_uncertainty: 0.01 },
                    EfficiencyPoint { x: 1.0, efficiency: 0.96, absolute_uncertainty: 0.01 },
                ],
            ),
            inverter_efficiency_by_load: curve(
                "inverter",
                "input_load_fraction",
                vec![
                    EfficiencyPoint { x: 0.1, efficiency: 0.97, absolute_uncertainty: 0.005 },
                    EfficiencyPoint { x: 1.0, efficiency: 0.985, absolute_uncertainty: 0.005 },
                ],
            ),
            propeller_efficiency_by_advance_ratio: curve(
                "propeller",
                "advance_ratio",
                vec![
                    EfficiencyPoint { x: 0.4, efficiency: 0.80, absolute_uncertainty: 0.02 },
                    EfficiencyPoint { x: 1.0, efficiency: 0.86, absolute_uncertainty: 0.02 },
                ],
            ),
            motor_rated_shaft_power_w: 80_000.0,
            inverter_rated_input_power_w: 90_000.0,
            propeller_diameter_m: 1.6,
            propeller_rpm: 2_400.0,
        };
        let hash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        for record in [
            &mut propulsion.motor_efficiency_by_load.evidence,
            &mut propulsion.inverter_efficiency_by_load.evidence,
            &mut propulsion.propeller_efficiency_by_advance_ratio.evidence,
        ] {
            record.sha256 = hash.into();
        }
        let payloads = BTreeMap::from([
            ("motor-dataset".to_string(), b"abc".to_vec()),
            ("inverter-dataset".to_string(), b"abc".to_vec()),
            ("propeller-dataset".to_string(), b"abc".to_vec()),
        ]);
        propulsion.verify_evidence_payloads(&payloads).unwrap();

        let mut missing = payloads.clone();
        missing.remove("inverter-dataset");
        assert!(matches!(
            propulsion.verify_evidence_payloads(&missing),
            Err(CharacterizationError::MissingEvidencePayload(id)) if id == "inverter-dataset"
        ));

        let mut corrupt = payloads;
        corrupt.insert("propeller-dataset".into(), b"not-the-artifact".to_vec());
        assert!(matches!(
            propulsion.verify_evidence_payloads(&corrupt),
            Err(CharacterizationError::EvidenceDigestMismatch { .. })
        ));
    }

    #[test]
    fn evidence_requires_sha256_digest() {
        let mut item = evidence("motor-data");
        item.sha256 = "not-a-digest".into();
        assert!(matches!(item.validate(), Err(CharacterizationError::InvalidEvidence(_))));
    }
}
