// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Closed-form heat-pump screening in physical units.
//!
//! This is an engineering feasibility model, not an equipment-performance curve.
//! It enforces the first-law balance and the Carnot heating-COP ceiling so a
//! higher-grade thermal output cannot appear without explicit electrical work.

use std::error::Error;
use std::fmt;

const KELVIN_OFFSET: f64 = 273.15;
const TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatPump {
    /// Maximum electrical input power accepted by the model (W).
    pub rated_electrical_power_w: f64,
    /// Heating coefficient of performance, `Q_hot / W_electric`.
    pub heating_cop: f64,
}

impl HeatPump {
    pub fn new(rated_electrical_power_w: f64, heating_cop: f64) -> Result<Self, HeatPumpError> {
        if !rated_electrical_power_w.is_finite() || rated_electrical_power_w <= 0.0 {
            return Err(HeatPumpError::InvalidRatedPower(rated_electrical_power_w));
        }
        if !heating_cop.is_finite() || heating_cop < 1.0 {
            return Err(HeatPumpError::InvalidHeatingCop(heating_cop));
        }
        Ok(Self {
            rated_electrical_power_w,
            heating_cop,
        })
    }

    /// Screen one steady-state heating operating point.
    ///
    /// `source_temperature_c` and `sink_temperature_c` are idealized reservoir
    /// temperatures for thermodynamic bounding. Real equipment needs approach
    /// temperatures, compressor maps, flow constraints, defrost behavior, etc.
    pub fn heating_step(
        &self,
        electrical_input_w: f64,
        source_temperature_c: f64,
        sink_temperature_c: f64,
    ) -> Result<HeatPumpFlow, HeatPumpError> {
        if !electrical_input_w.is_finite() || electrical_input_w < 0.0 {
            return Err(HeatPumpError::InvalidElectricalInput(electrical_input_w));
        }
        if electrical_input_w > self.rated_electrical_power_w + TOLERANCE {
            return Err(HeatPumpError::ElectricalInputExceedsRating {
                requested_w: electrical_input_w,
                rated_w: self.rated_electrical_power_w,
            });
        }

        let carnot_cop = carnot_heating_cop(source_temperature_c, sink_temperature_c)?;
        if self.heating_cop > carnot_cop + TOLERANCE {
            return Err(HeatPumpError::HeatingCopExceedsCarnot {
                requested_cop: self.heating_cop,
                carnot_cop,
            });
        }

        let delivered_heat_w = electrical_input_w * self.heating_cop;
        let source_heat_w = delivered_heat_w - electrical_input_w;
        Ok(HeatPumpFlow {
            electrical_input_w,
            source_heat_w,
            delivered_heat_w,
            source_temperature_c,
            sink_temperature_c,
            heating_cop: self.heating_cop,
            carnot_cop,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatPumpFlow {
    pub electrical_input_w: f64,
    /// Heat extracted from the source reservoir (W).
    pub source_heat_w: f64,
    /// Heat delivered to the hot sink (W).
    pub delivered_heat_w: f64,
    pub source_temperature_c: f64,
    pub sink_temperature_c: f64,
    pub heating_cop: f64,
    pub carnot_cop: f64,
}

impl HeatPumpFlow {
    /// First-law residual: `Q_hot - Q_source - W_electric`.
    pub fn conservation_residual_w(&self) -> f64 {
        self.delivered_heat_w - self.source_heat_w - self.electrical_input_w
    }

    pub fn conserved_within(&self, tolerance_w: f64) -> bool {
        tolerance_w.is_finite()
            && tolerance_w >= 0.0
            && self.conservation_residual_w().abs() <= tolerance_w
    }
}

/// Ideal reversible heating COP, `T_hot / (T_hot - T_cold)` using kelvin.
pub fn carnot_heating_cop(
    source_temperature_c: f64,
    sink_temperature_c: f64,
) -> Result<f64, HeatPumpError> {
    if !source_temperature_c.is_finite() || !sink_temperature_c.is_finite() {
        return Err(HeatPumpError::InvalidTemperature);
    }
    let source_k = source_temperature_c + KELVIN_OFFSET;
    let sink_k = sink_temperature_c + KELVIN_OFFSET;
    if source_k <= 0.0 || sink_k <= 0.0 {
        return Err(HeatPumpError::InvalidTemperature);
    }
    if sink_k <= source_k {
        return Err(HeatPumpError::InvalidTemperatureLift {
            source_temperature_c,
            sink_temperature_c,
        });
    }
    Ok(sink_k / (sink_k - source_k))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeatPumpError {
    InvalidTemperature,
    InvalidTemperatureLift {
        source_temperature_c: f64,
        sink_temperature_c: f64,
    },
    InvalidRatedPower(f64),
    InvalidHeatingCop(f64),
    InvalidElectricalInput(f64),
    ElectricalInputExceedsRating {
        requested_w: f64,
        rated_w: f64,
    },
    HeatingCopExceedsCarnot {
        requested_cop: f64,
        carnot_cop: f64,
    },
}

impl fmt::Display for HeatPumpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTemperature => write!(formatter, "heat-pump temperature must be finite and above absolute zero"),
            Self::InvalidTemperatureLift {
                source_temperature_c,
                sink_temperature_c,
            } => write!(
                formatter,
                "heating sink temperature {sink_temperature_c} C must exceed source temperature {source_temperature_c} C"
            ),
            Self::InvalidRatedPower(value) => {
                write!(formatter, "heat-pump rated electrical power must be positive and finite, got {value}")
            }
            Self::InvalidHeatingCop(value) => {
                write!(formatter, "heating COP must be finite and at least 1, got {value}")
            }
            Self::InvalidElectricalInput(value) => {
                write!(formatter, "electrical input must be finite and non-negative, got {value}")
            }
            Self::ElectricalInputExceedsRating { requested_w, rated_w } => write!(
                formatter,
                "requested heat-pump electrical input {requested_w} W exceeds rating {rated_w} W"
            ),
            Self::HeatingCopExceedsCarnot {
                requested_cop,
                carnot_cop,
            } => write!(
                formatter,
                "requested heating COP {requested_cop} exceeds Carnot ceiling {carnot_cop}"
            ),
        }
    }
}

impl Error for HeatPumpError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carnot_heating_cop_matches_textbook_relation() {
        let cop = carnot_heating_cop(0.0, 40.0).unwrap();
        let expected = (40.0 + KELVIN_OFFSET) / 40.0;
        assert!((cop - expected).abs() < 1e-12);
    }

    #[test]
    fn grade_upgrade_has_explicit_electrical_and_source_heat_cost() {
        let pump = HeatPump::new(25_000.0, 4.0).unwrap();
        let flow = pump.heating_step(20_000.0, 42.0, 65.0).unwrap();
        assert!((flow.delivered_heat_w - 80_000.0).abs() < 1e-9);
        assert!((flow.source_heat_w - 60_000.0).abs() < 1e-9);
        assert!(flow.conserved_within(1e-9));
        assert!(flow.carnot_cop > flow.heating_cop);
    }

    #[test]
    fn impossible_cop_above_carnot_is_rejected() {
        let pump = HeatPump::new(10_000.0, 10.0).unwrap();
        let error = pump.heating_step(5_000.0, 0.0, 40.0).unwrap_err();
        assert!(matches!(error, HeatPumpError::HeatingCopExceedsCarnot { .. }));
    }

    #[test]
    fn electrical_rating_is_enforced() {
        let pump = HeatPump::new(10_000.0, 3.0).unwrap();
        let error = pump.heating_step(10_001.0, 10.0, 45.0).unwrap_err();
        assert!(matches!(
            error,
            HeatPumpError::ElectricalInputExceedsRating { .. }
        ));
    }

    #[test]
    fn non_positive_temperature_lift_is_rejected() {
        let pump = HeatPump::new(10_000.0, 3.0).unwrap();
        assert!(matches!(
            pump.heating_step(5_000.0, 45.0, 40.0),
            Err(HeatPumpError::InvalidTemperatureLift { .. })
        ));
    }

    #[test]
    fn invalid_constructor_values_fail_closed() {
        assert!(matches!(
            HeatPump::new(0.0, 3.0),
            Err(HeatPumpError::InvalidRatedPower(_))
        ));
        assert!(matches!(
            HeatPump::new(10_000.0, 0.9),
            Err(HeatPumpError::InvalidHeatingCop(_))
        ));
    }
}
