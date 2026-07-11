// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Solar and wind generation models.
//!
//! Solar follows the NREL PVWatts v5 approach (NOCT-based cell-temperature
//! model, temperature derating, inverter efficiency, system losses) — the
//! same method Terra Atlas's `lib/forecast/generation-forecaster.ts` uses
//! (NREL/TP-6A20-62641), reimplemented here in Rust rather than reused
//! directly since that's TypeScript in a separate app. Wind follows the
//! standard cubic (power ∝ v³) turbine power-speed curve described in
//! IEC 61400-12-1.
//!
//! Both models take measured/forecasted weather (irradiance, wind speed) as
//! input — neither derives solar position or atmospheric conditions from
//! first principles, matching how Terra Atlas itself sources this data from
//! an external weather API rather than computing it analytically.

use serde::{Deserialize, Serialize};

/// A fixed-tilt PV array, PVWatts v5-style.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SolarArray {
    /// DC nameplate capacity at Standard Test Conditions (1000 W/m^2, 25°C cell temp), kW.
    pub rated_capacity_kw: f64,
    /// Nominal Operating Cell Temperature, °C (typical crystalline-silicon module: ~45°C).
    pub noct_c: f64,
    /// Power temperature coefficient, fraction per °C above 25°C (typically
    /// negative, e.g. -0.004 for crystalline silicon: power falls as cells heat up).
    pub temp_coefficient_per_c: f64,
    /// Inverter (DC->AC) conversion efficiency, [0, 1].
    pub inverter_efficiency: f64,
    /// Fractional system losses (soiling, wiring, mismatch, availability),
    /// [0, 1). PVWatts' own default is 0.14.
    pub system_losses: f64,
}

impl SolarArray {
    /// AC power output (kW) given plane-of-array irradiance (W/m^2) and
    /// ambient temperature (°C).
    pub fn ac_power_kw(&self, irradiance_w_per_m2: f64, ambient_temp_c: f64) -> f64 {
        if irradiance_w_per_m2 <= 0.0 {
            return 0.0;
        }
        // NOCT cell-temperature model.
        let cell_temp_c = ambient_temp_c + (self.noct_c - 20.0) / 800.0 * irradiance_w_per_m2;
        let temp_derate = (1.0 + self.temp_coefficient_per_c * (cell_temp_c - 25.0)).max(0.0);
        let dc_power_kw = self.rated_capacity_kw * (irradiance_w_per_m2 / 1000.0) * temp_derate;
        (dc_power_kw * (1.0 - self.system_losses) * self.inverter_efficiency).max(0.0)
    }
}

/// A wind turbine with the standard cut-in/rated/cut-out cubic power curve
/// (IEC 61400-12-1-shaped): power rises with the cube of wind speed between
/// cut-in and rated, is flat at rated capacity between rated and cut-out,
/// and is zero outside that range (including above cut-out, where the
/// turbine feathers/shuts down to avoid overspeed damage).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WindTurbine {
    pub rated_capacity_kw: f64,
    pub cut_in_speed_m_s: f64,
    pub rated_speed_m_s: f64,
    pub cut_out_speed_m_s: f64,
}

impl WindTurbine {
    pub fn power_kw(&self, wind_speed_m_s: f64) -> f64 {
        if wind_speed_m_s < self.cut_in_speed_m_s || wind_speed_m_s >= self.cut_out_speed_m_s {
            0.0
        } else if wind_speed_m_s < self.rated_speed_m_s {
            let v3 = wind_speed_m_s.powi(3);
            let vci3 = self.cut_in_speed_m_s.powi(3);
            let vr3 = self.rated_speed_m_s.powi(3);
            self.rated_capacity_kw * (v3 - vci3) / (vr3 - vci3)
        } else {
            self.rated_capacity_kw
        }
    }
}

/// Deterministic synthetic diurnal irradiance profile: zero outside
/// `[sunrise_hour, sunset_hour)`, a smooth sine-bell peak at solar noon
/// otherwise. For scenario/test harnesses only -- real deployments should
/// use measured/forecasted irradiance (e.g. Terra Atlas's Open-Meteo feed),
/// not this.
pub fn synthetic_irradiance_w_per_m2(
    time_of_day_hours: f64,
    peak_irradiance_w_per_m2: f64,
    sunrise_hour: f64,
    sunset_hour: f64,
) -> f64 {
    if time_of_day_hours <= sunrise_hour
        || time_of_day_hours >= sunset_hour
        || sunset_hour <= sunrise_hour
    {
        return 0.0;
    }
    let day_fraction = (time_of_day_hours - sunrise_hour) / (sunset_hour - sunrise_hour);
    (peak_irradiance_w_per_m2 * (std::f64::consts::PI * day_fraction).sin()).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_array() -> SolarArray {
        SolarArray {
            rated_capacity_kw: 100.0,
            noct_c: 45.0,
            temp_coefficient_per_c: -0.004,
            inverter_efficiency: 0.96,
            system_losses: 0.14,
        }
    }

    #[test]
    fn test_zero_irradiance_gives_zero_power() {
        let array = default_array();
        assert_eq!(array.ac_power_kw(0.0, 25.0), 0.0);
    }

    #[test]
    fn test_stc_equivalent_conditions_give_expected_power() {
        // Choose ambient_temp_c so cell_temp_c lands exactly at 25C (STC) at
        // full 1000 W/m^2 irradiance: 25 = ambient + (45-20)/800*1000 = ambient + 31.25
        let array = default_array();
        let ambient_for_stc_cell_temp = 25.0 - (array.noct_c - 20.0) / 800.0 * 1000.0;
        let power = array.ac_power_kw(1000.0, ambient_for_stc_cell_temp);
        // At exactly 25C cell temp, temp_derate = 1.0 exactly, so power =
        // rated * 1.0 * (1-losses) * inverter_eff, hand-computable.
        let expected =
            array.rated_capacity_kw * (1.0 - array.system_losses) * array.inverter_efficiency;
        assert!(
            (power - expected).abs() < 1e-9,
            "got {power}, expected {expected}"
        );
    }

    #[test]
    fn test_higher_ambient_temp_derates_power() {
        let array = default_array();
        let cool_power = array.ac_power_kw(800.0, 15.0);
        let hot_power = array.ac_power_kw(800.0, 40.0);
        assert!(
            hot_power < cool_power,
            "hotter ambient should derate power: cool={cool_power} hot={hot_power}"
        );
    }

    #[test]
    fn test_wind_below_cut_in_gives_zero() {
        let turbine = WindTurbine {
            rated_capacity_kw: 2000.0,
            cut_in_speed_m_s: 3.0,
            rated_speed_m_s: 12.0,
            cut_out_speed_m_s: 25.0,
        };
        assert_eq!(turbine.power_kw(2.0), 0.0);
    }

    #[test]
    fn test_wind_at_or_above_cut_out_gives_zero() {
        let turbine = WindTurbine {
            rated_capacity_kw: 2000.0,
            cut_in_speed_m_s: 3.0,
            rated_speed_m_s: 12.0,
            cut_out_speed_m_s: 25.0,
        };
        assert_eq!(turbine.power_kw(25.0), 0.0);
        assert_eq!(turbine.power_kw(30.0), 0.0);
    }

    #[test]
    fn test_wind_at_rated_speed_gives_full_rated_power() {
        let turbine = WindTurbine {
            rated_capacity_kw: 2000.0,
            cut_in_speed_m_s: 3.0,
            rated_speed_m_s: 12.0,
            cut_out_speed_m_s: 25.0,
        };
        assert_eq!(turbine.power_kw(12.0), 2000.0);
        assert_eq!(turbine.power_kw(18.0), 2000.0); // flat between rated and cut-out
    }

    #[test]
    fn test_wind_cubic_curve_matches_hand_computed_value() {
        let turbine = WindTurbine {
            rated_capacity_kw: 2000.0,
            cut_in_speed_m_s: 3.0,
            rated_speed_m_s: 12.0,
            cut_out_speed_m_s: 25.0,
        };
        // At 8 m/s: P = 2000 * (8^3 - 3^3) / (12^3 - 3^3) = 2000*(512-27)/(1728-27)
        let expected = 2000.0 * (512.0 - 27.0) / (1728.0 - 27.0);
        assert!((turbine.power_kw(8.0) - expected).abs() < 1e-9);
    }

    #[test]
    fn test_wind_power_monotonically_increases_between_cut_in_and_rated() {
        let turbine = WindTurbine {
            rated_capacity_kw: 2000.0,
            cut_in_speed_m_s: 3.0,
            rated_speed_m_s: 12.0,
            cut_out_speed_m_s: 25.0,
        };
        let mut prev = turbine.power_kw(3.0);
        let mut v = 3.5;
        while v < 12.0 {
            let p = turbine.power_kw(v);
            assert!(
                p >= prev,
                "power should be non-decreasing: v={v} p={p} prev={prev}"
            );
            prev = p;
            v += 0.5;
        }
    }

    #[test]
    fn test_synthetic_irradiance_zero_at_night() {
        assert_eq!(synthetic_irradiance_w_per_m2(2.0, 1000.0, 6.0, 18.0), 0.0);
        assert_eq!(synthetic_irradiance_w_per_m2(22.0, 1000.0, 6.0, 18.0), 0.0);
        assert_eq!(synthetic_irradiance_w_per_m2(6.0, 1000.0, 6.0, 18.0), 0.0);
        assert_eq!(synthetic_irradiance_w_per_m2(18.0, 1000.0, 6.0, 18.0), 0.0);
    }

    #[test]
    fn test_synthetic_irradiance_peaks_at_solar_noon() {
        let noon = synthetic_irradiance_w_per_m2(12.0, 1000.0, 6.0, 18.0);
        let morning = synthetic_irradiance_w_per_m2(8.0, 1000.0, 6.0, 18.0);
        let afternoon = synthetic_irradiance_w_per_m2(16.0, 1000.0, 6.0, 18.0);
        assert!((noon - 1000.0).abs() < 1e-6, "got {noon}");
        assert!(morning < noon && afternoon < noon);
    }
}
