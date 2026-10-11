// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Strict unit mapping for versioned interval-energy study inputs.
//!
//! Sol Atlas fixture profiles are interval energies (kWh/interval), while the
//! Symthaea scheduler consumes mean power (kW) sampled for a declared duration.
//! This module performs only that explicit unit conversion and validates row
//! correspondence. It does not parse or authenticate source artifacts, infer
//! a tariff timezone, enforce a reserve floor, or evaluate critical-load
//! service. In particular, carrying a critical profile is not evidence that
//! the aggregate scheduler has served it.

use serde::{Deserialize, Serialize};

const MAX_MAPPED_INTERVALS: usize = 1_000_000;
pub const ENERGY_STUDY_PROFILE_MAPPING_SCHEMA_VERSION: u16 = 1;

/// Parsed interval-energy inputs after their source bytes have been checked by
/// the upstream manifest/artifact verifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnergyStudyProfilesV1 {
    /// UTC Unix timestamp of the first interval start; no local-time inference.
    pub start_unix_seconds: i64,
    /// Fixed duration represented by every row, in seconds.
    pub interval_seconds: u32,
    /// Total demand energy in each interval, kWh/interval.
    pub load_kwh_per_interval: Vec<f64>,
    /// Critical subset of demand energy, kWh/interval.
    pub critical_load_kwh_per_interval: Vec<f64>,
    /// Renewable generation energy in each interval, kWh/interval.
    pub renewable_generation_kwh_per_interval: Vec<f64>,
    /// Grid-availability state for each interval.
    pub grid_available_by_interval: Vec<bool>,
}

/// Status intentionally remains unevaluated until a priority-aware service
/// model and independent checker consume the critical-load profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CriticalLoadServiceStatusV1 {
    NotEvaluatedByAggregateScheduler,
}

/// Mean-power representation of the same frozen interval inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MappedEnergyStudyProfilesV1 {
    pub schema_version: u16,
    pub start_unix_seconds: i64,
    pub interval_seconds: u32,
    pub interval_count: usize,
    /// Mean demand power over each interval, kW.
    pub load_kw: Vec<f64>,
    /// Mean critical-demand power over each interval, kW.
    pub critical_load_kw: Vec<f64>,
    /// Mean renewable-generation power over each interval, kW.
    pub renewable_generation_kw: Vec<f64>,
    pub grid_available_by_interval: Vec<bool>,
    /// This is an explicit non-pass; aggregate-only receipts cannot evaluate it.
    pub critical_load_service_status: CriticalLoadServiceStatusV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyProfileMappingError {
    InvalidIntervalDuration,
    EmptyProfile,
    TooManyIntervals,
    ProfileLengthMismatch,
    InvalidEnergyValue,
    CriticalLoadExceedsTotalLoad,
    PowerConversionOverflow,
}

impl std::fmt::Display for EnergyProfileMappingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidIntervalDuration => f.write_str("interval duration must be positive"),
            Self::EmptyProfile => f.write_str("energy study must contain at least one interval"),
            Self::TooManyIntervals => write!(
                f,
                "energy study exceeds the {MAX_MAPPED_INTERVALS}-interval mapping limit"
            ),
            Self::ProfileLengthMismatch => {
                f.write_str("energy-study profile lengths do not match")
            }
            Self::InvalidEnergyValue => {
                f.write_str("energy profile values must be finite and non-negative")
            }
            Self::CriticalLoadExceedsTotalLoad => {
                f.write_str("critical load energy exceeds total load energy")
            }
            Self::PowerConversionOverflow => {
                f.write_str("interval-energy to mean-power conversion overflowed")
            }
        }
    }
}

impl std::error::Error for EnergyProfileMappingError {}

impl EnergyStudyProfilesV1 {
    /// Convert kWh/interval to mean kW with kW = kWh / (seconds / 3600).
    ///
    /// The method preserves the UTC start time and grid-state ordinal exactly.
    /// It does not derive wall-clock tariff time from the opaque region ID.
    pub fn map_to_mean_power_v1(
        &self,
    ) -> Result<MappedEnergyStudyProfilesV1, EnergyProfileMappingError> {
        if self.interval_seconds == 0 {
            return Err(EnergyProfileMappingError::InvalidIntervalDuration);
        }

        let count = self.load_kwh_per_interval.len();
        if count == 0 {
            return Err(EnergyProfileMappingError::EmptyProfile);
        }
        if count > MAX_MAPPED_INTERVALS {
            return Err(EnergyProfileMappingError::TooManyIntervals);
        }
        if self.critical_load_kwh_per_interval.len() != count
            || self.renewable_generation_kwh_per_interval.len() != count
            || self.grid_available_by_interval.len() != count
        {
            return Err(EnergyProfileMappingError::ProfileLengthMismatch);
        }

        for ((&load, &critical), &generation) in self
            .load_kwh_per_interval
            .iter()
            .zip(&self.critical_load_kwh_per_interval)
            .zip(&self.renewable_generation_kwh_per_interval)
        {
            if [load, critical, generation]
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(EnergyProfileMappingError::InvalidEnergyValue);
            }
            if critical > load {
                return Err(EnergyProfileMappingError::CriticalLoadExceedsTotalLoad);
            }
        }

        let interval_hours = f64::from(self.interval_seconds) / 3600.0;
        if !interval_hours.is_finite() || interval_hours <= 0.0 {
            return Err(EnergyProfileMappingError::InvalidIntervalDuration);
        }

        let convert = |values: &[f64]| -> Result<Vec<f64>, EnergyProfileMappingError> {
            let mapped: Vec<f64> = values.iter().map(|value| value / interval_hours).collect();
            if mapped.iter().any(|value| !value.is_finite() || *value < 0.0) {
                return Err(EnergyProfileMappingError::PowerConversionOverflow);
            }
            Ok(mapped)
        };

        Ok(MappedEnergyStudyProfilesV1 {
            schema_version: ENERGY_STUDY_PROFILE_MAPPING_SCHEMA_VERSION,
            start_unix_seconds: self.start_unix_seconds,
            interval_seconds: self.interval_seconds,
            interval_count: count,
            load_kw: convert(&self.load_kwh_per_interval)?,
            critical_load_kw: convert(&self.critical_load_kwh_per_interval)?,
            renewable_generation_kw: convert(&self.renewable_generation_kwh_per_interval)?,
            grid_available_by_interval: self.grid_available_by_interval.clone(),
            critical_load_service_status:
                CriticalLoadServiceStatusV1::NotEvaluatedByAggregateScheduler,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sol_atlas_cloud_evening_fixture() -> EnergyStudyProfilesV1 {
        EnergyStudyProfilesV1 {
            start_unix_seconds: 1_797_000_000,
            interval_seconds: 900,
            load_kwh_per_interval: vec![2.0, 2.5, 3.0, 4.0, 5.0, 5.0, 3.5, 3.0],
            critical_load_kwh_per_interval: vec![1.0, 1.0, 1.2, 1.5, 1.5, 1.5, 1.2, 1.0],
            renewable_generation_kwh_per_interval: vec![0.0, 0.1, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0],
            grid_available_by_interval: vec![true, true, true, true, false, false, true, true],
        }
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn maps_frozen_15_minute_energy_profiles_to_mean_power() {
        let mapped = sol_atlas_cloud_evening_fixture()
            .map_to_mean_power_v1()
            .unwrap();

        assert_eq!(mapped.schema_version, 1);
        assert_eq!(mapped.start_unix_seconds, 1_797_000_000);
        assert_eq!(mapped.interval_seconds, 900);
        assert_eq!(mapped.interval_count, 8);
        assert_eq!(
            mapped.grid_available_by_interval,
            vec![true, true, true, true, false, false, true, true]
        );
        let expected_load_kw = [8.0, 10.0, 12.0, 16.0, 20.0, 20.0, 14.0, 12.0];
        for (actual, expected) in mapped.load_kw.iter().zip(expected_load_kw) {
            close(*actual, expected);
        }

        let expected_critical_load_kw = [4.0, 4.0, 4.8, 6.0, 6.0, 6.0, 4.8, 4.0];
        for (actual, expected) in mapped.critical_load_kw.iter().zip(expected_critical_load_kw) {
            close(*actual, expected);
        }

        let expected_generation_kw = [0.0, 0.4, 2.0, 4.0, 2.0, 0.0, 0.0, 0.0];
        for (actual, expected) in mapped.renewable_generation_kw.iter().zip(expected_generation_kw) {
            close(*actual, expected);
        }
        assert_eq!(
            mapped.critical_load_service_status,
            CriticalLoadServiceStatusV1::NotEvaluatedByAggregateScheduler
        );
    }

    #[test]
    fn rejects_bad_time_axis_lengths_and_critical_demand() {
        let mut input = sol_atlas_cloud_evening_fixture();
        input.interval_seconds = 0;
        assert_eq!(
            input.map_to_mean_power_v1(),
            Err(EnergyProfileMappingError::InvalidIntervalDuration)
        );

        let mut input = sol_atlas_cloud_evening_fixture();
        input.grid_available_by_interval.pop();
        assert_eq!(
            input.map_to_mean_power_v1(),
            Err(EnergyProfileMappingError::ProfileLengthMismatch)
        );

        let mut input = sol_atlas_cloud_evening_fixture();
        input.critical_load_kwh_per_interval[0] = 2.1;
        assert_eq!(
            input.map_to_mean_power_v1(),
            Err(EnergyProfileMappingError::CriticalLoadExceedsTotalLoad)
        );
    }

    #[test]
    fn rejects_nonfinite_negative_and_overflowing_energy_values() {
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            let mut input = sol_atlas_cloud_evening_fixture();
            input.load_kwh_per_interval[0] = invalid;
            assert_eq!(
                input.map_to_mean_power_v1(),
                Err(EnergyProfileMappingError::InvalidEnergyValue)
            );
        }

        let mut input = sol_atlas_cloud_evening_fixture();
        input.interval_seconds = 1;
        input.load_kwh_per_interval[0] = f64::MAX;
        assert_eq!(
            input.map_to_mean_power_v1(),
            Err(EnergyProfileMappingError::PowerConversionOverflow)
        );
    }
}
