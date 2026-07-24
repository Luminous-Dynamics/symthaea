// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conserved single-bucket land-water baseline.
//!
//! The bucket receives precipitation, loses water through a storage-limited
//! evapotranspiration flux, and spills excess storage as runoff. Under constant
//! forcing the model has a closed-form solution and an exact cumulative water
//! budget, making it useful as an oracle for larger land-surface simulations.
//! It does not resolve snow, groundwater, infiltration fronts, vegetation, or
//! routing between catchments.

use crate::error::{ModelError, require_finite, require_non_negative, require_positive};
use crate::transient::MAX_TRAJECTORY_STEPS;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydrologyBucket {
    /// Maximum liquid-water storage, mm water equivalent.
    pub capacity_mm: f64,
    /// Evapotranspiration at full storage, mm/day.
    pub potential_evapotranspiration_mm_per_day: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydrologyState {
    pub storage_mm: f64,
    pub cumulative_precipitation_mm: f64,
    pub cumulative_evapotranspiration_mm: f64,
    pub cumulative_runoff_mm: f64,
}

impl HydrologyState {
    pub fn soil_moisture_fraction(&self, capacity_mm: f64) -> f64 {
        self.storage_mm / capacity_mm
    }

    /// `initial storage + precipitation - ET - runoff - final storage`.
    pub fn budget_residual_mm(&self, initial_storage_mm: f64) -> f64 {
        initial_storage_mm + self.cumulative_precipitation_mm
            - self.cumulative_evapotranspiration_mm
            - self.cumulative_runoff_mm
            - self.storage_mm
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydrologySample {
    pub time_days: f64,
    pub storage_mm: f64,
    pub soil_moisture_fraction: f64,
    pub actual_evapotranspiration_mm_per_day: f64,
    pub runoff_mm_per_day: f64,
    pub cumulative_precipitation_mm: f64,
    pub cumulative_evapotranspiration_mm: f64,
    pub cumulative_runoff_mm: f64,
    pub budget_residual_mm: f64,
}

impl HydrologyBucket {
    fn validated_state(&self, state: HydrologyState) -> Result<HydrologyState, ModelError> {
        self.validate_storage(state.storage_mm)?;
        require_non_negative(
            "cumulative_precipitation_mm",
            state.cumulative_precipitation_mm,
        )?;
        require_non_negative(
            "cumulative_evapotranspiration_mm",
            state.cumulative_evapotranspiration_mm,
        )?;
        require_non_negative("cumulative_runoff_mm", state.cumulative_runoff_mm)?;
        Ok(state)
    }

    pub fn try_new(
        capacity_mm: f64,
        potential_evapotranspiration_mm_per_day: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            capacity_mm,
            potential_evapotranspiration_mm_per_day,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("capacity_mm", self.capacity_mm)?;
        require_non_negative(
            "potential_evapotranspiration_mm_per_day",
            self.potential_evapotranspiration_mm_per_day,
        )
    }

    pub fn validate_storage(&self, storage_mm: f64) -> Result<(), ModelError> {
        require_finite("storage_mm", storage_mm)?;
        if (0.0..=self.capacity_mm).contains(&storage_mm) {
            Ok(())
        } else {
            Err(ModelError::OutOfRange {
                parameter: "storage_mm",
                value: storage_mm,
                min: 0.0,
                max: self.capacity_mm,
            })
        }
    }

    /// Storage-limited evapotranspiration, mm/day.
    pub fn actual_evapotranspiration(&self, storage_mm: f64) -> Result<f64, ModelError> {
        self.validate()?;
        self.validate_storage(storage_mm)?;
        Ok(self.potential_evapotranspiration_mm_per_day * storage_mm / self.capacity_mm)
    }

    /// Exact state under constant precipitation for `elapsed_days`.
    pub fn exact_constant_precipitation(
        &self,
        initial_storage_mm: f64,
        precipitation_mm_per_day: f64,
        elapsed_days: f64,
    ) -> Result<HydrologyState, ModelError> {
        self.validate()?;
        self.validate_storage(initial_storage_mm)?;
        require_non_negative("precipitation_mm_per_day", precipitation_mm_per_day)?;
        require_non_negative("elapsed_days", elapsed_days)?;

        let cumulative_precipitation_mm = precipitation_mm_per_day * elapsed_days;
        require_finite("cumulative_precipitation_mm", cumulative_precipitation_mm)?;

        let et_capacity = self.potential_evapotranspiration_mm_per_day;
        if elapsed_days == 0.0 {
            return self.validated_state(HydrologyState {
                storage_mm: initial_storage_mm,
                cumulative_precipitation_mm: 0.0,
                cumulative_evapotranspiration_mm: 0.0,
                cumulative_runoff_mm: 0.0,
            });
        }

        if et_capacity == 0.0 {
            let unconstrained = initial_storage_mm + cumulative_precipitation_mm;
            let storage_mm = unconstrained.min(self.capacity_mm);
            return self.validated_state(HydrologyState {
                storage_mm,
                cumulative_precipitation_mm,
                cumulative_evapotranspiration_mm: 0.0,
                cumulative_runoff_mm: (unconstrained - self.capacity_mm).max(0.0),
            });
        }

        let rate = et_capacity / self.capacity_mm;
        let unconstrained_equilibrium = precipitation_mm_per_day / rate;
        let reaches_capacity =
            precipitation_mm_per_day > et_capacity && initial_storage_mm < self.capacity_mm;

        let hit_time = if reaches_capacity {
            let numerator = self.capacity_mm - unconstrained_equilibrium;
            let denominator = initial_storage_mm - unconstrained_equilibrium;
            let ratio = numerator / denominator;
            if ratio > 0.0 && ratio <= 1.0 {
                Some(-ratio.ln() / rate)
            } else {
                None
            }
        } else if precipitation_mm_per_day > et_capacity && initial_storage_mm == self.capacity_mm {
            Some(0.0)
        } else {
            None
        };

        if let Some(time_to_capacity) = hit_time
            && time_to_capacity <= elapsed_days
        {
            let cumulative_et_before =
                precipitation_mm_per_day * time_to_capacity + initial_storage_mm - self.capacity_mm;
            let saturated_duration = elapsed_days - time_to_capacity;
            let cumulative_evapotranspiration_mm =
                cumulative_et_before + et_capacity * saturated_duration;
            let cumulative_runoff_mm =
                (precipitation_mm_per_day - et_capacity) * saturated_duration;
            return self.validated_state(HydrologyState {
                storage_mm: self.capacity_mm,
                cumulative_precipitation_mm,
                cumulative_evapotranspiration_mm,
                cumulative_runoff_mm,
            });
        }

        let storage_mm = unconstrained_equilibrium
            + (initial_storage_mm - unconstrained_equilibrium) * (-rate * elapsed_days).exp();
        let storage_mm = storage_mm.clamp(0.0, self.capacity_mm);
        let cumulative_evapotranspiration_mm =
            cumulative_precipitation_mm + initial_storage_mm - storage_mm;
        self.validated_state(HydrologyState {
            storage_mm,
            cumulative_precipitation_mm,
            cumulative_evapotranspiration_mm,
            cumulative_runoff_mm: 0.0,
        })
    }

    /// Exact constant-precipitation trajectory including the initial state.
    pub fn exact_trajectory(
        &self,
        initial_storage_mm: f64,
        precipitation_mm_per_day: f64,
        dt_days: f64,
        steps: usize,
    ) -> Result<Vec<HydrologySample>, ModelError> {
        self.validate()?;
        self.validate_storage(initial_storage_mm)?;
        require_non_negative("precipitation_mm_per_day", precipitation_mm_per_day)?;
        require_positive("dt_days", dt_days)?;
        if steps == 0 {
            return Err(ModelError::ZeroSteps);
        }
        if steps > MAX_TRAJECTORY_STEPS {
            return Err(ModelError::TrajectoryTooLarge {
                requested: steps,
                maximum: MAX_TRAJECTORY_STEPS,
            });
        }
        let capacity = steps.checked_add(1).ok_or(ModelError::TrajectoryTooLarge {
            requested: usize::MAX,
            maximum: MAX_TRAJECTORY_STEPS,
        })?;
        require_finite("duration_days", dt_days * steps as f64)?;

        let mut samples = Vec::with_capacity(capacity);
        for step in 0..=steps {
            let time_days = step as f64 * dt_days;
            let state = self.exact_constant_precipitation(
                initial_storage_mm,
                precipitation_mm_per_day,
                time_days,
            )?;
            let actual_evapotranspiration_mm_per_day =
                self.actual_evapotranspiration(state.storage_mm)?;
            let runoff_mm_per_day = if state.storage_mm == self.capacity_mm {
                (precipitation_mm_per_day - actual_evapotranspiration_mm_per_day).max(0.0)
            } else {
                0.0
            };
            samples.push(HydrologySample {
                time_days,
                storage_mm: state.storage_mm,
                soil_moisture_fraction: state.soil_moisture_fraction(self.capacity_mm),
                actual_evapotranspiration_mm_per_day,
                runoff_mm_per_day,
                cumulative_precipitation_mm: state.cumulative_precipitation_mm,
                cumulative_evapotranspiration_mm: state.cumulative_evapotranspiration_mm,
                cumulative_runoff_mm: state.cumulative_runoff_mm,
                budget_residual_mm: state.budget_residual_mm(initial_storage_mm),
            });
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bucket() -> HydrologyBucket {
        HydrologyBucket::try_new(100.0, 5.0).unwrap()
    }

    #[test]
    fn drydown_matches_exponential_solution_and_closes_budget() {
        let state = bucket()
            .exact_constant_precipitation(100.0, 0.0, 20.0)
            .unwrap();
        let expected = 100.0 * (-1.0_f64).exp();
        assert!((state.storage_mm - expected).abs() < 1.0e-12);
        assert!(state.budget_residual_mm(100.0).abs() < 1.0e-12);
        assert_eq!(state.cumulative_runoff_mm, 0.0);
    }

    #[test]
    fn subcapacity_equilibrium_is_recovered() {
        let state = bucket()
            .exact_constant_precipitation(10.0, 2.5, 1_000.0)
            .unwrap();
        assert!((state.storage_mm - 50.0).abs() < 1.0e-10);
        assert_eq!(state.cumulative_runoff_mm, 0.0);
        assert!(state.budget_residual_mm(10.0).abs() < 1.0e-10);
    }

    #[test]
    fn sustained_surplus_fills_bucket_then_runs_off() {
        let state = bucket()
            .exact_constant_precipitation(20.0, 10.0, 100.0)
            .unwrap();
        assert_eq!(state.storage_mm, 100.0);
        assert!(state.cumulative_runoff_mm > 0.0);
        assert!(state.budget_residual_mm(20.0).abs() < 1.0e-10);
    }

    #[test]
    fn zero_et_bucket_has_exact_overflow_budget() {
        let model = HydrologyBucket::try_new(50.0, 0.0).unwrap();
        let state = model.exact_constant_precipitation(40.0, 3.0, 10.0).unwrap();
        assert_eq!(state.storage_mm, 50.0);
        assert_eq!(state.cumulative_runoff_mm, 20.0);
        assert_eq!(state.cumulative_evapotranspiration_mm, 0.0);
        assert_eq!(state.budget_residual_mm(40.0), 0.0);
    }

    #[test]
    fn trajectory_includes_initial_and_preserves_fraction_bounds() {
        let samples = bucket().exact_trajectory(25.0, 8.0, 1.0, 20).unwrap();
        assert_eq!(samples.len(), 21);
        assert_eq!(samples[0].time_days, 0.0);
        assert_eq!(samples[0].storage_mm, 25.0);
        assert!(samples.iter().all(|sample| {
            (0.0..=1.0).contains(&sample.soil_moisture_fraction)
                && sample.budget_residual_mm.abs() < 1.0e-10
        }));
    }
}
