// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact two-pool soil-carbon turnover baseline.
//!
//! A fast pool receives litter input, a fixed fraction of fast-pool decay is
//! transferred to a slow pool, and all remaining decay is respired. Constant
//! temperature and litter input admit a closed-form solution with an exact
//! carbon budget. Parameters are illustrative rate constants unless callers
//! provide an external calibration.

use crate::error::{
    ModelError, require_finite, require_fraction, require_non_negative, require_positive,
};
use crate::transient::MAX_TRAJECTORY_STEPS;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoilCarbonState {
    pub fast_carbon: f64,
    pub slow_carbon: f64,
    pub cumulative_litter_input: f64,
    pub cumulative_respiration: f64,
}

impl SoilCarbonState {
    pub fn total_carbon(&self) -> f64 {
        self.fast_carbon + self.slow_carbon
    }

    pub fn budget_residual(&self, initial_total_carbon: f64) -> f64 {
        initial_total_carbon + self.cumulative_litter_input
            - self.cumulative_respiration
            - self.total_carbon()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoilCarbonSample {
    pub time_years: f64,
    pub temperature_k: f64,
    pub fast_carbon: f64,
    pub slow_carbon: f64,
    pub total_carbon: f64,
    pub respiration_rate_per_year: f64,
    pub cumulative_litter_input: f64,
    pub cumulative_respiration: f64,
    pub budget_residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPoolSoilCarbon {
    /// Fast-pool decomposition rate at the reference temperature, year⁻¹.
    pub fast_decay_rate: f64,
    /// Slow-pool decomposition rate at the reference temperature, year⁻¹.
    pub slow_decay_rate: f64,
    /// Fraction of fast-pool decay transferred into the slow pool.
    pub fast_to_slow_fraction: f64,
    /// Reference temperature for the Q10 multiplier, K.
    pub reference_temperature_k: f64,
    /// Multiplicative rate increase per 10 K warming.
    pub q10: f64,
}

impl TwoPoolSoilCarbon {
    pub fn try_new(
        fast_decay_rate: f64,
        slow_decay_rate: f64,
        fast_to_slow_fraction: f64,
        reference_temperature_k: f64,
        q10: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            fast_decay_rate,
            slow_decay_rate,
            fast_to_slow_fraction,
            reference_temperature_k,
            q10,
        };
        model.validate()?;
        Ok(model)
    }

    /// Illustrative turnover parameters for numerical experiments.
    pub fn illustrative() -> Self {
        Self {
            fast_decay_rate: 0.7,
            slow_decay_rate: 0.03,
            fast_to_slow_fraction: 0.25,
            reference_temperature_k: 283.15,
            q10: 2.0,
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("fast_decay_rate", self.fast_decay_rate)?;
        require_positive("slow_decay_rate", self.slow_decay_rate)?;
        require_fraction("fast_to_slow_fraction", self.fast_to_slow_fraction)?;
        require_positive("reference_temperature_k", self.reference_temperature_k)?;
        require_positive("q10", self.q10)
    }

    pub fn validate_state(&self, state: SoilCarbonState) -> Result<(), ModelError> {
        require_non_negative("fast_carbon", state.fast_carbon)?;
        require_non_negative("slow_carbon", state.slow_carbon)?;
        require_non_negative("cumulative_litter_input", state.cumulative_litter_input)?;
        require_non_negative("cumulative_respiration", state.cumulative_respiration)
    }

    pub fn temperature_multiplier(&self, temperature_k: f64) -> Result<f64, ModelError> {
        self.validate()?;
        require_positive("temperature_k", temperature_k)?;
        let multiplier = self
            .q10
            .powf((temperature_k - self.reference_temperature_k) / 10.0);
        require_positive("temperature_multiplier", multiplier)?;
        Ok(multiplier)
    }

    pub fn effective_decay_rates(&self, temperature_k: f64) -> Result<(f64, f64), ModelError> {
        let multiplier = self.temperature_multiplier(temperature_k)?;
        Ok((
            self.fast_decay_rate * multiplier,
            self.slow_decay_rate * multiplier,
        ))
    }

    pub fn respiration_rate(
        &self,
        fast_carbon: f64,
        slow_carbon: f64,
        temperature_k: f64,
    ) -> Result<f64, ModelError> {
        require_non_negative("fast_carbon", fast_carbon)?;
        require_non_negative("slow_carbon", slow_carbon)?;
        let (fast_rate, slow_rate) = self.effective_decay_rates(temperature_k)?;
        Ok((1.0 - self.fast_to_slow_fraction) * fast_rate * fast_carbon + slow_rate * slow_carbon)
    }

    pub fn equilibrium(
        &self,
        litter_input_per_year: f64,
        temperature_k: f64,
    ) -> Result<(f64, f64), ModelError> {
        require_non_negative("litter_input_per_year", litter_input_per_year)?;
        let (fast_rate, slow_rate) = self.effective_decay_rates(temperature_k)?;
        let fast = litter_input_per_year / fast_rate;
        let slow = self.fast_to_slow_fraction * litter_input_per_year / slow_rate;
        Ok((fast, slow))
    }

    /// Exact state under constant temperature and litter input.
    pub fn exact_constant_environment(
        &self,
        initial_fast_carbon: f64,
        initial_slow_carbon: f64,
        litter_input_per_year: f64,
        temperature_k: f64,
        elapsed_years: f64,
    ) -> Result<SoilCarbonState, ModelError> {
        self.validate()?;
        require_non_negative("initial_fast_carbon", initial_fast_carbon)?;
        require_non_negative("initial_slow_carbon", initial_slow_carbon)?;
        require_non_negative("litter_input_per_year", litter_input_per_year)?;
        require_positive("temperature_k", temperature_k)?;
        require_non_negative("elapsed_years", elapsed_years)?;

        let (a, c) = self.effective_decay_rates(temperature_k)?;
        let b = self.fast_to_slow_fraction * a;
        let fast_equilibrium = litter_input_per_year / a;
        let slow_equilibrium = b * fast_equilibrium / c;
        let fast_offset = initial_fast_carbon - fast_equilibrium;
        let exp_a = (-a * elapsed_years).exp();
        let exp_c = (-c * elapsed_years).exp();
        let fast_carbon = fast_equilibrium + fast_offset * exp_a;
        let slow_carbon = if (a - c).abs() <= 1.0e-12 * a.max(c) {
            slow_equilibrium
                + (initial_slow_carbon - slow_equilibrium) * exp_a
                + b * fast_offset * elapsed_years * exp_a
        } else {
            let transferred_offset = b * fast_offset / (c - a);
            slow_equilibrium
                + transferred_offset * exp_a
                + (initial_slow_carbon - slow_equilibrium - transferred_offset) * exp_c
        };

        let cumulative_litter_input = litter_input_per_year * elapsed_years;
        require_finite("cumulative_litter_input", cumulative_litter_input)?;
        let initial_total = initial_fast_carbon + initial_slow_carbon;
        let cumulative_respiration =
            initial_total + cumulative_litter_input - fast_carbon - slow_carbon;
        let state = SoilCarbonState {
            fast_carbon: fast_carbon.max(0.0),
            slow_carbon: slow_carbon.max(0.0),
            cumulative_litter_input,
            cumulative_respiration: cumulative_respiration.max(0.0),
        };
        self.validate_state(state)?;
        Ok(state)
    }

    pub fn exact_trajectory(
        &self,
        initial_fast_carbon: f64,
        initial_slow_carbon: f64,
        litter_input_per_year: f64,
        temperature_k: f64,
        dt_years: f64,
        steps: usize,
    ) -> Result<Vec<SoilCarbonSample>, ModelError> {
        self.validate()?;
        require_positive("dt_years", dt_years)?;
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
        require_finite("duration_years", dt_years * steps as f64)?;
        let initial_total = initial_fast_carbon + initial_slow_carbon;
        let mut samples = Vec::with_capacity(capacity);
        for step in 0..=steps {
            let time_years = step as f64 * dt_years;
            let state = self.exact_constant_environment(
                initial_fast_carbon,
                initial_slow_carbon,
                litter_input_per_year,
                temperature_k,
                time_years,
            )?;
            samples.push(SoilCarbonSample {
                time_years,
                temperature_k,
                fast_carbon: state.fast_carbon,
                slow_carbon: state.slow_carbon,
                total_carbon: state.total_carbon(),
                respiration_rate_per_year: self.respiration_rate(
                    state.fast_carbon,
                    state.slow_carbon,
                    temperature_k,
                )?,
                cumulative_litter_input: state.cumulative_litter_input,
                cumulative_respiration: state.cumulative_respiration,
                budget_residual: state.budget_residual(initial_total),
            });
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equilibrium_fluxes_close() {
        let model = TwoPoolSoilCarbon::illustrative();
        let input = 4.0;
        let (fast, slow) = model.equilibrium(input, 283.15).unwrap();
        let respiration = model.respiration_rate(fast, slow, 283.15).unwrap();
        assert!((respiration - input).abs() < 1.0e-12);
    }

    #[test]
    fn exact_solution_preserves_carbon_budget() {
        let model = TwoPoolSoilCarbon::illustrative();
        let state = model
            .exact_constant_environment(10.0, 90.0, 3.0, 288.15, 75.0)
            .unwrap();
        assert!(state.fast_carbon >= 0.0 && state.slow_carbon >= 0.0);
        assert!(state.budget_residual(100.0).abs() < 1.0e-10);
    }

    #[test]
    fn warming_accelerates_turnover_and_reduces_equilibrium_stock() {
        let model = TwoPoolSoilCarbon::illustrative();
        let cool = model.equilibrium(3.0, 283.15).unwrap();
        let warm = model.equilibrium(3.0, 293.15).unwrap();
        assert!((warm.0 - 0.5 * cool.0).abs() < 1.0e-12);
        assert!((warm.1 - 0.5 * cool.1).abs() < 1.0e-12);
    }

    #[test]
    fn equal_rate_limit_is_finite_and_exact_at_initial_time() {
        let model = TwoPoolSoilCarbon::try_new(0.1, 0.1, 0.4, 283.15, 2.0).unwrap();
        let state = model
            .exact_constant_environment(12.0, 30.0, 2.0, 283.15, 0.0)
            .unwrap();
        assert_eq!(state.fast_carbon, 12.0);
        assert_eq!(state.slow_carbon, 30.0);
        assert_eq!(state.cumulative_respiration, 0.0);
    }

    #[test]
    fn trajectory_includes_initial_and_budget_evidence() {
        let model = TwoPoolSoilCarbon::illustrative();
        let samples = model
            .exact_trajectory(10.0, 90.0, 3.0, 283.15, 1.0, 100)
            .unwrap();
        assert_eq!(samples.len(), 101);
        assert_eq!(samples[0].time_years, 0.0);
        assert!(
            samples
                .iter()
                .all(|sample| sample.budget_residual.abs() < 1.0e-9)
        );
    }
}
