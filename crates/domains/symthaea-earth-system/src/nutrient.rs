// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conserved organic-mineral nutrient-cycle baseline.
//!
//! Organic nutrient is mineralized into an available pool. The mineral pool
//! receives atmospheric or external deposition and is depleted by plant uptake
//! and leaching. Under constant inputs the triangular linear system has an exact
//! solution and exact cumulative loss integrals. The model is intentionally
//! generic: callers decide whether the nutrient represents nitrogen, phosphorus,
//! or another conserved limiting element and must supply compatible units.

use crate::error::{ModelError, require_finite, require_non_negative, require_positive};
use crate::transient::MAX_TRAJECTORY_STEPS;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NutrientState {
    pub organic_pool: f64,
    pub mineral_pool: f64,
}

impl NutrientState {
    pub fn total(&self) -> f64 {
        self.organic_pool + self.mineral_pool
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NutrientSample {
    pub time: f64,
    pub state: NutrientState,
    pub mineralization_flux: f64,
    pub uptake_flux: f64,
    pub leaching_flux: f64,
    pub cumulative_organic_input: f64,
    pub cumulative_deposition: f64,
    pub cumulative_uptake: f64,
    pub cumulative_leaching: f64,
    pub budget_residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPoolNutrientCycle {
    /// Organic-pool mineralization rate per model-time unit.
    pub mineralization_rate: f64,
    /// Mineral-pool biological uptake rate per model-time unit.
    pub uptake_rate: f64,
    /// Mineral-pool leaching loss rate per model-time unit.
    pub leaching_rate: f64,
}

impl TwoPoolNutrientCycle {
    pub fn try_new(
        mineralization_rate: f64,
        uptake_rate: f64,
        leaching_rate: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            mineralization_rate,
            uptake_rate,
            leaching_rate,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("mineralization_rate", self.mineralization_rate)?;
        require_non_negative("uptake_rate", self.uptake_rate)?;
        require_non_negative("leaching_rate", self.leaching_rate)?;
        require_positive("total_mineral_loss_rate", self.total_mineral_loss_rate())
    }

    pub fn validate_state(&self, state: NutrientState) -> Result<(), ModelError> {
        require_non_negative("organic_pool", state.organic_pool)?;
        require_non_negative("mineral_pool", state.mineral_pool)
    }

    pub fn total_mineral_loss_rate(&self) -> f64 {
        self.uptake_rate + self.leaching_rate
    }

    pub fn equilibrium(
        &self,
        organic_input: f64,
        deposition: f64,
    ) -> Result<NutrientState, ModelError> {
        self.validate()?;
        require_non_negative("organic_input", organic_input)?;
        require_non_negative("deposition", deposition)?;
        Ok(NutrientState {
            organic_pool: organic_input / self.mineralization_rate,
            mineral_pool: (organic_input + deposition) / self.total_mineral_loss_rate(),
        })
    }

    pub fn derivative(
        &self,
        state: NutrientState,
        organic_input: f64,
        deposition: f64,
    ) -> Result<NutrientState, ModelError> {
        self.validate_state(state)?;
        require_non_negative("organic_input", organic_input)?;
        require_non_negative("deposition", deposition)?;
        let mineralization = self.mineralization_rate * state.organic_pool;
        Ok(NutrientState {
            organic_pool: organic_input - mineralization,
            mineral_pool: deposition + mineralization
                - self.total_mineral_loss_rate() * state.mineral_pool,
        })
    }

    pub fn exact_state(
        &self,
        initial: NutrientState,
        organic_input: f64,
        deposition: f64,
        elapsed_time: f64,
    ) -> Result<NutrientState, ModelError> {
        self.validate_state(initial)?;
        require_non_negative("organic_input", organic_input)?;
        require_non_negative("deposition", deposition)?;
        require_non_negative("elapsed_time", elapsed_time)?;
        if elapsed_time == 0.0 {
            return Ok(initial);
        }

        let km = self.mineralization_rate;
        let q = self.total_mineral_loss_rate();
        let organic_equilibrium = organic_input / km;
        let organic_offset = initial.organic_pool - organic_equilibrium;
        let exp_m = (-km * elapsed_time).exp();
        let exp_q = (-q * elapsed_time).exp();
        let organic_pool = organic_equilibrium + organic_offset * exp_m;

        let mineral_equilibrium = (organic_input + deposition) / q;
        let forcing_amplitude = km * organic_offset;
        let separation = (q - km) * elapsed_time;
        let mineral_pool = if separation.abs() <= 1.0e-4 {
            // Stable convolution limit as the two rates approach one another.
            let exprel = if separation.abs() <= 1.0e-8 {
                1.0 + separation / 2.0 + separation.powi(2) / 6.0 + separation.powi(3) / 24.0
            } else {
                separation.exp_m1() / separation
            };
            initial.mineral_pool * exp_q
                + (organic_input + deposition) * (-q * elapsed_time).exp_m1() / -q
                + forcing_amplitude * exp_q * elapsed_time * exprel
        } else {
            let mineralization_mode = forcing_amplitude / (q - km);
            let loss_mode = initial.mineral_pool - mineral_equilibrium - mineralization_mode;
            mineral_equilibrium + mineralization_mode * exp_m + loss_mode * exp_q
        };

        let state = NutrientState {
            organic_pool,
            mineral_pool,
        };
        require_finite("organic_pool", state.organic_pool)?;
        require_finite("mineral_pool", state.mineral_pool)?;
        self.validate_state(state)?;
        Ok(state)
    }

    fn integrated_mineral_pool(
        &self,
        initial: NutrientState,
        organic_input: f64,
        deposition: f64,
        elapsed_time: f64,
    ) -> Result<f64, ModelError> {
        let state = self.exact_state(initial, organic_input, deposition, elapsed_time)?;
        let total_input = (organic_input + deposition) * elapsed_time;
        let total_loss = total_input - (state.total() - initial.total());
        let scale = total_input
            .abs()
            .max(initial.total())
            .max(state.total())
            .max(1.0);
        let tolerance = 1.0e-12 * scale;
        let adjusted_loss = if total_loss < 0.0 && total_loss >= -tolerance {
            0.0
        } else {
            total_loss
        };
        let integral = adjusted_loss / self.total_mineral_loss_rate();
        require_non_negative("integrated_mineral_pool", integral)?;
        Ok(integral)
    }

    pub fn exact_sample(
        &self,
        initial: NutrientState,
        organic_input: f64,
        deposition: f64,
        elapsed_time: f64,
    ) -> Result<NutrientSample, ModelError> {
        let state = self.exact_state(initial, organic_input, deposition, elapsed_time)?;
        let integrated_mineral =
            self.integrated_mineral_pool(initial, organic_input, deposition, elapsed_time)?;
        let cumulative_organic_input = organic_input * elapsed_time;
        let cumulative_deposition = deposition * elapsed_time;
        let cumulative_uptake = self.uptake_rate * integrated_mineral;
        let cumulative_leaching = self.leaching_rate * integrated_mineral;
        let budget_residual =
            state.total() - initial.total() - cumulative_organic_input - cumulative_deposition
                + cumulative_uptake
                + cumulative_leaching;
        require_finite("nutrient_budget_residual", budget_residual)?;
        Ok(NutrientSample {
            time: elapsed_time,
            state,
            mineralization_flux: self.mineralization_rate * state.organic_pool,
            uptake_flux: self.uptake_rate * state.mineral_pool,
            leaching_flux: self.leaching_rate * state.mineral_pool,
            cumulative_organic_input,
            cumulative_deposition,
            cumulative_uptake,
            cumulative_leaching,
            budget_residual,
        })
    }

    pub fn exact_trajectory(
        &self,
        initial: NutrientState,
        organic_input: f64,
        deposition: f64,
        dt: f64,
        steps: usize,
    ) -> Result<Vec<NutrientSample>, ModelError> {
        require_positive("dt", dt)?;
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
        require_finite("duration", dt * steps as f64)?;
        let mut samples = Vec::with_capacity(capacity);
        for step in 0..=steps {
            samples.push(self.exact_sample(
                initial,
                organic_input,
                deposition,
                step as f64 * dt,
            )?);
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equilibrium_balances_both_pools() {
        let model = TwoPoolNutrientCycle::try_new(0.2, 0.08, 0.02).unwrap();
        let equilibrium = model.equilibrium(4.0, 1.0).unwrap();
        let tendency = model.derivative(equilibrium, 4.0, 1.0).unwrap();
        assert!(tendency.organic_pool.abs() < 1.0e-12);
        assert!(tendency.mineral_pool.abs() < 1.0e-12);
    }

    #[test]
    fn exact_solution_closes_the_nutrient_budget() {
        let model = TwoPoolNutrientCycle::try_new(0.2, 0.08, 0.02).unwrap();
        let sample = model
            .exact_sample(
                NutrientState {
                    organic_pool: 5.0,
                    mineral_pool: 2.0,
                },
                4.0,
                1.0,
                25.0,
            )
            .unwrap();
        assert!(sample.budget_residual.abs() < 1.0e-11);
        assert!(sample.state.organic_pool > 0.0);
        assert!(sample.state.mineral_pool > 0.0);
    }

    #[test]
    fn equal_rate_limit_is_finite_and_continuous() {
        let initial = NutrientState {
            organic_pool: 7.0,
            mineral_pool: 3.0,
        };
        let exact = TwoPoolNutrientCycle::try_new(0.2, 0.15, 0.05)
            .unwrap()
            .exact_state(initial, 1.0, 0.5, 4.0)
            .unwrap();
        let nearby = TwoPoolNutrientCycle::try_new(0.2, 0.15000001, 0.05)
            .unwrap()
            .exact_state(initial, 1.0, 0.5, 4.0)
            .unwrap();
        assert!(exact.organic_pool.is_finite());
        assert!(exact.mineral_pool.is_finite());
        assert!((exact.mineral_pool - nearby.mineral_pool).abs() < 1.0e-6);
    }

    #[test]
    fn nearly_equal_rates_avoid_mode_cancellation() {
        let initial = NutrientState {
            organic_pool: 7.0,
            mineral_pool: 3.0,
        };
        let equal = TwoPoolNutrientCycle::try_new(0.2, 0.15, 0.05)
            .unwrap()
            .exact_sample(initial, 1.0, 0.5, 100.0)
            .unwrap();
        let nearby = TwoPoolNutrientCycle::try_new(0.2, 0.1500000001, 0.05)
            .unwrap()
            .exact_sample(initial, 1.0, 0.5, 100.0)
            .unwrap();
        assert!((equal.state.mineral_pool - nearby.state.mineral_pool).abs() < 1.0e-8);
        assert!(equal.budget_residual.abs() < 1.0e-11);
        assert!(nearby.budget_residual.abs() < 1.0e-11);
    }

    #[test]
    fn excessive_trajectory_requests_fail_closed() {
        let model = TwoPoolNutrientCycle::try_new(0.2, 0.08, 0.02).unwrap();
        let error = model
            .exact_trajectory(
                NutrientState {
                    organic_pool: 1.0,
                    mineral_pool: 1.0,
                },
                1.0,
                0.0,
                1.0,
                MAX_TRAJECTORY_STEPS + 1,
            )
            .unwrap_err();
        assert!(matches!(error, ModelError::TrajectoryTooLarge { .. }));
    }
}
