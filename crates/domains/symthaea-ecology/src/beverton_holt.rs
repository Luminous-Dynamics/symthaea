// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Beverton-Holt discrete-generation density dependence.
//!
//! This map complements the Ricker model with a monotone, non-oscillatory
//! approach to carrying capacity. It has an exact finite-generation solution,
//! making it a useful oracle for discrete life-cycle simulations.

use crate::error::{ModelError, require_non_negative, require_positive};
use crate::integration::MAX_TRAJECTORY_STEPS;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BevertonHoltModel {
    /// Low-density per-generation multiplication factor. Must exceed one.
    pub reproduction_factor: f64,
    pub carrying_capacity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BevertonHoltSample {
    pub generation: usize,
    pub population: f64,
}

impl BevertonHoltModel {
    pub fn try_new(reproduction_factor: f64, carrying_capacity: f64) -> Result<Self, ModelError> {
        require_positive("reproduction_factor", reproduction_factor)?;
        if reproduction_factor <= 1.0 {
            return Err(ModelError::NonPositive {
                parameter: "reproduction_factor_minus_one",
                value: reproduction_factor - 1.0,
            });
        }
        require_positive("carrying_capacity", carrying_capacity)?;
        Ok(Self {
            reproduction_factor,
            carrying_capacity,
        })
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        Self::try_new(self.reproduction_factor, self.carrying_capacity).map(|_| ())
    }

    pub fn step(&self, population: f64) -> Result<f64, ModelError> {
        self.validate()?;
        require_non_negative("population", population)?;
        if population == 0.0 {
            return Ok(0.0);
        }
        let denominator =
            1.0 + (self.reproduction_factor - 1.0) * population / self.carrying_capacity;
        let next = self.reproduction_factor * population / denominator;
        require_non_negative("next_population", next)?;
        Ok(next)
    }

    /// Exact population after `generations` map applications.
    pub fn exact_population(
        &self,
        initial_population: f64,
        generations: usize,
    ) -> Result<f64, ModelError> {
        self.validate()?;
        require_non_negative("initial_population", initial_population)?;
        if initial_population == 0.0 {
            return Ok(0.0);
        }
        let decay = self.reproduction_factor.powf(-(generations as f64));
        let denominator = 1.0 + (self.carrying_capacity / initial_population - 1.0) * decay;
        let population = self.carrying_capacity / denominator;
        require_non_negative("population", population)?;
        Ok(population)
    }

    pub fn simulate(
        &self,
        initial_population: f64,
        generations: usize,
    ) -> Result<Vec<BevertonHoltSample>, ModelError> {
        self.validate()?;
        require_non_negative("initial_population", initial_population)?;
        if generations == 0 {
            return Err(ModelError::ZeroSteps);
        }
        if generations > MAX_TRAJECTORY_STEPS {
            return Err(ModelError::TrajectoryTooLarge {
                requested: generations,
                maximum: MAX_TRAJECTORY_STEPS,
            });
        }
        let capacity = generations
            .checked_add(1)
            .ok_or(ModelError::TrajectoryTooLarge {
                requested: usize::MAX,
                maximum: MAX_TRAJECTORY_STEPS,
            })?;
        let mut samples = Vec::with_capacity(capacity);
        let mut population = initial_population;
        samples.push(BevertonHoltSample {
            generation: 0,
            population,
        });
        for generation in 1..=generations {
            population = self.step(population)?;
            samples.push(BevertonHoltSample {
                generation,
                population,
            });
        }
        Ok(samples)
    }

    pub fn extinction_multiplier(&self) -> Result<f64, ModelError> {
        self.validate()?;
        Ok(self.reproduction_factor)
    }

    pub fn carrying_capacity_multiplier(&self) -> Result<f64, ModelError> {
        self.validate()?;
        Ok(1.0 / self.reproduction_factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> BevertonHoltModel {
        BevertonHoltModel::try_new(2.0, 100.0).unwrap()
    }

    #[test]
    fn exact_solution_matches_iteration() {
        let samples = model().simulate(10.0, 25).unwrap();
        let exact = model().exact_population(10.0, 25).unwrap();
        assert!((samples.last().unwrap().population - exact).abs() < 1.0e-12);
    }

    #[test]
    fn trajectories_approach_capacity_monotonically_from_both_sides() {
        let below = model().simulate(10.0, 20).unwrap();
        assert!(
            below
                .windows(2)
                .all(|pair| pair[1].population > pair[0].population)
        );
        let above = model().simulate(250.0, 20).unwrap();
        assert!(
            above
                .windows(2)
                .all(|pair| pair[1].population < pair[0].population)
        );
        assert!((below.last().unwrap().population - 100.0).abs() < 1.0e-3);
        assert!((above.last().unwrap().population - 100.0).abs() < 1.0e-3);
    }

    #[test]
    fn fixed_point_multipliers_have_expected_stability() {
        assert!(model().extinction_multiplier().unwrap() > 1.0);
        assert!(model().carrying_capacity_multiplier().unwrap() < 1.0);
        assert_eq!(model().step(0.0).unwrap(), 0.0);
        assert!((model().step(100.0).unwrap() - 100.0).abs() < 1.0e-12);
    }

    #[test]
    fn invalid_low_density_factor_is_rejected() {
        assert!(BevertonHoltModel::try_new(1.0, 100.0).is_err());
        assert!(BevertonHoltModel::try_new(0.5, 100.0).is_err());
    }
}
