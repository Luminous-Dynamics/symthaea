// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Discrete-generation Ricker density dependence.
//!
//! The map provides a compact baseline for ecological dynamics that cannot be
//! reduced to a continuous-time ODE. It exposes the positive fixed point and
//! its first period-doubling threshold without claiming that all parameter
//! values beyond the threshold are chaotic.

use crate::error::{ModelError, require_non_negative, require_positive};
use crate::integration::MAX_TRAJECTORY_STEPS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RickerFixedPointStability {
    Stable,
    PeriodDoublingThreshold,
    Unstable,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RickerSample {
    pub generation: usize,
    pub population: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RickerModel {
    /// Dimensionless per-generation density-independent growth parameter.
    pub growth_parameter: f64,
    pub carrying_capacity: f64,
}

impl RickerModel {
    pub fn try_new(growth_parameter: f64, carrying_capacity: f64) -> Result<Self, ModelError> {
        require_positive("growth_parameter", growth_parameter)?;
        require_positive("carrying_capacity", carrying_capacity)?;
        Ok(Self {
            growth_parameter,
            carrying_capacity,
        })
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("growth_parameter", self.growth_parameter)?;
        require_positive("carrying_capacity", self.carrying_capacity)
    }

    pub fn next_population(&self, population: f64) -> Result<f64, ModelError> {
        self.validate()?;
        require_non_negative("population", population)?;
        let next = population
            * (self.growth_parameter * (1.0 - population / self.carrying_capacity)).exp();
        if next.is_finite() && next >= 0.0 {
            Ok(next)
        } else {
            Err(ModelError::IntegrationDomainViolation {
                step: 1,
                component: "ricker_population",
                value: next,
            })
        }
    }

    pub fn positive_fixed_point(&self) -> f64 {
        self.carrying_capacity
    }

    /// Local multiplier at the positive fixed point: `1 - r`.
    pub fn positive_fixed_point_multiplier(&self) -> f64 {
        1.0 - self.growth_parameter
    }

    pub fn positive_fixed_point_stability(&self) -> RickerFixedPointStability {
        let multiplier = self.positive_fixed_point_multiplier().abs();
        let tolerance = 16.0 * f64::EPSILON * self.growth_parameter.max(1.0);
        if (multiplier - 1.0).abs() <= tolerance {
            RickerFixedPointStability::PeriodDoublingThreshold
        } else if multiplier < 1.0 {
            RickerFixedPointStability::Stable
        } else {
            RickerFixedPointStability::Unstable
        }
    }

    pub fn trajectory(
        &self,
        initial_population: f64,
        generations: usize,
    ) -> Result<Vec<RickerSample>, ModelError> {
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
        samples.push(RickerSample {
            generation: 0,
            population,
        });
        for generation in 1..=generations {
            population = self
                .next_population(population)
                .map_err(|error| match error {
                    ModelError::IntegrationDomainViolation {
                        component, value, ..
                    } => ModelError::IntegrationDomainViolation {
                        step: generation,
                        component,
                        value,
                    },
                    other => other,
                })?;
            samples.push(RickerSample {
                generation,
                population,
            });
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carrying_capacity_is_an_exact_fixed_point() {
        let model = RickerModel::try_new(1.5, 100.0).unwrap();
        assert!((model.next_population(100.0).unwrap() - 100.0).abs() < 1e-12);
        assert_eq!(model.next_population(0.0).unwrap(), 0.0);
    }

    #[test]
    fn local_stability_changes_at_r_equals_two() {
        assert_eq!(
            RickerModel::try_new(1.0, 100.0)
                .unwrap()
                .positive_fixed_point_stability(),
            RickerFixedPointStability::Stable
        );
        assert_eq!(
            RickerModel::try_new(2.0, 100.0)
                .unwrap()
                .positive_fixed_point_stability(),
            RickerFixedPointStability::PeriodDoublingThreshold
        );
        assert_eq!(
            RickerModel::try_new(2.5, 100.0)
                .unwrap()
                .positive_fixed_point_stability(),
            RickerFixedPointStability::Unstable
        );
    }

    #[test]
    fn stable_map_converges_to_positive_fixed_point() {
        let model = RickerModel::try_new(1.0, 100.0).unwrap();
        let samples = model.trajectory(10.0, 50).unwrap();
        assert_eq!(samples.len(), 51);
        assert_eq!(samples[0].generation, 0);
        assert!((samples.last().unwrap().population - 100.0).abs() < 1e-10);
    }

    #[test]
    fn unstable_fixed_point_produces_persistent_departures_without_chaos_claim() {
        let model = RickerModel::try_new(2.2, 100.0).unwrap();
        let samples = model.trajectory(99.0, 100).unwrap();
        let tail = &samples[80..];
        let maximum_departure = tail
            .iter()
            .map(|sample| (sample.population - 100.0).abs())
            .fold(0.0_f64, f64::max);
        assert!(maximum_departure > 1.0);
    }

    #[test]
    fn trajectory_allocation_is_bounded() {
        let model = RickerModel::try_new(1.0, 100.0).unwrap();
        assert!(matches!(
            model.trajectory(10.0, MAX_TRAJECTORY_STEPS + 1),
            Err(ModelError::TrajectoryTooLarge { .. })
        ));
    }
}
