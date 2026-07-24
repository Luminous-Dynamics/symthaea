// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Discrete community-succession transition oracle.
//!
//! A row-stochastic transition matrix moves abundance or area among a bounded
//! set of community states. The model preserves total mass exactly up to
//! floating-point roundoff and can solve a stationary composition by power
//! iteration when the supplied chain converges. It does not infer transition
//! probabilities, represent within-state demography, or guarantee uniqueness
//! for reducible or periodic chains.

use crate::error::{ModelError, require_finite, require_non_negative};

pub const MAX_SUCCESSION_STATES: usize = 16;
pub const MAX_SUCCESSION_GENERATIONS: usize = 100_000;

#[derive(Debug, Clone, PartialEq)]
pub struct SuccessionSample {
    pub generation: usize,
    pub composition: Vec<f64>,
    pub total_residual: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CommunitySuccession {
    states: usize,
    transition: Vec<f64>,
}

impl CommunitySuccession {
    pub fn try_new(states: usize, transition: Vec<f64>) -> Result<Self, ModelError> {
        if states == 0 || states > MAX_SUCCESSION_STATES {
            return Err(ModelError::OutOfRange {
                parameter: "succession_states",
                value: states as f64,
                min: 1.0,
                max: MAX_SUCCESSION_STATES as f64,
            });
        }
        let expected = states
            .checked_mul(states)
            .ok_or(ModelError::DimensionMismatch {
                context: "succession transition matrix",
                expected: usize::MAX,
                found: transition.len(),
            })?;
        if transition.len() != expected {
            return Err(ModelError::DimensionMismatch {
                context: "succession transition matrix",
                expected,
                found: transition.len(),
            });
        }
        let model = Self { states, transition };
        model.validate()?;
        Ok(model)
    }

    pub fn states(&self) -> usize {
        self.states
    }

    pub fn transition_probability(&self, from: usize, to: usize) -> Option<f64> {
        if from >= self.states || to >= self.states {
            None
        } else {
            Some(self.transition[from * self.states + to])
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.states == 0 || self.states > MAX_SUCCESSION_STATES {
            return Err(ModelError::OutOfRange {
                parameter: "succession_states",
                value: self.states as f64,
                min: 1.0,
                max: MAX_SUCCESSION_STATES as f64,
            });
        }
        let expected = self.states * self.states;
        if self.transition.len() != expected {
            return Err(ModelError::DimensionMismatch {
                context: "succession transition matrix",
                expected,
                found: self.transition.len(),
            });
        }
        for row in 0..self.states {
            let mut sum = 0.0;
            for column in 0..self.states {
                let value = self.transition[row * self.states + column];
                require_non_negative("transition_probability", value)?;
                sum += value;
            }
            if (sum - 1.0).abs() > 1.0e-12 {
                return Err(ModelError::OutOfRange {
                    parameter: "transition_row_sum",
                    value: sum,
                    min: 1.0 - 1.0e-12,
                    max: 1.0 + 1.0e-12,
                });
            }
        }
        Ok(())
    }

    pub fn validate_composition(&self, composition: &[f64]) -> Result<f64, ModelError> {
        if composition.len() != self.states {
            return Err(ModelError::DimensionMismatch {
                context: "succession composition",
                expected: self.states,
                found: composition.len(),
            });
        }
        let mut total = 0.0;
        for value in composition {
            require_non_negative("community_abundance", *value)?;
            total += *value;
        }
        require_finite("community_total", total)?;
        if total <= 0.0 {
            return Err(ModelError::NonPositive {
                parameter: "community_total",
                value: total,
            });
        }
        Ok(total)
    }

    pub fn project(&self, composition: &[f64]) -> Result<Vec<f64>, ModelError> {
        self.validate()?;
        let total = self.validate_composition(composition)?;
        let mut next = vec![0.0; self.states];
        for from in 0..self.states {
            for to in 0..self.states {
                next[to] += composition[from] * self.transition[from * self.states + to];
            }
        }
        let projected_total: f64 = next.iter().sum();
        let residual = projected_total - total;
        if residual.abs() > 1.0e-10 * total.max(1.0) {
            return Err(ModelError::IntegrationDomainViolation {
                step: 0,
                component: "succession_total_residual",
                value: residual,
            });
        }
        Ok(next)
    }

    pub fn trajectory(
        &self,
        initial: &[f64],
        generations: usize,
    ) -> Result<Vec<SuccessionSample>, ModelError> {
        self.validate()?;
        let total = self.validate_composition(initial)?;
        if generations == 0 {
            return Err(ModelError::ZeroSteps);
        }
        if generations > MAX_SUCCESSION_GENERATIONS {
            return Err(ModelError::TrajectoryTooLarge {
                requested: generations,
                maximum: MAX_SUCCESSION_GENERATIONS,
            });
        }
        let capacity = generations
            .checked_add(1)
            .ok_or(ModelError::TrajectoryTooLarge {
                requested: usize::MAX,
                maximum: MAX_SUCCESSION_GENERATIONS,
            })?;
        let mut samples = Vec::with_capacity(capacity);
        let mut composition = initial.to_vec();
        for generation in 0..=generations {
            samples.push(SuccessionSample {
                generation,
                total_residual: composition.iter().sum::<f64>() - total,
                composition: composition.clone(),
            });
            if generation < generations {
                composition = self.project(&composition)?;
            }
        }
        Ok(samples)
    }

    /// Solve one normalized stationary composition by power iteration.
    pub fn stationary_distribution(
        &self,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Vec<f64>, ModelError> {
        self.validate()?;
        require_non_negative("stationary_tolerance", tolerance)?;
        if tolerance == 0.0 {
            return Err(ModelError::NonPositive {
                parameter: "stationary_tolerance",
                value: tolerance,
            });
        }
        if max_iterations == 0 {
            return Err(ModelError::ZeroSteps);
        }
        if max_iterations > MAX_SUCCESSION_GENERATIONS {
            return Err(ModelError::TrajectoryTooLarge {
                requested: max_iterations,
                maximum: MAX_SUCCESSION_GENERATIONS,
            });
        }
        let mut composition = vec![1.0 / self.states as f64; self.states];
        for _ in 0..max_iterations {
            let next = self.project(&composition)?;
            let difference = next
                .iter()
                .zip(&composition)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            composition = next;
            if difference <= tolerance {
                let sum: f64 = composition.iter().sum();
                for value in &mut composition {
                    *value /= sum;
                }
                return Ok(composition);
            }
        }
        Err(ModelError::NoConvergence {
            context: "community succession stationary distribution",
            iterations: max_iterations,
        })
    }

    pub fn stationary_residual(&self, composition: &[f64]) -> Result<f64, ModelError> {
        let projected = self.project(composition)?;
        Ok(projected
            .iter()
            .zip(composition)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_state() -> CommunitySuccession {
        CommunitySuccession::try_new(2, vec![0.8, 0.2, 0.1, 0.9]).unwrap()
    }

    #[test]
    fn projection_conserves_total_abundance() {
        let next = two_state().project(&[30.0, 70.0]).unwrap();
        assert!((next.iter().sum::<f64>() - 100.0).abs() < 1.0e-12);
    }

    #[test]
    fn known_stationary_distribution_is_recovered() {
        let model = two_state();
        let stationary = model.stationary_distribution(1.0e-13, 10_000).unwrap();
        assert!((stationary[0] - 1.0 / 3.0).abs() < 1.0e-11);
        assert!((stationary[1] - 2.0 / 3.0).abs() < 1.0e-11);
        assert!(model.stationary_residual(&stationary).unwrap() < 1.0e-13);
    }

    #[test]
    fn trajectory_converges_without_losing_area() {
        let samples = two_state().trajectory(&[1.0, 0.0], 100).unwrap();
        let final_composition = &samples.last().unwrap().composition;
        assert!((final_composition[0] - 1.0 / 3.0).abs() < 1.0e-10);
        assert!(
            samples
                .iter()
                .all(|sample| sample.total_residual.abs() < 1.0e-12)
        );
    }

    #[test]
    fn malformed_transition_rows_fail_closed() {
        assert!(CommunitySuccession::try_new(2, vec![0.8, 0.3, 0.1, 0.9]).is_err());
        assert!(CommunitySuccession::try_new(2, vec![0.8, -0.2, 0.1, 0.9]).is_err());
    }

    #[test]
    fn stationary_iteration_budget_is_bounded() {
        let error = two_state()
            .stationary_distribution(1.0e-12, MAX_SUCCESSION_GENERATIONS + 1)
            .unwrap_err();
        assert!(matches!(error, ModelError::TrajectoryTooLarge { .. }));
    }

    #[test]
    fn periodic_chain_reports_nonconvergence() {
        let model = CommunitySuccession::try_new(2, vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        // Uniform initialization is stationary, so use the trajectory to show
        // periodic behavior and a one-step non-stationary composition residual.
        let samples = model.trajectory(&[1.0, 0.0], 3).unwrap();
        assert_eq!(samples[1].composition, vec![0.0, 1.0]);
        assert_eq!(samples[2].composition, vec![1.0, 0.0]);
        assert!(model.stationary_residual(&[1.0, 0.0]).unwrap() > 0.9);
    }
}
