// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Strong Allee-effect population dynamics.
//!
//! `dN/dt = r N (1 - N/K) (N/A - 1)` has two attracting equilibria,
//! extinction and carrying capacity, separated by an unstable threshold `A`.
//! It is a compact oracle for critical-population and restoration experiments,
//! not a universal model of low-density population processes.

use crate::error::{ModelError, require_non_negative, require_positive};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrongAlleeModel {
    pub growth_rate: f64,
    pub carrying_capacity: f64,
    pub threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlleeBasin {
    Extinction,
    Threshold,
    Persistence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlleeEquilibriumStability {
    Stable,
    Unstable,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlleeEquilibrium {
    pub population: f64,
    pub stability: AlleeEquilibriumStability,
}

impl StrongAlleeModel {
    pub fn try_new(
        growth_rate: f64,
        carrying_capacity: f64,
        threshold: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            growth_rate,
            carrying_capacity,
            threshold,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("growth_rate", self.growth_rate)?;
        require_positive("carrying_capacity", self.carrying_capacity)?;
        require_positive("threshold", self.threshold)?;
        if self.threshold >= self.carrying_capacity {
            return Err(ModelError::OutOfRange {
                parameter: "threshold",
                value: self.threshold,
                min: f64::MIN_POSITIVE,
                max: self.carrying_capacity,
            });
        }
        Ok(())
    }

    pub fn tendency(&self, population: f64) -> f64 {
        self.growth_rate
            * population
            * (1.0 - population / self.carrying_capacity)
            * (population / self.threshold - 1.0)
    }

    pub fn tendency_derivative(&self, population: f64) -> f64 {
        let capacity_factor = 1.0 - population / self.carrying_capacity;
        let threshold_factor = population / self.threshold - 1.0;
        self.growth_rate
            * (capacity_factor * threshold_factor
                - population * threshold_factor / self.carrying_capacity
                + population * capacity_factor / self.threshold)
    }

    pub fn recovery_diagnostic(
        &self,
        population: f64,
    ) -> Result<crate::recovery::RecoveryDiagnostic, ModelError> {
        self.validate()?;
        require_non_negative("population", population)?;
        crate::recovery::scalar_recovery_diagnostic(self.tendency_derivative(population))
    }

    pub fn try_tendency(&self, population: f64) -> Result<f64, ModelError> {
        self.validate()?;
        require_non_negative("population", population)?;
        Ok(self.tendency(population))
    }

    pub fn equilibria(&self) -> Result<[AlleeEquilibrium; 3], ModelError> {
        self.validate()?;
        Ok([
            AlleeEquilibrium {
                population: 0.0,
                stability: AlleeEquilibriumStability::Stable,
            },
            AlleeEquilibrium {
                population: self.threshold,
                stability: AlleeEquilibriumStability::Unstable,
            },
            AlleeEquilibrium {
                population: self.carrying_capacity,
                stability: AlleeEquilibriumStability::Stable,
            },
        ])
    }

    pub fn basin(&self, population: f64) -> Result<AlleeBasin, ModelError> {
        self.validate()?;
        require_non_negative("population", population)?;
        let scale = self.carrying_capacity.max(1.0);
        if (population - self.threshold).abs() <= 16.0 * f64::EPSILON * scale {
            Ok(AlleeBasin::Threshold)
        } else if population < self.threshold {
            Ok(AlleeBasin::Extinction)
        } else {
            Ok(AlleeBasin::Persistence)
        }
    }

    /// Guarded, timestamped trajectory including the initial population.
    pub fn try_simulate_timestamped(
        &self,
        initial_population: f64,
        dt: f64,
        steps: usize,
    ) -> Result<Vec<crate::integration::PopulationSample>, ModelError> {
        self.validate()?;
        require_non_negative("initial_population", initial_population)?;
        require_positive("dt", dt)?;
        if steps == 0 {
            return Err(ModelError::ZeroSteps);
        }
        crate::integration::simulate_non_negative_single(
            initial_population,
            dt,
            steps,
            |population| self.tendency(population),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> StrongAlleeModel {
        StrongAlleeModel::try_new(0.5, 100.0, 20.0).unwrap()
    }

    #[test]
    fn equilibria_and_stability_are_explicit() {
        let equilibria = model().equilibria().unwrap();
        assert_eq!(equilibria[0].population, 0.0);
        assert_eq!(equilibria[1].population, 20.0);
        assert_eq!(equilibria[2].population, 100.0);
        assert_eq!(equilibria[0].stability, AlleeEquilibriumStability::Stable);
        assert_eq!(equilibria[1].stability, AlleeEquilibriumStability::Unstable);
        assert_eq!(equilibria[2].stability, AlleeEquilibriumStability::Stable);
        assert!(
            equilibria
                .iter()
                .all(|equilibrium| model().tendency(equilibrium.population).abs() < 1e-12)
        );
    }

    #[test]
    fn threshold_separates_extinction_and_persistence_basins() {
        assert_eq!(model().basin(10.0).unwrap(), AlleeBasin::Extinction);
        assert_eq!(model().basin(20.0).unwrap(), AlleeBasin::Threshold);
        assert_eq!(model().basin(30.0).unwrap(), AlleeBasin::Persistence);
        assert!(model().tendency(10.0) < 0.0);
        assert!(model().tendency(30.0) > 0.0);
        assert!(model().tendency(120.0) < 0.0);
    }

    #[test]
    fn trajectories_move_toward_their_attractors() {
        let below = model().try_simulate_timestamped(10.0, 0.01, 1_000).unwrap();
        let above = model().try_simulate_timestamped(30.0, 0.01, 1_000).unwrap();
        assert!(below.last().unwrap().population < 10.0);
        assert!(above.last().unwrap().population > 30.0);
        assert_eq!(below.len(), 1_001);
        assert_eq!(below[0].time, 0.0);
    }

    #[test]
    fn extinction_equilibrium_is_a_valid_initial_state() {
        let trajectory = model().try_simulate_timestamped(0.0, 1.0, 10).unwrap();
        assert!(trajectory.iter().all(|sample| sample.population == 0.0));
    }

    #[test]
    fn invalid_thresholds_and_unresolved_steps_fail_closed() {
        assert!(StrongAlleeModel::try_new(0.5, 100.0, 100.0).is_err());
        assert!(matches!(
            model().try_simulate_timestamped(10.0, 100.0, 2),
            Err(ModelError::IntegrationDomainViolation { .. })
        ));
    }

    #[test]
    fn local_recovery_distinguishes_both_attractors_from_threshold() {
        let model = model();
        assert_eq!(
            model.recovery_diagnostic(0.0).unwrap().stability,
            crate::recovery::LinearStability::Stable
        );
        assert_eq!(
            model
                .recovery_diagnostic(model.threshold)
                .unwrap()
                .stability,
            crate::recovery::LinearStability::Unstable
        );
        assert_eq!(
            model
                .recovery_diagnostic(model.carrying_capacity)
                .unwrap()
                .stability,
            crate::recovery::LinearStability::Stable
        );
    }
}
