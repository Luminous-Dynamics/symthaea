// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Poisson Galton-Watson branching-process extinction oracle.
//!
//! This is the smallest finite-population stochastic baseline in the crate. It
//! reports extinction probabilities implied by a declared mean number of
//! offspring per individual; it does not generate random trajectories or model
//! density dependence, age structure, environmental variation, or genetics.

use crate::error::{ModelError, require_finite, require_non_negative};

const EXTINCTION_TOLERANCE: f64 = 1.0e-14;
const MAX_EXTINCTION_ITERATIONS: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchingRegime {
    Subcritical,
    Critical,
    Supercritical,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonBranchingProcess {
    /// Mean number of offspring per individual.
    pub mean_offspring: f64,
}

impl PoissonBranchingProcess {
    pub fn try_new(mean_offspring: f64) -> Result<Self, ModelError> {
        require_non_negative("mean_offspring", mean_offspring)?;
        Ok(Self { mean_offspring })
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_non_negative("mean_offspring", self.mean_offspring)
    }

    pub fn regime(&self) -> BranchingRegime {
        if self.mean_offspring < 1.0 {
            BranchingRegime::Subcritical
        } else if self.mean_offspring == 1.0 {
            BranchingRegime::Critical
        } else {
            BranchingRegime::Supercritical
        }
    }

    /// Probability that a lineage founded by one individual ultimately goes
    /// extinct. For the Poisson offspring law this is the smallest solution of
    /// `q = exp(mean * (q - 1))` in `[0, 1]`.
    pub fn ultimate_extinction_probability(&self) -> Result<f64, ModelError> {
        self.validate()?;
        if self.mean_offspring <= 1.0 {
            return Ok(1.0);
        }
        let mut probability = 0.0;
        for _ in 0..MAX_EXTINCTION_ITERATIONS {
            let next = (self.mean_offspring * (probability - 1.0)).exp();
            if (next - probability).abs() <= EXTINCTION_TOLERANCE {
                return Ok(next.clamp(0.0, 1.0));
            }
            probability = next;
        }
        Err(ModelError::NoConvergence {
            context: "poisson_branching_extinction_probability",
            iterations: MAX_EXTINCTION_ITERATIONS,
        })
    }

    /// Extinction probability by the specified generation for one founder.
    /// Generation zero means the founder is present, so the probability is 0.
    pub fn extinction_probability_by_generation(
        &self,
        generations: usize,
    ) -> Result<f64, ModelError> {
        self.validate()?;
        let mut probability = 0.0;
        for _ in 0..generations {
            probability = (self.mean_offspring * (probability - 1.0)).exp();
        }
        Ok(probability.clamp(0.0, 1.0))
    }

    /// Ultimate extinction probability for independent founder lineages.
    pub fn ultimate_extinction_probability_for_founders(
        &self,
        founders: usize,
    ) -> Result<f64, ModelError> {
        let one = self.ultimate_extinction_probability()?;
        if founders == 0 {
            return Ok(1.0);
        }
        Ok(one.powf(founders as f64))
    }

    /// Expected population after a fixed number of generations.
    pub fn expected_population(
        &self,
        initial_population: f64,
        generations: usize,
    ) -> Result<f64, ModelError> {
        self.validate()?;
        require_non_negative("initial_population", initial_population)?;
        let expected = initial_population * self.mean_offspring.powf(generations as f64);
        require_finite("expected_population", expected)?;
        Ok(expected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subcritical_and_critical_lineages_go_extinct_almost_surely() {
        for mean in [0.0, 0.5, 1.0] {
            let process = PoissonBranchingProcess::try_new(mean).unwrap();
            assert_eq!(process.ultimate_extinction_probability().unwrap(), 1.0);
        }
    }

    #[test]
    fn supercritical_poisson_extinction_root_matches_reference_value() {
        let process = PoissonBranchingProcess::try_new(2.0).unwrap();
        let probability = process.ultimate_extinction_probability().unwrap();
        assert!((probability - 0.203_187_869_979_98).abs() < 1e-12);
        let residual = probability - (2.0 * (probability - 1.0)).exp();
        assert!(residual.abs() < 1e-13);
    }

    #[test]
    fn finite_generation_extinction_converges_monotonically() {
        let process = PoissonBranchingProcess::try_new(1.5).unwrap();
        let q5 = process.extinction_probability_by_generation(5).unwrap();
        let q20 = process.extinction_probability_by_generation(20).unwrap();
        let ultimate = process.ultimate_extinction_probability().unwrap();
        assert!(q5 < q20);
        assert!(q20 < ultimate);
    }

    #[test]
    fn multiple_founders_reduce_supercritical_extinction_risk() {
        let process = PoissonBranchingProcess::try_new(2.0).unwrap();
        let one = process
            .ultimate_extinction_probability_for_founders(1)
            .unwrap();
        let ten = process
            .ultimate_extinction_probability_for_founders(10)
            .unwrap();
        assert!(ten < one);
        assert_eq!(
            process
                .ultimate_extinction_probability_for_founders(0)
                .unwrap(),
            1.0
        );
    }

    #[test]
    fn expectation_and_extinction_are_distinct_quantities() {
        let process = PoissonBranchingProcess::try_new(2.0).unwrap();
        assert_eq!(process.expected_population(3.0, 4).unwrap(), 48.0);
        assert!(process.ultimate_extinction_probability().unwrap() > 0.0);
    }
}
