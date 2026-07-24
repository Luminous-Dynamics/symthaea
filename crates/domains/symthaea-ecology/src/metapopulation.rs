// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Levins patch-occupancy metapopulation baseline.
//!
//! `dp/dt = c p (1-p) - e p` provides the smallest analytic spatial ecology
//! model with a colonization-extinction persistence threshold. Occupancy is a
//! fraction of suitable patches, not abundance within patches.

use crate::error::{ModelError, require_finite, require_non_negative, require_positive};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevinsMetapopulation {
    /// Colonization rate per unit time.
    pub colonization_rate: f64,
    /// Local extinction rate per unit time.
    pub extinction_rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetapopulationRegime {
    Extinction,
    Threshold,
    Persistence,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetapopulationEquilibrium {
    pub occupancy: f64,
    pub locally_stable: bool,
    pub hyperbolic: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetapopulationSample {
    pub time: f64,
    pub occupancy: f64,
}

impl LevinsMetapopulation {
    pub fn try_new(colonization_rate: f64, extinction_rate: f64) -> Result<Self, ModelError> {
        let model = Self {
            colonization_rate,
            extinction_rate,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_non_negative("colonization_rate", self.colonization_rate)?;
        require_non_negative("extinction_rate", self.extinction_rate)?;
        if self.colonization_rate == 0.0 && self.extinction_rate == 0.0 {
            return Err(ModelError::SingularCalibration {
                reason: "colonization and extinction cannot both be zero",
            });
        }
        Ok(())
    }

    pub fn tendency(&self, occupancy: f64) -> f64 {
        self.colonization_rate * occupancy * (1.0 - occupancy) - self.extinction_rate * occupancy
    }

    pub fn tendency_derivative(&self, occupancy: f64) -> f64 {
        self.colonization_rate * (1.0 - 2.0 * occupancy) - self.extinction_rate
    }

    pub fn recovery_diagnostic(
        &self,
        occupancy: f64,
    ) -> Result<crate::recovery::RecoveryDiagnostic, ModelError> {
        self.validate()?;
        require_occupancy(occupancy)?;
        crate::recovery::scalar_recovery_diagnostic(self.tendency_derivative(occupancy))
    }

    pub fn try_tendency(&self, occupancy: f64) -> Result<f64, ModelError> {
        self.validate()?;
        require_occupancy(occupancy)?;
        Ok(self.tendency(occupancy))
    }

    pub fn regime(&self) -> MetapopulationRegime {
        if self.colonization_rate > self.extinction_rate {
            MetapopulationRegime::Persistence
        } else if self.colonization_rate < self.extinction_rate {
            MetapopulationRegime::Extinction
        } else {
            MetapopulationRegime::Threshold
        }
    }

    pub fn persistence_equilibrium(&self) -> Option<f64> {
        if self.colonization_rate > self.extinction_rate {
            Some(1.0 - self.extinction_rate / self.colonization_rate)
        } else {
            None
        }
    }

    pub fn equilibria(&self) -> Result<Vec<MetapopulationEquilibrium>, ModelError> {
        self.validate()?;
        let mut equilibria = vec![MetapopulationEquilibrium {
            occupancy: 0.0,
            locally_stable: self.colonization_rate <= self.extinction_rate,
            hyperbolic: self.colonization_rate != self.extinction_rate,
        }];
        if let Some(occupancy) = self.persistence_equilibrium() {
            equilibria.push(MetapopulationEquilibrium {
                occupancy,
                locally_stable: true,
                hyperbolic: true,
            });
        }
        Ok(equilibria)
    }

    /// Exact occupancy at non-negative elapsed time.
    pub fn exact_occupancy(
        &self,
        initial_occupancy: f64,
        elapsed_time: f64,
    ) -> Result<f64, ModelError> {
        self.validate()?;
        require_occupancy(initial_occupancy)?;
        require_non_negative("elapsed_time", elapsed_time)?;
        if initial_occupancy == 0.0 || elapsed_time == 0.0 {
            return Ok(initial_occupancy);
        }
        let net_growth = self.colonization_rate - self.extinction_rate;
        let occupancy = if self.colonization_rate == 0.0 {
            initial_occupancy * (-self.extinction_rate * elapsed_time).exp()
        } else if net_growth.abs() <= 64.0 * f64::EPSILON {
            initial_occupancy / (1.0 + self.colonization_rate * initial_occupancy * elapsed_time)
        } else if net_growth > 0.0 {
            let carrying_occupancy = net_growth / self.colonization_rate;
            carrying_occupancy
                / (1.0
                    + (carrying_occupancy / initial_occupancy - 1.0)
                        * (-net_growth * elapsed_time).exp())
        } else {
            let exponential = (net_growth * elapsed_time).exp();
            initial_occupancy * exponential
                / (1.0
                    + self.colonization_rate * initial_occupancy * (exponential - 1.0) / net_growth)
        };
        require_occupancy_with_tolerance(occupancy)
    }

    /// Exact timestamped trajectory including the initial occupancy.
    pub fn exact_trajectory(
        &self,
        initial_occupancy: f64,
        dt: f64,
        steps: usize,
    ) -> Result<Vec<MetapopulationSample>, ModelError> {
        crate::integration::validate_trajectory_request(dt, steps)?;
        let mut samples = Vec::with_capacity(steps + 1);
        for step in 0..=steps {
            let time = step as f64 * dt;
            samples.push(MetapopulationSample {
                time,
                occupancy: self.exact_occupancy(initial_occupancy, time)?,
            });
        }
        Ok(samples)
    }
}

fn require_occupancy(value: f64) -> Result<(), ModelError> {
    require_finite("occupancy", value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ModelError::OutOfRange {
            parameter: "occupancy",
            value,
            min: 0.0,
            max: 1.0,
        })
    }
}

fn require_occupancy_with_tolerance(value: f64) -> Result<f64, ModelError> {
    require_finite("occupancy", value)?;
    let tolerance = 64.0 * f64::EPSILON;
    if value >= -tolerance && value <= 1.0 + tolerance {
        Ok(value.clamp(0.0, 1.0))
    } else {
        Err(ModelError::OutOfRange {
            parameter: "occupancy",
            value,
            min: 0.0,
            max: 1.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistence_threshold_and_equilibrium_are_explicit() {
        let persistent = LevinsMetapopulation::try_new(0.4, 0.1).unwrap();
        assert_eq!(persistent.regime(), MetapopulationRegime::Persistence);
        assert!((persistent.persistence_equilibrium().unwrap() - 0.75).abs() < 1e-12);
        let equilibria = persistent.equilibria().unwrap();
        assert!(!equilibria[0].locally_stable);
        assert!(equilibria[1].locally_stable);

        let extinct = LevinsMetapopulation::try_new(0.1, 0.4).unwrap();
        assert_eq!(extinct.regime(), MetapopulationRegime::Extinction);
        assert!(extinct.persistence_equilibrium().is_none());
        assert!(extinct.equilibria().unwrap()[0].locally_stable);
    }

    #[test]
    fn exact_solution_relaxes_to_the_correct_attractor() {
        let persistent = LevinsMetapopulation::try_new(0.4, 0.1).unwrap();
        let occupancy = persistent.exact_occupancy(0.2, 200.0).unwrap();
        assert!((occupancy - 0.75).abs() < 1e-12);

        let extinct = LevinsMetapopulation::try_new(0.1, 0.4).unwrap();
        assert!(extinct.exact_occupancy(0.8, 200.0).unwrap() < 1e-20);
    }

    #[test]
    fn threshold_case_has_inverse_time_decay() {
        let model = LevinsMetapopulation::try_new(0.5, 0.5).unwrap();
        let expected = 0.4 / (1.0 + 0.5 * 0.4 * 10.0);
        assert!((model.exact_occupancy(0.4, 10.0).unwrap() - expected).abs() < 1e-12);
        assert_eq!(model.regime(), MetapopulationRegime::Threshold);
        assert!(!model.equilibria().unwrap()[0].hyperbolic);
    }

    #[test]
    fn large_positive_time_reaches_persistence_without_overflow() {
        let model = LevinsMetapopulation::try_new(2.0, 0.1).unwrap();
        let equilibrium = model.persistence_equilibrium().unwrap();
        let occupancy = model.exact_occupancy(1.0e-12, 1.0e6).unwrap();
        assert!((occupancy - equilibrium).abs() < 1e-12);
    }

    #[test]
    fn occupancy_domain_is_enforced() {
        let model = LevinsMetapopulation::try_new(0.4, 0.1).unwrap();
        assert!(model.exact_occupancy(-0.1, 1.0).is_err());
        assert!(model.exact_occupancy(1.1, 1.0).is_err());
    }

    #[test]
    fn recovery_slows_near_the_colonization_extinction_threshold() {
        let far = LevinsMetapopulation::try_new(0.5, 0.1).unwrap();
        let far_time = far
            .recovery_diagnostic(far.persistence_equilibrium().unwrap())
            .unwrap()
            .e_folding_time
            .unwrap();
        let near = LevinsMetapopulation::try_new(0.1001, 0.1).unwrap();
        let near_time = near
            .recovery_diagnostic(near.persistence_equilibrium().unwrap())
            .unwrap()
            .e_folding_time
            .unwrap();
        assert!(near_time > far_time);
        let critical = LevinsMetapopulation::try_new(0.1, 0.1).unwrap();
        assert_eq!(
            critical.recovery_diagnostic(0.0).unwrap().stability,
            crate::recovery::LinearStability::Critical
        );
    }
}
