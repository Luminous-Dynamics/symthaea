// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Rosenzweig-MacArthur predator-prey dynamics.
//!
//! This extends classical Lotka-Volterra with logistic prey growth and a
//! saturating Holling type-II functional response. It is still a reduced-order
//! baseline, but unlike classical Lotka-Volterra it has bounded prey resources
//! and can express enrichment-driven loss of coexistence stability.

use crate::error::{ModelError, require_non_negative, require_positive};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RosenzweigMacArthur {
    pub prey_growth_rate: f64,
    pub prey_carrying_capacity: f64,
    pub attack_rate: f64,
    pub handling_time: f64,
    pub conversion_efficiency: f64,
    pub predator_mortality: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoexistenceStability {
    Stable,
    Unstable,
    Saddle,
    Degenerate,
    NoCoexistence,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnrichmentSlice {
    pub carrying_capacity: f64,
    pub coexistence_equilibrium: Option<(f64, f64)>,
    pub jacobian_trace: Option<f64>,
    pub jacobian_determinant: Option<f64>,
    pub stability: CoexistenceStability,
}

impl RosenzweigMacArthur {
    pub fn try_new(
        prey_growth_rate: f64,
        prey_carrying_capacity: f64,
        attack_rate: f64,
        handling_time: f64,
        conversion_efficiency: f64,
        predator_mortality: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            prey_growth_rate,
            prey_carrying_capacity,
            attack_rate,
            handling_time,
            conversion_efficiency,
            predator_mortality,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("prey_growth_rate", self.prey_growth_rate)?;
        require_positive("prey_carrying_capacity", self.prey_carrying_capacity)?;
        require_positive("attack_rate", self.attack_rate)?;
        require_non_negative("handling_time", self.handling_time)?;
        require_positive("conversion_efficiency", self.conversion_efficiency)?;
        require_positive("predator_mortality", self.predator_mortality)?;
        Ok(())
    }

    /// Prey consumed per predator per unit time.
    pub fn functional_response(&self, prey: f64) -> f64 {
        self.attack_rate * prey / (1.0 + self.attack_rate * self.handling_time * prey)
    }

    /// Derivative of the functional response with respect to prey density.
    pub fn functional_response_derivative(&self, prey: f64) -> f64 {
        self.attack_rate / (1.0 + self.attack_rate * self.handling_time * prey).powi(2)
    }

    pub fn derivatives(&self, prey: f64, predator: f64) -> (f64, f64) {
        let consumption = self.functional_response(prey);
        (
            self.prey_growth_rate * prey * (1.0 - prey / self.prey_carrying_capacity)
                - consumption * predator,
            (self.conversion_efficiency * consumption - self.predator_mortality) * predator,
        )
    }

    /// Positive coexistence equilibrium, when predator energetic gain can
    /// balance mortality and the required prey density lies below capacity.
    pub fn coexistence_equilibrium(&self) -> Option<(f64, f64)> {
        if self.validate().is_err() {
            return None;
        }
        let energetic_margin =
            self.conversion_efficiency - self.predator_mortality * self.handling_time;
        if energetic_margin <= 0.0 {
            return None;
        }
        let prey = self.predator_mortality / (self.attack_rate * energetic_margin);
        if prey <= 0.0 || prey >= self.prey_carrying_capacity {
            return None;
        }
        let predator = self.prey_growth_rate
            * (1.0 - prey / self.prey_carrying_capacity)
            * (1.0 + self.attack_rate * self.handling_time * prey)
            / self.attack_rate;
        (predator > 0.0).then_some((prey, predator))
    }

    pub fn jacobian(&self, prey: f64, predator: f64) -> [[f64; 2]; 2] {
        let response = self.functional_response(prey);
        let response_derivative = self.functional_response_derivative(prey);
        [
            [
                self.prey_growth_rate * (1.0 - 2.0 * prey / self.prey_carrying_capacity)
                    - response_derivative * predator,
                -response,
            ],
            [
                self.conversion_efficiency * response_derivative * predator,
                self.conversion_efficiency * response - self.predator_mortality,
            ],
        ]
    }

    pub fn coexistence_stability(&self) -> CoexistenceStability {
        let Some((prey, predator)) = self.coexistence_equilibrium() else {
            return CoexistenceStability::NoCoexistence;
        };
        let jacobian = self.jacobian(prey, predator);
        let trace = jacobian[0][0] + jacobian[1][1];
        let determinant = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
        let scale = trace.abs().max(determinant.abs().sqrt()).max(1.0);
        let tolerance = 1e-12 * scale;

        if determinant < -tolerance {
            CoexistenceStability::Saddle
        } else if determinant.abs() <= tolerance || trace.abs() <= tolerance {
            CoexistenceStability::Degenerate
        } else if trace < 0.0 {
            CoexistenceStability::Stable
        } else {
            CoexistenceStability::Unstable
        }
    }

    /// Analytic carrying-capacity threshold at which coexistence changes
    /// stability through the classical enrichment-driven Hopf boundary.
    /// Returns `None` when handling time is zero or coexistence is impossible.
    pub fn hopf_threshold_carrying_capacity(&self) -> Option<f64> {
        if self.validate().is_err() || self.handling_time == 0.0 {
            return None;
        }
        let energetic_margin =
            self.conversion_efficiency - self.predator_mortality * self.handling_time;
        if energetic_margin <= 0.0 {
            return None;
        }
        let coexistence_prey = self.predator_mortality / (self.attack_rate * energetic_margin);
        Some(1.0 / (self.attack_rate * self.handling_time) + 2.0 * coexistence_prey)
    }

    /// Deterministic carrying-capacity continuation with endpoint inclusion.
    pub fn enrichment_sweep(
        &self,
        minimum_capacity: f64,
        maximum_capacity: f64,
        points: usize,
    ) -> Result<Vec<EnrichmentSlice>, ModelError> {
        self.validate()?;
        require_positive("minimum_capacity", minimum_capacity)?;
        require_positive("maximum_capacity", maximum_capacity)?;
        if minimum_capacity >= maximum_capacity {
            return Err(ModelError::OutOfRange {
                parameter: "maximum_capacity",
                value: maximum_capacity,
                min: minimum_capacity,
                max: f64::INFINITY,
            });
        }
        if points < 2 {
            return Err(ModelError::InsufficientSamples {
                required: 2,
                found: points,
            });
        }

        let mut slices = Vec::with_capacity(points);
        for index in 0..points {
            let fraction = index as f64 / (points - 1) as f64;
            let carrying_capacity =
                minimum_capacity + fraction * (maximum_capacity - minimum_capacity);
            let model = Self::try_new(
                self.prey_growth_rate,
                carrying_capacity,
                self.attack_rate,
                self.handling_time,
                self.conversion_efficiency,
                self.predator_mortality,
            )?;
            let coexistence_equilibrium = model.coexistence_equilibrium();
            let (jacobian_trace, jacobian_determinant) =
                if let Some((prey, predator)) = coexistence_equilibrium {
                    let jacobian = model.jacobian(prey, predator);
                    (
                        Some(jacobian[0][0] + jacobian[1][1]),
                        Some(jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]),
                    )
                } else {
                    (None, None)
                };
            slices.push(EnrichmentSlice {
                carrying_capacity,
                coexistence_equilibrium,
                jacobian_trace,
                jacobian_determinant,
                stability: model.coexistence_stability(),
            });
        }
        Ok(slices)
    }

    pub fn simulate(&self, prey0: f64, predator0: f64, dt: f64, steps: usize) -> Vec<(f64, f64)> {
        crate::integration::simulate_pair_unchecked(
            prey0,
            predator0,
            dt,
            steps,
            |prey, predator| self.derivatives(prey, predator),
        )
    }

    pub fn try_simulate(
        &self,
        prey0: f64,
        predator0: f64,
        dt: f64,
        steps: usize,
    ) -> Result<Vec<(f64, f64)>, ModelError> {
        self.validate()?;
        require_positive("initial_prey", prey0)?;
        require_positive("initial_predator", predator0)?;
        require_positive("dt", dt)?;
        if steps == 0 {
            return Err(ModelError::ZeroSteps);
        }
        Ok(crate::integration::simulate_positive_pair(
            prey0,
            predator0,
            dt,
            steps,
            |prey, predator| self.derivatives(prey, predator),
        )?
        .into_iter()
        .map(|sample| (sample.first, sample.second))
        .collect())
    }

    /// Guarded RK4 trajectory with explicit timestamps and the initial state.
    pub fn try_simulate_timestamped(
        &self,
        prey0: f64,
        predator0: f64,
        dt: f64,
        steps: usize,
    ) -> Result<Vec<crate::integration::PopulationPairSample>, ModelError> {
        self.validate()?;
        require_positive("initial_prey", prey0)?;
        require_positive("initial_predator", predator0)?;
        require_positive("dt", dt)?;
        if steps == 0 {
            return Err(ModelError::ZeroSteps);
        }
        crate::integration::simulate_positive_pair(prey0, predator0, dt, steps, |prey, predator| {
            self.derivatives(prey, predator)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(carrying_capacity: f64) -> RosenzweigMacArthur {
        RosenzweigMacArthur::try_new(1.0, carrying_capacity, 0.1, 0.1, 0.5, 0.2).unwrap()
    }

    #[test]
    fn coexistence_equilibrium_zeroes_derivatives() {
        let m = model(20.0);
        let (prey, predator) = m.coexistence_equilibrium().unwrap();
        let (dprey, dpredator) = m.derivatives(prey, predator);
        assert!(dprey.abs() < 1e-12);
        assert!(dpredator.abs() < 1e-12);
    }

    #[test]
    fn enrichment_can_destabilize_coexistence() {
        assert_eq!(
            model(20.0).coexistence_stability(),
            CoexistenceStability::Stable
        );
        assert_eq!(
            model(200.0).coexistence_stability(),
            CoexistenceStability::Unstable
        );
    }

    #[test]
    fn insufficient_conversion_prevents_coexistence() {
        let m = RosenzweigMacArthur::try_new(1.0, 100.0, 0.1, 3.0, 0.5, 0.2).unwrap();
        assert_eq!(
            m.coexistence_stability(),
            CoexistenceStability::NoCoexistence
        );
    }

    #[test]
    fn checked_simulation_remains_positive_at_resolved_step() {
        let trajectory = model(20.0).try_simulate(5.0, 5.0, 0.001, 10_000).unwrap();
        assert!(trajectory.iter().all(|(prey, predator)| {
            prey.is_finite() && predator.is_finite() && *prey > 0.0 && *predator > 0.0
        }));
        assert!(matches!(
            model(20.0).try_simulate(5.0, 5.0, 100.0, 2),
            Err(ModelError::IntegrationDomainViolation { .. })
        ));
    }
    #[test]
    fn timestamped_trajectory_has_explicit_sample_contract() {
        let trajectory = model(20.0)
            .try_simulate_timestamped(5.0, 5.0, 0.001, 100)
            .unwrap();
        assert_eq!(trajectory.len(), 101);
        assert_eq!(trajectory[0].time, 0.0);
        assert!((trajectory.last().unwrap().time - 0.1).abs() < 1e-12);
    }

    #[test]
    fn analytic_hopf_threshold_zeroes_trace() {
        let base = model(20.0);
        let threshold = base.hopf_threshold_carrying_capacity().unwrap();
        let at_threshold = model(threshold);
        let (prey, predator) = at_threshold.coexistence_equilibrium().unwrap();
        let jacobian = at_threshold.jacobian(prey, predator);
        let trace = jacobian[0][0] + jacobian[1][1];
        assert!(trace.abs() < 1e-12);
        assert_eq!(
            at_threshold.coexistence_stability(),
            CoexistenceStability::Degenerate
        );
    }

    #[test]
    fn enrichment_sweep_crosses_stability_boundary() {
        let slices = model(20.0).enrichment_sweep(20.0, 200.0, 19).unwrap();
        assert_eq!(slices.len(), 19);
        assert_eq!(
            slices.first().unwrap().stability,
            CoexistenceStability::Stable
        );
        assert_eq!(
            slices.last().unwrap().stability,
            CoexistenceStability::Unstable
        );
        assert!(slices.iter().all(|slice| {
            slice
                .jacobian_determinant
                .map(|determinant| determinant > 0.0)
                .unwrap_or(true)
        }));
    }

    #[test]
    fn zero_handling_time_has_no_finite_hopf_threshold() {
        let model = RosenzweigMacArthur::try_new(1.0, 100.0, 0.1, 0.0, 0.5, 0.2).unwrap();
        assert_eq!(model.hopf_threshold_carrying_capacity(), None);
        assert_eq!(model.coexistence_stability(), CoexistenceStability::Stable);
    }
}
