// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Two-patch occupancy with directional colonization and local extinction.
//!
//! This is the smallest metapopulation model in the crate where connectivity
//! and extinction can differ by patch. The extinction equilibrium loses local
//! stability when the geometric mean colonization coupling exceeds the
//! geometric mean extinction rate: `c12 c21 > e1 e2`.

use crate::error::{ModelError, require_non_negative, require_positive};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPatchMetapopulation {
    /// Colonization pressure from patch 2 into patch 1, per unit time.
    pub colonization_12: f64,
    /// Colonization pressure from patch 1 into patch 2, per unit time.
    pub colonization_21: f64,
    pub extinction_1: f64,
    pub extinction_2: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoPatchRegime {
    Extinction,
    Threshold,
    Persistence,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPatchSample {
    pub time: f64,
    pub occupancy_1: f64,
    pub occupancy_2: f64,
}

impl TwoPatchMetapopulation {
    pub fn try_new(
        colonization_12: f64,
        colonization_21: f64,
        extinction_1: f64,
        extinction_2: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            colonization_12,
            colonization_21,
            extinction_1,
            extinction_2,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_non_negative("colonization_12", self.colonization_12)?;
        require_non_negative("colonization_21", self.colonization_21)?;
        require_positive("extinction_1", self.extinction_1)?;
        require_positive("extinction_2", self.extinction_2)
    }

    /// Dimensionless connectivity ratio. Persistence requires a value above 1.
    pub fn connectivity_ratio(&self) -> f64 {
        self.colonization_12 * self.colonization_21 / (self.extinction_1 * self.extinction_2)
    }

    pub fn regime(&self) -> TwoPatchRegime {
        let margin =
            self.colonization_12 * self.colonization_21 - self.extinction_1 * self.extinction_2;
        let scale = (self.colonization_12 * self.colonization_21)
            .abs()
            .max((self.extinction_1 * self.extinction_2).abs())
            .max(1.0);
        if margin.abs() <= 64.0 * f64::EPSILON * scale {
            TwoPatchRegime::Threshold
        } else if margin > 0.0 {
            TwoPatchRegime::Persistence
        } else {
            TwoPatchRegime::Extinction
        }
    }

    /// Dominant linear growth rate of an occupancy perturbation around zero.
    pub fn extinction_dominant_eigenvalue(&self) -> f64 {
        let half_trace = -0.5 * (self.extinction_1 + self.extinction_2);
        let half_difference = 0.5 * (self.extinction_1 - self.extinction_2);
        half_trace
            + (half_difference * half_difference + self.colonization_12 * self.colonization_21)
                .sqrt()
    }

    pub fn persistence_equilibrium(&self) -> Option<(f64, f64)> {
        if self.regime() != TwoPatchRegime::Persistence {
            return None;
        }
        let numerator =
            self.colonization_12 * self.colonization_21 - self.extinction_1 * self.extinction_2;
        Some((
            numerator / (self.colonization_21 * (self.colonization_12 + self.extinction_1)),
            numerator / (self.colonization_12 * (self.colonization_21 + self.extinction_2)),
        ))
    }

    pub fn derivatives(&self, occupancy_1: f64, occupancy_2: f64) -> (f64, f64) {
        (
            self.colonization_12 * occupancy_2 * (1.0 - occupancy_1)
                - self.extinction_1 * occupancy_1,
            self.colonization_21 * occupancy_1 * (1.0 - occupancy_2)
                - self.extinction_2 * occupancy_2,
        )
    }

    pub fn jacobian(&self, occupancy_1: f64, occupancy_2: f64) -> [[f64; 2]; 2] {
        [
            [
                -self.colonization_12 * occupancy_2 - self.extinction_1,
                self.colonization_12 * (1.0 - occupancy_1),
            ],
            [
                self.colonization_21 * (1.0 - occupancy_2),
                -self.colonization_21 * occupancy_1 - self.extinction_2,
            ],
        ]
    }

    pub fn try_simulate(
        &self,
        initial_occupancy_1: f64,
        initial_occupancy_2: f64,
        dt: f64,
        steps: usize,
    ) -> Result<Vec<TwoPatchSample>, ModelError> {
        self.validate()?;
        validate_occupancy(0, "initial_occupancy_1", initial_occupancy_1)?;
        validate_occupancy(0, "initial_occupancy_2", initial_occupancy_2)?;
        crate::integration::validate_trajectory_request(dt, steps)?;

        let mut samples = Vec::with_capacity(steps + 1);
        let mut state = (initial_occupancy_1, initial_occupancy_2);
        samples.push(TwoPatchSample {
            time: 0.0,
            occupancy_1: state.0,
            occupancy_2: state.1,
        });
        for step in 1..=steps {
            state = self.rk4_step(state, dt, step)?;
            samples.push(TwoPatchSample {
                time: step as f64 * dt,
                occupancy_1: state.0,
                occupancy_2: state.1,
            });
        }
        Ok(samples)
    }

    fn rk4_step(&self, state: (f64, f64), dt: f64, step: usize) -> Result<(f64, f64), ModelError> {
        let k1 = self.derivatives(state.0, state.1);
        let stage2 = (state.0 + 0.5 * dt * k1.0, state.1 + 0.5 * dt * k1.1);
        validate_occupancy(step, "occupancy_1_stage_2", stage2.0)?;
        validate_occupancy(step, "occupancy_2_stage_2", stage2.1)?;
        let k2 = self.derivatives(stage2.0, stage2.1);
        let stage3 = (state.0 + 0.5 * dt * k2.0, state.1 + 0.5 * dt * k2.1);
        validate_occupancy(step, "occupancy_1_stage_3", stage3.0)?;
        validate_occupancy(step, "occupancy_2_stage_3", stage3.1)?;
        let k3 = self.derivatives(stage3.0, stage3.1);
        let stage4 = (state.0 + dt * k3.0, state.1 + dt * k3.1);
        validate_occupancy(step, "occupancy_1_stage_4", stage4.0)?;
        validate_occupancy(step, "occupancy_2_stage_4", stage4.1)?;
        let k4 = self.derivatives(stage4.0, stage4.1);
        let next = (
            state.0 + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0,
            state.1 + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0,
        );
        validate_occupancy(step, "occupancy_1", next.0)?;
        validate_occupancy(step, "occupancy_2", next.1)?;
        Ok(next)
    }
}

fn validate_occupancy(step: usize, component: &'static str, value: f64) -> Result<(), ModelError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ModelError::IntegrationDomainViolation {
            step,
            component,
            value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_threshold_matches_extinction_eigenvalue() {
        let below = TwoPatchMetapopulation::try_new(0.2, 0.2, 0.3, 0.3).unwrap();
        assert_eq!(below.regime(), TwoPatchRegime::Extinction);
        assert!(below.extinction_dominant_eigenvalue() < 0.0);

        let threshold = TwoPatchMetapopulation::try_new(0.3, 0.3, 0.3, 0.3).unwrap();
        assert_eq!(threshold.regime(), TwoPatchRegime::Threshold);
        assert!(threshold.extinction_dominant_eigenvalue().abs() < 1.0e-12);

        let above = TwoPatchMetapopulation::try_new(0.5, 0.4, 0.2, 0.3).unwrap();
        assert_eq!(above.regime(), TwoPatchRegime::Persistence);
        assert!(above.extinction_dominant_eigenvalue() > 0.0);
    }

    #[test]
    fn persistence_equilibrium_zeroes_both_tendencies() {
        let model = TwoPatchMetapopulation::try_new(0.5, 0.4, 0.2, 0.3).unwrap();
        let equilibrium = model.persistence_equilibrium().unwrap();
        let derivative = model.derivatives(equilibrium.0, equilibrium.1);
        assert!(derivative.0.abs() < 1.0e-12);
        assert!(derivative.1.abs() < 1.0e-12);
        assert!((0.0..1.0).contains(&equilibrium.0));
        assert!((0.0..1.0).contains(&equilibrium.1));
    }

    #[test]
    fn trajectories_select_the_analytic_regime() {
        let persistent = TwoPatchMetapopulation::try_new(0.5, 0.4, 0.2, 0.3).unwrap();
        let target = persistent.persistence_equilibrium().unwrap();
        let samples = persistent.try_simulate(0.05, 0.05, 0.02, 10_000).unwrap();
        let final_state = samples.last().unwrap();
        assert!((final_state.occupancy_1 - target.0).abs() < 1.0e-8);
        assert!((final_state.occupancy_2 - target.1).abs() < 1.0e-8);

        let extinct = TwoPatchMetapopulation::try_new(0.1, 0.1, 0.3, 0.3).unwrap();
        let samples = extinct.try_simulate(0.5, 0.5, 0.02, 10_000).unwrap();
        let final_state = samples.last().unwrap();
        assert!(final_state.occupancy_1 < 1.0e-12);
        assert!(final_state.occupancy_2 < 1.0e-12);
    }
}
