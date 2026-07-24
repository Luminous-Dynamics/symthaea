// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hutchinson delayed-logistic population dynamics.
//!
//! `dN/dt = r N(t) [1 - N(t-τ)/K]` is the smallest deterministic baseline in
//! this crate with an explicit response delay. The positive equilibrium loses
//! local stability at `r τ = π/2`. Numerical trajectories use a fixed-step
//! method of steps with linearly interpolated history and require `dt <= τ` so
//! every delayed RK4 evaluation is already represented in the stored history.

use crate::error::{ModelError, require_finite, require_positive};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HutchinsonDelayLogistic {
    /// Intrinsic growth rate per unit time.
    pub growth_rate: f64,
    /// Carrying capacity in population units.
    pub carrying_capacity: f64,
    /// Response delay in the same time unit used by `growth_rate`.
    pub delay: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelayEquilibriumStability {
    Stable,
    Critical,
    Unstable,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DelayedLogisticSample {
    pub time: f64,
    pub population: f64,
    pub delayed_population: f64,
}

impl HutchinsonDelayLogistic {
    pub fn try_new(
        growth_rate: f64,
        carrying_capacity: f64,
        delay: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            growth_rate,
            carrying_capacity,
            delay,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("growth_rate", self.growth_rate)?;
        require_positive("carrying_capacity", self.carrying_capacity)?;
        require_positive("delay", self.delay)
    }

    /// Dimensionless delay controlling local stability of `N = K`.
    pub fn delay_number(&self) -> f64 {
        self.growth_rate * self.delay
    }

    /// The analytic Hopf threshold in delay units, `π/(2r)`.
    pub fn critical_delay(&self) -> f64 {
        core::f64::consts::FRAC_PI_2 / self.growth_rate
    }

    pub fn equilibrium_stability(&self) -> DelayEquilibriumStability {
        let margin = self.delay_number() - core::f64::consts::FRAC_PI_2;
        let tolerance = 64.0 * f64::EPSILON * self.delay_number().abs().max(1.0);
        if margin.abs() <= tolerance {
            DelayEquilibriumStability::Critical
        } else if margin < 0.0 {
            DelayEquilibriumStability::Stable
        } else {
            DelayEquilibriumStability::Unstable
        }
    }

    pub fn tendency(&self, population: f64, delayed_population: f64) -> f64 {
        self.growth_rate * population * (1.0 - delayed_population / self.carrying_capacity)
    }

    /// Fixed-step method-of-steps trajectory with constant prehistory equal to
    /// `initial_population`. The returned series includes the initial state.
    pub fn try_simulate(
        &self,
        initial_population: f64,
        dt: f64,
        steps: usize,
    ) -> Result<Vec<DelayedLogisticSample>, ModelError> {
        self.validate()?;
        require_positive("initial_population", initial_population)?;
        crate::integration::validate_trajectory_request(dt, steps)?;
        if dt > self.delay {
            return Err(ModelError::OutOfRange {
                parameter: "dt",
                value: dt,
                min: f64::from_bits(1),
                max: self.delay,
            });
        }

        let mut samples = Vec::with_capacity(steps + 1);
        samples.push(DelayedLogisticSample {
            time: 0.0,
            population: initial_population,
            delayed_population: initial_population,
        });

        for step in 1..=steps {
            let time = (step - 1) as f64 * dt;
            let population = samples[step - 1].population;
            let k1 = self.tendency(
                population,
                delayed_value(&samples, time - self.delay, dt, initial_population),
            );
            validate_value(step, "population_tendency_stage_1", k1, false)?;

            let stage2 = population + 0.5 * dt * k1;
            validate_value(step, "population_stage_2", stage2, true)?;
            let k2 = self.tendency(
                stage2,
                delayed_value(
                    &samples,
                    time + 0.5 * dt - self.delay,
                    dt,
                    initial_population,
                ),
            );
            validate_value(step, "population_tendency_stage_2", k2, false)?;

            let stage3 = population + 0.5 * dt * k2;
            validate_value(step, "population_stage_3", stage3, true)?;
            let k3 = self.tendency(
                stage3,
                delayed_value(
                    &samples,
                    time + 0.5 * dt - self.delay,
                    dt,
                    initial_population,
                ),
            );
            validate_value(step, "population_tendency_stage_3", k3, false)?;

            let stage4 = population + dt * k3;
            validate_value(step, "population_stage_4", stage4, true)?;
            let k4 = self.tendency(
                stage4,
                delayed_value(&samples, time + dt - self.delay, dt, initial_population),
            );
            validate_value(step, "population_tendency_stage_4", k4, false)?;

            let next = population + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            validate_value(step, "population", next, true)?;
            let next_time = step as f64 * dt;
            samples.push(DelayedLogisticSample {
                time: next_time,
                population: next,
                delayed_population: delayed_value(
                    &samples,
                    next_time - self.delay,
                    dt,
                    initial_population,
                ),
            });
        }
        Ok(samples)
    }
}

fn delayed_value(
    samples: &[DelayedLogisticSample],
    query_time: f64,
    dt: f64,
    history_value: f64,
) -> f64 {
    if query_time <= 0.0 {
        return history_value;
    }
    let position = query_time / dt;
    let lower = position.floor() as usize;
    let fraction = position - lower as f64;
    if fraction <= 16.0 * f64::EPSILON {
        return samples[lower].population;
    }
    let upper = lower + 1;
    samples[lower].population + fraction * (samples[upper].population - samples[lower].population)
}

fn validate_value(
    step: usize,
    component: &'static str,
    value: f64,
    positive: bool,
) -> Result<(), ModelError> {
    require_finite(component, value)?;
    if !positive || value > 0.0 {
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
    fn stability_changes_at_pi_over_two() {
        let growth_rate = 0.5;
        let critical = core::f64::consts::FRAC_PI_2 / growth_rate;
        assert_eq!(
            HutchinsonDelayLogistic::try_new(growth_rate, 100.0, 0.5 * critical)
                .unwrap()
                .equilibrium_stability(),
            DelayEquilibriumStability::Stable
        );
        assert_eq!(
            HutchinsonDelayLogistic::try_new(growth_rate, 100.0, critical)
                .unwrap()
                .equilibrium_stability(),
            DelayEquilibriumStability::Critical
        );
        assert_eq!(
            HutchinsonDelayLogistic::try_new(growth_rate, 100.0, 2.0 * critical)
                .unwrap()
                .equilibrium_stability(),
            DelayEquilibriumStability::Unstable
        );
    }

    #[test]
    fn carrying_capacity_is_an_exact_constant_history_solution() {
        let model = HutchinsonDelayLogistic::try_new(0.5, 100.0, 1.0).unwrap();
        let samples = model.try_simulate(100.0, 0.05, 200).unwrap();
        assert!(
            samples
                .iter()
                .all(|sample| (sample.population - 100.0).abs() < 1.0e-12)
        );
    }

    #[test]
    fn stable_delay_relaxes_toward_capacity() {
        let model = HutchinsonDelayLogistic::try_new(0.5, 100.0, 1.0).unwrap();
        let samples = model.try_simulate(10.0, 0.02, 4_000).unwrap();
        let final_population = samples.last().unwrap().population;
        assert!((final_population - 100.0).abs() < 1.0e-4);
        assert!(samples.iter().all(|sample| sample.population.is_finite()));
    }

    #[test]
    fn integration_step_must_resolve_the_delay() {
        let model = HutchinsonDelayLogistic::try_new(0.5, 100.0, 1.0).unwrap();
        assert!(model.try_simulate(10.0, 2.0, 10).is_err());
    }
}
