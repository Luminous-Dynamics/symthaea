// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Explicit, guarded integration contracts for positive two-population models.
//!
//! The legacy model methods retain their original return types. These helpers
//! add timestamped trajectories that always include the initial state and fail
//! closed when an RK4 stage leaves the positive finite domain.

use crate::error::{ModelError, require_finite, require_positive};

pub const MAX_TRAJECTORY_STEPS: usize = 1_000_000;

pub(crate) fn validate_step_count(steps: usize) -> Result<(), ModelError> {
    if steps == 0 {
        return Err(ModelError::ZeroSteps);
    }
    if steps > MAX_TRAJECTORY_STEPS {
        return Err(ModelError::TrajectoryTooLarge {
            requested: steps,
            maximum: MAX_TRAJECTORY_STEPS,
        });
    }
    steps.checked_add(1).ok_or(ModelError::TrajectoryTooLarge {
        requested: usize::MAX,
        maximum: MAX_TRAJECTORY_STEPS,
    })?;
    Ok(())
}

pub(crate) fn validate_trajectory_request(dt: f64, steps: usize) -> Result<(), ModelError> {
    require_positive("dt", dt)?;
    validate_step_count(steps)?;
    require_finite("trajectory_duration", dt * steps as f64)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PopulationPairSample {
    pub time: f64,
    pub first: f64,
    pub second: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PopulationSample {
    pub time: f64,
    pub population: f64,
}

/// Guarded scalar RK4 trajectory for models whose physical domain includes
/// zero population. The returned trajectory includes the initial state.
pub(crate) fn simulate_non_negative_single<F>(
    initial_population: f64,
    dt: f64,
    steps: usize,
    derivative: F,
) -> Result<Vec<PopulationSample>, ModelError>
where
    F: Fn(f64) -> f64,
{
    validate_trajectory_request(dt, steps)?;
    let mut samples = Vec::with_capacity(steps + 1);
    let mut population = initial_population;
    samples.push(PopulationSample {
        time: 0.0,
        population,
    });
    for step in 1..=steps {
        validate_non_negative_state(step - 1, "population", population)?;
        let k1 = derivative(population);
        validate_derivative(step, "population_derivative_stage_1", k1)?;
        let stage2 = population + 0.5 * dt * k1;
        validate_non_negative_state(step, "population_stage_2", stage2)?;
        let k2 = derivative(stage2);
        validate_derivative(step, "population_derivative_stage_2", k2)?;
        let stage3 = population + 0.5 * dt * k2;
        validate_non_negative_state(step, "population_stage_3", stage3)?;
        let k3 = derivative(stage3);
        validate_derivative(step, "population_derivative_stage_3", k3)?;
        let stage4 = population + dt * k3;
        validate_non_negative_state(step, "population_stage_4", stage4)?;
        let k4 = derivative(stage4);
        validate_derivative(step, "population_derivative_stage_4", k4)?;
        population += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        validate_non_negative_state(step, "population", population)?;
        samples.push(PopulationSample {
            time: step as f64 * dt,
            population,
        });
    }
    Ok(samples)
}

/// Allocation-free two-state RK4 used by the legacy permissive APIs.
///
/// The trajectory includes the initial state and `steps` advanced states. This
/// helper intentionally performs no domain validation; checked public methods
/// use [`simulate_positive_pair`] instead.
pub(crate) fn simulate_pair_unchecked<F>(
    initial_first: f64,
    initial_second: f64,
    dt: f64,
    steps: usize,
    derivatives: F,
) -> Vec<(f64, f64)>
where
    F: Fn(f64, f64) -> (f64, f64),
{
    if steps == 0 {
        return Vec::new();
    }
    let mut trajectory = Vec::with_capacity(steps + 1);
    let mut first = initial_first;
    let mut second = initial_second;
    trajectory.push((first, second));
    for _ in 0..steps {
        let k1 = derivatives(first, second);
        let k2 = derivatives(first + 0.5 * dt * k1.0, second + 0.5 * dt * k1.1);
        let k3 = derivatives(first + 0.5 * dt * k2.0, second + 0.5 * dt * k2.1);
        let k4 = derivatives(first + dt * k3.0, second + dt * k3.1);
        first += dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
        second += dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;
        trajectory.push((first, second));
    }
    trajectory
}

pub(crate) fn simulate_positive_pair<F>(
    initial_first: f64,
    initial_second: f64,
    dt: f64,
    steps: usize,
    derivatives: F,
) -> Result<Vec<PopulationPairSample>, ModelError>
where
    F: Fn(f64, f64) -> (f64, f64),
{
    validate_trajectory_request(dt, steps)?;
    let mut samples = Vec::with_capacity(steps + 1);
    let mut first = initial_first;
    let mut second = initial_second;
    let mut time = 0.0;
    samples.push(PopulationPairSample {
        time,
        first,
        second,
    });

    for step in 1..=steps {
        let (next_first, next_second) =
            rk4_positive_pair_step(first, second, dt, step, &derivatives)?;
        first = next_first;
        second = next_second;
        time = step as f64 * dt;
        samples.push(PopulationPairSample {
            time,
            first,
            second,
        });
    }
    Ok(samples)
}

fn rk4_positive_pair_step<F>(
    first: f64,
    second: f64,
    dt: f64,
    step: usize,
    derivatives: &F,
) -> Result<(f64, f64), ModelError>
where
    F: Fn(f64, f64) -> (f64, f64),
{
    validate_state(step - 1, "first_population", first)?;
    validate_state(step - 1, "second_population", second)?;

    let k1 = derivatives(first, second);
    validate_derivative(step, "first_derivative", k1.0)?;
    validate_derivative(step, "second_derivative", k1.1)?;

    let stage2 = (first + 0.5 * dt * k1.0, second + 0.5 * dt * k1.1);
    validate_state(step, "first_stage_2", stage2.0)?;
    validate_state(step, "second_stage_2", stage2.1)?;
    let k2 = derivatives(stage2.0, stage2.1);

    let stage3 = (first + 0.5 * dt * k2.0, second + 0.5 * dt * k2.1);
    validate_state(step, "first_stage_3", stage3.0)?;
    validate_state(step, "second_stage_3", stage3.1)?;
    let k3 = derivatives(stage3.0, stage3.1);

    let stage4 = (first + dt * k3.0, second + dt * k3.1);
    validate_state(step, "first_stage_4", stage4.0)?;
    validate_state(step, "second_stage_4", stage4.1)?;
    let k4 = derivatives(stage4.0, stage4.1);

    let next_first = first + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
    let next_second = second + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;
    validate_state(step, "first_population", next_first)?;
    validate_state(step, "second_population", next_second)?;
    Ok((next_first, next_second))
}

fn validate_non_negative_state(
    step: usize,
    component: &'static str,
    value: f64,
) -> Result<(), ModelError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(ModelError::IntegrationDomainViolation {
            step,
            component,
            value,
        })
    }
}

fn validate_state(step: usize, component: &'static str, value: f64) -> Result<(), ModelError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ModelError::IntegrationDomainViolation {
            step,
            component,
            value,
        })
    }
}

fn validate_derivative(step: usize, component: &'static str, value: f64) -> Result<(), ModelError> {
    if value.is_finite() {
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
    fn timestamp_contract_includes_initial_state() {
        let samples = simulate_positive_pair(1.0, 2.0, 0.5, 3, |_, _| (0.0, 0.0)).unwrap();
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0].time, 0.0);
        assert_eq!(samples[0].first, 1.0);
        assert_eq!(samples[3].time, 1.5);
    }

    #[test]
    fn guarded_step_rejects_domain_escape() {
        let error = simulate_positive_pair(1.0, 1.0, 1.0, 1, |first, second| {
            (-10.0 * first, -10.0 * second)
        })
        .unwrap_err();
        assert!(matches!(
            error,
            ModelError::IntegrationDomainViolation { .. }
        ));
    }

    #[test]
    fn unchecked_integrator_includes_initial_and_requested_steps() {
        let trajectory = simulate_pair_unchecked(1.0, 2.0, 0.5, 3, |_, _| (0.0, 0.0));
        assert_eq!(trajectory, vec![(1.0, 2.0); 4]);
        assert!(simulate_pair_unchecked(1.0, 2.0, 0.5, 0, |_, _| (0.0, 0.0)).is_empty());
    }
    #[test]
    fn non_negative_scalar_integrator_accepts_extinction_equilibrium() {
        let samples = simulate_non_negative_single(0.0, 1.0, 3, |_| 0.0).unwrap();
        assert_eq!(samples.len(), 4);
        assert!(samples.iter().all(|sample| sample.population == 0.0));
    }

    #[test]
    fn checked_trajectories_reject_unbounded_allocations_and_time_overflow() {
        assert!(matches!(
            validate_trajectory_request(1.0, MAX_TRAJECTORY_STEPS + 1),
            Err(ModelError::TrajectoryTooLarge { .. })
        ));
        assert!(validate_trajectory_request(f64::MAX, 2).is_err());
    }
}
