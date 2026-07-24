// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Time-varying environmental drivers and non-autonomous logistic replay.
//!
//! Timeline `time` uses the same caller-chosen unit as ecological growth rates.
//! Climate outputs expressed in seconds or years must be converted explicitly;
//! this crate does not silently reinterpret units.

use crate::environment::{
    EnvironmentalDrivers, LogisticEnvironmentCoupling, LogisticEnvironmentEvaluation,
};
use crate::error::{ModelError, require_finite, require_non_negative, require_positive};

/// A deterministic source of environmental drivers in caller-chosen time units.
pub trait EnvironmentalDriverSource {
    fn drivers_at(&self, time: f64) -> Result<EnvironmentalDrivers, ModelError>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvironmentalWaypoint {
    pub time: f64,
    pub drivers: EnvironmentalDrivers,
}

impl EnvironmentalWaypoint {
    pub fn try_new(time: f64, drivers: EnvironmentalDrivers) -> Result<Self, ModelError> {
        require_non_negative("waypoint_time", time)?;
        drivers.validate()?;
        Ok(Self { time, drivers })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentalTimeline {
    waypoints: Vec<EnvironmentalWaypoint>,
}

impl EnvironmentalTimeline {
    pub fn try_new(waypoints: Vec<EnvironmentalWaypoint>) -> Result<Self, ModelError> {
        if waypoints.is_empty() {
            return Err(ModelError::EmptySeries {
                series: "environmental waypoints",
            });
        }
        let mut previous_time = None;
        for (index, waypoint) in waypoints.iter().enumerate() {
            require_non_negative("waypoint_time", waypoint.time)?;
            waypoint.drivers.validate()?;
            if let Some(previous) = previous_time {
                if waypoint.time <= previous {
                    return Err(ModelError::NonMonotonicTime {
                        index,
                        previous,
                        current: waypoint.time,
                    });
                }
            }
            previous_time = Some(waypoint.time);
        }
        Ok(Self { waypoints })
    }

    pub fn waypoints(&self) -> &[EnvironmentalWaypoint] {
        &self.waypoints
    }

    /// Piecewise-linear interpolation with endpoint holds outside the supplied
    /// waypoint interval.
    pub fn at(&self, time: f64) -> Result<EnvironmentalDrivers, ModelError> {
        require_non_negative("timeline_time", time)?;
        let first = self.waypoints[0];
        if time <= first.time {
            return Ok(first.drivers);
        }
        let last = self.waypoints[self.waypoints.len() - 1];
        if time >= last.time {
            return Ok(last.drivers);
        }
        let upper = self
            .waypoints
            .partition_point(|waypoint| waypoint.time <= time);
        let lower = self.waypoints[upper - 1];
        let upper = self.waypoints[upper];
        let fraction = (time - lower.time) / (upper.time - lower.time);
        EnvironmentalDrivers::try_new(
            lerp(
                lower.drivers.temperature,
                upper.drivers.temperature,
                fraction,
            ),
            lerp(
                lower.drivers.productivity,
                upper.drivers.productivity,
                fraction,
            ),
            lerp(
                lower.drivers.disturbance,
                upper.drivers.disturbance,
                fraction,
            ),
        )
    }
}

impl EnvironmentalDriverSource for EnvironmentalTimeline {
    fn drivers_at(&self, time: f64) -> Result<EnvironmentalDrivers, ModelError> {
        self.at(time)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogisticEnvironmentSample {
    pub time: f64,
    pub population: f64,
    pub drivers: EnvironmentalDrivers,
    pub evaluation: LogisticEnvironmentEvaluation,
}

/// Integrate a logistic population through a changing environment. The result
/// includes the initial state and evaluates environmental drivers at all RK4
/// substeps.
pub fn simulate_logistic_environment(
    coupling: &LogisticEnvironmentCoupling,
    initial_population: f64,
    timeline: &EnvironmentalTimeline,
    dt: f64,
    steps: usize,
) -> Result<Vec<LogisticEnvironmentSample>, ModelError> {
    simulate_logistic_driver_source(coupling, initial_population, timeline, dt, steps)
}

/// Integrate a logistic population through any deterministic driver source.
pub fn simulate_logistic_driver_source<S: EnvironmentalDriverSource + ?Sized>(
    coupling: &LogisticEnvironmentCoupling,
    initial_population: f64,
    source: &S,
    dt: f64,
    steps: usize,
) -> Result<Vec<LogisticEnvironmentSample>, ModelError> {
    coupling.validate()?;
    require_positive("initial_population", initial_population)?;
    crate::integration::validate_trajectory_request(dt, steps)?;

    let mut samples = Vec::with_capacity(steps + 1);
    let mut population = initial_population;
    let mut time = 0.0;
    let drivers = source.drivers_at(time)?;
    samples.push(LogisticEnvironmentSample {
        time,
        population,
        drivers,
        evaluation: coupling.evaluate(drivers)?,
    });

    for step in 1..=steps {
        population = step_logistic_environment(coupling, source, population, time, dt, step)?;
        time = step as f64 * dt;
        let drivers = source.drivers_at(time)?;
        samples.push(LogisticEnvironmentSample {
            time,
            population,
            drivers,
            evaluation: coupling.evaluate(drivers)?,
        });
    }
    Ok(samples)
}

fn step_logistic_environment<S: EnvironmentalDriverSource + ?Sized>(
    coupling: &LogisticEnvironmentCoupling,
    source: &S,
    population: f64,
    time: f64,
    dt: f64,
    step: usize,
) -> Result<f64, ModelError> {
    let tendency = |at_time: f64, at_population: f64| -> Result<f64, ModelError> {
        require_positive("population", at_population)?;
        let model = coupling.effective_model(source.drivers_at(at_time)?)?;
        model.growth_rate(at_population)
    };

    let k1 = tendency(time, population)?;
    let stage2 = population + 0.5 * dt * k1;
    validate_population(step, "population_stage_2", stage2)?;
    let k2 = tendency(time + 0.5 * dt, stage2)?;
    let stage3 = population + 0.5 * dt * k2;
    validate_population(step, "population_stage_3", stage3)?;
    let k3 = tendency(time + 0.5 * dt, stage3)?;
    let stage4 = population + dt * k3;
    validate_population(step, "population_stage_4", stage4)?;
    let k4 = tendency(time + dt, stage4)?;
    let next = population + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
    validate_population(step, "population", next)?;
    Ok(next)
}

fn validate_population(step: usize, component: &'static str, value: f64) -> Result<(), ModelError> {
    require_finite(component, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ModelError::IntegrationDomainViolation {
            step,
            component,
            value,
        })
    }
}

fn lerp(start: f64, end: f64, fraction: f64) -> f64 {
    start + fraction * (end - start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GaussianThermalResponse, LogisticModel};

    fn coupling() -> LogisticEnvironmentCoupling {
        LogisticEnvironmentCoupling::try_new(
            LogisticModel::try_new(0.5, 100.0).unwrap(),
            GaussianThermalResponse::try_new(293.0, 10.0, 0.1).unwrap(),
            1.0,
            0.5,
        )
        .unwrap()
    }

    fn drivers(temperature: f64, productivity: f64) -> EnvironmentalDrivers {
        EnvironmentalDrivers::try_new(temperature, productivity, 0.0).unwrap()
    }

    #[test]
    fn timeline_interpolates_and_holds_endpoints() {
        let timeline = EnvironmentalTimeline::try_new(vec![
            EnvironmentalWaypoint::try_new(1.0, drivers(290.0, 1.0)).unwrap(),
            EnvironmentalWaypoint::try_new(3.0, drivers(294.0, 0.5)).unwrap(),
        ])
        .unwrap();
        assert_eq!(timeline.at(0.0).unwrap().temperature, 290.0);
        let middle = timeline.at(2.0).unwrap();
        assert!((middle.temperature - 292.0).abs() < 1e-12);
        assert!((middle.productivity - 0.75).abs() < 1e-12);
        assert_eq!(timeline.at(10.0).unwrap().temperature, 294.0);
    }

    #[test]
    fn constant_environment_matches_fixed_logistic_model() {
        let coupling = coupling();
        let environment = drivers(293.0, 1.0);
        let timeline = EnvironmentalTimeline::try_new(vec![
            EnvironmentalWaypoint::try_new(0.0, environment).unwrap(),
        ])
        .unwrap();
        let dt = 0.01;
        let steps = 100;
        let trajectory =
            simulate_logistic_environment(&coupling, 10.0, &timeline, dt, steps).unwrap();
        let exact = coupling
            .effective_model(environment)
            .unwrap()
            .population(10.0, dt * steps as f64)
            .unwrap();
        assert!((trajectory.last().unwrap().population - exact).abs() < 1e-9);
        assert_eq!(trajectory.len(), steps + 1);
    }

    #[test]
    fn declining_productivity_reduces_final_population() {
        let stable = EnvironmentalTimeline::try_new(vec![
            EnvironmentalWaypoint::try_new(0.0, drivers(293.0, 1.0)).unwrap(),
        ])
        .unwrap();
        let decline = EnvironmentalTimeline::try_new(vec![
            EnvironmentalWaypoint::try_new(0.0, drivers(293.0, 1.0)).unwrap(),
            EnvironmentalWaypoint::try_new(10.0, drivers(293.0, 0.2)).unwrap(),
        ])
        .unwrap();
        let stable_final = simulate_logistic_environment(&coupling(), 10.0, &stable, 0.01, 1000)
            .unwrap()
            .last()
            .unwrap()
            .population;
        let decline_final = simulate_logistic_environment(&coupling(), 10.0, &decline, 0.01, 1000)
            .unwrap()
            .last()
            .unwrap()
            .population;
        assert!(decline_final < stable_final);
    }

    #[test]
    fn non_monotonic_waypoints_are_rejected() {
        let result = EnvironmentalTimeline::try_new(vec![
            EnvironmentalWaypoint::try_new(1.0, drivers(293.0, 1.0)).unwrap(),
            EnvironmentalWaypoint::try_new(1.0, drivers(294.0, 1.0)).unwrap(),
        ]);
        assert!(matches!(result, Err(ModelError::NonMonotonicTime { .. })));
    }
}
