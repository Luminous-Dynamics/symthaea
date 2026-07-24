// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Quantitative oracle summaries for richer or agent-based simulations.
//!
//! These metrics do not assert that a stochastic ecological simulation should
//! exactly follow an ODE. They provide explicit distances from analytic
//! baselines so departures can be measured instead of described informally.

use crate::error::{ModelError, require_non_negative, require_positive};
use crate::{LogisticModel, LotkaVolterra};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorSummary {
    pub count: usize,
    pub mean_absolute_error: f64,
    pub root_mean_square_error: f64,
    pub maximum_absolute_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvariantDriftSummary {
    pub count: usize,
    pub initial_value: f64,
    pub root_mean_square_drift: f64,
    pub maximum_absolute_drift: f64,
}

/// Compare observed populations `(time, population)` with the closed-form
/// logistic trajectory starting from `initial_population`.
pub fn logistic_error_summary(
    model: &LogisticModel,
    initial_population: f64,
    observations: &[(f64, f64)],
) -> Result<ErrorSummary, ModelError> {
    model.validate()?;
    require_positive("initial_population", initial_population)?;
    if observations.is_empty() {
        return Err(ModelError::EmptySeries {
            series: "logistic observations",
        });
    }

    let mut absolute_sum = 0.0;
    let mut squared_sum = 0.0;
    let mut maximum_absolute_error: f64 = 0.0;
    for &(time, observed_population) in observations {
        require_non_negative("observation_time", time)?;
        require_non_negative("observed_population", observed_population)?;
        let predicted = model.population(initial_population, time)?;
        let error = observed_population - predicted;
        let absolute = error.abs();
        absolute_sum += absolute;
        squared_sum += error * error;
        maximum_absolute_error = maximum_absolute_error.max(absolute);
    }
    let count = observations.len();
    Ok(ErrorSummary {
        count,
        mean_absolute_error: absolute_sum / count as f64,
        root_mean_square_error: (squared_sum / count as f64).sqrt(),
        maximum_absolute_error,
    })
}

/// Measure numerical or empirical drift from the classical Lotka-Volterra
/// first integral over `(prey, predator)` samples.
pub fn lotka_volterra_invariant_drift(
    model: &LotkaVolterra,
    trajectory: &[(f64, f64)],
) -> Result<InvariantDriftSummary, ModelError> {
    model.validate()?;
    let Some(&(initial_prey, initial_predator)) = trajectory.first() else {
        return Err(ModelError::EmptySeries {
            series: "predator-prey trajectory",
        });
    };
    let initial_value = model.try_conserved_quantity(initial_prey, initial_predator)?;
    let mut squared_sum = 0.0;
    let mut maximum_absolute_drift: f64 = 0.0;
    for &(prey, predator) in trajectory {
        let value = model.try_conserved_quantity(prey, predator)?;
        let drift = value - initial_value;
        squared_sum += drift * drift;
        maximum_absolute_drift = maximum_absolute_drift.max(drift.abs());
    }
    let count = trajectory.len();
    Ok(InvariantDriftSummary {
        count,
        initial_value,
        root_mean_square_drift: (squared_sum / count as f64).sqrt(),
        maximum_absolute_drift,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_logistic_observations_have_zero_error() {
        let model = LogisticModel::try_new(0.5, 100.0).unwrap();
        let initial = 10.0;
        let observations: Vec<_> = [0.0, 1.0, 2.0, 4.0]
            .into_iter()
            .map(|time| (time, model.population(initial, time).unwrap()))
            .collect();
        let summary = logistic_error_summary(&model, initial, &observations).unwrap();
        assert_eq!(summary.count, observations.len());
        assert!(summary.maximum_absolute_error < 1e-12);
        assert!(summary.root_mean_square_error < 1e-12);
    }

    #[test]
    fn perturbed_logistic_observation_is_quantified() {
        let model = LogisticModel::try_new(0.5, 100.0).unwrap();
        let initial = 10.0;
        let exact = model.population(initial, 1.0).unwrap();
        let summary = logistic_error_summary(&model, initial, &[(1.0, exact + 2.0)]).unwrap();
        assert!((summary.mean_absolute_error - 2.0).abs() < 1e-12);
        assert!((summary.root_mean_square_error - 2.0).abs() < 1e-12);
        assert!((summary.maximum_absolute_error - 2.0).abs() < 1e-12);
    }

    #[test]
    fn lv_rk4_invariant_drift_is_small() {
        let model = LotkaVolterra::try_new(1.0, 0.1, 0.075, 1.5).unwrap();
        let trajectory = model.try_simulate(10.0, 5.0, 0.001, 20_000).unwrap();
        let summary = lotka_volterra_invariant_drift(&model, &trajectory).unwrap();
        assert_eq!(summary.count, trajectory.len());
        assert!(summary.maximum_absolute_drift < 1e-3);
    }

    #[test]
    fn empty_series_fail_closed() {
        let logistic = LogisticModel::try_new(0.5, 100.0).unwrap();
        assert!(logistic_error_summary(&logistic, 10.0, &[]).is_err());
        let lv = LotkaVolterra::try_new(1.0, 0.1, 0.075, 1.5).unwrap();
        assert!(lotka_volterra_invariant_drift(&lv, &[]).is_err());
    }
}
