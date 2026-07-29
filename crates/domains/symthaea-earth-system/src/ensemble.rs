// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic parameter ensembles for reduced-order climate experiments.
//!
//! These are Cartesian sweeps, not probabilistic Monte Carlo samples. Input
//! ordering is preserved, every member is explicit, and no distributional
//! meaning is attached unless the caller supplies one externally.

use crate::error::{ModelError, require_finite, require_positive};
use crate::transient::OneBoxClimateModel;

pub const MAX_ENSEMBLE_MEMBERS: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneBoxEnsembleCase {
    pub heat_capacity: f64,
    pub feedback: f64,
    pub forcing: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneBoxEnsembleOutcome {
    pub case: OneBoxEnsembleCase,
    pub equilibrium_temperature: f64,
    pub response_time_seconds: f64,
    pub horizon_temperature: f64,
    pub horizon_warming: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnsembleSummary {
    pub count: usize,
    pub minimum: f64,
    pub median: f64,
    pub mean: f64,
    pub maximum: f64,
}

pub fn run_one_box_ensemble(
    baseline_temperature: f64,
    initial_temperature: f64,
    horizon_seconds: f64,
    heat_capacities: &[f64],
    feedbacks: &[f64],
    forcings: &[f64],
) -> Result<Vec<OneBoxEnsembleOutcome>, ModelError> {
    require_positive("baseline_temperature", baseline_temperature)?;
    require_positive("initial_temperature", initial_temperature)?;
    require_finite("horizon_seconds", horizon_seconds)?;
    if horizon_seconds < 0.0 {
        return Err(ModelError::OutOfRange {
            parameter: "horizon_seconds",
            value: horizon_seconds,
            min: 0.0,
            max: f64::INFINITY,
        });
    }
    require_non_empty("heat_capacities", heat_capacities)?;
    require_non_empty("feedbacks", feedbacks)?;
    require_non_empty("forcings", forcings)?;
    let requested = heat_capacities
        .len()
        .checked_mul(feedbacks.len())
        .and_then(|count| count.checked_mul(forcings.len()))
        .ok_or(ModelError::EnsembleTooLarge {
            requested: usize::MAX,
            maximum: MAX_ENSEMBLE_MEMBERS,
        })?;
    if requested > MAX_ENSEMBLE_MEMBERS {
        return Err(ModelError::EnsembleTooLarge {
            requested,
            maximum: MAX_ENSEMBLE_MEMBERS,
        });
    }

    let mut outcomes = Vec::with_capacity(requested);
    for &heat_capacity in heat_capacities {
        for &feedback in feedbacks {
            let model = OneBoxClimateModel::try_new(heat_capacity, feedback, baseline_temperature)?;
            for &forcing in forcings {
                require_finite("forcing", forcing)?;
                let horizon_temperature =
                    model.exact_constant_forcing(initial_temperature, forcing, horizon_seconds)?;
                outcomes.push(OneBoxEnsembleOutcome {
                    case: OneBoxEnsembleCase {
                        heat_capacity,
                        feedback,
                        forcing,
                    },
                    equilibrium_temperature: model.equilibrium_temperature(forcing),
                    response_time_seconds: model.response_time(),
                    horizon_temperature,
                    horizon_warming: horizon_temperature - baseline_temperature,
                });
            }
        }
    }
    Ok(outcomes)
}

pub fn summarize_horizon_warming(
    outcomes: &[OneBoxEnsembleOutcome],
) -> Result<EnsembleSummary, ModelError> {
    if outcomes.is_empty() {
        return Err(ModelError::EmptySeries {
            series: "one-box ensemble outcomes",
        });
    }
    let mut values: Vec<f64> = outcomes
        .iter()
        .map(|outcome| outcome.horizon_warming)
        .collect();
    if let Some(value) = values.iter().find(|value| !value.is_finite()) {
        return Err(ModelError::NonFinite {
            parameter: "horizon_warming",
            value: *value,
        });
    }
    values.sort_by(f64::total_cmp);
    let count = values.len();
    let median = if count.is_multiple_of(2) {
        0.5 * (values[count / 2 - 1] + values[count / 2])
    } else {
        values[count / 2]
    };
    Ok(EnsembleSummary {
        count,
        minimum: values[0],
        median,
        mean: values.iter().sum::<f64>() / count as f64,
        maximum: values[count - 1],
    })
}

fn require_non_empty(parameter: &'static str, values: &[f64]) -> Result<(), ModelError> {
    if values.is_empty() {
        return Err(ModelError::EmptySeries { series: parameter });
    }
    for &value in values {
        require_finite(parameter, value)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transient::SECONDS_PER_YEAR;

    #[test]
    fn cartesian_order_is_deterministic() {
        let outcomes = run_one_box_ensemble(
            288.0,
            288.0,
            10.0 * SECONDS_PER_YEAR,
            &[2.0e8, 4.0e8],
            &[1.0, 2.0],
            &[2.0, 4.0],
        )
        .unwrap();
        assert_eq!(outcomes.len(), 8);
        assert_eq!(outcomes[0].case.heat_capacity, 2.0e8);
        assert_eq!(outcomes[0].case.feedback, 1.0);
        assert_eq!(outcomes[0].case.forcing, 2.0);
        assert_eq!(outcomes[1].case.forcing, 4.0);
        assert_eq!(outcomes[4].case.heat_capacity, 4.0e8);
    }

    #[test]
    fn lower_feedback_has_greater_long_horizon_warming() {
        let outcomes = run_one_box_ensemble(
            288.0,
            288.0,
            1000.0 * SECONDS_PER_YEAR,
            &[4.0e8],
            &[1.0, 2.0],
            &[4.0],
        )
        .unwrap();
        assert!(outcomes[0].horizon_warming > outcomes[1].horizon_warming);
    }

    #[test]
    fn summary_is_order_independent() {
        let mut outcomes = run_one_box_ensemble(
            288.0,
            288.0,
            100.0 * SECONDS_PER_YEAR,
            &[4.0e8],
            &[1.0, 2.0],
            &[2.0, 4.0],
        )
        .unwrap();
        let first = summarize_horizon_warming(&outcomes).unwrap();
        outcomes.reverse();
        let second = summarize_horizon_warming(&outcomes).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.count, 4);
        assert!(first.minimum <= first.median && first.median <= first.maximum);
    }

    #[test]
    fn oversized_ensemble_is_rejected_before_allocation() {
        let heat = vec![1.0; 1001];
        let feedback = vec![1.0; 101];
        assert!(matches!(
            run_one_box_ensemble(288.0, 288.0, 1.0, &heat, &feedback, &[1.0]),
            Err(ModelError::EnsembleTooLarge { .. })
        ));
    }
}
