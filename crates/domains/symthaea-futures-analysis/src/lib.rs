// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-futures-analysis
//!
//! Rare-event / branch-diversity / model-disagreement analysis for the Symthaea Futures
//! Laboratory. Scoped per the Phase 2 addendum
//! (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`, 2026-07-27, item 2): rather than a new
//! scenario simulation, this crate's first real content answers "does inter-model disagreement
//! at a checkpoint predict where the best individual model is more likely to be wrong?" against
//! data Phase 1 already knows how to produce (6 rungs x 2 scenario families).
//!
//! Deliberately minimal for now — a single, well-verified disagreement metric for the boolean
//! outcome space every Phase 1 rung so far has used, not a general framework for outcome spaces
//! this plan hasn't built a second consumer for yet.

use symthaea_futures_core::{ForecastDistribution, OutcomeRegion};

/// Extracts the probability mass a forecast assigns to `OutcomeRegion::Boolean(true)` — `0.0` if
/// no such branch exists. Centralizes logic that was previously hand-duplicated across several
/// `symthaea-futures-ensemble` examples (e.g. `horizon_decay_sweep.rs`'s `p_true`).
pub fn boolean_p_true(forecast: &ForecastDistribution) -> f64 {
    forecast
        .branches
        .iter()
        .find(|b| b.outcome == OutcomeRegion::Boolean(true))
        .map(|b| b.probability)
        .unwrap_or(0.0)
}

/// Population variance of [`boolean_p_true`] across `forecasts` — the disagreement metric the
/// Phase 2 addendum proposes checking against real backtest data. `None` if fewer than 2
/// forecasts (there's nothing to disagree about with 0 or 1 opinions).
pub fn boolean_disagreement_variance(forecasts: &[ForecastDistribution]) -> Option<f64> {
    if forecasts.len() < 2 {
        return None;
    }
    let ps: Vec<f64> = forecasts.iter().map(boolean_p_true).collect();
    let n = ps.len() as f64;
    let mean = ps.iter().sum::<f64>() / n;
    let variance = ps.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n;
    Some(variance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_futures_core::{ForecastBranch, Horizon, OutcomeSpaceId};

    fn boolean_forecast(p_true: f64) -> ForecastDistribution {
        ForecastDistribution {
            issued_at_tick: 0,
            horizon: Horizon(0),
            outcome_space: OutcomeSpaceId("test".to_string()),
            branches: vec![
                ForecastBranch {
                    probability: p_true,
                    outcome: OutcomeRegion::Boolean(true),
                    assumptions: Vec::new(),
                },
                ForecastBranch {
                    probability: 1.0 - p_true,
                    outcome: OutcomeRegion::Boolean(false),
                    assumptions: Vec::new(),
                },
            ],
            unsupported_mass: 0.0,
        }
    }

    #[test]
    fn boolean_p_true_extracts_the_true_branch() {
        assert_eq!(boolean_p_true(&boolean_forecast(0.73)), 0.73);
    }

    #[test]
    fn boolean_p_true_defaults_to_zero_with_no_true_branch() {
        let dist = ForecastDistribution {
            issued_at_tick: 0,
            horizon: Horizon(0),
            outcome_space: OutcomeSpaceId("test".to_string()),
            branches: vec![ForecastBranch {
                probability: 1.0,
                outcome: OutcomeRegion::Boolean(false),
                assumptions: Vec::new(),
            }],
            unsupported_mass: 0.0,
        };
        assert_eq!(boolean_p_true(&dist), 0.0);
    }

    #[test]
    fn disagreement_variance_none_with_fewer_than_two_forecasts() {
        assert_eq!(boolean_disagreement_variance(&[]), None);
        assert_eq!(
            boolean_disagreement_variance(&[boolean_forecast(0.5)]),
            None
        );
    }

    #[test]
    fn disagreement_variance_zero_on_perfect_agreement() {
        // Not assert_eq! against exactly 0.0 -- (0.4+0.4+0.4)/3 doesn't round-trip to bit-
        // identical 0.4 in f64, so the "identical inputs" variance is ~1e-33, not exactly 0.
        let forecasts = vec![
            boolean_forecast(0.4),
            boolean_forecast(0.4),
            boolean_forecast(0.4),
        ];
        let variance = boolean_disagreement_variance(&forecasts).unwrap();
        assert!(variance.abs() < 1e-15, "got {variance}");
    }

    #[test]
    fn disagreement_variance_matches_hand_computed_value() {
        // p = [0.2, 0.4, 0.6, 0.8], mean = 0.5
        // variance = ((-0.3)^2 + (-0.1)^2 + (0.1)^2 + (0.3)^2) / 4 = 0.2 / 4 = 0.05
        let forecasts = vec![
            boolean_forecast(0.2),
            boolean_forecast(0.4),
            boolean_forecast(0.6),
            boolean_forecast(0.8),
        ];
        let variance = boolean_disagreement_variance(&forecasts).unwrap();
        assert!((variance - 0.05).abs() < 1e-9, "got {variance}");
    }
}
