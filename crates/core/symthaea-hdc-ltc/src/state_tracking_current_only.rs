// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fail-closed scope gate for the first associative HLS learning experiment.
//!
//! The current associative decoder has a justified algebra for current one-hop
//! queries and current two-hop object-location composition. It deliberately does
//! not yet claim a temporal-addressing algebra for historical queries.
//!
//! These wrappers therefore validate the complete benchmark before delegating to
//! evaluation or training. In particular, `train_current_only_episode` performs
//! no state reset, trace update, recurrent step, loss evaluation, or parameter
//! update unless every query is current-state.

use crate::holographic_liquid::HolographicLiquidCell;
use crate::state_tracking_benchmark::StateTrackingBenchmark;
use crate::state_tracking_codec::StateTrackingCodec;
use crate::state_tracking_exact_training::{
    AssociativeEpisodeMetrics, ExactEpisodeTrainingConfig, ExactEpisodeTrainingError,
    ExactEpisodeTrainingReport, evaluate_associative_episode, train_exact_episode,
};
use std::fmt;

#[derive(Debug)]
pub enum CurrentOnlyTrainingError {
    /// Historical recall is outside the theorem established by the current
    /// associative decoder and must be introduced by a separate temporal model.
    HistoricalQueriesUnsupported { count: usize },
    Training(ExactEpisodeTrainingError),
}

impl fmt::Display for CurrentOnlyTrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HistoricalQueriesUnsupported { count } => write!(
                f,
                "current-only HLS experiment rejects {count} historical queries; an explicit temporal-addressing algebra is required"
            ),
            Self::Training(error) => write!(f, "current-only HLS experiment failed: {error}"),
        }
    }
}

impl std::error::Error for CurrentOnlyTrainingError {}

impl From<ExactEpisodeTrainingError> for CurrentOnlyTrainingError {
    fn from(value: ExactEpisodeTrainingError) -> Self {
        Self::Training(value)
    }
}

/// Validate that a benchmark contains only current-state queries.
///
/// This is intentionally a whole-benchmark preflight rather than a per-query
/// runtime check. A mixed benchmark fails before any recurrent state can move.
pub fn validate_current_only_benchmark(
    benchmark: &StateTrackingBenchmark,
) -> Result<(), CurrentOnlyTrainingError> {
    let historical_count = benchmark
        .queries
        .iter()
        .filter(|query| query.is_historical())
        .count();

    if historical_count == 0 {
        Ok(())
    } else {
        Err(CurrentOnlyTrainingError::HistoricalQueriesUnsupported {
            count: historical_count,
        })
    }
}

/// Evaluate the first associative experiment only on mathematically supported
/// current-state queries.
pub fn evaluate_current_only_episode(
    cell: &HolographicLiquidCell,
    benchmark: &StateTrackingBenchmark,
    codec: &StateTrackingCodec,
    loss_epsilon: f32,
) -> Result<AssociativeEpisodeMetrics, CurrentOnlyTrainingError> {
    validate_current_only_benchmark(benchmark)?;
    Ok(evaluate_associative_episode(
        cell,
        benchmark,
        codec,
        loss_epsilon,
    )?)
}

/// Train one episode only after the entire benchmark passes the current-state
/// scope gate.
pub fn train_current_only_episode(
    cell: &mut HolographicLiquidCell,
    benchmark: &StateTrackingBenchmark,
    codec: &StateTrackingCodec,
    config: &ExactEpisodeTrainingConfig,
) -> Result<ExactEpisodeTrainingReport, CurrentOnlyTrainingError> {
    validate_current_only_benchmark(benchmark)?;
    Ok(train_exact_episode(cell, benchmark, codec, config)?)
}
