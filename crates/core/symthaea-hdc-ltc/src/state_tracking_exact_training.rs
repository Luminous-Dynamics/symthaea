// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Episodic exact-gradient training for diagonal HLS state tracking.
//!
//! Parameters remain fixed for the complete generated world. Exact HLS
//! eligibility traces are advanced with every event, fixed associative query
//! losses are accumulated without changing recurrent state, and a single
//! optimizer update is applied only after the episode ends.
//!
//! This keeps each update interpretable as a gradient of a fixed-parameter
//! episode objective rather than mixing parameter changes into the trace being
//! differentiated.

use crate::continuous_hv::ContinuousHV;
use crate::hls_online_trace::{HlsEligibilityTrace, HlsTraceError, step_with_eligibility};
use crate::holographic_liquid::{HlsError, HlsParameters, HolographicLiquidCell};
use crate::state_tracking_associative::{
    AssociativeReadoutError, associative_query,
};
use crate::state_tracking_benchmark::{
    StateTrackingBenchmark, TrackingAnswer, TrackingScore,
};
use crate::state_tracking_codec::{StateTrackingCodec, TrackingCodecError};
use std::fmt;

#[derive(Debug, Clone)]
pub struct ExactEpisodeTrainingConfig {
    /// Positive optimizer learning rate. Applied as gradient descent.
    pub learning_rate: f32,
    /// Positive global L2 clip for the averaged episode gradient.
    pub gradient_norm_clip: f32,
    /// Positive absolute parameter bound passed to `apply_parameter_delta`.
    pub parameter_abs_bound: f32,
    /// Positive norm regularizer for the associative cosine loss.
    pub loss_epsilon: f32,
}

impl Default for ExactEpisodeTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-2,
            gradient_norm_clip: 1.0,
            parameter_abs_bound: 4.0,
            loss_epsilon: 1e-4,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssociativeEpisodeMetrics {
    pub mean_loss: f32,
    pub score: TrackingScore,
    pub query_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactEpisodeTrainingReport {
    /// Metrics from the episode before its parameter update is applied.
    pub pre_update: AssociativeEpisodeMetrics,
    /// Norm of the query-averaged exact gradient before clipping.
    pub mean_gradient_norm: f32,
    /// Norm after global clipping and before multiplication by learning rate.
    pub clipped_gradient_norm: f32,
    /// Parameter norm after the bounded update.
    pub parameter_norm_after: f32,
}

#[derive(Debug)]
pub enum ExactEpisodeTrainingError {
    InvalidLearningRate,
    InvalidGradientClip,
    InvalidParameterBound,
    InvalidLossEpsilon,
    NoQueries,
    DimensionMismatch { expected: usize, actual: usize },
    Codec(TrackingCodecError),
    Readout(AssociativeReadoutError),
    Trace(HlsTraceError),
    Cell(HlsError),
    Benchmark(String),
    NonFiniteGradient,
}

impl fmt::Display for ExactEpisodeTrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLearningRate => write!(f, "exact episode learning rate must be finite and positive"),
            Self::InvalidGradientClip => write!(f, "exact episode gradient clip must be finite and positive"),
            Self::InvalidParameterBound => write!(f, "exact episode parameter bound must be finite and positive"),
            Self::InvalidLossEpsilon => write!(f, "exact episode loss epsilon must be finite and positive"),
            Self::NoQueries => write!(f, "exact episode requires at least one scored query"),
            Self::DimensionMismatch { expected, actual } => write!(f, "exact episode dimension mismatch: expected {expected}, got {actual}"),
            Self::Codec(error) => write!(f, "exact episode codec error: {error}"),
            Self::Readout(error) => write!(f, "exact episode readout error: {error}"),
            Self::Trace(error) => write!(f, "exact episode trace error: {error}"),
            Self::Cell(error) => write!(f, "exact episode cell error: {error}"),
            Self::Benchmark(error) => write!(f, "exact episode benchmark error: {error}"),
            Self::NonFiniteGradient => write!(f, "exact episode gradient became non-finite"),
        }
    }
}

impl std::error::Error for ExactEpisodeTrainingError {}

impl From<TrackingCodecError> for ExactEpisodeTrainingError {
    fn from(value: TrackingCodecError) -> Self { Self::Codec(value) }
}
impl From<AssociativeReadoutError> for ExactEpisodeTrainingError {
    fn from(value: AssociativeReadoutError) -> Self { Self::Readout(value) }
}
impl From<HlsTraceError> for ExactEpisodeTrainingError {
    fn from(value: HlsTraceError) -> Self { Self::Trace(value) }
}
impl From<HlsError> for ExactEpisodeTrainingError {
    fn from(value: HlsError) -> Self { Self::Cell(value) }
}

/// Evaluate the fixed associative decoder on one complete world without mutating
/// the supplied cell or its parameters.
pub fn evaluate_associative_episode(
    cell: &HolographicLiquidCell,
    benchmark: &StateTrackingBenchmark,
    codec: &StateTrackingCodec,
    loss_epsilon: f32,
) -> Result<AssociativeEpisodeMetrics, ExactEpisodeTrainingError> {
    validate_loss_epsilon(loss_epsilon)?;
    check_dimensions(cell, codec)?;
    let mut eval_cell = cell.clone();
    eval_cell.reset();
    let mut predictions = Vec::with_capacity(benchmark.queries.len());
    let mut total_loss = 0.0_f32;
    let mut query_cursor = 0_usize;
    let mut previous_time = 0.0_f64;

    for (event_index, event) in benchmark.events.iter().enumerate() {
        let dt = event.time - previous_time;
        let event_vector = codec.encode_event(event, dt)?;
        eval_cell.step(dt as f32, &event_vector)?;
        previous_time = event.time;

        while query_cursor < benchmark.queries.len()
            && benchmark.queries[query_cursor].asked_after_event == event_index
        {
            let query = &benchmark.queries[query_cursor];
            let result = associative_query(codec, eval_cell.state(), query, event.time, loss_epsilon)?;
            total_loss += result.loss;
            predictions.push(result.decoded);
            query_cursor += 1;
        }
    }

    episode_metrics(benchmark, predictions, total_loss)
}

/// Accumulate the exact fixed-parameter episode gradient and apply one bounded
/// gradient-descent update after the final query.
pub fn train_exact_episode(
    cell: &mut HolographicLiquidCell,
    benchmark: &StateTrackingBenchmark,
    codec: &StateTrackingCodec,
    config: &ExactEpisodeTrainingConfig,
) -> Result<ExactEpisodeTrainingReport, ExactEpisodeTrainingError> {
    validate_training_config(config)?;
    check_dimensions(cell, codec)?;

    // State and eligibility are episode-local; parameters intentionally persist.
    cell.reset();
    let mut trace = HlsEligibilityTrace::zeros(cell.config().dim);
    let mut accumulated_gradient = HlsParameters::zeros(cell.config().dim);
    let mut predictions = Vec::<TrackingAnswer>::with_capacity(benchmark.queries.len());
    let mut total_loss = 0.0_f32;
    let mut query_cursor = 0_usize;
    let mut previous_time = 0.0_f64;

    for (event_index, event) in benchmark.events.iter().enumerate() {
        let dt = event.time - previous_time;
        let event_vector = codec.encode_event(event, dt)?;
        step_with_eligibility(cell, &mut trace, dt as f32, &event_vector)?;
        previous_time = event.time;

        while query_cursor < benchmark.queries.len()
            && benchmark.queries[query_cursor].asked_after_event == event_index
        {
            let query = &benchmark.queries[query_cursor];
            let result = associative_query(codec, cell.state(), query, event.time, config.loss_epsilon)?;
            let gradient = trace.parameter_gradient(&result.state_learning_signal)?;
            add_parameters_in_place(&mut accumulated_gradient, &gradient);
            total_loss += result.loss;
            predictions.push(result.decoded);
            query_cursor += 1;
        }
    }

    if query_cursor == 0 {
        return Err(ExactEpisodeTrainingError::NoQueries);
    }
    let pre_update = episode_metrics(benchmark, predictions, total_loss)?;

    scale_parameters_in_place(&mut accumulated_gradient, 1.0 / query_cursor as f32);
    let mean_gradient_norm = accumulated_gradient.l2_norm();
    if !mean_gradient_norm.is_finite() {
        return Err(ExactEpisodeTrainingError::NonFiniteGradient);
    }
    let clip_scale = if mean_gradient_norm > config.gradient_norm_clip && mean_gradient_norm > 0.0 {
        config.gradient_norm_clip / mean_gradient_norm
    } else {
        1.0
    };
    scale_parameters_in_place(&mut accumulated_gradient, clip_scale);
    let clipped_gradient_norm = accumulated_gradient.l2_norm();

    // Gradient descent. Parameters were fixed throughout all traced events and
    // queries above, so this update happens only after the exact episode gradient
    // has been fully accumulated.
    cell.apply_parameter_delta(
        &accumulated_gradient,
        -config.learning_rate,
        config.parameter_abs_bound,
    )?;
    let parameter_norm_after = cell.parameters().l2_norm();

    Ok(ExactEpisodeTrainingReport {
        pre_update,
        mean_gradient_norm,
        clipped_gradient_norm,
        parameter_norm_after,
    })
}

fn episode_metrics(
    benchmark: &StateTrackingBenchmark,
    predictions: Vec<TrackingAnswer>,
    total_loss: f32,
) -> Result<AssociativeEpisodeMetrics, ExactEpisodeTrainingError> {
    if predictions.is_empty() {
        return Err(ExactEpisodeTrainingError::NoQueries);
    }
    let score = benchmark
        .score(&predictions)
        .map_err(|error| ExactEpisodeTrainingError::Benchmark(error.to_string()))?;
    Ok(AssociativeEpisodeMetrics {
        mean_loss: total_loss / predictions.len() as f32,
        query_count: predictions.len(),
        score,
    })
}

fn validate_training_config(config: &ExactEpisodeTrainingConfig) -> Result<(), ExactEpisodeTrainingError> {
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(ExactEpisodeTrainingError::InvalidLearningRate);
    }
    if !config.gradient_norm_clip.is_finite() || config.gradient_norm_clip <= 0.0 {
        return Err(ExactEpisodeTrainingError::InvalidGradientClip);
    }
    if !config.parameter_abs_bound.is_finite() || config.parameter_abs_bound <= 0.0 {
        return Err(ExactEpisodeTrainingError::InvalidParameterBound);
    }
    validate_loss_epsilon(config.loss_epsilon)
}

fn validate_loss_epsilon(epsilon: f32) -> Result<(), ExactEpisodeTrainingError> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        Err(ExactEpisodeTrainingError::InvalidLossEpsilon)
    } else {
        Ok(())
    }
}

fn check_dimensions(
    cell: &HolographicLiquidCell,
    codec: &StateTrackingCodec,
) -> Result<(), ExactEpisodeTrainingError> {
    if cell.config().dim == codec.dim() {
        Ok(())
    } else {
        Err(ExactEpisodeTrainingError::DimensionMismatch {
            expected: cell.config().dim,
            actual: codec.dim(),
        })
    }
}

fn add_parameters_in_place(target: &mut HlsParameters, source: &HlsParameters) {
    add_field(&mut target.recurrent_weight, &source.recurrent_weight);
    add_field(&mut target.input_weight, &source.input_weight);
    add_field(&mut target.tau_state_weight, &source.tau_state_weight);
    add_field(&mut target.gate_state_weight, &source.gate_state_weight);
    add_field(&mut target.gate_input_weight, &source.gate_input_weight);
    add_field(&mut target.gate_bias, &source.gate_bias);
}

fn scale_parameters_in_place(parameters: &mut HlsParameters, scale: f32) {
    parameters.recurrent_weight.scale_in_place(scale);
    parameters.input_weight.scale_in_place(scale);
    parameters.tau_state_weight.scale_in_place(scale);
    parameters.gate_state_weight.scale_in_place(scale);
    parameters.gate_input_weight.scale_in_place(scale);
    parameters.gate_bias.scale_in_place(scale);
}

fn add_field(target: &mut ContinuousHV, source: &ContinuousHV) {
    debug_assert_eq!(target.dim(), source.dim());
    for (target_value, source_value) in target.values.iter_mut().zip(source.values.iter()) {
        *target_value += *source_value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous_hv::UnitaryRole;
    use crate::holographic_liquid::HlsConfig;
    use crate::state_tracking_benchmark::StateTrackingBenchmarkConfig;

    fn fixture(seed: u64) -> (StateTrackingBenchmark, StateTrackingCodec) {
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
            entities: 4,
            objects: 6,
            locations: 3,
            events: 48,
            query_every: 4,
            historical_query_rate: 0.5,
            seed,
            ..StateTrackingBenchmarkConfig::default()
        })
        .unwrap();
        let codec = StateTrackingCodec::from_benchmark_config(64, &benchmark.config, 500).unwrap();
        (benchmark, codec)
    }

    fn cell() -> HolographicLiquidCell {
        HolographicLiquidCell::try_new(
            HlsConfig {
                dim: 64,
                state_norm_limit: f32::INFINITY,
                ..HlsConfig::default()
            },
            700,
        )
        .unwrap()
    }

    #[test]
    fn evaluation_does_not_mutate_supplied_cell() {
        let (benchmark, codec) = fixture(1);
        let cell = cell();
        let before_state = cell.state().clone();
        let before_parameters = cell.parameters();
        let metrics = evaluate_associative_episode(&cell, &benchmark, &codec, 1e-4).unwrap();
        assert!(metrics.mean_loss.is_finite());
        assert_eq!(cell.state(), &before_state);
        assert_eq!(cell.parameters(), before_parameters);
    }

    #[test]
    fn exact_episode_update_is_finite_bounded_and_preserves_equivariance() {
        let (benchmark, codec) = fixture(2);
        let mut cell = cell();
        let before = cell.parameters();
        let report = train_exact_episode(
            &mut cell,
            &benchmark,
            &codec,
            &ExactEpisodeTrainingConfig {
                learning_rate: 1e-3,
                gradient_norm_clip: 0.25,
                parameter_abs_bound: 2.0,
                loss_epsilon: 1e-4,
            },
        )
        .unwrap();

        assert!(report.pre_update.mean_loss.is_finite());
        assert!(report.mean_gradient_norm.is_finite());
        assert!(report.clipped_gradient_norm <= 0.25001);
        assert!(report.parameter_norm_after.is_finite());
        assert_ne!(cell.parameters(), before);
        assert!(cell
            .parameters()
            .recurrent_weight
            .values
            .iter()
            .all(|value| value.abs() <= 2.0));

        cell.set_state(ContinuousHV::new_random(64, 900).scale(0.2)).unwrap();
        let input = ContinuousHV::new_random(64, 901).scale(0.2);
        let role = UnitaryRole::new(64, 902);
        let error = cell.binding_equivariance_error(&role, &input, 0.17).unwrap();
        assert!(error <= 1e-6, "post-training equivariance error={error}");
    }

    #[test]
    fn trainer_rejects_nonpositive_learning_rate_before_mutation() {
        let (benchmark, codec) = fixture(3);
        let mut cell = cell();
        let before = cell.parameters();
        let error = train_exact_episode(
            &mut cell,
            &benchmark,
            &codec,
            &ExactEpisodeTrainingConfig {
                learning_rate: 0.0,
                ..ExactEpisodeTrainingConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(error, ExactEpisodeTrainingError::InvalidLearningRate));
        assert_eq!(cell.parameters(), before);
    }
}
