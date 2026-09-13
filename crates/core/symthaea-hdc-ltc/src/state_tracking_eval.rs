// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Frozen-reservoir evaluation harness for the compositional state-tracking task.
//!
//! Recurrent parameters are frozen. A shared nearest-centroid readout is trained
//! on one generated world and evaluated on a disjoint seed. Every result also
//! includes a query-only control trained on the same labels; this makes identity
//! or dataset leakage visible instead of attributing it to recurrent memory.

use crate::contextual_holographic_liquid::{ContextualHlsError, ContextualHolographicLiquidCell};
use crate::continuous_hv::ContinuousHV;
use crate::holographic_liquid::{HlsError, HolographicLiquidCell};
use crate::neuron::HdcLtcUnifiedNeuron;
use crate::state_tracking_benchmark::{StateTrackingBenchmark, TrackingAnswer, TrackingQuery, TrackingScore};
use crate::state_tracking_codec::{StateTrackingCodec, TrackingCodecError};
use crate::state_tracking_readout::{TrackingPrototypeReadout, TrackingReadoutError};
use std::convert::Infallible;
use std::fmt;

/// Minimal recurrent interface required by the diagnostic harness.
pub trait FrozenTrackingReservoir: Clone {
    type Error: fmt::Display;

    fn tracking_dim(&self) -> usize;
    fn tracking_state(&self) -> &ContinuousHV;
    fn tracking_reset(&mut self);
    fn tracking_step(&mut self, dt: f32, input: &ContinuousHV) -> Result<(), Self::Error>;
}

impl FrozenTrackingReservoir for HolographicLiquidCell {
    type Error = HlsError;

    fn tracking_dim(&self) -> usize {
        self.config().dim
    }
    fn tracking_state(&self) -> &ContinuousHV {
        self.state()
    }
    fn tracking_reset(&mut self) {
        self.reset();
    }
    fn tracking_step(&mut self, dt: f32, input: &ContinuousHV) -> Result<(), Self::Error> {
        self.step(dt, input)
    }
}

impl FrozenTrackingReservoir for ContextualHolographicLiquidCell {
    type Error = ContextualHlsError;

    fn tracking_dim(&self) -> usize {
        self.config().dim
    }
    fn tracking_state(&self) -> &ContinuousHV {
        self.state()
    }
    fn tracking_reset(&mut self) {
        self.reset();
    }
    fn tracking_step(&mut self, dt: f32, input: &ContinuousHV) -> Result<(), Self::Error> {
        self.step(dt, input)
    }
}

impl FrozenTrackingReservoir for HdcLtcUnifiedNeuron {
    type Error = Infallible;

    fn tracking_dim(&self) -> usize {
        self.config().dim
    }
    fn tracking_state(&self) -> &ContinuousHV {
        self.state()
    }
    fn tracking_reset(&mut self) {
        self.reset();
    }
    fn tracking_step(&mut self, dt: f32, input: &ContinuousHV) -> Result<(), Self::Error> {
        self.evolve_closed_form(dt, input);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FrozenTrackingEvalConfig {
    /// Seed for the shared event/query codebook. It is independent of both
    /// benchmark-world seeds and recurrent initialization.
    pub codec_seed: u64,
}

impl Default for FrozenTrackingEvalConfig {
    fn default() -> Self {
        Self { codec_seed: 9001 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrozenTrackingEvalResult {
    /// Accuracy using recurrent state concatenated with the query encoding.
    pub score: TrackingScore,
    /// Control accuracy using only query encoding and no recurrent state.
    pub query_only_score: TrackingScore,
    pub train_queries: usize,
    pub test_queries: usize,
    pub observed_entity_classes: usize,
    pub observed_location_classes: usize,
    pub state_dimension: usize,
}

impl FrozenTrackingEvalResult {
    pub fn accuracy_gain_over_query_only(&self) -> f64 {
        self.score.accuracy() - self.query_only_score.accuracy()
    }

    pub fn compositional_gain_over_query_only(&self) -> f64 {
        self.score.compositional_accuracy() - self.query_only_score.compositional_accuracy()
    }

    pub fn historical_gain_over_query_only(&self) -> f64 {
        self.score.historical_accuracy() - self.query_only_score.historical_accuracy()
    }
}

#[derive(Debug)]
pub enum FrozenTrackingEvalError {
    BenchmarkCardinalityMismatch,
    IdenticalBenchmarkSeed(u64),
    Codec(TrackingCodecError),
    Readout(TrackingReadoutError),
    Benchmark(String),
    Reservoir(String),
}

impl fmt::Display for FrozenTrackingEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BenchmarkCardinalityMismatch => write!(
                f,
                "train/test benchmarks must have identical entity/object/location cardinalities"
            ),
            Self::IdenticalBenchmarkSeed(seed) => write!(
                f,
                "train/test benchmark seeds must be disjoint; both were {seed}"
            ),
            Self::Codec(error) => write!(f, "tracking codec error: {error}"),
            Self::Readout(error) => write!(f, "tracking readout error: {error}"),
            Self::Benchmark(error) => write!(f, "tracking benchmark error: {error}"),
            Self::Reservoir(error) => write!(f, "tracking reservoir error: {error}"),
        }
    }
}

impl std::error::Error for FrozenTrackingEvalError {}

impl From<TrackingCodecError> for FrozenTrackingEvalError {
    fn from(value: TrackingCodecError) -> Self {
        Self::Codec(value)
    }
}

impl From<TrackingReadoutError> for FrozenTrackingEvalError {
    fn from(value: TrackingReadoutError) -> Self {
        Self::Readout(value)
    }
}

/// Train shared diagnostic readouts on `train`, reset the frozen reservoir, and
/// evaluate both reservoir+query and query-only controls on `test`.
pub fn evaluate_frozen_reservoir<R>(
    reservoir: &R,
    train: &StateTrackingBenchmark,
    test: &StateTrackingBenchmark,
    config: FrozenTrackingEvalConfig,
) -> Result<FrozenTrackingEvalResult, FrozenTrackingEvalError>
where
    R: FrozenTrackingReservoir,
{
    ensure_compatible_benchmarks(train, test)?;
    let state_dim = reservoir.tracking_dim();
    let codec = StateTrackingCodec::from_benchmark_config(
        state_dim,
        &train.config,
        config.codec_seed,
    )?;

    let mut reservoir_readout =
        TrackingPrototypeReadout::from_benchmark_config(state_dim * 2, &train.config)?;
    let mut query_only_readout =
        TrackingPrototypeReadout::from_benchmark_config(state_dim, &train.config)?;

    let mut train_reservoir = reservoir.clone();
    train_reservoir.tracking_reset();
    for_each_query_context(
        &mut train_reservoir,
        train,
        &codec,
        |state, query_vector, query| {
            let feature = concatenate(state, query_vector);
            reservoir_readout.observe(&feature, query.expected)?;
            query_only_readout.observe(query_vector, query.expected)?;
            Ok(())
        },
    )?;

    let mut test_reservoir = reservoir.clone();
    test_reservoir.tracking_reset();
    let mut predictions = Vec::<TrackingAnswer>::with_capacity(test.queries.len());
    let mut query_only_predictions = Vec::<TrackingAnswer>::with_capacity(test.queries.len());

    for_each_query_context(
        &mut test_reservoir,
        test,
        &codec,
        |state, query_vector, query| {
            let feature = concatenate(state, query_vector);
            predictions.push(reservoir_readout.predict(&feature, query.kind)?);
            query_only_predictions.push(query_only_readout.predict(query_vector, query.kind)?);
            Ok(())
        },
    )?;

    let score = test
        .score(&predictions)
        .map_err(|error| FrozenTrackingEvalError::Benchmark(error.to_string()))?;
    let query_only_score = test
        .score(&query_only_predictions)
        .map_err(|error| FrozenTrackingEvalError::Benchmark(error.to_string()))?;

    Ok(FrozenTrackingEvalResult {
        score,
        query_only_score,
        train_queries: train.queries.len(),
        test_queries: test.queries.len(),
        observed_entity_classes: reservoir_readout.observed_entity_classes(),
        observed_location_classes: reservoir_readout.observed_location_classes(),
        state_dimension: state_dim,
    })
}

fn for_each_query_context<R, F>(
    reservoir: &mut R,
    benchmark: &StateTrackingBenchmark,
    codec: &StateTrackingCodec,
    mut on_query: F,
) -> Result<(), FrozenTrackingEvalError>
where
    R: FrozenTrackingReservoir,
    F: FnMut(&ContinuousHV, &ContinuousHV, &TrackingQuery) -> Result<(), FrozenTrackingEvalError>,
{
    let mut previous_time = 0.0_f64;
    let mut query_cursor = 0_usize;

    for (event_index, event) in benchmark.events.iter().enumerate() {
        let dt = event.time - previous_time;
        let event_vector = codec.encode_event(event, dt)?;
        reservoir
            .tracking_step(dt as f32, &event_vector)
            .map_err(|error| FrozenTrackingEvalError::Reservoir(error.to_string()))?;
        previous_time = event.time;

        while query_cursor < benchmark.queries.len()
            && benchmark.queries[query_cursor].asked_after_event == event_index
        {
            let query = &benchmark.queries[query_cursor];
            let query_vector = codec.encode_query(query, event.time)?;
            on_query(reservoir.tracking_state(), &query_vector, query)?;
            query_cursor += 1;
        }
    }

    Ok(())
}

fn concatenate(state: &ContinuousHV, query: &ContinuousHV) -> ContinuousHV {
    assert_eq!(state.dim(), query.dim(), "tracking feature dimension mismatch");
    let mut values = Vec::with_capacity(state.dim() + query.dim());
    values.extend_from_slice(&state.values);
    values.extend_from_slice(&query.values);
    ContinuousHV::from_values(values)
}

fn ensure_compatible_benchmarks(
    train: &StateTrackingBenchmark,
    test: &StateTrackingBenchmark,
) -> Result<(), FrozenTrackingEvalError> {
    if train.config.seed == test.config.seed {
        return Err(FrozenTrackingEvalError::IdenticalBenchmarkSeed(
            train.config.seed,
        ));
    }
    if train.config.entities != test.config.entities
        || train.config.objects != test.config.objects
        || train.config.locations != test.config.locations
    {
        return Err(FrozenTrackingEvalError::BenchmarkCardinalityMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holographic_liquid::{HlsActivation, HlsConfig};
    use crate::state_tracking_benchmark::StateTrackingBenchmarkConfig;

    fn benchmark(seed: u64) -> StateTrackingBenchmark {
        StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
            entities: 4,
            objects: 8,
            locations: 3,
            events: 96,
            query_every: 4,
            historical_query_rate: 0.5,
            seed,
            ..StateTrackingBenchmarkConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn rejects_identical_world_seed() {
        let train = benchmark(1);
        let test = benchmark(1);
        let reservoir = HolographicLiquidCell::try_new(
            HlsConfig {
                dim: 64,
                ..HlsConfig::default()
            },
            7,
        )
        .unwrap();
        assert!(matches!(
            evaluate_frozen_reservoir(
                &reservoir,
                &train,
                &test,
                FrozenTrackingEvalConfig::default()
            ),
            Err(FrozenTrackingEvalError::IdenticalBenchmarkSeed(1))
        ));
    }

    #[test]
    fn diagonal_hls_reports_model_and_query_only_controls() {
        let train = benchmark(1);
        let test = benchmark(2);
        let reservoir = HolographicLiquidCell::try_new(
            HlsConfig {
                dim: 128,
                activation: HlsActivation::Tanh,
                ..HlsConfig::default()
            },
            77,
        )
        .unwrap();
        let result = evaluate_frozen_reservoir(
            &reservoir,
            &train,
            &test,
            FrozenTrackingEvalConfig::default(),
        )
        .unwrap();
        assert_eq!(result.score.total, test.queries.len());
        assert_eq!(result.query_only_score.total, test.queries.len());
        assert!(result.score.accuracy().is_finite());
        assert!(result.query_only_score.accuracy().is_finite());
    }

    #[test]
    fn contextual_and_legacy_cells_share_identical_harness() {
        let train = benchmark(3);
        let test = benchmark(4);

        let contextual = ContextualHolographicLiquidCell::try_new(
            HlsConfig {
                dim: 128,
                ..HlsConfig::default()
            },
            vec![1, 7, 31],
            0.75,
            88,
        )
        .unwrap();
        let contextual_result = evaluate_frozen_reservoir(
            &contextual,
            &train,
            &test,
            FrozenTrackingEvalConfig::default(),
        )
        .unwrap();
        assert_eq!(contextual_result.score.total, test.queries.len());

        let legacy = HdcLtcUnifiedNeuron::new(
            crate::config::NeuronConfig {
                dim: 128,
                ..crate::config::NeuronConfig::default()
            },
            99,
        );
        let legacy_result = evaluate_frozen_reservoir(
            &legacy,
            &train,
            &test,
            FrozenTrackingEvalConfig::default(),
        )
        .unwrap();
        assert_eq!(legacy_result.score.total, test.queries.len());
    }
}
