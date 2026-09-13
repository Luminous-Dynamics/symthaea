// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Frozen-reservoir evaluation harness for the compositional state-tracking task.
//!
//! This is an intentionally conservative first diagnostic. Recurrent parameters
//! are frozen; only a shared nearest-centroid readout observes labeled training
//! queries. Train and test event streams are disjoint benchmark instances.

use crate::contextual_holographic_liquid::{ContextualHlsError, ContextualHolographicLiquidCell};
use crate::continuous_hv::ContinuousHV;
use crate::holographic_liquid::{HlsError, HolographicLiquidCell};
use crate::neuron::HdcLtcUnifiedNeuron;
use crate::state_tracking_benchmark::{
    StateTrackingBenchmark, TrackingAnswer, TrackingScore,
};
use crate::state_tracking_codec::{StateTrackingCodec, TrackingCodecError};
use crate::state_tracking_readout::{TrackingPrototypeReadout, TrackingReadoutError};
use std::convert::Infallible;
use std::fmt;

/// Minimal recurrent interface required by the benchmark harness.
pub trait FrozenTrackingReservoir: Clone {
    type Error: fmt::Display;

    fn tracking_dim(&self) -> usize;
    fn tracking_state(&self) -> &ContinuousHV;
    fn tracking_reset(&mut self);
    fn tracking_step(
        &mut self,
        dt: f32,
        input: &ContinuousHV,
    ) -> Result<(), Self::Error>;
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

    fn tracking_step(
        &mut self,
        dt: f32,
        input: &ContinuousHV,
    ) -> Result<(), Self::Error> {
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

    fn tracking_step(
        &mut self,
        dt: f32,
        input: &ContinuousHV,
    ) -> Result<(), Self::Error> {
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

    fn tracking_step(
        &mut self,
        dt: f32,
        input: &ContinuousHV,
    ) -> Result<(), Self::Error> {
        self.evolve_closed_form(dt, input);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FrozenTrackingEvalConfig {
    /// Seed for the shared event/query HDC codebook.
    pub codec_seed: u64,
}

impl Default for FrozenTrackingEvalConfig {
    fn default() -> Self {
        Self { codec_seed: 9001 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrozenTrackingEvalResult {
    pub score: TrackingScore,
    pub train_queries: usize,
    pub test_queries: usize,
    pub observed_entity_classes: usize,
    pub observed_location_classes: usize,
    /// Recurrent state dimension, excluding the diagnostic readout feature.
    pub state_dimension: usize,
}

#[derive(Debug)]
pub enum FrozenTrackingEvalError {
    BenchmarkCardinalityMismatch,
    Codec(TrackingCodecError),
    Readout(TrackingReadoutError),
    Reservoir(String),
}

impl fmt::Display for FrozenTrackingEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BenchmarkCardinalityMismatch => write!(
                f,
                "train/test benchmarks must have identical entity/object/location cardinalities"
            ),
            Self::Codec(error) => write!(f, "tracking codec error: {error}"),
            Self::Readout(error) => write!(f, "tracking readout error: {error}"),
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

/// Train only the shared prototype readout on one benchmark world and evaluate
/// a reset copy of the same frozen reservoir on a disjoint benchmark world.
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

    // The feature is a transparent concatenation of recurrent state and query
    // encoding rather than an architecture-specific learned query head.
    let feature_dim = state_dim * 2;
    let mut readout = TrackingPrototypeReadout::from_benchmark_config(feature_dim, &train.config)?;

    let mut train_reservoir = reservoir.clone();
    train_reservoir.tracking_reset();
    for_each_query_feature(&mut train_reservoir, train, &codec, |feature, query| {
        readout.observe(&feature, query.expected)?;
        Ok::<(), FrozenTrackingEvalError>(())
    })?;

    let mut test_reservoir = reservoir.clone();
    test_reservoir.tracking_reset();
    let mut predictions = Vec::<TrackingAnswer>::with_capacity(test.queries.len());
    for_each_query_feature(&mut test_reservoir, test, &codec, |feature, query| {
        predictions.push(readout.predict(&feature, query.kind)?);
        Ok::<(), FrozenTrackingEvalError>(())
    })?;

    let score = test
        .score(&predictions)
        .map_err(|error| FrozenTrackingEvalError::Reservoir(error.to_string()))?;

    Ok(FrozenTrackingEvalResult {
        score,
        train_queries: train.queries.len(),
        test_queries: test.queries.len(),
        observed_entity_classes: readout.observed_entity_classes(),
        observed_location_classes: readout.observed_location_classes(),
        state_dimension: state_dim,
    })
}

fn for_each_query_feature<R, F>(
    reservoir: &mut R,
    benchmark: &StateTrackingBenchmark,
    codec: &StateTrackingCodec,
    mut on_query: F,
) -> Result<(), FrozenTrackingEvalError>
where
    R: FrozenTrackingReservoir,
    F: FnMut(ContinuousHV, &crate::state_tracking_benchmark::TrackingQuery) -> Result<(), FrozenTrackingEvalError>,
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
            let feature = concatenate(reservoir.tracking_state(), &query_vector);
            on_query(feature, query)?;
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
    fn diagonal_hls_evaluation_executes_without_label_leakage() {
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

        assert_eq!(result.train_queries, train.queries.len());
        assert_eq!(result.test_queries, test.queries.len());
        assert_eq!(result.score.total, test.queries.len());
        assert!(result.score.accuracy().is_finite());
        assert!((0.0..=1.0).contains(&result.score.accuracy()));
    }

    #[test]
    fn contextual_hls_uses_same_evaluation_contract() {
        let train = benchmark(3);
        let test = benchmark(4);
        let reservoir = ContextualHolographicLiquidCell::try_new(
            HlsConfig {
                dim: 128,
                ..HlsConfig::default()
            },
            vec![1, 7, 31],
            0.75,
            88,
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
    }

    #[test]
    fn legacy_hdc_ltc_uses_same_evaluation_contract() {
        let train = benchmark(5);
        let test = benchmark(6);
        let reservoir = HdcLtcUnifiedNeuron::new(
            crate::config::NeuronConfig {
                dim: 128,
                ..crate::config::NeuronConfig::default()
            },
            99,
        );

        let result = evaluate_frozen_reservoir(
            &reservoir,
            &train,
            &test,
            FrozenTrackingEvalConfig::default(),
        )
        .unwrap();
        assert_eq!(result.score.total, test.queries.len());
    }
}
