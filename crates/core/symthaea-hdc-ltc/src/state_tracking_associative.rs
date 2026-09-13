// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fixed differentiable HDC associative readout for state tracking.
//!
//! The decoder has **no trainable parameters**. A deterministic unitary query
//! key `k(q)` unbinds the recurrent state:
//!
//! `prediction = k(q) ⊙ h`.
//!
//! The prediction is compared with the exact answer hypervector using a smooth
//! cosine loss. Since bipolar binding is an orthogonal diagonal transform, the
//! gradient with respect to recurrent state is just the same role applied to the
//! prediction-space gradient. This gives HLS eligibility traces an explicit
//! `dL/dh` without introducing a learned decoder that could hide weak memory.

use crate::continuous_hv::ContinuousHV;
use crate::state_tracking_benchmark::{TrackingAnswer, TrackingQuery};
use crate::state_tracking_codec::{StateTrackingCodec, TrackingCodecError};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum AssociativeReadoutError {
    Codec(TrackingCodecError),
    DimensionMismatch { expected: usize, actual: usize },
    InvalidEpsilon,
    NonFiniteState,
    NonFiniteLoss,
}

impl fmt::Display for AssociativeReadoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(error) => write!(f, "associative readout codec error: {error}"),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "associative readout dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidEpsilon => write!(f, "associative readout epsilon must be finite and positive"),
            Self::NonFiniteState => write!(f, "associative readout state contains a non-finite value"),
            Self::NonFiniteLoss => write!(f, "associative readout produced a non-finite loss or gradient"),
        }
    }
}

impl std::error::Error for AssociativeReadoutError {}

impl From<TrackingCodecError> for AssociativeReadoutError {
    fn from(value: TrackingCodecError) -> Self {
        Self::Codec(value)
    }
}

/// Complete fixed-decoder result for one query.
#[derive(Debug, Clone, PartialEq)]
pub struct AssociativeQueryResult {
    /// HDC vector obtained by unitary unbinding of recurrent state.
    pub prediction_vector: ContinuousHV,
    /// Nearest answer symbol under the benchmark codebook.
    pub decoded: TrackingAnswer,
    /// Regularized cosine similarity between prediction and target answer.
    pub cosine_similarity: f32,
    /// `1 - cosine_similarity`.
    pub loss: f32,
    /// Exact derivative `dL/dh` for the current recurrent state.
    pub state_learning_signal: ContinuousHV,
}

/// Query a recurrent state through a deterministic HDC associative decoder.
///
/// `epsilon` smooths the vector norms as
/// `sqrt(||v||^2 + epsilon^2)` so the loss and derivative remain finite even for
/// an all-zero recurrent state.
pub fn associative_query(
    codec: &StateTrackingCodec,
    state: &ContinuousHV,
    query: &TrackingQuery,
    asked_time: f64,
    epsilon: f32,
) -> Result<AssociativeQueryResult, AssociativeReadoutError> {
    if state.dim() != codec.dim() {
        return Err(AssociativeReadoutError::DimensionMismatch {
            expected: codec.dim(),
            actual: state.dim(),
        });
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(AssociativeReadoutError::InvalidEpsilon);
    }
    if state.values.iter().any(|value| !value.is_finite()) {
        return Err(AssociativeReadoutError::NonFiniteState);
    }

    let key = codec.query_key(query, asked_time)?;
    let prediction_vector = key.bind(state);
    let target = codec.answer_symbol(query.expected)?;

    let epsilon_sq = epsilon * epsilon;
    let prediction_sq = prediction_vector.dot(&prediction_vector);
    let target_sq = target.dot(target);
    let prediction_norm = (prediction_sq + epsilon_sq).sqrt();
    let target_norm = (target_sq + epsilon_sq).sqrt();
    let dot = prediction_vector.dot(target);
    let norm_product = prediction_norm * target_norm;
    let cosine_similarity = dot / norm_product;
    let loss = 1.0 - cosine_similarity;

    // d(1-cos(y,t))/dy = -t/(||y|| ||t||)
    //                       + (y·t)y/(||y||^3 ||t||)
    // where the y norm is epsilon-regularized above.
    let first_scale = -1.0 / norm_product;
    let second_scale = dot / (prediction_norm * prediction_norm * prediction_norm * target_norm);
    let prediction_gradient = ContinuousHV::from_values(
        target
            .values
            .iter()
            .zip(prediction_vector.values.iter())
            .map(|(&target_value, &prediction_value)| {
                first_scale * target_value + second_scale * prediction_value
            })
            .collect(),
    );
    let state_learning_signal = key.bind(&prediction_gradient);

    if !loss.is_finite()
        || !cosine_similarity.is_finite()
        || state_learning_signal
            .values
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(AssociativeReadoutError::NonFiniteLoss);
    }

    Ok(AssociativeQueryResult {
        decoded: codec.decode_answer_symbol(&prediction_vector, query.kind),
        prediction_vector,
        cosine_similarity,
        loss,
        state_learning_signal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_tracking_benchmark::{StateTrackingBenchmark, StateTrackingBenchmarkConfig};

    fn fixture() -> (StateTrackingBenchmark, StateTrackingCodec) {
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
            entities: 6,
            objects: 10,
            locations: 4,
            events: 80,
            query_every: 4,
            historical_query_rate: 1.0,
            seed: 44,
            ..StateTrackingBenchmarkConfig::default()
        })
        .unwrap();
        let codec = StateTrackingCodec::from_benchmark_config(128, &benchmark.config, 55).unwrap();
        (benchmark, codec)
    }

    #[test]
    fn zero_state_has_finite_loss_and_gradient() {
        let (benchmark, codec) = fixture();
        let query = &benchmark.queries[0];
        let asked_time = benchmark.events[query.asked_after_event].time;
        let result = associative_query(
            &codec,
            &ContinuousHV::new(codec.dim()),
            query,
            asked_time,
            1e-4,
        )
        .unwrap();
        assert!(result.loss.is_finite());
        assert!(result.cosine_similarity.is_finite());
        assert!(result
            .state_learning_signal
            .values
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn exact_association_decodes_target_with_near_zero_loss() {
        let (benchmark, codec) = fixture();
        let query = &benchmark.queries[0];
        let asked_time = benchmark.events[query.asked_after_event].time;
        let key = codec.query_key(query, asked_time).unwrap();
        let target = codec.answer_symbol(query.expected).unwrap();
        let perfect_memory = key.bind(target);
        let result = associative_query(&codec, &perfect_memory, query, asked_time, 1e-6).unwrap();
        assert_eq!(result.decoded, query.expected);
        assert!(result.cosine_similarity > 0.99999);
        assert!(result.loss < 1e-5, "loss={}", result.loss);
    }

    #[test]
    fn state_learning_signal_matches_central_finite_difference() {
        let (benchmark, codec) = fixture();
        let query = &benchmark.queries[0];
        let asked_time = benchmark.events[query.asked_after_event].time;
        let state = ContinuousHV::new_random(codec.dim(), 77).scale(0.2);
        let epsilon = 1e-4_f32;
        let result = associative_query(&codec, &state, query, asked_time, epsilon).unwrap();
        let fd_step = 1e-3_f32;

        for coordinate in [0_usize, 3, 17, 63, 127] {
            let mut plus = state.clone();
            plus.values[coordinate] += fd_step;
            let plus_loss = associative_query(&codec, &plus, query, asked_time, epsilon)
                .unwrap()
                .loss;

            let mut minus = state.clone();
            minus.values[coordinate] -= fd_step;
            let minus_loss = associative_query(&codec, &minus, query, asked_time, epsilon)
                .unwrap()
                .loss;

            let numerical = (plus_loss - minus_loss) / (2.0 * fd_step);
            let analytic = result.state_learning_signal.values[coordinate];
            let tolerance = 2e-4_f32.max(0.02 * numerical.abs());
            assert!(
                (analytic - numerical).abs() <= tolerance,
                "coordinate={coordinate} analytic={analytic} numerical={numerical} tolerance={tolerance}"
            );
        }
    }
}
