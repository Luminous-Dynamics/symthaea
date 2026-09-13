// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fixed differentiable HDC associative readout for state tracking.
//!
//! The decoder has **no trainable parameters**.
//!
//! Current one-hop queries use exact bipolar unbinding:
//!
//! ```text
//! value = key(relation, subject) ⊙ h
//! ```
//!
//! Current two-hop `ObjectLocation` queries compose two HDC lookups without a
//! learned decoder:
//!
//! ```text
//! owner_soft  = K_owner(object) ⊙ h
//! location_key = R_location ⊙ owner_soft
//! location_soft = location_key ⊙ h
//! ```
//!
//! Because entity identity symbols are themselves bipolar/self-keying, an ideal
//! owner retrieval can immediately participate in the second relation lookup.
//! For the soft differentiable path, each location output coordinate is
//! `r_i * k_i * h_i^2`, so the exact local derivative is
//! `2 * r_i * k_i * h_i`.
//!
//! Historical queries are deliberately rejected. A random lag role is not a
//! justified temporal-addressing algebra and must not be treated as evidence of
//! historical memory.

use crate::continuous_hv::ContinuousHV;
use crate::state_tracking_benchmark::{TrackingAnswer, TrackingQuery, TrackingQueryKind};
use crate::state_tracking_codec::{StateTrackingCodec, TrackingCodecError};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum AssociativeReadoutError {
    Codec(TrackingCodecError),
    DimensionMismatch { expected: usize, actual: usize },
    InvalidEpsilon,
    HistoricalQueryUnsupported,
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
            Self::InvalidEpsilon => write!(
                f,
                "associative readout epsilon must be finite and positive"
            ),
            Self::HistoricalQueryUnsupported => write!(
                f,
                "historical associative queries require an explicit temporal-addressing algebra"
            ),
            Self::NonFiniteState => {
                write!(f, "associative readout state contains a non-finite value")
            }
            Self::NonFiniteLoss => write!(
                f,
                "associative readout produced a non-finite loss or gradient"
            ),
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
    /// HDC vector produced by the fixed one-hop or two-hop decoder.
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

/// Query a recurrent state through the fixed HDC decoder.
///
/// `epsilon` smooths vector norms as `sqrt(||v||^2 + epsilon^2)` so the loss and
/// derivative remain finite even for an all-zero recurrent state.
pub fn associative_query(
    codec: &StateTrackingCodec,
    state: &ContinuousHV,
    query: &TrackingQuery,
    asked_time: f64,
    epsilon: f32,
) -> Result<AssociativeQueryResult, AssociativeReadoutError> {
    validate_inputs(codec, state, query, epsilon)?;

    // Keep the time-domain validation in the codec even though current queries
    // do not otherwise use asked_time in the algebra.
    let _ = codec.query_key(query, asked_time)?;

    let (prediction_vector, state_jacobian_diagonal) = match query.kind {
        TrackingQueryKind::EntityLocation { .. } | TrackingQueryKind::ObjectOwner { .. } => {
            let key = codec.query_key(query, asked_time)?;
            let prediction = key.bind(state);
            let jacobian = ContinuousHV::from_values(key.as_slice().to_vec());
            (prediction, jacobian)
        }
        TrackingQueryKind::ObjectLocation { object } => {
            // First hop: retrieve a soft owner identity.
            let owner_key = codec.object_owner_key(object)?;
            let owner_soft = owner_key.bind(state);
            let location_role = codec.entity_location_relation_role();

            // Second hop: use the recovered owner identity as the subject key for
            // entity->location lookup. With soft owner retrieval:
            // y_i = r_i * (k_i * h_i) * h_i = r_i*k_i*h_i^2.
            let mut prediction_values = Vec::with_capacity(state.dim());
            let mut jacobian_values = Vec::with_capacity(state.dim());
            for i in 0..state.dim() {
                let r = location_role.as_slice()[i];
                let k = owner_key.as_slice()[i];
                let h = state.values[i];
                prediction_values.push(r * owner_soft.values[i] * h);
                jacobian_values.push(2.0 * r * k * h);
            }
            (
                ContinuousHV::from_values(prediction_values),
                ContinuousHV::from_values(jacobian_values),
            )
        }
    };

    let target = codec.answer_symbol(query.expected)?;
    let (cosine_similarity, loss, prediction_gradient) =
        cosine_loss_and_gradient(&prediction_vector, target, epsilon);

    // The decoder Jacobian is diagonal for both one-hop and current two-hop
    // paths, so dL/dh is a componentwise contraction.
    let state_learning_signal = ContinuousHV::from_values(
        prediction_gradient
            .values
            .iter()
            .zip(state_jacobian_diagonal.values.iter())
            .map(|(&gradient, &jacobian)| gradient * jacobian)
            .collect(),
    );

    if !loss.is_finite()
        || !cosine_similarity.is_finite()
        || prediction_vector.values.iter().any(|value| !value.is_finite())
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

fn validate_inputs(
    codec: &StateTrackingCodec,
    state: &ContinuousHV,
    query: &TrackingQuery,
    epsilon: f32,
) -> Result<(), AssociativeReadoutError> {
    if state.dim() != codec.dim() {
        return Err(AssociativeReadoutError::DimensionMismatch {
            expected: codec.dim(),
            actual: state.dim(),
        });
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(AssociativeReadoutError::InvalidEpsilon);
    }
    if query.is_historical() {
        return Err(AssociativeReadoutError::HistoricalQueryUnsupported);
    }
    if state.values.iter().any(|value| !value.is_finite()) {
        return Err(AssociativeReadoutError::NonFiniteState);
    }
    Ok(())
}

fn cosine_loss_and_gradient(
    prediction: &ContinuousHV,
    target: &ContinuousHV,
    epsilon: f32,
) -> (f32, f32, ContinuousHV) {
    let epsilon_sq = epsilon * epsilon;
    let prediction_sq = prediction.dot(prediction);
    let target_sq = target.dot(target);
    let prediction_norm = (prediction_sq + epsilon_sq).sqrt();
    let target_norm = (target_sq + epsilon_sq).sqrt();
    let dot = prediction.dot(target);
    let norm_product = prediction_norm * target_norm;
    let cosine_similarity = dot / norm_product;
    let loss = 1.0 - cosine_similarity;

    // d(1-cos(y,t))/dy = -t/(||y|| ||t||)
    //                       + (y·t)y/(||y||^3 ||t||)
    let first_scale = -1.0 / norm_product;
    let second_scale =
        dot / (prediction_norm * prediction_norm * prediction_norm * target_norm);
    let gradient = ContinuousHV::from_values(
        target
            .values
            .iter()
            .zip(prediction.values.iter())
            .map(|(&target_value, &prediction_value)| {
                first_scale * target_value + second_scale * prediction_value
            })
            .collect(),
    );

    (cosine_similarity, loss, gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_tracking_benchmark::{
        StateTrackingBenchmark, StateTrackingBenchmarkConfig,
    };

    fn fixture(historical_query_rate: f64) -> (StateTrackingBenchmark, StateTrackingCodec) {
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
            entities: 6,
            objects: 10,
            locations: 4,
            events: 80,
            query_every: 4,
            historical_query_rate,
            seed: 44,
            ..StateTrackingBenchmarkConfig::default()
        })
        .unwrap();
        let codec = StateTrackingCodec::from_benchmark_config(128, &benchmark.config, 55).unwrap();
        (benchmark, codec)
    }

    fn current_query_of_kind(
        benchmark: &StateTrackingBenchmark,
        predicate: impl Fn(TrackingQueryKind) -> bool,
    ) -> &TrackingQuery {
        benchmark
            .queries
            .iter()
            .find(|query| !query.is_historical() && predicate(query.kind))
            .expect("required current query kind")
    }

    #[test]
    fn zero_state_has_finite_one_hop_loss_and_gradient() {
        let (benchmark, codec) = fixture(0.0);
        let query = current_query_of_kind(&benchmark, |kind| {
            matches!(kind, TrackingQueryKind::EntityLocation { .. })
        });
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
    fn exact_one_hop_association_decodes_target_with_near_zero_loss() {
        let (benchmark, codec) = fixture(0.0);
        let query = current_query_of_kind(&benchmark, |kind| {
            matches!(kind, TrackingQueryKind::ObjectOwner { .. })
        });
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
    fn one_hop_state_learning_signal_matches_central_finite_difference() {
        let (benchmark, codec) = fixture(0.0);
        let query = current_query_of_kind(&benchmark, |kind| {
            matches!(kind, TrackingQueryKind::EntityLocation { .. })
        });
        assert_gradient_matches_finite_difference(&benchmark, &codec, query, 77);
    }

    #[test]
    fn two_hop_state_learning_signal_matches_central_finite_difference() {
        let (benchmark, codec) = fixture(0.0);
        let query = current_query_of_kind(&benchmark, |kind| {
            matches!(kind, TrackingQueryKind::ObjectLocation { .. })
        });
        assert_gradient_matches_finite_difference(&benchmark, &codec, query, 78);
    }

    #[test]
    fn two_hop_path_is_not_the_direct_object_owner_unbinding() {
        let (benchmark, codec) = fixture(0.0);
        let query = current_query_of_kind(&benchmark, |kind| {
            matches!(kind, TrackingQueryKind::ObjectLocation { .. })
        });
        let asked_time = benchmark.events[query.asked_after_event].time;
        let state = ContinuousHV::new_random(codec.dim(), 91).scale(0.2);
        let result = associative_query(&codec, &state, query, asked_time, 1e-4).unwrap();
        let object = match query.kind {
            TrackingQueryKind::ObjectLocation { object } => object,
            _ => unreachable!(),
        };
        let direct_owner_unbinding = codec.object_owner_key(object).unwrap().bind(&state);
        assert_ne!(result.prediction_vector, direct_owner_unbinding);
    }

    #[test]
    fn historical_query_is_explicitly_unsupported() {
        let (benchmark, codec) = fixture(1.0);
        let query = benchmark
            .queries
            .iter()
            .find(|query| query.is_historical())
            .expect("historical query");
        let asked_time = benchmark.events[query.asked_after_event].time;
        let error = associative_query(
            &codec,
            &ContinuousHV::new(codec.dim()),
            query,
            asked_time,
            1e-4,
        )
        .unwrap_err();
        assert_eq!(error, AssociativeReadoutError::HistoricalQueryUnsupported);
    }

    fn assert_gradient_matches_finite_difference(
        benchmark: &StateTrackingBenchmark,
        codec: &StateTrackingCodec,
        query: &TrackingQuery,
        state_seed: u64,
    ) {
        let asked_time = benchmark.events[query.asked_after_event].time;
        let state = ContinuousHV::new_random(codec.dim(), state_seed).scale(0.2);
        let epsilon = 1e-4_f32;
        let result = associative_query(codec, &state, query, asked_time, epsilon).unwrap();
        let fd_step = 1e-3_f32;

        for coordinate in [0_usize, 3, 17, 63, 127] {
            let mut plus = state.clone();
            plus.values[coordinate] += fd_step;
            let plus_loss = associative_query(codec, &plus, query, asked_time, epsilon)
                .unwrap()
                .loss;

            let mut minus = state.clone();
            minus.values[coordinate] -= fd_step;
            let minus_loss = associative_query(codec, &minus, query, asked_time, epsilon)
                .unwrap()
                .loss;

            let numerical = (plus_loss - minus_loss) / (2.0 * fd_step);
            let analytic = result.state_learning_signal.values[coordinate];
            let tolerance = 3e-4_f32.max(0.03 * numerical.abs());
            assert!(
                (analytic - numerical).abs() <= tolerance,
                "coordinate={coordinate} analytic={analytic} numerical={numerical} tolerance={tolerance}"
            );
        }
    }
}
