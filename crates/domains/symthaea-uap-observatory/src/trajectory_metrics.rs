// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! OSPA(2) track-set evaluation over a fixed, aligned time window.
//!
//! OSPA(2) computes a time-aggregated distance for each truth/estimated-track
//! pair, then performs one global assignment between the two track sets. The
//! assignment is fixed across the window, so an identity swap cannot score as
//! perfect merely because unordered point sets are correct at every frame.
//!
//! This implements the finite discrete-window OSPA(2) formulation:
//! - per-pair time distance: weighted L_q aggregation of cutoff spatial error;
//! - outer track-set distance: OSPA assignment with order p and cardinality
//!   penalty c^p for each unmatched track, normalized by the larger set.
//!
//! It is NOT T-GOSPA: there is no explicit time-local assignment or switch
//! penalty. The 2026 T-GOSPA family is a separate, newer comparator.
//! This module does not implement track generation or data association.
//!
//! Track states are aligned by vector index within the caller-supplied window.
//! None represents absence at that frame. When both tracks are absent, the
//! per-frame contribution is zero; when exactly one is present, it is cutoff.
//! Weights are normalized internally. Assignment uses exact dynamic
//! programming over the smaller track set, capped at 20 tracks.

#![forbid(unsafe_code)]

use crate::triangulation::Vector3;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const MAX_OSPA2_ASSIGNMENT_SIDE: usize = 20;
pub const MAX_OSPA2_WINDOW_LENGTH: usize = 100_000;

/// One truth or estimated trajectory over a fixed, aligned time window.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackSeries {
    /// Stable identifier for validation/reporting only; never used to force
    /// an association between a truth ID and an estimate ID.
    pub track_id: String,
    /// One entry per frame in the common evaluation window.
    /// None means the object/track is absent at that frame.
    pub states: Vec<Option<Vector3>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ospa2Config {
    pub cutoff_m: f64,
    /// Outer OSPA order p.
    pub order: f64,
    /// Inner temporal aggregation order q.
    pub window_sum_order: f64,
    /// Optional nonnegative weights, one per frame. None means uniform.
    /// Supplied weights are normalized by their sum.
    pub window_weights: Option<Vec<f64>>,
    /// Every track must contain exactly this many frame slots.
    pub window_length: usize,
}

impl Ospa2Config {
    pub fn uniform(cutoff_m: f64, order: f64, window_sum_order: f64, window_length: usize) -> Self {
        Self {
            cutoff_m,
            order,
            window_sum_order,
            window_weights: None,
            window_length,
        }
    }

    fn normalized_weights(&self) -> Result<Vec<f64>, Ospa2Error> {
        if !self.cutoff_m.is_finite() || self.cutoff_m <= 0.0 {
            return Err(Ospa2Error::InvalidCutoff);
        }
        if !self.order.is_finite() || self.order < 1.0 || self.order > 4.0 {
            return Err(Ospa2Error::InvalidOrder);
        }
        if !self.window_sum_order.is_finite()
            || self.window_sum_order < 1.0
            || self.window_sum_order > 4.0
        {
            return Err(Ospa2Error::InvalidWindowSumOrder);
        }
        if self.window_length == 0 {
            return Err(Ospa2Error::InvalidWindowLength);
        }
        if self.window_length > MAX_OSPA2_WINDOW_LENGTH {
            return Err(Ospa2Error::WindowTooLong {
                count: self.window_length,
                maximum: MAX_OSPA2_WINDOW_LENGTH,
            });
        }

        match &self.window_weights {
            None => Ok(vec![1.0 / self.window_length as f64; self.window_length]),
            Some(weights) => {
                if weights.len() != self.window_length {
                    return Err(Ospa2Error::WindowWeightCount {
                        expected: self.window_length,
                        actual: weights.len(),
                    });
                }
                let mut total = 0.0;
                for (index, weight) in weights.iter().enumerate() {
                    if !weight.is_finite() || *weight < 0.0 {
                        return Err(Ospa2Error::InvalidWindowWeight { index });
                    }
                    total += *weight;
                }
                if !total.is_finite() || total <= 0.0 {
                    return Err(Ospa2Error::ZeroWindowWeightSum);
                }
                Ok(weights.iter().map(|weight| *weight / total).collect())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ospa2Breakdown {
    /// Overall OSPA(2) distance in metres.
    pub distance_m: f64,
    /// Normalized matched-track localization contribution before the p-root.
    pub localization_cost_m_p: f64,
    /// Normalized unmatched-track cardinality contribution before the p-root.
    pub cardinality_cost_m_p: f64,
    pub matched_tracks: usize,
    pub missed_truth_tracks: usize,
    pub false_estimated_tracks: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Ospa2Error {
    InvalidCutoff,
    InvalidOrder,
    InvalidWindowSumOrder,
    InvalidWindowLength,
    WindowTooLong {
        count: usize,
        maximum: usize,
    },
    WindowWeightCount {
        expected: usize,
        actual: usize,
    },
    InvalidWindowWeight {
        index: usize,
    },
    ZeroWindowWeightSum,
    EmptyTrackId {
        side: &'static str,
        index: usize,
    },
    DuplicateTrackId {
        side: &'static str,
        track_id: String,
    },
    TrackWindowLength {
        side: &'static str,
        track_index: usize,
        expected: usize,
        actual: usize,
    },
    NonFinitePosition {
        side: &'static str,
        track_index: usize,
        frame_index: usize,
    },
    AssignmentSideTooLarge {
        count: usize,
        maximum: usize,
    },
}

impl fmt::Display for Ospa2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCutoff => write!(f, "OSPA(2) cutoff must be finite and positive"),
            Self::InvalidOrder => write!(f, "OSPA(2) outer order p must lie in [1,4]"),
            Self::InvalidWindowSumOrder => {
                write!(f, "OSPA(2) temporal order q must lie in [1,4]")
            }
            Self::InvalidWindowLength => write!(f, "OSPA(2) window length must be nonzero"),
            Self::WindowTooLong { count, maximum } => write!(
                f,
                "OSPA(2) window has {count} frames; maximum supported is {maximum}"
            ),
            Self::WindowWeightCount { expected, actual } => {
                write!(
                    f,
                    "OSPA(2) has {actual} window weights; expected {expected}"
                )
            }
            Self::InvalidWindowWeight { index } => {
                write!(f, "OSPA(2) window weight {index} is negative or non-finite")
            }
            Self::ZeroWindowWeightSum => {
                write!(f, "OSPA(2) window weights must have a positive finite sum")
            }
            Self::EmptyTrackId { side, index } => {
                write!(f, "{side} track {index} has an empty identifier")
            }
            Self::DuplicateTrackId { side, track_id } => {
                write!(f, "{side} track identifier {track_id} is duplicated")
            }
            Self::TrackWindowLength {
                side,
                track_index,
                expected,
                actual,
            } => write!(
                f,
                "{side} track {track_index} has {actual} frames; expected {expected}"
            ),
            Self::NonFinitePosition {
                side,
                track_index,
                frame_index,
            } => write!(
                f,
                "{side} track {track_index} has a non-finite position at frame {frame_index}"
            ),
            Self::AssignmentSideTooLarge { count, maximum } => write!(
                f,
                "smaller track set has {count} tracks; maximum supported is {maximum}"
            ),
        }
    }
}

impl Error for Ospa2Error {}

fn validate_track_set(
    tracks: &[TrackSeries],
    side: &'static str,
    config: &Ospa2Config,
) -> Result<(), Ospa2Error> {
    let mut identifiers = BTreeSet::new();
    for (track_index, track) in tracks.iter().enumerate() {
        if track.track_id.trim().is_empty() {
            return Err(Ospa2Error::EmptyTrackId {
                side,
                index: track_index,
            });
        }
        if !identifiers.insert(track.track_id.as_str()) {
            return Err(Ospa2Error::DuplicateTrackId {
                side,
                track_id: track.track_id.clone(),
            });
        }
        if track.states.len() != config.window_length {
            return Err(Ospa2Error::TrackWindowLength {
                side,
                track_index,
                expected: config.window_length,
                actual: track.states.len(),
            });
        }
        for (frame_index, state) in track.states.iter().enumerate() {
            if state.is_some_and(|position| !position.is_finite()) {
                return Err(Ospa2Error::NonFinitePosition {
                    side,
                    track_index,
                    frame_index,
                });
            }
        }
    }
    Ok(())
}

fn track_pair_distance_with_weights(
    truth: &TrackSeries,
    estimate: &TrackSeries,
    config: &Ospa2Config,
    weights: &[f64],
) -> f64 {
    let mut weighted_sum = 0.0;
    for frame in 0..config.window_length {
        let frame_distance = match (truth.states[frame], estimate.states[frame]) {
            (Some(truth_position), Some(estimate_position)) => truth_position
                .subtract(estimate_position)
                .norm()
                .min(config.cutoff_m),
            (Some(_), None) | (None, Some(_)) => config.cutoff_m,
            (None, None) => 0.0,
        };
        weighted_sum += weights[frame] * frame_distance.powf(config.window_sum_order);
    }
    weighted_sum.powf(1.0 / config.window_sum_order)
}

/// Weighted time-aggregated base distance between two trajectories.
pub fn ospa2_track_distance(
    truth: &TrackSeries,
    estimate: &TrackSeries,
    config: &Ospa2Config,
) -> Result<f64, Ospa2Error> {
    let weights = config.normalized_weights()?;
    validate_track_set(std::slice::from_ref(truth), "truth", config)?;
    validate_track_set(std::slice::from_ref(estimate), "estimate", config)?;
    Ok(track_pair_distance_with_weights(
        truth, estimate, config, &weights,
    ))
}

/// Compute the OSPA(2) distance between two sets of complete track histories.
///
/// OSPA(2) fixes one assignment for each pair of whole tracks across the
/// evaluation window. It aggregates per-track temporal errors and adds a
/// cutoff penalty for every unmatched track. Uniform time weights are used
/// unless custom weights are supplied in config.
pub fn ospa2_track_sets(
    truth: &[TrackSeries],
    estimates: &[TrackSeries],
    config: &Ospa2Config,
) -> Result<Ospa2Breakdown, Ospa2Error> {
    let weights = config.normalized_weights()?;
    validate_track_set(truth, "truth", config)?;
    validate_track_set(estimates, "estimate", config)?;

    let truth_count = truth.len();
    let estimate_count = estimates.len();
    let maximum_count = truth_count.max(estimate_count);
    if maximum_count == 0 {
        return Ok(Ospa2Breakdown {
            distance_m: 0.0,
            localization_cost_m_p: 0.0,
            cardinality_cost_m_p: 0.0,
            matched_tracks: 0,
            missed_truth_tracks: 0,
            false_estimated_tracks: 0,
        });
    }

    let (rows, columns) = if truth_count >= estimate_count {
        (truth, estimates)
    } else {
        (estimates, truth)
    };
    let column_count = columns.len();
    if column_count > MAX_OSPA2_ASSIGNMENT_SIDE {
        return Err(Ospa2Error::AssignmentSideTooLarge {
            count: column_count,
            maximum: MAX_OSPA2_ASSIGNMENT_SIDE,
        });
    }

    // DP over the smaller track set. Skipping a row leaves one unmatched
    // track on the larger side; every column must be assigned for OSPA(2).
    let state_count = 1_usize << column_count;
    let mut costs = vec![f64::INFINITY; state_count];
    costs[0] = 0.0;

    for row in rows {
        let mut next = costs.clone();
        for mask in 0..state_count {
            let base = costs[mask];
            if !base.is_finite() {
                continue;
            }
            for (column_index, column) in columns.iter().enumerate() {
                let bit = 1_usize << column_index;
                if mask & bit != 0 {
                    continue;
                }
                let distance = track_pair_distance_with_weights(row, column, config, &weights);
                let candidate = base + distance.powf(config.order);
                let next_mask = mask | bit;
                if candidate < next[next_mask] {
                    next[next_mask] = candidate;
                }
            }
        }
        costs = next;
    }

    let matched_cost = costs[state_count - 1];
    if !matched_cost.is_finite() {
        return Err(Ospa2Error::AssignmentSideTooLarge {
            count: column_count,
            maximum: MAX_OSPA2_ASSIGNMENT_SIDE,
        });
    }

    let unmatched_count = maximum_count - column_count;
    let cardinality_cost = unmatched_count as f64 * config.cutoff_m.powf(config.order);
    let denominator = maximum_count as f64;
    let localization_cost_m_p = matched_cost / denominator;
    let cardinality_cost_m_p = cardinality_cost / denominator;
    let distance_m = (localization_cost_m_p + cardinality_cost_m_p).powf(1.0 / config.order);

    Ok(Ospa2Breakdown {
        distance_m,
        localization_cost_m_p,
        cardinality_cost_m_p,
        matched_tracks: column_count,
        missed_truth_tracks: truth_count.saturating_sub(estimate_count),
        false_estimated_tracks: estimate_count.saturating_sub(truth_count),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking_metrics::{GospaConfig, gospa_frame};

    fn track(track_id: &str, xs: &[Option<f64>]) -> TrackSeries {
        TrackSeries {
            track_id: track_id.to_string(),
            states: xs
                .iter()
                .map(|value| value.map(|x| Vector3::new(x, 0.0, 0.0)))
                .collect(),
        }
    }

    fn config(window_length: usize) -> Ospa2Config {
        Ospa2Config::uniform(10.0, 2.0, 2.0, window_length)
    }

    #[test]
    fn identical_track_sets_have_zero_ospa2_error() {
        let truth = vec![
            track("truth-a", &[Some(0.0), Some(1.0), Some(2.0)]),
            track("truth-b", &[Some(10.0), Some(11.0), Some(12.0)]),
        ];
        let estimates = vec![
            track("estimate-x", &[Some(0.0), Some(1.0), Some(2.0)]),
            track("estimate-y", &[Some(10.0), Some(11.0), Some(12.0)]),
        ];
        let result = ospa2_track_sets(&truth, &estimates, &config(3)).unwrap();
        assert_eq!(result.distance_m, 0.0);
        assert_eq!(result.localization_cost_m_p, 0.0);
        assert_eq!(result.cardinality_cost_m_p, 0.0);
        assert_eq!(result.matched_tracks, 2);
    }

    #[test]
    fn fixed_track_assignment_detects_identity_swaps_that_frame_gospa_misses() {
        let truth = vec![
            track("truth-a", &[Some(0.0), Some(10.0)]),
            track("truth-b", &[Some(10.0), Some(0.0)]),
        ];
        let estimates = vec![
            track("estimate-x", &[Some(0.0), Some(0.0)]),
            track("estimate-y", &[Some(10.0), Some(10.0)]),
        ];

        let perfect_frame = [Vector3::new(0.0, 0.0, 0.0), Vector3::new(10.0, 0.0, 0.0)];
        let frame_zero = gospa_frame(
            &perfect_frame,
            &perfect_frame,
            GospaConfig {
                cutoff_m: 10.0,
                order: 2.0,
                alpha: 2.0,
            },
        )
        .unwrap();
        let frame_one = frame_zero;
        assert_eq!(frame_zero.distance_m, 0.0);
        assert_eq!(frame_one.distance_m, 0.0);

        let result = ospa2_track_sets(&truth, &estimates, &config(2)).unwrap();
        assert!(result.distance_m > 0.0);
        assert!(result.localization_cost_m_p > 0.0);
        assert_eq!(result.matched_tracks, 2);
    }

    #[test]
    fn track_fragmentation_increases_cardinality_and_temporal_error() {
        let truth = vec![track(
            "truth-a",
            &[Some(0.0), Some(1.0), Some(2.0), Some(3.0)],
        )];
        let continuous = vec![track(
            "estimate-continuous",
            &[Some(0.0), Some(1.0), Some(2.0), Some(3.0)],
        )];
        let fragmented = vec![
            track("estimate-first-half", &[Some(0.0), Some(1.0), None, None]),
            track("estimate-second-half", &[None, None, Some(2.0), Some(3.0)]),
        ];
        let good = ospa2_track_sets(&truth, &continuous, &config(4)).unwrap();
        let split = ospa2_track_sets(&truth, &fragmented, &config(4)).unwrap();
        assert_eq!(good.distance_m, 0.0);
        assert!(split.distance_m > 0.0);
        assert_eq!(split.false_estimated_tracks, 1);
        assert_eq!(split.matched_tracks, 1);
    }

    #[test]
    fn missing_both_tracks_is_zero_but_one_sided_presence_costs_the_cutoff() {
        let absent = track("absent", &[None, None, None]);
        let present = track("present", &[None, Some(5.0), None]);
        let result = ospa2_track_distance(&absent, &present, &config(3)).unwrap();
        assert!((result - 10.0 / 3.0_f64.sqrt()).abs() < 1.0e-12);

        let both_absent = ospa2_track_distance(
            &absent,
            &track("also-absent", &[None, None, None]),
            &config(3),
        )
        .unwrap();
        assert_eq!(both_absent, 0.0);
    }

    #[test]
    fn custom_weights_are_normalized_and_permutation_invariant() {
        let truth = vec![
            track("truth-a", &[Some(0.0), Some(1.0), Some(2.0)]),
            track("truth-b", &[Some(10.0), Some(11.0), Some(12.0)]),
        ];
        let estimates = vec![
            track("estimate-x", &[Some(0.0), Some(1.5), Some(2.0)]),
            track("estimate-y", &[Some(10.5), Some(11.0), Some(12.0)]),
        ];
        let mut weighted = config(3);
        weighted.window_weights = Some(vec![1.0, 2.0, 1.0]);
        let mut scaled = config(3);
        scaled.window_weights = Some(vec![10.0, 20.0, 10.0]);
        let a = ospa2_track_sets(&truth, &estimates, &weighted).unwrap();
        let b = ospa2_track_sets(
            &truth.iter().cloned().rev().collect::<Vec<_>>(),
            &estimates.iter().cloned().rev().collect::<Vec<_>>(),
            &scaled,
        )
        .unwrap();
        assert!((a.distance_m - b.distance_m).abs() < 1.0e-12);
    }

    #[test]
    fn malformed_track_sets_fail_closed() {
        let duplicate = vec![track("same", &[Some(0.0)]), track("same", &[Some(1.0)])];
        assert!(matches!(
            ospa2_track_sets(&duplicate, &[], &config(1)),
            Err(Ospa2Error::DuplicateTrackId { .. })
        ));

        let too_short = vec![TrackSeries {
            track_id: "short".to_string(),
            states: Vec::new(),
        }];
        assert!(matches!(
            ospa2_track_sets(&too_short, &[], &config(1)),
            Err(Ospa2Error::TrackWindowLength { .. })
        ));

        let nan = vec![TrackSeries {
            track_id: "nan".to_string(),
            states: vec![Some(Vector3::new(f64::NAN, 0.0, 0.0))],
        }];
        assert!(matches!(
            ospa2_track_sets(&nan, &[], &config(1)),
            Err(Ospa2Error::NonFinitePosition { .. })
        ));
    }

    #[test]
    fn oversized_window_is_rejected_before_uniform_weight_allocation() {
        let config = Ospa2Config::uniform(10.0, 2.0, 2.0, MAX_OSPA2_WINDOW_LENGTH + 1);
        assert_eq!(
            ospa2_track_sets(&[], &[], &config),
            Err(Ospa2Error::WindowTooLong {
                count: MAX_OSPA2_WINDOW_LENGTH + 1,
                maximum: MAX_OSPA2_WINDOW_LENGTH,
            })
        );
    }

    #[test]
    fn oversized_track_assignment_side_is_rejected() {
        let many = (0..=MAX_OSPA2_ASSIGNMENT_SIDE)
            .map(|index| track(&format!("truth-{index}"), &[Some(index as f64)]))
            .collect::<Vec<_>>();
        assert!(matches!(
            ospa2_track_sets(&many, &many, &config(1)),
            Err(Ospa2Error::AssignmentSideTooLarge { .. })
        ));
    }
}
