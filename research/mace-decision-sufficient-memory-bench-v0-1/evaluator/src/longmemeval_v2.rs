// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact LongMemEval-V2 LAFS metric helper.
//!
//! The official benchmark integrates the best reachable accuracy over
//! log-scaled query-latency budgets from 1s to 200s, then reports the gain
//! against its fixed reference frontier. Accuracy is expressed in percentage
//! points and latency in seconds.
//!
//! Reference operating points are transcribed from the official benchmark
//! page: https://xiaowu0162.github.io/longmemeval-v2/
//! Re-check that page and its evaluation package before every submission.

const MIN_LATENCY_SECONDS: f64 = 1.0;
const MAX_LATENCY_SECONDS: f64 = 200.0;

/// One completed LongMemEval-V2 accuracy/latency operating point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OperatingPointV1 {
    pub accuracy_percent: f64,
    pub query_latency_seconds: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LafsErrorV1 {
    NonFiniteAccuracy,
    AccuracyOutOfRange,
    NonFiniteLatency,
    NegativeLatency,
}

/// Official Small-tier reference methods and operating points.
pub const SMALL_REFERENCE_V1: [OperatingPointV1; 6] = [
    OperatingPointV1 {
        accuracy_percent: 1.3,
        query_latency_seconds: 0.0,
    },
    OperatingPointV1 {
        accuracy_percent: 42.8,
        query_latency_seconds: 0.1,
    },
    OperatingPointV1 {
        accuracy_percent: 51.0,
        query_latency_seconds: 0.2,
    },
    OperatingPointV1 {
        accuracy_percent: 58.6,
        query_latency_seconds: 26.9,
    },
    OperatingPointV1 {
        accuracy_percent: 69.9,
        query_latency_seconds: 177.2,
    },
    OperatingPointV1 {
        accuracy_percent: 74.9,
        query_latency_seconds: 108.3,
    },
];

/// Official Medium-tier reference methods and operating points.
pub const MEDIUM_REFERENCE_V1: [OperatingPointV1; 6] = [
    OperatingPointV1 {
        accuracy_percent: 1.3,
        query_latency_seconds: 0.0,
    },
    OperatingPointV1 {
        accuracy_percent: 38.1,
        query_latency_seconds: 0.1,
    },
    OperatingPointV1 {
        accuracy_percent: 45.9,
        query_latency_seconds: 0.3,
    },
    OperatingPointV1 {
        accuracy_percent: 57.0,
        query_latency_seconds: 25.8,
    },
    OperatingPointV1 {
        accuracy_percent: 68.7,
        query_latency_seconds: 185.8,
    },
    OperatingPointV1 {
        accuracy_percent: 70.1,
        query_latency_seconds: 139.9,
    },
];

fn validate_point(point: OperatingPointV1) -> Result<(), LafsErrorV1> {
    if !point.accuracy_percent.is_finite() {
        return Err(LafsErrorV1::NonFiniteAccuracy);
    }
    if !(0.0..=100.0).contains(&point.accuracy_percent) {
        return Err(LafsErrorV1::AccuracyOutOfRange);
    }
    if !point.query_latency_seconds.is_finite() {
        return Err(LafsErrorV1::NonFiniteLatency);
    }
    if point.query_latency_seconds < 0.0 {
        return Err(LafsErrorV1::NegativeLatency);
    }
    Ok(())
}

/// Compute the official log-latency-weighted average frontier accuracy.
///
/// For each latency budget T, only operating points with latency <= T are
/// reachable. Points below 1s establish the initial frontier value; points
/// beyond 200s cannot affect the integral. Empty input yields zero.
pub fn lafs_score(points: &[OperatingPointV1]) -> Result<f64, LafsErrorV1> {
    for point in points {
        validate_point(*point)?;
    }

    let mut ordered = points.to_vec();
    ordered.sort_by(|left, right| {
        left.query_latency_seconds
            .total_cmp(&right.query_latency_seconds)
            .then_with(|| left.accuracy_percent.total_cmp(&right.accuracy_percent))
    });

    let mut area = 0.0_f64;
    let mut best_accuracy = 0.0_f64;
    let mut previous_latency = MIN_LATENCY_SECONDS;

    for point in ordered {
        let latency = point.query_latency_seconds;
        if latency <= MIN_LATENCY_SECONDS {
            best_accuracy = best_accuracy.max(point.accuracy_percent);
            continue;
        }
        if latency >= MAX_LATENCY_SECONDS {
            break;
        }

        area += best_accuracy * (latency / previous_latency).ln();
        previous_latency = latency;
        best_accuracy = best_accuracy.max(point.accuracy_percent);
    }

    area += best_accuracy * (MAX_LATENCY_SECONDS / previous_latency).ln();
    Ok(area / (MAX_LATENCY_SECONDS / MIN_LATENCY_SECONDS).ln())
}

/// Compute the candidate's LAFS gain over a fixed reference frontier.
///
/// A candidate cannot reduce the frontier because the reference points are
/// included in the union. Tiny negative floating-point noise is clamped to 0.
pub fn delta_lafs(
    reference: &[OperatingPointV1],
    candidate: &[OperatingPointV1],
) -> Result<f64, LafsErrorV1> {
    let reference_score = lafs_score(reference)?;
    let mut combined = Vec::with_capacity(reference.len() + candidate.len());
    combined.extend_from_slice(reference);
    combined.extend_from_slice(candidate);
    let combined_score = lafs_score(&combined)?;
    Ok((combined_score - reference_score).max(0.0))
}

pub fn delta_lafs_vs_small(
    candidate: &[OperatingPointV1],
) -> Result<f64, LafsErrorV1> {
    delta_lafs(&SMALL_REFERENCE_V1, candidate)
}

pub fn delta_lafs_vs_medium(
    candidate: &[OperatingPointV1],
) -> Result<f64, LafsErrorV1> {
    delta_lafs(&MEDIUM_REFERENCE_V1, candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(accuracy_percent: f64, query_latency_seconds: f64) -> OperatingPointV1 {
        OperatingPointV1 {
            accuracy_percent,
            query_latency_seconds,
        }
    }

    #[test]
    fn empty_operating_points_have_zero_lafs() {
        assert_eq!(lafs_score(&[]), Ok(0.0));
    }

    #[test]
    fn subsecond_operating_point_sets_frontier_from_one_second_onward() {
        let score = lafs_score(&[point(42.8, 0.1)]).expect("valid point");
        assert!((score - 42.8).abs() < 1.0e-10);
    }

    #[test]
    fn a_dominated_candidate_does_not_improve_the_frontier() {
        let gain = delta_lafs_vs_small(&[point(40.0, 1.0)]).expect("valid candidate");
        assert!(gain.abs() < 1.0e-10);
    }

    #[test]
    fn a_fast_higher_accuracy_point_improves_small_and_medium_frontiers() {
        let candidate = [point(60.0, 2.0)];
        assert!(delta_lafs_vs_small(&candidate).expect("small tier") > 0.0);
        assert!(delta_lafs_vs_medium(&candidate).expect("medium tier") > 0.0);
    }

    #[test]
    fn points_beyond_two_hundred_seconds_cannot_change_lafs() {
        let baseline = lafs_score(&SMALL_REFERENCE_V1).expect("reference");
        let mut points = SMALL_REFERENCE_V1.to_vec();
        points.push(point(100.0, 250.0));
        let with_slow_candidate = lafs_score(&points).expect("valid point");
        assert!((baseline - with_slow_candidate).abs() < 1.0e-10);
    }

    #[test]
    fn official_small_and_medium_reference_frontiers_match_known_values() {
        let small = lafs_score(&SMALL_REFERENCE_V1).expect("small reference");
        let medium = lafs_score(&MEDIUM_REFERENCE_V1).expect("medium reference");
        assert!((small - 55.764_846_936_380_05).abs() < 1.0e-9);
        assert!((medium - 51.074_089_482_305_1).abs() < 1.0e-9);
    }

    #[test]
    fn negative_latency_is_rejected() {
        assert_eq!(
            lafs_score(&[point(50.0, -0.1)]),
            Err(LafsErrorV1::NegativeLatency)
        );
    }

    #[test]
    fn invalid_accuracy_and_nonfinite_latency_are_rejected() {
        assert_eq!(
            lafs_score(&[point(101.0, 1.0)]),
            Err(LafsErrorV1::AccuracyOutOfRange)
        );
        assert_eq!(
            lafs_score(&[point(50.0, f64::INFINITY)]),
            Err(LafsErrorV1::NonFiniteLatency)
        );
    }

    #[test]
    fn same_operating_point_order_does_not_change_score() {
        let forward = [
            point(58.6, 26.9),
            point(74.9, 108.3),
            point(51.0, 0.2),
        ];
        let reverse = [forward[2], forward[1], forward[0]];
        assert_eq!(lafs_score(&forward), lafs_score(&reverse));
    }
}
