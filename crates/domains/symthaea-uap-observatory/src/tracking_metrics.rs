// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Small, dependency-free GOSPA reference for UAP multi-target evaluation.
//!
//! GOSPA separates localization error from missed and false targets. This is
//! important for anomaly observatories: a tracker must not appear accurate by
//! reporting only the few detections it happened to match.
//!
//! This module implements the standard finite-set GOSPA objective for a
//! single frame. Track-set temporal evaluation is implemented separately in
//! trajectory_metrics as OSPA(2). This module does not implement T-GOSPA's
//! explicit track-switch penalty, data association or a target tracker.
//! The assignment solver uses exact dynamic programming over the smaller of
//! the truth/estimate sets; that side is capped at 20 elements to bound memory.

#![forbid(unsafe_code)]

use crate::triangulation::Vector3;
use std::error::Error;
use std::fmt;

pub const MAX_GOSPA_ASSIGNMENT_SIDE: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GospaConfig {
    /// Localization cutoff in metres.
    pub cutoff_m: f64,
    /// Minkowski order p.
    pub order: f64,
    /// Cardinality penalty parameter alpha; standard GOSPA uses alpha=2.
    pub alpha: f64,
}

impl Default for GospaConfig {
    fn default() -> Self {
        Self {
            cutoff_m: 100.0,
            order: 2.0,
            alpha: 2.0,
        }
    }
}

impl GospaConfig {
    fn validate(self) -> Result<(), GospaError> {
        if !self.cutoff_m.is_finite() || self.cutoff_m <= 0.0 {
            return Err(GospaError::InvalidCutoff);
        }
        if !self.order.is_finite() || self.order < 1.0 || self.order > 4.0 {
            return Err(GospaError::InvalidOrder);
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.alpha > 2.0 {
            return Err(GospaError::InvalidAlpha);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GospaBreakdown {
    /// Full GOSPA distance in metres.
    pub distance_m: f64,
    /// Sum of matched, cutoff localization costs before taking the p-root.
    pub localization_cost_m_p: f64,
    /// Number of unmatched ground-truth targets.
    pub missed_targets: usize,
    /// Number of unmatched estimated targets / false tracks.
    pub false_targets: usize,
    /// Cardinality penalty contribution from missed targets.
    pub missed_cost_m_p: f64,
    /// Cardinality penalty contribution from false targets.
    pub false_cost_m_p: f64,
    /// Number of matched target pairs.
    pub matched_targets: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GospaError {
    InvalidCutoff,
    InvalidOrder,
    InvalidAlpha,
    NonFiniteTruthPosition { index: usize },
    NonFiniteEstimatePosition { index: usize },
    AssignmentSideTooLarge { count: usize, maximum: usize },
}

impl fmt::Display for GospaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCutoff => write!(f, "GOSPA cutoff must be finite and positive"),
            Self::InvalidOrder => write!(f, "GOSPA order must lie in [1,4]"),
            Self::InvalidAlpha => write!(f, "GOSPA alpha must lie in (0,2]"),
            Self::NonFiniteTruthPosition { index } => {
                write!(f, "truth target {index} has a non-finite position")
            }
            Self::NonFiniteEstimatePosition { index } => {
                write!(f, "estimated target {index} has a non-finite position")
            }
            Self::AssignmentSideTooLarge { count, maximum } => write!(
                f,
                "smaller assignment side has {count} targets; maximum supported is {maximum}"
            ),
        }
    }
}

impl Error for GospaError {}

/// Compute the standard finite-set GOSPA distance for one synchronized frame.
///
/// The metric minimizes
///
///   sum_matched min(distance, cutoff)^p
///     + (cutoff^p / alpha) * (missed + false)
///
/// and returns the p-th root. The assignment is solved exactly by dynamic
/// programming on the smaller set, so input order does not affect the score.
/// For alpha=2 this is the conventional GOSPA cardinality split.
pub fn gospa_frame(
    truth: &[Vector3],
    estimates: &[Vector3],
    config: GospaConfig,
) -> Result<GospaBreakdown, GospaError> {
    config.validate()?;
    for (index, position) in truth.iter().enumerate() {
        if !position.is_finite() {
            return Err(GospaError::NonFiniteTruthPosition { index });
        }
    }
    for (index, position) in estimates.iter().enumerate() {
        if !position.is_finite() {
            return Err(GospaError::NonFiniteEstimatePosition { index });
        }
    }

    // GOSPA is symmetric. Put the smaller cardinality on the bitmask side.
    let (rows, columns) = if estimates.len() <= truth.len() {
        (truth, estimates)
    } else {
        (estimates, truth)
    };
    let n = rows.len();
    let m = columns.len();
    if m > MAX_GOSPA_ASSIGNMENT_SIDE {
        return Err(GospaError::AssignmentSideTooLarge {
            count: m,
            maximum: MAX_GOSPA_ASSIGNMENT_SIDE,
        });
    }

    let states = 1_usize << m;
    let infinity = f64::INFINITY;
    let miss_penalty = config.cutoff_m.powf(config.order) / config.alpha;
    let mut costs = vec![infinity; states];
    costs[0] = 0.0;

    for row in rows {
        let mut next = vec![infinity; states];
        for mask in 0..states {
            let base = costs[mask];
            if !base.is_finite() {
                continue;
            }

            // Leave this row unmatched (missed truth, or false estimate if
            // the argument sets were swapped).
            let unmatched_cost = base + miss_penalty;
            if unmatched_cost < next[mask] {
                next[mask] = unmatched_cost;
            }

            // Match this row to one not-yet-assigned column.
            for (column_index, column) in columns.iter().enumerate() {
                let bit = 1_usize << column_index;
                if mask & bit != 0 {
                    continue;
                }
                let distance = row.subtract(*column).norm().min(config.cutoff_m);
                let candidate_cost = base + distance.powf(config.order);
                let next_mask = mask | bit;
                if candidate_cost < next[next_mask] {
                    next[next_mask] = candidate_cost;
                }
            }
        }
        costs = next;
    }

    let mut best_total = infinity;
    let mut best_localization = 0.0;
    let mut best_matches = 0_usize;
    for (mask, accumulated) in costs.iter().copied().enumerate() {
        if !accumulated.is_finite() {
            continue;
        }
        let matched = mask.count_ones() as usize;
        let unmatched_rows = n - matched;
        let unmatched_columns = m - matched;
        let cardinality_cost = miss_penalty * (unmatched_rows + unmatched_columns) as f64;
        let total = accumulated - miss_penalty * unmatched_rows as f64 + cardinality_cost;
        // At exactly the cutoff, matching and leaving both objects
        // unmatched can have the same GOSPA objective (alpha=2). Prefer the
        // maximum-cardinality assignment on an exact cost tie so the
        // localization/cardinality breakdown is deterministic and informative.
        if total < best_total || (total == best_total && matched > best_matches) {
            best_total = total;
            best_matches = matched;
            // Remove the per-row unmatched penalty already included in DP.
            best_localization = accumulated - miss_penalty * unmatched_rows as f64;
        }
    }

    let missed_targets = truth.len() - best_matches;
    let false_targets = estimates.len() - best_matches;
    let missed_cost_m_p = miss_penalty * missed_targets as f64;
    let false_cost_m_p = miss_penalty * false_targets as f64;
    let distance_m = best_total.powf(1.0 / config.order);

    Ok(GospaBreakdown {
        distance_m,
        localization_cost_m_p: best_localization,
        missed_targets,
        false_targets,
        missed_cost_m_p,
        false_cost_m_p,
        matched_targets: best_matches,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64, z: f64) -> Vector3 {
        Vector3::new(x, y, z)
    }

    fn brute_force_gospa(
        truth: &[Vector3],
        estimates: &[Vector3],
        config: GospaConfig,
    ) -> (f64, usize) {
        fn enumerate(
            row: usize,
            truth: &[Vector3],
            estimates: &[Vector3],
            used: &mut [bool],
            config: GospaConfig,
            cost: f64,
            matched: usize,
            best: &mut (f64, usize),
        ) {
            if row == truth.len() {
                let unmatched_estimates = used.iter().filter(|used| !**used).count();
                let total = cost
                    + unmatched_estimates as f64 * config.cutoff_m.powf(config.order)
                        / config.alpha;
                if total < best.0 || (total == best.0 && matched > best.1) {
                    *best = (total, matched);
                }
                return;
            }

            let penalty = config.cutoff_m.powf(config.order) / config.alpha;
            enumerate(
                row + 1,
                truth,
                estimates,
                used,
                config,
                cost + penalty,
                matched,
                best,
            );

            for column in 0..estimates.len() {
                if used[column] {
                    continue;
                }
                used[column] = true;
                let distance = truth[row].subtract(estimates[column]).norm();
                enumerate(
                    row + 1,
                    truth,
                    estimates,
                    used,
                    config,
                    cost + distance.min(config.cutoff_m).powf(config.order),
                    matched + 1,
                    best,
                );
                used[column] = false;
            }
        }

        let mut best = (f64::INFINITY, 0);
        enumerate(
            0,
            truth,
            estimates,
            &mut vec![false; estimates.len()],
            config,
            0.0,
            0,
            &mut best,
        );
        best
    }

    #[test]
    fn exact_match_has_zero_gospa_error() {
        let target = p(10.0, -5.0, 100.0);
        let result = gospa_frame(&[target], &[target], GospaConfig::default()).unwrap();
        assert_eq!(result.distance_m, 0.0);
        assert_eq!(result.localization_cost_m_p, 0.0);
        assert_eq!(result.matched_targets, 1);
        assert_eq!(result.missed_targets, 0);
        assert_eq!(result.false_targets, 0);
    }

    #[test]
    fn localization_and_cardinality_errors_are_separated() {
        let config = GospaConfig {
            cutoff_m: 10.0,
            order: 2.0,
            alpha: 2.0,
        };
        let matched = gospa_frame(&[p(0.0, 0.0, 0.0)], &[p(3.0, 0.0, 0.0)], config).unwrap();
        assert!((matched.localization_cost_m_p - 9.0).abs() < 1.0e-12);
        assert_eq!(matched.missed_targets, 0);
        assert_eq!(matched.false_targets, 0);
        assert!((matched.distance_m - 3.0).abs() < 1.0e-12);

        let missed = gospa_frame(&[p(0.0, 0.0, 0.0)], &[], config).unwrap();
        assert_eq!(missed.missed_targets, 1);
        assert_eq!(missed.false_targets, 0);
        assert!((missed.distance_m - (50.0_f64).sqrt()).abs() < 1.0e-12);

        let false_only = gospa_frame(&[], &[p(0.0, 0.0, 0.0)], config).unwrap();
        assert_eq!(false_only.missed_targets, 0);
        assert_eq!(false_only.false_targets, 1);
        assert!((false_only.distance_m - (50.0_f64).sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn assignment_is_global_not_greedy_and_is_permutation_invariant() {
        let config = GospaConfig {
            cutoff_m: 100.0,
            order: 2.0,
            alpha: 2.0,
        };
        let truth = [p(0.0, 0.0, 0.0), p(10.0, 0.0, 0.0)];
        // Greedy nearest-neighbour assignment takes 0->4 first and leaves
        // 10->-5 (cost 241); the global optimum is 0->-5 and 10->4 (cost 61).
        let estimate = [p(4.0, 0.0, 0.0), p(-5.0, 0.0, 0.0)];
        let forward = gospa_frame(&truth, &estimate, config).unwrap();
        let reversed = gospa_frame(
            &truth,
            &estimate.iter().copied().rev().collect::<Vec<_>>(),
            config,
        )
        .unwrap();
        assert!((forward.distance_m - reversed.distance_m).abs() < 1.0e-12);
        assert_eq!(forward.matched_targets, 2);
        assert!((forward.localization_cost_m_p - 61.0).abs() < 1.0e-12);
    }

    #[test]
    fn cutoff_limits_outlier_localization_cost() {
        let config = GospaConfig {
            cutoff_m: 10.0,
            order: 2.0,
            alpha: 2.0,
        };
        let result = gospa_frame(&[p(0.0, 0.0, 0.0)], &[p(1000.0, 0.0, 0.0)], config).unwrap();
        assert!((result.localization_cost_m_p - 100.0).abs() < 1.0e-12);
        assert_eq!(result.missed_targets, 0);
        assert_eq!(result.false_targets, 0);
        assert!((result.distance_m - 10.0).abs() < 1.0e-12);
    }

    #[test]
    fn dynamic_program_matches_exhaustive_assignment_oracle() {
        let config = GospaConfig {
            cutoff_m: 7.0,
            order: 2.0,
            alpha: 2.0,
        };
        for truth_count in 0..=4 {
            for estimate_count in 0..=4 {
                let truth = (0..truth_count)
                    .map(|index| {
                        p(
                            (index * 7 % 11) as f64,
                            (index * 3 % 5) as f64,
                            (index % 2) as f64,
                        )
                    })
                    .collect::<Vec<_>>();
                let estimates = (0..estimate_count)
                    .map(|index| {
                        p(
                            (index * 5 % 13) as f64 + 0.25,
                            (index * 2 % 7) as f64 - 0.5,
                            (index * 3 % 4) as f64,
                        )
                    })
                    .collect::<Vec<_>>();

                let expected = brute_force_gospa(&truth, &estimates, config);
                let actual = gospa_frame(&truth, &estimates, config).unwrap();
                assert!(
                    (actual.distance_m.powf(config.order) - expected.0).abs() < 1.0e-9,
                    "truth={truth_count}, estimates={estimate_count}, actual={actual:?}, expected={expected:?}"
                );
                assert_eq!(
                    actual.matched_targets, expected.1,
                    "truth={truth_count}, estimates={estimate_count}"
                );
                assert_eq!(actual.missed_targets, truth_count - expected.1);
                assert_eq!(actual.false_targets, estimate_count - expected.1);
            }
        }
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        assert_eq!(
            gospa_frame(
                &[],
                &[],
                GospaConfig {
                    cutoff_m: 0.0,
                    ..GospaConfig::default()
                }
            ),
            Err(GospaError::InvalidCutoff)
        );
        assert_eq!(
            gospa_frame(&[p(f64::NAN, 0.0, 0.0)], &[], GospaConfig::default()),
            Err(GospaError::NonFiniteTruthPosition { index: 0 })
        );
    }

    #[test]
    fn a_large_one_sided_set_uses_the_cardinality_fast_path() {
        // With no truth objects there is no assignment problem: all estimates
        // are false targets. The algorithm bitmasks the smaller side (zero),
        // so this must not be rejected merely because the other side is large.
        let estimates = vec![p(0.0, 0.0, 0.0); MAX_GOSPA_ASSIGNMENT_SIDE + 1];
        let result = gospa_frame(&[], &estimates, GospaConfig::default()).unwrap();
        assert_eq!(result.matched_targets, 0);
        assert_eq!(result.missed_targets, 0);
        assert_eq!(result.false_targets, MAX_GOSPA_ASSIGNMENT_SIDE + 1);
        let expected_cost = (MAX_GOSPA_ASSIGNMENT_SIDE + 1) as f64
            * GospaConfig::default()
                .cutoff_m
                .powf(GospaConfig::default().order)
            / GospaConfig::default().alpha;
        assert!((result.distance_m.powf(2.0) - expected_cost).abs() < 1.0e-8);
    }

    #[test]
    fn an_oversized_smaller_assignment_side_is_rejected_before_dp_allocation() {
        let count = MAX_GOSPA_ASSIGNMENT_SIDE + 1;
        let truth = (0..count)
            .map(|index| p(index as f64, 0.0, 0.0))
            .collect::<Vec<_>>();
        let estimates = (0..count)
            .map(|index| p(index as f64 + 0.1, 0.0, 0.0))
            .collect::<Vec<_>>();
        assert_eq!(
            gospa_frame(&truth, &estimates, GospaConfig::default()),
            Err(GospaError::AssignmentSideTooLarge {
                count,
                maximum: MAX_GOSPA_ASSIGNMENT_SIDE
            })
        );
    }

    #[test]
    fn exact_cutoff_tie_prefers_a_match_but_cost_remains_capped() {
        let config = GospaConfig {
            cutoff_m: 10.0,
            order: 2.0,
            alpha: 2.0,
        };
        let result = gospa_frame(&[p(0.0, 0.0, 0.0)], &[p(10.0, 0.0, 0.0)], config).unwrap();
        assert_eq!(result.matched_targets, 1);
        assert_eq!(result.missed_targets, 0);
        assert_eq!(result.false_targets, 0);
        assert!((result.localization_cost_m_p - 100.0).abs() < 1.0e-12);
        assert!((result.distance_m - 10.0).abs() < 1.0e-12);
    }
}
