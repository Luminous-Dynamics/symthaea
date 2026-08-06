// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Irregular-time task — §5.2 of `SYMTHAEA_TEMPORAL_BENCHMARK_V2_PLAN.md`.
//!
//! ```text
//! A --100ms--> B → C
//! A --20s----> B → D
//! ```
//!
//! Identical items in identical order; only the elapsed interval differs, and
//! the correct target differs with it. This is the family that tests the
//! distinctive closed-form-LTC claim directly — that **elapsed time itself**
//! changes the appropriate prediction, not merely event order. A multi-timescale
//! EMA bank has a credible structural case here too, which is what makes the
//! comparison fair rather than staged.
//!
//! # The target is a rule, not a lookup
//!
//! `target = f(interval)` via a threshold, so **every** interval has a defined
//! correct answer, including ones never trained on. Without that, held-out
//! intervals would have no ground truth and the interpolation/extrapolation
//! split the plan requires would be unanswerable.
//!
//! # Interpolation and extrapolation are reported separately
//!
//! Per §5.2: interpolation = unseen intervals *inside* the trained range;
//! extrapolation = intervals *outside* it. They test different claims — a
//! mechanism can interpolate by smoothing between seen values while failing
//! completely outside the range it was fit on — so collapsing them into one
//! "generalization" number would hide exactly the distinction worth measuring.

use symthaea_evidence_plane::task_validator::{self, TimedItem, TokenId};

/// Shared antecedent, identical across every branch.
pub const CUE_TOKEN: TokenId = 100;
/// Shared decision point. Identical token in every branch — only its incoming
/// interval differs.
pub const DECISION_TOKEN: TokenId = 200;
/// Target when the interval is at or below the threshold.
pub const FAST_TARGET: TokenId = 300;
/// Target when the interval is above it.
pub const SLOW_TARGET: TokenId = 301;

/// Interval separating the two targets.
pub const DEFAULT_THRESHOLD: f64 = 1.0;

/// Configuration for an irregular-time corpus.
#[derive(Debug, Clone)]
pub struct IrregularTimeConfig {
    /// Interval boundary: `<= threshold` yields [`FAST_TARGET`], above yields
    /// [`SLOW_TARGET`].
    pub threshold: f64,
    /// Intervals the model is trained on.
    pub train_intervals: Vec<f64>,
    /// Unseen intervals strictly inside the trained range.
    pub interpolation_intervals: Vec<f64>,
    /// Unseen intervals outside the trained range.
    pub extrapolation_intervals: Vec<f64>,
    /// Sequences generated per interval.
    pub repeats: usize,
}

impl Default for IrregularTimeConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            // Both sides of the threshold, well separated.
            train_intervals: vec![0.1, 0.2, 0.5, 5.0, 10.0, 20.0],
            // Inside [0.1, 20.0], never trained, both sides of the threshold.
            interpolation_intervals: vec![0.3, 0.8, 3.0, 15.0],
            // Outside [0.1, 20.0] on both ends.
            extrapolation_intervals: vec![0.02, 0.05, 40.0, 100.0],
            repeats: 16,
        }
    }
}

impl IrregularTimeConfig {
    /// Correct target for an interval. Total over the reals, which is what makes
    /// held-out intervals answerable.
    pub fn target_for(&self, interval: f64) -> TokenId {
        if interval <= self.threshold {
            FAST_TARGET
        } else {
            SLOW_TARGET
        }
    }

    /// Bin edges for [`task_validator::validate_timed`]. A single edge at the
    /// threshold, matching the rule that actually generates the targets —
    /// declared rather than inferred, so the discretization is auditable.
    pub fn bin_edges(&self) -> Vec<f64> {
        vec![self.threshold]
    }

    fn trained_range(&self) -> (f64, f64) {
        let lo = self
            .train_intervals
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let hi = self
            .train_intervals
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        (lo, hi)
    }
}

/// A validated irregular-time corpus, split per §5.2.
#[derive(Debug, Clone)]
pub struct IrregularTimeTask {
    pub train: Vec<Vec<TimedItem>>,
    pub interpolation: Vec<Vec<TimedItem>>,
    pub extrapolation: Vec<Vec<TimedItem>>,
    pub config: IrregularTimeConfig,
}

#[derive(Debug, Clone)]
pub enum GenerateError {
    /// An interpolation interval is not inside the trained range, or an
    /// extrapolation interval is not outside it. Either would silently
    /// mislabel which claim is being tested.
    MisclassifiedSplit {
        interval: f64,
        expected: &'static str,
    },
    /// All trained intervals fall on one side of the threshold, so there is
    /// nothing for timing to discriminate.
    SingleSidedTraining,
    /// The training corpus failed timed validation.
    ValidationFailed { violations: String },
    /// The training corpus is solvable from tokens alone, so timing is not
    /// load-bearing and this family tests nothing.
    SolvableWithoutTiming,
}

fn build(intervals: &[f64], config: &IrregularTimeConfig) -> Vec<Vec<TimedItem>> {
    let mut out = Vec::new();
    for _ in 0..config.repeats {
        for &interval in intervals {
            out.push(vec![
                TimedItem {
                    token: CUE_TOKEN,
                    dt_since_prev: 0.0,
                },
                // The discriminating interval rides on the decision point's own
                // arrival time — the quantity available at prediction time.
                TimedItem {
                    token: DECISION_TOKEN,
                    dt_since_prev: interval,
                },
                TimedItem {
                    token: config.target_for(interval),
                    dt_since_prev: 0.05,
                },
            ]);
        }
    }
    out
}

/// Build an irregular-time task and prove timing is what makes it solvable.
pub fn generate(config: IrregularTimeConfig) -> Result<IrregularTimeTask, GenerateError> {
    let (lo, hi) = config.trained_range();

    for &i in &config.interpolation_intervals {
        if i < lo || i > hi {
            return Err(GenerateError::MisclassifiedSplit {
                interval: i,
                expected: "inside the trained range",
            });
        }
    }
    for &i in &config.extrapolation_intervals {
        if i >= lo && i <= hi {
            return Err(GenerateError::MisclassifiedSplit {
                interval: i,
                expected: "outside the trained range",
            });
        }
    }
    let fast = config
        .train_intervals
        .iter()
        .any(|&i| i <= config.threshold);
    let slow = config.train_intervals.iter().any(|&i| i > config.threshold);
    if !(fast && slow) {
        return Err(GenerateError::SingleSidedTraining);
    }

    let train = build(&config.train_intervals, &config);

    // Timing must be the thing that makes it solvable: token-only must fail.
    let tokens_only: Vec<Vec<TokenId>> = train
        .iter()
        .map(|s| s.iter().map(|i| i.token).collect())
        .collect();
    if task_validator::validate(&tokens_only, 2).is_ok() {
        return Err(GenerateError::SolvableWithoutTiming);
    }

    task_validator::validate_timed(&train, 2, &config.bin_edges()).map_err(|(_, v)| {
        GenerateError::ValidationFailed {
            violations: format!("{v:?}"),
        }
    })?;

    Ok(IrregularTimeTask {
        interpolation: build(&config.interpolation_intervals, &config),
        extrapolation: build(&config.extrapolation_intervals, &config),
        train,
        config,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_task_validates_and_needs_timing() {
        let task = generate(IrregularTimeConfig::default()).expect("must validate");
        assert!(!task.train.is_empty());
        assert!(!task.interpolation.is_empty());
        assert!(!task.extrapolation.is_empty());
    }

    /// The load-bearing property, mirroring §5.1's: the same corpus must be
    /// unsolvable from tokens and solvable with timing. If tokens sufficed,
    /// this family would not be testing elapsed time at all.
    #[test]
    fn tokens_alone_are_insufficient_timing_is_sufficient() {
        let config = IrregularTimeConfig::default();
        let train = build(&config.train_intervals, &config);

        let tokens_only: Vec<Vec<TokenId>> = train
            .iter()
            .map(|s| s.iter().map(|i| i.token).collect())
            .collect();
        assert!(
            task_validator::validate(&tokens_only, 2).is_err(),
            "token sequences are identical across branches — must be unsolvable"
        );

        let m = task_validator::validate_timed(&train, 2, &config.bin_edges())
            .expect("timing must make it solvable");
        assert!(m.timing_cmi > 0.0);
        assert!(m.timed_oracle_accuracy > m.token_oracle_accuracy);
    }

    /// Held-out intervals must have defined answers, or the split is untestable.
    #[test]
    fn every_held_out_interval_has_ground_truth() {
        let c = IrregularTimeConfig::default();
        for &i in c
            .interpolation_intervals
            .iter()
            .chain(c.extrapolation_intervals.iter())
        {
            let t = c.target_for(i);
            assert!(
                t == FAST_TARGET || t == SLOW_TARGET,
                "undefined target for {i}"
            );
        }
    }

    /// A split that mislabels which claim it tests is worse than no split.
    #[test]
    fn misclassified_splits_are_rejected() {
        let mut c = IrregularTimeConfig::default();
        c.interpolation_intervals = vec![999.0]; // outside trained range
        assert!(matches!(
            generate(c),
            Err(GenerateError::MisclassifiedSplit { .. })
        ));

        let mut c = IrregularTimeConfig::default();
        c.extrapolation_intervals = vec![0.3]; // inside trained range
        assert!(matches!(
            generate(c),
            Err(GenerateError::MisclassifiedSplit { .. })
        ));
    }

    /// Training on one side of the threshold leaves nothing for timing to
    /// discriminate, and would produce a corpus that looks fine but tests nothing.
    #[test]
    fn single_sided_training_is_rejected() {
        let mut c = IrregularTimeConfig::default();
        c.train_intervals = vec![0.1, 0.2, 0.5]; // all below threshold
        c.interpolation_intervals = vec![0.3];
        c.extrapolation_intervals = vec![5.0];
        assert!(matches!(
            generate(c),
            Err(GenerateError::SingleSidedTraining)
        ));
    }

    /// Interpolation and extrapolation must be genuinely different regimes.
    #[test]
    fn extrapolation_lies_outside_the_trained_range() {
        let c = IrregularTimeConfig::default();
        let (lo, hi) = c.trained_range();
        assert!(
            c.interpolation_intervals
                .iter()
                .all(|&i| i >= lo && i <= hi)
        );
        assert!(c.extrapolation_intervals.iter().all(|&i| i < lo || i > hi));
        // And both ends are probed, not just one.
        assert!(c.extrapolation_intervals.iter().any(|&i| i < lo));
        assert!(c.extrapolation_intervals.iter().any(|&i| i > hi));
    }
}
