// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Per-trial trace analysis: learning curves, error bursts, strategy shifts, calibration.
//!
//! Stores per-trial outcomes instead of just aggregates, enabling fine-grained
//! analysis of learning trajectories, error clustering, and metacognitive calibration.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A single trial outcome with full context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialOutcome {
    /// Trial index within the benchmark run.
    pub trial_idx: usize,
    /// Condition label (e.g., "congruent", "incongruent", "nback_2").
    pub condition: String,
    /// Whether the response was correct.
    pub correct: bool,
    /// Reaction time in ticks (1 tick ≈ 50ms).
    pub rt_ticks: f64,
    /// Similarity score of the selected response to the target.
    pub similarity: f64,
    /// Confidence estimate (if available, 0.0 otherwise).
    pub confidence: f64,
    /// Index of the selected response alternative.
    pub response_idx: usize,
    /// Extra fields for benchmark-specific data.
    #[serde(default)]
    pub extra: BTreeMap<String, f64>,
}

/// A row in a learning curve (block-level summary).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialBlockRow {
    /// Block index (0-based).
    pub block: usize,
    /// Mean accuracy in this block.
    pub accuracy: f64,
    /// Mean RT in this block.
    pub mean_rt: f64,
    /// Number of trials in this block.
    pub n_trials: usize,
}

/// A burst of consecutive errors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBurst {
    /// Starting trial index.
    pub start_idx: usize,
    /// Length of the error burst (>= 3).
    pub length: usize,
    /// Condition of the first error in the burst.
    pub condition: String,
}

/// A detected strategy shift (change point in accuracy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyShift {
    /// Block index where the shift occurred.
    pub block: usize,
    /// Accuracy before the shift.
    pub accuracy_before: f64,
    /// Accuracy after the shift.
    pub accuracy_after: f64,
    /// Magnitude: accuracy_after - accuracy_before.
    pub delta: f64,
}

/// Calibration result from per-trial confidence vs. accuracy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Expected Calibration Error (binned).
    pub ece: f64,
    /// Goodman-Kruskal gamma (ordinal association of confidence and accuracy).
    pub gamma: f64,
}

/// Speed-accuracy tradeoff result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedAccuracyResult {
    /// Pearson correlation between RT and accuracy (per block).
    pub pearson_r: f64,
    /// Number of blocks used.
    pub n_blocks: usize,
}

/// Per-trial analysis utilities.
pub struct TrialAnalysis;

impl TrialAnalysis {
    /// Compute a learning curve from a trial trace, grouped into blocks.
    pub fn learning_curve(trace: &[TrialOutcome], block_size: usize) -> Vec<TrialBlockRow> {
        if trace.is_empty() || block_size == 0 {
            return Vec::new();
        }
        let mut rows = Vec::new();
        for (block_idx, chunk) in trace.chunks(block_size).enumerate() {
            let n = chunk.len();
            let correct = chunk.iter().filter(|t| t.correct).count();
            let mean_rt = chunk.iter().map(|t| t.rt_ticks).sum::<f64>() / n as f64;
            rows.push(TrialBlockRow {
                block: block_idx,
                accuracy: correct as f64 / n as f64,
                mean_rt,
                n_trials: n,
            });
        }
        rows
    }

    /// Detect bursts of 3+ consecutive errors.
    pub fn error_bursts(trace: &[TrialOutcome]) -> Vec<ErrorBurst> {
        let mut bursts = Vec::new();
        let mut i = 0;
        while i < trace.len() {
            if !trace[i].correct {
                let start = i;
                while i < trace.len() && !trace[i].correct {
                    i += 1;
                }
                let length = i - start;
                if length >= 3 {
                    bursts.push(ErrorBurst {
                        start_idx: start,
                        length,
                        condition: trace[start].condition.clone(),
                    });
                }
            } else {
                i += 1;
            }
        }
        bursts
    }

    /// Detect strategy shifts: blocks where accuracy changes by more than 15%.
    pub fn strategy_shifts(trace: &[TrialOutcome], block_size: usize) -> Vec<StrategyShift> {
        let curve = Self::learning_curve(trace, block_size);
        let mut shifts = Vec::new();
        for window in curve.windows(2) {
            let delta = window[1].accuracy - window[0].accuracy;
            if delta.abs() > 0.15 {
                shifts.push(StrategyShift {
                    block: window[1].block,
                    accuracy_before: window[0].accuracy,
                    accuracy_after: window[1].accuracy,
                    delta,
                });
            }
        }
        shifts
    }

    /// Compute calibration metrics (ECE and gamma) from per-trial confidence and accuracy.
    pub fn trial_calibration(trace: &[TrialOutcome]) -> CalibrationResult {
        if trace.is_empty() {
            return CalibrationResult {
                ece: 0.0,
                gamma: 0.0,
            };
        }

        // ECE: bin trials by confidence into 10 bins
        let n_bins = 10;
        let mut bin_correct = vec![0.0f64; n_bins];
        let mut bin_conf = vec![0.0f64; n_bins];
        let mut bin_count = vec![0usize; n_bins];

        for t in trace {
            let bin = ((t.confidence * n_bins as f64) as usize).min(n_bins - 1);
            bin_count[bin] += 1;
            bin_conf[bin] += t.confidence;
            if t.correct {
                bin_correct[bin] += 1.0;
            }
        }

        let n_total = trace.len() as f64;
        let mut ece = 0.0;
        for i in 0..n_bins {
            if bin_count[i] > 0 {
                let avg_conf = bin_conf[i] / bin_count[i] as f64;
                let avg_acc = bin_correct[i] / bin_count[i] as f64;
                ece += (bin_count[i] as f64 / n_total) * (avg_conf - avg_acc).abs();
            }
        }

        // Goodman-Kruskal gamma: concordant vs discordant pairs
        let mut concordant = 0u64;
        let mut discordant = 0u64;
        // Sample pairs to keep O(n log n) manageable — use stride for large traces
        let stride = if trace.len() > 500 {
            trace.len() / 250
        } else {
            1
        };
        let mut i = 0;
        while i < trace.len() {
            let mut j = i + stride;
            while j < trace.len() {
                let c1 = trace[i].confidence;
                let c2 = trace[j].confidence;
                let a1 = if trace[i].correct { 1.0 } else { 0.0 };
                let a2 = if trace[j].correct { 1.0 } else { 0.0 };
                let conf_diff = c1 - c2;
                let acc_diff = a1 - a2;
                if conf_diff * acc_diff > 0.0 {
                    concordant += 1;
                } else if conf_diff * acc_diff < 0.0 {
                    discordant += 1;
                }
                j += stride;
            }
            i += stride;
        }

        let gamma = if concordant + discordant > 0 {
            (concordant as f64 - discordant as f64) / (concordant as f64 + discordant as f64)
        } else {
            0.0
        };

        CalibrationResult { ece, gamma }
    }

    /// Compute speed-accuracy tradeoff: Pearson r between block mean RT and block accuracy.
    pub fn speed_accuracy_tradeoff(
        trace: &[TrialOutcome],
        block_size: usize,
    ) -> SpeedAccuracyResult {
        let curve = Self::learning_curve(trace, block_size);
        if curve.len() < 3 {
            return SpeedAccuracyResult {
                pearson_r: 0.0,
                n_blocks: curve.len(),
            };
        }

        let n = curve.len() as f64;
        let rts: Vec<f64> = curve.iter().map(|r| r.mean_rt).collect();
        let accs: Vec<f64> = curve.iter().map(|r| r.accuracy).collect();

        let mean_rt = rts.iter().sum::<f64>() / n;
        let mean_acc = accs.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_rt = 0.0;
        let mut var_acc = 0.0;
        for i in 0..curve.len() {
            let dr = rts[i] - mean_rt;
            let da = accs[i] - mean_acc;
            cov += dr * da;
            var_rt += dr * dr;
            var_acc += da * da;
        }

        let denom = (var_rt * var_acc).sqrt();
        let pearson_r = if denom > 1e-10 { cov / denom } else { 0.0 };

        SpeedAccuracyResult {
            pearson_r,
            n_blocks: curve.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trace(correct_pattern: &[bool], rts: &[f64]) -> Vec<TrialOutcome> {
        correct_pattern
            .iter()
            .zip(rts.iter())
            .enumerate()
            .map(|(i, (&c, &rt))| TrialOutcome {
                trial_idx: i,
                condition: "test".to_string(),
                correct: c,
                rt_ticks: rt,
                similarity: if c { 0.9 } else { 0.3 },
                confidence: if c { 0.8 } else { 0.4 },
                response_idx: 0,
                extra: BTreeMap::new(),
            })
            .collect()
    }

    #[test]
    fn test_learning_curve_basic() {
        let trace = make_trace(
            &[
                true, true, false, true, true, true, false, false, true, true,
            ],
            &[5.0, 4.0, 6.0, 3.0, 5.0, 4.0, 7.0, 8.0, 3.0, 4.0],
        );
        let curve = TrialAnalysis::learning_curve(&trace, 5);
        assert_eq!(curve.len(), 2);
        assert_eq!(curve[0].n_trials, 5);
        assert!((curve[0].accuracy - 0.8).abs() < 1e-10); // 4/5
        assert!((curve[1].accuracy - 0.6).abs() < 1e-10); // 3/5
    }

    #[test]
    fn test_learning_curve_empty() {
        let curve = TrialAnalysis::learning_curve(&[], 5);
        assert!(curve.is_empty());
    }

    #[test]
    fn test_error_bursts_detection() {
        // Pattern: 2 correct, 4 errors (burst!), 1 correct, 3 errors (burst!)
        let correct = [
            true, true, false, false, false, false, true, false, false, false,
        ];
        let rts = [5.0; 10];
        let trace = make_trace(&correct, &rts);
        let bursts = TrialAnalysis::error_bursts(&trace);
        assert_eq!(bursts.len(), 2);
        assert_eq!(bursts[0].start_idx, 2);
        assert_eq!(bursts[0].length, 4);
        assert_eq!(bursts[1].start_idx, 7);
        assert_eq!(bursts[1].length, 3);
    }

    #[test]
    fn test_error_bursts_no_burst() {
        let correct = [true, false, true, false, true, false, true];
        let rts = [5.0; 7];
        let trace = make_trace(&correct, &rts);
        let bursts = TrialAnalysis::error_bursts(&trace);
        assert!(bursts.is_empty());
    }

    #[test]
    fn test_strategy_shift_detection() {
        // Block 0: 20% accuracy, Block 1: 80% accuracy → shift
        let mut correct = vec![false; 10];
        correct[0] = true;
        correct[1] = true;
        // Block 1: 80%
        for i in 5..9 {
            correct[i] = true;
        }
        let rts = vec![5.0; 10];
        let trace = make_trace(&correct, &rts);
        let shifts = TrialAnalysis::strategy_shifts(&trace, 5);
        assert_eq!(shifts.len(), 1);
        assert!(shifts[0].delta > 0.15);
    }

    #[test]
    fn test_trial_calibration_perfect() {
        // Perfect calibration: high confidence = correct, low confidence = incorrect
        let mut trace = Vec::new();
        for i in 0..20 {
            trace.push(TrialOutcome {
                trial_idx: i,
                condition: "test".to_string(),
                correct: i < 10,
                rt_ticks: 5.0,
                similarity: 0.5,
                confidence: if i < 10 { 0.9 } else { 0.1 },
                response_idx: 0,
                extra: BTreeMap::new(),
            });
        }
        let cal = TrialAnalysis::trial_calibration(&trace);
        assert!(
            cal.gamma > 0.5,
            "gamma should be positive for well-calibrated: {}",
            cal.gamma
        );
        assert!(cal.ece.is_finite());
    }

    #[test]
    fn test_trial_calibration_empty() {
        let cal = TrialAnalysis::trial_calibration(&[]);
        assert!((cal.ece - 0.0).abs() < 1e-10);
        assert!((cal.gamma - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_speed_accuracy_tradeoff() {
        // Fast blocks are accurate, slow blocks are inaccurate → negative r
        let mut trace = Vec::new();
        for block in 0..4 {
            for i in 0..10 {
                let fast = block < 2;
                trace.push(TrialOutcome {
                    trial_idx: block * 10 + i,
                    condition: "test".to_string(),
                    correct: fast,
                    rt_ticks: if fast { 3.0 } else { 8.0 },
                    similarity: 0.5,
                    confidence: 0.5,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }
        let sat = TrialAnalysis::speed_accuracy_tradeoff(&trace, 10);
        assert_eq!(sat.n_blocks, 4);
        assert!(
            sat.pearson_r < 0.0,
            "expected negative SAT correlation, got {}",
            sat.pearson_r
        );
    }
}
