// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Adaptive staircase difficulty procedure.
//!
//! Wraps any `PsychBenchmark` + `DifficultyModel` to automatically converge
//! on the difficulty level where accuracy matches a target (default ~70.7%).
//!
//! Uses a **2-down-1-up** rule (Levitt, 1971): increase difficulty after
//! 2 consecutive correct blocks, decrease after 1 incorrect block.
//! This converges to the 70.7% accuracy threshold.
//!
//! Also supports **1-up-1-down** (50% threshold) and **3-down-1-up** (79.4%).

use serde::{Deserialize, Serialize};

use super::PsychBenchmark;
use super::config::BenchmarkConfig;
use super::report::BenchmarkResult;
#[cfg(test)]
use super::report::MetricValue;

/// Staircase rule determining convergence target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StaircaseRule {
    /// 1-down-1-up: converges to 50% threshold.
    OneDownOneUp,
    /// 2-down-1-up: converges to 70.7% threshold (standard).
    TwoDownOneUp,
    /// 3-down-1-up: converges to 79.4% threshold.
    ThreeDownOneUp,
}

impl StaircaseRule {
    /// Number of consecutive correct blocks needed before stepping up.
    fn correct_streak_needed(&self) -> usize {
        match self {
            StaircaseRule::OneDownOneUp => 1,
            StaircaseRule::TwoDownOneUp => 2,
            StaircaseRule::ThreeDownOneUp => 3,
        }
    }

    /// Target accuracy this rule converges to.
    pub fn target_accuracy(&self) -> f64 {
        match self {
            StaircaseRule::OneDownOneUp => 0.500,
            StaircaseRule::TwoDownOneUp => 0.707,
            StaircaseRule::ThreeDownOneUp => 0.794,
        }
    }
}

/// Configuration for a staircase run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaircaseConfig {
    /// Staircase rule.
    pub rule: StaircaseRule,
    /// Starting difficulty (0.0–1.0).
    pub start_difficulty: f64,
    /// Step size for difficulty adjustment.
    pub step_size: f64,
    /// Minimum step size (halved after each reversal).
    pub min_step: f64,
    /// Number of reversals to collect before stopping.
    pub max_reversals: usize,
    /// Accuracy threshold for "correct block" (proportion correct in a block).
    pub block_accuracy_threshold: f64,
    /// Maximum number of staircase steps.
    pub max_steps: usize,
}

impl Default for StaircaseConfig {
    fn default() -> Self {
        Self {
            rule: StaircaseRule::TwoDownOneUp,
            start_difficulty: 0.3,
            step_size: 0.10,
            min_step: 0.02,
            max_reversals: 8,
            block_accuracy_threshold: 0.5,
            max_steps: 30,
        }
    }
}

/// A single step in the staircase trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaircaseStep {
    /// Step index.
    pub step: usize,
    /// Difficulty level at this step.
    pub difficulty: f64,
    /// Block accuracy at this difficulty.
    pub accuracy: f64,
    /// Whether this was a reversal point.
    pub reversal: bool,
    /// Direction: +1 = harder, -1 = easier.
    pub direction: i8,
}

/// Result of a staircase procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaircaseResult {
    /// Benchmark name.
    pub benchmark: String,
    /// Estimated threshold difficulty (mean of last N reversal points).
    pub threshold: f64,
    /// Standard error of threshold estimate.
    pub threshold_se: f64,
    /// All staircase steps.
    pub trajectory: Vec<StaircaseStep>,
    /// Number of reversals collected.
    pub n_reversals: usize,
    /// Target accuracy for the rule used.
    pub target_accuracy: f64,
    /// Accuracy at the threshold difficulty (interpolated).
    pub accuracy_at_threshold: f64,
}

impl StaircaseResult {
    /// Format as markdown summary.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("## Staircase: {}\n\n", self.benchmark));
        out.push_str(&format!(
            "- Threshold: {:.3} (SE: {:.3})\n",
            self.threshold, self.threshold_se
        ));
        out.push_str(&format!(
            "- Target accuracy: {:.1}%\n",
            self.target_accuracy * 100.0
        ));
        out.push_str(&format!(
            "- Accuracy at threshold: {:.1}%\n",
            self.accuracy_at_threshold * 100.0
        ));
        out.push_str(&format!("- Reversals: {}\n", self.n_reversals));
        out.push_str(&format!("- Steps: {}\n\n", self.trajectory.len()));

        out.push_str("| Step | Difficulty | Accuracy | Reversal |\n");
        out.push_str("|------|-----------|----------|----------|\n");
        for s in &self.trajectory {
            out.push_str(&format!(
                "| {} | {:.3} | {:.3} | {} |\n",
                s.step,
                s.difficulty,
                s.accuracy,
                if s.reversal { "*" } else { "" }
            ));
        }
        out
    }
}

/// Run an adaptive staircase on any benchmark.
pub fn run_staircase(
    bench: &dyn PsychBenchmark,
    base_config: &BenchmarkConfig,
    staircase: &StaircaseConfig,
) -> StaircaseResult {
    let rule = staircase.rule;
    let needed = rule.correct_streak_needed();

    let mut difficulty = staircase.start_difficulty.clamp(0.0, 1.0);
    let mut step_size = staircase.step_size;
    let mut consecutive_correct = 0usize;
    let mut last_direction: i8 = 0; // +1=harder, -1=easier
    let mut reversals = Vec::new();
    let mut trajectory = Vec::new();

    for step in 0..staircase.max_steps {
        // Run benchmark at current difficulty
        let config = BenchmarkConfig {
            difficulty,
            ..base_config.clone()
        };
        let result = bench.run(&config);

        // Compute block accuracy from key metric
        let accuracy = extract_accuracy(&result, bench.name());
        let correct_block = accuracy >= staircase.block_accuracy_threshold;

        // Apply staircase rule
        let direction: i8;
        if correct_block {
            consecutive_correct += 1;
            if consecutive_correct >= needed {
                // Step up (harder)
                let new_diff = (difficulty + step_size).min(1.0);
                if new_diff >= 1.0 && difficulty >= 1.0 {
                    // At ceiling — force step down to avoid saturation
                    direction = -1;
                    difficulty = (difficulty - step_size).max(0.0);
                } else {
                    direction = 1;
                    difficulty = new_diff;
                }
                consecutive_correct = 0;
            } else {
                direction = 0; // hold
            }
        } else {
            // Step down (easier)
            let new_diff = (difficulty - step_size).max(0.0);
            if new_diff <= 0.0 && difficulty <= 0.0 {
                // At floor — force step up to avoid saturation
                direction = 1;
                difficulty = (difficulty + step_size).min(1.0);
            } else {
                direction = -1;
                difficulty = new_diff;
            }
            consecutive_correct = 0;
        }

        // Detect reversal
        let reversal = direction != 0 && last_direction != 0 && direction != last_direction;
        if reversal {
            reversals.push(difficulty);
            // Halve step size after reversal
            step_size = (step_size * 0.5).max(staircase.min_step);
        }

        if direction != 0 {
            last_direction = direction;
        }

        trajectory.push(StaircaseStep {
            step,
            difficulty,
            accuracy,
            reversal,
            direction,
        });

        if reversals.len() >= staircase.max_reversals {
            break;
        }
    }

    // Compute threshold from last N/2 reversals (discard early transient)
    let skip = reversals.len() / 2;
    let threshold_reversals: Vec<f64> = reversals.iter().skip(skip).copied().collect();
    let threshold = if threshold_reversals.is_empty() {
        difficulty
    } else {
        threshold_reversals.iter().sum::<f64>() / threshold_reversals.len() as f64
    };

    let threshold_se = if threshold_reversals.len() > 1 {
        let n = threshold_reversals.len() as f64;
        let mean = threshold;
        let var = threshold_reversals
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        (var / n).sqrt()
    } else {
        0.0
    };

    // Estimate accuracy at threshold via interpolation from trajectory
    let accuracy_at_threshold = trajectory
        .iter()
        .filter(|s| (s.difficulty - threshold).abs() < step_size * 2.0)
        .map(|s| s.accuracy)
        .sum::<f64>()
        / trajectory
            .iter()
            .filter(|s| (s.difficulty - threshold).abs() < step_size * 2.0)
            .count()
            .max(1) as f64;

    StaircaseResult {
        benchmark: bench.name().to_string(),
        threshold,
        threshold_se,
        trajectory,
        n_reversals: reversals.len(),
        target_accuracy: rule.target_accuracy(),
        accuracy_at_threshold,
    }
}

/// Extract the key accuracy metric from a benchmark result.
fn extract_accuracy(result: &BenchmarkResult, benchmark_name: &str) -> f64 {
    // Try benchmark-specific accuracy metrics first
    let keys = match benchmark_name {
        name if name.contains("Stroop") => vec!["incongruent_accuracy", "congruent_accuracy"],
        name if name.contains("GoNoGo") => vec!["overall_accuracy", "go_accuracy"],
        name if name.contains("WCST") => vec!["accuracy"],
        name if name.contains("FittsLaw") => vec!["accuracy"],
        name if name.contains("ArcFluid") => vec!["transfer_accuracy"],
        _ => vec!["accuracy", "overall_accuracy", "word_accuracy", "hit_rate"],
    };

    for key in keys {
        if let Some(mv) = result.metrics.get(key) {
            return mv.mean;
        }
    }

    // Fallback: average of all metrics that look like accuracy
    let acc_metrics: Vec<f64> = result
        .metrics
        .iter()
        .filter(|(k, _)| k.contains("accuracy") || k.contains("hit_rate"))
        .map(|(_, v)| v.mean)
        .collect();

    if !acc_metrics.is_empty() {
        acc_metrics.iter().sum::<f64>() / acc_metrics.len() as f64
    } else {
        0.5 // fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_staircase_rule_targets() {
        assert!((StaircaseRule::OneDownOneUp.target_accuracy() - 0.5).abs() < 1e-3);
        assert!((StaircaseRule::TwoDownOneUp.target_accuracy() - 0.707).abs() < 1e-3);
        assert!((StaircaseRule::ThreeDownOneUp.target_accuracy() - 0.794).abs() < 1e-3);
    }

    #[test]
    fn test_staircase_rule_streaks() {
        assert_eq!(StaircaseRule::OneDownOneUp.correct_streak_needed(), 1);
        assert_eq!(StaircaseRule::TwoDownOneUp.correct_streak_needed(), 2);
        assert_eq!(StaircaseRule::ThreeDownOneUp.correct_streak_needed(), 3);
    }

    #[test]
    fn test_staircase_default_config() {
        let cfg = StaircaseConfig::default();
        assert_eq!(cfg.rule, StaircaseRule::TwoDownOneUp);
        assert!((cfg.start_difficulty - 0.3).abs() < 1e-10);
        assert!(cfg.step_size > cfg.min_step);
    }

    #[test]
    fn test_staircase_on_stroop() {
        use crate::benchmarks::executive::StroopBenchmark;

        let base = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 5,
            ..Default::default()
        };
        let staircase_cfg = StaircaseConfig {
            max_reversals: 4,
            max_steps: 10,
            ..Default::default()
        };

        let result = run_staircase(&StroopBenchmark, &base, &staircase_cfg);
        assert_eq!(result.benchmark, "Executive::Stroop");
        assert!(result.threshold >= 0.0 && result.threshold <= 1.0);
        assert!(!result.trajectory.is_empty());
        assert!(result.target_accuracy > 0.5);
    }

    #[test]
    fn test_staircase_result_markdown() {
        let result = StaircaseResult {
            benchmark: "Test".to_string(),
            threshold: 0.45,
            threshold_se: 0.05,
            trajectory: vec![
                StaircaseStep {
                    step: 0,
                    difficulty: 0.3,
                    accuracy: 0.9,
                    reversal: false,
                    direction: 1,
                },
                StaircaseStep {
                    step: 1,
                    difficulty: 0.4,
                    accuracy: 0.6,
                    reversal: true,
                    direction: -1,
                },
            ],
            n_reversals: 1,
            target_accuracy: 0.707,
            accuracy_at_threshold: 0.75,
        };
        let md = result.to_markdown();
        assert!(md.contains("Threshold: 0.450"));
        assert!(md.contains("Reversals: 1"));
    }

    #[test]
    fn test_extract_accuracy_stroop() {
        let mut result = BenchmarkResult::new("Executive::Stroop", None);
        result.insert("incongruent_accuracy", MetricValue::from_samples(&[0.75]));
        let acc = extract_accuracy(&result, "Executive::Stroop");
        assert!((acc - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_extract_accuracy_fallback() {
        let mut result = BenchmarkResult::new("Unknown::Bench", None);
        result.insert("some_accuracy", MetricValue::from_samples(&[0.85]));
        let acc = extract_accuracy(&result, "Unknown::Bench");
        assert!((acc - 0.85).abs() < 1e-10);
    }
}
