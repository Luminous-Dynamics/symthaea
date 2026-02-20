//! Metacognitive Calibration benchmark.
//!
//! Tests "knowing what you know" — whether the system's confidence in
//! its retrieval judgments covaries with actual accuracy. Directly
//! validates HOT-2 Butlin indicator.
//!
//! Agent stores N items in HDC working memory, then retrieves one after
//! a variable delay (intervening items). Confidence = max similarity.
//! Difficulty: easy (3 items, delay 2), medium (5, delay 5), hard (7, delay 10).
//!
//! Human baselines (Fleming & Lau, 2014):
//! - calibration_error (ECE): 0.10-0.20
//! - discrimination_gamma: 0.40-0.60
//! - meta_d_prime_ratio: 0.70-0.90

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};
use symthaea_core::hdc::ContinuousHV;

/// Metacognitive Calibration benchmark.
pub struct MetacognitiveCalibrationBenchmark;

struct Difficulty {
    _name: &'static str,
    num_items: usize,
    delay: usize,
}

// Delays calibrated to create a difficulty gradient:
// Easy: most items survive in WM. Hard: most items evicted, retrieval fails.
const DIFFICULTIES: [Difficulty; 3] = [
    Difficulty { _name: "easy", num_items: 3, delay: 2 },
    Difficulty { _name: "medium", num_items: 5, delay: 5 },
    Difficulty { _name: "hard", num_items: 7, delay: 7 },
];

impl MetacognitiveCalibrationBenchmark {
    fn run_trial(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> CalibrationResult {
        let dim = config.dimension;
        let seed = config.trial_seed("metacognition", "calibration", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let mut all_confidences = Vec::new();
        let mut all_accuracies = Vec::new();
        let items_per_difficulty = 15;

        for diff in &DIFFICULTIES {
            let mut wm = WorkingMemory::new(WmConfig {
                dimension: dim,
                capacity: config.working_memory_capacity,
                ..Default::default()
            });

            for _item_trial in 0..items_per_difficulty {
                // Generate items to store
                let items: Vec<ContinuousHV> = (0..diff.num_items)
                    .map(|i| {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        ContinuousHV::random(dim, rng.wrapping_add(i as u64))
                    })
                    .collect();

                // Store items in working memory
                for item in &items {
                    wm.perceive(item.clone());
                }

                // Pick a target to retrieve later
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let target_idx = (rng % diff.num_items as u64) as usize;
                let target = items[target_idx].clone();

                // Delay: push intervening distractors
                for d in 0..diff.delay {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let distractor = ContinuousHV::random(dim, rng.wrapping_add(1000 + d as u64));
                    wm.perceive(distractor);
                }

                // Retrieve: find most similar item in WM to query
                let contents = wm.contents();
                if contents.is_empty() {
                    all_confidences.push(0.0);
                    all_accuracies.push(0.0);
                    continue;
                }

                let (_best_idx, best_sim) = contents
                    .iter()
                    .enumerate()
                    .map(|(i, hv)| (i, target.similarity(hv)))
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap_or((0, 0.0));

                // Confidence = max similarity (natural HDC metric), clamped to [0, 1]
                let confidence = (best_sim as f64).clamp(0.0, 1.0);

                // Accuracy: exact match has sim ~1.0, random has sim ~0.
                // Threshold 0.3 separates genuine matches from noise.
                let accurate = best_sim > 0.3;

                all_confidences.push(confidence);
                all_accuracies.push(if accurate { 1.0 } else { 0.0 });

                // Reset WM for next sub-trial
                wm = WorkingMemory::new(WmConfig {
                    dimension: dim,
                    capacity: config.working_memory_capacity,
                    ..Default::default()
                });
            }
        }

        compute_calibration_metrics(&all_confidences, &all_accuracies)
    }
}

/// Compute expected calibration error, discrimination, overconfidence, and resolution.
fn compute_calibration_metrics(confidences: &[f64], accuracies: &[f64]) -> CalibrationResult {
    let n = confidences.len();
    if n == 0 {
        return CalibrationResult {
            calibration_error: 1.0,
            discrimination: 0.0,
            overconfidence: 0.0,
            resolution: 0.0,
        };
    }

    // ECE: bin confidences into 10 bins, compute |mean_conf - mean_acc| per bin
    let num_bins = 10;
    let mut bin_conf_sums = vec![0.0f64; num_bins];
    let mut bin_acc_sums = vec![0.0f64; num_bins];
    let mut bin_counts = vec![0u32; num_bins];

    for i in 0..n {
        let bin = ((confidences[i] * num_bins as f64) as usize).min(num_bins - 1);
        bin_conf_sums[bin] += confidences[i];
        bin_acc_sums[bin] += accuracies[i];
        bin_counts[bin] += 1;
    }

    let mut ece = 0.0;
    for b in 0..num_bins {
        if bin_counts[b] > 0 {
            let mean_conf = bin_conf_sums[b] / bin_counts[b] as f64;
            let mean_acc = bin_acc_sums[b] / bin_counts[b] as f64;
            ece += (mean_conf - mean_acc).abs() * (bin_counts[b] as f64 / n as f64);
        }
    }

    // Goodman-Kruskal gamma: concordant vs discordant pairs
    let mut concordant = 0u64;
    let mut discordant = 0u64;
    // Sample pairs for efficiency (avoid O(n^2) for large n)
    let max_pairs = 5000;
    let step = if n > 100 { n / 100 } else { 1 };
    for i in (0..n).step_by(step) {
        for j in (i + 1..n).step_by(step) {
            let conf_diff = confidences[i] - confidences[j];
            let acc_diff = accuracies[i] - accuracies[j];
            let product = conf_diff * acc_diff;
            if product > 0.0 {
                concordant += 1;
            } else if product < 0.0 {
                discordant += 1;
            }
            if concordant + discordant > max_pairs as u64 {
                break;
            }
        }
        if concordant + discordant > max_pairs as u64 {
            break;
        }
    }

    let gamma = if concordant + discordant > 0 {
        (concordant as f64 - discordant as f64) / (concordant as f64 + discordant as f64)
    } else {
        0.0
    };

    // Overconfidence: mean(confidence - accuracy) when confidence > accuracy
    let mut over_sum = 0.0;
    let mut over_count = 0u32;
    for i in 0..n {
        if confidences[i] > accuracies[i] {
            over_sum += confidences[i] - accuracies[i];
            over_count += 1;
        }
    }
    let overconfidence = if over_count > 0 {
        over_sum / over_count as f64
    } else {
        0.0
    };

    // Resolution: variance of accuracy across confidence bins
    let overall_mean_acc = accuracies.iter().sum::<f64>() / n as f64;
    let mut resolution = 0.0;
    for b in 0..num_bins {
        if bin_counts[b] > 0 {
            let bin_mean_acc = bin_acc_sums[b] / bin_counts[b] as f64;
            resolution += (bin_counts[b] as f64 / n as f64) * (bin_mean_acc - overall_mean_acc).powi(2);
        }
    }

    CalibrationResult {
        calibration_error: ece,
        discrimination: gamma,
        overconfidence,
        resolution,
    }
}

struct CalibrationResult {
    calibration_error: f64,
    discrimination: f64,
    overconfidence: f64,
    resolution: f64,
}

impl PsychBenchmark for MetacognitiveCalibrationBenchmark {
    fn name(&self) -> &str {
        "Metacognition::Calibration"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut eces = Vec::new();
        let mut gammas = Vec::new();
        let mut overconfs = Vec::new();
        let mut resolutions = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            eces.push(r.calibration_error);
            gammas.push(r.discrimination);
            overconfs.push(r.overconfidence);
            resolutions.push(r.resolution);
        }

        result.insert("calibration_error_ece", MetricValue::from_samples(&eces));
        result.insert("discrimination_gamma", MetricValue::from_samples(&gammas));
        result.insert("overconfidence", MetricValue::from_samples(&overconfs));
        result.insert("resolution", MetricValue::from_samples(&resolutions));

        result.conditions = 3; // easy, medium, hard
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = MetacognitiveCalibrationBenchmark.run(&config);
        assert!(result.metrics.contains_key("calibration_error_ece"));
        assert!(result.metrics.contains_key("discrimination_gamma"));
    }

    #[test]
    fn test_calibration_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = MetacognitiveCalibrationBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_calibration_ece_bounded() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = MetacognitiveCalibrationBenchmark.run(&config);
        let ece = result.metrics["calibration_error_ece"].mean;
        assert!(ece >= 0.0 && ece <= 1.0, "ECE should be in [0, 1], got {}", ece);
    }
}
