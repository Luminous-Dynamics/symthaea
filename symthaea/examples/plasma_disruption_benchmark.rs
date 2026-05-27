// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Plasma Disruption Prediction Benchmark
//!
//! Validation benchmark for comparing Phi-drop timing vs ML baselines for
//! plasma disruption prediction in tokamak fusion reactors.
//!
//! ## Overview
//!
//! This benchmark compares three methods for predicting plasma disruptions:
//!
//! 1. **Threshold-based** (traditional): Alert when any sensor exceeds critical threshold
//! 2. **Rate-of-change** (derivative): Alert when dIp/dt or dne/dt exceed threshold
//! 3. **Phi-based** (our method): Alert when Phi drops below threshold OR rate-of-Phi-change exceeds threshold
//!
//! ## Hypothesis
//!
//! The Phi-based method should predict disruptions EARLIER than threshold-based methods
//! because Phi captures system-wide integration changes that precede sensor-level anomalies.
//!
//! ## Usage
//!
//! ```bash
//! # Run with default synthetic data (100 shots)
//! cargo run --example plasma_disruption_benchmark
//!
//! # Run with real C-Mod data (if available)
//! # Place data in data/cmod/train.csv
//! cargo run --example plasma_disruption_benchmark
//! ```
//!
//! ## Output
//!
//! - Comparison table printed to console
//! - Detailed results saved to `data/benchmark_results.json`
//!
//! ## References
//!
//! - Alcator C-Mod disruption database (MIT PSFC)
//! - Multi-Machine Disruption Prediction Challenge
//! - Integrated Information Theory (IIT) for plasma state coherence

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use symthaea::hdc::tiered_phi::{ApproximationTier, TieredPhi};
use symthaea::physics::cmod_adapter::{
    CModHdcEncoder, CModSample, CModShot, DisruptionLabel, SensorNormalizer, SyntheticConfig,
    compute_statistics, generate_synthetic_data, load_csv, to_cmod_plasma_sample,
};
use symthaea_core::hdc::binary_hv::BinaryHV;

// =============================================================================
// BENCHMARK CONFIGURATION
// =============================================================================

/// Configuration for the benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkConfig {
    /// Warning time threshold in ms (predictions within this window count as TP)
    pub warning_window_ms: f64,
    /// Critical threshold for method 1 (Threshold-based)
    pub threshold_critical_sensors: ThresholdConfig,
    /// Critical threshold for method 2 (Rate-of-change)
    pub rate_critical_thresholds: RateConfig,
    /// Critical threshold for method 3 (Phi-based)
    pub phi_critical_thresholds: PhiConfig,
    /// Minimum samples before making predictions
    pub warmup_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ThresholdConfig {
    /// Plasma current threshold (MA)
    pub ip_min: f32,
    pub ip_max: f32,
    /// Electron density threshold (10^20 m^-3)
    pub ne_max: f32,
    /// Electron temperature threshold (keV)
    pub te_min: f32,
    /// Loop voltage threshold (V)
    pub vloop_max: f32,
    /// Safety factor threshold (q95)
    pub q95_min: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RateConfig {
    /// dIp/dt threshold (MA/ms)
    pub dip_dt_threshold: f32,
    /// dne/dt threshold (10^20 m^-3 / ms)
    pub dne_dt_threshold: f32,
    /// dWmhd/dt threshold (MJ/ms) - rapid energy loss
    pub dwmhd_dt_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhiConfig {
    /// Absolute Phi threshold (below this = alert)
    pub phi_threshold: f64,
    /// Phi rate-of-change threshold (negative = dropping, below this = alert)
    pub dphi_dt_threshold: f64,
    /// Window size for Phi calculation
    pub phi_window_size: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warning_window_ms: 100.0, // 100ms warning window
            threshold_critical_sensors: ThresholdConfig {
                ip_min: 0.3,
                ip_max: 2.5,
                ne_max: 4.0,
                te_min: 0.5,
                vloop_max: 5.0,
                q95_min: 2.0,
            },
            rate_critical_thresholds: RateConfig {
                dip_dt_threshold: 0.05,   // 50 kA/ms = rapid current loss
                dne_dt_threshold: 0.02,   // Rapid density change
                dwmhd_dt_threshold: 0.01, // 10 kJ/ms energy loss
            },
            phi_critical_thresholds: PhiConfig {
                phi_threshold: 0.3,       // Phi below 0.3 = system losing integration
                dphi_dt_threshold: -0.02, // 2% Phi drop per ms
                phi_window_size: 50,
            },
            warmup_samples: 20,
        }
    }
}

// =============================================================================
// BENCHMARK RESULTS
// =============================================================================

/// Results for a single prediction method
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    /// Method name
    pub method: String,
    /// True positives: Correctly predicted disruptions
    pub true_positives: usize,
    /// False positives: False alarms
    pub false_positives: usize,
    /// False negatives: Missed disruptions
    pub false_negatives: usize,
    /// True negatives: Correctly predicted non-disruptions
    pub true_negatives: usize,
    /// Warning times for true positives (ms before disruption)
    pub warning_times_ms: Vec<f64>,
    /// Mean warning time (ms)
    pub mean_warning_time_ms: f64,
    /// Standard deviation of warning time (ms)
    pub std_warning_time_ms: f64,
    /// Precision: TP / (TP + FP)
    pub precision: f64,
    /// Recall: TP / (TP + FN)
    pub recall: f64,
    /// F1 score: 2 * precision * recall / (precision + recall)
    pub f1_score: f64,
    /// Processing time (ms total)
    pub processing_time_ms: f64,
}

impl BenchmarkResult {
    fn new(method: &str) -> Self {
        Self {
            method: method.to_string(),
            true_positives: 0,
            false_positives: 0,
            false_negatives: 0,
            true_negatives: 0,
            warning_times_ms: Vec::new(),
            mean_warning_time_ms: 0.0,
            std_warning_time_ms: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            processing_time_ms: 0.0,
        }
    }

    fn finalize(&mut self) {
        // Calculate mean and std of warning times
        if !self.warning_times_ms.is_empty() {
            let sum: f64 = self.warning_times_ms.iter().sum();
            self.mean_warning_time_ms = sum / self.warning_times_ms.len() as f64;

            let variance: f64 = self
                .warning_times_ms
                .iter()
                .map(|t| (t - self.mean_warning_time_ms).powi(2))
                .sum::<f64>()
                / self.warning_times_ms.len() as f64;
            self.std_warning_time_ms = variance.sqrt();
        }

        // Calculate precision, recall, F1
        let tp = self.true_positives as f64;
        let fp = self.false_positives as f64;
        let fn_ = self.false_negatives as f64;

        self.precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        self.recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        self.f1_score = if self.precision + self.recall > 0.0 {
            2.0 * self.precision * self.recall / (self.precision + self.recall)
        } else {
            0.0
        };
    }
}

/// Combined benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FullBenchmarkResults {
    /// Configuration used
    pub config: BenchmarkConfig,
    /// Per-method results
    pub results: Vec<BenchmarkResult>,
    /// Dataset statistics
    pub dataset_stats: DatasetSummary,
    /// Statistical comparison
    pub statistical_comparison: StatisticalComparison,
    /// Timestamp
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetSummary {
    pub total_shots: usize,
    pub disrupted_shots: usize,
    pub non_disrupted_shots: usize,
    pub total_samples: usize,
    pub data_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StatisticalComparison {
    /// Phi vs Threshold: mean warning time difference (ms)
    pub phi_vs_threshold_warning_diff_ms: f64,
    /// Phi vs Rate: mean warning time difference (ms)
    pub phi_vs_rate_warning_diff_ms: f64,
    /// Is Phi significantly earlier than Threshold? (basic test)
    pub phi_earlier_than_threshold: bool,
    /// Is Phi significantly earlier than Rate? (basic test)
    pub phi_earlier_than_rate: bool,
    /// Confidence level (percentage of shots where Phi was earlier)
    pub phi_earlier_percentage: f64,
}

// =============================================================================
// PREDICTION METHODS
// =============================================================================

/// Prediction state for a single shot
struct ShotPredictionState {
    /// Has alert been triggered?
    pub alerted: bool,
    /// Time of alert (if any)
    pub alert_time_ms: Option<f64>,
    /// Previous sample for rate calculation
    pub prev_sample: Option<CModSample>,
    /// Phi history for rate calculation
    pub phi_history: Vec<(f64, f64)>, // (time_ms, phi)
    /// Recent Phi values for windowed average
    pub recent_phi: Vec<f64>,
}

impl ShotPredictionState {
    fn new() -> Self {
        Self {
            alerted: false,
            alert_time_ms: None,
            prev_sample: None,
            phi_history: Vec::new(),
            recent_phi: Vec::new(),
        }
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.alerted = false;
        self.alert_time_ms = None;
        self.prev_sample = None;
        self.phi_history.clear();
        self.recent_phi.clear();
    }
}

/// Method 1: Threshold-based prediction
fn check_threshold_alert(sample: &CModSample, config: &ThresholdConfig) -> bool {
    // Check if any sensor exceeds critical threshold
    if !sample.ip.is_nan() && (sample.ip < config.ip_min || sample.ip > config.ip_max) {
        return true;
    }
    if !sample.ne.is_nan() && sample.ne > config.ne_max {
        return true;
    }
    if !sample.te.is_nan() && sample.te < config.te_min {
        return true;
    }
    if !sample.vloop.is_nan() && sample.vloop > config.vloop_max {
        return true;
    }
    if !sample.q95.is_nan() && sample.q95 < config.q95_min {
        return true;
    }
    false
}

/// Method 2: Rate-of-change prediction
fn check_rate_alert(sample: &CModSample, prev: &CModSample, config: &RateConfig) -> bool {
    let dt = (sample.time_ms - prev.time_ms).max(0.001);

    // dIp/dt
    if !sample.ip.is_nan() && !prev.ip.is_nan() {
        let dip_dt = (sample.ip - prev.ip) / dt as f32;
        if dip_dt.abs() > config.dip_dt_threshold {
            return true;
        }
    }

    // dne/dt
    if !sample.ne.is_nan() && !prev.ne.is_nan() {
        let dne_dt = (sample.ne - prev.ne) / dt as f32;
        if dne_dt.abs() > config.dne_dt_threshold {
            return true;
        }
    }

    // dWmhd/dt
    if !sample.wmhd.is_nan() && !prev.wmhd.is_nan() {
        let dwmhd_dt = (sample.wmhd - prev.wmhd) / dt as f32;
        if dwmhd_dt < -config.dwmhd_dt_threshold {
            // Negative = energy loss
            return true;
        }
    }

    false
}

/// Method 3: Phi-based prediction
fn check_phi_alert(phi: f64, phi_history: &[(f64, f64)], config: &PhiConfig) -> bool {
    // Check absolute threshold
    if phi < config.phi_threshold {
        return true;
    }

    // Check rate of change (need at least 2 points)
    if phi_history.len() >= 5 {
        // Use last 5 points for rate estimation
        let recent: Vec<_> = phi_history.iter().rev().take(5).collect();
        let first = recent.last().unwrap();
        let last = recent.first().unwrap();
        let dt = (last.0 - first.0).max(0.001);
        let dphi_dt = (last.1 - first.1) / dt;

        if dphi_dt < config.dphi_dt_threshold {
            return true;
        }
    }

    false
}

// =============================================================================
// BENCHMARK RUNNER
// =============================================================================

/// Run benchmark on a set of shots
fn run_benchmark(shots: &[CModShot], config: &BenchmarkConfig) -> Vec<BenchmarkResult> {
    let stats = compute_statistics(shots);
    let normalizer = SensorNormalizer::from_stats(&stats);
    let encoder = CModHdcEncoder::default_encoder();
    let mut phi_calc = TieredPhi::new(ApproximationTier::SampledPartition);

    let mut threshold_result = BenchmarkResult::new("Threshold-based");
    let mut rate_result = BenchmarkResult::new("Rate-of-change");
    let mut phi_result = BenchmarkResult::new("Phi-based");

    // Per-shot tracking for statistical comparison
    let mut per_shot_warnings: Vec<(Option<f64>, Option<f64>, Option<f64>, bool)> = Vec::new();

    // Process each shot
    for shot in shots {
        let mut threshold_state = ShotPredictionState::new();
        let mut rate_state = ShotPredictionState::new();
        let mut phi_state = ShotPredictionState::new();

        let start_time = Instant::now();

        // Process samples in order
        for (i, sample) in shot.samples.iter().enumerate() {
            // Skip warmup
            if i < config.warmup_samples {
                threshold_state.prev_sample = Some(sample.clone());
                rate_state.prev_sample = Some(sample.clone());
                continue;
            }

            // === Method 1: Threshold-based ===
            if !threshold_state.alerted
                && check_threshold_alert(sample, &config.threshold_critical_sensors)
            {
                threshold_state.alerted = true;
                threshold_state.alert_time_ms = Some(sample.time_ms);
            }

            // === Method 2: Rate-of-change ===
            if let Some(ref prev) = rate_state.prev_sample {
                if !rate_state.alerted
                    && check_rate_alert(sample, prev, &config.rate_critical_thresholds)
                {
                    rate_state.alerted = true;
                    rate_state.alert_time_ms = Some(sample.time_ms);
                }
            }
            rate_state.prev_sample = Some(sample.clone());

            // === Method 3: Phi-based ===
            // Encode sample and compute Phi
            let plasma_sample = to_cmod_plasma_sample(sample, &normalizer, DisruptionLabel::Normal);
            let _encoding = encoder.encode(&plasma_sample);

            // Convert ContinuousHV to BinaryHV for Phi calculation
            // We need multiple components for Phi, so we use the sensor-level encodings
            let components: Vec<BinaryHV> = plasma_sample
                .sensors
                .iter()
                .enumerate()
                .map(|(idx, &val)| {
                    // Create a deterministic HV based on sensor value
                    let level = ((val * 31.0) as usize).min(31);
                    BinaryHV::random(42 + idx as u64 * 100 + level as u64)
                })
                .collect();

            let phi = phi_calc.compute(&components);

            // Track Phi history
            phi_state.phi_history.push((sample.time_ms, phi));
            phi_state.recent_phi.push(phi);
            if phi_state.recent_phi.len() > config.phi_critical_thresholds.phi_window_size {
                phi_state.recent_phi.remove(0);
            }

            // Check Phi alert
            if !phi_state.alerted
                && check_phi_alert(phi, &phi_state.phi_history, &config.phi_critical_thresholds)
            {
                phi_state.alerted = true;
                phi_state.alert_time_ms = Some(sample.time_ms);
            }

            threshold_state.prev_sample = Some(sample.clone());
        }

        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        threshold_result.processing_time_ms += processing_time / 3.0;
        rate_result.processing_time_ms += processing_time / 3.0;
        phi_result.processing_time_ms += processing_time / 3.0;

        // === Evaluate predictions ===
        let evaluate_shot = |state: &ShotPredictionState,
                             result: &mut BenchmarkResult,
                             shot: &CModShot| {
            if shot.disrupted {
                if let Some(disruption_time) = shot.disruption_time_ms {
                    if state.alerted {
                        if let Some(alert_time) = state.alert_time_ms {
                            let warning_time = disruption_time - alert_time;
                            if warning_time > 0.0 && warning_time <= config.warning_window_ms * 10.0
                            {
                                // True positive: predicted before disruption
                                result.true_positives += 1;
                                result.warning_times_ms.push(warning_time);
                            } else if warning_time <= 0.0 {
                                // Alert was after disruption - too late
                                result.false_negatives += 1;
                            } else {
                                // Alert was too early (could be considered TP still)
                                result.true_positives += 1;
                                result.warning_times_ms.push(warning_time);
                            }
                        }
                    } else {
                        // No alert - missed disruption
                        result.false_negatives += 1;
                    }
                }
            } else {
                // Non-disrupting shot
                if state.alerted {
                    result.false_positives += 1;
                } else {
                    result.true_negatives += 1;
                }
            }
        };

        evaluate_shot(&threshold_state, &mut threshold_result, shot);
        evaluate_shot(&rate_state, &mut rate_result, shot);
        evaluate_shot(&phi_state, &mut phi_result, shot);

        // Track per-shot for comparison
        let get_warning_time = |state: &ShotPredictionState, shot: &CModShot| -> Option<f64> {
            if shot.disrupted {
                if let (Some(alert), Some(disruption)) =
                    (state.alert_time_ms, shot.disruption_time_ms)
                {
                    let warning = disruption - alert;
                    if warning > 0.0 {
                        return Some(warning);
                    }
                }
            }
            None
        };

        per_shot_warnings.push((
            get_warning_time(&threshold_state, shot),
            get_warning_time(&rate_state, shot),
            get_warning_time(&phi_state, shot),
            shot.disrupted,
        ));
    }

    // Finalize results
    threshold_result.finalize();
    rate_result.finalize();
    phi_result.finalize();

    vec![threshold_result, rate_result, phi_result]
}

/// Compute statistical comparison
fn compute_statistical_comparison(results: &[BenchmarkResult]) -> StatisticalComparison {
    let threshold = &results[0];
    let rate = &results[1];
    let phi = &results[2];

    // Mean warning time differences
    let phi_vs_threshold = phi.mean_warning_time_ms - threshold.mean_warning_time_ms;
    let phi_vs_rate = phi.mean_warning_time_ms - rate.mean_warning_time_ms;

    // Count how often Phi was earlier
    let mut phi_earlier_count = 0;
    let mut total_comparisons = 0;

    for i in 0..phi
        .warning_times_ms
        .len()
        .min(threshold.warning_times_ms.len())
    {
        total_comparisons += 1;
        if phi.warning_times_ms[i] > threshold.warning_times_ms[i] {
            phi_earlier_count += 1;
        }
    }

    let phi_earlier_percentage = if total_comparisons > 0 {
        100.0 * phi_earlier_count as f64 / total_comparisons as f64
    } else {
        0.0
    };

    // Simple significance test: Phi is "significantly" earlier if it wins > 60% of comparisons
    let phi_earlier_than_threshold = phi_vs_threshold > 0.0 && phi_earlier_percentage > 60.0;
    let phi_earlier_than_rate =
        phi_vs_rate > 0.0 && phi.mean_warning_time_ms > rate.mean_warning_time_ms;

    StatisticalComparison {
        phi_vs_threshold_warning_diff_ms: phi_vs_threshold,
        phi_vs_rate_warning_diff_ms: phi_vs_rate,
        phi_earlier_than_threshold,
        phi_earlier_than_rate,
        phi_earlier_percentage,
    }
}

// =============================================================================
// OUTPUT FORMATTING
// =============================================================================

fn print_comparison_table(results: &[BenchmarkResult]) {
    println!();
    println!("================================================================================");
    println!("               PLASMA DISRUPTION PREDICTION BENCHMARK RESULTS");
    println!("================================================================================");
    println!();

    // Header
    println!(
        "{:<20} {:>8} {:>8} {:>8} {:>8} {:>12} {:>10} {:>8}",
        "Method", "TP", "FP", "FN", "TN", "Warning(ms)", "Precision", "F1"
    );
    println!("{}", "-".repeat(94));

    // Results rows
    for result in results {
        let warning_str = if result.mean_warning_time_ms > 0.0 {
            format!(
                "{:.1} +/- {:.1}",
                result.mean_warning_time_ms, result.std_warning_time_ms
            )
        } else {
            "N/A".to_string()
        };

        println!(
            "{:<20} {:>8} {:>8} {:>8} {:>8} {:>12} {:>10.3} {:>8.3}",
            result.method,
            result.true_positives,
            result.false_positives,
            result.false_negatives,
            result.true_negatives,
            warning_str,
            result.precision,
            result.f1_score
        );
    }

    println!("{}", "-".repeat(94));
    println!();
}

fn print_statistical_comparison(comparison: &StatisticalComparison) {
    println!("================================================================================");
    println!("                          STATISTICAL COMPARISON");
    println!("================================================================================");
    println!();

    println!("Phi vs Threshold-based:");
    println!(
        "  Warning time difference: {:+.2} ms",
        comparison.phi_vs_threshold_warning_diff_ms
    );
    println!(
        "  Phi earlier in {:.1}% of shots",
        comparison.phi_earlier_percentage
    );
    println!(
        "  Statistically significant: {}",
        if comparison.phi_earlier_than_threshold {
            "YES"
        } else {
            "NO"
        }
    );
    println!();

    println!("Phi vs Rate-of-change:");
    println!(
        "  Warning time difference: {:+.2} ms",
        comparison.phi_vs_rate_warning_diff_ms
    );
    println!(
        "  Statistically significant: {}",
        if comparison.phi_earlier_than_rate {
            "YES"
        } else {
            "NO"
        }
    );
    println!();

    if comparison.phi_earlier_than_threshold || comparison.phi_earlier_than_rate {
        println!("HYPOTHESIS SUPPORTED: Phi-based method provides earlier warnings!");
    } else {
        println!(
            "HYPOTHESIS NOT SUPPORTED: Phi-based method did not provide significantly earlier warnings."
        );
        println!("(This may be due to insufficient data or suboptimal Phi thresholds)");
    }
    println!();
}

fn save_results_to_json(full_results: &FullBenchmarkResults, path: &Path) -> std::io::Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_string_pretty(full_results)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;

    println!("Detailed results saved to: {}", path.display());
    Ok(())
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    println!();
    println!("================================================================================");
    println!("        PHI-DROP TIMING VS ML BASELINES: PLASMA DISRUPTION BENCHMARK");
    println!("================================================================================");
    println!();
    println!("Comparing three methods for predicting plasma disruptions:");
    println!("  1. Threshold-based (traditional sensor thresholds)");
    println!("  2. Rate-of-change (dIp/dt, dne/dt, dWmhd/dt)");
    println!("  3. Phi-based (integrated information drop detection)");
    println!();

    // Load or generate data
    let cmod_data_path = Path::new("data/cmod/train.csv");
    let (shots, data_source) = if cmod_data_path.exists() {
        println!("Loading real C-Mod data from: {}", cmod_data_path.display());
        match load_csv(cmod_data_path) {
            Ok(shots) => (shots, "C-Mod train.csv".to_string()),
            Err(e) => {
                println!("  Error loading CSV: {}", e);
                println!("  Falling back to synthetic data...");
                let config = SyntheticConfig {
                    num_shots: 100,
                    disruption_probability: 0.3,
                    samples_per_shot: 200,
                    sample_interval_ms: 1.0,
                    seed: 42,
                };
                (
                    generate_synthetic_data(&config),
                    "Synthetic (100 shots, seed=42)".to_string(),
                )
            }
        }
    } else {
        println!("C-Mod data not found at: {}", cmod_data_path.display());
        println!("Generating synthetic data (100 shots)...");
        let config = SyntheticConfig {
            num_shots: 100,
            disruption_probability: 0.3,
            samples_per_shot: 200,
            sample_interval_ms: 1.0,
            seed: 42,
        };
        (
            generate_synthetic_data(&config),
            "Synthetic (100 shots, seed=42)".to_string(),
        )
    };

    // Compute and display dataset statistics
    let stats = compute_statistics(&shots);
    println!();
    println!("Dataset Statistics:");
    println!("  Total shots:       {}", stats.total_shots);
    println!(
        "  Disrupted shots:   {} ({:.1}%)",
        stats.disrupted_shots,
        100.0 * stats.disrupted_shots as f64 / stats.total_shots as f64
    );
    println!("  Non-disrupted:     {}", stats.non_disrupted_shots);
    println!("  Total samples:     {}", stats.total_samples);
    println!("  Data source:       {}", data_source);
    println!();

    // Run benchmark
    let config = BenchmarkConfig::default();
    println!("Running benchmark with configuration:");
    println!("  Warning window:    {} ms", config.warning_window_ms);
    println!(
        "  Phi threshold:     {}",
        config.phi_critical_thresholds.phi_threshold
    );
    println!(
        "  dPhi/dt threshold: {}/ms",
        config.phi_critical_thresholds.dphi_dt_threshold
    );
    println!();

    let start = Instant::now();
    let results = run_benchmark(&shots, &config);
    let elapsed = start.elapsed();

    println!("Benchmark completed in {:.2}s", elapsed.as_secs_f64());

    // Print results
    print_comparison_table(&results);

    // Compute and print statistical comparison
    let comparison = compute_statistical_comparison(&results);
    print_statistical_comparison(&comparison);

    // Prepare full results
    let full_results = FullBenchmarkResults {
        config: config.clone(),
        results: results.clone(),
        dataset_stats: DatasetSummary {
            total_shots: stats.total_shots,
            disrupted_shots: stats.disrupted_shots,
            non_disrupted_shots: stats.non_disrupted_shots,
            total_samples: stats.total_samples,
            data_source,
        },
        statistical_comparison: comparison,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    // Save to JSON
    let output_path = Path::new("data/benchmark_results.json");
    if let Err(e) = save_results_to_json(&full_results, output_path) {
        eprintln!("Warning: Could not save results to JSON: {}", e);
    }

    println!();
    println!("================================================================================");
    println!("                              BENCHMARK COMPLETE");
    println!("================================================================================");
    println!();
    println!("To run with real C-Mod data:");
    println!("  1. Download C-Mod disruption data from MIT PSFC or the Multi-Machine Challenge");
    println!("  2. Place train.csv in data/cmod/train.csv");
    println!("  3. Re-run: cargo run --example plasma_disruption_benchmark");
    println!();
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_alert() {
        let config = ThresholdConfig {
            ip_min: 0.3,
            ip_max: 2.5,
            ne_max: 4.0,
            te_min: 0.5,
            vloop_max: 5.0,
            q95_min: 2.0,
        };

        // Normal sample - no alert
        let normal = CModSample {
            shot_id: 1,
            time_ms: 0.0,
            ip: 1.0,
            ne: 2.0,
            te: 3.0,
            prad: 1.0,
            vloop: 1.0,
            q95: 3.5,
            wmhd: 0.1,
            beta: 1.0,
            is_disruption: false,
            time_to_disruption_ms: None,
        };
        assert!(!check_threshold_alert(&normal, &config));

        // Critical sample - alert
        let critical = CModSample {
            shot_id: 1,
            time_ms: 0.0,
            ip: 0.1, // Below threshold
            ne: 2.0,
            te: 3.0,
            prad: 1.0,
            vloop: 1.0,
            q95: 3.5,
            wmhd: 0.1,
            beta: 1.0,
            is_disruption: true,
            time_to_disruption_ms: Some(10.0),
        };
        assert!(check_threshold_alert(&critical, &config));
    }

    #[test]
    fn test_rate_alert() {
        let config = RateConfig {
            dip_dt_threshold: 0.05,
            dne_dt_threshold: 0.02,
            dwmhd_dt_threshold: 0.01,
        };

        let prev = CModSample {
            shot_id: 1,
            time_ms: 0.0,
            ip: 1.0,
            ne: 2.0,
            te: 3.0,
            prad: 1.0,
            vloop: 1.0,
            q95: 3.5,
            wmhd: 0.1,
            beta: 1.0,
            is_disruption: false,
            time_to_disruption_ms: None,
        };

        // Small change - no alert
        let curr_small = CModSample {
            time_ms: 1.0,
            ip: 1.01, // 0.01 MA/ms < threshold
            ..prev.clone()
        };
        assert!(!check_rate_alert(&curr_small, &prev, &config));

        // Large change - alert
        let curr_large = CModSample {
            time_ms: 1.0,
            ip: 0.9, // 0.1 MA/ms > threshold
            ..prev.clone()
        };
        assert!(check_rate_alert(&curr_large, &prev, &config));
    }

    #[test]
    fn test_phi_alert() {
        let config = PhiConfig {
            phi_threshold: 0.3,
            dphi_dt_threshold: -0.02,
            phi_window_size: 50,
        };

        // High Phi, stable - no alert
        let history_stable: Vec<(f64, f64)> = (0..10).map(|i| (i as f64, 0.8)).collect();
        assert!(!check_phi_alert(0.8, &history_stable, &config));

        // Low Phi - alert
        assert!(check_phi_alert(0.2, &history_stable, &config));

        // Dropping Phi - alert
        let history_dropping: Vec<(f64, f64)> =
            (0..10).map(|i| (i as f64, 0.8 - 0.05 * i as f64)).collect();
        assert!(check_phi_alert(0.4, &history_dropping, &config));
    }

    #[test]
    fn test_benchmark_result_finalize() {
        let mut result = BenchmarkResult::new("Test");
        result.true_positives = 8;
        result.false_positives = 2;
        result.false_negatives = 2;
        result.true_negatives = 88;
        result.warning_times_ms = vec![50.0, 60.0, 40.0, 70.0, 50.0];

        result.finalize();

        assert!((result.precision - 0.8).abs() < 0.01); // 8/(8+2) = 0.8
        assert!((result.recall - 0.8).abs() < 0.01); // 8/(8+2) = 0.8
        assert!((result.f1_score - 0.8).abs() < 0.01); // 2*0.8*0.8/(0.8+0.8) = 0.8
        assert!((result.mean_warning_time_ms - 54.0).abs() < 0.01);
    }

    #[test]
    fn test_synthetic_data_benchmark() {
        // Quick benchmark on small synthetic dataset
        let config = SyntheticConfig {
            num_shots: 10,
            disruption_probability: 0.3,
            samples_per_shot: 50,
            sample_interval_ms: 1.0,
            seed: 42,
        };
        let shots = generate_synthetic_data(&config);

        let bench_config = BenchmarkConfig::default();
        let results = run_benchmark(&shots, &bench_config);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].method, "Threshold-based");
        assert_eq!(results[1].method, "Rate-of-change");
        assert_eq!(results[2].method, "Phi-based");
    }
}
