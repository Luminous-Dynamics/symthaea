// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Plasma Threshold Tuning Optimizer
//!
//! Optimizes threshold parameters for plasma disruption prediction to achieve
//! the best tradeoff between early warning time and false positive rate.
//!
//! ## Optimization Objectives
//!
//! - **Maximize**: Mean warning time (early detection is better)
//! - **Minimize**: False positive rate (avoid unnecessary shutdowns)
//! - **Constraint**: Recall > 0.95 (must catch 95%+ of disruptions)
//!
//! ## Search Methods
//!
//! 1. **Grid Search**: Exhaustive search over parameter grid (5^5 = 3125 configs)
//! 2. **Random Search**: Sample 500 random configurations (often competitive)
//! 3. **Bayesian Optimization**: GP surrogate for efficient optimization
//!
//! ## Run
//!
//! ```bash
//! cargo run --example plasma_threshold_optimizer
//! ```

use std::fs::{self, File};
use std::io::Write as IoWrite;
use std::path::Path;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use symthaea::physics::cmod_adapter::{
    CModShot, DisruptionLabel, LabelConfig, SyntheticConfig, generate_synthetic_data, label_samples,
};
use symthaea::physics::plasma_control::{PlasmaControlConfig, StabilityRegime};

// =============================================================================
// THRESHOLD CONFIGURATION
// =============================================================================

/// Threshold configuration for plasma stability assessment
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ThresholdConfig {
    /// Phi threshold for stable regime [0.5, 0.8]
    pub phi_stable: f32,
    /// Phi threshold for warning regime [0.3, 0.6]
    pub phi_warning: f32,
    /// Phi threshold for critical regime [0.15, 0.4]
    pub phi_critical: f32,
    /// Rate of change threshold for rapid drops [-0.2, -0.05]
    pub phi_rate_threshold: f32,
    /// Volatility threshold for instability detection [0.1, 0.3]
    pub volatility_threshold: f32,
}

impl ThresholdConfig {
    /// Parameter bounds for search space
    pub const PHI_STABLE_BOUNDS: (f32, f32) = (0.5, 0.8);
    pub const PHI_WARNING_BOUNDS: (f32, f32) = (0.3, 0.6);
    pub const PHI_CRITICAL_BOUNDS: (f32, f32) = (0.15, 0.4);
    pub const PHI_RATE_BOUNDS: (f32, f32) = (-0.2, -0.05);
    pub const VOLATILITY_BOUNDS: (f32, f32) = (0.1, 0.3);

    /// Create a new threshold config with validation
    pub fn new(
        phi_stable: f32,
        phi_warning: f32,
        phi_critical: f32,
        phi_rate_threshold: f32,
        volatility_threshold: f32,
    ) -> Option<Self> {
        // Validate threshold ordering: stable > warning > critical
        if phi_stable <= phi_warning || phi_warning <= phi_critical {
            return None;
        }
        Some(Self {
            phi_stable,
            phi_warning,
            phi_critical,
            phi_rate_threshold,
            volatility_threshold,
        })
    }

    /// Generate a random valid configuration
    pub fn random(rng: &mut impl Rng) -> Self {
        loop {
            let phi_stable = rng.r#gen_range(Self::PHI_STABLE_BOUNDS.0..=Self::PHI_STABLE_BOUNDS.1);
            let phi_warning =
                rng.r#gen_range(Self::PHI_WARNING_BOUNDS.0..=Self::PHI_WARNING_BOUNDS.1);
            let phi_critical =
                rng.r#gen_range(Self::PHI_CRITICAL_BOUNDS.0..=Self::PHI_CRITICAL_BOUNDS.1);
            let phi_rate_threshold =
                rng.r#gen_range(Self::PHI_RATE_BOUNDS.0..=Self::PHI_RATE_BOUNDS.1);
            let volatility_threshold =
                rng.r#gen_range(Self::VOLATILITY_BOUNDS.0..=Self::VOLATILITY_BOUNDS.1);

            if let Some(config) = Self::new(
                phi_stable,
                phi_warning,
                phi_critical,
                phi_rate_threshold,
                volatility_threshold,
            ) {
                return config;
            }
        }
    }

    /// Convert to PlasmaControlConfig
    pub fn to_plasma_control_config(&self) -> PlasmaControlConfig {
        PlasmaControlConfig {
            phi_stable_threshold: self.phi_stable,
            phi_warning_threshold: self.phi_warning,
            phi_critical_threshold: self.phi_critical,
            rate_of_change_threshold: self.phi_rate_threshold,
            volatility_threshold: self.volatility_threshold,
            ..Default::default()
        }
    }

    /// Create from grid indices (5 levels per parameter)
    pub fn from_grid_indices(i: usize, j: usize, k: usize, l: usize, m: usize) -> Option<Self> {
        let levels = 5;
        let idx_to_val = |idx: usize, bounds: (f32, f32)| -> f32 {
            bounds.0 + (bounds.1 - bounds.0) * (idx as f32) / ((levels - 1) as f32)
        };

        let phi_stable = idx_to_val(i, Self::PHI_STABLE_BOUNDS);
        let phi_warning = idx_to_val(j, Self::PHI_WARNING_BOUNDS);
        let phi_critical = idx_to_val(k, Self::PHI_CRITICAL_BOUNDS);
        let phi_rate = idx_to_val(l, Self::PHI_RATE_BOUNDS);
        let volatility = idx_to_val(m, Self::VOLATILITY_BOUNDS);

        Self::new(phi_stable, phi_warning, phi_critical, phi_rate, volatility)
    }

    /// Get as parameter vector for GP
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.phi_stable as f64,
            self.phi_warning as f64,
            self.phi_critical as f64,
            self.phi_rate_threshold as f64,
            self.volatility_threshold as f64,
        ]
    }
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            phi_stable: 0.6,
            phi_warning: 0.4,
            phi_critical: 0.25,
            phi_rate_threshold: -0.1,
            volatility_threshold: 0.15,
        }
    }
}

// =============================================================================
// EVALUATION RESULTS
// =============================================================================

const WARNING_WINDOW_MS: f64 = 100.0;
const CRITICAL_WINDOW_MS: f64 = 20.0;

/// Results from evaluating a threshold configuration. Every confusion-matrix count uses a
/// shot/event as its unit. Censored or unobservable shots are excluded and reported separately.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResults {
    /// Disruption shots detected in the configured warning/critical window.
    pub true_positives: usize,
    /// Non-disruption shots with one or more alarms.
    pub false_positives: usize,
    /// Non-disruption shots with no alarms.
    pub true_negatives: usize,
    /// Disruption shots with an observed warning window but no alarm in that window.
    pub false_negatives: usize,
    /// Mean warning lead time in ms (one earliest alarm per detected disruption shot).
    pub mean_warning_time_ms: f64,
    /// Median warning lead time in ms.
    pub median_warning_time_ms: f64,
    /// Minimum warning lead time in ms.
    pub min_warning_time_ms: f64,
    /// Total raw samples, including excluded invalid samples.
    pub total_samples: usize,
    /// Samples excluded because a required Phi-proxy input was missing or non-finite.
    pub invalid_samples: usize,
    /// Disruption shots with at least one valid sample in the warning/critical label window.
    pub scoreable_disruption_shots: usize,
    /// Non-disruption shots with at least one valid sample.
    pub scoreable_non_disruption_shots: usize,
    /// Disruption shots with no valid sample in the warning/critical label window.
    pub unscored_disruption_shots: usize,
    /// Non-disruption shots with no valid samples.
    pub unscored_non_disruption_shots: usize,
    /// All disruption shots in the corpus, including unscored shots.
    pub total_disruption_shots: usize,
    /// All non-disruption shots in the corpus, including unscored shots.
    pub total_non_disruption_shots: usize,
    /// Per-shot decisions and explicit censoring reasons for audit/replay.
    pub shot_records: Vec<ShotEvaluationRecord>,
}

impl EvaluationResults {
    /// Event-level recall; undefined if there are no scoreable disruption shots.
    pub fn recall(&self) -> Option<f64> {
        let denominator = self.true_positives + self.false_negatives;
        (denominator > 0).then(|| self.true_positives as f64 / denominator as f64)
    }

    /// Event-level precision; undefined if there are no alarmed shots.
    pub fn precision(&self) -> Option<f64> {
        let denominator = self.true_positives + self.false_positives;
        (denominator > 0).then(|| self.true_positives as f64 / denominator as f64)
    }

    /// Event-level false-positive rate; undefined if there are no scoreable non-disruption shots.
    pub fn false_positive_rate(&self) -> Option<f64> {
        let denominator = self.false_positives + self.true_negatives;
        (denominator > 0).then(|| self.false_positives as f64 / denominator as f64)
    }

    /// Event-level F1; undefined if precision or recall is undefined.
    pub fn f1_score(&self) -> Option<f64> {
        let p = self.precision()?;
        let r = self.recall()?;
        Some(if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) })
    }
}

/// Render undefined metrics visibly instead of converting them to perfect-looking scores.
fn metric_display(value: Option<f64>) -> String {
    match value.filter(|value| value.is_finite()) {
        Some(value) => format!("{value:.4}"),
        None => "NotScoreable".to_string(),
    }
}

/// Inputs used by the Phi proxy must be present and finite before a sample can count.
fn has_valid_phi_inputs(sample: &symthaea::physics::cmod_adapter::CModSample) -> bool {
    sample.time_ms.is_finite()
        && sample.ip.is_finite()
        && sample.ne.is_finite()
        && sample.te.is_finite()
        && sample.q95.is_finite()
        && sample.prad.is_finite()
}

// =============================================================================
// EVALUATION FUNCTION
// =============================================================================

/// Reason an event could not be scored because the required observations were unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShotUnscoredReason {
    NoValidSampleInWarningWindow,
    NoValidSamples,
}

/// Per-shot audit record. A missing prediction means that this shot was not scoreable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShotEvaluationRecord {
    pub shot_id: u32,
    pub actual_disruption: bool,
    pub predicted_disruption: Option<bool>,
    pub warning_lead_time_ms: Option<f64>,
    pub sample_count: usize,
    pub valid_sample_count: usize,
    /// Zero-based indices into the original shot sample sequence.
    pub invalid_sample_indices: Vec<usize>,
    pub unscored_reason: Option<ShotUnscoredReason>,
}

/// Evaluate a threshold configuration with event-level, fail-closed scoring.
fn evaluate_config(config: &ThresholdConfig, shots: &[CModShot]) -> EvaluationResults {
    let label_config = LabelConfig {
        warning_window_ms: WARNING_WINDOW_MS,
        critical_window_ms: CRITICAL_WINDOW_MS,
    };

    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut true_negatives = 0;
    let mut false_negatives = 0;
    let mut warning_times_ms = Vec::new();
    let mut total_samples = 0;
    let mut invalid_samples = 0;
    let mut scoreable_disruption_shots = 0;
    let mut scoreable_non_disruption_shots = 0;
    let mut unscored_disruption_shots = 0;
    let mut unscored_non_disruption_shots = 0;
    let mut total_disruption_shots = 0;
    let mut total_non_disruption_shots = 0;
    let mut shot_records = Vec::with_capacity(shots.len());

    for shot in shots {
        if shot.disrupted {
            total_disruption_shots += 1;
        } else {
            total_non_disruption_shots += 1;
        }

        let labels = label_samples(
            shot,
            label_config.warning_window_ms,
            label_config.critical_window_ms,
        );
        let mut has_valid_sample = false;
        let mut valid_sample_count = 0;
        let mut invalid_sample_indices = Vec::new();
        let mut has_valid_warning_sample = false;
        let mut alarm_in_warning_window = false;
        let mut alarm_in_normal_operation = false;
        // The largest time-to-disruption among matching alarms is the earliest lead-time alarm.
        let mut earliest_warning_ttd: Option<f64> = None;

        for (sample_index, (sample, label)) in
            shot.samples.iter().zip(labels.iter()).enumerate()
        {
            total_samples += 1;
            if !has_valid_phi_inputs(sample) {
                invalid_samples += 1;
                invalid_sample_indices.push(sample_index);
                continue;
            }
            has_valid_sample = true;
            valid_sample_count += 1;

            let phi_proxy = compute_phi_proxy(sample);
            let regime = classify_regime(phi_proxy, config);
            let is_warning_or_critical = matches!(
                regime,
                StabilityRegime::Warning | StabilityRegime::Critical | StabilityRegime::Emergency
            );

            if shot.disrupted {
                if matches!(label, DisruptionLabel::Warning | DisruptionLabel::Critical) {
                    has_valid_warning_sample = true;
                    if is_warning_or_critical {
                        alarm_in_warning_window = true;
                        if let Some(ttd) = sample
                            .time_to_disruption_ms
                            .filter(|ttd| ttd.is_finite() && *ttd >= 0.0)
                        {
                            earliest_warning_ttd = Some(
                                earliest_warning_ttd.map_or(ttd, |current| current.max(ttd)),
                            );
                        }
                    }
                }
            } else if is_warning_or_critical && *label == DisruptionLabel::Normal {
                // All alarms within a non-disruption shot collapse to one event-level FP.
                alarm_in_normal_operation = true;
            }
        }

        if shot.disrupted {
            if !has_valid_warning_sample {
                unscored_disruption_shots += 1;
                shot_records.push(ShotEvaluationRecord {
                    shot_id: shot.shot_id,
                    actual_disruption: true,
                    predicted_disruption: None,
                    warning_lead_time_ms: None,
                    sample_count: shot.samples.len(),
                    valid_sample_count,
                    invalid_sample_indices: invalid_sample_indices.clone(),
                    unscored_reason: Some(ShotUnscoredReason::NoValidSampleInWarningWindow),
                });
                continue;
            }
            scoreable_disruption_shots += 1;
            let detected = alarm_in_warning_window;
            let lead_time = if detected { earliest_warning_ttd } else { None };
            if let Some(ttd) = lead_time {
                true_positives += 1;
                warning_times_ms.push(ttd);
            } else {
                false_negatives += 1;
            }
            shot_records.push(ShotEvaluationRecord {
                shot_id: shot.shot_id,
                actual_disruption: true,
                predicted_disruption: Some(detected),
                warning_lead_time_ms: lead_time,
                sample_count: shot.samples.len(),
                valid_sample_count,
                invalid_sample_indices: invalid_sample_indices.clone(),
                unscored_reason: None,
            });
        } else if !has_valid_sample {
            unscored_non_disruption_shots += 1;
            shot_records.push(ShotEvaluationRecord {
                shot_id: shot.shot_id,
                actual_disruption: false,
                predicted_disruption: None,
                warning_lead_time_ms: None,
                sample_count: shot.samples.len(),
                valid_sample_count,
                invalid_sample_indices: invalid_sample_indices.clone(),
                unscored_reason: Some(ShotUnscoredReason::NoValidSamples),
            });
        } else {
            scoreable_non_disruption_shots += 1;
            let predicted_disruption = alarm_in_normal_operation;
            if predicted_disruption {
                false_positives += 1;
            } else {
                true_negatives += 1;
            }
            shot_records.push(ShotEvaluationRecord {
                shot_id: shot.shot_id,
                actual_disruption: false,
                predicted_disruption: Some(predicted_disruption),
                warning_lead_time_ms: None,
                sample_count: shot.samples.len(),
                valid_sample_count,
                invalid_sample_indices: invalid_sample_indices.clone(),
                unscored_reason: None,
            });
        }
    }

    warning_times_ms.sort_by(f64::total_cmp);
    let mean_warning_time_ms = if warning_times_ms.is_empty() {
        0.0
    } else {
        warning_times_ms.iter().sum::<f64>() / warning_times_ms.len() as f64
    };
    let median_warning_time_ms = if warning_times_ms.is_empty() {
        0.0
    } else {
        let mid = warning_times_ms.len() / 2;
        if warning_times_ms.len() % 2 == 0 {
            (warning_times_ms[mid - 1] + warning_times_ms[mid]) / 2.0
        } else {
            warning_times_ms[mid]
        }
    };
    let min_warning_time_ms = warning_times_ms.first().copied().unwrap_or(0.0);

    EvaluationResults {
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
        mean_warning_time_ms,
        median_warning_time_ms,
        min_warning_time_ms,
        total_samples,
        invalid_samples,
        scoreable_disruption_shots,
        scoreable_non_disruption_shots,
        unscored_disruption_shots,
        unscored_non_disruption_shots,
        total_disruption_shots,
        total_non_disruption_shots,
        shot_records,
    }
}

/// Compute a Phi proxy from sample sensor values
fn compute_phi_proxy(sample: &symthaea::physics::cmod_adapter::CModSample) -> f64 {
    // Normalize key indicators to [0, 1] and combine
    // Higher Phi = more stable plasma

    // Plasma current stability (deviation from nominal)
    let ip_norm = (sample.ip as f64 / 1.0).clamp(0.0, 2.0) / 2.0; // Normalize to 1 MA nominal

    // Temperature stability
    let te_norm = ((sample.te as f64 - 0.5) / 5.0).clamp(0.0, 1.0); // Cold plasma = bad

    // Density limit proximity (lower is better for stability)
    let ne_norm = 1.0 - (sample.ne as f64 / 4.0).clamp(0.0, 1.0);

    // Safety factor (q95 > 2 is good)
    let q95_norm = ((sample.q95 as f64 - 1.5) / 3.5).clamp(0.0, 1.0);

    // Radiation fraction (lower is better)
    let prad_norm = 1.0 - (sample.prad as f64 / 5.0).clamp(0.0, 1.0);

    // Combine with weights
    let phi = 0.25 * ip_norm + 0.25 * te_norm + 0.2 * ne_norm + 0.15 * q95_norm + 0.15 * prad_norm;

    phi.clamp(0.0, 1.0)
}

/// Classify regime based on Phi proxy and thresholds
fn classify_regime(phi: f64, config: &ThresholdConfig) -> StabilityRegime {
    let phi = phi as f32;
    if phi >= config.phi_stable {
        StabilityRegime::Stable
    } else if phi >= config.phi_warning {
        StabilityRegime::Warning
    } else if phi >= config.phi_critical {
        StabilityRegime::Critical
    } else {
        StabilityRegime::Emergency
    }
}

// =============================================================================
// OBJECTIVE FUNCTION
// =============================================================================

/// Objective function for optimization
/// Returns negative infinity if constraint violated (infeasible)
fn objective(config: &ThresholdConfig, shots: &[CModShot]) -> f64 {
    let results = evaluate_config(config, shots);

    let Some(recall) = results.recall() else {
        return f64::NEG_INFINITY;
    };
    if recall < 0.95 {
        return f64::NEG_INFINITY;
    }
    let Some(false_positive_rate) = results.false_positive_rate() else {
        return f64::NEG_INFINITY;
    }

    // Synthetic-example objective only; this is not a validated operational safety criterion.
    results.mean_warning_time_ms - 100.0 * false_positive_rate
}

// =============================================================================
// SEARCH METHODS
// =============================================================================

/// Result of a search method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Best configuration found
    pub best_config: ThresholdConfig,
    /// Best objective value
    pub best_objective: f64,
    /// Evaluation results for best config
    pub best_results: EvaluationResults,
    /// Number of configurations evaluated
    pub configs_evaluated: usize,
    /// Search time in milliseconds
    pub search_time_ms: u64,
    /// Objective values history (for learning curve)
    pub objective_history: Vec<f64>,
}

/// Grid search over all parameter combinations
fn grid_search(shots: &[CModShot]) -> SearchResult {
    let start = Instant::now();
    let levels: usize = 5;

    let mut best_config = ThresholdConfig::default();
    let mut best_objective = f64::NEG_INFINITY;
    let mut best_results = evaluate_config(&best_config, shots);
    let mut configs_evaluated = 0;
    let mut objective_history = Vec::new();

    println!(
        "  Running grid search ({} total configurations)...",
        (levels as u32).pow(5)
    );

    for i in 0..levels {
        for j in 0..levels {
            for k in 0..levels {
                for l in 0..levels {
                    for m in 0..levels {
                        if let Some(config) = ThresholdConfig::from_grid_indices(i, j, k, l, m) {
                            configs_evaluated += 1;
                            let obj = objective(&config, shots);
                            objective_history.push(obj);

                            if obj > best_objective {
                                best_objective = obj;
                                best_config = config;
                                best_results = evaluate_config(&config, shots);
                            }
                        }
                    }
                }
            }
        }
        // Progress indicator
        print!(".");
        std::io::stdout().flush().unwrap();
    }
    println!();

    SearchResult {
        best_config,
        best_objective,
        best_results,
        configs_evaluated,
        search_time_ms: start.elapsed().as_millis() as u64,
        objective_history,
    }
}

/// Random search with specified number of samples
fn random_search(shots: &[CModShot], n_samples: usize, seed: u64) -> SearchResult {
    let start = Instant::now();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut best_config = ThresholdConfig::default();
    let mut best_objective = f64::NEG_INFINITY;
    let mut best_results = evaluate_config(&best_config, shots);
    let mut objective_history = Vec::new();

    println!("  Running random search ({} samples)...", n_samples);

    for i in 0..n_samples {
        let config = ThresholdConfig::random(&mut rng);
        let obj = objective(&config, shots);
        objective_history.push(obj);

        if obj > best_objective {
            best_objective = obj;
            best_config = config;
            best_results = evaluate_config(&config, shots);
        }

        // Progress indicator every 50 samples
        if (i + 1) % 50 == 0 {
            print!(".");
            std::io::stdout().flush().unwrap();
        }
    }
    println!();

    SearchResult {
        best_config,
        best_objective,
        best_results,
        configs_evaluated: n_samples,
        search_time_ms: start.elapsed().as_millis() as u64,
        objective_history,
    }
}

/// Simplified Bayesian optimization using random search with adaptive sampling
/// (Full GP implementation would require additional dependencies)
fn bayesian_optimization(shots: &[CModShot], n_iterations: usize, seed: u64) -> SearchResult {
    let start = Instant::now();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Store evaluated points
    let mut evaluated: Vec<(ThresholdConfig, f64)> = Vec::new();
    let mut best_config = ThresholdConfig::default();
    let mut best_objective = f64::NEG_INFINITY;
    let mut best_results = evaluate_config(&best_config, shots);
    let mut objective_history = Vec::new();

    println!(
        "  Running Bayesian-inspired optimization ({} iterations)...",
        n_iterations
    );

    // Initial random exploration (20% of budget)
    let explore_budget = n_iterations / 5;
    for _ in 0..explore_budget {
        let config = ThresholdConfig::random(&mut rng);
        let obj = objective(&config, shots);
        evaluated.push((config, obj));
        objective_history.push(obj);

        if obj > best_objective {
            best_objective = obj;
            best_config = config;
            best_results = evaluate_config(&config, shots);
        }
    }
    print!(".");
    std::io::stdout().flush().unwrap();

    // Exploitation with local search around best points
    for i in explore_budget..n_iterations {
        // Select a good point to explore around (top 20%)
        let mut sorted_evals: Vec<_> = evaluated.iter().filter(|(_, o)| o.is_finite()).collect();
        sorted_evals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let base_config = if sorted_evals.is_empty() {
            ThresholdConfig::random(&mut rng)
        } else {
            // Pick from top quartile with some randomness
            let top_k = (sorted_evals.len() / 4).max(1);
            let idx = rng.r#gen_range(0..top_k);
            sorted_evals[idx].0
        };

        // Perturb around the base configuration
        let perturbation = 0.1 * (1.0 - (i as f32 / n_iterations as f32)); // Decay perturbation
        let config = perturb_config(&base_config, perturbation, &mut rng);

        let obj = objective(&config, shots);
        evaluated.push((config, obj));
        objective_history.push(obj);

        if obj > best_objective {
            best_objective = obj;
            best_config = config;
            best_results = evaluate_config(&config, shots);
        }

        // Progress indicator
        if (i + 1) % 25 == 0 {
            print!(".");
            std::io::stdout().flush().unwrap();
        }
    }
    println!();

    SearchResult {
        best_config,
        best_objective,
        best_results,
        configs_evaluated: n_iterations,
        search_time_ms: start.elapsed().as_millis() as u64,
        objective_history,
    }
}

/// Perturb a configuration by a random amount within bounds
fn perturb_config(base: &ThresholdConfig, scale: f32, rng: &mut impl Rng) -> ThresholdConfig {
    loop {
        let mut perturb = |val: f32, bounds: (f32, f32)| -> f32 {
            let range = bounds.1 - bounds.0;
            let delta = (rng.r#gen::<f32>() - 0.5) * 2.0 * scale * range;
            (val + delta).clamp(bounds.0, bounds.1)
        };

        let phi_stable = perturb(base.phi_stable, ThresholdConfig::PHI_STABLE_BOUNDS);
        let phi_warning = perturb(base.phi_warning, ThresholdConfig::PHI_WARNING_BOUNDS);
        let phi_critical = perturb(base.phi_critical, ThresholdConfig::PHI_CRITICAL_BOUNDS);
        let phi_rate = perturb(base.phi_rate_threshold, ThresholdConfig::PHI_RATE_BOUNDS);
        let volatility = perturb(
            base.volatility_threshold,
            ThresholdConfig::VOLATILITY_BOUNDS,
        );

        if let Some(config) =
            ThresholdConfig::new(phi_stable, phi_warning, phi_critical, phi_rate, volatility)
        {
            return config;
        }
    }
}

// =============================================================================
// CROSS-VALIDATION
// =============================================================================

/// Cross-validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResults {
    /// Best config from training set
    pub best_config: ThresholdConfig,
    /// Train set objective
    pub train_objective: f64,
    /// Validation set objective
    pub val_objective: f64,
    /// Generalization gap (train - val)
    pub generalization_gap: f64,
    /// Train results
    pub train_results: EvaluationResults,
    /// Validation results
    pub val_results: EvaluationResults,
}

/// Split shots into train/val sets (80/20)
fn train_val_split(shots: &[CModShot], seed: u64) -> (Vec<CModShot>, Vec<CModShot>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..shots.len()).collect();

    // Fisher-Yates shuffle
    for i in (1..indices.len()).rev() {
        let j = rng.r#gen_range(0..=i);
        indices.swap(i, j);
    }

    let split_point = (shots.len() as f64 * 0.8) as usize;
    let train_indices = &indices[..split_point];
    let val_indices = &indices[split_point..];

    let train: Vec<CModShot> = train_indices.iter().map(|&i| shots[i].clone()).collect();
    let val: Vec<CModShot> = val_indices.iter().map(|&i| shots[i].clone()).collect();

    (train, val)
}

/// Run cross-validation
fn cross_validate(shots: &[CModShot], seed: u64) -> CrossValidationResults {
    println!("\n  Splitting data (80% train / 20% val)...");
    let (train_shots, val_shots) = train_val_split(shots, seed);
    println!(
        "    Train: {} shots, Val: {} shots",
        train_shots.len(),
        val_shots.len()
    );

    println!("  Optimizing on training set...");
    let search_result = random_search(&train_shots, 300, seed);

    let train_objective = search_result.best_objective;
    let val_objective = objective(&search_result.best_config, &val_shots);
    let val_results = evaluate_config(&search_result.best_config, &val_shots);

    CrossValidationResults {
        best_config: search_result.best_config,
        train_objective,
        val_objective,
        generalization_gap: train_objective - val_objective,
        train_results: search_result.best_results,
        val_results,
    }
}

// =============================================================================
// PARETO FRONTIER
// =============================================================================

/// A point on the Pareto frontier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub config: ThresholdConfig,
    pub warning_time_ms: f64,
    pub false_positive_rate: f64,
    pub recall: f64,
}

/// Compute Pareto frontier for warning time vs FPR
fn compute_pareto_frontier(shots: &[CModShot], n_samples: usize, seed: u64) -> Vec<ParetoPoint> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut points: Vec<ParetoPoint> = Vec::new();

    println!(
        "  Sampling {} configurations for Pareto analysis...",
        n_samples
    );

    for _ in 0..n_samples {
        let config = ThresholdConfig::random(&mut rng);
        let results = evaluate_config(&config, shots);

        // Undefined/degenerate metrics cannot qualify as feasible Pareto points.
        let (Some(recall), Some(false_positive_rate)) =
            (results.recall(), results.false_positive_rate())
        else {
            continue;
        };
        if recall >= 0.95 {
            points.push(ParetoPoint {
                config,
                warning_time_ms: results.mean_warning_time_ms,
                false_positive_rate,
                recall,
            });
        }
    }

    // Filter to Pareto optimal points
    // A point is Pareto optimal if no other point dominates it
    // (higher warning time AND lower FPR)
    let mut pareto: Vec<ParetoPoint> = Vec::new();

    for point in &points {
        let is_dominated = points.iter().any(|other| {
            other.warning_time_ms > point.warning_time_ms
                && other.false_positive_rate < point.false_positive_rate
        });

        if !is_dominated {
            pareto.push(point.clone());
        }
    }

    // Sort by warning time
    pareto.sort_by(|a, b| a.warning_time_ms.partial_cmp(&b.warning_time_ms).unwrap());

    pareto
}

// =============================================================================
// VISUALIZATION
// =============================================================================

/// Print ASCII learning curve
fn print_learning_curve(history: &[f64], title: &str) {
    println!("\n{}", title);
    println!("{}", "-".repeat(70));

    // Filter finite values and compute running best
    let mut running_best = f64::NEG_INFINITY;
    let running_bests: Vec<f64> = history
        .iter()
        .map(|&v| {
            if v.is_finite() && v > running_best {
                running_best = v;
            }
            running_best
        })
        .filter(|v| v.is_finite())
        .collect();

    if running_bests.is_empty() {
        println!("  No feasible solutions found");
        return;
    }

    let min_val = running_bests.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = running_bests
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let range = (max_val - min_val).max(1.0);

    let chart_width = 60;
    let chart_height = 12;
    let mut chart = vec![vec![' '; chart_width]; chart_height];

    // Plot running best
    for (i, &val) in running_bests.iter().enumerate() {
        let x = (i as f64 / running_bests.len() as f64 * (chart_width - 1) as f64) as usize;
        let y_norm = (val - min_val) / range;
        let y = ((1.0 - y_norm) * (chart_height - 1) as f64) as usize;
        chart[y.min(chart_height - 1)][x.min(chart_width - 1)] = '*';
    }

    // Print chart
    println!("Objective ^");
    for (i, row) in chart.iter().enumerate() {
        let y_val = max_val - (i as f64 / (chart_height - 1) as f64) * range;
        if i == 0 || i == chart_height - 1 || i == chart_height / 2 {
            print!("{:>8.1} |", y_val);
        } else {
            print!("         |");
        }
        for c in row {
            print!("{}", c);
        }
        println!("|");
    }
    println!("         +{}+", "-".repeat(chart_width));
    println!(
        "          0{}{} Iterations",
        " ".repeat(chart_width / 2 - 5),
        running_bests.len() / 2
    );
}

/// Print Pareto frontier as ASCII scatter plot
fn print_pareto_frontier(pareto: &[ParetoPoint]) {
    println!("\nPARETO FRONTIER (Warning Time vs False Positive Rate)");
    println!("{}", "-".repeat(70));

    if pareto.is_empty() {
        println!("  No Pareto-optimal points found");
        return;
    }

    let min_wt = pareto
        .iter()
        .map(|p| p.warning_time_ms)
        .fold(f64::INFINITY, f64::min);
    let max_wt = pareto
        .iter()
        .map(|p| p.warning_time_ms)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_fpr = pareto
        .iter()
        .map(|p| p.false_positive_rate)
        .fold(f64::INFINITY, f64::min);
    let max_fpr = pareto
        .iter()
        .map(|p| p.false_positive_rate)
        .fold(f64::NEG_INFINITY, f64::max);

    let wt_range = (max_wt - min_wt).max(1.0);
    let fpr_range = (max_fpr - min_fpr).max(0.01);

    let chart_width = 50;
    let chart_height = 15;
    let mut chart = vec![vec!['.'; chart_width]; chart_height];

    // Plot points (inverted Y axis: low FPR at top)
    for (i, point) in pareto.iter().enumerate() {
        let x = ((point.warning_time_ms - min_wt) / wt_range * (chart_width - 1) as f64) as usize;
        let y = ((point.false_positive_rate - min_fpr) / fpr_range * (chart_height - 1) as f64)
            as usize;
        let ch = if i < 10 {
            char::from_digit(i as u32, 10).unwrap()
        } else {
            '#'
        };
        chart[y.min(chart_height - 1)][x.min(chart_width - 1)] = ch;
    }

    // Print chart
    println!("FPR ^");
    for (i, row) in chart.iter().enumerate() {
        let y_val = min_fpr + (i as f64 / (chart_height - 1) as f64) * fpr_range;
        if i == 0 || i == chart_height - 1 {
            print!("{:>6.3} |", y_val);
        } else {
            print!("       |");
        }
        for c in row {
            print!("{}", c);
        }
        println!("|");
    }
    println!("       +{}+", "-".repeat(chart_width));
    println!(
        "        {:.0}{}{}ms Warning Time",
        min_wt,
        " ".repeat(chart_width - 10),
        max_wt
    );

    // Print point details
    println!("\nPareto-optimal configurations:");
    println!(
        "{:<4} {:>12} {:>10} {:>10}",
        "Pt", "Warning(ms)", "FPR", "Recall"
    );
    println!("{}", "-".repeat(40));
    for (i, point) in pareto.iter().take(10).enumerate() {
        println!(
            "{:<4} {:>12.1} {:>10.4} {:>10.4}",
            i, point.warning_time_ms, point.false_positive_rate, point.recall
        );
    }
}

// =============================================================================
// SAVE RESULTS
// =============================================================================

/// Machine-readable provenance for the corpus used by an optimization run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DatasetProvenance {
    Synthetic {
        generator: String,
        seed: u64,
        num_shots: usize,
        disruption_probability: f32,
        samples_per_shot: usize,
        sample_interval_ms: f64,
        warning_window_ms: f64,
        critical_window_ms: f64,
        time_alignment: String,
        label_definition: String,
        generator_version: String,
        source_revision: String,
        qualification_note: String,
    },
    Measured {
        dataset_id: String,
        dataset_version: String,
        content_sha256: String,
        time_alignment: String,
        label_definition: String,
    },
}

/// Complete optimization output. Schema version 1 is the first explicitly versioned form of
/// this previously unversioned example output.
#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizationOutput {
    pub schema_version: u32,
    pub dataset_provenance: DatasetProvenance,
    pub grid_search: Option<SearchResult>,
    pub random_search: Option<SearchResult>,
    pub bayesian_search: Option<SearchResult>,
    pub cross_validation: Option<CrossValidationResults>,
    pub pareto_frontier: Vec<ParetoPoint>,
    pub best_overall: ThresholdConfig,
    pub best_overall_objective: f64,
}

fn save_results(output: &OptimizationOutput, path: &Path) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(output)?;
    fs::create_dir_all(path.parent().unwrap())?;
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("  PLASMA THRESHOLD TUNING OPTIMIZER");
    println!("  Optimizing disruption prediction thresholds");
    println!("{}\n", "=".repeat(80));

    // =========================================================================
    // Generate synthetic data
    // =========================================================================

    println!("GENERATING SYNTHETIC DATA");
    println!("{}", "-".repeat(50));

    let synth_config = SyntheticConfig {
        num_shots: 200,
        disruption_probability: 0.35,
        samples_per_shot: 150,
        sample_interval_ms: 1.0,
        seed: 42,
    };

    let shots = generate_synthetic_data(&synth_config);
    let disrupted_count = shots.iter().filter(|s| s.disrupted).count();
    println!(
        "  Generated {} shots ({} disrupted, {} non-disrupted)",
        shots.len(),
        disrupted_count,
        shots.len() - disrupted_count
    );

    // =========================================================================
    // Baseline evaluation
    // =========================================================================

    println!("\n\nBASELINE EVALUATION (Default Thresholds)");
    println!("{}", "-".repeat(50));

    let baseline_config = ThresholdConfig::default();
    let baseline_results = evaluate_config(&baseline_config, &shots);
    let baseline_objective = objective(&baseline_config, &shots);

    println!(
        "  Phi thresholds: stable={:.2}, warning={:.2}, critical={:.2}",
        baseline_config.phi_stable, baseline_config.phi_warning, baseline_config.phi_critical
    );
    println!("  Event-level Recall: {}", metric_display(baseline_results.recall()));
    println!("  Event-level Precision: {}", metric_display(baseline_results.precision()));
    println!("  Event-level False Positive Rate: {}", metric_display(baseline_results.false_positive_rate()));
    println!(
        "  Scoreable shots: {} disruption, {} non-disruption; unscored: {} disruption, {} non-disruption; invalid samples: {}",
        baseline_results.scoreable_disruption_shots,
        baseline_results.scoreable_non_disruption_shots,
        baseline_results.unscored_disruption_shots,
        baseline_results.unscored_non_disruption_shots,
        baseline_results.invalid_samples
    );
    println!(
        "  Mean Warning Time: {:.1} ms",
        baseline_results.mean_warning_time_ms
    );
    println!("  Objective:        {:.2}", baseline_objective);

    // =========================================================================
    // Grid Search
    // =========================================================================

    println!("\n\nSEARCH METHOD 1: GRID SEARCH");
    println!("{}", "-".repeat(50));

    let grid_result = grid_search(&shots);

    println!("\n  Best configuration found:");
    println!(
        "    phi_stable:    {:.3}",
        grid_result.best_config.phi_stable
    );
    println!(
        "    phi_warning:   {:.3}",
        grid_result.best_config.phi_warning
    );
    println!(
        "    phi_critical:  {:.3}",
        grid_result.best_config.phi_critical
    );
    println!(
        "    phi_rate:      {:.3}",
        grid_result.best_config.phi_rate_threshold
    );
    println!(
        "    volatility:    {:.3}",
        grid_result.best_config.volatility_threshold
    );
    println!("  Objective:       {:.2}", grid_result.best_objective);
    println!(
        "  Event-level Recall: {}",
        metric_display(grid_result.best_results.recall())
    );
    println!(
        "  Event-level FPR: {}",
        metric_display(grid_result.best_results.false_positive_rate())
    );
    println!(
        "  Warning Time:    {:.1} ms",
        grid_result.best_results.mean_warning_time_ms
    );
    println!("  Configs evaluated: {}", grid_result.configs_evaluated);
    println!("  Search time:     {} ms", grid_result.search_time_ms);

    // =========================================================================
    // Random Search
    // =========================================================================

    println!("\n\nSEARCH METHOD 2: RANDOM SEARCH");
    println!("{}", "-".repeat(50));

    let random_result = random_search(&shots, 500, 12345);

    println!("\n  Best configuration found:");
    println!(
        "    phi_stable:    {:.3}",
        random_result.best_config.phi_stable
    );
    println!(
        "    phi_warning:   {:.3}",
        random_result.best_config.phi_warning
    );
    println!(
        "    phi_critical:  {:.3}",
        random_result.best_config.phi_critical
    );
    println!(
        "    phi_rate:      {:.3}",
        random_result.best_config.phi_rate_threshold
    );
    println!(
        "    volatility:    {:.3}",
        random_result.best_config.volatility_threshold
    );
    println!("  Objective:       {:.2}", random_result.best_objective);
    println!(
        "  Event-level Recall: {}",
        metric_display(random_result.best_results.recall())
    );
    println!(
        "  Event-level FPR: {}",
        metric_display(random_result.best_results.false_positive_rate())
    );
    println!(
        "  Warning Time:    {:.1} ms",
        random_result.best_results.mean_warning_time_ms
    );
    println!("  Configs evaluated: {}", random_result.configs_evaluated);
    println!("  Search time:     {} ms", random_result.search_time_ms);

    print_learning_curve(
        &random_result.objective_history,
        "Random Search Learning Curve (Running Best)",
    );

    // =========================================================================
    // Bayesian Optimization
    // =========================================================================

    println!("\n\nSEARCH METHOD 3: BAYESIAN-INSPIRED OPTIMIZATION");
    println!("{}", "-".repeat(50));

    let bayesian_result = bayesian_optimization(&shots, 200, 54321);

    println!("\n  Best configuration found:");
    println!(
        "    phi_stable:    {:.3}",
        bayesian_result.best_config.phi_stable
    );
    println!(
        "    phi_warning:   {:.3}",
        bayesian_result.best_config.phi_warning
    );
    println!(
        "    phi_critical:  {:.3}",
        bayesian_result.best_config.phi_critical
    );
    println!(
        "    phi_rate:      {:.3}",
        bayesian_result.best_config.phi_rate_threshold
    );
    println!(
        "    volatility:    {:.3}",
        bayesian_result.best_config.volatility_threshold
    );
    println!("  Objective:       {:.2}", bayesian_result.best_objective);
    println!(
        "  Event-level Recall: {}",
        metric_display(bayesian_result.best_results.recall())
    );
    println!(
        "  Event-level FPR: {}",
        metric_display(bayesian_result.best_results.false_positive_rate())
    );
    println!(
        "  Warning Time:    {:.1} ms",
        bayesian_result.best_results.mean_warning_time_ms
    );
    println!("  Configs evaluated: {}", bayesian_result.configs_evaluated);
    println!("  Search time:     {} ms", bayesian_result.search_time_ms);

    print_learning_curve(
        &bayesian_result.objective_history,
        "Bayesian Optimization Learning Curve (Running Best)",
    );

    // =========================================================================
    // Cross-Validation
    // =========================================================================

    println!("\n\nCROSS-VALIDATION");
    println!("{}", "-".repeat(50));

    let cv_results = cross_validate(&shots, 99999);

    println!("\n  Train Objective:    {:.2}", cv_results.train_objective);
    println!("  Val Objective:      {:.2}", cv_results.val_objective);
    println!("  Generalization Gap: {:.2}", cv_results.generalization_gap);
    println!("\n  Train Metrics:");
    println!("    Event-level Recall: {}", metric_display(cv_results.train_results.recall()));
    println!("    Event-level FPR: {}", metric_display(cv_results.train_results.false_positive_rate()));
    println!(
        "    Warning Time: {:.1} ms",
        cv_results.train_results.mean_warning_time_ms
    );
    println!("\n  Validation Metrics:");
    println!("    Event-level Recall: {}", metric_display(cv_results.val_results.recall()));
    println!("    Event-level FPR: {}", metric_display(cv_results.val_results.false_positive_rate()));
    println!(
        "    Warning Time: {:.1} ms",
        cv_results.val_results.mean_warning_time_ms
    );

    // =========================================================================
    // Pareto Frontier
    // =========================================================================

    println!("\n\nPARETO FRONTIER ANALYSIS");
    println!("{}", "-".repeat(50));

    let pareto = compute_pareto_frontier(&shots, 1000, 11111);
    print_pareto_frontier(&pareto);

    // =========================================================================
    // Method Comparison
    // =========================================================================

    println!("\n\nSEARCH METHOD COMPARISON");
    println!("{}", "-".repeat(70));
    println!(
        "{:<25} {:>12} {:>12} {:>12} {:>10}",
        "Method", "Objective", "Recall", "FPR", "Time(ms)"
    );
    println!("{}", "-".repeat(70));
    println!(
        "{:<25} {:>12.2} {:>12} {:>12} {:>10}",
        "Baseline (Default)", baseline_objective,
        metric_display(baseline_results.recall()),
        metric_display(baseline_results.false_positive_rate()), "-"
    );
    println!(
        "{:<25} {:>12.2} {:>12} {:>12} {:>10}",
        "Grid Search", grid_result.best_objective,
        metric_display(grid_result.best_results.recall()),
        metric_display(grid_result.best_results.false_positive_rate()),
        grid_result.search_time_ms
    );
    println!(
        "{:<25} {:>12.2} {:>12} {:>12} {:>10}",
        "Random Search (500)", random_result.best_objective,
        metric_display(random_result.best_results.recall()),
        metric_display(random_result.best_results.false_positive_rate()),
        random_result.search_time_ms
    );
    println!(
        "{:<25} {:>12.2} {:>12} {:>12} {:>10}",
        "Bayesian (200)", bayesian_result.best_objective,
        metric_display(bayesian_result.best_results.recall()),
        metric_display(bayesian_result.best_results.false_positive_rate()),
        bayesian_result.search_time_ms
    );

    // =========================================================================
    // Select Best Overall
    // =========================================================================

    // Find best overall configuration by objective value
    let candidates = [
        (grid_result.best_config, grid_result.best_objective),
        (random_result.best_config, random_result.best_objective),
        (bayesian_result.best_config, bayesian_result.best_objective),
    ];
    let selected = candidates
        .into_iter()
        .filter(|(_, candidate_objective)| candidate_objective.is_finite())
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let Some((best_overall, best_overall_obj)) = selected else {
        println!(
            "No scoreable, feasible threshold configuration was found. Refusing to serialize or present default thresholds as an optimizer winner."
        );
        return;
    };

    println!("\n\nBEST OVERALL CONFIGURATION");
    println!("{}", "-".repeat(50));
    println!("  phi_stable:    {:.3}", best_overall.phi_stable);
    println!("  phi_warning:   {:.3}", best_overall.phi_warning);
    println!("  phi_critical:  {:.3}", best_overall.phi_critical);
    println!("  phi_rate:      {:.3}", best_overall.phi_rate_threshold);
    println!("  volatility:    {:.3}", best_overall.volatility_threshold);
    println!("  Objective:     {:.2}", best_overall_obj);

    // =========================================================================
    // Save Results
    // =========================================================================

    println!("\n\nSAVING RESULTS");
    println!("{}", "-".repeat(50));

    let output = OptimizationOutput {
        schema_version: 1,
        dataset_provenance: DatasetProvenance::Synthetic {
            generator: "symthaea::physics::cmod_adapter::generate_synthetic_data".to_string(),
            seed: synth_config.seed,
            num_shots: synth_config.num_shots,
            disruption_probability: synth_config.disruption_probability,
            samples_per_shot: synth_config.samples_per_shot,
            sample_interval_ms: synth_config.sample_interval_ms,
            warning_window_ms: WARNING_WINDOW_MS,
            critical_window_ms: CRITICAL_WINDOW_MS,
            time_alignment: "sample time is milliseconds from shot start; time-to-disruption is disruption_time_ms minus sample.time_ms".to_string(),
            label_definition: "Warning when 20 ms < time-to-disruption <= 100 ms; Critical when 0 <= time-to-disruption <= 20 ms; post-disruption samples are not detections".to_string(),
            generator_version: "synthetic-cmod-generator-v1".to_string(),
            source_revision: option_env!("GIT_COMMIT_SHA").unwrap_or("unrecorded-at-build").to_string(),
            qualification_note: "Synthetic demonstration only; not measured evidence and not qualified for device operation or safety decisions.".to_string(),
        },
        grid_search: Some(grid_result),
        random_search: Some(random_result),
        bayesian_search: Some(bayesian_result),
        cross_validation: Some(cv_results),
        pareto_frontier: pareto,
        best_overall,
        best_overall_objective: best_overall_obj,
    };

    let data_path = Path::new("data/optimized_thresholds.json");
    match save_results(&output, data_path) {
        Ok(()) => println!("  Results saved to: {}", data_path.display()),
        Err(e) => println!("  Warning: Failed to save results: {}", e),
    }

    // =========================================================================
    // Apply to PlasmaControlConfig
    // =========================================================================

    println!("\n\nCREATING EXPLORATORY CONFIG (SYNTHETIC DATA ONLY; NO ACTUATION)");
    println!("{}", "-".repeat(50));

    let optimized_plasma_config = best_overall.to_plasma_control_config();
    println!("  Created an in-memory config from synthetic optimization outputs.");
    println!("  This config is unvalidated and must not be used for device operation or safety decisions.");
    println!(
        "    phi_stable_threshold:   {:.3}",
        optimized_plasma_config.phi_stable_threshold
    );
    println!(
        "    phi_warning_threshold:  {:.3}",
        optimized_plasma_config.phi_warning_threshold
    );
    println!(
        "    phi_critical_threshold: {:.3}",
        optimized_plasma_config.phi_critical_threshold
    );
    println!(
        "    rate_of_change_threshold: {:.3}",
        optimized_plasma_config.rate_of_change_threshold
    );
    println!(
        "    volatility_threshold:   {:.3}",
        optimized_plasma_config.volatility_threshold
    );

    // =========================================================================
    // Key Takeaways
    // =========================================================================

    println!("\n\n{}", "=".repeat(80));
    println!("KEY TAKEAWAYS");
    println!("{}", "=".repeat(80));

    let improvement = best_overall_obj - baseline_objective;
    let pct_improvement = if baseline_objective != 0.0 {
        improvement / baseline_objective.abs() * 100.0
    } else {
        0.0
    };

    println!(
        "
1. SYNTHETIC DEMONSTRATION ONLY
   - Baseline objective: {:.2}
   - Best optimized objective: {:.2}
   - Difference: {:.2} ({:+.1}%)
   - These values describe generated data, not a physical device.

2. SEARCH METHODS
   - Grid search: exhaustive over valid grid configurations.
   - Random search: stochastic baseline.
   - Bayesian-inspired search: adaptive random sampling, not full GP Bayesian optimization.

3. CONSTRAINT INTERPRETATION
   - Recall >= 0.95 is checked only on scoreable synthetic shots.
   - This does not establish real-device recall, stability, or safety.

4. EVENT-LEVEL METRICS
   - Confusion counts are per shot, not per sample.
   - Repeated alarms count once per shot; invalid/censored cases are reported separately.

5. NEXT RESEARCH STEP
   - Replace generated data with source-pinned, provenance-verified measured shots.
   - Split evaluation by shot/device/campaign and evaluate on a held-out corpus.
   - Report warning lead-time distributions, alarm burden, censoring, and uncertainty.

6. NOT FOR OPERATION
   - This example does not authorize hardware actuation or device-operation thresholds.
   - Independent validation is required for the exact target regime and diagnostics.
",
        baseline_objective,
        best_overall_obj,
        improvement,
        pct_improvement,
        if output.grid_search.is_some() {
            output.grid_search.as_ref().unwrap().configs_evaluated
        } else {
            0
        }
    );

    println!("{}", "=".repeat(80));
    println!("  Optimization complete!");
    println!("{}\n", "=".repeat(80));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_config_validation() {
        // Valid config
        let valid = ThresholdConfig::new(0.7, 0.5, 0.3, -0.1, 0.2);
        assert!(valid.is_some());

        // Invalid: warning > stable
        let invalid1 = ThresholdConfig::new(0.5, 0.7, 0.3, -0.1, 0.2);
        assert!(invalid1.is_none());

        // Invalid: critical > warning
        let invalid2 = ThresholdConfig::new(0.7, 0.3, 0.5, -0.1, 0.2);
        assert!(invalid2.is_none());
    }

    #[test]
    fn test_threshold_config_random() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..100 {
            let config = ThresholdConfig::random(&mut rng);
            assert!(config.phi_stable > config.phi_warning);
            assert!(config.phi_warning > config.phi_critical);
        }
    }

    #[test]
    fn test_evaluation_results_metrics() {
        let results = EvaluationResults {
            true_positives: 90,
            false_positives: 10,
            true_negatives: 100,
            false_negatives: 5,
            mean_warning_time_ms: 50.0,
            median_warning_time_ms: 45.0,
            min_warning_time_ms: 10.0,
            total_samples: 1000,
            invalid_samples: 0,
            scoreable_disruption_shots: 95,
            scoreable_non_disruption_shots: 110,
            unscored_disruption_shots: 0,
            unscored_non_disruption_shots: 0,
            total_disruption_shots: 95,
            total_non_disruption_shots: 110,
            shot_records: Vec::new(),
        };

        assert!((results.recall().unwrap() - 0.947).abs() < 0.01);
        assert!((results.precision().unwrap() - 0.9).abs() < 0.01);
        assert!((results.false_positive_rate().unwrap() - 0.091).abs() < 0.01);
        assert!(results.f1_score().is_some());
    }

    fn sample_for_test(shot_id: u32, time_ms: f64, unstable: bool) -> symthaea::physics::cmod_adapter::CModSample {
        use symthaea::physics::cmod_adapter::CModSample;
        let mut sample = CModSample::new(shot_id, time_ms);
        if unstable {
            sample.ip = 0.2;
            sample.te = 0.5;
            sample.ne = 4.0;
            sample.q95 = 1.5;
            sample.prad = 4.5;
        } else {
            sample.ip = 1.0;
            sample.te = 5.0;
            sample.ne = 1.5;
            sample.q95 = 3.5;
            sample.prad = 1.0;
        }
        sample
    }

    fn shot_for_test(shot_id: u32, disrupted: bool, sample_specs: &[(f64, bool)]) -> CModShot {
        let mut shot = CModShot::new(shot_id);
        shot.disrupted = disrupted;
        shot.disruption_time_ms = disrupted.then_some(100.0);
        for &(time_ms, unstable) in sample_specs {
            shot.add_sample(sample_for_test(shot_id, time_ms, unstable));
        }
        shot
    }

    #[test]
    fn event_metrics_are_invariant_to_pure_oversampling() {
        // The same negative event contains one alarm in the coarse stream and ten repeated
        // alarm samples in the dense stream. Event-level FP remains exactly one in both.
        let coarse = vec![
            shot_for_test(1, true, &[(10.0, true)]),
            shot_for_test(2, false, &[(0.0, true)]),
        ];
        let dense_negative_samples: Vec<(f64, bool)> =
            (0..40).map(|index| (index as f64, index % 4 == 0)).collect();
        let dense = vec![
            shot_for_test(1, true, &[(0.0, false), (5.0, false), (10.0, true)]),
            shot_for_test(2, false, &dense_negative_samples),
        ];

        let coarse_results = evaluate_config(&ThresholdConfig::default(), &coarse);
        let dense_results = evaluate_config(&ThresholdConfig::default(), &dense);

        for results in [&coarse_results, &dense_results] {
            assert_eq!(results.true_positives, 1);
            assert_eq!(results.false_positives, 1);
            assert_eq!(results.true_negatives, 0);
            assert_eq!(results.false_negatives, 0);
            assert_eq!(results.precision(), Some(0.5));
            assert_eq!(results.false_positive_rate(), Some(1.0));
        }
        assert_eq!(coarse_results.true_positives, dense_results.true_positives);
        assert_eq!(coarse_results.false_positives, dense_results.false_positives);
        assert_eq!(coarse_results.true_negatives, dense_results.true_negatives);
        assert_eq!(coarse_results.false_negatives, dense_results.false_negatives);
        assert_eq!(coarse_results.recall(), dense_results.recall());
        assert_eq!(coarse_results.precision(), dense_results.precision());
        assert_eq!(coarse_results.false_positive_rate(), dense_results.false_positive_rate());
    }

    #[test]
    fn undefined_metrics_are_not_scoreable() {
        let empty: Vec<CModShot> = Vec::new();
        let empty_results = evaluate_config(&ThresholdConfig::default(), &empty);
        assert_eq!(empty_results.recall(), None);
        assert_eq!(empty_results.precision(), None);
        assert_eq!(empty_results.false_positive_rate(), None);
        assert_eq!(empty_results.f1_score(), None);
        assert_eq!(objective(&ThresholdConfig::default(), &empty), f64::NEG_INFINITY);

        let no_disruptions = vec![shot_for_test(1, false, &[(0.0, false)])];
        let results = evaluate_config(&ThresholdConfig::default(), &no_disruptions);
        assert_eq!(results.recall(), None);
        assert_eq!(results.precision(), None);
        assert_eq!(results.f1_score(), None);
        assert_eq!(results.false_positive_rate(), Some(0.0));
        assert_eq!(objective(&ThresholdConfig::default(), &no_disruptions), f64::NEG_INFINITY);

        let false_alarm_only = vec![shot_for_test(4, false, &[(0.0, true)])];
        let results = evaluate_config(&ThresholdConfig::default(), &false_alarm_only);
        assert_eq!(results.recall(), None);
        assert_eq!(results.precision(), Some(0.0));
        assert_eq!(results.f1_score(), None);
        assert_eq!(results.false_positive_rate(), Some(1.0));
        assert_eq!(objective(&ThresholdConfig::default(), &false_alarm_only), f64::NEG_INFINITY);

        let no_negative_examples = vec![shot_for_test(2, true, &[(10.0, true)])];
        let results = evaluate_config(&ThresholdConfig::default(), &no_negative_examples);
        assert_eq!(results.recall(), Some(1.0));
        assert_eq!(results.precision(), Some(1.0));
        assert_eq!(results.false_positive_rate(), None);
        assert_eq!(results.f1_score(), Some(1.0));

        let no_predictions = vec![shot_for_test(3, true, &[(10.0, false)])];
        let results = evaluate_config(&ThresholdConfig::default(), &no_predictions);
        assert_eq!(results.recall(), Some(0.0));
        assert_eq!(results.precision(), None);
        assert_eq!(results.f1_score(), None);
    }

    #[test]
    fn missing_inputs_censor_unobservable_shots_instead_of_scoring_them() {
        let mut shot = CModShot::new(10);
        shot.disrupted = true;
        shot.disruption_time_ms = Some(100.0);
        let mut invalid = sample_for_test(10, 10.0, true);
        invalid.ip = f32::NAN;
        shot.add_sample(invalid);

        let results = evaluate_config(&ThresholdConfig::default(), &[shot]);
        assert_eq!(results.invalid_samples, 1);
        assert_eq!(results.unscored_disruption_shots, 1);
        assert_eq!(results.false_negatives, 0);
        assert_eq!(results.recall(), None);
        assert_eq!(results.shot_records.len(), 1);
        assert_eq!(results.shot_records[0].shot_id, 10);
        assert_eq!(
            results.shot_records[0].unscored_reason,
            Some(ShotUnscoredReason::NoValidSampleInWarningWindow)
        );
        assert_eq!(results.shot_records[0].sample_count, 1);
        assert_eq!(results.shot_records[0].valid_sample_count, 0);
        assert_eq!(results.shot_records[0].invalid_sample_indices, vec![0]);
    }

    #[test]
    fn post_disruption_alarm_does_not_count_as_a_true_positive() {
        let shot = shot_for_test(11, true, &[(10.0, false), (110.0, true)]);
        let results = evaluate_config(&ThresholdConfig::default(), &[shot]);
        assert_eq!(results.true_positives, 0);
        assert_eq!(results.false_negatives, 1);
        assert_eq!(results.recall(), Some(0.0));
    }

    #[test]
    fn repeated_alarms_count_once_and_keep_earliest_lead_time() {
        let shot = shot_for_test(12, true, &[(10.0, true), (15.0, true)]);
        let results = evaluate_config(&ThresholdConfig::default(), &[shot]);
        assert_eq!(results.true_positives, 1);
        assert_eq!(results.false_negatives, 0);
        assert_eq!(results.mean_warning_time_ms, 90.0);
    }

    #[test]
    fn test_grid_indices() {
        // Test that grid index 0,0,0,0,0 gives minimum bounds
        // phi_stable(0.5) > phi_warning(0.3) > phi_critical(0.15) is valid ordering
        let config = ThresholdConfig::from_grid_indices(0, 0, 0, 0, 0);
        assert!(config.is_some()); // min bounds are valid (0.5 > 0.3 > 0.15)

        // Test invalid case: phi_stable(0.5) == phi_warning at max (0.6) when both at low indices
        // Actually, let's test a case where the ordering is violated
        // j=4 gives phi_warning=0.6, which equals phi_stable at i=1 (0.575)
        // Need i=0 (phi_stable=0.5), j=4 (phi_warning=0.6) to get invalid (0.5 < 0.6)
        let config_invalid = ThresholdConfig::from_grid_indices(0, 4, 0, 0, 0);
        assert!(config_invalid.is_none()); // phi_stable(0.5) < phi_warning(0.6) is invalid

        // Test valid indices with clear ordering
        let config = ThresholdConfig::from_grid_indices(4, 2, 0, 2, 2);
        assert!(config.is_some()); // phi_stable(0.8) > phi_warning(0.45) > phi_critical(0.15)
    }

    #[test]
    fn test_phi_proxy() {
        use symthaea::physics::cmod_adapter::CModSample;

        let mut sample = CModSample::new(1, 0.0);
        sample.ip = 1.0;
        sample.te = 5.0;
        sample.ne = 1.5;
        sample.q95 = 3.5;
        sample.prad = 1.0;

        let phi = compute_phi_proxy(&sample);
        assert!(
            phi > 0.3 && phi < 0.9,
            "Stable sample should have moderate-high Phi"
        );

        // Disruption-like sample
        sample.ip = 0.2;
        sample.te = 0.5;
        sample.ne = 4.0;
        sample.q95 = 1.5;
        sample.prad = 4.5;

        let phi_disruption = compute_phi_proxy(&sample);
        assert!(
            phi_disruption < phi,
            "Disruption sample should have lower Phi"
        );
    }
}
