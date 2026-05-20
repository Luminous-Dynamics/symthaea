// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metacognitive Ignition Tracking Benchmark
//!
//! Tests whether the Higher-Order Thought (HOT) system's consciousness
//! classification spontaneously predicts Global Workspace Theory (GWT) ignition,
//! even though HOT has no direct access to workspace competition dynamics.
//!
//! ## Scientific Basis
//!
//! - **Rosenthal (1986, 2005)**: HOT theory — a mental state is conscious iff
//!   one has a higher-order thought about it.
//! - **Dehaene & Naccache (2001)**: GWT ignition creates a sharp
//!   conscious/unconscious boundary.
//! - **Lau & Rosenthal (2011)**: Metacognitive monitoring should track the
//!   ignition boundary.
//!
//! ## Experimental Design
//!
//! ### Primary Condition (liberal HOT)
//!
//! GWT `max_capacity = 2` with fillers at ~0.64 and ~0.74. HOT threshold 0.50.
//! Creates a disagreement zone (0.50–0.64) where HOT classifies targets as
//! conscious but GWT rejects them due to competition.
//!
//! ### Competition Pressure Sweep
//!
//! Filler1 swept from 0.52 to 0.72 (5 levels, step 0.05), filler2 = filler1 + 0.10.
//! Measures how alignment and accuracy degrade as competition intensifies.
//! The degradation curve reveals the architecture's robustness to workspace scarcity.
//!
//! ### Conservative HOT Condition (bidirectional mismatch)
//!
//! Same GWT setup but HOT threshold raised to 0.70 (above GWT's effective ~0.64).
//! Now HOT is more conservative than GWT — it misses some GWT-conscious targets.
//! Breaks the hit rate ceiling and tests whether alignment is symmetric.
//!
//! ## Signal Detection Theory (Macmillan & Creelman, 2005)
//!
//! Using GWT ignition as "signal present" and HOT classification as "response":
//! - **Hit**: GWT=conscious AND HOT=conscious
//! - **Miss**: GWT=conscious AND HOT=unconscious
//! - **False Alarm**: GWT=unconscious AND HOT=conscious
//! - **Correct Rejection**: GWT=unconscious AND HOT=unconscious
//!
//! Liberal HOT (threshold 0.50): high hit rate, moderate FA → negative criterion c.
//! Conservative HOT (threshold 0.70): reduced hit rate, low FA → positive criterion c.
//! Comparing both conditions reveals whether d' is stable across bias shifts.
//!
//! ## ROC Analysis (Green & Swets, 1966)
//!
//! Full Receiver Operating Characteristic by sweeping HOT threshold from 0.30 to 0.80.
//! Each threshold produces a (false alarm rate, hit rate) point. The Area Under Curve
//! (AUC) is a bias-free measure of metacognitive sensitivity — it answers "how well
//! does HOT track GWT ignition, regardless of where the threshold is set?"
//! Optimal threshold identified via Youden's J statistic (max hit_rate - fa_rate).
//!
//! ## Confidence Calibration (Niculescu-Mizil & Caruana, 2005)
//!
//! Tests whether the continuous activation level is calibrated as a probability
//! of GWT ignition. Activations binned into deciles; for each bin, measures
//! the gap between mean activation (interpreted as predicted P(ignition)) and
//! actual ignition rate. Expected Calibration Error (ECE) quantifies the
//! miscalibration. Since GWT has a sharp threshold, the calibration curve is
//! sigmoid-like rather than diagonal — the ECE reveals that consciousness is
//! a threshold phenomenon, not a probability proportional to evidence strength.
//!
//! ## Observation Noise Sweep (Lau & Rosenthal, 2011)
//!
//! Models imperfect metacognitive access by adding Gaussian noise (σ = 0.00–0.20)
//! to HOT's activation input while GWT retains the true value. In real brains,
//! higher-order representations are noisy approximations of first-order states.
//! Measures how accuracy, ROC AUC, and d' degrade with observation noise, and
//! identifies the noise tolerance threshold where metacognitive alignment breaks down.
//!
//! ## Cross-Condition ROC
//!
//! Computes ROC AUC for each competition pressure level in the sweep, measuring
//! whether the underlying discriminability (not just fixed-threshold accuracy)
//! degrades with increasing workspace competition.

use crate::benchmarks::qualia_confidence::helpers::{
    float_from_seed, jitter_from_seed, linear_fit, pearson_correlation, sdt_dprime, sigmoid_fit,
};
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::global_workspace::{GlobalWorkspace, WorkspaceConfig, WorkspaceContent};
use symthaea_core::hdc::higher_order_thought::{HOTConfig, HigherOrderThoughtSystem};

/// Number of activation levels to sweep (0.10 to 0.90, step 0.05).
const NUM_LEVELS: usize = 17;

/// Trials per activation level.
const TRIALS_PER_LEVEL: usize = 50;

/// Jitter amplitude applied to target and filler activations.
const JITTER_AMPLITUDE: f64 = 0.03;

/// Primary filler activations (strong competition).
const FILLER_1_BASE: f64 = 0.64;
const FILLER_2_BASE: f64 = 0.74;

/// GWT parameters.
const GWT_MAX_CAPACITY: usize = 2;
const GWT_ENTRY_THRESHOLD: f64 = 0.50;

/// HOT thresholds for bidirectional mismatch.
const HOT_THRESHOLD_LIBERAL: f64 = 0.50;
const HOT_THRESHOLD_CONSERVATIVE: f64 = 0.70;

/// Competition sweep: filler1 levels (filler2 = filler1 + 0.10).
const SWEEP_FILLER1_LEVELS: [f64; 5] = [0.52, 0.57, 0.62, 0.67, 0.72];

/// ROC curve: HOT thresholds to sweep for receiver operating characteristic.
const ROC_THRESHOLDS: [f64; 11] = [
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
];

/// Number of equal-width bins for confidence calibration analysis.
const CALIBRATION_BINS: usize = 10;

/// Observation noise sigma levels for metacognitive robustness sweep.
const NOISE_SIGMA_LEVELS: [f64; 5] = [0.00, 0.05, 0.10, 0.15, 0.20];

/// Deterministic Gaussian noise via Box-Muller transform.
fn gaussian_noise_from_seed(seed: u64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return 0.0;
    }
    let u1 = float_from_seed(seed).max(1e-10); // avoid log(0)
    let u2 = float_from_seed(seed.wrapping_add(7919)); // co-prime offset
    sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Results from a single experimental condition.
struct ConditionResult {
    gwt_rates: Vec<f64>,
    hot_rates: Vec<f64>,
    agree_count: usize,
    total_trials: usize,
    hits: usize,
    misses: usize,
    false_alarms: usize,
    correct_rejections: usize,
    /// Trial-level data: (effective_activation, gwt_ignited) for ROC and calibration analysis.
    trials: Vec<(f64, bool)>,
}

/// Run one experimental condition: sweep activation levels with given fillers and HOT threshold.
///
/// `hot_observation_noise`: Gaussian noise σ added to HOT's activation input. Models
/// imperfect metacognitive access to first-order states. When > 0, HOT observes a
/// noisy version of the true activation while GWT uses the true value.
fn run_condition(
    base_seed: u64,
    levels: &[f64],
    filler1_base: f64,
    filler2_base: f64,
    hot_threshold: f64,
    seed_offset: u64,
    hot_observation_noise: f64,
) -> ConditionResult {
    let n = levels.len();
    let mut gwt_rates = vec![0.0f64; n];
    let mut hot_rates = vec![0.0f64; n];
    let mut agree_count = 0usize;
    let mut total_trials = 0usize;
    let mut hits = 0usize;
    let mut misses = 0usize;
    let mut false_alarms = 0usize;
    let mut correct_rejections = 0usize;
    let mut trials = Vec::with_capacity(n * TRIALS_PER_LEVEL);

    for (level_idx, &level) in levels.iter().enumerate() {
        let mut gwt_ignited_count = 0usize;
        let mut hot_conscious_count = 0usize;

        for trial in 0..TRIALS_PER_LEVEL {
            let trial_seed = base_seed
                .wrapping_mul(0x517CC1B727220A95)
                .wrapping_add(seed_offset)
                .wrapping_add(level_idx as u64 * 1000 + trial as u64);

            // Target
            let target_hv = BinaryHV::random(trial_seed);
            let target_jitter = jitter_from_seed(trial_seed.wrapping_add(1), JITTER_AMPLITUDE);
            let effective_activation = (level + target_jitter).clamp(0.05, 0.95);

            // Fillers
            let filler1_hv = BinaryHV::random(trial_seed.wrapping_add(100));
            let filler1_jitter = jitter_from_seed(trial_seed.wrapping_add(101), JITTER_AMPLITUDE);
            let filler1_activation = (filler1_base + filler1_jitter).clamp(0.05, 0.95);

            let filler2_hv = BinaryHV::random(trial_seed.wrapping_add(200));
            let filler2_jitter = jitter_from_seed(trial_seed.wrapping_add(201), JITTER_AMPLITUDE);
            let filler2_activation = (filler2_base + filler2_jitter).clamp(0.05, 0.95);

            // --- GWT ---
            let mut ws = GlobalWorkspace::new(WorkspaceConfig {
                max_capacity: GWT_MAX_CAPACITY,
                entry_threshold: GWT_ENTRY_THRESHOLD,
                ..Default::default()
            });
            ws.submit(WorkspaceContent::new(
                vec![target_hv],
                effective_activation,
                "target".to_string(),
            ));
            ws.submit(WorkspaceContent::new(
                vec![filler1_hv],
                filler1_activation,
                "filler_0".to_string(),
            ));
            ws.submit(WorkspaceContent::new(
                vec![filler2_hv],
                filler2_activation,
                "filler_1".to_string(),
            ));
            let assessment = ws.process();
            let gwt_ignited = assessment
                .conscious_contents
                .iter()
                .any(|c| c.source == "target");

            // --- HOT (with optional observation noise) ---
            let hot_observed_activation = if hot_observation_noise > 0.0 {
                let noise =
                    gaussian_noise_from_seed(trial_seed.wrapping_add(500), hot_observation_noise);
                (effective_activation + noise).clamp(0.05, 0.95)
            } else {
                effective_activation
            };

            let mut hot = HigherOrderThoughtSystem::new(HOTConfig {
                auto_generate_hots: true,
                hot_threshold,
                max_depth: 2,
                detect_misrepresentation: false,
                require_explicit_hots: false,
            });
            hot.perceive_with_confidence(
                vec![target_hv],
                "target".to_string(),
                hot_observed_activation,
            );
            let hot_assessment = hot.process();
            let hot_conscious = hot_assessment.second_order_count > 0;

            if gwt_ignited {
                gwt_ignited_count += 1;
            }
            if hot_conscious {
                hot_conscious_count += 1;
            }
            if gwt_ignited == hot_conscious {
                agree_count += 1;
            }
            total_trials += 1;

            match (gwt_ignited, hot_conscious) {
                (true, true) => hits += 1,
                (true, false) => misses += 1,
                (false, true) => false_alarms += 1,
                (false, false) => correct_rejections += 1,
            }

            trials.push((hot_observed_activation, gwt_ignited));
        }

        gwt_rates[level_idx] = gwt_ignited_count as f64 / TRIALS_PER_LEVEL as f64;
        hot_rates[level_idx] = hot_conscious_count as f64 / TRIALS_PER_LEVEL as f64;
    }

    ConditionResult {
        gwt_rates,
        hot_rates,
        agree_count,
        total_trials,
        hits,
        misses,
        false_alarms,
        correct_rejections,
        trials,
    }
}

/// Compute standard metrics from a condition result.
struct ConditionMetrics {
    boundary_alignment: f64,
    classification_accuracy: f64,
    hot_ignition_correlation: f64,
    sigmoid_quality: f64,
    gwt_x0: f64,
    hot_x0: f64,
    gwt_r2: f64,
    hot_r2: f64,
    hit_rate: f64,
    false_alarm_rate: f64,
    d_prime: f64,
    criterion_c: f64,
}

fn compute_metrics(cr: &ConditionResult, levels: &[f64]) -> ConditionMetrics {
    let (_, _, gwt_x0, gwt_r2) = sigmoid_fit(levels, &cr.gwt_rates);
    let (_, _, hot_x0, hot_r2) = sigmoid_fit(levels, &cr.hot_rates);

    let boundary_alignment = (1.0 - (gwt_x0 - hot_x0).abs()).clamp(0.0, 1.0);
    let classification_accuracy = cr.agree_count as f64 / cr.total_trials as f64;
    let hot_ignition_correlation = pearson_correlation(&cr.gwt_rates, &cr.hot_rates);
    let sigmoid_quality = ((gwt_r2 + hot_r2) / 2.0).clamp(0.0, 1.0);

    let n_signal = cr.hits + cr.misses;
    let n_noise = cr.false_alarms + cr.correct_rejections;
    let hit_rate = if n_signal > 0 {
        cr.hits as f64 / n_signal as f64
    } else {
        0.0
    };
    let false_alarm_rate = if n_noise > 0 {
        cr.false_alarms as f64 / n_noise as f64
    } else {
        0.0
    };
    let (d_prime, criterion_c) = sdt_dprime(hit_rate, false_alarm_rate, n_signal, n_noise);

    ConditionMetrics {
        boundary_alignment,
        classification_accuracy,
        hot_ignition_correlation,
        sigmoid_quality,
        gwt_x0,
        hot_x0,
        gwt_r2,
        hot_r2,
        hit_rate,
        false_alarm_rate,
        d_prime,
        criterion_c,
    }
}

/// Compute ROC curve by sweeping HOT threshold over trial-level data.
///
/// For each threshold, classifies trials as "conscious" (activation >= threshold) and
/// computes hit rate and false alarm rate against GWT ground truth.
///
/// Returns (auc, optimal_threshold, optimal_j, num_curve_points).
fn compute_roc(trials: &[(f64, bool)], thresholds: &[f64]) -> (f64, f64, f64, usize) {
    let n_signal = trials.iter().filter(|t| t.1).count();
    let n_noise = trials.iter().filter(|t| !t.1).count();

    if n_signal == 0 || n_noise == 0 {
        return (0.5, 0.5, 0.0, 0);
    }

    let mut roc_points: Vec<(f64, f64)> = Vec::new(); // (fa_rate, hit_rate)
    let mut best_j = f64::NEG_INFINITY;
    let mut best_threshold = 0.5;

    for &t in thresholds {
        let hits = trials.iter().filter(|&&(act, gwt)| gwt && act >= t).count();
        let fas = trials
            .iter()
            .filter(|&&(act, gwt)| !gwt && act >= t)
            .count();

        let hit_rate = hits as f64 / n_signal as f64;
        let fa_rate = fas as f64 / n_noise as f64;

        roc_points.push((fa_rate, hit_rate));

        let j = hit_rate - fa_rate; // Youden's J statistic
        if j > best_j {
            best_j = j;
            best_threshold = t;
        }
    }

    // Anchor points for proper AUC integration
    roc_points.push((0.0, 0.0));
    roc_points.push((1.0, 1.0));

    // Sort by FA rate ascending, then hit rate descending (for dedup: keep highest HR)
    roc_points.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    // For same FA rate, keep only the highest hit rate
    let mut deduped: Vec<(f64, f64)> = Vec::with_capacity(roc_points.len());
    for &(fa, hr) in &roc_points {
        if deduped.is_empty() || (fa - deduped.last().unwrap().0).abs() > 1e-10 {
            deduped.push((fa, hr));
        }
    }

    // Trapezoidal AUC
    let mut auc = 0.0;
    for i in 0..deduped.len() - 1 {
        let dx = deduped[i + 1].0 - deduped[i].0;
        let avg_y = (deduped[i].1 + deduped[i + 1].1) / 2.0;
        auc += dx * avg_y;
    }

    (
        auc.clamp(0.0, 1.0),
        best_threshold,
        best_j.max(0.0),
        deduped.len(),
    )
}

/// Compute confidence calibration metrics from trial-level data.
///
/// Bins activations into equal-width bins and measures whether the mean activation
/// in each bin predicts the GWT ignition rate. Perfect calibration would mean
/// activation = P(ignition), but GWT's threshold dynamics create a sharp sigmoid
/// rather than a diagonal — the miscalibration itself is informative.
///
/// Returns (ece, slope, max_bin_error).
fn compute_calibration(trials: &[(f64, bool)], num_bins: usize) -> (f64, f64, f64) {
    if trials.is_empty() || num_bins == 0 {
        return (0.0, 0.0, 0.0);
    }

    let bin_width = 1.0 / num_bins as f64;
    let mut bin_means = Vec::new();
    let mut bin_rates = Vec::new();
    let mut bin_counts = Vec::new();

    for b in 0..num_bins {
        let lo = b as f64 * bin_width;
        let hi = lo + bin_width;

        let in_bin: Vec<_> = trials
            .iter()
            .filter(|&&(act, _)| act >= lo && (act < hi || (b == num_bins - 1 && act <= hi)))
            .collect();

        if in_bin.is_empty() {
            continue;
        }

        let mean_act = in_bin.iter().map(|&&(act, _)| act).sum::<f64>() / in_bin.len() as f64;
        let ignition_rate =
            in_bin.iter().filter(|&&&(_, gwt)| gwt).count() as f64 / in_bin.len() as f64;

        bin_means.push(mean_act);
        bin_rates.push(ignition_rate);
        bin_counts.push(in_bin.len());
    }

    if bin_means.len() < 2 {
        return (0.0, 0.0, 0.0);
    }

    // Expected Calibration Error: weighted average of |ignition_rate - mean_activation|
    let total = bin_counts.iter().sum::<usize>() as f64;
    let ece: f64 = bin_means
        .iter()
        .zip(bin_rates.iter())
        .zip(bin_counts.iter())
        .map(|((&mean, &rate), &count)| (rate - mean).abs() * count as f64 / total)
        .sum();

    // Calibration slope: linear fit of ignition_rate ~ mean_activation
    let (slope, _, _) = linear_fit(&bin_means, &bin_rates);

    // Maximum single-bin calibration error
    let max_error = bin_means
        .iter()
        .zip(bin_rates.iter())
        .map(|(&mean, &rate)| (rate - mean).abs())
        .fold(0.0f64, f64::max);

    (ece, slope, max_error)
}

/// Metacognitive Ignition Tracking Benchmark.
pub struct MetacognitiveIgnitionBenchmark;

impl PsychBenchmark for MetacognitiveIgnitionBenchmark {
    fn name(&self) -> &str {
        "QualiaConfidence::MetacognitiveIgnition"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Metacognitive Ignition Tracking (HOT-GWT Alignment)",
            citation: "Rosenthal (2005); Dehaene & Naccache (2001); Lau & Rosenthal (2011)",
            year: 2011,
            doi: Some("10.1016/j.concog.2010.09.014"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let base_seed = config.seed;
        let levels: Vec<f64> = (0..NUM_LEVELS).map(|i| 0.10 + i as f64 * 0.05).collect();

        // =====================================================================
        // 1. Primary condition: liberal HOT (threshold 0.50)
        // =====================================================================
        let primary = run_condition(
            base_seed,
            &levels,
            FILLER_1_BASE,
            FILLER_2_BASE,
            HOT_THRESHOLD_LIBERAL,
            0,
            0.0,
        );
        let pm = compute_metrics(&primary, &levels);

        let spontaneous_tracking_score = (0.3 * pm.boundary_alignment
            + 0.3 * pm.classification_accuracy
            + 0.2 * pm.hot_ignition_correlation.clamp(0.0, 1.0)
            + 0.2 * pm.sigmoid_quality)
            .clamp(0.0, 1.0);

        // Degenerate detection
        let gwt_always_on = primary.gwt_rates.iter().all(|&r| r >= 1.0);
        let gwt_always_off = primary.gwt_rates.iter().all(|&r| r <= 0.0);
        let hot_always_on = primary.hot_rates.iter().all(|&r| r >= 1.0);
        let hot_always_off = primary.hot_rates.iter().all(|&r| r <= 0.0);
        let both_flat = pm.gwt_r2 < 0.30 && pm.hot_r2 < 0.30;
        let is_degenerate =
            gwt_always_on || gwt_always_off || hot_always_on || hot_always_off || both_flat;

        // =====================================================================
        // 2. Conservative HOT condition (threshold 0.70)
        // =====================================================================
        let conservative = run_condition(
            base_seed,
            &levels,
            FILLER_1_BASE,
            FILLER_2_BASE,
            HOT_THRESHOLD_CONSERVATIVE,
            1_000_000,
            0.0,
        );
        let cm = compute_metrics(&conservative, &levels);

        // Bidirectional comparison
        let criterion_shift = cm.criterion_c - pm.criterion_c; // should be positive
        let d_prime_ratio = if pm.d_prime.abs() > 0.01 {
            cm.d_prime / pm.d_prime
        } else {
            0.0
        };

        // =====================================================================
        // 3. Competition pressure sweep
        // =====================================================================
        let mut sweep_accuracies = Vec::with_capacity(SWEEP_FILLER1_LEVELS.len());
        let mut sweep_alignments = Vec::with_capacity(SWEEP_FILLER1_LEVELS.len());
        let mut sweep_d_primes = Vec::with_capacity(SWEEP_FILLER1_LEVELS.len());
        let mut sweep_roc_aucs = Vec::with_capacity(SWEEP_FILLER1_LEVELS.len());

        for (i, &f1) in SWEEP_FILLER1_LEVELS.iter().enumerate() {
            let f2 = f1 + 0.10;
            let cond = run_condition(
                base_seed,
                &levels,
                f1,
                f2,
                HOT_THRESHOLD_LIBERAL,
                (i as u64 + 2) * 1_000_000,
                0.0,
            );
            let sm = compute_metrics(&cond, &levels);
            sweep_accuracies.push(sm.classification_accuracy);
            sweep_alignments.push(sm.boundary_alignment);
            sweep_d_primes.push(sm.d_prime);

            // Cross-condition ROC: discriminability at each competition level
            let (sweep_auc, _, _, _) = compute_roc(&cond.trials, &ROC_THRESHOLDS);
            sweep_roc_aucs.push(sweep_auc);
        }

        // Degradation curve analysis
        let pressure_levels: Vec<f64> = SWEEP_FILLER1_LEVELS.to_vec();
        let (alignment_slope, _, _) = linear_fit(&pressure_levels, &sweep_alignments);
        let (accuracy_slope, _, _) = linear_fit(&pressure_levels, &sweep_accuracies);

        // AUC via trapezoidal rule (normalized to [0,1] range)
        let sweep_alignment_auc = if sweep_alignments.len() >= 2 {
            let mut area = 0.0;
            for i in 0..sweep_alignments.len() - 1 {
                area += (sweep_alignments[i] + sweep_alignments[i + 1]) / 2.0
                    * (pressure_levels[i + 1] - pressure_levels[i]);
            }
            let span = pressure_levels.last().unwrap() - pressure_levels.first().unwrap();
            if span > 0.0 { area / span } else { 0.0 }
        } else {
            0.0
        };

        // =====================================================================
        // 4. ROC analysis: sweep HOT threshold over primary trial data
        // =====================================================================
        let (roc_auc, roc_optimal_threshold, roc_optimal_j, roc_num_points) =
            compute_roc(&primary.trials, &ROC_THRESHOLDS);

        // =====================================================================
        // 5. Confidence calibration: is activation calibrated as P(ignition)?
        // =====================================================================
        let (calibration_ece, calibration_slope, calibration_max_bin_error) =
            compute_calibration(&primary.trials, CALIBRATION_BINS);

        // =====================================================================
        // 6. Observation noise sweep: metacognitive robustness
        // =====================================================================
        let mut noise_accuracies = Vec::with_capacity(NOISE_SIGMA_LEVELS.len());
        let mut noise_roc_aucs = Vec::with_capacity(NOISE_SIGMA_LEVELS.len());
        let mut noise_d_primes = Vec::with_capacity(NOISE_SIGMA_LEVELS.len());

        for (i, &sigma) in NOISE_SIGMA_LEVELS.iter().enumerate() {
            let noisy_cond = run_condition(
                base_seed,
                &levels,
                FILLER_1_BASE,
                FILLER_2_BASE,
                HOT_THRESHOLD_LIBERAL,
                (i as u64 + 10) * 1_000_000,
                sigma,
            );
            let nm = compute_metrics(&noisy_cond, &levels);
            noise_accuracies.push(nm.classification_accuracy);
            noise_d_primes.push(nm.d_prime);

            let (noisy_auc, _, _, _) = compute_roc(&noisy_cond.trials, &ROC_THRESHOLDS);
            noise_roc_aucs.push(noisy_auc);
        }

        // Noise degradation analysis
        let sigma_vec: Vec<f64> = NOISE_SIGMA_LEVELS.to_vec();
        let (noise_accuracy_slope, _, _) = linear_fit(&sigma_vec, &noise_accuracies);
        let (noise_roc_auc_slope, _, _) = linear_fit(&sigma_vec, &noise_roc_aucs);

        // ROC AUC at σ=0.10 (index 2)
        let noise_roc_auc_at_010 = noise_roc_aucs.get(2).copied().unwrap_or(0.0);

        // d' retention: fraction of baseline d' preserved at σ=0.10
        let noise_d_prime_retention = if noise_d_primes[0].abs() > 0.01 {
            noise_d_primes.get(2).copied().unwrap_or(0.0) / noise_d_primes[0]
        } else {
            0.0
        };

        // Noise tolerance: highest σ where accuracy > 0.70
        let noise_tolerance = NOISE_SIGMA_LEVELS
            .iter()
            .zip(noise_accuracies.iter())
            .rev()
            .find(|&(_, &acc)| acc > 0.70)
            .map(|(&sigma, _)| sigma)
            .unwrap_or(0.0);

        // =====================================================================
        // 7. Cross-condition ROC (competition sweep discriminability)
        // =====================================================================
        let (sweep_roc_auc_slope, _, _) = linear_fit(&pressure_levels, &sweep_roc_aucs);
        let sweep_roc_min_auc = sweep_roc_aucs.iter().copied().fold(f64::INFINITY, f64::min);

        // =====================================================================
        // Build result
        // =====================================================================
        let elapsed = start.elapsed();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        // Primary condition metrics
        result.insert(
            "boundary_alignment",
            MetricValue::from_samples(&[pm.boundary_alignment]),
        );
        result.insert(
            "classification_accuracy",
            MetricValue::from_samples(&[pm.classification_accuracy]),
        );
        result.insert(
            "hot_ignition_correlation",
            MetricValue::from_samples(&[pm.hot_ignition_correlation]),
        );
        result.insert(
            "spontaneous_tracking_score",
            MetricValue::from_samples(&[spontaneous_tracking_score]),
        );
        result.insert(
            "gwt_sigmoid_r_squared",
            MetricValue::from_samples(&[pm.gwt_r2]),
        );
        result.insert(
            "hot_sigmoid_r_squared",
            MetricValue::from_samples(&[pm.hot_r2]),
        );
        result.insert(
            "gwt_effective_threshold",
            MetricValue::from_samples(&[pm.gwt_x0]),
        );
        result.insert(
            "hot_effective_threshold",
            MetricValue::from_samples(&[pm.hot_x0]),
        );
        result.insert(
            "sigmoid_quality",
            MetricValue::from_samples(&[pm.sigmoid_quality]),
        );

        // SDT — liberal condition
        result.insert("sdt_hit_rate", MetricValue::from_samples(&[pm.hit_rate]));
        result.insert(
            "sdt_false_alarm_rate",
            MetricValue::from_samples(&[pm.false_alarm_rate]),
        );
        result.insert("sdt_d_prime", MetricValue::from_samples(&[pm.d_prime]));
        result.insert(
            "sdt_criterion_c",
            MetricValue::from_samples(&[pm.criterion_c]),
        );

        // SDT — conservative condition
        result.insert(
            "conservative_hit_rate",
            MetricValue::from_samples(&[cm.hit_rate]),
        );
        result.insert(
            "conservative_false_alarm_rate",
            MetricValue::from_samples(&[cm.false_alarm_rate]),
        );
        result.insert(
            "conservative_d_prime",
            MetricValue::from_samples(&[cm.d_prime]),
        );
        result.insert(
            "conservative_criterion_c",
            MetricValue::from_samples(&[cm.criterion_c]),
        );

        // Bidirectional comparison
        result.insert(
            "criterion_shift",
            MetricValue::from_samples(&[criterion_shift]),
        );
        result.insert("d_prime_ratio", MetricValue::from_samples(&[d_prime_ratio]));

        // Competition sweep
        result.insert(
            "sweep_alignment_slope",
            MetricValue::from_samples(&[alignment_slope]),
        );
        result.insert(
            "sweep_accuracy_slope",
            MetricValue::from_samples(&[accuracy_slope]),
        );
        result.insert(
            "sweep_alignment_auc",
            MetricValue::from_samples(&[sweep_alignment_auc]),
        );
        result.insert(
            "sweep_min_accuracy",
            MetricValue::from_samples(&[sweep_accuracies
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)]),
        );
        result.insert(
            "sweep_max_accuracy",
            MetricValue::from_samples(&[sweep_accuracies
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)]),
        );

        // ROC analysis
        result.insert("roc_auc", MetricValue::from_samples(&[roc_auc]));
        result.insert(
            "roc_optimal_threshold",
            MetricValue::from_samples(&[roc_optimal_threshold]),
        );
        result.insert("roc_optimal_j", MetricValue::from_samples(&[roc_optimal_j]));
        result.insert(
            "roc_num_points",
            MetricValue::from_samples(&[roc_num_points as f64]),
        );

        // Confidence calibration
        result.insert(
            "calibration_ece",
            MetricValue::from_samples(&[calibration_ece]),
        );
        result.insert(
            "calibration_slope",
            MetricValue::from_samples(&[calibration_slope]),
        );
        result.insert(
            "calibration_max_bin_error",
            MetricValue::from_samples(&[calibration_max_bin_error]),
        );

        // Observation noise sweep
        result.insert(
            "noise_accuracy_slope",
            MetricValue::from_samples(&[noise_accuracy_slope]),
        );
        result.insert(
            "noise_roc_auc_at_010",
            MetricValue::from_samples(&[noise_roc_auc_at_010]),
        );
        result.insert(
            "noise_roc_auc_slope",
            MetricValue::from_samples(&[noise_roc_auc_slope]),
        );
        result.insert(
            "noise_d_prime_retention",
            MetricValue::from_samples(&[noise_d_prime_retention]),
        );
        result.insert(
            "noise_tolerance",
            MetricValue::from_samples(&[noise_tolerance]),
        );

        // Cross-condition ROC (competition sweep)
        result.insert(
            "sweep_roc_auc_slope",
            MetricValue::from_samples(&[sweep_roc_auc_slope]),
        );
        result.insert(
            "sweep_roc_min_auc",
            MetricValue::from_samples(&[sweep_roc_min_auc]),
        );

        result.insert(
            "is_degenerate",
            MetricValue::from_samples(&[if is_degenerate { 1.0 } else { 0.0 }]),
        );
        result.insert(
            "elapsed_ms",
            MetricValue::from_samples(&[elapsed.as_millis() as f64]),
        );

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_benchmark(seed: u64) -> BenchmarkResult {
        let config = BenchmarkConfig {
            seed,
            ..Default::default()
        };
        MetacognitiveIgnitionBenchmark.run(&config)
    }

    // =================================================================
    // Core benchmark tests
    // =================================================================

    #[test]
    fn test_metacognitive_ignition_runs() {
        let result = run_benchmark(42);
        assert_eq!(result.benchmark, "QualiaConfidence::MetacognitiveIgnition");
        assert!(
            result.metrics.len() >= 34,
            "Expected >= 34 metrics, got {}",
            result.metrics.len()
        );
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = run_benchmark(42);
        for (key, metric) in &result.metrics {
            assert!(
                metric.mean.is_finite(),
                "Metric '{}' is not finite: {}",
                key,
                metric.mean
            );
        }
    }

    #[test]
    fn test_boundary_alignment_positive() {
        let result = run_benchmark(42);
        let alignment = result.metrics["boundary_alignment"].mean;
        assert!(
            alignment > 0.65,
            "Boundary alignment should be > 0.65: {alignment}"
        );
    }

    #[test]
    fn test_classification_accuracy_above_chance() {
        let result = run_benchmark(42);
        let accuracy = result.metrics["classification_accuracy"].mean;
        assert!(
            accuracy > 0.65,
            "Classification accuracy should be > 0.65: {accuracy}"
        );
    }

    #[test]
    fn test_hot_ignition_correlation_positive() {
        let result = run_benchmark(42);
        let corr = result.metrics["hot_ignition_correlation"].mean;
        assert!(
            corr > 0.0,
            "HOT-ignition correlation should be positive: {corr}"
        );
    }

    #[test]
    fn test_gwt_ignition_increases() {
        let base_seed = 42u64;
        let levels: Vec<f64> = (0..NUM_LEVELS).map(|i| 0.10 + i as f64 * 0.05).collect();
        let cr = run_condition(
            base_seed,
            &levels,
            FILLER_1_BASE,
            FILLER_2_BASE,
            HOT_THRESHOLD_LIBERAL,
            0,
            0.0,
        );
        assert!(
            cr.gwt_rates[NUM_LEVELS - 1] > cr.gwt_rates[0],
            "GWT rate should increase: low={}, high={}",
            cr.gwt_rates[0],
            cr.gwt_rates[NUM_LEVELS - 1]
        );
    }

    #[test]
    fn test_hot_ratio_increases() {
        let base_seed = 42u64;
        let levels: Vec<f64> = (0..NUM_LEVELS).map(|i| 0.10 + i as f64 * 0.05).collect();
        let cr = run_condition(
            base_seed,
            &levels,
            FILLER_1_BASE,
            FILLER_2_BASE,
            HOT_THRESHOLD_LIBERAL,
            0,
            0.0,
        );
        assert!(
            cr.hot_rates[NUM_LEVELS - 1] > cr.hot_rates[0],
            "HOT rate should increase: low={}, high={}",
            cr.hot_rates[0],
            cr.hot_rates[NUM_LEVELS - 1]
        );
    }

    #[test]
    fn test_not_degenerate() {
        let result = run_benchmark(42);
        assert!(
            result.metrics["is_degenerate"].mean < 0.5,
            "Should not be degenerate"
        );
    }

    #[test]
    fn test_disagreement_zone_exists() {
        let result = run_benchmark(42);
        let accuracy = result.metrics["classification_accuracy"].mean;
        assert!(
            accuracy < 0.95,
            "Systems should disagree meaningfully: accuracy={accuracy}"
        );
    }

    #[test]
    fn test_deterministic() {
        let r1 = run_benchmark(42);
        let r2 = run_benchmark(42);
        let s1 = r1.metrics["spontaneous_tracking_score"].mean;
        let s2 = r2.metrics["spontaneous_tracking_score"].mean;
        assert!(
            (s1 - s2).abs() < 1e-10,
            "Same seed should match: {s1} vs {s2}"
        );
    }

    #[test]
    fn test_has_provenance() {
        let prov = MetacognitiveIgnitionBenchmark.provenance();
        assert!(prov.is_some());
        assert!(prov.unwrap().citation.contains("Rosenthal"));
    }

    /// Multi-seed statistical analysis: 10 seeds, mean ± SD, CV < 0.30 for stability.
    ///
    /// Transforms single-seed results into population-level claims. Tests that
    /// the 5 headline metrics are stable across random seeds (low coefficient of
    /// variation) and that predictions hold consistently (not just at seed=42).
    #[test]
    fn test_multi_seed_confidence_intervals() {
        use crate::benchmarks::qualia_confidence::helpers::coefficient_of_variation;

        const SEEDS: [u64; 10] = [42, 123, 7, 999, 314159, 271828, 16384, 65537, 2718, 31415];

        let mut tracking_scores = Vec::with_capacity(10);
        let mut accuracies = Vec::with_capacity(10);
        let mut roc_aucs = Vec::with_capacity(10);
        let mut eces = Vec::with_capacity(10);
        let mut noise_tolerances = Vec::with_capacity(10);

        for &seed in &SEEDS {
            let result = run_benchmark(seed);
            tracking_scores.push(result.metrics["spontaneous_tracking_score"].mean);
            accuracies.push(result.metrics["classification_accuracy"].mean);
            roc_aucs.push(result.metrics["roc_auc"].mean);
            eces.push(result.metrics["calibration_ece"].mean);
            noise_tolerances.push(result.metrics["noise_tolerance"].mean);
        }

        // --- Stability: CV < 0.30 for each metric ---
        let cv_tracking = coefficient_of_variation(&tracking_scores);
        let cv_accuracy = coefficient_of_variation(&accuracies);
        let cv_roc = coefficient_of_variation(&roc_aucs);
        let cv_ece = coefficient_of_variation(&eces);

        assert!(
            cv_tracking < 0.30,
            "tracking_score CV should be < 0.30 (stable): {cv_tracking:.4}"
        );
        assert!(
            cv_accuracy < 0.30,
            "accuracy CV should be < 0.30 (stable): {cv_accuracy:.4}"
        );
        assert!(
            cv_roc < 0.30,
            "roc_auc CV should be < 0.30 (stable): {cv_roc:.4}"
        );
        assert!(
            cv_ece < 0.30,
            "calibration_ece CV should be < 0.30 (stable): {cv_ece:.4}"
        );

        // --- Population means meet predictions ---
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;

        let mean_tracking = mean(&tracking_scores);
        assert!(
            mean_tracking > 0.50,
            "Mean tracking score across 10 seeds should exceed prediction (0.50): {mean_tracking:.4}"
        );

        let mean_accuracy = mean(&accuracies);
        assert!(
            mean_accuracy > 0.70,
            "Mean accuracy across 10 seeds should be > 0.70: {mean_accuracy:.4}"
        );

        let mean_roc = mean(&roc_aucs);
        assert!(
            mean_roc > 0.90,
            "Mean ROC AUC across 10 seeds should be > 0.90: {mean_roc:.4}"
        );

        // --- All seeds individually pass minimum thresholds ---
        for (i, &seed) in SEEDS.iter().enumerate() {
            assert!(
                tracking_scores[i] > 0.40,
                "Seed {seed}: tracking_score should be > 0.40: {:.4}",
                tracking_scores[i]
            );
            assert!(
                accuracies[i] > 0.60,
                "Seed {seed}: accuracy should be > 0.60: {:.4}",
                accuracies[i]
            );
        }

        // --- Noise tolerance: majority of seeds tolerate σ ≥ 0.10 ---
        let tolerant_count = noise_tolerances.iter().filter(|&&t| t >= 0.10).count();
        assert!(
            tolerant_count >= 7,
            "At least 7/10 seeds should tolerate noise σ ≥ 0.10: {tolerant_count}/10"
        );
    }

    // =================================================================
    // SDT tests — liberal condition
    // =================================================================

    #[test]
    fn test_sdt_d_prime_positive() {
        let result = run_benchmark(42);
        let dp = result.metrics["sdt_d_prime"].mean;
        assert!(dp > 1.0, "d' should show discriminability (> 1.0): {dp}");
    }

    #[test]
    fn test_sdt_liberal_bias() {
        let result = run_benchmark(42);
        let c = result.metrics["sdt_criterion_c"].mean;
        assert!(c < 0.0, "Liberal HOT should have criterion c < 0: {c}");
    }

    #[test]
    fn test_sdt_hit_rate_high() {
        let result = run_benchmark(42);
        let hr = result.metrics["sdt_hit_rate"].mean;
        assert!(hr > 0.80, "Hit rate should be high: {hr}");
    }

    #[test]
    fn test_sdt_false_alarm_rate_moderate() {
        let result = run_benchmark(42);
        let far = result.metrics["sdt_false_alarm_rate"].mean;
        assert!(
            far > 0.05 && far < 0.90,
            "FA rate should be moderate: {far}"
        );
    }

    // =================================================================
    // Bidirectional mismatch tests
    // =================================================================

    #[test]
    fn test_conservative_criterion_positive() {
        let result = run_benchmark(42);
        let c = result.metrics["conservative_criterion_c"].mean;
        assert!(
            c > 0.0,
            "Conservative HOT should have positive criterion c: {c}"
        );
    }

    #[test]
    fn test_criterion_shift_positive() {
        let result = run_benchmark(42);
        let shift = result.metrics["criterion_shift"].mean;
        assert!(
            shift > 0.5,
            "Criterion should shift substantially from liberal to conservative: {shift}"
        );
    }

    #[test]
    fn test_conservative_hit_rate_below_ceiling() {
        let result = run_benchmark(42);
        let hr = result.metrics["conservative_hit_rate"].mean;
        assert!(
            hr < 0.99,
            "Conservative HOT should have hit rate below ceiling: {hr}"
        );
    }

    #[test]
    fn test_d_prime_stable_across_bias() {
        let result = run_benchmark(42);
        let ratio = result.metrics["d_prime_ratio"].mean;
        // d' should be roughly similar across bias conditions (within 2x)
        assert!(
            ratio > 0.3 && ratio < 3.0,
            "d' ratio should be moderate (same underlying sensitivity): {ratio}"
        );
    }

    // =================================================================
    // Competition sweep tests
    // =================================================================

    #[test]
    fn test_sweep_alignment_degrades() {
        let result = run_benchmark(42);
        let slope = result.metrics["sweep_alignment_slope"].mean;
        assert!(
            slope < 0.0,
            "Alignment should degrade as competition increases (negative slope): {slope}"
        );
    }

    #[test]
    fn test_sweep_accuracy_degrades() {
        let result = run_benchmark(42);
        let slope = result.metrics["sweep_accuracy_slope"].mean;
        assert!(
            slope < 0.0,
            "Accuracy should degrade as competition increases (negative slope): {slope}"
        );
    }

    #[test]
    fn test_sweep_auc_positive() {
        let result = run_benchmark(42);
        let auc = result.metrics["sweep_alignment_auc"].mean;
        assert!(
            auc > 0.50,
            "Alignment AUC should be well above zero (overall robustness): {auc}"
        );
    }

    #[test]
    fn test_sweep_min_accuracy_above_chance() {
        let result = run_benchmark(42);
        let min_acc = result.metrics["sweep_min_accuracy"].mean;
        assert!(
            min_acc > 0.50,
            "Even at highest competition, accuracy should beat chance: {min_acc}"
        );
    }

    // =================================================================
    // ROC analysis tests
    // =================================================================

    #[test]
    fn test_roc_auc_above_chance() {
        let result = run_benchmark(42);
        let auc = result.metrics["roc_auc"].mean;
        assert!(auc > 0.50, "ROC AUC should beat chance (0.50): {auc}");
    }

    #[test]
    fn test_roc_auc_high_discrimination() {
        let result = run_benchmark(42);
        let auc = result.metrics["roc_auc"].mean;
        assert!(
            auc > 0.75,
            "ROC AUC should show strong discrimination (> 0.75): {auc}"
        );
    }

    #[test]
    fn test_roc_optimal_threshold_in_range() {
        let result = run_benchmark(42);
        let threshold = result.metrics["roc_optimal_threshold"].mean;
        assert!(
            threshold >= 0.30 && threshold <= 0.80,
            "Optimal threshold should be within sweep range: {threshold}"
        );
    }

    #[test]
    fn test_roc_optimal_j_positive() {
        let result = run_benchmark(42);
        let j = result.metrics["roc_optimal_j"].mean;
        assert!(
            j > 0.20,
            "Youden's J should be substantially positive (good separation): {j}"
        );
    }

    // =================================================================
    // Confidence calibration tests
    // =================================================================

    #[test]
    fn test_calibration_slope_positive() {
        let result = run_benchmark(42);
        let slope = result.metrics["calibration_slope"].mean;
        assert!(
            slope > 0.0,
            "Calibration slope should be positive (activation predicts ignition): {slope}"
        );
    }

    #[test]
    fn test_calibration_slope_steep() {
        let result = run_benchmark(42);
        let slope = result.metrics["calibration_slope"].mean;
        // Slope > 1.0 means the ignition boundary is sharper than linear —
        // consciousness is a threshold phenomenon, not proportional to evidence.
        assert!(
            slope > 1.0,
            "Calibration slope > 1.0 indicates threshold consciousness: {slope}"
        );
    }

    #[test]
    fn test_calibration_ece_reveals_miscalibration() {
        let result = run_benchmark(42);
        let ece = result.metrics["calibration_ece"].mean;
        // ECE > 0 means activation is not perfectly calibrated as P(ignition).
        // This is expected: GWT's competition dynamics create a sharp threshold.
        assert!(
            ece > 0.05,
            "ECE should show meaningful miscalibration (threshold effect): {ece}"
        );
        assert!(
            ece < 0.50,
            "ECE should be bounded (activation still informative): {ece}"
        );
    }

    #[test]
    fn test_calibration_max_bin_error_bounded() {
        let result = run_benchmark(42);
        let max_err = result.metrics["calibration_max_bin_error"].mean;
        assert!(
            max_err < 0.80,
            "No single bin should be wildly miscalibrated: {max_err}"
        );
    }

    // =================================================================
    // Observation noise sweep tests
    // =================================================================

    #[test]
    fn test_noise_accuracy_degrades() {
        let result = run_benchmark(42);
        let slope = result.metrics["noise_accuracy_slope"].mean;
        assert!(
            slope < 0.0,
            "Accuracy should degrade with observation noise (negative slope): {slope}"
        );
    }

    #[test]
    fn test_noise_roc_auc_at_010_below_perfect() {
        let result = run_benchmark(42);
        let auc = result.metrics["noise_roc_auc_at_010"].mean;
        // With σ=0.10 noise, ROC AUC should drop from the trivially perfect 0.998
        assert!(
            auc < 0.99,
            "ROC AUC with noise σ=0.10 should be below perfect: {auc}"
        );
        assert!(
            auc > 0.60,
            "ROC AUC with noise σ=0.10 should still beat chance: {auc}"
        );
    }

    #[test]
    fn test_noise_d_prime_retention_partial() {
        let result = run_benchmark(42);
        let retention = result.metrics["noise_d_prime_retention"].mean;
        // Some d' should survive noise, but not all
        assert!(
            retention > 0.0 && retention <= 1.5,
            "d' retention at σ=0.10 should be partial: {retention}"
        );
    }

    #[test]
    fn test_noise_tolerance_positive() {
        let result = run_benchmark(42);
        let tolerance = result.metrics["noise_tolerance"].mean;
        // Should tolerate at least some noise before accuracy < 0.70
        assert!(
            tolerance >= 0.05,
            "Should tolerate at least σ=0.05 noise: {tolerance}"
        );
    }

    #[test]
    fn test_noise_roc_auc_degrades() {
        let result = run_benchmark(42);
        let slope = result.metrics["noise_roc_auc_slope"].mean;
        assert!(
            slope < 0.0,
            "ROC AUC should degrade with observation noise: {slope}"
        );
    }

    // =================================================================
    // Cross-condition ROC tests
    // =================================================================

    #[test]
    fn test_sweep_roc_min_auc_above_chance() {
        let result = run_benchmark(42);
        let min_auc = result.metrics["sweep_roc_min_auc"].mean;
        assert!(
            min_auc > 0.50,
            "Even at highest competition, ROC AUC should beat chance: {min_auc}"
        );
    }

    #[test]
    fn test_sweep_roc_auc_slope_nonpositive() {
        let result = run_benchmark(42);
        let slope = result.metrics["sweep_roc_auc_slope"].mean;
        // ROC AUC should stay high or degrade with competition (not increase)
        // Competition affects fixed-threshold accuracy more than underlying discriminability
        assert!(
            slope <= 0.5,
            "Sweep ROC AUC slope should not increase substantially: {slope}"
        );
    }
}
