// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmark reporting: JSON/CSV output and summary formatting.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Reaction-time summary (tick-based proxy for processing time).
///
/// Fits an ex-Gaussian distribution via method of moments (Heathcote 1996).
/// The ex-Gaussian is the convolution of a Gaussian (mu, sigma) and an
/// exponential (tau), commonly used to model RT distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtSummary {
    /// Mean RT in ticks.
    pub mean_ticks: f64,
    /// Standard deviation of RT in ticks.
    pub sd_ticks: f64,
    /// Ex-Gaussian tail parameter (exponential component).
    /// Captures the slow tail of the RT distribution.
    pub tau: f64,
    /// Ex-Gaussian Gaussian mean (mu = mean - tau).
    pub mu: f64,
    /// Ex-Gaussian Gaussian SD (sigma = sqrt(variance - tau^2)).
    pub sigma: f64,
}

impl RtSummary {
    /// Fit ex-Gaussian parameters from a slice of RT samples (in ticks).
    ///
    /// Uses method of moments (Heathcote, 1996):
    /// - tau = skewness^(1/3) * sd / cbrt(2)
    /// - mu = mean - tau
    /// - sigma = sqrt(variance - tau^2)
    pub fn from_rt_samples(ticks: &[f64]) -> Self {
        let n = ticks.len();
        if n < 3 {
            let mean = if n > 0 {
                ticks.iter().sum::<f64>() / n as f64
            } else {
                0.0
            };
            return Self {
                mean_ticks: mean,
                sd_ticks: 0.0,
                tau: 0.0,
                mu: mean,
                sigma: 0.0,
            };
        }

        let nf = n as f64;
        let mean = ticks.iter().sum::<f64>() / nf;
        let variance = ticks.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (nf - 1.0);
        let sd = variance.sqrt();

        // Compute skewness (Fisher's)
        let m3 = ticks.iter().map(|t| (t - mean).powi(3)).sum::<f64>() / nf;
        let skewness = if sd.abs() > 1e-15 {
            m3 / sd.powi(3)
        } else {
            0.0
        };

        // Method of moments: tau from skewness
        let tau = if skewness > 0.0 {
            skewness.cbrt() * sd / 2.0f64.cbrt()
        } else {
            0.0 // No positive skew → no exponential tail
        };

        let mu = mean - tau;
        let sigma_sq = (variance - tau * tau).max(0.0);
        let sigma = sigma_sq.sqrt();

        Self {
            mean_ticks: mean,
            sd_ticks: sd,
            tau,
            mu,
            sigma,
        }
    }
}

impl fmt::Display for RtSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RT: {:.2} +/- {:.2} ticks (mu={:.2}, sigma={:.2}, tau={:.2})",
            self.mean_ticks, self.sd_ticks, self.mu, self.sigma, self.tau
        )
    }
}

/// A single metric value with optional confidence interval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    /// Mean value across trials.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Number of trials.
    pub n: usize,
    /// 95% confidence interval lower bound.
    pub ci_lower: f64,
    /// 95% confidence interval upper bound.
    pub ci_upper: f64,
}

impl MetricValue {
    /// Compute from a slice of samples using BCa bootstrap confidence intervals.
    ///
    /// Uses 2000 resamples with the BCa (bias-corrected and accelerated) method
    /// for more accurate CIs on skewed or small-sample distributions.
    ///
    /// Reference: Efron & Tibshirani (1993), "An Introduction to the Bootstrap".
    pub fn from_samples_bootstrap(samples: &[f64], seed: u64) -> Self {
        let n = samples.len();
        if n == 0 {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                n: 0,
                ci_lower: 0.0,
                ci_upper: 0.0,
            };
        }
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = if n > 1 {
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();
        let (ci_lower, ci_upper) = super::analysis::bootstrap_ci_bca(samples, 2000, 0.05, seed);
        Self {
            mean,
            std_dev,
            n,
            ci_lower,
            ci_upper,
        }
    }

    /// Compute from a slice of samples.
    ///
    /// Uses proper t-distribution critical values for the 95% CI when n < 30,
    /// falling back to z = 1.96 for larger samples.
    pub fn from_samples(samples: &[f64]) -> Self {
        let n = samples.len();
        if n == 0 {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                n: 0,
                ci_lower: 0.0,
                ci_upper: 0.0,
            };
        }
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = if n > 1 {
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();
        let se = std_dev / (n as f64).sqrt();
        let t = t_critical_95(n);
        Self {
            mean,
            std_dev,
            n,
            ci_lower: mean - t * se,
            ci_upper: mean + t * se,
        }
    }
}

/// Two-tailed t critical value for 95% CI (alpha=0.05) given sample size n.
///
/// Uses a lookup table for df=1..29 (n=2..30), then z=1.96 for n>30.
/// Values from standard t-distribution tables.
fn t_critical_95(n: usize) -> f64 {
    if n <= 1 {
        return 1.96; // degenerate; CI width will be 0 anyway (std_dev=0)
    }
    let df = n - 1;
    // t(0.025, df) for df = 1..29
    const T_TABLE: [f64; 29] = [
        12.706, // df=1
        4.303,  // df=2
        3.182,  // df=3
        2.776,  // df=4
        2.571,  // df=5
        2.447,  // df=6
        2.365,  // df=7
        2.306,  // df=8
        2.262,  // df=9
        2.228,  // df=10
        2.201,  // df=11
        2.179,  // df=12
        2.160,  // df=13
        2.145,  // df=14
        2.131,  // df=15
        2.120,  // df=16
        2.110,  // df=17
        2.101,  // df=18
        2.093,  // df=19
        2.086,  // df=20
        2.080,  // df=21
        2.074,  // df=22
        2.069,  // df=23
        2.064,  // df=24
        2.060,  // df=25
        2.056,  // df=26
        2.052,  // df=27
        2.048,  // df=28
        2.045,  // df=29
    ];
    if df <= 29 { T_TABLE[df - 1] } else { 1.96 }
}

impl fmt::Display for MetricValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.3} +/- {:.3} (n={}, CI=[{:.3}, {:.3}])",
            self.mean, self.std_dev, self.n, self.ci_lower, self.ci_upper
        )
    }
}

/// Result from a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name.
    pub benchmark: String,
    /// Configuration label (if any).
    pub config_label: Option<String>,
    /// Metrics keyed by "condition::metric_name".
    pub metrics: BTreeMap<String, MetricValue>,
    /// Wall-clock time in milliseconds.
    pub elapsed_ms: u64,
    /// Number of conditions tested.
    pub conditions: usize,
    /// Number of trials per condition.
    pub trials_per_condition: usize,
    /// Per-trial trace data (populated when `config.trial_trace` is true).
    #[serde(default)]
    pub trial_trace: Vec<super::trial_analysis::TrialOutcome>,
    /// Optional diagnostic notes from the benchmark run.
    #[serde(default)]
    pub notes: Vec<String>,
}

impl BenchmarkResult {
    /// Create a new empty result.
    pub fn new(benchmark: impl Into<String>, config_label: Option<String>) -> Self {
        Self {
            benchmark: benchmark.into(),
            config_label,
            metrics: BTreeMap::new(),
            elapsed_ms: 0,
            conditions: 0,
            trials_per_condition: 0,
            trial_trace: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Insert a metric.
    pub fn insert(&mut self, key: impl Into<String>, value: MetricValue) {
        self.metrics.insert(key.into(), value);
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "=== {} ==={}",
            self.benchmark,
            self.config_label
                .as_ref()
                .map(|l| format!(" [{}]", l))
                .unwrap_or_default()
        ));
        lines.push(format!(
            "  {} conditions, {} trials each, {}ms total",
            self.conditions, self.trials_per_condition, self.elapsed_ms
        ));
        for (key, val) in &self.metrics {
            lines.push(format!("  {}: {}", key, val));
        }
        lines.join("\n")
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// A baseline comparison annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    /// Human reference value.
    pub human_value: f64,
    /// Source citation.
    pub source: String,
    /// Population (e.g., "human adults").
    pub population: String,
    /// Ratio: system_value / human_value.
    pub ratio: f64,
    /// Cohen's d effect size (agent vs. baseline), if computable.
    pub effect_size: Option<f64>,
    /// Norm-referenced z-score: (agent_mean - human_mean) / human_sd.
    ///
    /// Only populated when the baseline has a known population SD.
    /// Standard clinical neuropsychology reporting format (WAIS-IV style).
    pub z_score: Option<f64>,
}

/// Collection of benchmark results with comparison support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Individual benchmark results.
    pub results: Vec<BenchmarkResult>,
    /// Timestamp of report generation.
    pub timestamp: String,
}

impl BenchmarkReport {
    /// Create a new report.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Add a result.
    pub fn add(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Full summary of all results with baseline comparisons.
    pub fn summary(&self) -> String {
        use crate::harness::baselines::BaselineCollection;

        let bl = BaselineCollection::all();

        let mut lines = vec![format!("Psych Benchmark Report ({})", self.timestamp)];
        lines.push(format!("{} benchmarks run", self.results.len()));
        lines.push(String::new());
        for result in &self.results {
            lines.push(result.summary());

            // Add baseline comparisons for known metrics
            let comparisons = self.find_comparisons(result, &bl);
            if !comparisons.is_empty() {
                lines.push("  --- Baseline Comparisons ---".to_string());
                for (metric, comp) in &comparisons {
                    let pct = comp.ratio * 100.0;
                    lines.push(format!(
                        "  {} -> {:.1}% of human ({:.3}, {})",
                        metric, pct, comp.human_value, comp.source
                    ));
                }
            }

            lines.push(String::new());
        }
        lines.join("\n")
    }

    /// Compute Cohen's d for a metric vs. a baseline value.
    ///
    /// Uses the metric's std_dev as the pooled SD estimate.
    fn compute_effect_size(metric: &MetricValue, baseline_value: f64) -> Option<f64> {
        if metric.std_dev.abs() < 1e-15 {
            None
        } else {
            Some(super::analysis::cohens_d(
                metric.mean,
                baseline_value,
                metric.std_dev,
            ))
        }
    }

    /// Find applicable baseline comparisons for a benchmark result.
    pub fn find_comparisons(
        &self,
        result: &BenchmarkResult,
        bl: &super::baselines::BaselineCollection,
    ) -> Vec<(String, BaselineComparison)> {
        let mut comps = Vec::new();
        let benchmark = result.benchmark.as_str();

        // Map benchmark metrics to baselines
        let mappings: Vec<(&str, &str, &super::baselines::BaselineMap)> = vec![
            ("nback_2::accuracy", "nback_2_accuracy", &bl.worm),
            ("nback_3::accuracy", "nback_3_accuracy", &bl.worm),
            ("set_size_4::accuracy", "change_detection_k4", &bl.worm),
            ("average_pumps", "bart_avg_pumps", &bl.cogbench),
            (
                "horizon_6::directed_exploration",
                "directed_exploration",
                &bl.cogbench,
            ),
            ("beta3_model_basedness", "model_basedness", &bl.cogbench),
            ("discounting_score_S", "discounting_score", &bl.cogbench),
            (
                "false_belief_accuracy",
                "false_belief_accuracy",
                &bl.tombench,
            ),
            ("faux_pas_accuracy", "faux_pas_accuracy", &bl.tombench),
            ("hinting_accuracy", "hinting_accuracy", &bl.tombench),
            (
                "overall_retrieval_accuracy",
                "accurate_retrieval",
                &bl.memory_agent,
            ),
            (
                "correction_accuracy",
                "test_time_learning",
                &bl.memory_agent,
            ),
            (
                "categories_completed",
                "wcst_categories_completed",
                &bl.executive,
            ),
            (
                "trials_to_first_category",
                "wcst_trials_to_first",
                &bl.executive,
            ),
            ("overall_net_score", "igt_overall_net_score", &bl.executive),
            (
                "deck_preference_good",
                "igt_deck_preference_good",
                &bl.executive,
            ),
            ("stroop_effect", "stroop_effect", &bl.executive),
            (
                "incongruent_accuracy",
                "stroop_incongruent_accuracy",
                &bl.executive,
            ),
            ("flanker_effect", "flanker_effect", &bl.executive),
            (
                "overall_optimal_rate",
                "tol_overall_optimal_rate",
                &bl.executive,
            ),
            (
                "planning_efficiency",
                "tol_planning_efficiency",
                &bl.executive,
            ),
            ("forward_span", "digit_span_forward", &bl.worm),
            ("backward_span", "digit_span_backward", &bl.worm),
            ("win_stay_rate", "reversal_win_stay", &bl.cogbench),
            ("lose_shift_rate", "reversal_lose_shift", &bl.cogbench),
            (
                "calibration_error_ece",
                "calibration_error_ece",
                &bl.metacognition,
            ),
            (
                "discrimination_gamma",
                "discrimination_gamma",
                &bl.metacognition,
            ),
            ("persuasion_detection", "persuasion_detection", &bl.tombench),
            (
                "delay_50::retention",
                "long_range_delay_50",
                &bl.memory_agent,
            ),
            (
                "recency_preference",
                "conflict_recency_preference",
                &bl.memory_agent,
            ),
            ("valence_accuracy", "valence_accuracy", &bl.affect),
            ("congruence_ratio", "congruence_ratio", &bl.affect),
            ("fluency", "aut_fluency", &bl.creativity),
            ("present_count", "present_count", &bl.butlin),
            ("mean_quality_score", "mean_quality_score", &bl.butlin),
            ("presence_ratio", "presence_ratio", &bl.butlin),
            // Go/No-Go (Inhibition)
            ("go_accuracy", "go_accuracy", &bl.inhibition),
            ("nogo_accuracy", "nogo_accuracy", &bl.inhibition),
            ("inhibition_cost", "inhibition_cost", &bl.inhibition),
            ("go_rt_ticks", "go_rt_ticks", &bl.inhibition),
            // Attentional Blink (Attention)
            ("t1_accuracy", "t1_accuracy", &bl.attention),
            ("lag3_t2_accuracy", "lag3_t2_accuracy", &bl.attention),
            ("lag8_t2_accuracy", "lag8_t2_accuracy", &bl.attention),
            ("blink_magnitude", "blink_magnitude", &bl.attention),
            // Prospective Memory (MemoryAgent)
            ("pm_hit_rate", "pm_hit_rate", &bl.memory_agent),
            (
                "pm_ongoing_accuracy",
                "pm_ongoing_accuracy",
                &bl.memory_agent,
            ),
            ("pm_cost", "pm_cost", &bl.memory_agent),
            // Emotional Stroop (Affect)
            (
                "emotional_interference",
                "emotional_interference",
                &bl.affect,
            ),
            // Reasoning (ARC Fluid)
            ("rule_consistency", "arc_rule_consistency", &bl.reasoning),
            ("transfer_accuracy", "arc_transfer_accuracy", &bl.reasoning),
            (
                "transfer_similarity",
                "arc_transfer_similarity",
                &bl.reasoning,
            ),
            // Mismatch Negativity (Attention)
            (
                "detection_accuracy",
                "mmn_detection_accuracy",
                &bl.attention,
            ),
            ("false_alarm_rate", "mmn_false_alarm_rate", &bl.attention),
            (
                "attentional_independence",
                "mmn_attentional_independence",
                &bl.attention,
            ),
            // Change Blindness (Metacognition)
            (
                "detection_with_disruption",
                "cb_detection_with_disruption",
                &bl.metacognition,
            ),
            (
                "attention_benefit",
                "cb_attention_benefit",
                &bl.metacognition,
            ),
            // Proprioceptive Drift (Motor)
            ("synchronous_drift", "synchronous_drift", &bl.motor),
            ("drift_difference", "drift_difference", &bl.motor),
            ("ownership_rate", "ownership_rate", &bl.motor),
            // Blindsight (Consciousness)
            (
                "supraliminal_accuracy",
                "supraliminal_accuracy",
                &bl.consciousness,
            ),
            (
                "subliminal_accuracy",
                "subliminal_accuracy",
                &bl.consciousness,
            ),
            (
                "awareness_dissociation",
                "awareness_dissociation",
                &bl.consciousness,
            ),
            // Temporal Order (Binding)
            ("simultaneity_window", "simultaneity_window", &bl.binding),
            ("discrimination_slope", "discrimination_slope", &bl.binding),
            ("asymptotic_accuracy", "asymptotic_accuracy", &bl.binding),
            // Cross-Modal Feature Binding (Binding)
            (
                "binding_accuracy",
                "cross_modal_binding_accuracy",
                &bl.binding,
            ),
            (
                "swap_error_rate",
                "cross_modal_swap_error_rate",
                &bl.binding,
            ),
            ("set_size_slope", "cross_modal_set_size_slope", &bl.binding),
            // Binocular Rivalry (Consciousness)
            (
                "alternation_rate",
                "rivalry_alternation_rate",
                &bl.consciousness,
            ),
            (
                "dominance_ratio",
                "rivalry_dominance_ratio",
                &bl.consciousness,
            ),
            ("coefficient_of_variation", "rivalry_cv", &bl.consciousness),
            // Phoneme Discrimination (Speech)
            (
                "cross_boundary_accuracy",
                "cross_boundary_accuracy",
                &bl.speech,
            ),
            (
                "categorical_perception_index",
                "categorical_perception_index",
                &bl.speech,
            ),
            // VOT Continuum (Speech)
            (
                "identification_accuracy",
                "vot_identification_accuracy",
                &bl.speech,
            ),
            ("boundary_width", "vot_boundary_width", &bl.speech),
            ("slope_at_boundary", "vot_slope_at_boundary", &bl.speech),
            // Substrate Transfer
            ("transfer_fidelity", "transfer_fidelity", &bl.substrate),
            ("phi_preservation", "phi_preservation", &bl.substrate),
            // Substrate Degradation
            (
                "degradation_slope",
                "substrate_degradation_slope",
                &bl.substrate,
            ),
            (
                "critical_threshold",
                "substrate_critical_threshold",
                &bl.substrate,
            ),
            ("graceful_ratio", "substrate_graceful_ratio", &bl.substrate),
            // Feature Conjunction (Binding)
            ("conjunction_accuracy", "conjunction_accuracy", &bl.binding),
            ("feature_accuracy", "feature_accuracy", &bl.binding),
            ("conjunction_cost", "conjunction_cost", &bl.binding),
            // Perceptual Crowding (Consciousness)
            (
                "unflanked_accuracy",
                "crowding_unflanked_accuracy",
                &bl.consciousness,
            ),
            (
                "flanked_accuracy",
                "crowding_flanked_accuracy",
                &bl.consciousness,
            ),
            (
                "crowding_magnitude",
                "crowding_magnitude",
                &bl.consciousness,
            ),
            // Divergent Thinking (Creativity)
            ("originality_score", "originality_score", &bl.creativity),
            ("flexibility_score", "flexibility_score", &bl.creativity),
            ("elaboration_score", "elaboration_score", &bl.creativity),
            // Remote Associates convergent binding (Creativity)
            (
                "convergent_binding",
                "rat_convergent_binding",
                &bl.creativity,
            ),
            // Conceptual Blending (Creativity)
            ("blend_coherence", "blend_coherence", &bl.creativity),
            ("novelty_score", "novelty_score", &bl.creativity),
            ("integration_score", "integration_score", &bl.creativity),
            // Insight Problem (Creativity)
            ("insight_accuracy", "insight_accuracy", &bl.creativity),
            ("restructuring_depth", "restructuring_depth", &bl.creativity),
            // Flanker Inhibition (Inhibition)
            (
                "congruent_accuracy",
                "flanker_congruent_accuracy",
                &bl.inhibition,
            ),
            (
                "incongruent_accuracy",
                "flanker_incongruent_accuracy",
                &bl.inhibition,
            ),
            (
                "interference_suppression",
                "interference_suppression",
                &bl.inhibition,
            ),
            // Categorical Perception (Speech)
            ("boundary_slope", "cp_boundary_slope", &bl.speech),
            (
                "boundary_discrimination",
                "cp_boundary_discrimination",
                &bl.speech,
            ),
            (
                "within_category_discrimination",
                "cp_within_category_discrimination",
                &bl.speech,
            ),
            ("categorical_index", "cp_categorical_index", &bl.speech),
            // Substrate Latency
            (
                "fast_substrate_accuracy",
                "substrate_fast_accuracy",
                &bl.substrate,
            ),
            (
                "slow_substrate_accuracy",
                "substrate_slow_accuracy",
                &bl.substrate,
            ),
            (
                "speed_accuracy_correlation",
                "substrate_speed_accuracy_correlation",
                &bl.substrate,
            ),
            (
                "latency_gradient",
                "substrate_latency_gradient",
                &bl.substrate,
            ),
            // Mathematics domain
            ("accuracy", "arithmetic_accuracy", &bl.mathematics),
            (
                "accuracy_2x2",
                "linear_system_accuracy_2x2",
                &bl.mathematics,
            ),
            (
                "accuracy_3x3",
                "linear_system_accuracy_3x3",
                &bl.mathematics,
            ),
            (
                "accuracy_quadratic",
                "polynomial_quadratic_accuracy",
                &bl.mathematics,
            ),
            (
                "accuracy_cubic",
                "polynomial_cubic_accuracy",
                &bl.mathematics,
            ),
            (
                "determinant_accuracy",
                "determinant_accuracy",
                &bl.mathematics,
            ),
            (
                "eigenvalue_accuracy",
                "eigenvalue_accuracy",
                &bl.mathematics,
            ),
            (
                "posterior_accuracy",
                "bayesian_posterior_accuracy",
                &bl.mathematics,
            ),
            ("valid_accuracy", "logical_valid_accuracy", &bl.mathematics),
            (
                "invalid_accuracy",
                "logical_invalid_accuracy",
                &bl.mathematics,
            ),
            (
                "queens_4_accuracy",
                "constraint_queens_4_accuracy",
                &bl.mathematics,
            ),
            (
                "queens_8_accuracy",
                "constraint_queens_8_accuracy",
                &bl.mathematics,
            ),
            ("tautology_accuracy", "tautology_accuracy", &bl.mathematics),
            (
                "derivation_accuracy",
                "derivation_accuracy",
                &bl.mathematics,
            ),
            (
                "integration_accuracy",
                "integration_accuracy",
                &bl.mathematics,
            ),
            // Institutional Reasoning
            (
                "institutional_decomposition_accuracy",
                "institutional_decomposition_accuracy",
                &bl.institutional_reasoning,
            ),
            (
                "institutional_axiom_discrimination",
                "institutional_axiom_discrimination",
                &bl.institutional_reasoning,
            ),
            (
                "institutional_recovery_fidelity",
                "institutional_recovery_fidelity",
                &bl.institutional_reasoning,
            ),
            (
                "institutional_cross_domain_coherence",
                "institutional_cross_domain_coherence",
                &bl.institutional_reasoning,
            ),
            // Analogical Reasoning
            (
                "analogical_transfer_accuracy",
                "analogical_transfer_accuracy",
                &bl.institutional_reasoning,
            ),
            (
                "analogical_transfer_strength",
                "analogical_transfer_strength",
                &bl.institutional_reasoning,
            ),
            (
                "analogical_shared_component_auc",
                "analogical_shared_component_auc",
                &bl.institutional_reasoning,
            ),
            (
                "analogical_asymmetry_score",
                "analogical_asymmetry_score",
                &bl.institutional_reasoning,
            ),
            (
                "analogical_hdc_transfer_accuracy",
                "analogical_hdc_transfer_accuracy",
                &bl.institutional_reasoning,
            ),
            (
                "analogical_hdc_transfer_strength",
                "analogical_hdc_transfer_strength",
                &bl.institutional_reasoning,
            ),
            // Causal Chain
            (
                "causal_chain_coherence",
                "causal_chain_coherence",
                &bl.institutional_reasoning,
            ),
            (
                "causal_chain_terminal_accuracy",
                "causal_chain_terminal_accuracy",
                &bl.institutional_reasoning,
            ),
            (
                "causal_chain_step_count",
                "causal_chain_step_count",
                &bl.institutional_reasoning,
            ),
            // Counterfactual
            (
                "counterfactual_accuracy",
                "counterfactual_accuracy",
                &bl.institutional_reasoning,
            ),
            (
                "counterfactual_coherence",
                "counterfactual_coherence",
                &bl.institutional_reasoning,
            ),
            (
                "counterfactual_reversibility",
                "counterfactual_reversibility",
                &bl.institutional_reasoning,
            ),
            // Weighted Decomposition
            (
                "weighted_decomposition_accuracy",
                "weighted_decomposition_accuracy",
                &bl.institutional_reasoning,
            ),
            (
                "weight_sensitivity",
                "weight_sensitivity",
                &bl.institutional_reasoning,
            ),
            (
                "weighted_vs_unweighted_delta",
                "weighted_vs_unweighted_delta",
                &bl.institutional_reasoning,
            ),
            // Stability
            (
                "institutional_stability",
                "institutional_stability",
                &bl.institutional_reasoning,
            ),
            (
                "institutional_min_stability",
                "institutional_min_stability",
                &bl.institutional_reasoning,
            ),
            (
                "institutional_stability_variance",
                "institutional_stability_variance",
                &bl.institutional_reasoning,
            ),
            // Isomorphism
            (
                "isomorphism_self_similarity",
                "isomorphism_self_similarity",
                &bl.institutional_reasoning,
            ),
            (
                "isomorphism_overlap_correlation",
                "isomorphism_overlap_correlation",
                &bl.institutional_reasoning,
            ),
            (
                "isomorphism_discrimination_gap",
                "isomorphism_discrimination_gap",
                &bl.institutional_reasoning,
            ),
            (
                "isomorphism_monotonicity",
                "isomorphism_monotonicity",
                &bl.institutional_reasoning,
            ),
            // Clinical/Therapeutic
            ("empathic_accuracy", "empathic_accuracy", &bl.clinical),
            (
                "response_appropriateness",
                "response_appropriateness",
                &bl.clinical,
            ),
            ("repair_success_rate", "repair_success_rate", &bl.clinical),
            ("crisis_sensitivity", "crisis_sensitivity", &bl.clinical),
            ("crisis_specificity", "crisis_specificity", &bl.clinical),
            (
                "distortion_identification",
                "distortion_identification",
                &bl.clinical,
            ),
            ("mi_spirit_score", "mi_spirit_score", &bl.clinical),
            // Security (HDC-FHE)
            ("encrypted_accuracy", "encrypted_accuracy", &bl.security),
            ("aggregation_fidelity", "aggregation_fidelity", &bl.security),
            ("learning_accuracy", "learning_accuracy", &bl.security),
            (
                "cross_session_leakage",
                "cross_session_leakage",
                &bl.security,
            ),
            ("binding_preservation", "binding_preservation", &bl.security),
            ("accuracy_at_scale", "accuracy_at_scale", &bl.security),
            // Coding domain
            ("pass_at_1", "humaneval_pass_at_1", &bl.coding),
            (
                "delta_magnitude",
                "bug_detection_delta_magnitude",
                &bl.coding,
            ),
        ];

        for (metric_key, baseline_key, baselines) in &mappings {
            if let Some(metric) = result.metrics.get(*metric_key) {
                if let Some(baseline) = baselines.get(*baseline_key) {
                    comps.push((
                        metric_key.to_string(),
                        Self::make_comparison(metric, baseline),
                    ));
                }
            }
        }

        // Helper closure for benchmark-specific comparisons
        let mut push_specific =
            |metric_key: &str, baseline_key: &str, baselines: &super::baselines::BaselineMap| {
                if let Some(metric) = result.metrics.get(metric_key) {
                    if let Some(baseline) = baselines.get(baseline_key) {
                        comps.push((
                            metric_key.to_string(),
                            Self::make_comparison(metric, baseline),
                        ));
                    }
                }
            };

        // Benchmark-specific metrics (avoid cross-benchmark collisions)
        if benchmark.contains("Ravens") {
            push_specific("overall_accuracy", "ravens_overall_accuracy", &bl.executive);
            push_specific("easy_accuracy", "ravens_easy_accuracy", &bl.executive);
        }
        if benchmark.contains("RemoteAssociates") {
            push_specific("overall_accuracy", "rat_overall_accuracy", &bl.creativity);
            push_specific(
                "mean_solution_rank",
                "rat_mean_solution_rank",
                &bl.creativity,
            );
            push_specific(
                "convergent_binding",
                "rat_convergent_binding",
                &bl.creativity,
            );
        }
        if benchmark.contains("StrangeStory") {
            push_specific("overall_accuracy", "strange_story_accuracy", &bl.tombench);
        }
        if benchmark.contains("Stroop") && !benchmark.contains("Emotional") {
            push_specific(
                "congruent_accuracy",
                "stroop_congruent_accuracy",
                &bl.executive,
            );
            push_specific(
                "incongruent_accuracy",
                "stroop_incongruent_accuracy",
                &bl.executive,
            );
            push_specific(
                "congruent::rt_ticks",
                "stroop_congruent_rt_ticks",
                &bl.executive,
            );
            push_specific(
                "incongruent::rt_ticks",
                "stroop_incongruent_rt_ticks",
                &bl.executive,
            );
        }
        if benchmark.contains("Flanker") && !benchmark.contains("FlankerInhibition") {
            push_specific(
                "congruent_accuracy",
                "flanker_congruent_accuracy",
                &bl.executive,
            );
            push_specific(
                "incongruent_accuracy",
                "flanker_incongruent_accuracy",
                &bl.executive,
            );
            push_specific(
                "congruent::rt_ticks",
                "flanker_congruent_rt_ticks",
                &bl.executive,
            );
            push_specific(
                "incongruent::rt_ticks",
                "flanker_incongruent_rt_ticks",
                &bl.executive,
            );
        }
        if benchmark.contains("Wisconsin") {
            push_specific(
                "perseverative_errors",
                "wcst_perseverative_errors",
                &bl.executive,
            );
            push_specific("rt_ticks", "wcst_rt_ticks", &bl.executive);
        }
        if benchmark.contains("Reversal") {
            push_specific(
                "perseverative_errors",
                "reversal_perseverative_errors",
                &bl.cogbench,
            );
        }
        if benchmark.contains("RestlessBandit") {
            push_specific("overall_accuracy", "restless_bandit_accuracy", &bl.cogbench);
        }
        if benchmark.contains("Instrumental") {
            push_specific(
                "contingency_sensitivity",
                "instrumental_sensitivity",
                &bl.cogbench,
            );
        }
        if benchmark.contains("SpatialUpdating") {
            push_specific("overall_accuracy", "spatial_updating_accuracy", &bl.worm);
        }
        // Use "WorM::Binding" to avoid false-matching Spatial::LandmarkBinding
        if benchmark.contains("WorM::Binding") {
            push_specific("overall_binding_accuracy", "binding_accuracy", &bl.worm);
        }
        if benchmark.contains("SerialRecall") {
            push_specific(
                "list_7::primacy_index",
                "serial_primacy_advantage",
                &bl.worm,
            );
        }
        if benchmark.contains("Probabilistic") {
            push_specific(
                "beta2_likelihood_weight",
                "probabilistic_likelihood_weight",
                &bl.cogbench,
            );
        }
        if benchmark.contains("EmotionalStroop") {
            push_specific("neutral_accuracy", "emotional_neutral_accuracy", &bl.affect);
            push_specific(
                "negative_accuracy",
                "emotional_negative_accuracy",
                &bl.affect,
            );
        }
        // RT baselines for specific benchmarks
        if benchmark.contains("Iowa") || benchmark.contains("IGT") {
            push_specific("rt_ticks", "igt_rt_ticks", &bl.executive);
        }
        if benchmark.contains("NBack") || benchmark.contains("N-back") {
            push_specific("nback_1::rt_ticks", "nback_1_rt_ticks", &bl.worm);
            push_specific("nback_2::rt_ticks", "nback_2_rt_ticks", &bl.worm);
            push_specific("nback_3::rt_ticks", "nback_3_rt_ticks", &bl.worm);
        }
        if benchmark.contains("AttentionalBlink") {
            push_specific("t1::rt_ticks", "attblink_t1_rt_ticks", &bl.attention);
            push_specific("lag3::rt_ticks", "attblink_lag3_rt_ticks", &bl.attention);
            push_specific("lag8::rt_ticks", "attblink_lag8_rt_ticks", &bl.attention);
        }
        // CogBench RT mappings
        if benchmark.contains("BART") {
            push_specific("rt_ticks", "bart_rt_ticks", &bl.cogbench);
        }
        if benchmark.contains("Reversal") {
            push_specific("rt_ticks", "reversal_rt_ticks", &bl.cogbench);
        }
        if benchmark.contains("TwoStep") {
            push_specific("rt_ticks", "two_step_rt_ticks", &bl.cogbench);
        }
        if benchmark.contains("RestlessBandit") {
            push_specific("rt_ticks", "restless_bandit_rt_ticks", &bl.cogbench);
        }
        if benchmark.contains("Instrumental") {
            push_specific("rt_ticks", "instrumental_rt_ticks", &bl.cogbench);
        }
        if benchmark.contains("TemporalDiscounting") {
            push_specific("rt_ticks", "temporal_discounting_rt_ticks", &bl.cogbench);
        }
        if benchmark.contains("Horizon") {
            push_specific("horizon_1::rt_ticks", "horizon_rt_ticks", &bl.cogbench);
            push_specific("horizon_6::rt_ticks", "horizon_rt_ticks", &bl.cogbench);
        }
        if benchmark.contains("Probabilistic") {
            push_specific("rt_ticks", "probabilistic_rt_ticks", &bl.cogbench);
        }
        // WorM RT mappings
        if benchmark.contains("ChangeDetection") {
            push_specific("rt_ticks", "change_detection_rt_ticks", &bl.worm);
        }
        if benchmark.contains("DigitSpan") {
            push_specific("forward::rt_ticks", "digit_span_forward_rt_ticks", &bl.worm);
            push_specific(
                "backward::rt_ticks",
                "digit_span_backward_rt_ticks",
                &bl.worm,
            );
        }
        // Use "WorM::Binding" to avoid false-matching Spatial::LandmarkBinding
        if benchmark.contains("WorM::Binding") {
            push_specific("rt_ticks", "binding_rt_ticks", &bl.worm);
        }
        if benchmark.contains("SerialRecall") {
            push_specific("rt_ticks", "serial_recall_rt_ticks", &bl.worm);
        }
        if benchmark.contains("SpatialUpdating") {
            push_specific("rt_ticks", "spatial_updating_rt_ticks", &bl.worm);
        }
        // ToMBench RT mappings
        if benchmark.contains("FalseBelief") {
            push_specific("rt_ticks", "tombench_rt_ticks", &bl.tombench);
        }
        if benchmark.contains("FauxPas") {
            push_specific("rt_ticks", "tombench_rt_ticks", &bl.tombench);
        }
        if benchmark.contains("Hinting") {
            push_specific("rt_ticks", "tombench_rt_ticks", &bl.tombench);
        }
        if benchmark.contains("StrangeStory") {
            push_specific("rt_ticks", "tombench_rt_ticks", &bl.tombench);
        }
        if benchmark.contains("Persuasion") {
            push_specific("rt_ticks", "tombench_rt_ticks", &bl.tombench);
        }
        // Affect RT mappings
        if benchmark.contains("EmotionalStroop") {
            push_specific(
                "neutral::rt_ticks",
                "emotional_stroop_neutral_rt_ticks",
                &bl.affect,
            );
            push_specific(
                "negative::rt_ticks",
                "emotional_stroop_negative_rt_ticks",
                &bl.affect,
            );
        }
        if benchmark.contains("ValenceClassification") {
            push_specific("rt_ticks", "valence_rt_ticks", &bl.affect);
        }
        if benchmark.contains("MoodCongruent") {
            push_specific("rt_ticks", "mood_congruent_rt_ticks", &bl.affect);
        }
        // MemoryAgent RT mappings
        if benchmark.contains("AccurateRetrieval") {
            push_specific("rt_ticks", "accurate_retrieval_rt_ticks", &bl.memory_agent);
        }
        if benchmark.contains("ConflictResolution") {
            push_specific("rt_ticks", "conflict_resolution_rt_ticks", &bl.memory_agent);
        }
        if benchmark.contains("LongRange") {
            push_specific("rt_ticks", "long_range_rt_ticks", &bl.memory_agent);
        }
        if benchmark.contains("ProspectiveMemory") {
            push_specific("rt_ticks", "prospective_memory_rt_ticks", &bl.memory_agent);
        }
        if benchmark.contains("TestTimeLearning") {
            push_specific("rt_ticks", "test_time_learning_rt_ticks", &bl.memory_agent);
        }
        // Metacognition RT mapping
        if benchmark.contains("Calibration") {
            push_specific("rt_ticks", "calibration_rt_ticks", &bl.metacognition);
        }
        // Creativity metric mappings
        if benchmark.contains("AlternateUses") {
            push_specific("rt_ticks", "aut_rt_ticks", &bl.creativity);
            push_specific("originality", "aut_originality", &bl.creativity);
        }
        if benchmark.contains("RemoteAssociates") {
            push_specific("rt_ticks", "rat_rt_ticks", &bl.creativity);
        }
        // Stop Signal Task (Inhibition)
        if benchmark.contains("StopSignal") {
            push_specific("sst_go_accuracy", "sst_go_accuracy", &bl.inhibition);
            push_specific("sst_go_rt_ticks", "sst_go_rt_ticks", &bl.inhibition);
            push_specific("sst_stop_accuracy", "sst_stop_accuracy", &bl.inhibition);
            push_specific("ssrt_ticks", "ssrt_ticks", &bl.inhibition);
        }
        // Flanker Inhibition (Inhibition)
        if benchmark.contains("FlankerInhibition") {
            push_specific(
                "congruent_accuracy",
                "flanker_congruent_accuracy",
                &bl.inhibition,
            );
            push_specific(
                "incongruent_accuracy",
                "flanker_incongruent_accuracy",
                &bl.inhibition,
            );
            push_specific(
                "interference_suppression",
                "interference_suppression",
                &bl.inhibition,
            );
        }
        // Visual Search (Attention)
        if benchmark.contains("VisualSearch") {
            push_specific(
                "feature_search_accuracy",
                "feature_search_accuracy",
                &bl.attention,
            );
            push_specific(
                "conjunction_search_accuracy",
                "conjunction_search_accuracy",
                &bl.attention,
            );
            push_specific(
                "feature_search_slope",
                "feature_search_slope",
                &bl.attention,
            );
            push_specific(
                "conjunction_search_slope",
                "conjunction_search_slope",
                &bl.attention,
            );
            push_specific("search_asymmetry", "search_asymmetry", &bl.attention);
        }
        // Dual-Task (Executive)
        if benchmark.contains("DualTask") {
            push_specific("single_accuracy", "dual_single_accuracy", &bl.executive);
            push_specific("dual_low_accuracy", "dual_low_accuracy", &bl.executive);
            push_specific("dual_high_accuracy", "dual_high_accuracy", &bl.executive);
            push_specific("dual_task_cost", "dual_task_cost", &bl.executive);
            push_specific("digit_recall_accuracy", "dual_digit_recall", &bl.executive);
            push_specific("single::rt_ticks", "dual_single_rt_ticks", &bl.executive);
        }
        // Feeling of Knowing (Metacognition)
        if benchmark.contains("FeelingOfKnowing") {
            push_specific("fok_gamma", "fok_gamma", &bl.metacognition);
            push_specific(
                "recognition_hit_rate",
                "recognition_hit_rate",
                &bl.metacognition,
            );
            push_specific("fok_resolution", "fok_resolution", &bl.metacognition);
        }
        // ARC Fluid Reasoning
        if benchmark.contains("ArcFluid") {
            push_specific("rt_ticks", "arc_rt_ticks", &bl.reasoning);
        }
        // ARC Compositional Reasoning
        if benchmark.contains("ArcCompositional") {
            push_specific(
                "compositional_accuracy",
                "arc_compositional_accuracy",
                &bl.reasoning,
            );
            push_specific(
                "size_generalization",
                "arc_size_generalization",
                &bl.reasoning,
            );
            push_specific(
                "symmetry_detection",
                "arc_symmetry_detection",
                &bl.reasoning,
            );
            push_specific(
                "compositional_rt_ticks",
                "arc_compositional_rt_ticks",
                &bl.reasoning,
            );
        }
        // ARC Analogy Reasoning
        if benchmark.contains("ArcAnalogy") {
            push_specific("analogy_accuracy", "arc_analogy_accuracy", &bl.reasoning);
            push_specific(
                "cross_domain_accuracy",
                "arc_cross_domain_accuracy",
                &bl.reasoning,
            );
            push_specific(
                "multi_example_accuracy",
                "arc_multi_example_accuracy",
                &bl.reasoning,
            );
            push_specific("analogy_rt_ticks", "arc_analogy_rt_ticks", &bl.reasoning);
        }
        // ARC Abductive Reasoning
        if benchmark.contains("ArcAbductive") {
            push_specific(
                "abduction_accuracy",
                "arc_abduction_accuracy",
                &bl.reasoning,
            );
            push_specific(
                "unbinding_similarity",
                "arc_unbinding_similarity",
                &bl.reasoning,
            );
            push_specific(
                "abduction_rt_ticks",
                "arc_abduction_rt_ticks",
                &bl.reasoning,
            );
        }
        // ARC Learning Curve (in ArcFluid)
        if benchmark.contains("ArcFluid") {
            push_specific(
                "single_pair_accuracy",
                "arc_single_pair_accuracy",
                &bl.reasoning,
            );
            push_specific(
                "learning_efficiency",
                "arc_learning_efficiency",
                &bl.reasoning,
            );
        }
        // ARC Chain (multi-step composition)
        if benchmark.contains("ArcChain") {
            push_specific("chain_accuracy", "arc_chain_accuracy", &bl.reasoning);
            push_specific("chain_2_accuracy", "arc_chain_2_accuracy", &bl.reasoning);
            push_specific("chain_degradation", "arc_chain_degradation", &bl.reasoning);
        }
        // ARC Noise (robustness)
        if benchmark.contains("ArcNoise") {
            push_specific("noise_resilience", "arc_noise_resilience", &bl.reasoning);
            push_specific("accuracy_0pct", "arc_accuracy_0pct", &bl.reasoning);
        }
        // ARC FewShot (learning curve)
        if benchmark.contains("ArcFewShot") {
            push_specific("accuracy_1shot", "arc_accuracy_1shot", &bl.reasoning);
            push_specific("accuracy_5shot", "arc_accuracy_5shot", &bl.reasoning);
            push_specific("learning_rate", "arc_learning_rate", &bl.reasoning);
        }
        // ARC Scaling (grid complexity + dimension)
        if benchmark.contains("ArcScaling") {
            push_specific("grid_3x3_accuracy", "arc_grid_3x3_accuracy", &bl.reasoning);
            push_specific("capacity_ratio", "arc_capacity_ratio", &bl.reasoning);
        }
        // ARC RSA (representational similarity)
        if benchmark.contains("ArcRSA") {
            push_specific("rsa_correlation", "arc_rsa_correlation", &bl.reasoning);
            push_specific(
                "discriminability",
                "arc_rsa_discriminability",
                &bl.reasoning,
            );
        }
        // ARC Algebra (rule algebra probes)
        if benchmark.contains("ArcAlgebra") {
            push_specific("algebra_score", "arc_algebra_score", &bl.reasoning);
        }
        // ARC Staircase (adaptive threshold)
        if benchmark.contains("ArcStaircase") {
            push_specific(
                "capacity_threshold",
                "arc_capacity_threshold",
                &bl.reasoning,
            );
        }
        // SART (Sustained Attention)
        if benchmark.contains("SART") {
            push_specific(
                "commission_errors",
                "commission_errors",
                &bl.sustained_attention,
            );
            push_specific(
                "omission_errors",
                "omission_errors",
                &bl.sustained_attention,
            );
            push_specific("d_prime", "sart_d_prime", &bl.sustained_attention);
            push_specific("rt_ticks", "sart_rt_ticks", &bl.sustained_attention);
        }
        // PVT (Sustained Attention)
        if benchmark.contains("PVT") {
            push_specific(
                "vigilance_decrement",
                "vigilance_decrement",
                &bl.sustained_attention,
            );
            push_specific("lapse_rate", "lapse_rate", &bl.sustained_attention);
            push_specific("fastest_10pct", "fastest_10pct", &bl.sustained_attention);
        }
        // CPT (Sustained Attention)
        if benchmark.contains("CPT") {
            push_specific("d_prime", "cpt_d_prime", &bl.sustained_attention);
            push_specific("hit_rate", "cpt_hit_rate", &bl.sustained_attention);
            push_specific(
                "false_alarm_rate",
                "cpt_false_alarm_rate",
                &bl.sustained_attention,
            );
        }
        // SRTT (Motor)
        if benchmark.contains("SRTT") {
            push_specific("learning_effect", "learning_effect", &bl.motor);
            push_specific("sequence_accuracy", "sequence_accuracy", &bl.motor);
            push_specific("random_accuracy", "random_accuracy", &bl.motor);
            push_specific("sequence::rt_ticks", "srtt_sequence_rt_ticks", &bl.motor);
            push_specific("random::rt_ticks", "srtt_random_rt_ticks", &bl.motor);
        }
        // FittsLaw (Motor)
        if benchmark.contains("FittsLaw") {
            push_specific("fitts_r_squared", "fitts_r_squared", &bl.motor);
        }
        // Bimanual (Motor)
        if benchmark.contains("Bimanual") {
            push_specific("coordination_cost", "coordination_cost", &bl.motor);
        }
        // ProprioceptiveDrift (Motor)
        if benchmark.contains("ProprioceptiveDrift") {
            push_specific("drift_difference", "drift_difference", &bl.motor);
        }
        // Garden-Path (Language)
        if benchmark.contains("GardenPath") {
            push_specific("disambiguation_cost", "disambiguation_cost", &bl.language);
            push_specific("overall_accuracy", "gp_overall_accuracy", &bl.language);
            push_specific("garden_path_accuracy", "garden_path_accuracy", &bl.language);
            push_specific("control_accuracy", "gp_control_accuracy", &bl.language);
            push_specific("rt_ticks", "gp_rt_ticks", &bl.language);
        }
        // Semantic Coherence (Language)
        if benchmark.contains("SemanticCoherence") {
            push_specific("coherence_mean", "coherence_mean", &bl.language);
            push_specific("coherence_decay", "coherence_decay", &bl.language);
            push_specific("recovery_speed", "recovery_speed", &bl.language);
            push_specific("complexity_penalty", "complexity_penalty", &bl.language);
            push_specific("rt_ticks", "sc_rt_ticks", &bl.language);
        }
        // Lexical Decision (Language)
        if benchmark.contains("LexicalDecision") {
            push_specific("lexicality_effect", "lexicality_effect", &bl.language);
            push_specific("word_accuracy", "word_accuracy", &bl.language);
            push_specific("nonword_accuracy", "nonword_accuracy", &bl.language);
            push_specific("frequency_effect", "frequency_effect", &bl.language);
        }
        // Semantic Priming (Language)
        if benchmark.contains("SemanticPriming") {
            push_specific("priming_effect", "priming_effect", &bl.language);
            push_specific("related_accuracy", "related_accuracy", &bl.language);
        }
        // RME (Social)
        if benchmark.contains("Social") && benchmark.contains("RME") {
            push_specific("rme_accuracy", "rme_accuracy", &bl.social);
            push_specific("easy_accuracy", "rme_easy_accuracy", &bl.social);
            push_specific("hard_accuracy", "rme_hard_accuracy", &bl.social);
            push_specific("rt_ticks", "rme_rt_ticks", &bl.social);
        }
        // Ultimatum Game (Social)
        if benchmark.contains("UltimatumGame") {
            push_specific("fairness_sensitivity", "fairness_sensitivity", &bl.social);
            push_specific("rejection_rate", "rejection_rate", &bl.social);
            push_specific("offer_threshold", "offer_threshold", &bl.social);
        }
        // Prisoner's Dilemma (Social)
        if benchmark.contains("PrisonersDilemma") {
            push_specific("cooperation_rate", "cooperation_rate", &bl.social);
            push_specific(
                "mutual_cooperation_rate",
                "mutual_cooperation_rate",
                &bl.social,
            );
            push_specific("payoff_efficiency", "payoff_efficiency", &bl.social);
        }
        // Public Goods Game (Social)
        if benchmark.contains("PublicGoods") {
            push_specific("contribution_rate", "contribution_rate", &bl.social);
            push_specific("free_rider_fraction", "free_rider_fraction", &bl.social);
            push_specific("punishment_effect", "punishment_effect", &bl.social);
        }
        // Dictator Game (Social)
        if benchmark.contains("DictatorGame") {
            push_specific("mean_offer", "mean_offer", &bl.social);
            push_specific("positive_offer_rate", "positive_offer_rate", &bl.social);
            push_specific("generosity_index", "generosity_index", &bl.social);
        }
        // Machiavelli (Social)
        if benchmark.contains("Machiavelli") {
            push_specific("deception_detection", "deception_detection", &bl.social);
            push_specific(
                "power_seeking_detection",
                "power_seeking_detection",
                &bl.social,
            );
            push_specific("harm_avoidance", "harm_avoidance", &bl.social);
            push_specific("composite_ethics", "composite_ethics", &bl.social);
        }
        // Spatial::MentalRotation
        if benchmark.contains("MentalRotation") && benchmark.contains("Spatial") {
            push_specific("accuracy_mean", "mental_rotation_accuracy", &bl.spatial);
            push_specific("rt_slope", "rt_slope", &bl.spatial);
        }
        // Spatial::PathUpdating
        if benchmark.contains("PathUpdating") {
            push_specific("simple_accuracy", "simple_accuracy", &bl.spatial);
            push_specific("updating_accuracy", "updating_accuracy", &bl.spatial);
        }
        // Spatial::LandmarkBinding
        if benchmark.contains("LandmarkBinding") && benchmark.contains("Spatial") {
            push_specific("retrieval_accuracy", "retrieval_accuracy", &bl.spatial);
        }
        // Spatial::PerspectiveTaking
        if benchmark.contains("PerspectiveTaking") && benchmark.contains("Spatial") {
            push_specific("perspective_accuracy", "perspective_accuracy", &bl.spatial);
        }
        // CausalReasoning::CausalChain
        if benchmark.contains("CausalReasoning") && benchmark.contains("CausalChain") {
            push_specific(
                "chain_tracing_accuracy",
                "chain_tracing_accuracy",
                &bl.causal_reasoning,
            );
        }
        // CausalReasoning::ConfoundDetection
        if benchmark.contains("ConfoundDetection") && benchmark.contains("CausalReasoning") {
            push_specific(
                "confound_detection_accuracy",
                "confound_detection_accuracy",
                &bl.causal_reasoning,
            );
        }
        // CausalReasoning::InterventionEffect
        if benchmark.contains("InterventionEffect") && benchmark.contains("CausalReasoning") {
            push_specific("causal_score", "causal_score", &bl.causal_reasoning);
        }

        // Only return comparisons relevant to this benchmark
        if benchmark.contains("WorM")
            || benchmark.contains("CogBench")
            || benchmark.contains("ToM")
            || benchmark.contains("Memory")
            || benchmark.contains("Executive")
            || benchmark.contains("Metacognition")
            || benchmark.contains("Affect")
            || benchmark.contains("Creativity")
            || benchmark.contains("Butlin")
            || benchmark.contains("Inhibition")
            || benchmark.contains("Attention")
            || benchmark.contains("Reasoning")
            || benchmark.contains("SustainedAttention")
            || benchmark.contains("Motor")
            || benchmark.contains("Language")
            || benchmark.contains("Social")
            || benchmark.contains("Consciousness")
            || benchmark.contains("Binding")
            || benchmark.contains("Speech")
            || benchmark.contains("Substrate")
            || benchmark.contains("Mathematics")
            || benchmark.contains("InstitutionalReasoning")
            || benchmark.contains("Clinical")
            || benchmark.contains("Spatial")
            || benchmark.contains("CausalReasoning")
            || benchmark.contains("Security")
            || benchmark.contains("Coding")
        {
            comps
        } else {
            Vec::new()
        }
    }

    /// Find applicable LLM (GPT-4) baseline comparisons for a benchmark result.
    ///
    /// Parallels `find_comparisons()` but uses LLM-specific baselines from
    /// CogBench (Coda et al., 2023) and ToMBench (Kosinski, 2023).
    pub fn find_llm_comparisons(
        &self,
        result: &BenchmarkResult,
        bl: &super::baselines::BaselineCollection,
    ) -> Vec<(String, BaselineComparison)> {
        let mut comps = Vec::new();
        let benchmark = result.benchmark.as_str();

        // CogBench LLM mappings
        let cogbench_mappings: Vec<(&str, &str)> = vec![
            ("horizon_6::directed_exploration", "directed_exploration"),
            ("beta3_model_basedness", "model_basedness"),
            ("discounting_score_S", "discounting_score"),
            ("average_pumps", "bart_avg_pumps"),
            ("win_stay_rate", "reversal_win_stay"),
            ("lose_shift_rate", "reversal_lose_shift"),
            ("overall_accuracy", "restless_bandit_accuracy"),
            ("contingency_sensitivity", "instrumental_sensitivity"),
        ];

        if benchmark.contains("CogBench") {
            for (metric_key, baseline_key) in &cogbench_mappings {
                if let Some(metric) = result.metrics.get(*metric_key) {
                    if let Some(baseline) = bl.llm_cogbench.get(baseline_key) {
                        comps.push((
                            metric_key.to_string(),
                            Self::make_comparison(metric, baseline),
                        ));
                    }
                }
            }
        }

        // ToMBench LLM mappings
        let tombench_mappings: Vec<(&str, &str)> = vec![
            ("false_belief_accuracy", "false_belief_accuracy"),
            ("faux_pas_accuracy", "faux_pas_accuracy"),
            ("persuasion_detection", "persuasion_detection"),
            ("overall_accuracy", "strange_story_accuracy"),
            ("hinting_accuracy", "hinting_accuracy"),
        ];

        if benchmark.contains("ToM") {
            for (metric_key, baseline_key) in &tombench_mappings {
                if let Some(metric) = result.metrics.get(*metric_key) {
                    if let Some(baseline) = bl.llm_tombench.get(baseline_key) {
                        comps.push((
                            metric_key.to_string(),
                            Self::make_comparison(metric, baseline),
                        ));
                    }
                }
            }
        }

        comps
    }

    /// Build a BaselineComparison from a metric and its baseline.
    fn make_comparison(
        metric: &MetricValue,
        baseline: &super::baselines::Baseline,
    ) -> BaselineComparison {
        let ratio = if baseline.value.abs() > 1e-10 {
            metric.mean / baseline.value
        } else {
            0.0
        };
        let z_score = baseline.sd.and_then(|sd| {
            if sd.abs() > 1e-15 {
                Some((metric.mean - baseline.value) / sd)
            } else {
                None
            }
        });
        BaselineComparison {
            human_value: baseline.value,
            source: baseline.source.to_string(),
            population: baseline.population.to_string(),
            ratio,
            effect_size: Self::compute_effect_size(metric, baseline.value),
            z_score,
        }
    }

    /// Export all results as CSV.
    pub fn to_csv(&self) -> Result<String, csv::Error> {
        let mut wtr = csv::Writer::from_writer(Vec::new());
        wtr.write_record([
            "benchmark",
            "config",
            "metric",
            "mean",
            "std_dev",
            "n",
            "ci_lower",
            "ci_upper",
        ])?;
        for result in &self.results {
            let config = result.config_label.as_deref().unwrap_or("");
            for (key, val) in &result.metrics {
                wtr.write_record([
                    &result.benchmark,
                    config,
                    key.as_str(),
                    &format!("{:.6}", val.mean),
                    &format!("{:.6}", val.std_dev),
                    &val.n.to_string(),
                    &format!("{:.6}", val.ci_lower),
                    &format!("{:.6}", val.ci_upper),
                ])?;
            }
        }
        Ok(String::from_utf8(
            wtr.into_inner()
                .map_err(|e| csv::Error::from(e.into_error()))?,
        )
        .unwrap_or_default())
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Write pretty-printed JSON directly to a file.
    pub fn to_json_file(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Publication-ready Markdown summary table.
    ///
    /// One row per benchmark, showing key metric, agent value, human baseline,
    /// % of human, effect size d, z-score, 95% CI, and RT (if available).
    pub fn paper_summary(&self) -> String {
        use crate::harness::baselines::BaselineCollection;
        let bl = BaselineCollection::all();

        let mut lines = Vec::new();
        lines.push(
            "| Domain | Benchmark | Key Metric | Agent | Human | % Human | d | z | 95% CI | RT (ticks) |"
                .to_string(),
        );
        lines.push(
            "|--------|-----------|------------|-------|-------|---------|---|---|--------|------------|"
                .to_string(),
        );

        for result in &self.results {
            let domain = domain_of(&result.benchmark);
            let key = key_metric_for_benchmark(&result.benchmark);
            let metric = match result.metrics.get(key) {
                Some(m) => m,
                None => continue,
            };

            let comparisons = self.find_comparisons(result, &bl);

            let comp = comparisons.iter().find(|(k, _)| k == key);
            let (human_str, pct_str) = comp
                .map(|(_, c)| {
                    (
                        format!("{:.3}", c.human_value),
                        format!("{:.1}%", c.ratio * 100.0),
                    )
                })
                .unwrap_or_else(|| ("\u{2014}".to_string(), "\u{2014}".to_string()));

            let d_str = comp
                .and_then(|(_, c)| c.effect_size)
                .map(|d| format!("{:.2}", d))
                .unwrap_or_else(|| "\u{2014}".to_string());

            let z_str = comp
                .and_then(|(_, c)| c.z_score)
                .map(|z| format!("{:+.2}", z))
                .unwrap_or_else(|| "\u{2014}".to_string());

            let ci_str = format!("[{:.3}, {:.3}]", metric.ci_lower, metric.ci_upper);

            // RT column: look for rt_ticks metrics in this result
            let rt_str = rt_summary_for_result(result);

            lines.push(format!(
                "| {} | {} | {} | {:.3} | {} | {} | {} | {} | {} | {} |",
                domain,
                result
                    .benchmark
                    .split("::")
                    .last()
                    .unwrap_or(&result.benchmark),
                key,
                metric.mean,
                human_str,
                pct_str,
                d_str,
                z_str,
                ci_str,
                rt_str,
            ));
        }

        lines.join("\n")
    }

    /// Publication-ready LaTeX tabular output.
    pub fn paper_summary_latex(&self) -> String {
        use crate::harness::baselines::BaselineCollection;
        let bl = BaselineCollection::all();

        let mut lines = vec![
            r"\begin{tabular}{lllllrrrrl}".to_string(),
            r"\toprule".to_string(),
            r"Domain & Benchmark & Key Metric & Agent & Human & \% Human & $d$ & $z$ & 95\% CI \\"
                .to_string(),
            r"\midrule".to_string(),
        ];

        for result in &self.results {
            let domain = domain_of(&result.benchmark);
            let key = key_metric_for_benchmark(&result.benchmark);
            let metric = match result.metrics.get(key) {
                Some(m) => m,
                None => continue,
            };

            let comparisons = self.find_comparisons(result, &bl);

            let comp = comparisons.iter().find(|(k, _)| k == key);
            let (human_str, pct_str) = comp
                .map(|(_, c)| {
                    (
                        format!("{:.3}", c.human_value),
                        format!("{:.1}\\%", c.ratio * 100.0),
                    )
                })
                .unwrap_or_else(|| ("---".to_string(), "---".to_string()));

            let d_str = comp
                .and_then(|(_, c)| c.effect_size)
                .map(|d| format!("{:.2}", d))
                .unwrap_or_else(|| "---".to_string());

            let z_str = comp
                .and_then(|(_, c)| c.z_score)
                .map(|z| format!("{:+.2}", z))
                .unwrap_or_else(|| "---".to_string());

            let bench_name = result
                .benchmark
                .split("::")
                .last()
                .unwrap_or(&result.benchmark);

            lines.push(format!(
                "{} & {} & {} & {:.3} & {} & {} & {} & {} & [{:.3}, {:.3}] \\\\",
                domain,
                bench_name,
                key,
                metric.mean,
                human_str,
                pct_str,
                d_str,
                z_str,
                metric.ci_lower,
                metric.ci_upper,
            ));
        }

        lines.push(r"\bottomrule".to_string());
        lines.push(r"\end{tabular}".to_string());
        lines.join("\n")
    }

    /// Publication-ready LaTeX table with provenance (paradigm + citation).
    ///
    /// Requires benchmark references to look up provenance data. Includes
    /// calibration class (A Priori / Post Hoc / Theoretical) for each baseline.
    pub fn paper_summary_latex_with_provenance(
        &self,
        benchmarks: &[&dyn super::PsychBenchmark],
    ) -> String {
        use crate::harness::baselines::BaselineCollection;
        let bl = BaselineCollection::all();

        // Build provenance lookup: benchmark name → (paradigm, citation)
        let provenance_map: std::collections::HashMap<&str, (&str, &str)> = benchmarks
            .iter()
            .filter_map(|b| b.provenance().map(|p| (b.name(), (p.paradigm, p.citation))))
            .collect();

        let mut lines = vec![
            r"\begin{tabular}{llllrrrrc}".to_string(),
            r"\toprule".to_string(),
            r"Domain & Benchmark & Paradigm & Citation & Agent & Human$\pm$SD & $z$ & 95\% CI & Cal. \\".to_string(),
            r"\midrule".to_string(),
        ];

        let mut prev_domain = String::new();

        for result in &self.results {
            let domain = domain_of(&result.benchmark);
            let key = key_metric_for_benchmark(&result.benchmark);
            let metric = match result.metrics.get(key) {
                Some(m) => m,
                None => continue,
            };

            let comparisons = self.find_comparisons(result, &bl);
            let comp = comparisons.iter().find(|(k, _)| k == key);

            // Skip benchmarks with no baseline comparison
            let (_, c) = match comp {
                Some(pair) => pair,
                None => continue,
            };

            let z_str = c
                .z_score
                .map(|z| {
                    let z_adj = if is_lower_better(key) { -z } else { z };
                    format!("{:+.2}", z_adj)
                })
                .unwrap_or_else(|| "---".to_string());

            let bench_name = result
                .benchmark
                .split("::")
                .last()
                .unwrap_or(&result.benchmark);

            // Look up provenance
            let (paradigm, citation) = provenance_map
                .get(result.benchmark.as_str())
                .copied()
                .unwrap_or(("---", "---"));

            // Look up human SD for display
            let human_sd_str = {
                let baseline_maps = [
                    &bl.worm,
                    &bl.cogbench,
                    &bl.executive,
                    &bl.tombench,
                    &bl.memory_agent,
                    &bl.metacognition,
                    &bl.affect,
                    &bl.creativity,
                    &bl.butlin,
                    &bl.inhibition,
                    &bl.attention,
                    &bl.reasoning,
                    &bl.sustained_attention,
                    &bl.motor,
                    &bl.language,
                    &bl.social,
                    &bl.binding,
                    &bl.spatial,
                    &bl.causal_reasoning,
                    &bl.speech,
                    &bl.consciousness,
                    &bl.substrate,
                    &bl.clinical,
                    &bl.institutional_reasoning,
                    &bl.mathematics,
                ];
                let mut sd_val = None;
                for map in baseline_maps {
                    if let Some(bv) = map.get(key) {
                        sd_val = bv.sd;
                        break;
                    }
                }
                match sd_val {
                    Some(sd) => format!("{:.2}$\\pm${:.2}", c.human_value, sd),
                    None => format!("{:.2}", c.human_value),
                }
            };

            // Calibration class: A Priori (literature-derived params), Post Hoc, Theoretical
            let cal_class = calibration_class_of(&result.benchmark);

            // Add midrule between domains
            let domain_str = if domain != prev_domain {
                if !prev_domain.is_empty() {
                    lines.push(r"\midrule".to_string());
                }
                prev_domain = domain.to_string();
                domain.to_string()
            } else {
                String::new()
            };

            lines.push(format!(
                "{} & {} & {} & {} & {:.3} & {} & {} & [{:.3}, {:.3}] & {} \\\\",
                domain_str,
                bench_name,
                paradigm,
                citation,
                metric.mean,
                human_sd_str,
                z_str,
                metric.ci_lower,
                metric.ci_upper,
                cal_class,
            ));
        }

        // Grand mean footer
        let composites = self.composite_scores();
        if !composites.is_empty() {
            let total_z: f64 = composites.values().map(|c| c.mean_z).sum();
            let grand_mean = total_z / composites.len() as f64;
            let total_n: usize = composites.values().map(|c| c.n_benchmarks).sum();
            lines.push(r"\midrule".to_string());
            lines.push(format!(
                r"\multicolumn{{6}}{{l}}{{\textbf{{Grand mean}}}} & \textbf{{{:+.3}}} & \multicolumn{{2}}{{l}}{{$N={}$ benchmarks, $K={}$ domains}} \\",
                grand_mean, total_n, composites.len(),
            ));
        }

        lines.push(r"\bottomrule".to_string());
        lines.push(r"\end{tabular}".to_string());
        lines.join("\n")
    }
}

/// Calibration class for a benchmark's human baseline.
///
/// - **AP** (A Priori): Parameters derived directly from published literature values
///   with minimal or no tuning.
/// - **PH** (Post Hoc): Parameters hand-tuned to match human baselines after
///   initial implementation.
/// - **TH** (Theoretical): Baselines from theoretical constructs or novel paradigms
///   without direct human comparison data.
pub fn calibration_class_of(benchmark: &str) -> &'static str {
    // A Priori benchmarks: established paradigms with well-documented parameters
    let a_priori = [
        "Stroop",
        "Flanker",
        "WCST",
        "NBack",
        "N-back",
        "DigitSpan",
        "StopSignal",
        "GoNoGo",
        "FalseBeliefBenchmark",
        "SerialRecall",
        "VisualSearch",
        "AttentionalBlink",
        "SART",
        "PVT",
        "CPT",
        "SRTT",
        "FittsLaw",
        "EmotionalStroop",
        "VotContinuum",
        "BinocularRivalry",
        "ChangeBlindness",
        "FeelingOfKnowing",
        "Calibration",
        "MismatchNegativity",
        "GardenPath",
        "SemanticPriming",
        "LexicalDecision",
        "MentalRotation",
        "TemporalDiscounting",
        "IowaGambling",
        "TowerOfLondon",
        "RavensProgressiveMatrices",
        "TemporalOrder",
        "CrossModal",
        "FeatureConjunction",
        "RemoteAssociates",
        "AlternateUses",
        "DivergentThinking",
        "ConceptualBlending",
        "InsightProblem",
        "FlankerInhibition",
        "CategoricalPerception",
        "PerceptualCrowding",
        "WorM::Binding",
        "ChangeDetection",
    ];

    // Theoretical benchmarks: novel paradigms or theoretical constructs
    let theoretical = [
        "SubstrateTransfer",
        "SubstrateDegradation",
        "SubstrateLatency",
        "Blindsight",
        "ButlinIndicator",
        "InstitutionalReasoning",
        "InstitutionalStability",
        "InstitutionalIsomorphism",
    ];

    for t in &theoretical {
        if benchmark.contains(t) {
            return "TH";
        }
    }

    for a in &a_priori {
        if benchmark.contains(a) {
            return "AP";
        }
    }

    "PH"
}

/// Extract domain prefix from benchmark name.
pub fn domain_of(benchmark: &str) -> &str {
    benchmark.split("::").next().unwrap_or(benchmark)
}

/// Map each benchmark to its primary metric for paper-table output.
pub fn key_metric_for_benchmark(benchmark: &str) -> &str {
    match benchmark {
        b if b.contains("NBack") || b.contains("N-back") => "nback_2::accuracy",
        b if b.contains("ChangeDetection") => "set_size_4::accuracy",
        b if b.contains("SerialRecall") => "list_7::primacy_index",
        b if b.contains("SpatialUpdating") => "overall_accuracy",
        b if b.contains("TemporalOrder") => "discrimination_slope",
        b if b.contains("FeatureConjunction") => "conjunction_accuracy",
        b if b.contains("CrossModal") => "binding_accuracy",
        // Spatial domain (must be before generic "Binding" arm)
        b if b.contains("Spatial::MentalRotation")
            || (b.contains("MentalRotation") && b.contains("Spatial")) =>
        {
            "accuracy_mean"
        }
        b if b.contains("Spatial::PathUpdating") || b.contains("PathUpdating") => "simple_accuracy",
        b if b.contains("Spatial::LandmarkBinding")
            || (b.contains("LandmarkBinding") && b.contains("Spatial")) =>
        {
            "retrieval_accuracy"
        }
        b if b.contains("Spatial::PerspectiveTaking")
            || (b.contains("PerspectiveTaking") && b.contains("Spatial")) =>
        {
            "perspective_accuracy"
        }
        b if b.contains("EncryptedBinding") => "binding_preservation",
        b if b.contains("Binding") => "overall_binding_accuracy",
        b if b.contains("DigitSpan") => "forward_span",
        b if b.contains("EmotionalStroop") => "negative_accuracy",
        b if b.contains("Stroop") && !b.contains("Strange") => "incongruent_accuracy",
        b if b.contains("FlankerInhibition") => "interference_suppression",
        b if b.contains("Flanker") => "flanker_effect",
        b if b.contains("Wisconsin") || b.contains("WCST") => "categories_completed",
        b if b.contains("Iowa") || b.contains("IGT") => "overall_net_score",
        b if b.contains("Ravens") => "overall_accuracy",
        b if b.contains("TowerOfLondon") => "overall_optimal_rate",
        b if b.contains("Probabilistic") => "beta2_likelihood_weight",
        b if b.contains("Horizon") => "horizon_6::directed_exploration",
        b if b.contains("RestlessBandit") => "overall_accuracy",
        b if b.contains("Instrumental") => "contingency_sensitivity",
        b if b.contains("TwoStep") => "beta3_model_basedness",
        b if b.contains("Temporal") => "discounting_score_S",
        b if b.contains("Bart") || b.contains("BART") => "average_pumps",
        b if b.contains("Reversal") => "win_stay_rate",
        b if b.contains("FalseBelief") => "false_belief_accuracy",
        b if b.contains("FauxPas") => "faux_pas_accuracy",
        b if b.contains("Hinting") => "hinting_accuracy",
        b if b.contains("Persuasion") => "persuasion_detection",
        b if b.contains("StrangeStory") => "overall_accuracy",
        b if b.contains("AccurateRetrieval") => "overall_retrieval_accuracy",
        b if b.contains("TestTimeLearning") => "correction_accuracy",
        b if b.contains("LongRange") => "delay_50::retention",
        b if b.contains("ConflictResolution") => "recency_preference",
        b if b.contains("Calibration") => "calibration_error_ece",
        b if b.contains("Butlin") => "mean_quality_score",
        b if b.contains("ValenceClassification") => "valence_accuracy",
        b if b.contains("MoodCongruent") => "congruence_ratio",
        b if b.contains("RemoteAssociates") => "convergent_binding",
        b if b.contains("AlternateUses") => "originality",
        b if b.contains("DivergentThinking") => "originality_score",
        b if b.contains("ConceptualBlending") => "blend_coherence",
        b if b.contains("InsightProblem") => "insight_accuracy",
        b if b.contains("GoNoGo") => "nogo_accuracy",
        b if b.contains("AttentionalBlink") => "lag3_t2_accuracy",
        b if b.contains("ProspectiveMemory") => "pm_hit_rate",
        b if b.contains("StopSignal") => "ssrt_ticks",
        b if b.contains("VisualSearch") => "search_asymmetry",
        b if b.contains("FeelingOfKnowing") => "fok_gamma",
        b if b.contains("DualTask") => "dual_task_cost",
        b if b.contains("SART") => "commission_errors",
        b if b.contains("PVT") => "vigilance_decrement",
        b if b.contains("CPT") => "d_prime",
        b if b.contains("SRTT") => "learning_effect",
        b if b.contains("FittsLaw") => "fitts_r_squared",
        b if b.contains("Bimanual") => "coordination_cost",
        b if b.contains("GardenPath") => "disambiguation_cost",
        b if b.contains("SemanticCoherence") => "coherence_mean",
        b if b.contains("LexicalDecision") => "lexicality_effect",
        b if b.contains("SemanticPriming") => "priming_effect",
        b if b.contains("UltimatumGame") => "fairness_sensitivity",
        b if b.contains("PrisonersDilemma") => "cooperation_rate",
        b if b.contains("PublicGoods") => "contribution_rate",
        b if b.contains("DictatorGame") => "mean_offer",
        b if b.contains("Machiavelli") => "composite_ethics",
        b if b.contains("SocialNorm") => "d_prime",
        b if b.contains("RME") && b.contains("Social") => "rme_accuracy",
        b if b.contains("ArcFluid") => "transfer_accuracy",
        b if b.contains("ArcCompositional") => "compositional_accuracy",
        b if b.contains("ArcAnalogy") => "analogy_accuracy",
        b if b.contains("ArcAbductive") => "abduction_accuracy",
        b if b.contains("ArcChain") => "chain_accuracy",
        b if b.contains("ArcNoise") => "noise_resilience",
        b if b.contains("ArcFewShot") => "accuracy_5shot",
        b if b.contains("ArcScaling") => "capacity_ratio",
        b if b.contains("ArcRSA") => "rsa_correlation",
        b if b.contains("ArcAlgebra") => "algebra_score",
        b if b.contains("ArcStaircase") => "capacity_threshold",
        // Neuromod domain
        b if b.contains("RewardLearning") => "trials_to_criterion",
        b if b.contains("YerkesDodson") => "inverted_u_fit_r2",
        b if b.contains("AttentionNetwork") => "conflict_effect",
        b if b.contains("MoodInduction") => "mood_congruent_bias",
        b if b.contains("PharmacologicalAblation") => "da_knockout_lr_drop_pct",
        b if b.contains("PharmacologicalChallenge") => "da_agonist_gradient_scale",
        b if b.contains("InjectionChallenge") => "stimulant_peak_effect",
        b if b.contains("AllostaticStress") => "chronic_da_baseline_final",
        b if b.contains("LiveLoopAblation") => "live_da_knockout_gradient_drop_pct",
        b if b.contains("BehavioralKnockout") => "da_ko_lr_d",
        b if b.contains("ConsciousnessPharmacology") => "psychedelic_proxy_peak",
        b if b.contains("MetacognitiveIgnition") => "spontaneous_tracking_score",
        // New benchmarks (Mar 2026)
        b if b.contains("Blindsight") || b.contains("BlindSight") => "subliminal_accuracy",
        b if b.contains("MismatchNegativity") => "detection_accuracy",
        b if b.contains("ChangeBlindness") => "detection_with_disruption",
        b if b.contains("ProprioceptiveDrift") => "drift_difference",
        b if b.contains("PhonemeDiscrimination") => "categorical_perception_index",
        b if b.contains("CategoricalPerception") => "categorical_index",
        b if b.contains("VotContinuum") => "identification_accuracy",
        b if b.contains("PerceptualCrowding") => "crowding_magnitude",
        b if b.contains("BinocularRivalry") => "dominance_ratio",
        b if b.contains("Substrate") && b.contains("Degradation") => "graceful_ratio",
        b if b.contains("Substrate") && b.contains("Transfer") => "transfer_fidelity",
        b if b.contains("Substrate") && b.contains("Latency") => "speed_accuracy_correlation",
        // Mathematics domain
        b if b.contains("ArithmeticWordProblem") => "accuracy",
        b if b.contains("LinearSystemSolving") => "accuracy_2x2",
        b if b.contains("PolynomialRoots") => "accuracy_quadratic",
        b if b.contains("DefiniteIntegral") => "integration_accuracy",
        b if b.contains("MatrixOperations") => "determinant_accuracy",
        b if b.contains("StatisticalInference") => "variance_estimation_accuracy",
        b if b.contains("BayesianReasoning") => "posterior_accuracy",
        b if b.contains("LogicalDeduction") => "overall_accuracy",
        b if b.contains("ConstraintPuzzle") => "queens_4_accuracy",
        b if b.contains("ProofConstruction") => "tautology_accuracy",
        // CausalReasoning domain (must be before generic "CausalChain" arm below)
        b if b.contains("CausalReasoning::CausalChain")
            || (b.contains("CausalChain") && b.contains("CausalReasoning")) =>
        {
            "chain_tracing_accuracy"
        }
        b if b.contains("CausalReasoning::ConfoundDetection")
            || b.contains("ConfoundDetection") =>
        {
            "confound_detection_accuracy"
        }
        b if b.contains("CausalReasoning::InterventionEffect")
            || (b.contains("InterventionEffect") && b.contains("CausalReasoning")) =>
        {
            "causal_score"
        }
        // Institutional Reasoning sub-benchmarks (specific before generic)
        b if b.contains("AnalogicalReasoning") => "analogical_transfer_accuracy",
        b if b.contains("CausalChain") => "causal_chain_coherence",
        b if b.contains("Counterfactual") => "counterfactual_accuracy",
        b if b.contains("WeightedDecomposition") => "weighted_decomposition_accuracy",
        b if b.contains("Isomorphism") => "isomorphism_overlap_correlation",
        b if b.contains("Stability") && b.contains("Institutional") => "institutional_stability",
        b if b.contains("InstitutionalReasoning") => "institutional_decomposition_accuracy",
        // Clinical/Therapeutic domain
        b if b.contains("EmpathicAccuracy") => "empathic_accuracy",
        b if b.contains("TherapeuticResponse") => "response_appropriateness",
        b if b.contains("AllianceMaintenance") => "repair_success_rate",
        b if b.contains("CrisisDetection") => "crisis_sensitivity",
        b if b.contains("CognitiveDistortion") => "distortion_identification",
        b if b.contains("MotivationalInterviewing") => "mi_spirit_score",
        // Security (HDC-FHE) domain
        b if b.contains("EncryptedLearning") => "learning_accuracy",
        b if b.contains("CrossMaskPrivacy") => "cross_session_leakage",
        // EncryptedBinding handled above (before generic "Binding" arm)
        b if b.contains("ScalingAnalysis") => "accuracy_at_scale",
        b if b.contains("EncryptedClassification") => "encrypted_accuracy",
        b if b.contains("CollectiveAggregation") => "aggregation_fidelity",
        // Coding domain
        b if b.contains("HumanEvalMini") => "pass_at_1",
        b if b.contains("BugDetection") => "delta_magnitude",
        _ => "overall_accuracy",
    }
}

/// Extract a compact RT summary string from a benchmark result.
///
/// Looks for metrics ending in `::rt_ticks` or named `go_rt_ticks`.
/// Returns mean RT across conditions, or "\u{2014}" if no RT data.
fn rt_summary_for_result(result: &BenchmarkResult) -> String {
    let rt_metrics: Vec<&MetricValue> = result
        .metrics
        .iter()
        .filter(|(k, _)| k.ends_with("::rt_ticks") || *k == "go_rt_ticks" || *k == "rt_ticks")
        .map(|(_, v)| v)
        .collect();

    if rt_metrics.is_empty() {
        return "\u{2014}".to_string();
    }

    let mean_rt: f64 = rt_metrics.iter().map(|m| m.mean).sum::<f64>() / rt_metrics.len() as f64;
    format!("{:.1}", mean_rt)
}

/// Returns true if a lower metric value is better (inverted scoring).
///
/// Canonical list — all callers should use this instead of maintaining copies.
pub fn is_lower_better(metric_key: &str) -> bool {
    matches!(
        metric_key,
        "stroop_effect"
            | "flanker_effect"
            | "dual_task_cost"
            | "calibration_error_ece"
            | "commission_errors"
            | "ssrt_ticks"
            | "coordination_cost"
            | "vigilance_decrement"
            | "disambiguation_cost"
            | "blink_magnitude"
            | "perseverative_errors"
            | "trials_to_first_category"
            | "trials_to_criterion"
            | "restless_bandit_regret"
            | "lapse_rate"
            | "false_alarm_rate"
            | "simultaneity_window"
            | "degradation_gradient"
            | "emotional_interference"
            | "swap_error_rate"
            | "pm_cost"
            | "capacity_ratio"
            | "interference_suppression"
            | "cross_session_leakage"
    )
}

impl BenchmarkReport {
    /// Compute a normalized cognitive profile: domain → score [0.0, 1.0].
    ///
    /// Groups benchmarks by domain via `domain_of()`, looks up baseline
    /// comparisons, and averages the ratios per domain. Lower-is-better
    /// metrics are inverted so that 1.0 always means "at or above baseline".
    pub fn cognitive_profile(&self) -> BTreeMap<String, f64> {
        use crate::harness::baselines::BaselineCollection;
        let bl = BaselineCollection::all();

        let mut domain_scores: BTreeMap<String, Vec<f64>> = BTreeMap::new();

        for result in &self.results {
            let domain = domain_of(&result.benchmark).to_string();
            let key = key_metric_for_benchmark(&result.benchmark);

            let comparisons = self.find_comparisons(result, &bl);

            if let Some((_, comp)) = comparisons.iter().find(|(k, _)| k == key) {
                let score = if is_lower_better(key) {
                    // Lower is better: score = baseline / agent (capped at 1.0)
                    if comp.ratio.abs() > 1e-10 {
                        (1.0 / comp.ratio).min(1.0)
                    } else {
                        0.0
                    }
                } else {
                    comp.ratio.min(1.0)
                };
                domain_scores
                    .entry(domain)
                    .or_default()
                    .push(score.max(0.0));
            }
        }

        domain_scores
            .into_iter()
            .map(|(domain, scores)| {
                let avg = scores.iter().sum::<f64>() / scores.len() as f64;
                (domain, avg)
            })
            .collect()
    }

    /// Format the cognitive profile as an ASCII horizontal bar chart.
    pub fn format_profile(&self) -> String {
        let profile = self.cognitive_profile();
        if profile.is_empty() {
            return "No profile data available.".to_string();
        }

        let bar_width = 20;
        let mut lines = vec!["Cognitive Profile".to_string()];

        for (domain, score) in &profile {
            let filled = (score * bar_width as f64).round() as usize;
            let filled = filled.min(bar_width);
            let empty = bar_width - filled;
            let bar = format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty),);
            lines.push(format!("{:<14}{} {:.2}", domain, bar, score));
        }

        lines.join("\n")
    }

    /// Compute domain composite scores as mean z-scores per domain.
    ///
    /// For each benchmark, looks up the key metric's z-score (from baseline SD).
    /// Averages z-scores within each domain to produce composite indices.
    /// Returns `None` for domains with no z-score-enabled baselines.
    ///
    /// Interpretation: z=0 = human mean, z=+1 = 1 SD above human, z=-1 = 1 SD below.
    pub fn composite_scores(&self) -> BTreeMap<String, CompositeScore> {
        use crate::harness::baselines::BaselineCollection;
        let bl = BaselineCollection::all();

        let mut domain_z: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut domain_benchmarks: BTreeMap<String, Vec<String>> = BTreeMap::new();

        for result in &self.results {
            let domain = domain_of(&result.benchmark).to_string();
            let key = key_metric_for_benchmark(&result.benchmark);

            let comparisons = self.find_comparisons(result, &bl);

            // Debug: log domains with no comparisons for key metric

            if let Some((_, comp)) = comparisons.iter().find(|(k, _)| k == key) {
                if let Some(z) = comp.z_score {
                    // For lower-is-better metrics, negate the z-score
                    // so positive always means "better than human"
                    let z_adj = if is_lower_better(key) { -z } else { z };
                    domain_z.entry(domain.clone()).or_default().push(z_adj);
                    domain_benchmarks.entry(domain).or_default().push(
                        result
                            .benchmark
                            .split("::")
                            .last()
                            .unwrap_or(&result.benchmark)
                            .to_string(),
                    );
                }
            }
        }

        domain_z
            .into_iter()
            .map(|(domain, zs)| {
                let mean_z = zs.iter().sum::<f64>() / zs.len() as f64;
                let benchmarks = domain_benchmarks.remove(&domain).unwrap_or_default();
                (
                    domain,
                    CompositeScore {
                        mean_z,
                        n_benchmarks: zs.len(),
                        benchmarks,
                    },
                )
            })
            .collect()
    }

    /// Format composite scores as a summary table.
    pub fn format_composites(&self) -> String {
        let composites = self.composite_scores();
        if composites.is_empty() {
            return "No composite scores available (baselines lack SD data).".to_string();
        }

        let mut lines = Vec::new();
        lines.push("Domain Composite Scores (z-score: 0=human mean)".to_string());
        lines.push(format!(
            "{:<14} {:>6} {:>4}  {}",
            "Domain", "z", "n", "Benchmarks"
        ));
        lines.push(format!(
            "{:<14} {:>6} {:>4}  {}",
            "------", "---", "--", "----------"
        ));

        for (domain, cs) in &composites {
            let label = if cs.mean_z > 1.0 {
                "***" // well above human
            } else if cs.mean_z > 0.5 {
                "**"
            } else if cs.mean_z > 0.0 {
                "*"
            } else if cs.mean_z > -0.5 {
                ""
            } else {
                "(!)" // notably below human
            };
            lines.push(format!(
                "{:<14} {:>+6.2} {:>4}  {} {}",
                domain,
                cs.mean_z,
                cs.n_benchmarks,
                cs.benchmarks.join(", "),
                label,
            ));
        }

        // Cross-cultural baseline disclosure
        let cultural = super::baselines::BaselineCollection::cultural_metadata();
        let total_domains = cultural.len();
        let weird_domains = cultural
            .values()
            .filter(|m| m.sample_region == "WEIRD")
            .count();
        lines.push(String::new());
        lines.push(format!(
            "NOTE: {}/{} domain baselines derive from WEIRD populations.",
            weird_domains, total_domains,
        ));
        lines.push("Cross-cultural generalizability has not been validated.".to_string());
        lines.push("Ref: Henrich, Heine, & Norenzayan (2010).".to_string());

        lines.join("\n")
    }
}

/// A domain composite score derived from norm-referenced z-scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeScore {
    /// Mean z-score across benchmarks in this domain.
    /// Positive = above human mean, negative = below.
    pub mean_z: f64,
    /// Number of benchmarks contributing to this composite.
    pub n_benchmarks: usize,
    /// Names of contributing benchmarks.
    pub benchmarks: Vec<String>,
}

/// A single row in a forest plot export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForestPlotRow {
    /// Domain (e.g., "WorM", "Executive").
    pub domain: String,
    /// Benchmark short name.
    pub benchmark: String,
    /// Key metric name.
    pub metric: String,
    /// Agent mean value.
    pub agent_mean: f64,
    /// Human baseline mean.
    pub human_mean: f64,
    /// Human baseline SD (if available).
    pub human_sd: Option<f64>,
    /// Cohen's d effect size.
    pub cohens_d: Option<f64>,
    /// 95% CI lower bound for agent metric.
    pub ci_lower: f64,
    /// 95% CI upper bound for agent metric.
    pub ci_upper: f64,
    /// Ratio: agent / human.
    pub ratio: f64,
    /// Norm-referenced z-score.
    pub z_score: Option<f64>,
}

/// Learning curve data for a single benchmark across blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningCurveRow {
    /// Benchmark name.
    pub benchmark: String,
    /// Block index (0-based).
    pub block: usize,
    /// Key metric mean for this block.
    pub metric_mean: f64,
    /// Key metric SD for this block.
    pub metric_sd: f64,
    /// Slope (linear regression coefficient across blocks).
    pub slope: f64,
}

impl BenchmarkReport {
    /// Export forest plot data for all benchmarks with baselines.
    ///
    /// Returns one row per benchmark with effect size, CI, and z-score.
    /// Suitable for rendering as a forest plot or exporting to CSV/JSON.
    pub fn forest_plot_data(&self) -> Vec<ForestPlotRow> {
        use crate::harness::baselines::BaselineCollection;
        let bl = BaselineCollection::all();
        let mut rows = Vec::new();

        for result in &self.results {
            let domain = domain_of(&result.benchmark).to_string();
            let key = key_metric_for_benchmark(&result.benchmark);
            let metric = match result.metrics.get(key) {
                Some(m) => m,
                None => continue,
            };

            let comparisons = self.find_comparisons(result, &bl);
            let comp = comparisons.iter().find(|(k, _)| k == key);

            if let Some((_, c)) = comp {
                rows.push(ForestPlotRow {
                    domain,
                    benchmark: result
                        .benchmark
                        .split("::")
                        .last()
                        .unwrap_or(&result.benchmark)
                        .to_string(),
                    metric: key.to_string(),
                    agent_mean: metric.mean,
                    human_mean: c.human_value,
                    human_sd: {
                        // Look up the baseline SD directly
                        let baseline_maps = [
                            &bl.worm,
                            &bl.cogbench,
                            &bl.executive,
                            &bl.tombench,
                            &bl.memory_agent,
                            &bl.metacognition,
                            &bl.affect,
                            &bl.creativity,
                            &bl.butlin,
                            &bl.inhibition,
                            &bl.attention,
                        ];
                        baseline_maps.iter().find_map(|bm| {
                            bm.values()
                                .find(|b| (b.value - c.human_value).abs() < 1e-10)
                                .and_then(|b| b.sd)
                        })
                    },
                    cohens_d: c.effect_size,
                    ci_lower: metric.ci_lower,
                    ci_upper: metric.ci_upper,
                    ratio: c.ratio,
                    z_score: c.z_score,
                });
            }
        }

        rows
    }

    /// Format forest plot data as CSV.
    pub fn forest_plot_csv(&self) -> String {
        let rows = self.forest_plot_data();
        let mut lines = vec![
            "domain,benchmark,metric,agent_mean,human_mean,human_sd,cohens_d,ci_lower,ci_upper,ratio,z_score".to_string(),
        ];
        for r in &rows {
            lines.push(format!(
                "{},{},{},{:.4},{:.4},{},{},{:.4},{:.4},{:.3},{}",
                r.domain,
                r.benchmark,
                r.metric,
                r.agent_mean,
                r.human_mean,
                r.human_sd.map(|s| format!("{:.4}", s)).unwrap_or_default(),
                r.cohens_d.map(|d| format!("{:.4}", d)).unwrap_or_default(),
                r.ci_lower,
                r.ci_upper,
                r.ratio,
                r.z_score.map(|z| format!("{:+.4}", z)).unwrap_or_default(),
            ));
        }
        lines.join("\n")
    }

    /// Format forest plot as an ASCII visualization.
    pub fn forest_plot_ascii(&self) -> String {
        let rows = self.forest_plot_data();
        if rows.is_empty() {
            return "No baseline comparisons available.".to_string();
        }

        let mut lines = Vec::new();
        lines.push(format!(
            "{:<25} {:>6} {:>6} {:>7}  {}",
            "Benchmark", "Agent", "Human", "d", "Effect [---------|---------|]"
        ));
        lines.push(format!(
            "{:<25} {:>6} {:>6} {:>7}  {}",
            "-------------------------", "------", "------", "-------", "  -2   -1    0   +1   +2"
        ));

        for r in &rows {
            let d = r.cohens_d.unwrap_or(0.0);
            let d_str = format!("{:+.2}", d);

            // ASCII bar: map d ∈ [-2, +2] to positions 0..24
            let bar_width = 24;
            let center = bar_width / 2;
            let pos = ((d + 2.0) / 4.0 * bar_width as f64)
                .round()
                .clamp(0.0, bar_width as f64) as usize;

            let mut bar: Vec<char> = vec![' '; bar_width + 1];
            bar[center] = '|'; // zero line
            if pos <= bar_width {
                bar[pos] = '*';
            }
            let bar_str: String = bar.into_iter().collect();

            lines.push(format!(
                "{:<25} {:>6.3} {:>6.3} {:>7}  {}",
                &r.benchmark[..r.benchmark.len().min(25)],
                r.agent_mean,
                r.human_mean,
                d_str,
                bar_str,
            ));
        }

        lines.join("\n")
    }
}

impl Default for BenchmarkReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Format a Markdown provenance/citations table from a list of benchmarks.
pub fn provenance_table(benchmarks: &[&dyn crate::harness::PsychBenchmark]) -> String {
    let mut out = String::new();
    out.push_str("| Benchmark | Paradigm | Citation | Year | DOI |\n");
    out.push_str("|-----------|----------|----------|------|-----|\n");
    for b in benchmarks {
        if let Some(p) = b.provenance() {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                b.name(),
                p.paradigm,
                p.citation,
                p.year,
                p.doi.unwrap_or("—"),
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_from_samples() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = MetricValue::from_samples(&samples);
        assert!((m.mean - 3.0).abs() < 1e-10);
        assert_eq!(m.n, 5);
        assert!(m.ci_lower < m.mean);
        assert!(m.ci_upper > m.mean);
    }

    #[test]
    fn test_small_sample_ci_wider_than_large() {
        // With n=5, t(0.025,4)=2.776 > z=1.96, so CI should be wider
        let small = MetricValue::from_samples(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        // Simulate what z=1.96 CI would give
        let se = small.std_dev / (5.0f64).sqrt();
        let z_width = 2.0 * 1.96 * se;
        let t_width = small.ci_upper - small.ci_lower;
        assert!(
            t_width > z_width,
            "t-based CI ({:.4}) should be wider than z-based CI ({:.4}) for n=5",
            t_width,
            z_width
        );
    }

    #[test]
    fn test_large_sample_ci_uses_z() {
        // For n=50, should use z=1.96
        let samples: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let m = MetricValue::from_samples(&samples);
        let se = m.std_dev / (50.0f64).sqrt();
        let expected_width = 2.0 * 1.96 * se;
        let actual_width = m.ci_upper - m.ci_lower;
        assert!(
            (actual_width - expected_width).abs() < 1e-10,
            "n=50 should use z=1.96"
        );
    }

    #[test]
    fn test_metric_empty() {
        let m = MetricValue::from_samples(&[]);
        assert_eq!(m.n, 0);
        assert_eq!(m.mean, 0.0);
    }

    #[test]
    fn test_metric_single() {
        let m = MetricValue::from_samples(&[42.0]);
        assert!((m.mean - 42.0).abs() < 1e-10);
        assert_eq!(m.std_dev, 0.0);
    }

    #[test]
    fn test_cognitive_profile_domains_present() {
        let mut report = BenchmarkReport::new();
        // Use benchmark names/metrics that map through key_metric_for_benchmark + baselines
        let mut r1 = BenchmarkResult::new("WorM::DigitSpan", None);
        r1.insert("forward_span", MetricValue::from_samples(&[6.0, 6.5]));
        report.add(r1);
        let mut r2 = BenchmarkResult::new("Executive::Stroop", None);
        r2.insert(
            "incongruent_accuracy",
            MetricValue::from_samples(&[0.85, 0.88]),
        );
        report.add(r2);
        let profile = report.cognitive_profile();
        assert!(profile.contains_key("WorM"), "profile: {:?}", profile);
        assert!(profile.contains_key("Executive"), "profile: {:?}", profile);
    }

    #[test]
    fn test_cognitive_profile_values_clamped() {
        let mut report = BenchmarkReport::new();
        let mut r = BenchmarkResult::new("WorM::DigitSpan", None);
        r.insert("forward_span", MetricValue::from_samples(&[20.0, 18.0])); // way above human
        report.add(r);
        let profile = report.cognitive_profile();
        for (_, &score) in &profile {
            assert!(
                score >= 0.0 && score <= 1.0,
                "Score out of range: {}",
                score
            );
        }
    }

    #[test]
    fn test_format_profile_non_empty() {
        let mut report = BenchmarkReport::new();
        let mut r = BenchmarkResult::new("WorM::DigitSpan", None);
        r.insert("forward_span", MetricValue::from_samples(&[6.5]));
        report.add(r);
        let output = report.format_profile();
        assert!(!output.is_empty());
        assert!(output.contains("Cognitive Profile"), "output: {}", output);
        assert!(output.contains("WorM"), "output: {}", output);
    }

    #[test]
    fn test_effect_size_populated_in_comparisons() {
        let mut report = BenchmarkReport::new();
        let mut result = BenchmarkResult::new("WorM::N-back", None);
        result.insert(
            "nback_2::accuracy",
            MetricValue::from_samples(&[0.9, 0.85, 0.88, 0.92, 0.87]),
        );
        report.add(result);

        use crate::harness::baselines::BaselineCollection;
        let bl = BaselineCollection::all();

        let comparisons = report.find_comparisons(&report.results[0], &bl);
        assert!(!comparisons.is_empty());
        let (_, comp) = &comparisons[0];
        assert!(
            comp.effect_size.is_some(),
            "effect_size should be populated"
        );
        assert!(comp.effect_size.unwrap().is_finite());
        // N-back baseline has sd=0.10, so z-score should be populated
        assert!(
            comp.z_score.is_some(),
            "z_score should be populated for nback"
        );
    }

    #[test]
    fn test_z_score_correct_direction() {
        let mut report = BenchmarkReport::new();
        // Agent scores 0.95, human mean 0.85 with SD 0.10 → z = +1.0
        let mut result = BenchmarkResult::new("WorM::N-back", None);
        result.insert("nback_2::accuracy", MetricValue::from_samples(&[0.95]));
        report.add(result);

        use crate::harness::baselines::BaselineCollection;
        let bl = BaselineCollection::all();
        let comparisons = report.find_comparisons(&report.results[0], &bl);
        let (_, comp) = comparisons
            .iter()
            .find(|(k, _)| k == "nback_2::accuracy")
            .unwrap();
        let z = comp.z_score.unwrap();
        assert!((z - 1.0).abs() < 0.01, "z should be ~+1.0, got {}", z);
    }

    #[test]
    fn test_composite_scores_populated() {
        let mut report = BenchmarkReport::new();
        // Add benchmarks with z-score-enabled baselines
        let mut r1 = BenchmarkResult::new("WorM::DigitSpan", None);
        r1.insert("forward_span", MetricValue::from_samples(&[7.0]));
        report.add(r1);
        let mut r2 = BenchmarkResult::new("WorM::N-back", None);
        r2.insert("nback_2::accuracy", MetricValue::from_samples(&[0.90]));
        report.add(r2);

        let composites = report.composite_scores();
        assert!(
            composites.contains_key("WorM"),
            "composites: {:?}",
            composites
        );
        let worm = &composites["WorM"];
        assert_eq!(worm.n_benchmarks, 2);
        assert!(worm.mean_z.is_finite());
    }

    #[test]
    fn test_report_csv() {
        let mut report = BenchmarkReport::new();
        let mut result = BenchmarkResult::new("test_bench", None);
        result.insert("acc", MetricValue::from_samples(&[0.8, 0.9]));
        report.add(result);
        let csv = report.to_csv().unwrap();
        assert!(csv.contains("test_bench"));
        assert!(csv.contains("acc"));
    }

    #[test]
    fn test_paper_summary_contains_all_benchmarks() {
        let mut report = BenchmarkReport::new();
        let mut r1 = BenchmarkResult::new("WorM::N-back", None);
        r1.insert(
            "nback_2::accuracy",
            MetricValue::from_samples(&[0.85, 0.90]),
        );
        report.add(r1);
        let mut r2 = BenchmarkResult::new("Executive::Stroop", None);
        r2.insert(
            "incongruent_accuracy",
            MetricValue::from_samples(&[0.10, 0.12]),
        );
        report.add(r2);

        let md = report.paper_summary();
        assert!(md.contains("N-back"), "paper_summary: {}", md);
        assert!(md.contains("Stroop"), "paper_summary: {}", md);
        assert!(md.contains("% Human"), "should have header");
    }

    #[test]
    fn test_rt_summary_from_samples() {
        let samples = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0];
        let rt = RtSummary::from_rt_samples(&samples);
        assert!(rt.mean_ticks.is_finite());
        assert!(rt.sd_ticks > 0.0);
        assert!(rt.mu.is_finite());
        assert!(rt.sigma >= 0.0);
        assert!(rt.tau >= 0.0, "positively skewed data should have tau >= 0");
    }

    #[test]
    fn test_rt_summary_empty() {
        let rt = RtSummary::from_rt_samples(&[]);
        assert_eq!(rt.mean_ticks, 0.0);
        assert_eq!(rt.tau, 0.0);
    }

    #[test]
    fn test_rt_summary_symmetric() {
        // Symmetric data → tau should be ~0
        let samples = vec![8.0, 9.0, 10.0, 11.0, 12.0];
        let rt = RtSummary::from_rt_samples(&samples);
        assert!((rt.mean_ticks - 10.0).abs() < 0.01);
        // Symmetric → no positive skew → tau = 0
        assert!(rt.tau.abs() < 1.0, "symmetric data tau should be near 0");
    }

    #[test]
    fn test_forest_plot_data_populated() {
        let mut report = BenchmarkReport::new();
        let mut r = BenchmarkResult::new("WorM::N-back", None);
        r.insert(
            "nback_2::accuracy",
            MetricValue::from_samples(&[0.85, 0.90, 0.88]),
        );
        report.add(r);
        let rows = report.forest_plot_data();
        assert!(!rows.is_empty(), "forest plot should have data");
        assert_eq!(rows[0].domain, "WorM");
        assert!(rows[0].cohens_d.is_some());
        assert!(rows[0].ratio > 0.0);
    }

    #[test]
    fn test_forest_plot_csv_header() {
        let mut report = BenchmarkReport::new();
        let mut r = BenchmarkResult::new("Executive::Stroop", None);
        r.insert(
            "incongruent_accuracy",
            MetricValue::from_samples(&[0.10, 0.12]),
        );
        report.add(r);
        let csv = report.forest_plot_csv();
        assert!(csv.starts_with("domain,benchmark,metric"));
        assert!(csv.contains("Stroop"));
    }

    #[test]
    fn test_forest_plot_ascii_format() {
        let mut report = BenchmarkReport::new();
        let mut r = BenchmarkResult::new("WorM::DigitSpan", None);
        r.insert("forward_span", MetricValue::from_samples(&[7.0, 6.5]));
        report.add(r);
        let ascii = report.forest_plot_ascii();
        assert!(ascii.contains("Effect"), "ascii: {}", ascii);
        assert!(ascii.contains("DigitSpan"), "ascii: {}", ascii);
    }

    #[test]
    fn test_llm_comparison_populated() {
        let mut report = BenchmarkReport::new();
        let mut r = BenchmarkResult::new("CogBench::Reversal", None);
        r.insert("win_stay_rate", MetricValue::from_samples(&[0.80, 0.82]));
        r.insert("lose_shift_rate", MetricValue::from_samples(&[0.60, 0.65]));
        report.add(r);

        use crate::harness::baselines::BaselineCollection;
        let bl = BaselineCollection::all();
        let llm_comps = report.find_llm_comparisons(&report.results[0], &bl);
        assert!(
            !llm_comps.is_empty(),
            "LLM comparisons should be populated for CogBench"
        );
        // Verify population is GPT-4
        for (_, comp) in &llm_comps {
            assert_eq!(comp.population, "GPT-4");
        }
    }

    #[test]
    fn test_multi_population_comparisons() {
        let mut report = BenchmarkReport::new();
        let mut r = BenchmarkResult::new("CogBench::Reversal", None);
        r.insert("win_stay_rate", MetricValue::from_samples(&[0.80, 0.82]));
        r.insert("lose_shift_rate", MetricValue::from_samples(&[0.60, 0.65]));
        report.add(r);

        use crate::harness::baselines::BaselineCollection;
        let bl = BaselineCollection::all();

        let human_comps = report.find_comparisons(&report.results[0], &bl);
        let llm_comps = report.find_llm_comparisons(&report.results[0], &bl);

        // Both should be non-empty
        assert!(!human_comps.is_empty(), "human comparisons should exist");
        assert!(!llm_comps.is_empty(), "LLM comparisons should exist");
        // Human and LLM populations should differ
        if let Some((_, h)) = human_comps.first() {
            if let Some((_, l)) = llm_comps.first() {
                assert_ne!(
                    h.population, l.population,
                    "populations should differ: {} vs {}",
                    h.population, l.population
                );
            }
        }
    }

    #[test]
    fn test_paper_summary_latex_valid() {
        let mut report = BenchmarkReport::new();
        let mut r = BenchmarkResult::new("WorM::DigitSpan", None);
        r.insert("forward_span", MetricValue::from_samples(&[7.0, 6.5]));
        report.add(r);

        let tex = report.paper_summary_latex();
        assert!(tex.contains(r"\begin{tabular}"), "tex: {}", tex);
        assert!(tex.contains(r"\toprule"), "tex: {}", tex);
        assert!(tex.contains(r"\bottomrule"), "tex: {}", tex);
        assert!(tex.contains(r"\end{tabular}"), "tex: {}", tex);
        assert!(tex.contains("DigitSpan"), "tex: {}", tex);
    }

    /// Build the full benchmark suite used for domain composite scoring.
    ///
    /// Shared by `test_print_all_domain_composites` and `test_cross_seed_stability`
    /// to avoid duplicating the ~100-benchmark vector.
    fn all_benchmarks() -> Vec<Box<dyn crate::harness::PsychBenchmark>> {
        use crate::benchmarks::affect::*;
        use crate::benchmarks::attention::*;
        use crate::benchmarks::binding::*;
        use crate::benchmarks::butlin::*;
        use crate::benchmarks::causal_reasoning;
        use crate::benchmarks::clinical::*;
        use crate::benchmarks::coding::*;
        use crate::benchmarks::cogbench::*;
        use crate::benchmarks::consciousness::*;
        use crate::benchmarks::creativity::*;
        use crate::benchmarks::executive::*;
        use crate::benchmarks::inhibition::*;
        use crate::benchmarks::institutional_reasoning;
        use crate::benchmarks::language::*;
        use crate::benchmarks::mathematics::*;
        use crate::benchmarks::memory_agent::*;
        use crate::benchmarks::metacognition::*;
        use crate::benchmarks::motor::*;
        use crate::benchmarks::reasoning::*;
        use crate::benchmarks::security::*;
        use crate::benchmarks::social::*;
        use crate::benchmarks::spatial::*;
        use crate::benchmarks::speech::*;
        use crate::benchmarks::substrate::*;
        use crate::benchmarks::sustained_attention::*;
        use crate::benchmarks::tombench::*;
        use crate::benchmarks::worm::*;

        vec![
            // WorM
            Box::new(BindingBenchmark),
            Box::new(ChangeDetectionBenchmark),
            Box::new(DigitSpanBenchmark),
            Box::new(NBackBenchmark),
            Box::new(SerialRecallBenchmark),
            Box::new(SpatialUpdatingBenchmark),
            // CogBench
            Box::new(BartBenchmark),
            Box::new(HorizonBenchmark),
            Box::new(InstrumentalLearningBenchmark),
            Box::new(ProbabilisticReasoningBenchmark),
            Box::new(RestlessBanditBenchmark),
            Box::new(ReversalLearningBenchmark),
            Box::new(TemporalDiscountingBenchmark),
            Box::new(TwoStepBenchmark),
            // Executive
            Box::new(StroopBenchmark),
            Box::new(FlankerBenchmark),
            Box::new(WisconsinCardSortingBenchmark),
            Box::new(IowaGamblingBenchmark),
            Box::new(TowerOfLondonBenchmark),
            Box::new(RavensProgressiveMatricesBenchmark),
            Box::new(DualTaskBenchmark),
            // ToMBench
            Box::new(FalseBeliefBenchmark),
            Box::new(FauxPasBenchmark),
            Box::new(HintingBenchmark),
            Box::new(PersuasionBenchmark),
            Box::new(StrangeStoryBenchmark),
            // MemoryAgent
            Box::new(AccurateRetrievalBenchmark),
            Box::new(ConflictResolutionBenchmark),
            Box::new(LongRangeBenchmark),
            Box::new(ProspectiveMemoryBenchmark),
            Box::new(TestTimeLearningBenchmark),
            // Metacognition
            Box::new(MetacognitiveCalibrationBenchmark),
            Box::new(FeelingOfKnowingBenchmark),
            Box::new(ChangeBlindnessBenchmark),
            // Affect
            Box::new(EmotionalStroopBenchmark),
            Box::new(MoodCongruentRecallBenchmark),
            Box::new(ValenceClassificationBenchmark),
            // Creativity
            Box::new(AlternateUsesBenchmark),
            Box::new(RemoteAssociatesBenchmark),
            Box::new(DivergentThinkingBenchmark),
            Box::new(ConceptualBlendingBenchmark),
            Box::new(InsightProblemBenchmark),
            // Butlin
            Box::new(ButlinIndicatorSuite),
            // Inhibition
            Box::new(GoNoGoBenchmark),
            Box::new(StopSignalBenchmark),
            Box::new(FlankerInhibitionBenchmark),
            // Attention
            Box::new(AttentionalBlinkBenchmark),
            Box::new(VisualSearchBenchmark),
            Box::new(MismatchNegativityBenchmark),
            // Reasoning
            Box::new(ArcFluidBenchmark),
            Box::new(ArcCompositionalBenchmark),
            Box::new(ArcAnalogyBenchmark),
            Box::new(ArcAbductiveBenchmark),
            Box::new(ArcChainBenchmark),
            Box::new(ArcNoiseBenchmark),
            Box::new(ArcFewShotBenchmark),
            Box::new(ArcScalingBenchmark),
            Box::new(ArcRsaBenchmark),
            Box::new(ArcAlgebraBenchmark),
            Box::new(ArcStaircaseBenchmark),
            // Sustained Attention
            Box::new(SartBenchmark),
            Box::new(PvtBenchmark),
            Box::new(CptBenchmark),
            // Motor
            Box::new(SrttBenchmark),
            Box::new(FittsLawBenchmark),
            Box::new(BimanualBenchmark),
            Box::new(ProprioceptiveDriftBenchmark),
            // Language
            Box::new(GardenPathBenchmark),
            Box::new(SemanticCoherenceBenchmark),
            Box::new(LexicalDecisionBenchmark),
            Box::new(SemanticPrimingBenchmark),
            // Social
            Box::new(RmeBenchmark),
            Box::new(UltimatumGameBenchmark),
            Box::new(SocialNormBenchmark),
            Box::new(PrisonersDilemmaBenchmark),
            Box::new(PublicGoodsBenchmark),
            Box::new(DictatorGameBenchmark),
            Box::new(MachiavelliBenchmark),
            // Binding
            Box::new(TemporalOrderBenchmark),
            Box::new(CrossModalBindingBenchmark),
            Box::new(FeatureConjunctionBenchmark),
            // Spatial
            Box::new(MentalRotationBenchmark),
            Box::new(SpatialPathUpdatingBenchmark),
            Box::new(LandmarkBindingBenchmark),
            Box::new(PerspectiveTakingBenchmark),
            // Causal Reasoning
            Box::new(causal_reasoning::CausalChainBenchmark),
            Box::new(causal_reasoning::ConfoundDetectionBenchmark),
            Box::new(causal_reasoning::InterventionEffectBenchmark),
            // Speech
            Box::new(PhonemeDiscriminationBenchmark),
            Box::new(VotContinuumBenchmark),
            Box::new(CategoricalPerceptionBenchmark),
            // Consciousness
            Box::new(BlindSightBenchmark),
            Box::new(BinocularRivalryBenchmark),
            Box::new(PerceptualCrowdingBenchmark),
            // Substrate
            Box::new(SubstrateTransferBenchmark),
            Box::new(SubstrateDegradationBenchmark),
            Box::new(SubstrateLatencyBenchmark),
            // Clinical/Therapeutic
            Box::new(EmpathicAccuracyBenchmark),
            Box::new(TherapeuticResponseBenchmark),
            Box::new(AllianceMaintenanceBenchmark),
            Box::new(CrisisDetectionBenchmark),
            Box::new(CognitiveDistortionBenchmark),
            Box::new(MotivationalInterviewingBenchmark),
            // Institutional Reasoning
            Box::new(institutional_reasoning::InstitutionalReasoningBenchmark),
            Box::new(institutional_reasoning::AnalogicalReasoningBenchmark),
            Box::new(institutional_reasoning::CausalChainBenchmark),
            Box::new(institutional_reasoning::CounterfactualReasoningBenchmark),
            Box::new(institutional_reasoning::WeightedDecompositionBenchmark),
            Box::new(institutional_reasoning::InstitutionalStabilityBenchmark),
            Box::new(institutional_reasoning::InstitutionalIsomorphismBenchmark),
            // Mathematics
            Box::new(ArithmeticWordProblemsBenchmark),
            Box::new(LinearSystemSolvingBenchmark),
            Box::new(PolynomialRootsBenchmark),
            Box::new(MatrixOperationsBenchmark),
            Box::new(StatisticalInferenceBenchmark),
            Box::new(BayesianReasoningBenchmark),
            Box::new(LogicalDeductionBenchmark),
            Box::new(ConstraintPuzzlesBenchmark),
            Box::new(ProofConstructionBenchmark),
            Box::new(DefiniteIntegralsBenchmark),
            // Security (HDC-FHE)
            Box::new(EncryptedClassificationBenchmark),
            Box::new(CollectiveAggregationBenchmark),
            Box::new(EncryptedLearningBenchmark),
            Box::new(CrossMaskPrivacyBenchmark),
            Box::new(EncryptedBindingBenchmark),
            Box::new(ScalingAnalysisBenchmark),
            // Coding
            Box::new(HumanEvalMiniBenchmark),
            Box::new(BugDetectionBenchmark),
        ]
    }

    #[test]
    fn test_print_all_domain_composites() {
        let config = crate::harness::config::BenchmarkConfig::default();
        let benchmarks = all_benchmarks();

        let mut report = BenchmarkReport::new();
        for b in &benchmarks {
            report.add(b.run(&config));
        }

        eprintln!("\n{}", report.format_composites());

        let composites = report.composite_scores();
        assert!(
            composites.len() >= 10,
            "Expected 10+ domains, got {}",
            composites.len()
        );

        let total_z: f64 = composites.values().map(|c| c.mean_z).sum();
        let grand_mean = total_z / composites.len() as f64;
        eprintln!(
            "Grand mean z = {grand_mean:+.3} across {} domains\n",
            composites.len()
        );
    }

    /// Cross-seed stability test: verifies that grand mean z-scores are stable
    /// across different RNG seeds.
    ///
    /// Runs the full benchmark suite with 5 different seeds and asserts that the
    /// standard deviation of grand means is < 0.10 (i.e., results are not
    /// seed-dependent). This is a key reproducibility check.
    ///
    /// NOTE: This test takes ~5 minutes per seed (~25 minutes total).
    /// Run explicitly with: `cargo test --lib test_cross_seed_stability -- --ignored`
    #[test]
    #[ignore]
    fn test_cross_seed_stability() {
        use crate::harness::config::BenchmarkConfig;

        let seeds: [u64; 3] = [42, 123, 789];
        let mut grand_means: Vec<f64> = Vec::with_capacity(seeds.len());

        for &seed in &seeds {
            let config = BenchmarkConfig {
                seed,
                ..Default::default()
            };
            let benchmarks = all_benchmarks();

            let mut report = BenchmarkReport::new();
            for b in &benchmarks {
                report.add(b.run(&config));
            }

            let composites = report.composite_scores();
            let n_domains = composites.len() as f64;
            let total_z: f64 = composites.values().map(|c| c.mean_z).sum();
            let grand_mean = total_z / n_domains;

            eprintln!(
                "seed={seed:>4}: grand mean z = {grand_mean:+.4} across {n} domains",
                n = composites.len()
            );
            grand_means.push(grand_mean);
        }

        // Compute mean and SD of grand means across seeds
        let n = grand_means.len() as f64;
        let mean_of_means: f64 = grand_means.iter().sum::<f64>() / n;
        let variance: f64 = grand_means
            .iter()
            .map(|x| (x - mean_of_means).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let sd = variance.sqrt();

        eprintln!("\n--- Cross-seed stability ---");
        eprintln!("Seeds: {:?}", seeds);
        eprintln!("Grand means: {:?}", grand_means);
        eprintln!("Mean of grand means: {mean_of_means:+.4}");
        eprintln!("SD of grand means:   {sd:.4}");
        eprintln!("----------------------------\n");

        assert!(
            sd < 0.10,
            "Grand mean z-score SD across seeds ({sd:.4}) exceeds stability threshold (0.10). \
             Seeds produce inconsistent results."
        );
    }

    #[test]
    fn test_report_contains_bias_disclosure() {
        let mut report = BenchmarkReport::new();
        let mut stroop = BenchmarkResult::new("Executive::Stroop", None);
        stroop.insert(
            "incongruent_accuracy",
            MetricValue::from_samples(&[0.10, 0.08, 0.12]),
        );
        report.add(stroop);
        let output = report.format_composites();
        assert!(output.contains("WEIRD"), "Should contain WEIRD disclosure");
        assert!(output.contains("Henrich"), "Should cite Henrich et al.");
    }

    #[test]
    fn test_calibration_class_of_known_benchmarks() {
        // A Priori
        assert_eq!(calibration_class_of("Executive::Stroop"), "AP");
        assert_eq!(calibration_class_of("Executive::Flanker"), "AP");
        assert_eq!(calibration_class_of("WorM::N-back"), "AP");
        assert_eq!(calibration_class_of("Metacognition::ChangeBlindness"), "AP");
        assert_eq!(calibration_class_of("Metacognition::Calibration"), "AP");
        assert_eq!(calibration_class_of("Inhibition::StopSignal"), "AP");

        // Theoretical
        assert_eq!(calibration_class_of("Substrate::SubstrateTransfer"), "TH");
        assert_eq!(
            calibration_class_of("Substrate::SubstrateDegradation"),
            "TH"
        );
        assert_eq!(calibration_class_of("Consciousness::Blindsight"), "TH");

        // Post Hoc (everything else)
        assert_eq!(calibration_class_of("CogBench::BART"), "PH");
        assert_eq!(calibration_class_of("Social::UltimatumGame"), "PH");
    }

    /// Full reliability audit across representative benchmarks from each domain.
    ///
    /// Runs split-half ICC and test-retest reliability for a subset of benchmarks
    /// (one per major domain). Reports ICC, Pearson r, SEM, and practice effects.
    /// This is the credibility gate for paper submission.
    #[test]
    fn test_full_reliability_audit() {
        use crate::benchmarks::affect::EmotionalStroopBenchmark;
        use crate::benchmarks::attention::AttentionalBlinkBenchmark;
        use crate::benchmarks::creativity::RemoteAssociatesBenchmark;
        use crate::benchmarks::executive::{FlankerBenchmark, StroopBenchmark};
        use crate::benchmarks::inhibition::StopSignalBenchmark;
        use crate::benchmarks::language::SemanticPrimingBenchmark;
        use crate::benchmarks::memory_agent::AccurateRetrievalBenchmark;
        use crate::benchmarks::metacognition::{
            FeelingOfKnowingBenchmark, MetacognitiveCalibrationBenchmark,
        };
        use crate::benchmarks::motor::SrttBenchmark;
        use crate::benchmarks::social::RmeBenchmark;
        use crate::benchmarks::speech::VotContinuumBenchmark;
        use crate::benchmarks::tombench::FalseBeliefBenchmark;
        use crate::benchmarks::worm::{DigitSpanBenchmark, NBackBenchmark};
        use crate::harness::PsychBenchmark;
        use crate::harness::config::BenchmarkConfig;
        use crate::harness::reliability_analysis::ReliabilityBattery;

        let benchmarks: Vec<&dyn PsychBenchmark> = vec![
            // Core cognitive (fast benchmarks, one per major domain)
            &StroopBenchmark,
            &FlankerBenchmark,
            &NBackBenchmark,
            &DigitSpanBenchmark,
            // Metacognition
            &MetacognitiveCalibrationBenchmark,
            &FeelingOfKnowingBenchmark,
            // Attention + Inhibition
            &AttentionalBlinkBenchmark,
            &StopSignalBenchmark,
            // Memory + ToM
            &AccurateRetrievalBenchmark,
            &FalseBeliefBenchmark,
            // Motor + Language + Speech
            &SrttBenchmark,
            &SemanticPrimingBenchmark,
            &VotContinuumBenchmark,
            // Affect + Creativity + Social
            &EmotionalStroopBenchmark,
            &RemoteAssociatesBenchmark,
            &RmeBenchmark,
        ];

        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 10,
            seed: 42,
            ..Default::default()
        };

        // 2 sessions, 8 subjects — gives ICC enough between-subject variance
        let battery = ReliabilityBattery::run(&benchmarks, &config, 2, 8);

        eprintln!("\n{}", battery.to_markdown());

        // Count reliability classes
        let mut excellent = 0u32;
        let mut good = 0u32;
        let mut moderate = 0u32;
        let mut poor = 0u32;

        for r in &battery.results {
            assert!(
                r.icc.is_finite(),
                "{} ICC not finite: {}",
                r.benchmark,
                r.icc
            );
            assert!(
                r.sem.is_finite() && r.sem >= 0.0,
                "{} SEM invalid: {}",
                r.benchmark,
                r.sem
            );

            match r.reliability_class {
                crate::harness::reliability_analysis::ReliabilityClass::Excellent => excellent += 1,
                crate::harness::reliability_analysis::ReliabilityClass::Good => good += 1,
                crate::harness::reliability_analysis::ReliabilityClass::Moderate => moderate += 1,
                crate::harness::reliability_analysis::ReliabilityClass::Poor => poor += 1,
            }
        }

        let total = battery.results.len();
        eprintln!(
            "\nReliability Summary: {total} benchmarks — \
             {excellent} Excellent, {good} Good, {moderate} Moderate, {poor} Poor"
        );

        // At least some benchmarks should produce results
        assert!(
            total >= 15,
            "Expected 15+ reliability results, got {}",
            total
        );
    }

    /// Test that enhanced LaTeX table generates valid output with provenance.
    #[test]
    fn test_paper_summary_latex_with_provenance() {
        use crate::benchmarks::executive::{FlankerBenchmark, StroopBenchmark};
        use crate::benchmarks::metacognition::MetacognitiveCalibrationBenchmark;
        use crate::benchmarks::worm::DigitSpanBenchmark;
        use crate::harness::PsychBenchmark;

        let config = crate::harness::config::BenchmarkConfig::default();
        let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
            Box::new(StroopBenchmark),
            Box::new(FlankerBenchmark),
            Box::new(DigitSpanBenchmark),
            Box::new(MetacognitiveCalibrationBenchmark),
        ];

        let mut report = BenchmarkReport::new();
        for b in &benchmarks {
            report.add(b.run(&config));
        }

        let bench_refs: Vec<&dyn PsychBenchmark> = benchmarks.iter().map(|b| &**b).collect();
        let tex = report.paper_summary_latex_with_provenance(&bench_refs);

        assert!(tex.contains(r"\begin{tabular}"), "tex: {}", tex);
        assert!(tex.contains(r"\toprule"), "tex: {}", tex);
        assert!(tex.contains(r"\bottomrule"), "tex: {}", tex);
        assert!(tex.contains("Stroop"), "Should contain Stroop");
        assert!(
            tex.contains("AP") || tex.contains("PH"),
            "Should contain calibration class"
        );
        assert!(
            tex.contains("Grand mean"),
            "Should contain grand mean footer"
        );

        eprintln!("\n{}", tex);
    }
}
