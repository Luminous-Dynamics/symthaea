//! Benchmark reporting: JSON/CSV output and summary formatting.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

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
    if df <= 29 {
        T_TABLE[df - 1]
    } else {
        1.96
    }
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
    fn find_comparisons(
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
            ("presence_ratio", "presence_ratio", &bl.butlin),
        ];

        for (metric_key, baseline_key, baselines) in &mappings {
            if let Some(metric) = result.metrics.get(*metric_key) {
                if let Some(baseline) = baselines.get(baseline_key) {
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
        }
        if benchmark.contains("StrangeStory") {
            push_specific("overall_accuracy", "strange_story_accuracy", &bl.tombench);
        }
        if benchmark.contains("Stroop") {
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
        }
        if benchmark.contains("Flanker") {
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
        }
        if benchmark.contains("Wisconsin") {
            push_specific(
                "perseverative_errors",
                "wcst_perseverative_errors",
                &bl.executive,
            );
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
        if benchmark.contains("Binding") {
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
        {
            comps
        } else {
            Vec::new()
        }
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
    /// % of human, effect size d, z-score, and 95% CI.
    pub fn paper_summary(&self) -> String {
        use crate::harness::baselines::BaselineCollection;
        let bl = BaselineCollection::all();

        let mut lines = Vec::new();
        lines.push(
            "| Domain | Benchmark | Key Metric | Agent | Human | % Human | d | z | 95% CI |"
                .to_string(),
        );
        lines.push(
            "|--------|-----------|------------|-------|-------|---------|---|---|--------|"
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

            lines.push(format!(
                "| {} | {} | {} | {:.3} | {} | {} | {} | {} | {} |",
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
        b if b.contains("Binding") => "overall_binding_accuracy",
        b if b.contains("DigitSpan") => "forward_span",
        b if b.contains("Stroop") && !b.contains("Strange") => "stroop_effect",
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
        b if b.contains("Butlin") => "present_count",
        b if b.contains("ValenceClassification") => "valence_accuracy",
        b if b.contains("MoodCongruent") => "congruence_ratio",
        b if b.contains("RemoteAssociates") => "overall_accuracy",
        b if b.contains("AlternateUses") => "fluency",
        _ => "overall_accuracy",
    }
}

/// Returns true if a lower metric value is better (inverted scoring).
fn is_lower_better(metric_key: &str) -> bool {
    matches!(
        metric_key,
        "calibration_error_ece"
            | "perseverative_errors"
            | "trials_to_first_category"
            | "restless_bandit_regret"
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

impl Default for BenchmarkReport {
    fn default() -> Self {
        Self::new()
    }
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
        r2.insert("stroop_effect", MetricValue::from_samples(&[0.08, 0.09]));
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
}
