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
        use crate::harness::baselines;

        let worm_bl = baselines::worm_baselines();
        let cog_bl = baselines::cogbench_baselines();
        let tom_bl = baselines::tombench_baselines();
        let mem_bl = baselines::memory_agent_baselines();
        let exec_bl = baselines::executive_baselines();
        let meta_bl = baselines::metacognition_baselines();
        let affect_bl = baselines::affect_baselines();
        let creativity_bl = baselines::creativity_baselines();
        let butlin_bl = baselines::butlin_baselines();

        let mut lines = vec![format!("Psych Benchmark Report ({})", self.timestamp)];
        lines.push(format!("{} benchmarks run", self.results.len()));
        lines.push(String::new());
        for result in &self.results {
            lines.push(result.summary());

            // Add baseline comparisons for known metrics
            let comparisons = self.find_comparisons(result, &worm_bl, &cog_bl, &tom_bl, &mem_bl, &exec_bl, &meta_bl, &affect_bl, &creativity_bl, &butlin_bl);
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
            Some(super::analysis::cohens_d(metric.mean, baseline_value, metric.std_dev))
        }
    }

    /// Find applicable baseline comparisons for a benchmark result.
    fn find_comparisons(
        &self,
        result: &BenchmarkResult,
        worm_bl: &std::collections::BTreeMap<&str, super::baselines::Baseline>,
        cog_bl: &std::collections::BTreeMap<&str, super::baselines::Baseline>,
        tom_bl: &std::collections::BTreeMap<&str, super::baselines::Baseline>,
        mem_bl: &std::collections::BTreeMap<&str, super::baselines::Baseline>,
        exec_bl: &std::collections::BTreeMap<&str, super::baselines::Baseline>,
        meta_bl: &std::collections::BTreeMap<&str, super::baselines::Baseline>,
        affect_bl: &std::collections::BTreeMap<&str, super::baselines::Baseline>,
        creativity_bl: &std::collections::BTreeMap<&str, super::baselines::Baseline>,
        butlin_bl: &std::collections::BTreeMap<&str, super::baselines::Baseline>,
    ) -> Vec<(String, BaselineComparison)> {
        let mut comps = Vec::new();
        let benchmark = result.benchmark.as_str();

        // Map benchmark metrics to baselines
        let mappings: Vec<(&str, &str, &std::collections::BTreeMap<&str, super::baselines::Baseline>)> = vec![
            ("nback_2::accuracy", "nback_2_accuracy", worm_bl),
            ("nback_3::accuracy", "nback_3_accuracy", worm_bl),
            ("set_size_4::accuracy", "change_detection_k4", worm_bl),
            ("average_pumps", "bart_avg_pumps", cog_bl),
            ("horizon_6::directed_exploration", "directed_exploration", cog_bl),
            ("beta3_model_basedness", "model_basedness", cog_bl),
            ("discounting_score_S", "discounting_score", cog_bl),
            ("false_belief_accuracy", "false_belief_accuracy", tom_bl),
            ("faux_pas_accuracy", "faux_pas_accuracy", tom_bl),
            ("hinting_accuracy", "hinting_accuracy", tom_bl),
            ("overall_retrieval_accuracy", "accurate_retrieval", mem_bl),
            ("correction_accuracy", "test_time_learning", mem_bl),
            // Executive (WCST-specific perseverative_errors handled below)
            ("categories_completed", "wcst_categories_completed", exec_bl),
            ("trials_to_first_category", "wcst_trials_to_first", exec_bl),
            ("overall_net_score", "igt_overall_net_score", exec_bl),
            ("deck_preference_good", "igt_deck_preference_good", exec_bl),
            // Note: overall_accuracy, easy_accuracy, congruent_accuracy,
            // incongruent_accuracy are benchmark-specific; matched below with
            // benchmark-name guards to avoid cross-benchmark collisions.
            // Stroop-specific
            ("stroop_effect", "stroop_effect", exec_bl),
            // Flanker-specific
            ("flanker_effect", "flanker_effect", exec_bl),
            // Tower of London
            ("overall_optimal_rate", "tol_overall_optimal_rate", exec_bl),
            ("planning_efficiency", "tol_planning_efficiency", exec_bl),
            // Digit Span
            ("forward_span", "digit_span_forward", worm_bl),
            ("backward_span", "digit_span_backward", worm_bl),
            // Reversal Learning
            ("win_stay_rate", "reversal_win_stay", cog_bl),
            ("lose_shift_rate", "reversal_lose_shift", cog_bl),
            // Metacognition
            ("calibration_error_ece", "calibration_error_ece", meta_bl),
            ("discrimination_gamma", "discrimination_gamma", meta_bl),
            // ToMBench: Persuasion
            ("persuasion_detection", "persuasion_detection", tom_bl),
            // MemoryAgent: Long-range
            ("delay_50::retention", "long_range_delay_50", mem_bl),
            // MemoryAgent: Conflict resolution
            ("recency_preference", "conflict_recency_preference", mem_bl),
            // Affect: Valence classification
            ("valence_accuracy", "valence_accuracy", affect_bl),
            // Affect: Mood-congruent recall
            ("congruence_ratio", "congruence_ratio", affect_bl),
            // Creativity: Alternate Uses fluency
            ("fluency", "aut_fluency", creativity_bl),
            // Butlin: Consciousness indicator count
            ("present_count", "present_count", butlin_bl),
            ("presence_ratio", "presence_ratio", butlin_bl),
        ];

        for (metric_key, baseline_key, baselines) in mappings {
            if let Some(metric) = result.metrics.get(metric_key) {
                if let Some(bl) = baselines.get(baseline_key) {
                    let ratio = if bl.value.abs() > 1e-10 {
                        metric.mean / bl.value
                    } else {
                        0.0
                    };
                    comps.push((
                        metric_key.to_string(),
                        BaselineComparison {
                            human_value: bl.value,
                            source: bl.source.to_string(),
                            population: bl.population.to_string(),
                            ratio,
                            effect_size: Self::compute_effect_size(metric, bl.value),
                        },
                    ));
                }
            }
        }

        // Helper closure for benchmark-specific comparisons
        let mut push_specific = |metric_key: &str, baseline_key: &str,
                                  baselines: &std::collections::BTreeMap<&str, super::baselines::Baseline>| {
            if let Some(metric) = result.metrics.get(metric_key) {
                if let Some(bl) = baselines.get(baseline_key) {
                    let ratio = if bl.value.abs() > 1e-10 { metric.mean / bl.value } else { 0.0 };
                    comps.push((metric_key.to_string(), BaselineComparison {
                        human_value: bl.value, source: bl.source.to_string(),
                        population: bl.population.to_string(), ratio,
                        effect_size: Self::compute_effect_size(metric, bl.value),
                    }));
                }
            }
        };

        // Benchmark-specific metrics (avoid cross-benchmark collisions)
        if benchmark.contains("Ravens") {
            push_specific("overall_accuracy", "ravens_overall_accuracy", exec_bl);
            push_specific("easy_accuracy", "ravens_easy_accuracy", exec_bl);
        }
        if benchmark.contains("RemoteAssociates") {
            push_specific("overall_accuracy", "rat_overall_accuracy", creativity_bl);
        }
        if benchmark.contains("StrangeStory") {
            push_specific("overall_accuracy", "strange_story_accuracy", tom_bl);
        }
        if benchmark.contains("Stroop") {
            push_specific("congruent_accuracy", "stroop_congruent_accuracy", exec_bl);
            push_specific("incongruent_accuracy", "stroop_incongruent_accuracy", exec_bl);
        }
        if benchmark.contains("Flanker") {
            push_specific("congruent_accuracy", "flanker_congruent_accuracy", exec_bl);
            push_specific("incongruent_accuracy", "flanker_incongruent_accuracy", exec_bl);
        }
        if benchmark.contains("Wisconsin") {
            push_specific("perseverative_errors", "wcst_perseverative_errors", exec_bl);
        }
        if benchmark.contains("Reversal") {
            push_specific("perseverative_errors", "reversal_perseverative_errors", cog_bl);
        }
        if benchmark.contains("RestlessBandit") {
            push_specific("overall_accuracy", "restless_bandit_accuracy", cog_bl);
        }
        if benchmark.contains("Instrumental") {
            push_specific("contingency_sensitivity", "instrumental_sensitivity", cog_bl);
        }
        if benchmark.contains("SpatialUpdating") {
            push_specific("overall_accuracy", "spatial_updating_accuracy", worm_bl);
        }
        if benchmark.contains("Binding") {
            push_specific("overall_binding_accuracy", "binding_accuracy", worm_bl);
        }
        if benchmark.contains("SerialRecall") {
            push_specific("list_7::primacy_index", "serial_primacy_advantage", worm_bl);
        }
        if benchmark.contains("Probabilistic") {
            push_specific("beta2_likelihood_weight", "probabilistic_likelihood_weight", cog_bl);
        }

        // Only return comparisons relevant to this benchmark
        if benchmark.contains("WorM") || benchmark.contains("CogBench")
            || benchmark.contains("ToM") || benchmark.contains("Memory")
            || benchmark.contains("Executive") || benchmark.contains("Metacognition")
            || benchmark.contains("Affect") || benchmark.contains("Creativity")
            || benchmark.contains("Butlin")
        {
            comps
        } else {
            Vec::new()
        }
    }

    /// Export all results as CSV.
    pub fn to_csv(&self) -> Result<String, csv::Error> {
        let mut wtr = csv::Writer::from_writer(Vec::new());
        wtr.write_record(["benchmark", "config", "metric", "mean", "std_dev", "n", "ci_lower", "ci_upper"])?;
        for result in &self.results {
            let config = result.config_label.as_deref().unwrap_or("");
            for (key, val) in &result.metrics {
                wtr.write_record(&[
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
        Ok(String::from_utf8(wtr.into_inner().map_err(|e| csv::Error::from(e.into_error()))?)
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
    /// One row per benchmark, showing key metric, agent value, human baseline, % of human, and 95% CI.
    pub fn paper_summary(&self) -> String {
        use crate::harness::baselines;

        let worm_bl = baselines::worm_baselines();
        let cog_bl = baselines::cogbench_baselines();
        let tom_bl = baselines::tombench_baselines();
        let mem_bl = baselines::memory_agent_baselines();
        let exec_bl = baselines::executive_baselines();
        let meta_bl = baselines::metacognition_baselines();
        let affect_bl = baselines::affect_baselines();
        let creativity_bl = baselines::creativity_baselines();
        let butlin_bl = baselines::butlin_baselines();

        let mut lines = Vec::new();
        lines.push("| Domain | Benchmark | Key Metric | Agent | Human | % of Human | d | 95% CI |".to_string());
        lines.push("|--------|-----------|------------|-------|-------|------------|---|--------|".to_string());

        for result in &self.results {
            let domain = domain_of(&result.benchmark);
            let key = key_metric_for_benchmark(&result.benchmark);
            let metric = match result.metrics.get(key) {
                Some(m) => m,
                None => continue,
            };

            let comparisons = self.find_comparisons(
                result, &worm_bl, &cog_bl, &tom_bl, &mem_bl, &exec_bl, &meta_bl,
                &affect_bl, &creativity_bl, &butlin_bl,
            );

            let comp = comparisons.iter().find(|(k, _)| k == key);
            let (human_str, pct_str) = comp
                .map(|(_, c)| (format!("{:.3}", c.human_value), format!("{:.1}%", c.ratio * 100.0)))
                .unwrap_or_else(|| ("—".to_string(), "—".to_string()));

            let d_str = comp
                .and_then(|(_, c)| c.effect_size)
                .map(|d| format!("{:.2}", d))
                .unwrap_or_else(|| "—".to_string());

            let ci_str = format!("[{:.3}, {:.3}]", metric.ci_lower, metric.ci_upper);

            lines.push(format!(
                "| {} | {} | {} | {:.3} | {} | {} | {} | {} |",
                domain,
                result.benchmark.split("::").last().unwrap_or(&result.benchmark),
                key,
                metric.mean,
                human_str,
                pct_str,
                d_str,
                ci_str,
            ));
        }

        lines.join("\n")
    }

    /// Publication-ready LaTeX tabular output.
    pub fn paper_summary_latex(&self) -> String {
        use crate::harness::baselines;

        let worm_bl = baselines::worm_baselines();
        let cog_bl = baselines::cogbench_baselines();
        let tom_bl = baselines::tombench_baselines();
        let mem_bl = baselines::memory_agent_baselines();
        let exec_bl = baselines::executive_baselines();
        let meta_bl = baselines::metacognition_baselines();
        let affect_bl = baselines::affect_baselines();
        let creativity_bl = baselines::creativity_baselines();
        let butlin_bl = baselines::butlin_baselines();

        let mut lines = Vec::new();
        lines.push(r"\begin{tabular}{llllrrrl}".to_string());
        lines.push(r"\toprule".to_string());
        lines.push(r"Domain & Benchmark & Key Metric & Agent & Human & \% Human & $d$ & 95\% CI \\".to_string());
        lines.push(r"\midrule".to_string());

        for result in &self.results {
            let domain = domain_of(&result.benchmark);
            let key = key_metric_for_benchmark(&result.benchmark);
            let metric = match result.metrics.get(key) {
                Some(m) => m,
                None => continue,
            };

            let comparisons = self.find_comparisons(
                result, &worm_bl, &cog_bl, &tom_bl, &mem_bl, &exec_bl, &meta_bl,
                &affect_bl, &creativity_bl, &butlin_bl,
            );

            let comp = comparisons.iter().find(|(k, _)| k == key);
            let (human_str, pct_str) = comp
                .map(|(_, c)| (format!("{:.3}", c.human_value), format!("{:.1}\\%", c.ratio * 100.0)))
                .unwrap_or_else(|| ("---".to_string(), "---".to_string()));

            let d_str = comp
                .and_then(|(_, c)| c.effect_size)
                .map(|d| format!("{:.2}", d))
                .unwrap_or_else(|| "---".to_string());

            let bench_name = result.benchmark.split("::").last().unwrap_or(&result.benchmark);

            lines.push(format!(
                "{} & {} & {} & {:.3} & {} & {} & {} & [{:.3}, {:.3}] \\\\",
                domain, bench_name, key, metric.mean, human_str, pct_str, d_str,
                metric.ci_lower, metric.ci_upper,
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
        use crate::harness::baselines;

        let worm_bl = baselines::worm_baselines();
        let cog_bl = baselines::cogbench_baselines();
        let tom_bl = baselines::tombench_baselines();
        let mem_bl = baselines::memory_agent_baselines();
        let exec_bl = baselines::executive_baselines();
        let meta_bl = baselines::metacognition_baselines();
        let affect_bl = baselines::affect_baselines();
        let creativity_bl = baselines::creativity_baselines();
        let butlin_bl = baselines::butlin_baselines();

        let mut domain_scores: BTreeMap<String, Vec<f64>> = BTreeMap::new();

        for result in &self.results {
            let domain = domain_of(&result.benchmark).to_string();
            let key = key_metric_for_benchmark(&result.benchmark);

            let comparisons = self.find_comparisons(
                result, &worm_bl, &cog_bl, &tom_bl, &mem_bl,
                &exec_bl, &meta_bl, &affect_bl, &creativity_bl, &butlin_bl,
            );

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
                domain_scores.entry(domain).or_default().push(score.max(0.0));
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
            let bar = format!(
                "{}{}",
                "\u{2588}".repeat(filled),
                "\u{2591}".repeat(empty),
            );
            lines.push(format!("{:<14}{} {:.2}", domain, bar, score));
        }

        lines.join("\n")
    }
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
            t_width, z_width
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
            assert!(score >= 0.0 && score <= 1.0, "Score out of range: {}", score);
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

        use crate::harness::baselines;
        let worm_bl = baselines::worm_baselines();
        let cog_bl = baselines::cogbench_baselines();
        let tom_bl = baselines::tombench_baselines();
        let mem_bl = baselines::memory_agent_baselines();
        let exec_bl = baselines::executive_baselines();
        let meta_bl = baselines::metacognition_baselines();
        let affect_bl = baselines::affect_baselines();
        let creativity_bl = baselines::creativity_baselines();
        let butlin_bl = baselines::butlin_baselines();

        let comparisons = report.find_comparisons(
            &report.results[0], &worm_bl, &cog_bl, &tom_bl, &mem_bl,
            &exec_bl, &meta_bl, &affect_bl, &creativity_bl, &butlin_bl,
        );
        assert!(!comparisons.is_empty());
        let (_, comp) = &comparisons[0];
        assert!(comp.effect_size.is_some(), "effect_size should be populated");
        assert!(comp.effect_size.unwrap().is_finite());
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
