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
        // 95% CI using t-approximation (z=1.96 for large n)
        let se = std_dev / (n as f64).sqrt();
        let z = 1.96;
        Self {
            mean,
            std_dev,
            n,
            ci_lower: mean - z * se,
            ci_upper: mean + z * se,
        }
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
        use super::baselines;

        let worm_bl = baselines::worm_baselines();
        let cog_bl = baselines::cogbench_baselines();
        let tom_bl = baselines::tombench_baselines();
        let mem_bl = baselines::memory_agent_baselines();
        let exec_bl = baselines::executive_baselines();
        let meta_bl = baselines::metacognition_baselines();

        let mut lines = vec![format!("Psych Benchmark Report ({})", self.timestamp)];
        lines.push(format!("{} benchmarks run", self.results.len()));
        lines.push(String::new());
        for result in &self.results {
            lines.push(result.summary());

            // Add baseline comparisons for known metrics
            let comparisons = self.find_comparisons(result, &worm_bl, &cog_bl, &tom_bl, &mem_bl, &exec_bl, &meta_bl);
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
            ("retrieval_accuracy", "accurate_retrieval", mem_bl),
            ("correction_accuracy", "test_time_learning", mem_bl),
            // Executive
            ("categories_completed", "wcst_categories_completed", exec_bl),
            ("perseverative_errors", "wcst_perseverative_errors", exec_bl),
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
                        },
                    ));
                }
            }
        }

        // Benchmark-specific metrics (avoid cross-benchmark collisions)
        if benchmark.contains("Ravens") {
            if let Some(metric) = result.metrics.get("overall_accuracy") {
                if let Some(bl) = exec_bl.get("ravens_overall_accuracy") {
                    let ratio = if bl.value.abs() > 1e-10 { metric.mean / bl.value } else { 0.0 };
                    comps.push(("overall_accuracy".to_string(), BaselineComparison {
                        human_value: bl.value, source: bl.source.to_string(),
                        population: bl.population.to_string(), ratio,
                    }));
                }
            }
            if let Some(metric) = result.metrics.get("easy_accuracy") {
                if let Some(bl) = exec_bl.get("ravens_easy_accuracy") {
                    let ratio = if bl.value.abs() > 1e-10 { metric.mean / bl.value } else { 0.0 };
                    comps.push(("easy_accuracy".to_string(), BaselineComparison {
                        human_value: bl.value, source: bl.source.to_string(),
                        population: bl.population.to_string(), ratio,
                    }));
                }
            }
        }

        // Stroop-specific: congruent_accuracy, incongruent_accuracy
        if benchmark.contains("Stroop") {
            for (metric_key, baseline_key) in [
                ("congruent_accuracy", "stroop_congruent_accuracy"),
                ("incongruent_accuracy", "stroop_incongruent_accuracy"),
            ] {
                if let Some(metric) = result.metrics.get(metric_key) {
                    if let Some(bl) = exec_bl.get(baseline_key) {
                        let ratio = if bl.value.abs() > 1e-10 { metric.mean / bl.value } else { 0.0 };
                        comps.push((metric_key.to_string(), BaselineComparison {
                            human_value: bl.value, source: bl.source.to_string(),
                            population: bl.population.to_string(), ratio,
                        }));
                    }
                }
            }
        }

        // Flanker-specific: congruent_accuracy, incongruent_accuracy
        if benchmark.contains("Flanker") {
            for (metric_key, baseline_key) in [
                ("congruent_accuracy", "flanker_congruent_accuracy"),
                ("incongruent_accuracy", "flanker_incongruent_accuracy"),
            ] {
                if let Some(metric) = result.metrics.get(metric_key) {
                    if let Some(bl) = exec_bl.get(baseline_key) {
                        let ratio = if bl.value.abs() > 1e-10 { metric.mean / bl.value } else { 0.0 };
                        comps.push((metric_key.to_string(), BaselineComparison {
                            human_value: bl.value, source: bl.source.to_string(),
                            population: bl.population.to_string(), ratio,
                        }));
                    }
                }
            }
        }

        // Only return comparisons relevant to this benchmark
        if benchmark.contains("WorM") || benchmark.contains("CogBench")
            || benchmark.contains("ToM") || benchmark.contains("Memory")
            || benchmark.contains("Executive") || benchmark.contains("Metacognition")
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
