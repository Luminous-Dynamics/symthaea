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

    /// Full summary of all results.
    pub fn summary(&self) -> String {
        let mut lines = vec![format!("Psych Benchmark Report ({})", self.timestamp)];
        lines.push(format!("{} benchmarks run", self.results.len()));
        lines.push(String::new());
        for result in &self.results {
            lines.push(result.summary());
            lines.push(String::new());
        }
        lines.join("\n")
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
