// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regression tracking: save baseline snapshots and compare against them.

use crate::harness::report::{BenchmarkReport, MetricValue};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Current schema version for snapshot files.
pub const SNAPSHOT_SCHEMA_VERSION: &str = "1.1";

fn default_schema_version() -> String {
    "1.0".to_string()
}

/// A snapshot of benchmark results for regression comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionSnapshot {
    /// Human-readable name for this snapshot (e.g., "v0.5.0-baseline").
    pub name: String,
    /// Timestamp of when the snapshot was taken.
    pub timestamp: String,
    /// Git commit hash at time of capture.
    pub git_hash: Option<String>,
    /// Free-form configuration summary.
    pub config_summary: String,
    /// Schema version for forward/backward compatibility.
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    /// benchmark_name -> metric_name -> MetricValue
    pub metrics: BTreeMap<String, BTreeMap<String, MetricValue>>,
}

/// Severity level for a regression result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionSeverity {
    /// Metric is within baseline confidence interval.
    Pass,
    /// 5-10% degradation below baseline CI.
    Warning,
    /// >10% degradation below baseline CI.
    Critical,
}

/// Comparison result for a single metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    /// Benchmark name.
    pub benchmark: String,
    /// Metric name within the benchmark.
    pub metric: String,
    /// Baseline mean value.
    pub baseline_mean: f64,
    /// Baseline 95% CI lower bound.
    pub baseline_ci_lower: f64,
    /// Baseline 95% CI upper bound.
    pub baseline_ci_upper: f64,
    /// Current mean value.
    pub current_mean: f64,
    /// Percentage change from baseline (negative = degradation).
    pub delta_pct: f64,
    /// Severity classification.
    pub severity: RegressionSeverity,
}

/// Full regression report comparing current results against a baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    /// Name of the baseline snapshot.
    pub baseline_name: String,
    /// Name of the current snapshot.
    pub current_name: String,
    /// Per-metric comparison results.
    pub results: Vec<RegressionResult>,
    /// Aggregate summary counts.
    pub summary: RegressionSummary,
}

/// Aggregate counts for a regression report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionSummary {
    /// Total number of metrics compared.
    pub total_metrics: usize,
    /// Number of metrics that passed.
    pub passed: usize,
    /// Number of metrics with warning-level regressions.
    pub warnings: usize,
    /// Number of metrics with critical regressions.
    pub critical: usize,
}

impl RegressionSnapshot {
    /// Create a snapshot from a [`BenchmarkReport`].
    pub fn from_report(report: &BenchmarkReport, name: &str) -> Self {
        let mut metrics = BTreeMap::new();
        for result in &report.results {
            let bench_metrics: BTreeMap<String, MetricValue> = result.metrics.clone();
            metrics.insert(result.benchmark.clone(), bench_metrics);
        }
        Self {
            name: name.to_string(),
            timestamp: report.timestamp.clone(),
            git_hash: None,
            config_summary: String::new(),
            schema_version: SNAPSHOT_SCHEMA_VERSION.to_string(),
            metrics,
        }
    }

    /// Set git hash (builder pattern).
    pub fn with_git_hash(mut self, hash: String) -> Self {
        self.git_hash = Some(hash);
        self
    }

    /// Serialize to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Save snapshot to a file.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = self.to_json().map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Load snapshot from a file.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json).map_err(std::io::Error::other)
    }

    /// Age of this snapshot in days (from timestamp to now).
    ///
    /// Returns `None` if the timestamp cannot be parsed.
    pub fn age_days(&self) -> Option<i64> {
        let ts = chrono::DateTime::parse_from_rfc3339(&self.timestamp).ok()?;
        let now = chrono::Utc::now();
        Some((now - ts.with_timezone(&chrono::Utc)).num_days())
    }

    /// Returns a staleness warning if the snapshot is older than `max_days`.
    ///
    /// Returns `None` if fresh or if the timestamp can't be parsed.
    pub fn staleness_warning(&self, max_days: i64) -> Option<String> {
        let age = self.age_days()?;
        if age > max_days {
            Some(format!(
                "Baseline \"{}\" is {} days old (generated {}). Consider regenerating.",
                self.name, age, self.timestamp
            ))
        } else {
            None
        }
    }

    /// Validate the snapshot for structural integrity.
    ///
    /// Returns `Ok(())` if valid, or a list of validation errors.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.name.is_empty() {
            errors.push("Snapshot name is empty".to_string());
        }

        if self.metrics.is_empty() {
            errors.push("Snapshot contains no benchmark metrics".to_string());
        }

        for (bench, bench_metrics) in &self.metrics {
            for (metric_name, val) in bench_metrics {
                if !val.mean.is_finite() {
                    errors.push(format!(
                        "{}/{}: mean is not finite ({:?})",
                        bench, metric_name, val.mean
                    ));
                }
                if !val.std_dev.is_finite() {
                    errors.push(format!(
                        "{}/{}: std_dev is not finite ({:?})",
                        bench, metric_name, val.std_dev
                    ));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Return the default snapshots directory for this crate.
    pub fn snapshot_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("snapshots")
    }

    /// Migrate a JSON string to the current schema version.
    ///
    /// Handles version-aware deserialization: v1.0 snapshots (no `schema_version`
    /// field) are upgraded to v1.1 with the field set.
    pub fn migrate(json: &str) -> Result<Self, serde_json::Error> {
        let mut snapshot: Self = serde_json::from_str(json)?;
        // v1.0 files lack schema_version; serde default fills "1.0"
        if snapshot.schema_version == "1.0" {
            snapshot.schema_version = SNAPSHOT_SCHEMA_VERSION.to_string();
        }
        Ok(snapshot)
    }
}

impl RegressionReport {
    /// Compare current results against a baseline snapshot.
    ///
    /// `warning_pct` and `critical_pct` are degradation thresholds expressed as
    /// fractions (e.g., 0.05 for 5%, 0.10 for 10%). A regression is flagged only
    /// when the current mean falls below the baseline confidence interval *and*
    /// the percentage drop exceeds the threshold.
    pub fn compare(
        baseline: &RegressionSnapshot,
        current: &RegressionSnapshot,
        warning_pct: f64,
        critical_pct: f64,
    ) -> Self {
        let mut results = Vec::new();

        for (bench_name, baseline_metrics) in &baseline.metrics {
            if let Some(current_metrics) = current.metrics.get(bench_name) {
                for (metric_name, baseline_val) in baseline_metrics {
                    if let Some(current_val) = current_metrics.get(metric_name) {
                        // Skip metrics where baseline is near zero (avoid div-by-zero)
                        if baseline_val.mean.abs() < 1e-10 {
                            results.push(RegressionResult {
                                benchmark: bench_name.clone(),
                                metric: metric_name.clone(),
                                baseline_mean: baseline_val.mean,
                                baseline_ci_lower: baseline_val.ci_lower,
                                baseline_ci_upper: baseline_val.ci_upper,
                                current_mean: current_val.mean,
                                delta_pct: 0.0,
                                severity: RegressionSeverity::Pass,
                            });
                            continue;
                        }

                        let delta_pct =
                            (current_val.mean - baseline_val.mean) / baseline_val.mean.abs();

                        // Detect regression: current mean dropped below baseline CI
                        let severity = if current_val.mean < baseline_val.ci_lower {
                            if delta_pct.abs() >= critical_pct {
                                RegressionSeverity::Critical
                            } else if delta_pct.abs() >= warning_pct {
                                RegressionSeverity::Warning
                            } else {
                                RegressionSeverity::Pass
                            }
                        } else {
                            RegressionSeverity::Pass
                        };

                        results.push(RegressionResult {
                            benchmark: bench_name.clone(),
                            metric: metric_name.clone(),
                            baseline_mean: baseline_val.mean,
                            baseline_ci_lower: baseline_val.ci_lower,
                            baseline_ci_upper: baseline_val.ci_upper,
                            current_mean: current_val.mean,
                            delta_pct,
                            severity,
                        });
                    }
                }
            }
        }

        let total = results.len();
        let warnings = results
            .iter()
            .filter(|r| r.severity == RegressionSeverity::Warning)
            .count();
        let critical = results
            .iter()
            .filter(|r| r.severity == RegressionSeverity::Critical)
            .count();

        Self {
            baseline_name: baseline.name.clone(),
            current_name: current.name.clone(),
            results,
            summary: RegressionSummary {
                total_metrics: total,
                passed: total - warnings - critical,
                warnings,
                critical,
            },
        }
    }

    /// Returns `true` if any metric has a Critical regression.
    pub fn has_critical(&self) -> bool {
        self.summary.critical > 0
    }

    /// Returns `true` if any metric has a Warning or Critical regression.
    pub fn has_regressions(&self) -> bool {
        self.summary.warnings > 0 || self.summary.critical > 0
    }

    /// Format as a human-readable summary string.
    pub fn format_summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Regression Report: {} vs {}\n",
            self.current_name, self.baseline_name
        ));
        out.push_str(&format!(
            "Metrics: {} total, {} pass, {} warning, {} critical\n\n",
            self.summary.total_metrics,
            self.summary.passed,
            self.summary.warnings,
            self.summary.critical
        ));

        // Only show non-passing metrics
        for r in &self.results {
            if r.severity != RegressionSeverity::Pass {
                let tag = match r.severity {
                    RegressionSeverity::Warning => "WARN",
                    RegressionSeverity::Critical => "CRIT",
                    RegressionSeverity::Pass => "PASS",
                };
                out.push_str(&format!(
                    "[{}] {}/{}: {:.4} -> {:.4} ({:+.1}%)\n",
                    tag,
                    r.benchmark,
                    r.metric,
                    r.baseline_mean,
                    r.current_mean,
                    r.delta_pct * 100.0
                ));
            }
        }

        if !self.has_regressions() {
            out.push_str("All metrics within baseline CI. No regressions detected.\n");
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_snapshot(name: &str, accuracy: f64) -> RegressionSnapshot {
        let mut metrics = BTreeMap::new();
        let mut bench_metrics = BTreeMap::new();
        bench_metrics.insert(
            "accuracy".to_string(),
            MetricValue {
                mean: accuracy,
                std_dev: 0.05,
                n: 20,
                ci_lower: accuracy - 0.02,
                ci_upper: accuracy + 0.02,
            },
        );
        metrics.insert("TestBench".to_string(), bench_metrics);
        RegressionSnapshot {
            name: name.to_string(),
            timestamp: "2026-02-21".to_string(),
            git_hash: None,
            config_summary: String::new(),
            schema_version: SNAPSHOT_SCHEMA_VERSION.to_string(),
            metrics,
        }
    }

    #[test]
    fn test_no_regression() {
        let baseline = make_test_snapshot("baseline", 0.90);
        let current = make_test_snapshot("current", 0.91);
        let report = RegressionReport::compare(&baseline, &current, 0.05, 0.10);
        assert!(!report.has_regressions());
        assert_eq!(report.summary.passed, 1);
    }

    #[test]
    fn test_warning_regression() {
        let baseline = make_test_snapshot("baseline", 0.90);
        let current = make_test_snapshot("current", 0.83); // ~7.8% drop, below CI
        let report = RegressionReport::compare(&baseline, &current, 0.05, 0.10);
        assert!(report.has_regressions());
        assert_eq!(report.summary.warnings, 1);
        assert_eq!(report.summary.critical, 0);
    }

    #[test]
    fn test_critical_regression() {
        let baseline = make_test_snapshot("baseline", 0.90);
        let current = make_test_snapshot("current", 0.78); // ~13.3% drop, below CI
        let report = RegressionReport::compare(&baseline, &current, 0.05, 0.10);
        assert!(report.has_critical());
        assert_eq!(report.summary.critical, 1);
    }

    #[test]
    fn test_snapshot_json_roundtrip() {
        let snapshot = make_test_snapshot("test", 0.85);
        let json = snapshot.to_json().unwrap();
        let loaded = RegressionSnapshot::from_json(&json).unwrap();
        assert_eq!(loaded.name, "test");
        let m = &loaded.metrics["TestBench"]["accuracy"];
        assert!((m.mean - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_format_summary() {
        let baseline = make_test_snapshot("v1.0", 0.90);
        let current = make_test_snapshot("v1.1", 0.78);
        let report = RegressionReport::compare(&baseline, &current, 0.05, 0.10);
        let summary = report.format_summary();
        assert!(summary.contains("CRIT"));
        assert!(summary.contains("TestBench"));
    }

    #[test]
    fn test_missing_benchmark_ignored() {
        let baseline = make_test_snapshot("baseline", 0.90);
        let mut current = make_test_snapshot("current", 0.90);
        current
            .metrics
            .insert("NewBench".to_string(), BTreeMap::new());
        let report = RegressionReport::compare(&baseline, &current, 0.05, 0.10);
        // Only compares benchmarks that exist in baseline
        assert_eq!(report.summary.total_metrics, 1);
    }

    #[test]
    fn test_near_zero_baseline_passes() {
        let mut baseline = make_test_snapshot("baseline", 0.90);
        // Insert a near-zero metric
        baseline.metrics.get_mut("TestBench").unwrap().insert(
            "near_zero".to_string(),
            MetricValue {
                mean: 0.0,
                std_dev: 0.0,
                n: 10,
                ci_lower: 0.0,
                ci_upper: 0.0,
            },
        );
        let mut current = make_test_snapshot("current", 0.90);
        current.metrics.get_mut("TestBench").unwrap().insert(
            "near_zero".to_string(),
            MetricValue {
                mean: 0.5,
                std_dev: 0.1,
                n: 10,
                ci_lower: 0.4,
                ci_upper: 0.6,
            },
        );
        let report = RegressionReport::compare(&baseline, &current, 0.05, 0.10);
        // Near-zero baseline should always pass (no div-by-zero)
        assert!(!report.has_regressions());
        assert_eq!(report.summary.total_metrics, 2);
    }

    #[test]
    fn test_snapshot_save_load_roundtrip() {
        let snapshot = make_test_snapshot("persist-test", 0.92);
        let dir = std::env::temp_dir();
        let path = dir.join("psych_bench_snapshot_test.json");
        snapshot.save(&path).unwrap();
        let loaded = RegressionSnapshot::load(&path).unwrap();
        assert_eq!(loaded.name, "persist-test");
        assert!((loaded.metrics["TestBench"]["accuracy"].mean - 0.92).abs() < 1e-10);
        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_with_git_hash() {
        let snapshot = make_test_snapshot("tagged", 0.85).with_git_hash("abc123def".to_string());
        assert_eq!(snapshot.git_hash.as_deref(), Some("abc123def"));
    }

    #[test]
    fn test_from_report() {
        use crate::harness::report::{BenchmarkReport, BenchmarkResult};
        let mut report = BenchmarkReport::new();
        let mut result = BenchmarkResult::new("MyBench", None);
        result.insert("score", MetricValue::from_samples(&[0.8, 0.9, 0.85]));
        report.add(result);
        let snapshot = RegressionSnapshot::from_report(&report, "from-report");
        assert_eq!(snapshot.name, "from-report");
        assert!(snapshot.metrics.contains_key("MyBench"));
        assert!(snapshot.metrics["MyBench"].contains_key("score"));
    }

    #[test]
    fn test_age_days_fresh() {
        let snapshot = make_test_snapshot("fresh", 0.90);
        // Snapshot was just created with chrono::Utc::now() in from_report path,
        // but our test helper uses a hardcoded timestamp. Override it.
        let mut s = snapshot;
        s.timestamp = chrono::Utc::now().to_rfc3339();
        let age = s.age_days().unwrap();
        assert!(
            age <= 1,
            "Fresh snapshot should be ~0 days old, got {}",
            age
        );
    }

    #[test]
    fn test_age_days_old() {
        let mut s = make_test_snapshot("old", 0.85);
        // Set timestamp to 60 days ago
        let old = chrono::Utc::now() - chrono::Duration::days(60);
        s.timestamp = old.to_rfc3339();
        let age = s.age_days().unwrap();
        assert!((age - 60).abs() <= 1, "Should be ~60 days old, got {}", age);
    }

    #[test]
    fn test_staleness_warning_fresh() {
        let mut s = make_test_snapshot("fresh", 0.90);
        s.timestamp = chrono::Utc::now().to_rfc3339();
        assert!(
            s.staleness_warning(30).is_none(),
            "Fresh snapshot shouldn't warn"
        );
    }

    #[test]
    fn test_staleness_warning_stale() {
        let mut s = make_test_snapshot("stale", 0.85);
        let old = chrono::Utc::now() - chrono::Duration::days(45);
        s.timestamp = old.to_rfc3339();
        let warning = s.staleness_warning(30);
        assert!(warning.is_some(), "Stale snapshot should warn");
        assert!(warning.unwrap().contains("45 days old"));
    }

    // --- Schema versioning tests ---

    #[test]
    fn test_schema_version_default() {
        // Deserializing a v1.0 JSON (no schema_version field) should default to "1.0"
        let json = r#"{
            "name": "old-baseline",
            "timestamp": "2026-01-01T00:00:00Z",
            "git_hash": null,
            "config_summary": "",
            "metrics": {}
        }"#;
        let snapshot: RegressionSnapshot = serde_json::from_str(json).unwrap();
        assert_eq!(snapshot.schema_version, "1.0");
    }

    #[test]
    fn test_schema_version_set_on_create() {
        use crate::harness::report::{BenchmarkReport, BenchmarkResult};
        let mut report = BenchmarkReport::new();
        let mut result = BenchmarkResult::new("Bench", None);
        result.insert("acc", MetricValue::from_samples(&[0.8, 0.9]));
        report.add(result);
        let snapshot = RegressionSnapshot::from_report(&report, "new");
        assert_eq!(snapshot.schema_version, SNAPSHOT_SCHEMA_VERSION);
    }

    #[test]
    fn test_validate_valid() {
        let s = make_test_snapshot("valid", 0.85);
        assert!(s.validate().is_ok());
    }

    #[test]
    fn test_validate_empty_name() {
        let mut s = make_test_snapshot("", 0.85);
        s.name = String::new();
        let errs = s.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("name is empty")));
    }

    #[test]
    fn test_validate_empty_metrics() {
        let s = RegressionSnapshot {
            name: "empty".to_string(),
            timestamp: "2026-02-26".to_string(),
            git_hash: None,
            config_summary: String::new(),
            schema_version: SNAPSHOT_SCHEMA_VERSION.to_string(),
            metrics: BTreeMap::new(),
        };
        let errs = s.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("no benchmark metrics")));
    }

    #[test]
    fn test_validate_nan_metric() {
        let mut s = make_test_snapshot("nan-test", 0.85);
        s.metrics.get_mut("TestBench").unwrap().insert(
            "bad".to_string(),
            MetricValue {
                mean: f64::NAN,
                std_dev: 0.1,
                n: 10,
                ci_lower: 0.0,
                ci_upper: 1.0,
            },
        );
        let errs = s.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("mean is not finite")));
    }

    #[test]
    fn test_migrate_v1_0_to_v1_1() {
        let json = r#"{
            "name": "legacy",
            "timestamp": "2026-01-15T00:00:00Z",
            "git_hash": "abc123",
            "config_summary": "dim=512",
            "metrics": {
                "Stroop": {
                    "interference": {"mean": 0.10, "std_dev": 0.02, "n": 50, "ci_lower": 0.09, "ci_upper": 0.11}
                }
            }
        }"#;
        let migrated = RegressionSnapshot::migrate(json).unwrap();
        assert_eq!(migrated.schema_version, SNAPSHOT_SCHEMA_VERSION);
        assert_eq!(migrated.name, "legacy");
        assert!(migrated.metrics.contains_key("Stroop"));
    }

    #[test]
    fn test_backward_compat_roundtrip() {
        // Create v1.1 snapshot, serialize, deserialize — schema_version preserved
        let s = make_test_snapshot("roundtrip", 0.90);
        assert_eq!(s.schema_version, SNAPSHOT_SCHEMA_VERSION);
        let json = s.to_json().unwrap();
        let loaded = RegressionSnapshot::from_json(&json).unwrap();
        assert_eq!(loaded.schema_version, SNAPSHOT_SCHEMA_VERSION);
        assert_eq!(loaded.name, "roundtrip");
    }
}
