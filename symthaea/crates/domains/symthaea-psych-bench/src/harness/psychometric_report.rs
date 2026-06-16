// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified psychometric assessment report.
//!
//! Orchestrates `CognitiveProfile`, `NeuromodCorrelationMatrix`, `StaircaseResult`,
//! and `NormativeReport` into a single comprehensive markdown report.

use serde::{Deserialize, Serialize};

use super::PsychBenchmark;
use super::cognitive_profile::CognitiveProfile;
use super::config::BenchmarkConfig;
use super::neuromod_correlation::NeuromodCorrelationMatrix;
use super::report::{BenchmarkReport, key_metric_for_benchmark};
use super::staircase::{StaircaseConfig, StaircaseResult, run_staircase};

/// A full psychometric assessment combining all analysis modules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsychometricReport {
    /// Cognitive domain profile (8-domain radar).
    pub profile: CognitiveProfile,
    /// Summary statistics.
    pub summary: ReportSummary,
    /// Per-benchmark details.
    pub benchmark_details: Vec<BenchmarkDetail>,
    /// Staircase results (if run).
    #[serde(default)]
    pub staircase_results: Vec<StaircaseResult>,
    /// Neuromod correlations (if available).
    pub neuromod_correlations: Option<NeuromodCorrelationMatrix>,
}

/// High-level summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Total benchmarks run.
    pub n_benchmarks: usize,
    /// Total metrics collected.
    pub n_metrics: usize,
    /// Overall composite score (0.0–1.0).
    pub overall_score: f64,
    /// Strongest cognitive domain.
    pub strongest_domain: String,
    /// Weakest cognitive domain.
    pub weakest_domain: String,
    /// Number of metrics exceeding human baseline (if normative data available).
    pub n_above_human: usize,
    /// Number of metrics below human baseline.
    pub n_below_human: usize,
}

/// Detail for a single benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkDetail {
    /// Benchmark name.
    pub benchmark: String,
    /// Key metric name.
    pub key_metric: String,
    /// Key metric value.
    pub key_value: f64,
    /// All metrics for this benchmark.
    pub metrics: Vec<(String, f64, f64)>, // (name, mean, std_dev)
    /// Number of trials.
    pub n_trials: usize,
    /// Elapsed time in ms.
    pub elapsed_ms: u64,
}

impl PsychometricReport {
    /// Generate a psychometric report from a benchmark report.
    pub fn from_report(report: &BenchmarkReport) -> Self {
        let profile = CognitiveProfile::from_report(report);

        let mut benchmark_details = Vec::new();
        let mut n_metrics = 0usize;

        for result in &report.results {
            let key_metric = key_metric_for_benchmark(&result.benchmark);
            let key_value = result
                .metrics
                .get(key_metric)
                .map(|m| m.mean)
                .unwrap_or(0.0);

            let metrics: Vec<(String, f64, f64)> = result
                .metrics
                .iter()
                .map(|(k, v)| (k.clone(), v.mean, v.std_dev))
                .collect();
            n_metrics += metrics.len();

            benchmark_details.push(BenchmarkDetail {
                benchmark: result.benchmark.clone(),
                key_metric: key_metric.to_string(),
                key_value,
                metrics,
                n_trials: result.trials_per_condition,
                elapsed_ms: result.elapsed_ms,
            });
        }

        let summary = ReportSummary {
            n_benchmarks: report.results.len(),
            n_metrics,
            overall_score: profile.overall,
            strongest_domain: profile.strongest.clone(),
            weakest_domain: profile.weakest.clone(),
            n_above_human: 0, // Set by normative comparison if available
            n_below_human: 0,
        };

        Self {
            profile,
            summary,
            benchmark_details,
            staircase_results: Vec::new(),
            neuromod_correlations: None,
        }
    }

    /// Add staircase results from adaptive threshold estimation.
    pub fn add_staircase_results(&mut self, results: Vec<StaircaseResult>) {
        self.staircase_results = results;
    }

    /// Add neuromod correlation matrix.
    pub fn add_neuromod_correlations(&mut self, matrix: NeuromodCorrelationMatrix) {
        self.neuromod_correlations = Some(matrix);
    }

    /// Run staircases on a set of benchmarks and add results.
    pub fn run_staircases(
        &mut self,
        benchmarks: &[&dyn PsychBenchmark],
        base_config: &BenchmarkConfig,
        staircase_config: &StaircaseConfig,
    ) {
        let results: Vec<StaircaseResult> = benchmarks
            .iter()
            .map(|b| run_staircase(*b, base_config, staircase_config))
            .collect();
        self.staircase_results = results;
    }

    /// Generate the full markdown report.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();

        // Title
        out.push_str("# Psychometric Assessment Report\n\n");

        // Executive summary
        out.push_str("## Executive Summary\n\n");
        out.push_str(&format!(
            "- **Benchmarks run**: {}\n",
            self.summary.n_benchmarks
        ));
        out.push_str(&format!(
            "- **Metrics collected**: {}\n",
            self.summary.n_metrics
        ));
        out.push_str(&format!(
            "- **Overall composite**: {:.1}%\n",
            self.summary.overall_score * 100.0
        ));
        out.push_str(&format!(
            "- **Strongest domain**: {}\n",
            self.summary.strongest_domain
        ));
        out.push_str(&format!(
            "- **Weakest domain**: {}\n\n",
            self.summary.weakest_domain
        ));

        // Cognitive profile (radar chart)
        out.push_str(&self.profile.to_radar_markdown());
        out.push('\n');

        // Benchmark details table
        out.push_str("## Benchmark Results\n\n");
        out.push_str("| Benchmark | Key Metric | Value | Trials | Time (ms) |\n");
        out.push_str("|-----------|-----------|-------|--------|----------|\n");
        for d in &self.benchmark_details {
            out.push_str(&format!(
                "| {} | {} | {:.3} | {} | {} |\n",
                d.benchmark, d.key_metric, d.key_value, d.n_trials, d.elapsed_ms
            ));
        }
        out.push('\n');

        // Staircase results
        if !self.staircase_results.is_empty() {
            out.push_str("## Adaptive Threshold Estimation\n\n");
            out.push_str("| Benchmark | Threshold | SE | Target | Reversals |\n");
            out.push_str("|-----------|-----------|------|--------|----------|\n");
            for s in &self.staircase_results {
                out.push_str(&format!(
                    "| {} | {:.3} | {:.3} | {:.1}% | {} |\n",
                    s.benchmark,
                    s.threshold,
                    s.threshold_se,
                    s.target_accuracy * 100.0,
                    s.n_reversals
                ));
            }
            out.push('\n');
        }

        // Neuromod correlations
        if let Some(ref matrix) = self.neuromod_correlations {
            out.push_str(&matrix.to_markdown());
            out.push('\n');

            let sig = matrix.significant(0.05);
            if !sig.is_empty() {
                out.push_str("### Significant Correlations (p < 0.05)\n\n");
                for c in &sig {
                    let dir = if c.r > 0.0 { "+" } else { "-" };
                    out.push_str(&format!(
                        "- **{} x {}**: r={:.3} ({}), p={:.3}\n",
                        c.x_name, c.y_name, c.r, dir, c.p_value
                    ));
                }
                out.push('\n');
            }
        }

        // Domain-level analysis
        out.push_str("## Domain Analysis\n\n");
        for d in &self.profile.domains {
            let bar_len = (d.score * 20.0).round() as usize;
            let bar = "#".repeat(bar_len);
            out.push_str(&format!(
                "### {} ({:.1}% — {})\n\n",
                d.domain,
                d.score * 100.0,
                d.interpretation
            ));
            out.push_str(&format!(
                "```\n[{}{}]\n```\n\n",
                bar,
                " ".repeat(20 - bar_len)
            ));
            for (bench, score) in &d.benchmark_scores {
                let short = bench.split("::").last().unwrap_or(bench);
                out.push_str(&format!("- {}: {:.1}%\n", short, score * 100.0));
            }
            out.push('\n');
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::report::{BenchmarkResult, MetricValue};

    fn make_report() -> BenchmarkReport {
        let mut report = BenchmarkReport::new();

        let mut stroop = BenchmarkResult::new("Executive::Stroop", None);
        stroop.insert("incongruent_accuracy", MetricValue::from_samples(&[0.10]));
        stroop.insert("congruent_accuracy", MetricValue::from_samples(&[0.95]));
        stroop.elapsed_ms = 50;
        stroop.trials_per_condition = 20;
        report.add(stroop);

        let mut nback = BenchmarkResult::new("WorM::N-back", None);
        nback.insert("nback_2::accuracy", MetricValue::from_samples(&[0.72]));
        nback.elapsed_ms = 30;
        nback.trials_per_condition = 20;
        report.add(nback);

        let mut pvt = BenchmarkResult::new("SustainedAttention::PVT", None);
        pvt.insert("vigilance_decrement", MetricValue::from_samples(&[0.15]));
        pvt.elapsed_ms = 20;
        pvt.trials_per_condition = 20;
        report.add(pvt);

        report
    }

    #[test]
    fn test_psychometric_report_from_report() {
        let report = make_report();
        let pr = PsychometricReport::from_report(&report);

        assert_eq!(pr.summary.n_benchmarks, 3);
        assert!(pr.summary.n_metrics > 0);
        assert!(pr.summary.overall_score >= 0.0);
        assert!(!pr.summary.strongest_domain.is_empty());
    }

    #[test]
    fn test_benchmark_details_populated() {
        let report = make_report();
        let pr = PsychometricReport::from_report(&report);

        assert_eq!(pr.benchmark_details.len(), 3);

        let stroop = pr
            .benchmark_details
            .iter()
            .find(|d| d.benchmark == "Executive::Stroop")
            .unwrap();
        assert_eq!(stroop.key_metric, "incongruent_accuracy");
        assert!((stroop.key_value - 0.10).abs() < 1e-10);
    }

    #[test]
    fn test_markdown_contains_sections() {
        let report = make_report();
        let pr = PsychometricReport::from_report(&report);
        let md = pr.to_markdown();

        assert!(md.contains("Psychometric Assessment Report"));
        assert!(md.contains("Executive Summary"));
        assert!(md.contains("Benchmark Results"));
        assert!(md.contains("Domain Analysis"));
    }

    #[test]
    fn test_markdown_contains_benchmarks() {
        let report = make_report();
        let pr = PsychometricReport::from_report(&report);
        let md = pr.to_markdown();

        assert!(md.contains("Executive::Stroop"));
        assert!(md.contains("WorM::N-back"));
        assert!(md.contains("SustainedAttention::PVT"));
    }

    #[test]
    fn test_add_staircase_results() {
        let report = make_report();
        let mut pr = PsychometricReport::from_report(&report);

        let staircase = StaircaseResult {
            benchmark: "Test".to_string(),
            threshold: 0.45,
            threshold_se: 0.05,
            trajectory: vec![],
            n_reversals: 4,
            target_accuracy: 0.707,
            accuracy_at_threshold: 0.72,
        };
        pr.add_staircase_results(vec![staircase]);
        assert_eq!(pr.staircase_results.len(), 1);

        let md = pr.to_markdown();
        assert!(md.contains("Adaptive Threshold Estimation"));
    }

    #[test]
    fn test_empty_report() {
        let report = BenchmarkReport::new();
        let pr = PsychometricReport::from_report(&report);

        assert_eq!(pr.summary.n_benchmarks, 0);
        assert_eq!(pr.summary.n_metrics, 0);
        assert!(pr.benchmark_details.is_empty());
    }

    #[test]
    fn test_key_metric_mapping() {
        assert_eq!(
            key_metric_for_benchmark("Executive::Stroop"),
            "incongruent_accuracy"
        );
        assert_eq!(
            key_metric_for_benchmark("Executive::WCST"),
            "categories_completed"
        );
        assert_eq!(
            key_metric_for_benchmark("WorM::N-back"),
            "nback_2::accuracy"
        );
        assert_eq!(
            key_metric_for_benchmark("SustainedAttention::PVT"),
            "vigilance_decrement"
        );
        assert_eq!(
            key_metric_for_benchmark("Metacognition::Calibration"),
            "calibration_error_ece"
        );
        assert_eq!(
            key_metric_for_benchmark("Reasoning::ArcFluid"),
            "transfer_accuracy"
        );
        assert_eq!(
            key_metric_for_benchmark("Unknown::Bench"),
            "overall_accuracy"
        );
    }
}
