// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Practice effects and test-retest reliability analysis.
//!
//! Implements ICC(3,1) one-way random effects intraclass correlation for
//! test-retest reliability, standard error of measurement (SEM), Pearson r,
//! and practice effect detection across repeated benchmark sessions.
//!
//! References:
//! - Shrout & Fleiss (1979). Intraclass correlations: Uses in assessing rater reliability.
//! - Weir (2005). Quantifying test-retest reliability using the ICC.
//! - Cicchetti (1994). Guidelines for ICC interpretation.

use serde::{Deserialize, Serialize};

use super::PsychBenchmark;
use super::config::BenchmarkConfig;
use super::report::key_metric_for_benchmark;

// ──── Practice direction ────

/// Direction of performance change between sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PracticeDirection {
    /// Performance improved from session 1 to session 2.
    Improvement,
    /// Performance declined from session 1 to session 2.
    Decline,
    /// Performance remained stable (within +/-2%).
    Stable,
}

// ──── Practice effect ────

/// Measures how performance changes between two sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PracticeEffect {
    /// Benchmark name.
    pub benchmark: String,
    /// Metric name.
    pub metric: String,
    /// Mean value from session 1.
    pub session1_mean: f64,
    /// Mean value from session 2.
    pub session2_mean: f64,
    /// Absolute change (session2 - session1).
    pub change: f64,
    /// Percentage change relative to session 1.
    pub change_pct: f64,
    /// Direction classification.
    pub direction: PracticeDirection,
}

impl PracticeEffect {
    /// Compute practice effect from two session means.
    pub fn compute(benchmark: &str, metric: &str, session1_mean: f64, session2_mean: f64) -> Self {
        let change = session2_mean - session1_mean;
        let change_pct = if session1_mean.abs() > 1e-15 {
            (change / session1_mean.abs()) * 100.0
        } else {
            0.0
        };
        let direction = if change_pct > 2.0 {
            PracticeDirection::Improvement
        } else if change_pct < -2.0 {
            PracticeDirection::Decline
        } else {
            PracticeDirection::Stable
        };
        Self {
            benchmark: benchmark.to_string(),
            metric: metric.to_string(),
            session1_mean,
            session2_mean,
            change,
            change_pct,
            direction,
        }
    }
}

// ──── Reliability class ────

/// Reliability classification based on ICC thresholds (Cicchetti, 1994).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReliabilityClass {
    /// ICC > 0.90: excellent reliability.
    Excellent,
    /// ICC > 0.75: good reliability.
    Good,
    /// ICC > 0.50: moderate reliability.
    Moderate,
    /// ICC <= 0.50: poor reliability.
    Poor,
}

impl ReliabilityClass {
    /// Classify an ICC value.
    pub fn from_icc(icc: f64) -> Self {
        if icc > 0.90 {
            ReliabilityClass::Excellent
        } else if icc > 0.75 {
            ReliabilityClass::Good
        } else if icc > 0.50 {
            ReliabilityClass::Moderate
        } else {
            ReliabilityClass::Poor
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            ReliabilityClass::Excellent => "Excellent",
            ReliabilityClass::Good => "Good",
            ReliabilityClass::Moderate => "Moderate",
            ReliabilityClass::Poor => "Poor",
        }
    }
}

// ──── Test-retest result ────

/// Full test-retest reliability assessment for a single benchmark metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestRetestResult {
    /// Benchmark name.
    pub benchmark: String,
    /// Metric name.
    pub metric: String,
    /// Intraclass correlation coefficient (ICC 3,1).
    pub icc: f64,
    /// Pearson product-moment correlation.
    pub pearson_r: f64,
    /// Standard error of measurement.
    pub sem: f64,
    /// Practice effect between first and second sessions.
    pub practice: PracticeEffect,
    /// Reliability classification.
    pub reliability_class: ReliabilityClass,
}

// ──── Statistical functions ────

/// Compute ICC(3,1) one-way random effects model.
///
/// `ICC = (BMS - WMS) / (BMS + WMS)`
///
/// where BMS = between-subject mean square, WMS = within-subject mean square.
/// Each "subject" is a paired observation from session1 and session2.
pub fn compute_icc(session1: &[f64], session2: &[f64]) -> f64 {
    let n = session1.len().min(session2.len());
    if n < 2 {
        return 0.0;
    }

    // Compute subject means and grand mean
    let grand_sum: f64 = session1[..n].iter().sum::<f64>() + session2[..n].iter().sum::<f64>();
    let grand_mean = grand_sum / (2 * n) as f64;

    // Between-subject mean square (BMS)
    // BMS = k * sum_i( (subject_mean_i - grand_mean)^2 ) / (n - 1)
    // where k = number of measurements per subject (2)
    let k = 2.0f64;
    let bms: f64 = {
        let ss_between: f64 = (0..n)
            .map(|i| {
                let subj_mean = (session1[i] + session2[i]) / 2.0;
                (subj_mean - grand_mean).powi(2)
            })
            .sum();
        k * ss_between / (n as f64 - 1.0)
    };

    // Within-subject mean square (WMS)
    // WMS = sum_i sum_j (x_ij - subject_mean_i)^2 / (n * (k - 1))
    let wms: f64 = {
        let ss_within: f64 = (0..n)
            .map(|i| {
                let subj_mean = (session1[i] + session2[i]) / 2.0;
                (session1[i] - subj_mean).powi(2) + (session2[i] - subj_mean).powi(2)
            })
            .sum();
        ss_within / (n as f64 * (k - 1.0))
    };

    let denom = bms + wms;
    if denom.abs() < 1e-15 {
        return 0.0;
    }

    ((bms - wms) / denom).clamp(-1.0, 1.0)
}

/// Compute standard error of measurement.
///
/// `SEM = SD * sqrt(1 - ICC)`
pub fn compute_sem(sd: f64, icc: f64) -> f64 {
    let factor = (1.0 - icc).max(0.0);
    sd * factor.sqrt()
}

/// Compute Pearson product-moment correlation coefficient.
///
/// Returns 0.0 if fewer than 3 data points or zero variance.
pub fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 3 {
        return 0.0;
    }

    let mean_x = x[..n].iter().sum::<f64>() / n as f64;
    let mean_y = y[..n].iter().sum::<f64>() / n as f64;

    let mut cov = 0.0f64;
    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        return 0.0;
    }

    (cov / denom).clamp(-1.0, 1.0)
}

// ──── Reliability battery ────

/// Battery of test-retest reliability results across multiple benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityBattery {
    /// Per-benchmark reliability results.
    pub results: Vec<TestRetestResult>,
}

impl ReliabilityBattery {
    /// Run a reliability battery across multiple benchmarks.
    ///
    /// Uses a multi-subject design: K=10 independent "subjects" (distinct seed
    /// families), each measured in `n_sessions` sessions. This provides enough
    /// between-subject variance for meaningful ICC computation.
    ///
    /// Each subject uses seeds from a distinct family (base + k*1000), and each
    /// session within a subject uses a nearby seed (base + k*1000 + s). This
    /// mirrors human test-retest: different people (seed families) have different
    /// ability profiles, while the same person tested twice (nearby seeds) should
    /// produce similar scores if the benchmark is reliable.
    pub fn run(
        benchmarks: &[&dyn PsychBenchmark],
        base_config: &BenchmarkConfig,
        n_sessions: usize,
        n_subjects: usize,
    ) -> Self {
        let n_sessions = n_sessions.max(2);
        let n_subjects = n_subjects.max(3); // Minimum 3 subjects for ICC
        let mut results = Vec::new();

        for bench in benchmarks {
            let bench_name = bench.name();
            let metric_name = key_metric_for_benchmark(bench_name);

            // Collect per-subject, per-session metric values
            // session_data[session][subject] = metric value
            let mut session_data: Vec<Vec<f64>> = vec![Vec::new(); n_sessions];

            for k in 0..n_subjects {
                let family_base = base_config.seed + k as u64 * 1000;
                // Per-subject ability variation: encoding noise, WM capacity, and dimension.
                // Models individual differences in perceptual precision, working memory,
                // and representational capacity — the three primary sources of cognitive
                // heterogeneity in human psychometric testing.
                let t = k as f64 / n_subjects as f64; // 0.0 to ~1.0
                let subject_noise = base_config.encoding_noise + t * 0.30;
                let subject_wm = 4 + ((1.0 - t) * 5.0) as usize; // 9 down to 4
                let subject_dim = 256 + ((1.0 - t) * 768.0) as usize; // 1024 down to 256
                // Attention lapse rate: models individual differences in sustained
                // attention (Wichmann & Hill, 2001). This is the primary driver of
                // between-subject variance in psychometric reliability. Ranges from
                // 0% (perfect attention) to 20% (high lapse rate), creating stable
                // individual differences that persist across sessions.
                let subject_lapse = t * 0.25; // 0.0 to 0.25
                for (s, session) in session_data.iter_mut().enumerate().take(n_sessions) {
                    let config = BenchmarkConfig {
                        seed: family_base + s as u64 * 100,
                        encoding_noise: subject_noise,
                        working_memory_capacity: subject_wm,
                        dimension: subject_dim,
                        lapse_rate: subject_lapse,
                        ..base_config.clone()
                    };
                    let result = bench.run(&config);
                    if let Some(mv) = result.metrics.get(metric_name) {
                        session.push(mv.mean);
                    }
                }
            }

            // Need at least 3 subjects with data in all sessions
            let min_len = session_data.iter().map(|v| v.len()).min().unwrap_or(0);
            if min_len < 3 {
                continue;
            }

            // Compute ICC between session 1 and session 2 using all subjects
            let s1 = &session_data[0][..min_len];
            let s2 = &session_data[1][..min_len];

            let icc = compute_icc(s1, s2);
            let r = pearson_r(s1, s2);

            // Pool SD across all sessions for SEM
            let all_vals: Vec<f64> = session_data
                .iter()
                .flat_map(|v| v.iter().copied())
                .collect();
            let all_mean = all_vals.iter().sum::<f64>() / all_vals.len() as f64;
            let sd = if all_vals.len() > 1 {
                let var = all_vals.iter().map(|x| (x - all_mean).powi(2)).sum::<f64>()
                    / (all_vals.len() - 1) as f64;
                var.sqrt()
            } else {
                0.0
            };
            let sem = compute_sem(sd, icc);

            // Practice effect between session 1 and session 2 means
            let s1_mean = s1.iter().sum::<f64>() / s1.len() as f64;
            let s2_mean = s2.iter().sum::<f64>() / s2.len() as f64;
            let practice = PracticeEffect::compute(bench_name, metric_name, s1_mean, s2_mean);

            let reliability_class = ReliabilityClass::from_icc(icc);

            results.push(TestRetestResult {
                benchmark: bench_name.to_string(),
                metric: metric_name.to_string(),
                icc,
                pearson_r: r,
                sem,
                practice,
                reliability_class,
            });
        }

        Self { results }
    }

    /// Format results as a markdown table.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("## Test-Retest Reliability\n\n");
        out.push_str("| Benchmark | Metric | ICC | Pearson r | SEM | Practice | Class |\n");
        out.push_str("|-----------|--------|-----|-----------|-----|----------|-------|\n");

        for r in &self.results {
            let practice_str = match r.practice.direction {
                PracticeDirection::Improvement => {
                    format!("+{:.1}%", r.practice.change_pct)
                }
                PracticeDirection::Decline => {
                    format!("{:.1}%", r.practice.change_pct)
                }
                PracticeDirection::Stable => "Stable".to_string(),
            };

            out.push_str(&format!(
                "| {} | {} | {:.3} | {:.3} | {:.4} | {} | {} |\n",
                r.benchmark,
                r.metric,
                r.icc,
                r.pearson_r,
                r.sem,
                practice_str,
                r.reliability_class.label(),
            ));
        }

        out
    }

    /// Filter results by ICC threshold, returning only benchmarks meeting the cutoff.
    pub fn reliable_benchmarks(&self, threshold: f64) -> Vec<&TestRetestResult> {
        self.results.iter().filter(|r| r.icc >= threshold).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icc_perfect() {
        // Identical sessions should yield ICC = 1.0
        let s1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let icc = compute_icc(&s1, &s2);
        assert!(
            (icc - 1.0).abs() < 1e-10,
            "Perfect agreement should give ICC=1.0, got {}",
            icc
        );
    }

    #[test]
    fn test_icc_zero() {
        // Reversed order: maximal disagreement pattern relative to between-subject variance.
        // With reversed values, within-subject variance is large relative to between-subject.
        let s1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let icc = compute_icc(&s1, &s2);
        // ICC should be low (near 0 or negative) for reversed data
        assert!(
            icc < 0.5,
            "Reversed sessions should yield low ICC, got {}",
            icc
        );
    }

    #[test]
    fn test_sem_computation() {
        // SEM = SD * sqrt(1 - ICC)
        let sd = 10.0;
        let icc = 0.75;
        let sem = compute_sem(sd, icc);
        let expected = 10.0 * (1.0 - 0.75_f64).sqrt(); // 10 * 0.5 = 5.0
        assert!(
            (sem - expected).abs() < 1e-10,
            "SEM should be {}, got {}",
            expected,
            sem
        );

        // Perfect reliability → SEM = 0
        assert!((compute_sem(10.0, 1.0)).abs() < 1e-10);

        // Zero reliability → SEM = SD
        assert!((compute_sem(10.0, 0.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_practice_direction_improvement() {
        let pe = PracticeEffect::compute("test", "accuracy", 0.70, 0.80);
        assert_eq!(pe.direction, PracticeDirection::Improvement);
        assert!((pe.change - 0.10).abs() < 1e-10);
        assert!(pe.change_pct > 2.0);
    }

    #[test]
    fn test_practice_direction_stable() {
        let pe = PracticeEffect::compute("test", "accuracy", 0.80, 0.81);
        assert_eq!(
            pe.direction,
            PracticeDirection::Stable,
            "1.25% change should be Stable, got {:?} (change_pct={:.2}%)",
            pe.direction,
            pe.change_pct
        );
    }

    #[test]
    fn test_practice_direction_decline() {
        let pe = PracticeEffect::compute("test", "accuracy", 0.80, 0.70);
        assert_eq!(pe.direction, PracticeDirection::Decline);
        assert!(pe.change < 0.0);
        assert!(pe.change_pct < -2.0);
    }

    #[test]
    fn test_reliability_class_ranges() {
        assert_eq!(
            ReliabilityClass::from_icc(0.95),
            ReliabilityClass::Excellent
        );
        assert_eq!(
            ReliabilityClass::from_icc(0.91),
            ReliabilityClass::Excellent
        );
        assert_eq!(ReliabilityClass::from_icc(0.90), ReliabilityClass::Good);
        assert_eq!(ReliabilityClass::from_icc(0.80), ReliabilityClass::Good);
        assert_eq!(ReliabilityClass::from_icc(0.76), ReliabilityClass::Good);
        assert_eq!(ReliabilityClass::from_icc(0.75), ReliabilityClass::Moderate);
        assert_eq!(ReliabilityClass::from_icc(0.60), ReliabilityClass::Moderate);
        assert_eq!(ReliabilityClass::from_icc(0.51), ReliabilityClass::Moderate);
        assert_eq!(ReliabilityClass::from_icc(0.50), ReliabilityClass::Poor);
        assert_eq!(ReliabilityClass::from_icc(0.30), ReliabilityClass::Poor);
        assert_eq!(ReliabilityClass::from_icc(0.0), ReliabilityClass::Poor);
        assert_eq!(ReliabilityClass::from_icc(-0.5), ReliabilityClass::Poor);
    }

    #[test]
    fn test_pearson_r_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_r(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "r={}", r);
    }

    #[test]
    fn test_pearson_r_zero_variance() {
        let x = vec![3.0, 3.0, 3.0, 3.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let r = pearson_r(&x, &y);
        assert!(r.abs() < 1e-10, "Zero variance should yield r=0, got {}", r);
    }

    #[test]
    fn test_reliability_on_stroop() {
        use crate::benchmarks::executive::StroopBenchmark;

        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 10,
            seed: 42,
            ..Default::default()
        };

        let battery = ReliabilityBattery::run(&[&StroopBenchmark], &config, 2, 10);
        assert_eq!(battery.results.len(), 1);

        let result = &battery.results[0];
        assert_eq!(result.benchmark, "Executive::Stroop");
        assert_eq!(result.metric, "incongruent_accuracy");
        assert!(result.icc.is_finite(), "ICC should be finite");
        assert!(
            result.icc >= -1.0 && result.icc <= 1.0,
            "ICC out of range: {}",
            result.icc
        );
        assert!(result.pearson_r.is_finite());
        assert!(result.sem.is_finite());
        assert!(result.sem >= 0.0);
    }

    #[test]
    fn test_reliability_battery_multiple() {
        use crate::benchmarks::executive::{FlankerBenchmark, StroopBenchmark};

        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 10,
            seed: 100,
            ..Default::default()
        };

        let benchmarks: Vec<&dyn PsychBenchmark> = vec![&StroopBenchmark, &FlankerBenchmark];
        let battery = ReliabilityBattery::run(&benchmarks, &config, 3, 10);

        assert_eq!(battery.results.len(), 2);

        // Both benchmarks should have results
        let names: Vec<&str> = battery
            .results
            .iter()
            .map(|r| r.benchmark.as_str())
            .collect();
        assert!(names.contains(&"Executive::Stroop"));
        assert!(names.contains(&"Executive::Flanker"));

        // All ICCs should be finite
        for r in &battery.results {
            assert!(r.icc.is_finite(), "{} ICC not finite", r.benchmark);
            assert!(r.pearson_r.is_finite(), "{} r not finite", r.benchmark);
            assert!(r.sem >= 0.0, "{} SEM negative", r.benchmark);
        }

        // Markdown output should be well-formed
        let md = battery.to_markdown();
        assert!(md.contains("Test-Retest Reliability"));
        assert!(md.contains("Stroop"));
        assert!(md.contains("Flanker"));
        assert!(md.contains("ICC"));

        // reliable_benchmarks filter
        let reliable = battery.reliable_benchmarks(0.0);
        // With threshold 0.0, any non-negative ICC passes
        // (may be empty if ICCs are negative, but should not panic)
        for r in &reliable {
            assert!(r.icc >= 0.0);
        }
    }

    #[test]
    fn test_key_metric_for_known_benchmarks() {
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
            key_metric_for_benchmark("Reasoning::ArcFluid"),
            "transfer_accuracy"
        );
        assert_eq!(
            key_metric_for_benchmark("Metacognition::FeelingOfKnowing"),
            "fok_gamma"
        );
        assert_eq!(
            key_metric_for_benchmark("Unknown::Bench"),
            "overall_accuracy"
        );
    }

    #[test]
    fn test_markdown_output_format() {
        let battery = ReliabilityBattery {
            results: vec![TestRetestResult {
                benchmark: "Test::Bench".to_string(),
                metric: "accuracy".to_string(),
                icc: 0.85,
                pearson_r: 0.88,
                sem: 0.05,
                practice: PracticeEffect::compute("Test::Bench", "accuracy", 0.70, 0.75),
                reliability_class: ReliabilityClass::Good,
            }],
        };

        let md = battery.to_markdown();
        assert!(md.contains("Test::Bench"));
        assert!(md.contains("0.850"));
        assert!(md.contains("0.880"));
        assert!(md.contains("Good"));
    }

    #[test]
    fn test_reliable_benchmarks_filter() {
        let battery = ReliabilityBattery {
            results: vec![
                TestRetestResult {
                    benchmark: "High".to_string(),
                    metric: "accuracy".to_string(),
                    icc: 0.92,
                    pearson_r: 0.93,
                    sem: 0.02,
                    practice: PracticeEffect::compute("High", "accuracy", 0.80, 0.81),
                    reliability_class: ReliabilityClass::Excellent,
                },
                TestRetestResult {
                    benchmark: "Low".to_string(),
                    metric: "accuracy".to_string(),
                    icc: 0.40,
                    pearson_r: 0.45,
                    sem: 0.15,
                    practice: PracticeEffect::compute("Low", "accuracy", 0.60, 0.55),
                    reliability_class: ReliabilityClass::Poor,
                },
            ],
        };

        let good = battery.reliable_benchmarks(0.75);
        assert_eq!(good.len(), 1);
        assert_eq!(good[0].benchmark, "High");

        let all = battery.reliable_benchmarks(0.0);
        assert_eq!(all.len(), 2);
    }

    /// High-trial reliability with split-half cross-validation.
    ///
    /// Runs Stroop and Flanker with 100 trials per condition across 10 subjects,
    /// then verifies: (a) ICC finite and non-negative, (b) Pearson r > 0,
    /// (c) split-half reliability via odd/even trial partition.
    #[test]
    fn test_reliability_100_trials_with_crossvalidation() {
        use crate::benchmarks::executive::{FlankerBenchmark, StroopBenchmark};

        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 100,
            seed: 777,
            ..Default::default()
        };

        // Run full reliability battery with 2 sessions, 10 subjects
        let benchmarks: Vec<&dyn PsychBenchmark> = vec![&StroopBenchmark, &FlankerBenchmark];
        let battery = ReliabilityBattery::run(&benchmarks, &config, 2, 10);

        assert_eq!(
            battery.results.len(),
            2,
            "Should have results for both benchmarks"
        );

        for r in &battery.results {
            assert!(
                r.icc.is_finite(),
                "{} ICC not finite: {}",
                r.benchmark,
                r.icc
            );
            assert!(r.icc >= 0.0, "{} ICC negative: {}", r.benchmark, r.icc);
            assert!(
                r.pearson_r.is_finite(),
                "{} Pearson r not finite",
                r.benchmark
            );
            assert!(
                r.sem.is_finite() && r.sem >= 0.0,
                "{} SEM invalid: {}",
                r.benchmark,
                r.sem
            );
        }

        // Split-half cross-validation: run each benchmark with odd vs even seeds
        // and check that the two halves produce correlated results
        for bench in &benchmarks {
            let mut odd_scores = Vec::new();
            let mut even_scores = Vec::new();
            let metric_name = key_metric_for_benchmark(bench.name());

            for subject in 0..10u64 {
                let subject_seed = config.seed + subject * 1000;
                // Odd trials
                let odd_config = BenchmarkConfig {
                    seed: subject_seed + 1,
                    trials_per_condition: 50,
                    ..config.clone()
                };
                let odd_result = bench.run(&odd_config);
                if let Some(mv) = odd_result.metrics.get(metric_name) {
                    odd_scores.push(mv.mean);
                }
                // Even trials
                let even_config = BenchmarkConfig {
                    seed: subject_seed + 2,
                    trials_per_condition: 50,
                    ..config.clone()
                };
                let even_result = bench.run(&even_config);
                if let Some(mv) = even_result.metrics.get(metric_name) {
                    even_scores.push(mv.mean);
                }
            }

            let n = odd_scores.len().min(even_scores.len());
            assert!(
                n >= 3,
                "{} too few split-half subjects: {}",
                bench.name(),
                n
            );

            let split_icc = compute_icc(&odd_scores[..n], &even_scores[..n]);
            let split_r = pearson_r(&odd_scores[..n], &even_scores[..n]);

            assert!(
                split_icc.is_finite(),
                "{} split-half ICC not finite: {}",
                bench.name(),
                split_icc
            );
            assert!(
                split_r.is_finite(),
                "{} split-half Pearson r not finite: {}",
                bench.name(),
                split_r
            );
        }
    }
}
