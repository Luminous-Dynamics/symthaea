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

use super::config::BenchmarkConfig;
use super::report::BenchmarkResult;
use super::PsychBenchmark;

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

// ──── Key metric lookup ────

/// Return the key metric name for a benchmark (same logic as cognitive_profile).
pub fn key_metric_for(benchmark_name: &str) -> &'static str {
    match benchmark_name {
        name if name.contains("Stroop") && name.contains("Emotional") => "emotional_interference",
        name if name.contains("Stroop") => "incongruent_accuracy",
        name if name.contains("WCST") => "accuracy",
        name if name.contains("Flanker") => "incongruent_accuracy",
        name if name.contains("TowerOfLondon") => "accuracy",
        name if name.contains("DualTask") => "dual_cost",
        name if name.contains("Ravens") => "accuracy",
        name if name.contains("IowaGambling") => "net_score",
        name if name.contains("N-back") => "hit_rate",
        name if name.contains("ChangeDetection") => "accuracy",
        name if name.contains("SerialRecall") => "accuracy",
        name if name.contains("SpatialUpdating") => "accuracy",
        name if name.contains("Binding") => "accuracy",
        name if name.contains("DigitSpan") => "max_span",
        name if name.contains("PVT") => "mean_rt_ticks",
        name if name.contains("CPT") => "hit_rate",
        name if name.contains("SART") => "accuracy",
        name if name.contains("FittsLaw") => "fitts_r_squared",
        name if name.contains("Bimanual") => "accuracy",
        name if name.contains("SRTT") => "accuracy",
        name if name.contains("GoNoGo") => "overall_accuracy",
        name if name.contains("StopSignal") => "accuracy",
        name if name.contains("VisualSearch") => "accuracy",
        name if name.contains("AttentionalBlink") => "t1_accuracy",
        name if name.contains("RME") => "accuracy",
        name if name.contains("SocialNorm") => "accuracy",
        name if name.contains("UltimatumGame") => "acceptance_rate",
        name if name.contains("FalseBelief") => "accuracy",
        name if name.contains("FauxPas") => "accuracy",
        name if name.contains("Persuasion") => "accuracy",
        name if name.contains("StrangeStory") => "accuracy",
        name if name.contains("Hinting") => "accuracy",
        name if name.contains("LexicalDecision") => "word_accuracy",
        name if name.contains("SemanticPriming") => "priming_effect",
        name if name.contains("SemanticCoherence") => "accuracy",
        name if name.contains("GardenPath") => "accuracy",
        name if name.contains("Calibration") => "calibration_ece",
        name if name.contains("FOK") => "gamma",
        name if name.contains("Arc") => "transfer_accuracy",
        name if name.contains("AccurateRetrieval") => "accuracy",
        name if name.contains("LongRange") => "accuracy",
        name if name.contains("ProspectiveMemory") => "pm_accuracy",
        name if name.contains("ConflictResolution") => "accuracy",
        name if name.contains("TestTimeLearning") => "accuracy",
        _ => "accuracy",
    }
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
    /// For each benchmark, runs `n_sessions` sessions with different seeds
    /// (base seed + 0, base seed + 1, ..., base seed + n_sessions - 1).
    /// Computes ICC and practice effects between consecutive session pairs.
    pub fn run(
        benchmarks: &[&dyn PsychBenchmark],
        base_config: &BenchmarkConfig,
        n_sessions: usize,
    ) -> Self {
        let n_sessions = n_sessions.max(2);
        let mut results = Vec::new();

        for bench in benchmarks {
            let bench_name = bench.name();
            let metric_name = key_metric_for(bench_name);

            // Run n_sessions with different seeds
            let mut session_results: Vec<BenchmarkResult> = Vec::with_capacity(n_sessions);
            for s in 0..n_sessions {
                let config = BenchmarkConfig {
                    seed: base_config.seed + s as u64,
                    ..base_config.clone()
                };
                session_results.push(bench.run(&config));
            }

            // Extract key metric means from each session
            let session_means: Vec<f64> = session_results
                .iter()
                .filter_map(|r| r.metrics.get(metric_name).map(|mv| mv.mean))
                .collect();

            if session_means.len() < 2 {
                continue;
            }

            // Compute ICC and Pearson r between consecutive session pairs.
            // For n_sessions > 2, use all consecutive pairs as "subjects".
            let mut s1_vals = Vec::new();
            let mut s2_vals = Vec::new();
            for pair in session_means.windows(2) {
                s1_vals.push(pair[0]);
                s2_vals.push(pair[1]);
            }

            let icc = compute_icc(&s1_vals, &s2_vals);
            let r = pearson_r(&s1_vals, &s2_vals);

            // Pool SD across all sessions for SEM
            let all_mean = session_means.iter().sum::<f64>() / session_means.len() as f64;
            let sd = if session_means.len() > 1 {
                let var = session_means
                    .iter()
                    .map(|x| (x - all_mean).powi(2))
                    .sum::<f64>()
                    / (session_means.len() - 1) as f64;
                var.sqrt()
            } else {
                0.0
            };
            let sem = compute_sem(sd, icc);

            // Practice effect between first and last session
            let practice = PracticeEffect::compute(
                bench_name,
                metric_name,
                session_means[0],
                session_means[session_means.len() - 1],
            );

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
        assert_eq!(ReliabilityClass::from_icc(0.95), ReliabilityClass::Excellent);
        assert_eq!(ReliabilityClass::from_icc(0.91), ReliabilityClass::Excellent);
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

        let battery = ReliabilityBattery::run(&[&StroopBenchmark], &config, 2);
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
        let battery = ReliabilityBattery::run(&benchmarks, &config, 3);

        assert_eq!(battery.results.len(), 2);

        // Both benchmarks should have results
        let names: Vec<&str> = battery.results.iter().map(|r| r.benchmark.as_str()).collect();
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
        assert_eq!(key_metric_for("Executive::Stroop"), "incongruent_accuracy");
        assert_eq!(key_metric_for("Executive::WCST"), "accuracy");
        assert_eq!(key_metric_for("WorM::N-back"), "hit_rate");
        assert_eq!(key_metric_for("Reasoning::ArcFluid"), "transfer_accuracy");
        assert_eq!(key_metric_for("Metacognition::FOK"), "gamma");
        assert_eq!(key_metric_for("Unknown::Bench"), "accuracy");
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
}
