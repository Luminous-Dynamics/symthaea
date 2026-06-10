// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Speed-Accuracy Tradeoff (SAT) curve analysis.
//!
//! Runs a benchmark at multiple time-pressure levels, fits the exponential SAT
//! function `d'(t) = lambda * (1 - e^{-beta * (t - delta)})` for `t > delta`
//! (0 otherwise), and evaluates whether the resulting curve is human-like.
//!
//! Reference: Wickelgren (1977), "Speed-accuracy tradeoff and information
//! processing dynamics", Acta Psychologica, 41(1), 67-85.

use serde::{Deserialize, Serialize};

use super::PsychBenchmark;
use super::config::BenchmarkConfig;
use super::report::BenchmarkResult;

/// A single point on the SAT curve: one (time_pressure, accuracy, mean_rt) triple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatPoint {
    /// Time pressure level (0.0 = none, 1.0 = maximum).
    pub time_pressure: f64,
    /// Accuracy at this pressure level (0.0 – 1.0).
    pub accuracy: f64,
    /// Mean reaction time in ticks at this pressure level.
    pub mean_rt: f64,
}

/// Fit parameters for the exponential SAT function:
/// `d'(t) = lambda * (1 - e^{-beta * (t - delta)})` for `t > delta`, else 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatFit {
    /// Asymptotic accuracy (lambda).
    pub asymptote: f64,
    /// Rate of approach to asymptote (beta).
    pub rate: f64,
    /// Processing-time intercept (delta): onset of above-chance performance.
    pub intercept: f64,
    /// Coefficient of determination (R^2) for the fit.
    pub r_squared: f64,
}

/// Full SAT curve for one benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatCurve {
    /// Benchmark name.
    pub benchmark: String,
    /// Measured SAT points (one per time-pressure level).
    pub points: Vec<SatPoint>,
    /// Fitted SAT function parameters.
    pub fit: SatFit,
    /// Whether the curve exhibits human-like SAT behavior.
    pub human_like: bool,
}

impl SatCurve {
    /// Check whether this curve is human-like.
    ///
    /// Human-like criteria:
    /// (a) accuracy decreases as time pressure increases (monotonic tendency),
    /// (b) RT decreases as time pressure increases (monotonic tendency),
    /// (c) the fit has R^2 > 0.7.
    pub fn is_human_like(&self) -> bool {
        if self.points.len() < 2 {
            return false;
        }

        // (a) Accuracy should generally decrease with increasing pressure.
        // We check that the first point has higher accuracy than the last.
        let first = &self.points[0];
        let last = &self.points[self.points.len() - 1];
        let accuracy_decreases = first.accuracy >= last.accuracy;

        // (b) RT should generally decrease with increasing pressure.
        let rt_decreases = first.mean_rt >= last.mean_rt;

        // (c) Good fit quality.
        let good_fit = self.fit.r_squared > 0.7;

        accuracy_decreases && rt_decreases && good_fit
    }

    /// Render the SAT curve as a markdown table with an ASCII chart.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!("## SAT Curve: {}\n\n", self.benchmark));

        // Fit summary
        out.push_str(&format!(
            "**Fit**: lambda={:.3}, beta={:.3}, delta={:.3}, R^2={:.3}\n",
            self.fit.asymptote, self.fit.rate, self.fit.intercept, self.fit.r_squared
        ));
        out.push_str(&format!(
            "**Human-like**: {}\n\n",
            if self.human_like { "Yes" } else { "No" }
        ));

        // Data table
        out.push_str("| Pressure | Accuracy | Mean RT |\n");
        out.push_str("|----------|----------|---------|\n");
        for p in &self.points {
            out.push_str(&format!(
                "| {:.2}     | {:.3}    | {:.2}   |\n",
                p.time_pressure, p.accuracy, p.mean_rt
            ));
        }
        out.push('\n');

        // ASCII chart: accuracy vs pressure
        let chart_width = 40;
        let chart_height = 10;
        out.push_str("```\n");
        out.push_str("Accuracy vs. Time Pressure\n");
        out.push_str("1.0 |\n");

        // Build a simple ASCII plot
        for row in (0..chart_height).rev() {
            let y_threshold = row as f64 / chart_height as f64;
            let mut line = format!("{:.1} |", y_threshold);
            for col in 0..chart_width {
                let x = col as f64 / (chart_width - 1) as f64;
                // Find nearest point
                let nearest = self.points.iter().min_by(|a, b| {
                    let da = (a.time_pressure - x).abs();
                    let db = (b.time_pressure - x).abs();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
                if let Some(pt) = nearest {
                    let dist = (pt.time_pressure - x).abs();
                    if dist < 0.5 / chart_width as f64
                        && pt.accuracy >= y_threshold
                        && pt.accuracy < y_threshold + 1.0 / chart_height as f64
                    {
                        line.push('*');
                    } else {
                        line.push(' ');
                    }
                } else {
                    line.push(' ');
                }
            }
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str("    +");
        out.push_str(&"-".repeat(chart_width));
        out.push_str("> pressure\n");
        out.push_str("```\n");

        out
    }
}

/// Battery of SAT curves across multiple benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatBattery {
    /// Individual SAT curves.
    pub curves: Vec<SatCurve>,
}

impl SatBattery {
    /// Run SAT analysis on multiple benchmarks.
    pub fn run(
        benchmarks: &[&dyn PsychBenchmark],
        base_config: &BenchmarkConfig,
        pressures: &[f64],
    ) -> Self {
        let curves = benchmarks
            .iter()
            .map(|bench| run_sat_curve(*bench, base_config, pressures))
            .collect();
        Self { curves }
    }

    /// Render the full battery as markdown.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# SAT Battery Report\n\n");

        // Summary table
        out.push_str("| Benchmark | Asymptote | Rate | Intercept | R^2 | Human-like |\n");
        out.push_str("|-----------|-----------|------|-----------|-----|------------|\n");
        for c in &self.curves {
            out.push_str(&format!(
                "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {} |\n",
                c.benchmark,
                c.fit.asymptote,
                c.fit.rate,
                c.fit.intercept,
                c.fit.r_squared,
                if c.human_like { "Yes" } else { "No" }
            ));
        }
        out.push('\n');

        let n_human = self.curves.iter().filter(|c| c.human_like).count();
        out.push_str(&format!(
            "**{}/{} benchmarks show human-like SAT curves.**\n\n",
            n_human,
            self.curves.len()
        ));

        // Individual curves
        for c in &self.curves {
            out.push_str(&c.to_markdown());
            out.push('\n');
        }

        out
    }
}

/// Run a benchmark at multiple time-pressure levels to produce an SAT curve.
pub fn run_sat_curve(
    bench: &dyn PsychBenchmark,
    base_config: &BenchmarkConfig,
    pressures: &[f64],
) -> SatCurve {
    let benchmark_name = bench.name().to_string();

    let mut points = Vec::with_capacity(pressures.len());
    for &pressure in pressures {
        let config = BenchmarkConfig {
            time_pressure: pressure.clamp(0.0, 1.0),
            ..base_config.clone()
        };
        let result = bench.run(&config);
        let (accuracy, mean_rt) = extract_accuracy_and_rt(&result, &benchmark_name);
        points.push(SatPoint {
            time_pressure: pressure,
            accuracy,
            mean_rt,
        });
    }

    let fit = fit_sat(&points);
    let mut curve = SatCurve {
        benchmark: benchmark_name,
        points,
        fit,
        human_like: false,
    };
    curve.human_like = curve.is_human_like();
    curve
}

/// Extract the key accuracy metric and mean RT from a benchmark result.
///
/// Reuses the key-metric logic from `cognitive_profile.rs`.
pub fn extract_accuracy_and_rt(result: &BenchmarkResult, benchmark_name: &str) -> (f64, f64) {
    let accuracy = extract_key_accuracy(result, benchmark_name);
    let mean_rt = extract_mean_rt(result, benchmark_name);
    (accuracy, mean_rt)
}

/// Extract accuracy from a result using benchmark-specific key metrics.
fn extract_key_accuracy(result: &BenchmarkResult, benchmark_name: &str) -> f64 {
    // Benchmark-specific accuracy keys (mirrors cognitive_profile::key_metric)
    let keys: Vec<&str> = match benchmark_name {
        name if name.contains("Stroop") && name.contains("Emotional") => {
            vec!["emotional_interference", "incongruent_accuracy"]
        }
        name if name.contains("Stroop") => vec!["incongruent_accuracy", "congruent_accuracy"],
        name if name.contains("Flanker") => vec!["accuracy"],
        name if name.contains("WCST") => vec!["accuracy"],
        name if name.contains("GoNoGo") => vec!["overall_accuracy", "go_accuracy"],
        name if name.contains("N-back") => vec!["hit_rate"],
        name if name.contains("VisualSearch") => vec!["accuracy"],
        name if name.contains("FittsLaw") => vec!["fitts_r_squared", "accuracy"],
        name if name.contains("Arc") => vec!["transfer_accuracy", "accuracy"],
        _ => vec!["accuracy", "overall_accuracy", "word_accuracy", "hit_rate"],
    };

    for key in keys {
        if let Some(mv) = result.metrics.get(key) {
            return mv.mean;
        }
    }

    // Fallback: average of accuracy-like metrics
    let acc_metrics: Vec<f64> = result
        .metrics
        .iter()
        .filter(|(k, _)| k.contains("accuracy") || k.contains("hit_rate"))
        .map(|(_, v)| v.mean)
        .collect();

    if !acc_metrics.is_empty() {
        acc_metrics.iter().sum::<f64>() / acc_metrics.len() as f64
    } else {
        0.5 // fallback: chance
    }
}

/// Extract mean RT from a result, trying benchmark-specific keys then fallbacks.
fn extract_mean_rt(result: &BenchmarkResult, benchmark_name: &str) -> f64 {
    // Try benchmark-specific RT keys
    let rt_keys: Vec<&str> = match benchmark_name {
        name if name.contains("Stroop") => vec![
            "incongruent::rt_ticks",
            "congruent::rt_ticks",
            "neutral::rt_ticks",
        ],
        name if name.contains("Flanker") => vec![
            "incongruent::rt_ticks",
            "congruent::rt_ticks",
            "neutral::rt_ticks",
        ],
        _ => vec!["rt_ticks", "mean_rt_ticks"],
    };

    for key in &rt_keys {
        if let Some(mv) = result.metrics.get(*key) {
            return mv.mean;
        }
    }

    // Fallback: average of all RT-like metrics
    let rt_metrics: Vec<f64> = result
        .metrics
        .iter()
        .filter(|(k, _)| k.contains("rt_ticks") || k.contains("mean_rt"))
        .map(|(_, v)| v.mean)
        .collect();

    if !rt_metrics.is_empty() {
        rt_metrics.iter().sum::<f64>() / rt_metrics.len() as f64
    } else {
        5.0 // fallback: moderate RT
    }
}

/// Fit the exponential SAT function to measured points using grid search.
///
/// The SAT function is: `d'(t) = lambda * (1 - e^{-beta * (t - delta)})` for
/// `t > delta`, else 0. Here `t` is derived from `mean_rt` (processing time)
/// and we fit lambda (asymptote), beta (rate), and delta (intercept).
///
/// Uses a coarse-to-fine grid search over parameter space, minimizing
/// sum-of-squared residuals.
pub fn fit_sat(points: &[SatPoint]) -> SatFit {
    if points.is_empty() {
        return SatFit {
            asymptote: 0.0,
            rate: 1.0,
            intercept: 0.0,
            r_squared: 0.0,
        };
    }

    // Use mean_rt as the time variable (processing time).
    // Higher RT = more processing time = higher accuracy expected.
    let max_accuracy = points
        .iter()
        .map(|p| p.accuracy)
        .fold(f64::NEG_INFINITY, f64::max);

    let min_rt = points
        .iter()
        .map(|p| p.mean_rt)
        .fold(f64::INFINITY, f64::min);

    let max_rt = points
        .iter()
        .map(|p| p.mean_rt)
        .fold(f64::NEG_INFINITY, f64::max);

    // Mean accuracy for R^2 computation
    let mean_acc = points.iter().map(|p| p.accuracy).sum::<f64>() / points.len() as f64;
    let ss_tot = points
        .iter()
        .map(|p| (p.accuracy - mean_acc).powi(2))
        .sum::<f64>();

    // Grid search ranges
    let lambda_range = if max_accuracy > 0.0 {
        (max_accuracy * 0.5, max_accuracy * 1.5)
    } else {
        (0.1, 1.0)
    };
    let beta_range = (0.01, 5.0);
    let delta_range = if min_rt > 0.0 {
        (0.0, min_rt)
    } else {
        (0.0, max_rt * 0.5)
    };

    let grid_steps = 20;
    let mut best_lambda = lambda_range.0;
    let mut best_beta = beta_range.0;
    let mut best_delta = delta_range.0;
    let mut best_sse = f64::MAX;

    // Coarse grid search
    for li in 0..grid_steps {
        let lambda =
            lambda_range.0 + (lambda_range.1 - lambda_range.0) * li as f64 / grid_steps as f64;
        for bi in 0..grid_steps {
            let beta = beta_range.0 + (beta_range.1 - beta_range.0) * bi as f64 / grid_steps as f64;
            for di in 0..grid_steps {
                let delta =
                    delta_range.0 + (delta_range.1 - delta_range.0) * di as f64 / grid_steps as f64;

                let sse = compute_sse(points, lambda, beta, delta);
                if sse < best_sse {
                    best_sse = sse;
                    best_lambda = lambda;
                    best_beta = beta;
                    best_delta = delta;
                }
            }
        }
    }

    // Fine grid search around the best coarse solution
    let fine_steps = 10;
    let lambda_step = (lambda_range.1 - lambda_range.0) / grid_steps as f64;
    let beta_step = (beta_range.1 - beta_range.0) / grid_steps as f64;
    let delta_step = (delta_range.1 - delta_range.0) / grid_steps as f64;

    let fine_lambda_range = (
        (best_lambda - lambda_step).max(0.001),
        best_lambda + lambda_step,
    );
    let fine_beta_range = ((best_beta - beta_step).max(0.001), best_beta + beta_step);
    let fine_delta_range = ((best_delta - delta_step).max(0.0), best_delta + delta_step);

    for li in 0..fine_steps {
        let lambda = fine_lambda_range.0
            + (fine_lambda_range.1 - fine_lambda_range.0) * li as f64 / fine_steps as f64;
        for bi in 0..fine_steps {
            let beta = fine_beta_range.0
                + (fine_beta_range.1 - fine_beta_range.0) * bi as f64 / fine_steps as f64;
            for di in 0..fine_steps {
                let delta = fine_delta_range.0
                    + (fine_delta_range.1 - fine_delta_range.0) * di as f64 / fine_steps as f64;

                let sse = compute_sse(points, lambda, beta, delta);
                if sse < best_sse {
                    best_sse = sse;
                    best_lambda = lambda;
                    best_beta = beta;
                    best_delta = delta;
                }
            }
        }
    }

    let r_squared = if ss_tot > 1e-15 {
        (1.0 - best_sse / ss_tot).clamp(0.0, 1.0)
    } else {
        // All accuracies are (nearly) equal; any constant model is perfect.
        if best_sse < 1e-15 { 1.0 } else { 0.0 }
    };

    SatFit {
        asymptote: best_lambda,
        rate: best_beta,
        intercept: best_delta,
        r_squared,
    }
}

/// Compute sum of squared errors for given SAT parameters.
fn compute_sse(points: &[SatPoint], lambda: f64, beta: f64, delta: f64) -> f64 {
    points
        .iter()
        .map(|p| {
            let predicted = sat_function(p.mean_rt, lambda, beta, delta);
            (p.accuracy - predicted).powi(2)
        })
        .sum()
}

/// Evaluate the SAT function at time `t`.
///
/// `d'(t) = lambda * (1 - e^{-beta * (t - delta)})` for `t > delta`, else 0.
fn sat_function(t: f64, lambda: f64, beta: f64, delta: f64) -> f64 {
    if t <= delta {
        0.0
    } else {
        lambda * (1.0 - (-beta * (t - delta)).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::executive::{FlankerBenchmark, StroopBenchmark};
    use crate::harness::report::MetricValue;

    #[test]
    fn test_sat_point_construction() {
        let pt = SatPoint {
            time_pressure: 0.5,
            accuracy: 0.85,
            mean_rt: 6.0,
        };
        assert!((pt.time_pressure - 0.5).abs() < 1e-10);
        assert!((pt.accuracy - 0.85).abs() < 1e-10);
        assert!((pt.mean_rt - 6.0).abs() < 1e-10);

        // Serde round-trip
        let json = serde_json::to_string(&pt).unwrap();
        let loaded: SatPoint = serde_json::from_str(&json).unwrap();
        assert!((loaded.time_pressure - pt.time_pressure).abs() < 1e-10);
        assert!((loaded.accuracy - pt.accuracy).abs() < 1e-10);
        assert!((loaded.mean_rt - pt.mean_rt).abs() < 1e-10);
    }

    #[test]
    fn test_sat_curve_on_stroop() {
        let base_config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 5,
            ..Default::default()
        };
        let pressures = [0.0, 0.5, 1.0];

        let curve = run_sat_curve(&StroopBenchmark, &base_config, &pressures);

        assert_eq!(curve.benchmark, "Executive::Stroop");
        assert_eq!(curve.points.len(), 3);
        assert!(curve.fit.asymptote.is_finite());
        assert!(curve.fit.rate.is_finite());
        assert!(curve.fit.intercept.is_finite());
        assert!(curve.fit.r_squared >= 0.0 && curve.fit.r_squared <= 1.0);

        // Points should have valid accuracy and RT
        for p in &curve.points {
            assert!(p.accuracy.is_finite(), "accuracy should be finite");
            assert!(p.mean_rt.is_finite(), "mean_rt should be finite");
        }
    }

    #[test]
    fn test_sat_fit_r_squared_bounded() {
        // Construct well-ordered points: higher RT -> higher accuracy
        let points = vec![
            SatPoint {
                time_pressure: 1.0,
                accuracy: 0.50,
                mean_rt: 3.0,
            },
            SatPoint {
                time_pressure: 0.5,
                accuracy: 0.75,
                mean_rt: 6.0,
            },
            SatPoint {
                time_pressure: 0.0,
                accuracy: 0.90,
                mean_rt: 10.0,
            },
        ];

        let fit = fit_sat(&points);
        assert!(
            fit.r_squared >= 0.0 && fit.r_squared <= 1.0,
            "R^2 should be in [0, 1], got {}",
            fit.r_squared
        );
        assert!(fit.asymptote.is_finite());
        assert!(fit.rate.is_finite());
        assert!(fit.intercept.is_finite());
    }

    #[test]
    fn test_sat_human_like_check() {
        // Human-like: accuracy and RT both decrease with pressure, good fit
        let human_curve = SatCurve {
            benchmark: "Test".to_string(),
            points: vec![
                SatPoint {
                    time_pressure: 0.0,
                    accuracy: 0.95,
                    mean_rt: 10.0,
                },
                SatPoint {
                    time_pressure: 0.5,
                    accuracy: 0.80,
                    mean_rt: 6.0,
                },
                SatPoint {
                    time_pressure: 1.0,
                    accuracy: 0.60,
                    mean_rt: 3.0,
                },
            ],
            fit: SatFit {
                asymptote: 0.95,
                rate: 0.5,
                intercept: 1.0,
                r_squared: 0.85,
            },
            human_like: false,
        };
        assert!(human_curve.is_human_like());

        // Non-human: accuracy increases with pressure (paradoxical)
        let bad_curve = SatCurve {
            benchmark: "Bad".to_string(),
            points: vec![
                SatPoint {
                    time_pressure: 0.0,
                    accuracy: 0.50,
                    mean_rt: 10.0,
                },
                SatPoint {
                    time_pressure: 1.0,
                    accuracy: 0.90,
                    mean_rt: 3.0,
                },
            ],
            fit: SatFit {
                asymptote: 0.9,
                rate: 0.5,
                intercept: 1.0,
                r_squared: 0.9,
            },
            human_like: false,
        };
        assert!(!bad_curve.is_human_like());

        // Non-human: poor fit
        let poor_fit = SatCurve {
            benchmark: "Poor".to_string(),
            points: vec![
                SatPoint {
                    time_pressure: 0.0,
                    accuracy: 0.95,
                    mean_rt: 10.0,
                },
                SatPoint {
                    time_pressure: 1.0,
                    accuracy: 0.60,
                    mean_rt: 3.0,
                },
            ],
            fit: SatFit {
                asymptote: 0.95,
                rate: 0.5,
                intercept: 1.0,
                r_squared: 0.3, // below 0.7 threshold
            },
            human_like: false,
        };
        assert!(!poor_fit.is_human_like());
    }

    #[test]
    fn test_extract_accuracy_stroop() {
        let mut result = BenchmarkResult::new("Executive::Stroop", None);
        result.insert(
            "incongruent_accuracy",
            MetricValue::from_samples(&[0.75, 0.80, 0.78]),
        );
        result.insert(
            "congruent_accuracy",
            MetricValue::from_samples(&[0.90, 0.92]),
        );
        result.insert(
            "incongruent::rt_ticks",
            MetricValue::from_samples(&[8.0, 9.0, 7.5]),
        );

        let (accuracy, mean_rt) = extract_accuracy_and_rt(&result, "Executive::Stroop");

        // Should pick incongruent_accuracy as key metric
        let expected_acc = (0.75 + 0.80 + 0.78) / 3.0;
        assert!(
            (accuracy - expected_acc).abs() < 1e-6,
            "expected accuracy ~{:.4}, got {:.4}",
            expected_acc,
            accuracy
        );

        // Should pick incongruent::rt_ticks
        let expected_rt = (8.0 + 9.0 + 7.5) / 3.0;
        assert!(
            (mean_rt - expected_rt).abs() < 1e-6,
            "expected rt ~{:.4}, got {:.4}",
            expected_rt,
            mean_rt
        );
    }

    #[test]
    fn test_sat_markdown_output() {
        let curve = SatCurve {
            benchmark: "Executive::Stroop".to_string(),
            points: vec![
                SatPoint {
                    time_pressure: 0.0,
                    accuracy: 0.90,
                    mean_rt: 9.0,
                },
                SatPoint {
                    time_pressure: 0.5,
                    accuracy: 0.75,
                    mean_rt: 6.0,
                },
                SatPoint {
                    time_pressure: 1.0,
                    accuracy: 0.55,
                    mean_rt: 3.5,
                },
            ],
            fit: SatFit {
                asymptote: 0.92,
                rate: 0.4,
                intercept: 1.5,
                r_squared: 0.88,
            },
            human_like: true,
        };

        let md = curve.to_markdown();
        assert!(md.contains("SAT Curve: Executive::Stroop"));
        assert!(md.contains("lambda=0.920"));
        assert!(md.contains("R^2=0.880"));
        assert!(md.contains("**Human-like**: Yes"));
        assert!(md.contains("| Pressure | Accuracy | Mean RT |"));
        assert!(md.contains("| 0.00"));
        assert!(md.contains("| 0.50"));
        assert!(md.contains("| 1.00"));
    }

    #[test]
    fn test_sat_battery_multiple() {
        let base_config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 5,
            ..Default::default()
        };
        let pressures = [0.0, 0.5, 1.0];
        let benchmarks: Vec<&dyn PsychBenchmark> = vec![&StroopBenchmark, &FlankerBenchmark];

        let battery = SatBattery::run(&benchmarks, &base_config, &pressures);

        assert_eq!(battery.curves.len(), 2);
        assert_eq!(battery.curves[0].benchmark, "Executive::Stroop");
        assert_eq!(battery.curves[1].benchmark, "Executive::Flanker");

        // Each curve should have 3 points
        for curve in &battery.curves {
            assert_eq!(curve.points.len(), 3);
        }

        // Markdown should contain both benchmarks
        let md = battery.to_markdown();
        assert!(md.contains("SAT Battery Report"));
        assert!(md.contains("Executive::Stroop"));
        assert!(md.contains("Executive::Flanker"));
    }

    #[test]
    fn test_sat_monotonic_pressure_reduces_rt() {
        let base_config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 10,
            seed: 42,
            ..Default::default()
        };
        let pressures = [0.0, 0.25, 0.5, 0.75, 1.0];

        let curve = run_sat_curve(&StroopBenchmark, &base_config, &pressures);

        assert_eq!(curve.points.len(), 5);

        // RT at pressure=0.0 should generally be >= RT at pressure=1.0
        // (more time pressure => faster but less accurate responses)
        let rt_no_pressure = curve.points[0].mean_rt;
        let rt_max_pressure = curve.points[curve.points.len() - 1].mean_rt;
        // Allow up to 0.5 ticks of tolerance for stochastic softmax sampling noise
        assert!(
            rt_no_pressure >= rt_max_pressure - 0.5,
            "RT should generally decrease with pressure: no_pressure={:.2} vs max_pressure={:.2}",
            rt_no_pressure,
            rt_max_pressure
        );
    }

    #[test]
    fn test_sat_fit_empty_points() {
        let fit = fit_sat(&[]);
        assert!((fit.asymptote).abs() < 1e-10);
        assert!((fit.r_squared).abs() < 1e-10);
    }

    #[test]
    fn test_sat_function_evaluation() {
        // Below intercept: should return 0
        let val = sat_function(1.0, 0.9, 0.5, 2.0);
        assert!((val).abs() < 1e-10, "below delta should be 0");

        // At intercept: should return 0
        let val = sat_function(2.0, 0.9, 0.5, 2.0);
        assert!((val).abs() < 1e-10, "at delta should be 0");

        // Well above intercept: should approach asymptote
        let val = sat_function(100.0, 0.9, 0.5, 2.0);
        assert!(
            (val - 0.9).abs() < 0.01,
            "well above delta should approach asymptote, got {}",
            val
        );

        // Just above intercept: should be small positive
        let val = sat_function(2.5, 0.9, 0.5, 2.0);
        assert!(val > 0.0 && val < 0.9, "just above delta: {}", val);
    }

    #[test]
    fn test_sat_serde_roundtrip() {
        let curve = SatCurve {
            benchmark: "Test".to_string(),
            points: vec![SatPoint {
                time_pressure: 0.5,
                accuracy: 0.8,
                mean_rt: 5.0,
            }],
            fit: SatFit {
                asymptote: 0.9,
                rate: 0.5,
                intercept: 1.0,
                r_squared: 0.85,
            },
            human_like: true,
        };

        let json = serde_json::to_string(&curve).unwrap();
        let loaded: SatCurve = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.benchmark, "Test");
        assert_eq!(loaded.points.len(), 1);
        assert!((loaded.fit.r_squared - 0.85).abs() < 1e-10);
        assert!(loaded.human_like);
    }
}
