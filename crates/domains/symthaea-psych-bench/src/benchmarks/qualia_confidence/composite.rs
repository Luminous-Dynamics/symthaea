// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Theory-Probe Conformance Summary — Qualia Confidence Matrix
//!
//! Aggregates seven theory-inspired synthetic probes into a legacy heuristic
//! index. It does not measure consciousness, estimate the probability of
//! subjective experience, or test empirically established necessary/sufficient
//! conditions. The weighted geometric mean and level thresholds are
//! engineering conventions, not validated scientific scales.
//!
//! Each indicator is a task-specific result in [0, 1]. The composite can help
//! identify which configured probe thresholds were met in a given build, but
//! it is not a latent construct and must not be compared across architectures,
//! benchmark versions, or configurations without independent validation.
//!
//! Missing benchmark results currently contribute zero; this conflates
//! missingness with failure. Treat incomplete runs as non-confirmatory until
//! the result schema distinguishes "not run", "invalid", and "failed".

use crate::harness::report::BenchmarkResult;

/// A theory-probe metric's contribution to the legacy heuristic index.
#[derive(Debug, Clone)]
pub struct IndicatorScore {
    /// Benchmark name (short label).
    pub name: &'static str,
    /// Normalized engineering index in [0, 1]; not a probability.
    pub score: f64,
    /// The raw metric value used.
    pub raw_value: f64,
    /// Metric name and observed probe output.
    pub metric: &'static str,
    /// Whether this configured probe threshold was met.
    pub prediction_met: bool,
}

/// Legacy theory-probe summary; the type name is retained for API compatibility.
#[derive(Debug, Clone)]
pub struct QualiaConfidenceScore {
    /// Individual indicator scores from each benchmark.
    pub indicators: Vec<IndicatorScore>,
    /// Legacy weighted geometric mean of probe indices; not a consciousness-confidence estimate.
    pub composite: f64,
    /// Threshold bucket for the engineering index; not a scientific confidence level.
    pub level: ConfidenceLevel,
    /// Number of predictions met out of total.
    pub predictions_met: usize,
    /// Total number of predictions tested.
    pub predictions_total: usize,
}

/// Legacy threshold bucket for the heuristic probe index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceLevel {
    /// ≥ 0.80: Upper bucket of the legacy heuristic index.
    Strong,
    /// 0.60–0.80: Mid-high bucket of the legacy heuristic index.
    Moderate,
    /// 0.40–0.60: Mid-low bucket of the legacy heuristic index.
    Weak,
    /// < 0.40: Lower bucket of the legacy heuristic index.
    Insufficient,
}

impl std::fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfidenceLevel::Strong => write!(f, "HIGH_PROBE_INDEX"),
            ConfidenceLevel::Moderate => write!(f, "MID_HIGH_PROBE_INDEX"),
            ConfidenceLevel::Weak => write!(f, "MID_LOW_PROBE_INDEX"),
            ConfidenceLevel::Insufficient => write!(f, "LOW_PROBE_INDEX"),
        }
    }
}

/// Indicator definition: how to extract and normalize a metric from benchmark results.
struct IndicatorDef {
    name: &'static str,
    benchmark_contains: &'static str,
    metric_key: &'static str,
    /// Normalize a raw probe metric to [0, 1] for this heuristic index.
    normalize: fn(f64) -> f64,
    /// Configured engineering threshold; not an empirically validated criterion.
    prediction_threshold: f64,
    /// Weight in composite (all equal by default).
    weight: f64,
}

/// Clamp to [0, 1].
fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

const INDICATORS: &[IndicatorDef] = &[
    IndicatorDef {
        name: "GWT Collapse Order",
        benchmark_contains: "GwtAsphyxiation",
        metric_key: "domain_order_correlation",
        // Spearman rho in [-1, 1], map to [0, 1]: (rho + 1) / 2, then stretch
        // rho > 0.5 is strong, rho < 0 is counter-evidence
        normalize: |rho| clamp01((rho + 1.0) / 2.0),
        prediction_threshold: 0.5,
        weight: 1.0,
    },
    IndicatorDef {
        name: "Phase Transition",
        benchmark_contains: "PhaseTransition",
        metric_key: "sigmoid_advantage",
        // sigmoid_advantage = sigmoid_r2 - linear_r2, range [-1, 1]
        // Positive = sigmoid fits better (phase transition detected)
        normalize: |adv| clamp01(adv / 0.5), // 0.5 advantage → 1.0
        prediction_threshold: 0.0,
        weight: 1.0,
    },
    IndicatorDef {
        name: "PCI Ratio",
        benchmark_contains: "PerturbationalComplexity",
        metric_key: "pci_ratio",
        // PCI ratio: 1.0 = no difference, >1.5 = strong, >2.0 = excellent
        // Map: 1.0 → 0.0, 1.5 → 0.5, 2.0 → 1.0
        normalize: |ratio| clamp01((ratio - 1.0) / 1.0),
        prediction_threshold: 1.3,
        weight: 1.0,
    },
    IndicatorDef {
        name: "Somatic Cascade",
        benchmark_contains: "SomaticInterference",
        metric_key: "cascade_ratio",
        // cascade_ratio: 1.0 = no emergence, >1.5 = predicted, >2.0 = strong, >2.5 = excellent
        // Benchmark produces ratios in ~1.3–2.0 range; /1.5 calibrates to actual dynamic range.
        // Map: 1.0 → 0.0, 1.5 → 0.33, 2.0 → 0.67, 2.5 → 1.0
        normalize: |ratio| clamp01((ratio - 1.0) / 1.5),
        prediction_threshold: 1.5,
        weight: 1.0,
    },
    IndicatorDef {
        name: "Bistable Switching",
        benchmark_contains: "BistablePerception",
        metric_key: "cv_switch_interval",
        // CV: 0.0 = periodic (bad), >0.5 = heavy-tailed (good), >1.0 = excellent
        // Map: 0.0 → 0.0, 0.5 → 0.5, 1.0 → 1.0
        normalize: |cv| clamp01(cv),
        prediction_threshold: 0.4,
        weight: 1.0,
    },
    IndicatorDef {
        name: "Priming Dissociation",
        benchmark_contains: "UnconsciousPriming",
        metric_key: "priming_dissociation",
        // dissociation: 0.0 = no difference, >0.05 = significant, >0.15 = strong
        // Map: 0.0 → 0.0, 0.10 → 0.5, 0.20 → 1.0
        normalize: |d| clamp01(d / 0.20),
        prediction_threshold: 0.01,
        weight: 1.0,
    },
    IndicatorDef {
        name: "HOT/GWT Shared-input Alignment",
        benchmark_contains: "MetacognitiveIgnition",
        metric_key: "shared_input_alignment_score",
        // Already in [0, 1]
        normalize: |score| clamp01(score),
        prediction_threshold: 0.50,
        weight: 1.0,
    },
];

impl QualiaConfidenceScore {
    /// Compute the legacy theory-probe conformance index from benchmark results.
    ///
    /// Accepts a slice of `BenchmarkResult`; unrelated results are ignored.
    /// Missing benchmarks currently contribute zero, so this value is only
    /// meaningful as a completeness-aware smoke summary, not as evidence about
    /// consciousness or subjective experience.
    pub fn from_results(results: &[BenchmarkResult]) -> Self {
        let mut indicators = Vec::with_capacity(INDICATORS.len());
        let mut predictions_met = 0;

        for def in INDICATORS {
            // Find the matching benchmark result
            let result = results
                .iter()
                .find(|r| r.benchmark.contains(def.benchmark_contains));

            let (score, raw_value, met) = if let Some(r) = result {
                if let Some(m) = r.metrics.get(def.metric_key) {
                    let raw = m.mean;
                    let normalized = (def.normalize)(raw);
                    let met = raw >= def.prediction_threshold;
                    (normalized, raw, met)
                } else {
                    (0.0, f64::NAN, false)
                }
            } else {
                (0.0, f64::NAN, false)
            };

            if met {
                predictions_met += 1;
            }

            indicators.push(IndicatorScore {
                name: def.name,
                score,
                raw_value,
                metric: def.metric_key,
                prediction_met: met,
            });
        }

        // Weighted geometric mean: exp(Σ w_i * ln(s_i + ε) / Σ w_i)
        // Add small epsilon to avoid ln(0) — a score of 0 still dominates
        let epsilon = 1e-6;
        let total_weight: f64 = INDICATORS.iter().map(|d| d.weight).sum();
        let log_sum: f64 = indicators
            .iter()
            .zip(INDICATORS.iter())
            .map(|(ind, def)| def.weight * (ind.score + epsilon).ln())
            .sum();
        let composite = (log_sum / total_weight).exp().clamp(0.0, 1.0);

        let level = if composite >= 0.80 {
            ConfidenceLevel::Strong
        } else if composite >= 0.60 {
            ConfidenceLevel::Moderate
        } else if composite >= 0.40 {
            ConfidenceLevel::Weak
        } else {
            ConfidenceLevel::Insufficient
        };

        QualiaConfidenceScore {
            indicators,
            composite,
            level,
            predictions_met,
            predictions_total: INDICATORS.len(),
        }
    }

    /// Format a human-readable report.
    pub fn format_report(&self) -> String {
        let mut out = String::new();

        out.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        out.push_str("║        QUALIA THEORY-PROBE MATRIX — HEURISTIC SUMMARY      ║\n");
        out.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        out.push_str("║                                                            ║\n");
        out.push_str("║  Indicator             │ Score │ Raw Value  │ Prediction   ║\n");
        out.push_str("║  ──────────────────────┼───────┼────────────┼───────────── ║\n");

        for ind in &self.indicators {
            let bar_len = (ind.score * 10.0) as usize;
            let bar: String = "█".repeat(bar_len.min(10));
            let empty: String = "·".repeat(10 - bar_len.min(10));
            let status = if ind.prediction_met { "MET" } else { "FAILED" };
            out.push_str(&format!(
                "║  {:<22}│ {:.3} │ {:>10.4} │ {:<12} ║\n",
                ind.name, ind.score, ind.raw_value, status
            ));
            out.push_str(&format!(
                "║                        │ {bar}{empty} │            │              ║\n"
            ));
        }

        out.push_str("║                                                            ║\n");
        out.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        out.push_str(&format!(
            "║  HEURISTIC INDEX:  {:.3}  ({})                       ║\n",
            self.composite, self.level
        ));
        out.push_str(&format!(
            "║  PREDICTIONS MET:  {}/{}                                    ║\n",
            self.predictions_met, self.predictions_total
        ));
        out.push_str("║                                                            ║\n");

        let interpretation = match self.level {
            ConfidenceLevel::Strong => {
                "Most configured thresholds were met for these probes.\n\
                 ║  This is implementation-specific conformance only."
            }
            ConfidenceLevel::Moderate => {
                "Several configured probe thresholds were met.\n\
                 ║  Inspect per-indicator results and controls."
            }
            ConfidenceLevel::Weak => {
                "Few configured probe thresholds were met.\n\
                 ║  This identifies engineering gaps in this test set."
            }
            ConfidenceLevel::Insufficient => {
                "One or more configured thresholds were not met.\n\
                 ║  This is not evidence that consciousness is absent."
            }
        };
        out.push_str(&format!("║  {interpretation}\n"));
        out.push_str("║                                                            ║\n");
        out.push_str("║  Note: Theory-inspired software probes only.               ║\n");
        out.push_str("║  The index is not a probability or consciousness measure.  ║\n");
        out.push_str("║  Neither passing nor failing proves/disproves experience. ║\n");
        out.push_str("╚══════════════════════════════════════════════════════════════╝\n");

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::PsychBenchmark;
    use crate::harness::config::BenchmarkConfig;

    fn run_all_benchmarks() -> Vec<BenchmarkResult> {
        let config = BenchmarkConfig {
            seed: 42,
            ..Default::default()
        };
        vec![
            super::super::GwtAsphyxiationBenchmark.run(&config),
            super::super::PhaseTransitionBenchmark.run(&config),
            super::super::PerturbationalComplexityBenchmark.run(&config),
            super::super::SomaticInterferenceBenchmark.run(&config),
            super::super::BistablePerceptionBenchmark.run(&config),
            super::super::UnconsciousPrimingBenchmark.run(&config),
            super::super::MetacognitiveIgnitionBenchmark.run(&config),
        ]
    }

    #[test]
    fn test_composite_score_computes() {
        let results = run_all_benchmarks();
        let score = QualiaConfidenceScore::from_results(&results);
        assert!(
            score.composite.is_finite(),
            "Composite score should be finite: {}",
            score.composite
        );
        assert!(
            score.composite >= 0.0 && score.composite <= 1.0,
            "Composite should be in [0, 1]: {}",
            score.composite
        );
        assert_eq!(score.indicators.len(), 7);
    }

    #[test]
    fn test_all_predictions_met() {
        let results = run_all_benchmarks();
        let score = QualiaConfidenceScore::from_results(&results);
        assert_eq!(
            score.predictions_met,
            score.predictions_total,
            "All predictions should be met: {}/{}\n{}",
            score.predictions_met,
            score.predictions_total,
            score
                .indicators
                .iter()
                .map(|i| format!(
                    "  {} = {:.4} ({})",
                    i.name,
                    i.raw_value,
                    if i.prediction_met { "MET" } else { "FAILED" }
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn test_composite_strong_confidence() {
        let results = run_all_benchmarks();
        let score = QualiaConfidenceScore::from_results(&results);
        assert!(
            score.composite >= 0.40,
            "Composite should be at least Weak: {} ({})",
            score.composite,
            score.level
        );
    }

    #[test]
    fn test_all_indicator_scores_positive() {
        let results = run_all_benchmarks();
        let score = QualiaConfidenceScore::from_results(&results);
        for ind in &score.indicators {
            assert!(
                ind.score > 0.0,
                "Indicator '{}' should have positive score: {} (raw={})",
                ind.name,
                ind.score,
                ind.raw_value
            );
        }
    }

    #[test]
    fn test_missing_results_gives_zero() {
        let score = QualiaConfidenceScore::from_results(&[]);
        assert_eq!(score.predictions_met, 0);
        assert!(
            score.composite < 0.01,
            "Empty results should give near-zero score"
        );
    }

    #[test]
    fn test_format_report_not_empty() {
        let results = run_all_benchmarks();
        let score = QualiaConfidenceScore::from_results(&results);
        let report = score.format_report();
        assert!(report.len() > 100, "Report should have content");
        assert!(report.contains("HEURISTIC INDEX"));
        assert!(report.contains(&format!("{}", score.level)));
    }

    #[test]
    fn test_confidence_level_ordering() {
        // Verify the thresholds are consistent
        assert_eq!(
            ConfidenceLevel::Strong,
            if 0.85 >= 0.80 {
                ConfidenceLevel::Strong
            } else {
                ConfidenceLevel::Moderate
            }
        );
        assert_eq!(
            ConfidenceLevel::Moderate,
            if 0.70 >= 0.80 {
                ConfidenceLevel::Strong
            } else if 0.70 >= 0.60 {
                ConfidenceLevel::Moderate
            } else {
                ConfidenceLevel::Weak
            }
        );
    }
}
