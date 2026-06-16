// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Confound Detection Benchmark — Simpson's Paradox in HDC space.
//!
//! Tests whether the system can detect when an apparent correlation between
//! two variables reverses upon stratification by a confounding variable.
//! This is the hallmark of Simpson's paradox and tests the system's ability
//! to distinguish correlation from causation.
//!
//! ## HDC Implementation
//!
//! Three variables (X, Y, C) are encoded as BinaryHVs. The composite X⊕Y
//! has a certain similarity to a "positive effect" template. When stratified
//! by C (unbinding C), the direction should reverse — the residual (X⊕Y)⊕C
//! should be LESS similar to the positive effect template than X⊕Y alone.
//!
//! ## References
//!
//! - Simpson, E. H. (1951). The interpretation of interaction in contingency tables.
//!   *JRSS Series B*.
//! - Pearl, J. (2014). Comment: Understanding Simpson's paradox.
//!   *The American Statistician*.
//! - Spirtes, P. et al. (2000). *Causation, Prediction, and Search*. MIT Press.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;

/// Confound Detection benchmark: Simpson's paradox in HDC space.
pub struct ConfoundDetectionBenchmark;

struct TrialResult {
    /// Can the system detect that X→Y is confounded by C?
    confound_detection_accuracy: f64,
    /// Does stratification by C change the apparent effect direction?
    reversal_magnitude: f64,
    /// Can the system identify which variable is the confounder?
    confounder_identification: f64,
}

impl ConfoundDetectionBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("causal_reasoning", "confound", trial_idx);

        let n_scenarios = 10;
        let mut detection_scores = Vec::with_capacity(n_scenarios);
        let mut reversal_scores = Vec::with_capacity(n_scenarios);
        let mut ident_scores = Vec::with_capacity(n_scenarios);

        for i in 0..n_scenarios {
            let s = seed.wrapping_add(i as u64 * 100);

            // Create causal variables
            let x = BinaryHV::random(s.wrapping_add(1));
            let y = BinaryHV::random(s.wrapping_add(2));
            let confounder = BinaryHV::random(s.wrapping_add(3));
            let irrelevant = BinaryHV::random(s.wrapping_add(4));

            // Effect template: X ⊕ Y (direct association)
            let direct_effect = x.bind(&y);

            // Confounded composite: X ⊕ Y ⊕ C (confounder mixed in)
            let confounded = direct_effect.bind(&confounder);

            // ── Test 1: Detection ─────────────────────────────────────
            // The confounded composite should be LESS similar to the
            // direct effect than the direct effect is to itself (1.0)
            let confounded_sim = confounded.cosine_similarity(&direct_effect);
            // Larger difference from 1.0 = easier to detect confounding
            let detection = (1.0 - confounded_sim) as f64;
            detection_scores.push(detection.clamp(0.0, 1.0));

            // ── Test 2: Reversal on stratification ────────────────────
            // Unbinding the confounder should recover the direct effect
            let adjusted = confounded.bind(&confounder);
            let adjusted_sim = adjusted.cosine_similarity(&direct_effect);

            // Reversal: how much does adjusting change the similarity?
            let reversal = (adjusted_sim - confounded_sim) as f64;
            reversal_scores.push(reversal.clamp(0.0, 1.0));

            // ── Test 3: Confounder identification ─────────────────────
            // Given candidates [confounder, irrelevant], which one when
            // unbinding produces a result closer to the direct effect?
            let unbind_confounder = confounded.bind(&confounder);
            let unbind_irrelevant = confounded.bind(&irrelevant);

            let sim_correct = unbind_confounder.cosine_similarity(&direct_effect);
            let sim_wrong = unbind_irrelevant.cosine_similarity(&direct_effect);

            let identified = if sim_correct > sim_wrong { 1.0 } else { 0.0 };
            ident_scores.push(identified);
        }

        let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;

        TrialResult {
            confound_detection_accuracy: avg(&detection_scores),
            reversal_magnitude: avg(&reversal_scores),
            confounder_identification: avg(&ident_scores),
        }
    }
}

impl PsychBenchmark for ConfoundDetectionBenchmark {
    fn name(&self) -> &str {
        "CausalReasoning::ConfoundDetection"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Confound Detection (Simpson's Paradox)",
            citation: "Pearl, J. (2014). Comment: Understanding Simpson's paradox. The American Statistician, 68(1), 8-13.",
            year: 2014,
            doi: Some("10.1080/00031305.2014.876829"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut detection_vals = Vec::new();
        let mut reversal_vals = Vec::new();
        let mut ident_vals = Vec::new();
        let mut trace = Vec::new();

        for t in 0..config.trials_per_condition {
            let r = self.run_trial(config, t);
            detection_vals.push(r.confound_detection_accuracy);
            reversal_vals.push(r.reversal_magnitude);
            ident_vals.push(r.confounder_identification);

            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert("reversal_magnitude".to_string(), r.reversal_magnitude);
                extra.insert(
                    "confounder_identification".to_string(),
                    r.confounder_identification,
                );
                trace.push(TrialOutcome {
                    trial_idx: t,
                    correct: r.confounder_identification > 0.5,
                    rt_ticks: 1.0,
                    confidence: r.confound_detection_accuracy,
                    similarity: r.confound_detection_accuracy,
                    response_idx: 0,
                    condition: "confound".to_string(),
                    extra,
                });
            }
        }

        result.insert(
            "confound_detection_accuracy",
            MetricValue::from_samples(&detection_vals),
        );
        result.insert(
            "reversal_magnitude",
            MetricValue::from_samples(&reversal_vals),
        );
        result.insert(
            "confounder_identification",
            MetricValue::from_samples(&ident_vals),
        );

        result.conditions = 3;
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> BenchmarkConfig {
        BenchmarkConfig {
            trials_per_condition: 10,
            ..BenchmarkConfig::default()
        }
    }

    #[test]
    fn test_confound_detection_runs() {
        let bench = ConfoundDetectionBenchmark;
        let result = bench.run(&default_config());
        assert!(result.benchmark.contains("ConfoundDetection"));
        assert!(result.metrics.contains_key("confound_detection_accuracy"));
    }

    #[test]
    fn test_confounder_identification_above_chance() {
        let bench = ConfoundDetectionBenchmark;
        let result = bench.run(&default_config());
        let ident = result.metrics["confounder_identification"].mean;
        // With XOR binding, unbinding the correct variable should perfectly
        // recover the direct effect. Identification should be well above chance (0.5).
        assert!(
            ident > 0.5,
            "Confounder identification should be above chance, got {ident}"
        );
    }

    #[test]
    fn test_reversal_positive() {
        let bench = ConfoundDetectionBenchmark;
        let result = bench.run(&default_config());
        let rev = result.metrics["reversal_magnitude"].mean;
        assert!(
            rev > 0.0,
            "Adjusting for confounder should improve recovery, got {rev}"
        );
    }

    #[test]
    fn test_provenance_present() {
        let bench = ConfoundDetectionBenchmark;
        let prov = bench.provenance().expect("Should have provenance");
        assert_eq!(prov.year, 2014);
        assert!(prov.citation.contains("Simpson"));
    }
}
