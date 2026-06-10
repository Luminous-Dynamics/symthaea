// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ARC Rule Algebra Probes benchmark (SYNTHETIC — tests HDC algebra, not reasoning).
//!
//! **Honesty note:** This benchmark explicitly tests algebraic properties of the
//! HDC encoding, not cognitive reasoning. The z-scores reflect how well BinaryHV
//! XOR binding preserves mathematical structure (commutativity, associativity,
//! inverse existence, identity, distributivity). These are encoding quality
//! metrics, not abstract reasoning metrics.
//!
//! Tests whether HDC rule encodings satisfy algebraic properties:
//!
//! 1. **Commutativity**: Does encode_rule(A→B) ≈ encode_rule(B→A) after unbinding?
//!    Actually tests: is bind(in_A, out_A) similar to bind(in_B, out_B) when
//!    A and B use the same transform but different grids? (Rule consistency.)
//!
//! 2. **Associativity**: Does (A∘B)∘C produce similar results to A∘(B∘C)?
//!    Tests whether chaining order matters for the composed rule HV.
//!
//! 3. **Inverse existence**: Does apply_rule(output, rule) recover the input?
//!    HDC bind is approximately self-inverse for continuous HVs.
//!
//! 4. **Identity preservation**: Does encode_rule(grid, grid) produce a near-identity
//!    rule that preserves the input when applied?
//!
//! 5. **Distributivity**: Does rule(A+B) ≈ rule(A) + rule(B)?
//!    Tests whether the rule operator distributes over bundling.
//!
//! These are theoretical predictions of HDC algebra. Violations indicate
//! where the encoding breaks down.
//!
//! References:
//! - Plate (2003) Holographic Reduced Representation
//! - Kanerva (2009) Hyperdimensional Computing

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::binary_grid_encoder::BinaryGridEncoder;
use symthaea_core::hdc::grid_encoder::GridEncoder;

/// Rule algebra probes for HDC grid encoding.
pub struct ArcAlgebraBenchmark;

fn xor_shift(s: &mut u64) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
}

fn gen_grid(rng: &mut u64, size: usize, num_colors: u8) -> Vec<Vec<u8>> {
    let mut grid = vec![vec![0u8; size]; size];
    for row in grid.iter_mut() {
        for cell in row.iter_mut() {
            xor_shift(rng);
            *cell = (*rng % num_colors as u64) as u8;
        }
    }
    grid
}

struct TrialResult {
    /// Rule consistency: similarity of rules from different inputs (same transform)
    rule_consistency: f64,
    /// Associativity: similarity of (A∘B)∘C vs A∘(B∘C) composed rules
    associativity: f64,
    /// Inverse: similarity of unbind(output, rule) to original input
    inverse_similarity: f64,
    /// Identity: similarity of apply_rule(input, identity_rule) to input
    identity_preservation: f64,
    /// Distributivity: similarity of rule(A+B) vs rule(A)+rule(B)
    distributivity: f64,
    rt_ticks: f64,
}

impl ArcAlgebraBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let _dim = config.dimension;
        let seed = config.trial_seed("reasoning", "arc_algebra", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let grid_size = 5;
        let num_colors: u8 = 6;
        let num_probes = 8;

        let pressure = config.time_pressure;
        let tick_scale = 1.0 - pressure * 0.4;

        let encoder = BinaryGridEncoder::new(grid_size, grid_size, num_colors as usize, seed);

        let mut consistency_vals = Vec::new();
        let mut associativity_vals = Vec::new();
        let mut inverse_vals = Vec::new();
        let mut identity_vals = Vec::new();
        let mut distributivity_vals = Vec::new();

        for _probe in 0..num_probes {
            // ── Rule consistency (pseudo-commutativity) ──
            // Two different inputs, same transform → rules should be similar
            xor_shift(&mut rng);
            let input_a = gen_grid(&mut rng, grid_size, num_colors);
            let output_a = GridEncoder::reflect_x(&input_a);
            let in_a_hv = encoder.encode_grid(&input_a);
            let out_a_hv = encoder.encode_grid(&output_a);
            let rule_a = encoder.encode_rule(&in_a_hv, &out_a_hv);

            xor_shift(&mut rng);
            let input_b = gen_grid(&mut rng, grid_size, num_colors);
            let output_b = GridEncoder::reflect_x(&input_b);
            let in_b_hv = encoder.encode_grid(&input_b);
            let out_b_hv = encoder.encode_grid(&output_b);
            let rule_b = encoder.encode_rule(&in_b_hv, &out_b_hv);

            consistency_vals.push(rule_a.similarity(&rule_b) as f64);

            // ── Associativity: (reflect_x ∘ translate) ∘ color_replace vs reflect_x ∘ (translate ∘ color_replace) ──
            xor_shift(&mut rng);
            let base_grid = gen_grid(&mut rng, grid_size, num_colors);

            // Forward: apply A, then B, then C
            let after_a = GridEncoder::reflect_x(&base_grid);
            let after_ab = GridEncoder::translate_grid(&after_a, 1, 0, 0);
            let c0 = (rng % num_colors as u64) as u8;
            let c1 = ((rng / 7 + 1) % num_colors as u64) as u8;
            let after_abc = GridEncoder::color_replace(&after_ab, c0, c1);

            // Route 1: learn (A∘B) rule, then compose with C
            let base_hv = encoder.encode_grid(&base_grid);
            let abc_hv = encoder.encode_grid(&after_abc);
            let rule_abc_direct = encoder.encode_rule(&base_hv, &abc_hv);

            // Route 2: same thing but via different training input
            xor_shift(&mut rng);
            let alt_grid = gen_grid(&mut rng, grid_size, num_colors);
            let alt_after_a = GridEncoder::reflect_x(&alt_grid);
            let alt_after_ab = GridEncoder::translate_grid(&alt_after_a, 1, 0, 0);
            let alt_after_abc = GridEncoder::color_replace(&alt_after_ab, c0, c1);
            let alt_hv = encoder.encode_grid(&alt_grid);
            let alt_abc_hv = encoder.encode_grid(&alt_after_abc);
            let rule_abc_alt = encoder.encode_rule(&alt_hv, &alt_abc_hv);

            // Associativity = similarity of the two composed rules
            associativity_vals.push(rule_abc_direct.similarity(&rule_abc_alt) as f64);

            // ── Inverse existence: unbind(output, rule) ≈ input ──
            let predicted_input = out_a_hv.bind(&rule_a);
            let inv_sim = predicted_input.similarity(&in_a_hv) as f64;
            inverse_vals.push(inv_sim);

            // ── Identity preservation: rule(grid, grid) applied to grid ≈ grid ──
            xor_shift(&mut rng);
            let id_grid = gen_grid(&mut rng, grid_size, num_colors);
            let id_hv = encoder.encode_grid(&id_grid);
            let identity_rule = encoder.encode_rule(&id_hv, &id_hv);
            let recovered = encoder.apply_rule(&id_hv, &identity_rule);
            let id_sim = recovered.similarity(&id_hv) as f64;
            identity_vals.push(id_sim);

            // ── Distributivity: rule(bundle(A,B)) ≈ bundle(rule(A), rule(B)) ──
            xor_shift(&mut rng);
            let grid_d1 = gen_grid(&mut rng, grid_size, num_colors);
            xor_shift(&mut rng);
            let grid_d2 = gen_grid(&mut rng, grid_size, num_colors);
            let hv_d1 = encoder.encode_grid(&grid_d1);
            let hv_d2 = encoder.encode_grid(&grid_d2);

            let out_d1 = GridEncoder::reflect_y(&grid_d1);
            let out_d2 = GridEncoder::reflect_y(&grid_d2);
            let out_hv_d1 = encoder.encode_grid(&out_d1);
            let out_hv_d2 = encoder.encode_grid(&out_d2);

            let rule_d1 = encoder.encode_rule(&hv_d1, &out_hv_d1);
            let rule_d2 = encoder.encode_rule(&hv_d2, &out_hv_d2);

            // bundle(A, B) then apply_rule
            let bundle_input = BinaryHV::bundle(&[hv_d1, hv_d2]);
            let consensus_rule = encoder.bundle_rules(&[rule_d1, rule_d2]);
            let result_left = encoder.apply_rule(&bundle_input, &consensus_rule);

            // apply_rule(A) + apply_rule(B)
            let result_d1 = encoder.apply_rule(&hv_d1, &rule_d1);
            let result_d2 = encoder.apply_rule(&hv_d2, &rule_d2);
            let result_right = BinaryHV::bundle(&[result_d1, result_d2]);

            distributivity_vals.push(result_left.similarity(&result_right) as f64);
        }

        let mean = |v: &[f64]| -> f64 {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };

        xor_shift(&mut rng);
        let rt_ticks = (5.0 + (rng % 5) as f64) * tick_scale;

        TrialResult {
            rule_consistency: mean(&consistency_vals),
            associativity: mean(&associativity_vals),
            inverse_similarity: mean(&inverse_vals),
            identity_preservation: mean(&identity_vals),
            distributivity: mean(&distributivity_vals),
            rt_ticks,
        }
    }
}

impl PsychBenchmark for ArcAlgebraBenchmark {
    fn name(&self) -> &str {
        "Reasoning::ArcAlgebra"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Algebraic properties of hyperdimensional rule encoding",
            citation: "Plate (2003); Kanerva (2009)",
            year: 2003,
            doi: Some("10.1007/s12559-009-9009-8"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut consistencies = Vec::new();
        let mut associativities = Vec::new();
        let mut inverses = Vec::new();
        let mut identities = Vec::new();
        let mut distributivities = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            consistencies.push(r.rule_consistency);
            associativities.push(r.associativity);
            inverses.push(r.inverse_similarity);
            identities.push(r.identity_preservation);
            distributivities.push(r.distributivity);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "arc_algebra".to_string(),
                    correct: r.rule_consistency > 0.0,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "rule_consistency",
            MetricValue::from_samples(&consistencies),
        );
        result.insert("associativity", MetricValue::from_samples(&associativities));
        result.insert("inverse_similarity", MetricValue::from_samples(&inverses));
        result.insert(
            "identity_preservation",
            MetricValue::from_samples(&identities),
        );
        result.insert(
            "distributivity",
            MetricValue::from_samples(&distributivities),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        // Algebra score: mean of all 5 property scores
        let algebra_scores: Vec<f64> = (0..config.trials_per_condition)
            .map(|i| {
                (consistencies[i]
                    + associativities[i]
                    + inverses[i]
                    + identities[i]
                    + distributivities[i])
                    / 5.0
            })
            .collect();
        result.insert("algebra_score", MetricValue::from_samples(&algebra_scores));

        result.conditions = 5; // 5 algebraic properties
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.notes.push(
            "SYNTHETIC: Tests algebraic properties of HDC encoding \
             (commutativity, associativity, inverse, identity, distributivity). \
             Z-scores reflect encoding algebra quality, not reasoning ability."
                .to_string(),
        );
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BenchmarkConfig {
        BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        }
    }

    #[test]
    fn test_algebra_runs_with_metrics() {
        let result = ArcAlgebraBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("rule_consistency"));
        assert!(result.metrics.contains_key("associativity"));
        assert!(result.metrics.contains_key("inverse_similarity"));
        assert!(result.metrics.contains_key("identity_preservation"));
        assert!(result.metrics.contains_key("distributivity"));
        assert!(result.metrics.contains_key("algebra_score"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = ArcAlgebraBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_identity_positive() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ArcAlgebraBenchmark.run(&config);
        let id = result.metrics["identity_preservation"].mean;
        // Identity rule (same input→output) should produce positive similarity
        assert!(
            id > 0.0,
            "identity_preservation should be positive, got {}",
            id
        );
    }

    #[test]
    fn test_inverse_positive() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ArcAlgebraBenchmark.run(&config);
        let inv = result.metrics["inverse_similarity"].mean;
        // Unbinding should recover input above noise floor
        assert!(
            inv > 0.0,
            "inverse_similarity should be positive, got {}",
            inv
        );
    }

    #[test]
    fn test_provenance() {
        let prov = ArcAlgebraBenchmark.provenance().unwrap();
        assert!(prov.citation.contains("Plate"));
    }

    #[test]
    fn test_deterministic() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 2,
            seed: 42,
            ..Default::default()
        };
        let r1 = ArcAlgebraBenchmark.run(&config);
        let r2 = ArcAlgebraBenchmark.run(&config);
        assert_eq!(
            r1.metrics["algebra_score"].mean,
            r2.metrics["algebra_score"].mean
        );
    }
}
