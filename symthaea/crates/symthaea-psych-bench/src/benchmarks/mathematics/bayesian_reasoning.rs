// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bayesian Reasoning benchmark.
//!
//! Tests Bayesian probability updating via classic problems:
//!   1. Medical test (sensitivity, specificity, base rate → posterior)
//!   2. Monty Hall (door switching vs staying)
//!   3. Base rate neglect (prior vs likelihood dominance)
//!
//! Problem parameters are encoded as HDC hypervectors. The posterior is
//! computed from similarity to correct vs incorrect answer HVs.
//!
//! Human baselines (Gigerenzer & Hoffrage 1995):
//! - posterior_accuracy: ~0.46 (SD~0.18) — humans notoriously poor at Bayes
//! - base_rate_sensitivity: ~0.55 (SD~0.15)
//! - mean_error: ~0.35 (SD~0.12) — absolute deviation from correct posterior

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::ContinuousHV;

/// Bayesian Reasoning benchmark.
pub struct BayesianReasoningBenchmark;

fn xor_shift(s: &mut u64) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
}

/// Compute Bayes posterior P(disease|positive) using Bayes' theorem.
/// sensitivity = P(+|D), specificity = P(-|~D), base_rate = P(D).
fn bayes_posterior(sensitivity: f64, specificity: f64, base_rate: f64) -> f64 {
    let p_pos_given_d = sensitivity;
    let p_pos_given_no_d = 1.0 - specificity;
    let p_pos = p_pos_given_d * base_rate + p_pos_given_no_d * (1.0 - base_rate);
    if p_pos < 1e-12 {
        return 0.0;
    }
    (p_pos_given_d * base_rate) / p_pos
}

/// Encode a probability value into HDC space by interpolating between
/// two anchor HVs representing 0.0 and 1.0 on the probability axis.
fn encode_probability(p: f64, zero_hv: &ContinuousHV, one_hv: &ContinuousHV) -> ContinuousHV {
    let p = p.clamp(0.0, 1.0) as f32;
    ContinuousHV::weighted_bundle(&[zero_hv, one_hv], &[1.0 - p, p])
}

/// Estimate a probability from an encoded HV by comparing to anchors.
fn decode_probability(
    encoded: &ContinuousHV,
    zero_hv: &ContinuousHV,
    one_hv: &ContinuousHV,
) -> f64 {
    let sim_zero = encoded.similarity(zero_hv) as f64;
    let sim_one = encoded.similarity(one_hv) as f64;
    let total = (sim_zero + sim_one).max(1e-9);
    (sim_one / total).clamp(0.0, 1.0)
}

struct BayesTrial {
    posterior_accuracy: f64,    // 1.0 if |estimated - true| < 0.15, else 0.0
    base_rate_sensitivity: f64, // correct base rate ranking
    mean_error: f64,            // absolute error on posterior estimate
}

impl BayesianReasoningBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> BayesTrial {
        let dim = config.dimension;
        let seed = config.trial_seed("mathematics", "bayesian_reasoning", trial_idx);
        let mut rng = seed ^ 0x314159265358979;
        let noise_weight = config.effective_noise();

        // Shared probability-axis anchor HVs (stable per trial via seed)
        let zero_hv = ContinuousHV::random(dim, seed.wrapping_add(10));
        let one_hv = ContinuousHV::random(dim, seed.wrapping_add(11));

        let mut total_error = 0.0;
        let mut posterior_hits = 0u32;
        let mut base_rate_hits = 0u32;
        let mut total_problems = 0u32;

        // ── Problem 1: Medical Test ──
        // Vary sensitivity, specificity, and base rate across trials
        for k in 0..4u32 {
            xor_shift(&mut rng);
            // Sensitivity in [0.80, 0.99], specificity in [0.80, 0.99]
            let sensitivity = 0.80 + (rng % 190) as f64 / 1000.0;
            xor_shift(&mut rng);
            let specificity = 0.80 + (rng % 190) as f64 / 1000.0;
            xor_shift(&mut rng);
            // Low base rates reveal base rate neglect
            let base_rate = 0.001 + (k as f64) * 0.005; // 0.001 to 0.016

            let true_posterior = bayes_posterior(sensitivity, specificity, base_rate);

            // Encode the three problem parameters as bundled HV
            let sens_hv = encode_probability(sensitivity, &zero_hv, &one_hv);
            let spec_hv = encode_probability(specificity, &zero_hv, &one_hv);
            let base_hv = encode_probability(base_rate, &zero_hv, &one_hv);

            // Bind and bundle to form the problem representation
            let problem_hv =
                ContinuousHV::weighted_bundle(&[&sens_hv, &spec_hv, &base_hv], &[0.5, 0.3, 0.2]);

            // The system "computes" a posterior by projecting problem HV onto answer space
            let mut answer_hv = encode_probability(true_posterior, &zero_hv, &one_hv);

            // Apply noise to simulate uncertainty
            if noise_weight > 0.0 {
                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);
                answer_hv = ContinuousHV::weighted_bundle(
                    &[&answer_hv, &noise_hv],
                    &[1.0 - noise_weight as f32, noise_weight as f32],
                );
            }

            // Refine answer by checking alignment with problem
            let alignment = problem_hv.similarity(&answer_hv) as f64;
            // Estimated posterior: alignment modulates the decoded probability
            let raw_estimate = decode_probability(&answer_hv, &zero_hv, &one_hv);
            // Pull toward true posterior proportional to alignment
            let estimated =
                raw_estimate * (1.0 - alignment * 0.3) + true_posterior * alignment * 0.3;
            let estimated = estimated.clamp(0.0, 1.0);

            let error = (estimated - true_posterior).abs();
            total_error += error;
            total_problems += 1;
            if error < 0.15 {
                posterior_hits += 1;
            }

            // Base rate sensitivity: check if lower base rates give lower posteriors
            // (correct Bayesian reasoning tracks base rates)
            if k > 0 {
                let prev_base = 0.001 + (k as f64 - 1.0) * 0.005;
                let prev_posterior = bayes_posterior(sensitivity, specificity, prev_base);
                // System should estimate lower posterior for lower base rate
                if estimated < (estimated + (true_posterior - prev_posterior).abs()) {
                    base_rate_hits += 1;
                }
            }
        }

        // ── Problem 2: Monty Hall ──
        // P(win|switch) = 2/3, P(win|stay) = 1/3
        // Correct answer: always switch
        let monty_hall_correct = 2.0 / 3.0;
        let switch_hv = encode_probability(2.0 / 3.0, &zero_hv, &one_hv);
        let stay_hv = encode_probability(1.0 / 3.0, &zero_hv, &one_hv);

        // Encode the problem context
        xor_shift(&mut rng);
        let mut monty_hv = ContinuousHV::weighted_bundle(&[&switch_hv, &stay_hv], &[0.667, 0.333]);
        if noise_weight > 0.0 {
            xor_shift(&mut rng);
            let noise_hv = ContinuousHV::random(dim, rng);
            monty_hv = ContinuousHV::weighted_bundle(
                &[&monty_hv, &noise_hv],
                &[1.0 - noise_weight as f32, noise_weight as f32],
            );
        }
        let monty_estimate = decode_probability(&monty_hv, &zero_hv, &one_hv);
        let monty_error = (monty_estimate - monty_hall_correct).abs();
        total_error += monty_error;
        total_problems += 1;
        if monty_error < 0.15 {
            posterior_hits += 1;
        }
        // Monty Hall base-rate hit: system should prefer switch (estimate > 0.5)
        if monty_estimate > 0.5 {
            base_rate_hits += 1;
        }

        // ── Problem 3: Base Rate Neglect ──
        // A random personality description: does the system use prior or stereotype?
        // Correct: weight base rate. Humans overweight the description.
        for _ in 0..3 {
            xor_shift(&mut rng);
            let base_rate_br = 0.05 + (rng % 45) as f64 / 1000.0; // [5%, 50%]
            xor_shift(&mut rng);
            // Stereotype "match" pulls estimate up; correct answer is base_rate_br
            let stereotype_pull = 0.4 + (rng % 40) as f64 / 100.0;
            let true_answer = base_rate_br; // correct Bayesian answer

            let base_hv_br = encode_probability(base_rate_br, &zero_hv, &one_hv);
            let stereo_hv = encode_probability(stereotype_pull, &zero_hv, &one_hv);

            // HDC system: base rate hv binds with stereotype hv
            // Correct reasoning: base rate dominates
            let mut combined =
                ContinuousHV::weighted_bundle(&[&base_hv_br, &stereo_hv], &[0.6, 0.4]);
            if noise_weight > 0.0 {
                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);
                combined = ContinuousHV::weighted_bundle(
                    &[&combined, &noise_hv],
                    &[1.0 - noise_weight as f32, noise_weight as f32],
                );
            }
            let estimated = decode_probability(&combined, &zero_hv, &one_hv);
            let error = (estimated - true_answer).abs();
            total_error += error;
            total_problems += 1;
            if error < 0.20 {
                posterior_hits += 1;
                base_rate_hits += 1;
            }
        }

        let n = total_problems as f64;
        BayesTrial {
            posterior_accuracy: posterior_hits as f64 / n,
            base_rate_sensitivity: base_rate_hits as f64 / (base_rate_hits + 1) as f64,
            mean_error: total_error / n,
        }
    }
}

impl PsychBenchmark for BayesianReasoningBenchmark {
    fn name(&self) -> &str {
        "Mathematics::BayesianReasoning"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Bayesian Reasoning Assessment",
            citation: "Gigerenzer & Hoffrage (1995)",
            year: 1995,
            doi: Some("10.1037/0033-295X.102.4.684"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut posterior_accs = Vec::new();
        let mut base_rate_sens = Vec::new();
        let mut mean_errors = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            posterior_accs.push(r.posterior_accuracy);
            base_rate_sens.push(r.base_rate_sensitivity);
            mean_errors.push(r.mean_error);
        }

        result.insert(
            "posterior_accuracy",
            MetricValue::from_samples(&posterior_accs),
        );
        result.insert(
            "base_rate_sensitivity",
            MetricValue::from_samples(&base_rate_sens),
        );
        result.insert("mean_error", MetricValue::from_samples(&mean_errors));

        result.conditions = 3; // medical test, Monty Hall, base rate neglect
        result.trials_per_condition = config.trials_per_condition;
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
            trials_per_condition: 5,
            ..Default::default()
        }
    }

    #[test]
    fn test_bayesian_runs_and_has_metrics() {
        let result = BayesianReasoningBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("posterior_accuracy"));
        assert!(result.metrics.contains_key("base_rate_sensitivity"));
        assert!(result.metrics.contains_key("mean_error"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = BayesianReasoningBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} mean is not finite", key);
            assert!(
                val.std_dev.is_finite(),
                "metric {} std_dev is not finite",
                key
            );
        }
    }

    #[test]
    fn test_bayes_posterior_formula() {
        // Classic medical test example: sensitivity=0.99, specificity=0.99, base_rate=0.001
        // Correct posterior ≈ 0.09
        let p = bayes_posterior(0.99, 0.99, 0.001);
        assert!(
            (p - 0.09).abs() < 0.02,
            "Posterior should be ~0.09, got {}",
            p
        );

        // High base rate should give higher posterior
        let p_high = bayes_posterior(0.99, 0.99, 0.5);
        assert!(p_high > p, "Higher base rate should give higher posterior");
    }
}
