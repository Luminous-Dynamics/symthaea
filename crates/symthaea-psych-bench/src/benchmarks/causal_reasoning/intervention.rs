// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Intervention Effect Benchmark — Pearl's do-operator in HDC space.
//!
//! Tests whether HDC composition algebra correctly distinguishes observational
//! from interventional reasoning. When we intervene on variable X (do(X=x)),
//! upstream causes of X are severed. The benchmark encodes causal DAGs as
//! HDC bound structures and tests whether "intervention unbinding" (severing
//! upstream edges) produces different predictions from mere conditioning.
//!
//! ## Key Claims Tested
//!
//! 1. **Intervention vs Observation**: Intervening on a mediator severs its
//!    upstream causes, producing a different residual than conditioning.
//! 2. **Backdoor Criterion**: Adjusting for confounders (unbinding them)
//!    should recover the causal effect from observational data.
//! 3. **Front-door Criterion**: When confounders are latent, mediator-based
//!    adjustment should approximate the interventional distribution.
//!
//! ## References
//!
//! - Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. 2nd ed.
//! - Pearl, J. (2000). Causal diagrams for empirical research. *Biometrika*.
//! - Bramley, N. R. et al. (2017). Formalizing Neurath's ship. *Cognition*.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;

/// Intervention Effect benchmark: do-operator in HDC space.
pub struct InterventionEffectBenchmark;

struct TrialResult {
    /// Whether intervention unbinding yields different residual than conditioning
    intervention_discrimination: f64,
    /// Backdoor adjustment accuracy: unbinding confounder recovers causal effect
    backdoor_accuracy: f64,
    /// Front-door accuracy: mediator-based adjustment approximates intervention
    frontdoor_accuracy: f64,
    /// Overall causal reasoning score
    causal_score: f64,
}

/// A simple 3-node causal structure: Cause → Mediator → Effect
/// with optional Confounder → {Cause, Effect}
struct CausalDag {
    cause: BinaryHV,
    mediator: BinaryHV,
    effect: BinaryHV,
    confounder: Option<BinaryHV>,
    /// Observational composite: XOR bind of all variables
    observational: BinaryHV,
    /// Interventional composite: mediator severed from cause
    interventional: BinaryHV,
}

impl CausalDag {
    fn new(seed: u64, with_confounder: bool) -> Self {
        let cause = BinaryHV::random(seed.wrapping_add(1));
        let mediator = BinaryHV::random(seed.wrapping_add(2));
        let effect = BinaryHV::random(seed.wrapping_add(3));

        // Observational: full DAG as bound composite
        // Cause ⊕ Mediator ⊕ Effect (⊕ Confounder if present)
        let mut observational = cause.bind(&mediator).bind(&effect);

        let confounder = if with_confounder {
            let c = BinaryHV::random(seed.wrapping_add(4));
            observational = observational.bind(&c);
            Some(c)
        } else {
            None
        };

        // Interventional: do(Mediator) severs Cause → Mediator edge
        // Represented as Mediator ⊕ Effect (cause influence removed)
        let interventional = mediator.bind(&effect);

        Self {
            cause,
            mediator,
            effect,
            confounder,
            observational,
            interventional,
        }
    }
}

impl InterventionEffectBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("causal_reasoning", "intervention", trial_idx);

        // ── Test 1: Intervention vs Observation discrimination ────────
        // do(Mediator) should produce a different residual than just
        // conditioning on Mediator (which leaves upstream edges intact)
        let n_dags = 10;
        let mut discrim_scores = Vec::with_capacity(n_dags);

        for i in 0..n_dags {
            let dag = CausalDag::new(seed.wrapping_add(i as u64 * 100), false);

            // Conditioning: unbind Mediator from full observational composite
            // Residual = Cause ⊕ Effect (upstream edges still present)
            let conditioned_residual = dag.observational.bind(&dag.mediator);

            // Intervention: unbind Mediator from interventional composite
            // Residual = Effect only (upstream severed)
            let intervened_residual = dag.interventional.bind(&dag.mediator);

            // These residuals should be DIFFERENT (cause is in one but not the other)
            let sim = conditioned_residual.cosine_similarity(&intervened_residual);
            // Lower similarity = better discrimination
            discrim_scores.push(1.0 - sim as f64);
        }
        let intervention_discrimination =
            discrim_scores.iter().sum::<f64>() / discrim_scores.len() as f64;

        // ── Test 2: Backdoor criterion ────────────────────────────────
        // With a confounder C → {Cause, Effect}, adjusting for C should
        // recover the causal effect Cause → Effect
        let mut backdoor_scores = Vec::with_capacity(n_dags);

        for i in 0..n_dags {
            let dag = CausalDag::new(seed.wrapping_add(1000 + i as u64 * 100), true);
            let confounder = dag.confounder.as_ref().unwrap();

            // Unadjusted: includes confounding
            let unadjusted = dag.observational.bind(&dag.mediator);

            // Adjusted: unbind confounder to remove confounding
            let adjusted = unadjusted.bind(confounder);

            // The adjusted residual should be closer to the true causal
            // effect (Cause alone) than the unadjusted residual
            let adjusted_to_cause = adjusted.cosine_similarity(&dag.cause);
            let unadjusted_to_cause = unadjusted.cosine_similarity(&dag.cause);

            // Score: how much adjustment improves recovery of the cause
            let improvement = (adjusted_to_cause - unadjusted_to_cause) as f64;
            backdoor_scores.push(improvement.clamp(0.0, 1.0));
        }
        let backdoor_accuracy = backdoor_scores.iter().sum::<f64>() / backdoor_scores.len() as f64;

        // ── Test 3: Front-door criterion ──────────────────────────────
        // When confounder is latent, use the mediator path to estimate
        // the causal effect: P(Y|do(X)) via mediator adjustment
        let mut frontdoor_scores = Vec::with_capacity(n_dags);

        for i in 0..n_dags {
            let dag = CausalDag::new(seed.wrapping_add(2000 + i as u64 * 100), true);

            // Front-door: estimate effect via Cause → Mediator → Effect path
            // Unbind cause from observational to get mediator+effect+confounder
            let via_mediator = dag.observational.bind(&dag.cause);

            // Then unbind mediator to isolate effect (+ confounder residual)
            let effect_estimate = via_mediator.bind(&dag.mediator);

            // Compare to true effect
            let sim = effect_estimate.cosine_similarity(&dag.effect);

            // With confounder present, front-door gives approximate recovery
            frontdoor_scores.push(sim as f64);
        }
        let frontdoor_accuracy =
            frontdoor_scores.iter().sum::<f64>() / frontdoor_scores.len() as f64;

        let causal_score =
            intervention_discrimination * 0.4 + backdoor_accuracy * 0.3 + frontdoor_accuracy * 0.3;

        TrialResult {
            intervention_discrimination,
            backdoor_accuracy,
            frontdoor_accuracy,
            causal_score,
        }
    }
}

impl PsychBenchmark for InterventionEffectBenchmark {
    fn name(&self) -> &str {
        "CausalReasoning::InterventionEffect"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Intervention Effect (do-calculus)",
            citation: "Pearl, J. (2009). Causality: Models, Reasoning, and Inference. 2nd ed. Cambridge UP.",
            year: 2009,
            doi: Some("10.1017/CBO9780511803161"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut intervention_vals = Vec::new();
        let mut backdoor_vals = Vec::new();
        let mut frontdoor_vals = Vec::new();
        let mut causal_vals = Vec::new();
        let mut trace = Vec::new();

        for t in 0..config.trials_per_condition {
            let r = self.run_trial(config, t);
            intervention_vals.push(r.intervention_discrimination);
            backdoor_vals.push(r.backdoor_accuracy);
            frontdoor_vals.push(r.frontdoor_accuracy);
            causal_vals.push(r.causal_score);

            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert(
                    "intervention_discrimination".to_string(),
                    r.intervention_discrimination,
                );
                extra.insert("backdoor_accuracy".to_string(), r.backdoor_accuracy);
                extra.insert("frontdoor_accuracy".to_string(), r.frontdoor_accuracy);
                trace.push(TrialOutcome {
                    trial_idx: t,
                    correct: r.causal_score > 0.3,
                    rt_ticks: 1.0,
                    confidence: r.causal_score,
                    similarity: r.causal_score,
                    response_idx: 0,
                    condition: "intervention".to_string(),
                    extra,
                });
            }
        }

        result.insert(
            "intervention_discrimination",
            MetricValue::from_samples(&intervention_vals),
        );
        result.insert(
            "backdoor_accuracy",
            MetricValue::from_samples(&backdoor_vals),
        );
        result.insert(
            "frontdoor_accuracy",
            MetricValue::from_samples(&frontdoor_vals),
        );
        result.insert("causal_score", MetricValue::from_samples(&causal_vals));

        result.conditions = 4;
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
    fn test_intervention_effect_runs() {
        let bench = InterventionEffectBenchmark;
        let result = bench.run(&default_config());
        assert!(result.benchmark.contains("InterventionEffect"));
        assert!(result.metrics.contains_key("causal_score"));
    }

    #[test]
    fn test_intervention_discrimination_positive() {
        let bench = InterventionEffectBenchmark;
        let result = bench.run(&default_config());
        let discrim = result.metrics["intervention_discrimination"].mean;
        assert!(
            discrim > 0.0,
            "Intervention should produce discriminable residuals, got {discrim}"
        );
    }

    #[test]
    fn test_causal_score_bounded() {
        let bench = InterventionEffectBenchmark;
        let result = bench.run(&default_config());
        let score = result.metrics["causal_score"].mean;
        assert!(
            score >= 0.0 && score <= 1.0,
            "Score should be in [0,1], got {score}"
        );
    }

    #[test]
    fn test_provenance_present() {
        let bench = InterventionEffectBenchmark;
        let prov = bench.provenance().expect("Should have provenance");
        assert_eq!(prov.year, 2009);
        assert!(prov.citation.contains("Pearl"));
    }
}
