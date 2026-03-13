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
        let mut observational = cause.xor(&mediator).xor(&effect);

        let confounder = if with_confounder {
            let c = BinaryHV::random(seed.wrapping_add(4));
            observational = observational.xor(&c);
            Some(c)
        } else {
            None
        };

        // Interventional: do(Mediator) severs Cause → Mediator edge
        // Represented as Mediator ⊕ Effect (cause influence removed)
        let interventional = mediator.xor(&effect);

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
            let conditioned_residual = dag.observational.xor(&dag.mediator);

            // Intervention: unbind Mediator from interventional composite
            // Residual = Effect only (upstream severed)
            let intervened_residual = dag.interventional.xor(&dag.mediator);

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
            let unadjusted = dag.observational.xor(&dag.mediator);

            // Adjusted: unbind confounder to remove confounding
            let adjusted = unadjusted.xor(confounder);

            // The adjusted residual should be closer to the true causal
            // effect (Cause alone) than the unadjusted residual
            let adjusted_to_cause = adjusted.cosine_similarity(&dag.cause);
            let unadjusted_to_cause = unadjusted.cosine_similarity(&dag.cause);

            // Score: how much adjustment improves recovery of the cause
            let improvement = (adjusted_to_cause - unadjusted_to_cause) as f64;
            backdoor_scores.push(improvement.max(0.0).min(1.0));
        }
        let backdoor_accuracy =
            backdoor_scores.iter().sum::<f64>() / backdoor_scores.len() as f64;

        // ── Test 3: Front-door criterion ──────────────────────────────
        // When confounder is latent, use the mediator path to estimate
        // the causal effect: P(Y|do(X)) via mediator adjustment
        let mut frontdoor_scores = Vec::with_capacity(n_dags);

        for i in 0..n_dags {
            let dag = CausalDag::new(seed.wrapping_add(2000 + i as u64 * 100), true);

            // Front-door: estimate effect via Cause → Mediator → Effect path
            // Unbind cause from observational to get mediator+effect+confounder
            let via_mediator = dag.observational.xor(&dag.cause);

            // Then unbind mediator to isolate effect (+ confounder residual)
            let effect_estimate = via_mediator.xor(&dag.mediator);

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
        "causal_reasoning/intervention"
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
        let n_trials = config.trials_per_condition;
        let mut trials = Vec::with_capacity(n_trials);
        let mut intervention_sum = 0.0;
        let mut backdoor_sum = 0.0;
        let mut frontdoor_sum = 0.0;
        let mut causal_sum = 0.0;

        for t in 0..n_trials {
            let result = self.run_trial(config, t);
            intervention_sum += result.intervention_discrimination;
            backdoor_sum += result.backdoor_accuracy;
            frontdoor_sum += result.frontdoor_accuracy;
            causal_sum += result.causal_score;

            trials.push(TrialOutcome {
                correct: result.causal_score > 0.3,
                rt_ticks: None,
                confidence: Some(result.causal_score as f32),
                condition: "intervention".to_string(),
            });
        }

        let n = n_trials as f64;
        let mut metrics = BTreeMap::new();
        metrics.insert(
            "intervention_discrimination".to_string(),
            MetricValue::new(intervention_sum / n),
        );
        metrics.insert(
            "backdoor_accuracy".to_string(),
            MetricValue::new(backdoor_sum / n),
        );
        metrics.insert(
            "frontdoor_accuracy".to_string(),
            MetricValue::new(frontdoor_sum / n),
        );
        metrics.insert(
            "causal_score".to_string(),
            MetricValue::new(causal_sum / n),
        );

        BenchmarkResult {
            benchmark_name: self.name().to_string(),
            domain: "causal_reasoning".to_string(),
            metrics,
            trials,
            config: config.clone(),
            label: config.label.clone(),
        }
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
        assert_eq!(result.domain, "causal_reasoning");
        assert!(result.metrics.contains_key("causal_score"));
    }

    #[test]
    fn test_intervention_discrimination_positive() {
        let bench = InterventionEffectBenchmark;
        let result = bench.run(&default_config());
        let discrim = result.metrics["intervention_discrimination"].value;
        assert!(
            discrim > 0.0,
            "Intervention should produce discriminable residuals, got {discrim}"
        );
    }

    #[test]
    fn test_causal_score_bounded() {
        let bench = InterventionEffectBenchmark;
        let result = bench.run(&default_config());
        let score = result.metrics["causal_score"].value;
        assert!(score >= 0.0 && score <= 1.0, "Score should be in [0,1], got {score}");
    }

    #[test]
    fn test_provenance_present() {
        let bench = InterventionEffectBenchmark;
        let prov = bench.provenance().expect("Should have provenance");
        assert_eq!(prov.year, 2009);
        assert!(prov.citation.contains("Pearl"));
    }
}
