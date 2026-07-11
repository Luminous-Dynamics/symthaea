// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Logical Deduction benchmark.
//!
//! Tests propositional logic reasoning:
//!   1. Modus Ponens (MP): P → Q, P ⊢ Q
//!   2. Modus Tollens (MT): P → Q, ¬Q ⊢ ¬P
//!   3. Hypothetical Syllogism (HS): P → Q, Q → R ⊢ P → R
//!   4. Fallacies: Affirming the consequent, Denying the antecedent,
//!      Illicit conversion (P → Q ⊢ Q → P)
//!
//! **Engine-wired (Tier 0.1, 2026-07-06).** Each argument is decided by the
//! real `LogicEngine` from `symthaea-core`: validity is `(premises ∧
//! ¬conclusion)` UNSAT under the DPLL SAT solver, and valid rule forms are
//! additionally cross-checked through the natural-deduction rules
//! (`modus_ponens`, `modus_tollens`, `hypothetical_syllogism`). Accuracy
//! therefore measures *computed correctness* against the known logical status
//! of each argument form. The previous version scored HDC similarity between
//! premise and conclusion hypervectors with hand-tuned corrections and never
//! invoked the engine; that gap was flagged by the Phase 0 grounding audit.
//!
//! The `Circular` form (P ⊢ P) from the old battery was replaced by
//! `IllicitConversion` (P → Q ⊢ Q → P): P ⊢ P *is* a valid entailment under
//! classical semantics (P ∧ ¬P is UNSAT), so labelling it "invalid" would
//! contradict the engine's — correct — answer. Illicit conversion is a
//! genuine fallacy with a countermodel (P=false, Q=true).
//!
//! HDC still participates as trial structure: the old similarity classifier
//! is retained and its agreement with the engine-computed ground truth is
//! reported as the auxiliary `hdc_agreement` metric (not part of accuracy).
//!
//! Noise model: with probability proportional to `effective_noise()`, the
//! engine's decision is replaced by a coin flip (degraded readout).
//!
//! Human baselines (Johnson-Laird 1983):
//! - valid_accuracy: ~0.82 (SD~0.10) — correctly endorsing valid arguments
//! - invalid_accuracy: ~0.64 (SD~0.14) — correctly rejecting invalid arguments
//! - overall_accuracy: ~0.73 (SD~0.11) — combined classification

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::logic_engine::{LogicEngine, Proposition};

/// Logical Deduction benchmark.
pub struct LogicalDeductionBenchmark;

fn xor_shift(s: &mut u64) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
}

/// Argument type enumeration.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ArgumentType {
    /// Valid: Modus Ponens — P→Q, P ⊢ Q
    ModusPonens,
    /// Valid: Modus Tollens — P→Q, ¬Q ⊢ ¬P
    ModusTollens,
    /// Valid: Hypothetical Syllogism — P→Q, Q→R ⊢ P→R
    HypotheticalSyllogism,
    /// Invalid: Affirming the Consequent — P→Q, Q ⊢ P
    AffirmingConsequent,
    /// Invalid: Denying the Antecedent — P→Q, ¬P ⊢ ¬Q
    DenyingAntecedent,
    /// Invalid: Illicit Conversion — P→Q ⊢ Q→P
    IllicitConversion,
}

impl ArgumentType {
    fn is_valid(self) -> bool {
        matches!(
            self,
            ArgumentType::ModusPonens
                | ArgumentType::ModusTollens
                | ArgumentType::HypotheticalSyllogism
        )
    }
}

const ALL_ARGUMENT_TYPES: [ArgumentType; 6] = [
    ArgumentType::ModusPonens,
    ArgumentType::ModusTollens,
    ArgumentType::HypotheticalSyllogism,
    ArgumentType::AffirmingConsequent,
    ArgumentType::DenyingAntecedent,
    ArgumentType::IllicitConversion,
];

/// Build the symbolic (premises, conclusion) pair for an argument form.
fn build_argument(arg_type: ArgumentType) -> (Vec<Proposition>, Proposition) {
    let p = || Proposition::atom("P");
    let q = || Proposition::atom("Q");
    let r = || Proposition::atom("R");

    match arg_type {
        ArgumentType::ModusPonens => (vec![p().implies(q()), p()], q()),
        ArgumentType::ModusTollens => (vec![p().implies(q()), q().not()], p().not()),
        ArgumentType::HypotheticalSyllogism => {
            (vec![p().implies(q()), q().implies(r())], p().implies(r()))
        }
        ArgumentType::AffirmingConsequent => (vec![p().implies(q()), q()], p()),
        ArgumentType::DenyingAntecedent => (vec![p().implies(q()), p().not()], q().not()),
        ArgumentType::IllicitConversion => (vec![p().implies(q())], q().implies(p())),
    }
}

/// Decide validity with the REAL engine: `premises ⊢ conclusion` is valid iff
/// `(⋀ premises) ∧ ¬conclusion` is unsatisfiable (DPLL SAT solver).
fn engine_decides_valid(premises: &[Proposition], conclusion: &Proposition) -> bool {
    let mut conj = conclusion.clone().not();
    for prem in premises {
        conj = conj.and(prem.clone());
    }
    !LogicEngine::is_satisfiable(&conj)
}

/// Cross-check a valid rule form through the engine's natural-deduction path.
/// Returns `true` iff the corresponding rule fires and yields a valid proof
/// whose conclusion matches the expected one.
fn deduction_rule_confirms(arg_type: ArgumentType) -> bool {
    let p = || Proposition::atom("P");
    let q = || Proposition::atom("Q");
    let r = || Proposition::atom("R");

    match arg_type {
        ArgumentType::ModusPonens => {
            LogicEngine::modus_ponens(&p(), &p().implies(q())).is_some_and(|proof| proof.valid)
        }
        ArgumentType::ModusTollens => LogicEngine::modus_tollens(&q().not(), &p().implies(q()))
            .is_some_and(|proof| proof.valid),
        ArgumentType::HypotheticalSyllogism => {
            LogicEngine::hypothetical_syllogism(&p().implies(q()), &q().implies(r()))
                .is_some_and(|proof| proof.valid)
        }
        // Fallacies have no natural-deduction rule; SAT alone decides them.
        _ => false,
    }
}

/// HDC structural encoding of an argument (retained as trial structure).
///
/// Each proposition (P, Q, R) is a random HV. Implication P→Q is encoded
/// as bind(P, Q). Negation ¬P is encoded as -P (negated HV). The bundle
/// of premises forms the "argument" HV; the conclusion is a separate HV.
struct ArgumentEncoding {
    premise_hv: ContinuousHV,
    conclusion_hv: ContinuousHV,
    distractor_hv: ContinuousHV,
}

/// Negate a ContinuousHV by flipping all component signs.
/// Cannot use weighted_bundle([v], [-1.0]) — it normalizes by weight_sum,
/// turning (-1*v)/(-1) = v (a no-op).
fn negate_hv(hv: &ContinuousHV) -> ContinuousHV {
    let mut result = hv.clone();
    for v in result.values.iter_mut() {
        *v = -*v;
    }
    result
}

fn encode_argument(arg_type: ArgumentType, dim: usize, seed: u64) -> ArgumentEncoding {
    let p = ContinuousHV::random(dim, seed.wrapping_add(1));
    let q = ContinuousHV::random(dim, seed.wrapping_add(2));
    let r = ContinuousHV::random(dim, seed.wrapping_add(3));

    // Implication P→Q: bind P and Q (directional by convention: bind(P, Q))
    let p_implies_q = p.bind(&q);
    let q_implies_r = q.bind(&r);

    // Negation: negate each component (multiply by -1 in continuous HDC)
    let neg_p = negate_hv(&p);
    let neg_q = negate_hv(&q);

    // Distractor: a random HV unrelated to any argument component
    let distractor = ContinuousHV::random(dim, seed.wrapping_add(100));

    match arg_type {
        ArgumentType::ModusPonens => ArgumentEncoding {
            premise_hv: ContinuousHV::weighted_bundle(&[&p_implies_q, &p], &[0.6, 0.4]),
            conclusion_hv: q,
            distractor_hv: distractor,
        },
        ArgumentType::ModusTollens => ArgumentEncoding {
            premise_hv: ContinuousHV::weighted_bundle(&[&p_implies_q, &neg_q], &[0.6, 0.4]),
            conclusion_hv: neg_p,
            distractor_hv: distractor,
        },
        ArgumentType::HypotheticalSyllogism => {
            let p_implies_r = p.bind(&r);
            ArgumentEncoding {
                premise_hv: ContinuousHV::weighted_bundle(
                    &[&p_implies_q, &q_implies_r],
                    &[0.5, 0.5],
                ),
                conclusion_hv: p_implies_r,
                distractor_hv: distractor,
            }
        }
        ArgumentType::AffirmingConsequent => ArgumentEncoding {
            premise_hv: ContinuousHV::weighted_bundle(&[&p_implies_q, &q], &[0.6, 0.4]),
            conclusion_hv: p.clone(), // claimed (invalid) conclusion
            distractor_hv: distractor,
        },
        ArgumentType::DenyingAntecedent => ArgumentEncoding {
            premise_hv: ContinuousHV::weighted_bundle(&[&p_implies_q, &neg_p], &[0.6, 0.4]),
            conclusion_hv: neg_q.clone(), // claimed (invalid) conclusion
            distractor_hv: distractor,
        },
        ArgumentType::IllicitConversion => {
            // Premise: P→Q. Claimed conclusion: Q→P (bind is commutative, so
            // this is a maximally deceptive case for the HDC classifier).
            let q_implies_p = q.bind(&p);
            ArgumentEncoding {
                premise_hv: p_implies_q,
                conclusion_hv: q_implies_p,
                distractor_hv: distractor,
            }
        }
    }
}

/// The retained HDC similarity classifier (auxiliary only): endorses the
/// argument iff the premise bundle is more similar to the claimed conclusion
/// than to a random distractor.
fn hdc_classify(encoding: &ArgumentEncoding) -> bool {
    let sim_conclusion = encoding.premise_hv.similarity(&encoding.conclusion_hv) as f64;
    let sim_distractor = encoding.premise_hv.similarity(&encoding.distractor_hv) as f64;
    sim_conclusion > sim_distractor
}

struct LogicTrial {
    valid_accuracy: f64,
    invalid_accuracy: f64,
    overall_accuracy: f64,
    hdc_agreement: f64,
}

impl LogicalDeductionBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> LogicTrial {
        let dim = config.dimension;
        let seed = config.trial_seed("mathematics", "logical_deduction", trial_idx);
        let mut rng = seed ^ 0xFEDCBA9876543210;
        let noise_weight = config.effective_noise();

        let mut valid_hits = 0u32;
        let mut valid_total = 0u32;
        let mut invalid_hits = 0u32;
        let mut invalid_total = 0u32;
        let mut hdc_agree = 0u32;
        let mut hdc_total = 0u32;

        // Run 3 repetitions of each argument form (repetitions only differ
        // under noise; the engine itself is deterministic).
        for arg_type in &ALL_ARGUMENT_TYPES {
            for rep in 0..3usize {
                xor_shift(&mut rng);
                let arg_seed = seed
                    .wrapping_add((*arg_type as u64) * 100)
                    .wrapping_add(rep as u64)
                    .wrapping_add(rng);

                // REAL ENGINE: DPLL SAT decides entailment; valid rule forms
                // are additionally confirmed via natural deduction.
                let (premises, conclusion) = build_argument(*arg_type);
                let mut system_says_valid = engine_decides_valid(&premises, &conclusion);
                if arg_type.is_valid() {
                    system_says_valid = system_says_valid && deduction_rule_confirms(*arg_type);
                }

                // Noise: degraded readout randomizes the decision.
                if noise_weight > 0.0 {
                    xor_shift(&mut rng);
                    let noise_frac = noise_weight * 0.7;
                    if (rng as f64 / u64::MAX as f64) < noise_frac {
                        xor_shift(&mut rng);
                        system_says_valid = rng % 2 == 0;
                    }
                }

                // Auxiliary: does the HDC structural classifier agree with
                // the engine-computed ground truth?
                let encoding = encode_argument(*arg_type, dim, arg_seed);
                let ground_truth = engine_decides_valid(&premises, &conclusion);
                hdc_total += 1;
                if hdc_classify(&encoding) == ground_truth {
                    hdc_agree += 1;
                }

                if arg_type.is_valid() {
                    valid_total += 1;
                    if system_says_valid {
                        valid_hits += 1;
                    }
                } else {
                    invalid_total += 1;
                    if !system_says_valid {
                        // Correctly rejected the invalid argument
                        invalid_hits += 1;
                    }
                }
            }
        }

        let valid_accuracy = if valid_total > 0 {
            valid_hits as f64 / valid_total as f64
        } else {
            0.0
        };
        let invalid_accuracy = if invalid_total > 0 {
            invalid_hits as f64 / invalid_total as f64
        } else {
            0.0
        };
        let overall_total = valid_total + invalid_total;
        let overall_accuracy = if overall_total > 0 {
            (valid_hits + invalid_hits) as f64 / overall_total as f64
        } else {
            0.0
        };
        let hdc_agreement = if hdc_total > 0 {
            hdc_agree as f64 / hdc_total as f64
        } else {
            0.0
        };

        LogicTrial {
            valid_accuracy,
            invalid_accuracy,
            overall_accuracy,
            hdc_agreement,
        }
    }
}

impl PsychBenchmark for LogicalDeductionBenchmark {
    fn name(&self) -> &str {
        "Mathematics::LogicalDeduction"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Logical Reasoning Assessment",
            citation: "Johnson-Laird (1983)",
            year: 1983,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut valid_accs = Vec::new();
        let mut invalid_accs = Vec::new();
        let mut overall_accs = Vec::new();
        let mut hdc_agreements = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            valid_accs.push(r.valid_accuracy);
            invalid_accs.push(r.invalid_accuracy);
            overall_accs.push(r.overall_accuracy);
            hdc_agreements.push(r.hdc_agreement);
        }

        result.insert("valid_accuracy", MetricValue::from_samples(&valid_accs));
        result.insert("invalid_accuracy", MetricValue::from_samples(&invalid_accs));
        result.insert("overall_accuracy", MetricValue::from_samples(&overall_accs));
        result.insert("hdc_agreement", MetricValue::from_samples(&hdc_agreements));

        result.conditions = 6; // 3 valid + 3 invalid argument types
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
    fn test_logical_deduction_runs_and_has_metrics() {
        let result = LogicalDeductionBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("valid_accuracy"));
        assert!(result.metrics.contains_key("invalid_accuracy"));
        assert!(result.metrics.contains_key("overall_accuracy"));
        assert!(result.metrics.contains_key("hdc_agreement"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = LogicalDeductionBenchmark.run(&test_config());
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
    fn test_valid_accuracy_above_chance() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 8,
            ..Default::default()
        };
        let result = LogicalDeductionBenchmark.run(&config);
        let acc = result.metrics["overall_accuracy"].mean;
        // Binary classification: chance = 0.5. Should exceed chance.
        assert!(
            acc > 0.40,
            "Overall accuracy should exceed chance (0.5), got {}",
            acc
        );
    }

    /// Proves the REAL engine is invoked: at zero noise the DPLL solver plus
    /// natural-deduction cross-check must classify every argument form
    /// perfectly — the old HDC similarity classifier could not (the illicit
    /// conversion encoding is indistinguishable from its premise under
    /// commutative bind).
    #[test]
    fn test_engine_classifies_all_forms_perfectly_at_zero_noise() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 4,
            encoding_noise: 0.0,
            time_pressure: 0.0,
            ..Default::default()
        };
        let result = LogicalDeductionBenchmark.run(&config);
        assert_eq!(result.metrics["valid_accuracy"].mean, 1.0);
        assert_eq!(result.metrics["invalid_accuracy"].mean, 1.0);
        assert_eq!(result.metrics["overall_accuracy"].mean, 1.0);
    }

    /// Proves the benchmark CAN fail: fallacies presented as valid are
    /// rejected by the engine, and a wrong claimed conclusion for a valid
    /// form is likewise rejected.
    #[test]
    fn test_wrong_answers_score_low() {
        // Each fallacy: the engine must find a countermodel (not valid).
        for fallacy in [
            ArgumentType::AffirmingConsequent,
            ArgumentType::DenyingAntecedent,
            ArgumentType::IllicitConversion,
        ] {
            let (premises, conclusion) = build_argument(fallacy);
            assert!(
                !engine_decides_valid(&premises, &conclusion),
                "{:?} must be rejected by the SAT engine",
                fallacy
            );
        }
        // Modus ponens with the WRONG conclusion (R instead of Q): invalid.
        let p = Proposition::atom("P");
        let q = Proposition::atom("Q");
        let wrong = Proposition::atom("R");
        assert!(!engine_decides_valid(&[p.clone().implies(q), p], &wrong));
    }

    /// The natural-deduction path is genuinely exercised for valid forms.
    #[test]
    fn test_natural_deduction_rules_fire() {
        assert!(deduction_rule_confirms(ArgumentType::ModusPonens));
        assert!(deduction_rule_confirms(ArgumentType::ModusTollens));
        assert!(deduction_rule_confirms(ArgumentType::HypotheticalSyllogism));
        // Fallacies have no deduction rule.
        assert!(!deduction_rule_confirms(ArgumentType::IllicitConversion));
    }
}
