// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! UAL-P2: Second-order conditioning.
//!
//! Per `SYMTHAEA_UAL_PHASE1_PROTOCOLS_2026-07-27.md`'s Protocol P2, revised in
//! the claim-integrity repair pass (2026-07-27) after an independent review
//! plus this codebase's own direct algebraic verification found two real
//! defects in the original design (both fixed here, not just relabeled):
//!
//! 1. **The original behavioral criterion (`b_inferred_value - c_value > 0`)
//!    was mathematically incapable of detecting success.** A and C are
//!    reward-matched (both ~p=0.8), so `E[a_value] == E[c_value]` — a
//!    *perfectly correct* retrieval of A would still average to zero
//!    difference against C across runs. Replaced with a **retrieval-identity**
//!    criterion (does querying with B rank A above C at a rate above chance?)
//!    as the primary behavioral signal, plus a separate **value-transfer**
//!    check against a fourth, never-conditioned neutral stimulus `D`
//!    (stays at the uninformative prior 0.5) — this is the comparison that
//!    can actually detect "B acquired value", which C-vs-B alone could not.
//!
//! 2. **The original "confidence-weighted pairing" schedule mechanism was
//!    algebraically inert, not just imperfect.** `bind_hv = b_hv.bind(&a_hv)`
//!    is the *same fixed vector* on every pairing step (A_hv/B_hv never
//!    change within a run); accumulating an identical vector via
//!    `weighted_bundle` under *any* sequence of positive weights always
//!    produces a vector collinear with `bind_hv`, and `.normalize()` erases
//!    scale entirely — so `Blocked` vs `Interleaved` are direction-identical
//!    regardless of confidence-weighting. Confirmed by
//!    `schedule_produces_algebraically_identical_normalized_memory` (proof by
//!    executable test, not just derivation) using both the zero and a
//!    nontrivial initial `memory` state. The confidence-weighting has been
//!    **removed** (plain fixed-rate accumulation) rather than retained as an
//!    unexplained knob that doesn't do anything — P2's honest finding is that
//!    this specific mechanism is schedule-invariant by construction, unlike
//!    P4a's (see `p4a_recombination.rs`'s module doc for why P4a's case is
//!    genuinely different: its accumulator mixes two *non-collinear* targets).
//!
//! **System under test**: `SystemUnderTest::BenchmarkLocalHdcLearner`
//! (renamed from the misleading `FullSymthaea` — this mechanism does not
//! construct or call `CognitiveLoopService`, and is not live Symthaea).
//!
//! Three `ContinuousHV` stimuli A/B/C generated at near-chance mutual
//! similarity (ruling out the "representational similarity" alternative
//! explanation by construction), plus a fourth, D, generated the same way
//! and never touched by any conditioning or pairing at all. A acquires real
//! reward value via delta-rule conditioning; B is paired with A (never
//! directly rewarded) via bind-and-accumulate relational memory; C is an
//! independent, separately-seeded reward-matched control stimulus that never
//! participates in any pairing.
//!
//! **Retrieval mechanism caveat** (from the claim-integrity repair pass'
//! bounded HDC binding/unbinding audit, `hdc_binding_properties.rs`):
//! querying with `.bind(&b_hv)` rather than `.bind(&b_hv.inverse())` does not
//! perform textbook VSA unbinding — it works via a real, reproducible,
//! *directionally correct* self-squared-term bias (`sim(A⊗B², A)` has a
//! systematically positive expectation; `sim(A⊗B², C)` does not, for
//! independent zero-mean components) rather than clean algebraic recovery of
//! A. This is disclosed, not hidden: see the audit file for the measured
//! properties of this crate's actual `ContinuousHV::random` distribution.

use super::common::{generate_near_chance_hv, next_seed};
use super::report::{
    FunctionalOutcome, Presence, SystemUnderTest, UalProbeReport, UalRuntimeQualification,
    UalSchedule, combine_schedule_reports,
};
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::ContinuousHV;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineRung {
    /// Rungs 1+2 collapsed: real per-stimulus (A/C) value learning, zero
    /// relational chaining at all — B never gets any stored value.
    ValueTableNoChaining,
    /// Rung 3: explicit graph edge B->A, value copied directly (calibration
    /// case — passes by construction, no "understanding" required).
    GraphPropagation,
    /// Rung 4: real bind/bundle memory formation, but A/C values frozen at
    /// 0.5 — isolates representational transfer from value transfer.
    StaticBindingNoValueLearning,
    /// Rung 5: the benchmark-local candidate mechanism under test. NOT live
    /// Symthaea (see module doc; formerly misleadingly named `FullSymthaea`).
    CandidateHdcLearner,
}

#[derive(Debug, Clone, Copy)]
enum StepKind {
    CondA,
    PairBA,
}

struct P2TrialOutcome {
    a_value: f64,
    c_value: f64,
    d_value: f64,
    /// similarity(query, A) - similarity(query, C).
    margin_a_vs_c: f32,
    /// similarity(query, A) - similarity(query, D). D is never conditioned
    /// or paired at all, so this isolates "B carries a signal at all" from
    /// "B specifically resembles A more than C" (both reward-matched).
    margin_a_vs_d: f32,
    neighbor_is_a_over_c: bool,
    retrieval_had_signal: bool,
    memory_norm: f32,
}

pub struct P2SecondOrder;

const PHASE1_A_TRIALS: usize = 40;
const PHASE1_C_TRIALS: usize = 40;
const PHASE2_TRIALS: usize = 40;
const SIMILARITY_TIE_EPS: f32 = 0.05;
const NEAR_CHANCE_THRESHOLD: f32 = 0.1;
const MEMORY_LEARNING_RATE: f32 = 0.1;
const LR_POS: f64 = 0.15;
const LR_NEG: f64 = 0.25;

impl P2SecondOrder {
    fn condition_stimulus(dyn_rng: &mut u64, apply_value_learning: bool, n_trials: usize) -> f64 {
        let mut value = 0.5_f64;
        for _ in 0..n_trials {
            let roll = (next_seed(dyn_rng) % 10000) as f64 / 10000.0;
            let reward = if roll < 0.8 { 1.0 } else { 0.0 };
            if apply_value_learning {
                let rpe = reward - value;
                let lr = if rpe >= 0.0 { LR_POS } else { LR_NEG };
                value += lr * rpe;
                value = value.clamp(0.0, 1.0);
            }
        }
        value
    }

    fn build_ab_sequence(schedule: UalSchedule, shuffle_seed: u64) -> Vec<StepKind> {
        let mut steps = Vec::with_capacity(PHASE1_A_TRIALS + PHASE2_TRIALS);
        steps.extend(std::iter::repeat(StepKind::CondA).take(PHASE1_A_TRIALS));
        steps.extend(std::iter::repeat(StepKind::PairBA).take(PHASE2_TRIALS));
        if schedule == UalSchedule::Interleaved {
            let mut rng = shuffle_seed ^ 0x9E3779B97F4A7C15;
            for i in (1..steps.len()).rev() {
                let j = (next_seed(&mut rng) as usize) % (i + 1);
                steps.swap(i, j);
            }
        }
        steps
    }

    fn run_trial(
        &self,
        config: &BenchmarkConfig,
        run_idx: usize,
        schedule: UalSchedule,
        rung: BaselineRung,
    ) -> P2TrialOutcome {
        let dim = config.dimension;
        let a_hv = ContinuousHV::random(dim, config.trial_seed("ual", "p2_stim_a", run_idx));
        let b_hv = generate_near_chance_hv(
            dim,
            config.trial_seed("ual", "p2_stim_b", run_idx),
            &[&a_hv],
            NEAR_CHANCE_THRESHOLD,
            50,
        );
        let c_hv = generate_near_chance_hv(
            dim,
            config.trial_seed("ual", "p2_stim_c", run_idx),
            &[&a_hv, &b_hv],
            NEAR_CHANCE_THRESHOLD,
            50,
        );
        let d_hv = generate_near_chance_hv(
            dim,
            config.trial_seed("ual", "p2_stim_d", run_idx),
            &[&a_hv, &b_hv, &c_hv],
            NEAR_CHANCE_THRESHOLD,
            50,
        );

        let apply_value_learning = !matches!(rung, BaselineRung::StaticBindingNoValueLearning);
        let apply_memory_formation = matches!(
            rung,
            BaselineRung::StaticBindingNoValueLearning | BaselineRung::CandidateHdcLearner
        );

        // C and D are independent by construction: dedicated RNG streams,
        // run outside the A/B sequence entirely, never touch `memory`. D is
        // additionally never conditioned at all (stays at the uninformative
        // prior), unlike C which earns its own real, reward-matched value.
        let mut dyn_rng_c = config.trial_seed("ual", "p2_dynamics_c", run_idx) ^ 0x9E3779B97F4A7C15;
        let c_value =
            Self::condition_stimulus(&mut dyn_rng_c, apply_value_learning, PHASE1_C_TRIALS);
        let d_value = 0.5_f64;

        let steps =
            Self::build_ab_sequence(schedule, config.trial_seed("ual", "p2_shuffle", run_idx));
        let mut dyn_rng_a = config.trial_seed("ual", "p2_dynamics_a", run_idx) ^ 0x9E3779B97F4A7C15;
        let mut a_value = 0.5_f64;
        let mut memory = ContinuousHV::zero(dim);

        for step in &steps {
            match step {
                StepKind::CondA => {
                    let roll = (next_seed(&mut dyn_rng_a) % 10000) as f64 / 10000.0;
                    let reward = if roll < 0.8 { 1.0 } else { 0.0 };
                    if apply_value_learning {
                        let rpe = reward - a_value;
                        let lr = if rpe >= 0.0 { LR_POS } else { LR_NEG };
                        a_value += lr * rpe;
                        a_value = a_value.clamp(0.0, 1.0);
                    }
                }
                StepKind::PairBA => {
                    if apply_memory_formation {
                        // Fixed-rate accumulation (no confidence-weighting —
                        // see module doc: weighting was proven algebraically
                        // inert for this single-fixed-target accumulator).
                        let bind_hv = b_hv.bind(&a_hv);
                        memory = ContinuousHV::weighted_bundle(
                            &[&memory, &bind_hv],
                            &[1.0 - MEMORY_LEARNING_RATE, MEMORY_LEARNING_RATE],
                        );
                    }
                }
            }
        }
        let memory_norm_pre = memory.norm();
        if apply_memory_formation {
            memory = memory.normalize();
        }

        if matches!(rung, BaselineRung::GraphPropagation) {
            return P2TrialOutcome {
                a_value,
                c_value,
                d_value,
                margin_a_vs_c: 1.0,
                margin_a_vs_d: 1.0,
                neighbor_is_a_over_c: true,
                retrieval_had_signal: true,
                memory_norm: memory_norm_pre,
            };
        }

        let query = memory.bind(&b_hv);
        let sim_a = query.similarity(&a_hv);
        let sim_c = query.similarity(&c_hv);
        let sim_d = query.similarity(&d_hv);
        let margin_a_vs_c = sim_a - sim_c;
        let margin_a_vs_d = sim_a - sim_d;
        let retrieval_had_signal = margin_a_vs_c.abs() > SIMILARITY_TIE_EPS;
        let neighbor_is_a_over_c = sim_a >= sim_c;

        P2TrialOutcome {
            a_value,
            c_value,
            d_value,
            margin_a_vs_c,
            margin_a_vs_d,
            neighbor_is_a_over_c,
            retrieval_had_signal,
            memory_norm: memory_norm_pre,
        }
    }

    pub fn ual_report(&self, config: &BenchmarkConfig, n: usize) -> UalProbeReport {
        let blocked = self.schedule_report(config, n, UalSchedule::Blocked);
        let interleaved = self.schedule_report(config, n, UalSchedule::Interleaved);
        combine_schedule_reports("UAL-P2", &blocked, &interleaved)
    }

    fn schedule_report(
        &self,
        config: &BenchmarkConfig,
        n: usize,
        schedule: UalSchedule,
    ) -> UalProbeReport {
        let mut retrieval_correct = Vec::with_capacity(n);
        let mut margin_a_vs_d = Vec::with_capacity(n);
        let mut a_values = Vec::with_capacity(n);
        let mut memory_norms = Vec::with_capacity(n);
        for i in 0..n {
            let o = self.run_trial(config, i, schedule, BaselineRung::CandidateHdcLearner);
            retrieval_correct.push(if o.neighbor_is_a_over_c { 1.0 } else { 0.0 });
            margin_a_vs_d.push(o.margin_a_vs_d as f64);
            a_values.push(o.a_value);
            memory_norms.push(o.memory_norm as f64);
        }
        let retrieval_metric =
            MetricValue::from_samples_bootstrap(&retrieval_correct, config.seed ^ 0xABCDEF);
        // Primary behavioral criterion: retrieval-identity rate above chance
        // (0.5), CI-excluding-chance rather than a point estimate.
        let behavioral_expression = if retrieval_metric.ci_lower > 0.5 {
            Presence::Observed
        } else {
            Presence::NotObserved
        };
        let mean_margin_ad = margin_a_vs_d.iter().sum::<f64>() / margin_a_vs_d.len() as f64;
        let internal = if mean_margin_ad > 0.05 {
            Presence::Observed
        } else {
            Presence::NotObserved
        };

        let mean_a_value = a_values.iter().sum::<f64>() / a_values.len() as f64;
        let positive_control_passed = mean_a_value > 0.6;
        let mean_memory_norm = memory_norms.iter().sum::<f64>() / memory_norms.len() as f64;
        let manipulation_applied = mean_memory_norm > 1e-6;

        let blocked_steps = Self::build_ab_sequence(UalSchedule::Blocked, config.seed);
        let interleaved_steps = Self::build_ab_sequence(UalSchedule::Interleaved, config.seed);
        let schedule_valid = format!("{blocked_steps:?}") != format!("{interleaved_steps:?}");

        let qualification = UalRuntimeQualification {
            manipulation_applied,
            positive_control_passed,
            // D's own value never moves (never conditioned, never paired) —
            // structural guarantee by construction (d_value is a literal
            // constant in `run_trial`), not re-derived per run.
            negative_controls_passed: true,
            // Structural: `leakage_zeroing_memory_collapses_retrieval_signal`
            // pins that zeroing `memory` collapses retrieval to no-signal.
            leakage_checks_passed: true,
            // Covered by dedicated rung tests
            // (`value_table_no_chaining_shows_no_signal_for_b`,
            // `graph_propagation_copies_a_value_exactly`).
            baseline_ladder_valid: true,
            signal_usable: n > 0 && retrieval_metric.mean.is_finite(),
            schedule_valid,
        };

        UalProbeReport::new(
            "UAL-P2",
            SystemUnderTest::BenchmarkLocalHdcLearner,
            qualification,
            behavioral_expression,
            internal,
        )
        .with_note(format!(
            "retrieval-identity(A over C) rate={:.4} [{:.4},{:.4}] under {:?}; mean(sim_A - sim_D)={:.4}; mean A_value={:.4}",
            retrieval_metric.mean, retrieval_metric.ci_lower, retrieval_metric.ci_upper, schedule, mean_margin_ad, mean_a_value
        ))
    }
}

impl PsychBenchmark for P2SecondOrder {
    fn name(&self) -> &str {
        "Ual::P2SecondOrder"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "UAL-P2: Second-Order Conditioning",
            citation: "Birch, Ginsburg & Jablonka (2020); Rizley & Rescorla (1972)",
            year: 2020,
            doi: Some("10.1007/s10539-020-09772-0"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut retrieval = Vec::new();
        let mut margins = Vec::new();
        for trial in 0..config.trials_per_condition {
            let o = self.run_trial(
                config,
                trial,
                UalSchedule::Blocked,
                BaselineRung::CandidateHdcLearner,
            );
            retrieval.push(if o.neighbor_is_a_over_c { 1.0 } else { 0.0 });
            margins.push(o.margin_a_vs_d as f64);
        }
        result.insert(
            "retrieval_identity_rate",
            MetricValue::from_samples(&retrieval),
        );
        result.insert("margin_a_vs_d", MetricValue::from_samples(&margins));
        result.conditions = 4; // A, B, C, D
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BenchmarkConfig {
        BenchmarkConfig {
            trials_per_condition: 30,
            dimension: 512,
            ..Default::default()
        }
    }

    #[test]
    fn p2_runs_and_produces_finite_metrics() {
        let result = P2SecondOrder.run(&config());
        assert!(result.metrics["retrieval_identity_rate"].mean.is_finite());
        assert!(result.metrics["margin_a_vs_d"].mean.is_finite());
    }

    /// Stimuli must be near-chance similar to each other by construction.
    #[test]
    fn stimuli_are_near_chance_similar() {
        let cfg = config();
        let a = ContinuousHV::random(cfg.dimension, cfg.trial_seed("ual", "p2_stim_a", 0));
        let b = generate_near_chance_hv(
            cfg.dimension,
            cfg.trial_seed("ual", "p2_stim_b", 0),
            &[&a],
            NEAR_CHANCE_THRESHOLD,
            50,
        );
        assert!(a.similarity(&b).abs() < NEAR_CHANCE_THRESHOLD);
    }

    /// Baseline rung 1+2 (collapsed): no relational chaining at all -> B's
    /// retrieval carries no signal (margin near zero, no drift toward A).
    #[test]
    fn value_table_no_chaining_shows_no_signal_for_b() {
        let cfg = config();
        let n = 30;
        let mut had_signal = 0usize;
        for i in 0..n {
            let o = P2SecondOrder.run_trial(
                &cfg,
                i,
                UalSchedule::Blocked,
                BaselineRung::ValueTableNoChaining,
            );
            if o.retrieval_had_signal {
                had_signal += 1;
            }
        }
        assert!(
            had_signal as f64 / (n as f64) < 0.2,
            "value-table rung should show retrieval signal on B in only a small minority of runs by chance: {had_signal}/{n}"
        );
    }

    /// Baseline rung 3 (graph propagation): passes by construction.
    #[test]
    fn graph_propagation_copies_a_value_exactly() {
        let cfg = config();
        for i in 0..10 {
            let o = P2SecondOrder.run_trial(
                &cfg,
                i,
                UalSchedule::Blocked,
                BaselineRung::GraphPropagation,
            );
            assert!(o.neighbor_is_a_over_c);
        }
    }

    /// Baseline rung 4: correct geometric retrieval (neighbor_is_a_over_c)
    /// despite frozen values -> isolates representational transfer from
    /// value transfer.
    #[test]
    fn static_binding_retrieves_correct_neighbor_but_value_stays_frozen() {
        let cfg = config();
        let n = 30;
        let mut correct_neighbor = 0usize;
        for i in 0..n {
            let o = P2SecondOrder.run_trial(
                &cfg,
                i,
                UalSchedule::Blocked,
                BaselineRung::StaticBindingNoValueLearning,
            );
            if o.neighbor_is_a_over_c {
                correct_neighbor += 1;
            }
            assert!(
                (o.a_value - 0.5).abs() < 1e-9,
                "frozen-value rung must report the frozen default, not a real value: {}",
                o.a_value
            );
        }
        assert!(
            correct_neighbor as f64 / n as f64 > 0.7,
            "static binding should still correctly identify A as the nearest neighbor most of the time: {correct_neighbor}/{n}"
        );
    }

    /// Positive control (`id: "p2-direct-pairing-signal"`): A's own direct
    /// conditioning must reach a reward-matching asymptote.
    #[test]
    fn positive_control_a_conditioning_reaches_asymptote() {
        let cfg = config();
        let n = 30;
        let mut vals = Vec::new();
        for i in 0..n {
            let o = P2SecondOrder.run_trial(
                &cfg,
                i,
                UalSchedule::Blocked,
                BaselineRung::CandidateHdcLearner,
            );
            vals.push(o.a_value);
        }
        let mean = vals.iter().sum::<f64>() / n as f64;
        assert!(
            mean > 0.6,
            "A's value should approach its 0.8 reward asymptote: mean={mean}"
        );
    }

    /// Negative control 1 (shuffled-pairing / C): C never participates in
    /// any pairing -- structural guarantee, pinned as a regression test.
    #[test]
    fn negative_control_c_value_is_independent_of_pairing() {
        let cfg = config();
        for i in 0..10 {
            let o = P2SecondOrder.run_trial(
                &cfg,
                i,
                UalSchedule::Blocked,
                BaselineRung::CandidateHdcLearner,
            );
            let mut isolated_rng = cfg.trial_seed("ual", "p2_dynamics_c", i) ^ 0x9E3779B97F4A7C15;
            let isolated_c =
                P2SecondOrder::condition_stimulus(&mut isolated_rng, true, PHASE1_C_TRIALS);
            assert!((o.c_value - isolated_c).abs() < 1e-12);
        }
    }

    /// Negative control 2 (neutral D): D is never conditioned or paired at
    /// all -- must stay pinned at the uninformative prior.
    #[test]
    fn negative_control_d_value_never_moves_from_prior() {
        let cfg = config();
        for i in 0..10 {
            let o = P2SecondOrder.run_trial(
                &cfg,
                i,
                UalSchedule::Blocked,
                BaselineRung::CandidateHdcLearner,
            );
            assert!((o.d_value - 0.5).abs() < 1e-12);
        }
    }

    /// Sham control: same seed run twice must be bit-identical.
    #[test]
    fn sham_determinism_same_seed_same_result() {
        let cfg = config();
        let a = P2SecondOrder.run_trial(
            &cfg,
            5,
            UalSchedule::Blocked,
            BaselineRung::CandidateHdcLearner,
        );
        let b = P2SecondOrder.run_trial(
            &cfg,
            5,
            UalSchedule::Blocked,
            BaselineRung::CandidateHdcLearner,
        );
        assert_eq!(a.neighbor_is_a_over_c, b.neighbor_is_a_over_c);
        assert!((a.margin_a_vs_c - b.margin_a_vs_c).abs() < 1e-6);
    }

    /// Leakage test: zeroing `memory` must collapse retrieval to "no
    /// signal".
    #[test]
    fn leakage_zeroing_memory_collapses_retrieval_signal() {
        let cfg = config();
        let dim = cfg.dimension;
        let a_hv = ContinuousHV::random(dim, cfg.trial_seed("ual", "p2_stim_a", 0));
        let b_hv = generate_near_chance_hv(
            dim,
            cfg.trial_seed("ual", "p2_stim_b", 0),
            &[&a_hv],
            NEAR_CHANCE_THRESHOLD,
            50,
        );
        let c_hv = generate_near_chance_hv(
            dim,
            cfg.trial_seed("ual", "p2_stim_c", 0),
            &[&a_hv, &b_hv],
            NEAR_CHANCE_THRESHOLD,
            50,
        );
        let zero_memory = ContinuousHV::zero(dim);
        let query = zero_memory.bind(&b_hv);
        let sim_a = query.similarity(&a_hv);
        let sim_c = query.similarity(&c_hv);
        assert!(
            (sim_a - sim_c).abs() < SIMILARITY_TIE_EPS,
            "zeroed memory must produce no discriminating signal: sim_a={sim_a}, sim_c={sim_c}"
        );
    }

    /// **Claim-integrity repair pass — algebraic proof, not just narrative.**
    /// `bind_hv` is the same fixed vector on every `PairBA` step within a
    /// run, so accumulating it via `weighted_bundle` under *any* positive
    /// weight sequence yields a vector collinear with `bind_hv`; normalizing
    /// erases scale. Blocked and Interleaved must therefore produce
    /// direction-identical normalized memory (up to float rounding), from
    /// BOTH a zero start and a nontrivial pre-seeded start (ruling out the
    /// "residual initialization contamination" caveat: if only the
    /// nonzero-initialization case differed, that would indicate
    /// initialization persistence, not schedule-sensitive pairing).
    #[test]
    fn schedule_produces_algebraically_identical_normalized_memory() {
        let cfg = config();
        let dim = cfg.dimension;
        let run_idx = 3;
        let a_hv = ContinuousHV::random(dim, cfg.trial_seed("ual", "p2_stim_a", run_idx));
        let b_hv = generate_near_chance_hv(
            dim,
            cfg.trial_seed("ual", "p2_stim_b", run_idx),
            &[&a_hv],
            NEAR_CHANCE_THRESHOLD,
            50,
        );
        let bind_hv = b_hv.bind(&a_hv);

        let accumulate = |schedule: UalSchedule, initial: ContinuousHV| -> ContinuousHV {
            let steps = P2SecondOrder::build_ab_sequence(
                schedule,
                cfg.trial_seed("ual", "p2_shuffle", run_idx),
            );
            let mut memory = initial;
            for step in &steps {
                if matches!(step, StepKind::PairBA) {
                    memory = ContinuousHV::weighted_bundle(
                        &[&memory, &bind_hv],
                        &[1.0 - MEMORY_LEARNING_RATE, MEMORY_LEARNING_RATE],
                    );
                }
            }
            memory.normalize()
        };

        // Case 1: zero initialization (matches production `run_trial`).
        let blocked_zero = accumulate(UalSchedule::Blocked, ContinuousHV::zero(dim));
        let interleaved_zero = accumulate(UalSchedule::Interleaved, ContinuousHV::zero(dim));
        assert!(
            blocked_zero.similarity(&interleaved_zero) > 1.0 - 1e-4,
            "zero-init: blocked/interleaved normalized memory should be direction-identical, sim={}",
            blocked_zero.similarity(&interleaved_zero)
        );

        // Case 2: nontrivial pre-seeded initialization — if schedule
        // differences appeared ONLY here, that would mean initialization
        // persistence, not confidence-sensitive pairing (the caveat this
        // test is designed to also rule out).
        let seeded_init = ContinuousHV::random(dim, 0xDEADBEEF);
        let blocked_seeded = accumulate(UalSchedule::Blocked, seeded_init.clone());
        let interleaved_seeded = accumulate(UalSchedule::Interleaved, seeded_init);
        assert!(
            blocked_seeded.similarity(&interleaved_seeded) > 1.0 - 1e-4,
            "seeded-init: blocked/interleaved normalized memory should also be direction-identical, sim={}",
            blocked_seeded.similarity(&interleaved_seeded)
        );
    }

    #[test]
    fn ual_report_produces_valid_three_field_report() {
        let cfg = config();
        let report = P2SecondOrder.ual_report(&cfg, 40);
        if report.functional_outcome == FunctionalOutcome::Demonstrated {
            assert_eq!(report.behavioral_expression, Presence::Observed);
        }
        assert!(
            !report.notes.is_empty(),
            "ual_report must always attach at least one schedule note"
        );
    }
}
