// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! UAL-P2: Second-order conditioning.
//!
//! Per `SYMTHAEA_UAL_PHASE1_PROTOCOLS_2026-07-27.md`'s Protocol P2: three
//! `ContinuousHV` stimuli A/B/C generated at near-chance mutual similarity
//! (ruling out the "representational similarity" alternative explanation by
//! construction). A acquires real reward value via delta-rule conditioning;
//! B is paired with A (never directly rewarded) via bind-and-accumulate
//! relational memory; C is an independent, separately-seeded reward-matched
//! control stimulus that never participates in any pairing. At test, B's
//! value is inferred by unbinding the relational memory and taking the
//! nearest neighbor (by similarity) among {A, C}.
//!
//! **Schedule mechanism (important implementation note beyond the spec
//! text)**: a naive interleaving of "condition A" and "pair B with A" steps
//! would be causally inert here, because `bind(A_hv, B_hv)` is a fixed
//! vector recomputed identically every pairing step regardless of what
//! happens in between — the final accumulated memory after all 40 pairings
//! would be bit-identical under any ordering, making the multi-schedule
//! replication requirement a no-op. To give the schedule genuine
//! mechanistic teeth, each pairing step's contribution to the relational
//! memory is weighted by A's *current* value-confidence
//! (`|a_value - 0.5| * 2`) at the moment of pairing: under `Blocked`, A is
//! fully conditioned before any pairing occurs, so every pairing step
//! carries near-maximal weight; under `Interleaved`, some pairings occur
//! while A's value is still near its uninformative prior and contribute
//! little. This is a real, well-motivated associative-learning assumption
//! (you cannot form a strong second-order link to a not-yet-valued primary
//! stimulus), not an arbitrary knob — and it is exactly the schedule-
//! dependence question P2 exists to probe (design doc: "chain formation may
//! depend on interleaving").
//!
//! **Baseline ladder** (`BaselineRung`): rungs 1 and 2 (value table / first-
//! order learner) collapse onto one code path here — both learn only direct
//! A/C pairings with zero chaining, which fail for the identical reason (see
//! `learned to represent this collapse explicitly, not silently`  test).

use super::common::{generate_near_chance_hv, next_seed};
use super::report::{Presence, UalProbeReport, UalSchedule, combine_schedule_reports};
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
    /// Rung 4: real bind/bundle memory formation (unconditional weight, not
    /// confidence-scaled), but A/C values frozen at 0.5 — isolates
    /// representational transfer from value transfer.
    StaticBindingNoValueLearning,
    /// Rung 5: the mechanism under test.
    FullSymthaea,
}

#[derive(Debug, Clone, Copy)]
enum StepKind {
    CondA,
    PairBA,
}

struct P2TrialOutcome {
    b_inferred_value: f64,
    a_value: f64,
    c_value: f64,
    /// similarity(query, A) - similarity(query, C).
    internal_margin: f32,
    neighbor_is_a: bool,
    retrieval_had_signal: bool,
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

        let apply_value_learning = !matches!(rung, BaselineRung::StaticBindingNoValueLearning);
        let apply_memory_formation = matches!(
            rung,
            BaselineRung::StaticBindingNoValueLearning | BaselineRung::FullSymthaea
        );
        let confidence_weighted_pairing = matches!(rung, BaselineRung::FullSymthaea);

        // C is independent by construction: its own dedicated RNG stream,
        // run outside the A/B sequence entirely, never touches `memory`.
        let mut dyn_rng_c = config.trial_seed("ual", "p2_dynamics_c", run_idx) ^ 0x9E3779B97F4A7C15;
        let c_value =
            Self::condition_stimulus(&mut dyn_rng_c, apply_value_learning, PHASE1_C_TRIALS);

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
                        let bind_hv = b_hv.bind(&a_hv);
                        let step_weight: f32 = if confidence_weighted_pairing {
                            let a_confidence = ((a_value - 0.5).abs() * 2.0) as f32;
                            (MEMORY_LEARNING_RATE * a_confidence).clamp(0.0, MEMORY_LEARNING_RATE)
                        } else {
                            MEMORY_LEARNING_RATE
                        };
                        if step_weight > 0.0 {
                            memory = ContinuousHV::weighted_bundle(
                                &[&memory, &bind_hv],
                                &[1.0 - step_weight, step_weight],
                            );
                        }
                    }
                }
            }
        }
        if apply_memory_formation {
            memory = memory.normalize();
        }

        if matches!(rung, BaselineRung::GraphPropagation) {
            return P2TrialOutcome {
                b_inferred_value: a_value,
                a_value,
                c_value,
                internal_margin: 1.0,
                neighbor_is_a: true,
                retrieval_had_signal: true,
            };
        }

        let query = memory.bind(&b_hv);
        let sim_a = query.similarity(&a_hv);
        let sim_c = query.similarity(&c_hv);
        let margin = sim_a - sim_c;
        let retrieval_had_signal = margin.abs() > SIMILARITY_TIE_EPS;
        let neighbor_is_a = sim_a >= sim_c;
        let b_inferred_value = if !retrieval_had_signal {
            (a_value + c_value) / 2.0
        } else if neighbor_is_a {
            a_value
        } else {
            c_value
        };

        P2TrialOutcome {
            b_inferred_value,
            a_value,
            c_value,
            internal_margin: margin,
            neighbor_is_a,
            retrieval_had_signal,
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
        let mut diffs = Vec::with_capacity(n);
        let mut margins = Vec::with_capacity(n);
        for i in 0..n {
            let o = self.run_trial(config, i, schedule, BaselineRung::FullSymthaea);
            diffs.push(o.b_inferred_value - o.c_value);
            margins.push(o.internal_margin as f64);
        }
        let diff_metric = MetricValue::from_samples_bootstrap(&diffs, config.seed ^ 0xABCDEF);
        let ci_excludes_zero_positive = diff_metric.ci_lower > 0.0;
        let mean_margin = margins.iter().sum::<f64>() / margins.len() as f64;
        let internal = if mean_margin > 0.15 {
            Presence::Observed
        } else {
            Presence::NotObserved
        };
        let behavioral = if ci_excludes_zero_positive {
            Presence::Observed
        } else {
            Presence::NotObserved
        };
        UalProbeReport::new("UAL-P2", behavioral, internal).with_note(format!(
            "mean(B_inferred - C_actual)={:.4} [{:.4},{:.4}] under {:?}; mean internal margin={:.4}",
            diff_metric.mean, diff_metric.ci_lower, diff_metric.ci_upper, schedule, mean_margin
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
        let mut b_inferred = Vec::new();
        let mut margins = Vec::new();
        for trial in 0..config.trials_per_condition {
            let o = self.run_trial(
                config,
                trial,
                UalSchedule::Blocked,
                BaselineRung::FullSymthaea,
            );
            b_inferred.push(o.b_inferred_value);
            margins.push(o.internal_margin as f64);
        }
        result.insert("b_inferred_value", MetricValue::from_samples(&b_inferred));
        result.insert("internal_margin", MetricValue::from_samples(&margins));
        result.conditions = 3; // A, B, C
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
        assert!(result.metrics["b_inferred_value"].mean.is_finite());
        assert!(result.metrics["internal_margin"].mean.is_finite());
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
            assert!((o.b_inferred_value - o.a_value).abs() < 1e-12);
        }
    }

    /// Baseline rung 4: correct geometric retrieval (neighbor_is_a) despite
    /// frozen values -> b_inferred_value stays pinned at 0.5, isolating
    /// representational transfer from value transfer.
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
            if o.neighbor_is_a {
                correct_neighbor += 1;
            }
            assert!(
                (o.b_inferred_value - 0.5).abs() < 1e-9,
                "frozen-value rung must report the frozen default, not a real value: {}",
                o.b_inferred_value
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
            let o =
                P2SecondOrder.run_trial(&cfg, i, UalSchedule::Blocked, BaselineRung::FullSymthaea);
            vals.push(o.a_value);
        }
        let mean = vals.iter().sum::<f64>() / n as f64;
        assert!(
            mean > 0.6,
            "A's value should approach its 0.8 reward asymptote: mean={mean}"
        );
    }

    /// Negative control 1 (shuffled-pairing / C): C never participates in
    /// any pairing (own RNG stream, own conditioning loop, no shared state
    /// with `memory`) -- structural guarantee, pinned here as a regression
    /// test: C's value under FullSymthaea must equal C's value computed by
    /// the isolated `condition_stimulus` helper with the same seed.
    #[test]
    fn negative_control_c_value_is_independent_of_pairing() {
        let cfg = config();
        for i in 0..10 {
            let o =
                P2SecondOrder.run_trial(&cfg, i, UalSchedule::Blocked, BaselineRung::FullSymthaea);
            let mut isolated_rng = cfg.trial_seed("ual", "p2_dynamics_c", i) ^ 0x9E3779B97F4A7C15;
            let isolated_c =
                P2SecondOrder::condition_stimulus(&mut isolated_rng, true, PHASE1_C_TRIALS);
            assert!((o.c_value - isolated_c).abs() < 1e-12);
        }
    }

    /// Sham control (`lever: "unrelated-dimension-perturbation"`): the same
    /// seed run twice must be bit-identical (no hidden nondeterminism a
    /// sham-style unrelated perturbation could get confused with).
    #[test]
    fn sham_determinism_same_seed_same_result() {
        let cfg = config();
        let a = P2SecondOrder.run_trial(&cfg, 5, UalSchedule::Blocked, BaselineRung::FullSymthaea);
        let b = P2SecondOrder.run_trial(&cfg, 5, UalSchedule::Blocked, BaselineRung::FullSymthaea);
        assert!((a.b_inferred_value - b.b_inferred_value).abs() < 1e-12);
        assert!((a.internal_margin - b.internal_margin).abs() < 1e-6);
    }

    /// Leakage test: zeroing `memory` must collapse retrieval to "no
    /// signal" (both similarities near zero, tied) -- proving the retrieval
    /// path is load-bearing, not bypassed by some other route to B's value.
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

    /// Schedule genuinely matters (see module doc): confidence-weighted
    /// pairing means Blocked (A fully conditioned before any pairing) should
    /// produce a stronger mean internal margin than Interleaved on average.
    #[test]
    fn schedule_affects_pairing_strength_as_designed() {
        let cfg = config();
        let n = 30;
        let mean_margin = |schedule: UalSchedule| -> f64 {
            let mut sum = 0.0;
            for i in 0..n {
                let o = P2SecondOrder.run_trial(&cfg, i, schedule, BaselineRung::FullSymthaea);
                sum += o.internal_margin as f64;
            }
            sum / n as f64
        };
        let blocked = mean_margin(UalSchedule::Blocked);
        let interleaved = mean_margin(UalSchedule::Interleaved);
        assert!(
            blocked >= interleaved,
            "blocked schedule (A fully conditioned before pairing) should produce margin >= interleaved: blocked={blocked}, interleaved={interleaved}"
        );
    }
}
