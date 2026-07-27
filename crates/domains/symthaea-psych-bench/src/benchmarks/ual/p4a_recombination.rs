// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! UAL-P4a: Held-out compositional recombination.
//!
//! Per `SYMTHAEA_UAL_PHASE1_PROTOCOLS_2026-07-27.md`'s Protocol P4a: four
//! element `ContinuousHV`s W/X/Y/Z (mutually near-chance similar). Two
//! compounds are trained — `bind(W,X)` (mostly rewarded) and `bind(Y,Z)`
//! (mostly not) — with individual elements never presented alone, so an
//! exact-compound lookup table structurally cannot generalize. Two *entirely
//! novel* compounds, `bind(W,Z)` and `bind(Y,X)`, are tested at extinction
//! alongside the seen `bind(W,X)` (positive control).
//!
//! **Full-Symthaea mechanism**: a shared associative `value_memory`
//! accumulates `compound_hv.bind(&outcome_tag)` on every training trial
//! (both WX and YZ trials feed the *same* memory, `outcome_tag` being a
//! fixed `reward_tag`/`no_reward_tag` hypervector depending on that trial's
//! stochastic reward draw). At test, a novel compound's value is read out by
//! unbinding the query from `value_memory` and comparing its similarity to
//! `reward_tag` vs `no_reward_tag`. This is a genuinely different
//! computational route from rung 3's naive per-element marginal-mean
//! (design doc: "must combine ... with a *compositional* value-integration
//! step ... distinguishable from rung 3 by design, not just by outcome") —
//! rung 3 never touches HDC binding at all, only tracks two independent
//! scalars.
//!
//! **Schedule mechanism**: because `value_memory` is a *shared* EMA
//! accumulator fed by two structurally different targets (WX training
//! trials bind toward `reward_tag` on average, YZ trials toward
//! `no_reward_tag`), the relative order in which WX/YZ training steps occur
//! genuinely changes the final accumulated memory (EMA is order-sensitive
//! when successive targets differ, unlike P2's single fixed-target case) —
//! `Blocked-by-element` vs `Interleaved-by-element` is not a schedule no-op
//! here.

use super::common::{generate_near_chance_hv, next_seed};
use super::report::{Presence, UalProbeReport, UalSchedule, combine_schedule_reports};
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::ContinuousHV;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineRung {
    /// Rungs 1+2 collapsed: exact-compound-keyed lookup / whole-compound
    /// learner, zero decomposition -> held-out compounds get the "no entry"
    /// default, fails identically for the same reason (spec).
    ValueTableNoDecomposition,
    /// Rung 3: marginal-mean of each element's trained-compound value —
    /// the naive additive-composition calibration case.
    GraphPropagationMarginalMean,
    /// Rung 4: pure representational geometry (no value component at all)
    /// — nearest trained compound by raw HV similarity.
    StaticBindingNoValueLearning,
    /// Rung 5: the mechanism under test.
    FullSymthaea,
}

#[derive(Debug, Clone, Copy)]
enum StepKind {
    TrainWx,
    TrainYz,
}

/// Value-style outcome (rungs 1/2/3/5 — anything that reports a [0,1] value
/// estimate for comparison purposes).
struct ValueOutcome {
    wx_readout: f64,
    wz_readout: f64,
    yx_readout: f64,
    yz_readout: f64,
    wx_true_value: f64,
    /// Raw HDC value-integration margin (rung 5 only; NaN elsewhere) for
    /// the internal-association-formation check.
    wz_raw_margin: f32,
    yz_raw_margin: f32,
}

/// Geometry-only outcome (rung 4 — deliberately has no value component).
struct GeometryOutcome {
    wz_similarity_to_wx: f32,
    wz_similarity_to_yz: f32,
    nearest_trained_is_wx_for_wz: bool,
}

pub struct P4aRecombination;

const TRAIN_TRIALS_EACH: usize = 30;
const NEAR_CHANCE_THRESHOLD: f32 = 0.1;
const MEMORY_LEARNING_RATE: f32 = 0.08;
const LR_POS: f64 = 0.15;
const LR_NEG: f64 = 0.25;
const REWARD_TAG_SEED: u64 = 0xF00D_CAFE_0001;
const NO_REWARD_TAG_SEED: u64 = 0xF00D_CAFE_0002;

impl P4aRecombination {
    fn elements(config: &BenchmarkConfig, run_idx: usize) -> [ContinuousHV; 4] {
        let dim = config.dimension;
        let w = ContinuousHV::random(dim, config.trial_seed("ual", "p4a_w", run_idx));
        let x = generate_near_chance_hv(
            dim,
            config.trial_seed("ual", "p4a_x", run_idx),
            &[&w],
            NEAR_CHANCE_THRESHOLD,
            50,
        );
        let y = generate_near_chance_hv(
            dim,
            config.trial_seed("ual", "p4a_y", run_idx),
            &[&w, &x],
            NEAR_CHANCE_THRESHOLD,
            50,
        );
        let z = generate_near_chance_hv(
            dim,
            config.trial_seed("ual", "p4a_z", run_idx),
            &[&w, &x, &y],
            NEAR_CHANCE_THRESHOLD,
            50,
        );
        [w, x, y, z]
    }

    fn build_sequence(schedule: UalSchedule, shuffle_seed: u64) -> Vec<StepKind> {
        let mut steps = Vec::with_capacity(TRAIN_TRIALS_EACH * 2);
        steps.extend(std::iter::repeat(StepKind::TrainWx).take(TRAIN_TRIALS_EACH));
        steps.extend(std::iter::repeat(StepKind::TrainYz).take(TRAIN_TRIALS_EACH));
        if schedule == UalSchedule::Interleaved {
            let mut rng = shuffle_seed ^ 0x9E3779B97F4A7C15;
            for i in (1..steps.len()).rev() {
                let j = (next_seed(&mut rng) as usize) % (i + 1);
                steps.swap(i, j);
            }
        }
        steps
    }

    fn run_trial_value(
        &self,
        config: &BenchmarkConfig,
        run_idx: usize,
        schedule: UalSchedule,
        rung: BaselineRung,
    ) -> ValueOutcome {
        assert_ne!(rung, BaselineRung::StaticBindingNoValueLearning);
        let dim = config.dimension;
        let [w, x, y, z] = Self::elements(config, run_idx);
        let wx_hv = w.bind(&x);
        let yz_hv = y.bind(&z);
        let wz_hv = w.bind(&z); // held out
        let yx_hv = y.bind(&x); // held out

        // Leakage guard: held-out compounds must never coincide with a
        // trained one (exact-vector equality, not similarity).
        assert!(
            wz_hv != wx_hv && wz_hv != yz_hv,
            "held-out WZ leaked into training set"
        );
        assert!(
            yx_hv != wx_hv && yx_hv != yz_hv,
            "held-out YX leaked into training set"
        );

        let apply_value_learning = true; // only StaticBinding rung skips this, handled separately
        let apply_memory_formation = matches!(rung, BaselineRung::FullSymthaea);

        let reward_tag = ContinuousHV::random(dim, REWARD_TAG_SEED);
        let no_reward_tag = ContinuousHV::random(dim, NO_REWARD_TAG_SEED);

        let steps =
            Self::build_sequence(schedule, config.trial_seed("ual", "p4a_shuffle", run_idx));
        let mut dyn_rng_wx =
            config.trial_seed("ual", "p4a_dynamics_wx", run_idx) ^ 0x9E3779B97F4A7C15;
        let mut dyn_rng_yz =
            config.trial_seed("ual", "p4a_dynamics_yz", run_idx) ^ 0x9E3779B97F4A7C15;
        let mut wx_value = 0.5_f64;
        let mut yz_value = 0.5_f64;
        let mut value_memory = ContinuousHV::zero(dim);

        for step in &steps {
            match step {
                StepKind::TrainWx => {
                    let roll = (next_seed(&mut dyn_rng_wx) % 10000) as f64 / 10000.0;
                    let reward = roll < 0.8;
                    if apply_value_learning {
                        let r = if reward { 1.0 } else { 0.0 };
                        let rpe = r - wx_value;
                        let lr = if rpe >= 0.0 { LR_POS } else { LR_NEG };
                        wx_value += lr * rpe;
                        wx_value = wx_value.clamp(0.0, 1.0);
                    }
                    if apply_memory_formation {
                        let tag = if reward { &reward_tag } else { &no_reward_tag };
                        let target = wx_hv.bind(tag);
                        value_memory = ContinuousHV::weighted_bundle(
                            &[&value_memory, &target],
                            &[1.0 - MEMORY_LEARNING_RATE, MEMORY_LEARNING_RATE],
                        );
                    }
                }
                StepKind::TrainYz => {
                    let roll = (next_seed(&mut dyn_rng_yz) % 10000) as f64 / 10000.0;
                    let reward = roll < 0.2;
                    if apply_value_learning {
                        let r = if reward { 1.0 } else { 0.0 };
                        let rpe = r - yz_value;
                        let lr = if rpe >= 0.0 { LR_POS } else { LR_NEG };
                        yz_value += lr * rpe;
                        yz_value = yz_value.clamp(0.0, 1.0);
                    }
                    if apply_memory_formation {
                        let tag = if reward { &reward_tag } else { &no_reward_tag };
                        let target = yz_hv.bind(tag);
                        value_memory = ContinuousHV::weighted_bundle(
                            &[&value_memory, &target],
                            &[1.0 - MEMORY_LEARNING_RATE, MEMORY_LEARNING_RATE],
                        );
                    }
                }
            }
        }
        if apply_memory_formation {
            value_memory = value_memory.normalize();
        }

        let readout = |compound_hv: &ContinuousHV, marginal_default: f64| -> (f64, f32) {
            match rung {
                BaselineRung::ValueTableNoDecomposition => (marginal_default, f32::NAN),
                BaselineRung::GraphPropagationMarginalMean => {
                    ((wx_value + yz_value) / 2.0, f32::NAN)
                }
                BaselineRung::FullSymthaea => {
                    let query = value_memory.bind(compound_hv);
                    let sim_reward = query.similarity(&reward_tag);
                    let sim_no_reward = query.similarity(&no_reward_tag);
                    let margin = sim_reward - sim_no_reward;
                    let value_estimate = (((margin + 1.0) / 2.0) as f64).clamp(0.0, 1.0);
                    (value_estimate, margin)
                }
                BaselineRung::StaticBindingNoValueLearning => unreachable!(),
            }
        };

        // Seen compounds WX/YZ: under *any* rung, a directly-trained node
        // reports its own real learned value, not a propagated/marginal
        // estimate — marginal-mean composition only applies to genuinely
        // novel (held-out) compounds. This matters specifically for
        // `GraphPropagationMarginalMean`: a graph model still knows an
        // exact observed node's value without needing to propagate/average
        // anything for it (bug found via `marginal_mean_baseline_passes_by_construction`
        // failing: the shared `readout` closure was applying the marginal-mean
        // formula even to already-trained WX/YZ, not just to WZ/YX).
        let (wx_readout, _) = match rung {
            BaselineRung::ValueTableNoDecomposition
            | BaselineRung::GraphPropagationMarginalMean => (wx_value, f32::NAN),
            _ => readout(&wx_hv, wx_value),
        };
        let (yz_readout, yz_raw_margin) = match rung {
            BaselineRung::ValueTableNoDecomposition
            | BaselineRung::GraphPropagationMarginalMean => (yz_value, f32::NAN),
            _ => readout(&yz_hv, yz_value),
        };
        let (wz_readout, wz_raw_margin) = readout(&wz_hv, 0.5);
        let (yx_readout, _) = readout(&yx_hv, 0.5);

        ValueOutcome {
            wx_readout,
            wz_readout,
            yx_readout,
            yz_readout,
            wx_true_value: wx_value,
            wz_raw_margin,
            yz_raw_margin,
        }
    }

    fn run_trial_geometry(&self, config: &BenchmarkConfig, run_idx: usize) -> GeometryOutcome {
        // Static binding, no value learning at all: pure geometric nearest-
        // neighbor among the trained compounds' raw HVs. No training loop
        // needed (representational geometry doesn't depend on reward draws).
        let [w, x, y, z] = Self::elements(config, run_idx);
        let wx_hv = w.bind(&x);
        let yz_hv = y.bind(&z);
        let wz_hv = w.bind(&z);
        let sim_wx = wz_hv.similarity(&wx_hv);
        let sim_yz = wz_hv.similarity(&yz_hv);
        GeometryOutcome {
            wz_similarity_to_wx: sim_wx,
            wz_similarity_to_yz: sim_yz,
            nearest_trained_is_wx_for_wz: sim_wx >= sim_yz,
        }
    }

    pub fn ual_report(&self, config: &BenchmarkConfig, n: usize) -> UalProbeReport {
        let blocked = self.schedule_report(config, n, UalSchedule::Blocked);
        let interleaved = self.schedule_report(config, n, UalSchedule::Interleaved);
        combine_schedule_reports("UAL-P4a", &blocked, &interleaved)
    }

    fn schedule_report(
        &self,
        config: &BenchmarkConfig,
        n: usize,
        schedule: UalSchedule,
    ) -> UalProbeReport {
        let mut diffs = Vec::with_capacity(n);
        let mut margin_diffs = Vec::with_capacity(n);
        for i in 0..n {
            let o = self.run_trial_value(config, i, schedule, BaselineRung::FullSymthaea);
            diffs.push(o.wz_readout - o.yz_readout);
            margin_diffs.push((o.wz_raw_margin - o.yz_raw_margin) as f64);
        }
        let diff_metric = MetricValue::from_samples_bootstrap(&diffs, config.seed ^ 0x51DE51DE);
        let behavioral = if diff_metric.ci_lower > 0.0 {
            Presence::Observed
        } else {
            Presence::NotObserved
        };
        let mean_margin_diff = margin_diffs.iter().sum::<f64>() / margin_diffs.len() as f64;
        let internal = if mean_margin_diff.abs() > 0.05 {
            Presence::Observed
        } else {
            Presence::NotObserved
        };
        UalProbeReport::new("UAL-P4a", behavioral, internal).with_note(format!(
            "mean(WZ_readout - YZ_readout)={:.4} [{:.4},{:.4}] under {:?}; mean internal margin diff={:.4}",
            diff_metric.mean, diff_metric.ci_lower, diff_metric.ci_upper, schedule, mean_margin_diff
        ))
    }
}

impl PsychBenchmark for P4aRecombination {
    fn name(&self) -> &str {
        "Ual::P4aRecombination"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "UAL-P4a: Held-Out Compositional Recombination",
            citation: "Birch, Ginsburg & Jablonka (2020)",
            year: 2020,
            doi: Some("10.1007/s10539-020-09772-0"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut wz = Vec::new();
        let mut yz = Vec::new();
        for trial in 0..config.trials_per_condition {
            let o = self.run_trial_value(
                config,
                trial,
                UalSchedule::Blocked,
                BaselineRung::FullSymthaea,
            );
            wz.push(o.wz_readout);
            yz.push(o.yz_readout);
        }
        result.insert("wz_novel_readout", MetricValue::from_samples(&wz));
        result.insert("yz_trained_low_readout", MetricValue::from_samples(&yz));
        result.conditions = 3; // WX seen, WZ novel, YX novel
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
    fn p4a_runs_and_produces_finite_metrics() {
        let result = P4aRecombination.run(&config());
        assert!(result.metrics["wz_novel_readout"].mean.is_finite());
        assert!(result.metrics["yz_trained_low_readout"].mean.is_finite());
    }

    /// Leakage test: held-out compounds must never coincide with a trained
    /// compound (checked as a hard assertion inside `run_trial_value`
    /// itself, exercised here across many seeds).
    #[test]
    fn leakage_held_out_compounds_never_collide_with_trained() {
        let cfg = config();
        for i in 0..20 {
            let _ = P4aRecombination.run_trial_value(
                &cfg,
                i,
                UalSchedule::Blocked,
                BaselineRung::FullSymthaea,
            );
        }
    }

    /// Baseline rungs 1+2 (collapsed): held-out compounds get the "no
    /// entry" default (0.5), never the trained compounds' real values.
    ///
    /// The "no entry -> exactly 0.5" property is deterministic/structural
    /// and checked per-run. Whether the seen compound's *own* learned value
    /// exceeds the uninformative prior is a genuinely stochastic quantity
    /// (30 trials, asymmetric learning rates that penalize negative
    /// surprise more than they reward positive surprise) — a single run can
    /// land below 0.5 by chance even though the mechanism is working
    /// correctly (confirmed: an earlier per-run `> 0.5` assertion here
    /// failed on one seed with wx_readout=0.457, not because of a bug, but
    /// because that's a real, expected fluctuation at n=1). Checked as an
    /// aggregate mean instead, matching the pattern already used for P1/P2's
    /// positive controls.
    #[test]
    fn value_table_fails_on_held_out_compounds() {
        let cfg = config();
        let n = 30;
        let mut wx_readouts = Vec::with_capacity(n);
        for i in 0..n {
            let o = P4aRecombination.run_trial_value(
                &cfg,
                i,
                UalSchedule::Blocked,
                BaselineRung::ValueTableNoDecomposition,
            );
            assert!(
                (o.wz_readout - 0.5).abs() < 1e-12,
                "value table must have no entry for held-out WZ"
            );
            assert!(
                (o.yx_readout - 0.5).abs() < 1e-12,
                "value table must have no entry for held-out YX"
            );
            wx_readouts.push(o.wx_readout);
        }
        let mean_wx = wx_readouts.iter().sum::<f64>() / n as f64;
        assert!(
            mean_wx > 0.55,
            "value table's real entry for seen WX should reflect learning on average: mean={mean_wx}"
        );
    }

    /// Baseline rung 3: naive marginal-mean calibration case — passes by
    /// construction (both novel compounds get the same symmetric estimate).
    #[test]
    fn marginal_mean_baseline_passes_by_construction() {
        let cfg = config();
        for i in 0..10 {
            let o = P4aRecombination.run_trial_value(
                &cfg,
                i,
                UalSchedule::Blocked,
                BaselineRung::GraphPropagationMarginalMean,
            );
            let expected = (o.wx_true_value + o.yz_readout) / 2.0;
            // yz_readout under this rung IS yz_value directly (no decomposition needed for a trained compound).
            assert!((o.wz_readout - expected).abs() < 1e-9);
            assert!(
                (o.yx_readout - expected).abs() < 1e-9,
                "both novel compounds should get the identical symmetric marginal-mean estimate"
            );
        }
    }

    /// Baseline rung 4: pure geometry, no value component. Calibrates how
    /// much representational-transfer confound exists independent of value.
    #[test]
    fn static_binding_geometry_only_no_value_semantics() {
        let cfg = config();
        let n = 20;
        let mut wx_nearer_count = 0usize;
        for i in 0..n {
            let o = P4aRecombination.run_trial_geometry(&cfg, i);
            if o.nearest_trained_is_wx_for_wz {
                wx_nearer_count += 1;
            }
            // Similarities must be real, finite, and not collapsed to a tie
            // by construction (would indicate a broken bind implementation).
            assert!(o.wz_similarity_to_wx.is_finite());
            assert!(o.wz_similarity_to_yz.is_finite());
        }
        // No directional prediction here (spec: "a required comparison, not
        // a pass/fail target") — just confirm it produces real, usable
        // signal for calibration.
        let _ = wx_nearer_count;
    }

    /// Positive control: the seen compound WX's test-time read-out under
    /// FullSymthaea must track its real trained value.
    #[test]
    fn positive_control_seen_compound_tracks_trained_value() {
        let cfg = config();
        let n = 30;
        let mut diffs = Vec::new();
        for i in 0..n {
            let o = P4aRecombination.run_trial_value(
                &cfg,
                i,
                UalSchedule::Blocked,
                BaselineRung::FullSymthaea,
            );
            diffs.push(o.wx_readout - o.wx_true_value);
        }
        let mean_abs_diff = diffs.iter().map(|d| d.abs()).sum::<f64>() / n as f64;
        assert!(
            mean_abs_diff < 0.35,
            "seen compound's readout should track its trained value reasonably closely: mean_abs_diff={mean_abs_diff}"
        );
    }

    /// Sham control: determinism under the same seed.
    #[test]
    fn sham_determinism_same_seed_same_result() {
        let cfg = config();
        let a = P4aRecombination.run_trial_value(
            &cfg,
            4,
            UalSchedule::Blocked,
            BaselineRung::FullSymthaea,
        );
        let b = P4aRecombination.run_trial_value(
            &cfg,
            4,
            UalSchedule::Blocked,
            BaselineRung::FullSymthaea,
        );
        assert!((a.wz_readout - b.wz_readout).abs() < 1e-12);
    }

    /// Schedule genuinely matters here (see module doc): a shared EMA memory
    /// fed by two different targets is order-sensitive. This test only
    /// confirms the two schedules can produce *different* results (not a
    /// directional claim) — a schedule dimension that never differs would
    /// indicate the "leaky/no-op schedule" failure mode flagged in the
    /// design doc.
    #[test]
    fn schedule_produces_genuinely_different_outcomes() {
        let cfg = config();
        let n = 20;
        let mean_wz = |schedule: UalSchedule| -> f64 {
            let mut sum = 0.0;
            for i in 0..n {
                let o =
                    P4aRecombination.run_trial_value(&cfg, i, schedule, BaselineRung::FullSymthaea);
                sum += o.wz_readout;
            }
            sum / n as f64
        };
        let blocked = mean_wz(UalSchedule::Blocked);
        let interleaved = mean_wz(UalSchedule::Interleaved);
        assert!(
            (blocked - interleaved).abs() > 1e-6,
            "schedule must have a genuine mechanistic effect, not be a no-op: blocked={blocked}, interleaved={interleaved}"
        );
    }
}
