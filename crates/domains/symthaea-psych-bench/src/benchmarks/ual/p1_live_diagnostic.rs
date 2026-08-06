// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! UAL-P1 live-Symthaea reversal-learning diagnostic pilot.
//!
//! **This is deliberately an exploratory pilot, NOT a formal UAL probe
//! verdict.** Every other file in `benchmarks/ual/` runs against a
//! benchmark-local candidate HDC learner
//! (`SystemUnderTest::BenchmarkLocalHdcLearner`) and reports through the
//! full fail-closed `UalProbeReport`/preregistered-floor machinery. This
//! file instead drives the REAL `CognitiveLoopService` (`SystemUnderTest::
//! LiveSymthaea`, reserved in `report.rs` for exactly this use, previously
//! unused by any probe) via the already-existing, already-proven
//! `harness::live_runner::CognitiveLoopBenchmarkRunner` — used elsewhere for
//! Butlin-suite live-integration tests — to answer a cheaper, prior
//! question: does she already show any hint of reversal-learning in her
//! real substrate, before any decision to invest in building new UAL
//! capacity? A clean or messy result here licenses at most "diagnostic
//! pilot observed/did not observe a hint of X" — never "UAL demonstrated,"
//! never a claim backed by the preregistered-floor rigor the rest of this
//! module uses. If this pilot looks promising, that is itself the trigger
//! to scope a real preregistered follow-up (floor, power analysis, control
//! ladder) matching the rest of UAL's rigor — not a decision made here.
//!
//! **Task design**: two near-orthogonal arm hypervectors, a single fixed
//! neutral trial cue fed every trial via `cycle_with_hv` (the interesting
//! dynamics are expected to come from the CfC network's own persisting
//! recurrent state across trials, not from varying the cue), and a reward
//! contingency that favors arm 0 for the first half of trials then flips to
//! favor arm 1 (the reversal). `CognitiveLoopBenchmarkRunner::run_benchmark`
//! auto-delivers reward per trial based on whether the network's own
//! (deterministic, cosine-similarity-based, no explicit exploration/
//! temperature) response matched the currently-correct arm.
//!
//! **Scope**: P1 only. P2 (second-order conditioning) and P4a
//! (compositional generalization) both need stimuli-without-reward phases
//! that `run_benchmark`'s always-reward-every-trial contract can't express
//! without new production-harness work — explicitly deferred, not attempted
//! here.
//!
//! Requires the `symthaea-backend` feature (already used by `cogbench`/
//! `butlin`; already dormant in CI — no workflow passes it, so this only
//! ever builds/runs manually, consistent with the existing convention).

#![cfg(feature = "symthaea-backend")]

use super::common::generate_near_chance_hv;
use crate::harness::config::BenchmarkConfig;
use crate::harness::live_runner::{LoopDrivable, LoopStimulus};
use symthaea_core::hdc::ContinuousHV;

/// A live-Symthaea reversal-learning pilot task: `n_trials` total, reward
/// contingency flips from favoring arm 0 to favoring arm 1 exactly at the
/// midpoint.
pub struct P1LiveReversal {
    pub n_trials: usize,
}

impl LoopDrivable for P1LiveReversal {
    fn generate_stimuli(&self, config: &BenchmarkConfig) -> Vec<LoopStimulus> {
        let dim = config.dimension;
        let arm_a = ContinuousHV::random(dim, config.trial_seed("ual_live", "p1_arm_a", 0));
        let arm_b = generate_near_chance_hv(
            dim,
            config.trial_seed("ual_live", "p1_arm_b", 0),
            &[&arm_a],
            0.1,
            50,
        );
        let cue = ContinuousHV::random(dim, config.trial_seed("ual_live", "p1_cue", 0));
        let half = self.n_trials / 2;

        (0..self.n_trials)
            .map(|i| {
                let correct_idx = if i < half { 0 } else { 1 };
                LoopStimulus {
                    stimulus_hv: cue.clone(),
                    alternatives: vec![arm_a.clone(), arm_b.clone()],
                    correct_idx,
                    condition: if i < half {
                        "pre_reversal".to_string()
                    } else {
                        "post_reversal".to_string()
                    },
                    trial_idx: i,
                }
            })
            .collect()
    }

    fn loop_name(&self) -> &str {
        "UAL-P1-Live-Reversal-Pilot"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::live_runner::CognitiveLoopBenchmarkRunner;

    /// Sanity check only: `generate_stimuli` produces the expected trial
    /// count and reward-contingency flip, without needing a live
    /// `CognitiveLoopService` at all. Runs under default `cargo test`
    /// (no `symthaea-backend`-specific construction cost).
    #[test]
    fn generates_expected_trial_structure() {
        let bench = P1LiveReversal { n_trials: 40 };
        let config = BenchmarkConfig {
            dimension: 128,
            ..Default::default()
        };
        let stimuli = bench.generate_stimuli(&config);
        assert_eq!(stimuli.len(), 40);
        assert!(stimuli[0..20].iter().all(|s| s.correct_idx == 0));
        assert!(stimuli[20..40].iter().all(|s| s.correct_idx == 1));
        // All trials share the same cue and alternatives (only the reward
        // contingency changes) -- the interesting dynamics must come from
        // the network's own state, not from varying the stimulus content.
        assert!(
            stimuli
                .iter()
                .all(|s| s.stimulus_hv.similarity(&stimuli[0].stimulus_hv) > 0.999)
        );
    }

    /// **Exploratory pilot — requires the full `CognitiveLoopService`
    /// construction, too slow for default `cargo test`.** Run manually:
    /// `cargo test -p symthaea-psych-bench --features symthaea-backend
    /// --lib benchmarks::ual::p1_live_diagnostic -- --ignored --nocapture`
    ///
    /// Prints per-block accuracy (4 blocks of 50 trials: two pre-reversal,
    /// two post-reversal) plus mean reward/prediction-error/psi across the
    /// full run. See `SYMTHAEA_UAL_LIVE_DIAGNOSTIC_P1_2026-07-28.md` for the
    /// captured result and honest interpretation -- this test asserts
    /// nothing about the outcome itself (unlike the rest of UAL's
    /// preregistered probes), only that the run completes and produces
    /// finite numbers, since this is a pilot, not a confirmatory test.
    #[test]
    #[ignore = "requires full CognitiveLoopService construction; run manually with --ignored --nocapture"]
    fn p1_live_reversal_diagnostic() {
        let mut runner = CognitiveLoopBenchmarkRunner::new("ual_p1_live_diagnostic_seed")
            .expect("failed to construct CognitiveLoopService");
        let dim = runner.service_mut().state_dim();
        let config = BenchmarkConfig {
            dimension: dim,
            seed: 42,
            ..Default::default()
        };
        let bench = P1LiveReversal { n_trials: 200 };
        let result = runner.run_benchmark(&bench, &config);

        assert_eq!(result.trials.len(), 200);
        assert!(result.mean_prediction_error.is_finite());
        assert!(result.mean_reward.is_finite());

        let block_accuracy = |lo: usize, hi: usize| -> f64 {
            let slice = &result.trials[lo..hi];
            slice.iter().filter(|t| t.correct).count() as f64 / slice.len() as f64
        };
        eprintln!("=== UAL-P1 live-Symthaea reversal diagnostic (n=200, dim={dim}) ===");
        eprintln!(
            "block 0 (trials   0- 49, pre-reversal,  arm0 correct):  accuracy={:.4}",
            block_accuracy(0, 50)
        );
        eprintln!(
            "block 1 (trials  50- 99, pre-reversal,  arm0 correct):  accuracy={:.4}",
            block_accuracy(50, 100)
        );
        eprintln!(
            "block 2 (trials 100-149, post-reversal, arm1 correct):  accuracy={:.4}",
            block_accuracy(100, 150)
        );
        eprintln!(
            "block 3 (trials 150-199, post-reversal, arm1 correct):  accuracy={:.4}",
            block_accuracy(150, 200)
        );
        eprintln!(
            "overall accuracy={:.4}; mean_reward={:.4}; mean_prediction_error={:.4}; mean_psi={:.4}",
            result.accuracy,
            result.mean_reward,
            result.mean_prediction_error,
            result.trials.iter().map(|t| t.psi as f64).sum::<f64>() / result.trials.len() as f64
        );
    }

    /// **Root-cause trace for the diagnostic pilot's block 0->1 collapse**
    /// (`SYMTHAEA_UAL_LIVE_DIAGNOSTIC_P1_2026-07-28.md`). Bypasses
    /// `run_benchmark`/`LoopDrivable` entirely and hand-drives the same
    /// sequence via `service_mut()` directly, so every trial's raw
    /// similarity scores (not just the final response/correct bool) are
    /// visible. No production code touched -- this only reads
    /// `CycleResult`/`cosine_similarity_f64`, both already public.
    ///
    /// Same seed/profile/dimension/trial structure as the diagnostic pilot,
    /// so a byte-identical block-accuracy pattern here (vs. the frozen
    /// 0.64/0.00/0.46/0.00 result) is itself a determinism check: does the
    /// SAME configuration reliably reproduce the SAME collapse, or does it
    /// vary run to run despite a fixed seed (which would point at hidden
    /// non-determinism rather than a real, stable dynamical property)?
    ///
    /// **Must replicate `CognitiveLoopBenchmarkRunner::warmup()`'s 100-cycle
    /// pre-trial warmup exactly** (`ContinuousHV::random(dim, 0xDEAD_BEEF)`,
    /// 100 `cycle_with_hv` calls) -- `warmup()` itself is private to
    /// `harness::live_runner` and not reachable from this module. Skipping
    /// it was an earlier bug in this trace (caught by the block-accuracy
    /// mismatch against the frozen result, 0.58/0.98/0.00/0.00 vs the real
    /// 0.64/0.00/0.46/0.00) -- not a finding about Symthaea's own dynamics.
    #[test]
    #[ignore = "requires full CognitiveLoopService construction; run manually with --ignored --nocapture"]
    fn p1_live_reversal_diagnostic_trace() {
        let mut runner = CognitiveLoopBenchmarkRunner::new("ual_p1_live_diagnostic_seed")
            .expect("failed to construct CognitiveLoopService");
        let dim = runner.service_mut().state_dim();
        let config = BenchmarkConfig {
            dimension: dim,
            seed: 42,
            ..Default::default()
        };
        let bench = P1LiveReversal { n_trials: 200 };
        let stimuli = bench.generate_stimuli(&config);

        let warmup_hv = ContinuousHV::random(dim, 0xDEAD_BEEF);
        for _ in 0..100 {
            let _ = runner.service_mut().cycle_with_hv(&warmup_hv);
        }

        eprintln!("=== UAL-P1 live-Symthaea collapse trace (n=200, dim={dim}) ===");
        eprintln!(
            "trial\tcorrect_idx\tresponse\tcorrect\tsim_a\tsim_b\treward\tda\tpe\tpsi\ttraining_loss"
        );

        let mut block_correct = [0u32; 4];
        for stim in &stimuli {
            let result = runner.service_mut().cycle_with_hv(&stim.stimulus_hv);
            let out_f64: Vec<f64> = result.output.iter().map(|&v| v as f64).collect();
            let sim_a = symthaea_core::math::cosine_similarity_f64(
                &out_f64,
                &stim.alternatives[0]
                    .values
                    .iter()
                    .map(|&v| v as f64)
                    .collect::<Vec<_>>(),
            );
            let sim_b = symthaea_core::math::cosine_similarity_f64(
                &out_f64,
                &stim.alternatives[1]
                    .values
                    .iter()
                    .map(|&v| v as f64)
                    .collect::<Vec<_>>(),
            );
            let response = if sim_a >= sim_b { 0 } else { 1 };
            let correct = response == stim.correct_idx;
            let reward = if correct { 0.8 } else { -0.5 };
            runner.service_mut().provide_reward(reward);

            let block = stim.trial_idx / 50;
            if correct {
                block_correct[block] += 1;
            }

            eprintln!(
                "{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:?}",
                stim.trial_idx,
                stim.correct_idx,
                response,
                correct,
                sim_a,
                sim_b,
                reward,
                result.metadata.neuromod.dopamine_effective,
                result.prediction_error,
                result.metadata.consciousness.consciousness_level,
                result.training_loss
            );
        }

        eprintln!("=== block accuracy (trace re-derivation) ===");
        for (i, c) in block_correct.iter().enumerate() {
            eprintln!("block {i}: accuracy={:.4}", *c as f64 / 50.0);
        }
    }
}
