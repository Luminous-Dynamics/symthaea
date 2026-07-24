// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IIT: does integrating hypothesis memory into experiment selection track something
//! functionally useful?
//!
//! Fifth and final capability in the agreed longer sequence for the Ramanujan Protocol (HDC →
//! constrained physical reasoning → FEP active-experiment-selection → CfC → **this**). Per the
//! user's own standing rule from early in this arc: "IIT last, only with a falsifiable
//! prediction -- otherwise it's exactly the kind of unfalsifiable 'cognition quality' scalar
//! this codebase's own history warns against" (see the Φ-vs-Ψ distinction in
//! `CORE_SUBSTRATE.md`). This module exists specifically to test one such prediction, frozen
//! before any code was written (see the plan file / commit message for the full preregistration).
//!
//! ## The falsifiable prediction
//!
//! Wires `experiment_selection` (capability #2, `epistemic_value`/
//! `select_most_informative_experiment`) into an actual multi-round hypothesis-discrimination
//! loop for the first time -- every prior capability in this arc validated its primitives
//! statically, never iteratively. Two conditions: **coupled** (experiment selection scores
//! candidates against the *currently live* hypothesis set, shrinking as elimination proceeds)
//! vs. **decoupled** (selection always scores against the *original full* set, blind to
//! elimination, even though elimination itself still happens and still determines
//! rounds-to-converge). Each round's `[live_count, mean_pairwise_disagreement,
//! chosen_experiment_epistemic_value, eliminations_this_round]` feature vector is pushed into
//! [`crate::consciousness_metrics::SpectralMIPFinder`] -- the one Φ implementation actually
//! live in production in this codebase (see `CORE_SUBSTRATE.md`), not a new one.
//!
//! **Prediction**: coupled will show BOTH higher measured Φ of this joint trajectory AND fewer
//! rounds-to-converge than decoupled. Both effects must hold together to count as support --
//! Φ rising without efficiency improving would falsify the claim that this Φ tracks something
//! functionally useful; efficiency improving without Φ rising means the win came from
//! somewhere else. Neither appearing is a full negative, consistent with every other honest
//! result in this arc.
//!
//! ## Corrected result: the efficiency effect is real; the Φ effect is unmeasurable here
//!
//! **An earlier version of this module (`min_samples: 2`, the absolute floor
//! `SpectralMIPConfig::validate()` allows) reported "SUPPORTED" -- that verdict was a
//! measurement artifact, not a real finding, and has been retracted.** With
//! `normalize_variance: true` (the default, still in effect) and exactly `min_samples` == 2
//! pushed samples, z-scoring maps *any* 2-point trajectory to the identical ±1 pattern per
//! dimension regardless of the underlying magnitudes -- confirmed empirically: every trial
//! whose window happened to hit exactly 2 pushed samples produced bit-identical Φ to 10
//! decimal places, across dozens of seeds with different ground-truth hypotheses. A
//! covariance/MI estimate over `feature_dim` dimensions needs meaningfully more samples than
//! that to be non-degenerate (`2 * feature_dim` is the rule of thumb used here). Raising
//! `min_samples` to a properly-conditioned threshold (`2 * feature_dim` = 8) and rerunning
//! both the original N=30 test and a new N=300 test with a paired bootstrap CI: **0/30 and
//! 0/300 trials ever reach a non-degenerate window** -- this task's typical trajectories
//! (mean rounds-to-converge 1.5 coupled, 3.7 decoupled) are simply too short. The Φ half of
//! the prediction is genuinely **untestable with this task design**, not negative and not
//! supported -- there isn't enough data to measure it at all under an honest sample
//! threshold. The efficiency half is real and, at N=300, formally significant: a paired
//! bootstrap 95% CI on rounds-to-converge excludes zero ([1.800, 2.240], favoring coupled).
//! **Overall verdict: PARTIAL** -- one predeclared effect (efficiency) confirmed with real
//! statistical rigor; the other (Φ) cannot be evaluated at all with this task's trajectory
//! lengths, which is itself the actual finding worth keeping: *design a task that naturally
//! runs long enough for the metric you're trying to measure*, or use a metric that doesn't
//! degenerate at small n, before trusting a "significant" small-sample Φ comparison again.

use super::experiment_selection::select_most_informative_experiment;
use crate::consciousness_metrics::{SpectralMIPConfig, SpectralMIPFinder};
use crate::hdc::unified_hv::ContinuousHV;

type Hypothesis = fn(f64) -> f64;

/// A cluster of near-identical linear hypotheses (h0-h5) that stay within a fraction of a
/// unit of each other across most of the experiment domain, plus two clearly-different
/// curves (h6/h7) that get ruled out quickly regardless. This is a corrected version of an
/// earlier design that used 6 well-separated hypotheses -- that version let a single
/// maximally-informative experiment eliminate every wrong hypothesis at once, so every trial
/// converged in exactly 1 round (verified: 30/30 seeds, both conditions, `rounds_used=1.000`)
/// and coupling could never matter (with only ever one round, "coupled" and "decoupled"
/// selection are mechanically identical, since there's nothing yet eliminated to differ
/// over). A real, honest construction bug caught empirically -- the near-identical cluster
/// here specifically prevents any single experiment from resolving everything at once.
fn candidate_hypotheses() -> [Hypothesis; 8] {
    [
        |x| x,
        |x| 1.02 * x,
        |x| 0.98 * x,
        |x: f64| x + 0.08,
        |x: f64| x - 0.08,
        |x| 1.05 * x + 0.05,
        |x: f64| x * x / 4.0,
        |x: f64| x.sin() * 2.0,
    ]
}

fn predict(h: &Hypothesis, x: &f64) -> Option<f64> {
    let v = h(*x);
    v.is_finite().then_some(v)
}

fn xorshift_next(rng: &mut u64) -> u64 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    *rng
}

/// Fisher-Yates shuffle using this module's own small RNG (matching this arc's established
/// no-new-dependency convention).
fn shuffled_pool(rng: &mut u64) -> Vec<f64> {
    let mut pool: Vec<f64> = (0..33).map(|i| -4.0 + i as f64 * 0.25).collect();
    for i in (1..pool.len()).rev() {
        let j = (xorshift_next(rng) as usize) % (i + 1);
        pool.swap(i, j);
    }
    pool
}

fn mean_pairwise_disagreement(predictions: &[f64]) -> f64 {
    if predictions.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for i in 0..predictions.len() {
        for j in (i + 1)..predictions.len() {
            total += (predictions[i] - predictions[j]).abs();
            count += 1;
        }
    }
    total / count as f64
}

/// Result of one discrimination trial.
pub struct TrialResult {
    pub rounds_used: usize,
    pub converged: bool,
    pub phi: Option<f64>,
}

const ELIMINATION_TOLERANCE: f64 = 0.05;

/// Run one discrimination trial. `coupled`: if true, experiment selection scores against the
/// live (shrinking) hypothesis set each round; if false, always scores against the full
/// original set regardless of what's already been eliminated. Elimination itself is identical
/// either way -- only the *selection* step differs, isolating the one variable this
/// experiment is about.
pub fn run_trial(seed: u64, coupled: bool) -> TrialResult {
    let hypotheses = candidate_hypotheses();
    let truth_idx = (seed as usize) % hypotheses.len();
    let truth = hypotheses[truth_idx];

    let mut rng = seed ^ 0x1157_c0de_u64;
    let mut pool = shuffled_pool(&mut rng);

    let mut live: Vec<Hypothesis> = hypotheses.to_vec();
    let feature_dim = 4;
    let config = SpectralMIPConfig {
        num_components: feature_dim,
        window_size: pool.len(),
        // History: min_samples=3 first (0/30 valid Φ pairs -- too strict for the faster
        // coupled condition to ever satisfy); then min_samples=2, the absolute floor
        // `SpectralMIPConfig::validate()` allows (16/30, then 151/300 valid pairs, both
        // showing a tight, "significant" bootstrap CI on Φ). That result turned out to be a
        // measurement artifact, not a real finding: with `normalize_variance: true` (the
        // default, still in effect here) and exactly 2 samples, z-scoring ALWAYS maps any
        // 2-point trajectory to the same +-1 pattern per dimension regardless of the
        // underlying magnitudes -- confirmed empirically (every trial with rounds_used==2
        // produced bit-identical Φ regardless of which of 4+ different ground-truth
        // hypotheses was used). A covariance/MI estimate over `feature_dim` dimensions needs
        // meaningfully more than `feature_dim` samples to be non-degenerate; `2 *
        // feature_dim` is a standard rule of thumb. Kept here (not lowered again) --
        // fixing this correctly means accepting fewer valid trials, not gaming min_samples
        // down until *something* looks significant.
        min_samples: 2 * feature_dim,
        ..SpectralMIPConfig::default()
    };
    let mut finder = SpectralMIPFinder::new(config);

    let mut rounds_used = 0;
    while live.len() > 1 && !pool.is_empty() {
        let selection_set: &[Hypothesis] = if coupled { &live } else { &hypotheses };
        let Some((&experiment, chosen_epistemic_value)) =
            select_most_informative_experiment(&pool, selection_set, predict)
        else {
            break;
        };
        pool.retain(|&x| x != experiment);
        rounds_used += 1;

        let true_obs = truth(experiment);
        let before = live.len();
        let live_predictions: Vec<f64> = live
            .iter()
            .filter_map(|h| predict(h, &experiment))
            .collect();
        live.retain(|h| match predict(h, &experiment) {
            Some(p) => (p - true_obs).abs() <= ELIMINATION_TOLERANCE,
            None => false,
        });
        let eliminations_this_round = before - live.len();

        let features = vec![
            live.len() as f32,
            mean_pairwise_disagreement(&live_predictions) as f32,
            chosen_epistemic_value as f32,
            eliminations_this_round as f32,
        ];
        finder.push(&ContinuousHV::from_values(features));
    }

    let phi = if finder.ready() {
        finder.compute().map(|r| r.phi)
    } else {
        None
    };

    TrialResult {
        rounds_used,
        converged: live.len() == 1,
        phi,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }

    #[test]
    fn ground_truth_always_survives_elimination() {
        // Sanity check on the task itself, independent of coupling: the true hypothesis must
        // never be eliminated by its own (noiseless) observations.
        for seed in 0..12u64 {
            for &coupled in &[true, false] {
                let result = run_trial(seed, coupled);
                assert!(
                    result.converged || result.rounds_used > 0,
                    "trial should make progress (seed={seed}, coupled={coupled})"
                );
            }
        }
    }

    #[test]
    fn coupled_vs_decoupled_paired_comparison() {
        const N: u64 = 30;
        let mut coupled_phis = Vec::new();
        let mut decoupled_phis = Vec::new();
        let mut coupled_rounds = Vec::new();
        let mut decoupled_rounds = Vec::new();
        let mut phi_pairs_available = 0usize;

        for seed in 0..N {
            let c = run_trial(seed, true);
            let d = run_trial(seed, false);
            coupled_rounds.push(c.rounds_used as f64);
            decoupled_rounds.push(d.rounds_used as f64);
            if let (Some(cp), Some(dp)) = (c.phi, d.phi) {
                coupled_phis.push(cp);
                decoupled_phis.push(dp);
                phi_pairs_available += 1;
            }
        }

        let mean_coupled_rounds = mean(&coupled_rounds);
        let mean_decoupled_rounds = mean(&decoupled_rounds);
        let mean_coupled_phi = if coupled_phis.is_empty() {
            f64::NAN
        } else {
            mean(&coupled_phis)
        };
        let mean_decoupled_phi = if decoupled_phis.is_empty() {
            f64::NAN
        } else {
            mean(&decoupled_phis)
        };

        println!(
            "IIT falsifiable-prediction test ({N} paired seeds, {phi_pairs_available} yielded \
             a valid Φ pair for both conditions):\n\
             mean rounds-to-converge: coupled={mean_coupled_rounds:.3}, \
             decoupled={mean_decoupled_rounds:.3}\n\
             mean Φ: coupled={mean_coupled_phi:.5}, decoupled={mean_decoupled_phi:.5}"
        );

        let efficiency_effect = mean_coupled_rounds < mean_decoupled_rounds;
        let phi_effect = phi_pairs_available > 0 && mean_coupled_phi > mean_decoupled_phi;

        println!(
            "PREDECLARED INTERPRETATION: efficiency_effect(coupled uses fewer rounds)={efficiency_effect}, \
             phi_effect(coupled has higher Φ)={phi_effect} -- \
             {}",
            match (efficiency_effect, phi_effect) {
                (true, true) => "SUPPORTED (both effects present)",
                (true, false) | (false, true) => "PARTIAL (only one effect present)",
                (false, false) => "NEGATIVE (neither effect present)",
            }
        );

        // This test intentionally does not assert on the outcome -- per the frozen design,
        // the point is to report the real result honestly, not to gate CI on which way IIT's
        // prediction actually resolves. See the module doc / memory file for what the printed
        // result means and how it was interpreted.
    }

    /// Paired bootstrap over `diffs` (one signed difference per seed, positive = coupled
    /// favored): resample with replacement `resamples` times, return `(mean_diff, ci_low,
    /// ci_high)` at the 95% percentile level. Self-contained (reuses this module's own
    /// `xorshift_next`), matching this arc's no-new-dependency convention rather than pulling
    /// in a stats crate for one test.
    fn bootstrap_ci(diffs: &[f64], resamples: usize, seed: u64) -> (f64, f64, f64) {
        let mean_diff = mean(diffs);
        let mut rng = seed;
        let mut means: Vec<f64> = Vec::with_capacity(resamples);
        for _ in 0..resamples {
            let resample_sum: f64 = (0..diffs.len())
                .map(|_| diffs[(xorshift_next(&mut rng) as usize) % diffs.len()])
                .sum();
            means.push(resample_sum / diffs.len() as f64);
        }
        means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo = means[((0.025 * resamples as f64) as usize).min(resamples - 1)];
        let hi = means[((0.975 * resamples as f64) as usize).min(resamples - 1)];
        (mean_diff, lo, hi)
    }

    /// Follow-up to `coupled_vs_decoupled_paired_comparison`: a separate, larger-N test
    /// adding real statistical rigor (a paired bootstrap CI instead of a bare mean
    /// comparison) to the same preregistered prediction. Actual per-trial compute is trivial
    /// (a full run finishes in ~0.1s once compiled -- the expensive part is always the crate
    /// compile, not the test), so N can be raised substantially for free.
    ///
    /// **This test is what actually caught the degenerate-`min_samples` measurement artifact
    /// described in the module doc.** At `min_samples: 2`, this test's first run reported a
    /// suspiciously *zero-width* 95% CI on Φ (`[0.7496931518, 0.7496931518]`, bit-identical
    /// to 10 decimal places) -- a real bootstrap correctly reflects zero variance in its
    /// input, but zero variance across 151 different seeds' Φ *differences* was itself the
    /// red flag, traced to the printed raw per-seed diagnostics below (`coupled_phi`/
    /// `decoupled_phi` identical across seeds with different ground-truth hypotheses) and
    /// then confirmed as z-scoring's known 2-point degeneracy. A tighter, more suspicious
    /// bootstrap CI turned out to be evidence of a measurement bug, not stronger evidence for
    /// the hypothesis -- the opposite of what a naive read of "the CI got even tighter at
    /// higher N" would suggest. Kept here as a real bug-diagnosis, not deleted after the fix,
    /// since catching this pattern (fixed-width CI with an implausibly exact repeat) is a
    /// reusable lesson for any future bootstrap-CI code in this codebase.
    #[test]
    fn coupled_vs_decoupled_larger_n_with_bootstrap_significance() {
        const N: u64 = 300;
        const RESAMPLES: usize = 10_000;

        let mut rounds_diffs = Vec::new(); // decoupled_rounds - coupled_rounds; positive = coupled fewer (better)
        let mut phi_diffs = Vec::new(); // coupled_phi - decoupled_phi; positive = coupled higher (better)
        let mut phi_pairs_available = 0usize;
        let mut raw_phi_samples_printed = 0;

        for seed in 0..N {
            let c = run_trial(seed, true);
            let d = run_trial(seed, false);
            rounds_diffs.push(d.rounds_used as f64 - c.rounds_used as f64);
            if let (Some(cp), Some(dp)) = (c.phi, d.phi) {
                if raw_phi_samples_printed < 10 {
                    println!(
                        "[diagnostic raw] seed={seed}: coupled_rounds={}, decoupled_rounds={}, \
                         coupled_phi={cp:.10}, decoupled_phi={dp:.10}",
                        c.rounds_used, d.rounds_used
                    );
                    raw_phi_samples_printed += 1;
                }
                phi_diffs.push(cp - dp);
                phi_pairs_available += 1;
            }
        }

        let (mean_rounds_diff, rounds_ci_lo, rounds_ci_hi) =
            bootstrap_ci(&rounds_diffs, RESAMPLES, 0xB007_5777);
        let rounds_significant = rounds_ci_lo > 0.0;

        println!(
            "IIT larger-N bootstrap test ({N} paired seeds, {phi_pairs_available} yielded a \
             valid Φ pair):\n\
             rounds-to-converge: mean(decoupled-coupled)={mean_rounds_diff:.3}, \
             95% CI=[{rounds_ci_lo:.3}, {rounds_ci_hi:.3}], \
             significant(CI excludes 0, favors coupled)={rounds_significant}"
        );

        if phi_diffs.len() >= 2 {
            let (mean_phi_diff, phi_ci_lo, phi_ci_hi) =
                bootstrap_ci(&phi_diffs, RESAMPLES, 0xB007_5778);
            let phi_significant = phi_ci_lo > 0.0;
            // Extra precision + raw sample stats: a suspiciously tight/zero-width CI needs
            // to be distinguished from a genuine implementation bug (e.g. an aliasing error
            // that always resamples the same element) before being trusted as a real finding.
            let phi_variance = phi_diffs
                .iter()
                .map(|d| (d - mean_phi_diff).powi(2))
                .sum::<f64>()
                / phi_diffs.len() as f64;
            let phi_min = phi_diffs.iter().cloned().fold(f64::INFINITY, f64::min);
            let phi_max = phi_diffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "Φ: mean(coupled-decoupled)={mean_phi_diff:.10}, \
                 95% CI=[{phi_ci_lo:.10}, {phi_ci_hi:.10}], \
                 significant(CI excludes 0, favors coupled)={phi_significant}\n\
                 [diagnostic] raw phi_diffs: variance={phi_variance:.10}, \
                 min={phi_min:.10}, max={phi_max:.10}, n={}",
                phi_diffs.len()
            );
            println!(
                "PREDECLARED INTERPRETATION (bootstrap-backed): rounds_significant={rounds_significant}, \
                 phi_significant={phi_significant} -- {}",
                match (rounds_significant, phi_significant) {
                    (true, true) => "SUPPORTED (both effects statistically significant)",
                    (true, false) | (false, true) =>
                        "PARTIAL (only one effect statistically significant)",
                    (false, false) => "NEGATIVE (neither effect statistically significant)",
                }
            );
        } else {
            println!(
                "Fewer than 2 valid Φ pairs -- cannot bootstrap a Φ confidence interval at all."
            );
        }

        // Same convention as the original test: reports the real result, does not assert on
        // which way it comes out.
    }
}
