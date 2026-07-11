// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 0 ground-truth tests, per `ALIFE_PLAN_2026-07-08.md` §0c.
//!
//! Two falsifiable claims, each checked against an explicit baseline computed in the same
//! test run (never a hardcoded target number to "look alive" against):
//!
//! 1. `perceiving_organism_tracks_resource_better_than_constant_baseline` — perception is doing
//!    real work, not theater.
//! 2. `select_action_causally_changes_behavior_and_outcomes` — action selection is doing real
//!    work, not theater.
//!
//! **Correction (Phase 1 development, same day):** this file originally had a second claim —
//! `fep_guided_action_selection_beats_uniform_random_for_energy_homeostasis` — asserting
//! FEP-guided action selection gave a real homeostatic advantage over uniform-random actions.
//! That claim relied on a bug in `Organism::tick`: the energy-gain calculation used the
//! *belief-gated* observation instead of the true `resource_level`, letting a slowly-adapting
//! belief "hallucinate" real energy income from a resource that had actually run out. Once fixed
//! (Phase 1's breakeven-share calibration collapsed toward ~0, exposing it), this test was
//! re-run at 20 seeds and again at 4x the training ticks — the gap did not survive either check
//! (guided was statistically indistinguishable from, and occasionally slightly worse than,
//! random). Per this project's standing rule against weak "didn't get worse" gates
//! (`feedback_regression_vs_improvement_gate`), the response is not to loosen the threshold
//! until it passes — it's to retract the overclaim and replace it with what's actually true:
//! `select_action()` is genuinely wired in and causally determines outcomes, but does not
//! (currently, under this task/training horizon) prove a homeostatic advantage over random
//! action selection. Both are honest, falsifiable claims; only the first survived.

use symthaea_alife::{Environment, Organism, OrganismConfig};

/// Small self-contained xorshift64, deliberately separate from anything internal to the crate
/// under test — this is the test harness's own RNG for the "uniform random" baseline, not a
/// reuse of the organism's or environment's machinery.
struct TestRng(u64);

impl TestRng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    fn next_unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f64) / (u64::MAX as f64)
    }

    fn next_action(&mut self, num_actions: usize) -> usize {
        ((self.next_unit() * num_actions as f64) as usize).min(num_actions - 1)
    }
}

#[test]
fn perceiving_organism_tracks_resource_better_than_constant_baseline() {
    // Averaged over several seeds rather than one, since the effect (organism belief update
    // converges to a fixed point that balances the observation gradient against a pull toward
    // the generative model's prior mean, per ActiveInferenceAgent::update_belief) is real but
    // modest -- checking it's consistent, not a single-seed fluke, is part of the honesty bar.
    const TICKS: u64 = 1500;
    const SEEDS: &[u64] = &[1, 2, 3, 4, 5];
    const CONSTANT_BASELINE_BELIEF: f64 = 0.5; // ActiveInferenceAgent's own untouched default.

    let mut organism_mae_sum = 0.0;
    let mut baseline_mae_sum = 0.0;

    for &seed in SEEDS {
        let env = Environment::default();
        let mut organism = Organism::new(OrganismConfig::default(), seed);

        let mut organism_abs_err = 0.0;
        let mut constant_baseline_abs_err = 0.0;

        for t in 0..TICKS {
            let resource = env.resource_at(t);
            let tick = organism.tick(resource, None);
            organism_abs_err += (tick.belief_resource - resource).abs();
            constant_baseline_abs_err += (CONSTANT_BASELINE_BELIEF - resource).abs();
        }

        organism_mae_sum += organism_abs_err / TICKS as f64;
        baseline_mae_sum += constant_baseline_abs_err / TICKS as f64;
    }

    let organism_mae = organism_mae_sum / SEEDS.len() as f64;
    let baseline_mae = baseline_mae_sum / SEEDS.len() as f64;

    // The real, structural effect measured during Phase 0 development was a consistent ~13-14%
    // reduction. Phase 3's real Landauer/Prigogine energy drain (organism.rs) has a genuine,
    // honest side effect on this specific number: it raises average homeostatic deficit
    // slightly, which (via BoundaryModulators::from_energy_deficit) closes blanket permeability
    // slightly, which reduces how much of the true resource signal reaches perceive() through
    // gate_observation -- re-measured at ~7% after Phase 3 landed. Rather than keep shrinking
    // the physics constants to preserve the old ~14% number (which would erode Phase 3's own
    // claim to being a *real*, measurable cost), the threshold below was moved to match what's
    // actually true now: still a real, non-trivial, consistent improvement (if perceive() were
    // theater, this ratio would be ~1.0, not comfortably under it), just a smaller one.
    assert!(
        organism_mae < baseline_mae * 0.95,
        "perceive() should make belief track the resource signal measurably better than a \
         static default, averaged over {} seeds: organism_mae={organism_mae:.4}, baseline_mae={baseline_mae:.4}",
        SEEDS.len()
    );
}

#[test]
fn select_action_causally_changes_behavior_and_outcomes() {
    // Same seed, same environment, same starting state -- the only difference is whether the
    // executed action comes from the agent's own select_action() or a uniform-random override.
    // If select_action() were theater (called but its result discarded), guided and random would
    // be identical. Checks two independent signatures of a real causal effect: the sequence of
    // actions actually taken diverges substantially, and so do the resulting energy
    // trajectories.
    const TICKS: u64 = 3000;
    const SEEDS: &[u64] = &[1, 2, 3, 4, 5, 6];

    let mut divergent_action_fraction_sum = 0.0;
    let mut energy_divergence_sum = 0.0;

    for &seed in SEEDS {
        let env = Environment::default();
        let cfg = OrganismConfig::default();

        let mut guided = Organism::new(cfg, seed);
        let mut random = Organism::new(cfg, seed);
        let mut rng = TestRng::new(seed ^ 0xC0FFEE);

        let mut divergent_actions = 0u64;
        let mut energy_abs_diff_sum = 0.0;

        for t in 0..TICKS {
            let resource = env.resource_at(t);

            let guided_tick = guided.tick(resource, None);
            let random_action = rng.next_action(symthaea_alife::organism::Action::COUNT);
            let random_tick = random.tick(resource, Some(random_action));

            if guided_tick.action != random_tick.action {
                divergent_actions += 1;
            }
            energy_abs_diff_sum += (guided_tick.energy - random_tick.energy).abs();
        }

        divergent_action_fraction_sum += divergent_actions as f64 / TICKS as f64;
        energy_divergence_sum += energy_abs_diff_sum / TICKS as f64;
    }

    let mean_divergent_action_fraction = divergent_action_fraction_sum / SEEDS.len() as f64;
    let mean_energy_divergence = energy_divergence_sum / SEEDS.len() as f64;

    assert!(
        mean_divergent_action_fraction > 0.2,
        "guided and random should actually choose different actions a substantial fraction of \
         the time, averaged over {} seeds: mean_divergent_action_fraction={mean_divergent_action_fraction:.3}",
        SEEDS.len()
    );
    assert!(
        mean_energy_divergence > 0.02,
        "guided and random should produce measurably different energy trajectories, averaged \
         over {} seeds: mean_energy_divergence={mean_energy_divergence:.4}",
        SEEDS.len()
    );
}
