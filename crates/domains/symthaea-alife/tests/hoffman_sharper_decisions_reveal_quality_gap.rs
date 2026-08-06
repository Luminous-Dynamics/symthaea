// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Follow-up to `hoffman_coarse_wins_decomposition.rs`'s finding that decision-correctness (vs. a
//! ground-truth oracle) was statistically identical between fine and coarse perception at the
//! crate's default `action_temperature=1.0` (50.40% vs 50.04%). That result's own write-up in
//! `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md` speculated the ~50% ceiling might be an artifact
//! of a near-uniform softmax near the Forage/Rest crossover, not evidence resolution genuinely
//! doesn't matter -- lower `action_temperature` (sharper, more greedily exploitative action
//! selection) should let a real, if modest, belief-quality difference actually manifest as
//! behavior instead of being smoothed into softmax noise.
//!
//! Verified: an exploratory sweep (not committed here) from `action_temperature=1.0` down to
//! `0.0001` found the fine-vs-coarse decision-correctness gap grows monotonically and
//! substantially as temperature drops, from a negligible ~0.4 points (default) to a clean,
//! 8-of-8-seed-consistent ~2.7 points at `temperature=0.01` -- while coarse's energy advantage
//! *shrinks* over the same range (though it never fully closes at any well-behaved temperature
//! tried). **This test locks in `temperature=0.01` as a clean, reproducible demonstration that
//! decision-quality CAN separate in this substrate** -- the earlier finding's "resolution buys
//! nothing" conclusion was specific to the default temperature, not a general property of the
//! decision mechanism.
//!
//! **Not committed, and explicitly flagged as unresolved**: the same exploratory sweep found a
//! qualitative reversal at the most extreme temperature tried (`0.0001`) -- fine's mean energy
//! exceeded coarse's, and the decision-correctness gap collapsed. Whether this is a genuine
//! dynamical effect (e.g. a feedback bifurcation, since `action_temperature` this low makes
//! action selection essentially deterministic given belief, and belief update rate itself depends
//! on energy via blanket permeability -- a real nonlinear coupling) or an artifact of some other
//! kind was not diagnosed. The softmax computation itself is not suspected (the max-subtraction
//! stability trick in `select_action` keeps every exponent argument in `(-inf, 0]`, so it
//! underflows cleanly to zero rather than overflowing to `inf`/`NaN` even at very low
//! temperature) -- but this hasn't been independently confirmed by tracing actual values at that
//! temperature. Left as genuinely open further work, not force-fit into either finding above.

use symthaea_alife::{Environment, Organism, OrganismConfig};

const SEEDS: &[u64] = &[1, 2, 3, 4, 5, 6, 7, 8];
const TICKS: u64 = 4_000;
const GRAIN_FINE: f64 = 0.02;
const GRAIN_COARSE: f64 = 0.4;
const FORAGE: usize = 0;
const SHARP_TEMPERATURE: f64 = 0.01;

fn calibrated_environment() -> Environment {
    Environment {
        mean: 0.20,
        amplitude: 0.20,
        period: 200.0,
        noise_seed: 0xA5A5_1234_DEAD_BEEF,
        noise_amplitude: 0.02,
    }
}

fn calibrated_config(grain: f64, temperature: f64) -> OrganismConfig {
    OrganismConfig {
        forage_efficiency: 0.6,
        perceptual_grain: Some(grain),
        resource_prior: 0.0,
        action_temperature: temperature,
        ..OrganismConfig::default() // resource_preference: 1.0 (already fixed)
    }
}

struct Result {
    mean_energy: f64,
    decision_correctness_rate: f64,
}

fn run_with_decomposition(grain: f64, temperature: f64, seed: u64) -> Result {
    let mut organism = Organism::new(calibrated_config(grain, temperature), seed);
    let env = calibrated_environment();
    let mut sum_energy = 0.0;
    let mut matches = 0u64;
    let mut count = 0u64;

    for t in 0..TICKS {
        let r_true = env.resource_at(t);

        // Oracle: same technique as hoffman_coarse_wins_decomposition.rs -- clone the live
        // agent before tick(), override only the resource belief to ground truth.
        let mut oracle_agent = organism.agent.clone();
        oracle_agent.belief.mean[0] = r_true;
        let oracle_probs = oracle_agent.select_action().action_probabilities;
        let oracle_action = if oracle_probs[FORAGE] >= oracle_probs[1] {
            FORAGE
        } else {
            1
        };

        let tick = organism.tick(r_true, None);

        if t >= TICKS / 4 {
            sum_energy += tick.energy;
            if tick.action == oracle_action {
                matches += 1;
            }
            count += 1;
        }
    }

    Result {
        mean_energy: sum_energy / count.max(1) as f64,
        decision_correctness_rate: matches as f64 / count.max(1) as f64,
    }
}

#[test]
fn manipulation_check_energy_stays_moderate_at_sharp_temperature() {
    // Prerequisite: sharpening the decision policy shouldn't push either strategy into
    // saturation (the failure mode that masked every pre-VOI-calibration experiment).
    for &grain in &[GRAIN_FINE, GRAIN_COARSE] {
        let mut sum = 0.0;
        for &seed in SEEDS {
            sum += run_with_decomposition(grain, SHARP_TEMPERATURE, seed).mean_energy;
        }
        let mean = sum / SEEDS.len() as f64;
        assert!(
            (0.2..0.85).contains(&mean),
            "grain={grain}: expected moderate, non-saturated mean energy at \
             temperature={SHARP_TEMPERATURE}, got {mean:.4}"
        );
    }
}

#[test]
fn fine_shows_higher_decision_correctness_than_coarse_at_sharp_temperature() {
    // Unlike the default temperature=1.0 (statistically identical, ~50% for both), a sharper
    // decision policy should let fine's real (if modest) resolution advantage manifest as
    // behavior -- a directional claim this test asserts per-seed, not just in aggregate.
    let mut fine_wins = 0;
    let mut fine_rates = Vec::new();
    let mut coarse_rates = Vec::new();

    for &seed in SEEDS {
        let fine = run_with_decomposition(GRAIN_FINE, SHARP_TEMPERATURE, seed);
        let coarse = run_with_decomposition(GRAIN_COARSE, SHARP_TEMPERATURE, seed);
        fine_rates.push(fine.decision_correctness_rate);
        coarse_rates.push(coarse.decision_correctness_rate);
        if fine.decision_correctness_rate > coarse.decision_correctness_rate {
            fine_wins += 1;
        }
    }

    eprintln!(
        "sharp-temperature decision correctness (temperature={SHARP_TEMPERATURE}): \
         fine={fine_rates:?} coarse={coarse_rates:?} fine_wins={fine_wins}/{}",
        SEEDS.len()
    );

    assert_eq!(
        fine_wins,
        SEEDS.len(),
        "expected fine to show higher decision-correctness than coarse in every seed at \
         temperature={SHARP_TEMPERATURE}: fine={fine_rates:?}, coarse={coarse_rates:?}"
    );
}

#[test]
fn coarse_still_wins_on_energy_despite_its_lower_decision_correctness() {
    // The sharper temperature narrows but does not close coarse's fitness advantage -- the
    // headline Fitness-Beats-Truth result still holds even where decision-quality genuinely
    // does separate, arguably a *stronger* form of Hoffman's thesis than the temperature=1.0
    // result: truth-tracking perception can lose on fitness even when it demonstrably makes
    // better decisions.
    let mut coarse_wins = 0;
    for &seed in SEEDS {
        let fine = run_with_decomposition(GRAIN_FINE, SHARP_TEMPERATURE, seed);
        let coarse = run_with_decomposition(GRAIN_COARSE, SHARP_TEMPERATURE, seed);
        if coarse.mean_energy > fine.mean_energy {
            coarse_wins += 1;
        }
    }
    assert_eq!(
        coarse_wins,
        SEEDS.len(),
        "expected coarse to still win on mean energy in every seed at \
         temperature={SHARP_TEMPERATURE}, despite fine's higher decision-correctness"
    );
}
