// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exploratory sweep, not a committed test: does lowering `action_temperature` (sharper, less
//! stochastic softmax action selection) let decision-correctness separate between fine and
//! coarse perception at the VOI-calibrated crossover, per
//! `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md`'s "Coarse-wins gap decomposition" section's own
//! flagged next question?
//!
//! Hypothesis: the ~50% decision-correctness ceiling found at `action_temperature=1.0` (the
//! crate default) may be an artifact of a near-uniform softmax near the Forage/Rest crossover,
//! not evidence resolution genuinely doesn't matter. Lowering temperature makes `select_action`
//! more greedily exploitative -- if a real, if small, belief-quality difference exists between
//! fine and coarse, sharper decisions should let it actually manifest as behavior instead of
//! being smoothed into noise.
//!
//! Same technique as `tests/hoffman_coarse_wins_decomposition.rs` (oracle = clone the live
//! agent, override belief to ground truth, compare argmax against the real tick's actual
//! action) -- reused verbatim, just swept over `action_temperature` instead of held at 1.0.
//! Must also re-verify the moderate-energy precondition at each temperature, since sharper
//! decisions could push organisms toward saturation (the failure mode that masked every
//! pre-VOI-calibration experiment).

use symthaea_alife::{Environment, Organism, OrganismConfig};

const SEEDS: &[u64] = &[1, 2, 3, 4, 5, 6, 7, 8];
const TICKS: u64 = 4_000;
const GRAIN_FINE: f64 = 0.02;
const GRAIN_COARSE: f64 = 0.4;
const FORAGE: usize = 0;
const TEMPERATURES: &[f64] = &[
    1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001,
];

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

fn main() {
    println!("Hoffman action_temperature sweep -- decision-correctness vs. energy-moderation");
    println!(
        "{:>5} | {:>8} {:>8} | {:>8} {:>8} | moderate?",
        "temp", "fine_E", "coarse_E", "fine_C%", "coarse_C%"
    );

    for &temp in TEMPERATURES {
        let mut fine_energy = Vec::new();
        let mut coarse_energy = Vec::new();
        let mut fine_correct = Vec::new();
        let mut coarse_correct = Vec::new();

        for &seed in SEEDS {
            let fine = run_with_decomposition(GRAIN_FINE, temp, seed);
            let coarse = run_with_decomposition(GRAIN_COARSE, temp, seed);
            fine_energy.push(fine.mean_energy);
            coarse_energy.push(coarse.mean_energy);
            fine_correct.push(fine.decision_correctness_rate);
            coarse_correct.push(coarse.decision_correctness_rate);
        }

        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let fine_e = mean(&fine_energy);
        let coarse_e = mean(&coarse_energy);
        let fine_c = mean(&fine_correct);
        let coarse_c = mean(&coarse_correct);
        let moderate = (0.2..0.85).contains(&fine_e) && (0.2..0.85).contains(&coarse_e);

        println!(
            "{:>5.2} | {:>8.4} {:>8.4} | {:>8.4} {:>8.4} | {}",
            temp,
            fine_e,
            coarse_e,
            fine_c,
            coarse_c,
            if moderate { "yes" } else { "NO (saturated)" }
        );
        println!("        fine_correct_per_seed={fine_correct:?}");
        println!("        coarse_correct_per_seed={coarse_correct:?}");
    }
}
