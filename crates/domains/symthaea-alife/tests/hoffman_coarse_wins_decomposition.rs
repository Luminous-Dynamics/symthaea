// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Decomposes `hoffman_voi_calibrated_positive_control.rs`'s headline result -- coarse-grained
//! perception wins decisively in all 8 seeds, zero overlap -- into its two candidate causes, per
//! `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md`'s "VOI-calibrated positive control" section's own
//! flagged follow-up: "instrument `bits_processed` and action-correctness (vs. the analytically
//! known `r*`) per tick to decompose exactly how much of the gap is resolution cost vs. residual
//! decision-quality difference."
//!
//! ## Method
//!
//! Reuses that test's exact calibration (`forage_efficiency=0.6`, `Environment{mean:0.20,
//! amplitude:0.20}`, `resource_prior=0.0`, `GRAIN_FINE=0.02`/`GRAIN_COARSE=0.4`, same 8 seeds,
//! same 4,000 ticks, same burn-in window) so the decomposition is measured on literally the same
//! runs whose net outcome is already established -- not a fresh, potentially non-comparable setup.
//!
//! No new production code. `OrganismTick` already exposes everything needed per-tick:
//! `bits_processed` (Shannon-entropy resolution cost), `physical_cost` (the Landauer/Prigogine
//! energy actually debited for that cost, in the same units as `energy` -- directly comparable to
//! the overall energy gap), and `action` (what the organism actually did).
//!
//! The one new technique, built entirely from existing public API (`ActiveInferenceAgent: Clone`,
//! `agent.belief.mean` writable, `select_action()` public -- the same pattern
//! `hoffman_efe_rest_structurally_dominates.rs` already established): before each real `tick()`
//! call, clone the organism's live agent (capturing its actual belief state going into this
//! tick's decision -- same energy belief, same accumulated learning, same RNG state as the real
//! organism), override *only* the resource belief to the tick's true (ground-truth) resource
//! level, and call `select_action()` on the clone. This "oracle" answers: what would this exact
//! organism, at this exact point in its life, have chosen if its resource perception were
//! perfect? Comparing the oracle's deterministic argmax (over `action_probabilities`, not the
//! real agent's own stochastic softmax draw) against the real tick's actual `action` gives a
//! per-tick decision-correctness signal, isolating the resource channel specifically -- exactly
//! what `perceptual_grain` controls, and nothing else. The clone is discarded after each
//! comparison; it never affects the real organism's own RNG stream or belief.

use symthaea_alife::{Environment, Organism, OrganismConfig};

const SEEDS: &[u64] = &[1, 2, 3, 4, 5, 6, 7, 8];
const TICKS: u64 = 4_000;
const GRAIN_FINE: f64 = 0.02;
const GRAIN_COARSE: f64 = 0.4;
const FORAGE: usize = 0;

fn calibrated_environment() -> Environment {
    Environment {
        mean: 0.20,
        amplitude: 0.20,
        period: 200.0,
        noise_seed: 0xA5A5_1234_DEAD_BEEF,
        noise_amplitude: 0.02,
    }
}

fn calibrated_config(grain: f64) -> OrganismConfig {
    OrganismConfig {
        forage_efficiency: 0.6,
        perceptual_grain: Some(grain),
        resource_prior: 0.0,
        ..OrganismConfig::default() // resource_preference: 1.0 (already fixed)
    }
}

struct Decomposition {
    mean_energy: f64,
    total_bits_processed: f64,
    total_physical_cost: f64,
    decision_correctness_rate: f64,
}

fn run_with_decomposition(grain: f64, seed: u64) -> Decomposition {
    let mut organism = Organism::new(calibrated_config(grain), seed);
    let env = calibrated_environment();
    let mut sum_energy = 0.0;
    let mut sum_bits = 0.0;
    let mut sum_physical_cost = 0.0;
    let mut matches = 0u64;
    let mut count = 0u64;

    for t in 0..TICKS {
        let r_true = env.resource_at(t);

        // Oracle: what would this organism choose right now if resource perception were exact?
        // Clone BEFORE tick() so this captures the belief state the real tick is about to use.
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
            sum_bits += tick.bits_processed;
            sum_physical_cost += tick.physical_cost;
            if tick.action == oracle_action {
                matches += 1;
            }
            count += 1;
        }
    }

    Decomposition {
        mean_energy: sum_energy / count.max(1) as f64,
        total_bits_processed: sum_bits,
        total_physical_cost: sum_physical_cost,
        decision_correctness_rate: matches as f64 / count.max(1) as f64,
    }
}

#[test]
fn manipulation_check_fine_processes_more_bits_and_pays_more_physical_cost() {
    // Prerequisite for interpreting the decomposition below: the resolution-cost channel must
    // actually differ in the direction `perceptual_grain` implies, and by a real margin -- not
    // just report `bits_processed` exists, but confirm fine genuinely costs more.
    for &seed in SEEDS {
        let fine = run_with_decomposition(GRAIN_FINE, seed);
        let coarse = run_with_decomposition(GRAIN_COARSE, seed);
        assert!(
            fine.total_bits_processed > coarse.total_bits_processed,
            "seed={seed}: expected fine to process more bits than coarse, \
             fine={:.4} coarse={:.4}",
            fine.total_bits_processed,
            coarse.total_bits_processed
        );
        assert!(
            fine.total_physical_cost > coarse.total_physical_cost,
            "seed={seed}: expected fine to pay more physical (Landauer/Prigogine) cost than \
             coarse, fine={:.6} coarse={:.6}",
            fine.total_physical_cost,
            coarse.total_physical_cost
        );
    }
}

#[test]
fn decompose_coarse_wins_gap_into_resolution_cost_and_decision_quality() {
    let mut fine_energy = Vec::new();
    let mut coarse_energy = Vec::new();
    let mut fine_bits = Vec::new();
    let mut coarse_bits = Vec::new();
    let mut fine_physical_cost = Vec::new();
    let mut coarse_physical_cost = Vec::new();
    let mut fine_correctness = Vec::new();
    let mut coarse_correctness = Vec::new();

    for &seed in SEEDS {
        let fine = run_with_decomposition(GRAIN_FINE, seed);
        let coarse = run_with_decomposition(GRAIN_COARSE, seed);
        fine_energy.push(fine.mean_energy);
        coarse_energy.push(coarse.mean_energy);
        fine_bits.push(fine.total_bits_processed);
        coarse_bits.push(coarse.total_bits_processed);
        fine_physical_cost.push(fine.total_physical_cost);
        coarse_physical_cost.push(coarse.total_physical_cost);
        fine_correctness.push(fine.decision_correctness_rate);
        coarse_correctness.push(coarse.decision_correctness_rate);
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let fine_energy_mean = mean(&fine_energy);
    let coarse_energy_mean = mean(&coarse_energy);
    let fine_bits_mean = mean(&fine_bits);
    let coarse_bits_mean = mean(&coarse_bits);
    let fine_cost_mean = mean(&fine_physical_cost);
    let coarse_cost_mean = mean(&coarse_physical_cost);
    let fine_correct_mean = mean(&fine_correctness);
    let coarse_correct_mean = mean(&coarse_correctness);

    eprintln!(
        "Hoffman coarse-wins decomposition (N=8 seeds):\n\
         energy:            fine={fine_energy_mean:.4} coarse={coarse_energy_mean:.4} \
         gap={:.4}\n\
         bits_processed:    fine={fine_bits_mean:.2} coarse={coarse_bits_mean:.2} \
         gap={:.2}\n\
         physical_cost:     fine={fine_cost_mean:.6} coarse={coarse_cost_mean:.6} \
         gap={:.6}\n\
         decision_correct%: fine={:.4} coarse={:.4} gap={:.4}\n\
         fine_energy_per_seed={fine_energy:?}\n\
         coarse_energy_per_seed={coarse_energy:?}\n\
         fine_correctness_per_seed={fine_correctness:?}\n\
         coarse_correctness_per_seed={coarse_correctness:?}",
        coarse_energy_mean - fine_energy_mean,
        coarse_bits_mean - fine_bits_mean, // note: fine - coarse convention flips for bits (fine costs more)
        fine_cost_mean - coarse_cost_mean,
        fine_correct_mean,
        coarse_correct_mean,
        fine_correct_mean - coarse_correct_mean,
    );

    // Sanity bounds only -- report, don't assume a direction on decision quality. The resolution
    // gap (fine costs more) is the only pre-committed direction, since that's a structural
    // property of `perceptual_grain`'s Landauer tax, not an open empirical question.
    assert!(
        fine_bits_mean > coarse_bits_mean,
        "resolution-cost component: fine should process more bits than coarse, \
         fine={fine_bits_mean:.2} coarse={coarse_bits_mean:.2}"
    );
    assert!(
        (0.0..=1.0).contains(&fine_correct_mean) && (0.0..=1.0).contains(&coarse_correct_mean),
        "decision-correctness rates must be valid probabilities: fine={fine_correct_mean}, \
         coarse={coarse_correct_mean}"
    );
}
