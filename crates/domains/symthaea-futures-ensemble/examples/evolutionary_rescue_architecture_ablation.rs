// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phase 2.2C-i (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`): the architecture-isolation
//! ablation that Phase 2.2B-ii's correction called for. Phase 2.2B-ii found the trait-augmented
//! FEP model (2D) scores worse than census-only (1D), and that the PRIVILEGED (perfect-info)
//! arm scores *worst* of the three trait-augmented conditions -- which rules out "the signal was
//! too noisy" but does NOT by itself prove the harm is architectural, because that comparison
//! confounded information content (trait vs. none) with model capacity (1D vs. 2D).
//!
//! This experiment removes that confound: **every arm here holds `state_dim=2` fixed**,
//! reusing `FepCensusPlusTraitGenerator` completely unchanged. Only what occupies the second
//! channel differs:
//! 1. `constant` -- a fixed, informationless value every tick (architecture-only control).
//! 2. `duplicate_census` -- the population channel's own value, copied into the second slot (a
//!    second, maximally-predictive, well-formed channel).
//! 3. `independent_noise` -- pure noise from its own dedicated RNG stream, uncorrelated with
//!    anything (generic irrelevant-dimensionality drag).
//! 4. `monotonic_ramp` -- **added same day, after the first run of this ablation surfaced a new
//!    puzzle**: all three trait-shaped arms below scored *worse* than `constant`/
//!    `duplicate_census`/`independent_noise`, with the exact-ground-truth arm scoring *worst of
//!    all six*. This arm tests one specific, cheap-to-check hypothesis for why: a purely
//!    deterministic, content-free linear ramp (same slow-rising *shape* as the real trait
//!    signal, zero relation to any world's actual state, identical across every seed) -- if this
//!    also underperforms like the trait arms do, that's evidence the slow-monotonic *shape*
//!    itself (not the trait's specific information) is what interacts badly with this FEP
//!    configuration's cross-dimensional coupling.
//! 5. `shuffled_trait` -- Phase 2.2B-ii's condition 3, reused from the fixture unchanged.
//! 6. `real_noisy_trait` -- Phase 2.2B-ii's condition 2, reused from the fixture unchanged.
//! 7. `privileged_trait` -- Phase 2.2B-ii's condition 4, reused from the fixture unchanged.
//!
//! `fep_census_only` (1D) is printed as a labeled REFERENCE POINT only, per the plan's own
//! instruction not to treat any 2D-vs-1D comparison as primary until the 2D arms are understood
//! relative to each other first.
//!
//! Prerequisite: `cargo run --release --example evolutionary_rescue_generate_worlds -p
//! symthaea-futures-ensemble` first, to produce the fixture files this script loads.
//!
//! Run: `cargo run --release --example evolutionary_rescue_architecture_ablation -p symthaea-futures-ensemble`
//!
//! Deliberately unchanged from Phase 2.2B-ii: `HORIZON`/`CHECKPOINT_STRIDE` (the
//! horizon-resolution limitation that write-up also disclosed is a *separate* concern, scoped to
//! Phase 2.2C-ii, not fixed here -- this experiment isolates one confound at a time).

#[path = "support/evolutionary_rescue_common.rs"]
mod common;

use common::SerializedWorld;
use symthaea_futures_calibration::{BrierScore, ScoringRule};
use symthaea_futures_core::{ForecastOutput, Horizon, OutcomeRegion, TrajectoryGenerator};
use symthaea_futures_ensemble::evolutionary_rescue::{
    FepCensusOnlyGenerator, FepCensusPlusTraitGenerator, TRAIT_CEILING, TRAIT_FLOOR,
};
use symthaea_futures_symtropy::evolutionary_rescue::{
    EvolutionaryRescueObservation, EvolutionaryRescueSample,
};

const HORIZON: u64 = 300;
const CHECKPOINT_STRIDE: u64 = 300;

const ARM_NAMES: [&str; 7] = [
    "constant",
    "duplicate_census",
    "independent_noise",
    "monotonic_ramp",
    "shuffled_trait",
    "real_noisy_trait",
    "privileged_trait",
];

/// The ramp's endpoint value -- the midpoint of the pre-extinction trait levels this scenario
/// family's own worlds actually reach (0.37-0.61 across the 10 recorded seeds), not invented.
const RAMP_TARGET: f64 = 0.48;

/// xorshift64 -- own independent stream, matching this scenario family's established RNG
/// convention (never derived from the simulation's or any other policy's own state).
fn xorshift64_next_unit(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

/// Builds a synthetic second-channel observation stream with the *exact same* tick/sample
/// presence pattern as `census_obs` (so all 6 arms see information at identical times -- only
/// the second channel's content differs), by calling `second_channel` once per observed tick.
fn build_synthetic(
    census_obs: &[EvolutionaryRescueObservation],
    mut second_channel: impl FnMut(usize, &EvolutionaryRescueObservation) -> f64,
) -> Vec<EvolutionaryRescueObservation> {
    census_obs
        .iter()
        .enumerate()
        .map(|(i, obs)| EvolutionaryRescueObservation {
            tick: obs.tick,
            sample: obs.sample.map(|s| EvolutionaryRescueSample {
                sampled_alive_count: s.sampled_alive_count,
                observed_mean_forage_efficiency: Some(second_channel(i, obs)),
            }),
        })
        .collect()
}

/// The seven 2D observation streams for one world, holding architecture (`state_dim=2`) fixed
/// and varying only what the second channel carries.
struct ArmStreams {
    constant: Vec<EvolutionaryRescueObservation>,
    duplicate_census: Vec<EvolutionaryRescueObservation>,
    independent_noise: Vec<EvolutionaryRescueObservation>,
    monotonic_ramp: Vec<EvolutionaryRescueObservation>,
    shuffled_trait: Vec<EvolutionaryRescueObservation>,
    real_noisy_trait: Vec<EvolutionaryRescueObservation>,
    privileged_trait: Vec<EvolutionaryRescueObservation>,
}

fn build_arm_streams(world: &SerializedWorld) -> ArmStreams {
    let reference = world
        .census_obs
        .first()
        .and_then(|o| o.sample)
        .map(|s| s.sampled_alive_count as f64)
        .filter(|&r| r > 0.0)
        .unwrap_or(1.0);

    let midpoint = (TRAIT_FLOOR + TRAIT_CEILING) / 2.0;
    let constant = build_synthetic(&world.census_obs, |_, _| midpoint);

    // Inverts normalize_trait's mapping so that, once FepCensusPlusTraitGenerator re-applies
    // normalize_trait internally, the model receives a second-channel value EXACTLY equal to
    // the population channel's own normalized value at that tick -- a genuine duplicate, not an
    // approximation.
    let duplicate_census = build_synthetic(&world.census_obs, |_, obs| {
        let pop_fraction = obs
            .sample
            .map(|s| (s.sampled_alive_count as f64 / reference).clamp(0.0, 1.0))
            .unwrap_or(0.5);
        TRAIT_FLOOR + pop_fraction * (TRAIT_CEILING - TRAIT_FLOOR)
    });

    // Own independent stream, seeded well clear of every other RNG this scenario family uses
    // (simulation seed, noisy-trait noise seed, shuffle seed).
    let mut noise_state = world.seed.wrapping_mul(0xC0FFEE_u64).wrapping_add(0xACE1);
    if noise_state == 0 {
        noise_state = 1;
    }
    let independent_noise = build_synthetic(&world.census_obs, |_, _| {
        TRAIT_FLOOR + xorshift64_next_unit(&mut noise_state) * (TRAIT_CEILING - TRAIT_FLOOR)
    });

    // Purely deterministic, content-free, identical across every seed at a given tick -- a
    // linear rise from TRAIT_FLOOR to RAMP_TARGET over the full recorded trajectory length,
    // matching the real trait signal's slow-monotonic *shape* while carrying zero information
    // about this (or any) world's actual state.
    let total_ticks = world.census_obs.len().saturating_sub(1).max(1) as f64;
    let monotonic_ramp = build_synthetic(&world.census_obs, |_, obs| {
        let progress = (obs.tick as f64 / total_ticks).clamp(0.0, 1.0);
        TRAIT_FLOOR + progress * (RAMP_TARGET - TRAIT_FLOOR)
    });

    ArmStreams {
        constant,
        duplicate_census,
        independent_noise,
        monotonic_ramp,
        shuffled_trait: world.shuffled_obs.clone(),
        real_noisy_trait: world.noisy_obs.clone(),
        privileged_trait: world.privileged_obs.clone(),
    }
}

fn brier_of(output: ForecastOutput, actual: &OutcomeRegion, acc: &mut (f64, usize)) {
    if let ForecastOutput::Distribution(dist) = output {
        acc.0 += BrierScore
            .score(&dist, actual)
            .expect("scoring a validated forecast cannot fail")
            .get();
        acc.1 += 1;
    }
}

/// Mean Brier per arm for one world, plus the 1D reference point. Index order matches
/// `ARM_NAMES`; the 8th slot is the `fep_census_only` reference.
fn score_world(world: &SerializedWorld) -> [(f64, usize); 8] {
    let arms = build_arm_streams(world);
    let streams: [&Vec<EvolutionaryRescueObservation>; 7] = [
        &arms.constant,
        &arms.duplicate_census,
        &arms.independent_noise,
        &arms.monotonic_ramp,
        &arms.shuffled_trait,
        &arms.real_noisy_trait,
        &arms.privileged_trait,
    ];

    let fep_2d = FepCensusPlusTraitGenerator::default();
    let fep_1d = FepCensusOnlyGenerator::default();

    let mut scores: [(f64, usize); 8] = [(0.0, 0); 8];

    let mut checkpoint = 0u64;
    while checkpoint + HORIZON < world.trajectory.len() as u64 {
        let actual = OutcomeRegion::Boolean(world.trajectory[(checkpoint + HORIZON) as usize]);
        let idx = checkpoint as usize;

        for (arm_i, stream) in streams.iter().enumerate() {
            let history = stream[..=idx].to_vec();
            brier_of(
                fep_2d.generate(&history, Horizon(HORIZON)),
                &actual,
                &mut scores[arm_i],
            );
        }

        let census_history = world.census_obs[..=idx].to_vec();
        brier_of(
            fep_1d.generate(&census_history, Horizon(HORIZON)),
            &actual,
            &mut scores[7],
        );

        checkpoint += CHECKPOINT_STRIDE;
    }

    scores
}

fn mean(acc: (f64, usize)) -> f64 {
    if acc.1 > 0 {
        acc.0 / acc.1 as f64
    } else {
        f64::NAN
    }
}

fn print_paired(label: &str, a_name: &str, b_name: &str, diffs: &[(u64, f64)]) {
    println!("\n{label}: {a_name} minus {b_name} (negative = {a_name} better)");
    for &(seed, d) in diffs {
        println!("  seed={seed:4}  diff={d:+.4}");
    }
    let mean_diff: f64 = diffs.iter().map(|&(_, d)| d).sum::<f64>() / diffs.len() as f64;
    let negative_count = diffs.iter().filter(|&&(_, d)| d < 0.0).count();
    println!(
        "  mean diff = {mean_diff:+.4}   ({negative_count}/{} seeds favor {a_name})",
        diffs.len()
    );
}

fn main() {
    println!("== Phase 2.2C-i: architecture-isolation ablation (state_dim=2 held fixed) ==");
    println!(
        "HORIZON={HORIZON} CHECKPOINT_STRIDE={CHECKPOINT_STRIDE} (unchanged from Phase 2.2B-ii)\n"
    );

    let train_worlds: Vec<SerializedWorld> = common::TRAIN_SEEDS
        .iter()
        .map(|&s| common::load_world(s))
        .collect();
    let test_worlds: Vec<SerializedWorld> = common::TEST_SEEDS
        .iter()
        .map(|&s| common::load_world(s))
        .collect();

    for (label, worlds) in [("Train", &train_worlds), ("Test (held-out)", &test_worlds)] {
        println!("-- {label} seeds --");
        let mut aggregate: [(f64, usize); 8] = [(0.0, 0); 8];
        let mut per_seed: Vec<(u64, [(f64, usize); 8])> = Vec::new();

        for world in worlds {
            let scores = score_world(world);
            for i in 0..8 {
                aggregate[i].0 += scores[i].0;
                aggregate[i].1 += scores[i].1;
            }
            per_seed.push((world.seed, scores));
        }

        for (i, name) in ARM_NAMES.iter().enumerate() {
            println!(
                "  {name:20} mean Brier = {:.4}  (n={})",
                mean(aggregate[i]),
                aggregate[i].1
            );
        }
        println!(
            "  {:20} mean Brier = {:.4}  (n={})  [REFERENCE, 1D -- not a primary comparison yet]",
            "fep_census_only",
            mean(aggregate[7]),
            aggregate[7].1
        );

        if label.starts_with("Test") {
            let idx = |name: &str| ARM_NAMES.iter().position(|&n| n == name).unwrap();
            let diffs_for = |a: usize, b: usize| -> Vec<(u64, f64)> {
                per_seed
                    .iter()
                    .map(|&(seed, s)| (seed, mean(s[a]) - mean(s[b])))
                    .collect()
            };

            println!("\n== Primary paired-by-seed comparisons (held-out test seeds only) ==");
            print_paired(
                "Real vs. shuffled trait",
                "real_noisy_trait",
                "shuffled_trait",
                &diffs_for(idx("real_noisy_trait"), idx("shuffled_trait")),
            );
            print_paired(
                "Exact vs. constant",
                "privileged_trait",
                "constant",
                &diffs_for(idx("privileged_trait"), idx("constant")),
            );
            print_paired(
                "Duplicate-census vs. constant",
                "duplicate_census",
                "constant",
                &diffs_for(idx("duplicate_census"), idx("constant")),
            );
            print_paired(
                "Independent noise vs. constant",
                "independent_noise",
                "constant",
                &diffs_for(idx("independent_noise"), idx("constant")),
            );
            print_paired(
                "Monotonic ramp vs. constant (does the slow-rising SHAPE alone hurt?)",
                "monotonic_ramp",
                "constant",
                &diffs_for(idx("monotonic_ramp"), idx("constant")),
            );
            print_paired(
                "Real trait vs. monotonic ramp (does real trait content hurt MORE than just \
                 having this shape?)",
                "real_noisy_trait",
                "monotonic_ramp",
                &diffs_for(idx("real_noisy_trait"), idx("monotonic_ramp")),
            );

            println!(
                "\nNote on statistical power: only 5 held-out seeds -- the counts above are \
                 reported directly (per-seed sign agreement), not dressed up as a p-value that \
                 would be uninformative at this N."
            );
        }
        println!();
    }
}
