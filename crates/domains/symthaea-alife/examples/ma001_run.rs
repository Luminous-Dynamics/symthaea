// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001 — Partner-Conditioned Social Learning: calibration + confirmatory run + swap
//! intervention, per `ALIFE_MA001_PARTNER_CONDITIONED_POLICY_PLAN_2026-07-26.md`.
//!
//! Run: `cargo run -p symthaea-alife --example ma001_run`
//!
//! Execution order matches the plan's §10/execution-order exactly: calibration (seed 9999,
//! never pooled into the confirmatory result) freezes the divergence-margin threshold *before*
//! any confirmatory seed is analyzed; only then do the 10 confirmatory seeds run.

use symthaea_alife::ma001::{Condition, Ma001Config, Ma001Run, agent_divergence_score};

/// Flat per-organism resource share, **not** density-divided by population size. Genesis's own
/// `4.0/n` formula was calibrated against populations starting near n=16 (giving ~0.25/organism)
/// -- MA-001A's fixed population of 100 organisms under that same formula gives only ~0.04/organism,
/// well below what sustains metabolism at the crate's default costs/forage rate (found by an
/// initial run where every organism starved to death in every condition -- density-division exists
/// to guard against *unbounded growth* in a reproducing population, which MA-001A's fixed,
/// non-reproducing population structurally cannot do, so it isn't needed here at all).
fn resource_fn() -> impl FnMut(usize) -> f64 {
    move |_n| 0.25
}

struct SeedResult {
    mean_divergence: f64,
    eligible_agents: usize,
    alive_agents: usize,
}

fn run_condition(condition: Condition, seed: u64, cfg: Ma001Config) -> SeedResult {
    let mut run = Ma001Run::new(condition, seed, cfg, None);
    run.run(resource_fn());
    let alive_agents = run.organisms.iter().filter(|o| !o.is_dead()).count();
    let scores: Vec<f64> = run
        .organisms
        .iter()
        .enumerate()
        .filter(|(_, o)| !o.is_dead())
        .filter_map(|(i, _)| {
            agent_divergence_score(
                &run.analysis_counts[i],
                cfg.dirichlet_alpha,
                cfg.min_encounters_per_partner,
            )
        })
        .collect();
    let eligible_agents = scores.len();
    let mean_divergence = scores.iter().sum::<f64>() / eligible_agents.max(1) as f64;
    SeedResult {
        mean_divergence,
        eligible_agents,
        alive_agents,
    }
}

/// Per-organism classification of the swap intervention's result (plan §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SwapReading {
    HistoryFollowing,
    IdentityFollowing,
    Ambiguous,
    Ineligible,
}

fn read_swap_outcome(run: &Ma001Run, idx: usize, alpha: f64, min_encounters: u32) -> SwapReading {
    let Some((highest, lowest)) = run.swapped_partners[idx] else {
        return SwapReading::Ineligible;
    };
    let pre = &run.pre_swap_counts[idx];
    let post = &run.post_swap_counts[idx];
    let (Some(pre_high), Some(pre_low), Some(post_high)) =
        (pre.get(&highest), pre.get(&lowest), post.get(&highest))
    else {
        return SwapReading::Ineligible;
    };
    if pre_high.iter().sum::<u32>() < min_encounters
        || pre_low.iter().sum::<u32>() < min_encounters
        || post_high.iter().sum::<u32>() < min_encounters
    {
        return SwapReading::Ineligible;
    }
    let d_pre = symthaea_alife::ma001::jensen_shannon_divergence(
        &symthaea_alife::ma001::dirichlet_smooth(*pre_high, alpha),
        &symthaea_alife::ma001::dirichlet_smooth(*pre_low, alpha),
    );
    if d_pre < 0.01 {
        // No meaningful pre-swap differentiation between these two partners to begin with --
        // the swap can't tell us anything for this organism.
        return SwapReading::Ineligible;
    }
    let post_smoothed = symthaea_alife::ma001::dirichlet_smooth(*post_high, alpha);
    let d_same = symthaea_alife::ma001::jensen_shannon_divergence(
        &post_smoothed,
        &symthaea_alife::ma001::dirichlet_smooth(*pre_high, alpha),
    );
    let d_swapped = symthaea_alife::ma001::jensen_shannon_divergence(
        &post_smoothed,
        &symthaea_alife::ma001::dirichlet_smooth(*pre_low, alpha),
    );
    // Require a real gap between the two readings before calling it either way.
    if (d_same - d_swapped).abs() < 0.02 {
        SwapReading::Ambiguous
    } else if d_swapped < d_same {
        SwapReading::HistoryFollowing
    } else {
        SwapReading::IdentityFollowing
    }
}

fn main() {
    let cfg = Ma001Config::default();
    println!(
        "MA-001 -- population={}, total_ticks={}, burn_in={}, shuffle_epoch={}\n",
        cfg.population(),
        cfg.total_ticks,
        cfg.burn_in_ticks,
        cfg.shuffle_epoch_ticks
    );

    // --- Calibration (seed 9999, never pooled into the confirmatory result) ---
    println!("=== Calibration (seed 9999) ===");
    let bound_cal = run_condition(Condition::Bound, 9999, cfg);
    let shuffled_cal = run_condition(Condition::Shuffled, 9999, cfg);
    let nohist_cal = run_condition(Condition::NoHistory, 9999, cfg);
    println!(
        "Bound:     mean={:.4} eligible={} alive={}",
        bound_cal.mean_divergence, bound_cal.eligible_agents, bound_cal.alive_agents
    );
    println!(
        "Shuffled:  mean={:.4} eligible={} alive={}",
        shuffled_cal.mean_divergence, shuffled_cal.eligible_agents, shuffled_cal.alive_agents
    );
    println!(
        "NoHistory: mean={:.4} eligible={} alive={}",
        nohist_cal.mean_divergence, nohist_cal.eligible_agents, nohist_cal.alive_agents
    );
    // Frozen rule (decided before seeing this number, not after): both Shuffled and NoHistory
    // lack correctly-bound history, so their own mutual gap is a direct empirical estimate of
    // pure noise between two conditions that are equally null. Margin = 2x that gap.
    let noise_floor = (shuffled_cal.mean_divergence - nohist_cal.mean_divergence).abs();
    let margin = 2.0 * noise_floor;
    println!(
        "Calibrated margin = 2 x |Shuffled - NoHistory| = 2 x {noise_floor:.4} = {margin:.4}\n"
    );

    // --- Confirmatory (seeds 1..=10, MA-001A / LifetimeOnly) ---
    println!("=== Confirmatory MA-001A (seeds 1..=10) ===");
    let mut passes = 0u32;
    for seed in 1..=10u64 {
        let bound = run_condition(Condition::Bound, seed, cfg);
        let shuffled = run_condition(Condition::Shuffled, seed, cfg);
        let nohist = run_condition(Condition::NoHistory, seed, cfg);
        let gap = bound.mean_divergence - shuffled.mean_divergence.max(nohist.mean_divergence);
        let pass = gap >= margin;
        if pass {
            passes += 1;
        }
        println!(
            "seed {seed:2}: Bound={:.4} (n={}) Shuffled={:.4} (n={}) NoHistory={:.4} (n={}) gap={gap:.4} {}",
            bound.mean_divergence,
            bound.eligible_agents,
            shuffled.mean_divergence,
            shuffled.eligible_agents,
            nohist.mean_divergence,
            nohist.eligible_agents,
            if pass { "PASS" } else { "fail" }
        );
    }
    println!("Primary comparison: {passes}/10 seeds pass (gap >= calibrated margin {margin:.4})\n");

    // --- Swap intervention (Bound condition, swap at tick 600, 300-tick pre/post windows) ---
    println!("=== Swap intervention (Bound, swap at tick 600) ===");
    let mut history_following = 0u32;
    let mut identity_following = 0u32;
    let mut ambiguous = 0u32;
    let mut ineligible = 0u32;
    for seed in 1..=10u64 {
        let mut run = Ma001Run::new(Condition::Bound, seed, cfg, Some((600, 300)));
        run.run(resource_fn());
        for idx in 0..run.organisms.len() {
            match read_swap_outcome(
                &run,
                idx,
                cfg.dirichlet_alpha,
                cfg.min_encounters_per_partner,
            ) {
                SwapReading::HistoryFollowing => history_following += 1,
                SwapReading::IdentityFollowing => identity_following += 1,
                SwapReading::Ambiguous => ambiguous += 1,
                SwapReading::Ineligible => ineligible += 1,
            }
        }
    }
    let classified = history_following + identity_following + ambiguous;
    println!(
        "Across 10 seeds x {} organisms: history-following={history_following}, identity-following={identity_following}, ambiguous={ambiguous}, ineligible={ineligible} (of {classified} classifiable)",
        cfg.population()
    );

    // --- Interpretation ladder (plan §9) ---
    println!("\n=== Interpretation ===");
    let primary_majority = passes > 5;
    let swap_majority_history = classified > 0 && history_following * 2 > classified;
    let verdict = if primary_majority && swap_majority_history {
        "STRONG POSITIVE"
    } else if primary_majority {
        "PARTIAL POSITIVE (differentiation without a clear causal swap result)"
    } else {
        "NULL or ARCHITECTURE LIMITATION (see full report for which)"
    };
    println!("{verdict}");
}
