// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001A-delta — population-scale rerun of MA-001 with the validated raw-observation delta
//! rule in place of each organism's default Hebbian+TD pathway, per
//! `ALIFE_MA001A_DELTA_RERUN_PLAN_2026-07-26.md`.
//!
//! Run: `cargo run -p symthaea-alife --example ma001a_delta_run --release`
//!
//! Execution order matches the plan's §8 exit criteria: ecological-viability check first, then
//! calibration (seed 9999, never pooled into the confirmatory result) freezes the divergence
//! margin *before* any confirmatory seed is analyzed, then the 10 confirmatory seeds, then the
//! swap intervention.

use symthaea_alife::ma001::{Condition, Ma001Config, Ma001Run, agent_divergence_score};
use symthaea_alife::ma001l::DeltaRuleConfig;

/// Identical flat per-organism resource share to MA-001A's own original run (`ma001_run.rs`) --
/// unchanged, since this rerun's ecological-viability gate is specifically about checking whether
/// the *learning mechanism* (not the resource formula) shifts survival, holding everything else
/// fixed.
fn resource_fn() -> impl FnMut(usize) -> f64 {
    move |_n| 0.25
}

struct SeedResult {
    mean_divergence: f64,
    eligible_agents: usize,
    alive_agents: usize,
}

fn run_condition(condition: Condition, seed: u64, cfg: Ma001Config) -> SeedResult {
    let (mut run, delta_rules) =
        Ma001Run::new_with_delta_rule(condition, seed, cfg, None, DeltaRuleConfig::default());
    run.run_with_delta_rule(resource_fn(), &delta_rules);
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

/// Per-organism classification of the swap intervention's result (plan §2, reused from MA-001A §6
/// unchanged).
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
        "MA-001A-delta -- population={}, total_ticks={}, burn_in={}, shuffle_epoch={}\n",
        cfg.population(),
        cfg.total_ticks,
        cfg.burn_in_ticks,
        cfg.shuffle_epoch_ticks
    );

    // --- Ecological viability gate + calibration (seed 9999, never pooled into confirmatory) ---
    println!("=== Ecological viability gate + calibration (seed 9999) ===");
    println!(
        "Reference (original MA-001A, default learning): Bound alive=98 Shuffled alive=98 NoHistory alive=100\n"
    );
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
    let viability_ok = [&bound_cal, &shuffled_cal, &nohist_cal]
        .iter()
        .all(|r| r.alive_agents >= cfg.population() * 9 / 10);
    println!(
        "Viability check (>=90% alive in all 3 conditions): {}\n",
        if viability_ok {
            "PASS"
        } else {
            "FAIL -- do not trust confirmatory seeds below"
        }
    );

    let noise_floor = (shuffled_cal.mean_divergence - nohist_cal.mean_divergence).abs();
    let margin = 2.0 * noise_floor;
    println!(
        "Calibrated margin (delta rule, freshly calibrated, NOT reused from MA-001A's original 0.0001) = 2 x |Shuffled - NoHistory| = 2 x {noise_floor:.4} = {margin:.4}\n"
    );

    // --- Confirmatory (seeds 1..=10, delta-rule mechanism) ---
    println!("=== Confirmatory MA-001A-delta (seeds 1..=10) ===");
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
    println!("=== Swap intervention (Bound, swap at tick 600, delta-rule mechanism) ===");
    let mut history_following = 0u32;
    let mut identity_following = 0u32;
    let mut ambiguous = 0u32;
    let mut ineligible = 0u32;
    for seed in 1..=10u64 {
        let (mut run, delta_rules) = Ma001Run::new_with_delta_rule(
            Condition::Bound,
            seed,
            cfg,
            Some((600, 300)),
            DeltaRuleConfig::default(),
        );
        run.run_with_delta_rule(resource_fn(), &delta_rules);
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

    // --- Interpretation ladder (plan §7, extends MA-001A's original §9) ---
    println!("\n=== Interpretation ===");
    let primary_majority = passes > 5;
    let swap_majority_history = classified > 0 && history_following * 2 > classified;
    let verdict = if primary_majority && swap_majority_history {
        "STRONG POSITIVE -- the mechanism-level fix (MA-001L/R-delta) restored population-scale policy differentiation. Closes the full population-null -> mechanism-diagnosis -> mechanism-fix -> population-confirmation chain."
    } else if primary_majority {
        "PARTIAL POSITIVE (differentiation without a clear causal swap result)"
    } else {
        "NULL or ARCHITECTURE LIMITATION -- despite the validated delta rule, population-scale free-choice policy still does not differentiate by partner history. This would mean MA-001A's original null was NOT (fully) explained by the learning-rule limitation MA-001R/L/R-delta diagnosed; some other population-scale factor (ecology, free action selection under real EFE competition, or something not yet identified) is the actual bottleneck. See the plan's §7 addendum."
    };
    println!("{verdict}");
}
