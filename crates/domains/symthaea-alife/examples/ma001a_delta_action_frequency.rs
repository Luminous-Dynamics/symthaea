// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001A-delta action-frequency diagnostic — direct follow-up to
//! `ALIFE_MA001A_DELTA_RERUN_PLAN_2026-07-26.md` §9's suspected explanation for the population
//! rerun's null: a POLICY gap, not a representation gap. `Action::Transfer`'s pragmatic (EFE)
//! value may rarely be favored over `Forage`/`Rest` under real ecological competition, regardless
//! of what the transition model has learned -- meaning the learned social→physical coupling could
//! exist in the model while almost never being exercised by real action selection. This is exactly
//! the secondary check `ALIFE_MA001R_SOCIAL_PHYSICAL_COUPLING_PLAN_2026-07-26.md` §9 named
//! ("does model sensitivity reach action selection at all") but deferred, and which was never run
//! anywhere in this research arc.
//!
//! `Ma001Run::analysis_counts` already records per-organism, per-partner raw action counts
//! (`[Forage, Rest, Transfer]`, per `Action::index()`) during the analysis window -- this
//! diagnostic reuses that existing data, computed fresh from the Bound condition (delta-rule
//! mechanism, calibration seed 9999 -- the same run already used for MA-001A-delta's own
//! calibration, not pooled into any confirmatory claim).
//!
//! Run: `cargo run -p symthaea-alife --example ma001a_delta_action_frequency --release`

use symthaea_alife::ma001::{Condition, Ma001Config, Ma001Run};
use symthaea_alife::ma001l::DeltaRuleConfig;

fn resource_fn() -> impl FnMut(usize) -> f64 {
    move |_n| 0.25
}

fn main() {
    let cfg = Ma001Config::default();
    println!(
        "MA-001A-delta action-frequency diagnostic -- population={} total_ticks={} burn_in={}\n",
        cfg.population(),
        cfg.total_ticks,
        cfg.burn_in_ticks
    );

    let (mut run, delta_rules) = Ma001Run::new_with_delta_rule(
        Condition::Bound,
        9999,
        cfg,
        None,
        DeltaRuleConfig::default(),
    );
    run.run_with_delta_rule(resource_fn(), &delta_rules);

    let alive = run.organisms.iter().filter(|o| !o.is_dead()).count();
    println!("alive={alive}/{}\n", cfg.population());

    // Aggregate action counts across the whole population, within the analysis window
    // (burn_in_ticks..total_ticks), summed over every partner each organism met.
    let mut total_forage = 0u64;
    let mut total_rest = 0u64;
    let mut total_transfer = 0u64;
    for counts_by_partner in &run.analysis_counts {
        for counts in counts_by_partner.values() {
            total_forage += counts[0] as u64;
            total_rest += counts[1] as u64;
            total_transfer += counts[2] as u64;
        }
    }
    let total_actions = total_forage + total_rest + total_transfer;
    println!("=== Aggregate action frequency across the whole population (analysis window) ===");
    println!(
        "  Forage:   {total_forage:>8} ({:.2}%)",
        100.0 * total_forage as f64 / total_actions.max(1) as f64
    );
    println!(
        "  Rest:     {total_rest:>8} ({:.2}%)",
        100.0 * total_rest as f64 / total_actions.max(1) as f64
    );
    println!(
        "  Transfer: {total_transfer:>8} ({:.2}%)",
        100.0 * total_transfer as f64 / total_actions.max(1) as f64
    );
    println!("  Total: {total_actions}\n");

    // Per-organism Transfer-selection rate distribution -- is Transfer selection itself rare
    // and/or extremely uneven across the population, or broadly but thinly spread?
    let mut per_organism_transfer_rate: Vec<f64> = run
        .analysis_counts
        .iter()
        .enumerate()
        .filter(|(i, _)| !run.organisms[*i].is_dead())
        .map(|(_, counts_by_partner)| {
            let mut forage = 0u64;
            let mut rest = 0u64;
            let mut transfer = 0u64;
            for counts in counts_by_partner.values() {
                forage += counts[0] as u64;
                rest += counts[1] as u64;
                transfer += counts[2] as u64;
            }
            let total = forage + rest + transfer;
            if total == 0 {
                0.0
            } else {
                transfer as f64 / total as f64
            }
        })
        .collect();
    per_organism_transfer_rate.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = per_organism_transfer_rate.len().max(1);
    let mean_rate: f64 = per_organism_transfer_rate.iter().sum::<f64>() / n as f64;
    let median_rate = per_organism_transfer_rate[per_organism_transfer_rate.len() / 2];
    let zero_transfer_organisms = per_organism_transfer_rate
        .iter()
        .filter(|&&r| r == 0.0)
        .count();
    println!(
        "=== Per-organism Transfer-selection rate (fraction of this organism's own actions that were Transfer) ==="
    );
    println!("  mean:   {mean_rate:.4}");
    println!("  median: {median_rate:.4}");
    println!(
        "  min:    {:.4}",
        per_organism_transfer_rate.first().copied().unwrap_or(0.0)
    );
    println!(
        "  max:    {:.4}",
        per_organism_transfer_rate.last().copied().unwrap_or(0.0)
    );
    println!(
        "  organisms with ZERO Transfer selections: {zero_transfer_organisms}/{} ({:.1}%)\n",
        per_organism_transfer_rate.len(),
        100.0 * zero_transfer_organisms as f64 / per_organism_transfer_rate.len().max(1) as f64
    );

    // Per-organism, per-partner Transfer-selection rate spread -- does the SAME organism select
    // Transfer at meaningfully different rates for different partners (the raw behavioral
    // signature the divergence metric is trying to detect), even before Dirichlet smoothing?
    let mut max_within_organism_spread = 0.0f64;
    let mut organisms_with_meaningful_spread = 0usize;
    for (i, counts_by_partner) in run.analysis_counts.iter().enumerate() {
        if run.organisms[i].is_dead() {
            continue;
        }
        let rates: Vec<f64> = counts_by_partner
            .values()
            .filter(|c| c.iter().sum::<u32>() >= cfg.min_encounters_per_partner)
            .map(|c| {
                let total = c.iter().sum::<u32>() as f64;
                c[2] as f64 / total
            })
            .collect();
        if rates.len() < 2 {
            continue;
        }
        let min_r = rates.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_r = rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let spread = max_r - min_r;
        max_within_organism_spread = max_within_organism_spread.max(spread);
        if spread > 0.10 {
            organisms_with_meaningful_spread += 1;
        }
    }
    println!(
        "=== Within-organism, across-partner Transfer-rate spread (eligible partners only) ==="
    );
    println!("  max spread observed across the whole population: {max_within_organism_spread:.4}");
    println!(
        "  organisms with >10 percentage-point spread across their eligible partners: {organisms_with_meaningful_spread}\n"
    );

    println!("=== Interpretation ===");
    if total_transfer == 0 {
        println!(
            "VERDICT: Transfer is NEVER selected across the entire population -- this fully \
            explains the null: with zero Transfer selections, no partner-conditioned Transfer \
            differentiation could ever be observed regardless of what the underlying model has \
            learned. Action::Transfer's pragmatic value structurally loses to Forage/Rest under \
            this configuration's real EFE competition."
        );
    } else if (total_transfer as f64 / total_actions.max(1) as f64) < 0.01 {
        println!(
            "VERDICT: Transfer is selected extremely rarely (<1% of all actions) -- consistent \
            with the suspected policy gap: even if the model has learned genuine social-context \
            sensitivity for Transfer specifically, there are too few Transfer-selection events per \
            organism-partner pair to produce a detectable differentiated action distribution. This \
            would explain the null WITHOUT needing any deeper claim about the learning mechanism \
            itself being at fault."
        );
    } else if max_within_organism_spread < 0.05 {
        println!(
            "VERDICT: Transfer is selected often enough to matter, but its rate barely varies \
            across a given organism's own eligible partners (max spread {max_within_organism_spread:.4}) \
            -- this points at the coupling not reaching the pragmatic-value computation at all \
            (or being swamped by something else in it), not simply insufficient Transfer volume. \
            A different next step than the low-Transfer-rate case: look at ExpectedFreeEnergyComputer's \
            own pragmatic-value formula for whether it's sensitive to social dims at all under real \
            (not scripted) resource/energy conditions."
        );
    } else {
        println!(
            "VERDICT: Transfer selection is common enough and shows real within-organism spread \
            across partners -- this does NOT obviously support the policy-gap hypothesis as \
            stated. The null may have a different explanation than action-selection frequency \
            alone; report these numbers honestly and reconsider."
        );
    }
}
