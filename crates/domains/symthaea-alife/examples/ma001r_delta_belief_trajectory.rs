// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001R-delta — belief trajectory diagnostic. Pure measurement, no fix attempted.
//!
//! Directly logs `organism.agent.belief.mean[2..6]` (the four social dimensions:
//! partner_present, given_to_partner, received_from_partner, encounter_count) tick-by-tick while
//! running the *original* belief-based mechanism (`Ma001rProbe::run_with_delta_rule`, NOT the
//! `_from_observation` fix), to independently confirm or refute the "belief inertia/smearing"
//! hypothesis MA-001R-delta v1's diagnosis rested on but never directly measured. Closes the gap
//! `ALIFE_MA001L_CONTEXTUAL_TRANSITION_LEARNING_PLAN_2026-07-26.md` §13 explicitly disclosed:
//! "the specific claim that [incremental belief update] is the full causal explanation for the
//! observed sign-flip has not been independently re-verified by a targeted follow-up (e.g.
//! logging belief.mean[2..6]'s actual trajectory over the training run to confirm it smears
//! rather than tracks)."
//!
//! Run: `cargo run -p symthaea-alife --example ma001r_delta_belief_trajectory --release`

use symthaea_alife::ledger::compress_for_observation;
use symthaea_alife::ma001l::{DeltaRuleConfig, DeltaRuleLearner};
use symthaea_alife::ma001r::{Ma001rConfig, Ma001rProbe, Schedule};
use symthaea_alife::organism::OrganismConfig;

const SEED: u64 = 1;
const SOCIAL_DIM_NAMES: [&str; 4] = [
    "partner_present",
    "given_to_partner",
    "received_from_partner",
    "encounter_count",
];

fn social_cfg() -> OrganismConfig {
    OrganismConfig {
        social_enabled: true,
        ..OrganismConfig::default()
    }
}

/// The four raw social fields Context A/B would contribute to a freshly-gated observation --
/// i.e. what a mechanism that "tracks cleanly" should alternate exactly between, each tick.
/// Mirrors `Ma001rProbe::realized_state`'s / `ma001l`'s own construction: `partner_present=1.0`
/// always, remaining three fields are `compress_for_observation` of the raw `InteractionRecord`.
fn context_raw_social_fields(is_context_a: bool) -> [f64; 4] {
    if is_context_a {
        // context_a(): given_to_partner=2.0, received_from_partner=2.0, encounter_count=20
        [
            1.0,
            compress_for_observation(2.0),
            compress_for_observation(2.0),
            compress_for_observation(20.0),
        ]
    } else {
        // context_b(): InteractionRecord::default() -- all zero
        [1.0, 0.0, 0.0, 0.0]
    }
}

fn main() {
    let cfg = Ma001rConfig::default();
    println!(
        "MA-001R-delta belief trajectory diagnostic -- resource_level={} outcome_a={} outcome_b={} training_ticks={}\n",
        cfg.resource_level, cfg.outcome_a, cfg.outcome_b, cfg.training_ticks
    );

    let mut probe = Ma001rProbe::new(social_cfg(), SEED, cfg);
    probe.set_learning_pathway(false, false);
    let delta_rule = DeltaRuleLearner::new(DeltaRuleConfig::default(), &probe.organism.agent.model);

    let raw_a = context_raw_social_fields(true);
    let raw_b = context_raw_social_fields(false);
    println!(
        "Context A raw social fields: partner_present={:.4} given_to_partner={:.4} received_from_partner={:.4} encounter_count={:.4}",
        raw_a[0], raw_a[1], raw_a[2], raw_a[3]
    );
    println!(
        "Context B raw social fields: partner_present={:.4} given_to_partner={:.4} received_from_partner={:.4} encounter_count={:.4}",
        raw_b[0], raw_b[1], raw_b[2], raw_b[3]
    );
    println!(
        "True raw |A-B| difference per dim: {:.4} {:.4} {:.4} {:.4}\n",
        (raw_a[0] - raw_b[0]).abs(),
        (raw_a[1] - raw_b[1]).abs(),
        (raw_a[2] - raw_b[2]).abs(),
        (raw_a[3] - raw_b[3]).abs()
    );

    println!(
        "tick  ctx  belief[2..6] = partner_present  given_to_partner  received_from_partner  encounter_count"
    );

    let total_ticks = cfg.training_ticks;
    // Every belief.mean[2..6] snapshot, recorded regardless of print verbosity, so the late-window
    // amplitude analysis below sees the full trajectory, not just the printed samples.
    let mut history: Vec<(u64, bool, [f64; 4])> = Vec::with_capacity(total_ticks as usize);

    for t in 0..total_ticks {
        probe.run_with_delta_rule(&delta_rule, t, 1, Schedule::Bound);
        let is_context_a = t % 2 == 0;
        let belief = &probe.organism.agent.belief.mean;
        let snapshot = [belief[2], belief[3], belief[4], belief[5]];
        history.push((t, is_context_a, snapshot));

        let print_this_tick = t < 40 || t % 100 == 0 || t >= total_ticks - 5;
        if print_this_tick {
            let ctx_label = if is_context_a { "A" } else { "B" };
            println!(
                "{t:5} {ctx_label:>4}  {:.4}  {:.4}  {:.4}  {:.4}",
                snapshot[0], snapshot[1], snapshot[2], snapshot[3]
            );
        }
    }

    // Late-window amplitude analysis (ticks 1900..2000, the last 100 recorded, per task spec).
    let window_start = total_ticks.saturating_sub(100);
    let late_window = &history[window_start as usize..];
    println!("\n=== Late-window analysis (ticks {window_start}..{total_ticks}) ===");

    let mut amplitude = [0.0f64; 4];
    let mut mean_a = [0.0f64; 4];
    let mut mean_b = [0.0f64; 4];
    let mut count_a = 0usize;
    let mut count_b = 0usize;
    for dim in 0..4 {
        let values: Vec<f64> = late_window.iter().map(|(_, _, snap)| snap[dim]).collect();
        let max = values.iter().cloned().fold(f64::MIN, f64::max);
        let min = values.iter().cloned().fold(f64::MAX, f64::min);
        amplitude[dim] = max - min;
    }
    for (_, is_a, snap) in late_window.iter() {
        if *is_a {
            count_a += 1;
            for dim in 0..4 {
                mean_a[dim] += snap[dim];
            }
        } else {
            count_b += 1;
            for dim in 0..4 {
                mean_b[dim] += snap[dim];
            }
        }
    }
    for dim in 0..4 {
        mean_a[dim] /= count_a.max(1) as f64;
        mean_b[dim] /= count_b.max(1) as f64;
    }

    println!("  ticks presenting context A in this window: {count_a}, context B: {count_b}");
    println!(
        "  dim                     amplitude(max-min)  mean|ctx=A  mean|ctx=B  true_raw_|A-B|  tracking_ratio"
    );
    for dim in 0..4 {
        let true_diff = (raw_a[dim] - raw_b[dim]).abs();
        let observed_diff = (mean_a[dim] - mean_b[dim]).abs();
        let tracking_ratio = if true_diff > 1e-9 {
            observed_diff / true_diff
        } else {
            f64::NAN
        };
        println!(
            "  {:22} {:18.6}  {:10.4}  {:10.4}  {:14.4}  {:.4}",
            SOCIAL_DIM_NAMES[dim],
            amplitude[dim],
            mean_a[dim],
            mean_b[dim],
            true_diff,
            tracking_ratio
        );
    }

    println!(
        "\nInterpretation guide: tracking_ratio near 1.0 (and amplitude close to true_raw_|A-B|) means \
belief cleanly alternates between the two contexts' true field values each tick (tracks). \
tracking_ratio near 0 (and small amplitude relative to true_raw_|A-B|) means belief has settled \
onto a single damped/smeared value that barely distinguishes the two contexts (smears). Judge \
from the actual numbers above, not this guide alone."
    );
}
