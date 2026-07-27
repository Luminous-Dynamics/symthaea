// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001A-delta EFE-probability diagnostic — direct follow-up to
//! `ALIFE_MA001A_DELTA_RERUN_PLAN_2026-07-26.md` §10's reframed hypothesis: the near-exact
//! 1/3-1/3-1/3 action-selection split found there is consistent with action selection being close
//! to *state-insensitive altogether* at `action_temperature: 1.0` -- a softmax whose underlying
//! EFE differences between actions are small relative to the temperature would produce exactly
//! this pattern. This example measures the actual post-softmax `action_probabilities`
//! (`OrganismTick::action_probabilities`, exposing `ActiveInferenceAgent::select_action`'s own
//! already-computed result) directly for organisms at the end of a real delta-rule population
//! run -- not just the downstream realized action counts §10 already measured.
//!
//! Probing is deliberately done ONLY after the real run completes, once per organism: calling
//! `act_social` mutates the organism's real belief/RNG state (via `perceive()`/`select_action()`),
//! which cannot be cleanly reverted just by restoring `energy`/`ledger` -- probing mid-run would
//! silently contaminate the confirmatory run's own real trajectory. Probing at the very end avoids
//! this: there is no "after" to contaminate.
//!
//! Run: `cargo run -p symthaea-alife --example ma001a_delta_efe_probability_check --release`

use symthaea_alife::ma001::{Condition, Ma001Config, Ma001Run};
use symthaea_alife::ma001l::DeltaRuleConfig;

fn resource_fn() -> impl FnMut(usize) -> f64 {
    move |_n| 0.25
}

fn main() {
    let cfg = Ma001Config::default();
    println!(
        "MA-001A-delta EFE-probability diagnostic -- population={} total_ticks={} burn_in={}\n",
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

    println!(
        "=== Post-softmax action_probabilities at the end of the run (first 10 living organisms, each probed with its real current partner) ==="
    );
    let mut spreads = Vec::new();
    let mut sampled = 0;
    for i in 0..run.organisms.len() {
        if run.organisms[i].is_dead() {
            continue;
        }
        let partner = run.peek_observed_partner_ctx(i, cfg.total_ticks.saturating_sub(1));
        // This is the run's very last tick -- probing here is a genuine "no after to contaminate"
        // read: the run itself is finished, this organism's own state is never advanced further.
        let (tick, _pending) = run.organisms[i].act_social(0.25, None, partner);
        let probs = &tick.action_probabilities;
        let max_p = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_p = probs.iter().cloned().fold(f64::INFINITY, f64::min);
        let spread = max_p - min_p;
        spreads.push(spread);
        println!(
            "  organism {i}: P(Forage)={:.4} P(Rest)={:.4} P(Transfer)={:.4} spread(max-min)={:.4}",
            probs.first().copied().unwrap_or(0.0),
            probs.get(1).copied().unwrap_or(0.0),
            probs.get(2).copied().unwrap_or(0.0),
            spread
        );
        sampled += 1;
        if sampled >= 10 {
            break;
        }
    }

    // Same-organism, different-context comparison: does presenting a DIFFERENT partner context
    // to the SAME organism (same belief state) change its action_probabilities at all? This
    // directly tests context-sensitivity of the softmax itself, isolated from any specific real
    // partner assignment.
    println!(
        "\n=== Same organism (idx 0), synthetic Context-A-like vs Context-B-like partner comparison ==="
    );
    if !run.organisms[0].is_dead() {
        use symthaea_alife::{AgentId, InteractionRecord};
        let rich_context = (
            AgentId::UNALLOCATED,
            InteractionRecord {
                given_to_partner: 2.0,
                received_from_partner: 2.0,
                encounter_count: 20,
            },
        );
        let blank_context = (AgentId::UNALLOCATED, InteractionRecord::default());
        let (tick_rich, _p1) = run.organisms[0].act_social(0.25, None, Some(rich_context));
        let (tick_blank, _p2) = run.organisms[0].act_social(0.25, None, Some(blank_context));
        println!(
            "  rich context:  P(Forage)={:.4} P(Rest)={:.4} P(Transfer)={:.4}",
            tick_rich
                .action_probabilities
                .first()
                .copied()
                .unwrap_or(0.0),
            tick_rich
                .action_probabilities
                .get(1)
                .copied()
                .unwrap_or(0.0),
            tick_rich
                .action_probabilities
                .get(2)
                .copied()
                .unwrap_or(0.0)
        );
        println!(
            "  blank context: P(Forage)={:.4} P(Rest)={:.4} P(Transfer)={:.4}",
            tick_blank
                .action_probabilities
                .first()
                .copied()
                .unwrap_or(0.0),
            tick_blank
                .action_probabilities
                .get(1)
                .copied()
                .unwrap_or(0.0),
            tick_blank
                .action_probabilities
                .get(2)
                .copied()
                .unwrap_or(0.0)
        );
        println!(
            "  (both calls run on organism 0 AFTER the real run finished -- no contamination risk, this organism is not used further)"
        );
    }

    let mean_spread: f64 = spreads.iter().sum::<f64>() / spreads.len().max(1) as f64;
    println!(
        "\nMean spread (max-min) across {} sampled organisms: {mean_spread:.4}",
        spreads.len()
    );

    println!("\n=== Interpretation ===");
    println!(
        "If P(Forage)/P(Rest)/P(Transfer) are all close to 0.333 with small spread for every \
        sampled organism, that directly confirms the state-insensitivity hypothesis: the softmax \
        itself produces near-uniform output regardless of belief state, fully explaining sec 10's \
        near-1/3-1/3-1/3 realized action counts without needing to invoke any deeper claim about \
        the learning mechanism. If spread is large (one action strongly favored per organism, \
        varying meaningfully organism to organism), the near-uniform AGGREGATE counts from sec 10 \
        would instead indicate the population AVERAGES out large individual preferences, which \
        would need a different explanation. The rich-vs-blank context comparison directly tests \
        whether presenting a different social history to the identical belief state changes the \
        softmax output at all -- if rich and blank produce near-identical probabilities, that is \
        additional direct evidence the learned coupling (if any) isn't reaching action selection. \
        Read the raw numbers above directly -- this diagnostic deliberately prints them rather than \
        pre-computing a verdict, since the right threshold for 'close to uniform' is itself part of \
        what's being judged here."
    );
}
