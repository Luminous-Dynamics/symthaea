// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regeneration Agent EFE Investigation
//!
//! `regeneration_statistical_replication.rs` found that the regeneration
//! agent's chosen action was *exactly* identical across 30 independent
//! seeds, even after fixing a real RNG-sharing bug -- pointing at something
//! structural rather than shared randomness: the hypothesis that this
//! untrained agent's expected-free-energy computation is so strongly
//! peaked for this task's observation shape that softmax action selection
//! is, in practice, a near-deterministic argmax.
//!
//! This tests that hypothesis directly. `ActiveInferenceAgent::select_action`
//! already returns `ActionSelectionResult::action_probabilities` -- the
//! actual softmax distribution over all 4 actions -- so no new
//! instrumentation is needed anywhere; this just logs it day by day over a
//! real regeneration trajectory, using a standalone agent instance
//! constructed identically to `crate::regeneration_agent::RegenerationAgent`
//! (same config, same goals, same seed-derivation convention) so the
//! diagnostic reflects the real thing. The tissue itself is advanced under
//! ordinary (legacy) dynamics -- this agent's choices are never fed back
//! into it -- so what's being inspected is purely "what would this agent's
//! probability distribution look like, given the observation sequence a
//! real episode actually produces."
//!
//! **Result: the hypothesis is wrong, and the real explanation is more
//! useful.** Action probabilities stay close to uniform throughout (not
//! peaked at all) and the chosen action genuinely varies day to day -- this
//! agent is not behaving deterministically. What actually explains the
//! exact-0.0000-every-seed finding is logged in an added `eligible` column:
//! `regenerative_proliferate_with_boost` only affects cells that are BOTH
//! wound-boundary AND still progenitor-type, and that count is 0 on every
//! single day of this scenario -- by the time a 20-day-matured organoid gets
//! amputated, essentially no wound-adjacent cells are still progenitors.
//! No matter which multiplier gets chosen, by the agent or the legacy flat
//! rate, there is nothing left to apply it to. This isn't an agent-behavior
//! finding at all -- it's a scenario-design gap that made the entire
//! mechanism under test inert for every controller, discovered only by
//! actually looking rather than stopping at the first plausible-sounding
//! explanation.
//!
//! Run: cargo run -p symthaea-cell-foundry --example regeneration_agent_efe_investigation

use symthaea_cell_foundry::build_radial_bipolar_template;
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

const ACTION_LABELS: [&str; 4] = ["0.0x", "0.5x", "1.0x", "2.0x"];
/// Mirrors `crate::bioelectric::REGENERATION_TIMEOUT_DAYS` (private).
const REGENERATION_TIMEOUT_DAYS: f64 = 200.0;
/// Mirrors the seed-derivation convention in `advance_regeneration`
/// (`self.creation_seed.wrapping_add(0xA6E7_FEB0)`).
const AGENT_SEED_SALT: u64 = 0xA6E7_FEB0;

fn main() {
    let seed = 10_000u64; // same first seed regeneration_statistical_replication.rs used
    let cells = 150;
    let maturation_days = 20;
    let boundary_r = 0.2;
    let recovery_days = 60;

    // Same config crate::regeneration_agent::RegenerationAgent::new() uses.
    let config = ActiveInferenceAgentConfig {
        state_dim: 4,
        obs_dim: 4,
        num_actions: ACTION_LABELS.len(),
        ..Default::default()
    };
    let mut agent = ActiveInferenceAgent::new(config);
    agent.set_rng_seed(seed.wrapping_add(AGENT_SEED_SALT));
    agent.set_goals(vec![0.0, 0.0, 0.0, 0.0], 2.0);

    let mut organoid = build_radial_bipolar_template(seed, cells, maturation_days, boundary_r);
    organoid.amputate(0.6, 2.0);

    println!(
        "Logging the regeneration agent's actual action-probability distribution \
         over a real {recovery_days}-day recovery trajectory (seed={seed})...\n"
    );
    println!(
        "{:>4} {:>11} {:>10} {:>9} | {:>44} | {:>6}",
        "day",
        "discrepancy",
        "wound_frac",
        "eligible",
        "action_probabilities [0.0x,0.5x,1.0x,2.0x]",
        "chosen"
    );

    for day in 1..=recovery_days {
        organoid.advance_day();

        let n = organoid.field.num_cells().max(1) as f64;
        let discrepancy = organoid.morphology_discrepancy().unwrap_or(1.0);
        // The actual gate `regenerative_proliferate_with_boost` applies: a
        // cell must be BOTH wound-boundary AND still progenitor-type for
        // any boost multiplier to have any effect on it at all. If this is
        // ~0 throughout, the chosen multiplier is a no-op regardless of
        // which one gets picked, which would fully explain why varying it
        // never changed the tissue's outcome.
        let eligible_count = (0..organoid.field.num_cells())
            .filter(|&i| {
                organoid.field.bioelectric.wound_boundary[i]
                    && organoid.field.cells[i].cell_type.is_progenitor()
            })
            .count();
        let defected_fraction = organoid
            .field
            .bioelectric
            .defected
            .iter()
            .filter(|&&d| d)
            .count() as f64
            / n;
        let wound_boundary_fraction = organoid
            .field
            .bioelectric
            .wound_boundary
            .iter()
            .filter(|&&w| w)
            .count() as f64
            / n;
        let days_since_wound_frac = (day as f64 / REGENERATION_TIMEOUT_DAYS).min(1.0);

        let obs = Observation::new(
            vec![
                discrepancy,
                days_since_wound_frac,
                defected_fraction,
                wound_boundary_fraction,
            ],
            1.0,
            "regeneration_state",
        );
        agent.perceive(&obs);
        let result = agent.select_action();
        agent.act(result.action);
        agent.learn_from_outcome(result.action, &obs);

        let probs: Vec<String> = result
            .action_probabilities
            .iter()
            .map(|p| format!("{p:.3}"))
            .collect();
        println!(
            "{day:>4} {discrepancy:>11.4} {wound_boundary_fraction:>10.4} {eligible_count:>9} | [{:>44}] | {:>6}",
            probs.join(", "),
            ACTION_LABELS[result.action]
        );
    }

    println!();
    println!(
        "The near-deterministic-argmax hypothesis this investigation set out to test is \
         REFUTED by the data above: action_probabilities stay close to uniform (~0.25 \
         each) throughout, and the chosen action genuinely varies day to day (0.0x, \
         2.0x, 0.5x, 1.0x, ...). The agent really is sampling a fairly flat distribution \
         -- it isn't behaving deterministically at all."
    );
    println!(
        "The 'eligible' column reveals the real explanation instead: it is 0 on every \
         single day. regenerative_proliferate_with_boost only ever affects cells that \
         are BOTH wound-boundary AND still progenitor-type -- and by the time this \
         scenario's 20-day-matured tissue gets amputated, essentially none of the \
         wound-adjacent cells are still progenitors (they've already differentiated). \
         So no matter which multiplier gets chosen -- by the agent OR the legacy flat \
         rate -- there is nothing left for it to act on. This isn't about agent \
         behavior at all: the entire proliferation-boost mechanism has zero effect in \
         this specific scenario, for any controller. That's what actually explains why \
         regeneration_statistical_replication.rs found an exact 0.0000 difference \
         across all 30 seeds, and it's a real gap in that scenario's design (not the \
         mechanism itself), worth fixing by choosing an amputation/maturation \
         combination that actually leaves eligible cells at the wound if this agent is \
         developed further."
    );
}
