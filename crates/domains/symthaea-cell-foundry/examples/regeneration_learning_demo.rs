// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regeneration Learning Demo
//!
//! `regeneration_agent_demo.rs` showed a single amputation-recovery episode
//! where a freshly-initialized active-inference agent converged to the same
//! policy as the legacy flat rate -- an honest but weak test, since a
//! cold-start agent has had no experience yet to learn from. This demo asks
//! the more meaningful question: does the *same* agent, given repeated
//! experience across many amputation-recovery cycles on the same organoid
//! (a real experimental paradigm -- planaria in lab studies are routinely
//! cut and allowed to regenerate repeatedly), get any better at picking a
//! proliferation-boost policy over time?
//!
//! This is architecturally already supported with no production-code
//! changes: `NeuralOrganoid::amputate` resets the wound clock but leaves
//! `target_morphology` and the lazily-constructed, persisted
//! `regeneration_agent` untouched, so the same agent instance backs every
//! episode.
//!
//! **Metric note.** An earlier draft of this demo tracked "days to
//! converge" (discrepancy dropping to/below the crate's
//! `MORPHOLOGY_CONVERGENCE_TOLERANCE = 0.05`). Empirically, that threshold
//! is essentially never reached within realistic recovery windows in this
//! model, in either condition -- confirmed by running both legacy and
//! FEP-driven regeneration for 80 days per episode across 10 episodes with
//! zero convergences either way. Rather than report an all-"did not
//! converge" table (true, but uninformative), this demo instead tracks the
//! actual discrepancy value reached after a fixed-length episode, which is
//! a continuous signal available every run.
//!
//! Reports real final-discrepancy-per-episode for both legacy and
//! FEP-driven regeneration, honestly -- legacy has no state to learn from,
//! so it should stay roughly flat across episodes (a useful contrast, not
//! a strawman); whether FEP shows a real trend is an open question this
//! demo answers empirically, not by assertion.
//!
//! Run: cargo run -p symthaea-cell-foundry --example regeneration_learning_demo

use symthaea_cell_foundry::build_radial_bipolar_template;

/// Fixed length of every episode's recovery window.
const EPISODE_DAYS: u32 = 40;
/// Radial boundary for the imposed bipolar target pattern -- matches this
/// crate's own equifinality experiments. Without a real imposed pattern
/// like this, a captured target has little spatial structure to actually
/// recover.
const BOUNDARY_R: f32 = 0.2;

/// Final discrepancy reached at the end of each fixed-length episode.
fn run_episodes(
    seed: u64,
    cells: usize,
    maturation_days: u32,
    num_episodes: u32,
    fep_enabled: bool,
) -> Vec<f64> {
    let mut organoid = build_radial_bipolar_template(seed, cells, maturation_days, BOUNDARY_R);
    organoid.set_fep_regeneration_enabled(fep_enabled);

    let mut final_discrepancies = Vec::new();
    for _episode in 0..num_episodes {
        organoid.amputate(0.6, 2.0);
        for _ in 0..EPISODE_DAYS {
            organoid.advance_day();
        }
        final_discrepancies.push(organoid.morphology_discrepancy().unwrap_or(1.0));
    }
    final_discrepancies
}

fn print_episodes(label: &str, discrepancies: &[f64]) {
    println!("  {label}:");
    for (i, d) in discrepancies.iter().enumerate() {
        println!("    episode {:2}: final discrepancy = {d:.4}", i + 1);
    }
}

fn mean_of_first_and_last_half(discrepancies: &[f64]) -> (f64, f64) {
    let half = discrepancies.len() / 2;
    let first_half_mean = discrepancies[..half].iter().sum::<f64>() / half as f64;
    let second_half_mean =
        discrepancies[half..].iter().sum::<f64>() / (discrepancies.len() - half) as f64;
    (first_half_mean, second_half_mean)
}

fn main() {
    let seed = 11;
    let cells = 150;
    let maturation_days = 20;
    let num_episodes = 10;

    println!(
        "Running {num_episodes} repeated amputation-recovery episodes on the same \
         organoid ({cells} cells, seed={seed}, {maturation_days}d maturation, \
         {EPISODE_DAYS}d recovery window per episode)...\n"
    );

    println!("Legacy dynamics (flat rate every episode -- no state to learn from):");
    let legacy = run_episodes(seed, cells, maturation_days, num_episodes, false);
    print_episodes("legacy", &legacy);
    println!();

    println!("FEP-driven dynamics (same agent instance persists across all episodes):");
    let fep = run_episodes(seed, cells, maturation_days, num_episodes, true);
    print_episodes("fep_driven", &fep);
    println!();

    for (label, discrepancies) in [("legacy", &legacy), ("fep_driven", &fep)] {
        let (first, second) = mean_of_first_and_last_half(discrepancies);
        println!(
            "{label}: mean final discrepancy, first half of episodes = {first:.4}, \
             second half = {second:.4} ({})",
            if second < first {
                "improved"
            } else if second > first {
                "got worse"
            } else {
                "no change"
            }
        );
    }
    println!();
    println!(
        "This is reported as an honest empirical result. A flat legacy trend is \
         expected (it has no learning mechanism at all). Whether the FEP-driven \
         agent shows genuine improvement across repeated experience is the open \
         question this demo actually tests, not a predetermined conclusion."
    );
}
