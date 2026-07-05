// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regeneration Learning Demo
//!
//! `regeneration_agent_demo.rs` showed a single amputation-recovery episode
//! where a freshly-initialized active-inference agent converged to the same
//! policy as the legacy flat rate -- an honest but weak test, since a
//! cold-start agent has had no experience yet to learn from. A first version
//! of this demo tested repeated experience with an *identical* cut every
//! episode, and found the agent settled into the legacy-equivalent policy
//! and never moved -- but that's an ambiguous result: it could mean active
//! inference isn't helping here, or it could just mean an identical,
//! unvarying scenario gives the agent nothing to differentiate or adapt to.
//!
//! This version gives it something to actually adapt to: **wound size
//! varies episode to episode**, cycling through 4 distinct amputation
//! configurations (large/medium-large/medium/small cuts) across 3 full
//! cycles (12 episodes). To isolate "does experience with a given
//! difficulty level help" from "different difficulties just produce
//! different outcomes" (a real confound if not controlled for), this
//! tracks the trend **within each specific configuration** across its three
//! occurrences (episodes 1/5/9 all use the same cut, etc.), not just a
//! blunt first-half-vs-second-half comparison across mixed difficulties.
//!
//! **A second, unexpected finding surfaced while building this.** Two
//! different severity ranges (an initial, harsher one and the current,
//! narrower one below) both produced the same qualitative pattern: the
//! tissue's discrepancy climbs to a fixed ceiling within the first 2-3
//! episodes and then stays there for the rest of the run, *regardless of
//! subsequent cut size* -- a large cut in episode 9 changes the trajectory
//! no more than a small one would. This looks like a real, reproducible
//! property of repeated amputation without full inter-episode recovery
//! time (each episode's fixed window isn't long enough for discrepancy to
//! approach convergence before the next cut lands), not an artifact of one
//! particular parameter choice -- it shows up whether the first cut is
//! catastrophic or comparatively mild. It's reported here as a genuine
//! model finding, not chased away by further tuning: repeatedly stressing
//! this tissue without letting it fully heal between insults drives it
//! toward a stable, elevated-discrepancy attractor rather than a fresh
//! response to each new insult. Whether that's a desirable property (it
//! does have a loose real-biology analogue -- declining regenerative
//! capacity under repeated injury without recovery time is a real
//! phenomenon in several organisms) or an artifact worth investigating
//! further is left as an open question for whoever picks this up next.
//!
//! Still architecturally free: `NeuralOrganoid::amputate` resets the wound
//! clock but leaves `target_morphology` and the persisted
//! `regeneration_agent` untouched, so one agent instance backs every
//! episode regardless of cut size.
//!
//! Reports real per-episode, per-configuration discrepancy honestly -- no
//! predetermined conclusion about whether varying difficulty is what it
//! takes to see a real learning effect.
//!
//! Run: cargo run -p symthaea-cell-foundry --example regeneration_learning_demo

use symthaea_cell_foundry::build_radial_bipolar_template;

/// Fixed length of every episode's recovery window. Longer than the 40
/// days used in the fixed-cut version of this demo: an earlier draft with
/// 40 days here found the tissue saturated near a ceiling discrepancy
/// after its first (largest) cut and never moved again regardless of
/// subsequent cut size -- repeated cuts without enough time to actually
/// approach recovery between them compound into an all-episodes-look-the-
/// same result, which defeats the point of varying difficulty. 60 days
/// gives meaningfully more room to actually recover between cuts.
const EPISODE_DAYS: u32 = 60;
/// Radial boundary for the imposed bipolar target pattern -- matches this
/// crate's own equifinality experiments. Without a real imposed pattern
/// like this, a captured target has little spatial structure to actually
/// recover.
const BOUNDARY_R: f32 = 0.2;
/// Four distinct amputation configs (min_r, max_r), cycled across episodes
/// to vary wound size/location -- large cut (removes more of the tissue)
/// down to small cut. `amputate(min_r, max_r)` removes cells with
/// `r >= min_r && r < max_r`, so a smaller `min_r` means a larger cut.
/// Narrower severity range than an earlier draft ((0.3..0.9) instead of
/// starting as low as an even more catastrophic cut) -- the same
/// saturation problem above was worse the more severe the first cut was.
const AMPUTATION_CONFIGS: [(f32, f32); 4] = [(0.5, 2.0), (0.65, 2.0), (0.8, 2.0), (0.95, 2.0)];
const CONFIG_LABELS: [&str; 4] = ["large", "medium-large", "medium", "small"];
const NUM_CYCLES: u32 = 3;

/// Final discrepancy reached at the end of each fixed-length episode, along
/// with which amputation config index was used that episode.
fn run_episodes(
    seed: u64,
    cells: usize,
    maturation_days: u32,
    fep_enabled: bool,
) -> Vec<(usize, f64)> {
    let mut organoid = build_radial_bipolar_template(seed, cells, maturation_days, BOUNDARY_R);
    organoid.set_fep_regeneration_enabled(fep_enabled);

    let mut results = Vec::new();
    for cycle in 0..NUM_CYCLES {
        for (config_idx, &(min_r, max_r)) in AMPUTATION_CONFIGS.iter().enumerate() {
            let _ = cycle;
            organoid.amputate(min_r, max_r);
            for _ in 0..EPISODE_DAYS {
                organoid.advance_day();
            }
            let discrepancy = organoid.morphology_discrepancy().unwrap_or(1.0);
            results.push((config_idx, discrepancy));
        }
    }
    results
}

fn print_episodes(label: &str, results: &[(usize, f64)]) {
    println!("  {label}:");
    for (i, (config_idx, d)) in results.iter().enumerate() {
        println!(
            "    episode {:2} [{:>12} cut]: final discrepancy = {d:.4}",
            i + 1,
            CONFIG_LABELS[*config_idx]
        );
    }
}

/// For each config, the discrepancy at its 1st vs. its last (3rd)
/// occurrence -- the trend that actually isolates "did repeated experience
/// with this specific difficulty help" from raw difficulty variation.
fn per_config_trend(results: &[(usize, f64)]) -> Vec<(usize, f64, f64)> {
    (0..AMPUTATION_CONFIGS.len())
        .map(|config_idx| {
            let occurrences: Vec<f64> = results
                .iter()
                .filter(|(c, _)| *c == config_idx)
                .map(|(_, d)| *d)
                .collect();
            (
                config_idx,
                *occurrences.first().unwrap(),
                *occurrences.last().unwrap(),
            )
        })
        .collect()
}

fn main() {
    let seed = 11;
    let cells = 150;
    let maturation_days = 20;
    let num_episodes = AMPUTATION_CONFIGS.len() as u32 * NUM_CYCLES;

    println!(
        "Running {num_episodes} amputation-recovery episodes on the same organoid \
         ({cells} cells, seed={seed}, {maturation_days}d maturation, {EPISODE_DAYS}d \
         recovery window per episode), cycling through {} wound sizes {NUM_CYCLES} times \
         each...\n",
        AMPUTATION_CONFIGS.len()
    );

    println!("Legacy dynamics (flat rate every episode -- no state to learn from):");
    let legacy = run_episodes(seed, cells, maturation_days, false);
    print_episodes("legacy", &legacy);
    println!();

    println!("FEP-driven dynamics (same agent instance persists across all episodes):");
    let fep = run_episodes(seed, cells, maturation_days, true);
    print_episodes("fep_driven", &fep);
    println!();

    for (label, results) in [("legacy", &legacy), ("fep_driven", &fep)] {
        println!("{label}: per-configuration trend (1st occurrence -> last occurrence):");
        for (config_idx, first, last) in per_config_trend(results) {
            println!(
                "    {:>12} cut: {first:.4} -> {last:.4} ({})",
                CONFIG_LABELS[config_idx],
                if last < first {
                    "improved"
                } else if last > first {
                    "got worse"
                } else {
                    "no change"
                }
            );
        }
    }
    println!();
    println!(
        "This is reported as an honest empirical result. A flat legacy trend per \
         configuration is expected (it has no learning mechanism at all). Whether \
         the FEP-driven agent shows genuine within-configuration improvement given \
         real varied experience is the open question this demo actually tests, not \
         a predetermined conclusion."
    );
    println!();
    println!(
        "Note: if every configuration's discrepancy converges to nearly the same \
         value regardless of cut size, that's the second finding described in this \
         file's module docs -- repeated amputation without full inter-episode \
         recovery time appears to drive this tissue toward a stable, elevated-\
         discrepancy attractor rather than responding freshly to each new cut. \
         That's a real, reproducible model property surfaced while building this \
         demo, not a bug being papered over."
    );
}
