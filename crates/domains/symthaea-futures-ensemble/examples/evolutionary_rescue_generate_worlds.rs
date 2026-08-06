// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phase 2.2C engineering prerequisite (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`):
//! serialize the 10 evolutionary-rescue world trajectories (5 train + 5 test seeds) once, so
//! `evolutionary_rescue_architecture_ablation.rs` and any future model-comparison experiment on
//! this scenario family can replay against them without re-running ~110,000 simulation ticks
//! each time.
//!
//! Run: `cargo run --release --example evolutionary_rescue_generate_worlds -p symthaea-futures-ensemble`
//!
//! Expensive (full ~11,000-tick simulation per seed, 10 seeds) -- this is the one script in the
//! Phase 2.2C pair that's supposed to be slow. Writes
//! `crates/domains/symthaea-futures-ensemble/fixtures/evolutionary_rescue/world_<seed>.json`.

#[path = "support/evolutionary_rescue_common.rs"]
mod common;

fn main() {
    let dir = common::fixtures_dir();
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("failed to create fixtures dir {}: {e}", dir.display()));

    let all_seeds: Vec<u64> = common::TRAIN_SEEDS
        .iter()
        .chain(common::TEST_SEEDS.iter())
        .copied()
        .collect();

    println!(
        "== Generating {} evolutionary-rescue world fixtures (format {}) ==",
        all_seeds.len(),
        common::WORLD_FORMAT_VERSION
    );

    for seed in all_seeds {
        let world = common::record_world(seed);
        let path = common::fixture_path(seed);
        let json = serde_json::to_vec_pretty(&world)
            .unwrap_or_else(|e| panic!("failed to serialize world for seed {seed}: {e}"));
        std::fs::write(&path, json)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
        println!(
            "  seed={seed:4}  max_population={:3}  first_extinction_tick={:?}  -> {}",
            world.max_population,
            world.first_extinction_tick,
            path.display()
        );
    }

    println!("\nDone. Run evolutionary_rescue_architecture_ablation next.");
}
