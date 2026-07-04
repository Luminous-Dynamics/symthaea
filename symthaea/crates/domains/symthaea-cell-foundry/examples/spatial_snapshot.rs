// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spatial Snapshot Capture
//!
//! Captures per-cell (position, Vmem, wound-boundary) snapshots over time
//! for two comparisons, small and short enough to embed in a visualization:
//!
//! - `amputate_open` / `amputate_blocked`: the same cut, gap junctions open
//!   vs. blocked (neighbour-propagation recovery).
//! - `scramble_blocked_no_homing` / `scramble_blocked_homing`: the same
//!   total Vmem scramble, gap junctions blocked in both, with vs. without
//!   positional homing (the mechanism that can recover even with no
//!   surviving neighbour template).
//!
//! Writes `spatial_snapshot.json`: an array of `{label, frames: [{day,
//! cells: [{x,y,z,v,w}]}]}`.
//!
//! Run: cargo run -p symthaea-cell-foundry --example spatial_snapshot

use serde::Serialize;
use symthaea_cell_foundry::experiments::build_radial_bipolar_template;
use symthaea_cell_foundry::morphogenetic_consciousness::NeuralOrganoid;

#[derive(Serialize)]
struct CellSnapshot {
    x: f32,
    y: f32,
    z: f32,
    v: f32,
    w: bool,
}

#[derive(Serialize)]
struct Frame {
    day: u32,
    cells: Vec<CellSnapshot>,
}

#[derive(Serialize)]
struct RunResult {
    label: String,
    frames: Vec<Frame>,
}

fn round3(v: f32) -> f32 {
    (v * 1000.0).round() / 1000.0
}

fn snapshot(organoid: &NeuralOrganoid, day: u32) -> Frame {
    let cells = (0..organoid.field.num_cells())
        .map(|i| {
            let p = organoid.field.cells[i].position;
            CellSnapshot {
                x: round3(p[0]),
                y: round3(p[1]),
                z: round3(p[2]),
                v: round3(organoid.field.bioelectric.vmem[i]),
                w: organoid.field.bioelectric.wound_boundary[i],
            }
        })
        .collect();
    Frame { day, cells }
}

fn run_and_capture(
    mut organoid: NeuralOrganoid,
    recovery_days: u32,
    sample_every: u32,
    label: &str,
) -> RunResult {
    let mut frames = vec![snapshot(&organoid, 0)];
    for day in 1..=recovery_days {
        organoid.advance_day();
        if day % sample_every == 0 {
            frames.push(snapshot(&organoid, day));
        }
    }
    RunResult {
        label: label.to_string(),
        frames,
    }
}

fn main() {
    let cells = 80;
    let maturation_days = 10;
    let boundary_r = 0.2;
    let recovery_days = 16;
    let sample_every = 2;

    println!(
        "Building small template ({cells} cells, {maturation_days}d maturation, r={boundary_r})..."
    );
    let template = build_radial_bipolar_template(31, cells, maturation_days, boundary_r);
    println!("  template cells: {}", template.field.num_cells());
    println!();

    let mut runs = Vec::new();

    // Condition set A: same cut, open vs blocked.
    for (permeability, label) in [(1.0f32, "amputate_open"), (0.0f32, "amputate_blocked")] {
        let mut o = template.clone();
        o.set_gap_junction_permeability(permeability);
        o.amputate(0.8, 2.0);
        println!("Running {label} (permeability={permeability})...");
        runs.push(run_and_capture(o, recovery_days, sample_every, label));
    }

    // Condition set B: same total scramble, gap junctions blocked in both,
    // with vs. without positional homing.
    for (homing, label) in [
        (false, "scramble_blocked_no_homing"),
        (true, "scramble_blocked_homing"),
    ] {
        let mut o = template.clone();
        o.set_gap_junction_permeability(0.0);
        o.set_positional_homing(homing);
        o.scramble_vmem(777);
        println!("Running {label} (homing={homing})...");
        runs.push(run_and_capture(o, recovery_days, sample_every, label));
    }

    println!();
    for r in &runs {
        println!(
            "  {}: {} frames, {} cells at final frame",
            r.label,
            r.frames.len(),
            r.frames.last().unwrap().cells.len()
        );
    }

    let json = serde_json::to_string(&runs).expect("serialize");
    std::fs::write("spatial_snapshot.json", &json).expect("write spatial_snapshot.json");
    println!();
    println!("Wrote spatial_snapshot.json ({} bytes)", json.len());
}
