// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cancer Rescue Snapshot
//!
//! Captures per-cell (position, Vmem, defected, cell-type) snapshots over
//! time for the same scenario as `cancer_rescue_demo.rs` -- a healthy,
//! patterned organoid that develops a permanent cancer-like defected
//! region -- under two conditions: `naive` (default parameters, no
//! intervention) and `rescued` (the specific intervention parameters
//! `cancer_rescue_demo.rs` found via `search_intervention`: gap_junction_
//! permeability=1.0, positional_homing_rate=0.3, gap_junction_diffusion_
//! rate=0.8832, potassium_channel_block=0.6425). This replays that
//! already-discovered configuration rather than re-running the (slow,
//! stochastic) search, so this example is fast and deterministic.
//!
//! Writes `cancer_rescue_snapshot.json`: `{label, frames: [{day, cells:
//! [{x,y,z,v,defected,cell_type}]}]}` for each condition -- same shape
//! convention as `spatial_snapshot.rs`.
//!
//! Run: cargo run -p symthaea-cell-foundry --example cancer_rescue_snapshot

use serde::Serialize;
use symthaea_cell_foundry::build_radial_bipolar_template;
use symthaea_cell_foundry::morphogenetic_consciousness::NeuralOrganoid;

#[derive(Serialize)]
struct CellSnapshot {
    x: f32,
    y: f32,
    z: f32,
    v: f32,
    defected: bool,
    cell_type: &'static str,
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

fn cell_type_label(organoid: &NeuralOrganoid, i: usize) -> &'static str {
    let ct = organoid.field.cells[i].cell_type;
    if ct.is_neuron() {
        "neuron"
    } else if ct.is_glial() {
        "glial"
    } else if ct.is_progenitor() {
        "progenitor"
    } else {
        "undifferentiated"
    }
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
                defected: organoid.field.bioelectric.defected[i],
                cell_type: cell_type_label(organoid, i),
            }
        })
        .collect();
    Frame { day, cells }
}

fn run_and_capture(mut organoid: NeuralOrganoid, days: u32, label: &str) -> RunResult {
    let mut frames = vec![snapshot(&organoid, 0)];
    for day in 1..=days {
        organoid.advance_day();
        frames.push(snapshot(&organoid, day));
    }
    RunResult {
        label: label.to_string(),
        frames,
    }
}

fn main() {
    let seed = 33;
    // Smaller than cancer_rescue_demo.rs's 150 cells / 15 recovery days --
    // that combination let the defected population's faster proliferation
    // rate balloon final cell counts to ~1,900 (a 1.5MB JSON export and a
    // visually cluttered point cloud). This scale keeps the final frame in
    // the low hundreds of cells, which reads far more clearly as a
    // recognizable spatial pattern.
    let cells = 120;
    let maturation_days = 20;
    let boundary_r = 0.2;
    let defection_establish_days = 3;
    let recovery_days = 8;

    println!("Building a healthy, patterned organoid ({cells} cells, seed={seed})...");
    let mut organoid = build_radial_bipolar_template(seed, cells, maturation_days, boundary_r);
    let target = organoid
        .target_morphology
        .clone()
        .expect("captured by build_radial_bipolar_template");

    println!("Inducing a local defection (cancer-analog) near the tissue's core...");
    let marked = organoid.induce_local_defection(|p| {
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        r < 0.5
    });
    println!("  marked {marked} cells as defected");

    println!("Letting the defection establish itself for {defection_establish_days} days...");
    for _ in 0..defection_establish_days {
        organoid.advance_day();
    }
    let template = organoid;

    let mut runs = Vec::new();

    println!("Running naive (default parameters, no intervention)...");
    let mut naive = template.clone();
    naive.target_morphology = Some(target.clone());
    runs.push(run_and_capture(naive, recovery_days, "naive"));

    println!("Running rescued (cancer_rescue_demo.rs's discovered intervention)...");
    let mut rescued = template.clone();
    rescued.target_morphology = Some(target.clone());
    rescued.set_gap_junction_permeability(1.0);
    rescued.set_positional_homing(true);
    rescued.set_positional_homing_rate(0.3);
    rescued.set_gap_junction_diffusion_rate(0.8832);
    rescued.set_ion_channel_model_enabled(true);
    rescued.set_potassium_channel_block(0.6425);
    runs.push(run_and_capture(rescued, recovery_days, "rescued"));

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
    std::fs::write("cancer_rescue_snapshot.json", &json)
        .expect("write cancer_rescue_snapshot.json");
    println!();
    println!("Wrote cancer_rescue_snapshot.json ({} bytes)", json.len());
}
