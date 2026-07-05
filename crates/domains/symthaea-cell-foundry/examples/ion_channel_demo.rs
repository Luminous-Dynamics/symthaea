// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ion Channel Demo
//!
//! Demonstrates the opt-in ion-channel conductance model
//! (`NeuralOrganoid::set_ion_channel_model_enabled`, see
//! `crate::ion_channels`/`crate::bioelectric`) against the ad hoc
//! fate-drift dynamics it can replace, and against a pharmacological
//! phenocopy: blocking K+ channels prevents the Vmem hyperpolarization
//! that would otherwise accompany normal genetic differentiation.
//!
//! Runs the same seeded developmental trajectory three ways:
//! - `legacy`: the default ad hoc drift toward one of two hardcoded target
//!   voltages.
//! - `ion_channel_model`: the opt-in conductance-weighted model, Vmem
//!   settling toward real Nernst-derived reversal potentials.
//! - `ion_channel_model_k_blocked`: the same model with K+ channels fully
//!   pharmacologically blocked (`set_potassium_channel_block(0.0)`) --
//!   cells still commit to Neuron/Glial fate genetically (same activator/
//!   gene-expression program), but can no longer electrically hyperpolarize.
//!
//! Run: cargo run -p symthaea-cell-foundry --example ion_channel_demo

use symthaea_cell_foundry::morphogenetic_consciousness::NeuralOrganoid;

struct DayStats {
    day: u32,
    neuron_glial_count: usize,
    mean_vmem_all: f32,
    mean_vmem_neuron_glial: Option<f32>,
}

fn stats(organoid: &NeuralOrganoid, day: u32) -> DayStats {
    let n = organoid.field.num_cells();
    let vmem = &organoid.field.bioelectric.vmem;
    let mean_vmem_all = vmem.iter().sum::<f32>() / n as f32;

    let mut ng_sum = 0.0f32;
    let mut ng_count = 0usize;
    for i in 0..n {
        let ct = organoid.field.cells[i].cell_type;
        if ct.is_neuron() || ct.is_glial() {
            ng_sum += vmem[i];
            ng_count += 1;
        }
    }
    let mean_vmem_neuron_glial = if ng_count > 0 {
        Some(ng_sum / ng_count as f32)
    } else {
        None
    };

    DayStats {
        day,
        neuron_glial_count: ng_count,
        mean_vmem_all,
        mean_vmem_neuron_glial,
    }
}

fn run(
    seed: u64,
    cells: usize,
    days: u32,
    sample_every: u32,
    ion_channel_model: bool,
    potassium_block: f32,
) -> Vec<DayStats> {
    let mut organoid = NeuralOrganoid::new(cells, seed);
    organoid.set_ion_channel_model_enabled(ion_channel_model);
    if ion_channel_model {
        organoid.set_potassium_channel_block(potassium_block);
    }
    let mut trace = vec![stats(&organoid, 0)];
    for day in 1..=days {
        organoid.advance_day();
        if day % sample_every == 0 {
            trace.push(stats(&organoid, day));
        }
    }
    trace
}

fn print_trace(label: &str, trace: &[DayStats]) {
    println!("  {label}:");
    println!(
        "    {:>4} {:>10} {:>14} {:>18}",
        "day", "n/g cells", "mean vmem (all)", "mean vmem (n/g)"
    );
    for s in trace {
        let ng = s
            .mean_vmem_neuron_glial
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "--".to_string());
        println!(
            "    {:>4} {:>10} {:>14.4} {:>18}",
            s.day, s.neuron_glial_count, s.mean_vmem_all, ng
        );
    }
}

fn main() {
    let seed = 42;
    let cells = 150;
    let days = 60;
    let sample_every = 5;

    println!("Running {days}-day trajectories with {cells} cells (seed={seed})...\n");

    println!("Legacy dynamics (ad hoc fate-drift, the default):");
    let legacy = run(seed, cells, days, sample_every, false, 1.0);
    print_trace("legacy", &legacy);
    println!();

    println!("Ion-channel model (Nernst-derived reversal potentials, K+ open):");
    let ion_channel_open = run(seed, cells, days, sample_every, true, 1.0);
    print_trace("ion_channel_model", &ion_channel_open);
    println!();

    println!("Ion-channel model with K+ channels pharmacologically blocked:");
    let ion_channel_blocked = run(seed, cells, days, sample_every, true, 0.0);
    print_trace("ion_channel_model_k_blocked", &ion_channel_blocked);
    println!();

    let final_ng = |trace: &[DayStats]| trace.last().and_then(|s| s.mean_vmem_neuron_glial);
    match (
        final_ng(&legacy),
        final_ng(&ion_channel_open),
        final_ng(&ion_channel_blocked),
    ) {
        (Some(l), Some(o), Some(b)) => {
            println!(
                "Final mean Vmem of Neuron/Glial cells: legacy={l:.4}, ion_channel_open={o:.4}, ion_channel_k_blocked={b:.4}"
            );
            assert!(
                o < b,
                "K+ channel blockade should leave differentiated cells less \
                 hyperpolarized than the open-channel condition: open={o}, blocked={b}"
            );
            println!(
                "K+ channel blockade prevents hyperpolarization despite identical \
                 genetic differentiation -- a bioelectric/genetic dissociation, the \
                 same class of result as this crate's existing gap-junction-blocker \
                 experiments, via a different (ion-channel-specific) mechanism."
            );
        }
        _ => {
            println!(
                "(No Neuron/Glial cells differentiated in one or more conditions \
                 within {days} days -- try more cells, more days, or a different seed.)"
            );
        }
    }
}
