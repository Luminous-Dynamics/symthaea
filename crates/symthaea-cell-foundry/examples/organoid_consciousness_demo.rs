// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Digital Organoid Consciousness Demo
//!
//! Grows a cerebral organoid from 50 stem cells over 200 developmental days,
//! tracking consciousness emergence via Phi measurement.
//!
//! Run: cargo run -p symthaea-cell-foundry --example organoid_consciousness_demo

use symthaea_cell_foundry::digital_organoid::DevelopmentalStage;
use symthaea_cell_foundry::organoid_pipeline::{OrganoidPipeline, OrganoidPipelineConfig};

fn main() {
    println!("======================================================");
    println!("   Digital Organoid Consciousness Emergence Demo");
    println!("======================================================");
    println!("  Growing neural tissue from stem cells...");
    println!("  Monitoring consciousness via Phi measurement");
    println!("  Ethics gate at Phi >= 0.3");
    println!("======================================================");
    println!();

    let config = OrganoidPipelineConfig {
        initial_cells: 50,
        max_cells: 5_000,
        seed: 42,
        ethics_phi_threshold: 0.3,
        target_day: 200,
        enable_lfp_recording: true,
        morphogenetic_steps_per_day: 3,
    };

    let mut pipeline = OrganoidPipeline::new(config);

    println!("Day  | Cells  | Neurons | Syn/N  | Phi      | Stage");
    println!("-----|--------|---------|--------|----------|------------------");

    let result = pipeline.run();

    // Print trajectory (every 5th day + final day)
    for snap in &result.developmental_trajectory {
        if snap.day % 5 == 0 || snap.day == result.final_day {
            println!(
                "{:4} | {:6} | {:5.1}%  | {:5.2}  | {:8.6} | {}",
                snap.day,
                snap.cell_count,
                snap.neuron_fraction * 100.0,
                snap.synapse_density,
                snap.phi,
                stage_name(snap.stage),
            );
        }
    }

    println!();
    println!("=== Results ===");
    println!("Final day:        {}", result.final_day);
    println!("Final Phi:        {:.6}", result.final_phi);
    println!("Peak Phi:         {:.6}", result.peak_phi);
    println!("Cell count:       {}", result.cell_count);
    println!("Neuron count:     {}", result.neuron_count);
    println!("Synapse count:    {}", result.synapse_count);
    println!("Final stage:      {:?}", result.final_stage);
    println!("Ethics halt:      {}", result.halted_by_ethics);
    if let Some(reason) = &result.halt_reason {
        println!("Halt reason:      {}", reason);
    }

    // Consciousness onset analysis
    println!();
    println!("=== Consciousness Onset ===");
    let _phi_curve = pipeline.phi_curve();
    for threshold in [0.01, 0.05, 0.1, 0.2, 0.3] {
        if let Some(day) = pipeline.consciousness_onset_day(threshold) {
            println!("Phi > {:.2} at day: {}", threshold, day);
        }
    }

    // LFP analysis for last recorded day
    if let Some(last) = result.developmental_trajectory.last() {
        if let Some(lfp) = &last.lfp {
            println!();
            println!("=== LFP Power Spectrum (Day {}) ===", last.day);
            println!("Delta (0.5-4 Hz):  {:.4}", lfp.delta_power);
            println!("Theta (4-8 Hz):    {:.4}", lfp.theta_power);
            println!("Alpha (8-13 Hz):   {:.4}", lfp.alpha_power);
            println!("Beta (13-30 Hz):   {:.4}", lfp.beta_power);
            println!("Gamma (30-100 Hz): {:.4}", lfp.gamma_power);
        }
    }
}

fn stage_name(stage: DevelopmentalStage) -> &'static str {
    match stage {
        DevelopmentalStage::EarlyProliferation => "Early Proliferation",
        DevelopmentalStage::NeuralInduction => "Neural Induction",
        DevelopmentalStage::Patterning => "Patterning",
        DevelopmentalStage::Synaptogenesis => "Synaptogenesis",
        DevelopmentalStage::MaturationI => "Maturation I",
        DevelopmentalStage::MaturationII => "Maturation II",
    }
}
