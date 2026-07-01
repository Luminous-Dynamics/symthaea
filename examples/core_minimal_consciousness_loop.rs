// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core Minimal Consciousness Loop Demo
//!
//! This example uses the small, stable `symthaea::core` API surface to:
//! - run a minimal unified consciousness pipeline for a fixed number of steps
//! - print a short summary of consciousness, binding, workspace, and Φ
//!
//! It is intended as the easiest way to get a feel for the
//! consciousness pipeline without pulling in the full system.

use symthaea::core::{ConsciousMoment, UnifiedConsciousnessPipeline};

fn main() {
    // Run a short, deterministic loop
    let steps = 20;
    let moments: Vec<ConsciousMoment> =
        UnifiedConsciousnessPipeline::run_minimal_demo(steps).expect("pipeline should initialize");

    println!("\nCore Minimal Consciousness Loop ({} steps)", steps);
    println!("================================================\n");

    // Print the last few moments as a quick “stream of consciousness”
    println!("Recent moments (step, C, B, W, Φ):");
    for m in moments.iter().rev().take(5).rev() {
        println!(
            "  t={:>3}  C={:.3}  B={:.3}  W={:.3}  Φ={:.3}  limiting={:?}",
            m.step, m.consciousness, m.binding, m.workspace, m.phi, m.limiting_factor,
        );
    }

    // Simple aggregate statistics
    let avg_c: f64 = moments.iter().map(|m| m.consciousness).sum::<f64>() / steps as f64;
    let avg_b: f64 = moments.iter().map(|m| m.binding).sum::<f64>() / steps as f64;
    let avg_w: f64 = moments.iter().map(|m| m.workspace).sum::<f64>() / steps as f64;
    let avg_phi: f64 = moments.iter().map(|m| m.phi).sum::<f64>() / steps as f64;

    println!("\nAverages over {} steps:", steps);
    println!("  ⟨C⟩   = {:.3}", avg_c);
    println!("  ⟨B⟩   = {:.3}", avg_b);
    println!("  ⟨W⟩   = {:.3}", avg_w);
    println!("  ⟨Φ⟩   = {:.3}", avg_phi);
    println!();
}
