// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Thermodynamic Observatory
//!
//! Runs the cognitive loop with thermodynamic modules enabled,
//! observing the unified state and physics bridge in real time.
//!
//! Run: `cargo run --example thermodynamic_observatory`

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Symthaea Thermodynamic Observatory                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let config = CognitiveLoopConfig {
        enable_consciousness_thermodynamics: true,
        enable_hierarchical_free_energy: true,
        enable_primitive_consciousness: true,
        enable_virtual_body: true,
        learning_threshold: 0.0,
        ..CognitiveLoopConfig::with_cfc()
    };

    let mut svc = match CognitiveLoopService::new(config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed: {e}");
            return;
        }
    };

    let inputs = [
        "analyzing patterns in consciousness data",
        "integrating information across modalities",
        "exploring novel cognitive architectures",
        "reflecting on thermodynamic equilibrium",
        "ALERT unexpected phase transition cascade entropy spike",
    ];

    println!(
        "{:>4} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6} {:>7}",
        "Cyc", "Entropy", "FE", "Temp", "Health", "Carnot", "Onsagr", "Insgt", "Phase"
    );
    println!("{}", "─".repeat(72));

    for i in 0..60 {
        // Perturbation at cycles 30-39
        let input = if (30..40).contains(&i) {
            inputs[4]
        } else {
            inputs[i % 4]
        };

        let _result = svc.cycle(input);
        let s = svc.thermodynamic_state();
        let b = svc.thermodynamic_bridge();

        let phase_str = format!("{:?}", s.consciousness_phase);
        let phase_short = &phase_str[..phase_str.len().min(7)];

        if i % 3 == 0 || i == 30 || b.insight_detected {
            println!(
                "{:>4} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.4} {:>6} {:>7}",
                i,
                s.canonical_entropy,
                s.canonical_free_energy,
                s.effective_temperature,
                s.dissipative_health,
                b.carnot_efficiency,
                b.onsager_asymmetry,
                if b.insight_detected { "YES" } else { "-" },
                phase_short,
            );
        }
    }

    println!("\n── Final State ─────────────────────────────────────────────────\n");
    let s = svc.thermodynamic_state();
    let b = svc.thermodynamic_bridge();
    println!(
        "  Entropy={:.4}  FE={:.4}  Temp={:.4}  Health={:.4}",
        s.canonical_entropy, s.canonical_free_energy, s.effective_temperature, s.dissipative_health
    );
    println!(
        "  Order={:.4}  Lambda={:.4}  HFE={:.4}",
        s.order_parameter, s.lambda_parameter, s.hierarchical_free_energy
    );
    println!(
        "  Carnot={:.4}  Attention_bits={:.1}  Landauer={:.4}",
        b.carnot_efficiency, b.attention_demon_bits, b.memory_landauer_cost
    );
    println!(
        "  Onsager={:.6}  Jarzynski_dF={:.6}  Prigogine={}",
        b.onsager_asymmetry, b.jarzynski_free_energy, b.prigogine_violated
    );
    println!(
        "  Insight_prob={:.6}  Memory_suppressed={}",
        b.insight_probability,
        svc.thermodynamic_memory_suppressed()
    );
    println!("  Regime={:?}  Phase={:?}", s.regime, s.consciousness_phase);
    println!("\n  ✓ Observatory complete (60 cycles)");
}