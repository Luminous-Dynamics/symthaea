//! End-to-end manufacturing consciousness demo.
//!
//! Demonstrates the full pipeline:
//!   ManufacturingReading → ManufacturingTwin → FEP output →
//!   Autonomy loop state transitions → FabricationEventData
//!
//! Run: cargo run -p symthaea-fabrication-kernel --example manufacturing_demo

use symthaea_fabrication_kernel::{
    autonomy_loop::{AutonomyEvent, AutonomyLoop},
    manufacturing::{ManufacturingReading, ManufacturingTwin},
};

fn main() {
    println!("=== Manufacturing Consciousness Pipeline Demo ===\n");

    // ── 1. ManufacturingTwin: FEP-based process monitoring ──────────────
    println!("--- Step 1: ManufacturingTwin (FEP Process Monitor) ---");
    let mut twin = ManufacturingTwin::new();

    // Set reference (ideal) state
    let reference = ManufacturingReading {
        tolerance: 0.95,
        surface_quality: 0.90,
        throughput: 0.85,
        energy_cost: 0.20,
    };
    twin.set_reference(&reference);

    // Simulate 5 cycles of normal operation
    for i in 0..5 {
        let reading = ManufacturingReading {
            tolerance: 0.93 - (i as f64 * 0.01),
            surface_quality: 0.88 - (i as f64 * 0.005),
            throughput: 0.85,
            energy_cost: 0.22 + (i as f64 * 0.01),
        };
        let output = twin.step(&reading, 0.05);
        println!(
            "  Cycle {}: FE={:.4}  Safety={:?}  Action={:?}",
            i, output.free_energy, output.safety_level, output.recommended_action
        );
    }

    // Simulate degradation
    println!("\n  [Degradation event: tolerance dropping]");
    for i in 5..10 {
        let reading = ManufacturingReading {
            tolerance: 0.80 - (i as f64 * 0.05),
            surface_quality: 0.70,
            throughput: 0.60,
            energy_cost: 0.50,
        };
        let output = twin.step(&reading, 0.05);
        println!(
            "  Cycle {}: FE={:.4}  Safety={:?}  Action={:?}",
            i, output.free_energy, output.safety_level, output.recommended_action
        );
    }

    // ── 2. Autonomy Loop: state machine with consciousness bridge ───────
    println!("\n--- Step 2: Autonomy Loop (State Machine → FabricationEvents) ---");
    let mut autonomy = AutonomyLoop::new();

    // Happy path
    let transitions: Vec<(&str, AutonomyEvent)> = vec![
        (
            "Print started",
            AutonomyEvent::PrintStarted("JOB-001".into()),
        ),
        (
            "Print completed (q=0.92)",
            AutonomyEvent::PrintCompleted(0.92),
        ),
        ("QC passed", AutonomyEvent::QcPassed),
    ];

    for (desc, event) in &transitions {
        let ok = autonomy.apply(event.clone());
        let fab_events = autonomy.to_fabrication_events();
        println!(
            "  {} → state={:?} valid={} fab_events={}",
            desc,
            autonomy.state(),
            ok,
            fab_events.len()
        );
        for fe in &fab_events {
            println!("    → {:?}", fe.kind);
        }
    }

    // Failure path
    println!("\n  [Failure path]");
    let mut autonomy2 = AutonomyLoop::new();
    let failures: Vec<(&str, AutonomyEvent)> = vec![
        (
            "Print started",
            AutonomyEvent::PrintStarted("JOB-002".into()),
        ),
        (
            "Print completed (q=0.3)",
            AutonomyEvent::PrintCompleted(0.3),
        ),
        ("QC failed", AutonomyEvent::QcFailed("Delamination".into())),
    ];

    for (desc, event) in &failures {
        autonomy2.apply(event.clone());
        let fab_events = autonomy2.to_fabrication_events();
        println!(
            "  {} → state={:?} fab_events={}",
            desc,
            autonomy2.state(),
            fab_events.len()
        );
        for fe in &fab_events {
            println!("    → {:?}", fe.kind);
        }
    }

    // ── 3. Summary ──────────────────────────────────────────────────────
    println!("\n--- Pipeline Summary ---");
    println!("In the full Symthaea system, FabricationEvents flow into");
    println!("FabricationManager (CognitiveSubsystem, interval 47) which:");
    println!("  • Cincinnati anomaly (sev>0.5) → NE phasic burst [Aston-Jones 2005]");
    println!("  • Print success (q>0.5) → DA phasic burst [Schultz 1997]");
    println!("  • Emergency halt → NE surge + 5-HT dip [Sapolsky 2004]");
    println!("  • Quality trend up → 5-HT baseline rise [Crockett 2009]");
    println!("  • High PoGF (>0.7) → Oxytocin boost [Zak 2012]");
    println!("\nThese modulate consciousness level, learning rate, and exploration.");
    println!("\n=== Demo complete ===");
}
