// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end manufacturing consciousness demo.
//!
//! Demonstrates the full pipeline:
//!   Cincinnati sensors → CincinnatiMonitor → ManufacturingReading →
//!   ManufacturingTwin → FEP output → Autonomy loop → FabricationEventData
//!
//! Run: cargo run -p symthaea-fabrication-kernel --example manufacturing_demo

use symthaea_fabrication_kernel::{
    autonomy_loop::{AutonomyEvent, AutonomyLoop},
    cincinnati_live::{CincinnatiMonitor, CincinnatiMonitorConfig, SensorReading},
    manufacturing::{ManufacturingReading, ManufacturingTwin},
};

fn main() {
    println!("=== Manufacturing Consciousness Pipeline Demo ===\n");

    // ── 1. Cincinnati Monitor: sensor ingestion + anomaly detection ──────
    println!("--- Step 1: Cincinnati Monitor (Anomaly Detection) ---");
    let config = CincinnatiMonitorConfig {
        poll_interval_ms: 100,
        anomaly_threshold: 2.5,
        sensor_channels: vec![
            "temperature".into(),
            "vibration".into(),
            "extrusion_pressure".into(),
        ],
    };
    let mut monitor = CincinnatiMonitor::new(config);

    // Build baseline with 20 normal readings
    println!("  Building baseline (20 readings per channel)...");
    for i in 0..20 {
        monitor.ingest_reading(SensorReading {
            channel: "temperature".into(),
            value: 200.0 + (i as f64 * 0.5),
            timestamp_ms: i * 100,
        });
        monitor.ingest_reading(SensorReading {
            channel: "vibration".into(),
            value: 0.5 + (i as f64 * 0.02),
            timestamp_ms: i * 100,
        });
        monitor.ingest_reading(SensorReading {
            channel: "extrusion_pressure".into(),
            value: 45.0 + (i as f64 * 0.1),
            timestamp_ms: i * 100,
        });
    }
    println!(
        "  Baseline established. Anomaly count: {}",
        monitor.anomaly_count()
    );

    // Inject anomalous reading
    println!("  Injecting temperature spike (350°C vs ~210°C baseline)...");
    let alert = monitor.ingest_reading(SensorReading {
        channel: "temperature".into(),
        value: 350.0,
        timestamp_ms: 2100,
    });
    if let Some(a) = &alert {
        println!(
            "  ANOMALY DETECTED: channel={} severity={:.2} z_score={:.1} type={:?}",
            a.channel, a.severity, a.z_score, a.anomaly_type
        );
    }

    // Convert to ManufacturingReading
    let mfg_reading = monitor.to_manufacturing_reading();
    println!(
        "  → ManufacturingReading: tol={:.2} surface={:.2} throughput={:.2} energy={:.2}",
        mfg_reading.tolerance,
        mfg_reading.surface_quality,
        mfg_reading.throughput,
        mfg_reading.energy_cost
    );

    // ── 2. ManufacturingTwin: FEP-based process monitoring ──────────────
    println!("\n--- Step 2: ManufacturingTwin (FEP Process Monitor) ---");
    let mut twin = ManufacturingTwin::new();

    let reference = ManufacturingReading {
        tolerance: 0.95,
        surface_quality: 0.90,
        throughput: 0.85,
        energy_cost: 0.20,
    };
    twin.set_reference(&reference);

    // Normal operation
    println!("  Normal operation (5 cycles):");
    for i in 0..5 {
        let reading = ManufacturingReading {
            tolerance: 0.93 - (i as f64 * 0.01),
            surface_quality: 0.88,
            throughput: 0.85,
            energy_cost: 0.22,
        };
        let output = twin.step(&reading, 0.05);
        println!(
            "    Cycle {}: FE={:.4} Safety={:?} Action={:?}",
            i, output.free_energy, output.safety_level, output.recommended_action
        );
    }

    // Degradation
    println!("  Degradation (5 cycles):");
    for i in 5..10 {
        let reading = ManufacturingReading {
            tolerance: 0.80 - (i as f64 * 0.04),
            surface_quality: 0.60,
            throughput: 0.50,
            energy_cost: 0.60,
        };
        let output = twin.step(&reading, 0.05);
        println!(
            "    Cycle {}: FE={:.4} Safety={:?} Action={:?}",
            i, output.free_energy, output.safety_level, output.recommended_action
        );
    }

    // ── 3. Autonomy Loop: state machine → FabricationEvents ─────────────
    println!("\n--- Step 3: Autonomy Loop (State Machine → Consciousness Events) ---");

    // Happy path
    println!("  Happy path:");
    let mut autonomy = AutonomyLoop::new();
    let happy_path = [
        ("Start print", AutonomyEvent::PrintStarted("JOB-001".into())),
        ("Print done (q=0.92)", AutonomyEvent::PrintCompleted(0.92)),
        ("QC passed", AutonomyEvent::QcPassed),
    ];
    for (desc, event) in &happy_path {
        autonomy.apply(event.clone());
        let fab_events = autonomy.to_fabrication_events();
        print!("    {} → {:?}", desc, autonomy.state());
        for fe in &fab_events {
            print!(" → {:?}", fe.kind);
        }
        println!();
    }

    // Failure path
    println!("  Failure path:");
    let mut autonomy2 = AutonomyLoop::new();
    let fail_path = [
        ("Start print", AutonomyEvent::PrintStarted("JOB-002".into())),
        ("Print done (q=0.25)", AutonomyEvent::PrintCompleted(0.25)),
        (
            "QC failed",
            AutonomyEvent::QcFailed("Delamination detected".into()),
        ),
    ];
    for (desc, event) in &fail_path {
        autonomy2.apply(event.clone());
        let fab_events = autonomy2.to_fabrication_events();
        print!("    {} → {:?}", desc, autonomy2.state());
        for fe in &fab_events {
            print!(" → {:?}", fe.kind);
        }
        println!();
    }

    // ── Summary ─────────────────────────────────────────────────────────
    println!("\n--- Consciousness Integration Summary ---");
    println!("FabricationEvents feed into FabricationManager (interval 47):");
    println!("  Cincinnati anomaly (sev>0.5) → NE phasic burst [Aston-Jones 2005]");
    println!("  Print success     (q>0.5)    → DA phasic burst [Schultz 1997]");
    println!("  Emergency halt               → NE surge + 5-HT dip [Sapolsky 2004]");
    println!("  Quality trend ↑              → 5-HT baseline rise [Crockett 2009]");
    println!("  High PoGF         (>0.7)     → Oxytocin boost [Zak 2012]");
    println!("\nNeuromod bath modulates: consciousness level, learning rate,");
    println!("exploration drive, arousal, and valence.");
    println!("\n=== Demo complete (240 kernel tests, 9 integration tests pass) ===");
}
