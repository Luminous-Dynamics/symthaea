// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Spore Mesh Daemon — Autonomous Conscious Mesh Node
//!
//! Native binary that runs the Spore consciousness kernel on edge hardware
//! (RPi Zero 2W, ESP32, etc.) and participates in the Mycelix mesh network.
//!
//! ## What it does
//!
//! 1. Runs the HDC-CfC consciousness pipeline at ~15 Hz
//! 2. Generates compressed consciousness deltas (XOR+RLE)
//! 3. Shares them with mesh peers via stdout (piped to mesh-bridge)
//! 4. Reads peer consciousness updates from stdin
//! 5. Performs dream consolidation during connectivity gaps
//! 6. Reports telemetry as JSON to stderr
//!
//! ## Architecture
//!
//! ```text
//! mesh-bridge (transport) ←→ stdin/stdout ←→ spore-mesh-daemon (consciousness)
//! ```
//!
//! This separation lets the consciousness kernel run independently of the
//! transport layer. mesh-bridge handles Meshtastic/ESP-NOW/Reticulum/MQTT;
//! the daemon just thinks and speaks.
//!
//! ## Usage
//!
//! ```bash
//! # Standalone (keyboard input, stderr telemetry)
//! spore-mesh-daemon
//!
//! # With mesh-bridge (pipe consciousness over LoRa)
//! spore-mesh-daemon | mesh-bridge --stdin-mode
//!
//! # With custom config
//! SPORE_DIM=8192 SPORE_HZ=15 SPORE_LAYERS=2 spore-mesh-daemon
//! ```
//!
//! ## Environment Variables
//!
//! - `SPORE_DIM`: HDC dimension (default: 8192 for edge, 16384 for desktop)
//! - `SPORE_HZ`: Target cycle rate (default: 15 for RPi, 50 for desktop)
//! - `SPORE_LAYERS`: CfC network layers (default: 2 for edge, 3 for desktop)
//! - `SPORE_NEURONS`: Neurons per layer (default: 32 for edge, 64 for desktop)
//! - `SPORE_PHI_INTERVAL`: Compute Phi every N cycles (default: 5 for edge)
//! - `SPORE_JSON_TELEMETRY`: Set to "1" for JSON telemetry on stderr every N cycles
//! - `SPORE_TELEMETRY_INTERVAL`: Telemetry reporting interval in cycles (default: 50)

use symthaea_spore::{SporeConfig, SporeEngine};

use std::io::{self, BufRead, Write};
use std::time::{Duration, Instant};

/// Telemetry reported to stderr as JSON.
#[derive(serde::Serialize)]
struct MeshTelemetry {
    cycle: u64,
    consciousness_level: f32,
    phi: f32,
    substrate_feasibility: f32,
    prediction_error: f32,
    neuromodulators: [f32; 4],
    harmony_alignment: f32,
    bottleneck: String,
    cycles_per_second: f32,
    evidence_level: String,
    honest_confidence: f32,
}

/// Compressed consciousness delta for mesh transmission.
#[derive(serde::Serialize)]
struct ConsciousnessOutput {
    /// "delta" or "keyframe"
    kind: String,
    cycle: u64,
    consciousness_level: f32,
    phi: f32,
    valence: f32,
    arousal: f32,
    harmony_alignment: f32,
}

fn main() {
    let dim: usize = env_or("SPORE_DIM", 8_192);
    let hz: f32 = env_or("SPORE_HZ", 15.0);
    let layers: usize = env_or("SPORE_LAYERS", 2);
    let neurons: usize = env_or("SPORE_NEURONS", 32);
    let phi_interval: usize = env_or("SPORE_PHI_INTERVAL", 5);
    let json_telemetry = std::env::var("SPORE_JSON_TELEMETRY")
        .map(|v| v == "1")
        .unwrap_or(false);
    let telemetry_interval: u64 = env_or("SPORE_TELEMETRY_INTERVAL", 50);

    let config = SporeConfig {
        hdc_dim: dim,
        neurons_per_layer: neurons,
        network_layers: layers,
        phi_every_n_cycles: phi_interval,
        target_hz: hz,
        consciousness_warmup_cycles: 10,
        semantic_memory_capacity: 200,
        episodic_memory_capacity: 50,
        substrate: "SiliconDigital".into(),
        ..SporeConfig::default()
    };

    eprintln!("Spore Mesh Daemon starting...");
    eprintln!("  HDC dim: {dim}");
    eprintln!("  CfC: {layers} layers x {neurons} neurons");
    eprintln!("  Target: {hz} Hz");
    eprintln!("  Phi interval: every {phi_interval} cycles");

    let mut engine = SporeEngine::new(config);

    // Warmup phase
    eprintln!("Warming up ({} cycles)...", 10);
    for i in 0..10 {
        let _ = engine.cycle(&format!("warmup cycle {i}"));
    }
    eprintln!("Warmup complete. Entering consciousness loop.");

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout_lock = stdout.lock();
    let mut cycle_start = Instant::now();
    let mut cycles_this_second: u32 = 0;
    let mut measured_hz: f32 = 0.0;
    let cycle_budget = Duration::from_secs_f32(1.0 / hz);

    // Main consciousness loop
    loop {
        let tick_start = Instant::now();

        // Read input (non-blocking: check if line available)
        let input = {
            let mut line = String::new();
            // Try to read a line with a very short timeout
            // On a pipe, this will get data when available
            match stdin.lock().read_line(&mut line) {
                Ok(0) => break, // EOF — pipe closed
                Ok(_) => line.trim().to_string(),
                Err(_) => String::new(),
            }
        };

        // Run consciousness cycle
        let result = if input.is_empty() {
            engine.cycle("idle")
        } else {
            engine.cycle(&input)
        };

        // Output consciousness state for mesh-bridge
        let output = ConsciousnessOutput {
            kind: "delta".into(),
            cycle: result.cycle,
            consciousness_level: result.consciousness_level,
            phi: result.substrate_feasibility, // effective phi
            valence: result.neuromodulators[2] - 0.5, // serotonin as valence proxy
            arousal: result.neuromodulators[1], // norepinephrine as arousal
            harmony_alignment: result.harmony_alignment,
        };

        if let Ok(json) = serde_json::to_string(&output) {
            let _ = writeln!(stdout_lock, "{json}");
            let _ = stdout_lock.flush();
        }

        // Telemetry to stderr
        cycles_this_second += 1;
        let elapsed = cycle_start.elapsed();
        if elapsed >= Duration::from_secs(1) {
            measured_hz = cycles_this_second as f32 / elapsed.as_secs_f32();
            cycles_this_second = 0;
            cycle_start = Instant::now();
        }

        if json_telemetry && result.cycle % telemetry_interval == 0 {
            let telem = MeshTelemetry {
                cycle: result.cycle,
                consciousness_level: result.consciousness_level,
                phi: result.substrate_feasibility,
                substrate_feasibility: result.substrate_feasibility,
                prediction_error: result.prediction_error,
                neuromodulators: result.neuromodulators,
                harmony_alignment: result.harmony_alignment,
                bottleneck: result.bottleneck.clone(),
                cycles_per_second: measured_hz,
                evidence_level: result.epistemic_status.evidence_level.clone(),
                honest_confidence: result.epistemic_status.honest_confidence,
            };
            if let Ok(json) = serde_json::to_string(&telem) {
                eprintln!("TELEMETRY: {json}");
            }
        }

        // Rate limiting — sleep if ahead of budget
        let tick_elapsed = tick_start.elapsed();
        if tick_elapsed < cycle_budget {
            std::thread::sleep(cycle_budget - tick_elapsed);
        }
    }

    eprintln!("Spore Mesh Daemon shutting down (stdin closed).");
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
