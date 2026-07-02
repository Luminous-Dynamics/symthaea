// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full pipeline integration: biometrics → Φ → proc-gen → macro bridge → physics.
//!
//! Demonstrates all consciousness-physics coupling systems working together
//! in a single simulation loop. This is the "plug in a heart rate monitor,
//! watch the dungeon reshape" pipeline.
//!
//! Run: cargo run --example full_pipeline -p symtropy-consciousness-physics

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::{
    biometrics::{BiometricHistory, BiometricSample},
    coupling::ConsciousnessField,
    macro_bridge::{MacroWorldModifiers, apply_macro_modifiers},
    proc_gen::{EnvironmentEvent, PhiEnvironmentCoupler},
};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const NUM_AGENTS: usize = 8;
const TICKS: usize = 600; // 10 seconds at 60Hz
const DT: f64 = 1.0 / 60.0;

fn main() {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  Symtropy Full Pipeline Integration");
    eprintln!("  biometrics → Φ → proc-gen → macro bridge → physics");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    // ── 1. Create physics world and consciousness field ──────────────
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, -9.81]));
    let mut field = ConsciousnessField::<2>::new();

    let mut handles = Vec::new();
    for i in 0..NUM_AGENTS {
        let x = i as f64 * 4.0;
        let h = world.add_sphere(Point::new([x, 5.0]), 1.0, 1.0);
        field.register(h, 200.0, 3.0);
        handles.push(h);
    }

    // ── 2. Create subsystems ─────────────────────────────────────────
    // Biometric history (simulating a player's physiological signals)
    let mut biometrics = BiometricHistory::new(30);

    // Proc-gen coupler (environment responds to collective Φ)
    let mut env_coupler = PhiEnvironmentCoupler::new(60);

    // Macro bridge modifiers (civilization-sim → physics)
    let mut macro_mods = MacroWorldModifiers::from_sim(
        0.7,   // stability
        0.1,   // oppression
        0.3,   // gini
        false, // no emergency
        0.6,   // production
        0.05,  // faction harmony bias
    );

    // ── 3. Simulation loop ───────────────────────────────────────────
    eprintln!(
        "Running {} ticks ({:.1}s at {:.0} Hz)...\n",
        TICKS,
        TICKS as f64 * DT,
        1.0 / DT
    );

    let mut total_events = 0usize;
    let mut regime_transitions = 0usize;

    for tick in 0..TICKS {
        // ── Phase A: Simulate biometric input ────────────────────────
        // Heart rate rises over time (simulating exercise/stress)
        let t = tick as f64 / TICKS as f64;
        let hr = 60.0 + 40.0 * t; // 60 → 100 bpm
        let hrv = 80.0 - 50.0 * t; // 80 → 30 ms (decreasing with stress)
        let gamma = 0.3 + 0.5 * (1.0 - t); // decreasing gamma coherence
        let gsr = 2.0 + 15.0 * t; // rising galvanic skin response

        biometrics.push(BiometricSample {
            heart_rate_bpm: Some(hr),
            hrv_rmssd: Some(hrv),
            eeg_gamma_coherence: Some(gamma),
            gsr_microsiemens: Some(gsr),
            respiration_bpm: Some(12.0 + 6.0 * t),
            eeg_alpha_power: Some(30.0 - 15.0 * t),
            eeg_theta_power: Some(10.0 + 5.0 * t),
            eeg_beta_power: Some(8.0 + 4.0 * t),
        });

        // ── Phase B: Convert biometrics → consciousness inputs ──────
        let player_inputs = biometrics.to_consciousness_inputs();

        // ── Phase C: Update all agents' consciousness ────────────────
        // Player (agent 0) gets biometric inputs; others get default
        let default_inputs = ConsciousnessInputs {
            phi: 0.5,
            broadcast: 0.5,
            working_memory: 0.5,
            attention: 0.5,
            recurrence: 0.5,
            embodiment: 0.5,
            knowledge: 0.5,
            synchrony: 0.5,
        };

        for (i, &h) in handles.iter().enumerate() {
            let inputs = if i == 0 {
                &player_inputs
            } else {
                &default_inputs
            };
            let pos = world
                .body(h)
                .map(|b| {
                    let p = b.position();
                    Point::new([p[0], p[1]])
                })
                .unwrap_or_else(Point::origin);
            field.update_entity(h, inputs, pos);
        }

        // ── Phase D: Macro bridge — civilization modifiers → physics ─
        // Simulate governance destabilization over time
        if tick == TICKS / 2 {
            eprintln!(
                "  [tick {}] Emergency declared! Governance destabilizing.",
                tick
            );
            macro_mods = MacroWorldModifiers::from_sim(
                0.3,  // stability drops
                0.4,  // oppression rises
                0.5,  // inequality increases
                true, // emergency active!
                0.4,  // production falls
                -0.1, // faction disharmony
            );
        }

        apply_macro_modifiers(&mut field, &macro_mods, 0.1);

        // ── Phase E: Proc-gen — collective Φ reshapes environment ────
        let collective_phi: f64 = handles
            .iter()
            .filter_map(|h| field.entities.get(h).map(|e| e.phi()))
            .sum::<f64>()
            / NUM_AGENTS as f64;

        env_coupler.tick(collective_phi);
        let events = env_coupler.drain_events();
        for event in &events {
            match event {
                EnvironmentEvent::RegimeTransition { from, to, phi } => {
                    eprintln!(
                        "  [tick {}] Regime: {} → {} (Φ={:.3})",
                        tick,
                        from.name(),
                        to.name(),
                        phi
                    );
                    regime_transitions += 1;
                }
                EnvironmentEvent::PortalActivated { phi } => {
                    eprintln!("  [tick {}] Portal ACTIVATED (Φ={:.3})", tick, phi);
                }
                EnvironmentEvent::PortalDeactivated { phi } => {
                    eprintln!("  [tick {}] Portal deactivated (Φ={:.3})", tick, phi);
                }
                _ => {}
            }
        }
        total_events += events.len();

        // ── Phase F: Prediction error decay + physics step ───────────
        field.tick_prediction_errors();
        world.step_with_callback(DT, &mut field);

        // ── Telemetry (every 60 ticks = ~1s) ─────────────────────────
        if tick % 60 == 0 {
            let profile = env_coupler.current_profile();
            let alive = handles
                .iter()
                .filter(|h| {
                    field
                        .entities
                        .get(h)
                        .map(|e| e.phi() > 0.0)
                        .unwrap_or(false)
                })
                .count();

            eprintln!(
                "  t={:>4} | Φ_collective={:.3} | regime={:>12} | obstacles={:.2} | wells={} | sanctuaries={} | alive={}/{}",
                tick,
                collective_phi,
                env_coupler.current_profile().regime.name(),
                profile.obstacle_density,
                profile.energy_well_count,
                profile.sanctuary_count,
                alive,
                NUM_AGENTS,
            );
        }
    }

    // ── Results ──────────────────────────────────────────────────────
    let final_phi: f64 = handles
        .iter()
        .filter_map(|h| field.entities.get(h).map(|e| e.phi()))
        .sum::<f64>()
        / NUM_AGENTS as f64;
    let final_profile = env_coupler.current_profile();

    eprintln!("\n═══════════════════════════════════════════════════════════");
    eprintln!("  Results after {} ticks:", TICKS);
    eprintln!("  Final collective Φ:    {:.4}", final_phi);
    eprintln!("  Final regime:          {}", final_profile.regime.name());
    eprintln!("  Environment events:    {}", total_events);
    eprintln!("  Regime transitions:    {}", regime_transitions);
    eprintln!("  Portal active:         {}", final_profile.portal_active);
    eprintln!(
        "  Obstacle density:      {:.2}",
        final_profile.obstacle_density
    );
    eprintln!(
        "  Energy wells:          {}",
        final_profile.energy_well_count
    );
    eprintln!(
        "  Darkness level:        {:.2}",
        final_profile.darkness_level
    );
    eprintln!("═══════════════════════════════════════════════════════════");

    // Sanity assertions
    assert!(
        final_phi >= 0.0 && final_phi <= 1.0,
        "Φ out of range: {}",
        final_phi
    );
    assert!(final_profile.obstacle_density >= 0.0 && final_profile.obstacle_density <= 1.0);
    eprintln!("\n  ✓ All pipeline stages executed successfully.");
}
