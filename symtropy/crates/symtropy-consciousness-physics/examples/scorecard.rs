// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Comprehensive scorecard: validates all engine systems in a single run.
//!
//! Run: cargo run -p symtropy-consciousness-physics --example scorecard --release
//!
//! Outputs timestamped CSV to stdout, summary to stderr.

use nalgebra::SVector;
use symtropy_consciousness_physics::convergence::ConvergenceDetector;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{
    CollapseRecoveryMode, ConsciousnessField, ThermodynamicConstants,
};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;
use symthaea_consciousness_equation::ConsciousnessInputs;

const AGENTS: usize = 20;
const TICKS: usize = 10_000;
const DT: f64 = 1.0 / 64.0;
const SAMPLE_INTERVAL: usize = 50;

fn main() {
    eprintln!("=== Symtropy Engine Scorecard ===");
    eprintln!("Agents: {AGENTS}, Ticks: {TICKS}, DT: {DT}");

    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    // Research-grade constants: scarcity-driven dynamics
    consciousness.constants = ThermodynamicConstants::research();

    // Energy wells
    let well_positions = vec![
        SVector::from([30.0, 30.0]),
        SVector::from([-30.0, -30.0]),
        SVector::from([0.0, 40.0]),
    ];

    // Spawn agents
    let mut rng = simple_rng(42);
    let mut handles = Vec::new();

    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 80.0;
        let y = (rng_f64(&mut rng) - 0.5) * 80.0;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(body) = world.body_mut(h) {
            body.linear_damping = 0.3;
        }
        consciousness.register(h, consciousness.constants.initial_energy, consciousness.constants.harmony_range);

        if let Some(entity) = consciousness.entities.get_mut(&h) {
            match i % 4 {
                0 => entity.harmony_activations = [0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8],
                1 => entity.harmony_activations = [0.2, 0.1, 0.9, 0.1, 0.1, 0.1, 0.8, 0.2],
                2 => entity.harmony_activations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                _ => entity.harmony_activations = [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.3],
            }
        }
        handles.push(h);
    }

    // Tracking
    let mut jphi_detector = ConvergenceDetector::new(200, 1e-3);
    let mut cooperation_events = 0u64;
    let mut total_rescues = 0u64;
    let mut peak_phi = 0.0f64;
    let mut peak_phi_tick = 0usize;
    let mut cooperation_emerged_tick: Option<usize> = None;
    // Manual energy tracking (ledger only captures collision energy, not maintenance/regen)
    let mut manual_energy_consumed = 0.0f64;
    let mut manual_energy_regenerated = 0.0f64;

    let mut prev_consumed = 0.0f64;
    let mut prev_regenerated = 0.0f64;

    // CSV header
    println!("tick,collective_phi,alive,collapsed,avg_energy,avg_pred_error,jphi,conservation_error,clustering,cooperation_events,rescues,avg_motor_precision");

    for tick in 0..TICKS {
        // 1. Update consciousness
        for &h in &handles {
            let collapsed = consciousness
                .entities
                .get(&h)
                .map(|e| e.energy.is_collapsed())
                .unwrap_or(true);
            let inputs = ConsciousnessInputs {
                phi: if collapsed { 0.0 } else { 0.5 },
                broadcast: 0.6,
                working_memory: 0.5,
                attention: 0.5,
                recurrence: 0.4,
                embodiment: 0.6,
                knowledge: 0.4,
                synchrony: 0.5,
            };
            consciousness.update_entity(h, &inputs, Point::origin());
        }

        // 2. FEP gradient movement
        let agent_data: Vec<(SVector<f64, 2>, [f64; 8])> = handles
            .iter()
            .filter_map(|&h| {
                let body = world.body(h)?;
                let entity = consciousness.entities.get(&h)?;
                Some((body.position().0, entity.harmony_activations))
            })
            .collect();

        let well_data: Vec<(SVector<f64, 2>, f64)> =
            well_positions.iter().map(|&p| (p, 1.0)).collect();

        for &h in &handles {
            let Some(body) = world.body(h) else { continue };
            let Some(entity) = consciousness.entities.get(&h) else { continue };
            if entity.energy.is_collapsed() { continue; }

            let pos = body.position().0;
            let ef = entity.energy.fraction_remaining();
            let harmony = entity.harmony_activations;

            let nearby: Vec<_> = agent_data
                .iter()
                .filter(|(p, _)| (p - pos).norm() > 2.0)
                .cloned()
                .collect();

            let dir = fep_gradient::free_energy_gradient(&pos, ef, &harmony, &nearby, &well_data, None, 0.0);
            if let Some(body) = world.body_mut(h) {
                body.linear_velocity = dir * 30.0;
            }
        }

        // 3. Maintenance + ambient regen + well regen
        let regen_mult = consciousness.resource_regeneration_multiplier();
        for &h in &handles {
            if let Some(entity) = consciousness.entities.get_mut(&h) {
                entity.energy.tick_reset();
                let phi = entity.phi();
                let maintenance = consciousness.constants.consciousness_maintenance_per_tick * (1.0 + phi * 0.5);
                let consumed = entity.energy.consume(maintenance);
                manual_energy_consumed += consumed;
                let ambient = consciousness.constants.ambient_regen_rate * regen_mult;
                let actual_ambient = entity.energy.regenerate(ambient);
                manual_energy_regenerated += actual_ambient;

                // Well proximity regen
                if let Some(body) = world.body(h) {
                    let pos = body.position().0;
                    for &well_pos in &well_positions {
                        if (pos - well_pos).norm() < 40.0 {
                            let well_regen = consciousness.constants.energy_well_regen_rate;
                            let actual_well = entity.energy.regenerate(well_regen);
                            manual_energy_regenerated += actual_well;
                        }
                    }
                }
            }
        }

        // 4. Harmony resonance + offloading + collapse recovery
        let mut tick_rescues = 0u64;
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);
                let (harm_a, harm_b, collapsed_a, collapsed_b) = {
                    let ea = consciousness.entities.get(&ha);
                    let eb = consciousness.entities.get(&hb);
                    match (ea, eb) {
                        (Some(a), Some(b)) => (
                            a.harmony_activations,
                            b.harmony_activations,
                            a.energy.is_collapsed(),
                            b.energy.is_collapsed(),
                        ),
                        _ => continue,
                    }
                };

                let resonance = HarmonyField::<2>::resonance(&harm_a, &harm_b);
                if resonance > 0.5 {
                    let regen = consciousness.constants.harmony_resonance_regen_rate
                        * (resonance - 0.5) * 2.0;
                    if let Some(e) = consciousness.entities.get_mut(&ha) {
                        let actual = e.energy.regenerate(regen);
                        manual_energy_regenerated += actual;
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        let actual = e.energy.regenerate(regen);
                        manual_energy_regenerated += actual;
                    }
                    cooperation_events += 1;

                    // Collapse recovery
                    if resonance > consciousness.constants.collapse_recovery_harmony_threshold {
                        if collapsed_a {
                            if let Some(e) = consciousness.entities.get_mut(&ha) {
                                let actual = e.energy.regenerate(50.0);
                                manual_energy_regenerated += actual;
                                tick_rescues += 1;
                            }
                        }
                        if collapsed_b {
                            if let Some(e) = consciousness.entities.get_mut(&hb) {
                                let actual = e.energy.regenerate(50.0);
                                manual_energy_regenerated += actual;
                                tick_rescues += 1;
                            }
                        }
                    }
                }
            }
        }
        total_rescues += tick_rescues;

        // 4b. Record ALL energy flows in ledger (after all consume/regenerate calls)
        let delta_consumed = manual_energy_consumed - prev_consumed;
        let delta_regenerated = manual_energy_regenerated - prev_regenerated;
        if delta_consumed > 0.0 {
            consciousness.ledger.record_action(delta_consumed, consciousness.collective_phi);
        }
        if delta_regenerated > 0.0 {
            consciousness.ledger.record_regeneration(delta_regenerated);
        }
        prev_consumed = manual_energy_consumed;
        prev_regenerated = manual_energy_regenerated;

        // 5. Prediction errors + physics + thermo
        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        let balance = consciousness.tick_thermodynamics();

        // Track collapses
        let alive = handles.iter()
            .filter(|h| consciousness.entities.get(h).map(|e| !e.energy.is_collapsed()).unwrap_or(false))
            .count();
        let collapsed = AGENTS - alive;

        // Track peak phi
        if consciousness.collective_phi > peak_phi {
            peak_phi = consciousness.collective_phi;
            peak_phi_tick = tick;
        }

        // Track J/Phi convergence
        if let Some(jphi) = balance.joules_per_phi {
            jphi_detector.push(jphi);
        }

        // Track cooperation emergence
        if cooperation_emerged_tick.is_none() {
            let clustering = avg_nearest_neighbor(&world, &handles);
            if clustering < 5.0 {
                cooperation_emerged_tick = Some(tick);
            }
        }

        // Output CSV
        if tick % SAMPLE_INTERVAL == 0 {
            let avg_energy: f64 = handles.iter()
                .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
                .sum::<f64>() / AGENTS as f64;
            let avg_pred_error: f64 = handles.iter()
                .filter_map(|h| consciousness.entities.get(h).map(|e| e.prediction_error))
                .sum::<f64>() / AGENTS as f64;
            let avg_motor: f64 = handles.iter()
                .filter_map(|h| consciousness.entities.get(h).map(|e| e.motor_precision))
                .sum::<f64>() / AGENTS as f64;
            let clustering = avg_nearest_neighbor(&world, &handles);
            let jphi_str = balance.joules_per_phi
                .map(|j| format!("{j:.4}"))
                .unwrap_or_else(|| "N/A".to_string());

            println!(
                "{tick},{:.4},{alive},{collapsed},{avg_energy:.2},{avg_pred_error:.4},{jphi_str},{:.6},{clustering:.2},{cooperation_events},{total_rescues},{avg_motor:.4}",
                consciousness.collective_phi,
                balance.conservation_error,
            );
        }
    }

    // ========= SCORECARD =========
    let final_alive = handles.iter()
        .filter(|h| consciousness.entities.get(h).map(|e| !e.energy.is_collapsed()).unwrap_or(false))
        .count();
    let jphi_converged = jphi_detector.is_converged();
    let jphi_final = if jphi_converged {
        format!("{:.4}", jphi_detector.rolling_mean())
    } else {
        "NOT CONVERGED".to_string()
    };
    let manual_balance = manual_energy_consumed - manual_energy_regenerated;
    let manual_error = if manual_energy_consumed > 0.0 {
        manual_balance.abs() / manual_energy_consumed
    } else {
        0.0
    };

    eprintln!("\n╔══════════════════════════════════════════════════════╗");
    eprintln!("║            SYMTROPY ENGINE SCORECARD                ║");
    eprintln!("╠══════════════════════════════════════════════════════╣");
    eprintln!("║ Final alive:           {:>3}/{:<3}                       ║", final_alive, AGENTS);
    eprintln!("║ Peak Φ:               {:.4} (tick {:<5})              ║", peak_phi, peak_phi_tick);
    eprintln!("║ J/Φ equilibrium:      {:<24}       ║", jphi_final);
    eprintln!("║ Energy consumed:      {:<12.1} J                    ║", manual_energy_consumed);
    eprintln!("║ Energy regenerated:   {:<12.1} J                    ║", manual_energy_regenerated);
    eprintln!("║ Energy balance:       {:<12.1} J ({:.1}%)            ║", manual_balance, manual_error * 100.0);
    eprintln!("║ Cooperation events:   {:<12}                       ║", cooperation_events);
    eprintln!("║ Total rescues:        {:<12}                       ║", total_rescues);
    eprintln!("║ Cooperation emerged:  {:<24}       ║",
        cooperation_emerged_tick.map(|t| format!("tick {t}")).unwrap_or("NEVER".to_string()));
    eprintln!("╚══════════════════════════════════════════════════════╝");

    // Verdicts
    // TRUE conservation check: does entity budget match expected?
    let current_total: f64 = handles.iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
        .sum();
    let initial_total = AGENTS as f64 * consciousness.constants.initial_energy;
    let expected_total = initial_total + manual_energy_regenerated - manual_energy_consumed;
    let budget_error = (current_total - expected_total).abs();
    let budget_error_pct = if expected_total.abs() > 1e-10 {
        budget_error / expected_total * 100.0
    } else {
        0.0
    };

    eprintln!("\n── Consciousness Energy Budget (game layer) ──");
    eprintln!("  Initial total:       {:.1} J ({} × {:.0})", initial_total, AGENTS, consciousness.constants.initial_energy);
    eprintln!("  + Actual regen:      {:.1} J (after cap overflow)", manual_energy_regenerated);
    eprintln!("  - Consumed:          {:.1} J (maintenance + movement)", manual_energy_consumed);
    eprintln!("  = Expected budgets:  {:.1} J", expected_total);
    eprintln!("  Actual budgets:      {:.1} J", current_total);
    eprintln!("  Budget error:        {:.1} J ({:.2}%)", budget_error, budget_error_pct);

    eprintln!("\n── Physics Dissipation (separate system) ──");
    eprintln!("  Damping dissipation: {:.1} J (KE → heat, tracked by integrator)", consciousness.ledger.lifetime_dissipated);
    eprintln!("  Note: damping removes kinetic energy, NOT consciousness energy.");

    eprintln!("\nVerdicts:");
    eprintln!("  Survival:     {}", if final_alive > AGENTS / 2 { "PASS (>50% alive)" } else { "FAIL" });
    eprintln!("  Cooperation:  {}", if cooperation_events > 0 { "PASS (emerged)" } else { "FAIL (never)" });
    eprintln!("  J/Φ:          {}", if jphi_converged { "PASS (converged)" } else { "INCONCLUSIVE" });
    let budget_verdict = if budget_error_pct < 1.0 {
        "PASS (< 1%)".to_string()
    } else {
        format!("FAIL ({:.1}%)", budget_error_pct)
    };
    eprintln!("  Budget:       {budget_verdict}");
}

fn avg_nearest_neighbor(world: &PhysicsWorld<2>, handles: &[symtropy_physics::BodyHandle]) -> f64 {
    let positions: Vec<SVector<f64, 2>> = handles.iter()
        .filter_map(|h| world.body(*h).map(|b| b.position().0))
        .collect();
    if positions.len() < 2 { return f64::MAX; }

    let mut total = 0.0;
    for (i, p) in positions.iter().enumerate() {
        let nearest = positions.iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, q)| (p - q).norm())
            .fold(f64::MAX, f64::min);
        total += nearest;
    }
    total / positions.len() as f64
}

fn simple_rng(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}
fn rng_f64(state: &mut u64) -> f64 {
    *state = simple_rng(*state);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}
