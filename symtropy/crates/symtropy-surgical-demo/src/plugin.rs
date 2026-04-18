// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Main plugin.

use bevy::prelude::*;
use symthaea_surgical::simulator::SurgicalPhysicsSimulator;
use symthaea_surgical::types::SurgicalSafetyLevel;

use crate::camera;
use crate::consciousness_bridge;
use crate::hud;
use crate::resources::*;
use crate::visualization;

pub struct SurgicalDemoPlugin;

impl Plugin for SurgicalDemoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SurgicalResources::new())
            .insert_resource(SimTime::default())
            .add_systems(
                Startup,
                (
                    camera::setup_camera,
                    visualization::setup_scene,
                    hud::setup_hud,
                ),
            )
            .add_systems(
                Update,
                (
                    update_sim_time,
                    step_surgical,
                    visualization::update_arm_visual,
                    visualization::update_cautery_glow,
                    visualization::update_critical_structure_pulse,
                    hud::update_hud,
                    print_status,
                )
                    .chain(),
            );
    }
}

fn update_sim_time(time: Res<Time>, mut sim_time: ResMut<SimTime>) {
    sim_time.elapsed += time.delta_secs_f64();
}

fn step_surgical(time: Res<Time>, mut surg: ResMut<SurgicalResources>) {
    let dt = time.delta_secs_f64();
    if dt <= 0.0 || dt > 0.1 {
        return;
    }

    let state = surg.simulator.state().clone();

    // Observation signals:
    // - proximity_norm: 1.0 when at the critical structure, 0.0 when far away
    //   (Simulator returns a distance in mm; 4 mm is "critically close").
    let proximity_norm = ((8.0 - state.critical_structure_distance) / 8.0).clamp(0.0, 1.0);
    // - trocar_norm: already in [0,1] from the simulator.
    let trocar_norm = state.trocar_compliance.clamp(0.0, 1.0);
    // - tip force magnitude (tissue contact)
    let force_mag = state.force_magnitude();
    let force_norm = (force_mag / 3.0).clamp(0.0, 1.0);

    let last_pe = surg.last_prediction_error;
    // Danger: weighted mix of the three proximity-like signals.
    let danger = (proximity_norm * 0.55 + trocar_norm * 0.25 + force_norm * 0.20).min(1.0);

    let (phi, safety, _gain) = consciousness_bridge::consciousness_tick(
        &mut surg.robot_agent,
        last_pe,
        danger,
        proximity_norm,
        trocar_norm,
    );
    surg.current_phi = phi;
    surg.current_safety = safety;

    // Platform-provided Φ → level with torque gain + hard cautery interlock
    let level = SurgicalSafetyLevel::from_phi(phi);
    surg.current_level = level;

    let cmd = surg.controller.compute(&state, level);
    surg.last_effort = cmd.control_effort();
    surg.last_cautery = cmd.cautery;
    surg.last_jaw = cmd.jaw;

    surg.simulator.step(&cmd, dt);

    // HDC → PE
    let new_state = surg.simulator.state().clone();
    let current_hv = surg.encoder.encode(&new_state);
    let pe = if let Some(ref prev) = surg.last_perception {
        let sim = current_hv.similarity(prev);
        (1.0 - sim.max(0.0)).min(1.0) as f32
    } else {
        0.0
    };
    surg.last_prediction_error = pe;
    surg.last_perception = Some(current_hv);
}

fn print_status(sim_time: Res<SimTime>, surg: Res<SurgicalResources>) {
    let tick = (sim_time.elapsed * 2.0) as u64;
    let prev = ((sim_time.elapsed - 0.016) * 2.0) as u64;
    if tick == prev {
        return;
    }
    let st = surg.simulator.state();
    println!(
        "[{:>6.2}s] tip=({:+.1},{:+.1},{:+.1})mm dist_crit={:.2}mm trocar={:.2} force={:.1}N Φ={:.3} level={:?} {:?} cautery={:.2} jaw={:.2} PE={:.3}",
        sim_time.elapsed,
        st.tip_position[0],
        st.tip_position[1],
        st.tip_position[2],
        st.critical_structure_distance,
        st.trocar_compliance,
        st.force_magnitude(),
        surg.current_phi,
        surg.current_level,
        surg.current_safety,
        surg.last_cautery,
        surg.last_jaw,
        surg.last_prediction_error,
    );
}
