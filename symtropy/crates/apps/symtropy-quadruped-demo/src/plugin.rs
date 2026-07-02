// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Main plugin.

use bevy::prelude::*;
use symthaea_quadruped::simulator::QuadrupedPhysicsSimulator;
use symthaea_quadruped::types::GaitType;

use crate::camera;
use crate::consciousness_bridge;
use crate::hud;
use crate::resources::*;
use crate::visualization;

pub struct QuadrupedDemoPlugin;

impl Plugin for QuadrupedDemoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(QuadrupedResources::new())
            .insert_resource(Terrain::default())
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
                    step_quadruped,
                    visualization::update_base_visual,
                    visualization::update_leg_visual,
                    visualization::update_terrain_cue_visual,
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

/// Core per-frame loop: terrain → consciousness → Φ→gait → controller →
/// physics → HDC PE.
fn step_quadruped(
    time: Res<Time>,
    sim_time: Res<SimTime>,
    terrain: Res<Terrain>,
    mut q: ResMut<QuadrupedResources>,
) {
    let dt = time.delta_secs_f64();
    if dt <= 0.0 || dt > 0.1 {
        return;
    }

    let state = q.simulator.state().clone();

    // 1. Sample terrain roughness (no external-force API — this drives
    //    the observation vector and the danger signal, not the physics)
    let (roughness, _patch_x) = terrain.field.sample(sim_time.elapsed);
    q.last_terrain_roughness = roughness;

    // 2. Consciousness tick
    let height_norm = (state.base_position[2] / 0.45).clamp(0.0, 1.0);
    let last_pe = q.last_prediction_error;
    // Danger: rough terrain + PE-from-stance (if the prior step saw high
    // joint deviation, the HDC already registered a novelty — feeds forward).
    let danger = (roughness * 0.6 + last_pe as f64 * 0.4).min(1.0);
    let (phi, safety, _gain) = consciousness_bridge::consciousness_tick(
        &mut q.robot_agent,
        last_pe,
        danger,
        height_norm,
        roughness,
    );
    q.current_phi = phi;
    q.current_safety = safety;

    // 3. Platform-provided: map Φ → gait; apply to simulator
    let gait = GaitType::from_phi(phi);
    q.current_gait = gait;
    q.simulator.set_gait(gait);

    // 4. Stance-hold PD torques, gated by gait mode
    let cmd = q.controller.compute(&state, gait);
    q.last_effort = cmd.control_effort();

    // 5. Step physics (CPG at gait frequency drives forward motion)
    q.simulator.step(&cmd, dt);

    // 6. HDC → PE
    let new_state = q.simulator.state().clone();
    let current_hv = q.encoder.encode(&new_state);
    let pe = if let Some(ref prev) = q.last_perception {
        let sim = current_hv.similarity(prev);
        (1.0 - sim.max(0.0)).min(1.0) as f32
    } else {
        0.0
    };
    q.last_prediction_error = pe;
    q.last_perception = Some(current_hv);
}

fn print_status(sim_time: Res<SimTime>, q: Res<QuadrupedResources>) {
    let tick = (sim_time.elapsed * 2.0) as u64;
    let prev = ((sim_time.elapsed - 0.016) * 2.0) as u64;
    if tick == prev {
        return;
    }
    let st = q.simulator.state();
    let fwd_vel = st.base_linear_velocity[0];
    println!(
        "[{:>6.2}s] pos=({:+.2},{:+.2},{:+.2})m fwd={:.2}m/s gait={:?} Φ={:.3} {:?} effort={:.2} PE={:.3} rough={:.2}",
        sim_time.elapsed,
        st.base_position[0],
        st.base_position[1],
        st.base_position[2],
        fwd_vel,
        q.current_gait,
        q.current_phi,
        q.current_safety,
        q.last_effort,
        q.last_prediction_error,
        q.last_terrain_roughness,
    );
}
