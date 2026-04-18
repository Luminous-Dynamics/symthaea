// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Main plugin: wires resources + ordered per-frame systems.

use bevy::prelude::*;
use symthaea_auv::simulator::AuvPhysicsSimulator;

use crate::camera;
use crate::consciousness_bridge;
use crate::controller::gain_scale;
use crate::hud;
use crate::resources::*;
use crate::visualization;

pub struct AuvDemoPlugin;

impl Plugin for AuvDemoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(AuvResources::new())
            .insert_resource(WaypointPath::default())
            .insert_resource(Current::default())
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
                    step_auv,
                    update_waypoint,
                    visualization::update_auv_visual,
                    visualization::update_waypoint_visual,
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

/// Core per-frame loop: current → controller → consciousness → gain → physics → PE.
fn step_auv(
    time: Res<Time>,
    sim_time: Res<SimTime>,
    waypoints: Res<WaypointPath>,
    current: Res<Current>,
    mut auv: ResMut<AuvResources>,
) {
    let dt = time.delta_secs_f64();
    if dt <= 0.0 || dt > 0.1 {
        return;
    }

    auv.controller.target = waypoints.current();

    // 1. Sample underwater current at current depth
    let state = auv.simulator.state().clone();
    let (force, intensity) = current.source.sample(sim_time.elapsed, state.depth);
    auv.last_current_force = force;
    auv.last_current_intensity = intensity;
    auv.simulator.apply_external_force(force);

    // 2. Build waypoint command
    let cmd = auv.controller.compute(&state);

    // 3. Consciousness tick
    let depth_norm = (state.depth / 25.0).clamp(0.0, 1.0);
    let current_norm = intensity;
    let last_pe = auv.last_prediction_error;
    let danger = (intensity + last_pe as f64 * 0.3).min(1.0);
    let (phi, safety, gain) = consciousness_bridge::consciousness_tick(
        &mut auv.robot_agent,
        last_pe,
        danger,
        depth_norm,
        current_norm,
    );
    auv.current_phi = phi;
    auv.current_safety = safety;
    auv.current_motor_gain = gain;

    // 4. Scale by motor gain
    let scaled = gain_scale(cmd, gain);
    auv.last_thruster_effort = scaled.control_effort();

    // 5. Step physics
    auv.simulator.step(&scaled, dt);

    // 6. HDC encode → cosine dissimilarity PE
    let new_state = auv.simulator.state().clone();
    let current_hv = auv.encoder.encode(&new_state);
    let pe = if let Some(ref prev) = auv.last_perception {
        let sim = current_hv.similarity(prev);
        (1.0 - sim.max(0.0)).min(1.0) as f32
    } else {
        0.0
    };
    auv.last_prediction_error = pe;
    auv.last_perception = Some(current_hv);
}

fn update_waypoint(auv: Res<AuvResources>, mut waypoints: ResMut<WaypointPath>) {
    let st = auv.simulator.state();
    waypoints.advance_if_reached(st.position, st.depth, 2.0);
}

fn print_status(sim_time: Res<SimTime>, auv: Res<AuvResources>, waypoints: Res<WaypointPath>) {
    let tick = (sim_time.elapsed * 2.0) as u64;
    let prev = ((sim_time.elapsed - 0.016) * 2.0) as u64;
    if tick == prev {
        return;
    }
    let st = auv.simulator.state();
    let cur_mag = (auv.last_current_force[0].powi(2) + auv.last_current_force[1].powi(2)).sqrt();
    println!(
        "[{:>6.2}s] pos=({:+.1},{:+.1}) depth={:>5.2}m Φ={:.3} {:?} gain={:.2} PE={:.3} effort={:.2} current={:.0}N wp={}/{} lap={}",
        sim_time.elapsed,
        st.position[0],
        st.position[1],
        st.depth,
        auv.current_phi,
        auv.current_safety,
        auv.current_motor_gain,
        auv.last_prediction_error,
        auv.last_thruster_effort,
        cur_mag,
        waypoints.current_index + 1,
        waypoints.waypoints.len(),
        waypoints.laps_completed,
    );
}
