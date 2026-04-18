// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Main plugin.

use bevy::prelude::*;
use symthaea_orbital::simulator::OrbitalPhysicsSimulator;

use crate::camera;
use crate::consciousness_bridge;
use crate::controller::gain_scale;
use crate::hud;
use crate::resources::*;
use crate::visualization;

pub struct OrbitalDemoPlugin;

impl Plugin for OrbitalDemoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(OrbitalResources::new())
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
                    step_orbital,
                    visualization::update_spacecraft_visual,
                    visualization::update_arm_visual,
                    visualization::update_lighting_cue,
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

/// Core per-frame loop: controller → consciousness → gain → physics → PE.
///
/// Internal dynamics (arm torques → reaction-wheel dump → attitude) all
/// live inside the simulator; no external-force injection.
fn step_orbital(time: Res<Time>, mut orbital: ResMut<OrbitalResources>) {
    let dt = time.delta_secs_f64();
    if dt <= 0.0 || dt > 0.1 {
        return;
    }

    // Compress 90-min orbit into something visible: 1 real-second = 60 sim-seconds.
    let sim_dt = dt * 60.0;

    let state = orbital.simulator.state().clone();

    // Build the raw PD command
    let cmd = orbital.controller.compute(&state);

    // Derive observation signals from orbital state
    let attitude_drift = (state
        .spacecraft_angular_velocity
        .iter()
        .map(|v| v * v)
        .sum::<f64>())
    .sqrt();
    // attitude_drift in rad/s; 0.02 rad/s ≈ "tumbling"
    let drift_norm = (attitude_drift / 0.02).clamp(0.0, 1.0);
    let last_pe = orbital.last_prediction_error;
    // Danger rises when out of comm window, in eclipse, or tumbling.
    let danger =
        ((1.0 - state.comm_window) * 0.4 + (1.0 - state.solar_exposure) * 0.2 + drift_norm * 0.4)
            .min(1.0);

    let (phi, safety, gain) = consciousness_bridge::consciousness_tick(
        &mut orbital.robot_agent,
        last_pe,
        danger,
        state.comm_window,
        drift_norm,
    );
    orbital.current_phi = phi;
    orbital.current_safety = safety;
    orbital.current_motor_gain = gain;

    // Scale + step
    let scaled = gain_scale(cmd, gain);
    orbital.last_effort = scaled.control_effort();
    orbital.simulator.step(&scaled, sim_dt);

    // Accumulate spacecraft attitude for the visualization (scaled-axis integration)
    let av = orbital.simulator.state().spacecraft_angular_velocity;
    let axis = Vec3::new(av[0] as f32, av[1] as f32, av[2] as f32) * sim_dt as f32;
    let delta_rot = Quat::from_scaled_axis(axis);
    orbital.spacecraft_attitude = (orbital.spacecraft_attitude * delta_rot).normalize();

    // HDC → PE
    let new_state = orbital.simulator.state().clone();
    let current_hv = orbital.encoder.encode(&new_state);
    let pe = if let Some(ref prev) = orbital.last_perception {
        let sim = current_hv.similarity(prev);
        (1.0 - sim.max(0.0)).min(1.0) as f32
    } else {
        0.0
    };
    orbital.last_prediction_error = pe;
    orbital.last_perception = Some(current_hv);
}

fn print_status(sim_time: Res<SimTime>, orbital: Res<OrbitalResources>) {
    let tick = (sim_time.elapsed * 2.0) as u64;
    let prev = ((sim_time.elapsed - 0.016) * 2.0) as u64;
    if tick == prev {
        return;
    }
    let st = orbital.simulator.state();
    let drift = (st
        .spacecraft_angular_velocity
        .iter()
        .map(|v| v * v)
        .sum::<f64>())
    .sqrt();
    println!(
        "[{:>6.2}s] ee=({:+.2},{:+.2},{:+.2}) comm={:.0}% sun={:.0}% drift={:.4}rad/s Φ={:.3} {:?} gain={:.2} eff={:.2} PE={:.3}",
        sim_time.elapsed,
        st.ee_position[0],
        st.ee_position[1],
        st.ee_position[2],
        st.comm_window * 100.0,
        st.solar_exposure * 100.0,
        drift,
        orbital.current_phi,
        orbital.current_safety,
        orbital.current_motor_gain,
        orbital.last_effort,
        orbital.last_prediction_error,
    );
}
