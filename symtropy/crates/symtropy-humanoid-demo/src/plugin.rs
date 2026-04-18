// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Main plugin.

use bevy::prelude::*;
use symthaea_humanoid::simulator::HumanoidPhysicsSimulator;

use crate::camera;
use crate::consciousness_bridge;
use crate::controller::gain_scale;
use crate::hud;
use crate::resources::*;
use crate::visualization;

pub struct HumanoidDemoPlugin;

impl Plugin for HumanoidDemoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(HumanoidResources::new())
            .insert_resource(Push::default())
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
                    step_humanoid,
                    visualization::update_humanoid_visual,
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

fn step_humanoid(
    time: Res<Time>,
    sim_time: Res<SimTime>,
    push: Res<Push>,
    mut h: ResMut<HumanoidResources>,
) {
    let dt = time.delta_secs_f64();
    if dt <= 0.0 || dt > 0.1 {
        return;
    }

    // 1. Apply push disturbance to the torso
    let (force, intensity) = push.source.sample(sim_time.elapsed);
    h.last_push_force = force;
    h.last_push_intensity = intensity;
    h.simulator.apply_external_force(force);

    // 2. Build balance command
    let state = h.simulator.state().clone();
    let cmd = h.controller.compute(&state);

    // 3. Consciousness tick
    let upright = state.uprightness().clamp(0.0, 1.0);
    let push_norm = intensity;
    let last_pe = h.last_prediction_error;
    // Danger rises as uprightness falls or push grows. Uprightness=1
    // means upright; 0 means horizontal; (1 - upright) is the fall risk.
    let danger = ((1.0 - upright) * 0.65 + push_norm * 0.35).min(1.0);
    let (phi, safety, gain) = consciousness_bridge::consciousness_tick(
        &mut h.robot_agent,
        last_pe,
        danger,
        upright,
        push_norm,
    );
    h.current_phi = phi;
    h.current_safety = safety;
    h.current_motor_gain = gain;

    // 4. Scale + step
    let scaled = gain_scale(cmd, gain);
    h.last_effort = scaled.control_effort();
    h.simulator.step(&scaled, dt);

    // 5. HDC → PE
    let new_state = h.simulator.state().clone();
    let current_hv = h.encoder.encode(&new_state);
    let pe = if let Some(ref prev) = h.last_perception {
        let sim = current_hv.similarity(prev);
        (1.0 - sim.max(0.0)).min(1.0) as f32
    } else {
        0.0
    };
    h.last_prediction_error = pe;
    h.last_perception = Some(current_hv);
}

fn print_status(sim_time: Res<SimTime>, h: Res<HumanoidResources>) {
    let tick = (sim_time.elapsed * 2.0) as u64;
    let prev = ((sim_time.elapsed - 0.016) * 2.0) as u64;
    if tick == prev {
        return;
    }
    let st = h.simulator.state();
    let push_mag = (h.last_push_force[0].powi(2) + h.last_push_force[1].powi(2)).sqrt();
    println!(
        "[{:>6.2}s] upright={:.3} root_z={:.2}m head_z={:.2}m Φ={:.3} {:?} gain={:.2} effort={:.2} PE={:.3} push={:.0}N",
        sim_time.elapsed,
        st.uprightness(),
        st.root_height,
        st.head_height,
        h.current_phi,
        h.current_safety,
        h.current_motor_gain,
        h.last_effort,
        h.last_prediction_error,
        push_mag,
    );
}
