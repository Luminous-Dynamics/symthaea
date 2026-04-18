// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Main plugin.

use bevy::prelude::*;
use symthaea_exoskeleton::simulator::ExoskeletonPhysicsSimulator;
use symthaea_exoskeleton::types::AssistanceMode;

use crate::camera;
use crate::consciousness_bridge;
use crate::hud;
use crate::resources::*;
use crate::visualization;

pub struct ExoskeletonDemoPlugin;

impl Plugin for ExoskeletonDemoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ExoskeletonResources::new())
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
                    step_exoskeleton,
                    visualization::update_leg_visual,
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

/// Core per-frame loop.
///
/// Flow:
/// 1. Measure balance proxy (CoP magnitude) + battery — observation inputs
/// 2. Consciousness tick → Φ → AssistanceMode (platform-provided API)
/// 3. Controller computes PD torques toward target pose, scaled by mode factors
/// 4. Step physics (simulator adds human CPG torques internally)
/// 5. HDC encode new state → cosine-dissim PE for next tick
fn step_exoskeleton(time: Res<Time>, mut exo: ResMut<ExoskeletonResources>) {
    let dt = time.delta_secs_f64();
    if dt <= 0.0 || dt > 0.1 {
        return;
    }

    let state = exo.simulator.state().clone();

    // Observation inputs
    let cop_mag =
        (state.center_of_pressure[0].powi(2) + state.center_of_pressure[1].powi(2)).sqrt();
    // A walking human's CoP sweeps ~10 cm; >15 cm is unstable
    let balance_norm = (cop_mag / 0.15).clamp(0.0, 1.0);
    let battery_norm = state.battery_soc.clamp(0.0, 1.0);
    let last_pe = exo.last_prediction_error;
    // Danger rises when CoP drifts or battery drops — both argue for less
    // aggressive exo authority.
    let danger = (balance_norm * 0.7 + (1.0 - battery_norm) * 0.3).min(1.0);

    let (phi, safety, _gain) = consciousness_bridge::consciousness_tick(
        &mut exo.robot_agent,
        last_pe,
        danger,
        balance_norm,
        battery_norm,
    );
    exo.current_phi = phi;
    exo.current_safety = safety;

    // Platform-provided: map Φ → assistance mode → factors
    let mode = AssistanceMode::from_phi(phi);
    exo.current_mode = mode;

    // Controller computes assistive torques scaled by mode factors
    let cmd = exo.controller.compute(&state, mode);
    exo.last_exo_effort = cmd.control_effort();

    // Step physics (simulator adds human CPG torques internally)
    exo.simulator.step(&cmd, dt);

    // HDC → PE
    let new_state = exo.simulator.state().clone();
    let current_hv = exo.encoder.encode(&new_state);
    let pe = if let Some(ref prev) = exo.last_perception {
        let sim = current_hv.similarity(prev);
        (1.0 - sim.max(0.0)).min(1.0) as f32
    } else {
        0.0
    };
    exo.last_prediction_error = pe;
    exo.last_perception = Some(current_hv);
}

fn print_status(sim_time: Res<SimTime>, exo: Res<ExoskeletonResources>) {
    let tick = (sim_time.elapsed * 2.0) as u64;
    let prev = ((sim_time.elapsed - 0.016) * 2.0) as u64;
    if tick == prev {
        return;
    }
    let st = exo.simulator.state();
    println!(
        "[{:>6.2}s] Φ={:.3} mode={:?} {:?} effort={:.2} PE={:.3} CoP=({:+.02},{:+.02})m SoC={:.0}% hum_rms={:.1}",
        sim_time.elapsed,
        exo.current_phi,
        exo.current_mode,
        exo.current_safety,
        exo.last_exo_effort,
        exo.last_prediction_error,
        st.center_of_pressure[0],
        st.center_of_pressure[1],
        st.battery_soc * 100.0,
        rms(&st.human_torques),
    );
}

fn rms(v: &[f64]) -> f64 {
    let n = v.len().max(1) as f64;
    (v.iter().map(|x| x * x).sum::<f64>() / n).sqrt()
}
