// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Main plugin: resources + ordered per-frame systems.

use bevy::prelude::*;
use symthaea_helicopter::simulator::HelicopterPhysicsSimulator;
use symtropy_consciousness_physics::safety::sprint_floor_gain;

use crate::camera;
use crate::consciousness_bridge;
use crate::controller::gain_scale;
use crate::hud;
use crate::resources::*;
use crate::visualization;

/// Per-platform Φ-gate thresholds for the SAR-helicopter demo, using
/// the empirically-validated `sprint_floor_gain` primitive from the
/// Φ-gated-safety paper (commits `38dc8b1fd9..317baad595`, promoted
/// to library at `52e3fb710f`). Mirrors flight-demo (`8d61e348d9`),
/// vehicle-demo (`c2f2fb46c8`), and AUV-demo.
///
/// **Starting values** inherited from the manipulator study's measured
/// Φ band [0.099, 0.145]. Helicopter observation vector (altitude /
/// wind-intensity / attitude) differs from the manipulator's (danger /
/// PE / effort / stiffness), so the empirical band may drift —
/// recalibrate with a `MANIP_BENCH_PHI_TRACE`-style capture under a
/// representative Dryden gust schedule.
// Recalibrated 2026-04-19 from 0.135 → 0.125 after commit `996750d12b`
// (FEP wiring into ConsciousnessInputs) shifted the empirical Φ band
// from [0.099, 0.145] to [0.088, 0.133]. Same relative band position.
const SPRINT_THRESHOLD: f64 = 0.125;
const FLOOR_GAIN: f64 = 0.3;

pub struct HelicopterDemoPlugin;

impl Plugin for HelicopterDemoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(HelicopterResources::new())
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
                    step_helicopter,
                    visualization::update_helicopter_visual,
                    visualization::update_rotor_spin,
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

/// Core per-frame loop: wind → controller → consciousness → gain → physics → PE.
fn step_helicopter(time: Res<Time>, mut heli: ResMut<HelicopterResources>) {
    let dt = time.delta_secs_f64();
    if dt <= 0.0 || dt > 0.1 {
        return;
    }

    // 1. Sample the Dryden wind model (drag area ≈ 2 m² for R44-class fuselage)
    let state = heli.simulator.state().clone();
    let airspeed = state.linear_velocity;
    let wind_force = heli.wind.step(dt, airspeed, 2.0);
    heli.last_wind_force = wind_force.force;
    heli.last_wind_speed = wind_force.wind_speed();
    heli.simulator.apply_external_force(wind_force.force);

    // 2. Build station-hold command
    let cmd = heli.controller.compute(&state);

    // 3. Consciousness tick
    let altitude_norm = (state.position[2] / 40.0).clamp(0.0, 1.0);
    // Normalize wind force magnitude onto [0,1] assuming 300 N is "strong"
    let wind_norm = (wind_force.force_magnitude() / 300.0).clamp(0.0, 1.0);
    let last_pe = heli.last_prediction_error;
    // Off-station distance also contributes to danger
    let off_station =
        ((state.position[0].powi(2) + state.position[1].powi(2)).sqrt() / 15.0).clamp(0.0, 1.0);
    let danger = (wind_norm + off_station * 0.5).min(1.0);
    let (phi, safety, _default_gain) = consciousness_bridge::consciousness_tick(
        &mut heli.robot_agent,
        last_pe,
        danger,
        altitude_norm,
        wind_norm,
    );
    // Use the empirically-validated SprintFloor mapping instead of the
    // default `SafetyTier::motor_gain()` — hardcoded 0.6/0.3/0.1
    // thresholds don't match this platform's empirical Φ band.
    let gain = sprint_floor_gain(phi, SPRINT_THRESHOLD, FLOOR_GAIN);
    heli.current_phi = phi;
    heli.current_safety = safety;
    heli.current_motor_gain = gain;

    // 4. Scale by motor gain
    let scaled = gain_scale(cmd, gain);
    heli.last_collective = scaled.collective;

    // 5. Step physics
    heli.simulator.step(&scaled, dt);

    // 6. HDC encode → cosine dissimilarity PE
    let new_state = heli.simulator.state().clone();
    // Spin main rotor visually using the rotor RPM from the simulator
    let rpm = new_state.main_rotor_rpm;
    let spin_rate_rad_per_s = rpm * std::f64::consts::TAU / 60.0;
    heli.last_rotor_spin_angle = (heli.last_rotor_spin_angle + (spin_rate_rad_per_s * dt) as f32)
        .rem_euclid(std::f32::consts::TAU);

    let current_hv = heli.encoder.encode(&new_state);
    let pe = if let Some(ref prev) = heli.last_perception {
        let sim = current_hv.similarity(prev);
        (1.0 - sim.max(0.0)).min(1.0) as f32
    } else {
        0.0
    };
    heli.last_prediction_error = pe;
    heli.last_perception = Some(current_hv);
}

fn print_status(sim_time: Res<SimTime>, heli: Res<HelicopterResources>) {
    let tick = (sim_time.elapsed * 2.0) as u64;
    let prev = ((sim_time.elapsed - 0.016) * 2.0) as u64;
    if tick == prev {
        return;
    }
    let st = heli.simulator.state();
    let off = (st.position[0].powi(2) + st.position[1].powi(2)).sqrt();
    let wind_mag = (heli.last_wind_force[0].powi(2)
        + heli.last_wind_force[1].powi(2)
        + heli.last_wind_force[2].powi(2))
    .sqrt();
    println!(
        "[{:>6.2}s] alt={:>5.2}m off={:>5.2}m rpm={:>5.0} coll={:.2} Φ={:.3} {:?} gain={:.2} PE={:.3} wind={:.0}N ({:.1}m/s)",
        sim_time.elapsed,
        st.position[2],
        off,
        st.main_rotor_rpm,
        heli.last_collective,
        heli.current_phi,
        heli.current_safety,
        heli.current_motor_gain,
        heli.last_prediction_error,
        wind_mag,
        heli.last_wind_speed,
    );
}
