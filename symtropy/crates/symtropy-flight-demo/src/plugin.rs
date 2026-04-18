// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Main plugin: wires resources + ordered per-frame systems.

use bevy::prelude::*;
use symthaea_flight::simulator::PhysicsSimulator;
use symtropy_consciousness_physics::safety::sprint_floor_gain;

use crate::camera;
use crate::consciousness_bridge;
use crate::controller::gain_scale;
use crate::hud;
use crate::resources::*;
use crate::visualization;

/// Per-platform Φ-gate thresholds for the quadrotor demo. These use the
/// empirically-validated `sprint_floor_gain` shape
/// (`symtropy-consciousness-physics::safety`) whose 2-part sufficiency
/// was proven in the manipulator Monte Carlo study — see commits
/// `38dc8b1fd9..317baad595` and the promoting primitive at `52e3fb710f`.
///
/// **Starting values**, inherited from the manipulator's measured Φ band
/// [0.099, 0.145]. The `RoboticAgent::tick` pipeline and underlying
/// `MasterConsciousnessEquation` are the same across platforms, so Φ
/// distributions should be similar in shape — but the flight-demo's
/// observation vector (altitude / attitude) differs from the
/// manipulator's (PE / danger / control-effort / stiffness), so the
/// empirical band may drift.
///
/// To recalibrate: add a trace-capture block to `step_quadrotor`
/// mirroring `manipulator_benchmark`'s `MANIP_BENCH_PHI_TRACE=1`,
/// record 40 s of Φ samples, then set `SPRINT_THRESHOLD` to a value near
/// the 95th percentile of the observed band and `FLOOR_GAIN` above
/// whatever thrust level keeps altitude held in a light wind.
// Recalibrated 2026-04-19 from 0.135 → 0.125 after commit `996750d12b`
// (FEP wiring into ConsciousnessInputs) shifted the empirical Φ band
// from [0.099, 0.145] to [0.088, 0.133]. The new value sits at the
// same relative position within the new band as 0.135 did in the old
// one (~78 % up the range). See `phi_trace_multi/` CSVs for data.
const SPRINT_THRESHOLD: f64 = 0.125;
const FLOOR_GAIN: f64 = 0.3;

pub struct FlightDemoPlugin;

impl Plugin for FlightDemoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(QuadrotorState::new())
            .insert_resource(WaypointPath::default())
            .insert_resource(Wind::default())
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
                    step_quadrotor,
                    update_waypoint,
                    visualization::update_quadrotor_visual,
                    visualization::update_waypoint_visual,
                    hud::update_hud,
                    print_status,
                )
                    .chain(),
            );
    }
}

/// Advance simulation time.
fn update_sim_time(time: Res<Time>, mut sim_time: ResMut<SimTime>) {
    sim_time.elapsed += time.delta_secs_f64();
}

/// Core per-frame loop: wind → controller → consciousness → gain-scale → physics → PE.
///
/// Flow mirrors the manipulator demo's `step_phi_arm`:
/// 1. Compute disturbance (wind gust)
/// 2. Build PID command for the current waypoint
/// 3. Run consciousness tick (PE + danger → motor gain)
/// 4. Scale command by gain (interpolates thrust between hover and commanded)
/// 5. Step physics with scaled command + external wind force
/// 6. Encode new state as HDC, derive PE from cosine dissimilarity vs previous
fn step_quadrotor(
    time: Res<Time>,
    sim_time: Res<SimTime>,
    waypoints: Res<WaypointPath>,
    wind: Res<Wind>,
    mut quad: ResMut<QuadrotorState>,
) {
    let dt = time.delta_secs_f64();
    if dt <= 0.0 || dt > 0.1 {
        return;
    }

    // Set current target
    quad.controller.target = waypoints.current();

    // 1. Wind gust → external force + danger intensity
    let (force, gust_intensity) = wind.source.sample(sim_time.elapsed);
    quad.last_gust_force = force;
    quad.last_gust_intensity = gust_intensity;
    quad.simulator.apply_external_force(force);

    // 2. PID command toward current waypoint (Crazyflie mass = 0.027 kg)
    let state = quad.simulator.state().clone();
    let cmd = quad.controller.compute(&state, 0.027);

    // 3. Consciousness tick — PE from last cycle + danger from gust intensity
    let altitude_norm = (state.position[2] / 3.0).clamp(0.0, 1.0);
    let (roll, pitch, _yaw) = state.euler_angles();
    let attitude_norm = ((roll.abs() + pitch.abs()) / std::f64::consts::PI).clamp(0.0, 1.0);
    let last_pe = quad.last_prediction_error;
    let danger = (gust_intensity + attitude_norm * 0.5).min(1.0);
    let (phi, safety, _default_gain) = consciousness_bridge::consciousness_tick(
        &mut quad.robot_agent,
        last_pe,
        danger,
        altitude_norm,
        attitude_norm,
    );
    // Use the empirically-validated SprintFloor mapping instead of the
    // default `SafetyTier::motor_gain()` — whose hardcoded 0.6/0.3/0.1
    // thresholds are known to pin Φ at a single tier under this
    // platform's Φ distribution (see module-level doc comment).
    let gain = sprint_floor_gain(phi, SPRINT_THRESHOLD, FLOOR_GAIN);
    quad.current_phi = phi;
    quad.current_safety = safety;
    quad.current_motor_gain = gain;

    // 4. Scale command by motor gain
    let scaled = gain_scale(cmd, gain);
    quad.last_thrust = scaled.thrust;

    // 5. Step physics
    quad.simulator.step(&scaled, dt);

    // 6. Encode new state → HDC, compute PE via cosine dissimilarity
    let new_state = quad.simulator.state().clone();
    let current_hv = quad.encoder.encode(&new_state);
    let pe = if let Some(ref prev) = quad.last_perception {
        let sim = current_hv.similarity(prev);
        (1.0 - sim.max(0.0)).min(1.0) as f32
    } else {
        0.0
    };
    quad.last_prediction_error = pe;
    quad.last_perception = Some(current_hv);
}

/// Advance the waypoint index when the quadrotor is within 0.35 m of the target.
fn update_waypoint(quad: Res<QuadrotorState>, mut waypoints: ResMut<WaypointPath>) {
    let pos = quad.simulator.state().position;
    waypoints.advance_if_reached(pos, 0.35);
}

/// Print a compact status line every ~0.5 s of sim time.
fn print_status(sim_time: Res<SimTime>, quad: Res<QuadrotorState>, waypoints: Res<WaypointPath>) {
    let tick = (sim_time.elapsed * 2.0) as u64;
    let prev = ((sim_time.elapsed - 0.016) * 2.0) as u64;
    if tick == prev {
        return;
    }
    let st = quad.simulator.state();
    let (roll, pitch, _) = st.euler_angles();
    println!(
        "[{:>6.2}s] alt={:>5.2}m rp=({:>+.2},{:>+.2}) Φ={:.3} {:?} gain={:.2} PE={:.3} gust={:.2}N wp={}/{} cyc={}",
        sim_time.elapsed,
        st.position[2],
        roll,
        pitch,
        quad.current_phi,
        quad.current_safety,
        quad.current_motor_gain,
        quad.last_prediction_error,
        (quad.last_gust_force[0].powi(2) + quad.last_gust_force[1].powi(2)).sqrt(),
        waypoints.current_index + 1,
        waypoints.waypoints.len(),
        waypoints.cycles_completed,
    );
}
