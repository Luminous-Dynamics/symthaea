// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Main plugin: wires resources + ordered per-frame systems.

use bevy::prelude::*;
use symthaea_vehicle::simulator::VehiclePhysicsSimulator;

use crate::camera;
use crate::consciousness_bridge;
use crate::controller::gain_scale;
use crate::hud;
use crate::resources::*;
use crate::visualization;

pub struct VehicleDemoPlugin;

impl Plugin for VehicleDemoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(VehicleResources::new())
            .insert_resource(WaypointPath::default())
            .insert_resource(Ice::default())
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
                    step_vehicle,
                    update_waypoint,
                    visualization::update_vehicle_visual,
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

/// Core per-frame loop mirroring flight-demo.
fn step_vehicle(
    time: Res<Time>,
    waypoints: Res<WaypointPath>,
    ice: Res<Ice>,
    mut vehicle: ResMut<VehicleResources>,
) {
    let dt = time.delta_secs_f64();
    if dt <= 0.0 || dt > 0.1 {
        return;
    }

    vehicle.controller.target = waypoints.current();

    // 1. Sample ice field at current position
    let state = vehicle.simulator.state().clone();
    let pos = [state.position_x, state.position_y];
    let (friction, gust, intensity) = ice.field.sample(pos);
    vehicle.current_friction = friction;
    vehicle.current_gust = gust;
    vehicle.current_ice_intensity = intensity;
    vehicle.simulator.set_friction_scale(friction);
    vehicle.simulator.apply_external_force(gust);

    // 2. Build Stanley command toward current waypoint
    let cmd = vehicle.controller.compute(&state);

    // 3. Consciousness tick
    let speed_norm = (state.speed / 15.0).clamp(0.0, 1.0);
    let slip_norm =
        ((state.tire_slip_front.abs() + state.tire_slip_rear.abs()) / 0.52).clamp(0.0, 1.0);
    let last_pe = vehicle.last_prediction_error;
    let danger = (intensity + slip_norm * 0.5).min(1.0);
    let (phi, safety, gain) = consciousness_bridge::consciousness_tick(
        &mut vehicle.robot_agent,
        last_pe,
        danger,
        speed_norm,
        slip_norm,
    );
    vehicle.current_phi = phi;
    vehicle.current_safety = safety;
    vehicle.current_motor_gain = gain;

    // 4. Scale by motor gain (brake is pass-through so safety isn't compromised)
    let scaled = gain_scale(cmd, gain);
    vehicle.last_steering = scaled.steering;
    vehicle.last_throttle = scaled.throttle;
    vehicle.last_brake = scaled.brake;

    // 5. Step physics
    vehicle.simulator.step(&scaled, dt);

    // 6. HDC encode → cosine dissimilarity PE
    let new_state = vehicle.simulator.state().clone();
    let current_hv = vehicle.encoder.encode(&new_state);
    let pe = if let Some(ref prev) = vehicle.last_perception {
        let sim = current_hv.similarity(prev);
        (1.0 - sim.max(0.0)).min(1.0) as f32
    } else {
        0.0
    };
    vehicle.last_prediction_error = pe;
    vehicle.last_perception = Some(current_hv);
}

fn update_waypoint(vehicle: Res<VehicleResources>, mut waypoints: ResMut<WaypointPath>) {
    let st = vehicle.simulator.state();
    waypoints.advance_if_reached([st.position_x, st.position_y], 3.0);
}

fn print_status(
    sim_time: Res<SimTime>,
    vehicle: Res<VehicleResources>,
    waypoints: Res<WaypointPath>,
) {
    let tick = (sim_time.elapsed * 2.0) as u64;
    let prev = ((sim_time.elapsed - 0.016) * 2.0) as u64;
    if tick == prev {
        return;
    }
    let st = vehicle.simulator.state();
    println!(
        "[{:>6.2}s] v={:>5.2}m/s μ={:.2} slip=({:+.2},{:+.2}) Φ={:.3} {:?} gain={:.2} PE={:.3} wp={}/{} lap={}",
        sim_time.elapsed,
        st.speed,
        vehicle.current_friction,
        st.tire_slip_front,
        st.tire_slip_rear,
        vehicle.current_phi,
        vehicle.current_safety,
        vehicle.current_motor_gain,
        vehicle.last_prediction_error,
        waypoints.current_index + 1,
        waypoints.waypoints.len(),
        waypoints.laps_completed,
    );
}
