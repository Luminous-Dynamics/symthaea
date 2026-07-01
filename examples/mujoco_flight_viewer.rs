// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Interactive 3D viewer for MuJoCo Crazyflie 2 flight simulation.
//!
//! Launches a passive viewer window showing the quadrotor in real-time
//! while running the HDC-LTC-FEP control loop.
//!
//! Run: `cargo run --example mujoco_flight_viewer --features multirotor-mujoco --release`

use symthaea::multirotor::{
    controller::FlightController,
    encoder::QuadrotorHdcEncoder,
    fep_agent::{ActiveInferenceFlightAgent, FlightFepConfig},
    mujoco_sim::MuJoCoSimulator,
    types::*,
    viewer::FlightViewer,
};
use symthaea_core::genesis::GenesisSeed;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     MuJoCo Flight Viewer — Crazyflie 2 HDC-LTC-FEP           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    let config = FlightConfig::default();
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);

    let mut sim = MuJoCoSimulator::from_primitive();
    sim.reset(0.1);

    let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
    let mut controller = FlightController::new(&genesis, &config);
    let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());

    let setpoint = FlightSetpoint::hover();
    let dt = config.motor_dt();
    let cognitive_interval = config.cognitive_interval();
    let pd_gains = PdGains::default();

    let mut viewer = FlightViewer::new(&sim, "Symthaea Flight — HDC-LTC-FEP Quadrotor");
    let mut fep_result = fep_agent.initial_step(sim.state(), &setpoint);

    println!("Viewer launched. Close window to exit.");
    println!("Hover setpoint: {:?}", setpoint.position);
    println!();

    let mut step = 0;
    while viewer.is_running() {
        let state = sim.state().clone();
        let sensor_hv = encoder.encode(&state);
        let mut command = controller.forward(&sensor_hv, dt as f32);

        if let Some(noise) = &fep_result.exploration_noise {
            command = command.with_noise(noise);
        }

        sim.step(&command, dt);

        // Training
        if step % config.train_every == 0 {
            let target = pd_baseline(&state, &setpoint, &pd_gains);
            let lr = config.learning_rate * fep_result.learning_rate_factor;
            controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));
        }

        // Cognitive tick
        if step % cognitive_interval == 0 {
            fep_result = fep_agent.step(sim.state(), &setpoint);
            if (fep_result.tau_factor - 1.0).abs() > 0.01 {
                controller.modulate_tau(fep_result.tau_factor);
            }
        }

        // Render every 10 steps (~50Hz visual update)
        if step % 10 == 0 {
            viewer.render(&mut sim);
        }

        // Print status periodically
        if step % 500 == 0 {
            let err = setpoint.position_error_magnitude(&state);
            println!(
                "Step {:5} | Alt: {:.3}m | Error: {:.4}m | FE: {:.3} | τ: {:.2}",
                step,
                state.altitude(),
                err,
                fep_result.free_energy,
                fep_result.tau_factor
            );
        }

        step += 1;
        std::thread::sleep(std::time::Duration::from_secs_f64(dt));
    }

    println!();
    println!("Viewer closed after {} frames.", viewer.frame_count());
}
