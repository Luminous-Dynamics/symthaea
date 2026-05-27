// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::thread;
use std::time::{Duration, Instant};

use symthaea_humanoid::simulator::{HumanoidPhysicsSimulator, MuJoCoHumanoidSimulator};
use symthaea_humanoid::types::HumanoidCommand;
use symthaea_humanoid::types::HumanoidConfig;

use mujoco_rs::viewer::MjViewer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 INITIALIZING HIGH-PERFORMANCE NATIVE SIMULATION ENGINE...");

    let humanoid_config = HumanoidConfig::default();
    let physics_dt = humanoid_config.physics_dt();
    let target_refresh_dt = 0.025f64; // 40Hz UI step (25ms)
    let physics_substeps = (target_refresh_dt / physics_dt).round().max(1.0) as usize;

    // Load updated model asset configuration matrix
    println!("🚀 Loading high-fidelity humanoid asset tree...");
    let mut physics = MuJoCoHumanoidSimulator::from_bundled_asset()
        .map_err(|e| format!("❌ Failed loading bundled MJCF schema: {:?}", e))?;

    let total_actuators = physics.data_mut().ctrl_mut().len();

    // Capture initial reference positions to prime the posture hold
    let mut stance_targets = vec![0.0f32; total_actuators];
    for i in 0..total_actuators {
        if i + 7 < physics.data_mut().qpos().len() {
            stance_targets[i] = physics.data_mut().qpos()[i + 7] as f32;
        }
    }

    // Launch modern passive graphics rendering context
    println!("🖥️ Launching non-blocking viewer graphics context...");
    let mut window = MjViewer::launch_passive(physics.model_arc().clone(), 0)
        .map_err(|e| format!("❌ Viewport failure: {:?}", e))?;

    let start_time = Instant::now();
    let mut last_time = Instant::now();
    let mut frame_idx: u64 = 0;

    println!("🏁 RUNNING REAL-TIME HIGH-SPEED SIMULATION LOOP... Close window to exit.");

    while window.running() {
        let elapsed = start_time.elapsed().as_secs_f64();
        last_time = Instant::now();

        // MOTOR SUBSTEPS: Execute catch-up updates at 500Hz frequency
        for _ in 0..physics_substeps {
            let mut command = HumanoidCommand::zero_for(total_actuators);

            // Read direct mechanical references safely from the simulator handle
            let qpos = physics.data_mut().qpos();
            let qvel = physics.data_mut().qvel();

            // Software PD Posture Controller
            let kp = 140.0f32;
            let kd = 8.0f32;

            for i in 0..total_actuators {
                let current_angle = qpos[i + 7] as f32;
                let current_velocity = qvel[i + 6] as f32;
                let target_angle = stance_targets[i];

                // Synchronize raw torque inputs to maintain an upright posture
                command.torques[i] = kp * (target_angle - current_angle) - kd * current_velocity;
            }

            physics.step(&command, physics_dt);
        }

        // HIGH PERFORMANCE REPLACEMENT: Non-blocking split interface flush
        window.sync_data(physics.data_mut());
        window.render();

        if frame_idx % 40 == 0 {
            let current_state = physics.state();
            println!(
                "🏁 [Frame {:04}] Real Time: {:.2}s | Sim Clock: {:.2}s | Frame Pipeline: Optimized",
                frame_idx, elapsed, current_state.timestamp
            );
        }

        frame_idx += 1;
        thread::sleep(Duration::from_millis(25));
    }

    println!("✅ Runtime session closed cleanly.");
    Ok(())
}
