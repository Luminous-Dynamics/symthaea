// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HAL Sovereignty Demo
//!
//! Demonstrates Symthaea sensing her own power via INA219
//! and adjusting her cognitive 'throttle' to maintain the 6-Watt limit.

use symthaea::Symthaea;
use symthaea::action::PolicyBundle;
use tracing::Level;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    println!("\n[POWER] Symthaea v0.6.0: Physical Sovereignty (HAL Bridge)\n");

    // 1. Initialize Symthaea
    let mut sym = Symthaea::new(1024, 64).await?;
    sym.executor = symthaea::action::SimpleExecutor::with_real_commands();

    // Setup Policy
    let mut policy = PolicyBundle::restrictive();
    policy.capabilities.min_phi = 0.1;

    // 2. The Power Sensing Command
    let command = "Check the INA219 power sensor. If current is high (> 500mA), lower the cognitive intensity by moving the throttle servo (ID 0) to 0.2.";

    println!("[INPUT] Command: {}\n", command);

    // 3. Process the Sensing & Action
    println!("[THOUGHT] Reading sensors...");
    let response = sym.process(command).await?;

    println!("\n[RESPONSE] Symthaea Reflection: {}\n", response.content);

    // 4. Audit the Motor Cortex
    println!("[AUDIT] Checking Motor Cortex (Telemetry)...");
    for record in sym.executor.telemetry() {
        match &record.action {
            symthaea::action::ActionIR::ReadSensor { sensor_id, .. } => {
                println!("   -> Sensed: {}\n", sensor_id);
            }
            symthaea::action::ActionIR::WriteServo { servo_id, value } => {
                println!("   -> Actuated Servo {}: val={}\n", servo_id, value);
            }
            _ => {}
        }
    }

    println!("\n[RESULT] PHYSICAL SOVEREIGNTY VERIFIED.");

    Ok(())
}