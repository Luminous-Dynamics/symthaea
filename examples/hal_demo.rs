// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HAL Integration Demo (v5)
//!
//! Demonstrates the full mock HAL pipeline with v5 features:
//! - Sensor recording/replay for offline debugging
//! - Health check before servo enable
//! - Telemetry export (JSON) after run
//! - Position readback verification
//! - Sensor degradation detection
//!
//! Run with: cargo run --example hal_demo --features hal

#[cfg(not(feature = "hal"))]
fn main() {
    eprintln!("This example requires the `hal` feature.");
    eprintln!("Run with: cargo run --example hal_demo --features hal");
}

#[cfg(feature = "hal")]
fn main() {
    use symthaea_hal::calibration::CalibrationProfile;
    use symthaea_hal::interlock::SafetyInterlock;
    use symthaea_hal::mock::{MockHalSensor, MockI2cBus};
    use symthaea_hal::recording::RecordingAdapter;
    use symthaea_hal::runtime::HalRuntime;
    use symthaea_hal::servo::ServoOutput;
    use symthaea_humanoid::types::HumanoidCommand;

    println!("=== symthaea-hal v5 integration demo ===\n");

    // 1. Create servo output with mock I2C buses
    let cal = CalibrationProfile::default_21();
    let bus0 = MockI2cBus::new();
    let bus1 = MockI2cBus::new();
    let mut servo = ServoOutput::new(bus0, bus1, cal);
    servo.init(50.0).expect("servo init failed");
    servo.enable();
    println!("[OK] Servo output initialized (2× PCA9685 mock, 50 Hz)");

    // 2. Safety interlock
    let interlock = SafetyInterlock::new();
    println!("[OK] Safety interlock created (watchdog=100ms, max_torque=0.9)");

    // 3. Build runtime with builder pattern
    let imu_readings: Vec<Vec<f32>> = (0..50)
        .map(|i| {
            let t = i as f32 * 0.02;
            vec![
                0.0,            // accel_x
                0.0,            // accel_y
                1.0,            // accel_z (gravity)
                t.sin() * 10.0, // gyro_x (simulated wobble)
                0.0,            // gyro_y
                0.0,            // gyro_z
            ]
        })
        .collect();

    // Wrap IMU in RecordingAdapter for offline debugging
    let raw_imu = MockHalSensor::new("mock_imu", imu_readings);
    let recording_imu = RecordingAdapter::new(Box::new(raw_imu));

    let mut runtime = HalRuntime::builder(servo, interlock)
        .with_tick_hz(50.0)
        .with_sensor(Box::new(recording_imu))
        .build();
    println!("[OK] Runtime built (50 Hz, 1 recorded sensor)\n");

    // 4. Health check before running
    let health = runtime.health();
    println!("--- Pre-run health check ---");
    println!("  sensors_ok:     {}", health.sensors_ok);
    println!("  servos_enabled: {}", health.servos_enabled);
    println!("  interlock_ok:   {}", health.interlock_ok);
    println!("  estop_active:   {}", health.estop_active);
    println!("  is_ready:       {}", health.is_ready());
    if !health.issues.is_empty() {
        for issue in &health.issues {
            println!("  issue: {}", issue);
        }
    }
    println!();

    // 5. Run 20 ticks with reactive controller
    println!("Running 20 ticks...\n");
    let mut tick = 0u32;

    let result = runtime.run(Some(20), |readings: &[Option<Vec<f32>>]| {
        tick += 1;
        let gyro_x = readings
            .first()
            .and_then(|r: &Option<Vec<f32>>| r.as_ref())
            .map(|v: &Vec<f32>| v.get(3).copied().unwrap_or(0.0))
            .unwrap_or(0.0);

        // Simple reactive controller: gyro_x → abdomen_y torque (joint 0)
        let mut cmd = HumanoidCommand::zero();
        cmd.torques[0] = (gyro_x * 0.05).clamp(-1.0, 1.0);

        if tick <= 5 || tick % 5 == 0 {
            println!(
                "  tick {:2}: gyro_x={:+6.2}°/s → torque[0]={:+.4}",
                tick, gyro_x, cmd.torques[0]
            );
        }
        cmd
    });

    match result {
        Ok(count) => {
            println!("\n[OK] Completed {} ticks", count);
            println!(
                "     Final pulse[0] = {} µs",
                runtime.servo().last_pulses()[0]
            );
        }
        Err(e) => {
            println!("\n[ERR] Runtime stopped: {}", e);
        }
    }

    // 6. Telemetry export
    let telemetry = runtime.telemetry();
    println!("\n--- Telemetry ---");
    println!("  tick_count:      {}", telemetry.tick_count);
    println!("  actual_hz:       {:.1}", telemetry.actual_hz);
    println!(
        "  latency p50/p95/p99: {:.1}/{:.1}/{:.1} µs",
        telemetry.p50_tick_us, telemetry.p95_tick_us, telemetry.p99_tick_us
    );
    println!("  deadline_misses: {}", telemetry.deadline_misses);
    println!("  jitter_us:       {:.1}", telemetry.jitter_us);

    // JSON export
    let json = telemetry.to_json_pretty().unwrap_or_default();
    println!("\n--- Telemetry JSON (first 200 chars) ---");
    println!("  {}...", &json[..json.len().min(200)]);

    // 7. Position readback
    println!("\n--- Position readback ---");
    match runtime.servo_mut().read_positions() {
        Ok(positions) => {
            let nonzero_count = positions.iter().filter(|&&p| p != 0).count();
            println!("  {} of 21 channels have non-zero position", nonzero_count);
            for (i, &p) in positions.iter().enumerate().take(5) {
                if p != 0 {
                    println!("    joint {:2}: OFF count = {}", i, p);
                }
            }
        }
        Err(e) => println!("  readback error: {}", e),
    }

    // 8. Verify commanded vs actual positions
    match runtime.servo_mut().verify_positions() {
        Ok(ref mismatches) => {
            if mismatches.is_empty() {
                println!("  All positions match commanded values");
            } else {
                println!("  {} mismatches detected:", mismatches.len());
                for &(joint, cmd, actual) in mismatches.iter().take(5) {
                    println!(
                        "    joint {}: commanded {} µs, actual OFF={}",
                        joint, cmd, actual
                    );
                }
            }
        }
        Err(e) => println!("  verify error: {}", e),
    }

    // 9. Sensor degradation check
    let degraded = runtime.degraded_sensors();
    if degraded.is_empty() {
        println!("\n[OK] No degraded sensors");
    } else {
        println!("\n[WARN] {} degraded sensor(s):", degraded.len());
        for (kind, name, count) in &degraded {
            println!("  {} '{}': {} consecutive Nones", kind, name, count);
        }
    }

    println!("\n=== demo complete ===");
}
