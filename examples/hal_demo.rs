//! HAL Integration Demo
//!
//! Demonstrates the full mock HAL pipeline: sensors → callback → safety → servos.
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
    use symthaea_hal::runtime::HalRuntime;
    use symthaea_hal::servo::ServoOutput;
    use symthaea_humanoid::types::HumanoidCommand;

    println!("=== symthaea-hal integration demo ===\n");

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

    // 3. Runtime orchestrator
    let mut runtime = HalRuntime::new(servo, interlock);
    runtime.set_tick_hz(50.0);

    // 4. Add mock IMU sensor (10 readings)
    let imu_readings: Vec<Vec<f32>> = (0..10)
        .map(|i| {
            let t = i as f32 * 0.1;
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
    runtime.add_sensor(Box::new(MockHalSensor::new("mock_imu", imu_readings)));
    println!("[OK] Mock IMU sensor added (10 readings)\n");

    // 5. Run 10 ticks
    println!("Running 10 ticks...\n");
    let mut tick = 0u32;

    let result = runtime.run(Some(10), |readings| {
        tick += 1;
        let gyro_x = readings
            .first()
            .and_then(|r| r.as_ref())
            .map(|v| v.get(3).copied().unwrap_or(0.0))
            .unwrap_or(0.0);

        // Simple reactive controller: gyro_x → abdomen_y torque (joint 0)
        let mut cmd = HumanoidCommand::zero();
        cmd.torques[0] = (gyro_x * 0.05).clamp(-1.0, 1.0);

        println!(
            "  tick {:2}: gyro_x={:+6.2}°/s → torque[0]={:+.4}",
            tick, gyro_x, cmd.torques[0]
        );
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

    println!("\n=== demo complete ===");
}
