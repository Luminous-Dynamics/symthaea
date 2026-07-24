// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HAL servo calibration CLI.
//!
//! Sweeps each joint to find endpoints and saves a JSON calibration profile.
//! Supports both mock (dry-run) and real hardware modes.
//!
//! ```text
//! hal-calibrate [--dry-run] [--bus /dev/i2c-1] [--output calibration.json] [--joint N]
//! hal-calibrate --test --joint 3 [--step 50] [--delay-ms 200]
//! ```
//!
//! Build with: `cargo build -p symthaea-hal --features calibrate`

use clap::Parser;
use embedded_hal::i2c::I2c;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use symthaea_hal::calibration::CalibrationProfile;
use symthaea_hal::mock::{MockHalSensor, MockI2cBus};
use symthaea_hal::pca9685::Pca9685;
use symthaea_hal::sensor::HalSensorAdapter;
use symthaea_hal::{HalError, HalResult};
use symthaea_humanoid::types::NUM_ACTUATORS;

/// Servo calibration tool for symthaea-hal.
#[derive(Parser)]
#[command(name = "hal-calibrate", version, about = "Calibrate servo endpoints")]
struct Cli {
    /// Use mock I2C bus (no hardware required).
    #[arg(long)]
    dry_run: bool,

    /// Explicitly acknowledge that hardware outputs may move. Required outside dry-run.
    #[arg(long)]
    arm: bool,

    /// I2C bus device path (ignored in --dry-run mode).
    #[arg(long, default_value = "/dev/i2c-1")]
    bus: String,

    /// Output file for the calibration profile (JSON).
    #[arg(long, short, default_value = "calibration.json")]
    output: PathBuf,

    /// Calibrate only a specific joint (0-indexed). Omit to calibrate all.
    #[arg(long)]
    joint: Option<usize>,

    /// PWM frequency in Hz.
    #[arg(long, default_value = "50.0")]
    frequency: f64,

    /// Test sweep mode: sweep a joint from min to max in steps.
    #[arg(long)]
    test: bool,

    /// Step size in µs for test sweep (default 50).
    #[arg(long, default_value = "50")]
    step: u16,

    /// Delay between steps in milliseconds (default 200).
    #[arg(long, default_value = "200")]
    delay_ms: u64,

    /// Enable INA219 current monitoring during test sweep.
    #[arg(long)]
    monitor: bool,

    /// Current threshold in amps for stall detection (default 1.5).
    #[arg(long, default_value = "1.5")]
    stall_threshold: f32,

    /// INA219 shunt resistance in ohms (default 0.1).
    #[arg(long, default_value = "0.1")]
    shunt_resistance: f32,
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(&cli) {
        eprintln!("Calibration failed: {e}");
        std::process::exit(1);
    }
}

fn run(cli: &Cli) -> HalResult<()> {
    validate_cli(cli)?;

    println!("=== symthaea-hal calibration ===");
    println!(
        "Mode: {}",
        if cli.dry_run {
            "dry-run (mock)"
        } else {
            "hardware"
        }
    );
    println!("Output: {}", cli.output.display());

    let mut profile = load_or_create_profile(&cli.output)?;
    profile.validate()?;
    let joints = resolve_joints(cli.joint, &profile)?;

    if !cli.dry_run {
        #[cfg(feature = "linux")]
        {
            return run_hardware(cli, &mut profile, &joints);
        }
        #[cfg(not(feature = "linux"))]
        {
            return Err(HalError::Safety(
                "hardware mode requires the `linux` feature; use --dry-run for testing".to_string(),
            ));
        }
    }

    let mut pca0 = Pca9685::new(MockI2cBus::new(), 0x40);
    let mut pca1 = Pca9685::new(MockI2cBus::new(), 0x41);
    let operation = (|| -> HalResult<()> {
        pca0.init(cli.frequency)?;
        pca1.init(cli.frequency)?;

        if cli.test {
            for &joint in &joints {
                if cli.monitor {
                    println!("  (monitor mode: using mock sensor data in dry-run)");
                    let mock_readings = vec![vec![0.3, 5.0]; 200];
                    let mut sensor = MockHalSensor::new("ina219-mock", mock_readings);
                    run_test_sweep_monitored(
                        &mut pca0,
                        &mut pca1,
                        &profile,
                        joint,
                        cli.step,
                        cli.delay_ms,
                        &mut sensor,
                        cli.stall_threshold,
                    )?;
                } else {
                    run_test_sweep(
                        &mut pca0,
                        &mut pca1,
                        &profile,
                        joint,
                        cli.step,
                        cli.delay_ms,
                    )?;
                }
            }
        } else {
            run_calibration(&mut pca0, &mut pca1, &mut profile, &joints)?;
            save_profile(&profile, &cli.output)?;
        }
        Ok(())
    })();

    finish_with_shutdown(operation, shutdown_boards(&mut pca0, &mut pca1))?;
    println!("All servos off. Done.");
    Ok(())
}

fn validate_cli(cli: &Cli) -> HalResult<()> {
    if !cli.dry_run && !cli.arm {
        return Err(HalError::Safety(
            "hardware calibration requires explicit --arm acknowledgement".to_string(),
        ));
    }
    if !cli.frequency.is_finite() || !(24.0..=1526.0).contains(&cli.frequency) {
        return Err(HalError::Safety(format!(
            "PWM frequency must be finite and within 24..=1526 Hz, got {}",
            cli.frequency
        )));
    }
    if cli.step == 0 {
        return Err(HalError::Safety(
            "sweep step must be greater than zero".to_string(),
        ));
    }
    if cli.monitor {
        if !cli.stall_threshold.is_finite() || cli.stall_threshold <= 0.0 {
            return Err(HalError::Safety(format!(
                "stall threshold must be finite and positive, got {}",
                cli.stall_threshold
            )));
        }
        if !cli.shunt_resistance.is_finite() || cli.shunt_resistance <= 0.0 {
            return Err(HalError::Safety(format!(
                "shunt resistance must be finite and positive, got {}",
                cli.shunt_resistance
            )));
        }
    }
    Ok(())
}

// ============================================================================
// Hardware path (linux feature only)
// ============================================================================

#[cfg(feature = "linux")]
fn run_hardware(cli: &Cli, profile: &mut CalibrationProfile, joints: &[usize]) -> HalResult<()> {
    use linux_embedded_hal::I2cdev;
    use std::cell::RefCell;
    use symthaea_hal::RefCellDevice;

    let dev = I2cdev::new(&cli.bus).map_err(|e| HalError::I2c {
        bus: cli.bus.clone(),
        detail: e.to_string(),
    })?;
    let bus = RefCell::new(dev);
    let mut pca0 = Pca9685::new(RefCellDevice::new(&bus), 0x40);
    let mut pca1 = Pca9685::new(RefCellDevice::new(&bus), 0x41);

    let operation = (|| -> HalResult<()> {
        pca0.init(cli.frequency)?;
        pca1.init(cli.frequency)?;

        if cli.test {
            for &joint in joints {
                if cli.monitor {
                    use std::sync::Mutex;
                    use symthaea_hal::{EmbeddedSensor, Ina219Decoder, MutexDevice};

                    let sensor_bus =
                        Mutex::new(I2cdev::new(&cli.bus).map_err(|e| HalError::I2c {
                            bus: cli.bus.clone(),
                            detail: e.to_string(),
                        })?);
                    let mut sensor = EmbeddedSensor::new(
                        MutexDevice::new(&sensor_bus),
                        Ina219Decoder::new(cli.shunt_resistance),
                    );
                    if !sensor.probe()? {
                        return Err(HalError::Safety(
                            "INA219 current monitor probe failed".to_string(),
                        ));
                    }
                    run_test_sweep_monitored(
                        &mut pca0,
                        &mut pca1,
                        profile,
                        joint,
                        cli.step,
                        cli.delay_ms,
                        &mut sensor,
                        cli.stall_threshold,
                    )?;
                } else {
                    run_test_sweep(&mut pca0, &mut pca1, profile, joint, cli.step, cli.delay_ms)?;
                }
            }
        } else {
            run_calibration(&mut pca0, &mut pca1, profile, joints)?;
            save_profile(profile, &cli.output)?;
        }
        Ok(())
    })();

    finish_with_shutdown(operation, shutdown_boards(&mut pca0, &mut pca1))?;
    println!("All servos off. Done.");
    Ok(())
}

// ============================================================================
// Generic calibration functions
// ============================================================================

/// Write a pulse to the correct PCA9685 board for the given joint.
fn write_pulse<I: I2c>(
    joint: usize,
    pulse_us: u16,
    pca0: &mut Pca9685<I>,
    pca1: &mut Pca9685<I>,
) -> HalResult<()> {
    if joint >= NUM_ACTUATORS {
        return Err(HalError::Safety(format!(
            "joint index {joint} is out of range"
        )));
    }
    if joint < 16 {
        pca0.set_pulse_us(joint as u8, pulse_us)
    } else {
        pca1.set_pulse_us((joint - 16) as u8, pulse_us)
    }
}

/// Interactive calibration loop: sweep to center, prompt for min/max/reversed.
fn run_calibration<I: I2c>(
    pca0: &mut Pca9685<I>,
    pca1: &mut Pca9685<I>,
    profile: &mut CalibrationProfile,
    joints: &[usize],
) -> HalResult<()> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();

    for &joint_idx in joints {
        let joint = &profile.joints[joint_idx];
        println!("\n--- Joint {} ({}) ---", joint_idx, joint.name);
        println!(
            "Current: pulse {}–{} µs, angle {:.0}°–{:.0}°, reversed={}",
            joint.pulse_min_us,
            joint.pulse_max_us,
            joint.angle_min_deg,
            joint.angle_max_deg,
            joint.reversed
        );

        // Sweep to center
        let center = joint.center_pulse_us();
        write_pulse(joint_idx, center, pca0, pca1)?;
        println!("  Moved to center ({} µs)", center);

        // Ask for min
        print!("  Enter min pulse (µs) [{}]: ", joint.pulse_min_us);
        io::stdout()
            .flush()
            .map_err(|e| HalError::Calibration(format!("failed to flush prompt: {e}")))?;
        let min = read_u16_or_default(&mut reader, joint.pulse_min_us)?;

        // Ask for max
        print!("  Enter max pulse (µs) [{}]: ", joint.pulse_max_us);
        io::stdout()
            .flush()
            .map_err(|e| HalError::Calibration(format!("failed to flush prompt: {e}")))?;
        let max = read_u16_or_default(&mut reader, joint.pulse_max_us)?;

        // Ask for reversed
        print!(
            "  Reversed? (y/n) [{}]: ",
            if joint.reversed { "y" } else { "n" }
        );
        io::stdout()
            .flush()
            .map_err(|e| HalError::Calibration(format!("failed to flush prompt: {e}")))?;
        let reversed = read_bool_or_default(&mut reader, joint.reversed)?;

        if min >= max {
            return Err(HalError::Calibration(format!(
                "joint {joint_idx} requires min pulse < max pulse, got {min}..{max}"
            )));
        }

        // Update profile
        let j = profile.joint_mut(joint_idx).ok_or_else(|| {
            HalError::Calibration(format!("joint {joint_idx} disappeared from profile"))
        })?;
        j.pulse_min_us = min;
        j.pulse_max_us = max;
        j.reversed = reversed;

        println!("  Updated: pulse {}–{} µs, reversed={}", min, max, reversed);
    }
    profile.validate()?;
    Ok(())
}

/// Test sweep: move a joint from min to max in steps, then back.
fn run_test_sweep<I: I2c>(
    pca0: &mut Pca9685<I>,
    pca1: &mut Pca9685<I>,
    profile: &CalibrationProfile,
    joint: usize,
    step: u16,
    delay_ms: u64,
) -> HalResult<()> {
    if step == 0 {
        return Err(HalError::Safety(
            "sweep step must be greater than zero".to_string(),
        ));
    }
    let cal = profile
        .joint(joint)
        .ok_or_else(|| HalError::Calibration(format!("joint {joint} out of range")))?;
    cal.validate(joint)?;

    println!("\n--- Test sweep: joint {} ({}) ---", joint, cal.name);
    println!(
        "  Range: {} → {} µs, step {} µs, delay {} ms",
        cal.pulse_min_us, cal.pulse_max_us, step, delay_ms
    );

    let delay = std::time::Duration::from_millis(delay_ms);
    for pulse in (cal.pulse_min_us..=cal.pulse_max_us).step_by(step as usize) {
        println!("  pulse = {} µs", pulse);
        write_pulse(joint, pulse, pca0, pca1)?;
        std::thread::sleep(delay);
    }
    for pulse in (cal.pulse_min_us..=cal.pulse_max_us)
        .rev()
        .step_by(step as usize)
    {
        println!("  pulse = {} µs", pulse);
        write_pulse(joint, pulse, pca0, pca1)?;
        std::thread::sleep(delay);
    }

    let center = cal.center_pulse_us();
    write_pulse(joint, center, pca0, pca1)?;
    println!("  Returned to center ({} µs)", center);
    Ok(())
}

// ============================================================================
// Monitored sweep
// ============================================================================

/// Read current status from a sensor and classify it.
///
/// Returns `(current_a, voltage_v, status)` where status is "OK", "STALL",
/// or "NO_DATA" if the sensor returned nothing.
fn read_current_status(
    sensor: &mut dyn HalSensorAdapter,
    stall_threshold: f32,
) -> (f32, f32, &'static str) {
    match sensor.read_raw() {
        Some(values) => {
            let Some(current) = values.first().copied().filter(|v| v.is_finite()) else {
                return (0.0, 0.0, "NO_DATA");
            };
            let Some(voltage) = values.get(1).copied().filter(|v| v.is_finite()) else {
                return (current, 0.0, "NO_DATA");
            };
            let status = if current.abs() >= stall_threshold {
                "STALL"
            } else {
                "OK"
            };
            (current, voltage, status)
        }
        None => (0.0, 0.0, "NO_DATA"),
    }
}

/// Test sweep with current monitoring at each step.
fn run_test_sweep_monitored<I: I2c>(
    pca0: &mut Pca9685<I>,
    pca1: &mut Pca9685<I>,
    profile: &CalibrationProfile,
    joint: usize,
    step: u16,
    delay_ms: u64,
    sensor: &mut dyn HalSensorAdapter,
    stall_threshold: f32,
) -> HalResult<()> {
    if step == 0 {
        return Err(HalError::Safety(
            "sweep step must be greater than zero".to_string(),
        ));
    }
    if !stall_threshold.is_finite() || stall_threshold <= 0.0 {
        return Err(HalError::Safety(format!(
            "invalid stall threshold: {stall_threshold}"
        )));
    }
    let cal = profile
        .joint(joint)
        .ok_or_else(|| HalError::Calibration(format!("joint {joint} out of range")))?;
    cal.validate(joint)?;
    println!("\n--- Monitored sweep: joint {} ({}) ---", joint, cal.name);
    println!(
        "  Range: {} → {} µs, step {} µs, delay {} ms, stall threshold {:.2}A",
        cal.pulse_min_us, cal.pulse_max_us, step, delay_ms, stall_threshold
    );
    println!(
        "  {:>8} {:>10} {:>10} {:>8}",
        "pulse", "current", "voltage", "status"
    );
    println!("  {:-<8} {:-<10} {:-<10} {:-<8}", "", "", "", "");

    let delay = std::time::Duration::from_millis(delay_ms);
    for pulse in (cal.pulse_min_us..=cal.pulse_max_us).step_by(step as usize) {
        write_pulse(joint, pulse, pca0, pca1)?;
        std::thread::sleep(delay);
        let (current, voltage, status) = read_current_status(sensor, stall_threshold);
        println!(
            "  {:>8} {:>9.3}A {:>9.3}V {:>8}",
            pulse, current, voltage, status
        );
        match status {
            "OK" => {}
            "STALL" => {
                return Err(HalError::Safety(format!(
                    "stall detected on joint {joint} at pulse {pulse} µs ({:.3} A)",
                    current.abs()
                )));
            }
            _ => {
                return Err(HalError::Safety(format!(
                    "current telemetry unavailable on joint {joint} at pulse {pulse} µs"
                )));
            }
        }
    }

    let center = cal.center_pulse_us();
    write_pulse(joint, center, pca0, pca1)?;
    println!("  Returned to center ({} µs)", center);
    println!("  No stall points detected.");
    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

fn load_or_create_profile(path: &Path) -> HalResult<CalibrationProfile> {
    if path.exists() {
        println!("Loading existing profile from {}", path.display());
        CalibrationProfile::load(path)
    } else {
        Ok(CalibrationProfile::default_21())
    }
}

fn save_profile(profile: &CalibrationProfile, path: &Path) -> HalResult<()> {
    profile.save(path)?;
    println!("\nProfile saved to {}", path.display());
    Ok(())
}

fn resolve_joints(joint: Option<usize>, profile: &CalibrationProfile) -> HalResult<Vec<usize>> {
    if let Some(joint) = joint {
        if joint >= profile.joints.len() {
            return Err(HalError::Calibration(format!(
                "joint {joint} out of range (0-{})",
                profile.joints.len().saturating_sub(1)
            )));
        }
        Ok(vec![joint])
    } else {
        Ok((0..profile.joints.len()).collect())
    }
}

fn shutdown_boards<I: I2c>(pca0: &mut Pca9685<I>, pca1: &mut Pca9685<I>) -> HalResult<()> {
    let board0 = pca0.all_off();
    let board1 = pca1.all_off();
    match (board0, board1) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(e), Ok(())) | (Ok(()), Err(e)) => Err(e),
        (Err(e0), Err(e1)) => Err(HalError::Safety(format!(
            "failed to disable both PWM boards: board0={e0}; board1={e1}"
        ))),
    }
}

fn finish_with_shutdown<T>(operation: HalResult<T>, shutdown: HalResult<()>) -> HalResult<T> {
    match (operation, shutdown) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(operation_error), Err(shutdown_error)) => Err(HalError::Safety(format!(
            "operation failed: {operation_error}; shutdown also failed: {shutdown_error}"
        ))),
    }
}

fn read_u16_or_default(reader: &mut impl BufRead, default: u16) -> HalResult<u16> {
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| HalError::Calibration(format!("failed to read calibration input: {e}")))?;
    let trimmed = line.trim();
    if trimmed.is_empty() {
        Ok(default)
    } else {
        trimmed
            .parse()
            .map_err(|e| HalError::Calibration(format!("invalid pulse width '{trimmed}': {e}")))
    }
}

fn read_bool_or_default(reader: &mut impl BufRead, default: bool) -> HalResult<bool> {
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| HalError::Calibration(format!("failed to read calibration input: {e}")))?;
    let trimmed = line.trim().to_lowercase();
    if trimmed.is_empty() {
        return Ok(default);
    }
    match trimmed.as_str() {
        "y" | "yes" | "true" | "1" => Ok(true),
        "n" | "no" | "false" | "0" => Ok(false),
        _ => Err(HalError::Calibration(format!(
            "invalid boolean response '{trimmed}'"
        ))),
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_hal::mock::MockI2cBus;

    #[test]
    fn test_write_pulse_dispatches_board() {
        let mut pca0 = Pca9685::new(MockI2cBus::new(), 0x40);
        let mut pca1 = Pca9685::new(MockI2cBus::new(), 0x41);
        pca0.init(50.0).unwrap();
        pca1.init(50.0).unwrap();

        // Joint 3 → board 0
        write_pulse(3, 1500, &mut pca0, &mut pca1).unwrap();
        // Joint 17 → board 1, channel 1
        write_pulse(17, 1200, &mut pca0, &mut pca1).unwrap();

        // No panic is success — mock bus records transactions
    }

    #[test]
    fn test_run_test_sweep_completes() {
        let mut pca0 = Pca9685::new(MockI2cBus::new(), 0x40);
        let mut pca1 = Pca9685::new(MockI2cBus::new(), 0x41);
        pca0.init(50.0).unwrap();
        pca1.init(50.0).unwrap();
        let profile = CalibrationProfile::default_21();

        // Sweep joint 0 with 0ms delay (fast test)
        run_test_sweep(&mut pca0, &mut pca1, &profile, 0, 200, 0).unwrap();
        // No panic is success
    }

    #[test]
    fn test_monitored_sweep_with_mock_sensor() {
        let mut pca0 = Pca9685::new(MockI2cBus::new(), 0x40);
        let mut pca1 = Pca9685::new(MockI2cBus::new(), 0x41);
        pca0.init(50.0).unwrap();
        pca1.init(50.0).unwrap();
        let profile = CalibrationProfile::default_21();

        // Normal readings (below stall threshold)
        let readings: Vec<Vec<f32>> = (0..100).map(|_| vec![0.5, 5.0]).collect();
        let mut sensor = MockHalSensor::new("ina219", readings);
        run_test_sweep_monitored(&mut pca0, &mut pca1, &profile, 0, 200, 0, &mut sensor, 1.5)
            .unwrap();
    }

    #[test]
    fn test_monitored_sweep_with_empty_sensor() {
        let mut pca0 = Pca9685::new(MockI2cBus::new(), 0x40);
        let mut pca1 = Pca9685::new(MockI2cBus::new(), 0x41);
        pca0.init(50.0).unwrap();
        pca1.init(50.0).unwrap();
        let profile = CalibrationProfile::default_21();

        // Empty sensor: no data
        let mut sensor = MockHalSensor::new("ina219", vec![]);
        assert!(
            run_test_sweep_monitored(&mut pca0, &mut pca1, &profile, 0, 200, 0, &mut sensor, 1.5,)
                .is_err()
        );
    }

    #[test]
    fn test_zero_step_rejected() {
        let mut pca0 = Pca9685::new(MockI2cBus::new(), 0x40);
        let mut pca1 = Pca9685::new(MockI2cBus::new(), 0x41);
        pca0.init(50.0).unwrap();
        pca1.init(50.0).unwrap();
        let profile = CalibrationProfile::default_21();
        assert!(run_test_sweep(&mut pca0, &mut pca1, &profile, 0, 0, 0).is_err());
    }

    #[test]
    fn test_monitored_sweep_stops_on_first_stall() {
        let mut pca0 = Pca9685::new(MockI2cBus::new(), 0x40);
        let mut pca1 = Pca9685::new(MockI2cBus::new(), 0x41);
        pca0.init(50.0).unwrap();
        pca1.init(50.0).unwrap();
        let profile = CalibrationProfile::default_21();
        let mut sensor = MockHalSensor::new("ina219", vec![vec![2.0, 5.0]]);
        assert!(
            run_test_sweep_monitored(&mut pca0, &mut pca1, &profile, 0, 200, 0, &mut sensor, 1.5,)
                .is_err()
        );
    }

    #[test]
    fn test_invalid_numeric_input_is_rejected() {
        let mut input = std::io::Cursor::new(b"not-a-number\n".to_vec());
        assert!(read_u16_or_default(&mut input, 1500).is_err());
    }

    #[test]
    fn test_invalid_boolean_input_is_rejected() {
        let mut input = std::io::Cursor::new(b"maybe\n".to_vec());
        assert!(read_bool_or_default(&mut input, false).is_err());
    }

    #[test]
    fn test_read_current_status_normal() {
        let mut sensor = MockHalSensor::new("ina219", vec![vec![0.5, 5.0]]);
        let (current, voltage, status) = read_current_status(&mut sensor, 1.5);
        assert!((current - 0.5).abs() < 0.001);
        assert!((voltage - 5.0).abs() < 0.001);
        assert_eq!(status, "OK");
    }

    #[test]
    fn test_read_current_status_stall() {
        let mut sensor = MockHalSensor::new("ina219", vec![vec![2.0, 4.8]]);
        let (current, _voltage, status) = read_current_status(&mut sensor, 1.5);
        assert!((current - 2.0).abs() < 0.001);
        assert_eq!(status, "STALL");
    }

    #[test]
    fn test_read_current_status_no_data() {
        let mut sensor = MockHalSensor::new("ina219", vec![]);
        let (current, voltage, status) = read_current_status(&mut sensor, 1.5);
        assert!((current - 0.0).abs() < f32::EPSILON);
        assert!((voltage - 0.0).abs() < f32::EPSILON);
        assert_eq!(status, "NO_DATA");
    }
}
