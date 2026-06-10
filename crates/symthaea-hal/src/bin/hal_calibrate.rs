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

/// Servo calibration tool for symthaea-hal.
#[derive(Parser)]
#[command(name = "hal-calibrate", version, about = "Calibrate servo endpoints")]
struct Cli {
    /// Use mock I2C bus (no hardware required).
    #[arg(long)]
    dry_run: bool,

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

    let mut profile = load_or_create_profile(&cli.output);
    let joints = resolve_joints(cli.joint, &profile);

    if !cli.dry_run {
        #[cfg(feature = "linux")]
        {
            run_hardware(&cli, &mut profile, &joints);
            return;
        }
        #[cfg(not(feature = "linux"))]
        {
            eprintln!("Hardware mode requires the `linux` feature. Use --dry-run for testing.");
            std::process::exit(1);
        }
    }

    // Mock (dry-run) path
    let mut pca0 = Pca9685::new(MockI2cBus::new(), 0x40);
    let mut pca1 = Pca9685::new(MockI2cBus::new(), 0x41);
    pca0.init(cli.frequency).unwrap();
    pca1.init(cli.frequency).unwrap();

    if cli.test {
        for &j in &joints {
            if cli.monitor {
                println!("  (monitor mode: using mock sensor data in dry-run)");
                // Mock sensor: [current_a, voltage_v] per reading, enough for a full sweep
                let mock_readings: Vec<Vec<f32>> = (0..200)
                    .map(|i| vec![0.3 + (i as f32) * 0.01, 5.0])
                    .collect();
                let mut sensor = MockHalSensor::new("ina219-mock", mock_readings);
                run_test_sweep_monitored(
                    &mut pca0,
                    &mut pca1,
                    &profile,
                    j,
                    cli.step,
                    cli.delay_ms,
                    &mut sensor,
                    cli.stall_threshold,
                );
            } else {
                run_test_sweep(&mut pca0, &mut pca1, &profile, j, cli.step, cli.delay_ms);
            }
        }
    } else {
        run_calibration(&mut pca0, &mut pca1, &mut profile, &joints);
        save_profile(&profile, &cli.output);
    }

    pca0.all_off().unwrap();
    pca1.all_off().unwrap();
    println!("All servos off. Done.");
}

// ============================================================================
// Hardware path (linux feature only)
// ============================================================================

#[cfg(feature = "linux")]
fn run_hardware(cli: &Cli, profile: &mut CalibrationProfile, joints: &[usize]) {
    use linux_embedded_hal::I2cdev;
    use std::cell::RefCell;
    use symthaea_hal::RefCellDevice;

    let dev = I2cdev::new(&cli.bus).unwrap_or_else(|e| {
        eprintln!("Failed to open I2C bus {}: {}", cli.bus, e);
        std::process::exit(1);
    });
    let bus = RefCell::new(dev);
    let mut pca0 = Pca9685::new(RefCellDevice::new(&bus), 0x40);
    let mut pca1 = Pca9685::new(RefCellDevice::new(&bus), 0x41);
    pca0.init(cli.frequency).unwrap();
    pca1.init(cli.frequency).unwrap();

    if cli.test {
        for &j in joints {
            if cli.monitor {
                use std::sync::Mutex;
                use symthaea_hal::{EmbeddedSensor, Ina219Decoder, MutexDevice};
                let bus_mtx = Mutex::new(I2cdev::new(&cli.bus).unwrap_or_else(|e| {
                    eprintln!("Failed to open I2C bus for sensor: {}", e);
                    std::process::exit(1);
                }));
                let mut sensor = EmbeddedSensor::new(
                    MutexDevice::new(&bus_mtx),
                    Ina219Decoder::new(cli.shunt_resistance),
                );
                run_test_sweep_monitored(
                    &mut pca0,
                    &mut pca1,
                    profile,
                    j,
                    cli.step,
                    cli.delay_ms,
                    &mut sensor,
                    cli.stall_threshold,
                );
            } else {
                run_test_sweep(&mut pca0, &mut pca1, profile, j, cli.step, cli.delay_ms);
            }
        }
    } else {
        run_calibration(&mut pca0, &mut pca1, profile, joints);
        save_profile(profile, &cli.output);
    }

    pca0.all_off().unwrap();
    pca1.all_off().unwrap();
    println!("All servos off. Done.");
}

// ============================================================================
// Generic calibration functions
// ============================================================================

/// Write a pulse to the correct PCA9685 board for the given joint.
fn write_pulse<I: I2c>(joint: usize, pulse_us: u16, pca0: &mut Pca9685<I>, pca1: &mut Pca9685<I>) {
    if joint < 16 {
        let _ = pca0.set_pulse_us(joint as u8, pulse_us);
    } else {
        let _ = pca1.set_pulse_us((joint - 16) as u8, pulse_us);
    }
}

/// Interactive calibration loop: sweep to center, prompt for min/max/reversed.
fn run_calibration<I: I2c>(
    pca0: &mut Pca9685<I>,
    pca1: &mut Pca9685<I>,
    profile: &mut CalibrationProfile,
    joints: &[usize],
) {
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
        write_pulse(joint_idx, center, pca0, pca1);
        println!("  Moved to center ({} µs)", center);

        // Ask for min
        print!("  Enter min pulse (µs) [{}]: ", joint.pulse_min_us);
        io::stdout().flush().unwrap();
        let min = read_u16_or_default(&mut reader, joint.pulse_min_us);

        // Ask for max
        print!("  Enter max pulse (µs) [{}]: ", joint.pulse_max_us);
        io::stdout().flush().unwrap();
        let max = read_u16_or_default(&mut reader, joint.pulse_max_us);

        // Ask for reversed
        print!(
            "  Reversed? (y/n) [{}]: ",
            if joint.reversed { "y" } else { "n" }
        );
        io::stdout().flush().unwrap();
        let reversed = read_bool_or_default(&mut reader, joint.reversed);

        // Update profile
        let j = profile.joint_mut(joint_idx).unwrap();
        j.pulse_min_us = min;
        j.pulse_max_us = max;
        j.reversed = reversed;

        println!("  Updated: pulse {}–{} µs, reversed={}", min, max, reversed);
    }
}

/// Test sweep: move a joint from min to max in steps, then back.
fn run_test_sweep<I: I2c>(
    pca0: &mut Pca9685<I>,
    pca1: &mut Pca9685<I>,
    profile: &CalibrationProfile,
    joint: usize,
    step: u16,
    delay_ms: u64,
) {
    let cal = &profile.joints[joint];
    println!("\n--- Test sweep: joint {} ({}) ---", joint, cal.name);
    println!(
        "  Range: {} → {} µs, step {} µs, delay {} ms",
        cal.pulse_min_us, cal.pulse_max_us, step, delay_ms
    );

    let delay = std::time::Duration::from_millis(delay_ms);

    // Sweep up
    let mut pulse = cal.pulse_min_us;
    while pulse <= cal.pulse_max_us {
        println!("  pulse = {} µs", pulse);
        write_pulse(joint, pulse, pca0, pca1);
        std::thread::sleep(delay);
        pulse = pulse.saturating_add(step);
    }

    // Sweep down
    pulse = cal.pulse_max_us;
    while pulse >= cal.pulse_min_us {
        println!("  pulse = {} µs", pulse);
        write_pulse(joint, pulse, pca0, pca1);
        std::thread::sleep(delay);
        if pulse < step {
            break;
        }
        pulse = pulse.saturating_sub(step);
    }

    // Return to center
    let center = cal.center_pulse_us();
    write_pulse(joint, center, pca0, pca1);
    println!("  Returned to center ({} µs)", center);
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
            let current = values.first().copied().unwrap_or(0.0);
            let voltage = values.get(1).copied().unwrap_or(0.0);
            let status = if current >= stall_threshold {
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
) {
    let cal = &profile.joints[joint];
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
    let mut stall_points: Vec<u16> = Vec::new();

    // Sweep up
    let mut pulse = cal.pulse_min_us;
    while pulse <= cal.pulse_max_us {
        write_pulse(joint, pulse, pca0, pca1);
        std::thread::sleep(delay);
        let (current, voltage, status) = read_current_status(sensor, stall_threshold);
        println!(
            "  {:>8} {:>9.3}A {:>9.3}V {:>8}",
            pulse, current, voltage, status
        );
        if status == "STALL" {
            stall_points.push(pulse);
        }
        pulse = pulse.saturating_add(step);
    }

    // Return to center
    let center = cal.center_pulse_us();
    write_pulse(joint, center, pca0, pca1);
    println!("  Returned to center ({} µs)", center);

    // Summary
    if stall_points.is_empty() {
        println!("  No stall points detected.");
    } else {
        println!(
            "  WARNING: {} stall points detected at pulses: {:?}",
            stall_points.len(),
            stall_points
        );
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn load_or_create_profile(path: &Path) -> CalibrationProfile {
    if path.exists() {
        println!("Loading existing profile from {}", path.display());
        CalibrationProfile::load(path).unwrap_or_else(|e| {
            eprintln!("Failed to load profile: {}. Starting fresh.", e);
            CalibrationProfile::default_21()
        })
    } else {
        CalibrationProfile::default_21()
    }
}

fn save_profile(profile: &CalibrationProfile, path: &Path) {
    match profile.save(path) {
        Ok(()) => println!("\nProfile saved to {}", path.display()),
        Err(e) => {
            eprintln!("\nFailed to save profile: {}", e);
            std::process::exit(1);
        }
    }
}

fn resolve_joints(joint: Option<usize>, profile: &CalibrationProfile) -> Vec<usize> {
    if let Some(j) = joint {
        if j >= profile.joints.len() {
            eprintln!("Joint {} out of range (0-{})", j, profile.joints.len() - 1);
            std::process::exit(1);
        }
        vec![j]
    } else {
        (0..profile.joints.len()).collect()
    }
}

fn read_u16_or_default(reader: &mut impl BufRead, default: u16) -> u16 {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap_or(0);
    let trimmed = line.trim();
    if trimmed.is_empty() {
        default
    } else {
        trimmed.parse().unwrap_or(default)
    }
}

fn read_bool_or_default(reader: &mut impl BufRead, default: bool) -> bool {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap_or(0);
    let trimmed = line.trim().to_lowercase();
    if trimmed.is_empty() {
        default
    } else {
        matches!(trimmed.as_str(), "y" | "yes" | "true" | "1")
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
        write_pulse(3, 1500, &mut pca0, &mut pca1);
        // Joint 17 → board 1, channel 1
        write_pulse(17, 1200, &mut pca0, &mut pca1);

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
        run_test_sweep(&mut pca0, &mut pca1, &profile, 0, 200, 0);
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
        run_test_sweep_monitored(&mut pca0, &mut pca1, &profile, 0, 200, 0, &mut sensor, 1.5);
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
        run_test_sweep_monitored(&mut pca0, &mut pca1, &profile, 0, 200, 0, &mut sensor, 1.5);
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
