// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Servo output: `HumanoidCommand` → PCA9685 batch write with slew-rate limiting.
//!
//! [`ServoOutput`] bridges the cognitive loop's 21-float torque vector to
//! two PCA9685 boards via calibrated pulse-width conversion and slew-rate
//! limiting for smooth, safe motion.
//!
//! # Board Layout
//!
//! ```text
//! Board 0 (0x40): joints  0–15  (abdomen + legs + right_shoulder1)
//! Board 1 (0x41): joints 16–20  (remaining arms, channels 0–4)
//! ```

use embedded_hal::i2c::I2c;
use embedded_hal_bus::i2c::{MutexDevice, RefCellDevice};
use std::cell::RefCell;
use std::sync::Mutex;
use symthaea_humanoid::types::{HumanoidCommand, NUM_ACTUATORS};
use tracing::debug;

use crate::calibration::CalibrationProfile;
use crate::error::HalResult;
use crate::pca9685::{CHANNELS, Pca9685};

// ============================================================================
// SERVO OUTPUT
// ============================================================================

/// Maximum slew rate in µs per tick (limits how fast servos move per update).
const DEFAULT_SLEW_RATE_US: u16 = 100;

/// Servo output controller for 21-joint humanoid.
///
/// Owns two PCA9685 boards and a calibration profile. Each `apply()` call:
/// 1. Converts torques → pulse widths via calibration
/// 2. Applies slew-rate limiting
/// 3. Batch-writes to the PCA9685 boards
pub struct ServoOutput<I> {
    board0: Pca9685<I>,
    board1: Pca9685<I>,
    calibration: CalibrationProfile,
    /// Last commanded pulse widths (for slew-rate limiting).
    last_pulses: [u16; NUM_ACTUATORS],
    /// Maximum change in µs per update cycle.
    slew_rate_us: u16,
    /// Whether servo output is enabled.
    enabled: bool,
}

impl<I: I2c> ServoOutput<I> {
    /// Create a new servo output with two I2C buses (or the same bus) and calibration.
    ///
    /// - `bus0`: I2C bus for board 0 (address 0x40, joints 0–15)
    /// - `bus1`: I2C bus for board 1 (address 0x41, joints 16–20)
    pub fn new(bus0: I, bus1: I, calibration: CalibrationProfile) -> Self {
        let center = calibration.joints[0].center_pulse_us();
        Self {
            board0: Pca9685::new(bus0, 0x40),
            board1: Pca9685::new(bus1, 0x41),
            calibration,
            last_pulses: [center; NUM_ACTUATORS],
            slew_rate_us: DEFAULT_SLEW_RATE_US,
            enabled: false,
        }
    }

    /// Get a reference to board 0 (joints 0–15).
    pub fn board0(&self) -> &Pca9685<I> {
        &self.board0
    }

    /// Get a reference to board 1 (joints 16–20).
    pub fn board1(&self) -> &Pca9685<I> {
        &self.board1
    }

    /// Initialize both PCA9685 boards at the given PWM frequency (typically 50 Hz).
    pub fn init(&mut self, frequency_hz: f64) -> HalResult<()> {
        self.board0.init(frequency_hz)?;
        self.board1.init(frequency_hz)?;
        debug!("servo output initialized at {}Hz", frequency_hz);
        Ok(())
    }

    /// Enable servo output (allows `apply()` to write to hardware).
    pub fn enable(&mut self) {
        self.enabled = true;
        debug!("servo output enabled");
    }

    /// Disable servo output and turn off all channels.
    pub fn disable(&mut self) -> HalResult<()> {
        self.enabled = false;
        self.board0.all_off()?;
        self.board1.all_off()?;
        debug!("servo output disabled");
        Ok(())
    }

    /// Whether output is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set the maximum slew rate (µs per update tick).
    pub fn set_slew_rate(&mut self, us_per_tick: u16) {
        self.slew_rate_us = us_per_tick;
    }

    /// Set the calibration profile.
    pub fn set_calibration(&mut self, cal: CalibrationProfile) {
        self.calibration = cal;
    }

    /// Apply a `HumanoidCommand` to the servos.
    ///
    /// Returns `Ok(())` if disabled (no-op). On enabled, converts torques
    /// to pulses, applies slew-rate limiting, and writes to both boards.
    pub fn apply(&mut self, command: &HumanoidCommand) -> HalResult<()> {
        if !self.enabled {
            return Ok(());
        }

        // 1. Convert torques → target pulse widths
        let targets = self.calibration.torques_to_pulses(&command.torques);

        // 2. Slew-rate limiting
        let mut pulses = [0u16; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS {
            pulses[i] = slew_limit(self.last_pulses[i], targets[i], self.slew_rate_us);
        }
        self.last_pulses = pulses;

        // 3. Write to board 0 (joints 0–15)
        let board0_pulses: Vec<u16> = pulses[..CHANNELS].to_vec();
        self.board0.set_pulse_batch(0, &board0_pulses)?;

        // 4. Write to board 1 (joints 16–20, mapped to channels 0–4)
        let board1_pulses: Vec<u16> = pulses[CHANNELS..].to_vec();
        self.board1.set_pulse_batch(0, &board1_pulses)?;

        Ok(())
    }

    /// Get the last commanded pulse widths.
    pub fn last_pulses(&self) -> &[u16; NUM_ACTUATORS] {
        &self.last_pulses
    }

    /// Move all servos to their center (neutral) position.
    pub fn center_all(&mut self) -> HalResult<()> {
        let zero = HumanoidCommand::zero();
        // Temporarily bypass slew-rate for immediate centering
        let saved = self.slew_rate_us;
        self.slew_rate_us = u16::MAX;
        let result = self.apply(&zero);
        self.slew_rate_us = saved;
        result
    }

    /// Get a reference to the calibration profile.
    pub fn calibration(&self) -> &CalibrationProfile {
        &self.calibration
    }

    /// Read back the OFF counts for all 21 servo channels across both boards.
    ///
    /// Returns an array of 21 OFF counts (12-bit). This is a diagnostic
    /// operation (21 I2C reads), not intended for per-tick use.
    pub fn read_positions(&mut self) -> HalResult<[u16; NUM_ACTUATORS]> {
        let mut counts = [0u16; NUM_ACTUATORS];
        for (i, count) in counts.iter_mut().enumerate().take(CHANNELS) {
            let (_on, off) = self.board0.read_channel(i as u8)?;
            *count = off;
        }
        for i in 0..(NUM_ACTUATORS - CHANNELS) {
            let (_on, off) = self.board1.read_channel(i as u8)?;
            counts[CHANNELS + i] = off;
        }
        Ok(counts)
    }

    /// Compare last commanded pulses against actual readback from PCA9685.
    ///
    /// Returns a list of mismatches: `(joint_index, commanded_pulse_us, actual_off_count)`.
    /// An empty vector means all positions match. This is a diagnostic
    /// operation (21 I2C reads), not intended for per-tick use.
    pub fn verify_positions(&mut self) -> HalResult<Vec<(usize, u16, u16)>> {
        let actual = self.read_positions()?;
        let period_us = self.board0.period_us();
        let mut mismatches = Vec::new();
        for (i, &pulse) in self.last_pulses.iter().enumerate() {
            let expected_count = ((pulse as f64 / period_us) * 4096.0).round() as u16;
            let expected_count = expected_count.min(4095);
            if actual[i] != expected_count {
                mismatches.push((i, pulse, actual[i]));
            }
        }
        Ok(mismatches)
    }
}

// ============================================================================
// SHARED-BUS CONSTRUCTORS
// ============================================================================

impl<'a, I: I2c> ServoOutput<RefCellDevice<'a, I>> {
    /// Create a servo output sharing a single I2C bus via `RefCell` (single-threaded).
    ///
    /// Both PCA9685 boards use the same bus wrapped in a `RefCell`.
    /// This is the simplest approach when all access happens on one thread.
    pub fn new_shared(bus: &'a RefCell<I>, calibration: CalibrationProfile) -> Self {
        let center = calibration.joints[0].center_pulse_us();
        Self {
            board0: Pca9685::new(RefCellDevice::new(bus), 0x40),
            board1: Pca9685::new(RefCellDevice::new(bus), 0x41),
            calibration,
            last_pulses: [center; NUM_ACTUATORS],
            slew_rate_us: DEFAULT_SLEW_RATE_US,
            enabled: false,
        }
    }
}

impl<'a, I: I2c + Send> ServoOutput<MutexDevice<'a, I>> {
    /// Create a servo output sharing a single I2C bus via `Mutex` (thread-safe).
    ///
    /// Both PCA9685 boards use the same bus wrapped in a `std::sync::Mutex`.
    /// Use this when sensor reads and servo writes happen on different threads.
    pub fn new_shared_mutex(bus: &'a Mutex<I>, calibration: CalibrationProfile) -> Self {
        let center = calibration.joints[0].center_pulse_us();
        Self {
            board0: Pca9685::new(MutexDevice::new(bus), 0x40),
            board1: Pca9685::new(MutexDevice::new(bus), 0x41),
            calibration,
            last_pulses: [center; NUM_ACTUATORS],
            slew_rate_us: DEFAULT_SLEW_RATE_US,
            enabled: false,
        }
    }
}

// ============================================================================
// SLEW-RATE LIMITER
// ============================================================================

/// Limit the step from `current` to `target` by at most `max_step` µs.
fn slew_limit(current: u16, target: u16, max_step: u16) -> u16 {
    if target > current {
        let delta = target - current;
        if delta > max_step {
            current + max_step
        } else {
            target
        }
    } else {
        let delta = current - target;
        if delta > max_step {
            current - max_step
        } else {
            target
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockI2cBus;

    fn make_servo() -> ServoOutput<MockI2cBus> {
        let bus0 = MockI2cBus::new();
        let bus1 = MockI2cBus::new();
        let cal = CalibrationProfile::default_21();
        ServoOutput::new(bus0, bus1, cal)
    }

    #[test]
    fn test_servo_disabled_noop() {
        let mut servo = make_servo();
        assert!(!servo.is_enabled());
        let cmd = HumanoidCommand::zero();
        servo.apply(&cmd).unwrap(); // no-op, no error
    }

    #[test]
    fn test_servo_enable_apply() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable();
        assert!(servo.is_enabled());

        let cmd = HumanoidCommand::zero();
        servo.apply(&cmd).unwrap();

        // All pulses should be center (1500)
        for &p in servo.last_pulses() {
            assert_eq!(p, 1500);
        }
    }

    #[test]
    fn test_servo_disable_turns_off() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable();
        servo.disable().unwrap();
        assert!(!servo.is_enabled());
    }

    #[test]
    fn test_slew_rate_limiting() {
        // From 1500 to 2500 with max step 100 → 1600
        assert_eq!(slew_limit(1500, 2500, 100), 1600);
        // From 1500 to 500 with max step 100 → 1400
        assert_eq!(slew_limit(1500, 500, 100), 1400);
        // Within range: no limiting
        assert_eq!(slew_limit(1500, 1550, 100), 1550);
        // Exact match
        assert_eq!(slew_limit(1500, 1500, 100), 1500);
    }

    #[test]
    fn test_servo_slew_rate_applied() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable();
        servo.set_slew_rate(50); // very slow

        // Command full +1 → target 2500, but slew from 1500 limits to 1550
        let mut cmd = HumanoidCommand::zero();
        cmd.torques[0] = 1.0;
        servo.apply(&cmd).unwrap();
        assert_eq!(servo.last_pulses()[0], 1550);

        // Second apply: 1550 → 1600
        servo.apply(&cmd).unwrap();
        assert_eq!(servo.last_pulses()[0], 1600);
    }

    #[test]
    fn test_servo_new_shared_refcell() {
        let bus = RefCell::new(MockI2cBus::new());
        let cal = CalibrationProfile::default_21();
        let mut servo = ServoOutput::new_shared(&bus, cal);
        servo.init(50.0).unwrap();
        servo.enable();
        let cmd = HumanoidCommand::zero();
        servo.apply(&cmd).unwrap();
        for &p in servo.last_pulses() {
            assert_eq!(p, 1500);
        }
    }

    #[test]
    fn test_servo_new_shared_mutex() {
        let bus = Mutex::new(MockI2cBus::new());
        let cal = CalibrationProfile::default_21();
        let mut servo = ServoOutput::new_shared_mutex(&bus, cal);
        servo.init(50.0).unwrap();
        servo.enable();
        let cmd = HumanoidCommand::zero();
        servo.apply(&cmd).unwrap();
        for &p in servo.last_pulses() {
            assert_eq!(p, 1500);
        }
    }

    // ── Position readback tests ──────────────────────────────────────

    #[test]
    fn test_read_positions_returns_21_elements() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        let positions = servo.read_positions().unwrap();
        assert_eq!(positions.len(), NUM_ACTUATORS);
    }

    #[test]
    fn test_verify_positions_detects_mismatches() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable();

        // Apply zero command → last_pulses = 1500
        let cmd = HumanoidCommand::zero();
        servo.apply(&cmd).unwrap();
        assert_eq!(servo.last_pulses()[0], 1500);

        // Mock returns 0 for all reads → mismatch with commanded 1500
        let mismatches = servo.verify_positions().unwrap();
        // 1500µs at 50Hz → count ≈ 307, mock returns 0
        assert!(!mismatches.is_empty(), "should detect mismatches");
        // All 21 joints should mismatch (all commanded 1500, all read 0)
        assert_eq!(mismatches.len(), NUM_ACTUATORS);
        assert_eq!(mismatches[0].1, 1500); // commanded pulse
        assert_eq!(mismatches[0].2, 0); // actual off count from mock
    }

    #[test]
    fn test_center_all() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable();

        // Move away from center first
        let mut cmd = HumanoidCommand::zero();
        cmd.torques[0] = 1.0;
        servo.set_slew_rate(u16::MAX);
        servo.apply(&cmd).unwrap();
        assert_eq!(servo.last_pulses()[0], 2500);

        // Center all should return to 1500 immediately
        servo.set_slew_rate(50);
        servo.center_all().unwrap();
        assert_eq!(servo.last_pulses()[0], 1500);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn slew_limit_output_bounded(current in 500u16..2500, target in 500u16..2500, max_step in 1u16..500) {
            let result = slew_limit(current, target, max_step);
            // Result must be between current and target (inclusive)
            let lo = current.min(target);
            let hi = current.max(target);
            prop_assert!(result >= lo, "result {} < min(current={}, target={})", result, current, target);
            prop_assert!(result <= hi, "result {} > max(current={}, target={})", result, current, target);
        }

        #[test]
        fn slew_limit_converges(current in 500u16..2500, target in 500u16..2500) {
            // With max_step = u16::MAX, result should equal target
            let result = slew_limit(current, target, u16::MAX);
            prop_assert_eq!(result, target);
        }

        #[test]
        fn slew_limit_max_step_respected(current in 500u16..2500, target in 500u16..2500, max_step in 1u16..500) {
            let result = slew_limit(current, target, max_step);
            let delta = if result > current {
                result - current
            } else {
                current - result
            };
            prop_assert!(delta <= max_step, "delta {} > max_step {}", delta, max_step);
        }
    }
}
