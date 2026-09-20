// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Servo output: `HumanoidCommand` → PCA9685 batch write with slew-rate limiting.
//!
//! [`ServoOutput`] bridges the legacy 21-value command path to two PCA9685
//! boards. The two-board write is explicitly **not atomic**: one board can
//! accept a write before the other reports an I2C error. [`ServoActuationReceipt`]
//! preserves that execution cut instead of flattening it into `Result<(), _>`.
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

use crate::calibration::{CalibrationProfile, JointCalibration};
use crate::error::{HalError, HalResult};
use crate::pca9685::{CHANNELS, Pca9685};

// ============================================================================
// ACTUATION RECEIPTS
// ============================================================================

/// Evidence available for one physical controller endpoint after an actuation
/// attempt.
///
/// `InDoubt` is deliberate: an I2C error does not prove that the controller
/// accepted no bytes before the failure became visible to the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointActuationDisposition {
    /// No write was attempted for this endpoint.
    NotAttempted,
    /// The host-side I2C write returned success.
    ///
    /// This proves controller-write acceptance only, not physical joint motion.
    WriteAccepted,
    /// A write was attempted but its controller-side effect is not known.
    InDoubt,
}

/// Overall disposition of one logical servo command attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActuationDisposition {
    /// Servo output was disabled, so no physical write was attempted.
    DisabledNoOp,
    /// Command validation/conversion failed before any physical write.
    NotDispatched,
    /// Every endpoint write returned success.
    Completed,
    /// At least one endpoint is known accepted while another is not proven
    /// accepted.
    Partial,
    /// No endpoint is proven accepted, but at least one attempted write has an
    /// uncertain controller-side effect.
    InDoubt,
}

/// Evidence for one logical servo command attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServoActuationReceipt {
    /// Monotonic sequence assigned only to enabled command attempts.
    pub command_sequence: Option<u64>,
    pub board0: EndpointActuationDisposition,
    pub board1: EndpointActuationDisposition,
    pub disposition: ActuationDisposition,
}

/// Typed failure that retains the actuation receipt alongside the underlying
/// HAL error.
#[derive(Debug)]
pub struct ServoActuationFailure {
    pub receipt: ServoActuationReceipt,
    pub error: HalError,
}

impl std::fmt::Display for ServoActuationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "servo actuation {:?} sequence {:?} (board0={:?}, board1={:?}): {}",
            self.receipt.disposition,
            self.receipt.command_sequence,
            self.receipt.board0,
            self.receipt.board1,
            self.error
        )
    }
}

impl std::error::Error for ServoActuationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Compatibility conversion for callers that still expose `HalResult`.
///
/// The typed receipt remains available from `ServoOutput::last_actuation_receipt`
/// and the servo fault latch; this conversion exists so the current runtime can
/// stop on the error without pretending it was an ordinary atomic failure.
impl From<ServoActuationFailure> for HalError {
    fn from(failure: ServoActuationFailure) -> Self {
        HalError::Safety(failure.to_string())
    }
}

// ============================================================================
// SERVO OUTPUT
// ============================================================================

/// Maximum slew rate in µs per tick (limits how fast servos move per update).
const DEFAULT_SLEW_RATE_US: u16 = 100;

/// Servo output controller for 21-joint humanoid.
///
/// Owns two PCA9685 boards and a calibration profile. Each `apply()` call:
/// 1. Converts the legacy command to pulse widths via calibration
/// 2. Applies slew-rate limiting
/// 3. Batch-writes board 0 then board 1
/// 4. Returns endpoint-specific execution evidence
///
/// A `Partial` or `InDoubt` attempt latches a fault and disables further
/// discretionary writes. A confirmed shutdown does not erase that history or
/// clear the latch; reconstruction/reconciliation is required before reuse.
pub struct ServoOutput<I> {
    board0: Pca9685<I>,
    board1: Pca9685<I>,
    calibration: CalibrationProfile,
    /// Last pulse widths supported by successful host-side write evidence.
    last_pulses: [u16; NUM_ACTUATORS],
    /// Maximum change in µs per update cycle.
    slew_rate_us: u16,
    /// Whether both PWM boards completed initialization.
    initialized: bool,
    /// Whether ordinary servo output is enabled.
    enabled: bool,
    /// Whether the most recent shutdown was confirmed on both PWM boards.
    shutdown_verified: bool,
    /// Next sequence allocated to an enabled logical command attempt.
    next_command_sequence: u64,
    /// Most recent command-attempt receipt, including failures/no-ops.
    last_actuation_receipt: Option<ServoActuationReceipt>,
    /// Latched consequential write fault. Never cleared implicitly.
    fault_latched: Option<ActuationDisposition>,
    /// Exact receipt that caused the consequential fault latch. Unlike
    /// `last_actuation_receipt`, this is not overwritten by later rejected
    /// commands.
    latched_fault_receipt: Option<ServoActuationReceipt>,
}

impl<I: I2c> ServoOutput<I> {
    /// Create a new servo output with two I2C buses (or the same bus) and calibration.
    ///
    /// - `bus0`: I2C bus for board 0 (address 0x40, joints 0–15)
    /// - `bus1`: I2C bus for board 1 (address 0x41, joints 16–20)
    pub fn new(bus0: I, bus1: I, calibration: CalibrationProfile) -> Self {
        let center = calibration
            .joints
            .first()
            .map(JointCalibration::center_pulse_us)
            .unwrap_or(1500);
        Self {
            board0: Pca9685::new(bus0, 0x40),
            board1: Pca9685::new(bus1, 0x41),
            calibration,
            last_pulses: [center; NUM_ACTUATORS],
            slew_rate_us: DEFAULT_SLEW_RATE_US,
            initialized: false,
            enabled: false,
            shutdown_verified: true,
            next_command_sequence: 1,
            last_actuation_receipt: None,
            fault_latched: None,
            latched_fault_receipt: None,
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
        if let Some(disposition) = self.fault_latched {
            return Err(HalError::Safety(format!(
                "servo actuation fault {disposition:?} is latched; reconstruct/reconcile before initialization"
            )));
        }
        self.calibration.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(HalError::Safety(format!(
                "invalid servo PWM frequency: {frequency_hz}"
            )));
        }
        let period_us = 1_000_000.0 / frequency_hz;
        for (index, joint) in self.calibration.joints.iter().enumerate() {
            if joint.pulse_max_us as f64 >= period_us {
                return Err(HalError::Calibration(format!(
                    "joint {index} ({}) pulse max {} µs does not fit PWM period {:.1} µs",
                    joint.name, joint.pulse_max_us, period_us
                )));
            }
        }

        self.initialized = false;
        self.enabled = false;
        self.shutdown_verified = false;

        let init_result = (|| -> HalResult<()> {
            self.board0.init(frequency_hz)?;
            self.board1.init(frequency_hz)?;
            self.board0.all_off()?;
            self.board1.all_off()?;
            Ok(())
        })();

        if let Err(init_error) = init_result {
            let board0 = self.board0.all_off();
            let board1 = self.board1.all_off();
            self.shutdown_verified = board0.is_ok() && board1.is_ok();
            if !self.shutdown_verified {
                self.fault_latched = Some(ActuationDisposition::InDoubt);
            }
            return match (board0, board1) {
                (Ok(()), Ok(())) => Err(init_error),
                (Err(shutdown_error), Ok(())) | (Ok(()), Err(shutdown_error)) => {
                    Err(HalError::Safety(format!(
                        "servo initialization failed: {init_error}; shutdown failed: {shutdown_error}"
                    )))
                }
                (Err(e0), Err(e1)) => Err(HalError::Safety(format!(
                    "servo initialization failed: {init_error}; both shutdowns failed: board0={e0}; board1={e1}"
                ))),
            };
        }

        self.initialized = true;
        self.shutdown_verified = true;
        debug!(
            "servo output initialized and verified off at {}Hz",
            frequency_hz
        );
        Ok(())
    }

    /// Enable servo output (allows `apply()` to write to hardware).
    pub fn enable(&mut self) -> HalResult<()> {
        if let Some(disposition) = self.fault_latched {
            return Err(HalError::Safety(format!(
                "servo actuation fault {disposition:?} is latched; ordinary output cannot resume"
            )));
        }
        if !self.initialized {
            return Err(HalError::Safety(
                "servo output cannot be enabled before initialization".to_string(),
            ));
        }
        if !self.shutdown_verified {
            return Err(HalError::Safety(
                "servo output cannot be enabled from an unverified shutdown state".to_string(),
            ));
        }
        self.calibration.validate()?;
        self.enabled = true;
        self.shutdown_verified = false;
        debug!("servo output enabled");
        Ok(())
    }

    /// Disable servo output and turn off all channels.
    ///
    /// Both boards are attempted even if the first write fails. A failed
    /// shutdown is recorded as unverified so health reporting cannot mistake
    /// a requested shutdown for confirmed de-energization. A successful
    /// shutdown does not clear a prior actuation fault latch.
    pub fn disable(&mut self) -> HalResult<()> {
        let board0 = self.board0.all_off();
        let board1 = self.board1.all_off();
        self.enabled = false;
        self.shutdown_verified = board0.is_ok() && board1.is_ok();
        if !self.shutdown_verified {
            self.fault_latched = Some(ActuationDisposition::InDoubt);
        }

        match (board0, board1) {
            (Ok(()), Ok(())) => {
                debug!("servo output disabled");
                Ok(())
            }
            (Err(e), Ok(())) => Err(e),
            (Ok(()), Err(e)) => Err(e),
            (Err(e0), Err(e1)) => Err(crate::error::HalError::Safety(format!(
                "failed to disable both PWM boards: board0={e0}; board1={e1}"
            ))),
        }
    }

    /// Whether both PWM boards completed initialization.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Whether ordinary output is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Whether both PWM boards confirmed the most recent all-off request.
    pub fn shutdown_verified(&self) -> bool {
        self.shutdown_verified
    }

    /// Consequential write fault currently latched by the servo boundary.
    pub fn fault_latched(&self) -> Option<ActuationDisposition> {
        self.fault_latched
    }

    /// Exact consequential receipt that caused the current fault latch.
    pub fn latched_fault_receipt(&self) -> Option<&ServoActuationReceipt> {
        self.latched_fault_receipt.as_ref()
    }

    /// Most recent command-attempt receipt.
    pub fn last_actuation_receipt(&self) -> Option<&ServoActuationReceipt> {
        self.last_actuation_receipt.as_ref()
    }

    /// Set the maximum slew rate (µs per update tick).
    pub fn set_slew_rate(&mut self, us_per_tick: u16) {
        self.slew_rate_us = us_per_tick;
    }

    /// Set the calibration profile.
    pub fn set_calibration(&mut self, cal: CalibrationProfile) -> HalResult<()> {
        if self.enabled {
            return Err(HalError::Safety(
                "cannot replace calibration while servo output is enabled".to_string(),
            ));
        }
        if let Some(disposition) = self.fault_latched {
            return Err(HalError::Safety(format!(
                "cannot replace calibration while servo actuation fault {disposition:?} is latched"
            )));
        }
        cal.validate()?;
        self.calibration = cal;
        Ok(())
    }

    /// Apply a legacy `HumanoidCommand` to the servos and return endpoint-specific
    /// execution evidence.
    ///
    /// This method does not claim physical motion. `WriteAccepted` means only
    /// that the host-side I2C transaction returned success.
    pub fn apply(
        &mut self,
        command: &HumanoidCommand,
    ) -> Result<ServoActuationReceipt, ServoActuationFailure> {
        if let Some(disposition) = self.fault_latched {
            let receipt = ServoActuationReceipt {
                command_sequence: None,
                board0: EndpointActuationDisposition::NotAttempted,
                board1: EndpointActuationDisposition::NotAttempted,
                disposition: ActuationDisposition::NotDispatched,
            };
            self.last_actuation_receipt = Some(receipt);
            return Err(ServoActuationFailure {
                receipt,
                error: HalError::Safety(format!(
                    "servo actuation fault {disposition:?} is latched; ordinary command rejected"
                )),
            });
        }

        if !self.enabled {
            let receipt = ServoActuationReceipt {
                command_sequence: None,
                board0: EndpointActuationDisposition::NotAttempted,
                board1: EndpointActuationDisposition::NotAttempted,
                disposition: ActuationDisposition::DisabledNoOp,
            };
            self.last_actuation_receipt = Some(receipt);
            return Ok(receipt);
        }

        if self.next_command_sequence == u64::MAX {
            let receipt = ServoActuationReceipt {
                command_sequence: None,
                board0: EndpointActuationDisposition::NotAttempted,
                board1: EndpointActuationDisposition::NotAttempted,
                disposition: ActuationDisposition::NotDispatched,
            };
            self.last_actuation_receipt = Some(receipt);
            return Err(ServoActuationFailure {
                receipt,
                error: HalError::Safety("servo command sequence exhausted".to_string()),
            });
        }
        let command_sequence = self.next_command_sequence;
        self.next_command_sequence += 1;

        // 1. Convert command → target pulse widths. A conversion/validation
        // failure occurs before any physical write and is therefore
        // NotDispatched rather than InDoubt.
        let targets = match self.calibration.torques_to_pulses(&command.torques) {
            Ok(targets) => targets,
            Err(error) => {
                let receipt = ServoActuationReceipt {
                    command_sequence: Some(command_sequence),
                    board0: EndpointActuationDisposition::NotAttempted,
                    board1: EndpointActuationDisposition::NotAttempted,
                    disposition: ActuationDisposition::NotDispatched,
                };
                self.last_actuation_receipt = Some(receipt);
                return Err(ServoActuationFailure { receipt, error });
            }
        };

        // 2. Slew-rate limiting.
        let mut pulses = [0u16; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS {
            pulses[i] = slew_limit(self.last_pulses[i], targets[i], self.slew_rate_us);
        }

        // 3. Board 0 transaction. An I2C error is InDoubt rather than proof of
        // non-dispatch. Do not update shadow state without successful return.
        if let Err(error) = self.board0.set_pulse_batch(0, &pulses[..CHANNELS]) {
            let receipt = ServoActuationReceipt {
                command_sequence: Some(command_sequence),
                board0: EndpointActuationDisposition::InDoubt,
                board1: EndpointActuationDisposition::NotAttempted,
                disposition: ActuationDisposition::InDoubt,
            };
            self.last_actuation_receipt = Some(receipt);
            self.latch_actuation_fault(receipt);
            return Err(ServoActuationFailure { receipt, error });
        }
        self.last_pulses[..CHANNELS].copy_from_slice(&pulses[..CHANNELS]);

        // 4. Board 1 transaction. Board 0 is already known accepted; therefore
        // a board 1 I2C error is a genuine partial execution cut.
        if let Err(error) = self.board1.set_pulse_batch(0, &pulses[CHANNELS..]) {
            let receipt = ServoActuationReceipt {
                command_sequence: Some(command_sequence),
                board0: EndpointActuationDisposition::WriteAccepted,
                board1: EndpointActuationDisposition::InDoubt,
                disposition: ActuationDisposition::Partial,
            };
            self.last_actuation_receipt = Some(receipt);
            self.latch_actuation_fault(receipt);
            return Err(ServoActuationFailure { receipt, error });
        }
        self.last_pulses[CHANNELS..].copy_from_slice(&pulses[CHANNELS..]);

        let receipt = ServoActuationReceipt {
            command_sequence: Some(command_sequence),
            board0: EndpointActuationDisposition::WriteAccepted,
            board1: EndpointActuationDisposition::WriteAccepted,
            disposition: ActuationDisposition::Completed,
        };
        self.last_actuation_receipt = Some(receipt);
        Ok(receipt)
    }

    fn latch_actuation_fault(&mut self, receipt: ServoActuationReceipt) {
        debug_assert!(matches!(
            receipt.disposition,
            ActuationDisposition::Partial | ActuationDisposition::InDoubt
        ));
        self.fault_latched = Some(receipt.disposition);
        self.latched_fault_receipt = Some(receipt);
        self.enabled = false;
        self.shutdown_verified = false;
    }

    /// Get the last pulse widths supported by successful controller-write
    /// evidence.
    pub fn last_pulses(&self) -> &[u16; NUM_ACTUATORS] {
        &self.last_pulses
    }

    /// Move all servos to their center (neutral) position.
    ///
    /// Legacy behavior: temporarily bypasses normal slew-rate limiting. This is
    /// not a generally qualified safety reflex and remains tracked separately.
    pub fn center_all(&mut self) -> HalResult<()> {
        let zero = HumanoidCommand::zero();
        let saved = self.slew_rate_us;
        self.slew_rate_us = u16::MAX;
        let result = self.apply(&zero).map(|_| ()).map_err(HalError::from);
        self.slew_rate_us = saved;
        result
    }

    /// Get a reference to the calibration profile.
    pub fn calibration(&self) -> &CalibrationProfile {
        &self.calibration
    }

    /// Read the PWM OFF registers for all 21 channels across both boards.
    ///
    /// This confirms only what the PCA9685 latched. It does **not** measure
    /// physical servo or joint position; encoder or potentiometer feedback is
    /// required for that claim. This performs 21 I2C reads and is not intended
    /// for per-tick use.
    pub fn read_pwm_registers(&mut self) -> HalResult<[u16; NUM_ACTUATORS]> {
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

    /// Compare commanded pulses against PCA9685 register readback.
    ///
    /// Returns `(joint_index, commanded_pulse_us, latched_off_count)` entries.
    /// An empty vector proves register agreement only, not physical motion.
    pub fn verify_pwm_latch(&mut self) -> HalResult<Vec<(usize, u16, u16)>> {
        let actual = self.read_pwm_registers()?;
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

    /// Backward-compatible alias for [`Self::read_pwm_registers`].
    #[deprecated(note = "PCA9685 readback is not physical position; use read_pwm_registers")]
    pub fn read_positions(&mut self) -> HalResult<[u16; NUM_ACTUATORS]> {
        self.read_pwm_registers()
    }

    /// Backward-compatible alias for [`Self::verify_pwm_latch`].
    #[deprecated(note = "PCA9685 readback verifies only the PWM latch; use verify_pwm_latch")]
    pub fn verify_positions(&mut self) -> HalResult<Vec<(usize, u16, u16)>> {
        self.verify_pwm_latch()
    }
}

// ============================================================================
// SHARED-BUS CONSTRUCTORS
// ============================================================================

impl<'a, I: I2c> ServoOutput<RefCellDevice<'a, I>> {
    /// Create a servo output sharing a single I2C bus via `RefCell` (single-threaded).
    pub fn new_shared(bus: &'a RefCell<I>, calibration: CalibrationProfile) -> Self {
        let center = calibration
            .joints
            .first()
            .map(JointCalibration::center_pulse_us)
            .unwrap_or(1500);
        Self {
            board0: Pca9685::new(RefCellDevice::new(bus), 0x40),
            board1: Pca9685::new(RefCellDevice::new(bus), 0x41),
            calibration,
            last_pulses: [center; NUM_ACTUATORS],
            slew_rate_us: DEFAULT_SLEW_RATE_US,
            initialized: false,
            enabled: false,
            shutdown_verified: true,
            next_command_sequence: 1,
            last_actuation_receipt: None,
            fault_latched: None,
            latched_fault_receipt: None,
        }
    }
}

impl<'a, I: I2c + Send> ServoOutput<MutexDevice<'a, I>> {
    /// Create a servo output sharing a single I2C bus via `Mutex` (thread-safe).
    pub fn new_shared_mutex(bus: &'a Mutex<I>, calibration: CalibrationProfile) -> Self {
        let center = calibration
            .joints
            .first()
            .map(JointCalibration::center_pulse_us)
            .unwrap_or(1500);
        Self {
            board0: Pca9685::new(MutexDevice::new(bus), 0x40),
            board1: Pca9685::new(MutexDevice::new(bus), 0x41),
            calibration,
            last_pulses: [center; NUM_ACTUATORS],
            slew_rate_us: DEFAULT_SLEW_RATE_US,
            initialized: false,
            enabled: false,
            shutdown_verified: true,
            next_command_sequence: 1,
            last_actuation_receipt: None,
            fault_latched: None,
            latched_fault_receipt: None,
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
    use crate::mock::{MockI2cBus, MockI2cError};
    use embedded_hal::i2c::{ErrorType, Operation};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    fn make_servo() -> ServoOutput<MockI2cBus> {
        let bus0 = MockI2cBus::new();
        let bus1 = MockI2cBus::new();
        let cal = CalibrationProfile::default_21();
        ServoOutput::new(bus0, bus1, cal)
    }

    struct SwitchableFailBus {
        inner: MockI2cBus,
        fail_next: Arc<AtomicBool>,
    }

    impl SwitchableFailBus {
        fn new(fail_next: Arc<AtomicBool>) -> Self {
            Self {
                inner: MockI2cBus::new(),
                fail_next,
            }
        }
    }

    impl ErrorType for SwitchableFailBus {
        type Error = MockI2cError;
    }

    impl I2c for SwitchableFailBus {
        fn transaction(
            &mut self,
            address: u8,
            operations: &mut [Operation<'_>],
        ) -> Result<(), Self::Error> {
            if self.fail_next.swap(false, Ordering::SeqCst) {
                return Err(MockI2cError {
                    kind: embedded_hal::i2c::ErrorKind::Other,
                });
            }
            self.inner.transaction(address, operations)
        }
    }

    fn make_switchable_servo(
    ) -> (
        ServoOutput<SwitchableFailBus>,
        Arc<AtomicBool>,
        Arc<AtomicBool>,
    ) {
        let fail0 = Arc::new(AtomicBool::new(false));
        let fail1 = Arc::new(AtomicBool::new(false));
        let cal = CalibrationProfile::default_21();
        (
            ServoOutput::new(
                SwitchableFailBus::new(fail0.clone()),
                SwitchableFailBus::new(fail1.clone()),
                cal,
            ),
            fail0,
            fail1,
        )
    }

    #[test]
    fn test_servo_disabled_noop_is_typed() {
        let mut servo = make_servo();
        assert!(!servo.is_enabled());
        let receipt = servo.apply(&HumanoidCommand::zero()).unwrap();
        assert_eq!(receipt.disposition, ActuationDisposition::DisabledNoOp);
        assert_eq!(receipt.command_sequence, None);
        assert_eq!(receipt.board0, EndpointActuationDisposition::NotAttempted);
        assert_eq!(receipt.board1, EndpointActuationDisposition::NotAttempted);
    }

    #[test]
    fn test_servo_enable_requires_initialization() {
        let mut servo = make_servo();
        assert!(servo.enable().is_err());
        assert!(!servo.is_enabled());
    }

    #[test]
    fn test_calibration_cannot_change_while_enabled() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        assert!(
            servo
                .set_calibration(CalibrationProfile::default_21())
                .is_err()
        );
    }

    #[test]
    fn test_servo_enable_apply_returns_completed_receipt() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        assert!(servo.is_enabled());
        assert!(!servo.shutdown_verified());

        let receipt = servo.apply(&HumanoidCommand::zero()).unwrap();
        assert_eq!(receipt.command_sequence, Some(1));
        assert_eq!(receipt.disposition, ActuationDisposition::Completed);
        assert_eq!(receipt.board0, EndpointActuationDisposition::WriteAccepted);
        assert_eq!(receipt.board1, EndpointActuationDisposition::WriteAccepted);

        for &p in servo.last_pulses() {
            assert_eq!(p, 1500);
        }
    }

    #[test]
    fn board0_error_is_indoubt_and_board1_not_attempted() {
        let (mut servo, fail0, _fail1) = make_switchable_servo();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        servo.set_slew_rate(u16::MAX);
        fail0.store(true, Ordering::SeqCst);

        let mut cmd = HumanoidCommand::zero();
        cmd.torques[0] = 1.0;
        cmd.torques[16] = 1.0;
        let failure = servo.apply(&cmd).unwrap_err();

        assert_eq!(failure.receipt.disposition, ActuationDisposition::InDoubt);
        assert_eq!(
            failure.receipt.board0,
            EndpointActuationDisposition::InDoubt
        );
        assert_eq!(
            failure.receipt.board1,
            EndpointActuationDisposition::NotAttempted
        );
        assert_eq!(servo.last_pulses()[0], 1500);
        assert_eq!(servo.last_pulses()[16], 1500);
        assert_eq!(servo.fault_latched(), Some(ActuationDisposition::InDoubt));
        assert_eq!(
            servo.latched_fault_receipt().unwrap().command_sequence,
            Some(1)
        );
        assert!(!servo.is_enabled());
        assert!(!servo.shutdown_verified());
    }

    #[test]
    fn board1_error_preserves_partial_board0_success_and_latches_fault() {
        let (mut servo, _fail0, fail1) = make_switchable_servo();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        servo.set_slew_rate(u16::MAX);
        fail1.store(true, Ordering::SeqCst);

        let mut cmd = HumanoidCommand::zero();
        cmd.torques[0] = 1.0;
        cmd.torques[16] = 1.0;
        let failure = servo.apply(&cmd).unwrap_err();

        assert_eq!(failure.receipt.disposition, ActuationDisposition::Partial);
        assert_eq!(
            failure.receipt.board0,
            EndpointActuationDisposition::WriteAccepted
        );
        assert_eq!(
            failure.receipt.board1,
            EndpointActuationDisposition::InDoubt
        );
        assert_eq!(servo.last_pulses()[0], 2500);
        assert_eq!(servo.last_pulses()[16], 1500);
        assert_eq!(servo.fault_latched(), Some(ActuationDisposition::Partial));
        assert!(!servo.is_enabled());
        assert!(!servo.shutdown_verified());

        let second = servo.apply(&HumanoidCommand::zero()).unwrap_err();
        assert_eq!(
            second.receipt.disposition,
            ActuationDisposition::NotDispatched
        );
        assert_eq!(second.receipt.command_sequence, None);
        let latched = servo.latched_fault_receipt().unwrap();
        assert_eq!(latched.command_sequence, Some(1));
        assert_eq!(latched.disposition, ActuationDisposition::Partial);
        assert_eq!(latched.board0, EndpointActuationDisposition::WriteAccepted);
        assert_eq!(latched.board1, EndpointActuationDisposition::InDoubt);
    }

    #[test]
    fn verified_shutdown_does_not_erase_partial_history_or_clear_latch() {
        let (mut servo, _fail0, fail1) = make_switchable_servo();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        fail1.store(true, Ordering::SeqCst);
        let failure = servo.apply(&HumanoidCommand::zero()).unwrap_err();
        assert_eq!(failure.receipt.disposition, ActuationDisposition::Partial);

        servo.disable().unwrap();
        assert!(servo.shutdown_verified());
        assert_eq!(servo.fault_latched(), Some(ActuationDisposition::Partial));
        assert_eq!(
            servo.latched_fault_receipt().unwrap().disposition,
            ActuationDisposition::Partial
        );
        assert!(servo.enable().is_err());
    }

    #[test]
    fn test_servo_disable_turns_off() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        servo.disable().unwrap();
        assert!(!servo.is_enabled());
        assert!(servo.shutdown_verified());
    }

    #[test]
    fn test_slew_rate_limiting() {
        assert_eq!(slew_limit(1500, 2500, 100), 1600);
        assert_eq!(slew_limit(1500, 500, 100), 1400);
        assert_eq!(slew_limit(1500, 1550, 100), 1550);
        assert_eq!(slew_limit(1500, 1500, 100), 1500);
    }

    #[test]
    fn test_servo_slew_rate_applied() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        servo.set_slew_rate(50);

        let mut cmd = HumanoidCommand::zero();
        cmd.torques[0] = 1.0;
        servo.apply(&cmd).unwrap();
        assert_eq!(servo.last_pulses()[0], 1550);

        servo.apply(&cmd).unwrap();
        assert_eq!(servo.last_pulses()[0], 1600);
    }

    #[test]
    fn test_servo_new_shared_refcell() {
        let bus = RefCell::new(MockI2cBus::new());
        let cal = CalibrationProfile::default_21();
        let mut servo = ServoOutput::new_shared(&bus, cal);
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        servo.apply(&HumanoidCommand::zero()).unwrap();
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
        servo.enable().unwrap();
        servo.apply(&HumanoidCommand::zero()).unwrap();
        for &p in servo.last_pulses() {
            assert_eq!(p, 1500);
        }
    }

    #[test]
    fn test_read_pwm_registers_returns_21_elements() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        let positions = servo.read_pwm_registers().unwrap();
        assert_eq!(positions.len(), NUM_ACTUATORS);
    }

    #[test]
    fn test_verify_pwm_latch_detects_mismatches() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();

        servo.apply(&HumanoidCommand::zero()).unwrap();
        assert_eq!(servo.last_pulses()[0], 1500);

        let mismatches = servo.verify_pwm_latch().unwrap();
        assert!(!mismatches.is_empty(), "should detect mismatches");
        assert_eq!(mismatches.len(), NUM_ACTUATORS);
        assert_eq!(mismatches[0].1, 1500);
        assert_eq!(mismatches[0].2, 0);
    }

    #[test]
    fn test_center_all() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();

        let mut cmd = HumanoidCommand::zero();
        cmd.torques[0] = 1.0;
        servo.set_slew_rate(u16::MAX);
        servo.apply(&cmd).unwrap();
        assert_eq!(servo.last_pulses()[0], 2500);

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
            let lo = current.min(target);
            let hi = current.max(target);
            prop_assert!(result >= lo, "result {} < min(current={}, target={})", result, current, target);
            prop_assert!(result <= hi, "result {} > max(current={}, target={})", result, current, target);
        }

        #[test]
        fn slew_limit_converges(current in 500u16..2500, target in 500u16..2500) {
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
