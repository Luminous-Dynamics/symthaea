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
//! Startup pulse values are command-reference state only. They are initialized
//! independently from each joint's calibration and must not be interpreted as
//! observed physical joint positions.
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
use std::time::Duration;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointActuationDisposition {
    NotAttempted,
    /// Host-side I2C write returned success. This is not physical-motion evidence.
    WriteAccepted,
    /// A write was attempted but its controller-side effect is not known.
    InDoubt,
}

/// Overall disposition of one logical servo command attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActuationDisposition {
    DisabledNoOp,
    NotDispatched,
    Completed,
    Partial,
    InDoubt,
}

/// Evidence for one logical servo command attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServoActuationReceipt {
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

impl From<ServoActuationFailure> for HalError {
    fn from(failure: ServoActuationFailure) -> Self {
        HalError::Safety(failure.to_string())
    }
}

// ============================================================================
// COMMAND-REFERENCE KNOWLEDGE
// ============================================================================

/// What the HAL actually knows about one pulse command reference.
///
/// Neither variant claims physical joint position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandReferenceKnowledge {
    /// Synthetic/configured baseline derived from this actuator's calibration.
    ConfiguredCalibrationReference,
    /// A host-side controller write for this actuator's board returned success.
    WriteAccepted,
}

// ============================================================================
// SERVO OUTPUT
// ============================================================================

/// Legacy maximum slew step in µs per call. Retained until HalRuntime migrates
/// fully onto [`ServoOutput::apply_with_elapsed`].
const DEFAULT_SLEW_RATE_US: u16 = 100;
/// Transitional PWM-level velocity limit equivalent to 100 µs per 20 ms.
const DEFAULT_MAX_PULSE_VELOCITY_US_PER_SECOND: f64 = 5_000.0;
/// Fail closed on unexpectedly large elapsed intervals.
const DEFAULT_MAX_SLEW_GAP: Duration = Duration::from_millis(100);

pub struct ServoOutput<I> {
    board0: Pca9685<I>,
    board1: Pca9685<I>,
    calibration: CalibrationProfile,
    /// Last controller-command reference supported by the knowledge array below.
    last_pulses: [u16; NUM_ACTUATORS],
    command_reference_knowledge: [CommandReferenceKnowledge; NUM_ACTUATORS],
    /// Legacy µs-per-call limit. Runtime migration will remove this path.
    slew_rate_us: u16,
    /// Explicit elapsed-time PWM velocity limit.
    max_pulse_velocity_us_per_second: f64,
    /// Maximum accepted monotonic elapsed interval for one slew calculation.
    max_slew_gap: Duration,
    initialized: bool,
    enabled: bool,
    shutdown_verified: bool,
    next_command_sequence: u64,
    last_actuation_receipt: Option<ServoActuationReceipt>,
    fault_latched: Option<ActuationDisposition>,
    latched_fault_receipt: Option<ServoActuationReceipt>,
}

impl<I: I2c> ServoOutput<I> {
    pub fn new(bus0: I, bus1: I, calibration: CalibrationProfile) -> Self {
        let last_pulses = configured_reference_pulses(&calibration);
        Self {
            board0: Pca9685::new(bus0, 0x40),
            board1: Pca9685::new(bus1, 0x41),
            calibration,
            last_pulses,
            command_reference_knowledge: [
                CommandReferenceKnowledge::ConfiguredCalibrationReference;
                NUM_ACTUATORS
            ],
            slew_rate_us: DEFAULT_SLEW_RATE_US,
            max_pulse_velocity_us_per_second: DEFAULT_MAX_PULSE_VELOCITY_US_PER_SECOND,
            max_slew_gap: DEFAULT_MAX_SLEW_GAP,
            initialized: false,
            enabled: false,
            shutdown_verified: true,
            next_command_sequence: 1,
            last_actuation_receipt: None,
            fault_latched: None,
            latched_fault_receipt: None,
        }
    }

    pub fn board0(&self) -> &Pca9685<I> {
        &self.board0
    }

    pub fn board1(&self) -> &Pca9685<I> {
        &self.board1
    }

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
            (Err(e0), Err(e1)) => Err(HalError::Safety(format!(
                "failed to disable both PWM boards: board0={e0}; board1={e1}"
            ))),
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn shutdown_verified(&self) -> bool {
        self.shutdown_verified
    }

    pub fn fault_latched(&self) -> Option<ActuationDisposition> {
        self.fault_latched
    }

    pub fn latched_fault_receipt(&self) -> Option<&ServoActuationReceipt> {
        self.latched_fault_receipt.as_ref()
    }

    pub fn last_actuation_receipt(&self) -> Option<&ServoActuationReceipt> {
        self.last_actuation_receipt.as_ref()
    }

    /// Command-reference provenance for every actuator. This is deliberately
    /// not called physical-state knowledge.
    pub fn command_reference_knowledge(
        &self,
    ) -> &[CommandReferenceKnowledge; NUM_ACTUATORS] {
        &self.command_reference_knowledge
    }

    /// Transitional legacy setter. This still means µs per call, not per second.
    pub fn set_slew_rate(&mut self, us_per_tick: u16) {
        self.slew_rate_us = us_per_tick;
    }

    /// Configure the explicit PWM velocity limit used by `apply_with_elapsed`.
    pub fn set_max_pulse_velocity_us_per_second(&mut self, rate: f64) -> HalResult<()> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err(HalError::Safety(format!(
                "pulse velocity must be finite and positive, got {rate}"
            )));
        }
        self.max_pulse_velocity_us_per_second = rate;
        Ok(())
    }

    pub fn max_pulse_velocity_us_per_second(&self) -> f64 {
        self.max_pulse_velocity_us_per_second
    }

    /// Configure the maximum accepted elapsed interval for one slew step.
    pub fn set_max_slew_gap(&mut self, max_gap: Duration) -> HalResult<()> {
        if max_gap.is_zero() {
            return Err(HalError::Safety(
                "maximum slew gap must be positive".to_string(),
            ));
        }
        self.max_slew_gap = max_gap;
        Ok(())
    }

    pub fn max_slew_gap(&self) -> Duration {
        self.max_slew_gap
    }

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
        self.last_pulses = configured_reference_pulses(&cal);
        self.command_reference_knowledge = [
            CommandReferenceKnowledge::ConfiguredCalibrationReference;
            NUM_ACTUATORS
        ];
        self.calibration = cal;
        Ok(())
    }

    /// Legacy per-call slew path retained only until runtime migration.
    pub fn apply(
        &mut self,
        command: &HumanoidCommand,
    ) -> Result<ServoActuationReceipt, ServoActuationFailure> {
        self.apply_with_max_step(command, self.slew_rate_us)
    }

    /// Apply a command using an explicit elapsed monotonic duration.
    ///
    /// The caller owns the clock theorem. This API consumes only a `Duration`,
    /// never UTC/wall-clock time or an implicit tick count.
    pub fn apply_with_elapsed(
        &mut self,
        command: &HumanoidCommand,
        elapsed: Duration,
    ) -> Result<ServoActuationReceipt, ServoActuationFailure> {
        // Preserve disabled/fault-latched semantics before timing validation.
        if !self.enabled || self.fault_latched.is_some() {
            return self.apply(command);
        }

        let max_step = match pulse_step_for_elapsed(
            self.max_pulse_velocity_us_per_second,
            elapsed,
            self.max_slew_gap,
        ) {
            Ok(max_step) => max_step,
            Err(error) => {
                let receipt = ServoActuationReceipt {
                    command_sequence: None,
                    board0: EndpointActuationDisposition::NotAttempted,
                    board1: EndpointActuationDisposition::NotAttempted,
                    disposition: ActuationDisposition::NotDispatched,
                };
                self.last_actuation_receipt = Some(receipt);
                return Err(ServoActuationFailure { receipt, error });
            }
        };

        self.apply_with_max_step(command, max_step)
    }

    fn apply_with_max_step(
        &mut self,
        command: &HumanoidCommand,
        max_step: u16,
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

        let mut pulses = [0u16; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS {
            pulses[i] = slew_limit(self.last_pulses[i], targets[i], max_step);
        }

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
        self.command_reference_knowledge[..CHANNELS]
            .fill(CommandReferenceKnowledge::WriteAccepted);

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
        self.command_reference_knowledge[CHANNELS..]
            .fill(CommandReferenceKnowledge::WriteAccepted);

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

    /// Last controller command references. These are not physical positions.
    pub fn last_pulses(&self) -> &[u16; NUM_ACTUATORS] {
        &self.last_pulses
    }

    /// Legacy immediate-centering path. Not a generally qualified safety reflex.
    pub fn center_all(&mut self) -> HalResult<()> {
        let zero = HumanoidCommand::zero();
        self.apply_with_max_step(&zero, u16::MAX)
            .map(|_| ())
            .map_err(HalError::from)
    }

    pub fn calibration(&self) -> &CalibrationProfile {
        &self.calibration
    }

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

    #[deprecated(note = "PCA9685 readback is not physical position; use read_pwm_registers")]
    pub fn read_positions(&mut self) -> HalResult<[u16; NUM_ACTUATORS]> {
        self.read_pwm_registers()
    }

    #[deprecated(note = "PCA9685 readback verifies only the PWM latch; use verify_pwm_latch")]
    pub fn verify_positions(&mut self) -> HalResult<Vec<(usize, u16, u16)>> {
        self.verify_pwm_latch()
    }
}

impl<'a, I: I2c> ServoOutput<RefCellDevice<'a, I>> {
    pub fn new_shared(bus: &'a RefCell<I>, calibration: CalibrationProfile) -> Self {
        let last_pulses = configured_reference_pulses(&calibration);
        Self {
            board0: Pca9685::new(RefCellDevice::new(bus), 0x40),
            board1: Pca9685::new(RefCellDevice::new(bus), 0x41),
            calibration,
            last_pulses,
            command_reference_knowledge: [
                CommandReferenceKnowledge::ConfiguredCalibrationReference;
                NUM_ACTUATORS
            ],
            slew_rate_us: DEFAULT_SLEW_RATE_US,
            max_pulse_velocity_us_per_second: DEFAULT_MAX_PULSE_VELOCITY_US_PER_SECOND,
            max_slew_gap: DEFAULT_MAX_SLEW_GAP,
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
    pub fn new_shared_mutex(bus: &'a Mutex<I>, calibration: CalibrationProfile) -> Self {
        let last_pulses = configured_reference_pulses(&calibration);
        Self {
            board0: Pca9685::new(MutexDevice::new(bus), 0x40),
            board1: Pca9685::new(MutexDevice::new(bus), 0x41),
            calibration,
            last_pulses,
            command_reference_knowledge: [
                CommandReferenceKnowledge::ConfiguredCalibrationReference;
                NUM_ACTUATORS
            ],
            slew_rate_us: DEFAULT_SLEW_RATE_US,
            max_pulse_velocity_us_per_second: DEFAULT_MAX_PULSE_VELOCITY_US_PER_SECOND,
            max_slew_gap: DEFAULT_MAX_SLEW_GAP,
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

fn configured_reference_pulses(
    calibration: &CalibrationProfile,
) -> [u16; NUM_ACTUATORS] {
    std::array::from_fn(|index| {
        calibration
            .joints
            .get(index)
            .map(JointCalibration::center_pulse_us)
            .unwrap_or(1500)
    })
}

fn pulse_step_for_elapsed(
    rate_us_per_second: f64,
    elapsed: Duration,
    max_gap: Duration,
) -> HalResult<u16> {
    if !rate_us_per_second.is_finite() || rate_us_per_second <= 0.0 {
        return Err(HalError::Safety(format!(
            "pulse velocity must be finite and positive, got {rate_us_per_second}"
        )));
    }
    if elapsed.is_zero() {
        return Err(HalError::Safety(
            "elapsed slew duration must be positive".to_string(),
        ));
    }
    if elapsed > max_gap {
        return Err(HalError::Safety(format!(
            "elapsed slew duration {:?} exceeds configured maximum {:?}",
            elapsed, max_gap
        )));
    }
    let allowed = rate_us_per_second * elapsed.as_secs_f64();
    if !allowed.is_finite() || allowed < 0.0 {
        return Err(HalError::Safety(
            "computed pulse slew allowance is invalid".to_string(),
        ));
    }
    Ok(allowed.floor().clamp(0.0, u16::MAX as f64) as u16)
}

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
        ServoOutput::new(
            MockI2cBus::new(),
            MockI2cBus::new(),
            CalibrationProfile::default_21(),
        )
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
        (
            ServoOutput::new(
                SwitchableFailBus::new(fail0.clone()),
                SwitchableFailBus::new(fail1.clone()),
                CalibrationProfile::default_21(),
            ),
            fail0,
            fail1,
        )
    }

    #[test]
    fn heterogeneous_calibration_centers_are_kept_per_actuator() {
        let mut cal = CalibrationProfile::default_21();
        cal.joints[0].pulse_min_us = 400;
        cal.joints[0].pulse_max_us = 1600;
        cal.joints[1].pulse_min_us = 1000;
        cal.joints[1].pulse_max_us = 2400;
        let servo = ServoOutput::new(MockI2cBus::new(), MockI2cBus::new(), cal);
        assert_eq!(servo.last_pulses()[0], 1000);
        assert_eq!(servo.last_pulses()[1], 1700);
        assert!(servo
            .command_reference_knowledge()
            .iter()
            .all(|state| *state == CommandReferenceKnowledge::ConfiguredCalibrationReference));
    }

    #[test]
    fn elapsed_time_slew_two_10ms_steps_equal_one_20ms_step() {
        let mut a = make_servo();
        let mut b = make_servo();
        for servo in [&mut a, &mut b] {
            servo.init(50.0).unwrap();
            servo.enable().unwrap();
            servo.set_max_pulse_velocity_us_per_second(5_000.0).unwrap();
        }
        let mut cmd = HumanoidCommand::zero();
        cmd.torques[0] = 1.0;

        a.apply_with_elapsed(&cmd, Duration::from_millis(10)).unwrap();
        a.apply_with_elapsed(&cmd, Duration::from_millis(10)).unwrap();
        b.apply_with_elapsed(&cmd, Duration::from_millis(20)).unwrap();

        assert_eq!(a.last_pulses()[0], 1600);
        assert_eq!(b.last_pulses()[0], 1600);
    }

    #[test]
    fn elapsed_time_slew_rejects_zero_and_oversized_gap_before_dispatch() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        let zero = servo
            .apply_with_elapsed(&HumanoidCommand::zero(), Duration::ZERO)
            .unwrap_err();
        assert_eq!(zero.receipt.disposition, ActuationDisposition::NotDispatched);
        assert_eq!(zero.receipt.command_sequence, None);

        let large = servo
            .apply_with_elapsed(&HumanoidCommand::zero(), Duration::from_millis(101))
            .unwrap_err();
        assert_eq!(large.receipt.disposition, ActuationDisposition::NotDispatched);
        assert_eq!(large.receipt.command_sequence, None);
        assert_eq!(servo.last_pulses()[0], 1500);
    }

    #[test]
    fn successful_write_promotes_command_reference_not_physical_state() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        servo
            .apply_with_elapsed(&HumanoidCommand::zero(), Duration::from_millis(20))
            .unwrap();
        assert!(servo
            .command_reference_knowledge()
            .iter()
            .all(|state| *state == CommandReferenceKnowledge::WriteAccepted));
    }

    #[test]
    fn test_servo_disabled_noop_is_typed() {
        let mut servo = make_servo();
        let receipt = servo.apply(&HumanoidCommand::zero()).unwrap();
        assert_eq!(receipt.disposition, ActuationDisposition::DisabledNoOp);
        assert_eq!(receipt.command_sequence, None);
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
        assert!(servo
            .set_calibration(CalibrationProfile::default_21())
            .is_err());
    }

    #[test]
    fn test_servo_enable_apply_returns_completed_receipt() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        let receipt = servo.apply(&HumanoidCommand::zero()).unwrap();
        assert_eq!(receipt.command_sequence, Some(1));
        assert_eq!(receipt.disposition, ActuationDisposition::Completed);
        assert_eq!(receipt.board0, EndpointActuationDisposition::WriteAccepted);
        assert_eq!(receipt.board1, EndpointActuationDisposition::WriteAccepted);
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
        assert_eq!(failure.receipt.board0, EndpointActuationDisposition::InDoubt);
        assert_eq!(failure.receipt.board1, EndpointActuationDisposition::NotAttempted);
        assert_eq!(servo.last_pulses()[0], 1500);
        assert_eq!(servo.last_pulses()[16], 1500);
        assert_eq!(servo.fault_latched(), Some(ActuationDisposition::InDoubt));
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
        assert_eq!(failure.receipt.board0, EndpointActuationDisposition::WriteAccepted);
        assert_eq!(failure.receipt.board1, EndpointActuationDisposition::InDoubt);
        assert_eq!(servo.last_pulses()[0], 2500);
        assert_eq!(servo.last_pulses()[16], 1500);
        assert_eq!(
            servo.command_reference_knowledge()[0],
            CommandReferenceKnowledge::WriteAccepted
        );
        assert_eq!(
            servo.command_reference_knowledge()[16],
            CommandReferenceKnowledge::ConfiguredCalibrationReference
        );

        let second = servo.apply(&HumanoidCommand::zero()).unwrap_err();
        assert_eq!(second.receipt.disposition, ActuationDisposition::NotDispatched);
        let latched = servo.latched_fault_receipt().unwrap();
        assert_eq!(latched.command_sequence, Some(1));
        assert_eq!(latched.disposition, ActuationDisposition::Partial);
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
        let mut servo = ServoOutput::new_shared(&bus, CalibrationProfile::default_21());
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        servo.apply(&HumanoidCommand::zero()).unwrap();
        assert!(servo.last_pulses().iter().all(|pulse| *pulse == 1500));
    }

    #[test]
    fn test_servo_new_shared_mutex() {
        let bus = Mutex::new(MockI2cBus::new());
        let mut servo = ServoOutput::new_shared_mutex(&bus, CalibrationProfile::default_21());
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        servo.apply(&HumanoidCommand::zero()).unwrap();
        assert!(servo.last_pulses().iter().all(|pulse| *pulse == 1500));
    }

    #[test]
    fn test_read_pwm_registers_returns_21_elements() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        assert_eq!(servo.read_pwm_registers().unwrap().len(), NUM_ACTUATORS);
    }

    #[test]
    fn test_verify_pwm_latch_detects_mismatches() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        servo.apply(&HumanoidCommand::zero()).unwrap();
        let mismatches = servo.verify_pwm_latch().unwrap();
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
        servo.center_all().unwrap();
        assert_eq!(servo.last_pulses()[0], 1500);
    }

    #[test]
    fn pulse_step_elapsed_validation() {
        assert_eq!(
            pulse_step_for_elapsed(
                5_000.0,
                Duration::from_millis(20),
                Duration::from_millis(100)
            )
            .unwrap(),
            100
        );
        assert!(pulse_step_for_elapsed(5_000.0, Duration::ZERO, Duration::from_millis(100)).is_err());
        assert!(pulse_step_for_elapsed(5_000.0, Duration::from_millis(101), Duration::from_millis(100)).is_err());
        assert!(pulse_step_for_elapsed(f64::NAN, Duration::from_millis(20), Duration::from_millis(100)).is_err());
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
            prop_assert!(result >= lo);
            prop_assert!(result <= hi);
        }

        #[test]
        fn slew_limit_converges(current in 500u16..2500, target in 500u16..2500) {
            prop_assert_eq!(slew_limit(current, target, u16::MAX), target);
        }

        #[test]
        fn slew_limit_max_step_respected(current in 500u16..2500, target in 500u16..2500, max_step in 1u16..500) {
            let result = slew_limit(current, target, max_step);
            let delta = result.abs_diff(current);
            prop_assert!(delta <= max_step);
        }
    }
}
