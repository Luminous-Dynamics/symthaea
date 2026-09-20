// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Read-only operational/session-state projection for [`ServoOutput`].
//!
//! This module intentionally does **not** create a second actuator state
//! machine. It derives one closed status from the executor's existing
//! authoritative fields so callers do not have to reconstruct meaning from
//! `initialized + enabled + shutdown_verified + fault_latched` themselves.
//!
//! Current operational state is not historical effect evidence:
//!
//! ```text
//! ServoSessionState != ServoActuationReceipt
//! ```

use embedded_hal::i2c::I2c;

use crate::runtime::HalRuntime;
use crate::servo::{ActuationDisposition, ServoOutput};

/// Closed projection of the current servo executor/session status.
///
/// This is observability/currentness state only. It is not an authority token,
/// physical-position proof, or substitute for typed actuation/shutdown receipts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServoSessionState {
    /// The controller stack has not completed initialization. `shutdown_verified`
    /// records only whether the most recent all-off theorem is currently true.
    Uninitialized { shutdown_verified: bool },
    /// Initialized and both controller endpoints most recently confirmed all-off.
    VerifiedOff,
    /// Initialized, enabled, and inside a live executor-owned timing session.
    Live,
    /// The previous live timing session expired. No new command was dispatched,
    /// but prior PWM may remain latched until all-off is verified and the
    /// executor is explicitly re-enabled.
    StaleTiming { shutdown_verified: bool },
    /// A consequential command/shutdown uncertainty remains latched.
    FaultLatched {
        disposition: ActuationDisposition,
        shutdown_verified: bool,
    },
    /// Existing fields form a combination outside the closed state theorem.
    /// This must be surfaced rather than silently guessed into a normal state.
    Inconsistent {
        initialized: bool,
        enabled: bool,
        shutdown_verified: bool,
        fault_latched: Option<ActuationDisposition>,
    },
}

impl<I: I2c> ServoOutput<I> {
    /// Derive the current closed servo-session state from authoritative executor
    /// fields. No state is mutated by this projection.
    pub fn session_state(&self) -> ServoSessionState {
        let initialized = self.is_initialized();
        let enabled = self.is_enabled();
        let shutdown_verified = self.shutdown_verified();
        let fault_latched = self.fault_latched();

        if let Some(disposition) = fault_latched {
            return if matches!(
                disposition,
                ActuationDisposition::Partial | ActuationDisposition::InDoubt
            ) {
                ServoSessionState::FaultLatched {
                    disposition,
                    shutdown_verified,
                }
            } else {
                ServoSessionState::Inconsistent {
                    initialized,
                    enabled,
                    shutdown_verified,
                    fault_latched,
                }
            };
        }

        match (initialized, enabled, shutdown_verified) {
            (false, false, verified) => ServoSessionState::Uninitialized {
                shutdown_verified: verified,
            },
            (true, false, true) => ServoSessionState::VerifiedOff,
            (true, true, false) => ServoSessionState::Live,
            // Under the current executor theorem this is the only stable
            // initialized/no-fault/no-shutdown state: #4983 revoked a live
            // timing session because its monotonic gap became stale.
            (true, false, false) => ServoSessionState::StaleTiming {
                shutdown_verified: false,
            },
            _ => ServoSessionState::Inconsistent {
                initialized,
                enabled,
                shutdown_verified,
                fault_latched,
            },
        }
    }
}

impl<I: I2c> HalRuntime<I> {
    /// Return the exact same typed servo-session projection exposed by the
    /// underlying executor. Runtime callers therefore do not need to reconstruct
    /// servo state from health strings or individual booleans.
    pub fn servo_session_state(&self) -> ServoSessionState {
        self.servo().session_state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CalibrationProfile;
    use crate::SafetyInterlock;
    use crate::mock::MockI2cBus;
    use std::time::Duration;
    use symthaea_humanoid::types::HumanoidCommand;

    fn make_servo() -> ServoOutput<MockI2cBus> {
        ServoOutput::new(
            MockI2cBus::new(),
            MockI2cBus::new(),
            CalibrationProfile::default_21(),
        )
    }

    #[test]
    fn constructor_is_uninitialized_but_verified_off() {
        let servo = make_servo();
        assert_eq!(
            servo.session_state(),
            ServoSessionState::Uninitialized {
                shutdown_verified: true
            }
        );
    }

    #[test]
    fn successful_init_projects_verified_off_then_enable_projects_live() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        assert_eq!(servo.session_state(), ServoSessionState::VerifiedOff);
        servo.enable().unwrap();
        assert_eq!(servo.session_state(), ServoSessionState::Live);
    }

    #[test]
    fn runtime_exposes_the_same_session_projection() {
        let mut servo = make_servo();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        let runtime = HalRuntime::new(servo, SafetyInterlock::new());

        assert_eq!(runtime.servo_session_state(), runtime.servo().session_state());
        assert_eq!(runtime.servo_session_state(), ServoSessionState::Live);
    }

    #[test]
    fn stale_monotonic_session_is_not_projected_as_verified_off() {
        let mut servo = make_servo();
        servo.set_max_slew_gap(Duration::from_millis(1)).unwrap();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        std::thread::sleep(Duration::from_millis(3));

        let failure = servo.apply(&HumanoidCommand::zero()).unwrap_err();
        assert_eq!(failure.receipt.disposition, ActuationDisposition::NotDispatched);
        assert_eq!(
            servo.session_state(),
            ServoSessionState::StaleTiming {
                shutdown_verified: false
            }
        );
    }

    #[test]
    fn verified_all_off_after_stale_session_is_currently_verified_off_projection() {
        let mut servo = make_servo();
        servo.set_max_slew_gap(Duration::from_millis(1)).unwrap();
        servo.init(50.0).unwrap();
        servo.enable().unwrap();
        std::thread::sleep(Duration::from_millis(3));
        let _ = servo.apply(&HumanoidCommand::zero());

        servo.disable().unwrap();
        assert_eq!(servo.session_state(), ServoSessionState::VerifiedOff);
    }
}
