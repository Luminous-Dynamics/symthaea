// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-command admission over the legacy HAL safety interlock.
//!
//! The legacy [`SafetyInterlock::filter_command`] safely supports existing
//! callers by clamping torque and applying prediction-error gain reduction.
//! Those transformations are incompatible with a future exact-command dispatch
//! theorem because the command leaving the interlock may differ from the command
//! finalized and committed upstream.
//!
//! `ExactCommandSafetyInterlock` preserves the existing e-stop, trip, watchdog,
//! configuration, shape, torque-limit, and prediction-error behavior, but never
//! exposes a transformed command. If the legacy interlock would change even one
//! command bit, exact admission latches a safety fault and refuses the command.
//! Existing callers can continue using `SafetyInterlock` directly.

use symthaea_humanoid::types::HumanoidCommand;

use crate::error::{HalError, HalResult};
use crate::interlock::{SafetyConfig, SafetyInterlock};

/// Compatibility-preserving exact-command safety boundary.
pub struct ExactCommandSafetyInterlock {
    inner: SafetyInterlock,
}

impl ExactCommandSafetyInterlock {
    pub fn new() -> Self {
        Self::from_interlock(SafetyInterlock::new())
    }

    pub fn with_config(config: SafetyConfig) -> Self {
        Self::from_interlock(SafetyInterlock::with_config(config))
    }

    pub const fn from_interlock(inner: SafetyInterlock) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &SafetyInterlock {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut SafetyInterlock {
        &mut self.inner
    }

    pub fn into_inner(self) -> SafetyInterlock {
        self.inner
    }

    /// Admit a command only if the complete legacy safety path would pass it
    /// through bit-for-bit unchanged.
    ///
    /// A would-be clamp or prediction-error gain reduction is treated as a
    /// request for upstream re-finalization, not as permission for HAL to create
    /// a different command after authority/evidence binding.
    pub fn admit_exact(&mut self, command: &HumanoidCommand) -> HalResult<()> {
        let filtered = self.inner.filter_command(command)?;
        if !commands_bitwise_equal(command, &filtered) {
            return Err(self.inner.trip_safety(
                "exact-command admission refused a post-finalization HAL transformation",
            ));
        }
        Ok(())
    }
}

impl Default for ExactCommandSafetyInterlock {
    fn default() -> Self {
        Self::new()
    }
}

fn commands_bitwise_equal(left: &HumanoidCommand, right: &HumanoidCommand) -> bool {
    left.torques.len() == right.torques.len()
        && left
            .torques
            .iter()
            .zip(right.torques.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn already_safe_command_is_admitted_unchanged() {
        let mut interlock = ExactCommandSafetyInterlock::new();
        let mut command = HumanoidCommand::zero();
        command.torques[0] = 0.5;
        assert!(interlock.admit_exact(&command).is_ok());
        assert!(!interlock.inner().is_tripped());
    }

    #[test]
    fn torque_clamp_is_refused_instead_of_exposed_as_new_command() {
        let mut interlock = ExactCommandSafetyInterlock::new();
        let mut command = HumanoidCommand::zero();
        command.torques[0] = 1.0; // legacy default clamps this to 0.9
        let error = interlock.admit_exact(&command).unwrap_err();
        assert!(matches!(error, HalError::Safety(_)));
        assert!(interlock.inner().is_tripped());
    }

    #[test]
    fn prediction_error_derating_is_refused_instead_of_mutating_command() {
        let mut interlock = ExactCommandSafetyInterlock::new();
        interlock.inner_mut().set_prediction_error(1.0);
        let mut command = HumanoidCommand::zero();
        command.torques[0] = 0.5;
        let error = interlock.admit_exact(&command).unwrap_err();
        assert!(matches!(error, HalError::Safety(_)));
        assert!(interlock.inner().is_tripped());
    }

    #[test]
    fn e_stop_remains_fail_stop() {
        let mut interlock = ExactCommandSafetyInterlock::new();
        interlock.inner().trigger_estop();
        assert!(matches!(
            interlock.admit_exact(&HumanoidCommand::zero()),
            Err(HalError::EStop)
        ));
        assert!(interlock.inner().is_tripped());
    }
}
