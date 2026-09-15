// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical exact-command identity and typed HAL safety admission.
//!
//! XCA-v1 deliberately binds authority to the canonical command bytes themselves.
//! The command is small, so this avoids making a cryptographic digest collision
//! assumption part of the execution-authority theorem. Evidence and transport
//! layers may derive their own qualified digests from [`ExactCommandIdentityV1`]
//! later, but those digests do not replace the exact bytes at this boundary.
//!
//! This module does **not** grant operator authority or execution permission and
//! does not constrain the legacy [`ServoOutput`](crate::servo::ServoOutput) API.

use serde::{Deserialize, Serialize};
use symthaea_humanoid::types::HumanoidCommand;

use crate::error::{HalError, HalResult};
use crate::interlock::{SafetyConfig, SafetyInterlock};

/// Domain/version prefix included literally in every XCA-v1 command identity.
pub const EXACT_COMMAND_DOMAIN_V1: &[u8] = b"symthaea.hal.exact-command.v1\0";

/// Conservative allocation bound for the generic XCA-v1 identity format.
///
/// The current legacy HAL safety interlock accepts a much smaller fixed actuator
/// count; this bound only prevents an authority-identity parser from accepting an
/// effectively unbounded vector before the backend-specific shape gate runs.
pub const MAX_EXACT_COMMAND_ACTUATORS_V1: usize = 4096;

/// Canonical, versioned bytes identifying one exact proposed command.
///
/// The bytes are private and immutable after construction. Equality therefore
/// means literal byte equality, not semantic approximation or hash equality.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExactCommandIdentityV1(Vec<u8>);

impl ExactCommandIdentityV1 {
    /// Borrow the complete canonical authority identity.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// Immutable proposed exact command paired with its canonical identity.
#[derive(Debug)]
pub struct ProposedExactCommandV1 {
    command: HumanoidCommand,
    identity: ExactCommandIdentityV1,
}

impl ProposedExactCommandV1 {
    /// Validate and freeze one exact command proposal.
    pub fn new(command: HumanoidCommand) -> HalResult<Self> {
        let identity = ExactCommandIdentityV1(canonical_command_bytes_v1(&command)?);
        Ok(Self { command, identity })
    }

    /// Borrow the exact command represented by this proposal.
    pub fn command(&self) -> &HumanoidCommand {
        &self.command
    }

    /// Borrow the immutable canonical identity for this proposal.
    pub fn identity(&self) -> &ExactCommandIdentityV1 {
        &self.identity
    }
}

/// Successful HAL safety admission for one exact command identity.
///
/// This is a safety artifact only. It is not an operator grant, execution permit,
/// independent-safety lease, or proof that any effect occurred. Its only field is
/// private so downstream callers cannot fabricate an admitted command directly.
#[derive(Debug)]
pub struct SafetyAdmittedExactCommandV1 {
    proposed: ProposedExactCommandV1,
}

impl SafetyAdmittedExactCommandV1 {
    /// Borrow the exact command that passed HAL safety unchanged.
    pub fn command(&self) -> &HumanoidCommand {
        self.proposed.command()
    }

    /// Borrow the exact canonical identity bound by this admission.
    pub fn identity(&self) -> &ExactCommandIdentityV1 {
        self.proposed.identity()
    }
}

/// Safety boundary that admits a finalized command only when the legacy HAL
/// safety path would leave every command bit unchanged.
pub struct ExactCommandSafetyInterlockV1 {
    inner: SafetyInterlock,
}

impl ExactCommandSafetyInterlockV1 {
    /// Create an exact-command safety interlock with the default safety policy.
    pub fn new() -> Self {
        Self::from_interlock(SafetyInterlock::new())
    }

    /// Create an exact-command safety interlock with an explicit safety policy.
    pub fn with_config(config: SafetyConfig) -> Self {
        Self::from_interlock(SafetyInterlock::with_config(config))
    }

    /// Wrap an existing legacy safety interlock.
    pub const fn from_interlock(inner: SafetyInterlock) -> Self {
        Self { inner }
    }

    /// Borrow the underlying interlock for status and e-stop handling.
    pub const fn inner(&self) -> &SafetyInterlock {
        &self.inner
    }

    /// Mutably borrow the underlying interlock for existing sensor/monitor input.
    ///
    /// This does not grant execution authority. XCA-C must independently bind an
    /// operator/policy decision and a single-use permit at the effect boundary.
    pub fn inner_mut(&mut self) -> &mut SafetyInterlock {
        &mut self.inner
    }

    /// Admit one proposal only if the complete legacy safety path accepts it and
    /// would emit the same torque bits exactly.
    pub fn admit_exact(
        &mut self,
        proposed: ProposedExactCommandV1,
    ) -> HalResult<SafetyAdmittedExactCommandV1> {
        let filtered = self.inner.filter_command(proposed.command())?;
        if !commands_bitwise_equal(proposed.command(), &filtered) {
            return Err(self.inner.trip_safety(
                "exact-command admission refused a post-finalization HAL transformation",
            ));
        }
        Ok(SafetyAdmittedExactCommandV1 { proposed })
    }
}

impl Default for ExactCommandSafetyInterlockV1 {
    fn default() -> Self {
        Self::new()
    }
}

/// Produce the canonical XCA-v1 bytes for a command.
///
/// Encoding is deliberately language-neutral and contains its domain/version:
///
/// ```text
/// "symthaea.hal.exact-command.v1\\0"      fixed bytes
/// actuator_count                           u32 little-endian
/// torque[0]                                exact IEEE-754 f32 bits, LE
/// ...
/// torque[n-1]                              exact IEEE-754 f32 bits, LE
/// ```
///
/// `+0.0` and `-0.0` are intentionally different exact commands. NaNs,
/// infinities, empty commands, normalized values outside `[-1, 1]`, and commands
/// above the explicit v1 bound are rejected before identity construction.
pub fn canonical_command_bytes_v1(command: &HumanoidCommand) -> HalResult<Vec<u8>> {
    if command.torques.is_empty() {
        return Err(HalError::Safety(
            "exact command must contain at least one actuator".to_string(),
        ));
    }
    if command.torques.len() > MAX_EXACT_COMMAND_ACTUATORS_V1 {
        return Err(HalError::Safety(format!(
            "exact command actuator count {} exceeds v1 bound {}",
            command.torques.len(),
            MAX_EXACT_COMMAND_ACTUATORS_V1
        )));
    }
    let count = u32::try_from(command.torques.len()).map_err(|_| {
        HalError::Safety("exact command actuator count exceeds u32".to_string())
    })?;

    let mut out = Vec::with_capacity(EXACT_COMMAND_DOMAIN_V1.len() + 4 + command.torques.len() * 4);
    out.extend_from_slice(EXACT_COMMAND_DOMAIN_V1);
    out.extend_from_slice(&count.to_le_bytes());
    for (index, torque) in command.torques.iter().copied().enumerate() {
        if !torque.is_finite() {
            return Err(HalError::Safety(format!(
                "exact command contains non-finite torque at index {index}"
            )));
        }
        if !(-1.0..=1.0).contains(&torque) {
            return Err(HalError::Safety(format!(
                "exact command torque at index {index} is outside [-1, 1]: {torque}"
            )));
        }
        out.extend_from_slice(&torque.to_bits().to_le_bytes());
    }
    Ok(out)
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
    fn canonical_identity_matches_independent_language_neutral_golden() {
        let command = HumanoidCommand {
            torques: vec![0.0, -0.0, 0.5],
        };
        let identity = ProposedExactCommandV1::new(command).unwrap();
        assert_eq!(
            identity.identity().as_bytes(),
            &[
                0x73, 0x79, 0x6d, 0x74, 0x68, 0x61, 0x65, 0x61, 0x2e, 0x68, 0x61, 0x6c, 0x2e,
                0x65, 0x78, 0x61, 0x63, 0x74, 0x2d, 0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64,
                0x2e, 0x76, 0x31, 0x00, // domain
                0x03, 0x00, 0x00, 0x00, // actuator count
                0x00, 0x00, 0x00, 0x00, // +0.0
                0x00, 0x00, 0x00, 0x80, // -0.0
                0x00, 0x00, 0x00, 0x3f, // +0.5
            ]
        );
    }

    #[test]
    fn one_bit_command_change_changes_exact_identity() {
        let plus_zero = ProposedExactCommandV1::new(HumanoidCommand {
            torques: vec![0.0],
        })
        .unwrap();
        let minus_zero = ProposedExactCommandV1::new(HumanoidCommand {
            torques: vec![-0.0],
        })
        .unwrap();
        assert_ne!(plus_zero.identity(), minus_zero.identity());
    }

    #[test]
    fn already_safe_command_produces_typed_admission() {
        let mut interlock = ExactCommandSafetyInterlockV1::new();
        let mut command = HumanoidCommand::zero();
        command.torques[0] = 0.5;
        let proposed = ProposedExactCommandV1::new(command).unwrap();
        let expected_identity = proposed.identity().clone();
        let admitted = interlock.admit_exact(proposed).unwrap();
        assert_eq!(admitted.identity(), &expected_identity);
        assert_eq!(admitted.command().torques[0].to_bits(), 0.5f32.to_bits());
        assert!(!interlock.inner().is_tripped());
    }

    #[test]
    fn torque_clamp_is_refused_instead_of_creating_new_exact_command() {
        let mut interlock = ExactCommandSafetyInterlockV1::new();
        let mut command = HumanoidCommand::zero();
        command.torques[0] = 1.0; // legacy default max torque is 0.9
        let proposed = ProposedExactCommandV1::new(command).unwrap();
        let error = interlock.admit_exact(proposed).unwrap_err();
        assert!(matches!(error, HalError::Safety(_)));
        assert!(interlock.inner().is_tripped());
    }

    #[test]
    fn prediction_error_derating_is_refused_instead_of_mutating_identity() {
        let mut interlock = ExactCommandSafetyInterlockV1::new();
        interlock.inner_mut().set_prediction_error(1.0);
        let mut command = HumanoidCommand::zero();
        command.torques[0] = 0.5;
        let proposed = ProposedExactCommandV1::new(command).unwrap();
        let error = interlock.admit_exact(proposed).unwrap_err();
        assert!(matches!(error, HalError::Safety(_)));
        assert!(interlock.inner().is_tripped());
    }

    #[test]
    fn malformed_exact_commands_are_rejected_before_admission() {
        assert!(ProposedExactCommandV1::new(HumanoidCommand { torques: vec![] }).is_err());
        assert!(
            ProposedExactCommandV1::new(HumanoidCommand {
                torques: vec![f32::NAN],
            })
            .is_err()
        );
        assert!(
            ProposedExactCommandV1::new(HumanoidCommand {
                torques: vec![1.1],
            })
            .is_err()
        );
        assert!(
            ProposedExactCommandV1::new(HumanoidCommand {
                torques: vec![0.0; MAX_EXACT_COMMAND_ACTUATORS_V1 + 1],
            })
            .is_err()
        );
    }
}
