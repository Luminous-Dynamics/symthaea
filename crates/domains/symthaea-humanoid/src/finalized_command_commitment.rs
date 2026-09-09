// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Collision-resistant identity for the exact humanoid command selected for dispatch.
//!
//! This module creates evidence identity only. A command commitment is not proof
//! that the command came from the authoritative execution pipeline, is still
//! current, or may be dispatched. Later physical admission must bind this exact
//! digest to current authority, configuration, calibration, safety, and backend
//! session state before an actuator sink can use it.

use crate::morphology::HumanoidMorphology;
use crate::types::{ActuationMode, CommandValidationError, HumanoidCommand};

/// Current commitment schema.
pub const FINALIZED_HUMANOID_COMMAND_SCHEMA_VERSION: u16 = 1;
/// Domain separator for exact humanoid command commitments.
pub const FINALIZED_HUMANOID_COMMAND_DOMAIN: &[u8] =
    b"symthaea.humanoid.finalized-command.v1\0";

/// Collision-resistant identity of one exact command interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FinalizedHumanoidCommandDigest(pub [u8; 32]);

/// Non-authoritative evidence describing the exact command bytes and semantics
/// committed by [`commit_finalized_humanoid_command`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalizedHumanoidCommandCommitment {
    digest: FinalizedHumanoidCommandDigest,
    morphology_schema_id: String,
    actuation_mode: ActuationMode,
    actuator_count: usize,
}

impl FinalizedHumanoidCommandCommitment {
    pub const fn digest(&self) -> FinalizedHumanoidCommandDigest {
        self.digest
    }

    pub fn morphology_schema_id(&self) -> &str {
        &self.morphology_schema_id
    }

    pub const fn actuation_mode(&self) -> ActuationMode {
        self.actuation_mode
    }

    pub const fn actuator_count(&self) -> usize {
        self.actuator_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinalizedCommandCommitmentError {
    InvalidCommand(CommandValidationError),
    LengthOverflow,
}

impl std::fmt::Display for FinalizedCommandCommitmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCommand(source) => write!(f, "invalid finalized humanoid command: {source:?}"),
            Self::LengthOverflow => f.write_str("finalized command length exceeds canonical encoding"),
        }
    }
}

impl std::error::Error for FinalizedCommandCommitmentError {}

impl From<CommandValidationError> for FinalizedCommandCommitmentError {
    fn from(value: CommandValidationError) -> Self {
        Self::InvalidCommand(value)
    }
}

/// Commit the exact command values together with the semantics required to
/// interpret them physically.
///
/// Floating-point values are committed by their exact IEEE-754 bit patterns.
/// This deliberately identifies the exact dispatch representation rather than
/// an approximate or numerically equivalent command.
pub fn commit_finalized_humanoid_command(
    command: &HumanoidCommand,
    morphology: HumanoidMorphology,
    actuation_mode: ActuationMode,
) -> Result<FinalizedHumanoidCommandCommitment, FinalizedCommandCommitmentError> {
    command.validate_for(morphology.num_actuators(), actuation_mode)?;

    let morphology_schema_id = morphology.schema_id().to_owned();
    let morphology_len = u32::try_from(morphology_schema_id.len())
        .map_err(|_| FinalizedCommandCommitmentError::LengthOverflow)?;
    let actuator_count = u32::try_from(command.num_actuators())
        .map_err(|_| FinalizedCommandCommitmentError::LengthOverflow)?;

    let mut hasher = blake3::Hasher::new();
    hasher.update(&(FINALIZED_HUMANOID_COMMAND_DOMAIN.len() as u32).to_be_bytes());
    hasher.update(FINALIZED_HUMANOID_COMMAND_DOMAIN);
    hasher.update(&FINALIZED_HUMANOID_COMMAND_SCHEMA_VERSION.to_be_bytes());
    hasher.update(&morphology_len.to_be_bytes());
    hasher.update(morphology_schema_id.as_bytes());
    hasher.update(&[actuation_mode_tag(actuation_mode)]);
    hasher.update(&actuator_count.to_be_bytes());
    for value in &command.torques {
        hasher.update(&value.to_bits().to_be_bytes());
    }

    Ok(FinalizedHumanoidCommandCommitment {
        digest: FinalizedHumanoidCommandDigest(*hasher.finalize().as_bytes()),
        morphology_schema_id,
        actuation_mode,
        actuator_count: command.num_actuators(),
    })
}

fn actuation_mode_tag(mode: ActuationMode) -> u8 {
    match mode {
        ActuationMode::NormalizedTorque => 0,
        ActuationMode::TorqueNewtonMetres => 1,
        ActuationMode::NormalizedPosition => 2,
        ActuationMode::PositionTargetRadians => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_command_commitment_is_deterministic() {
        let mut command = HumanoidCommand::zero();
        command.torques[0] = 0.25;
        let left = commit_finalized_humanoid_command(
            &command,
            HumanoidMorphology::Dmc21,
            ActuationMode::NormalizedTorque,
        )
        .unwrap();
        let right = commit_finalized_humanoid_command(
            &command,
            HumanoidMorphology::Dmc21,
            ActuationMode::NormalizedTorque,
        )
        .unwrap();
        assert_eq!(left.digest(), right.digest());
        assert_eq!(left.actuator_count(), HumanoidMorphology::Dmc21.num_actuators());
        assert_eq!(left.morphology_schema_id(), HumanoidMorphology::Dmc21.schema_id());
    }

    #[test]
    fn one_bit_command_change_changes_digest() {
        let baseline = HumanoidCommand::zero();
        let mut changed = baseline.clone();
        changed.torques[0] = f32::from_bits(1);
        let baseline = commit_finalized_humanoid_command(
            &baseline,
            HumanoidMorphology::Dmc21,
            ActuationMode::NormalizedTorque,
        )
        .unwrap();
        let changed = commit_finalized_humanoid_command(
            &changed,
            HumanoidMorphology::Dmc21,
            ActuationMode::NormalizedTorque,
        )
        .unwrap();
        assert_ne!(baseline.digest(), changed.digest());
    }

    #[test]
    fn physical_interpretation_changes_digest() {
        let command = HumanoidCommand::zero();
        let normalized = commit_finalized_humanoid_command(
            &command,
            HumanoidMorphology::Dmc21,
            ActuationMode::NormalizedTorque,
        )
        .unwrap();
        let newton_metres = commit_finalized_humanoid_command(
            &command,
            HumanoidMorphology::Dmc21,
            ActuationMode::TorqueNewtonMetres,
        )
        .unwrap();
        assert_ne!(normalized.digest(), newton_metres.digest());
    }

    #[test]
    fn malformed_or_nonfinite_command_cannot_be_committed() {
        let wrong_count = HumanoidCommand::zero_for(1);
        assert!(matches!(
            commit_finalized_humanoid_command(
                &wrong_count,
                HumanoidMorphology::Dmc21,
                ActuationMode::NormalizedTorque,
            ),
            Err(FinalizedCommandCommitmentError::InvalidCommand(
                CommandValidationError::ActuatorCount { .. }
            ))
        ));

        let mut nonfinite = HumanoidCommand::zero();
        nonfinite.torques[0] = f32::NAN;
        assert!(matches!(
            commit_finalized_humanoid_command(
                &nonfinite,
                HumanoidMorphology::Dmc21,
                ActuationMode::NormalizedTorque,
            ),
            Err(FinalizedCommandCommitmentError::InvalidCommand(
                CommandValidationError::NonFiniteValue { .. }
            ))
        ));
    }
}
