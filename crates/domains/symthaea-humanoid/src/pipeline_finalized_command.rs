// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Process-local proof that one exact command crossed the humanoid finalization path.
//!
//! A bare [`FinalizedHumanoidCommandCommitment`] is only content identity: callers
//! may commit any structurally valid command. This module adds a stronger,
//! non-serializable wrapper that can only be constructed by invoking the real
//! [`HumanoidExecutionPipeline::finalize_prepared`] transition and immediately
//! committing the exact command it returned.
//!
//! This still does not grant physical dispatch authority. Future HAL admission
//! should consume this value together with verified mission authority, current
//! qualification, physical health, epistemic currentness, configuration,
//! calibration, safety epoch, and backend-session state.

use crate::execution::{
    HumanoidAuthorityEnvelope, HumanoidExecutionPipeline, HumanoidExecutionReport,
    HumanoidPreparedCommand,
};
use crate::finalized_command_commitment::{
    FinalizedCommandCommitmentError, FinalizedHumanoidCommandCommitment,
    FinalizedHumanoidCommandDigest, commit_finalized_humanoid_command,
};
use crate::morphology::HumanoidMorphology;
use crate::types::{ActuationMode, HumanoidCommand, HumanoidState};

/// Process-local evidence that one exact command crossed the authoritative
/// humanoid finalization path.
///
/// Intentionally not `Clone`, `Copy`, `Serialize`, or `Deserialize`.
#[derive(Debug)]
pub struct PipelineFinalizedHumanoidCommand {
    command: HumanoidCommand,
    report: HumanoidExecutionReport,
    commitment: FinalizedHumanoidCommandCommitment,
}

impl PipelineFinalizedHumanoidCommand {
    /// Exact command produced by the final safety projection.
    pub fn command(&self) -> &HumanoidCommand {
        &self.command
    }

    /// Execution evidence emitted by the same finalization transition.
    pub fn report(&self) -> &HumanoidExecutionReport {
        &self.report
    }

    /// Exact command-identity evidence bound to this finalized value.
    pub fn commitment(&self) -> &FinalizedHumanoidCommandCommitment {
        &self.commitment
    }

    pub const fn digest(&self) -> FinalizedHumanoidCommandDigest {
        self.commitment.digest()
    }

    pub fn morphology_schema_id(&self) -> &str {
        self.commitment.morphology_schema_id()
    }

    pub const fn actuation_mode(&self) -> ActuationMode {
        self.commitment.actuation_mode()
    }
}

/// Cross the existing humanoid finalization path and immediately bind the exact
/// projected command to collision-resistant identity.
///
/// Keeping this as a separate function avoids changing the compatibility shape
/// of [`HumanoidExecutionResult`] while giving future physical dispatch code a
/// stronger type it can require.
pub fn finalize_and_commit_humanoid_command(
    pipeline: &mut HumanoidExecutionPipeline,
    prepared: HumanoidPreparedCommand,
    state: &HumanoidState,
    authority: HumanoidAuthorityEnvelope,
    actuation_mode: ActuationMode,
    dt: f64,
) -> Result<PipelineFinalizedHumanoidCommand, FinalizedCommandCommitmentError> {
    let morphology = pipeline.morphology();
    let finalized = pipeline.finalize_prepared(prepared, state, authority, actuation_mode, dt);
    let commitment =
        commit_finalized_humanoid_command(&finalized.command, morphology, actuation_mode)?;

    Ok(PipelineFinalizedHumanoidCommand {
        command: finalized.command,
        report: finalized.report,
        commitment,
    })
}

/// Convenience guard for callers that need to compare this value with the
/// exact morphology expected by a later physical-runtime admission.
pub fn finalized_command_matches_morphology(
    finalized: &PipelineFinalizedHumanoidCommand,
    morphology: HumanoidMorphology,
) -> bool {
    finalized.morphology_schema_id() == morphology.schema_id()
        && finalized.command().num_actuators() == morphology.num_actuators()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::HumanoidTask;

    #[test]
    fn pipeline_finalization_owns_exact_commitment() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut pipeline = HumanoidExecutionPipeline::new(morphology);
        let state = HumanoidState::standing_for(morphology);
        let baseline = HumanoidCommand::zero_for(morphology.num_actuators());
        let residual = HumanoidCommand::zero_for(morphology.num_actuators());
        let prepared = pipeline.prepare(
            HumanoidTask::Stand,
            &state,
            &baseline,
            &residual,
            1.0,
            0.0,
        );

        let finalized = finalize_and_commit_humanoid_command(
            &mut pipeline,
            prepared,
            &state,
            HumanoidAuthorityEnvelope::fully_admitted(),
            ActuationMode::NormalizedTorque,
            0.02,
        )
        .unwrap();

        let recomputed = commit_finalized_humanoid_command(
            finalized.command(),
            morphology,
            ActuationMode::NormalizedTorque,
        )
        .unwrap();
        assert_eq!(finalized.digest(), recomputed.digest());
        assert!(finalized_command_matches_morphology(&finalized, morphology));
    }

    #[test]
    fn pipeline_finalized_type_tracks_actuation_semantics() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut pipeline = HumanoidExecutionPipeline::new(morphology);
        let state = HumanoidState::standing_for(morphology);
        let baseline = HumanoidCommand::zero_for(morphology.num_actuators());
        let residual = HumanoidCommand::zero_for(morphology.num_actuators());
        let prepared = pipeline.prepare(
            HumanoidTask::Stand,
            &state,
            &baseline,
            &residual,
            1.0,
            0.0,
        );

        let finalized = finalize_and_commit_humanoid_command(
            &mut pipeline,
            prepared,
            &state,
            HumanoidAuthorityEnvelope::fully_admitted(),
            ActuationMode::NormalizedTorque,
            0.02,
        )
        .unwrap();
        assert_eq!(finalized.actuation_mode(), ActuationMode::NormalizedTorque);
    }
}
