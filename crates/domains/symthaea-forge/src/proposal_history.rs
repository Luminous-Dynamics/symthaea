// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generation-boundary history context for proposal-complete Forge observations.
//!
//! The existing sequence theorem conditions concrete trials on the accepted state entering their
//! generation. Proposal-policy research also needs that same state for no-op attempts and for
//! unselected families. This module derives a history view for every proposal observation row from
//! only the validated accepted steps whose generation is strictly earlier than the row's generation.
//! Siblings in one generation therefore share the same history and parent even when some siblings
//! never produced a concrete candidate.

use crate::proposal_dataset::{
    ForgeProposalDatasetError, ForgeProposalObservationRow, ForgeProposalObservationTable,
};
use crate::sequence_learning::{
    ForgeFamilyHistory, ForgeSearchFamilySequence, ForgeSequenceLearningError,
};
use crate::sequence_stats::{ForgeHistoryOrder, ForgeSequenceStatsError};
use serde::Serialize;
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalHistoryError {
    #[error(transparent)]
    Dataset(#[from] ForgeProposalDatasetError),
    #[error(transparent)]
    Sequence(#[from] ForgeSequenceLearningError),
    #[error(transparent)]
    SequenceStats(#[from] ForgeSequenceStatsError),
    #[error("proposal observation table and sequence belong to different semantic runs")]
    RunMismatch,
    #[error("proposal observation table and sequence use different generator identities")]
    GeneratorMismatch,
    #[error("proposal observation table and sequence bind different concrete family-trial sets")]
    TrialSetMismatch,
    #[error("proposal row parent artifact is not the accepted parent entering its generation")]
    ParentMismatch,
    #[error("conditioned proposal row identity does not match canonical fields")]
    RowIdentityMismatch,
    #[error("conditioned proposal table is not in canonical source-row order")]
    NonCanonicalTable,
    #[error("conditioned proposal table identity does not match canonical fields")]
    TableIdentityMismatch,
}

/// Exact accepted state entering one Forge generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalGenerationState {
    generation: u64,
    parent_artifact_id: ContentId,
    history_before: ForgeFamilyHistory,
}

impl ForgeProposalGenerationState {
    fn from_sequence(
        sequence: &ForgeSearchFamilySequence,
        generation: u64,
    ) -> Result<Self, ForgeProposalHistoryError> {
        sequence.validate()?;
        let mut parent_artifact_id = sequence.baseline_artifact_id().clone();
        let mut families = Vec::new();
        for step in sequence.accepted_steps() {
            if step.generation() >= generation {
                break;
            }
            if step.parent_artifact_id() != &parent_artifact_id {
                return Err(ForgeProposalHistoryError::ParentMismatch);
            }
            parent_artifact_id = step.candidate_artifact_id().clone();
            families.push(step.family_id().clone());
        }
        let history_before = ForgeFamilyHistory::new(sequence.generator_id().clone(), families)?;
        Ok(Self {
            generation,
            parent_artifact_id,
            history_before,
        })
    }

    pub fn generation(&self) -> u64 { self.generation }
    pub fn parent_artifact_id(&self) -> &ContentId { &self.parent_artifact_id }
    pub fn history_before(&self) -> &ForgeFamilyHistory { &self.history_before }

    /// Explicit bounded-order history view for a later model specification.
    pub fn history_suffix(
        &self,
        order: &ForgeHistoryOrder,
    ) -> Result<ForgeFamilyHistory, ForgeProposalHistoryError> {
        order.validate()?;
        Ok(self.history_before.suffix(usize::from(order.depth()))?)
    }
}

/// One proposal-complete family row paired with the accepted state entering its generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeConditionedProposalObservationRow {
    id: ContentId,
    source: ForgeProposalObservationRow,
    state: ForgeProposalGenerationState,
}

impl ForgeConditionedProposalObservationRow {
    fn new(
        source: ForgeProposalObservationRow,
        state: ForgeProposalGenerationState,
    ) -> Result<Self, ForgeProposalHistoryError> {
        source.validate()?;
        state.history_before.validate()?;
        if source.generation() != state.generation
            || source.parent_artifact_id() != &state.parent_artifact_id
            || source.family_id().generator_id() != state.history_before.generator_id()
        {
            return Err(ForgeProposalHistoryError::ParentMismatch);
        }
        let id = derive_row_id(source.id(), state.history_before.id(), &state.parent_artifact_id);
        let row = Self { id, source, state };
        row.validate()?;
        Ok(row)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn source(&self) -> &ForgeProposalObservationRow { &self.source }
    pub fn state(&self) -> &ForgeProposalGenerationState { &self.state }
    pub fn history_before(&self) -> &ForgeFamilyHistory { self.state.history_before() }

    pub fn history_suffix(
        &self,
        order: &ForgeHistoryOrder,
    ) -> Result<ForgeFamilyHistory, ForgeProposalHistoryError> {
        self.state.history_suffix(order)
    }

    pub fn validate(&self) -> Result<(), ForgeProposalHistoryError> {
        self.source.validate()?;
        self.state.history_before.validate()?;
        if self.source.generation() != self.state.generation
            || self.source.parent_artifact_id() != &self.state.parent_artifact_id
            || self.source.family_id().generator_id() != self.state.history_before.generator_id()
        {
            return Err(ForgeProposalHistoryError::ParentMismatch);
        }
        let expected = derive_row_id(
            self.source.id(),
            self.state.history_before.id(),
            &self.state.parent_artifact_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalHistoryError::RowIdentityMismatch)
        }
    }
}

fn derive_row_id(
    source_row_id: &ContentId,
    history_id: &ContentId,
    parent_artifact_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-conditioned-proposal-observation-row.v1",
        [
            source_row_id.as_str().as_bytes(),
            history_id.as_str().as_bytes(),
            parent_artifact_id.as_str().as_bytes(),
        ],
    )
}

/// Run-scoped proposal observations with generation-boundary accepted history attached to every row.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeConditionedProposalObservationTable {
    id: ContentId,
    run_id: ContentId,
    generator_id: ContentId,
    source_table_id: ContentId,
    sequence_id: ContentId,
    family_trial_set_id: ContentId,
    rows: Vec<ForgeConditionedProposalObservationRow>,
}

impl ForgeConditionedProposalObservationTable {
    pub fn from_table(
        source: &ForgeProposalObservationTable,
        sequence: &ForgeSearchFamilySequence,
    ) -> Result<Self, ForgeProposalHistoryError> {
        source.validate()?;
        sequence.validate()?;
        if source.run_id() != sequence.run_id() {
            return Err(ForgeProposalHistoryError::RunMismatch);
        }
        if source.generator_id() != sequence.generator_id() {
            return Err(ForgeProposalHistoryError::GeneratorMismatch);
        }
        if source.family_trial_set_id() != sequence.source_family_trial_set_id() {
            return Err(ForgeProposalHistoryError::TrialSetMismatch);
        }

        let rows = source
            .rows()
            .iter()
            .map(|row| {
                let state = ForgeProposalGenerationState::from_sequence(sequence, row.generation())?;
                ForgeConditionedProposalObservationRow::new(row.clone(), state)
            })
            .collect::<Result<Vec<_>, ForgeProposalHistoryError>>()?;
        let id = derive_table_id(source.id(), sequence.id(), &rows);
        let table = Self {
            id,
            run_id: source.run_id().clone(),
            generator_id: source.generator_id().clone(),
            source_table_id: source.id().clone(),
            sequence_id: sequence.id().clone(),
            family_trial_set_id: source.family_trial_set_id().clone(),
            rows,
        };
        table.validate_for(source, sequence)?;
        Ok(table)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn run_id(&self) -> &ContentId { &self.run_id }
    pub fn generator_id(&self) -> &ContentId { &self.generator_id }
    pub fn source_table_id(&self) -> &ContentId { &self.source_table_id }
    pub fn sequence_id(&self) -> &ContentId { &self.sequence_id }
    pub fn family_trial_set_id(&self) -> &ContentId { &self.family_trial_set_id }
    pub fn rows(&self) -> &[ForgeConditionedProposalObservationRow] { &self.rows }

    pub fn validate(&self) -> Result<(), ForgeProposalHistoryError> {
        let mut previous_key: Option<(u64, u64)> = None;
        for row in &self.rows {
            row.validate()?;
            if row.source().family_id().generator_id() != &self.generator_id {
                return Err(ForgeProposalHistoryError::GeneratorMismatch);
            }
            let key = (
                row.source().attempt_id().ordinal(),
                row.source().family_index(),
            );
            if previous_key.is_some_and(|previous| previous >= key) {
                return Err(ForgeProposalHistoryError::NonCanonicalTable);
            }
            previous_key = Some(key);
        }
        let expected = derive_table_id_from_fields(&self.source_table_id, &self.sequence_id, &self.rows);
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalHistoryError::TableIdentityMismatch)
        }
    }

    pub fn validate_for(
        &self,
        source: &ForgeProposalObservationTable,
        sequence: &ForgeSearchFamilySequence,
    ) -> Result<(), ForgeProposalHistoryError> {
        source.validate()?;
        sequence.validate()?;
        self.validate()?;
        if self.run_id != *source.run_id()
            || self.run_id != *sequence.run_id()
            || self.generator_id != *source.generator_id()
            || self.generator_id != *sequence.generator_id()
            || self.source_table_id != *source.id()
            || self.sequence_id != *sequence.id()
            || self.family_trial_set_id != *source.family_trial_set_id()
            || self.family_trial_set_id != *sequence.source_family_trial_set_id()
            || self.rows.len() != source.rows().len()
        {
            return Err(ForgeProposalHistoryError::NonCanonicalTable);
        }
        for (conditioned, original) in self.rows.iter().zip(source.rows()) {
            if conditioned.source() != original {
                return Err(ForgeProposalHistoryError::NonCanonicalTable);
            }
            let expected_state =
                ForgeProposalGenerationState::from_sequence(sequence, original.generation())?;
            if conditioned.state() != &expected_state {
                return Err(ForgeProposalHistoryError::NonCanonicalTable);
            }
        }
        Ok(())
    }
}

fn derive_table_id(
    source_table_id: &ContentId,
    sequence_id: &ContentId,
    rows: &[ForgeConditionedProposalObservationRow],
) -> ContentId {
    derive_table_id_from_fields(source_table_id, sequence_id, rows)
}

fn derive_table_id_from_fields(
    source_table_id: &ContentId,
    sequence_id: &ContentId,
    rows: &[ForgeConditionedProposalObservationRow],
) -> ContentId {
    let count = (rows.len() as u64).to_be_bytes();
    let mut parts = vec![
        source_table_id.as_str().as_bytes().to_vec(),
        sequence_id.as_str().as_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        rows.iter()
            .map(|row| row.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-conditioned-proposal-observation-table.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family_learning::ForgeTransformationFamilyId;

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    #[test]
    fn history_suffix_is_explicitly_bounded_by_order() {
        let generator = cid("generator", "g");
        let families = ["A", "B", "C"]
            .into_iter()
            .map(|name| ForgeTransformationFamilyId::new(generator.clone(), name).unwrap())
            .collect::<Vec<_>>();
        let state = ForgeProposalGenerationState {
            generation: 3,
            parent_artifact_id: cid("artifact", "p"),
            history_before: ForgeFamilyHistory::new(generator, families.clone()).unwrap(),
        };
        let order = ForgeHistoryOrder::new(2).unwrap();
        let suffix = state.history_suffix(&order).unwrap();
        assert_eq!(suffix.families(), &families[1..]);
    }
}
