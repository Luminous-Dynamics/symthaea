// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Model-ready, authority-free proposal observation rows.
//!
//! This module deliberately stops before estimation. It projects one already-qualified proposal
//! history into one canonical row for every `(attempt, policy family)` pair, including families with
//! zero eligible sites and families that were eligible but not selected. Exposure remains exact
//! integer evidence; no floating propensity, inverse-propensity estimator, causal effect, ranking,
//! search steering, promotion, or runtime authority is computed here.

use crate::family_learning::{
    ForgeFamilyLearningError, ForgeFamilyTrialSet, ForgeTransformationFamilyId,
};
use crate::proposal_exposure::{
    ForgeProposalDecision, ForgeProposalExposureError, ForgeProposalPolicy,
};
use crate::proposal_qualification::{
    ForgeProposalQualificationError, ForgeQualifiedProposalEvidence,
};
use crate::trace::{ForgeAttemptId, ForgeTraceEvent};
use crate::trials::ForgeTrialOutcome;
use serde::Serialize;
use std::collections::BTreeMap;
use symthaea_algorithms::discovery::{DiscoveryError, DiscoveryRun};
use symthaea_algorithms::ledger::DiscoveryEventKind;
use symthaea_algorithms::observation::ObservationStore;
use symthaea_algorithms::{ContentId, ImplementationRecord, RegistryError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalDatasetError {
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error(transparent)]
    Exposure(#[from] ForgeProposalExposureError),
    #[error(transparent)]
    Qualification(#[from] ForgeProposalQualificationError),
    #[error(transparent)]
    Family(#[from] ForgeFamilyLearningError),
    #[error("qualified proposal evidence does not belong to the supplied semantic run")]
    RunMismatch,
    #[error("qualified proposal evidence does not belong to the supplied baseline implementation")]
    BaselineMismatch,
    #[error("qualified proposal evidence does not belong to the supplied proposal policy")]
    PolicyMismatch,
    #[error("qualified proposal evidence does not bind the reconstructed family trial set")]
    TrialSetMismatch,
    #[error("proposal-stage trace contains a duplicate or malformed attempt event")]
    ProposalEventMismatch,
    #[error("concrete family trial disagrees with the selected proposal family")]
    TrialSelectionMismatch,
    #[error("proposal observation count cannot be represented")]
    CountOverflow,
    #[error("proposal observation row is internally inconsistent")]
    InvalidRow,
    #[error("proposal observation row identity does not match canonical fields")]
    RowIdentityMismatch,
    #[error("proposal observation table is not canonical")]
    NonCanonicalTable,
    #[error("proposal observation table identity does not match canonical fields")]
    TableIdentityMismatch,
}

/// Exact proposal decision for one family on one attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum ForgeProposalRowDecision {
    /// No family had any eligible site on this attempt.
    NoEligibleSites,
    /// This family was not the selected family. Its own eligible-site count may be zero or nonzero.
    NotSelected,
    /// This family was selected at the exact local/global site coordinates.
    Selected {
        site_index: u64,
        global_pair_index: u64,
    },
}

/// What happened after this family-level proposal decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "kind", content = "trial_outcome")]
pub enum ForgeProposalRowOutcome {
    /// This family was not selected, including the all-zero no-eligible-sites case.
    Unselected,
    /// A pair was selected but did not produce a distinct candidate artifact/trial.
    SelectedNoCandidate,
    /// A distinct candidate existed and reached this exact concrete Forge trial outcome.
    Concrete(ForgeTrialOutcome),
}

/// One canonical `(attempt, family)` observation with exact proposal exposure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalObservationRow {
    id: ContentId,
    exposure_id: ContentId,
    attempt_id: ForgeAttemptId,
    generation: u64,
    parent_artifact_id: ContentId,
    family_index: u64,
    family_id: ForgeTransformationFamilyId,
    eligible_sites: u64,
    total_eligible_sites: u64,
    decision: ForgeProposalRowDecision,
    outcome: ForgeProposalRowOutcome,
}

impl ForgeProposalObservationRow {
    #[allow(clippy::too_many_arguments)]
    fn new(
        exposure_id: ContentId,
        attempt_id: ForgeAttemptId,
        parent_artifact_id: ContentId,
        family_index: u64,
        family_id: ForgeTransformationFamilyId,
        eligible_sites: u64,
        total_eligible_sites: u64,
        decision: ForgeProposalRowDecision,
        outcome: ForgeProposalRowOutcome,
    ) -> Result<Self, ForgeProposalDatasetError> {
        let generation = attempt_id.generation();
        let mut row = Self {
            id: ContentId::derive("symthaea.forge-proposal-observation-row.uninitialized", [
                b"v1".as_slice(),
            ]),
            exposure_id,
            attempt_id,
            generation,
            parent_artifact_id,
            family_index,
            family_id,
            eligible_sites,
            total_eligible_sites,
            decision,
            outcome,
        };
        row.id = derive_row_id(&row);
        row.validate()?;
        Ok(row)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn exposure_id(&self) -> &ContentId { &self.exposure_id }
    pub fn attempt_id(&self) -> &ForgeAttemptId { &self.attempt_id }
    pub fn generation(&self) -> u64 { self.generation }
    pub fn parent_artifact_id(&self) -> &ContentId { &self.parent_artifact_id }
    pub fn family_index(&self) -> u64 { self.family_index }
    pub fn family_id(&self) -> &ForgeTransformationFamilyId { &self.family_id }
    pub fn eligible_sites(&self) -> u64 { self.eligible_sites }
    pub fn total_eligible_sites(&self) -> u64 { self.total_eligible_sites }
    pub fn decision(&self) -> &ForgeProposalRowDecision { &self.decision }
    pub fn outcome(&self) -> ForgeProposalRowOutcome { self.outcome }

    /// Exact family proposal mass `(eligible family sites, all eligible sites)`.
    /// A zero-site family in a non-empty opportunity set returns `(0, total)` rather than `None`.
    pub fn family_probability_weight(&self) -> Option<(u64, u64)> {
        (self.total_eligible_sites > 0)
            .then_some((self.eligible_sites, self.total_eligible_sites))
    }

    /// Exact selected-pair proposal mass `(1, all eligible sites)` when this row was selected.
    pub fn selected_pair_probability_weight(&self) -> Option<(u64, u64)> {
        matches!(self.decision, ForgeProposalRowDecision::Selected { .. })
            .then_some((1, self.total_eligible_sites))
    }

    pub fn validate(&self) -> Result<(), ForgeProposalDatasetError> {
        self.attempt_id
            .validate()
            .map_err(|_| ForgeProposalDatasetError::InvalidRow)?;
        self.family_id.validate()?;
        if self.generation != self.attempt_id.generation()
            || self.eligible_sites > self.total_eligible_sites
        {
            return Err(ForgeProposalDatasetError::InvalidRow);
        }
        match (&self.decision, self.outcome) {
            (ForgeProposalRowDecision::NoEligibleSites, ForgeProposalRowOutcome::Unselected)
                if self.total_eligible_sites == 0 && self.eligible_sites == 0 => {}
            (ForgeProposalRowDecision::NotSelected, ForgeProposalRowOutcome::Unselected)
                if self.total_eligible_sites > 0 => {}
            (
                ForgeProposalRowDecision::Selected {
                    site_index,
                    global_pair_index,
                },
                ForgeProposalRowOutcome::SelectedNoCandidate
                | ForgeProposalRowOutcome::Concrete(_),
            ) if self.total_eligible_sites > 0
                && *site_index < self.eligible_sites
                && *global_pair_index < self.total_eligible_sites => {}
            _ => return Err(ForgeProposalDatasetError::InvalidRow),
        }
        if derive_row_id(self) == self.id {
            Ok(())
        } else {
            Err(ForgeProposalDatasetError::RowIdentityMismatch)
        }
    }
}

fn row_outcome_tag(outcome: ForgeProposalRowOutcome) -> &'static [u8] {
    match outcome {
        ForgeProposalRowOutcome::Unselected => b"unselected",
        ForgeProposalRowOutcome::SelectedNoCandidate => b"selected-no-candidate",
        ForgeProposalRowOutcome::Concrete(ForgeTrialOutcome::RejectedCompilation) => {
            b"concrete-rejected-compilation"
        }
        ForgeProposalRowOutcome::Concrete(ForgeTrialOutcome::RejectedCorrectness) => {
            b"concrete-rejected-correctness"
        }
        ForgeProposalRowOutcome::Concrete(ForgeTrialOutcome::RejectedEvaluation) => {
            b"concrete-rejected-evaluation"
        }
        ForgeProposalRowOutcome::Concrete(ForgeTrialOutcome::ValidNotSelected) => {
            b"concrete-valid-not-selected"
        }
        ForgeProposalRowOutcome::Concrete(ForgeTrialOutcome::SelectedForContinuation) => {
            b"concrete-selected-for-continuation"
        }
        ForgeProposalRowOutcome::Concrete(ForgeTrialOutcome::Interrupted) => b"concrete-interrupted",
    }
}

fn derive_row_id(row: &ForgeProposalObservationRow) -> ContentId {
    let generation = row.generation.to_be_bytes();
    let family_index = row.family_index.to_be_bytes();
    let eligible = row.eligible_sites.to_be_bytes();
    let total = row.total_eligible_sites.to_be_bytes();
    let mut parts = vec![
        row.exposure_id.as_str().as_bytes().to_vec(),
        row.attempt_id.as_content_id().as_str().as_bytes().to_vec(),
        generation.to_vec(),
        row.parent_artifact_id.as_str().as_bytes().to_vec(),
        family_index.to_vec(),
        row.family_id.as_content_id().as_str().as_bytes().to_vec(),
        eligible.to_vec(),
        total.to_vec(),
    ];
    match &row.decision {
        ForgeProposalRowDecision::NoEligibleSites => parts.push(b"no-eligible-sites".to_vec()),
        ForgeProposalRowDecision::NotSelected => parts.push(b"not-selected".to_vec()),
        ForgeProposalRowDecision::Selected {
            site_index,
            global_pair_index,
        } => {
            parts.push(b"selected".to_vec());
            parts.push(site_index.to_be_bytes().to_vec());
            parts.push(global_pair_index.to_be_bytes().to_vec());
        }
    }
    parts.push(row_outcome_tag(row.outcome).to_vec());
    ContentId::derive(
        "symthaea.forge-proposal-observation-row.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProposalEventKind {
    CandidateGenerated,
    GeneratorNoOp,
}

/// Canonical run-scoped observation table. Rows are ordered by attempt ordinal and then by the
/// frozen proposal-policy family order.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalObservationTable {
    id: ContentId,
    run_id: ContentId,
    baseline_implementation_id: ContentId,
    generator_id: ContentId,
    policy_id: ContentId,
    qualified_evidence_id: ContentId,
    family_trial_set_id: ContentId,
    families: Vec<ForgeTransformationFamilyId>,
    proposal_stage_attempts: u64,
    rows: Vec<ForgeProposalObservationRow>,
}

impl ForgeProposalObservationTable {
    #[allow(clippy::too_many_arguments)]
    pub fn from_qualified(
        run: &DiscoveryRun,
        baseline: &ImplementationRecord,
        trace: &[ForgeTraceEvent],
        observations: &ObservationStore,
        policy: &ForgeProposalPolicy,
        qualified: &ForgeQualifiedProposalEvidence,
    ) -> Result<Self, ForgeProposalDatasetError> {
        run.validate()?;
        baseline.validate()?;
        policy.validate()?;
        qualified.validate_identity()?;
        qualified.semantic_archive().validate_for(run, policy)?;

        if qualified.run_id() != &run.id || qualified.qualification().run_id() != &run.id {
            return Err(ForgeProposalDatasetError::RunMismatch);
        }
        if qualified.baseline_implementation_id() != baseline.id.as_content_id() {
            return Err(ForgeProposalDatasetError::BaselineMismatch);
        }
        if qualified.qualification().policy_id() != policy.id()
            || qualified.semantic_archive().policy_id() != policy.id()
            || policy.generator_id() != &run.generator_id
        {
            return Err(ForgeProposalDatasetError::PolicyMismatch);
        }

        let family_trials = ForgeFamilyTrialSet::from_trace(run, baseline, trace, observations)?;
        if family_trials.id() != qualified.qualification().family_trial_set_id()
            || family_trials.source_batch_id() != qualified.qualification().concrete_batch_id()
        {
            return Err(ForgeProposalDatasetError::TrialSetMismatch);
        }

        let expected_attempts = u64::try_from(qualified.semantic_archive().exposures().len())
            .map_err(|_| ForgeProposalDatasetError::CountOverflow)?;
        if expected_attempts != qualified.qualification().proposal_stage_attempts() {
            return Err(ForgeProposalDatasetError::NonCanonicalTable);
        }

        let mut proposal_events = BTreeMap::<String, ProposalEventKind>::new();
        for event in trace {
            let kind = match event.kind {
                DiscoveryEventKind::CandidateGenerated => ProposalEventKind::CandidateGenerated,
                DiscoveryEventKind::GeneratorNoOp => ProposalEventKind::GeneratorNoOp,
                _ => continue,
            };
            let attempt = event
                .attempt_id
                .as_ref()
                .ok_or(ForgeProposalDatasetError::ProposalEventMismatch)?;
            if proposal_events
                .insert(attempt.as_content_id().as_str().to_string(), kind)
                .is_some()
            {
                return Err(ForgeProposalDatasetError::ProposalEventMismatch);
            }
        }

        let mut trials_by_attempt = family_trials
            .trials()
            .iter()
            .map(|trial| {
                (
                    trial.attempt_id().as_content_id().as_str().to_string(),
                    trial,
                )
            })
            .collect::<BTreeMap<_, _>>();

        let mut rows = Vec::new();
        for exposure in qualified.semantic_archive().exposures() {
            let attempt_key = exposure.attempt_id().as_content_id().as_str();
            let proposal_event = proposal_events
                .remove(attempt_key)
                .ok_or(ForgeProposalDatasetError::ProposalEventMismatch)?;
            let concrete_trial = trials_by_attempt.remove(attempt_key);

            match (exposure.decision(), proposal_event, concrete_trial) {
                (
                    ForgeProposalDecision::NoEligibleSites,
                    ProposalEventKind::GeneratorNoOp,
                    None,
                ) => {}
                (
                    ForgeProposalDecision::Selected(selection),
                    ProposalEventKind::CandidateGenerated,
                    Some(trial),
                ) if selection.family_id() == trial.family_id() => {}
                (
                    ForgeProposalDecision::Selected(_),
                    ProposalEventKind::GeneratorNoOp,
                    None,
                ) => {}
                _ => return Err(ForgeProposalDatasetError::TrialSelectionMismatch),
            }

            for (family_index, (family, opportunity)) in policy
                .families()
                .iter()
                .zip(exposure.opportunities())
                .enumerate()
            {
                if opportunity.family_id() != family {
                    return Err(ForgeProposalDatasetError::PolicyMismatch);
                }
                let family_index = u64::try_from(family_index)
                    .map_err(|_| ForgeProposalDatasetError::CountOverflow)?;
                let selected = match exposure.decision() {
                    ForgeProposalDecision::NoEligibleSites => None,
                    ForgeProposalDecision::Selected(selection)
                        if selection.family_id() == family => Some(selection),
                    ForgeProposalDecision::Selected(_) => None,
                };
                let decision = match (exposure.decision(), selected) {
                    (ForgeProposalDecision::NoEligibleSites, _) => {
                        ForgeProposalRowDecision::NoEligibleSites
                    }
                    (_, Some(selection)) => ForgeProposalRowDecision::Selected {
                        site_index: selection.site_index(),
                        global_pair_index: selection.global_pair_index(),
                    },
                    _ => ForgeProposalRowDecision::NotSelected,
                };
                let outcome = match selected {
                    None => ForgeProposalRowOutcome::Unselected,
                    Some(_) => match concrete_trial {
                        Some(trial) => ForgeProposalRowOutcome::Concrete(trial.outcome()),
                        None => ForgeProposalRowOutcome::SelectedNoCandidate,
                    },
                };
                rows.push(ForgeProposalObservationRow::new(
                    exposure.id().clone(),
                    exposure.attempt_id().clone(),
                    exposure.parent_artifact_id().clone(),
                    family_index,
                    family.clone(),
                    opportunity.eligible_sites(),
                    exposure.total_eligible_sites(),
                    decision,
                    outcome,
                )?);
            }
        }

        if !proposal_events.is_empty() || !trials_by_attempt.is_empty() {
            return Err(ForgeProposalDatasetError::ProposalEventMismatch);
        }

        let families = policy.families().to_vec();
        let baseline_implementation_id = baseline.id.as_content_id().clone();
        let generator_id = run.generator_id.clone();
        let policy_id = policy.id().clone();
        let qualified_evidence_id = qualified.id().clone();
        let family_trial_set_id = family_trials.id().clone();
        let proposal_stage_attempts = qualified.qualification().proposal_stage_attempts();
        let id = derive_table_id(
            &run.id,
            &baseline_implementation_id,
            &generator_id,
            &policy_id,
            &qualified_evidence_id,
            &family_trial_set_id,
            &families,
            proposal_stage_attempts,
            &rows,
        );
        let table = Self {
            id,
            run_id: run.id.clone(),
            baseline_implementation_id,
            generator_id,
            policy_id,
            qualified_evidence_id,
            family_trial_set_id,
            families,
            proposal_stage_attempts,
            rows,
        };
        table.validate()?;
        Ok(table)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn run_id(&self) -> &ContentId { &self.run_id }
    pub fn baseline_implementation_id(&self) -> &ContentId { &self.baseline_implementation_id }
    pub fn generator_id(&self) -> &ContentId { &self.generator_id }
    pub fn policy_id(&self) -> &ContentId { &self.policy_id }
    pub fn qualified_evidence_id(&self) -> &ContentId { &self.qualified_evidence_id }
    pub fn family_trial_set_id(&self) -> &ContentId { &self.family_trial_set_id }
    pub fn families(&self) -> &[ForgeTransformationFamilyId] { &self.families }
    pub fn proposal_stage_attempts(&self) -> u64 { self.proposal_stage_attempts }
    pub fn rows(&self) -> &[ForgeProposalObservationRow] { &self.rows }

    pub fn rows_for_family(
        &self,
        family: &ForgeTransformationFamilyId,
    ) -> impl Iterator<Item = &ForgeProposalObservationRow> {
        self.rows.iter().filter(move |row| row.family_id() == family)
    }

    pub fn rows_for_attempt(
        &self,
        attempt: &ForgeAttemptId,
    ) -> impl Iterator<Item = &ForgeProposalObservationRow> {
        self.rows.iter().filter(move |row| row.attempt_id() == attempt)
    }

    pub fn validate(&self) -> Result<(), ForgeProposalDatasetError> {
        for family in &self.families {
            family.validate()?;
            if family.generator_id() != &self.generator_id {
                return Err(ForgeProposalDatasetError::NonCanonicalTable);
            }
        }
        if self.families.is_empty() {
            return Err(ForgeProposalDatasetError::NonCanonicalTable);
        }
        let family_count = u64::try_from(self.families.len())
            .map_err(|_| ForgeProposalDatasetError::CountOverflow)?;
        let expected_rows = self
            .proposal_stage_attempts
            .checked_mul(family_count)
            .ok_or(ForgeProposalDatasetError::CountOverflow)?;
        if u64::try_from(self.rows.len()).map_err(|_| ForgeProposalDatasetError::CountOverflow)?
            != expected_rows
        {
            return Err(ForgeProposalDatasetError::NonCanonicalTable);
        }

        let chunk_size = self.families.len();
        let mut previous_ordinal = None;
        for chunk in self.rows.chunks_exact(chunk_size) {
            let first = &chunk[0];
            let ordinal = first.attempt_id().ordinal();
            if previous_ordinal.is_some_and(|previous| ordinal <= previous) {
                return Err(ForgeProposalDatasetError::NonCanonicalTable);
            }
            previous_ordinal = Some(ordinal);

            let mut total_sum = 0u64;
            let mut selected_count = 0u64;
            let mut selected_global = None;
            let mut offset = 0u64;
            for (index, (row, family)) in chunk.iter().zip(&self.families).enumerate() {
                row.validate()?;
                let index_u64 = u64::try_from(index)
                    .map_err(|_| ForgeProposalDatasetError::CountOverflow)?;
                if row.family_index() != index_u64
                    || row.family_id() != family
                    || row.attempt_id() != first.attempt_id()
                    || row.exposure_id() != first.exposure_id()
                    || row.generation() != first.generation()
                    || row.parent_artifact_id() != first.parent_artifact_id()
                    || row.total_eligible_sites() != first.total_eligible_sites()
                {
                    return Err(ForgeProposalDatasetError::NonCanonicalTable);
                }
                total_sum = total_sum
                    .checked_add(row.eligible_sites())
                    .ok_or(ForgeProposalDatasetError::CountOverflow)?;
                if let ForgeProposalRowDecision::Selected {
                    site_index,
                    global_pair_index,
                } = row.decision()
                {
                    selected_count = selected_count
                        .checked_add(1)
                        .ok_or(ForgeProposalDatasetError::CountOverflow)?;
                    let expected_global = offset
                        .checked_add(*site_index)
                        .ok_or(ForgeProposalDatasetError::CountOverflow)?;
                    if expected_global != *global_pair_index {
                        return Err(ForgeProposalDatasetError::NonCanonicalTable);
                    }
                    selected_global = Some(*global_pair_index);
                }
                offset = offset
                    .checked_add(row.eligible_sites())
                    .ok_or(ForgeProposalDatasetError::CountOverflow)?;
            }
            if total_sum != first.total_eligible_sites() {
                return Err(ForgeProposalDatasetError::NonCanonicalTable);
            }
            if first.total_eligible_sites() == 0 {
                if selected_count != 0
                    || chunk.iter().any(|row| {
                        !matches!(row.decision(), ForgeProposalRowDecision::NoEligibleSites)
                    })
                {
                    return Err(ForgeProposalDatasetError::NonCanonicalTable);
                }
            } else if selected_count != 1
                || selected_global.is_none_or(|index| index >= first.total_eligible_sites())
                || chunk.iter().any(|row| {
                    matches!(row.decision(), ForgeProposalRowDecision::NoEligibleSites)
                })
            {
                return Err(ForgeProposalDatasetError::NonCanonicalTable);
            }
        }

        let expected = derive_table_id(
            &self.run_id,
            &self.baseline_implementation_id,
            &self.generator_id,
            &self.policy_id,
            &self.qualified_evidence_id,
            &self.family_trial_set_id,
            &self.families,
            self.proposal_stage_attempts,
            &self.rows,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalDatasetError::TableIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_table_id(
    run_id: &ContentId,
    baseline_implementation_id: &ContentId,
    generator_id: &ContentId,
    policy_id: &ContentId,
    qualified_evidence_id: &ContentId,
    family_trial_set_id: &ContentId,
    families: &[ForgeTransformationFamilyId],
    proposal_stage_attempts: u64,
    rows: &[ForgeProposalObservationRow],
) -> ContentId {
    let family_count = (families.len() as u64).to_be_bytes();
    let attempts = proposal_stage_attempts.to_be_bytes();
    let row_count = (rows.len() as u64).to_be_bytes();
    let mut parts = vec![
        run_id.as_str().as_bytes().to_vec(),
        baseline_implementation_id.as_str().as_bytes().to_vec(),
        generator_id.as_str().as_bytes().to_vec(),
        policy_id.as_str().as_bytes().to_vec(),
        qualified_evidence_id.as_str().as_bytes().to_vec(),
        family_trial_set_id.as_str().as_bytes().to_vec(),
        family_count.to_vec(),
    ];
    parts.extend(
        families
            .iter()
            .map(|family| family.as_content_id().as_str().as_bytes().to_vec()),
    );
    parts.push(attempts.to_vec());
    parts.push(row_count.to_vec());
    parts.extend(
        rows.iter()
            .map(|row| row.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-proposal-observation-table.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::full_source_artifact_id;
    use crate::mutations::{ComparisonOperatorSwap, Mutator, NumericLiteralPerturb};
    use crate::observations as forge_observations;
    use crate::proposal_recording::proposal_policy_for_mutator;
    use crate::proposal_trace::{ForgeRawProposalArchive, ForgeRawProposalRecord};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use symthaea_algorithms::discovery::{DiscoveryPolicy, SearchBudget};
    use symthaea_algorithms::{
        AlgorithmProvenance, AlgorithmRecord, DeterminismRequirement, DiscoveryRisk, ProblemSpec,
        SemanticGuarantee,
    };

    fn semantic_fixture(
        baseline_artifact: ContentId,
        mutator: &Mutator,
    ) -> (DiscoveryRun, ImplementationRecord, ForgeProposalPolicy) {
        let problem = ProblemSpec::new(
            "proposal-observation-table-test",
            "Exact fixture.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["matches oracle".into()],
            DiscoveryRisk::Ordinary,
        )
        .unwrap();
        let algorithm = AlgorithmRecord::new(
            problem.id.clone(),
            "baseline",
            "fixture baseline",
            AlgorithmProvenance::HumanAuthored,
        )
        .unwrap();
        let baseline = ImplementationRecord::new(
            problem.id.clone(),
            algorithm.id,
            "repo://src/f.rs",
            baseline_artifact,
            None,
        )
        .unwrap();
        let run = DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            ContentId::derive("generator", [b"forge-proposal-observation-v1".as_slice()]),
            "abc123",
            SearchBudget::new(8, 2, 8).unwrap(),
            7,
        )
        .unwrap();
        let policy = proposal_policy_for_mutator(&run, mutator).unwrap();
        (run, baseline, policy)
    }

    fn no_candidate_history(
        baseline_artifact: ContentId,
        run: &DiscoveryRun,
        mutator: &Mutator,
        body: &mut syn::Block,
        reason: &str,
    ) -> (Vec<ForgeTraceEvent>, ObservationStore, ForgeRawProposalArchive) {
        let attempt = ForgeAttemptId::derive(&baseline_artifact, run.seed, 0, 0);
        let mut rng = StdRng::seed_from_u64(11);
        let recorded = mutator.mutate_one_recorded(body, &mut rng);
        let raw_record = ForgeRawProposalRecord::from_recorded(
            attempt.clone(),
            baseline_artifact.clone(),
            &recorded,
        )
        .unwrap();
        let no_candidate = forge_observations::no_candidate(
            &attempt,
            reason,
            &baseline_artifact,
        )
        .unwrap();
        let summary = forge_observations::search_summary(1, 1, 0, 0, 0, 0, 0, None, None).unwrap();
        let trace = vec![
            ForgeTraceEvent::no_candidate(attempt, 0, no_candidate.id().clone()),
            ForgeTraceEvent::completed(summary.id().clone()),
        ];
        let observations = ObservationStore::from_objects(vec![no_candidate, summary]).unwrap();
        let raw = ForgeRawProposalArchive::from_records(vec![raw_record]).unwrap();
        (trace, observations, raw)
    }

    #[test]
    fn no_eligible_attempt_keeps_one_zero_exposure_row_per_policy_family() {
        let artifact = full_source_artifact_id("fn f() -> bool { true }\n");
        let mutator = Mutator::new(vec![Box::new(ComparisonOperatorSwap)]);
        let (run, baseline, policy) = semantic_fixture(artifact.clone(), &mutator);
        let mut body: syn::Block = syn::parse_str("{ true }").unwrap();
        let (trace, observations, raw) = no_candidate_history(
            artifact,
            &run,
            &mutator,
            &mut body,
            "no-eligible-ast-mutation",
        );
        let qualified = ForgeQualifiedProposalEvidence::from_raw(
            &run,
            &baseline,
            &trace,
            &observations,
            &policy,
            &raw,
        )
        .unwrap();
        let table = ForgeProposalObservationTable::from_qualified(
            &run,
            &baseline,
            &trace,
            &observations,
            &policy,
            &qualified,
        )
        .unwrap();
        assert_eq!(table.rows().len(), 1);
        let row = &table.rows()[0];
        assert_eq!(row.family_probability_weight(), None);
        assert!(matches!(row.decision(), ForgeProposalRowDecision::NoEligibleSites));
        assert_eq!(row.outcome(), ForgeProposalRowOutcome::Unselected);
        table.validate().unwrap();
    }

    #[test]
    fn selected_no_candidate_is_distinct_from_no_eligible_sites() {
        let artifact = full_source_artifact_id("fn f() -> i64 { 0 }\n");
        let mutator = Mutator::new(vec![Box::new(NumericLiteralPerturb { max_fraction: 0.1 })]);
        let (run, baseline, policy) = semantic_fixture(artifact.clone(), &mutator);
        let mut body: syn::Block = syn::parse_str("{ 0 }").unwrap();
        let (trace, observations, raw) = no_candidate_history(
            artifact,
            &run,
            &mutator,
            &mut body,
            "mutation-rendered-identical-source",
        );
        let qualified = ForgeQualifiedProposalEvidence::from_raw(
            &run,
            &baseline,
            &trace,
            &observations,
            &policy,
            &raw,
        )
        .unwrap();
        let table = ForgeProposalObservationTable::from_qualified(
            &run,
            &baseline,
            &trace,
            &observations,
            &policy,
            &qualified,
        )
        .unwrap();
        let row = &table.rows()[0];
        assert_eq!(row.family_probability_weight(), Some((1, 1)));
        assert_eq!(row.selected_pair_probability_weight(), Some((1, 1)));
        assert!(matches!(row.decision(), ForgeProposalRowDecision::Selected { .. }));
        assert_eq!(row.outcome(), ForgeProposalRowOutcome::SelectedNoCandidate);
    }

    #[test]
    fn zero_site_nonselected_family_remains_in_selected_attempt() {
        let artifact = full_source_artifact_id("fn f() -> i64 { 0 }\n");
        let mutator = Mutator::new(vec![
            Box::new(NumericLiteralPerturb { max_fraction: 0.1 }),
            Box::new(ComparisonOperatorSwap),
        ]);
        let (run, baseline, policy) = semantic_fixture(artifact.clone(), &mutator);
        let mut body: syn::Block = syn::parse_str("{ 0 }").unwrap();
        let (trace, observations, raw) = no_candidate_history(
            artifact,
            &run,
            &mutator,
            &mut body,
            "mutation-rendered-identical-source",
        );
        let qualified = ForgeQualifiedProposalEvidence::from_raw(
            &run,
            &baseline,
            &trace,
            &observations,
            &policy,
            &raw,
        )
        .unwrap();
        let table = ForgeProposalObservationTable::from_qualified(
            &run,
            &baseline,
            &trace,
            &observations,
            &policy,
            &qualified,
        )
        .unwrap();
        assert_eq!(table.rows().len(), 2);
        assert!(matches!(table.rows()[0].decision(), ForgeProposalRowDecision::Selected { .. }));
        assert_eq!(table.rows()[1].eligible_sites(), 0);
        assert_eq!(table.rows()[1].family_probability_weight(), Some((0, 1)));
        assert!(matches!(table.rows()[1].decision(), ForgeProposalRowDecision::NotSelected));
        assert_eq!(table.rows()[1].outcome(), ForgeProposalRowOutcome::Unselected);
        table.validate().unwrap();
    }
}
