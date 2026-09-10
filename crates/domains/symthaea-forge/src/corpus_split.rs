// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Outcome-independent train/validation/holdout partitions for Forge sequence learning.
//!
//! Partition rank is derived only from a frozen salt and semantic `DiscoveryRun` identity. It does
//! not use the outcome-bearing cohort or sequence-batch identity, preventing observed search results
//! from reshuffling runs between training and holdout. The final manifest still binds the exact
//! cohort so the dataset snapshot itself is auditable.

use crate::sequence_stats::{ExactContextForgeSequenceCohort, ForgeSequenceStatsError};
use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeCorpusSplitError {
    #[error(transparent)]
    SequenceStats(#[from] ForgeSequenceStatsError),
    #[error("training, validation, and holdout partitions must each contain at least one run")]
    EmptyPartition,
    #[error("corpus split run-count arithmetic overflow")]
    CountOverflow,
    #[error("split specification run count does not equal the exact-context sequence cohort")]
    RunCountMismatch,
    #[error("split specification identity does not match canonical fields")]
    SpecIdentityMismatch,
    #[error("corpus assignment identity does not match canonical fields")]
    AssignmentIdentityMismatch,
    #[error("corpus split contains a duplicate run")]
    DuplicateRun,
    #[error("corpus split role counts do not match the frozen specification")]
    RoleCountMismatch,
    #[error("corpus split does not cover exactly the supplied sequence cohort")]
    CohortCoverageMismatch,
    #[error("corpus split identity does not match canonical fields")]
    SplitIdentityMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForgeCorpusRole {
    Training,
    Validation,
    Holdout,
}

impl ForgeCorpusRole {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::Training => b"training",
            Self::Validation => b"validation",
            Self::Holdout => b"holdout",
        }
    }
}

/// Frozen exact run counts plus an externally chosen content-addressed partition salt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeCorpusSplitSpec {
    id: ContentId,
    training_runs: u64,
    validation_runs: u64,
    holdout_runs: u64,
    salt_id: ContentId,
}

impl ForgeCorpusSplitSpec {
    pub fn new(
        training_runs: u64,
        validation_runs: u64,
        holdout_runs: u64,
        salt_id: ContentId,
    ) -> Result<Self, ForgeCorpusSplitError> {
        if training_runs == 0 || validation_runs == 0 || holdout_runs == 0 {
            return Err(ForgeCorpusSplitError::EmptyPartition);
        }
        let total = training_runs
            .checked_add(validation_runs)
            .and_then(|value| value.checked_add(holdout_runs))
            .ok_or(ForgeCorpusSplitError::CountOverflow)?;
        let id = derive_spec_id(training_runs, validation_runs, holdout_runs, &salt_id, total);
        Ok(Self {
            id,
            training_runs,
            validation_runs,
            holdout_runs,
            salt_id,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn training_runs(&self) -> u64 { self.training_runs }
    pub fn validation_runs(&self) -> u64 { self.validation_runs }
    pub fn holdout_runs(&self) -> u64 { self.holdout_runs }
    pub fn salt_id(&self) -> &ContentId { &self.salt_id }

    pub fn total_runs(&self) -> Result<u64, ForgeCorpusSplitError> {
        self.training_runs
            .checked_add(self.validation_runs)
            .and_then(|value| value.checked_add(self.holdout_runs))
            .ok_or(ForgeCorpusSplitError::CountOverflow)
    }

    pub fn validate(&self) -> Result<(), ForgeCorpusSplitError> {
        let rebuilt = Self::new(
            self.training_runs,
            self.validation_runs,
            self.holdout_runs,
            self.salt_id.clone(),
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeCorpusSplitError::SpecIdentityMismatch)
        }
    }
}

fn derive_spec_id(
    training: u64,
    validation: u64,
    holdout: u64,
    salt_id: &ContentId,
    total: u64,
) -> ContentId {
    let training = training.to_be_bytes();
    let validation = validation.to_be_bytes();
    let holdout = holdout.to_be_bytes();
    let total = total.to_be_bytes();
    ContentId::derive(
        "symthaea.forge-corpus-split-spec.v1",
        [
            training.as_slice(),
            validation.as_slice(),
            holdout.as_slice(),
            total.as_slice(),
            salt_id.as_str().as_bytes(),
        ],
    )
}

/// One semantic discovery run's immutable corpus role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeCorpusAssignment {
    id: ContentId,
    run_id: ContentId,
    sequence_batch_id: ContentId,
    rank_id: ContentId,
    role: ForgeCorpusRole,
}

impl ForgeCorpusAssignment {
    fn new(
        run_id: ContentId,
        sequence_batch_id: ContentId,
        salt_id: &ContentId,
        role: ForgeCorpusRole,
    ) -> Self {
        let rank_id = derive_rank_id(salt_id, &run_id);
        let id = derive_assignment_id(&run_id, &sequence_batch_id, &rank_id, role);
        Self {
            id,
            run_id,
            sequence_batch_id,
            rank_id,
            role,
        }
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn run_id(&self) -> &ContentId { &self.run_id }
    pub fn sequence_batch_id(&self) -> &ContentId { &self.sequence_batch_id }
    pub fn rank_id(&self) -> &ContentId { &self.rank_id }
    pub fn role(&self) -> ForgeCorpusRole { self.role }

    fn validate(&self, salt_id: &ContentId) -> Result<(), ForgeCorpusSplitError> {
        let rank = derive_rank_id(salt_id, &self.run_id);
        let id = derive_assignment_id(
            &self.run_id,
            &self.sequence_batch_id,
            &rank,
            self.role,
        );
        if rank == self.rank_id && id == self.id {
            Ok(())
        } else {
            Err(ForgeCorpusSplitError::AssignmentIdentityMismatch)
        }
    }
}

fn derive_rank_id(salt_id: &ContentId, run_id: &ContentId) -> ContentId {
    // Deliberately excludes cohort/batch/outcome identity.
    ContentId::derive(
        "symthaea.forge-corpus-run-rank.v1",
        [salt_id.as_str().as_bytes(), run_id.as_str().as_bytes()],
    )
}

fn derive_assignment_id(
    run_id: &ContentId,
    sequence_batch_id: &ContentId,
    rank_id: &ContentId,
    role: ForgeCorpusRole,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-corpus-assignment.v1",
        [
            run_id.as_str().as_bytes(),
            sequence_batch_id.as_str().as_bytes(),
            rank_id.as_str().as_bytes(),
            role.tag(),
        ],
    )
}

#[derive(Clone)]
struct RunMember {
    run_id: ContentId,
    batch_id: ContentId,
}

/// Immutable partition manifest for one exact-context sequence cohort.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeSequenceCorpusSplit {
    id: ContentId,
    cohort_id: ContentId,
    spec: ForgeCorpusSplitSpec,
    assignments: Vec<ForgeCorpusAssignment>,
}

impl ForgeSequenceCorpusSplit {
    pub fn from_cohort(
        cohort: &ExactContextForgeSequenceCohort,
        spec: ForgeCorpusSplitSpec,
    ) -> Result<Self, ForgeCorpusSplitError> {
        cohort.validate()?;
        spec.validate()?;
        if cohort.run_count() != spec.total_runs()? {
            return Err(ForgeCorpusSplitError::RunCountMismatch);
        }
        let members = cohort
            .batches()
            .iter()
            .map(|batch| RunMember {
                run_id: batch.run_id().clone(),
                batch_id: batch.id().clone(),
            })
            .collect::<Vec<_>>();
        let assignments = build_assignments(&members, &spec)?;
        let id = derive_split_id(cohort.id(), spec.id(), &assignments);
        let split = Self {
            id,
            cohort_id: cohort.id().clone(),
            spec,
            assignments,
        };
        split.validate_for(cohort)?;
        Ok(split)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn cohort_id(&self) -> &ContentId { &self.cohort_id }
    pub fn spec(&self) -> &ForgeCorpusSplitSpec { &self.spec }
    pub fn assignments(&self) -> &[ForgeCorpusAssignment] { &self.assignments }

    pub fn assignments_for(
        &self,
        role: ForgeCorpusRole,
    ) -> impl Iterator<Item = &ForgeCorpusAssignment> {
        self.assignments.iter().filter(move |assignment| assignment.role == role)
    }

    pub fn validate(&self) -> Result<(), ForgeCorpusSplitError> {
        self.spec.validate()?;
        validate_assignments(&self.assignments, &self.spec)?;
        if derive_split_id(&self.cohort_id, self.spec.id(), &self.assignments) == self.id {
            Ok(())
        } else {
            Err(ForgeCorpusSplitError::SplitIdentityMismatch)
        }
    }

    pub fn validate_for(
        &self,
        cohort: &ExactContextForgeSequenceCohort,
    ) -> Result<(), ForgeCorpusSplitError> {
        cohort.validate()?;
        self.validate()?;
        if self.cohort_id != *cohort.id() || cohort.run_count() != self.spec.total_runs()? {
            return Err(ForgeCorpusSplitError::CohortCoverageMismatch);
        }
        let members = cohort
            .batches()
            .iter()
            .map(|batch| RunMember {
                run_id: batch.run_id().clone(),
                batch_id: batch.id().clone(),
            })
            .collect::<Vec<_>>();
        let expected = build_assignments(&members, &self.spec)?;
        if expected == self.assignments {
            Ok(())
        } else {
            Err(ForgeCorpusSplitError::CohortCoverageMismatch)
        }
    }
}

fn build_assignments(
    members: &[RunMember],
    spec: &ForgeCorpusSplitSpec,
) -> Result<Vec<ForgeCorpusAssignment>, ForgeCorpusSplitError> {
    if u64::try_from(members.len()).map_err(|_| ForgeCorpusSplitError::CountOverflow)?
        != spec.total_runs()?
    {
        return Err(ForgeCorpusSplitError::RunCountMismatch);
    }
    let mut ranked = members
        .iter()
        .map(|member| {
            (
                derive_rank_id(spec.salt_id(), &member.run_id),
                member.run_id.clone(),
                member.batch_id.clone(),
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let training_end = usize::try_from(spec.training_runs)
        .map_err(|_| ForgeCorpusSplitError::CountOverflow)?;
    let validation_end = usize::try_from(
        spec.training_runs
            .checked_add(spec.validation_runs)
            .ok_or(ForgeCorpusSplitError::CountOverflow)?,
    )
    .map_err(|_| ForgeCorpusSplitError::CountOverflow)?;

    let mut assignments = ranked
        .into_iter()
        .enumerate()
        .map(|(index, (_, run_id, batch_id))| {
            let role = if index < training_end {
                ForgeCorpusRole::Training
            } else if index < validation_end {
                ForgeCorpusRole::Validation
            } else {
                ForgeCorpusRole::Holdout
            };
            ForgeCorpusAssignment::new(run_id, batch_id, spec.salt_id(), role)
        })
        .collect::<Vec<_>>();
    assignments.sort_by(|a, b| a.run_id.cmp(&b.run_id));
    validate_assignments(&assignments, spec)?;
    Ok(assignments)
}

fn validate_assignments(
    assignments: &[ForgeCorpusAssignment],
    spec: &ForgeCorpusSplitSpec,
) -> Result<(), ForgeCorpusSplitError> {
    if u64::try_from(assignments.len()).map_err(|_| ForgeCorpusSplitError::CountOverflow)?
        != spec.total_runs()?
    {
        return Err(ForgeCorpusSplitError::RoleCountMismatch);
    }
    let mut seen = BTreeSet::new();
    let mut previous: Option<&ContentId> = None;
    let mut training = 0u64;
    let mut validation = 0u64;
    let mut holdout = 0u64;
    for assignment in assignments {
        assignment.validate(spec.salt_id())?;
        if !seen.insert(assignment.run_id.as_str().to_string())
            || previous.is_some_and(|run_id| run_id >= &assignment.run_id)
        {
            return Err(ForgeCorpusSplitError::DuplicateRun);
        }
        previous = Some(&assignment.run_id);
        match assignment.role {
            ForgeCorpusRole::Training => training = training.checked_add(1).ok_or(ForgeCorpusSplitError::CountOverflow)?,
            ForgeCorpusRole::Validation => validation = validation.checked_add(1).ok_or(ForgeCorpusSplitError::CountOverflow)?,
            ForgeCorpusRole::Holdout => holdout = holdout.checked_add(1).ok_or(ForgeCorpusSplitError::CountOverflow)?,
        }
    }
    if training == spec.training_runs
        && validation == spec.validation_runs
        && holdout == spec.holdout_runs
    {
        Ok(())
    } else {
        Err(ForgeCorpusSplitError::RoleCountMismatch)
    }
}

fn derive_split_id(
    cohort_id: &ContentId,
    spec_id: &ContentId,
    assignments: &[ForgeCorpusAssignment],
) -> ContentId {
    let count = (assignments.len() as u64).to_be_bytes();
    let mut parts = vec![
        cohort_id.as_str().as_bytes().to_vec(),
        spec_id.as_str().as_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        assignments
            .iter()
            .map(|assignment| assignment.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-sequence-corpus-split.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn members(batch_suffix: &str) -> Vec<RunMember> {
        (0..6)
            .map(|index| RunMember {
                run_id: cid("run", &format!("run-{index}")),
                batch_id: cid("batch", &format!("{batch_suffix}-{index}")),
            })
            .collect()
    }

    #[test]
    fn split_has_exact_non_overlapping_role_counts() {
        let spec = ForgeCorpusSplitSpec::new(3, 1, 2, cid("salt", "frozen-v1")).unwrap();
        let assignments = build_assignments(&members("a"), &spec).unwrap();
        assert_eq!(assignments.len(), 6);
        assert_eq!(assignments.iter().filter(|a| a.role() == ForgeCorpusRole::Training).count(), 3);
        assert_eq!(assignments.iter().filter(|a| a.role() == ForgeCorpusRole::Validation).count(), 1);
        assert_eq!(assignments.iter().filter(|a| a.role() == ForgeCorpusRole::Holdout).count(), 2);
        assert!(validate_assignments(&assignments, &spec).is_ok());
    }

    #[test]
    fn role_assignment_does_not_depend_on_outcome_bearing_batch_identity() {
        let spec = ForgeCorpusSplitSpec::new(3, 1, 2, cid("salt", "frozen-v1")).unwrap();
        let a = build_assignments(&members("outcomes-a"), &spec).unwrap();
        let b = build_assignments(&members("outcomes-b"), &spec).unwrap();
        let roles_a = a.iter().map(|x| (x.run_id().clone(), x.role(), x.rank_id().clone())).collect::<Vec<_>>();
        let roles_b = b.iter().map(|x| (x.run_id().clone(), x.role(), x.rank_id().clone())).collect::<Vec<_>>();
        assert_eq!(roles_a, roles_b);
        assert!(a.iter().zip(&b).any(|(left, right)| left.id() != right.id()));
    }

    #[test]
    fn changing_frozen_salt_changes_run_rank_identity() {
        let a = ForgeCorpusSplitSpec::new(3, 1, 2, cid("salt", "v1")).unwrap();
        let b = ForgeCorpusSplitSpec::new(3, 1, 2, cid("salt", "v2")).unwrap();
        let members = members("same");
        let split_a = build_assignments(&members, &a).unwrap();
        let split_b = build_assignments(&members, &b).unwrap();
        assert!(split_a
            .iter()
            .zip(&split_b)
            .any(|(left, right)| left.rank_id() != right.rank_id()));
    }
}
