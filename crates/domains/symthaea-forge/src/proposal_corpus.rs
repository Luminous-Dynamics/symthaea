// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Role-isolated proposal corpora over the frozen Forge sequence split.
//!
//! The existing sequence corpus split decides run roles from only a frozen salt and semantic run
//! identity. This module binds one proposal-complete observation table to every assigned run, then
//! exposes separate training and validation types. Holdout remains identity-only: there is
//! intentionally no public holdout-row dataset constructor in v1. A future frozen-model evaluator
//! must introduce the explicit authority boundary that is allowed to open those rows.

use crate::corpus_split::{
    ForgeCorpusAssignment, ForgeCorpusRole, ForgeCorpusSplitError, ForgeSequenceCorpusSplit,
};
use crate::family_learning::ForgeTransformationFamilyId;
use crate::proposal_dataset::{
    ForgeProposalDatasetError, ForgeProposalObservationRow, ForgeProposalObservationTable,
};
use crate::sequence_stats::{ExactContextForgeSequenceCohort, ForgeSequenceStatsError};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::{ContentId, ProblemId};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalCorpusError {
    #[error(transparent)]
    CorpusSplit(#[from] ForgeCorpusSplitError),
    #[error(transparent)]
    SequenceStats(#[from] ForgeSequenceStatsError),
    #[error(transparent)]
    Dataset(#[from] ForgeProposalDatasetError),
    #[error("proposal corpus exact-context sequence cohort is unexpectedly empty")]
    EmptyCohort,
    #[error("proposal corpus does not contain exactly one observation table per frozen run assignment")]
    TableCoverageMismatch,
    #[error("proposal corpus contains more than one observation table for the same semantic run")]
    DuplicateRunTable,
    #[error("proposal observation table run does not match its frozen corpus assignment")]
    RunMismatch,
    #[error("proposal observation table baseline does not match the exact-context cohort")]
    BaselineMismatch,
    #[error("proposal observation table generator does not match the exact-context cohort")]
    GeneratorMismatch,
    #[error("proposal observation tables do not share one exact proposal policy")]
    PolicyMismatch,
    #[error("proposal observation tables do not share one exact ordered family set")]
    FamilyOrderMismatch,
    #[error("proposal corpus member identity does not match canonical fields")]
    MemberIdentityMismatch,
    #[error("proposal corpus manifest is not in canonical run order or contains duplicates")]
    NonCanonicalManifest,
    #[error("proposal corpus manifest identity does not match canonical fields")]
    ManifestIdentityMismatch,
    #[error("proposal corpus manifest does not bind the supplied frozen sequence cohort/split")]
    ManifestScopeMismatch,
    #[error("role dataset contains a table not assigned to that frozen corpus role")]
    WrongRoleTable,
    #[error("role dataset is missing an assigned table")]
    MissingRoleTable,
    #[error("role dataset identity does not match canonical fields")]
    RoleDatasetIdentityMismatch,
    #[error("holdout seal identity does not match canonical fields")]
    HoldoutSealIdentityMismatch,
    #[error("proposal corpus count cannot be represented")]
    CountOverflow,
}

fn role_tag(role: ForgeCorpusRole) -> &'static [u8] {
    match role {
        ForgeCorpusRole::Training => b"training",
        ForgeCorpusRole::Validation => b"validation",
        ForgeCorpusRole::Holdout => b"holdout",
    }
}

/// Identity-only binding between one frozen run assignment and one proposal observation table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalCorpusMember {
    id: ContentId,
    assignment_id: ContentId,
    run_id: ContentId,
    sequence_batch_id: ContentId,
    rank_id: ContentId,
    proposal_table_id: ContentId,
    role: ForgeCorpusRole,
}

impl ForgeProposalCorpusMember {
    fn from_assignment(
        assignment: &ForgeCorpusAssignment,
        table: &ForgeProposalObservationTable,
    ) -> Result<Self, ForgeProposalCorpusError> {
        if assignment.run_id() != table.run_id() {
            return Err(ForgeProposalCorpusError::RunMismatch);
        }
        let id = derive_member_id(
            assignment.id(),
            assignment.run_id(),
            assignment.sequence_batch_id(),
            assignment.rank_id(),
            table.id(),
            assignment.role(),
        );
        Ok(Self {
            id,
            assignment_id: assignment.id().clone(),
            run_id: assignment.run_id().clone(),
            sequence_batch_id: assignment.sequence_batch_id().clone(),
            rank_id: assignment.rank_id().clone(),
            proposal_table_id: table.id().clone(),
            role: assignment.role(),
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn assignment_id(&self) -> &ContentId { &self.assignment_id }
    pub fn run_id(&self) -> &ContentId { &self.run_id }
    pub fn sequence_batch_id(&self) -> &ContentId { &self.sequence_batch_id }
    pub fn rank_id(&self) -> &ContentId { &self.rank_id }
    pub fn proposal_table_id(&self) -> &ContentId { &self.proposal_table_id }
    pub fn role(&self) -> ForgeCorpusRole { self.role }

    pub fn validate(&self) -> Result<(), ForgeProposalCorpusError> {
        let expected = derive_member_id(
            &self.assignment_id,
            &self.run_id,
            &self.sequence_batch_id,
            &self.rank_id,
            &self.proposal_table_id,
            self.role,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalCorpusError::MemberIdentityMismatch)
        }
    }
}

fn derive_member_id(
    assignment_id: &ContentId,
    run_id: &ContentId,
    sequence_batch_id: &ContentId,
    rank_id: &ContentId,
    proposal_table_id: &ContentId,
    role: ForgeCorpusRole,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-corpus-member.v1",
        [
            assignment_id.as_str().as_bytes(),
            run_id.as_str().as_bytes(),
            sequence_batch_id.as_str().as_bytes(),
            rank_id.as_str().as_bytes(),
            proposal_table_id.as_str().as_bytes(),
            role_tag(role),
        ],
    )
}

/// Frozen binding from an exact-context sequence split to proposal-complete run tables.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalCorpusManifest {
    id: ContentId,
    sequence_cohort_id: ContentId,
    corpus_split_id: ContentId,
    problem_id: ProblemId,
    baseline_implementation_id: ContentId,
    context_id: ContentId,
    generator_id: ContentId,
    policy_id: ContentId,
    families: Vec<ForgeTransformationFamilyId>,
    members: Vec<ForgeProposalCorpusMember>,
}

impl ForgeProposalCorpusManifest {
    /// Freeze the corpus assignment against already-built observation tables without taking
    /// ownership of those potentially large tables. The manifest stores identities only.
    pub fn from_tables(
        cohort: &ExactContextForgeSequenceCohort,
        split: &ForgeSequenceCorpusSplit,
        tables: &[ForgeProposalObservationTable],
    ) -> Result<Self, ForgeProposalCorpusError> {
        cohort.validate()?;
        split.validate_for(cohort)?;
        let first_batch = cohort
            .batches()
            .first()
            .ok_or(ForgeProposalCorpusError::EmptyCohort)?;
        let family_batch = first_batch.family_batch();
        let problem_id = family_batch.problem_id().clone();
        let baseline_implementation_id = family_batch
            .baseline_implementation_id()
            .as_content_id()
            .clone();
        let context_id = family_batch.context_id().clone();
        let generator_id = family_batch.generator_id().clone();

        let mut tables_by_run = BTreeMap::<String, &ForgeProposalObservationTable>::new();
        let mut policy_id: Option<ContentId> = None;
        let mut families: Option<Vec<ForgeTransformationFamilyId>> = None;
        for table in tables {
            table.validate()?;
            if table.baseline_implementation_id() != &baseline_implementation_id {
                return Err(ForgeProposalCorpusError::BaselineMismatch);
            }
            if table.generator_id() != &generator_id {
                return Err(ForgeProposalCorpusError::GeneratorMismatch);
            }
            match &policy_id {
                Some(expected) if table.policy_id() != expected => {
                    return Err(ForgeProposalCorpusError::PolicyMismatch);
                }
                None => policy_id = Some(table.policy_id().clone()),
                _ => {}
            }
            match &families {
                Some(expected) if table.families() != expected.as_slice() => {
                    return Err(ForgeProposalCorpusError::FamilyOrderMismatch);
                }
                None => families = Some(table.families().to_vec()),
                _ => {}
            }
            let key = table.run_id().as_str().to_string();
            if tables_by_run.insert(key, table).is_some() {
                return Err(ForgeProposalCorpusError::DuplicateRunTable);
            }
        }

        if tables_by_run.len() != split.assignments().len() {
            return Err(ForgeProposalCorpusError::TableCoverageMismatch);
        }
        let policy_id = policy_id.ok_or(ForgeProposalCorpusError::TableCoverageMismatch)?;
        let families = families.ok_or(ForgeProposalCorpusError::TableCoverageMismatch)?;

        let mut members = Vec::with_capacity(split.assignments().len());
        for assignment in split.assignments() {
            let table = tables_by_run
                .remove(assignment.run_id().as_str())
                .ok_or(ForgeProposalCorpusError::TableCoverageMismatch)?;
            members.push(ForgeProposalCorpusMember::from_assignment(assignment, table)?);
        }
        if !tables_by_run.is_empty() {
            return Err(ForgeProposalCorpusError::TableCoverageMismatch);
        }
        members.sort_by(|left, right| left.run_id().cmp(right.run_id()));

        let sequence_cohort_id = cohort.id().clone();
        let corpus_split_id = split.id().clone();
        let id = derive_manifest_id(
            &sequence_cohort_id,
            &corpus_split_id,
            &problem_id,
            &baseline_implementation_id,
            &context_id,
            &generator_id,
            &policy_id,
            &families,
            &members,
        );
        let manifest = Self {
            id,
            sequence_cohort_id,
            corpus_split_id,
            problem_id,
            baseline_implementation_id,
            context_id,
            generator_id,
            policy_id,
            families,
            members,
        };
        manifest.validate_for(cohort, split)?;
        Ok(manifest)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn sequence_cohort_id(&self) -> &ContentId { &self.sequence_cohort_id }
    pub fn corpus_split_id(&self) -> &ContentId { &self.corpus_split_id }
    pub fn problem_id(&self) -> &ProblemId { &self.problem_id }
    pub fn baseline_implementation_id(&self) -> &ContentId { &self.baseline_implementation_id }
    pub fn context_id(&self) -> &ContentId { &self.context_id }
    pub fn generator_id(&self) -> &ContentId { &self.generator_id }
    pub fn policy_id(&self) -> &ContentId { &self.policy_id }
    pub fn families(&self) -> &[ForgeTransformationFamilyId] { &self.families }
    pub fn members(&self) -> &[ForgeProposalCorpusMember] { &self.members }

    pub fn members_for(
        &self,
        role: ForgeCorpusRole,
    ) -> impl Iterator<Item = &ForgeProposalCorpusMember> {
        self.members.iter().filter(move |member| member.role() == role)
    }

    pub fn validate(&self) -> Result<(), ForgeProposalCorpusError> {
        if self.families.is_empty() || self.members.is_empty() {
            return Err(ForgeProposalCorpusError::NonCanonicalManifest);
        }
        for family in &self.families {
            family.validate().map_err(ForgeProposalDatasetError::from)?;
            if family.generator_id() != &self.generator_id {
                return Err(ForgeProposalCorpusError::GeneratorMismatch);
            }
        }

        let mut previous_run: Option<&ContentId> = None;
        let mut assignment_ids = BTreeSet::new();
        let mut run_ids = BTreeSet::new();
        let mut table_ids = BTreeSet::new();
        let mut role_counts = [0u64; 3];
        for member in &self.members {
            member.validate()?;
            if previous_run.is_some_and(|previous| previous >= member.run_id())
                || !assignment_ids.insert(member.assignment_id().as_str().to_string())
                || !run_ids.insert(member.run_id().as_str().to_string())
                || !table_ids.insert(member.proposal_table_id().as_str().to_string())
            {
                return Err(ForgeProposalCorpusError::NonCanonicalManifest);
            }
            previous_run = Some(member.run_id());
            let index = match member.role() {
                ForgeCorpusRole::Training => 0,
                ForgeCorpusRole::Validation => 1,
                ForgeCorpusRole::Holdout => 2,
            };
            role_counts[index] = role_counts[index]
                .checked_add(1)
                .ok_or(ForgeProposalCorpusError::CountOverflow)?;
        }
        if role_counts.iter().any(|count| *count == 0) {
            return Err(ForgeProposalCorpusError::NonCanonicalManifest);
        }

        let expected = derive_manifest_id(
            &self.sequence_cohort_id,
            &self.corpus_split_id,
            &self.problem_id,
            &self.baseline_implementation_id,
            &self.context_id,
            &self.generator_id,
            &self.policy_id,
            &self.families,
            &self.members,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalCorpusError::ManifestIdentityMismatch)
        }
    }

    pub fn validate_for(
        &self,
        cohort: &ExactContextForgeSequenceCohort,
        split: &ForgeSequenceCorpusSplit,
    ) -> Result<(), ForgeProposalCorpusError> {
        cohort.validate()?;
        split.validate_for(cohort)?;
        self.validate()?;
        if self.sequence_cohort_id != *cohort.id()
            || self.corpus_split_id != *split.id()
            || self.members.len() != split.assignments().len()
        {
            return Err(ForgeProposalCorpusError::ManifestScopeMismatch);
        }
        let first_batch = cohort
            .batches()
            .first()
            .ok_or(ForgeProposalCorpusError::EmptyCohort)?;
        let family_batch = first_batch.family_batch();
        if &self.problem_id != family_batch.problem_id()
            || &self.baseline_implementation_id
                != family_batch.baseline_implementation_id().as_content_id()
            || &self.context_id != family_batch.context_id()
            || &self.generator_id != family_batch.generator_id()
        {
            return Err(ForgeProposalCorpusError::ManifestScopeMismatch);
        }

        let assignments = split
            .assignments()
            .iter()
            .map(|assignment| (assignment.run_id().as_str().to_string(), assignment))
            .collect::<BTreeMap<_, _>>();
        for member in &self.members {
            let assignment = assignments
                .get(member.run_id().as_str())
                .ok_or(ForgeProposalCorpusError::ManifestScopeMismatch)?;
            if member.assignment_id() != assignment.id()
                || member.sequence_batch_id() != assignment.sequence_batch_id()
                || member.rank_id() != assignment.rank_id()
                || member.role() != assignment.role()
            {
                return Err(ForgeProposalCorpusError::ManifestScopeMismatch);
            }
        }
        Ok(())
    }

    /// Identity-only holdout seal. It contains no proposal observation rows.
    pub fn holdout_seal(&self) -> Result<ForgeProposalHoldoutSeal, ForgeProposalCorpusError> {
        ForgeProposalHoldoutSeal::from_manifest(self)
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_manifest_id(
    sequence_cohort_id: &ContentId,
    corpus_split_id: &ContentId,
    problem_id: &ProblemId,
    baseline_implementation_id: &ContentId,
    context_id: &ContentId,
    generator_id: &ContentId,
    policy_id: &ContentId,
    families: &[ForgeTransformationFamilyId],
    members: &[ForgeProposalCorpusMember],
) -> ContentId {
    let family_count = (families.len() as u64).to_be_bytes();
    let member_count = (members.len() as u64).to_be_bytes();
    let mut parts = vec![
        sequence_cohort_id.as_str().as_bytes().to_vec(),
        corpus_split_id.as_str().as_bytes().to_vec(),
        problem_id.as_content_id().as_str().as_bytes().to_vec(),
        baseline_implementation_id.as_str().as_bytes().to_vec(),
        context_id.as_str().as_bytes().to_vec(),
        generator_id.as_str().as_bytes().to_vec(),
        policy_id.as_str().as_bytes().to_vec(),
        family_count.to_vec(),
    ];
    parts.extend(
        families
            .iter()
            .map(|family| family.as_content_id().as_str().as_bytes().to_vec()),
    );
    parts.push(member_count.to_vec());
    parts.extend(
        members
            .iter()
            .map(|member| member.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-proposal-corpus-manifest.v1",
        parts.iter().map(Vec::as_slice),
    )
}

fn validate_table_scope(
    manifest: &ForgeProposalCorpusManifest,
    table: &ForgeProposalObservationTable,
) -> Result<(), ForgeProposalCorpusError> {
    table.validate()?;
    if table.baseline_implementation_id() != manifest.baseline_implementation_id() {
        return Err(ForgeProposalCorpusError::BaselineMismatch);
    }
    if table.generator_id() != manifest.generator_id() {
        return Err(ForgeProposalCorpusError::GeneratorMismatch);
    }
    if table.policy_id() != manifest.policy_id() {
        return Err(ForgeProposalCorpusError::PolicyMismatch);
    }
    if table.families() != manifest.families() {
        return Err(ForgeProposalCorpusError::FamilyOrderMismatch);
    }
    Ok(())
}

fn role_tables(
    manifest: &ForgeProposalCorpusManifest,
    role: ForgeCorpusRole,
    tables: Vec<ForgeProposalObservationTable>,
) -> Result<Vec<ForgeProposalObservationTable>, ForgeProposalCorpusError> {
    manifest.validate()?;
    let expected = manifest
        .members_for(role)
        .map(|member| (member.run_id().as_str().to_string(), member))
        .collect::<BTreeMap<_, _>>();
    if expected.is_empty() {
        return Err(ForgeProposalCorpusError::MissingRoleTable);
    }

    let mut provided = BTreeMap::<String, ForgeProposalObservationTable>::new();
    for table in tables {
        validate_table_scope(manifest, &table)?;
        let key = table.run_id().as_str().to_string();
        if !expected.contains_key(&key) {
            return Err(ForgeProposalCorpusError::WrongRoleTable);
        }
        if provided.insert(key, table).is_some() {
            return Err(ForgeProposalCorpusError::DuplicateRunTable);
        }
    }

    if provided.len() != expected.len() {
        return Err(ForgeProposalCorpusError::MissingRoleTable);
    }
    let mut canonical = Vec::with_capacity(expected.len());
    for (run_key, member) in expected {
        let table = provided
            .remove(&run_key)
            .ok_or(ForgeProposalCorpusError::MissingRoleTable)?;
        if table.id() != member.proposal_table_id() || table.run_id() != member.run_id() {
            return Err(ForgeProposalCorpusError::TableCoverageMismatch);
        }
        canonical.push(table);
    }
    canonical.sort_by(|left, right| left.run_id().cmp(right.run_id()));
    Ok(canonical)
}

fn derive_role_dataset_id(
    manifest_id: &ContentId,
    role: ForgeCorpusRole,
    tables: &[ForgeProposalObservationTable],
) -> ContentId {
    let count = (tables.len() as u64).to_be_bytes();
    let mut parts = vec![
        manifest_id.as_str().as_bytes().to_vec(),
        role_tag(role).to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        tables
            .iter()
            .map(|table| table.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-proposal-role-dataset.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// Tables explicitly assigned to training. Construction rejects every validation/holdout table.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalTrainingSet {
    id: ContentId,
    manifest_id: ContentId,
    tables: Vec<ForgeProposalObservationTable>,
}

impl ForgeProposalTrainingSet {
    pub fn from_manifest(
        manifest: &ForgeProposalCorpusManifest,
        tables: Vec<ForgeProposalObservationTable>,
    ) -> Result<Self, ForgeProposalCorpusError> {
        let tables = role_tables(manifest, ForgeCorpusRole::Training, tables)?;
        let id = derive_role_dataset_id(manifest.id(), ForgeCorpusRole::Training, &tables);
        Ok(Self {
            id,
            manifest_id: manifest.id().clone(),
            tables,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn manifest_id(&self) -> &ContentId { &self.manifest_id }
    pub fn tables(&self) -> &[ForgeProposalObservationTable] { &self.tables }
    pub fn rows(&self) -> impl Iterator<Item = &ForgeProposalObservationRow> {
        self.tables.iter().flat_map(|table| table.rows())
    }

    pub fn validate_for(
        &self,
        manifest: &ForgeProposalCorpusManifest,
    ) -> Result<(), ForgeProposalCorpusError> {
        let rebuilt = Self::from_manifest(manifest, self.tables.clone())?;
        if rebuilt.id == self.id && rebuilt.manifest_id == self.manifest_id {
            Ok(())
        } else {
            Err(ForgeProposalCorpusError::RoleDatasetIdentityMismatch)
        }
    }
}

/// Tables explicitly assigned to validation. Construction rejects every training/holdout table.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalValidationSet {
    id: ContentId,
    manifest_id: ContentId,
    tables: Vec<ForgeProposalObservationTable>,
}

impl ForgeProposalValidationSet {
    pub fn from_manifest(
        manifest: &ForgeProposalCorpusManifest,
        tables: Vec<ForgeProposalObservationTable>,
    ) -> Result<Self, ForgeProposalCorpusError> {
        let tables = role_tables(manifest, ForgeCorpusRole::Validation, tables)?;
        let id = derive_role_dataset_id(manifest.id(), ForgeCorpusRole::Validation, &tables);
        Ok(Self {
            id,
            manifest_id: manifest.id().clone(),
            tables,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn manifest_id(&self) -> &ContentId { &self.manifest_id }
    pub fn tables(&self) -> &[ForgeProposalObservationTable] { &self.tables }
    pub fn rows(&self) -> impl Iterator<Item = &ForgeProposalObservationRow> {
        self.tables.iter().flat_map(|table| table.rows())
    }

    pub fn validate_for(
        &self,
        manifest: &ForgeProposalCorpusManifest,
    ) -> Result<(), ForgeProposalCorpusError> {
        let rebuilt = Self::from_manifest(manifest, self.tables.clone())?;
        if rebuilt.id == self.id && rebuilt.manifest_id == self.manifest_id {
            Ok(())
        } else {
            Err(ForgeProposalCorpusError::RoleDatasetIdentityMismatch)
        }
    }
}

/// One holdout member identity. No observation rows are stored here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalHoldoutMember {
    assignment_id: ContentId,
    run_id: ContentId,
    proposal_table_id: ContentId,
}

impl ForgeProposalHoldoutMember {
    pub fn assignment_id(&self) -> &ContentId { &self.assignment_id }
    pub fn run_id(&self) -> &ContentId { &self.run_id }
    pub fn proposal_table_id(&self) -> &ContentId { &self.proposal_table_id }
}

/// Identity-only holdout commitment. v1 intentionally provides no method that returns holdout rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalHoldoutSeal {
    id: ContentId,
    manifest_id: ContentId,
    members: Vec<ForgeProposalHoldoutMember>,
}

impl ForgeProposalHoldoutSeal {
    fn from_manifest(
        manifest: &ForgeProposalCorpusManifest,
    ) -> Result<Self, ForgeProposalCorpusError> {
        manifest.validate()?;
        let members = manifest
            .members_for(ForgeCorpusRole::Holdout)
            .map(|member| ForgeProposalHoldoutMember {
                assignment_id: member.assignment_id().clone(),
                run_id: member.run_id().clone(),
                proposal_table_id: member.proposal_table_id().clone(),
            })
            .collect::<Vec<_>>();
        if members.is_empty() {
            return Err(ForgeProposalCorpusError::NonCanonicalManifest);
        }
        let id = derive_holdout_seal_id(manifest.id(), &members);
        Ok(Self {
            id,
            manifest_id: manifest.id().clone(),
            members,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn manifest_id(&self) -> &ContentId { &self.manifest_id }
    pub fn members(&self) -> &[ForgeProposalHoldoutMember] { &self.members }

    pub fn validate_for(
        &self,
        manifest: &ForgeProposalCorpusManifest,
    ) -> Result<(), ForgeProposalCorpusError> {
        let rebuilt = Self::from_manifest(manifest)?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeProposalCorpusError::HoldoutSealIdentityMismatch)
        }
    }
}

fn derive_holdout_seal_id(
    manifest_id: &ContentId,
    members: &[ForgeProposalHoldoutMember],
) -> ContentId {
    let count = (members.len() as u64).to_be_bytes();
    let mut parts = vec![manifest_id.as_str().as_bytes().to_vec(), count.to_vec()];
    for member in members {
        parts.push(member.assignment_id.as_str().as_bytes().to_vec());
        parts.push(member.run_id.as_str().as_bytes().to_vec());
        parts.push(member.proposal_table_id.as_str().as_bytes().to_vec());
    }
    ContentId::derive(
        "symthaea.forge-proposal-holdout-seal.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn synthetic_member(
        run: &str,
        table: &str,
        role: ForgeCorpusRole,
    ) -> ForgeProposalCorpusMember {
        let assignment_id = cid("assignment", run);
        let run_id = cid("run", run);
        let sequence_batch_id = cid("sequence-batch", run);
        let rank_id = cid("rank", run);
        let proposal_table_id = cid("proposal-table", table);
        let id = derive_member_id(
            &assignment_id,
            &run_id,
            &sequence_batch_id,
            &rank_id,
            &proposal_table_id,
            role,
        );
        ForgeProposalCorpusMember {
            id,
            assignment_id,
            run_id,
            sequence_batch_id,
            rank_id,
            proposal_table_id,
            role,
        }
    }

    #[test]
    fn member_identity_binds_role_and_table() {
        let training = synthetic_member("run-a", "table-a", ForgeCorpusRole::Training);
        let holdout = synthetic_member("run-a", "table-a", ForgeCorpusRole::Holdout);
        let other_table = synthetic_member("run-a", "table-b", ForgeCorpusRole::Training);
        assert_ne!(training.id(), holdout.id());
        assert_ne!(training.id(), other_table.id());
        training.validate().unwrap();
    }

    #[test]
    fn holdout_seal_identity_binds_exact_table_ids_without_rows() {
        let manifest_id = cid("manifest", "m1");
        let a = ForgeProposalHoldoutMember {
            assignment_id: cid("assignment", "a"),
            run_id: cid("run", "a"),
            proposal_table_id: cid("table", "a"),
        };
        let mut b = a.clone();
        b.proposal_table_id = cid("table", "b");
        assert_ne!(
            derive_holdout_seal_id(&manifest_id, &[a]),
            derive_holdout_seal_id(&manifest_id, &[b]),
        );
    }
}
