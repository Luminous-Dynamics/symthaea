// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content-blind planning for explicit correction propagation through dependent memory.
//!
//! This module does not mutate a backing store. It computes a deterministic,
//! provenance-aware propagation plan. A later executor must bind the plan to
//! actual privacy/memory graph mutations and fresh snapshot/index receipts.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MEMORY_CORRECTION_PROPAGATION_SCHEMA_V1: &str =
    "symthaea.communication.memory-correction-propagation.v1";
const MAX_ARTIFACTS_V1: usize = 4096;
const MAX_SOURCES_PER_ARTIFACT_V1: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrectionDurabilityV1 {
    TurnOnly,
    Session,
    DurableExplicit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryRetentionV1 {
    Turn,
    Session,
    Durable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryArtifactRoleV1 {
    SourceMemory,
    DerivedMemory,
    IndexOrEmbedding,
    ExternalExport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryCorrectionActionV1 {
    NoEffectOutsideScope,
    ShadowForCurrentScope,
    Invalidate,
    RecomputeRequired,
    SupersedeWithFreshIdentity,
    RetainIndependentBasis,
    BlockedUnknownDependency,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrectionDirectiveRefV1 {
    pub correction_id: String,
    pub correction_receipt_commitment: String,
    pub semantic_key: String,
    pub context_id: String,
    pub durability: CorrectionDurabilityV1,
    pub corrected_source_ids: BTreeSet<String>,
    pub correction_epoch: u64,
}

impl CorrectionDirectiveRefV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        correction_id: impl Into<String>,
        correction_receipt_commitment: impl Into<String>,
        semantic_key: impl Into<String>,
        context_id: impl Into<String>,
        durability: CorrectionDurabilityV1,
        corrected_source_ids: impl IntoIterator<Item = String>,
        correction_epoch: u64,
    ) -> Result<Self, MemoryCorrectionPropagationErrorV1> {
        let corrected_source_ids: BTreeSet<String> = corrected_source_ids
            .into_iter()
            .map(canonical_id)
            .collect::<Result<_, _>>()?;
        if corrected_source_ids.is_empty() {
            return Err(MemoryCorrectionPropagationErrorV1::MissingCorrectedSource);
        }
        if correction_epoch == 0 {
            return Err(MemoryCorrectionPropagationErrorV1::InvalidEpoch);
        }
        let correction_receipt_commitment = correction_receipt_commitment.into();
        if !is_blake3_commitment(&correction_receipt_commitment) {
            return Err(MemoryCorrectionPropagationErrorV1::InvalidCommitment);
        }
        Ok(Self {
            correction_id: canonical_id(correction_id.into())?,
            correction_receipt_commitment,
            semantic_key: canonical_key(semantic_key.into())?,
            context_id: canonical_key(context_id.into())?,
            durability,
            corrected_source_ids,
            correction_epoch,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryArtifactDependencyV1 {
    pub artifact_id: String,
    pub semantic_key: String,
    pub context_id: String,
    pub role: MemoryArtifactRoleV1,
    pub retention: MemoryRetentionV1,
    pub dependency_complete: bool,
    pub source_ids: BTreeSet<String>,
    /// The artifact's own derivation policy states that the remaining known
    /// sources are sufficient after affected sources are removed.
    pub independent_basis_sufficient: bool,
    pub derivation_policy_ref: String,
    pub external_copy_possible: bool,
}

impl MemoryArtifactDependencyV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        artifact_id: impl Into<String>,
        semantic_key: impl Into<String>,
        context_id: impl Into<String>,
        role: MemoryArtifactRoleV1,
        retention: MemoryRetentionV1,
        dependency_complete: bool,
        source_ids: impl IntoIterator<Item = String>,
        independent_basis_sufficient: bool,
        derivation_policy_ref: impl Into<String>,
        external_copy_possible: bool,
    ) -> Result<Self, MemoryCorrectionPropagationErrorV1> {
        let source_ids: BTreeSet<String> = source_ids
            .into_iter()
            .map(canonical_id)
            .collect::<Result<_, _>>()?;
        if source_ids.len() > MAX_SOURCES_PER_ARTIFACT_V1 {
            return Err(MemoryCorrectionPropagationErrorV1::TooManySources);
        }
        if independent_basis_sufficient && !dependency_complete {
            return Err(
                MemoryCorrectionPropagationErrorV1::IndependentBasisWithoutCompleteDependencies,
            );
        }
        Ok(Self {
            artifact_id: canonical_id(artifact_id.into())?,
            semantic_key: canonical_key(semantic_key.into())?,
            context_id: canonical_key(context_id.into())?,
            role,
            retention,
            dependency_complete,
            source_ids,
            independent_basis_sufficient,
            derivation_policy_ref: canonical_ref(derivation_policy_ref.into())?,
            external_copy_possible,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryDependencySnapshotV1 {
    pub snapshot_id: String,
    pub policy_version: String,
    pub artifacts: BTreeMap<String, MemoryArtifactDependencyV1>,
    pub commitment: String,
}

impl MemoryDependencySnapshotV1 {
    pub fn new(
        snapshot_id: impl Into<String>,
        policy_version: impl Into<String>,
        artifacts: Vec<MemoryArtifactDependencyV1>,
    ) -> Result<Self, MemoryCorrectionPropagationErrorV1> {
        if artifacts.len() > MAX_ARTIFACTS_V1 {
            return Err(MemoryCorrectionPropagationErrorV1::TooManyArtifacts);
        }
        let mut artifact_map = BTreeMap::new();
        for artifact in artifacts {
            if artifact_map.insert(artifact.artifact_id.clone(), artifact).is_some() {
                return Err(MemoryCorrectionPropagationErrorV1::DuplicateArtifactId);
            }
        }
        let mut snapshot = Self {
            snapshot_id: canonical_id(snapshot_id.into())?,
            policy_version: canonical_key(policy_version.into())?,
            artifacts: artifact_map,
            commitment: String::new(),
        };
        snapshot.commitment = snapshot_commitment(&snapshot);
        Ok(snapshot)
    }

    pub fn validate(&self) -> Result<(), MemoryCorrectionPropagationErrorV1> {
        if self.commitment != snapshot_commitment(self) {
            return Err(MemoryCorrectionPropagationErrorV1::SnapshotCommitmentMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryCorrectionDecisionV1 {
    pub artifact_id: String,
    pub action: MemoryCorrectionActionV1,
    pub corrected_dependencies: Vec<String>,
    pub remaining_sources: Vec<String>,
    pub fresh_identity_required: bool,
    pub external_deletion_unproven: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryCorrectionPropagationPlanV1 {
    pub schema: String,
    pub operation_id: String,
    pub operation_epoch: u64,
    pub correction_id: String,
    pub correction_receipt_commitment: String,
    pub correction_epoch: u64,
    pub durability: CorrectionDurabilityV1,
    pub snapshot_id: String,
    pub snapshot_commitment: String,
    pub policy_version: String,
    pub decisions: Vec<MemoryCorrectionDecisionV1>,
    pub blocked_unknown_dependency_count: u64,
    pub fresh_identity_required_count: u64,
    pub external_deletion_unproven_count: u64,
    pub commitment: String,
}

impl MemoryCorrectionPropagationPlanV1 {
    pub fn validate(
        &self,
        correction: &CorrectionDirectiveRefV1,
        snapshot: &MemoryDependencySnapshotV1,
    ) -> Result<(), MemoryCorrectionPropagationErrorV1> {
        snapshot.validate()?;
        let expected = plan_memory_correction(
            &self.operation_id,
            self.operation_epoch,
            correction,
            snapshot,
        )?;
        if &expected != self {
            return Err(MemoryCorrectionPropagationErrorV1::PlanMismatch);
        }
        if self.commitment != plan_commitment(self) {
            return Err(MemoryCorrectionPropagationErrorV1::PlanCommitmentMismatch);
        }
        Ok(())
    }
}

pub fn plan_memory_correction(
    operation_id: &str,
    operation_epoch: u64,
    correction: &CorrectionDirectiveRefV1,
    snapshot: &MemoryDependencySnapshotV1,
) -> Result<MemoryCorrectionPropagationPlanV1, MemoryCorrectionPropagationErrorV1> {
    snapshot.validate()?;
    let operation_id = canonical_id(operation_id.to_owned())?;
    if operation_epoch == 0 {
        return Err(MemoryCorrectionPropagationErrorV1::InvalidEpoch);
    }

    let mut decisions = Vec::with_capacity(snapshot.artifacts.len());
    for artifact in snapshot.artifacts.values() {
        decisions.push(decide_artifact(correction, artifact));
    }
    decisions.sort_by(|a, b| a.artifact_id.cmp(&b.artifact_id));

    let blocked_unknown_dependency_count = decisions
        .iter()
        .filter(|d| d.action == MemoryCorrectionActionV1::BlockedUnknownDependency)
        .count() as u64;
    let fresh_identity_required_count = decisions
        .iter()
        .filter(|d| d.fresh_identity_required)
        .count() as u64;
    let external_deletion_unproven_count = decisions
        .iter()
        .filter(|d| d.external_deletion_unproven)
        .count() as u64;

    let mut plan = MemoryCorrectionPropagationPlanV1 {
        schema: MEMORY_CORRECTION_PROPAGATION_SCHEMA_V1.into(),
        operation_id,
        operation_epoch,
        correction_id: correction.correction_id.clone(),
        correction_receipt_commitment: correction.correction_receipt_commitment.clone(),
        correction_epoch: correction.correction_epoch,
        durability: correction.durability,
        snapshot_id: snapshot.snapshot_id.clone(),
        snapshot_commitment: snapshot.commitment.clone(),
        policy_version: snapshot.policy_version.clone(),
        decisions,
        blocked_unknown_dependency_count,
        fresh_identity_required_count,
        external_deletion_unproven_count,
        commitment: String::new(),
    };
    plan.commitment = plan_commitment(&plan);
    Ok(plan)
}

fn decide_artifact(
    correction: &CorrectionDirectiveRefV1,
    artifact: &MemoryArtifactDependencyV1,
) -> MemoryCorrectionDecisionV1 {
    let in_scope = artifact.semantic_key == correction.semantic_key
        && artifact.context_id == correction.context_id;
    if !in_scope {
        return decision(
            artifact,
            MemoryCorrectionActionV1::NoEffectOutsideScope,
            vec![],
            artifact.source_ids.iter().cloned().collect(),
            false,
        );
    }

    let directly_corrected = correction.corrected_source_ids.contains(&artifact.artifact_id);
    if directly_corrected {
        let action = match correction.durability {
            CorrectionDurabilityV1::TurnOnly => MemoryCorrectionActionV1::ShadowForCurrentScope,
            CorrectionDurabilityV1::Session => match artifact.retention {
                MemoryRetentionV1::Durable => MemoryCorrectionActionV1::ShadowForCurrentScope,
                MemoryRetentionV1::Turn | MemoryRetentionV1::Session => {
                    MemoryCorrectionActionV1::Invalidate
                }
            },
            CorrectionDurabilityV1::DurableExplicit => {
                MemoryCorrectionActionV1::SupersedeWithFreshIdentity
            }
        };
        let fresh_identity_required =
            action == MemoryCorrectionActionV1::SupersedeWithFreshIdentity;
        return decision(
            artifact,
            action,
            vec![artifact.artifact_id.clone()],
            artifact.source_ids.iter().cloned().collect(),
            fresh_identity_required,
        );
    }

    if !artifact.dependency_complete {
        return decision(
            artifact,
            MemoryCorrectionActionV1::BlockedUnknownDependency,
            vec![],
            artifact.source_ids.iter().cloned().collect(),
            false,
        );
    }

    let corrected_dependencies: Vec<String> = artifact
        .source_ids
        .intersection(&correction.corrected_source_ids)
        .cloned()
        .collect();
    let remaining_sources: Vec<String> = artifact
        .source_ids
        .difference(&correction.corrected_source_ids)
        .cloned()
        .collect();

    if corrected_dependencies.is_empty() {
        return decision(
            artifact,
            MemoryCorrectionActionV1::RetainIndependentBasis,
            vec![],
            remaining_sources,
            false,
        );
    }

    let action = match correction.durability {
        CorrectionDurabilityV1::TurnOnly => MemoryCorrectionActionV1::ShadowForCurrentScope,
        CorrectionDurabilityV1::Session => match artifact.retention {
            MemoryRetentionV1::Durable => MemoryCorrectionActionV1::ShadowForCurrentScope,
            MemoryRetentionV1::Turn | MemoryRetentionV1::Session => {
                if artifact.independent_basis_sufficient && !remaining_sources.is_empty() {
                    MemoryCorrectionActionV1::RetainIndependentBasis
                } else {
                    MemoryCorrectionActionV1::Invalidate
                }
            }
        },
        CorrectionDurabilityV1::DurableExplicit => {
            if artifact.independent_basis_sufficient && !remaining_sources.is_empty() {
                MemoryCorrectionActionV1::RetainIndependentBasis
            } else {
                match artifact.role {
                    MemoryArtifactRoleV1::DerivedMemory => {
                        MemoryCorrectionActionV1::RecomputeRequired
                    }
                    MemoryArtifactRoleV1::SourceMemory
                    | MemoryArtifactRoleV1::IndexOrEmbedding
                    | MemoryArtifactRoleV1::ExternalExport => {
                        MemoryCorrectionActionV1::Invalidate
                    }
                }
            }
        }
    };

    let fresh_identity_required = action == MemoryCorrectionActionV1::RecomputeRequired;
    decision(
        artifact,
        action,
        corrected_dependencies,
        remaining_sources,
        fresh_identity_required,
    )
}

fn decision(
    artifact: &MemoryArtifactDependencyV1,
    action: MemoryCorrectionActionV1,
    corrected_dependencies: Vec<String>,
    remaining_sources: Vec<String>,
    fresh_identity_required: bool,
) -> MemoryCorrectionDecisionV1 {
    let destructive_or_recomputing = matches!(
        action,
        MemoryCorrectionActionV1::Invalidate
            | MemoryCorrectionActionV1::RecomputeRequired
            | MemoryCorrectionActionV1::SupersedeWithFreshIdentity
    );
    MemoryCorrectionDecisionV1 {
        artifact_id: artifact.artifact_id.clone(),
        action,
        corrected_dependencies,
        remaining_sources,
        fresh_identity_required,
        external_deletion_unproven: artifact.external_copy_possible && destructive_or_recomputing,
    }
}

fn snapshot_commitment(snapshot: &MemoryDependencySnapshotV1) -> String {
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, "symthaea-memory-correction-snapshot-v1");
    put_str(&mut hasher, &snapshot.snapshot_id);
    put_str(&mut hasher, &snapshot.policy_version);
    put_u64(&mut hasher, snapshot.artifacts.len() as u64);
    for artifact in snapshot.artifacts.values() {
        put_str(&mut hasher, &artifact.artifact_id);
        put_str(&mut hasher, &artifact.semantic_key);
        put_str(&mut hasher, &artifact.context_id);
        put_u8(&mut hasher, role_tag(artifact.role));
        put_u8(&mut hasher, retention_tag(artifact.retention));
        put_bool(&mut hasher, artifact.dependency_complete);
        put_bool(&mut hasher, artifact.independent_basis_sufficient);
        put_bool(&mut hasher, artifact.external_copy_possible);
        put_str(&mut hasher, &artifact.derivation_policy_ref);
        put_u64(&mut hasher, artifact.source_ids.len() as u64);
        for source in &artifact.source_ids {
            put_str(&mut hasher, source);
        }
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn plan_commitment(plan: &MemoryCorrectionPropagationPlanV1) -> String {
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, "symthaea-memory-correction-plan-v1");
    put_str(&mut hasher, &plan.schema);
    put_str(&mut hasher, &plan.operation_id);
    put_u64(&mut hasher, plan.operation_epoch);
    put_str(&mut hasher, &plan.correction_id);
    put_str(&mut hasher, &plan.correction_receipt_commitment);
    put_u64(&mut hasher, plan.correction_epoch);
    put_u8(&mut hasher, durability_tag(plan.durability));
    put_str(&mut hasher, &plan.snapshot_id);
    put_str(&mut hasher, &plan.snapshot_commitment);
    put_str(&mut hasher, &plan.policy_version);
    put_u64(&mut hasher, plan.decisions.len() as u64);
    for decision in &plan.decisions {
        put_str(&mut hasher, &decision.artifact_id);
        put_u8(&mut hasher, action_tag(decision.action));
        put_bool(&mut hasher, decision.fresh_identity_required);
        put_bool(&mut hasher, decision.external_deletion_unproven);
        put_u64(&mut hasher, decision.corrected_dependencies.len() as u64);
        for source in &decision.corrected_dependencies {
            put_str(&mut hasher, source);
        }
        put_u64(&mut hasher, decision.remaining_sources.len() as u64);
        for source in &decision.remaining_sources {
            put_str(&mut hasher, source);
        }
    }
    put_u64(&mut hasher, plan.blocked_unknown_dependency_count);
    put_u64(&mut hasher, plan.fresh_identity_required_count);
    put_u64(&mut hasher, plan.external_deletion_unproven_count);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn canonical_id(value: String) -> Result<String, MemoryCorrectionPropagationErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 256 {
        return Err(MemoryCorrectionPropagationErrorV1::InvalidIdentity);
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':' | '/'))
    {
        return Err(MemoryCorrectionPropagationErrorV1::InvalidIdentity);
    }
    Ok(value)
}

fn canonical_key(value: String) -> Result<String, MemoryCorrectionPropagationErrorV1> {
    let value = value.trim().to_ascii_lowercase();
    if value.is_empty() || value.len() > 256 {
        return Err(MemoryCorrectionPropagationErrorV1::InvalidKey);
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':' | '/'))
    {
        return Err(MemoryCorrectionPropagationErrorV1::InvalidKey);
    }
    Ok(value)
}

fn canonical_ref(value: String) -> Result<String, MemoryCorrectionPropagationErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 1024 {
        return Err(MemoryCorrectionPropagationErrorV1::InvalidReference);
    }
    Ok(value)
}

fn is_blake3_commitment(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("blake3:")
        && value[7..].bytes().all(|b| b.is_ascii_hexdigit())
}

fn put_str(hasher: &mut blake3::Hasher, value: &str) {
    put_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}
fn put_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}
fn put_u8(hasher: &mut blake3::Hasher, value: u8) {
    hasher.update(&[value]);
}
fn put_bool(hasher: &mut blake3::Hasher, value: bool) {
    put_u8(hasher, u8::from(value));
}

const fn durability_tag(value: CorrectionDurabilityV1) -> u8 {
    match value {
        CorrectionDurabilityV1::TurnOnly => 1,
        CorrectionDurabilityV1::Session => 2,
        CorrectionDurabilityV1::DurableExplicit => 3,
    }
}
const fn retention_tag(value: MemoryRetentionV1) -> u8 {
    match value {
        MemoryRetentionV1::Turn => 1,
        MemoryRetentionV1::Session => 2,
        MemoryRetentionV1::Durable => 3,
    }
}
const fn role_tag(value: MemoryArtifactRoleV1) -> u8 {
    match value {
        MemoryArtifactRoleV1::SourceMemory => 1,
        MemoryArtifactRoleV1::DerivedMemory => 2,
        MemoryArtifactRoleV1::IndexOrEmbedding => 3,
        MemoryArtifactRoleV1::ExternalExport => 4,
    }
}
const fn action_tag(value: MemoryCorrectionActionV1) -> u8 {
    match value {
        MemoryCorrectionActionV1::NoEffectOutsideScope => 1,
        MemoryCorrectionActionV1::ShadowForCurrentScope => 2,
        MemoryCorrectionActionV1::Invalidate => 3,
        MemoryCorrectionActionV1::RecomputeRequired => 4,
        MemoryCorrectionActionV1::SupersedeWithFreshIdentity => 5,
        MemoryCorrectionActionV1::RetainIndependentBasis => 6,
        MemoryCorrectionActionV1::BlockedUnknownDependency => 7,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryCorrectionPropagationErrorV1 {
    InvalidIdentity,
    InvalidKey,
    InvalidReference,
    InvalidCommitment,
    InvalidEpoch,
    MissingCorrectedSource,
    TooManySources,
    TooManyArtifacts,
    DuplicateArtifactId,
    IndependentBasisWithoutCompleteDependencies,
    SnapshotCommitmentMismatch,
    PlanCommitmentMismatch,
    PlanMismatch,
}
