// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Typed, candidate-only weight-space programs for Symthaea.
//!
//! This module deliberately stops before live model mutation. It validates a
//! program against an immutable, digest-bound parameter snapshot and returns a
//! materialized candidate. A model-specific adapter must still pass the
//! candidate through that model's transactional SWI replacement API and
//! independently evaluate it before any promotion.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::hdc::hdc_ltc_unified::HdcLtcParameterGroupId;

/// Stable, architecture-aware identity for a CfC serialized parameter group.
///
/// This adapter-neutral identity mirrors the SWI CfC manifest. The main-crate
/// adapter must map it explicitly to the model's native identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CfCWeightGroupId {
    CellInputProjection { cell: usize },
    CellRecurrentProjection { cell: usize },
    CellHiddenBias { cell: usize },
    CellTimeConstant { cell: usize },
    CellBackboneWeight { cell: usize, layer: usize },
    CellBackboneBias { cell: usize, layer: usize },
    OutputProjection,
    OutputBias,
}

/// Stable, architecture-aware identity for a parameter group.
///
/// It does not flatten CfC and HDC-LTC mathematical semantics into string keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightProgramTarget {
    CfC(CfCWeightGroupId),
    HdcLtc(HdcLtcParameterGroupId),
}

/// Semantic role used to check proposed values against model-specific invariants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterGroupRole {
    Projection,
    Bias,
    TimeConstant,
    Hypervector,
    Generic,
}

/// Numerical constraints exported by a model-specific snapshot adapter.
///
/// Constraints are descriptive preconditions, not defaults guessed by WSP.
/// Adapters should populate them from the architecture's actual configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ParameterGroupConstraints {
    pub min_value: Option<f32>,
    pub max_value: Option<f32>,
    pub min_l2_norm: Option<f32>,
    pub max_l2_norm: Option<f32>,
    pub max_l2_delta: Option<f32>,
}

/// Immutable parameter-group snapshot supplied by a model adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterGroupSnapshot {
    pub target: WeightProgramTarget,
    pub role: ParameterGroupRole,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    pub constraints: ParameterGroupConstraints,
}

/// Immutable model context against which a WeightProgram is authorized to run.
///
/// The architecture, manifest, model and target-specification digests must be
/// exact values produced or bound by the model/task adapter. A stale digest rejects the
/// proposal rather than rebasing it implicitly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightModelSnapshot {
    pub format_version: u32,
    pub architecture_digest: String,
    pub manifest_digest: String,
    pub model_digest: String,
    pub target_specification_digest: String,
    pub groups: Vec<ParameterGroupSnapshot>,
}

/// A bounded operation over one typed parameter group.
///
/// Each group may be targeted only once per program in v1. This avoids
/// ambiguous operation ordering and makes the edit budget easy to audit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WeightProgramOperation {
    ReplaceGroup {
        target: WeightProgramTarget,
        values: Vec<f32>,
    },
    AddDelta {
        target: WeightProgramTarget,
        offset: usize,
        values: Vec<f32>,
    },
    ScaleRange {
        target: WeightProgramTarget,
        start: usize,
        end: usize,
        factor: f32,
    },
}

/// Reviewable, bounded proposal for changing a model's parameter groups.
///
/// The proposal is not executable authority. It can only create an isolated
/// candidate after its exact base digest, target specification, evidence,
/// operation shapes, model invariants and edit budgets have been validated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightProgram {
    pub format_version: u32,
    pub architecture_digest: String,
    pub manifest_digest: String,
    pub base_model_digest: String,
    pub target_specification_digest: String,
    pub fault_hypothesis_ids: Vec<String>,
    pub evidence_receipt_ids: Vec<String>,
    pub max_changed_parameters: usize,
    pub max_total_l2_delta: f32,
    pub max_abs_delta: f32,
    pub operations: Vec<WeightProgramOperation>,
}

/// Summary produced when a candidate is materialized.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightProgramReceipt {
    pub format_version: u32,
    pub base_model_digest: String,
    pub program_digest: String,
    pub candidate_snapshot_digest: String,
    pub edited_group_count: usize,
    pub changed_parameter_count: usize,
    pub total_l2_delta: f32,
    pub max_abs_delta: f32,
}

/// Candidate parameter state. It is detached from the live model.
///
/// The candidate snapshot digest is a content digest for this generic
/// representation, not a replacement for the adapter's official model
/// fingerprint. Before evaluation/promotion, the adapter must recompute its
/// native fingerprint and use its existing transactional edit API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidatedWeightCandidate {
    pub groups: Vec<ParameterGroupSnapshot>,
    pub receipt: WeightProgramReceipt,
}

/// Fail-closed validation errors for WeightProgram proposals.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum WeightProgramError {
    #[error("unsupported WeightProgram format version: {0}")]
    UnsupportedProgramVersion(u32),
    #[error("unsupported model snapshot format version: {0}")]
    UnsupportedSnapshotVersion(u32),
    #[error("digest is not exactly 64 lowercase hexadecimal characters: {0}")]
    InvalidDigest(&'static str),
    #[error("program digest does not match the model snapshot: {0}")]
    StaleDigest(&'static str),
    #[error("missing fault hypothesis or evidence receipt")]
    MissingProvenance,
    #[error("duplicate provenance identifier: {0}")]
    DuplicateProvenance(String),
    #[error("model snapshot has no parameter groups")]
    EmptySnapshot,
    #[error("duplicate parameter group target")]
    DuplicateSnapshotTarget,
    #[error("program has no operations")]
    EmptyProgram,
    #[error("duplicate parameter group target in one program")]
    DuplicateProgramTarget,
    #[error("parameter group target is absent from the snapshot")]
    UnknownTarget,
    #[error("parameter group shape product overflows")]
    ShapeOverflow,
    #[error("parameter group shape does not match value count")]
    ShapeMismatch,
    #[error("parameter group contains a non-finite value")]
    NonFiniteSnapshotValue,
    #[error("operation contains a non-finite value")]
    NonFiniteOperationValue,
    #[error("invalid parameter constraints")]
    InvalidConstraints,
    #[error("baseline parameter group violates its declared constraints")]
    BaselineConstraintViolation,
    #[error("replacement value count does not match target group")]
    ReplacementLengthMismatch,
    #[error("delta range is empty, overflowing, or outside the target group")]
    InvalidDeltaRange,
    #[error("scale range is empty, reversed, or outside the target group")]
    InvalidScaleRange,
    #[error("edited candidate violates the target group's value or norm constraints")]
    CandidateConstraintViolation,
    #[error("per-group L2 edit norm exceeds the declared group budget")]
    GroupDeltaBudgetExceeded,
    #[error("changed parameter count exceeds the declared budget")]
    ChangedParameterBudgetExceeded,
    #[error("total L2 edit norm exceeds the declared budget")]
    TotalDeltaBudgetExceeded,
    #[error("maximum absolute parameter change exceeds the declared budget")]
    MaxDeltaBudgetExceeded,
}

impl WeightProgram {
    /// Validate and materialize a detached candidate against an immutable
    /// snapshot. The source snapshot is never mutated, including on failure.
    pub fn validate_and_materialize(
        &self,
        snapshot: &WeightModelSnapshot,
    ) -> Result<ValidatedWeightCandidate, WeightProgramError> {
        if self.format_version != 1 {
            return Err(WeightProgramError::UnsupportedProgramVersion(
                self.format_version,
            ));
        }
        if snapshot.format_version != 1 {
            return Err(WeightProgramError::UnsupportedSnapshotVersion(
                snapshot.format_version,
            ));
        }

        validate_digest(&self.architecture_digest)?;
        validate_digest(&self.manifest_digest)?;
        validate_digest(&self.base_model_digest)?;
        validate_digest(&self.target_specification_digest)?;
        validate_digest(&snapshot.architecture_digest)?;
        validate_digest(&snapshot.manifest_digest)?;
        validate_digest(&snapshot.model_digest)?;
        validate_digest(&snapshot.target_specification_digest)?;

        if self.architecture_digest != snapshot.architecture_digest {
            return Err(WeightProgramError::StaleDigest("architecture"));
        }
        if self.manifest_digest != snapshot.manifest_digest {
            return Err(WeightProgramError::StaleDigest("manifest"));
        }
        if self.base_model_digest != snapshot.model_digest {
            return Err(WeightProgramError::StaleDigest("base model"));
        }
        if self.target_specification_digest != snapshot.target_specification_digest {
            return Err(WeightProgramError::StaleDigest("target specification"));
        }
        if self.fault_hypothesis_ids.is_empty() || self.evidence_receipt_ids.is_empty() {
            return Err(WeightProgramError::MissingProvenance);
        }
        validate_unique_strings(&self.fault_hypothesis_ids)?;
        validate_unique_strings(&self.evidence_receipt_ids)?;

        if snapshot.groups.is_empty() {
            return Err(WeightProgramError::EmptySnapshot);
        }
        if self.operations.is_empty() {
            return Err(WeightProgramError::EmptyProgram);
        }
        if self.max_changed_parameters == 0
            || !self.max_total_l2_delta.is_finite()
            || self.max_total_l2_delta <= 0.0
            || !self.max_abs_delta.is_finite()
            || self.max_abs_delta <= 0.0
        {
            return Err(WeightProgramError::InvalidConstraints);
        }

        let mut snapshot_targets = HashSet::with_capacity(snapshot.groups.len());
        for group in &snapshot.groups {
            if !snapshot_targets.insert(group.target.clone()) {
                return Err(WeightProgramError::DuplicateSnapshotTarget);
            }
            validate_group(group, true)?;
        }

        let mut program_targets = HashSet::with_capacity(self.operations.len());
        let mut candidate_groups = snapshot.groups.clone();
        let mut edited_group_count = 0usize;

        for operation in &self.operations {
            let target = operation.target();
            if !program_targets.insert(target.clone()) {
                return Err(WeightProgramError::DuplicateProgramTarget);
            }
            let index = candidate_groups
                .iter()
                .position(|group| group.target == *target)
                .ok_or(WeightProgramError::UnknownTarget)?;
            let group = &mut candidate_groups[index];
            apply_operation(operation, group)?;
            validate_group(group, false)?;
            edited_group_count += 1;
        }

        let mut changed_parameter_count = 0usize;
        let mut total_delta_sq = 0.0_f64;
        let mut max_abs_delta = 0.0_f32;

        for (before, after) in snapshot.groups.iter().zip(candidate_groups.iter()) {
            let mut group_delta_sq = 0.0_f64;
            for (old, new) in before.values.iter().zip(after.values.iter()) {
                let delta = (*new - *old).abs();
                if delta > 0.0 {
                    changed_parameter_count += 1;
                    let delta_sq = f64::from(delta) * f64::from(delta);
                    total_delta_sq += delta_sq;
                    group_delta_sq += delta_sq;
                    max_abs_delta = max_abs_delta.max(delta);
                }
            }
            if before
                .constraints
                .max_l2_delta
                .is_some_and(|limit| group_delta_sq.sqrt() > f64::from(limit))
            {
                return Err(WeightProgramError::GroupDeltaBudgetExceeded);
            }
        }

        if changed_parameter_count > self.max_changed_parameters {
            return Err(WeightProgramError::ChangedParameterBudgetExceeded);
        }
        let total_l2_delta = total_delta_sq.sqrt() as f32;
        if total_l2_delta > self.max_total_l2_delta {
            return Err(WeightProgramError::TotalDeltaBudgetExceeded);
        }
        if max_abs_delta > self.max_abs_delta {
            return Err(WeightProgramError::MaxDeltaBudgetExceeded);
        }

        let program_digest = json_digest(b"symthaea-weight-program-v1\0", self)?;
        let candidate_snapshot_digest = candidate_digest(
            &snapshot.architecture_digest,
            &snapshot.manifest_digest,
            &snapshot.model_digest,
            &candidate_groups,
        )?;

        Ok(ValidatedWeightCandidate {
            groups: candidate_groups,
            receipt: WeightProgramReceipt {
                format_version: 1,
                base_model_digest: snapshot.model_digest.clone(),
                program_digest,
                candidate_snapshot_digest,
                edited_group_count,
                changed_parameter_count,
                total_l2_delta,
                max_abs_delta,
            },
        })
    }
}

impl WeightProgramOperation {
    fn target(&self) -> &WeightProgramTarget {
        match self {
            Self::ReplaceGroup { target, .. }
            | Self::AddDelta { target, .. }
            | Self::ScaleRange { target, .. } => target,
        }
    }
}

fn validate_digest(value: &str) -> Result<(), WeightProgramError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(WeightProgramError::InvalidDigest("digest"));
    }
    Ok(())
}

fn validate_unique_strings(values: &[String]) -> Result<(), WeightProgramError> {
    let mut seen = HashSet::with_capacity(values.len());
    for value in values {
        let normalized = value.trim();
        if normalized.is_empty() {
            return Err(WeightProgramError::MissingProvenance);
        }
        if !seen.insert(normalized.to_owned()) {
            return Err(WeightProgramError::DuplicateProvenance(
                normalized.to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_group(
    group: &ParameterGroupSnapshot,
    validate_baseline: bool,
) -> Result<(), WeightProgramError> {
    if group.shape.is_empty() {
        return Err(WeightProgramError::ShapeMismatch);
    }
    let expected_len = group
        .shape
        .iter()
        .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
        .ok_or(WeightProgramError::ShapeOverflow)?;
    if expected_len != group.values.len() || group.values.is_empty() {
        return Err(WeightProgramError::ShapeMismatch);
    }
    if group.values.iter().any(|value| !value.is_finite()) {
        return Err(WeightProgramError::NonFiniteSnapshotValue);
    }
    validate_group_constraints(&group.constraints)?;

    let norm = l2_norm(&group.values);
    if !norm.is_finite() {
        return Err(WeightProgramError::NonFiniteSnapshotValue);
    }
    let violates = group.values.iter().any(|value| {
        group.constraints.min_value.is_some_and(|min| *value < min)
            || group.constraints.max_value.is_some_and(|max| *value > max)
    }) || group
        .constraints
        .min_l2_norm
        .is_some_and(|min| norm < f64::from(min))
        || group
            .constraints
            .max_l2_norm
            .is_some_and(|max| norm > f64::from(max));
    if violates {
        return Err(if validate_baseline {
            WeightProgramError::BaselineConstraintViolation
        } else {
            WeightProgramError::CandidateConstraintViolation
        });
    }

    match &group.target {
        WeightProgramTarget::CfC(id) => {
            let expected_role = match id {
                CfCWeightGroupId::CellInputProjection { .. }
                | CfCWeightGroupId::CellRecurrentProjection { .. }
                | CfCWeightGroupId::CellBackboneWeight { .. }
                | CfCWeightGroupId::OutputProjection => ParameterGroupRole::Projection,
                CfCWeightGroupId::CellHiddenBias { .. }
                | CfCWeightGroupId::CellBackboneBias { .. }
                | CfCWeightGroupId::OutputBias => ParameterGroupRole::Bias,
                CfCWeightGroupId::CellTimeConstant { .. } => ParameterGroupRole::TimeConstant,
            };
            if group.role != expected_role {
                return Err(WeightProgramError::InvalidConstraints);
            }
            if matches!(id, CfCWeightGroupId::CellTimeConstant { .. })
                && !group
                    .constraints
                    .min_value
                    .is_some_and(|minimum| minimum > 0.0)
            {
                return Err(WeightProgramError::InvalidConstraints);
            }
        }
        WeightProgramTarget::HdcLtc(_)
            if group.role != ParameterGroupRole::Hypervector
                || !group
                    .constraints
                    .max_l2_norm
                    .is_some_and(|maximum| maximum <= 2.0) =>
        {
            return Err(WeightProgramError::InvalidConstraints);
        }
        WeightProgramTarget::HdcLtc(_) => {}
    }

    Ok(())
}

fn validate_group_constraints(
    constraints: &ParameterGroupConstraints,
) -> Result<(), WeightProgramError> {
    for value in [constraints.min_value, constraints.max_value]
        .into_iter()
        .flatten()
    {
        if !value.is_finite() {
            return Err(WeightProgramError::InvalidConstraints);
        }
    }
    for value in [
        constraints.min_l2_norm,
        constraints.max_l2_norm,
        constraints.max_l2_delta,
    ]
    .into_iter()
    .flatten()
    {
        if !value.is_finite() || value < 0.0 {
            return Err(WeightProgramError::InvalidConstraints);
        }
    }
    if constraints
        .min_value
        .zip(constraints.max_value)
        .is_some_and(|(a, b)| a > b)
        || constraints
            .min_l2_norm
            .zip(constraints.max_l2_norm)
            .is_some_and(|(a, b)| a > b)
    {
        return Err(WeightProgramError::InvalidConstraints);
    }
    Ok(())
}

fn apply_operation(
    operation: &WeightProgramOperation,
    group: &mut ParameterGroupSnapshot,
) -> Result<(), WeightProgramError> {
    match operation {
        WeightProgramOperation::ReplaceGroup { values, .. } => {
            if values.len() != group.values.len() {
                return Err(WeightProgramError::ReplacementLengthMismatch);
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(WeightProgramError::NonFiniteOperationValue);
            }
            group.values.clone_from(values);
        }
        WeightProgramOperation::AddDelta { offset, values, .. } => {
            if values.is_empty() {
                return Err(WeightProgramError::InvalidDeltaRange);
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(WeightProgramError::NonFiniteOperationValue);
            }
            let end = offset
                .checked_add(values.len())
                .ok_or(WeightProgramError::InvalidDeltaRange)?;
            let target = group
                .values
                .get_mut(*offset..end)
                .ok_or(WeightProgramError::InvalidDeltaRange)?;
            for (value, delta) in target.iter_mut().zip(values) {
                *value += *delta;
                if !value.is_finite() {
                    return Err(WeightProgramError::NonFiniteOperationValue);
                }
            }
        }
        WeightProgramOperation::ScaleRange {
            start, end, factor, ..
        } => {
            if !factor.is_finite() {
                return Err(WeightProgramError::NonFiniteOperationValue);
            }
            if start >= end {
                return Err(WeightProgramError::InvalidScaleRange);
            }
            let target = group
                .values
                .get_mut(*start..*end)
                .ok_or(WeightProgramError::InvalidScaleRange)?;
            for value in target {
                *value *= *factor;
                if !value.is_finite() {
                    return Err(WeightProgramError::NonFiniteOperationValue);
                }
            }
        }
    }
    Ok(())
}

fn l2_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt()
}

fn json_digest<T: Serialize>(domain: &[u8], value: &T) -> Result<String, WeightProgramError> {
    let encoded = serde_json::to_vec(value).map_err(|_| WeightProgramError::InvalidConstraints)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

fn candidate_digest(
    architecture_digest: &str,
    manifest_digest: &str,
    base_model_digest: &str,
    groups: &[ParameterGroupSnapshot],
) -> Result<String, WeightProgramError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-weight-program-candidate-v1\0");
    hasher.update(architecture_digest.as_bytes());
    hasher.update(manifest_digest.as_bytes());
    hasher.update(base_model_digest.as_bytes());

    for group in groups {
        let metadata =
            serde_json::to_vec(&(&group.target, group.role, &group.shape, &group.constraints))
                .map_err(|_| WeightProgramError::InvalidConstraints)?;
        hasher.update(&(metadata.len() as u64).to_le_bytes());
        hasher.update(&metadata);
        hasher.update(&(group.shape.len() as u64).to_le_bytes());
        for dimension in &group.shape {
            hasher.update(&(*dimension as u64).to_le_bytes());
        }
        hasher.update(&(group.values.len() as u64).to_le_bytes());
        for value in &group.values {
            hasher.update(&value.to_bits().to_le_bytes());
        }
    }
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn snapshot() -> WeightModelSnapshot {
        WeightModelSnapshot {
            format_version: 1,
            architecture_digest: digest('a'),
            manifest_digest: digest('b'),
            model_digest: digest('c'),
            target_specification_digest: digest('d'),
            groups: vec![
                ParameterGroupSnapshot {
                    target: WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant {
                        cell: 0,
                    }),
                    role: ParameterGroupRole::TimeConstant,
                    shape: vec![2],
                    values: vec![0.5, 0.75],
                    constraints: ParameterGroupConstraints {
                        min_value: Some(0.01),
                        max_value: Some(10.0),
                        max_l2_delta: Some(0.5),
                        ..Default::default()
                    },
                },
                ParameterGroupSnapshot {
                    target: WeightProgramTarget::HdcLtc(HdcLtcParameterGroupId::GateBias),
                    role: ParameterGroupRole::Hypervector,
                    shape: vec![2],
                    values: vec![0.1, -0.1],
                    constraints: ParameterGroupConstraints {
                        max_l2_norm: Some(2.0),
                        max_l2_delta: Some(0.5),
                        ..Default::default()
                    },
                },
            ],
        }
    }

    fn program(operation: WeightProgramOperation) -> WeightProgram {
        WeightProgram {
            format_version: 1,
            architecture_digest: digest('a'),
            manifest_digest: digest('b'),
            base_model_digest: digest('c'),
            target_specification_digest: digest('d'),
            fault_hypothesis_ids: vec!["fault:tau-high".into()],
            evidence_receipt_ids: vec!["receipt:trace-01".into()],
            max_changed_parameters: 2,
            max_total_l2_delta: 1.0,
            max_abs_delta: 0.5,
            operations: vec![operation],
        }
    }

    #[test]
    fn valid_program_materializes_detached_candidate() {
        let base = snapshot();
        let proposal = program(WeightProgramOperation::AddDelta {
            target: WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant { cell: 0 }),
            offset: 0,
            values: vec![0.1, 0.0],
        });
        let candidate = proposal
            .validate_and_materialize(&base)
            .expect("valid program");
        assert_eq!(base.groups[0].values, vec![0.5, 0.75]);
        assert_eq!(candidate.groups[0].values, vec![0.6, 0.75]);
        assert_eq!(candidate.receipt.edited_group_count, 1);
        assert_eq!(candidate.receipt.changed_parameter_count, 1);
        assert_eq!(candidate.receipt.base_model_digest, digest('c'));
        assert_eq!(candidate.receipt.program_digest.len(), 64);
        assert_eq!(candidate.receipt.candidate_snapshot_digest.len(), 64);
    }

    #[test]
    fn stale_model_digest_is_rejected_before_materialization() {
        let base = snapshot();
        let mut proposal = program(WeightProgramOperation::ScaleRange {
            target: WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant { cell: 0 }),
            start: 0,
            end: 2,
            factor: 1.1,
        });
        proposal.base_model_digest = digest('e');
        assert_eq!(
            proposal.validate_and_materialize(&base),
            Err(WeightProgramError::StaleDigest("base model"))
        );
    }

    #[test]
    fn invalid_tau_candidate_is_rejected_without_changing_snapshot() {
        let base = snapshot();
        let proposal = program(WeightProgramOperation::ReplaceGroup {
            target: WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant { cell: 0 }),
            values: vec![0.5, -0.1],
        });
        assert_eq!(
            proposal.validate_and_materialize(&base),
            Err(WeightProgramError::CandidateConstraintViolation)
        );
        assert_eq!(base.groups[0].values, vec![0.5, 0.75]);
    }

    #[test]
    fn duplicate_target_operations_are_rejected() {
        let base = snapshot();
        let target = WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant { cell: 0 });
        let mut proposal = program(WeightProgramOperation::ScaleRange {
            target: target.clone(),
            start: 0,
            end: 1,
            factor: 1.1,
        });
        proposal.operations.push(WeightProgramOperation::AddDelta {
            target,
            offset: 1,
            values: vec![0.1],
        });
        assert_eq!(
            proposal.validate_and_materialize(&base),
            Err(WeightProgramError::DuplicateProgramTarget)
        );
    }

    #[test]
    fn target_specification_digest_is_bound() {
        let base = snapshot();
        let mut proposal = program(WeightProgramOperation::ScaleRange {
            target: WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant { cell: 0 }),
            start: 0,
            end: 1,
            factor: 1.1,
        });
        proposal.target_specification_digest = digest('e');
        assert_eq!(
            proposal.validate_and_materialize(&base),
            Err(WeightProgramError::StaleDigest("target specification"))
        );
    }

    #[test]
    fn cfc_group_role_mismatch_is_rejected() {
        let mut base = snapshot();
        base.groups[0].role = ParameterGroupRole::Projection;
        let proposal = program(WeightProgramOperation::ScaleRange {
            target: WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant { cell: 0 }),
            start: 0,
            end: 1,
            factor: 1.1,
        });
        assert_eq!(
            proposal.validate_and_materialize(&base),
            Err(WeightProgramError::InvalidConstraints)
        );
    }

    #[test]
    fn per_group_delta_budget_is_enforced() {
        let base = snapshot();
        let proposal = program(WeightProgramOperation::AddDelta {
            target: WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant { cell: 0 }),
            offset: 0,
            values: vec![0.4, 0.4],
        });
        assert_eq!(
            proposal.validate_and_materialize(&base),
            Err(WeightProgramError::GroupDeltaBudgetExceeded)
        );
    }

    #[test]
    fn change_budget_is_enforced() {
        let base = snapshot();
        let mut proposal = program(WeightProgramOperation::ReplaceGroup {
            target: WeightProgramTarget::HdcLtc(HdcLtcParameterGroupId::GateBias),
            values: vec![0.2, -0.2],
        });
        proposal.max_changed_parameters = 1;
        assert_eq!(
            proposal.validate_and_materialize(&base),
            Err(WeightProgramError::ChangedParameterBudgetExceeded)
        );
    }

    #[test]
    fn candidate_digest_is_deterministic() {
        let base = snapshot();
        let proposal = program(WeightProgramOperation::ScaleRange {
            target: WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant { cell: 0 }),
            start: 0,
            end: 2,
            factor: 1.1,
        });
        let first = proposal
            .validate_and_materialize(&base)
            .expect("first candidate");
        let second = proposal
            .validate_and_materialize(&base)
            .expect("second candidate");
        assert_eq!(
            first.receipt.candidate_snapshot_digest,
            second.receipt.candidate_snapshot_digest
        );
    }
}
