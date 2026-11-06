// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Candidate-only adapter from generic WeightPrograms to the native CfC SWI API.
//!
//! This adapter creates an immutable snapshot from the network's native
//! fingerprint and manifest, validates a WeightProgram, and applies the
//! resulting replacements only to a cloned CfCNetwork. It never mutates the
//! caller's live network.

use symthaea_core::weight_space_program::{
    CfCWeightGroupId, ParameterGroupConstraints, ParameterGroupRole,
    ParameterGroupSnapshot, WeightModelSnapshot, WeightProgram, WeightProgramError,
    WeightProgramReceipt, WeightProgramTarget, ValidatedWeightCandidate,
};
use thiserror::Error;

use super::network::{
    CfCNetwork, CfCNetworkConfig, CfCParameterGroupId, CfCParameterGroupReplacement,
    CfCParameterManifest, CfCWeightLoadError,
};
use super::types::MIN_TAU;

/// Candidate CfC network produced without changing the source network.
#[derive(Debug, Clone)]
pub struct CfCWeightProgramCandidate {
    /// Isolated candidate network. The caller's source network is unchanged.
    pub network: CfCNetwork,
    /// Generic WeightProgram validation receipt.
    pub receipt: WeightProgramReceipt,
    /// Native fingerprint of the source model.
    pub base_native_model_digest: String,
    /// Native fingerprint recomputed from the materialized candidate network.
    pub candidate_native_model_digest: String,
}

/// Failure while adapting a generic WeightProgram to the native CfC SWI API.
#[derive(Debug, Error)]
pub enum CfCWeightProgramAdapterError {
    #[error(transparent)]
    Program(#[from] WeightProgramError),
    #[error(transparent)]
    NativeEdit(#[from] CfCWeightLoadError),
    #[error("CfC fingerprint failed: {0}")]
    Fingerprint(String),
    #[error("CfC manifest range is invalid for group {0}")]
    ManifestRange(String),
    #[error("WeightProgram contains a non-CfC target for a CfC candidate")]
    NonCfCTarget,
}

/// Build a digest-bound generic snapshot from the native CfC model.
///
/// The target-specification digest is supplied by the caller because task
/// identity belongs to the evaluation/repair request, not to the neural
/// network's architecture. It is checked exactly by WeightProgram validation.
pub fn cfc_weight_model_snapshot(
    network: &CfCNetwork,
    target_specification_digest: impl Into<String>,
) -> Result<WeightModelSnapshot, CfCWeightProgramAdapterError> {
    let fingerprint = network
        .fingerprint()
        .map_err(CfCWeightProgramAdapterError::Fingerprint)?;
    let manifest = network.parameter_manifest();
    let flat_weights = network.get_weights();
    let groups = snapshot_groups(&manifest, &flat_weights)?;

    Ok(WeightModelSnapshot {
        format_version: 1,
        architecture_digest: fingerprint.architecture_digest,
        manifest_digest: fingerprint.manifest_digest,
        model_digest: fingerprint.model_digest,
        target_specification_digest: target_specification_digest.into(),
        groups,
    })
}

/// Validate a WeightProgram and materialize it on an isolated CfCNetwork clone.
///
/// The native transactional edit API rechecks the exact base model digest and
/// validates all replacements before mutating the clone. The candidate's
/// native fingerprint is then recomputed. No live model promotion occurs.
pub fn materialize_cfc_weight_program(
    network: &CfCNetwork,
    target_specification_digest: impl Into<String>,
    program: &WeightProgram,
) -> Result<CfCWeightProgramCandidate, CfCWeightProgramAdapterError> {
    let snapshot = cfc_weight_model_snapshot(network, target_specification_digest)?;
    let candidate = program.validate_and_materialize(&snapshot)?;
    let replacements = native_replacements(&snapshot, &candidate)?;

    let mut candidate_network = network.clone();
    candidate_network.try_apply_parameter_group_replacements(
        &snapshot.model_digest,
        &replacements,
    )?;

    let candidate_fingerprint = candidate_network
        .fingerprint()
        .map_err(CfCWeightProgramAdapterError::Fingerprint)?;

    Ok(CfCWeightProgramCandidate {
        network: candidate_network,
        receipt: candidate.receipt,
        base_native_model_digest: snapshot.model_digest,
        candidate_native_model_digest: candidate_fingerprint.model_digest,
    })
}

fn snapshot_groups(
    manifest: &CfCParameterManifest,
    flat_weights: &[f32],
) -> Result<Vec<ParameterGroupSnapshot>, CfCWeightProgramAdapterError> {
    let mut groups = Vec::with_capacity(manifest.groups.len());
    for group in &manifest.groups {
        let end = group
            .offset
            .checked_add(group.len)
            .ok_or_else(|| CfCWeightProgramAdapterError::ManifestRange(group.name.clone()))?;
        let values = flat_weights
            .get(group.offset..end)
            .ok_or_else(|| CfCWeightProgramAdapterError::ManifestRange(group.name.clone()))?
            .to_vec();

        let (target, role, constraints) = match &group.id {
            CfCParameterGroupId::CellInputProjection { cell } => (
                CfCWeightGroupId::CellInputProjection { cell: *cell },
                ParameterGroupRole::Projection,
                ParameterGroupConstraints::default(),
            ),
            CfCParameterGroupId::CellRecurrentProjection { cell } => (
                CfCWeightGroupId::CellRecurrentProjection { cell: *cell },
                ParameterGroupRole::Projection,
                ParameterGroupConstraints::default(),
            ),
            CfCParameterGroupId::CellHiddenBias { cell } => (
                CfCWeightGroupId::CellHiddenBias { cell: *cell },
                ParameterGroupRole::Bias,
                ParameterGroupConstraints::default(),
            ),
            CfCParameterGroupId::CellTimeConstant { cell } => (
                CfCWeightGroupId::CellTimeConstant { cell: *cell },
                ParameterGroupRole::TimeConstant,
                ParameterGroupConstraints {
                    min_value: Some(MIN_TAU),
                    ..Default::default()
                },
            ),
            CfCParameterGroupId::CellBackboneWeight { cell, layer } => (
                CfCWeightGroupId::CellBackboneWeight {
                    cell: *cell,
                    layer: *layer,
                },
                ParameterGroupRole::Projection,
                ParameterGroupConstraints::default(),
            ),
            CfCParameterGroupId::CellBackboneBias { cell, layer } => (
                CfCWeightGroupId::CellBackboneBias {
                    cell: *cell,
                    layer: *layer,
                },
                ParameterGroupRole::Bias,
                ParameterGroupConstraints::default(),
            ),
            CfCParameterGroupId::OutputProjection => (
                CfCWeightGroupId::OutputProjection,
                ParameterGroupRole::Projection,
                ParameterGroupConstraints::default(),
            ),
            CfCParameterGroupId::OutputBias => (
                CfCWeightGroupId::OutputBias,
                ParameterGroupRole::Bias,
                ParameterGroupConstraints::default(),
            ),
        };

        groups.push(ParameterGroupSnapshot {
            target: WeightProgramTarget::CfC(target),
            role,
            shape: group.shape.clone(),
            values,
            constraints,
        });
    }
    Ok(groups)
}

fn native_replacements(
    snapshot: &WeightModelSnapshot,
    candidate: &ValidatedWeightCandidate,
) -> Result<Vec<CfCParameterGroupReplacement>, CfCWeightProgramAdapterError> {
    if snapshot.groups.len() != candidate.groups.len() {
        return Err(CfCWeightProgramAdapterError::NonCfCTarget);
    }

    let mut replacements = Vec::new();
    for (before, after) in snapshot.groups.iter().zip(candidate.groups.iter()) {
        if before.target != after.target || before.shape != after.shape {
            return Err(CfCWeightProgramAdapterError::NonCfCTarget);
        }
        if before.values == after.values {
            continue;
        }
        let WeightProgramTarget::CfC(target) = &after.target else {
            return Err(CfCWeightProgramAdapterError::NonCfCTarget);
        };
        replacements.push(CfCParameterGroupReplacement {
            id: native_group_id(target),
            values: after.values.clone(),
        });
    }
    Ok(replacements)
}

fn native_group_id(target: &CfCWeightGroupId) -> CfCParameterGroupId {
    match target {
        CfCWeightGroupId::CellInputProjection { cell } => {
            CfCParameterGroupId::CellInputProjection { cell: *cell }
        }
        CfCWeightGroupId::CellRecurrentProjection { cell } => {
            CfCParameterGroupId::CellRecurrentProjection { cell: *cell }
        }
        CfCWeightGroupId::CellHiddenBias { cell } => {
            CfCParameterGroupId::CellHiddenBias { cell: *cell }
        }
        CfCWeightGroupId::CellTimeConstant { cell } => {
            CfCParameterGroupId::CellTimeConstant { cell: *cell }
        }
        CfCWeightGroupId::CellBackboneWeight { cell, layer } => {
            CfCParameterGroupId::CellBackboneWeight {
                cell: *cell,
                layer: *layer,
            }
        }
        CfCWeightGroupId::CellBackboneBias { cell, layer } => {
            CfCParameterGroupId::CellBackboneBias {
                cell: *cell,
                layer: *layer,
            }
        }
        CfCWeightGroupId::OutputProjection => CfCParameterGroupId::OutputProjection,
        CfCWeightGroupId::OutputBias => CfCParameterGroupId::OutputBias,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::weight_space_program::WeightProgramOperation;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    #[test]
    fn candidate_materialization_does_not_mutate_live_cfc_network() {
        let config = CfCNetworkConfig {
            input_dim: 2,
            hidden_dim: 4,
            num_layers: 1,
            output_dim: 2,
            ..Default::default()
        };
        let network = CfCNetwork::new(config);
        let task_digest = digest('d');
        let snapshot = cfc_weight_model_snapshot(&network, task_digest.clone())
            .expect("snapshot");

        let tau_group = snapshot
            .groups
            .iter()
            .find(|group| {
                group.target
                    == WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant { cell: 0 })
            })
            .expect("cell zero tau group");
        let base_value = tau_group.values[0];

        let program = WeightProgram {
            format_version: 1,
            architecture_digest: snapshot.architecture_digest.clone(),
            manifest_digest: snapshot.manifest_digest.clone(),
            base_model_digest: snapshot.model_digest.clone(),
            target_specification_digest: task_digest,
            fault_hypothesis_ids: vec!["fault:tau-scale".into()],
            evidence_receipt_ids: vec!["receipt:diagnostic-01".into()],
            max_changed_parameters: tau_group.values.len(),
            max_total_l2_delta: 10.0,
            max_abs_delta: 10.0,
            operations: vec![WeightProgramOperation::ScaleRange {
                target: tau_group.target.clone(),
                start: 0,
                end: tau_group.values.len(),
                factor: 1.1,
            }],
        };

        let base_digest_before = network.fingerprint().expect("base fingerprint").model_digest;
        let candidate = materialize_cfc_weight_program(&network, digest('d'), &program)
            .expect("isolated candidate");
        let base_digest_after = network.fingerprint().expect("base fingerprint").model_digest;

        assert_eq!(base_digest_before, base_digest_after);
        assert_eq!(candidate.base_native_model_digest, base_digest_before);
        assert_ne!(
            candidate.candidate_native_model_digest,
            candidate.base_native_model_digest
        );
        let candidate_tau = candidate.network.flattened_tau();
        assert!((candidate_tau[0] - (base_value * 1.1)).abs() < 1e-4);
    }

    #[test]
    fn invalid_target_specification_is_rejected_before_clone_edit() {
        let config = CfCNetworkConfig {
            input_dim: 2,
            hidden_dim: 4,
            num_layers: 1,
            output_dim: 2,
            ..Default::default()
        };
        let network = CfCNetwork::new(config);
        let snapshot = cfc_weight_model_snapshot(&network, digest('d')).expect("snapshot");
        let program = WeightProgram {
            format_version: 1,
            architecture_digest: snapshot.architecture_digest.clone(),
            manifest_digest: snapshot.manifest_digest.clone(),
            base_model_digest: snapshot.model_digest.clone(),
            target_specification_digest: digest('e'),
            fault_hypothesis_ids: vec!["fault:tau-scale".into()],
            evidence_receipt_ids: vec!["receipt:diagnostic-01".into()],
            max_changed_parameters: 1,
            max_total_l2_delta: 1.0,
            max_abs_delta: 1.0,
            operations: vec![WeightProgramOperation::ScaleRange {
                target: WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant { cell: 0 }),
                start: 0,
                end: 1,
                factor: 1.1,
            }],
        };
        assert!(matches!(
            materialize_cfc_weight_program(&network, digest('d'), &program),
            Err(CfCWeightProgramAdapterError::Program(
                WeightProgramError::StaleDigest("target specification")
            ))
        ));
    }
}
