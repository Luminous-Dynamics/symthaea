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
use super::types::{CfCConfig, MIN_TAU};

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
    candidate_network.reset_optimizer_state_after_parameter_edit();

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

    fn permute_rows(values: &[f32], rows: usize, cols: usize, permutation: &[usize]) -> Vec<f32> {
        let mut result = vec![0.0; values.len()];
        for new_row in 0..rows {
            let old_row = permutation[new_row];
            for col in 0..cols {
                result[new_row * cols + col] = values[old_row * cols + col];
            }
        }
        result
    }

    fn permute_square(values: &[f32], dimension: usize, permutation: &[usize]) -> Vec<f32> {
        let mut result = vec![0.0; values.len()];
        for new_row in 0..dimension {
            for new_col in 0..dimension {
                result[new_row * dimension + new_col] =
                    values[permutation[new_row] * dimension + permutation[new_col]];
            }
        }
        result
    }

    fn permute_columns(
        values: &[f32],
        rows: usize,
        cols: usize,
        permutation: &[usize],
    ) -> Vec<f32> {
        let mut result = vec![0.0; values.len()];
        for row in 0..rows {
            for new_col in 0..cols {
                result[row * cols + new_col] = values[row * cols + permutation[new_col]];
            }
        }
        result
    }

    #[test]
    fn consistent_hidden_unit_permutation_preserves_cfc_function() {
        let config = CfCNetworkConfig {
            input_dim: 2,
            hidden_dim: 4,
            num_layers: 1,
            output_dim: 2,
            cell_config: CfCConfig {
                input_dim: 2,
                hidden_dim: 4,
                use_backbone: false,
                dropout: 0.0,
                ..Default::default()
            },
            residual: false,
            bidirectional: false,
            enable_online_learning: false,
            ..Default::default()
        };
        let original = CfCNetwork::new(config);
        let snapshot =
            cfc_weight_model_snapshot(&original, digest('d')).expect("snapshot");
        let permutation = [2usize, 0, 3, 1];
        let get = |id: CfCWeightGroupId| {
            snapshot
                .groups
                .iter()
                .find(|group| group.target == WeightProgramTarget::CfC(id.clone()))
                .expect("parameter group")
                .values
                .clone()
        };

        let input = get(CfCWeightGroupId::CellInputProjection { cell: 0 });
        let recurrent = get(CfCWeightGroupId::CellRecurrentProjection { cell: 0 });
        let bias = get(CfCWeightGroupId::CellHiddenBias { cell: 0 });
        let tau = get(CfCWeightGroupId::CellTimeConstant { cell: 0 });
        let output = get(CfCWeightGroupId::OutputProjection);
        let task_digest = digest('d');
        let program = WeightProgram {
            format_version: 1,
            architecture_digest: snapshot.architecture_digest.clone(),
            manifest_digest: snapshot.manifest_digest.clone(),
            base_model_digest: snapshot.model_digest.clone(),
            target_specification_digest: task_digest.clone(),
            fault_hypothesis_ids: vec!["hypothesis:function-preserving-permutation".into()],
            evidence_receipt_ids: vec!["receipt:permutation-probe-set-v1".into()],
            max_changed_parameters: 40,
            max_total_l2_delta: 100.0,
            max_abs_delta: 100.0,
            operations: vec![
                WeightProgramOperation::ReplaceGroup {
                    target: WeightProgramTarget::CfC(CfCWeightGroupId::CellInputProjection { cell: 0 }),
                    values: permute_rows(&input, 4, 2, &permutation),
                },
                WeightProgramOperation::ReplaceGroup {
                    target: WeightProgramTarget::CfC(CfCWeightGroupId::CellRecurrentProjection { cell: 0 }),
                    values: permute_square(&recurrent, 4, &permutation),
                },
                WeightProgramOperation::ReplaceGroup {
                    target: WeightProgramTarget::CfC(CfCWeightGroupId::CellHiddenBias { cell: 0 }),
                    values: permutation.iter().map(|index| bias[*index]).collect(),
                },
                WeightProgramOperation::ReplaceGroup {
                    target: WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant { cell: 0 }),
                    values: permutation.iter().map(|index| tau[*index]).collect(),
                },
                WeightProgramOperation::ReplaceGroup {
                    target: WeightProgramTarget::CfC(CfCWeightGroupId::OutputProjection),
                    values: permute_columns(&output, 2, 4, &permutation),
                },
            ],
        };
        let candidate = materialize_cfc_weight_program(&original, task_digest, &program)
            .expect("function-preserving WSP candidate");
        let permuted = candidate.network;

        assert_ne!(
            candidate.base_native_model_digest,
            candidate.candidate_native_model_digest,
            "the parameter fingerprint must distinguish equivalent parameterizations"
        );
        assert_eq!(candidate.receipt.edited_group_count, 5);
        assert!(candidate.receipt.changed_parameter_count <= 40);

        let inputs = [
            ndarray::Array1::from_vec(vec![0.25, -0.75]),
            ndarray::Array1::from_vec(vec![-0.4, 0.6]),
            ndarray::Array1::from_vec(vec![0.9, 0.1]),
            ndarray::Array1::from_vec(vec![-0.2, -0.8]),
        ];
        let mut original_eval = original.clone();
        let mut permuted_eval = permuted.clone();
        let mut squared_error = 0.0_f64;
        for input in &inputs {
            let expected = original_eval.forward(input, 0.1);
            let actual = permuted_eval.forward(input, 0.1);
            for (a, b) in expected.iter().zip(actual.iter()) {
                squared_error += f64::from((*a - *b).powi(2));
            }
        }
        assert!(
            squared_error.sqrt() < 1e-5,
            "permutation-only parameter change must preserve outputs, error={}",
            squared_error.sqrt()
        );
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
    fn planted_tau_fault_is_repaired_on_clone_and_stale_program_rejects_healthy_model() {
        let config = CfCNetworkConfig {
            input_dim: 2,
            hidden_dim: 4,
            num_layers: 1,
            output_dim: 2,
            ..Default::default()
        };
        let healthy = CfCNetwork::new(config);
        let mut faulty = healthy.clone();
        let task_digest = digest('d');
        let healthy_snapshot =
            cfc_weight_model_snapshot(&healthy, task_digest.clone()).expect("healthy snapshot");
        let tau_target = WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant {
            cell: 0,
        });
        let healthy_tau = healthy_snapshot
            .groups
            .iter()
            .find(|group| group.target == tau_target)
            .expect("healthy tau")
            .values
            .clone();

        let faulty_values = healthy_tau.iter().map(|value| value * 3.0).collect::<Vec<_>>();
        let faulty_base_digest = faulty.fingerprint().expect("faulty base").model_digest;
        faulty
            .try_apply_parameter_group_replacements(
                &faulty_base_digest,
                &[CfCParameterGroupReplacement {
                    id: native_group_id(&CfCWeightGroupId::CellTimeConstant { cell: 0 }),
                    values: faulty_values,
                }],
            )
            .expect("plant tau-scale fault");

        // Seed a nonzero recurrent state and stale optimizer counters so the
        // adapter must reset them on the candidate, without touching the source.
        let _ = faulty.forward(
            &ndarray::Array1::from_vec(vec![0.25, -0.75]),
            0.2,
        );
        let source_state_norm = faulty.cells[0]
            .state()
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!(source_state_norm > 0.0);
        faulty.adam_states[0].t = 7;
        faulty.adam_output.t = 7;

        let faulty_snapshot =
            cfc_weight_model_snapshot(&faulty, task_digest.clone()).expect("faulty snapshot");
        let faulty_digest_before = faulty_snapshot.model_digest.clone();
        let tau_len = faulty_snapshot
            .groups
            .iter()
            .find(|group| group.target == tau_target)
            .expect("faulty tau")
            .values
            .len();
        let repair = WeightProgram {
            format_version: 1,
            architecture_digest: faulty_snapshot.architecture_digest.clone(),
            manifest_digest: faulty_snapshot.manifest_digest.clone(),
            base_model_digest: faulty_snapshot.model_digest.clone(),
            target_specification_digest: task_digest.clone(),
            fault_hypothesis_ids: vec!["fault:tau-scale-3x".into()],
            evidence_receipt_ids: vec!["receipt:impulse-response-01".into()],
            max_changed_parameters: tau_len,
            max_total_l2_delta: 100.0,
            max_abs_delta: 100.0,
            operations: vec![WeightProgramOperation::ScaleRange {
                target: tau_target,
                start: 0,
                end: tau_len,
                factor: 1.0 / 3.0,
            }],
        };

        let candidate =
            materialize_cfc_weight_program(&faulty, task_digest.clone(), &repair)
                .expect("candidate repair");
        assert_eq!(
            faulty.fingerprint().expect("faulty remains unchanged").model_digest,
            faulty_digest_before
        );
        let source_state_after = faulty.cells[0]
            .state()
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert_eq!(source_state_after, source_state_norm);
        assert_eq!(faulty.adam_states[0].t, 7);
        assert_eq!(faulty.adam_output.t, 7);

        let candidate_state_norm = candidate.network.cells[0]
            .state()
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert_eq!(candidate_state_norm, 0.0);
        assert!(candidate.network.adam_states.iter().all(|state| state.t == 0));
        assert_eq!(candidate.network.adam_output.t, 0);

        let repaired_tau = candidate.network.flattened_tau();
        for (repaired, expected) in repaired_tau.iter().zip(healthy_tau.iter()) {
            assert!((repaired - expected).abs() < 1e-5);
        }

        // Behavioral check: use a fixed held-out sequence on separate clones.
        let inputs = [
            ndarray::Array1::from_vec(vec![0.25, -0.75]),
            ndarray::Array1::from_vec(vec![-0.4, 0.6]),
            ndarray::Array1::from_vec(vec![0.9, 0.1]),
            ndarray::Array1::from_vec(vec![-0.2, -0.8]),
        ];
        let mut healthy_eval = healthy.clone();
        let mut faulty_eval = faulty.clone();
        faulty_eval.reset_states_only();
        let mut repaired_eval = candidate.network.clone();
        let mut faulty_output_error = 0.0_f64;
        let mut repaired_output_error = 0.0_f64;
        for input in &inputs {
            let expected = healthy_eval.forward(input, 0.2);
            let faulty_output = faulty_eval.forward(input, 0.2);
            let repaired_output = repaired_eval.forward(input, 0.2);
            for ((expected, faulty), repaired) in expected
                .iter()
                .zip(faulty_output.iter())
                .zip(repaired_output.iter())
            {
                faulty_output_error += f64::from((*expected - *faulty).powi(2));
                repaired_output_error += f64::from((*expected - *repaired).powi(2));
            }
        }
        assert!(
            faulty_output_error.sqrt() > 1e-9,
            "the planted tau fault must affect held-out outputs"
        );
        assert!(
            repaired_output_error.sqrt() < 1e-6,
            "the isolated repair should recover the healthy output sequence"
        );

        let healthy_now = cfc_weight_model_snapshot(&healthy, task_digest).expect("healthy current");
        assert_eq!(
            repair.validate_and_materialize(&healthy_now),
            Err(WeightProgramError::StaleDigest("base model"))
        );
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
