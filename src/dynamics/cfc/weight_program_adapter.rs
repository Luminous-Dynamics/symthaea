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


/// One frozen functional probe used to construct hidden-unit activation signatures.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CfCFunctionalProbe {
    /// Input vector for one CfC time step.
    pub input: Vec<f32>,
    /// Elapsed time for this step.
    pub dt: f32,
}

/// Activation signatures bound to one model fingerprint and one ordered probe set.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CfCActivationSignatureSet {
    /// Schema version for serialized signature receipts.
    pub format_version: u32,
    /// Native model fingerprint from which these signatures were measured.
    pub model_digest: String,
    /// Digest of the ordered elapsed-time/input probe sequence.
    pub probe_digest: String,
    /// Layer whose hidden units were measured.
    pub layer_index: usize,
    /// Signatures indexed by hidden unit then probe index.
    pub signatures: Vec<Vec<f64>>,
}

/// Activation signatures for one layer, indexed by hidden unit then probe index.
///
/// The source model is cloned and reset before the probe sequence. The caller's
/// network is never mutated. These signatures are diagnostic metadata only.
pub fn cfc_hidden_unit_activation_signatures(
    network: &CfCNetwork,
    layer_index: usize,
    probes: &[CfCFunctionalProbe],
) -> Result<CfCActivationSignatureSet, CfCPermutationAlignmentError> {
    if probes.is_empty() {
        return Err(CfCPermutationAlignmentError::EmptyProbeSet);
    }
    if layer_index >= network.config().num_layers {
        return Err(CfCPermutationAlignmentError::InvalidLayerIndex {
            layer_index,
            layer_count: network.config().num_layers,
        });
    }

    let model_digest = network
        .fingerprint()
        .map_err(CfCPermutationAlignmentError::FingerprintFailed)?
        .model_digest;
    let probe_digest = cfc_probe_set_digest(layer_index, probes);

    let mut evaluation = network.clone();
    evaluation.reset();
    let hidden_dim = network.config().hidden_dim;
    let mut signatures = vec![Vec::with_capacity(probes.len()); hidden_dim];

    for (probe_index, probe) in probes.iter().enumerate() {
        // Probes always enter through the network's external input interface.
        // The requested layer is observed after the preceding layers propagate them.
        let expected_input_dim = network.config().input_dim;
        if probe.input.len() != expected_input_dim {
            return Err(CfCPermutationAlignmentError::ProbeInputDimensionMismatch {
                probe_index,
                expected: expected_input_dim,
                actual: probe.input.len(),
            });
        }
        if !probe.dt.is_finite() || probe.dt < 0.0 {
            return Err(CfCPermutationAlignmentError::InvalidProbeElapsedTime {
                probe_index,
                dt: probe.dt,
            });
        }
        if let Some(value_index) = probe.input.iter().position(|value| !value.is_finite()) {
            return Err(CfCPermutationAlignmentError::NonFiniteProbeInput {
                probe_index,
                value_index,
            });
        }

        let input = ndarray::Array1::from_vec(probe.input.clone());
        let _ = evaluation.forward(&input, probe.dt);
        let states = evaluation.state();
        let layer_state = states.get(layer_index).ok_or(
            CfCPermutationAlignmentError::InvalidLayerIndex {
                layer_index,
                layer_count: states.len(),
            },
        )?;
        for (unit_index, value) in layer_state.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(CfCPermutationAlignmentError::NonFiniteActivation {
                    layer_index,
                    unit_index,
                    probe_index,
                });
            }
            signatures[unit_index].push(f64::from(value));
        }
    }

    Ok(CfCActivationSignatureSet {
        format_version: 1,
        model_digest,
        probe_digest,
        layer_index,
        signatures,
    })
}

fn is_lowercase_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn cfc_probe_set_digest(layer_index: usize, probes: &[CfCFunctionalProbe]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-cfc-functional-probes-v1\0");
    hasher.update(&(layer_index as u64).to_le_bytes());
    hasher.update(&(probes.len() as u64).to_le_bytes());
    for probe in probes {
        hasher.update(&(probe.input.len() as u64).to_le_bytes());
        for value in &probe.input {
            hasher.update(&value.to_le_bytes());
        }
        hasher.update(&probe.dt.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

/// Result of matching source hidden units to candidate hidden units by probes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CfCHiddenUnitAlignment {
    /// Schema version for serialized alignment receipts.
    pub format_version: u32,
    /// Source model fingerprint to which the mapping is bound.
    pub source_model_digest: String,
    /// Candidate model fingerprint to which the mapping is bound.
    pub candidate_model_digest: String,
    /// Ordered functional probe-set digest used for this alignment.
    pub probe_digest: String,
    /// Layer whose units were matched.
    pub layer_index: usize,
    /// For each source unit, the index of its corresponding candidate unit.
    pub source_to_candidate: Vec<usize>,
    /// Mean squared activation-signature error across all matched units/probes.
    pub mean_squared_activation_error: f64,
    /// Minimum global assignment-cost increase when any selected pair is forbidden.
    ///
    /// A value near zero indicates that multiple assignments explain the probes
    /// equally well. Positive infinity means there is no alternative complete
    /// assignment. Consumers should abstain from alignment when this margin is
    /// below a task-calibrated threshold.
    pub minimum_assignment_margin: Option<f64>,
}

/// Fail-closed errors for functional hidden-unit correspondence.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CfCPermutationAlignmentError {
    #[error("at least one hidden unit is required")]
    EmptyHiddenUnitSet,
    #[error("source and candidate have different hidden-unit counts: {source} vs {candidate}")]
    HiddenUnitCountMismatch { source: usize, candidate: usize },
    #[error("{side} activation signature set has unsupported format version {version}")]
    UnsupportedSignatureVersion { side: &'static str, version: u32 },
    #[error("{side} activation signature set has an invalid {field} digest")]
    InvalidSignatureDigest {
        side: &'static str,
        field: &'static str,
    },
    #[error("activation signature set for {side} is empty")]
    EmptySignatureSet { side: &'static str },
    #[error("{side} activation signature {unit_index} has {actual} probes, expected {expected}")]
    SignatureLengthMismatch {
        side: &'static str,
        unit_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("{side} activation signature {unit_index} contains a non-finite value at probe {probe_index}")]
    NonFiniteSignature {
        side: &'static str,
        unit_index: usize,
        probe_index: usize,
    },
    #[error("no finite complete hidden-unit assignment exists")]
    NoFeasibleAssignment,
    #[error("source and candidate probe digests differ: {source} vs {candidate}")]
    ProbeSetMismatch { source: String, candidate: String },
    #[error("source and candidate layer indices differ: {source} vs {candidate}")]
    LayerMismatch { source: usize, candidate: usize },
    #[error("CfC fingerprint failed while binding activation signatures: {0}")]
    FingerprintFailed(String),
    #[error("at least one functional probe is required")]
    EmptyProbeSet,
    #[error("layer index {layer_index} is outside 0..{layer_count}")]
    InvalidLayerIndex {
        layer_index: usize,
        layer_count: usize,
    },
    #[error("probe {probe_index} input has length {actual}, expected {expected}")]
    ProbeInputDimensionMismatch {
        probe_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("probe {probe_index} has invalid elapsed time {dt}")]
    InvalidProbeElapsedTime { probe_index: usize, dt: f32 },
    #[error("probe {probe_index} input element {value_index} is non-finite")]
    NonFiniteProbeInput {
        probe_index: usize,
        value_index: usize,
    },
    #[error("layer {layer_index}, unit {unit_index}, probe {probe_index} produced a non-finite activation")]
    NonFiniteActivation {
        layer_index: usize,
        unit_index: usize,
        probe_index: usize,
    },
}

/// Infer a one-to-one hidden-unit correspondence from frozen activation probes.
///
/// This is a diagnostic alignment, not an edit: it does not reorder tensors,
/// alter native fingerprints, or authorize a WeightProgram. Matching minimizes
/// total mean-squared activation-signature distance with a deterministic
/// Hungarian assignment. The global assignment margin exposes ambiguity.
pub fn align_cfc_hidden_unit_signatures(
    source_signature_set: &CfCActivationSignatureSet,
    candidate_signature_set: &CfCActivationSignatureSet,
) -> Result<CfCHiddenUnitAlignment, CfCPermutationAlignmentError> {
    for (side, signature_set) in [
        ("source", source_signature_set),
        ("candidate", candidate_signature_set),
    ] {
        if signature_set.format_version != 1 {
            return Err(CfCPermutationAlignmentError::UnsupportedSignatureVersion {
                side,
                version: signature_set.format_version,
            });
        }
        if !is_lowercase_hex_digest(&signature_set.model_digest) {
            return Err(CfCPermutationAlignmentError::InvalidSignatureDigest {
                side,
                field: "model",
            });
        }
        if !is_lowercase_hex_digest(&signature_set.probe_digest) {
            return Err(CfCPermutationAlignmentError::InvalidSignatureDigest {
                side,
                field: "probe",
            });
        }
    }
    if source_signature_set.probe_digest != candidate_signature_set.probe_digest {
        return Err(CfCPermutationAlignmentError::ProbeSetMismatch {
            source: source_signature_set.probe_digest.clone(),
            candidate: candidate_signature_set.probe_digest.clone(),
        });
    }
    if source_signature_set.layer_index != candidate_signature_set.layer_index {
        return Err(CfCPermutationAlignmentError::LayerMismatch {
            source: source_signature_set.layer_index,
            candidate: candidate_signature_set.layer_index,
        });
    }
    let source_signatures = &source_signature_set.signatures;
    let candidate_signatures = &candidate_signature_set.signatures;

    if source_signatures.is_empty() {
        return Err(CfCPermutationAlignmentError::EmptyHiddenUnitSet);
    }
    if candidate_signatures.is_empty() {
        return Err(CfCPermutationAlignmentError::EmptySignatureSet { side: "candidate" });
    }
    if source_signatures.len() != candidate_signatures.len() {
        return Err(CfCPermutationAlignmentError::HiddenUnitCountMismatch {
            source: source_signatures.len(),
            candidate: candidate_signatures.len(),
        });
    }

    let probe_count = source_signatures[0].len();
    if probe_count == 0 {
        return Err(CfCPermutationAlignmentError::EmptySignatureSet { side: "source" });
    }
    for (side, signatures) in [
        ("source", source_signatures),
        ("candidate", candidate_signatures),
    ] {
        for (unit_index, signature) in signatures.iter().enumerate() {
            if signature.len() != probe_count {
                return Err(CfCPermutationAlignmentError::SignatureLengthMismatch {
                    side,
                    unit_index,
                    expected: probe_count,
                    actual: signature.len(),
                });
            }
            if let Some(probe_index) = signature.iter().position(|value| !value.is_finite()) {
                return Err(CfCPermutationAlignmentError::NonFiniteSignature {
                    side,
                    unit_index,
                    probe_index,
                });
            }
        }
    }

    let unit_count = source_signatures.len();
    let costs = source_signatures
        .iter()
        .map(|source| {
            candidate_signatures
                .iter()
                .map(|candidate| {
                    source
                        .iter()
                        .zip(candidate.iter())
                        .map(|(left, right)| (left - right).powi(2))
                        .sum::<f64>()
                        / probe_count as f64
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let (source_to_candidate, best_total_cost) =
        minimum_cost_assignment(&costs).ok_or(CfCPermutationAlignmentError::NoFeasibleAssignment)?;

    let minimum_assignment_margin =
        alternating_cycle_assignment_margin(&costs, &source_to_candidate);

    Ok(CfCHiddenUnitAlignment {
        format_version: 1,
        source_model_digest: source_signature_set.model_digest.clone(),
        candidate_model_digest: candidate_signature_set.model_digest.clone(),
        probe_digest: source_signature_set.probe_digest.clone(),
        layer_index: source_signature_set.layer_index,
        source_to_candidate,
        mean_squared_activation_error: best_total_cost / unit_count as f64,
        minimum_assignment_margin,
    })
}

/// Compute the smallest cost increase of any reassignment that forbids one
/// matched pair. Alternative assignments decompose into alternating cycles.
/// Floyd-Warshall over the row-transition graph computes all shortest cycles in
/// O(n^3), avoiding n separate O(n^3) Hungarian re-solves.
fn alternating_cycle_assignment_margin(
    costs: &[Vec<f64>],
    assignment: &[usize],
) -> Option<f64> {
    let n = assignment.len();
    if n <= 1 {
        return None;
    }

    let mut distance = vec![vec![f64::INFINITY; n]; n];
    for source_row in 0..n {
        let source_cost = costs[source_row][assignment[source_row]];
        for next_row in 0..n {
            if source_row == next_row {
                continue;
            }
            let reassigned_cost = costs[source_row][assignment[next_row]];
            if reassigned_cost.is_finite() && source_cost.is_finite() {
                distance[source_row][next_row] = reassigned_cost - source_cost;
            }
        }
    }

    for intermediate in 0..n {
        for source in 0..n {
            if !distance[source][intermediate].is_finite() {
                continue;
            }
            for destination in 0..n {
                if !distance[intermediate][destination].is_finite() {
                    continue;
                }
                let through = distance[source][intermediate]
                    + distance[intermediate][destination];
                if through < distance[source][destination] {
                    distance[source][destination] = through;
                }
            }
        }
    }

    let mut margin = f64::INFINITY;
    for source in 0..n {
        for next_row in 0..n {
            if source == next_row || !distance[next_row][source].is_finite() {
                continue;
            }
            let cycle_cost = costs[source][assignment[next_row]]
                - costs[source][assignment[source]]
                + distance[next_row][source];
            if cycle_cost.is_finite() {
                margin = margin.min(cycle_cost.max(0.0));
            }
        }
    }
    margin.is_finite().then_some(margin)
}

fn minimum_cost_assignment(costs: &[Vec<f64>]) -> Option<(Vec<usize>, f64)> {
    let n = costs.len();
    if n == 0 || costs.iter().any(|row| row.len() != n) {
        return None;
    }

    // One-indexed Hungarian algorithm. Positive infinity marks a forbidden edge.
    let mut row_potential = vec![0.0_f64; n + 1];
    let mut column_potential = vec![0.0_f64; n + 1];
    let mut matched_row = vec![0_usize; n + 1];
    let mut predecessor = vec![0_usize; n + 1];

    for row in 1..=n {
        matched_row[0] = row;
        let mut current_column = 0_usize;
        let mut minimum_reduced_cost = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[current_column] = true;
            let current_row = matched_row[current_column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0_usize;

            for column in 1..=n {
                if used[column] {
                    continue;
                }
                let edge_cost = costs[current_row - 1][column - 1];
                if edge_cost.is_finite() {
                    let reduced_cost =
                        edge_cost - row_potential[current_row] - column_potential[column];
                    if reduced_cost < minimum_reduced_cost[column] {
                        minimum_reduced_cost[column] = reduced_cost;
                        predecessor[column] = current_column;
                    }
                }
                if minimum_reduced_cost[column] < delta {
                    delta = minimum_reduced_cost[column];
                    next_column = column;
                }
            }

            if !delta.is_finite() {
                return None;
            }
            for column in 0..=n {
                if used[column] {
                    row_potential[matched_row[column]] += delta;
                    column_potential[column] -= delta;
                } else if minimum_reduced_cost[column].is_finite() {
                    minimum_reduced_cost[column] -= delta;
                }
            }

            current_column = next_column;
            if matched_row[current_column] == 0 {
                break;
            }
        }

        loop {
            let previous_column = predecessor[current_column];
            matched_row[current_column] = matched_row[previous_column];
            current_column = previous_column;
            if current_column == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![usize::MAX; n];
    for column in 1..=n {
        if matched_row[column] == 0 {
            return None;
        }
        assignment[matched_row[column] - 1] = column - 1;
    }
    let mut total_cost = 0.0_f64;
    for (row, column) in assignment.iter().copied().enumerate() {
        let edge_cost = *costs.get(row)?.get(column)?;
        if !edge_cost.is_finite() {
            return None;
        }
        total_cost += edge_cost;
    }
    total_cost
        .is_finite()
        .then_some((assignment, total_cost))
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::weight_space_program::WeightProgramOperation;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn signature_set(
        signatures: Vec<Vec<f64>>,
        model: char,
        probe: char,
    ) -> CfCActivationSignatureSet {
        CfCActivationSignatureSet {
            format_version: 1,
            model_digest: digest(model),
            probe_digest: digest(probe),
            layer_index: 0,
            signatures,
        }
    }

    #[test]
    fn activation_signature_receipt_binds_probe_order_and_is_repeatable() {
        let config = CfCNetworkConfig {
            input_dim: 1,
            hidden_dim: 2,
            num_layers: 1,
            output_dim: 1,
            cell_config: CfCConfig {
                input_dim: 1,
                hidden_dim: 2,
                use_backbone: false,
                dropout: 0.0,
                ..Default::default()
            },
            residual: false,
            bidirectional: false,
            enable_online_learning: false,
            ..Default::default()
        };
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(
            "symthaea-cfc-probe-digest-test-v1",
        );
        let network = CfCNetwork::from_genesis(config, &genesis, "probe-digest");
        let probes = vec![
            CfCFunctionalProbe {
                input: vec![0.2],
                dt: 0.05,
            },
            CfCFunctionalProbe {
                input: vec![-0.7],
                dt: 0.2,
            },
            CfCFunctionalProbe {
                input: vec![0.9],
                dt: 0.1,
            },
        ];
        let first =
            cfc_hidden_unit_activation_signatures(&network, 0, &probes).expect("first receipt");
        let repeated =
            cfc_hidden_unit_activation_signatures(&network, 0, &probes).expect("repeat receipt");
        let mut reordered_probes = probes.clone();
        reordered_probes.reverse();
        let reordered =
            cfc_hidden_unit_activation_signatures(&network, 0, &reordered_probes)
                .expect("reordered receipt");

        assert_eq!(first.probe_digest, repeated.probe_digest);
        assert_ne!(first.probe_digest, reordered.probe_digest);
        assert_eq!(first.model_digest, reordered.model_digest);
        assert_eq!(first.signatures, repeated.signatures);
    }

    #[test]
    fn alternating_cycle_margin_matches_small_exact_assignment_fixture() {
        let costs = vec![
            vec![0.0, 4.0, 9.0],
            vec![7.0, 0.0, 5.0],
            vec![3.0, 8.0, 0.0],
        ];
        let assignment = vec![0, 1, 2];
        assert_eq!(
            alternating_cycle_assignment_margin(&costs, &assignment),
            Some(11.0)
        );
    }

    fn enumerate_assignment_costs(
        costs: &[Vec<f64>],
        row: usize,
        used_columns: &mut [bool],
        accumulated: f64,
        totals: &mut Vec<f64>,
    ) {
        if row == costs.len() {
            totals.push(accumulated);
            return;
        }
        for column in 0..costs.len() {
            if used_columns[column] {
                continue;
            }
            used_columns[column] = true;
            enumerate_assignment_costs(
                costs,
                row + 1,
                used_columns,
                accumulated + costs[row][column],
                totals,
            );
            used_columns[column] = false;
        }
    }

    #[test]
    fn assignment_margin_matches_brute_force_second_best_cost() {
        let mut random_state = 0x5eed_cafe_d15c_a11u64;
        for dimension in 2..=6 {
            for _case in 0..12 {
                let mut costs = vec![vec![0.0; dimension]; dimension];
                for row in &mut costs {
                    for value in row {
                        random_state = random_state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        *value = f64::from((random_state >> 32) as u32 % 10_000) / 101.0;
                    }
                }

                let (assignment, best_cost) =
                    minimum_cost_assignment(&costs).expect("complete finite assignment");
                let mut totals = Vec::new();
                enumerate_assignment_costs(
                    &costs,
                    0,
                    &mut vec![false; dimension],
                    0.0,
                    &mut totals,
                );
                totals.sort_by(f64::total_cmp);
                let brute_best = totals[0];
                let brute_second = totals[1];
                let brute_margin = brute_second - brute_best;
                let measured_margin =
                    alternating_cycle_assignment_margin(&costs, &assignment)
                        .expect("complete finite matrix has an alternative assignment");

                assert!(
                    (best_cost - brute_best).abs() < 1e-9,
                    "Hungarian optimum mismatch at dimension {dimension}"
                );
                assert!(
                    (measured_margin - brute_margin).abs() < 1e-8,
                    "assignment margin mismatch at dimension {dimension}: {measured_margin} vs {brute_margin}"
                );
            }
        }
    }

    #[test]
    fn activation_signature_alignment_recovers_hidden_unit_permutation() {
        let source = vec![
            vec![0.0, 0.2, 0.8, -0.1],
            vec![0.7, -0.3, 0.1, 0.9],
            vec![-0.2, 0.4, -0.6, 0.3],
        ];
        // candidate[new_unit] = source[permutation[new_unit]]
        let permutation = [2usize, 0, 1];
        let candidate = permutation
            .iter()
            .map(|source_unit| source[*source_unit].clone())
            .collect::<Vec<_>>();
        let source = signature_set(source, 'a', 'c');
        let candidate = signature_set(candidate, 'b', 'c');
        let alignment =
            align_cfc_hidden_unit_signatures(&source, &candidate).expect("alignment");

        assert_eq!(alignment.source_to_candidate, vec![1, 2, 0]);
        assert_eq!(alignment.source_model_digest, digest('a'));
        assert_eq!(alignment.candidate_model_digest, digest('b'));
        assert_eq!(alignment.probe_digest, digest('p'));
        assert!(alignment.mean_squared_activation_error < 1e-15);
        assert!(alignment.minimum_assignment_margin.is_some_and(|margin| margin > 0.0));
    }

    #[test]
    fn activation_signatures_use_hidden_dimension_for_deeper_layers() {
        let genesis =
            symthaea_core::genesis::GenesisSeed::from_phrase("cfc-deep-layer-probe-v1");
        let config = CfCNetworkConfig {
            input_dim: 3,
            hidden_dim: 4,
            num_layers: 2,
            output_dim: 2,
            residual: false,
            ..Default::default()
        };
        let network = CfCNetwork::from_genesis(config, &genesis, "two-layer-probe");
        let probes = vec![CfCFunctionalProbe {
            input: vec![0.2, -0.3, 0.7],
            dt: 0.05,
        }];
        let signatures =
            cfc_hidden_unit_activation_signatures(&network, 1, &probes)
                .expect("layer one is observed after external input propagation");
        assert_eq!(signatures.signatures.len(), 4);
        assert!(signatures
            .signatures
            .iter()
            .all(|signature| signature.len() == 1));

        let invalid = vec![CfCFunctionalProbe {
            input: vec![0.2, -0.3, 0.7, 0.1],
            dt: 0.05,
        }];
        assert!(matches!(
            cfc_hidden_unit_activation_signatures(&network, 1, &invalid),
            Err(CfCPermutationAlignmentError::ProbeInputDimensionMismatch {
                expected: 3,
                actual: 4,
                ..
            })
        ));
    }

    #[test]
    fn activation_signature_alignment_exposes_ambiguous_units() {
        let source = vec![vec![0.0, 1.0], vec![0.0, 1.0], vec![2.0, 3.0]];
        let candidate = vec![vec![2.0, 3.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let source = signature_set(source, 'a', 'c');
        let candidate = signature_set(candidate, 'b', 'c');
        let alignment =
            align_cfc_hidden_unit_signatures(&source, &candidate).expect("alignment");

        assert!(alignment.mean_squared_activation_error < 1e-15);
        assert!(alignment.minimum_assignment_margin.is_some_and(|margin| margin < 1e-15));
    }

    #[test]
    fn activation_alignment_rejects_different_probe_sets_and_layers() {
        let source = signature_set(vec![vec![0.0, 1.0]], 'a', 'c');
        let different_probes = signature_set(vec![vec![0.0, 1.0]], 'b', 'd');
        assert!(matches!(
            align_cfc_hidden_unit_signatures(&source, &different_probes),
            Err(CfCPermutationAlignmentError::ProbeSetMismatch { .. })
        ));

        let mut different_layer = signature_set(vec![vec![0.0, 1.0]], 'b', 'c');
        different_layer.layer_index = 1;
        assert!(matches!(
            align_cfc_hidden_unit_signatures(&source, &different_layer),
            Err(CfCPermutationAlignmentError::LayerMismatch { .. })
        ));
    }

    #[test]
    fn activation_signature_alignment_rejects_malformed_inputs() {
        assert!(matches!(
            align_cfc_hidden_unit_signatures(
                &signature_set(Vec::new(), 'a', 'c'),
                &signature_set(Vec::new(), 'b', 'c')
            ),
            Err(CfCPermutationAlignmentError::EmptyHiddenUnitSet)
        ));
        assert!(matches!(
            align_cfc_hidden_unit_signatures(
                &signature_set(vec![vec![0.0]], 'a', 'c'),
                &signature_set(vec![vec![0.0, 1.0]], 'b', 'c')
            ),
            Err(CfCPermutationAlignmentError::SignatureLengthMismatch { .. })
        ));
        assert!(matches!(
            align_cfc_hidden_unit_signatures(
                &signature_set(vec![vec![f64::NAN]], 'a', 'c'),
                &signature_set(vec![vec![0.0]], 'b', 'c')
            ),
            Err(CfCPermutationAlignmentError::NonFiniteSignature { .. })
        ));
        assert!(matches!(
            align_cfc_hidden_unit_signatures(
                &signature_set(vec![vec![0.0], vec![1.0]], 'a', 'c'),
                &signature_set(vec![vec![0.0]], 'b', 'c')
            ),
            Err(CfCPermutationAlignmentError::HiddenUnitCountMismatch { .. })
        ));

        let mut invalid_digest = signature_set(vec![vec![0.0]], 'a', 'c');
        invalid_digest.model_digest = "not-a-256-bit-digest".into();
        assert!(matches!(
            align_cfc_hidden_unit_signatures(
                &invalid_digest,
                &signature_set(vec![vec![0.0]], 'b', 'c')
            ),
            Err(CfCPermutationAlignmentError::InvalidSignatureDigest {
                side: "source",
                field: "model"
            })
        ));

        let mut unsupported_version = signature_set(vec![vec![0.0]], 'a', 'c');
        unsupported_version.format_version = 2;
        assert!(matches!(
            align_cfc_hidden_unit_signatures(
                &unsupported_version,
                &signature_set(vec![vec![0.0]], 'b', 'c')
            ),
            Err(CfCPermutationAlignmentError::UnsupportedSignatureVersion {
                side: "source",
                version: 2
            })
        ));
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
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(
            "symthaea-swi-permutation-equivalence-v1",
        );
        let original = CfCNetwork::from_genesis(config, &genesis, "permutation-equivalence");
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
        assert!(candidate.receipt.changed_parameter_count > 0);
        assert!(candidate.receipt.changed_parameter_count <= 40);

        let inconsistent_program = WeightProgram {
            format_version: 1,
            architecture_digest: snapshot.architecture_digest.clone(),
            manifest_digest: snapshot.manifest_digest.clone(),
            base_model_digest: snapshot.model_digest.clone(),
            target_specification_digest: digest('d'),
            fault_hypothesis_ids: vec!["negative-control:partial-permutation".into()],
            evidence_receipt_ids: vec!["receipt:permutation-probe-set-v1".into()],
            max_changed_parameters: 8,
            max_total_l2_delta: 100.0,
            max_abs_delta: 100.0,
            operations: vec![WeightProgramOperation::ReplaceGroup {
                target: WeightProgramTarget::CfC(CfCWeightGroupId::CellInputProjection { cell: 0 }),
                values: permute_rows(&input, 4, 2, &permutation),
            }],
        };
        let inconsistent =
            materialize_cfc_weight_program(&original, digest('d'), &inconsistent_program)
                .expect("deliberately incomplete permutation candidate");

        let inputs = [
            ndarray::Array1::from_vec(vec![0.25, -0.75]),
            ndarray::Array1::from_vec(vec![-0.4, 0.6]),
            ndarray::Array1::from_vec(vec![0.9, 0.1]),
            ndarray::Array1::from_vec(vec![-0.2, -0.8]),
        ];
        let probes = inputs
            .iter()
            .map(|input| CfCFunctionalProbe {
                input: input.to_vec(),
                dt: 0.1,
            })
            .collect::<Vec<_>>();
        let source_state_before = original.state();
        let source_signatures =
            cfc_hidden_unit_activation_signatures(&original, 0, &probes)
                .expect("source activation signatures");
        let candidate_signatures =
            cfc_hidden_unit_activation_signatures(&permuted, 0, &probes)
                .expect("candidate activation signatures");
        assert_eq!(original.state(), source_state_before);
        assert_eq!(source_signatures.model_digest, snapshot.model_digest);
        assert_eq!(
            candidate_signatures.model_digest,
            candidate.candidate_native_model_digest
        );
        assert_eq!(source_signatures.probe_digest, candidate_signatures.probe_digest);
        let alignment = align_cfc_hidden_unit_signatures(
            &source_signatures,
            &candidate_signatures,
        )
        .expect("activation-based hidden-unit alignment");
        assert_eq!(alignment.source_to_candidate, vec![1, 3, 0, 2]);
        assert_eq!(alignment.source_model_digest, snapshot.model_digest);
        assert_eq!(
            alignment.candidate_model_digest,
            candidate.candidate_native_model_digest
        );
        assert_eq!(alignment.probe_digest, source_signatures.probe_digest);
        assert_eq!(alignment.layer_index, 0);
        assert!(alignment.mean_squared_activation_error < 1e-10);
        assert!(alignment.minimum_assignment_margin.is_some_and(|margin| margin > 0.0));

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

        let mut original_negative_control = original.clone();
        let mut inconsistent_negative_control = inconsistent.network;
        let mut negative_control_squared_error = 0.0_f64;
        for input in &inputs {
            let expected = original_negative_control.forward(input, 0.1);
            let actual = inconsistent_negative_control.forward(input, 0.1);
            for (a, b) in expected.iter().zip(actual.iter()) {
                negative_control_squared_error += f64::from((*a - *b).powi(2));
            }
        }
        assert!(
            negative_control_squared_error.sqrt() > 1e-6,
            "partial permutation must change the function, error={}",
            negative_control_squared_error.sqrt()
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
