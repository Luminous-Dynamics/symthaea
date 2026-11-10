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
    CfCWeightGroupId, ParameterGroupConstraints, ParameterGroupRole, ParameterGroupSnapshot,
    ValidatedWeightCandidate, WeightModelSnapshot, WeightProgram, WeightProgramError,
    WeightProgramReceipt, WeightProgramTarget,
};
use symthaea_fep::gaussian_model_information_gain::{
    GaussianModelInformationGainConfig, GaussianModelInformationGainError,
    GaussianObservationAction, GaussianObservationHypothesis,
    evaluate_gaussian_model_information_gains,
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
    candidate_network
        .try_apply_parameter_group_replacements(&snapshot.model_digest, &replacements)?;
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
    /// Architecture digest: config, topology, layer semantics, and dimensions.
    pub architecture_digest: String,
    /// Parameter manifest digest: semantic groups, shapes, and coordinate order.
    pub manifest_digest: String,
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

    let fingerprint = network
        .fingerprint()
        .map_err(CfCPermutationAlignmentError::FingerprintFailed)?;
    let model_digest = fingerprint.model_digest;
    let architecture_digest = fingerprint.architecture_digest;
    let manifest_digest = fingerprint.manifest_digest;
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
        let layer_state =
            states
                .get(layer_index)
                .ok_or(CfCPermutationAlignmentError::InvalidLayerIndex {
                    layer_index,
                    layer_count: states.len(),
                })?;
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
        architecture_digest,
        manifest_digest,
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
    /// Shared architecture digest required for valid permutation correspondence.
    pub architecture_digest: String,
    /// Shared parameter-manifest digest required for coordinate correspondence.
    pub manifest_digest: String,
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

/// Measured score for one candidate diagnostic probe appended to a frozen probe set.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CfCProbeCandidateScore {
    pub candidate_index: usize,
    /// Digest of the complete augmented probe set, not only the newly added probe.
    pub augmented_probe_digest: String,
    pub mean_squared_activation_error: f64,
    /// Accumulated assignment evidence margin. None means the layer has no alternative assignment.
    pub assignment_margin: Option<f64>,
}

/// Deterministic probe-selection receipt for reducing hidden-unit correspondence ambiguity.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CfCDisambiguatingProbeSelection {
    pub format_version: u32,
    pub source_model_digest: String,
    pub candidate_model_digest: String,
    pub architecture_digest: String,
    pub manifest_digest: String,
    pub selected_candidate_index: usize,
    pub selected_probe: CfCFunctionalProbe,
    pub selected_augmented_probe_digest: String,
    pub baseline_assignment_margin: Option<f64>,
    pub baseline_mean_squared_activation_error: Option<f64>,
    pub selected_assignment_margin: f64,
    pub selected_mean_squared_activation_error: f64,
    /// All evaluated probes in original candidate order.
    pub candidate_scores: Vec<CfCProbeCandidateScore>,
}

/// Acceptance thresholds required before a caller uses an inferred unit mapping
/// to transform parameter groups. These are task-calibrated tolerances, not
/// universal constants.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CfCAlignmentAcceptancePolicy {
    /// Maximum allowed mean squared activation-signature error.
    pub max_mean_squared_activation_error: f64,
    /// Minimum allowed gap to the best alternative complete assignment.
    /// A value of zero only rejects malformed/non-finite ambiguity metadata.
    pub min_assignment_margin: f64,
}

/// Posterior mass over one exact source-to-candidate hidden-unit permutation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CfCPermutationHypothesisPosterior {
    /// Mapping from source unit index to candidate unit index.
    pub source_to_candidate: Vec<usize>,
    /// Posterior probability after the existing functional probes.
    pub posterior_probability: f64,
}

/// Prospective FEP score for one candidate functional probe.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CfCGaussianPermutationProbeScore {
    pub candidate_index: usize,
    pub augmented_probe_digest: String,
    pub model_information_gain_nats: f64,
    pub expected_posterior_permutation_entropy_nats: f64,
}

/// Provenance class for a scalar activation-projection noise calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CfCProjectionNoiseCalibrationMethod {
    RepeatedProjectionRuns,
    DevicePrecisionStudy,
    StochasticActivationModel,
    IndependentBootstrap,
}

impl CfCProjectionNoiseCalibrationMethod {
    fn as_str(self) -> &'static str {
        match self {
            Self::RepeatedProjectionRuns => "repeated_projection_runs",
            Self::DevicePrecisionStudy => "device_precision_study",
            Self::StochasticActivationModel => "stochastic_activation_model",
            Self::IndependentBootstrap => "independent_bootstrap",
        }
    }
}

/// Empirical noise estimate bound to the exact CfC pair and scalar projection.
///
/// Samples are repeated measurements of the same scalar projection under a
/// frozen probe corpus. The unbiased sample standard deviation is used as the
/// Gaussian observation scale. A zero-variance sample is rejected: callers
/// must use a deterministic margin objective or establish a justified noise
/// model rather than inject a convenient non-zero value.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CfCProjectionNoiseCalibration {
    pub format_version: u32,
    pub source_model_digest: String,
    pub candidate_model_digest: String,
    pub architecture_digest: String,
    pub manifest_digest: String,
    pub layer_index: usize,
    pub projection_digest: String,
    pub calibration_probe_digest: String,
    pub method: CfCProjectionNoiseCalibrationMethod,
    pub sample_count: usize,
    /// Ordered repeated scalar-projection samples used to reproduce statistics and receipt digest.
    pub repeated_projection_samples: Vec<f64>,
    pub sample_mean: f64,
    pub observation_std_dev: f64,
    pub calibration_digest: String,
}

/// Compute a source/candidate-bound projection-noise receipt from repeated
/// measurements of the deterministic projection on a frozen calibration set.
///
/// The caller is responsible for collecting repeated measurements under the
/// declared method and producing the calibration probe-corpus digest. The
/// constructor calculates the statistics and binds the evidence to the exact
/// model pair, layer, and projection so it cannot be reused after an edit.
pub fn calibrate_cfc_projection_observation_noise(
    source: &CfCNetwork,
    candidate: &CfCNetwork,
    layer_index: usize,
    calibration_probe_digest: &str,
    method: CfCProjectionNoiseCalibrationMethod,
    repeated_projection_samples: &[f64],
) -> Result<CfCProjectionNoiseCalibration, CfCPermutationAlignmentError> {
    if repeated_projection_samples.len() < 2 {
        return Err(
            CfCPermutationAlignmentError::InsufficientNoiseCalibrationSamples {
                actual: repeated_projection_samples.len(),
                minimum: 2,
            },
        );
    }
    if repeated_projection_samples.len() > MAX_NOISE_CALIBRATION_SAMPLES {
        return Err(
            CfCPermutationAlignmentError::TooManyNoiseCalibrationSamples {
                actual: repeated_projection_samples.len(),
                maximum: MAX_NOISE_CALIBRATION_SAMPLES,
            },
        );
    }
    if !is_lowercase_hex_digest(calibration_probe_digest) {
        return Err(
            CfCPermutationAlignmentError::InvalidNoiseCalibrationDigest {
                field: "calibration_probe_digest",
            },
        );
    }
    if let Some(index) = repeated_projection_samples
        .iter()
        .position(|sample| !sample.is_finite())
    {
        return Err(CfCPermutationAlignmentError::NonFiniteNoiseCalibrationSample { index });
    }

    let source_fingerprint = source
        .fingerprint()
        .map_err(CfCPermutationAlignmentError::FingerprintFailed)?;
    let candidate_fingerprint = candidate
        .fingerprint()
        .map_err(CfCPermutationAlignmentError::FingerprintFailed)?;
    if source_fingerprint.architecture_digest != candidate_fingerprint.architecture_digest {
        return Err(CfCPermutationAlignmentError::ArchitectureMismatch {
            source: source_fingerprint.architecture_digest,
            candidate: candidate_fingerprint.architecture_digest,
        });
    }
    if source_fingerprint.manifest_digest != candidate_fingerprint.manifest_digest {
        return Err(CfCPermutationAlignmentError::ManifestMismatch {
            source: source_fingerprint.manifest_digest,
            candidate: candidate_fingerprint.manifest_digest,
        });
    }
    if layer_index >= source.config().num_layers {
        return Err(CfCPermutationAlignmentError::InvalidLayerIndex {
            layer_index,
            layer_count: source.config().num_layers,
        });
    }

    let sample_mean =
        repeated_projection_samples.iter().sum::<f64>() / repeated_projection_samples.len() as f64;
    let sample_variance = repeated_projection_samples
        .iter()
        .map(|sample| (sample - sample_mean).powi(2))
        .sum::<f64>()
        / (repeated_projection_samples.len() - 1) as f64;
    let observation_std_dev = sample_variance.sqrt();
    if !observation_std_dev.is_finite() || observation_std_dev <= 0.0 {
        return Err(CfCPermutationAlignmentError::DegenerateNoiseCalibration);
    }

    let projection = deterministic_permutation_projection(
        &source_fingerprint.architecture_digest,
        layer_index,
        source.config().hidden_dim,
    );
    let projection_digest = digest_projection(&projection);
    let calibration_digest = digest_cfc_projection_noise_calibration(
        &source_fingerprint.model_digest,
        &candidate_fingerprint.model_digest,
        &source_fingerprint.architecture_digest,
        &source_fingerprint.manifest_digest,
        layer_index,
        &projection_digest,
        calibration_probe_digest,
        method,
        repeated_projection_samples,
        sample_mean,
        observation_std_dev,
    );

    Ok(CfCProjectionNoiseCalibration {
        format_version: 1,
        source_model_digest: source_fingerprint.model_digest,
        candidate_model_digest: candidate_fingerprint.model_digest,
        architecture_digest: source_fingerprint.architecture_digest,
        manifest_digest: source_fingerprint.manifest_digest,
        layer_index,
        projection_digest,
        calibration_probe_digest: calibration_probe_digest.to_owned(),
        method,
        sample_count: repeated_projection_samples.len(),
        repeated_projection_samples: repeated_projection_samples.to_vec(),
        sample_mean,
        observation_std_dev,
        calibration_digest,
    })
}

fn digest_cfc_projection_noise_calibration(
    source_model_digest: &str,
    candidate_model_digest: &str,
    architecture_digest: &str,
    manifest_digest: &str,
    layer_index: usize,
    projection_digest: &str,
    calibration_probe_digest: &str,
    method: CfCProjectionNoiseCalibrationMethod,
    samples: &[f64],
    sample_mean: f64,
    observation_std_dev: f64,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-cfc-projection-noise-calibration-v1\0");
    for value in [
        source_model_digest,
        candidate_model_digest,
        architecture_digest,
        manifest_digest,
        projection_digest,
        calibration_probe_digest,
        method.as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update(&(layer_index as u64).to_le_bytes());
    hasher.update(&(samples.len() as u64).to_le_bytes());
    hasher.update(&sample_mean.to_le_bytes());
    hasher.update(&observation_std_dev.to_le_bytes());
    for sample in samples {
        hasher.update(&sample.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

/// Revalidate calibration statistics, digest, and binding to the current model pair.
fn validate_cfc_projection_noise_calibration(
    calibration: &CfCProjectionNoiseCalibration,
    source_model_digest: &str,
    candidate_model_digest: &str,
    architecture_digest: &str,
    manifest_digest: &str,
    layer_index: usize,
    projection_digest: &str,
) -> Result<(), CfCPermutationAlignmentError> {
    if calibration.format_version != 1 {
        return Err(CfCPermutationAlignmentError::MalformedNoiseCalibration {
            field: "format_version",
        });
    }
    for (field, value) in [
        (
            "source_model_digest",
            calibration.source_model_digest.as_str(),
        ),
        (
            "candidate_model_digest",
            calibration.candidate_model_digest.as_str(),
        ),
        (
            "architecture_digest",
            calibration.architecture_digest.as_str(),
        ),
        ("manifest_digest", calibration.manifest_digest.as_str()),
        ("projection_digest", calibration.projection_digest.as_str()),
        (
            "calibration_probe_digest",
            calibration.calibration_probe_digest.as_str(),
        ),
        (
            "calibration_digest",
            calibration.calibration_digest.as_str(),
        ),
    ] {
        if !is_lowercase_hex_digest(value) {
            return Err(CfCPermutationAlignmentError::InvalidNoiseCalibrationDigest { field });
        }
    }
    for (field, expected, actual) in [
        (
            "source model digest",
            source_model_digest,
            calibration.source_model_digest.as_str(),
        ),
        (
            "candidate model digest",
            candidate_model_digest,
            calibration.candidate_model_digest.as_str(),
        ),
        (
            "architecture digest",
            architecture_digest,
            calibration.architecture_digest.as_str(),
        ),
        (
            "manifest digest",
            manifest_digest,
            calibration.manifest_digest.as_str(),
        ),
        (
            "projection digest",
            projection_digest,
            calibration.projection_digest.as_str(),
        ),
    ] {
        if expected != actual {
            return Err(CfCPermutationAlignmentError::StaleNoiseCalibration { field });
        }
    }
    if calibration.layer_index != layer_index {
        return Err(CfCPermutationAlignmentError::StaleNoiseCalibration {
            field: "layer index",
        });
    }
    if calibration.sample_count != calibration.repeated_projection_samples.len()
        || calibration.sample_count < 2
        || calibration.sample_count > MAX_NOISE_CALIBRATION_SAMPLES
    {
        return Err(CfCPermutationAlignmentError::MalformedNoiseCalibration {
            field: "sample_count",
        });
    }
    if let Some(index) = calibration
        .repeated_projection_samples
        .iter()
        .position(|sample| !sample.is_finite())
    {
        return Err(CfCPermutationAlignmentError::NonFiniteNoiseCalibrationSample { index });
    }

    let recomputed_mean = calibration.repeated_projection_samples.iter().sum::<f64>()
        / calibration.sample_count as f64;
    let recomputed_variance = calibration
        .repeated_projection_samples
        .iter()
        .map(|sample| (sample - recomputed_mean).powi(2))
        .sum::<f64>()
        / (calibration.sample_count - 1) as f64;
    let recomputed_std_dev = recomputed_variance.sqrt();
    let mean_tolerance = 1e-12 * recomputed_mean.abs().max(1.0);
    let std_tolerance = 1e-12 * recomputed_std_dev.abs().max(1.0);
    if !calibration.sample_mean.is_finite()
        || !calibration.observation_std_dev.is_finite()
        || calibration.observation_std_dev <= 0.0
        || (calibration.sample_mean - recomputed_mean).abs() > mean_tolerance
        || (calibration.observation_std_dev - recomputed_std_dev).abs() > std_tolerance
    {
        return Err(CfCPermutationAlignmentError::MalformedNoiseCalibration {
            field: "sample statistics",
        });
    }

    let expected_digest = digest_cfc_projection_noise_calibration(
        &calibration.source_model_digest,
        &calibration.candidate_model_digest,
        &calibration.architecture_digest,
        &calibration.manifest_digest,
        calibration.layer_index,
        &calibration.projection_digest,
        &calibration.calibration_probe_digest,
        calibration.method,
        &calibration.repeated_projection_samples,
        calibration.sample_mean,
        calibration.observation_std_dev,
    );
    if expected_digest != calibration.calibration_digest {
        return Err(CfCPermutationAlignmentError::MalformedNoiseCalibration {
            field: "calibration_digest",
        });
    }
    Ok(())
}

/// Source-bound receipt for Gaussian-channel EIG over an exact small permutation space.
///
/// The likelihood is a scalar Gaussian observation: a deterministic projection of
/// candidate hidden activations with an explicit observation standard deviation.
/// The standard deviation is an experimental assumption that must be calibrated;
/// this receipt does not infer or justify it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CfCGaussianPermutationEIGSelection {
    pub format_version: u32,
    pub source_model_digest: String,
    pub candidate_model_digest: String,
    pub architecture_digest: String,
    pub manifest_digest: String,
    pub layer_index: usize,
    pub projection_digest: String,
    pub existing_probe_digest: Option<String>,
    pub selected_candidate_index: usize,
    pub selected_probe: CfCFunctionalProbe,
    pub selected_augmented_probe_digest: String,
    pub noise_calibration_digest: String,
    pub noise_calibration_method: CfCProjectionNoiseCalibrationMethod,
    pub noise_calibration_sample_count: usize,
    pub observation_std_dev: f64,
    pub quadrature_points: usize,
    pub quadrature_tail_sigma: f64,
    pub permutation_hypothesis_count: usize,
    pub current_permutation_entropy_nats: f64,
    pub selected_expected_posterior_entropy_nats: f64,
    pub selected_model_information_gain_nats: f64,
    /// Posterior after existing probes, before the selected new probe is observed.
    pub permutation_posterior: Vec<CfCPermutationHypothesisPosterior>,
    /// Every candidate probe in input order, for reproducible action comparison.
    pub candidate_scores: Vec<CfCGaussianPermutationProbeScore>,
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
    #[error(
        "{side} activation signature {unit_index} contains a non-finite value at probe {probe_index}"
    )]
    NonFiniteSignature {
        side: &'static str,
        unit_index: usize,
        probe_index: usize,
    },
    #[error("no finite complete hidden-unit assignment exists")]
    NoFeasibleAssignment,
    #[error("source and candidate architecture digests differ: {source} vs {candidate}")]
    ArchitectureMismatch { source: String, candidate: String },
    #[error("source and candidate parameter-manifest digests differ: {source} vs {candidate}")]
    ManifestMismatch { source: String, candidate: String },
    #[error("source and candidate probe digests differ: {source} vs {candidate}")]
    ProbeSetMismatch { source: String, candidate: String },
    #[error("source and candidate layer indices differ: {source} vs {candidate}")]
    LayerMismatch { source: usize, candidate: usize },
    #[error("CfC fingerprint failed while binding activation signatures: {0}")]
    FingerprintFailed(String),
    #[error("at least one functional probe is required")]
    EmptyProbeSet,
    #[error("at least one candidate disambiguating probe is required")]
    EmptyCandidateProbeSet,
    #[error("a one-unit hidden layer has no alternative permutation to disambiguate")]
    NoAlternativeAssignment,
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
    #[error(
        "layer {layer_index}, unit {unit_index}, probe {probe_index} produced a non-finite activation"
    )]
    NonFiniteActivation {
        layer_index: usize,
        unit_index: usize,
        probe_index: usize,
    },
    #[error("alignment acceptance policy requires finite non-negative thresholds")]
    InvalidAcceptancePolicy,
    #[error("alignment receipt is malformed: {field}")]
    MalformedAlignmentReceipt { field: &'static str },
    #[error("alignment error {measured} exceeds maximum {maximum}")]
    ActivationErrorAboveThreshold { measured: f64, maximum: f64 },
    #[error("alignment assignment margin {measured:?} is below required minimum {minimum}")]
    AssignmentAmbiguous { measured: Option<f64>, minimum: f64 },
    #[error("alignment receipt does not match the current {field} fingerprint")]
    StaleAlignment { field: &'static str },
    #[error(
        "permutation canonicalization supports only non-residual, non-backbone, unidirectional CfC stacks"
    )]
    UnsupportedCanonicalizationTopology,
    #[error("required CfC parameter group is absent from the candidate snapshot: {0}")]
    MissingCanonicalizationGroup(String),
    #[error("CfC parameter group has an unexpected shape for canonicalization: {group}")]
    InvalidCanonicalizationGroupShape { group: String },
    #[error(
        "exact permutation EIG supports at most {maximum_hypotheses} mappings; layer has {hidden_dim} units"
    )]
    ExactPermutationSpaceTooLarge {
        hidden_dim: usize,
        maximum_hypotheses: usize,
    },
    #[error(transparent)]
    GaussianInformationGain(#[from] GaussianModelInformationGainError),
    #[error("permutation activation signatures have an invalid hidden-unit dimension")]
    InvalidPermutationSignatureDimension,
    #[error(
        "Gaussian permutation EIG accepts at most {maximum_probes} candidate probes, received {actual_probes}"
    )]
    TooManyGaussianEIGCandidateProbes {
        actual_probes: usize,
        maximum_probes: usize,
    },
    #[error("permutation posterior normalization failed")]
    InvalidPermutationPosterior,
    #[error(
        "noise calibration has {actual} repeated projection samples; at least {minimum} are required"
    )]
    InsufficientNoiseCalibrationSamples { actual: usize, minimum: usize },
    #[error("noise calibration has {actual} samples; maximum supported is {maximum}")]
    TooManyNoiseCalibrationSamples { actual: usize, maximum: usize },
    #[error("noise calibration sample {index} is non-finite")]
    NonFiniteNoiseCalibrationSample { index: usize },
    #[error("noise calibration has zero or invalid sample variance")]
    DegenerateNoiseCalibration,
    #[error("noise calibration receipt has an invalid {field} digest")]
    InvalidNoiseCalibrationDigest { field: &'static str },
    #[error("noise calibration does not match the current {field}")]
    StaleNoiseCalibration { field: &'static str },
    #[error("noise calibration receipt is malformed: {field}")]
    MalformedNoiseCalibration { field: &'static str },
    #[error("canonicalization would not change any parameters")]
    NoOpCanonicalization,
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
        if !is_lowercase_hex_digest(&signature_set.architecture_digest) {
            return Err(CfCPermutationAlignmentError::InvalidSignatureDigest {
                side,
                field: "architecture",
            });
        }
        if !is_lowercase_hex_digest(&signature_set.manifest_digest) {
            return Err(CfCPermutationAlignmentError::InvalidSignatureDigest {
                side,
                field: "manifest",
            });
        }
        if !is_lowercase_hex_digest(&signature_set.probe_digest) {
            return Err(CfCPermutationAlignmentError::InvalidSignatureDigest {
                side,
                field: "probe",
            });
        }
    }
    if source_signature_set.architecture_digest != candidate_signature_set.architecture_digest {
        return Err(CfCPermutationAlignmentError::ArchitectureMismatch {
            source: source_signature_set.architecture_digest.clone(),
            candidate: candidate_signature_set.architecture_digest.clone(),
        });
    }
    if source_signature_set.manifest_digest != candidate_signature_set.manifest_digest {
        return Err(CfCPermutationAlignmentError::ManifestMismatch {
            source: source_signature_set.manifest_digest.clone(),
            candidate: candidate_signature_set.manifest_digest.clone(),
        });
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
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let (source_to_candidate, best_total_cost) = minimum_cost_assignment(&costs)
        .ok_or(CfCPermutationAlignmentError::NoFeasibleAssignment)?;

    let minimum_assignment_margin =
        alternating_cycle_assignment_margin(&costs, &source_to_candidate);

    Ok(CfCHiddenUnitAlignment {
        format_version: 1,
        source_model_digest: source_signature_set.model_digest.clone(),
        candidate_model_digest: candidate_signature_set.model_digest.clone(),
        architecture_digest: source_signature_set.architecture_digest.clone(),
        manifest_digest: source_signature_set.manifest_digest.clone(),
        probe_digest: source_signature_set.probe_digest.clone(),
        layer_index: source_signature_set.layer_index,
        source_to_candidate,
        mean_squared_activation_error: best_total_cost / (unit_count * probe_count) as f64,
        minimum_assignment_margin,
    })
}

/// Build a candidate-only WeightProgram that maps the candidate's hidden-unit
/// coordinates into the source network's coordinate order.
///
/// The alignment maps each source unit to its candidate index. For the supported
/// one-layer topology, the exact function-preserving reparameterization applies
/// the same row permutation to input weights, both axes of recurrent weights,
/// hidden bias and time constants, and the matching column permutation to the
/// output projection. Output bias is unchanged.
///
/// This is coordinate canonicalization, not weight transfer: it permutes the
/// candidate's own tensors and therefore preserves the candidate's function.
/// The returned program is still only a proposal; callers must materialize it
/// through the candidate-only adapter, evaluate independent held-out sequences,
/// and keep the source network unchanged.
pub fn build_cfc_hidden_unit_canonicalization_program(
    source: &CfCNetwork,
    candidate: &CfCNetwork,
    target_specification_digest: impl Into<String>,
    alignment: &CfCHiddenUnitAlignment,
    policy: CfCAlignmentAcceptancePolicy,
) -> Result<WeightProgram, CfCPermutationAlignmentError> {
    validate_cfc_alignment_for_parameter_transfer(alignment, policy)?;

    let source_fingerprint = source
        .fingerprint()
        .map_err(CfCPermutationAlignmentError::FingerprintFailed)?;
    let candidate_fingerprint = candidate
        .fingerprint()
        .map_err(CfCPermutationAlignmentError::FingerprintFailed)?;

    if alignment.source_model_digest != source_fingerprint.model_digest {
        return Err(CfCPermutationAlignmentError::StaleAlignment {
            field: "source model",
        });
    }
    if alignment.candidate_model_digest != candidate_fingerprint.model_digest {
        return Err(CfCPermutationAlignmentError::StaleAlignment {
            field: "candidate model",
        });
    }
    if alignment.architecture_digest != candidate_fingerprint.architecture_digest
        || source_fingerprint.architecture_digest != candidate_fingerprint.architecture_digest
    {
        return Err(CfCPermutationAlignmentError::ArchitectureMismatch {
            source: alignment.architecture_digest.clone(),
            candidate: candidate_fingerprint.architecture_digest,
        });
    }
    if alignment.manifest_digest != candidate_fingerprint.manifest_digest
        || source_fingerprint.manifest_digest != candidate_fingerprint.manifest_digest
    {
        return Err(CfCPermutationAlignmentError::ManifestMismatch {
            source: alignment.manifest_digest.clone(),
            candidate: candidate_fingerprint.manifest_digest,
        });
    }

    let config = candidate.config();
    if config.num_layers != 1
        || config.residual
        || config.bidirectional
        || config.cell_config.use_backbone
        || alignment.layer_index != 0
        || alignment.source_to_candidate.len() != config.hidden_dim
    {
        return Err(CfCPermutationAlignmentError::UnsupportedCanonicalizationTopology);
    }

    let task_digest = target_specification_digest.into();
    let snapshot =
        cfc_weight_model_snapshot(candidate, task_digest.clone()).map_err(|error| match error {
            CfCWeightProgramAdapterError::Fingerprint(message) => {
                CfCPermutationAlignmentError::FingerprintFailed(message)
            }
            _ => CfCPermutationAlignmentError::MissingCanonicalizationGroup(
                "candidate snapshot".to_owned(),
            ),
        })?;
    let hidden_dim = config.hidden_dim;
    let mut operations = Vec::with_capacity(5);
    let mut changed_parameter_count = 0_usize;
    let mut squared_delta = 0.0_f64;
    let mut max_abs_delta = 0.0_f32;

    let mut add_replacement = |group_id: CfCWeightGroupId,
                               expected_shape: &[usize],
                               replacement: Vec<f32>|
     -> Result<(), CfCPermutationAlignmentError> {
        let target = WeightProgramTarget::CfC(group_id.clone());
        let group = snapshot
            .groups
            .iter()
            .find(|group| group.target == target)
            .ok_or_else(|| {
                CfCPermutationAlignmentError::MissingCanonicalizationGroup(format!("{group_id:?}"))
            })?;
        if group.shape != expected_shape || group.values.len() != replacement.len() {
            return Err(
                CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
                    group: format!("{group_id:?}"),
                },
            );
        }
        for (before, after) in group
            .values
            .iter()
            .copied()
            .zip(replacement.iter().copied())
        {
            let delta = after - before;
            if delta != 0.0 {
                changed_parameter_count += 1;
                squared_delta += f64::from(delta).powi(2);
                max_abs_delta = max_abs_delta.max(delta.abs());
            }
        }
        operations.push(WeightProgramOperation::ReplaceGroup {
            target,
            values: replacement,
        });
        Ok(())
    };

    let input_group = CfCWeightGroupId::CellInputProjection { cell: 0 };
    let input_snapshot = snapshot
        .groups
        .iter()
        .find(|group| group.target == WeightProgramTarget::CfC(input_group.clone()))
        .ok_or_else(|| {
            CfCPermutationAlignmentError::MissingCanonicalizationGroup(format!("{input_group:?}"))
        })?;
    let input_shape = input_snapshot.shape.clone();
    if input_shape.len() != 2 || input_shape[0] != hidden_dim {
        return Err(
            CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
                group: format!("{input_group:?}"),
            },
        );
    }
    let permuted_input = permute_matrix_rows_checked(
        &input_snapshot.values,
        input_shape[0],
        input_shape[1],
        &alignment.source_to_candidate,
    )
    .ok_or_else(
        || CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
            group: format!("{input_group:?}"),
        },
    )?;
    add_replacement(input_group, &input_shape, permuted_input)?;

    let recurrent_group = CfCWeightGroupId::CellRecurrentProjection { cell: 0 };
    let recurrent_snapshot = snapshot
        .groups
        .iter()
        .find(|group| group.target == WeightProgramTarget::CfC(recurrent_group.clone()))
        .ok_or_else(|| {
            CfCPermutationAlignmentError::MissingCanonicalizationGroup(format!(
                "{recurrent_group:?}"
            ))
        })?;
    let recurrent_shape = recurrent_snapshot.shape.clone();
    if recurrent_shape != [hidden_dim, hidden_dim] {
        return Err(
            CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
                group: format!("{recurrent_group:?}"),
            },
        );
    }
    let permuted_recurrent = permute_square_checked(
        &recurrent_snapshot.values,
        hidden_dim,
        &alignment.source_to_candidate,
    )
    .ok_or_else(
        || CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
            group: format!("{recurrent_group:?}"),
        },
    )?;
    add_replacement(recurrent_group, &recurrent_shape, permuted_recurrent)?;

    for group_id in [
        CfCWeightGroupId::CellHiddenBias { cell: 0 },
        CfCWeightGroupId::CellTimeConstant { cell: 0 },
    ] {
        let group = snapshot
            .groups
            .iter()
            .find(|group| group.target == WeightProgramTarget::CfC(group_id.clone()))
            .ok_or_else(|| {
                CfCPermutationAlignmentError::MissingCanonicalizationGroup(format!("{group_id:?}"))
            })?;
        if group.shape != [hidden_dim] || group.values.len() != hidden_dim {
            return Err(
                CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
                    group: format!("{group_id:?}"),
                },
            );
        }
        let replacement = alignment
            .source_to_candidate
            .iter()
            .map(|index| group.values[*index])
            .collect::<Vec<_>>();
        add_replacement(group_id, &[hidden_dim], replacement)?;
    }

    let output_group = CfCWeightGroupId::OutputProjection;
    let output_snapshot = snapshot
        .groups
        .iter()
        .find(|group| group.target == WeightProgramTarget::CfC(output_group.clone()))
        .ok_or_else(|| {
            CfCPermutationAlignmentError::MissingCanonicalizationGroup(format!("{output_group:?}"))
        })?;
    let output_shape = output_snapshot.shape.clone();
    if output_shape != [config.output_dim, hidden_dim] {
        return Err(
            CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
                group: format!("{output_group:?}"),
            },
        );
    }
    let permuted_output = permute_matrix_columns_checked(
        &output_snapshot.values,
        config.output_dim,
        hidden_dim,
        &alignment.source_to_candidate,
    )
    .ok_or_else(
        || CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
            group: format!("{output_group:?}"),
        },
    )?;
    add_replacement(output_group, &output_shape, permuted_output)?;
    drop(add_replacement);

    if changed_parameter_count == 0 {
        return Err(CfCPermutationAlignmentError::NoOpCanonicalization);
    }

    let total_l2_delta = squared_delta.sqrt() as f32;
    let budget_epsilon = 1.0e-6_f32;
    let max_total_l2_delta = total_l2_delta * (1.0 + 1.0e-6) + budget_epsilon;
    let max_abs_delta = max_abs_delta * (1.0 + 1.0e-6) + budget_epsilon;
    let mut evidence_hasher = blake3::Hasher::new();
    evidence_hasher.update(b"symthaea-cfc-hidden-unit-alignment-receipt-v1\0");
    evidence_hasher.update(alignment.source_model_digest.as_bytes());
    evidence_hasher.update(alignment.candidate_model_digest.as_bytes());
    evidence_hasher.update(alignment.architecture_digest.as_bytes());
    evidence_hasher.update(alignment.manifest_digest.as_bytes());
    evidence_hasher.update(alignment.probe_digest.as_bytes());
    evidence_hasher.update(&(alignment.layer_index as u64).to_le_bytes());
    for index in alignment.source_to_candidate.iter().copied() {
        evidence_hasher.update(&(index as u64).to_le_bytes());
    }
    evidence_hasher.update(&alignment.mean_squared_activation_error.to_le_bytes());
    if let Some(margin) = alignment.minimum_assignment_margin {
        evidence_hasher.update(&margin.to_le_bytes());
    }

    Ok(WeightProgram {
        format_version: 1,
        architecture_digest: snapshot.architecture_digest,
        manifest_digest: snapshot.manifest_digest,
        base_model_digest: snapshot.model_digest,
        target_specification_digest: task_digest,
        fault_hypothesis_ids: vec!["symmetry:hidden-unit-permutation-canonicalization".to_owned()],
        evidence_receipt_ids: vec![format!("alignment:{}", evidence_hasher.finalize().to_hex())],
        max_changed_parameters: changed_parameter_count,
        max_total_l2_delta,
        max_abs_delta,
        operations,
    })
}

/// Build a candidate-only WeightProgram that canonicalizes every hidden layer
/// of a feed-forward CfC stack into the source network's coordinate system.
///
/// For layer i, the candidate-to-source coordinate map is applied to:
/// - rows of that layer's input projection;
/// - columns of that layer's input projection using the previous layer's map;
/// - both axes of that layer's recurrent projection;
/// - that layer's hidden bias and time constants;
/// - columns of the final output projection using the final layer's map.
///
/// Residual, bidirectional, and backbone topologies are rejected because their
/// extra paths require additional symmetry actions that are not represented by
/// this v1 parameter-group transform. No network is mutated by this function.
pub fn build_cfc_network_hidden_unit_canonicalization_program(
    source: &CfCNetwork,
    candidate: &CfCNetwork,
    target_specification_digest: impl Into<String>,
    layer_alignments: &[CfCHiddenUnitAlignment],
    policy: CfCAlignmentAcceptancePolicy,
) -> Result<WeightProgram, CfCPermutationAlignmentError> {
    let source_fingerprint = source
        .fingerprint()
        .map_err(CfCPermutationAlignmentError::FingerprintFailed)?;
    let candidate_fingerprint = candidate
        .fingerprint()
        .map_err(CfCPermutationAlignmentError::FingerprintFailed)?;

    if source_fingerprint.architecture_digest != candidate_fingerprint.architecture_digest {
        return Err(CfCPermutationAlignmentError::ArchitectureMismatch {
            source: source_fingerprint.architecture_digest,
            candidate: candidate_fingerprint.architecture_digest,
        });
    }
    if source_fingerprint.manifest_digest != candidate_fingerprint.manifest_digest {
        return Err(CfCPermutationAlignmentError::ManifestMismatch {
            source: source_fingerprint.manifest_digest,
            candidate: candidate_fingerprint.manifest_digest,
        });
    }

    let config = candidate.config();
    if config.num_layers == 0
        || config.residual
        || config.bidirectional
        || config.cell_config.use_backbone
        || layer_alignments.len() != config.num_layers
    {
        return Err(CfCPermutationAlignmentError::UnsupportedCanonicalizationTopology);
    }

    let mut alignment_by_layer = vec![None; config.num_layers];
    for alignment in layer_alignments {
        validate_cfc_alignment_for_parameter_transfer(alignment, policy.clone())?;
        if alignment.source_model_digest != source_fingerprint.model_digest {
            return Err(CfCPermutationAlignmentError::StaleAlignment {
                field: "source model",
            });
        }
        if alignment.candidate_model_digest != candidate_fingerprint.model_digest {
            return Err(CfCPermutationAlignmentError::StaleAlignment {
                field: "candidate model",
            });
        }
        if alignment.architecture_digest != candidate_fingerprint.architecture_digest {
            return Err(CfCPermutationAlignmentError::ArchitectureMismatch {
                source: alignment.architecture_digest.clone(),
                candidate: candidate_fingerprint.architecture_digest.clone(),
            });
        }
        if alignment.manifest_digest != candidate_fingerprint.manifest_digest {
            return Err(CfCPermutationAlignmentError::ManifestMismatch {
                source: alignment.manifest_digest.clone(),
                candidate: candidate_fingerprint.manifest_digest.clone(),
            });
        }
        let layer_index = alignment.layer_index;
        if layer_index >= config.num_layers || alignment_by_layer[layer_index].is_some() {
            return Err(CfCPermutationAlignmentError::MalformedAlignmentReceipt {
                field: "layer_alignments",
            });
        }
        if alignment.source_to_candidate.len() != config.hidden_dim {
            return Err(CfCPermutationAlignmentError::HiddenUnitCountMismatch {
                source: config.hidden_dim,
                candidate: alignment.source_to_candidate.len(),
            });
        }
        alignment_by_layer[layer_index] = Some(alignment);
    }

    if alignment_by_layer.iter().any(Option::is_none) {
        return Err(CfCPermutationAlignmentError::MalformedAlignmentReceipt {
            field: "missing layer alignment",
        });
    }
    let alignments = alignment_by_layer
        .into_iter()
        .map(|alignment| alignment.expect("all layer alignments checked"))
        .collect::<Vec<_>>();
    let permutations = alignments
        .iter()
        .map(|alignment| alignment.source_to_candidate.clone())
        .collect::<Vec<_>>();

    let task_digest = target_specification_digest.into();
    let snapshot =
        cfc_weight_model_snapshot(candidate, task_digest.clone()).map_err(|error| match error {
            CfCWeightProgramAdapterError::Fingerprint(message) => {
                CfCPermutationAlignmentError::FingerprintFailed(message)
            }
            _ => CfCPermutationAlignmentError::MissingCanonicalizationGroup(
                "candidate snapshot".to_owned(),
            ),
        })?;

    let hidden_dim = config.hidden_dim;
    let mut operations = Vec::with_capacity(4 * config.num_layers + 1);
    let mut changed_parameter_count = 0_usize;
    let mut squared_delta = 0.0_f64;
    let mut max_abs_delta = 0.0_f32;

    let mut add_replacement = |group_id: CfCWeightGroupId,
                               expected_shape: &[usize],
                               replacement: Vec<f32>|
     -> Result<(), CfCPermutationAlignmentError> {
        let target = WeightProgramTarget::CfC(group_id.clone());
        let group = snapshot
            .groups
            .iter()
            .find(|group| group.target == target)
            .ok_or_else(|| {
                CfCPermutationAlignmentError::MissingCanonicalizationGroup(format!("{group_id:?}"))
            })?;
        if group.shape != expected_shape || group.values.len() != replacement.len() {
            return Err(
                CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
                    group: format!("{group_id:?}"),
                },
            );
        }
        for (before, after) in group
            .values
            .iter()
            .copied()
            .zip(replacement.iter().copied())
        {
            let delta = after - before;
            if delta != 0.0 {
                changed_parameter_count += 1;
                squared_delta += f64::from(delta).powi(2);
                max_abs_delta = max_abs_delta.max(delta.abs());
            }
        }
        operations.push(WeightProgramOperation::ReplaceGroup {
            target,
            values: replacement,
        });
        Ok(())
    };

    for layer_index in 0..config.num_layers {
        let permutation = &permutations[layer_index];
        let input_width = if layer_index == 0 {
            config.input_dim
        } else {
            hidden_dim
        };

        let input_group = CfCWeightGroupId::CellInputProjection { cell: layer_index };
        let input_snapshot = snapshot
            .groups
            .iter()
            .find(|group| group.target == WeightProgramTarget::CfC(input_group.clone()))
            .ok_or_else(|| {
                CfCPermutationAlignmentError::MissingCanonicalizationGroup(format!(
                    "{input_group:?}"
                ))
            })?;
        let input_shape = input_snapshot.shape.clone();
        if input_shape != [hidden_dim, input_width] {
            return Err(
                CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
                    group: format!("{input_group:?}"),
                },
            );
        }
        let mut permuted_input = permute_matrix_rows_checked(
            &input_snapshot.values,
            hidden_dim,
            input_width,
            permutation,
        )
        .ok_or_else(|| {
            CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
                group: format!("{input_group:?}"),
            }
        })?;
        if layer_index > 0 {
            permuted_input = permute_matrix_columns_checked(
                &permuted_input,
                hidden_dim,
                input_width,
                &permutations[layer_index - 1],
            )
            .ok_or_else(|| {
                CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
                    group: format!("{input_group:?}"),
                }
            })?;
        }
        add_replacement(input_group, &input_shape, permuted_input)?;

        let recurrent_group = CfCWeightGroupId::CellRecurrentProjection { cell: layer_index };
        let recurrent_snapshot = snapshot
            .groups
            .iter()
            .find(|group| group.target == WeightProgramTarget::CfC(recurrent_group.clone()))
            .ok_or_else(|| {
                CfCPermutationAlignmentError::MissingCanonicalizationGroup(format!(
                    "{recurrent_group:?}"
                ))
            })?;
        let recurrent_shape = recurrent_snapshot.shape.clone();
        if recurrent_shape != [hidden_dim, hidden_dim] {
            return Err(
                CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
                    group: format!("{recurrent_group:?}"),
                },
            );
        }
        let permuted_recurrent =
            permute_square_checked(&recurrent_snapshot.values, hidden_dim, permutation)
                .ok_or_else(
                    || CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
                        group: format!("{recurrent_group:?}"),
                    },
                )?;
        add_replacement(recurrent_group, &recurrent_shape, permuted_recurrent)?;

        for group_id in [
            CfCWeightGroupId::CellHiddenBias { cell: layer_index },
            CfCWeightGroupId::CellTimeConstant { cell: layer_index },
        ] {
            let group = snapshot
                .groups
                .iter()
                .find(|group| group.target == WeightProgramTarget::CfC(group_id.clone()))
                .ok_or_else(|| {
                    CfCPermutationAlignmentError::MissingCanonicalizationGroup(format!(
                        "{group_id:?}"
                    ))
                })?;
            if group.shape != [hidden_dim] || group.values.len() != hidden_dim {
                return Err(
                    CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
                        group: format!("{group_id:?}"),
                    },
                );
            }
            let replacement = permutation
                .iter()
                .map(|index| group.values[*index])
                .collect::<Vec<_>>();
            add_replacement(group_id, &[hidden_dim], replacement)?;
        }
    }

    let output_group = CfCWeightGroupId::OutputProjection;
    let output_snapshot = snapshot
        .groups
        .iter()
        .find(|group| group.target == WeightProgramTarget::CfC(output_group.clone()))
        .ok_or_else(|| {
            CfCPermutationAlignmentError::MissingCanonicalizationGroup(format!("{output_group:?}"))
        })?;
    let output_shape = output_snapshot.shape.clone();
    if output_shape != [config.output_dim, hidden_dim] {
        return Err(
            CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
                group: format!("{output_group:?}"),
            },
        );
    }
    let permuted_output = permute_matrix_columns_checked(
        &output_snapshot.values,
        config.output_dim,
        hidden_dim,
        permutations.last().expect("at least one layer"),
    )
    .ok_or_else(
        || CfCPermutationAlignmentError::InvalidCanonicalizationGroupShape {
            group: format!("{output_group:?}"),
        },
    )?;
    add_replacement(output_group, &output_shape, permuted_output)?;
    drop(add_replacement);

    if changed_parameter_count == 0 {
        return Err(CfCPermutationAlignmentError::NoOpCanonicalization);
    }

    let total_l2_delta = squared_delta.sqrt() as f32;
    let budget_epsilon = 1.0e-6_f32;
    let max_total_l2_delta = total_l2_delta * (1.0 + 1.0e-6) + budget_epsilon;
    let max_abs_delta = max_abs_delta * (1.0 + 1.0e-6) + budget_epsilon;

    let mut evidence_hasher = blake3::Hasher::new();
    evidence_hasher.update(b"symthaea-cfc-network-hidden-unit-alignment-v1\0");
    evidence_hasher.update(alignments[0].source_model_digest.as_bytes());
    evidence_hasher.update(alignments[0].candidate_model_digest.as_bytes());
    evidence_hasher.update(snapshot.architecture_digest.as_bytes());
    evidence_hasher.update(snapshot.manifest_digest.as_bytes());
    for alignment in &alignments {
        evidence_hasher.update(&(alignment.layer_index as u64).to_le_bytes());
        evidence_hasher.update(alignment.probe_digest.as_bytes());
        for index in alignment.source_to_candidate.iter().copied() {
            evidence_hasher.update(&(index as u64).to_le_bytes());
        }
        evidence_hasher.update(&alignment.mean_squared_activation_error.to_le_bytes());
        if let Some(margin) = alignment.minimum_assignment_margin {
            evidence_hasher.update(&margin.to_le_bytes());
        }
    }

    Ok(WeightProgram {
        format_version: 1,
        architecture_digest: snapshot.architecture_digest,
        manifest_digest: snapshot.manifest_digest,
        base_model_digest: snapshot.model_digest,
        target_specification_digest: task_digest,
        fault_hypothesis_ids: vec![
            "symmetry:multilayer-hidden-unit-permutation-canonicalization".to_owned(),
        ],
        evidence_receipt_ids: vec![format!("alignment:{}", evidence_hasher.finalize().to_hex())],
        max_changed_parameters: changed_parameter_count,
        max_total_l2_delta,
        max_abs_delta,
        operations,
    })
}

fn permute_matrix_rows_checked(
    values: &[f32],
    rows: usize,
    columns: usize,
    permutation: &[usize],
) -> Option<Vec<f32>> {
    if values.len() != rows.checked_mul(columns)? || permutation.len() != rows {
        return None;
    }
    let mut seen = vec![false; rows];
    for index in permutation.iter().copied() {
        if index >= rows || seen[index] {
            return None;
        }
        seen[index] = true;
    }
    let mut result = vec![0.0; values.len()];
    for new_row in 0..rows {
        let old_row = permutation[new_row];
        let source = old_row * columns;
        let destination = new_row * columns;
        result[destination..destination + columns]
            .copy_from_slice(&values[source..source + columns]);
    }
    Some(result)
}

fn permute_square_checked(
    values: &[f32],
    dimension: usize,
    permutation: &[usize],
) -> Option<Vec<f32>> {
    if values.len() != dimension.checked_mul(dimension)? {
        return None;
    }
    let rows = permute_matrix_rows_checked(values, dimension, dimension, permutation)?;
    permute_matrix_columns_checked(&rows, dimension, dimension, permutation)
}

fn permute_matrix_columns_checked(
    values: &[f32],
    rows: usize,
    columns: usize,
    permutation: &[usize],
) -> Option<Vec<f32>> {
    if values.len() != rows.checked_mul(columns)? || permutation.len() != columns {
        return None;
    }
    let mut seen = vec![false; columns];
    for index in permutation.iter().copied() {
        if index >= columns || seen[index] {
            return None;
        }
        seen[index] = true;
    }
    let mut result = vec![0.0; values.len()];
    for row in 0..rows {
        for new_column in 0..columns {
            result[row * columns + new_column] = values[row * columns + permutation[new_column]];
        }
    }
    Some(result)
}

/// Select the next functional probe that most increases assignment confidence.
///
/// Existing probes are treated as frozen evidence. Each candidate probe is
/// appended in turn and the complete signature assignment is recomputed.
/// Candidates are ranked lexicographically by accumulated assignment margin,
/// then by lower mean-squared activation error, then by original index.
///
/// This is deterministic active probe design, not Bayesian expected information
/// gain: no observation-noise likelihood or posterior over latent mappings is
/// assumed. It never edits either network.
pub fn select_cfc_disambiguating_probe(
    source: &CfCNetwork,
    candidate: &CfCNetwork,
    layer_index: usize,
    existing_probes: &[CfCFunctionalProbe],
    candidate_probes: &[CfCFunctionalProbe],
) -> Result<CfCDisambiguatingProbeSelection, CfCPermutationAlignmentError> {
    if candidate_probes.is_empty() {
        return Err(CfCPermutationAlignmentError::EmptyCandidateProbeSet);
    }

    let baseline_alignment = if existing_probes.is_empty() {
        None
    } else {
        let source_set =
            cfc_hidden_unit_activation_signatures(source, layer_index, existing_probes)?;
        let candidate_set =
            cfc_hidden_unit_activation_signatures(candidate, layer_index, existing_probes)?;
        Some(align_cfc_hidden_unit_signatures(
            &source_set,
            &candidate_set,
        )?)
    };

    let baseline_assignment_margin = baseline_alignment
        .as_ref()
        .and_then(|alignment| alignment.minimum_assignment_margin);
    let baseline_mean_squared_activation_error = baseline_alignment
        .as_ref()
        .map(|alignment| alignment.mean_squared_activation_error);

    let mut candidate_scores = Vec::with_capacity(candidate_probes.len());
    let mut best_index = None::<usize>;

    for (candidate_index, probe) in candidate_probes.iter().enumerate() {
        let mut augmented_probes = Vec::with_capacity(existing_probes.len() + 1);
        augmented_probes.extend_from_slice(existing_probes);
        augmented_probes.push(probe.clone());

        let source_set =
            cfc_hidden_unit_activation_signatures(source, layer_index, &augmented_probes)?;
        let candidate_set =
            cfc_hidden_unit_activation_signatures(candidate, layer_index, &augmented_probes)?;
        let alignment = align_cfc_hidden_unit_signatures(&source_set, &candidate_set)?;
        let assignment_margin = alignment.minimum_assignment_margin;

        candidate_scores.push(CfCProbeCandidateScore {
            candidate_index,
            augmented_probe_digest: source_set.probe_digest,
            mean_squared_activation_error: alignment.mean_squared_activation_error,
            assignment_margin,
        });

        if best_index.is_none_or(|best| {
            let measured = &candidate_scores[candidate_index];
            let previous = &candidate_scores[best];
            match measured
                .assignment_margin
                .unwrap_or(0.0)
                .total_cmp(&previous.assignment_margin.unwrap_or(0.0))
            {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Less => false,
                std::cmp::Ordering::Equal => {
                    match previous
                        .mean_squared_activation_error
                        .total_cmp(&measured.mean_squared_activation_error)
                    {
                        std::cmp::Ordering::Greater => true,
                        std::cmp::Ordering::Less => false,
                        std::cmp::Ordering::Equal => candidate_index < best,
                    }
                }
            }
        }) {
            best_index = Some(candidate_index);
        }
    }

    let selected_candidate_index =
        best_index.ok_or(CfCPermutationAlignmentError::NoFeasibleAssignment)?;
    let selected_score = &candidate_scores[selected_candidate_index];
    let selected_assignment_margin = selected_score
        .assignment_margin
        .ok_or(CfCPermutationAlignmentError::NoAlternativeAssignment)?;
    let selected_probe = candidate_probes[selected_candidate_index].clone();

    let source_fingerprint = source
        .fingerprint()
        .map_err(CfCPermutationAlignmentError::FingerprintFailed)?;
    let candidate_fingerprint = candidate
        .fingerprint()
        .map_err(CfCPermutationAlignmentError::FingerprintFailed)?;
    if source_fingerprint.architecture_digest != candidate_fingerprint.architecture_digest {
        return Err(CfCPermutationAlignmentError::ArchitectureMismatch {
            source: source_fingerprint.architecture_digest,
            candidate: candidate_fingerprint.architecture_digest,
        });
    }
    if source_fingerprint.manifest_digest != candidate_fingerprint.manifest_digest {
        return Err(CfCPermutationAlignmentError::ManifestMismatch {
            source: source_fingerprint.manifest_digest,
            candidate: candidate_fingerprint.manifest_digest,
        });
    }

    Ok(CfCDisambiguatingProbeSelection {
        format_version: 1,
        source_model_digest: source_fingerprint.model_digest,
        candidate_model_digest: candidate_fingerprint.model_digest,
        architecture_digest: source_fingerprint.architecture_digest,
        manifest_digest: source_fingerprint.manifest_digest,
        selected_candidate_index,
        selected_probe,
        selected_augmented_probe_digest: selected_score.augmented_probe_digest.clone(),
        baseline_assignment_margin,
        baseline_mean_squared_activation_error,
        selected_assignment_margin,
        selected_mean_squared_activation_error: selected_score.mean_squared_activation_error,
        candidate_scores,
    })
}

const MAX_EXACT_PERMUTATION_HYPOTHESES: usize = 120;
const MAX_GAUSSIAN_EIG_CANDIDATE_PROBES: usize = 32;
const MAX_NOISE_CALIBRATION_SAMPLES: usize = 4096;

/// Select a CfC functional probe by prospective mutual information over exact
/// hidden-unit permutation hypotheses.
///
/// This is the first Bayesian/FEP probe selector for the symmetry workflow.
/// It is deliberately restricted to small layers (at most 6 units with the
/// current 1024-hypothesis ceiling). The likelihood assumes a scalar Gaussian
/// observation channel with the caller-supplied measurement standard deviation.
/// Existing probes update the permutation posterior; candidate probes are then
/// scored by I(Permutation; projected activation | probe). It selects a probe
/// only—it never reorders tensors, creates a WeightProgram, or accepts edits.
pub fn select_cfc_disambiguating_probe_by_gaussian_eig(
    source: &CfCNetwork,
    candidate: &CfCNetwork,
    layer_index: usize,
    existing_probes: &[CfCFunctionalProbe],
    candidate_probes: &[CfCFunctionalProbe],
    noise_calibration: &CfCProjectionNoiseCalibration,
    quadrature_points: usize,
) -> Result<CfCGaussianPermutationEIGSelection, CfCPermutationAlignmentError> {
    if candidate_probes.is_empty() {
        return Err(CfCPermutationAlignmentError::EmptyCandidateProbeSet);
    }
    if candidate_probes.len() > MAX_GAUSSIAN_EIG_CANDIDATE_PROBES {
        return Err(
            CfCPermutationAlignmentError::TooManyGaussianEIGCandidateProbes {
                actual_probes: candidate_probes.len(),
                maximum_probes: MAX_GAUSSIAN_EIG_CANDIDATE_PROBES,
            },
        );
    }
    let source_fingerprint = source
        .fingerprint()
        .map_err(CfCPermutationAlignmentError::FingerprintFailed)?;
    let candidate_fingerprint = candidate
        .fingerprint()
        .map_err(CfCPermutationAlignmentError::FingerprintFailed)?;
    if source_fingerprint.architecture_digest != candidate_fingerprint.architecture_digest {
        return Err(CfCPermutationAlignmentError::ArchitectureMismatch {
            source: source_fingerprint.architecture_digest,
            candidate: candidate_fingerprint.architecture_digest,
        });
    }
    if source_fingerprint.manifest_digest != candidate_fingerprint.manifest_digest {
        return Err(CfCPermutationAlignmentError::ManifestMismatch {
            source: source_fingerprint.manifest_digest,
            candidate: candidate_fingerprint.manifest_digest,
        });
    }

    let hidden_dim = source.config().hidden_dim;
    let hypothesis_count = exact_permutation_count(hidden_dim);
    if hypothesis_count.is_none_or(|count| count > MAX_EXACT_PERMUTATION_HYPOTHESES) {
        return Err(
            CfCPermutationAlignmentError::ExactPermutationSpaceTooLarge {
                hidden_dim,
                maximum_hypotheses: MAX_EXACT_PERMUTATION_HYPOTHESES,
            },
        );
    }
    let permutations = enumerate_hidden_unit_permutations(hidden_dim);
    if permutations.is_empty() {
        return Err(CfCPermutationAlignmentError::InvalidPermutationSignatureDimension);
    }

    let projection = deterministic_permutation_projection(
        &source_fingerprint.architecture_digest,
        layer_index,
        hidden_dim,
    );
    let projection_digest = digest_projection(&projection);
    validate_cfc_projection_noise_calibration(
        noise_calibration,
        &source_fingerprint.model_digest,
        &candidate_fingerprint.model_digest,
        &source_fingerprint.architecture_digest,
        &source_fingerprint.manifest_digest,
        layer_index,
        &projection_digest,
    )?;
    let observation_std_dev = noise_calibration.observation_std_dev;
    let existing_probe_digest = if existing_probes.is_empty() {
        None
    } else {
        Some(cfc_probe_set_digest(layer_index, existing_probes))
    };

    let mut log_posterior = vec![-(permutations.len() as f64).ln(); permutations.len()];
    if !existing_probes.is_empty() {
        let source_signatures =
            cfc_hidden_unit_activation_signatures(source, layer_index, existing_probes)?;
        let candidate_signatures =
            cfc_hidden_unit_activation_signatures(candidate, layer_index, existing_probes)?;
        let _ = align_cfc_hidden_unit_signatures(&source_signatures, &candidate_signatures)?;

        validate_signature_hidden_dimension(&source_signatures, hidden_dim)?;
        validate_signature_hidden_dimension(&candidate_signatures, hidden_dim)?;
        for probe_index in 0..existing_probes.len() {
            let observed =
                projected_candidate_activation(&candidate_signatures, probe_index, &projection);
            let residuals = permutations
                .iter()
                .map(|permutation| {
                    let predicted = projected_permuted_source_activation(
                        &source_signatures,
                        probe_index,
                        &projection,
                        permutation,
                    );
                    (observed - predicted).powi(2)
                })
                .collect::<Vec<_>>();
            for (log_weight, squared_error) in log_posterior.iter_mut().zip(residuals) {
                *log_weight -= squared_error / (2.0 * observation_std_dev * observation_std_dev);
            }
        }
    }

    let posterior_probabilities = normalize_log_weights(&log_posterior)
        .ok_or(CfCPermutationAlignmentError::InvalidPermutationPosterior)?;
    let current_entropy = categorical_entropy_nats(&posterior_probabilities);
    let permutation_hypotheses = permutations
        .iter()
        .zip(posterior_probabilities.iter().copied())
        .enumerate()
        .map(
            |(index, (permutation, prior_probability))| GaussianObservationHypothesis {
                hypothesis_id: format!(
                    "perm-{index}:{}",
                    permutation
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join("-")
                ),
                prior_probability,
            },
        )
        .collect::<Vec<_>>();

    let mut augmented_probe_digests = Vec::with_capacity(candidate_probes.len());
    let mut actions = Vec::with_capacity(candidate_probes.len());
    for (candidate_index, probe) in candidate_probes.iter().enumerate() {
        let mut augmented_probes = Vec::with_capacity(existing_probes.len() + 1);
        augmented_probes.extend_from_slice(existing_probes);
        augmented_probes.push(probe.clone());

        let source_signatures =
            cfc_hidden_unit_activation_signatures(source, layer_index, &augmented_probes)?;
        let candidate_signatures =
            cfc_hidden_unit_activation_signatures(candidate, layer_index, &augmented_probes)?;
        let _ = align_cfc_hidden_unit_signatures(&source_signatures, &candidate_signatures)?;
        validate_signature_hidden_dimension(&source_signatures, hidden_dim)?;
        validate_signature_hidden_dimension(&candidate_signatures, hidden_dim)?;

        let final_probe_index = augmented_probes.len() - 1;
        let outcome_means = permutations
            .iter()
            .map(|permutation| {
                projected_permuted_source_activation(
                    &source_signatures,
                    final_probe_index,
                    &projection,
                    permutation,
                )
            })
            .collect::<Vec<_>>();
        augmented_probe_digests.push(cfc_probe_set_digest(layer_index, &augmented_probes));
        actions.push(GaussianObservationAction {
            action_id: format!("probe-{candidate_index:06}"),
            outcome_means,
        });
    }

    let information_gains = evaluate_gaussian_model_information_gains(
        &permutation_hypotheses,
        &actions,
        GaussianModelInformationGainConfig {
            observation_std_dev,
            quadrature_points,
            tail_sigma: 8.0,
        },
    )?;
    let mut selected_index = 0_usize;
    for index in 1..information_gains.len() {
        let order = information_gains[index]
            .model_information_gain_nats
            .total_cmp(&information_gains[selected_index].model_information_gain_nats);
        if order == std::cmp::Ordering::Greater {
            selected_index = index;
        }
    }

    let scores = information_gains
        .iter()
        .enumerate()
        .map(|(candidate_index, gain)| CfCGaussianPermutationProbeScore {
            candidate_index,
            augmented_probe_digest: augmented_probe_digests[candidate_index].clone(),
            model_information_gain_nats: gain.model_information_gain_nats,
            expected_posterior_permutation_entropy_nats: gain.expected_posterior_model_entropy_nats,
        })
        .collect::<Vec<_>>();
    let selected_gain = &information_gains[selected_index];

    Ok(CfCGaussianPermutationEIGSelection {
        format_version: 1,
        source_model_digest: source_fingerprint.model_digest,
        candidate_model_digest: candidate_fingerprint.model_digest,
        architecture_digest: source_fingerprint.architecture_digest,
        manifest_digest: source_fingerprint.manifest_digest,
        layer_index,
        projection_digest,
        existing_probe_digest,
        selected_candidate_index: selected_index,
        selected_probe: candidate_probes[selected_index].clone(),
        selected_augmented_probe_digest: augmented_probe_digests[selected_index].clone(),
        noise_calibration_digest: noise_calibration.calibration_digest.clone(),
        noise_calibration_method: noise_calibration.method,
        noise_calibration_sample_count: noise_calibration.sample_count,
        observation_std_dev,
        quadrature_points,
        quadrature_tail_sigma: 8.0,
        permutation_hypothesis_count: permutations.len(),
        current_permutation_entropy_nats: current_entropy,
        selected_expected_posterior_entropy_nats: selected_gain
            .expected_posterior_model_entropy_nats,
        selected_model_information_gain_nats: selected_gain.model_information_gain_nats,
        permutation_posterior: permutations
            .into_iter()
            .zip(posterior_probabilities)
            .map(
                |(source_to_candidate, posterior_probability)| CfCPermutationHypothesisPosterior {
                    source_to_candidate,
                    posterior_probability,
                },
            )
            .collect(),
        candidate_scores: scores,
    })
}

fn exact_permutation_count(hidden_dim: usize) -> Option<usize> {
    if hidden_dim == 0 {
        return None;
    }
    let mut count = 1_usize;
    for factor in 2..=hidden_dim {
        count = count.checked_mul(factor)?;
        if count > MAX_EXACT_PERMUTATION_HYPOTHESES {
            return Some(count);
        }
    }
    Some(count)
}

fn enumerate_hidden_unit_permutations(hidden_dim: usize) -> Vec<Vec<usize>> {
    fn visit(
        position: usize,
        current: &mut Vec<usize>,
        used: &mut [bool],
        results: &mut Vec<Vec<usize>>,
    ) {
        if position == used.len() {
            results.push(current.clone());
            return;
        }
        for candidate_index in 0..used.len() {
            if used[candidate_index] {
                continue;
            }
            used[candidate_index] = true;
            current.push(candidate_index);
            visit(position + 1, current, used, results);
            current.pop();
            used[candidate_index] = false;
        }
    }

    if hidden_dim == 0 {
        return Vec::new();
    }
    let mut results = Vec::new();
    visit(
        0,
        &mut Vec::with_capacity(hidden_dim),
        &mut vec![false; hidden_dim],
        &mut results,
    );
    results
}

fn deterministic_permutation_projection(
    architecture_digest: &str,
    layer_index: usize,
    hidden_dim: usize,
) -> Vec<f64> {
    let mut projection = Vec::with_capacity(hidden_dim);
    for unit_index in 0..hidden_dim {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-cfc-permutation-eig-projection-v1\0");
        hasher.update(architecture_digest.as_bytes());
        hasher.update(&(layer_index as u64).to_le_bytes());
        hasher.update(&(unit_index as u64).to_le_bytes());
        let bytes = hasher.finalize();
        let raw = u64::from_le_bytes([
            bytes.as_bytes()[0],
            bytes.as_bytes()[1],
            bytes.as_bytes()[2],
            bytes.as_bytes()[3],
            bytes.as_bytes()[4],
            bytes.as_bytes()[5],
            bytes.as_bytes()[6],
            bytes.as_bytes()[7],
        ]);
        projection.push(2.0 * (raw as f64 / u64::MAX as f64) - 1.0);
    }
    let norm = projection
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm > 1e-15 {
        for value in &mut projection {
            *value /= norm;
        }
    } else if let Some(first) = projection.first_mut() {
        *first = 1.0;
    }
    projection
}

fn digest_projection(projection: &[f64]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-cfc-permutation-projection-digest-v1\0");
    hasher.update(&(projection.len() as u64).to_le_bytes());
    for value in projection {
        hasher.update(&value.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn validate_signature_hidden_dimension(
    signatures: &CfCActivationSignatureSet,
    expected: usize,
) -> Result<(), CfCPermutationAlignmentError> {
    if signatures.signatures.len() != expected {
        return Err(CfCPermutationAlignmentError::InvalidPermutationSignatureDimension);
    }
    Ok(())
}

fn projected_candidate_activation(
    signatures: &CfCActivationSignatureSet,
    probe_index: usize,
    projection: &[f64],
) -> f64 {
    signatures
        .signatures
        .iter()
        .zip(projection)
        .map(|(signature, weight)| signature[probe_index] * weight)
        .sum()
}

fn projected_permuted_source_activation(
    signatures: &CfCActivationSignatureSet,
    probe_index: usize,
    projection: &[f64],
    source_to_candidate: &[usize],
) -> f64 {
    let mut total = 0.0;
    for (source_unit, candidate_unit) in source_to_candidate.iter().copied().enumerate() {
        total += signatures.signatures[source_unit][probe_index] * projection[candidate_unit];
    }
    total
}

fn normalize_log_weights(log_weights: &[f64]) -> Option<Vec<f64>> {
    let maximum = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return None;
    }
    let unnormalized = log_weights
        .iter()
        .map(|value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let normalizer = unnormalized.iter().sum::<f64>();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return None;
    }
    Some(
        unnormalized
            .into_iter()
            .map(|value| value / normalizer)
            .collect(),
    )
}

fn categorical_entropy_nats(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

/// Fail-closed gate for using an activation alignment in parameter transfer.
///
/// The alignment must be a well-formed, versioned receipt with valid source,
/// candidate, and probe digests; a one-to-one complete mapping; finite error;
/// and an unambiguous assignment satisfying the caller's task-calibrated
/// tolerances. This validates evidence only: it does not apply or authorize an
/// edit by itself.
pub fn validate_cfc_alignment_for_parameter_transfer(
    alignment: &CfCHiddenUnitAlignment,
    policy: CfCAlignmentAcceptancePolicy,
) -> Result<(), CfCPermutationAlignmentError> {
    if !policy.max_mean_squared_activation_error.is_finite()
        || policy.max_mean_squared_activation_error < 0.0
        || !policy.min_assignment_margin.is_finite()
        || policy.min_assignment_margin < 0.0
    {
        return Err(CfCPermutationAlignmentError::InvalidAcceptancePolicy);
    }
    if alignment.format_version != 1 {
        return Err(CfCPermutationAlignmentError::MalformedAlignmentReceipt {
            field: "format_version",
        });
    }
    if !is_lowercase_hex_digest(&alignment.source_model_digest) {
        return Err(CfCPermutationAlignmentError::MalformedAlignmentReceipt {
            field: "source_model_digest",
        });
    }
    if !is_lowercase_hex_digest(&alignment.candidate_model_digest) {
        return Err(CfCPermutationAlignmentError::MalformedAlignmentReceipt {
            field: "candidate_model_digest",
        });
    }
    if !is_lowercase_hex_digest(&alignment.architecture_digest) {
        return Err(CfCPermutationAlignmentError::MalformedAlignmentReceipt {
            field: "architecture_digest",
        });
    }
    if !is_lowercase_hex_digest(&alignment.manifest_digest) {
        return Err(CfCPermutationAlignmentError::MalformedAlignmentReceipt {
            field: "manifest_digest",
        });
    }
    if !is_lowercase_hex_digest(&alignment.probe_digest) {
        return Err(CfCPermutationAlignmentError::MalformedAlignmentReceipt {
            field: "probe_digest",
        });
    }
    let unit_count = alignment.source_to_candidate.len();
    if unit_count == 0 {
        return Err(CfCPermutationAlignmentError::MalformedAlignmentReceipt {
            field: "source_to_candidate",
        });
    }
    let mut seen = vec![false; unit_count];
    for candidate_unit in alignment.source_to_candidate.iter().copied() {
        if candidate_unit >= unit_count || seen[candidate_unit] {
            return Err(CfCPermutationAlignmentError::MalformedAlignmentReceipt {
                field: "source_to_candidate",
            });
        }
        seen[candidate_unit] = true;
    }
    if !alignment.mean_squared_activation_error.is_finite()
        || alignment.mean_squared_activation_error < 0.0
    {
        return Err(CfCPermutationAlignmentError::MalformedAlignmentReceipt {
            field: "mean_squared_activation_error",
        });
    }
    if alignment.mean_squared_activation_error > policy.max_mean_squared_activation_error {
        return Err(
            CfCPermutationAlignmentError::ActivationErrorAboveThreshold {
                measured: alignment.mean_squared_activation_error,
                maximum: policy.max_mean_squared_activation_error,
            },
        );
    }

    // A one-unit layer has no alternative permutation. Every larger layer
    // must carry a finite measured assignment margin.
    if unit_count > 1 {
        let Some(margin) = alignment.minimum_assignment_margin else {
            return Err(CfCPermutationAlignmentError::AssignmentAmbiguous {
                measured: None,
                minimum: policy.min_assignment_margin,
            });
        };
        if !margin.is_finite() || margin < policy.min_assignment_margin {
            return Err(CfCPermutationAlignmentError::AssignmentAmbiguous {
                measured: Some(margin),
                minimum: policy.min_assignment_margin,
            });
        }
    }
    Ok(())
}

/// Compute the smallest cost increase of any reassignment that forbids one
/// matched pair. Alternative assignments decompose into alternating cycles.
/// Floyd-Warshall over the row-transition graph computes all shortest cycles in
/// O(n^3), avoiding n separate O(n^3) Hungarian re-solves.
fn alternating_cycle_assignment_margin(costs: &[Vec<f64>], assignment: &[usize]) -> Option<f64> {
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
                let through = distance[source][intermediate] + distance[intermediate][destination];
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
    total_cost.is_finite().then_some((assignment, total_cost))
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::sequential_weight_program::{
        SequentialEvidenceBatch, SequentialWeightProgramTemplate,
        accept_verified_sequential_candidate, begin_sequential_edit_session,
        prepare_sequential_candidate,
    };
    use symthaea_core::weight_space_program::WeightProgramOperation;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn noise_calibration(
        source: &CfCNetwork,
        candidate: &CfCNetwork,
        layer_index: usize,
        calibration_digest: char,
    ) -> CfCProjectionNoiseCalibration {
        calibrate_cfc_projection_observation_noise(
            source,
            candidate,
            layer_index,
            &digest(calibration_digest),
            CfCProjectionNoiseCalibrationMethod::RepeatedProjectionRuns,
            &[0.10, 0.12, 0.08, 0.11, 0.09, 0.105, 0.095, 0.115],
        )
        .expect("repeated projection noise calibration")
    }

    fn signature_set(
        signatures: Vec<Vec<f64>>,
        model: char,
        probe: char,
    ) -> CfCActivationSignatureSet {
        CfCActivationSignatureSet {
            format_version: 1,
            model_digest: digest(model),
            architecture_digest: digest('x'),
            manifest_digest: digest('y'),
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
        let genesis =
            symthaea_core::genesis::GenesisSeed::from_phrase("symthaea-cfc-probe-digest-test-v1");
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
        let reordered = cfc_hidden_unit_activation_signatures(&network, 0, &reordered_probes)
            .expect("reordered receipt");

        assert_eq!(first.probe_digest, repeated.probe_digest);
        assert_ne!(first.probe_digest, reordered.probe_digest);
        assert_eq!(first.model_digest, reordered.model_digest);
        assert_eq!(first.signatures, repeated.signatures);
    }

    fn brute_force_best_cost(costs: &[Vec<f64>]) -> Option<f64> {
        fn visit(row: usize, costs: &[Vec<f64>], used: &mut [bool], running: f64, best: &mut f64) {
            if row == costs.len() {
                *best = best.min(running);
                return;
            }
            for column in 0..costs.len() {
                let edge = costs[row][column];
                if !used[column] && edge.is_finite() {
                    used[column] = true;
                    visit(row + 1, costs, used, running + edge, best);
                    used[column] = false;
                }
            }
        }

        if costs.is_empty() || costs.iter().any(|row| row.len() != costs.len()) {
            return None;
        }
        let mut best = f64::INFINITY;
        visit(0, costs, &mut vec![false; costs.len()], 0.0, &mut best);
        best.is_finite().then_some(best)
    }

    #[test]
    fn hungarian_and_alternating_cycle_margin_match_exhaustive_assignments() {
        let matrices = [
            vec![
                vec![0.0, 4.0, 9.0],
                vec![7.0, 0.0, 5.0],
                vec![3.0, 8.0, 0.0],
            ],
            vec![
                vec![0.2, 0.7, 1.5, 0.9],
                vec![0.8, 0.1, 0.6, 1.4],
                vec![1.1, 0.4, 0.3, 0.5],
                vec![0.6, 1.0, 0.8, 0.2],
            ],
            vec![
                vec![0.0, 0.0, 4.0],
                vec![0.0, 0.0, 3.0],
                vec![2.0, 5.0, 0.0],
            ],
        ];

        for costs in matrices {
            let (assignment, total) =
                minimum_cost_assignment(&costs).expect("Hungarian assignment");
            let exhaustive = brute_force_best_cost(&costs).expect("exhaustive assignment");
            assert!((total - exhaustive).abs() < 1e-10);

            let mut exhaustive_margin = f64::INFINITY;
            for (row, column) in assignment.iter().copied().enumerate() {
                let mut forbidden = costs.clone();
                forbidden[row][column] = f64::INFINITY;
                let alternative = brute_force_best_cost(&forbidden);
                if let Some(alternative_cost) = alternative {
                    exhaustive_margin = exhaustive_margin.min((alternative_cost - total).max(0.0));
                }
            }
            let cycle_margin = alternating_cycle_assignment_margin(&costs, &assignment);
            if exhaustive_margin.is_infinite() {
                assert!(cycle_margin.is_infinite());
            } else {
                assert!((cycle_margin - exhaustive_margin).abs() < 1e-10);
            }
        }
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
                let measured_margin = alternating_cycle_assignment_margin(&costs, &assignment)
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
    fn diagnostic_probe_selection_is_deterministic_and_selects_best_margin() {
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("cfc-probe-selection-v1");
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
        let source = CfCNetwork::from_genesis(config, &genesis, "probe-selection");
        let candidate = source.clone();
        let existing = vec![CfCFunctionalProbe {
            input: vec![0.2, -0.3],
            dt: 0.05,
        }];
        let candidates = vec![
            CfCFunctionalProbe {
                input: vec![0.2, 0.7],
                dt: 0.05,
            },
            CfCFunctionalProbe {
                input: vec![-0.8, 0.1],
                dt: 0.2,
            },
            CfCFunctionalProbe {
                input: vec![0.9, -0.6],
                dt: 0.1,
            },
        ];

        let selection =
            select_cfc_disambiguating_probe(&source, &candidate, 0, &existing, &candidates)
                .expect("select probe");
        let repeated =
            select_cfc_disambiguating_probe(&source, &candidate, 0, &existing, &candidates)
                .expect("repeat probe selection");

        assert_eq!(
            selection.selected_candidate_index,
            repeated.selected_candidate_index
        );
        assert_eq!(
            selection.selected_augmented_probe_digest,
            repeated.selected_augmented_probe_digest
        );
        assert_eq!(selection.candidate_scores, repeated.candidate_scores);
        assert_eq!(selection.candidate_scores.len(), candidates.len());
        assert!(selection.baseline_assignment_margin.is_some());
        assert!(
            selection.selected_assignment_margin >= selection.baseline_assignment_margin.unwrap()
        );
        assert_eq!(
            selection.selected_assignment_margin,
            selection.candidate_scores[selection.selected_candidate_index]
                .assignment_margin
                .expect("multi-unit assignment margin")
        );
        assert_eq!(
            selection.selected_probe,
            candidates[selection.selected_candidate_index]
        );
        assert_eq!(
            source.fingerprint().unwrap().model_digest,
            candidate.fingerprint().unwrap().model_digest
        );
    }

    #[test]
    fn diagnostic_probe_selection_rejects_empty_candidates_and_single_unit_layers() {
        let genesis =
            symthaea_core::genesis::GenesisSeed::from_phrase("cfc-probe-selection-edge-v1");
        let config = CfCNetworkConfig {
            input_dim: 1,
            hidden_dim: 1,
            num_layers: 1,
            output_dim: 1,
            cell_config: CfCConfig {
                input_dim: 1,
                hidden_dim: 1,
                use_backbone: false,
                dropout: 0.0,
                ..Default::default()
            },
            residual: false,
            bidirectional: false,
            enable_online_learning: false,
            ..Default::default()
        };
        let network = CfCNetwork::from_genesis(config, &genesis, "one-unit");
        let probe = CfCFunctionalProbe {
            input: vec![0.5],
            dt: 0.1,
        };
        assert!(matches!(
            select_cfc_disambiguating_probe(&network, &network, 0, &[], &[]),
            Err(CfCPermutationAlignmentError::EmptyCandidateProbeSet)
        ));
        assert!(matches!(
            select_cfc_disambiguating_probe(&network, &network, 0, &[], &[probe]),
            Err(CfCPermutationAlignmentError::NoAlternativeAssignment)
        ));
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
        let alignment = align_cfc_hidden_unit_signatures(&source, &candidate).expect("alignment");

        assert_eq!(alignment.source_to_candidate, vec![1, 2, 0]);
        assert_eq!(alignment.source_model_digest, digest('a'));
        assert_eq!(alignment.candidate_model_digest, digest('b'));
        assert_eq!(alignment.architecture_digest, digest('x'));
        assert_eq!(alignment.manifest_digest, digest('y'));
        assert_eq!(alignment.probe_digest, digest('c'));
        assert!(alignment.mean_squared_activation_error < 1e-15);
        assert!(
            alignment
                .minimum_assignment_margin
                .is_some_and(|margin| margin > 0.0)
        );
    }

    #[test]
    fn activation_signatures_use_hidden_dimension_for_deeper_layers() {
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("cfc-deep-layer-probe-v1");
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
        let signatures = cfc_hidden_unit_activation_signatures(&network, 1, &probes)
            .expect("layer one is observed after external input propagation");
        assert_eq!(signatures.signatures.len(), 4);
        assert!(
            signatures
                .signatures
                .iter()
                .all(|signature| signature.len() == 1)
        );

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
    fn alignment_transfer_gate_accepts_well_separated_mapping() {
        let source = signature_set(
            vec![
                vec![0.0, 0.2, 0.8],
                vec![1.0, -0.4, 0.3],
                vec![-0.5, 0.6, -0.1],
            ],
            'a',
            'c',
        );
        let candidate = signature_set(
            vec![
                source.signatures[2].clone(),
                source.signatures[0].clone(),
                source.signatures[1].clone(),
            ],
            'b',
            'c',
        );
        let alignment = align_cfc_hidden_unit_signatures(&source, &candidate).expect("alignment");
        validate_cfc_alignment_for_parameter_transfer(
            &alignment,
            CfCAlignmentAcceptancePolicy {
                max_mean_squared_activation_error: 1e-12,
                min_assignment_margin: 1e-6,
            },
        )
        .expect("well-separated correspondence is acceptable");
    }

    #[test]
    fn alignment_transfer_gate_rejects_ambiguous_or_malformed_receipts() {
        let ambiguous_source = signature_set(vec![vec![0.0, 1.0], vec![0.0, 1.0]], 'a', 'c');
        let ambiguous_candidate = signature_set(vec![vec![0.0, 1.0], vec![0.0, 1.0]], 'b', 'c');
        let ambiguous = align_cfc_hidden_unit_signatures(&ambiguous_source, &ambiguous_candidate)
            .expect("ambiguous alignment is still diagnosable");
        assert!(matches!(
            validate_cfc_alignment_for_parameter_transfer(
                &ambiguous,
                CfCAlignmentAcceptancePolicy {
                    max_mean_squared_activation_error: 1e-12,
                    min_assignment_margin: 1e-8,
                }
            ),
            Err(CfCPermutationAlignmentError::AssignmentAmbiguous { .. })
        ));

        let mut malformed = ambiguous.clone();
        malformed.source_to_candidate = vec![0, 0];
        assert!(matches!(
            validate_cfc_alignment_for_parameter_transfer(
                &malformed,
                CfCAlignmentAcceptancePolicy {
                    max_mean_squared_activation_error: 1.0,
                    min_assignment_margin: 0.0,
                }
            ),
            Err(CfCPermutationAlignmentError::MalformedAlignmentReceipt {
                field: "source_to_candidate"
            })
        ));
    }

    #[test]
    fn alignment_transfer_gate_rejects_activation_error_and_invalid_policy() {
        let source = signature_set(vec![vec![0.0, 1.0], vec![1.0, 0.0]], 'a', 'c');
        let candidate = signature_set(vec![vec![0.1, 1.1], vec![1.1, 0.1]], 'b', 'c');
        let alignment = align_cfc_hidden_unit_signatures(&source, &candidate).expect("alignment");
        assert!(matches!(
            validate_cfc_alignment_for_parameter_transfer(
                &alignment,
                CfCAlignmentAcceptancePolicy {
                    max_mean_squared_activation_error: 0.0,
                    min_assignment_margin: 0.0,
                }
            ),
            Err(CfCPermutationAlignmentError::ActivationErrorAboveThreshold { .. })
        ));
        assert!(matches!(
            validate_cfc_alignment_for_parameter_transfer(
                &alignment,
                CfCAlignmentAcceptancePolicy {
                    max_mean_squared_activation_error: f64::NAN,
                    min_assignment_margin: 0.0,
                }
            ),
            Err(CfCPermutationAlignmentError::InvalidAcceptancePolicy)
        ));
    }

    #[test]
    fn assignment_margin_accumulates_evidence_without_changing_mean_fit_error() {
        let source_once = signature_set(vec![vec![0.0], vec![1.0], vec![3.0]], 'a', 'c');
        let candidate_once = signature_set(vec![vec![3.1], vec![0.1], vec![1.1]], 'b', 'c');
        let once =
            align_cfc_hidden_unit_signatures(&source_once, &candidate_once).expect("one probe");

        let source_twice = signature_set(
            vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![3.0, 3.0]],
            'a',
            'd',
        );
        let candidate_twice = signature_set(
            vec![vec![3.1, 3.1], vec![0.1, 0.1], vec![1.1, 1.1]],
            'b',
            'd',
        );
        let twice =
            align_cfc_hidden_unit_signatures(&source_twice, &candidate_twice).expect("two probes");

        assert!(
            (twice.mean_squared_activation_error - once.mean_squared_activation_error).abs()
                < 1e-12
        );
        let one_margin = once.minimum_assignment_margin.expect("one-probe margin");
        let two_margin = twice.minimum_assignment_margin.expect("two-probe margin");
        assert!((two_margin - 2.0 * one_margin).abs() < 1e-10);
    }

    #[test]
    fn activation_signature_alignment_exposes_ambiguous_units() {
        let source = vec![vec![0.0, 1.0], vec![0.0, 1.0], vec![2.0, 3.0]];
        let candidate = vec![vec![2.0, 3.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let source = signature_set(source, 'a', 'c');
        let candidate = signature_set(candidate, 'b', 'c');
        let alignment = align_cfc_hidden_unit_signatures(&source, &candidate).expect("alignment");

        assert!(alignment.mean_squared_activation_error < 1e-15);
        assert!(
            alignment
                .minimum_assignment_margin
                .is_some_and(|margin| margin < 1e-15)
        );
    }

    #[test]
    fn activation_alignment_rejects_different_architectures_and_manifests() {
        let source = signature_set(vec![vec![0.0, 1.0]], 'a', 'c');
        let mut different_architecture = signature_set(vec![vec![0.0, 1.0]], 'b', 'c');
        different_architecture.architecture_digest = digest('u');
        assert!(matches!(
            align_cfc_hidden_unit_signatures(&source, &different_architecture),
            Err(CfCPermutationAlignmentError::ArchitectureMismatch { .. })
        ));

        let mut different_manifest = signature_set(vec![vec![0.0, 1.0]], 'b', 'c');
        different_manifest.manifest_digest = digest('v');
        assert!(matches!(
            align_cfc_hidden_unit_signatures(&source, &different_manifest),
            Err(CfCPermutationAlignmentError::ManifestMismatch { .. })
        ));
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
        let snapshot = cfc_weight_model_snapshot(&original, digest('d')).expect("snapshot");
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
                    target: WeightProgramTarget::CfC(CfCWeightGroupId::CellInputProjection {
                        cell: 0,
                    }),
                    values: permute_rows(&input, 4, 2, &permutation),
                },
                WeightProgramOperation::ReplaceGroup {
                    target: WeightProgramTarget::CfC(CfCWeightGroupId::CellRecurrentProjection {
                        cell: 0,
                    }),
                    values: permute_square(&recurrent, 4, &permutation),
                },
                WeightProgramOperation::ReplaceGroup {
                    target: WeightProgramTarget::CfC(CfCWeightGroupId::CellHiddenBias { cell: 0 }),
                    values: permutation.iter().map(|index| bias[*index]).collect(),
                },
                WeightProgramOperation::ReplaceGroup {
                    target: WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant {
                        cell: 0,
                    }),
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
            candidate.base_native_model_digest, candidate.candidate_native_model_digest,
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
        let source_signatures = cfc_hidden_unit_activation_signatures(&original, 0, &probes)
            .expect("source activation signatures");
        let candidate_signatures = cfc_hidden_unit_activation_signatures(&permuted, 0, &probes)
            .expect("candidate activation signatures");
        assert_eq!(original.state(), source_state_before);
        assert_eq!(source_signatures.model_digest, snapshot.model_digest);
        assert_eq!(
            candidate_signatures.model_digest,
            candidate.candidate_native_model_digest
        );
        assert_eq!(
            source_signatures.probe_digest,
            candidate_signatures.probe_digest
        );
        let alignment = align_cfc_hidden_unit_signatures(&source_signatures, &candidate_signatures)
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
        assert!(
            alignment
                .minimum_assignment_margin
                .is_some_and(|margin| margin > 0.0)
        );

        let canonicalization = build_cfc_hidden_unit_canonicalization_program(
            &original,
            &permuted,
            digest('d'),
            &alignment,
            CfCAlignmentAcceptancePolicy {
                max_mean_squared_activation_error: 1e-10,
                min_assignment_margin: 0.0,
            },
        )
        .expect("verified coordinate canonicalization program");
        assert_eq!(canonicalization.operations.len(), 5);
        assert_eq!(
            canonicalization.fault_hypothesis_ids,
            vec!["symmetry:hidden-unit-permutation-canonicalization"]
        );
        let canonicalized =
            materialize_cfc_weight_program(&permuted, digest('d'), &canonicalization)
                .expect("canonicalized candidate");
        assert_eq!(
            canonicalized.candidate_native_model_digest,
            snapshot.model_digest
        );
        assert_eq!(
            permuted
                .fingerprint()
                .expect("permuted source remains immutable")
                .model_digest,
            candidate.candidate_native_model_digest
        );
        assert_eq!(
            original
                .fingerprint()
                .expect("original remains immutable")
                .model_digest,
            snapshot.model_digest
        );

        // The coordinate change must preserve the candidate's function and
        // recover the source coordinate representation for this exact fixture.
        let canonical_signatures =
            cfc_hidden_unit_activation_signatures(&canonicalized.network, 0, &probes)
                .expect("canonicalized signatures");
        assert_eq!(
            canonical_signatures.signatures,
            source_signatures.signatures
        );
        let mut permuted_function = permuted.clone();
        let mut canonicalized_function = canonicalized.network.clone();
        for input in &inputs {
            let before = permuted_function.forward(input, 0.1);
            let after = canonicalized_function.forward(input, 0.1);
            for (left, right) in before.iter().zip(after.iter()) {
                assert!((left - right).abs() < 1e-6);
            }
        }

        // A stale alignment cannot authorize a parameter rewrite after either
        // model fingerprint changes.
        assert!(matches!(
            build_cfc_hidden_unit_canonicalization_program(
                &original,
                &inconsistent.network,
                digest('d'),
                &alignment,
                CfCAlignmentAcceptancePolicy {
                    max_mean_squared_activation_error: 1e-10,
                    min_assignment_margin: 0.0,
                },
            ),
            Err(CfCPermutationAlignmentError::StaleAlignment {
                field: "candidate model"
            })
        ));

        // Verify the learned correspondence on a disjoint probe sequence.
        // No assignment is recomputed on the held-out probes.
        let held_out_probes = [
            CfCFunctionalProbe {
                input: vec![0.71, -0.19],
                dt: 0.03,
            },
            CfCFunctionalProbe {
                input: vec![-0.63, 0.42],
                dt: 0.17,
            },
            CfCFunctionalProbe {
                input: vec![0.08, 0.94],
                dt: 0.29,
            },
        ];
        let source_held_out = cfc_hidden_unit_activation_signatures(&original, 0, &held_out_probes)
            .expect("source held-out signatures");
        let candidate_held_out =
            cfc_hidden_unit_activation_signatures(&permuted, 0, &held_out_probes)
                .expect("candidate held-out signatures");
        assert_ne!(source_held_out.probe_digest, alignment.probe_digest);
        assert_eq!(
            source_held_out.probe_digest,
            candidate_held_out.probe_digest
        );
        assert_eq!(
            source_held_out.architecture_digest,
            alignment.architecture_digest
        );
        assert_eq!(
            candidate_held_out.architecture_digest,
            alignment.architecture_digest
        );
        assert_eq!(source_held_out.manifest_digest, alignment.manifest_digest);
        assert_eq!(
            candidate_held_out.manifest_digest,
            alignment.manifest_digest
        );
        let mut held_out_squared_error = 0.0_f64;
        let mut held_out_coordinate_count = 0usize;
        for (source_unit, candidate_unit) in
            alignment.source_to_candidate.iter().copied().enumerate()
        {
            for (source_activation, candidate_activation) in source_held_out.signatures[source_unit]
                .iter()
                .zip(candidate_held_out.signatures[candidate_unit].iter())
            {
                held_out_squared_error += (source_activation - candidate_activation).powi(2);
                held_out_coordinate_count += 1;
            }
        }
        let held_out_mse = held_out_squared_error / held_out_coordinate_count as f64;
        assert!(
            held_out_mse < 1e-10,
            "the selection-time mapping must generalize to held-out probes, mse={held_out_mse}"
        );

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
    fn multilayer_canonicalization_propagates_each_layer_permutation_through_adjacent_projections()
    {
        let genesis =
            symthaea_core::genesis::GenesisSeed::from_phrase("cfc-multilayer-canonicalization-v1");
        let config = CfCNetworkConfig {
            input_dim: 2,
            hidden_dim: 3,
            num_layers: 2,
            output_dim: 2,
            cell_config: CfCConfig {
                input_dim: 2,
                hidden_dim: 3,
                use_backbone: false,
                dropout: 0.0,
                ..Default::default()
            },
            residual: false,
            bidirectional: false,
            enable_online_learning: false,
            ..Default::default()
        };
        let source = CfCNetwork::from_genesis(config, &genesis, "multilayer-source");
        let mut permuted = source.clone();
        let task_digest = digest('d');
        let snapshot =
            cfc_weight_model_snapshot(&source, task_digest.clone()).expect("source snapshot");
        let layer_permutations = [vec![2usize, 0, 1], vec![1usize, 2, 0]];

        let get = |id: CfCWeightGroupId| {
            snapshot
                .groups
                .iter()
                .find(|group| group.target == WeightProgramTarget::CfC(id.clone()))
                .expect("parameter group")
                .values
                .clone()
        };
        let mut replacements = Vec::new();
        for layer_index in 0..2 {
            let permutation = &layer_permutations[layer_index];
            let input_id = CfCWeightGroupId::CellInputProjection { cell: layer_index };
            let input = get(input_id.clone());
            let input_width = if layer_index == 0 { 2 } else { 3 };
            let mut permuted_input = permute_rows(&input, 3, input_width, permutation);
            if layer_index > 0 {
                permuted_input = permute_columns(
                    &permuted_input,
                    3,
                    input_width,
                    &layer_permutations[layer_index - 1],
                );
            }
            replacements.push(CfCParameterGroupReplacement {
                id: native_group_id(&input_id),
                values: permuted_input,
            });

            let recurrent_id = CfCWeightGroupId::CellRecurrentProjection { cell: layer_index };
            replacements.push(CfCParameterGroupReplacement {
                id: native_group_id(&recurrent_id),
                values: permute_square(&get(recurrent_id.clone()), 3, permutation),
            });

            for group_id in [
                CfCWeightGroupId::CellHiddenBias { cell: layer_index },
                CfCWeightGroupId::CellTimeConstant { cell: layer_index },
            ] {
                let values = get(group_id.clone());
                replacements.push(CfCParameterGroupReplacement {
                    id: native_group_id(&group_id),
                    values: permutation.iter().map(|index| values[*index]).collect(),
                });
            }
        }

        let output_id = CfCWeightGroupId::OutputProjection;
        replacements.push(CfCParameterGroupReplacement {
            id: native_group_id(&output_id),
            values: permute_columns(&get(output_id.clone()), 2, 3, &layer_permutations[1]),
        });
        let source_digest = source
            .fingerprint()
            .expect("source fingerprint")
            .model_digest;
        let permuted_digest_before = permuted
            .fingerprint()
            .expect("permuted base fingerprint")
            .model_digest;
        permuted
            .try_apply_parameter_group_replacements(&permuted_digest_before, &replacements)
            .expect("apply coupled two-layer symmetry");

        let probe_set = vec![
            CfCFunctionalProbe {
                input: vec![0.25, -0.75],
                dt: 0.05,
            },
            CfCFunctionalProbe {
                input: vec![-0.4, 0.6],
                dt: 0.2,
            },
            CfCFunctionalProbe {
                input: vec![0.9, 0.1],
                dt: 0.1,
            },
            CfCFunctionalProbe {
                input: vec![-0.2, -0.8],
                dt: 0.15,
            },
        ];
        let mut alignments = Vec::new();
        for layer_index in 0..2 {
            let source_signatures =
                cfc_hidden_unit_activation_signatures(&source, layer_index, &probe_set)
                    .expect("source layer signatures");
            let candidate_signatures =
                cfc_hidden_unit_activation_signatures(&permuted, layer_index, &probe_set)
                    .expect("permuted layer signatures");
            let alignment =
                align_cfc_hidden_unit_signatures(&source_signatures, &candidate_signatures)
                    .expect("layer hidden-unit alignment");
            let expected_inverse = match layer_index {
                0 => vec![1, 2, 0],
                1 => vec![2, 0, 1],
                _ => unreachable!(),
            };
            assert_eq!(alignment.source_to_candidate, expected_inverse);
            assert!(alignment.mean_squared_activation_error < 1e-10);
            assert!(
                alignment
                    .minimum_assignment_margin
                    .is_some_and(|margin| margin > 0.0)
            );
            alignments.push(alignment);
        }

        let canonicalization = build_cfc_network_hidden_unit_canonicalization_program(
            &source,
            &permuted,
            task_digest.clone(),
            &alignments,
            CfCAlignmentAcceptancePolicy {
                max_mean_squared_activation_error: 1e-10,
                min_assignment_margin: 0.0,
            },
        )
        .expect("multi-layer canonicalization program");
        assert_eq!(canonicalization.operations.len(), 9);

        let canonicalized =
            materialize_cfc_weight_program(&permuted, task_digest, &canonicalization)
                .expect("candidate-only multi-layer canonicalization");
        assert_eq!(canonicalized.receipt.edited_group_count, 9);
        assert!(canonicalized.receipt.changed_parameter_count > 0);
        assert_ne!(
            canonicalized.candidate_native_model_digest,
            permuted_digest_before
        );
        assert_eq!(canonicalized.candidate_native_model_digest, source_digest);

        let source_digest_after = source
            .fingerprint()
            .expect("source remains unchanged")
            .model_digest;
        let permuted_digest_after = permuted
            .fingerprint()
            .expect("permuted source remains unchanged")
            .model_digest;
        assert_eq!(source_digest_after, source_digest);
        assert_eq!(permuted_digest_after, permuted_digest_before);

        let inputs = probe_set
            .iter()
            .map(|probe| ndarray::Array1::from_vec(probe.input.clone()))
            .collect::<Vec<_>>();
        let mut source_eval = source.clone();
        let mut permuted_eval = permuted.clone();
        let mut canonical_eval = canonicalized.network.clone();
        let mut source_permuted_error = 0.0_f64;
        let mut source_canonical_error = 0.0_f64;
        for (input, probe) in inputs.iter().zip(probe_set.iter()) {
            let expected = source_eval.forward(input, probe.dt);
            let equivalent = permuted_eval.forward(input, probe.dt);
            let restored = canonical_eval.forward(input, probe.dt);
            for ((a, b), restored_value) in
                expected.iter().zip(equivalent.iter()).zip(restored.iter())
            {
                source_permuted_error += f64::from((*a - *b).powi(2));
                source_canonical_error += f64::from((*a - *restored_value).powi(2));
            }
        }
        assert!(source_permuted_error.sqrt() < 1e-5);
        assert!(source_canonical_error.sqrt() < 1e-5);

        assert!(matches!(
            build_cfc_network_hidden_unit_canonicalization_program(
                &source,
                &permuted,
                digest('d'),
                &alignments[..1],
                CfCAlignmentAcceptancePolicy {
                    max_mean_squared_activation_error: 1e-10,
                    min_assignment_margin: 0.0,
                },
            ),
            Err(CfCPermutationAlignmentError::UnsupportedCanonicalizationTopology)
        ));
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
        let snapshot = cfc_weight_model_snapshot(&network, task_digest.clone()).expect("snapshot");

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

        let base_digest_before = network
            .fingerprint()
            .expect("base fingerprint")
            .model_digest;
        let candidate = materialize_cfc_weight_program(&network, digest('d'), &program)
            .expect("isolated candidate");
        let base_digest_after = network
            .fingerprint()
            .expect("base fingerprint")
            .model_digest;

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
        let tau_target = WeightProgramTarget::CfC(CfCWeightGroupId::CellTimeConstant { cell: 0 });
        let healthy_tau = healthy_snapshot
            .groups
            .iter()
            .find(|group| group.target == tau_target)
            .expect("healthy tau")
            .values
            .clone();

        let faulty_values = healthy_tau
            .iter()
            .map(|value| value * 3.0)
            .collect::<Vec<_>>();
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
        let _ = faulty.forward(&ndarray::Array1::from_vec(vec![0.25, -0.75]), 0.2);
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

        let candidate = materialize_cfc_weight_program(&faulty, task_digest.clone(), &repair)
            .expect("candidate repair");
        assert_eq!(
            faulty
                .fingerprint()
                .expect("faulty remains unchanged")
                .model_digest,
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
        assert!(
            candidate
                .network
                .adam_states
                .iter()
                .all(|state| state.t == 0)
        );
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

        let healthy_now =
            cfc_weight_model_snapshot(&healthy, task_digest).expect("healthy current");
        assert_eq!(
            repair.validate_and_materialize(&healthy_now),
            Err(WeightProgramError::StaleDigest("base model"))
        );
    }

    #[test]
    fn sequential_evidence_program_executes_through_native_cfc_candidate_boundary() {
        let genesis =
            symthaea_core::genesis::GenesisSeed::from_phrase("cfc-sequential-wsp-integration-v1");
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
        let source = CfCNetwork::from_genesis(config, &genesis, "sequential-wsp");
        let task_digest = digest('d');
        let initial_snapshot =
            cfc_weight_model_snapshot(&source, task_digest.clone()).expect("initial snapshot");
        let tau_group_id = CfCWeightGroupId::CellTimeConstant { cell: 0 };
        let tau_group = initial_snapshot
            .groups
            .iter()
            .find(|group| group.target == WeightProgramTarget::CfC(tau_group_id.clone()))
            .expect("cell tau parameter group");
        let template = SequentialWeightProgramTemplate {
            format_version: 1,
            architecture_digest: initial_snapshot.architecture_digest.clone(),
            manifest_digest: initial_snapshot.manifest_digest.clone(),
            target_specification_digest: task_digest.clone(),
            fault_hypothesis_ids: vec!["fault:recurrent-timescale-drift".into()],
            operations: vec![WeightProgramOperation::ScaleRange {
                target: tau_group.target.clone(),
                start: 0,
                end: tau_group.values.len(),
                factor: 1.01,
            }],
            max_changed_parameters_per_batch: tau_group.values.len(),
            max_total_l2_delta_per_batch: 1.0,
            max_abs_delta_per_batch: 1.0,
            max_cumulative_l2_path: 2.0,
            max_cumulative_parameter_change_events: 2 * tau_group.values.len(),
            max_batches: 2,
        };
        let session_id = digest('s');
        let session =
            begin_sequential_edit_session(session_id.clone(), &template, &initial_snapshot)
                .expect("begin native CfC sequential session");

        let first_batch = SequentialEvidenceBatch {
            session_id: session_id.clone(),
            batch_index: 0,
            parent_candidate_model_digest: initial_snapshot.model_digest.clone(),
            evidence_batch_digest: digest('e'),
            observation_manifest_digest: digest('o'),
        };
        let first_prepared =
            prepare_sequential_candidate(&session, &template, &initial_snapshot, &first_batch)
                .expect("prepare first evidence batch");
        let first_native =
            materialize_cfc_weight_program(&source, task_digest.clone(), &first_prepared.program)
                .expect("materialize first native CfC candidate");
        assert_eq!(
            source
                .fingerprint()
                .expect("source remains immutable")
                .model_digest,
            initial_snapshot.model_digest
        );
        assert_ne!(
            first_native.candidate_native_model_digest,
            initial_snapshot.model_digest
        );

        // The session must not advance merely because a detached candidate exists.
        assert_eq!(session.next_batch_index, 0);
        assert!(matches!(
            accept_verified_sequential_candidate(
                &session,
                &template,
                &first_prepared,
                &first_native.candidate_native_model_digest,
                &digest('v'),
                false,
            ),
            Err(
                symthaea_core::sequential_weight_program::SequentialWeightProgramError::VerifierRejectedCandidate
            )
        ));
        assert_eq!(session.next_batch_index, 0);

        // Only a positive verifier attestation over the exact native candidate
        // advances the reusable edit program to the next evidence batch.
        let (session_after_first, first_receipt) = accept_verified_sequential_candidate(
            &session,
            &template,
            &first_prepared,
            &first_native.candidate_native_model_digest,
            &digest('v'),
            true,
        )
        .expect("accept verified first native candidate");
        assert_eq!(first_receipt.batch_index, 0);
        assert_eq!(
            session_after_first.latest_candidate_model_digest,
            first_native.candidate_native_model_digest
        );
        assert_eq!(session_after_first.next_batch_index, 1);

        let second_snapshot = cfc_weight_model_snapshot(&first_native.network, task_digest.clone())
            .expect("second parent snapshot");
        assert_eq!(
            second_snapshot.model_digest,
            session_after_first.latest_candidate_model_digest
        );
        let second_batch = SequentialEvidenceBatch {
            session_id,
            batch_index: 1,
            parent_candidate_model_digest: second_snapshot.model_digest.clone(),
            evidence_batch_digest: digest('g'),
            observation_manifest_digest: digest('p'),
        };
        let second_prepared = prepare_sequential_candidate(
            &session_after_first,
            &template,
            &second_snapshot,
            &second_batch,
        )
        .expect("prepare second evidence batch against latest candidate");
        let second_native = materialize_cfc_weight_program(
            &first_native.network,
            task_digest,
            &second_prepared.program,
        )
        .expect("materialize second native CfC candidate");
        let (session_after_second, second_receipt) = accept_verified_sequential_candidate(
            &session_after_first,
            &template,
            &second_prepared,
            &second_native.candidate_native_model_digest,
            &digest('w'),
            true,
        )
        .expect("accept verified second native candidate");

        assert_eq!(second_receipt.batch_index, 1);
        assert_eq!(session_after_second.next_batch_index, 2);
        assert_eq!(
            session_after_second.latest_candidate_model_digest,
            second_native.candidate_native_model_digest
        );
        assert_ne!(
            second_native.candidate_native_model_digest,
            first_native.candidate_native_model_digest
        );
        assert!(session_after_second.cumulative_l2_path > session_after_first.cumulative_l2_path);
        assert_eq!(
            session_after_second.cumulative_parameter_change_events,
            2 * tau_group.values.len()
        );
        assert_eq!(
            source
                .fingerprint()
                .expect("original source still immutable")
                .model_digest,
            initial_snapshot.model_digest
        );
    }

    #[test]
    fn gaussian_permutation_eig_selects_a_receipt_bound_probe() {
        let config = CfCNetworkConfig {
            input_dim: 2,
            hidden_dim: 3,
            num_layers: 1,
            output_dim: 2,
            cell_config: CfCConfig {
                input_dim: 2,
                hidden_dim: 3,
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
            "symthaea-cfc-gaussian-permutation-eig-v1",
        );
        let source = CfCNetwork::from_genesis(config, &genesis, "gaussian-permutation-eig");
        let candidate = source.clone();
        let probes = vec![
            CfCFunctionalProbe {
                input: vec![0.2, -0.8],
                dt: 0.05,
            },
            CfCFunctionalProbe {
                input: vec![0.9, 0.1],
                dt: 0.2,
            },
            CfCFunctionalProbe {
                input: vec![-0.6, 0.5],
                dt: 0.1,
            },
        ];

        let calibration = noise_calibration(&source, &candidate, 0, 'n');
        let receipt = select_cfc_disambiguating_probe_by_gaussian_eig(
            &source,
            &candidate,
            0,
            &[],
            &probes,
            &calibration,
            65,
        )
        .expect("small exact permutation EIG selection");

        assert_eq!(receipt.format_version, 1);
        assert_eq!(
            receipt.noise_calibration_digest,
            calibration.calibration_digest
        );
        assert_eq!(
            receipt.noise_calibration_method,
            CfCProjectionNoiseCalibrationMethod::RepeatedProjectionRuns
        );
        assert_eq!(receipt.noise_calibration_sample_count, 8);
        assert_eq!(receipt.observation_std_dev, calibration.observation_std_dev);
        assert_eq!(receipt.permutation_hypothesis_count, 6);
        assert_eq!(receipt.candidate_scores.len(), probes.len());
        assert_eq!(receipt.permutation_posterior.len(), 6);
        assert!(receipt.current_permutation_entropy_nats > 0.0);
        assert!(receipt.selected_model_information_gain_nats > 0.0);
        assert!(
            receipt.selected_expected_posterior_entropy_nats
                < receipt.current_permutation_entropy_nats
        );
        assert_eq!(
            receipt.selected_probe,
            probes[receipt.selected_candidate_index]
        );
        assert_eq!(
            receipt.selected_augmented_probe_digest,
            cfc_probe_set_digest(0, &[receipt.selected_probe.clone()])
        );
        let posterior_mass = receipt
            .permutation_posterior
            .iter()
            .map(|hypothesis| hypothesis.posterior_probability)
            .sum::<f64>();
        assert!((posterior_mass - 1.0).abs() < 1e-12);
        assert_eq!(
            source.fingerprint().expect("source digest").model_digest,
            receipt.source_model_digest
        );
        assert_eq!(
            candidate
                .fingerprint()
                .expect("candidate digest")
                .model_digest,
            receipt.candidate_model_digest
        );
    }

    #[test]
    fn gaussian_permutation_eig_updates_posterior_from_existing_probes() {
        let config = CfCNetworkConfig {
            input_dim: 2,
            hidden_dim: 3,
            num_layers: 1,
            output_dim: 2,
            cell_config: CfCConfig {
                input_dim: 2,
                hidden_dim: 3,
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
            "symthaea-cfc-gaussian-permutation-posterior-v1",
        );
        let source = CfCNetwork::from_genesis(config, &genesis, "gaussian-permutation-posterior");
        let candidate = source.clone();
        let existing = vec![
            CfCFunctionalProbe {
                input: vec![0.2, -0.8],
                dt: 0.05,
            },
            CfCFunctionalProbe {
                input: vec![0.9, 0.1],
                dt: 0.2,
            },
            CfCFunctionalProbe {
                input: vec![-0.6, 0.5],
                dt: 0.1,
            },
            CfCFunctionalProbe {
                input: vec![-0.1, -0.9],
                dt: 0.15,
            },
        ];
        let candidates = vec![
            CfCFunctionalProbe {
                input: vec![0.75, -0.25],
                dt: 0.12,
            },
            CfCFunctionalProbe {
                input: vec![-0.85, 0.3],
                dt: 0.08,
            },
        ];

        let calibration = noise_calibration(&source, &candidate, 0, 'p');
        let receipt = select_cfc_disambiguating_probe_by_gaussian_eig(
            &source,
            &candidate,
            0,
            &existing,
            &candidates,
            &calibration,
            65,
        )
        .expect("posterior update and EIG selection");

        assert!(receipt.existing_probe_digest.is_some());
        assert!(
            receipt.current_permutation_entropy_nats < (6.0_f64).ln(),
            "observed probes should reduce the initially uniform permutation entropy"
        );
        let posterior_max = receipt
            .permutation_posterior
            .iter()
            .map(|hypothesis| hypothesis.posterior_probability)
            .fold(0.0_f64, f64::max);
        assert!(posterior_max > 0.95);
    }

    #[test]
    fn projection_noise_calibration_rejects_degenerate_or_malformed_samples() {
        let config = CfCNetworkConfig {
            input_dim: 2,
            hidden_dim: 3,
            num_layers: 1,
            output_dim: 2,
            ..Default::default()
        };
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(
            "symthaea-cfc-noise-calibration-negative-controls",
        );
        let network = CfCNetwork::from_genesis(config, &genesis, "noise-calibration");
        assert!(matches!(
            calibrate_cfc_projection_observation_noise(
                &network,
                &network,
                0,
                &digest('n'),
                CfCProjectionNoiseCalibrationMethod::RepeatedProjectionRuns,
                &[0.2, 0.2, 0.2]
            ),
            Err(CfCPermutationAlignmentError::DegenerateNoiseCalibration)
        ));
        assert!(matches!(
            calibrate_cfc_projection_observation_noise(
                &network,
                &network,
                0,
                &digest('n'),
                CfCProjectionNoiseCalibrationMethod::RepeatedProjectionRuns,
                &[0.2]
            ),
            Err(CfCPermutationAlignmentError::InsufficientNoiseCalibrationSamples { .. })
        ));
        assert!(matches!(
            calibrate_cfc_projection_observation_noise(
                &network,
                &network,
                0,
                &digest('n'),
                CfCProjectionNoiseCalibrationMethod::RepeatedProjectionRuns,
                &[0.2, f64::NAN]
            ),
            Err(CfCPermutationAlignmentError::NonFiniteNoiseCalibrationSample { index: 1 })
        ));
    }

    #[test]
    fn gaussian_permutation_eig_rejects_noise_calibration_from_stale_candidate() {
        let config = CfCNetworkConfig {
            input_dim: 2,
            hidden_dim: 3,
            num_layers: 1,
            output_dim: 2,
            ..Default::default()
        };
        let source_genesis =
            symthaea_core::genesis::GenesisSeed::from_phrase("noise-calibration-source");
        let stale_genesis =
            symthaea_core::genesis::GenesisSeed::from_phrase("noise-calibration-stale-candidate");
        let source = CfCNetwork::from_genesis(config.clone(), &source_genesis, "source");
        let calibration_candidate = source.clone();
        let stale_candidate = CfCNetwork::from_genesis(config, &stale_genesis, "stale");
        let calibration = noise_calibration(&source, &calibration_candidate, 0, 's');
        let candidate_probes = vec![CfCFunctionalProbe {
            input: vec![0.2, -0.4],
            dt: 0.1,
        }];

        assert!(matches!(
            select_cfc_disambiguating_probe_by_gaussian_eig(
                &source,
                &stale_candidate,
                0,
                &[],
                &candidate_probes,
                &calibration,
                65
            ),
            Err(CfCPermutationAlignmentError::StaleNoiseCalibration {
                field: "candidate model digest"
            })
        ));
    }

    #[test]
    fn exact_permutation_eig_stops_before_factorial_allocation() {
        assert_eq!(exact_permutation_count(3), Some(6));
        assert_eq!(exact_permutation_count(5), Some(120));
        assert_eq!(exact_permutation_count(6), Some(720));
        assert_eq!(exact_permutation_count(0), None);
    }

    #[test]
    fn every_four_unit_permutation_round_trips_through_canonicalization() {
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
            "symthaea-swi-permutation-orbit-v1",
        );
        let source = CfCNetwork::from_genesis(config, &genesis, "permutation-orbit");
        let task_digest = digest('d');
        let snapshot =
            cfc_weight_model_snapshot(&source, task_digest.clone()).expect("source snapshot");
        let permutations = enumerate_hidden_unit_permutations(4);
        assert_eq!(permutations.len(), 24, "the complete S4 orbit must be covered");

        let get = |id: CfCWeightGroupId| {
            snapshot
                .groups
                .iter()
                .find(|group| group.target == WeightProgramTarget::CfC(id.clone()))
                .expect("parameter group")
                .values
                .clone()
        };
        let input_projection = get(CfCWeightGroupId::CellInputProjection { cell: 0 });
        let recurrent_projection = get(CfCWeightGroupId::CellRecurrentProjection { cell: 0 });
        let hidden_bias = get(CfCWeightGroupId::CellHiddenBias { cell: 0 });
        let time_constants = get(CfCWeightGroupId::CellTimeConstant { cell: 0 });
        let output_projection = get(CfCWeightGroupId::OutputProjection);

        let probes = vec![
            CfCFunctionalProbe {
                input: vec![0.25, -0.75],
                dt: 0.05,
            },
            CfCFunctionalProbe {
                input: vec![-0.4, 0.6],
                dt: 0.2,
            },
            CfCFunctionalProbe {
                input: vec![0.9, 0.1],
                dt: 0.1,
            },
            CfCFunctionalProbe {
                input: vec![-0.2, -0.8],
                dt: 0.15,
            },
            CfCFunctionalProbe {
                input: vec![0.71, -0.19],
                dt: 0.03,
            },
        ];
        let source_signatures =
            cfc_hidden_unit_activation_signatures(&source, 0, &probes).expect("source signatures");
        let source_digest = source
            .fingerprint()
            .expect("source fingerprint")
            .model_digest;

        let mut maximum_precanonical_function_error = 0.0_f64;
        let mut maximum_canonical_function_error = 0.0_f64;
        let mut verified_orbit_members = 0usize;

        for (permutation_index, permutation) in permutations.iter().enumerate() {
            let program = WeightProgram {
                format_version: 1,
                architecture_digest: snapshot.architecture_digest.clone(),
                manifest_digest: snapshot.manifest_digest.clone(),
                base_model_digest: snapshot.model_digest.clone(),
                target_specification_digest: task_digest.clone(),
                fault_hypothesis_ids: vec![
                    "hypothesis:function-preserving-hidden-unit-permutation".to_owned(),
                ],
                evidence_receipt_ids: vec![format!("receipt:permutation-orbit-{permutation_index}")],
                max_changed_parameters: 40,
                max_total_l2_delta: 100.0,
                max_abs_delta: 100.0,
                operations: vec![
                    WeightProgramOperation::ReplaceGroup {
                        target: WeightProgramTarget::CfC(
                            CfCWeightGroupId::CellInputProjection { cell: 0 },
                        ),
                        values: permute_rows(&input_projection, 4, 2, permutation),
                    },
                    WeightProgramOperation::ReplaceGroup {
                        target: WeightProgramTarget::CfC(
                            CfCWeightGroupId::CellRecurrentProjection { cell: 0 },
                        ),
                        values: permute_square(&recurrent_projection, 4, permutation),
                    },
                    WeightProgramOperation::ReplaceGroup {
                        target: WeightProgramTarget::CfC(
                            CfCWeightGroupId::CellHiddenBias { cell: 0 },
                        ),
                        values: permutation.iter().map(|index| hidden_bias[*index]).collect(),
                    },
                    WeightProgramOperation::ReplaceGroup {
                        target: WeightProgramTarget::CfC(
                            CfCWeightGroupId::CellTimeConstant { cell: 0 },
                        ),
                        values: permutation.iter().map(|index| time_constants[*index]).collect(),
                    },
                    WeightProgramOperation::ReplaceGroup {
                        target: WeightProgramTarget::CfC(CfCWeightGroupId::OutputProjection),
                        values: permute_columns(&output_projection, 2, 4, permutation),
                    },
                ],
            };
            let permuted_candidate = materialize_cfc_weight_program(
                &source,
                task_digest.clone(),
                &program,
            )
            .expect("function-preserving permutation candidate");

            // The raw parameter digest must distinguish a different coordinate
            // representation, even when the represented function is unchanged.
            if permutation.as_slice() != &[0, 1, 2, 3] {
                assert_ne!(
                    permuted_candidate.candidate_native_model_digest,
                    source_digest
                );
            }

            let candidate_signatures = cfc_hidden_unit_activation_signatures(
                &permuted_candidate.network,
                0,
                &probes,
            )
            .expect("candidate signatures");
            let alignment = align_cfc_hidden_unit_signatures(
                &source_signatures,
                &candidate_signatures,
            )
            .expect("permutation-orbit alignment");

            let mut expected_source_to_candidate = vec![0usize; 4];
            for (new_index, old_index) in permutation.iter().copied().enumerate() {
                expected_source_to_candidate[old_index] = new_index;
            }
            assert_eq!(
                alignment.source_to_candidate,
                expected_source_to_candidate,
                "permutation orbit member {permutation_index}: {permutation:?}"
            );
            assert!(
                alignment.mean_squared_activation_error < 1e-10,
                "permutation orbit member {permutation_index} alignment error {}",
                alignment.mean_squared_activation_error
            );
            assert!(
                alignment.minimum_assignment_margin.is_some_and(|margin| margin > 0.0),
                "permutation orbit member {permutation_index} must have a unique registered assignment"
            );

            let canonicalization = build_cfc_hidden_unit_canonicalization_program(
                &source,
                &permuted_candidate.network,
                task_digest.clone(),
                &alignment,
                CfCAlignmentAcceptancePolicy {
                    max_mean_squared_activation_error: 1e-10,
                    min_assignment_margin: 0.0,
                },
            )
            .expect("canonicalization program from orbit alignment");
            let canonicalized = materialize_cfc_weight_program(
                &permuted_candidate.network,
                task_digest.clone(),
                &canonicalization,
            )
            .expect("canonicalized orbit candidate");
            assert_eq!(
                canonicalized.candidate_native_model_digest,
                source_digest,
                "inverse transport must recover the exact source parameter representation"
            );

            let mut source_eval = source.clone();
            let mut permuted_eval = permuted_candidate.network.clone();
            let mut canonical_eval = canonicalized.network;
            for probe in &probes {
                let input = ndarray::Array1::from_vec(probe.input.clone());
                let expected = source_eval.forward(&input, probe.dt);
                let equivalent = permuted_eval.forward(&input, probe.dt);
                let restored = canonical_eval.forward(&input, probe.dt);
                for ((source_value, permuted_value), canonical_value) in
                    expected.iter().zip(equivalent.iter()).zip(restored.iter())
                {
                    maximum_precanonical_function_error = maximum_precanonical_function_error
                        .max(f64::from((*source_value - *permuted_value).abs()));
                    maximum_canonical_function_error = maximum_canonical_function_error
                        .max(f64::from((*source_value - *canonical_value).abs()));
                }
            }
            verified_orbit_members += 1;
        }

        assert_eq!(verified_orbit_members, 24);
        assert!(
            maximum_precanonical_function_error < 1e-5,
            "function-preserving orbit error {maximum_precanonical_function_error}"
        );
        assert!(
            maximum_canonical_function_error < 1e-5,
            "canonical transport function error {maximum_canonical_function_error}"
        );
        assert_eq!(
            source.fingerprint().expect("source remains immutable").model_digest,
            source_digest
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
