//! Fleet-robust, multi-device deployment envelopes for hardware-aware NAS.
//!
//! The evaluator refuses to average away a device failure. Every required target profile
//! must provide one correctly bound observation for every registered objective. Explicit
//! infeasibility is a hard failure; Failed/NotMeasured/missing evidence is incomplete and
//! cannot produce a fleet-feasible label.
//!
//! Objective normalization ranges, units and directions must be identical across target
//! registries. Fully measured candidates are aggregated by the minimum normalized utility
//! across the fleet, preserving each target's device-specific result and cross-device spread.
//!
//! This is a deterministic table-evidence evaluator. It does not guarantee generalization to
//! unseen boards, firmware, temperature, voltage, workloads or production device variation.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::hardware_objectives::{
    compute_hardware_profile_sha256, score_objective_observation,
    validate_hardware_profile, validate_objective_registry, HardwareProfileManifest,
    ObjectiveDirection, ObjectiveNormalization, ObjectiveObservation, ObjectiveRegistry,
    ScoredMeasurement,
};

const FLEET_SCHEMA: &str = "symthaea.nas.fleet-deployment-envelope";
const FLEET_VERSION: &str = "0.1.0";
const FLEET_DOMAIN: &[u8] = b"symthaea.nas.fleet-deployment-manifest.v0.1";
const EVIDENCE_DOMAIN: &[u8] = b"symthaea.nas.fleet-target-evidence.v0.1";
const EVALUATION_DOMAIN: &[u8] = b"symthaea.nas.fleet-candidate-evaluation.v0.1";
const DIGEST_HEX_LEN: usize = 64;
const FLOAT_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FleetTarget {
    /// Stable fleet slot ID; must equal the referenced HardwareProfileManifest.profile_id.
    pub target_id: String,
    pub hardware_profile_sha256: String,
    pub objective_registry_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FleetObjective {
    pub objective_id: String,
    pub value_unit: String,
    pub direction: ObjectiveDirection,
    pub normalization: ObjectiveNormalization,
    /// Minimum normalized utility required on every required target. Must be in [0,1].
    pub minimum_utility: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FleetDeploymentManifest {
    pub schema: String,
    pub version: String,
    pub fleet_id: String,
    pub benchmark_id: String,
    /// Complete set of required targets, strictly sorted by target_id.
    pub targets: Vec<FleetTarget>,
    /// Complete set of common objective coordinates, strictly sorted by objective_id.
    pub objectives: Vec<FleetObjective>,
    pub fleet_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FleetEvidenceDisposition {
    Measured,
    ExplicitlyInfeasible,
    NotMeasured,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FleetTargetObjectiveEvidence {
    pub target_id: String,
    pub hardware_profile_sha256: String,
    pub objective_id: String,
    pub disposition: FleetEvidenceDisposition,
    pub raw_value: Option<f64>,
    pub normalized_utility: Option<f64>,
    pub value_unit: Option<String>,
    pub acquisition_cost_unit: Option<String>,
    pub acquisition_cost_units: Option<u64>,
    pub replicate_id: Option<String>,
    pub sample_count: Option<usize>,
    pub source_artifact_sha256: Option<String>,
    pub reason_code: Option<String>,
    pub retryable: Option<bool>,
    pub evidence_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FleetEvidenceKey {
    pub target_id: String,
    pub hardware_profile_sha256: String,
    pub objective_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FleetObjectiveEnvelope {
    pub objective_id: String,
    pub value_unit: String,
    pub worst_target_id: String,
    pub worst_utility: f64,
    pub best_target_id: String,
    pub best_utility: f64,
    pub cross_target_spread: f64,
    pub required_minimum_utility: f64,
    pub satisfies_all_target_floor: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FleetCandidateStatus {
    FeasibleAllRequiredTargets,
    InfeasibleAtLeastOneRequiredTarget,
    IncompleteEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FleetConstraintViolation {
    pub target_id: String,
    pub objective_id: String,
    pub normalized_utility: f64,
    pub required_minimum_utility: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FleetCandidateEvaluation {
    pub schema: String,
    pub version: String,
    pub fleet_id: String,
    pub fleet_sha256: String,
    pub candidate_id: String,
    pub candidate_config_sha256: String,
    pub status: FleetCandidateStatus,
    /// One sorted evidence record per observed target/objective pair.
    pub evidence: Vec<FleetTargetObjectiveEvidence>,
    /// Explicitly absent when any required target/objective lacks a measured value.
    pub objective_envelopes: Option<Vec<FleetObjectiveEnvelope>>,
    pub missing_evidence: Vec<FleetEvidenceKey>,
    pub constraint_violations: Vec<FleetConstraintViolation>,
    pub infeasibility_reasons: Vec<String>,
    pub incomplete_reasons: Vec<String>,
    pub evidence_sha256: String,
    pub evaluation_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FleetEnvelopeError {
    pub field: String,
    pub message: String,
}

impl FleetEnvelopeError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for FleetEnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for FleetEnvelopeError {}

pub fn compute_fleet_deployment_manifest_sha256(
    manifest: &FleetDeploymentManifest,
) -> Result<String, FleetEnvelopeError> {
    validate_fleet_manifest_fields(manifest)?;
    let mut canonical = manifest.clone();
    canonical.fleet_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| FleetEnvelopeError::new("manifest", error.to_string()))?;
    Ok(digest_bytes(FLEET_DOMAIN, &bytes))
}

pub fn validate_fleet_deployment_manifest(
    manifest: &FleetDeploymentManifest,
) -> Result<(), FleetEnvelopeError> {
    validate_fleet_manifest_fields(manifest)?;
    if !is_sha256(&manifest.fleet_sha256)
        || compute_fleet_deployment_manifest_sha256(manifest)? != manifest.fleet_sha256
    {
        return Err(FleetEnvelopeError::new(
            "fleet_sha256",
            "fleet manifest digest does not match canonical fields",
        ));
    }
    Ok(())
}

fn validate_fleet_manifest_fields(
    manifest: &FleetDeploymentManifest,
) -> Result<(), FleetEnvelopeError> {
    if manifest.schema != FLEET_SCHEMA || manifest.version != FLEET_VERSION {
        return Err(FleetEnvelopeError::new(
            "schema_version",
            "unsupported fleet manifest schema/version",
        ));
    }
    if manifest.fleet_id.trim().is_empty() || manifest.benchmark_id.trim().is_empty() {
        return Err(FleetEnvelopeError::new(
            "identity",
            "fleet_id and benchmark_id must be non-empty",
        ));
    }
    if manifest.targets.is_empty() || manifest.objectives.is_empty() {
        return Err(FleetEnvelopeError::new(
            "manifest",
            "at least one target and one objective are required",
        ));
    }
    let mut target_ids = HashSet::with_capacity(manifest.targets.len());
    let mut profile_ids = HashSet::with_capacity(manifest.targets.len());
    let mut previous_target: Option<&str> = None;
    for (index, target) in manifest.targets.iter().enumerate() {
        if target.target_id.trim().is_empty()
            || previous_target.is_some_and(|previous| previous >= target.target_id.as_str())
            || !target_ids.insert(target.target_id.as_str())
        {
            return Err(FleetEnvelopeError::new(
                format!("targets[{index}].target_id"),
                "target IDs must be non-empty, unique, and strictly sorted",
            ));
        }
        previous_target = Some(&target.target_id);
        validate_digest(
            &format!("targets[{index}].hardware_profile_sha256"),
            &target.hardware_profile_sha256,
        )?;
        validate_digest(
            &format!("targets[{index}].objective_registry_sha256"),
            &target.objective_registry_sha256,
        )?;
        if !profile_ids.insert(target.hardware_profile_sha256.as_str()) {
            return Err(FleetEnvelopeError::new(
                format!("targets[{index}].hardware_profile_sha256"),
                "a hardware profile may appear only once in the required fleet",
            ));
        }
    }

    let mut previous_objective: Option<&str> = None;
    let mut objective_ids = HashSet::with_capacity(manifest.objectives.len());
    for (index, objective) in manifest.objectives.iter().enumerate() {
        if objective.objective_id.trim().is_empty()
            || objective.value_unit.trim().is_empty()
            || previous_objective.is_some_and(|previous| previous >= objective.objective_id.as_str())
            || !objective_ids.insert(objective.objective_id.as_str())
        {
            return Err(FleetEnvelopeError::new(
                format!("objectives[{index}]"),
                "objective IDs must be non-empty, unique and strictly sorted; units must be non-empty",
            ));
        }
        previous_objective = Some(&objective.objective_id);
        let minimum = objective.normalization.raw_min;
        let maximum = objective.normalization.raw_max;
        if !minimum.is_finite() || !maximum.is_finite() || maximum <= minimum {
            return Err(FleetEnvelopeError::new(
                format!("objectives[{index}].normalization"),
                "normalization bounds must be finite and strictly increasing",
            ));
        }
        if !objective.minimum_utility.is_finite()
            || !(0.0..=1.0).contains(&objective.minimum_utility)
        {
            return Err(FleetEnvelopeError::new(
                format!("objectives[{index}].minimum_utility"),
                "hard minimum utility must be finite and in [0,1]",
            ));
        }
    }
    Ok(())
}

/// Validate all per-target objective registries against one fleet manifest.
pub fn validate_fleet_objective_registries(
    manifest: &FleetDeploymentManifest,
    registries: &[ObjectiveRegistry],
) -> Result<(), FleetEnvelopeError> {
    validate_fleet_deployment_manifest(manifest)?;
    if registries.len() != manifest.targets.len() {
        return Err(FleetEnvelopeError::new(
            "registries",
            "exactly one objective registry is required for every required fleet target",
        ));
    }
    let mut by_digest: HashMap<&str, &ObjectiveRegistry> = HashMap::with_capacity(registries.len());
    for (index, registry) in registries.iter().enumerate() {
        validate_objective_registry(registry)
            .map_err(|error| FleetEnvelopeError::new(format!("registries[{index}]"), error.to_string()))?;
        if by_digest.insert(&registry.registry_sha256, registry).is_some() {
            return Err(FleetEnvelopeError::new(
                format!("registries[{index}].registry_sha256"),
                "duplicate objective registry digest",
            ));
        }
    }

    let mut common_definitions: Option<BTreeMap<&str, (&str, ObjectiveDirection, f64, f64)>> = None;
    for (target_index, target) in manifest.targets.iter().enumerate() {
        let registry = by_digest.get(target.objective_registry_sha256.as_str()).ok_or_else(|| {
            FleetEnvelopeError::new(
                format!("targets[{target_index}].objective_registry_sha256"),
                "required target objective registry is missing",
            )
        })?;
        if registry.benchmark_id != manifest.benchmark_id {
            return Err(FleetEnvelopeError::new(
                format!("targets[{target_index}].benchmark_id"),
                "target registry belongs to a different benchmark",
            ));
        }
        let profile = registry
            .hardware_profiles
            .iter()
            .find(|profile| profile.profile_sha256 == target.hardware_profile_sha256)
            .ok_or_else(|| {
                FleetEnvelopeError::new(
                    format!("targets[{target_index}].hardware_profile_sha256"),
                    "target registry does not include the exact hardware profile",
                )
            })?;
        validate_hardware_profile(profile)
            .map_err(|error| FleetEnvelopeError::new("hardware_profile", error.to_string()))?;
        if compute_hardware_profile_sha256(profile)
            .map_err(|error| FleetEnvelopeError::new("hardware_profile", error.to_string()))?
            != target.hardware_profile_sha256
            || profile.profile_id != target.target_id
        {
            return Err(FleetEnvelopeError::new(
                format!("targets[{target_index}].hardware_profile_sha256"),
                "profile digest or target/profile ID does not match the frozen fleet manifest",
            ));
        }

        let mut definitions = BTreeMap::new();
        for objective in &manifest.objectives {
            let definition = registry
                .objectives
                .iter()
                .find(|definition| definition.objective_id == objective.objective_id)
                .ok_or_else(|| {
                    FleetEnvelopeError::new(
                        format!("targets[{target_index}].objectives"),
                        format!("objective {:?} is absent from target registry", objective.objective_id),
                    )
                })?;
            if definition.value_unit != objective.value_unit
                || definition.direction != objective.direction
                || !approximately_equal(definition.normalization.raw_min, objective.normalization.raw_min)
                || !approximately_equal(definition.normalization.raw_max, objective.normalization.raw_max)
                || definition.hardware_profile_sha256.as_deref()
                    != Some(target.hardware_profile_sha256.as_str())
            {
                return Err(FleetEnvelopeError::new(
                    format!("targets[{target_index}].objectives.{}", objective.objective_id),
                    "unit, direction, normalization or hardware-profile scope differs from fleet manifest",
                ));
            }
            definitions.insert(
                objective.objective_id.as_str(),
                (
                    definition.value_unit.as_str(),
                    definition.direction,
                    definition.normalization.raw_min,
                    definition.normalization.raw_max,
                ),
            );
        }
        if definitions.len() != registry.objectives.len() {
            return Err(FleetEnvelopeError::new(
                format!("targets[{target_index}].objectives"),
                "each target registry must contain exactly the fleet's declared objective set",
            ));
        }
        match &common_definitions {
            None => common_definitions = Some(definitions),
            Some(common) if common == &definitions => {}
            Some(_) => {
                return Err(FleetEnvelopeError::new(
                    format!("targets[{target_index}].objectives"),
                    "objective definitions are not comparable across the fleet",
                ));
            }
        }
    }
    Ok(())
}

/// Evaluate one candidate against every required device. No averaging or missing-value
/// imputation is used: the robust objective is the worst normalized utility across all
/// required fleet targets.
pub fn evaluate_fleet_candidate(
    manifest: &FleetDeploymentManifest,
    registries: &[ObjectiveRegistry],
    candidate_id: &str,
    candidate_config_sha256: &str,
    observations: &[ObjectiveObservation],
) -> Result<FleetCandidateEvaluation, FleetEnvelopeError> {
    validate_fleet_objective_registries(manifest, registries)?;
    if candidate_id.trim().is_empty() {
        return Err(FleetEnvelopeError::new("candidate_id", "must be non-empty"));
    }
    validate_digest("candidate_config_sha256", candidate_config_sha256)?;

    let registry_by_digest: HashMap<&str, &ObjectiveRegistry> = registries
        .iter()
        .map(|registry| (registry.registry_sha256.as_str(), registry))
        .collect();
    let target_by_profile: HashMap<&str, &FleetTarget> = manifest
        .targets
        .iter()
        .map(|target| (target.hardware_profile_sha256.as_str(), target))
        .collect();

    let mut seen = HashSet::with_capacity(observations.len());
    let mut evidence = Vec::with_capacity(observations.len());
    for (index, observation) in observations.iter().enumerate() {
        if observation.candidate_id != candidate_id {
            return Err(FleetEnvelopeError::new(
                format!("observations[{index}].candidate_id"),
                "observation belongs to a different candidate",
            ));
        }
        if observation.candidate_config_sha256 != candidate_config_sha256 {
            return Err(FleetEnvelopeError::new(
                format!("observations[{index}].candidate_config_sha256"),
                "candidate configuration differs across fleet evidence",
            ));
        }
        let profile_sha = observation.hardware_profile_sha256.as_deref().ok_or_else(|| {
            FleetEnvelopeError::new(
                format!("observations[{index}].hardware_profile_sha256"),
                "fleet evidence must identify an exact hardware profile",
            )
        })?;
        let target = target_by_profile.get(profile_sha).ok_or_else(|| {
            FleetEnvelopeError::new(
                format!("observations[{index}].hardware_profile_sha256"),
                "observation belongs to a profile outside the required fleet",
            )
        })?;
        if observation.registry_sha256 != target.objective_registry_sha256 {
            return Err(FleetEnvelopeError::new(
                format!("observations[{index}].registry_sha256"),
                "observation registry does not match its target profile registry",
            ));
        }
        let registry = registry_by_digest
            .get(target.objective_registry_sha256.as_str())
            .ok_or_else(|| FleetEnvelopeError::new("registries", "target registry disappeared"))?;
        let key = (target.target_id.as_str(), observation.objective_id.as_str());
        if !seen.insert(key) {
            return Err(FleetEnvelopeError::new(
                format!("observations[{index}]"),
                "duplicate target/objective observation",
            ));
        }
        let scored = score_objective_observation(registry, observation)
            .map_err(|error| FleetEnvelopeError::new(format!("observations[{index}]"), error.to_string()))?;
        evidence.push(evidence_from_scored(target, &observation.objective_id, scored));
    }

    evidence.sort_by(|left, right| {
        left.target_id
            .cmp(&right.target_id)
            .then_with(|| left.objective_id.cmp(&right.objective_id))
    });

    let mut missing_evidence = Vec::new();
    let mut infeasibility_reasons = Vec::new();
    let mut incomplete_reasons = Vec::new();
    let mut value_by_objective: BTreeMap<&str, Vec<(&str, f64)>> = BTreeMap::new();

    for target in &manifest.targets {
        for objective in &manifest.objectives {
            let item = evidence.iter().find(|item| {
                item.target_id == target.target_id && item.objective_id == objective.objective_id
            });
            match item {
                None => missing_evidence.push(FleetEvidenceKey {
                    target_id: target.target_id.clone(),
                    hardware_profile_sha256: target.hardware_profile_sha256.clone(),
                    objective_id: objective.objective_id.clone(),
                }),
                Some(item) => match item.disposition {
                    FleetEvidenceDisposition::Measured => {
                        let utility = item.normalized_utility.ok_or_else(|| {
                            FleetEnvelopeError::new("evidence", "measured evidence is missing normalized utility")
                        })?;
                        value_by_objective
                            .entry(&objective.objective_id)
                            .or_default()
                            .push((&target.target_id, utility));
                    }
                    FleetEvidenceDisposition::ExplicitlyInfeasible => {
                        infeasibility_reasons.push(format!(
                            "{} / {}: {}",
                            target.target_id,
                            objective.objective_id,
                            item.reason_code.as_deref().unwrap_or("explicitly_infeasible")
                        ));
                    }
                    FleetEvidenceDisposition::NotMeasured => incomplete_reasons.push(format!(
                        "{} / {}: not_measured ({})",
                        target.target_id,
                        objective.objective_id,
                        item.reason_code.as_deref().unwrap_or("unknown")
                    )),
                    FleetEvidenceDisposition::Failed => incomplete_reasons.push(format!(
                        "{} / {}: failed ({})",
                        target.target_id,
                        objective.objective_id,
                        item.reason_code.as_deref().unwrap_or("unknown")
                    )),
                },
            }
        }
    }

    let has_incomplete = !missing_evidence.is_empty() || !incomplete_reasons.is_empty();
    let all_measured = !has_incomplete && infeasibility_reasons.is_empty();
    // A known hard-threshold violation is conclusive even if a different target's
    // evidence is still missing. Unknown outcomes never erase known infeasibility.
    let mut constraint_violations = Vec::new();
    for item in &evidence {
        if item.disposition != FleetEvidenceDisposition::Measured {
            continue;
        }
        let objective = manifest
            .objectives
            .iter()
            .find(|objective| objective.objective_id == item.objective_id)
            .ok_or_else(|| FleetEnvelopeError::new("evidence.objective_id", "objective absent from manifest"))?;
        let utility = item.normalized_utility.ok_or_else(|| {
            FleetEnvelopeError::new("evidence.normalized_utility", "measured evidence lacks normalized utility")
        })?;
        if utility + FLOAT_TOLERANCE < objective.minimum_utility {
            constraint_violations.push(FleetConstraintViolation {
                target_id: item.target_id.clone(),
                objective_id: item.objective_id.clone(),
                normalized_utility: utility,
                required_minimum_utility: objective.minimum_utility,
            });
        }
    }
    let objective_envelopes = if all_measured {
        let mut envelopes = Vec::with_capacity(manifest.objectives.len());
        for objective in &manifest.objectives {
            let values = value_by_objective
                .get(objective.objective_id.as_str())
                .ok_or_else(|| FleetEnvelopeError::new("objective_envelope", "missing objective values"))?;
            if values.len() != manifest.targets.len() {
                return Err(FleetEnvelopeError::new(
                    "objective_envelope",
                    "every fleet objective must have one measured value on each target",
                ));
            }
            let (worst_target, worst_utility) = values
                .iter()
                .min_by(|left, right| left.1.total_cmp(&right.1))
                .copied()
                .ok_or_else(|| FleetEnvelopeError::new("objective_envelope", "empty target values"))?;
            let (best_target, best_utility) = values
                .iter()
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .copied()
                .ok_or_else(|| FleetEnvelopeError::new("objective_envelope", "empty target values"))?;
            let satisfies = worst_utility + FLOAT_TOLERANCE >= objective.minimum_utility;
            envelopes.push(FleetObjectiveEnvelope {
                objective_id: objective.objective_id.clone(),
                value_unit: objective.value_unit.clone(),
                worst_target_id: worst_target.to_owned(),
                worst_utility,
                best_target_id: best_target.to_owned(),
                best_utility,
                cross_target_spread: best_utility - worst_utility,
                required_minimum_utility: objective.minimum_utility,
                satisfies_all_target_floor: satisfies,
            });
        }
        Some(envelopes)
    } else {
        None
    };

    let status = if !infeasibility_reasons.is_empty() || !constraint_violations.is_empty() {
        FleetCandidateStatus::InfeasibleAtLeastOneRequiredTarget
    } else if has_incomplete {
        FleetCandidateStatus::IncompleteEvidence
    } else {
        FleetCandidateStatus::FeasibleAllRequiredTargets
    };

    let evidence_sha256 = compute_fleet_evidence_sha256(&evidence)?;
    let mut evaluation = FleetCandidateEvaluation {
        schema: "symthaea.nas.fleet-candidate-evaluation".to_owned(),
        version: FLEET_VERSION.to_owned(),
        fleet_id: manifest.fleet_id.clone(),
        fleet_sha256: manifest.fleet_sha256.clone(),
        candidate_id: candidate_id.to_owned(),
        candidate_config_sha256: candidate_config_sha256.to_owned(),
        status,
        evidence,
        objective_envelopes,
        missing_evidence,
        constraint_violations,
        infeasibility_reasons,
        incomplete_reasons,
        evidence_sha256,
        evaluation_sha256: String::new(),
    };
    evaluation.evaluation_sha256 = compute_evaluation_sha256(&evaluation)?;
    Ok(evaluation)
}

/// Recompute an evaluation from the original observations; never accept a caller-built
/// aggregate status as evidence.
pub fn verify_fleet_candidate_evaluation(
    manifest: &FleetDeploymentManifest,
    registries: &[ObjectiveRegistry],
    candidate_id: &str,
    candidate_config_sha256: &str,
    observations: &[ObjectiveObservation],
    supplied: &FleetCandidateEvaluation,
) -> Result<(), FleetEnvelopeError> {
    let recomputed = evaluate_fleet_candidate(
        manifest,
        registries,
        candidate_id,
        candidate_config_sha256,
        observations,
    )?;
    if &recomputed != supplied {
        return Err(FleetEnvelopeError::new(
            "evaluation",
            "supplied fleet evaluation differs from recomputed observations and manifest",
        ));
    }
    Ok(())
}

fn evidence_from_scored(
    target: &FleetTarget,
    objective_id: &str,
    scored: ScoredMeasurement,
) -> FleetTargetObjectiveEvidence {
    match scored {
        ScoredMeasurement::Measured {
            raw_value,
            normalized_utility,
            value_unit,
            acquisition_cost,
            replicate_id,
            sample_count,
            source_artifact_sha256,
        } => FleetTargetObjectiveEvidence {
            target_id: target.target_id.clone(),
            hardware_profile_sha256: target.hardware_profile_sha256.clone(),
            objective_id: objective_id.to_owned(),
            disposition: FleetEvidenceDisposition::Measured,
            raw_value: Some(raw_value),
            normalized_utility: Some(normalized_utility),
            value_unit: Some(value_unit),
            acquisition_cost_unit: Some(acquisition_cost.resource_unit),
            acquisition_cost_units: Some(acquisition_cost.units),
            replicate_id: Some(replicate_id),
            sample_count: Some(sample_count),
            source_artifact_sha256: Some(source_artifact_sha256),
            reason_code: None,
            retryable: None,
            evidence_sha256: None,
        },
        ScoredMeasurement::Infeasible { reason_code, source_artifact_sha256 } => {
            FleetTargetObjectiveEvidence {
                target_id: target.target_id.clone(),
                hardware_profile_sha256: target.hardware_profile_sha256.clone(),
                objective_id: objective_id.to_owned(),
                disposition: FleetEvidenceDisposition::ExplicitlyInfeasible,
                raw_value: None,
                normalized_utility: None,
                value_unit: None,
                acquisition_cost_unit: None,
                acquisition_cost_units: None,
                replicate_id: None,
                sample_count: None,
                source_artifact_sha256: Some(source_artifact_sha256),
                reason_code: Some(reason_code),
                retryable: None,
                evidence_sha256: None,
            }
        }
        ScoredMeasurement::NotMeasured { reason_code } => FleetTargetObjectiveEvidence {
            target_id: target.target_id.clone(),
            hardware_profile_sha256: target.hardware_profile_sha256.clone(),
            objective_id: objective_id.to_owned(),
            disposition: FleetEvidenceDisposition::NotMeasured,
            raw_value: None,
            normalized_utility: None,
            value_unit: None,
            acquisition_cost_unit: None,
            acquisition_cost_units: None,
            replicate_id: None,
            sample_count: None,
            source_artifact_sha256: None,
            reason_code: Some(reason_code),
            retryable: None,
            evidence_sha256: None,
        },
        ScoredMeasurement::Failed { reason_code, retryable, evidence_sha256 } => {
            FleetTargetObjectiveEvidence {
                target_id: target.target_id.clone(),
                hardware_profile_sha256: target.hardware_profile_sha256.clone(),
                objective_id: objective_id.to_owned(),
                disposition: FleetEvidenceDisposition::Failed,
                raw_value: None,
                normalized_utility: None,
                value_unit: None,
                acquisition_cost_unit: None,
                acquisition_cost_units: None,
                replicate_id: None,
                sample_count: None,
                source_artifact_sha256: None,
                reason_code: Some(reason_code),
                retryable: Some(retryable),
                evidence_sha256: Some(evidence_sha256),
            }
        }
    }
}

fn compute_fleet_evidence_sha256(
    evidence: &[FleetTargetObjectiveEvidence],
) -> Result<String, FleetEnvelopeError> {
    let bytes = serde_json::to_vec(evidence)
        .map_err(|error| FleetEnvelopeError::new("evidence", error.to_string()))?;
    Ok(digest_bytes(EVIDENCE_DOMAIN, &bytes))
}

fn compute_evaluation_sha256(
    evaluation: &FleetCandidateEvaluation,
) -> Result<String, FleetEnvelopeError> {
    let mut canonical = evaluation.clone();
    canonical.evaluation_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| FleetEnvelopeError::new("evaluation", error.to_string()))?;
    Ok(digest_bytes(EVALUATION_DOMAIN, &bytes))
}

pub fn verify_fleet_candidate_evaluation_digest(
    evaluation: &FleetCandidateEvaluation,
) -> Result<(), FleetEnvelopeError> {
    if !is_sha256(&evaluation.evidence_sha256)
        || compute_fleet_evidence_sha256(&evaluation.evidence)? != evaluation.evidence_sha256
        || !is_sha256(&evaluation.evaluation_sha256)
        || compute_evaluation_sha256(evaluation)? != evaluation.evaluation_sha256
    {
        return Err(FleetEnvelopeError::new(
            "evaluation_sha256",
            "fleet evidence or evaluation digest does not recompute",
        ));
    }
    Ok(())
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    encode_sha256(hasher.finalize().as_slice())
}

fn validate_digest(field: &str, value: &str) -> Result<(), FleetEnvelopeError> {
    if !is_sha256(value) {
        return Err(FleetEnvelopeError::new(
            field,
            "must be a lowercase hexadecimal SHA-256 digest",
        ));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == DIGEST_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn approximately_equal(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= FLOAT_TOLERANCE * left.abs().max(right.abs()).max(1.0)
}

fn encode_sha256(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware_objectives::{
        compute_hardware_profile_sha256, compute_objective_registry_sha256,
        HardwareProfileManifest, MeasurementCost, MeasurementOutcome, ObjectiveDefinition,
        ObjectiveSource,
    };

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn profile(target_id: &str, marker: char) -> HardwareProfileManifest {
        let mut value = HardwareProfileManifest {
            profile_id: target_id.into(),
            device_model: format!("MCU-{target_id}"),
            board_id: format!("board-{target_id}"),
            board_revision: "rev-a".into(),
            firmware_sha256: sha(marker),
            runtime_sha256: sha('b'),
            compiler_toolchain_sha256: sha('c'),
            quantization_spec_sha256: sha('d'),
            measurement_protocol_sha256: sha(marker),
            instrument_model: format!("instrument-{target_id}"),
            instrument_id: format!("instrument-id-{target_id}"),
            sampling_frequency_hz: 100_000,
            profile_sha256: String::new(),
        };
        value.profile_sha256 = compute_hardware_profile_sha256(&value).unwrap();
        value
    }

    fn registry(target: &HardwareProfileManifest) -> ObjectiveRegistry {
        let mut value = ObjectiveRegistry {
            schema: "symthaea.nas.objective-registry".into(),
            version: "0.1.0".into(),
            benchmark_id: "fleet-fixture".into(),
            objectives: vec![
                ObjectiveDefinition {
                    objective_id: "accuracy".into(),
                    value_unit: "fraction".into(),
                    direction: ObjectiveDirection::Maximize,
                    source: ObjectiveSource::PhysicalDeviceMeasurement,
                    normalization: ObjectiveNormalization { raw_min: 0.0, raw_max: 1.0 },
                    measurement_protocol_sha256: target.measurement_protocol_sha256.clone(),
                    hardware_profile_sha256: Some(target.profile_sha256.clone()),
                    measurement_cost_unit: "query".into(),
                    measurement_cost_model_sha256: sha('e'),
                },
                ObjectiveDefinition {
                    objective_id: "latency".into(),
                    value_unit: "milliseconds".into(),
                    direction: ObjectiveDirection::Minimize,
                    source: ObjectiveSource::PhysicalDeviceMeasurement,
                    normalization: ObjectiveNormalization { raw_min: 0.0, raw_max: 100.0 },
                    measurement_protocol_sha256: target.measurement_protocol_sha256.clone(),
                    hardware_profile_sha256: Some(target.profile_sha256.clone()),
                    measurement_cost_unit: "query".into(),
                    measurement_cost_model_sha256: sha('f'),
                },
            ],
            hardware_profiles: vec![target.clone()],
            registry_sha256: String::new(),
        };
        value.registry_sha256 = compute_objective_registry_sha256(&value).unwrap();
        value
    }

    fn manifest(
        profiles: &[HardwareProfileManifest],
        registries: &[ObjectiveRegistry],
        accuracy_floor: f64,
        latency_utility_floor: f64,
    ) -> FleetDeploymentManifest {
        let mut value = FleetDeploymentManifest {
            schema: FLEET_SCHEMA.into(),
            version: FLEET_VERSION.into(),
            fleet_id: "mcu-fleet-a".into(),
            benchmark_id: "fleet-fixture".into(),
            targets: profiles
                .iter()
                .zip(registries)
                .map(|(profile, registry)| FleetTarget {
                    target_id: profile.profile_id.clone(),
                    hardware_profile_sha256: profile.profile_sha256.clone(),
                    objective_registry_sha256: registry.registry_sha256.clone(),
                })
                .collect(),
            objectives: vec![
                FleetObjective {
                    objective_id: "accuracy".into(),
                    value_unit: "fraction".into(),
                    direction: ObjectiveDirection::Maximize,
                    normalization: ObjectiveNormalization { raw_min: 0.0, raw_max: 1.0 },
                    minimum_utility: accuracy_floor,
                },
                FleetObjective {
                    objective_id: "latency".into(),
                    value_unit: "milliseconds".into(),
                    direction: ObjectiveDirection::Minimize,
                    normalization: ObjectiveNormalization { raw_min: 0.0, raw_max: 100.0 },
                    minimum_utility: latency_utility_floor,
                },
            ],
            fleet_sha256: String::new(),
        };
        value.targets.sort_by(|left, right| left.target_id.cmp(&right.target_id));
        value.fleet_sha256 = compute_fleet_deployment_manifest_sha256(&value).unwrap();
        value
    }

    fn measured(
        candidate_id: &str,
        config: &str,
        target: &HardwareProfileManifest,
        registry: &ObjectiveRegistry,
        objective_id: &str,
        raw_value: f64,
        sequence: u64,
    ) -> ObjectiveObservation {
        let definition = registry.objectives.iter().find(|item| item.objective_id == objective_id).unwrap();
        ObjectiveObservation {
            candidate_id: candidate_id.into(),
            candidate_config_sha256: config.into(),
            objective_id: objective_id.into(),
            registry_sha256: registry.registry_sha256.clone(),
            measurement_protocol_sha256: definition.measurement_protocol_sha256.clone(),
            hardware_profile_sha256: Some(target.profile_sha256.clone()),
            outcome: MeasurementOutcome::Measured {
                raw_value_bits: raw_value.to_bits(),
                replicate_id: format!("rep-{sequence}"),
                sample_count: 10,
                source_artifact_sha256: sha('9'),
                cost: MeasurementCost {
                    resource_unit: "query".into(),
                    units: 1,
                },
            },
        }
    }

    fn fixture() -> (Vec<HardwareProfileManifest>, Vec<ObjectiveRegistry>, FleetDeploymentManifest, String) {
        let profiles = vec![profile("nrf52840", '1'), profile("nrf5340", '2')];
        let registries = profiles.iter().map(registry).collect::<Vec<_>>();
        let manifest = manifest(&profiles, &registries, 0.8, 0.6);
        (profiles, registries, manifest, sha('a'))
    }

    fn complete_observations(
        profiles: &[HardwareProfileManifest],
        registries: &[ObjectiveRegistry],
        accuracy: [f64; 2],
        latency: [f64; 2],
    ) -> Vec<ObjectiveObservation> {
        let mut rows = Vec::new();
        for index in 0..profiles.len() {
            rows.push(measured("candidate-A", &sha('a'), &profiles[index], &registries[index], "accuracy", accuracy[index], index as u64));
            rows.push(measured("candidate-A", &sha('a'), &profiles[index], &registries[index], "latency", latency[index], (index + 10) as u64));
        }
        rows
    }

    #[test]
    fn worst_device_utility_not_average_controls_fleet_feasibility() {
        let (profiles, registries, mut manifest, config) = fixture();
        // Mean latency utility is 0.725, but the worst target is 0.65 < 0.70.
        manifest.objectives[1].minimum_utility = 0.70;
        manifest.fleet_sha256 =
            compute_fleet_deployment_manifest_sha256(&manifest).unwrap();
        let observations = complete_observations(
            &profiles,
            &registries,
            [0.99, 0.81],
            [20.0, 35.0],
        );
        let result = evaluate_fleet_candidate(
            &manifest, &registries, "candidate-A", &config, &observations,
        ).unwrap();
        assert_eq!(result.status, FleetCandidateStatus::InfeasibleAtLeastOneRequiredTarget);
        let latency = result.objective_envelopes.as_ref().unwrap().iter()
            .find(|item| item.objective_id == "latency").unwrap();
        assert!((latency.worst_utility - 0.65).abs() < 1e-12);
        assert!((latency.cross_target_spread - 0.15).abs() < 1e-12);
        assert_eq!(result.constraint_violations.len(), 1);
        assert_eq!(result.constraint_violations[0].target_id, "nrf5340");
    }

    #[test]
    fn one_target_below_a_hard_floor_makes_the_fleet_infeasible() {
        let (profiles, registries, mut manifest, config) = fixture();
        manifest.objectives[0].minimum_utility = 0.85;
        manifest.fleet_sha256 = compute_fleet_deployment_manifest_sha256(&manifest).unwrap();
        let observations = complete_observations(
            &profiles, &registries, [0.99, 0.81], [20.0, 25.0],
        );
        let result = evaluate_fleet_candidate(
            &manifest, &registries, "candidate-A", &config, &observations,
        ).unwrap();
        assert_eq!(result.status, FleetCandidateStatus::InfeasibleAtLeastOneRequiredTarget);
        assert_eq!(result.constraint_violations.len(), 1);
        assert_eq!(result.constraint_violations[0].target_id, "nrf5340");
    }

    #[test]
    fn missing_required_target_evidence_is_incomplete_not_feasible() {
        let (profiles, registries, manifest, config) = fixture();
        let observations = complete_observations(
            &profiles[..1], &registries[..1], [0.95, 0.95], [20.0, 20.0],
        );
        let result = evaluate_fleet_candidate(
            &manifest, &registries, "candidate-A", &config, &observations,
        ).unwrap();
        assert_eq!(result.status, FleetCandidateStatus::IncompleteEvidence);
        assert_eq!(result.missing_evidence.len(), 2);
        assert!(result.objective_envelopes.is_none());
    }

    #[test]
    fn explicit_hardware_infeasibility_is_not_imputed_as_zero_or_missing() {
        let (profiles, registries, manifest, config) = fixture();
        let mut observations = complete_observations(
            &profiles[..1], &registries[..1], [0.95, 0.95], [20.0, 20.0],
        );
        observations.push(ObjectiveObservation {
            candidate_id: "candidate-A".into(),
            candidate_config_sha256: config.clone(),
            objective_id: "accuracy".into(),
            registry_sha256: registries[1].registry_sha256.clone(),
            measurement_protocol_sha256: registries[1].objectives[0].measurement_protocol_sha256.clone(),
            hardware_profile_sha256: Some(profiles[1].profile_sha256.clone()),
            outcome: MeasurementOutcome::Infeasible {
                reason_code: "flash_budget_exceeded".into(),
                source_artifact_sha256: sha('8'),
            },
        });
        let result = evaluate_fleet_candidate(
            &manifest, &registries, "candidate-A", &config, &observations,
        ).unwrap();
        assert_eq!(result.status, FleetCandidateStatus::InfeasibleAtLeastOneRequiredTarget);
        assert!(result.infeasibility_reasons[0].contains("flash_budget_exceeded"));
    }

    #[test]
    fn known_floor_violation_dominates_unrelated_incomplete_evidence() {
        let (profiles, registries, mut manifest, config) = fixture();
        manifest.objectives[0].minimum_utility = 0.80;
        manifest.fleet_sha256 =
            compute_fleet_deployment_manifest_sha256(&manifest).unwrap();
        let observations = complete_observations(
            &profiles[..1], &registries[..1], [0.40, 0.95], [20.0, 20.0],
        );
        let result = evaluate_fleet_candidate(
            &manifest, &registries, "candidate-A", &config, &observations,
        ).unwrap();
        assert_eq!(result.status, FleetCandidateStatus::InfeasibleAtLeastOneRequiredTarget);
        assert_eq!(result.missing_evidence.len(), 2);
        assert_eq!(result.constraint_violations.len(), 1);
        assert_eq!(result.constraint_violations[0].target_id, "nrf52840");
    }

    #[test]
    fn not_measured_and_failed_observations_remain_incomplete() {
        let (profiles, registries, manifest, config) = fixture();
        let mut observations = complete_observations(
            &profiles[..1], &registries[..1], [0.95, 0.95], [20.0, 20.0],
        );
        observations.push(ObjectiveObservation {
            candidate_id: "candidate-A".into(),
            candidate_config_sha256: config.clone(),
            objective_id: "accuracy".into(),
            registry_sha256: registries[1].registry_sha256.clone(),
            measurement_protocol_sha256: registries[1].objectives[0].measurement_protocol_sha256.clone(),
            hardware_profile_sha256: Some(profiles[1].profile_sha256.clone()),
            outcome: MeasurementOutcome::Failed {
                reason_code: "instrument_timeout".into(),
                retryable: true,
                evidence_sha256: sha('7'),
            },
        });
        observations.push(ObjectiveObservation {
            candidate_id: "candidate-A".into(),
            candidate_config_sha256: config.clone(),
            objective_id: "latency".into(),
            registry_sha256: registries[1].registry_sha256.clone(),
            measurement_protocol_sha256: registries[1].objectives[1].measurement_protocol_sha256.clone(),
            hardware_profile_sha256: Some(profiles[1].profile_sha256.clone()),
            outcome: MeasurementOutcome::NotMeasured {
                reason_code: "not_collected".into(),
            },
        });
        let result = evaluate_fleet_candidate(
            &manifest, &registries, "candidate-A", &config, &observations,
        ).unwrap();
        assert_eq!(result.status, FleetCandidateStatus::IncompleteEvidence);
        assert_eq!(result.incomplete_reasons.len(), 2);
        assert!(result.incomplete_reasons.iter().any(|reason| reason.contains("instrument_timeout")));
        assert!(result.incomplete_reasons.iter().any(|reason| reason.contains("not_collected")));
    }

    #[test]
    fn duplicate_profile_objective_pair_is_rejected() {
        let (profiles, registries, manifest, config) = fixture();
        let mut observations = complete_observations(
            &profiles, &registries, [0.95, 0.95], [20.0, 20.0],
        );
        observations.push(observations[0].clone());
        assert!(evaluate_fleet_candidate(
            &manifest, &registries, "candidate-A", &config, &observations,
        ).is_err());
    }

    #[test]
    fn candidate_configuration_replay_across_targets_is_rejected() {
        let (profiles, registries, manifest, config) = fixture();
        let mut observations = complete_observations(
            &profiles, &registries, [0.95, 0.95], [20.0, 20.0],
        );
        observations[2].candidate_config_sha256 = sha('b');
        assert!(evaluate_fleet_candidate(
            &manifest, &registries, "candidate-A", &config, &observations,
        ).is_err());
    }

    #[test]
    fn target_registry_and_profile_mismatch_are_rejected() {
        let (profiles, registries, manifest, config) = fixture();
        let mut observations = complete_observations(
            &profiles, &registries, [0.95, 0.95], [20.0, 20.0],
        );
        observations[0].hardware_profile_sha256 = Some(sha('0'));
        assert!(evaluate_fleet_candidate(
            &manifest, &registries, "candidate-A", &config, &observations,
        ).is_err());
    }

    #[test]
    fn inconsistent_normalization_between_profiles_is_rejected() {
        let (profiles, mut registries, mut manifest, config) = fixture();
        registries[1].objectives[0].normalization.raw_max = 2.0;
        registries[1].registry_sha256 =
            compute_objective_registry_sha256(&registries[1]).unwrap();
        manifest.targets[1].objective_registry_sha256 = registries[1].registry_sha256.clone();
        manifest.fleet_sha256 = compute_fleet_deployment_manifest_sha256(&manifest).unwrap();
        assert!(validate_fleet_objective_registries(&manifest, &registries).is_err());
        let _ = (profiles, config);
    }

    #[test]
    fn evaluation_receipt_is_bound_to_all_target_evidence_and_replays() {
        let (profiles, registries, manifest, config) = fixture();
        let observations = complete_observations(
            &profiles, &registries, [0.95, 0.91], [20.0, 25.0],
        );
        let result = evaluate_fleet_candidate(
            &manifest, &registries, "candidate-A", &config, &observations,
        ).unwrap();
        verify_fleet_candidate_evaluation_digest(&result).unwrap();
        verify_fleet_candidate_evaluation(
            &manifest, &registries, "candidate-A", &config, &observations, &result,
        ).unwrap();
        let mut changed = result.clone();
        changed.status = FleetCandidateStatus::IncompleteEvidence;
        changed.evaluation_sha256 = compute_evaluation_sha256(&changed).unwrap();
        verify_fleet_candidate_evaluation_digest(&changed).unwrap();
        assert!(verify_fleet_candidate_evaluation(
            &manifest, &registries, "candidate-A", &config, &observations, &changed,
        ).is_err());
    }

    #[test]
    fn manifest_rejects_unsorted_duplicate_and_invalid_floors() {
        let (profiles, registries, mut manifest, _) = fixture();
        manifest.targets.reverse();
        assert!(validate_fleet_manifest_fields(&manifest).is_err());
        manifest.targets.sort_by(|left, right| left.target_id.cmp(&right.target_id));
        manifest.objectives[0].minimum_utility = 1.5;
        assert!(validate_fleet_manifest_fields(&manifest).is_err());
        let _ = (profiles, registries);
    }
}
