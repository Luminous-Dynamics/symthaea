// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Matched transition-geometry diagnostics for the live meta-reasoning shadow.
//!
//! This module deliberately does **not** populate a canonical objective axis. Production
//! `ReasoningChain::execute_primitive` currently computes `phi_contribution` as a scaled Hamming
//! distance between the input and output hypervectors. That is useful transition-geometry
//! telemetry, but it is not a measurement of integration, irreducibility, consciousness, task
//! correctness, or expected utility.
//!
//! The live path therefore remains:
//! 1. bind lightweight historical candidates to the exact live primitive identity without
//!    changing their existing scoring fields;
//! 2. run the established historical/V2/all-unknown-V3 path unchanged;
//! 3. run a matched preregistered operator panel as side-channel diagnostic telemetry only.
//!
//! Any future mapping from this diagnostic into a policy objective requires separate empirical
//! qualification against externally scored outcomes.

use super::reasoning_shadow_meta::{
    ShadowMetaObservation, ShadowMetaStats, ShadowQualifiedMetaReasoner as BaselineShadowReasoner,
    V3ShadowObservation,
};
use crate::consciousness::meta_reasoning::{
    MetaCognitiveReasoner, MetaReasoningConfig, MetaReasoningResult,
};
use crate::consciousness::primitive_evolution::{CandidatePrimitive, EvolutionConfig};
use crate::consciousness::primitive_reasoning::{ReasoningChain, TransformationType};
use crate::consciousness::ActivePrimitive;
use crate::hdc::BinaryHV;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

pub const TRANSITION_GEOMETRY_SHADOW_VERSION: &str = "rq-006y-transition-geometry-shadow-v1";
pub const TRANSITION_GEOMETRY_PANEL_VERSION: &str = "rq-006y-transition-geometry-panel-v1";
pub const LIVE_META_IDENTITY_BINDING_VERSION: &str = "rq-006y-live-meta-identity-binding-v1";
pub const TRANSITION_GEOMETRY_SCALE: f64 = 0.1;
pub const TRANSITION_GEOMETRY_PANEL: [TransformationType; 4] = [
    TransformationType::Bundle,
    TransformationType::Resonate,
    TransformationType::Abstract,
    TransformationType::Ground,
];

const NUMERIC_TOLERANCE: f64 = 1.0e-12;
const EXECUTION_COMMITMENT_DOMAIN: &[u8] = b"symthaea/reasoning/transition-geometry/execution/v1";
const PANEL_COMMITMENT_DOMAIN: &[u8] = b"symthaea/reasoning/transition-geometry/panel/v1";

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateIdentityBindingReport {
    pub binding_version: String,
    pub attempted: usize,
    pub bound: usize,
    pub missing: usize,
    pub ambiguous: usize,
}

impl CandidateIdentityBindingReport {
    pub fn fully_bound(&self) -> bool {
        self.attempted > 0
            && self.bound == self.attempted
            && self.missing == 0
            && self.ambiguous == 0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransitionOperatorObservation {
    pub transformation: TransformationType,
    pub output_digest: String,
    /// Legacy production field, retained as raw provenance only.
    pub raw_legacy_phi_contribution: f64,
    /// `raw_legacy_phi_contribution / 0.1`; exactly a normalized transition-distance proxy under
    /// the current production executor, not an integration score.
    pub normalized_transition_distance: f64,
    pub execution_commitment: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateTransitionGeometry {
    pub candidate_id: String,
    pub primitive_encoding_digest: String,
    pub input_digest: String,
    pub observations: Vec<TransitionOperatorObservation>,
    pub transition_min: f64,
    pub transition_max: f64,
    pub transition_span: f64,
    pub panel_commitment: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransitionGeometryReport {
    pub panel_version: String,
    pub input_digest: String,
    pub profiles: Vec<CandidateTransitionGeometry>,
    pub global_transition_min: f64,
    pub global_transition_max: f64,
    pub global_transition_span: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TransitionGeometryShadowStats {
    pub attempts: u64,
    pub successes: u64,
    pub errors: u64,
    pub identity_binding_failures: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransitionGeometryShadowObservation {
    pub shadow_version: String,
    pub attempt: u64,
    pub baseline_v3_request_count: Option<usize>,
    pub identity_binding: CandidateIdentityBindingReport,
    pub candidate_profiles: usize,
    pub input_digest: Option<String>,
    pub global_transition_min: Option<f64>,
    pub global_transition_max: Option<f64>,
    pub global_transition_span: Option<f64>,
    pub error: Option<String>,
}

/// Public cognitive-loop compatibility wrapper.
///
/// The established baseline shadow remains behavior-authoritative. Transition geometry is
/// diagnostic-only and cannot fulfill an `IntegrationProxy` evidence request.
pub struct TransitionGeometryShadowQualifiedMetaReasoner {
    inner: BaselineShadowReasoner,
    transition_stats: TransitionGeometryShadowStats,
    last_transition_observation: Option<TransitionGeometryShadowObservation>,
    last_identity_binding_report: Option<CandidateIdentityBindingReport>,
}

impl TransitionGeometryShadowQualifiedMetaReasoner {
    pub fn new(
        evolution_config: EvolutionConfig,
        meta_config: MetaReasoningConfig,
    ) -> Result<Self> {
        Ok(Self {
            inner: BaselineShadowReasoner::new(evolution_config, meta_config)?,
            transition_stats: TransitionGeometryShadowStats::default(),
            last_transition_observation: None,
            last_identity_binding_report: None,
        })
    }

    pub fn meta_reason(
        &mut self,
        query: &str,
        primitives: Vec<CandidatePrimitive>,
        chain: &mut ReasoningChain,
    ) -> Result<MetaReasoningResult> {
        self.last_identity_binding_report = None;
        self.inner.meta_reason(query, primitives, chain)
    }

    pub fn meta_reason_with_active_evidence(
        &mut self,
        query: &str,
        primitives: Vec<CandidatePrimitive>,
        active_primitives: &[ActivePrimitive],
        chain: &mut ReasoningChain,
    ) -> Result<MetaReasoningResult> {
        let frozen_input = chain.question;
        let (bound_primitives, identity_binding) =
            bind_candidate_identities(primitives, active_primitives);
        let candidate_ids = bound_primitives
            .iter()
            .map(|candidate| candidate.name.clone())
            .collect::<Vec<_>>();
        self.last_identity_binding_report = Some(identity_binding.clone());

        // Historical reasoning + V2 + all-unknown V3 execute first and remain authoritative.
        let legacy_result = self.inner.meta_reason_with_active_evidence(
            query,
            bound_primitives,
            active_primitives,
            chain,
        )?;

        let attempt = self.transition_stats.attempts;
        self.transition_stats.attempts = self.transition_stats.attempts.saturating_add(1);
        let baseline_v3_request_count = self
            .inner
            .last_v3_shadow_observation()
            .map(|observation| observation.request_count);

        if !identity_binding.fully_bound() {
            self.transition_stats.errors = self.transition_stats.errors.saturating_add(1);
            self.transition_stats.identity_binding_failures = self
                .transition_stats
                .identity_binding_failures
                .saturating_add(1);
            let error = format!(
                "transition geometry withheld: identity binding incomplete (bound={}, missing={}, ambiguous={})",
                identity_binding.bound, identity_binding.missing, identity_binding.ambiguous
            );
            self.last_transition_observation = Some(TransitionGeometryShadowObservation {
                shadow_version: TRANSITION_GEOMETRY_SHADOW_VERSION.into(),
                attempt,
                baseline_v3_request_count,
                identity_binding,
                candidate_profiles: 0,
                input_digest: None,
                global_transition_min: None,
                global_transition_max: None,
                global_transition_span: None,
                error: Some(error),
            });
            return Ok(legacy_result);
        }

        match measure_transition_geometry(active_primitives, &candidate_ids, frozen_input) {
            Ok(report) => {
                self.transition_stats.successes = self.transition_stats.successes.saturating_add(1);
                tracing::debug!(
                    target: "symthaea::reasoning_transition_geometry",
                    attempt,
                    candidates = report.profiles.len(),
                    input_digest = %report.input_digest,
                    transition_min = report.global_transition_min,
                    transition_max = report.global_transition_max,
                    transition_span = report.global_transition_span,
                    baseline_v3_request_count = baseline_v3_request_count.unwrap_or(0),
                    "matched transition-geometry diagnostic observed without objective mutation"
                );
                self.last_transition_observation = Some(TransitionGeometryShadowObservation {
                    shadow_version: TRANSITION_GEOMETRY_SHADOW_VERSION.into(),
                    attempt,
                    baseline_v3_request_count,
                    identity_binding,
                    candidate_profiles: report.profiles.len(),
                    input_digest: Some(report.input_digest),
                    global_transition_min: Some(report.global_transition_min),
                    global_transition_max: Some(report.global_transition_max),
                    global_transition_span: Some(report.global_transition_span),
                    error: None,
                });
            }
            Err(err) => {
                self.transition_stats.errors = self.transition_stats.errors.saturating_add(1);
                self.last_transition_observation = Some(TransitionGeometryShadowObservation {
                    shadow_version: TRANSITION_GEOMETRY_SHADOW_VERSION.into(),
                    attempt,
                    baseline_v3_request_count,
                    identity_binding,
                    candidate_profiles: 0,
                    input_digest: None,
                    global_transition_min: None,
                    global_transition_max: None,
                    global_transition_span: None,
                    error: Some(err.to_string()),
                });
            }
        }

        Ok(legacy_result)
    }

    pub fn legacy(&self) -> &MetaCognitiveReasoner {
        self.inner.legacy()
    }

    pub fn shadow_stats(&self) -> &ShadowMetaStats {
        self.inner.shadow_stats()
    }

    pub fn last_shadow_observation(&self) -> Option<&ShadowMetaObservation> {
        self.inner.last_shadow_observation()
    }

    pub fn last_v3_shadow_observation(&self) -> Option<&V3ShadowObservation> {
        self.inner.last_v3_shadow_observation()
    }

    pub fn canonical_next_sequence(&self) -> u64 {
        self.inner.canonical_next_sequence()
    }

    pub fn transition_geometry_stats(&self) -> &TransitionGeometryShadowStats {
        &self.transition_stats
    }

    pub fn last_transition_geometry_observation(
        &self,
    ) -> Option<&TransitionGeometryShadowObservation> {
        self.last_transition_observation.as_ref()
    }

    pub fn last_identity_binding_report(&self) -> Option<&CandidateIdentityBindingReport> {
        self.last_identity_binding_report.as_ref()
    }
}

pub fn measure_transition_geometry(
    active_primitives: &[ActivePrimitive],
    candidate_ids: &[String],
    frozen_input: BinaryHV,
) -> std::result::Result<TransitionGeometryReport, TransitionGeometryError> {
    if candidate_ids.is_empty() {
        return Err(TransitionGeometryError::EmptyCandidateSet);
    }

    let mut active_by_name: HashMap<&str, Option<&ActivePrimitive>> = HashMap::new();
    for active in active_primitives {
        let name = active.primitive.name.as_str();
        active_by_name
            .entry(name)
            .and_modify(|slot| *slot = None)
            .or_insert(Some(active));
    }

    let input_digest = digest_hv(&frozen_input);
    let mut profiles = Vec::with_capacity(candidate_ids.len());
    let mut global_min = f64::INFINITY;
    let mut global_max = f64::NEG_INFINITY;

    for candidate_id in candidate_ids {
        let active = match active_by_name.get(candidate_id.as_str()) {
            Some(Some(active)) => *active,
            Some(None) => {
                return Err(TransitionGeometryError::AmbiguousActivePrimitive(
                    candidate_id.clone(),
                ));
            }
            None => {
                return Err(TransitionGeometryError::MissingActivePrimitive(
                    candidate_id.clone(),
                ));
            }
        };

        let encoding_digest = digest_hv(&active.primitive.encoding);
        let mut observations = Vec::with_capacity(TRANSITION_GEOMETRY_PANEL.len());
        let mut transition_min = f64::INFINITY;
        let mut transition_max = f64::NEG_INFINITY;

        for transformation in TRANSITION_GEOMETRY_PANEL {
            let mut chain = ReasoningChain::new(frozen_input);
            chain
                .execute_primitive(&active.primitive, transformation)
                .map_err(|err| TransitionGeometryError::ExecutionFailed {
                    candidate_id: candidate_id.clone(),
                    transformation,
                    error: err.to_string(),
                })?;
            if chain.executions.len() != 1 {
                return Err(TransitionGeometryError::UnexpectedExecutionCount {
                    candidate_id: candidate_id.clone(),
                    transformation,
                    found: chain.executions.len(),
                });
            }
            let execution = &chain.executions[0];
            if execution.input != frozen_input {
                return Err(TransitionGeometryError::InputMismatch {
                    candidate_id: candidate_id.clone(),
                    transformation,
                });
            }
            if execution.primitive.name != *candidate_id {
                return Err(TransitionGeometryError::CandidateMismatch {
                    expected: candidate_id.clone(),
                    found: execution.primitive.name.clone(),
                });
            }
            if execution.transformation != transformation {
                return Err(TransitionGeometryError::TransformationMismatch {
                    candidate_id: candidate_id.clone(),
                    expected: transformation,
                    found: execution.transformation,
                });
            }

            let raw = execution.phi_contribution;
            if !raw.is_finite()
                || raw < -NUMERIC_TOLERANCE
                || raw > TRANSITION_GEOMETRY_SCALE + NUMERIC_TOLERANCE
            {
                return Err(TransitionGeometryError::InvalidLegacyContribution {
                    candidate_id: candidate_id.clone(),
                    transformation,
                    value: raw,
                });
            }
            let normalized = (raw / TRANSITION_GEOMETRY_SCALE).clamp(0.0, 1.0);
            let output_digest = digest_hv(&execution.output);
            let execution_commitment = execution_commitment(
                candidate_id,
                &active.primitive.encoding,
                &frozen_input,
                &execution.output,
                transformation,
                raw,
            );
            transition_min = transition_min.min(normalized);
            transition_max = transition_max.max(normalized);
            observations.push(TransitionOperatorObservation {
                transformation,
                output_digest,
                raw_legacy_phi_contribution: raw,
                normalized_transition_distance: normalized,
                execution_commitment,
            });
        }

        let transition_span = transition_max - transition_min;
        let panel_commitment = panel_commitment(
            candidate_id,
            &encoding_digest,
            &input_digest,
            &observations,
        );
        global_min = global_min.min(transition_min);
        global_max = global_max.max(transition_max);
        profiles.push(CandidateTransitionGeometry {
            candidate_id: candidate_id.clone(),
            primitive_encoding_digest: encoding_digest,
            input_digest: input_digest.clone(),
            observations,
            transition_min,
            transition_max,
            transition_span,
            panel_commitment,
        });
    }

    let global_transition_span = global_max - global_min;
    Ok(TransitionGeometryReport {
        panel_version: TRANSITION_GEOMETRY_PANEL_VERSION.into(),
        input_digest,
        profiles,
        global_transition_min: global_min,
        global_transition_max: global_max,
        global_transition_span,
    })
}

fn bind_candidate_identities(
    mut candidates: Vec<CandidatePrimitive>,
    active_primitives: &[ActivePrimitive],
) -> (Vec<CandidatePrimitive>, CandidateIdentityBindingReport) {
    let mut active_by_name: HashMap<&str, Option<&ActivePrimitive>> = HashMap::new();
    for active in active_primitives {
        let name = active.primitive.name.as_str();
        active_by_name
            .entry(name)
            .and_modify(|slot| *slot = None)
            .or_insert(Some(active));
    }

    let mut report = CandidateIdentityBindingReport {
        binding_version: LIVE_META_IDENTITY_BINDING_VERSION.into(),
        attempted: candidates.len(),
        ..CandidateIdentityBindingReport::default()
    };

    for candidate in &mut candidates {
        match active_by_name.get(candidate.name.as_str()) {
            Some(Some(active)) => {
                candidate.tier = active.primitive.tier;
                candidate.definition = active.primitive.definition.clone();
                candidate.encoding = active.primitive.encoding;
                report.bound += 1;
            }
            Some(None) => report.ambiguous += 1,
            None => report.missing += 1,
        }
    }

    (candidates, report)
}

fn digest_hv(hv: &BinaryHV) -> String {
    blake3::hash(&hv.0).to_hex().to_string()
}

fn execution_commitment(
    candidate_id: &str,
    encoding: &BinaryHV,
    input: &BinaryHV,
    output: &BinaryHV,
    transformation: TransformationType,
    raw_contribution: f64,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, EXECUTION_COMMITMENT_DOMAIN);
    hash_str(&mut hasher, TRANSITION_GEOMETRY_PANEL_VERSION);
    hash_str(&mut hasher, candidate_id);
    hash_bytes(&mut hasher, &encoding.0);
    hash_bytes(&mut hasher, &input.0);
    hash_bytes(&mut hasher, &output.0);
    hash_u64(&mut hasher, transformation_tag(transformation));
    hash_u64(&mut hasher, raw_contribution.to_bits());
    hasher.finalize().to_hex().to_string()
}

fn panel_commitment(
    candidate_id: &str,
    encoding_digest: &str,
    input_digest: &str,
    observations: &[TransitionOperatorObservation],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, PANEL_COMMITMENT_DOMAIN);
    hash_str(&mut hasher, TRANSITION_GEOMETRY_PANEL_VERSION);
    hash_str(&mut hasher, candidate_id);
    hash_str(&mut hasher, encoding_digest);
    hash_str(&mut hasher, input_digest);
    hash_u64(&mut hasher, observations.len() as u64);
    for observation in observations {
        hash_u64(&mut hasher, transformation_tag(observation.transformation));
        hash_str(&mut hasher, &observation.output_digest);
        hash_u64(
            &mut hasher,
            observation.raw_legacy_phi_contribution.to_bits(),
        );
        hash_str(&mut hasher, &observation.execution_commitment);
    }
    hasher.finalize().to_hex().to_string()
}

const fn transformation_tag(transformation: TransformationType) -> u64 {
    match transformation {
        TransformationType::Bind => 0,
        TransformationType::Bundle => 1,
        TransformationType::Permute => 2,
        TransformationType::Resonate => 3,
        TransformationType::Abstract => 4,
        TransformationType::Ground => 5,
    }
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hash_bytes(hasher, value.as_bytes());
}

fn hash_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hash_u64(hasher, value.len() as u64);
    hasher.update(value);
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

#[derive(Debug)]
pub enum TransitionGeometryError {
    EmptyCandidateSet,
    MissingActivePrimitive(String),
    AmbiguousActivePrimitive(String),
    ExecutionFailed {
        candidate_id: String,
        transformation: TransformationType,
        error: String,
    },
    UnexpectedExecutionCount {
        candidate_id: String,
        transformation: TransformationType,
        found: usize,
    },
    InputMismatch {
        candidate_id: String,
        transformation: TransformationType,
    },
    CandidateMismatch {
        expected: String,
        found: String,
    },
    TransformationMismatch {
        candidate_id: String,
        expected: TransformationType,
        found: TransformationType,
    },
    InvalidLegacyContribution {
        candidate_id: String,
        transformation: TransformationType,
        value: f64,
    },
}

impl fmt::Display for TransitionGeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidateSet => write!(f, "transition geometry requires candidates"),
            Self::MissingActivePrimitive(id) => {
                write!(f, "candidate `{id}` is absent from the active primitive snapshot")
            }
            Self::AmbiguousActivePrimitive(id) => {
                write!(f, "candidate `{id}` appears more than once in the active snapshot")
            }
            Self::ExecutionFailed {
                candidate_id,
                transformation,
                error,
            } => write!(
                f,
                "candidate `{candidate_id}` {transformation:?} execution failed: {error}"
            ),
            Self::UnexpectedExecutionCount {
                candidate_id,
                transformation,
                found,
            } => write!(
                f,
                "candidate `{candidate_id}` {transformation:?} expected one execution, found {found}"
            ),
            Self::InputMismatch {
                candidate_id,
                transformation,
            } => write!(
                f,
                "candidate `{candidate_id}` {transformation:?} did not preserve the frozen input"
            ),
            Self::CandidateMismatch { expected, found } => write!(
                f,
                "transition diagnostic executed candidate `{found}`, expected `{expected}`"
            ),
            Self::TransformationMismatch {
                candidate_id,
                expected,
                found,
            } => write!(
                f,
                "candidate `{candidate_id}` executed {found:?}, expected {expected:?}"
            ),
            Self::InvalidLegacyContribution {
                candidate_id,
                transformation,
                value,
            } => write!(
                f,
                "candidate `{candidate_id}` {transformation:?} legacy contribution {value} is outside the frozen scale"
            ),
        }
    }
}

impl std::error::Error for TransitionGeometryError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::epistemic_tiers::EpistemicCoordinate;
    use crate::consciousness::ActivationReason;
    use symthaea_core::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier};

    fn active(name: &str, activation: f64) -> ActivePrimitive {
        ActivePrimitive {
            primitive: PrimitiveSystem::global()
                .get(name)
                .unwrap_or_else(|| panic!("fixture primitive `{name}` must exist"))
                .clone(),
            activation,
            activation_reason: ActivationReason::BottomUp {
                input_similarity: activation,
            },
            duration: 2,
        }
    }

    fn placeholder_candidate(name: &str) -> CandidatePrimitive {
        CandidatePrimitive {
            name: name.into(),
            tier: PrimitiveTier::NSM,
            definition: name.into(),
            fitness: 0.731,
            encoding: BinaryHV::random(42),
            epistemic_coordinate: EpistemicCoordinate::axiom(),
            harmonic_alignment: 0.617,
        }
    }

    #[test]
    fn bind_local_transition_is_input_invariant_and_excluded_from_panel() {
        assert!(!TRANSITION_GEOMETRY_PANEL.contains(&TransformationType::Bind));
        assert!(!TRANSITION_GEOMETRY_PANEL.contains(&TransformationType::Permute));
        let primitive = active("NSM_KNOW", 0.8).primitive;
        let mut first = ReasoningChain::new(BinaryHV::random(30_001));
        let mut second = ReasoningChain::new(BinaryHV::random(30_002));
        first
            .execute_primitive(&primitive, TransformationType::Bind)
            .unwrap();
        second
            .execute_primitive(&primitive, TransformationType::Bind)
            .unwrap();
        let first_normalized = first.executions[0].phi_contribution / TRANSITION_GEOMETRY_SCALE;
        let second_normalized = second.executions[0].phi_contribution / TRANSITION_GEOMETRY_SCALE;
        assert!((first_normalized - second_normalized).abs() < 1.0e-12);
        assert!((first_normalized - primitive.encoding.density() as f64).abs() < 1.0e-6);
    }

    #[test]
    fn exact_identity_binding_preserves_scoring_inputs() {
        let original = placeholder_candidate("NSM_KNOW");
        let fitness = original.fitness;
        let epistemic = original.epistemic_coordinate;
        let harmonic = original.harmonic_alignment;
        let actives = [active("NSM_KNOW", 0.8)];
        let (bound, report) = bind_candidate_identities(vec![original], &actives);
        let candidate = &bound[0];
        assert!(report.fully_bound());
        assert_eq!(candidate.tier, actives[0].primitive.tier);
        assert_eq!(candidate.definition, actives[0].primitive.definition);
        assert_eq!(candidate.encoding, actives[0].primitive.encoding);
        assert_eq!(candidate.fitness, fitness);
        assert_eq!(candidate.epistemic_coordinate, epistemic);
        assert_eq!(candidate.harmonic_alignment, harmonic);
    }

    #[test]
    fn missing_and_duplicate_identity_are_never_invented() {
        let missing = placeholder_candidate("NSM_DO");
        let missing_encoding = missing.encoding;
        let (bound_missing, missing_report) =
            bind_candidate_identities(vec![missing], &[active("NSM_KNOW", 0.8)]);
        assert_eq!(missing_report.missing, 1);
        assert_eq!(bound_missing[0].encoding, missing_encoding);

        let original = placeholder_candidate("NSM_KNOW");
        let original_encoding = original.encoding;
        let mut duplicate = active("NSM_KNOW", 0.7);
        duplicate.primitive.encoding = BinaryHV::random(30_003);
        let actives = [active("NSM_KNOW", 0.8), duplicate];
        let (bound_duplicate, duplicate_report) =
            bind_candidate_identities(vec![original], &actives);
        assert_eq!(duplicate_report.ambiguous, 1);
        assert_eq!(bound_duplicate[0].encoding, original_encoding);
    }

    #[test]
    fn matched_panel_is_deterministic_and_binds_candidate_input_and_operators() {
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.6)];
        let ids = vec!["NSM_KNOW".into(), "NSM_DO".into()];
        let input = BinaryHV::random(30_004);
        let first = measure_transition_geometry(&actives, &ids, input).unwrap();
        let second = measure_transition_geometry(&actives, &ids, input).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.profiles.len(), 2);
        for profile in &first.profiles {
            assert_eq!(profile.input_digest, first.input_digest);
            assert_eq!(profile.observations.len(), TRANSITION_GEOMETRY_PANEL.len());
            for (observation, expected) in
                profile.observations.iter().zip(TRANSITION_GEOMETRY_PANEL)
            {
                assert_eq!(observation.transformation, expected);
                assert!((0.0..=1.0).contains(&observation.normalized_transition_distance));
            }
            let min = profile
                .observations
                .iter()
                .map(|item| item.normalized_transition_distance)
                .fold(f64::INFINITY, f64::min);
            let max = profile
                .observations
                .iter()
                .map(|item| item.normalized_transition_distance)
                .fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(profile.transition_min, min);
            assert_eq!(profile.transition_max, max);
            assert_eq!(profile.transition_span, max - min);
        }
    }

    #[test]
    fn live_diagnostic_does_not_fulfill_v3_objective_requests() {
        let mut reasoner = TransitionGeometryShadowQualifiedMetaReasoner::new(
            EvolutionConfig::default(),
            MetaReasoningConfig::default(),
        )
        .unwrap();
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.6)];
        let mut chain = ReasoningChain::new(BinaryHV::random(30_005));
        let result = reasoner.meta_reason_with_active_evidence(
            "evidence experiment research theory scientific",
            vec![
                placeholder_candidate("NSM_KNOW"),
                placeholder_candidate("NSM_DO"),
            ],
            &actives,
            &mut chain,
        );
        assert!(result.is_ok());
        assert!(reasoner
            .last_identity_binding_report()
            .is_some_and(CandidateIdentityBindingReport::fully_bound));
        let baseline = reasoner.last_v3_shadow_observation().unwrap();
        // Two candidates x three genuinely unknown policy objectives remain six requests. Running
        // the transition diagnostic must not make IntegrationProxy appear observed.
        assert_eq!(baseline.request_count, 6);
        assert_eq!(reasoner.transition_geometry_stats().successes, 1);
        let diagnostic = reasoner.last_transition_geometry_observation().unwrap();
        assert_eq!(diagnostic.baseline_v3_request_count, Some(6));
        assert_eq!(diagnostic.candidate_profiles, 2);
        assert!(diagnostic.error.is_none());
    }

    #[test]
    fn incomplete_identity_withholds_diagnostic_but_not_historical_result() {
        let mut reasoner = TransitionGeometryShadowQualifiedMetaReasoner::new(
            EvolutionConfig::default(),
            MetaReasoningConfig::default(),
        )
        .unwrap();
        let mut chain = ReasoningChain::new(BinaryHV::random(30_006));
        let result = reasoner.meta_reason_with_active_evidence(
            "evidence experiment research theory scientific",
            vec![
                placeholder_candidate("NSM_KNOW"),
                placeholder_candidate("NSM_DO"),
            ],
            &[active("NSM_KNOW", 0.8)],
            &mut chain,
        );
        assert!(result.is_ok());
        assert_eq!(reasoner.transition_geometry_stats().attempts, 1);
        assert_eq!(reasoner.transition_geometry_stats().successes, 0);
        assert_eq!(reasoner.transition_geometry_stats().identity_binding_failures, 1);
        assert!(reasoner
            .last_transition_geometry_observation()
            .unwrap()
            .error
            .is_some());
    }
}
