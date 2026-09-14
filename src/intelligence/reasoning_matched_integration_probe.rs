// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Matched counterfactual IntegrationProxy probe for the live V3 candidate set.
//!
//! A single XOR `Bind` probe is deliberately insufficient: because BinaryHV similarity is Hamming
//! similarity, `similarity(input, input XOR primitive)` depends only on the primitive's bit density
//! and is independent of the frozen input. V2 therefore excludes that degenerate operator from
//! evidence and uses a preregistered panel of candidate-sensitive, input-sensitive production
//! transformations. The emitted interval is the exact min/max local-transition proxy across that
//! frozen panel. It is an operator-sensitivity envelope, not a confidence interval, historical
//! performance estimate, IIT Phi, consciousness evidence, reasoning correctness, or utility.

use super::reasoning_active_primitive_evidence::{
    adapt_active_primitive_evidence, ActivePrimitiveEvidenceError, ActivePrimitiveEvidenceReport,
};
use super::reasoning_context_competition::{ContextCompetitionPolicy, ContextHypothesis};
use super::reasoning_evidence_seeking::{
    plan_with_evidence, EvidenceSeekingPlanReport, EvidenceSeekingPlannerError,
};
use super::reasoning_integration_history::LOCAL_TRANSITION_CONTRIBUTION_SCALE;
use super::reasoning_objective_evidence::{
    CandidateObjectiveEvidence, ObjectiveEvidence, ObjectiveEvidenceError,
};
use crate::consciousness::primitive_reasoning::{ReasoningChain, TransformationType};
use crate::consciousness::ActivePrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use symthaea_core::hdc::BinaryHV;

pub const MATCHED_INTEGRATION_PROBE_VERSION: &str = "rq-006y-matched-integration-probe-v2";
const EXECUTION_COMMITMENT_DOMAIN: &[u8] = b"symthaea/reasoning/matched-integration-probe/v2";
const PANEL_COMMITMENT_DOMAIN: &[u8] = b"symthaea/reasoning/matched-integration-panel/v2";
const NUMERIC_TOLERANCE: f64 = 1.0e-12;

/// Frozen v2 panel. `Bind` is excluded because its local-transition score is input-invariant under
/// XOR/Hamming geometry. `Permute` is excluded because it does not depend on candidate identity.
pub const MATCHED_TRANSFORMATION_PANEL: [TransformationType; 4] = [
    TransformationType::Bundle,
    TransformationType::Resonate,
    TransformationType::Abstract,
    TransformationType::Ground,
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchedOperatorObservation {
    pub transformation: TransformationType,
    pub output_digest: String,
    pub raw_local_contribution: f64,
    pub normalized_integration_proxy: f64,
    pub execution_commitment: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchedIntegrationProbeProfile {
    pub candidate_id: String,
    pub primitive_encoding_digest: String,
    pub input_digest: String,
    pub operator_observations: Vec<MatchedOperatorObservation>,
    /// Exact min across the frozen operator panel.
    pub integration_lower: f64,
    /// Exact max across the frozen operator panel.
    pub integration_upper: f64,
    pub panel_commitment: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchedIntegrationProbeReport {
    pub probe_version: String,
    pub input_digest: String,
    pub transformation_panel: Vec<TransformationType>,
    pub profiles: Vec<MatchedIntegrationProbeProfile>,
    /// Candidate evidence in the same order as the active-evidence report.
    pub candidates: Vec<CandidateObjectiveEvidence>,
    /// Minimum lower bound across candidate operator envelopes.
    pub integration_min: f64,
    /// Maximum upper bound across candidate operator envelopes.
    pub integration_max: f64,
    pub integration_spread: f64,
}

pub fn probe_active_primitive_integration(
    active: &[ActivePrimitive],
    candidate_ids: &[String],
    frozen_input: BinaryHV,
) -> Result<
    (ActivePrimitiveEvidenceReport, MatchedIntegrationProbeReport),
    MatchedIntegrationProbeError,
> {
    let active_report = adapt_active_primitive_evidence(active, candidate_ids)?;
    let input_digest = digest_hv(&frozen_input);

    let mut active_by_name = HashMap::with_capacity(active.len());
    for primitive in active {
        if active_by_name
            .insert(primitive.primitive.name.as_str(), primitive)
            .is_some()
        {
            return Err(MatchedIntegrationProbeError::DuplicateActivePrimitive(
                primitive.primitive.name.clone(),
            ));
        }
    }

    let mut profiles = Vec::with_capacity(active_report.profiles.len());
    let mut candidates = Vec::with_capacity(active_report.profiles.len());
    let mut integration_min = f64::INFINITY;
    let mut integration_max = f64::NEG_INFINITY;

    for evidence_profile in &active_report.profiles {
        let active_primitive = active_by_name
            .get(evidence_profile.candidate_id.as_str())
            .copied()
            .ok_or_else(|| {
                MatchedIntegrationProbeError::MissingActivePrimitive(
                    evidence_profile.candidate_id.clone(),
                )
            })?;
        let encoding_digest = digest_hv(&active_primitive.primitive.encoding);
        if encoding_digest != evidence_profile.observed_primitive.encoding_digest {
            return Err(MatchedIntegrationProbeError::EncodingDigestMismatch {
                candidate_id: evidence_profile.candidate_id.clone(),
                active_digest: encoding_digest,
                evidence_digest: evidence_profile.observed_primitive.encoding_digest.clone(),
            });
        }

        let mut operator_observations = Vec::with_capacity(MATCHED_TRANSFORMATION_PANEL.len());
        let mut candidate_min = f64::INFINITY;
        let mut candidate_max = f64::NEG_INFINITY;

        for transformation in MATCHED_TRANSFORMATION_PANEL {
            let observation = execute_matched_operator(
                &evidence_profile.candidate_id,
                &active_primitive.primitive,
                frozen_input,
                transformation,
            )?;
            candidate_min = candidate_min.min(observation.normalized_integration_proxy);
            candidate_max = candidate_max.max(observation.normalized_integration_proxy);
            operator_observations.push(observation);
        }

        if !candidate_min.is_finite() || !candidate_max.is_finite() || candidate_min > candidate_max
        {
            return Err(MatchedIntegrationProbeError::InvalidOperatorEnvelope {
                candidate_id: evidence_profile.candidate_id.clone(),
                lower: candidate_min,
                upper: candidate_max,
            });
        }

        let panel_commitment = panel_commitment(
            &evidence_profile.candidate_id,
            &active_primitive.primitive.encoding,
            &frozen_input,
            &operator_observations,
        );

        let mut candidate = evidence_profile.objective_evidence.clone();
        candidate.validate()?;
        if candidate.integration_proxy.is_observed() {
            return Err(MatchedIntegrationProbeError::AxisAlreadyObserved(
                evidence_profile.candidate_id.clone(),
            ));
        }
        let harmonic_before = candidate.harmonic_alignment.clone();
        let epistemic_before = candidate.epistemic_grounding.clone();
        let mut evidence_refs = vec![
            format!("probe-input:{input_digest}"),
            format!("candidate-encoding:{encoding_digest}"),
            format!("probe-panel:{panel_commitment}"),
            format!("probe-operator-count:{}", MATCHED_TRANSFORMATION_PANEL.len()),
        ];
        evidence_refs.extend(
            operator_observations
                .iter()
                .map(|observation| format!("probe-execution:{}", observation.execution_commitment)),
        );
        candidate.integration_proxy = ObjectiveEvidence::observed_interval(
            MATCHED_INTEGRATION_PROBE_VERSION,
            evidence_refs,
            candidate_min,
            candidate_max,
        )?;
        candidate.validate()?;
        if candidate.harmonic_alignment != harmonic_before
            || candidate.epistemic_grounding != epistemic_before
        {
            return Err(MatchedIntegrationProbeError::CrossAxisMutation(
                evidence_profile.candidate_id.clone(),
            ));
        }

        integration_min = integration_min.min(candidate_min);
        integration_max = integration_max.max(candidate_max);
        profiles.push(MatchedIntegrationProbeProfile {
            candidate_id: evidence_profile.candidate_id.clone(),
            primitive_encoding_digest: encoding_digest,
            input_digest: input_digest.clone(),
            operator_observations,
            integration_lower: candidate_min,
            integration_upper: candidate_max,
            panel_commitment,
        });
        candidates.push(candidate);
    }

    if profiles.is_empty() {
        return Err(MatchedIntegrationProbeError::EmptyProbeSet);
    }
    let integration_spread = integration_max - integration_min;
    Ok((
        active_report,
        MatchedIntegrationProbeReport {
            probe_version: MATCHED_INTEGRATION_PROBE_VERSION.into(),
            input_digest,
            transformation_panel: MATCHED_TRANSFORMATION_PANEL.to_vec(),
            profiles,
            candidates,
            integration_min,
            integration_max,
            integration_spread,
        },
    ))
}

pub fn plan_active_primitive_with_matched_integration(
    hypotheses: &[ContextHypothesis],
    context_policy: ContextCompetitionPolicy,
    active: &[ActivePrimitive],
    candidate_ids: &[String],
    frozen_input: BinaryHV,
) -> Result<
    (
        ActivePrimitiveEvidenceReport,
        MatchedIntegrationProbeReport,
        EvidenceSeekingPlanReport,
    ),
    MatchedIntegrationProbeError,
> {
    let (active_report, probe_report) =
        probe_active_primitive_integration(active, candidate_ids, frozen_input)?;
    let plan = plan_with_evidence(hypotheses, context_policy, &probe_report.candidates)?;
    Ok((active_report, probe_report, plan))
}

fn execute_matched_operator(
    candidate_id: &str,
    primitive: &symthaea_core::hdc::primitive_system::Primitive,
    frozen_input: BinaryHV,
    transformation: TransformationType,
) -> Result<MatchedOperatorObservation, MatchedIntegrationProbeError> {
    let mut chain = ReasoningChain::new(frozen_input);
    chain
        .execute_primitive(primitive, transformation)
        .map_err(|err| MatchedIntegrationProbeError::ExecutionFailed {
            candidate_id: candidate_id.to_owned(),
            transformation,
            error: err.to_string(),
        })?;
    if chain.executions.len() != 1 {
        return Err(MatchedIntegrationProbeError::UnexpectedExecutionCount {
            candidate_id: candidate_id.to_owned(),
            transformation,
            found: chain.executions.len(),
        });
    }
    let execution = &chain.executions[0];
    if execution.input != frozen_input {
        return Err(MatchedIntegrationProbeError::InputMismatch {
            candidate_id: candidate_id.to_owned(),
            transformation,
        });
    }
    if execution.primitive.name != candidate_id {
        return Err(MatchedIntegrationProbeError::CandidateIdentityMismatch {
            expected: candidate_id.to_owned(),
            found: execution.primitive.name.clone(),
            transformation,
        });
    }
    if execution.transformation != transformation {
        return Err(MatchedIntegrationProbeError::TransformationMismatch {
            candidate_id: candidate_id.to_owned(),
            expected: transformation,
            found: execution.transformation,
        });
    }
    let raw = execution.phi_contribution;
    if !raw.is_finite()
        || raw < -NUMERIC_TOLERANCE
        || raw > LOCAL_TRANSITION_CONTRIBUTION_SCALE + NUMERIC_TOLERANCE
    {
        return Err(MatchedIntegrationProbeError::InvalidContribution {
            candidate_id: candidate_id.to_owned(),
            transformation,
            value: raw,
        });
    }
    let normalized = (raw / LOCAL_TRANSITION_CONTRIBUTION_SCALE).clamp(0.0, 1.0);
    let output_digest = digest_hv(&execution.output);
    let execution_commitment = execution_commitment(
        candidate_id,
        &primitive.encoding,
        &frozen_input,
        &execution.output,
        transformation,
        raw,
    );
    Ok(MatchedOperatorObservation {
        transformation,
        output_digest,
        raw_local_contribution: raw,
        normalized_integration_proxy: normalized,
        execution_commitment,
    })
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
    hash_str(&mut hasher, MATCHED_INTEGRATION_PROBE_VERSION);
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
    encoding: &BinaryHV,
    input: &BinaryHV,
    observations: &[MatchedOperatorObservation],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, PANEL_COMMITMENT_DOMAIN);
    hash_str(&mut hasher, MATCHED_INTEGRATION_PROBE_VERSION);
    hash_str(&mut hasher, candidate_id);
    hash_bytes(&mut hasher, &encoding.0);
    hash_bytes(&mut hasher, &input.0);
    hash_u64(&mut hasher, observations.len() as u64);
    for observation in observations {
        hash_u64(&mut hasher, transformation_tag(observation.transformation));
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
pub enum MatchedIntegrationProbeError {
    EmptyProbeSet,
    DuplicateActivePrimitive(String),
    MissingActivePrimitive(String),
    EncodingDigestMismatch {
        candidate_id: String,
        active_digest: String,
        evidence_digest: String,
    },
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
    CandidateIdentityMismatch {
        expected: String,
        found: String,
        transformation: TransformationType,
    },
    TransformationMismatch {
        candidate_id: String,
        expected: TransformationType,
        found: TransformationType,
    },
    InvalidContribution {
        candidate_id: String,
        transformation: TransformationType,
        value: f64,
    },
    InvalidOperatorEnvelope {
        candidate_id: String,
        lower: f64,
        upper: f64,
    },
    AxisAlreadyObserved(String),
    CrossAxisMutation(String),
    ActivePrimitive(ActivePrimitiveEvidenceError),
    Objective(ObjectiveEvidenceError),
    Planner(EvidenceSeekingPlannerError),
}

impl fmt::Display for MatchedIntegrationProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyProbeSet => write!(f, "matched integration probe requires candidates"),
            Self::DuplicateActivePrimitive(id) => {
                write!(f, "active primitive snapshot contains duplicate candidate `{id}`")
            }
            Self::MissingActivePrimitive(id) => {
                write!(f, "candidate `{id}` is absent from the active primitive snapshot")
            }
            Self::EncodingDigestMismatch {
                candidate_id,
                active_digest,
                evidence_digest,
            } => write!(
                f,
                concat!(
                    "candidate `{candidate_id}` active encoding `{active_digest}` ",
                    "differs from evidence profile `{evidence_digest}`"
                )
            ),
            Self::ExecutionFailed {
                candidate_id,
                transformation,
                error,
            } => write!(
                f,
                "candidate `{candidate_id}` matched {transformation:?} probe failed: {error}"
            ),
            Self::UnexpectedExecutionCount {
                candidate_id,
                transformation,
                found,
            } => write!(
                f,
                concat!(
                    "candidate `{candidate_id}` matched {transformation:?} probe ",
                    "expected one execution, found {found}"
                )
            ),
            Self::InputMismatch {
                candidate_id,
                transformation,
            } => write!(
                f,
                concat!(
                    "candidate `{candidate_id}` matched {transformation:?} probe ",
                    "did not preserve frozen input"
                )
            ),
            Self::CandidateIdentityMismatch {
                expected,
                found,
                transformation,
            } => write!(
                f,
                concat!(
                    "matched {transformation:?} probe executed candidate `{found}`, ",
                    "expected `{expected}`"
                )
            ),
            Self::TransformationMismatch {
                candidate_id,
                expected,
                found,
            } => write!(
                f,
                "candidate `{candidate_id}` matched probe executed {found:?}, expected {expected:?}"
            ),
            Self::InvalidContribution {
                candidate_id,
                transformation,
                value,
            } => write!(
                f,
                concat!(
                    "candidate `{candidate_id}` {transformation:?} contribution {value} ",
                    "is outside the frozen local-transition scale"
                )
            ),
            Self::InvalidOperatorEnvelope {
                candidate_id,
                lower,
                upper,
            } => write!(
                f,
                "candidate `{candidate_id}` produced invalid operator envelope [{lower}, {upper}]"
            ),
            Self::AxisAlreadyObserved(id) => write!(
                f,
                concat!(
                    "candidate `{id}` already has IntegrationProxy evidence; ",
                    "matched probe will not overwrite it"
                )
            ),
            Self::CrossAxisMutation(id) => write!(
                f,
                "matched integration probe mutated a non-integration objective for candidate `{id}`"
            ),
            Self::ActivePrimitive(err) => write!(f, "active primitive evidence error: {err}"),
            Self::Objective(err) => write!(f, "objective evidence error: {err}"),
            Self::Planner(err) => write!(f, "V3 planning error: {err}"),
        }
    }
}

impl std::error::Error for MatchedIntegrationProbeError {}

impl From<ActivePrimitiveEvidenceError> for MatchedIntegrationProbeError {
    fn from(value: ActivePrimitiveEvidenceError) -> Self {
        Self::ActivePrimitive(value)
    }
}

impl From<ObjectiveEvidenceError> for MatchedIntegrationProbeError {
    fn from(value: ObjectiveEvidenceError) -> Self {
        Self::Objective(value)
    }
}

impl From<EvidenceSeekingPlannerError> for MatchedIntegrationProbeError {
    fn from(value: EvidenceSeekingPlannerError) -> Self {
        Self::Planner(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::reasoning_evidence_seeking::EvidenceSeekingOutcome;
    use crate::consciousness::context_aware_evolution::ReasoningContext;
    use crate::consciousness::{ActivationReason, ActivePrimitive};
    use symthaea_core::hdc::primitive_system::PrimitiveSystem;

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

    fn hypothesis() -> ContextHypothesis {
        ContextHypothesis {
            context: ReasoningContext::CreativeExploration,
            support: 0.9,
            source: "fixture-context".into(),
            evidence_refs: vec!["query".into()],
        }
    }

    #[test]
    fn xor_bind_degeneracy_is_explicit_and_excluded_from_panel() {
        let active_primitive = active("NSM_KNOW", 0.8);
        let primitive = &active_primitive.primitive;
        let mut first = ReasoningChain::new(BinaryHV::random(40));
        let mut second = ReasoningChain::new(BinaryHV::random(41));
        first
            .execute_primitive(primitive, TransformationType::Bind)
            .unwrap();
        second
            .execute_primitive(primitive, TransformationType::Bind)
            .unwrap();
        assert_eq!(
            first.executions[0].phi_contribution.to_bits(),
            second.executions[0].phi_contribution.to_bits()
        );
        assert!(!MATCHED_TRANSFORMATION_PANEL.contains(&TransformationType::Bind));
        assert!(!MATCHED_TRANSFORMATION_PANEL.contains(&TransformationType::Permute));
    }

    #[test]
    fn same_input_candidate_and_panel_are_deterministic() {
        let actives = [active("NSM_KNOW", 0.8)];
        let ids = vec!["NSM_KNOW".into()];
        let input = BinaryHV::random(42);
        let (_, first) = probe_active_primitive_integration(&actives, &ids, input).unwrap();
        let (_, second) = probe_active_primitive_integration(&actives, &ids, input).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn every_candidate_uses_identical_input_and_frozen_panel() {
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.7)];
        let ids = vec!["NSM_KNOW".into(), "NSM_DO".into()];
        let input = BinaryHV::random(43);
        let (_, report) = probe_active_primitive_integration(&actives, &ids, input).unwrap();
        assert_eq!(report.profiles.len(), 2);
        assert_eq!(
            report.transformation_panel,
            MATCHED_TRANSFORMATION_PANEL.to_vec()
        );
        for profile in &report.profiles {
            assert_eq!(profile.input_digest, report.input_digest);
            assert_eq!(profile.operator_observations.len(), MATCHED_TRANSFORMATION_PANEL.len());
            assert_eq!(
                profile
                    .operator_observations
                    .iter()
                    .map(|observation| observation.transformation)
                    .collect::<Vec<_>>(),
                MATCHED_TRANSFORMATION_PANEL.to_vec()
            );
        }
    }

    #[test]
    fn probe_observes_only_bounded_integration_axis() {
        let actives = [active("NSM_KNOW", 0.8)];
        let ids = vec!["NSM_KNOW".into()];
        let (_, report) =
            probe_active_primitive_integration(&actives, &ids, BinaryHV::random(44)).unwrap();
        assert_eq!(report.candidates[0].observed_axes(), 1);
        assert!(report.candidates[0].integration_proxy.is_observed());
        assert!(!report.candidates[0].harmonic_alignment.is_observed());
        assert!(!report.candidates[0].epistemic_grounding.is_observed());
        let profile = &report.profiles[0];
        let min = profile
            .operator_observations
            .iter()
            .map(|observation| observation.normalized_integration_proxy)
            .reduce(f64::min)
            .unwrap();
        let max = profile
            .operator_observations
            .iter()
            .map(|observation| observation.normalized_integration_proxy)
            .reduce(f64::max)
            .unwrap();
        assert_eq!(profile.integration_lower.to_bits(), min.to_bits());
        assert_eq!(profile.integration_upper.to_bits(), max.to_bits());
    }

    #[test]
    fn commitments_bind_input_candidate_and_all_operator_executions() {
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.7)];
        let know = vec!["NSM_KNOW".into()];
        let do_id = vec!["NSM_DO".into()];
        let input_a = BinaryHV::random(45);
        let input_b = BinaryHV::random(46);
        let (_, first) = probe_active_primitive_integration(&actives, &know, input_a).unwrap();
        let (_, changed_input) =
            probe_active_primitive_integration(&actives, &know, input_b).unwrap();
        let (_, changed_candidate) =
            probe_active_primitive_integration(&actives, &do_id, input_a).unwrap();
        assert_ne!(first.input_digest, changed_input.input_digest);
        assert_ne!(
            first.profiles[0].panel_commitment,
            changed_input.profiles[0].panel_commitment
        );
        assert_ne!(
            first.profiles[0].primitive_encoding_digest,
            changed_candidate.profiles[0].primitive_encoding_digest
        );
        assert_ne!(
            first.profiles[0].panel_commitment,
            changed_candidate.profiles[0].panel_commitment
        );
        assert!(first.profiles[0]
            .operator_observations
            .iter()
            .all(|observation| !observation.output_digest.is_empty()));
    }

    #[test]
    fn missing_active_candidate_fails_closed() {
        let actives = [active("NSM_KNOW", 0.8)];
        let ids = vec!["NSM_DO".into()];
        assert!(
            probe_active_primitive_integration(&actives, &ids, BinaryHV::random(47)).is_err()
        );
    }

    #[test]
    fn matched_panel_composes_with_v3_without_fabricating_other_axes() {
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.7)];
        let ids = vec!["NSM_KNOW".into(), "NSM_DO".into()];
        let (_, probe, plan) = plan_active_primitive_with_matched_integration(
            &[hypothesis()],
            ContextCompetitionPolicy::development_v1(),
            &actives,
            &ids,
            BinaryHV::random(48),
        )
        .unwrap();
        assert_eq!(
            probe
                .candidates
                .iter()
                .map(|candidate| candidate.observed_axes())
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
        assert!(matches!(
            plan.outcome,
            EvidenceSeekingOutcome::NeedEvidence { .. }
        ));
    }
}
