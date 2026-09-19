// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1H: bounded claim adjudication for the first perceptual study.
//!
//! This layer does not perform statistics. It converts only the already-
//! validated P1G1 registered endpoint states into narrow study conclusions and
//! carries P1A's machine-readable nonclaims into the final human-evidence
//! artifact. Primary and key-secondary conclusions remain independent.

use crate::evidence_digest::{
    canonical_json_sha256,
    perceptual_analysis_plan::FrozenPerceptualAnalysisSpecV1,
    perceptual_glmm_result::{
        FrozenPerceptualGlmmExecutionBundleV1, RegisteredEndpointGlmmExecutionV1,
        RegisteredGlmmFailureKindV1, RegisteredNullDecisionV1,
        validate_perceptual_glmm_execution_bundle,
    },
    perceptual_model_input::FrozenPerceptualModelInputBundleV1,
    perceptual_study_protocol::{
        ForbiddenPerceptualClaimV1, FrozenPerceptualStudyProtocolV1, MEL003_FIXED_SEEDS,
        PerceptualTaskV1,
    },
    perceptual_unblinding::FrozenPerceptualDerivedDatasetV1,
};
use serde::{Deserialize, Serialize};

pub const PERCEPTUAL_CLAIM_ADJUDICATION_VERSION: &str =
    "mel003-perceptual-claim-adjudication-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualClaimRoleV1 {
    Primary,
    KeySecondary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundedPerceptualClaimV1 {
    /// Registered listeners discriminated the exact registered baseline and
    /// intervention stimuli above chance on the ABX endpoint.
    RegisteredStimuliDiscriminatedAboveChance,
    /// Registered listeners selected the intervention above chance as the arm
    /// containing more of the preregistered local re-articulation/activity cue.
    RegisteredInterventionSelectedForRearticulationAboveChance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualClaimNotEstablishedReasonV1 {
    RegisteredNullNotRejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail")]
pub enum PerceptualClaimStateV1 {
    EstablishedWithinRegisteredScope {
        claim: BoundedPerceptualClaimV1,
    },
    NotEstablished {
        reason: PerceptualClaimNotEstablishedReasonV1,
    },
    Inconclusive {
        registered_model_failure: RegisteredGlmmFailureKindV1,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredEndpointClaimV1 {
    pub role: PerceptualClaimRoleV1,
    pub task: PerceptualTaskV1,
    pub state: PerceptualClaimStateV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualClaimAdjudicationV1 {
    pub adjudication_version: String,
    pub protocol_sha256: String,
    pub derived_dataset_sha256: String,
    pub model_input_bundle_sha256: String,
    pub glmm_execution_bundle_sha256: String,
    pub registered_item_seeds: [u64; 8],
    pub primary: RegisteredEndpointClaimV1,
    pub key_secondary: RegisteredEndpointClaimV1,
    /// Exact P1A nonclaim registry, carried forward so the final result cannot
    /// silently shed its epistemic boundaries.
    pub forbidden_claims: Vec<ForbiddenPerceptualClaimV1>,
    pub adjudication_sha256: String,
}

#[derive(Serialize)]
struct ClaimAdjudicationCommitment<'a> {
    adjudication_version: &'a str,
    protocol_sha256: &'a str,
    derived_dataset_sha256: &'a str,
    model_input_bundle_sha256: &'a str,
    glmm_execution_bundle_sha256: &'a str,
    registered_item_seeds: [u64; 8],
    primary: &'a RegisteredEndpointClaimV1,
    key_secondary: &'a RegisteredEndpointClaimV1,
    forbidden_claims: &'a [ForbiddenPerceptualClaimV1],
}

pub fn perceptual_claim_adjudication_commitment(
    adjudication: &FrozenPerceptualClaimAdjudicationV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ClaimAdjudicationCommitment {
        adjudication_version: &adjudication.adjudication_version,
        protocol_sha256: &adjudication.protocol_sha256,
        derived_dataset_sha256: &adjudication.derived_dataset_sha256,
        model_input_bundle_sha256: &adjudication.model_input_bundle_sha256,
        glmm_execution_bundle_sha256: &adjudication.glmm_execution_bundle_sha256,
        registered_item_seeds: adjudication.registered_item_seeds,
        primary: &adjudication.primary,
        key_secondary: &adjudication.key_secondary,
        forbidden_claims: &adjudication.forbidden_claims,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualClaimAdjudicationIssueV1 {
    InvalidProtocol,
    InvalidGlmmExecution,
    ProtocolSerializationFailed,
    ProtocolDatasetDigestMismatch,
    AnalysisSpecSerializationFailed,
    ProtocolAnalysisSpecDigestMismatch,
    WrongVersion,
    ProtocolDigestMismatch,
    DerivedDatasetDigestMismatch,
    ModelInputBundleDigestMismatch,
    GlmmExecutionBundleDigestMismatch,
    WrongSeedPanel,
    PrimaryConclusionMismatch,
    SecondaryConclusionMismatch,
    ForbiddenClaimRegistryMismatch,
    InvalidDigest { field: String },
    AdjudicationDigestMismatch,
    SerializationFailed,
}

pub fn adjudicate_perceptual_claims(
    protocol: &FrozenPerceptualStudyProtocolV1,
    dataset: &FrozenPerceptualDerivedDatasetV1,
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
    model_inputs: &FrozenPerceptualModelInputBundleV1,
    glmm_execution: &FrozenPerceptualGlmmExecutionBundleV1,
) -> Result<FrozenPerceptualClaimAdjudicationV1, Vec<PerceptualClaimAdjudicationIssueV1>> {
    let issues = predecessor_issues(
        protocol,
        dataset,
        analysis_spec,
        model_inputs,
        glmm_execution,
    );
    if !issues.is_empty() {
        return Err(issues);
    }

    let mut forbidden_claims = protocol.forbidden_claims.clone();
    forbidden_claims.sort();

    let mut adjudication = FrozenPerceptualClaimAdjudicationV1 {
        adjudication_version: PERCEPTUAL_CLAIM_ADJUDICATION_VERSION.into(),
        protocol_sha256: canonical_json_sha256(protocol)
            .map_err(|_| vec![PerceptualClaimAdjudicationIssueV1::SerializationFailed])?,
        derived_dataset_sha256: dataset.dataset_sha256.clone(),
        model_input_bundle_sha256: model_inputs.bundle_sha256.clone(),
        glmm_execution_bundle_sha256: glmm_execution.bundle_sha256.clone(),
        registered_item_seeds: MEL003_FIXED_SEEDS,
        primary: conclusion_from_execution(
            PerceptualClaimRoleV1::Primary,
            analysis_spec.primary_task,
            &glmm_execution.primary,
        ),
        key_secondary: conclusion_from_execution(
            PerceptualClaimRoleV1::KeySecondary,
            analysis_spec.key_secondary_task,
            &glmm_execution.key_secondary,
        ),
        forbidden_claims,
        adjudication_sha256: String::new(),
    };
    adjudication.adjudication_sha256 = perceptual_claim_adjudication_commitment(&adjudication)
        .map_err(|_| vec![PerceptualClaimAdjudicationIssueV1::SerializationFailed])?;

    let validation = validate_perceptual_claim_adjudication(
        protocol,
        dataset,
        analysis_spec,
        model_inputs,
        glmm_execution,
        &adjudication,
    );
    if validation.is_empty() {
        Ok(adjudication)
    } else {
        Err(validation)
    }
}

pub fn validate_perceptual_claim_adjudication(
    protocol: &FrozenPerceptualStudyProtocolV1,
    dataset: &FrozenPerceptualDerivedDatasetV1,
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
    model_inputs: &FrozenPerceptualModelInputBundleV1,
    glmm_execution: &FrozenPerceptualGlmmExecutionBundleV1,
    adjudication: &FrozenPerceptualClaimAdjudicationV1,
) -> Vec<PerceptualClaimAdjudicationIssueV1> {
    let mut issues = predecessor_issues(
        protocol,
        dataset,
        analysis_spec,
        model_inputs,
        glmm_execution,
    );
    if adjudication.adjudication_version != PERCEPTUAL_CLAIM_ADJUDICATION_VERSION {
        issues.push(PerceptualClaimAdjudicationIssueV1::WrongVersion);
    }
    match canonical_json_sha256(protocol) {
        Ok(value) if value == adjudication.protocol_sha256 => {}
        Ok(_) => issues.push(PerceptualClaimAdjudicationIssueV1::ProtocolDigestMismatch),
        Err(_) => issues.push(PerceptualClaimAdjudicationIssueV1::ProtocolSerializationFailed),
    }
    if adjudication.derived_dataset_sha256 != dataset.dataset_sha256 {
        issues.push(PerceptualClaimAdjudicationIssueV1::DerivedDatasetDigestMismatch);
    }
    if adjudication.model_input_bundle_sha256 != model_inputs.bundle_sha256 {
        issues.push(PerceptualClaimAdjudicationIssueV1::ModelInputBundleDigestMismatch);
    }
    if adjudication.glmm_execution_bundle_sha256 != glmm_execution.bundle_sha256 {
        issues.push(PerceptualClaimAdjudicationIssueV1::GlmmExecutionBundleDigestMismatch);
    }
    if adjudication.registered_item_seeds != MEL003_FIXED_SEEDS {
        issues.push(PerceptualClaimAdjudicationIssueV1::WrongSeedPanel);
    }

    let expected_primary = conclusion_from_execution(
        PerceptualClaimRoleV1::Primary,
        analysis_spec.primary_task,
        &glmm_execution.primary,
    );
    if adjudication.primary != expected_primary {
        issues.push(PerceptualClaimAdjudicationIssueV1::PrimaryConclusionMismatch);
    }
    let expected_secondary = conclusion_from_execution(
        PerceptualClaimRoleV1::KeySecondary,
        analysis_spec.key_secondary_task,
        &glmm_execution.key_secondary,
    );
    if adjudication.key_secondary != expected_secondary {
        issues.push(PerceptualClaimAdjudicationIssueV1::SecondaryConclusionMismatch);
    }

    let mut expected_forbidden = protocol.forbidden_claims.clone();
    expected_forbidden.sort();
    let mut found_forbidden = adjudication.forbidden_claims.clone();
    found_forbidden.sort();
    if found_forbidden != expected_forbidden {
        issues.push(PerceptualClaimAdjudicationIssueV1::ForbiddenClaimRegistryMismatch);
    }

    for (field, digest) in [
        ("protocol_sha256", adjudication.protocol_sha256.as_str()),
        (
            "derived_dataset_sha256",
            adjudication.derived_dataset_sha256.as_str(),
        ),
        (
            "model_input_bundle_sha256",
            adjudication.model_input_bundle_sha256.as_str(),
        ),
        (
            "glmm_execution_bundle_sha256",
            adjudication.glmm_execution_bundle_sha256.as_str(),
        ),
        ("adjudication_sha256", adjudication.adjudication_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(PerceptualClaimAdjudicationIssueV1::InvalidDigest {
                field: field.into(),
            });
        }
    }
    match perceptual_claim_adjudication_commitment(adjudication) {
        Ok(value) if value == adjudication.adjudication_sha256 => {}
        Ok(_) => issues.push(PerceptualClaimAdjudicationIssueV1::AdjudicationDigestMismatch),
        Err(_) => issues.push(PerceptualClaimAdjudicationIssueV1::SerializationFailed),
    }
    issues
}

fn predecessor_issues(
    protocol: &FrozenPerceptualStudyProtocolV1,
    dataset: &FrozenPerceptualDerivedDatasetV1,
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
    model_inputs: &FrozenPerceptualModelInputBundleV1,
    glmm_execution: &FrozenPerceptualGlmmExecutionBundleV1,
) -> Vec<PerceptualClaimAdjudicationIssueV1> {
    let mut issues = Vec::new();
    if !protocol.validate().is_empty() {
        issues.push(PerceptualClaimAdjudicationIssueV1::InvalidProtocol);
    }
    if !validate_perceptual_glmm_execution_bundle(
        dataset,
        analysis_spec,
        model_inputs,
        glmm_execution,
    )
    .is_empty()
    {
        issues.push(PerceptualClaimAdjudicationIssueV1::InvalidGlmmExecution);
    }
    match canonical_json_sha256(protocol) {
        Ok(value) if value == dataset.protocol_sha256 => {}
        Ok(_) => issues.push(PerceptualClaimAdjudicationIssueV1::ProtocolDatasetDigestMismatch),
        Err(_) => issues.push(PerceptualClaimAdjudicationIssueV1::ProtocolSerializationFailed),
    }
    match canonical_json_sha256(analysis_spec) {
        Ok(value) if value == protocol.analysis_spec_sha256 => {}
        Ok(_) => {
            issues.push(PerceptualClaimAdjudicationIssueV1::ProtocolAnalysisSpecDigestMismatch)
        }
        Err(_) => issues.push(
            PerceptualClaimAdjudicationIssueV1::AnalysisSpecSerializationFailed,
        ),
    }
    issues
}

fn conclusion_from_execution(
    role: PerceptualClaimRoleV1,
    task: PerceptualTaskV1,
    execution: &RegisteredEndpointGlmmExecutionV1,
) -> RegisteredEndpointClaimV1 {
    let state = match execution {
        RegisteredEndpointGlmmExecutionV1::Completed(result) => match result.decision {
            RegisteredNullDecisionV1::RejectedInRegisteredDirection => {
                let claim = match role {
                    PerceptualClaimRoleV1::Primary => {
                        BoundedPerceptualClaimV1::RegisteredStimuliDiscriminatedAboveChance
                    }
                    PerceptualClaimRoleV1::KeySecondary => {
                        BoundedPerceptualClaimV1::RegisteredInterventionSelectedForRearticulationAboveChance
                    }
                };
                PerceptualClaimStateV1::EstablishedWithinRegisteredScope { claim }
            }
            RegisteredNullDecisionV1::NotRejected => PerceptualClaimStateV1::NotEstablished {
                reason: PerceptualClaimNotEstablishedReasonV1::RegisteredNullNotRejected,
            },
        },
        RegisteredEndpointGlmmExecutionV1::Inconclusive(result) => {
            PerceptualClaimStateV1::Inconclusive {
                registered_model_failure: result.failure_kind,
            }
        }
    };
    RegisteredEndpointClaimV1 { role, task, state }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::{
        perceptual_glmm_result::{
            CompletedRegisteredGlmmFitV1, InconclusiveRegisteredGlmmFitV1,
            PERCEPTUAL_GLMM_ENDPOINT_RESULT_VERSION,
        },
        perceptual_model_input::RegisteredEndpointRoleV1,
    };

    const DIGEST: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn completed(
        role: RegisteredEndpointRoleV1,
        task: PerceptualTaskV1,
        decision: RegisteredNullDecisionV1,
    ) -> RegisteredEndpointGlmmExecutionV1 {
        RegisteredEndpointGlmmExecutionV1::Completed(CompletedRegisteredGlmmFitV1 {
            result_version: PERCEPTUAL_GLMM_ENDPOINT_RESULT_VERSION.into(),
            role,
            task,
            input_sha256: DIGEST.into(),
            observation_count: 16,
            participant_count: 2,
            item_count: 8,
            intercept_log_odds: 0.0,
            standard_error: 1.0,
            wald_z: 0.0,
            one_sided_p_value: 0.5,
            multiplicity_adjusted_p_value: match role {
                RegisteredEndpointRoleV1::Primary => None,
                RegisteredEndpointRoleV1::KeySecondary => Some(0.5),
            },
            confidence_level: 0.95,
            confidence_interval_lower_log_odds: -1.96,
            confidence_interval_upper_log_odds: 1.96,
            odds_ratio: 1.0,
            participant_random_intercept_variance: 0.1,
            item_random_intercept_variance: 0.1,
            log_likelihood: -10.0,
            optimizer_iterations: 10,
            maximum_absolute_gradient: 1.0e-8,
            converged: true,
            singular_fit: false,
            hessian_positive_definite: true,
            fallback_model_executed: false,
            decision,
            raw_result_artifact_sha256: DIGEST.into(),
            stdout_sha256: DIGEST.into(),
            stderr_sha256: DIGEST.into(),
        })
    }

    #[test]
    fn primary_rejection_maps_only_to_discrimination_claim() {
        let execution = completed(
            RegisteredEndpointRoleV1::Primary,
            PerceptualTaskV1::AbxDiscrimination,
            RegisteredNullDecisionV1::RejectedInRegisteredDirection,
        );
        let conclusion = conclusion_from_execution(
            PerceptualClaimRoleV1::Primary,
            PerceptualTaskV1::AbxDiscrimination,
            &execution,
        );
        assert_eq!(
            conclusion.state,
            PerceptualClaimStateV1::EstablishedWithinRegisteredScope {
                claim: BoundedPerceptualClaimV1::RegisteredStimuliDiscriminatedAboveChance,
            }
        );
    }

    #[test]
    fn directional_rejection_does_not_become_preference_claim() {
        let execution = completed(
            RegisteredEndpointRoleV1::KeySecondary,
            PerceptualTaskV1::DirectionalRearticulation2Afc,
            RegisteredNullDecisionV1::RejectedInRegisteredDirection,
        );
        let conclusion = conclusion_from_execution(
            PerceptualClaimRoleV1::KeySecondary,
            PerceptualTaskV1::DirectionalRearticulation2Afc,
            &execution,
        );
        assert_eq!(
            conclusion.state,
            PerceptualClaimStateV1::EstablishedWithinRegisteredScope {
                claim: BoundedPerceptualClaimV1::RegisteredInterventionSelectedForRearticulationAboveChance,
            }
        );
    }

    #[test]
    fn non_rejection_stays_not_established() {
        let execution = completed(
            RegisteredEndpointRoleV1::Primary,
            PerceptualTaskV1::AbxDiscrimination,
            RegisteredNullDecisionV1::NotRejected,
        );
        assert!(matches!(
            conclusion_from_execution(
                PerceptualClaimRoleV1::Primary,
                PerceptualTaskV1::AbxDiscrimination,
                &execution,
            )
            .state,
            PerceptualClaimStateV1::NotEstablished { .. }
        ));
    }

    #[test]
    fn registered_model_failure_stays_inconclusive() {
        let execution = RegisteredEndpointGlmmExecutionV1::Inconclusive(
            InconclusiveRegisteredGlmmFitV1 {
                result_version: PERCEPTUAL_GLMM_ENDPOINT_RESULT_VERSION.into(),
                role: RegisteredEndpointRoleV1::Primary,
                task: PerceptualTaskV1::AbxDiscrimination,
                input_sha256: DIGEST.into(),
                observation_count: 16,
                participant_count: 2,
                item_count: 8,
                failure_kind: RegisteredGlmmFailureKindV1::FailedToConverge,
                diagnostic_code: "max-iterations".into(),
                fallback_model_executed: false,
                raw_result_artifact_sha256: DIGEST.into(),
                stdout_sha256: DIGEST.into(),
                stderr_sha256: DIGEST.into(),
            },
        );
        assert_eq!(
            conclusion_from_execution(
                PerceptualClaimRoleV1::Primary,
                PerceptualTaskV1::AbxDiscrimination,
                &execution,
            )
            .state,
            PerceptualClaimStateV1::Inconclusive {
                registered_model_failure: RegisteredGlmmFailureKindV1::FailedToConverge,
            }
        );
    }
}
