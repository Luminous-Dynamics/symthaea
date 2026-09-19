// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1G1: registered crossed-binomial GLMM execution/result contract.
//!
//! The actual fitter is an explicitly versioned dependency bound by P1B's
//! execution identity. This module does not implement or substitute a model.
//! It validates that the exact P1G0 inputs were analyzed under the registered
//! model and that each endpoint is represented as either a completed fit with
//! diagnostics or an inconclusive registered-model failure.

use crate::evidence_digest::{
    canonical_json_sha256,
    perceptual_analysis_plan::{
        FrozenPerceptualAnalysisSpecV1, ModelFailurePolicyV1,
        PerceptualAnalysisExecutionIdentityV1,
    },
    perceptual_model_input::{
        FrozenPerceptualModelInputBundleV1, RegisteredEndpointModelInputV1,
        RegisteredEndpointRoleV1, validate_perceptual_model_input_bundle,
    },
    perceptual_study_protocol::{PerceptualTaskV1, SecondaryMultiplicityPolicyV1},
    perceptual_unblinding::FrozenPerceptualDerivedDatasetV1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PERCEPTUAL_GLMM_EXECUTION_BUNDLE_VERSION: &str =
    "mel003-perceptual-glmm-execution-bundle-v1";
pub const PERCEPTUAL_GLMM_ENDPOINT_RESULT_VERSION: &str =
    "mel003-perceptual-glmm-endpoint-result-v1";
pub const WALD_P_VALUE_CHECK_TOLERANCE: f64 = 1.0e-6;
pub const NUMERIC_RELATIVE_TOLERANCE: f64 = 1.0e-8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegisteredInferenceProcedureV1 {
    /// Test the intercept against logit(0.5)=0 in the preregistered positive
    /// direction using the fitted intercept / standard-error Wald statistic.
    OneSidedWaldInterceptAgainstChanceLogit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegisteredIntervalProcedureV1 {
    TwoSidedWald,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegisteredNullDecisionV1 {
    RejectedInRegisteredDirection,
    NotRejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegisteredGlmmFailureKindV1 {
    FailedToConverge,
    SingularFit,
    InvalidHessian,
    NumericalFailure,
    EngineFailure,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletedRegisteredGlmmFitV1 {
    pub result_version: String,
    pub role: RegisteredEndpointRoleV1,
    pub task: PerceptualTaskV1,
    pub input_sha256: String,
    pub observation_count: usize,
    pub participant_count: usize,
    pub item_count: usize,
    pub intercept_log_odds: f64,
    pub standard_error: f64,
    pub wald_z: f64,
    pub one_sided_p_value: f64,
    /// Primary has no multiplicity adjustment. The v1 key-secondary family has
    /// one endpoint, so Holm-adjusted p must equal its raw p.
    pub multiplicity_adjusted_p_value: Option<f64>,
    pub confidence_level: f64,
    pub confidence_interval_lower_log_odds: f64,
    pub confidence_interval_upper_log_odds: f64,
    pub odds_ratio: f64,
    pub participant_random_intercept_variance: f64,
    pub item_random_intercept_variance: f64,
    pub log_likelihood: f64,
    pub optimizer_iterations: u32,
    pub maximum_absolute_gradient: f64,
    pub converged: bool,
    pub singular_fit: bool,
    pub hessian_positive_definite: bool,
    pub fallback_model_executed: bool,
    pub decision: RegisteredNullDecisionV1,
    pub raw_result_artifact_sha256: String,
    pub stdout_sha256: String,
    pub stderr_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InconclusiveRegisteredGlmmFitV1 {
    pub result_version: String,
    pub role: RegisteredEndpointRoleV1,
    pub task: PerceptualTaskV1,
    pub input_sha256: String,
    pub observation_count: usize,
    pub participant_count: usize,
    pub item_count: usize,
    pub failure_kind: RegisteredGlmmFailureKindV1,
    pub diagnostic_code: String,
    pub fallback_model_executed: bool,
    pub raw_result_artifact_sha256: String,
    pub stdout_sha256: String,
    pub stderr_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", content = "result", deny_unknown_fields)]
pub enum RegisteredEndpointGlmmExecutionV1 {
    Completed(CompletedRegisteredGlmmFitV1),
    Inconclusive(InconclusiveRegisteredGlmmFitV1),
}

impl RegisteredEndpointGlmmExecutionV1 {
    pub fn role(&self) -> RegisteredEndpointRoleV1 {
        match self {
            Self::Completed(result) => result.role,
            Self::Inconclusive(result) => result.role,
        }
    }

    pub fn task(&self) -> PerceptualTaskV1 {
        match self {
            Self::Completed(result) => result.task,
            Self::Inconclusive(result) => result.task,
        }
    }

    pub fn input_sha256(&self) -> &str {
        match self {
            Self::Completed(result) => &result.input_sha256,
            Self::Inconclusive(result) => &result.input_sha256,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualGlmmExecutionBundleV1 {
    pub bundle_version: String,
    pub model_input_bundle_sha256: String,
    pub analysis_spec_sha256: String,
    pub execution: PerceptualAnalysisExecutionIdentityV1,
    pub inference_procedure: RegisteredInferenceProcedureV1,
    pub interval_procedure: RegisteredIntervalProcedureV1,
    pub primary: RegisteredEndpointGlmmExecutionV1,
    pub key_secondary: RegisteredEndpointGlmmExecutionV1,
    pub bundle_sha256: String,
}

#[derive(Serialize)]
struct GlmmExecutionBundleCommitment<'a> {
    bundle_version: &'a str,
    model_input_bundle_sha256: &'a str,
    analysis_spec_sha256: &'a str,
    execution: &'a PerceptualAnalysisExecutionIdentityV1,
    inference_procedure: RegisteredInferenceProcedureV1,
    interval_procedure: RegisteredIntervalProcedureV1,
    primary: &'a RegisteredEndpointGlmmExecutionV1,
    key_secondary: &'a RegisteredEndpointGlmmExecutionV1,
}

pub fn glmm_execution_bundle_commitment(
    bundle: &FrozenPerceptualGlmmExecutionBundleV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&GlmmExecutionBundleCommitment {
        bundle_version: &bundle.bundle_version,
        model_input_bundle_sha256: &bundle.model_input_bundle_sha256,
        analysis_spec_sha256: &bundle.analysis_spec_sha256,
        execution: &bundle.execution,
        inference_procedure: bundle.inference_procedure,
        interval_procedure: bundle.interval_procedure,
        primary: &bundle.primary,
        key_secondary: &bundle.key_secondary,
    })
}

pub fn seal_perceptual_glmm_execution_bundle(
    bundle: &mut FrozenPerceptualGlmmExecutionBundleV1,
) -> Result<(), serde_json::Error> {
    bundle.bundle_sha256 = glmm_execution_bundle_commitment(bundle)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualGlmmExecutionIssueV1 {
    InvalidModelInputBundle,
    InvalidAnalysisSpec,
    AnalysisSpecSerializationFailed,
    WrongBundleVersion,
    ModelInputBundleDigestMismatch,
    AnalysisSpecDigestMismatch,
    ExecutionIdentityMismatch,
    WrongInferenceProcedure,
    WrongIntervalProcedure,
    EndpointRoleMismatch { endpoint: String },
    EndpointTaskMismatch { endpoint: String },
    EndpointInputDigestMismatch { endpoint: String },
    WrongEndpointResultVersion { endpoint: String },
    EndpointCountMismatch { endpoint: String, field: String },
    InvalidCompletedNumeric { endpoint: String, field: String },
    WaldStatisticMismatch { endpoint: String },
    WaldPValueMismatch { endpoint: String },
    InvalidConfidenceInterval { endpoint: String },
    OddsRatioMismatch { endpoint: String },
    CompletedFitNotConverged { endpoint: String },
    CompletedFitMarkedSingular { endpoint: String },
    CompletedFitInvalidHessian { endpoint: String },
    FallbackModelExecuted { endpoint: String },
    WrongPrimaryMultiplicityAdjustment,
    WrongSecondaryMultiplicityAdjustment,
    NullDecisionMismatch { endpoint: String },
    EmptyFailureDiagnostic { endpoint: String },
    InvalidDigest { field: String },
    BundleDigestMismatch,
    SerializationFailed,
}

pub fn validate_perceptual_glmm_execution_bundle(
    dataset: &FrozenPerceptualDerivedDatasetV1,
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
    model_inputs: &FrozenPerceptualModelInputBundleV1,
    bundle: &FrozenPerceptualGlmmExecutionBundleV1,
) -> Vec<PerceptualGlmmExecutionIssueV1> {
    let mut issues = Vec::new();
    if !validate_perceptual_model_input_bundle(dataset, analysis_spec, model_inputs).is_empty() {
        issues.push(PerceptualGlmmExecutionIssueV1::InvalidModelInputBundle);
    }
    if !analysis_spec.validate().is_empty() {
        issues.push(PerceptualGlmmExecutionIssueV1::InvalidAnalysisSpec);
    }
    if analysis_spec.model_failure_policy
        != ModelFailurePolicyV1::InconclusiveNoAutomaticSubstitution
    {
        issues.push(PerceptualGlmmExecutionIssueV1::InvalidAnalysisSpec);
    }
    if bundle.bundle_version != PERCEPTUAL_GLMM_EXECUTION_BUNDLE_VERSION {
        issues.push(PerceptualGlmmExecutionIssueV1::WrongBundleVersion);
    }
    if bundle.model_input_bundle_sha256 != model_inputs.bundle_sha256 {
        issues.push(PerceptualGlmmExecutionIssueV1::ModelInputBundleDigestMismatch);
    }
    match canonical_json_sha256(analysis_spec) {
        Ok(value) if value == bundle.analysis_spec_sha256 => {}
        Ok(_) => issues.push(PerceptualGlmmExecutionIssueV1::AnalysisSpecDigestMismatch),
        Err(_) => issues.push(PerceptualGlmmExecutionIssueV1::AnalysisSpecSerializationFailed),
    }
    if bundle.execution != analysis_spec.execution {
        issues.push(PerceptualGlmmExecutionIssueV1::ExecutionIdentityMismatch);
    }
    if bundle.inference_procedure
        != RegisteredInferenceProcedureV1::OneSidedWaldInterceptAgainstChanceLogit
    {
        issues.push(PerceptualGlmmExecutionIssueV1::WrongInferenceProcedure);
    }
    if bundle.interval_procedure != RegisteredIntervalProcedureV1::TwoSidedWald {
        issues.push(PerceptualGlmmExecutionIssueV1::WrongIntervalProcedure);
    }

    validate_endpoint_execution(
        "primary",
        RegisteredEndpointRoleV1::Primary,
        analysis_spec.primary_task,
        &model_inputs.primary,
        analysis_spec,
        &bundle.primary,
        &mut issues,
    );
    validate_endpoint_execution(
        "key_secondary",
        RegisteredEndpointRoleV1::KeySecondary,
        analysis_spec.key_secondary_task,
        &model_inputs.key_secondary,
        analysis_spec,
        &bundle.key_secondary,
        &mut issues,
    );

    for (field, digest) in [
        ("model_input_bundle_sha256", bundle.model_input_bundle_sha256.as_str()),
        ("analysis_spec_sha256", bundle.analysis_spec_sha256.as_str()),
        (
            "execution.analysis_program_sha256",
            bundle.execution.analysis_program_sha256.as_str(),
        ),
        ("execution.environment_sha256", bundle.execution.environment_sha256.as_str()),
        ("bundle_sha256", bundle.bundle_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(PerceptualGlmmExecutionIssueV1::InvalidDigest {
                field: field.into(),
            });
        }
    }
    match glmm_execution_bundle_commitment(bundle) {
        Ok(value) if value == bundle.bundle_sha256 => {}
        Ok(_) => issues.push(PerceptualGlmmExecutionIssueV1::BundleDigestMismatch),
        Err(_) => issues.push(PerceptualGlmmExecutionIssueV1::SerializationFailed),
    }
    issues
}

fn validate_endpoint_execution(
    endpoint_name: &str,
    expected_role: RegisteredEndpointRoleV1,
    expected_task: PerceptualTaskV1,
    input: &RegisteredEndpointModelInputV1,
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
    execution: &RegisteredEndpointGlmmExecutionV1,
    issues: &mut Vec<PerceptualGlmmExecutionIssueV1>,
) {
    if execution.role() != expected_role {
        issues.push(PerceptualGlmmExecutionIssueV1::EndpointRoleMismatch {
            endpoint: endpoint_name.into(),
        });
    }
    if execution.task() != expected_task {
        issues.push(PerceptualGlmmExecutionIssueV1::EndpointTaskMismatch {
            endpoint: endpoint_name.into(),
        });
    }
    if execution.input_sha256() != input.input_sha256 {
        issues.push(PerceptualGlmmExecutionIssueV1::EndpointInputDigestMismatch {
            endpoint: endpoint_name.into(),
        });
    }

    let expected_observations = input.observations.len();
    let expected_participants = input
        .observations
        .iter()
        .map(|row| row.participant_token.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let expected_items = input
        .observations
        .iter()
        .map(|row| (row.item_id.as_str(), row.seed))
        .collect::<BTreeSet<_>>()
        .len();

    match execution {
        RegisteredEndpointGlmmExecutionV1::Completed(result) => {
            if result.result_version != PERCEPTUAL_GLMM_ENDPOINT_RESULT_VERSION {
                issues.push(PerceptualGlmmExecutionIssueV1::WrongEndpointResultVersion {
                    endpoint: endpoint_name.into(),
                });
            }
            validate_counts(
                endpoint_name,
                result.observation_count,
                result.participant_count,
                result.item_count,
                expected_observations,
                expected_participants,
                expected_items,
                issues,
            );
            validate_completed_fit(endpoint_name, expected_role, result, analysis_spec, issues);
        }
        RegisteredEndpointGlmmExecutionV1::Inconclusive(result) => {
            if result.result_version != PERCEPTUAL_GLMM_ENDPOINT_RESULT_VERSION {
                issues.push(PerceptualGlmmExecutionIssueV1::WrongEndpointResultVersion {
                    endpoint: endpoint_name.into(),
                });
            }
            validate_counts(
                endpoint_name,
                result.observation_count,
                result.participant_count,
                result.item_count,
                expected_observations,
                expected_participants,
                expected_items,
                issues,
            );
            if result.diagnostic_code.trim().is_empty() {
                issues.push(PerceptualGlmmExecutionIssueV1::EmptyFailureDiagnostic {
                    endpoint: endpoint_name.into(),
                });
            }
            if result.fallback_model_executed {
                issues.push(PerceptualGlmmExecutionIssueV1::FallbackModelExecuted {
                    endpoint: endpoint_name.into(),
                });
            }
            validate_endpoint_digests(
                endpoint_name,
                &result.raw_result_artifact_sha256,
                &result.stdout_sha256,
                &result.stderr_sha256,
                issues,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_counts(
    endpoint_name: &str,
    observations: usize,
    participants: usize,
    items: usize,
    expected_observations: usize,
    expected_participants: usize,
    expected_items: usize,
    issues: &mut Vec<PerceptualGlmmExecutionIssueV1>,
) {
    for (field, found, expected) in [
        ("observation_count", observations, expected_observations),
        ("participant_count", participants, expected_participants),
        ("item_count", items, expected_items),
    ] {
        if found != expected {
            issues.push(PerceptualGlmmExecutionIssueV1::EndpointCountMismatch {
                endpoint: endpoint_name.into(),
                field: field.into(),
            });
        }
    }
}

fn validate_completed_fit(
    endpoint_name: &str,
    role: RegisteredEndpointRoleV1,
    result: &CompletedRegisteredGlmmFitV1,
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
    issues: &mut Vec<PerceptualGlmmExecutionIssueV1>,
) {
    for (field, value) in [
        ("intercept_log_odds", result.intercept_log_odds),
        ("standard_error", result.standard_error),
        ("wald_z", result.wald_z),
        ("one_sided_p_value", result.one_sided_p_value),
        ("confidence_level", result.confidence_level),
        (
            "confidence_interval_lower_log_odds",
            result.confidence_interval_lower_log_odds,
        ),
        (
            "confidence_interval_upper_log_odds",
            result.confidence_interval_upper_log_odds,
        ),
        ("odds_ratio", result.odds_ratio),
        (
            "participant_random_intercept_variance",
            result.participant_random_intercept_variance,
        ),
        (
            "item_random_intercept_variance",
            result.item_random_intercept_variance,
        ),
        ("log_likelihood", result.log_likelihood),
        ("maximum_absolute_gradient", result.maximum_absolute_gradient),
    ] {
        if !value.is_finite() {
            issues.push(PerceptualGlmmExecutionIssueV1::InvalidCompletedNumeric {
                endpoint: endpoint_name.into(),
                field: field.into(),
            });
        }
    }
    if result.standard_error <= 0.0
        || result.one_sided_p_value < 0.0
        || result.one_sided_p_value > 1.0
        || result.confidence_level != analysis_spec.confidence_level
        || result.odds_ratio <= 0.0
        || result.participant_random_intercept_variance < 0.0
        || result.item_random_intercept_variance < 0.0
        || result.maximum_absolute_gradient < 0.0
        || result.optimizer_iterations == 0
    {
        issues.push(PerceptualGlmmExecutionIssueV1::InvalidCompletedNumeric {
            endpoint: endpoint_name.into(),
            field: "completed_fit_constraints".into(),
        });
    }

    let expected_z = result.intercept_log_odds / result.standard_error;
    if !approximately_equal(result.wald_z, expected_z, NUMERIC_RELATIVE_TOLERANCE) {
        issues.push(PerceptualGlmmExecutionIssueV1::WaldStatisticMismatch {
            endpoint: endpoint_name.into(),
        });
    }
    let expected_p = one_sided_standard_normal_survival(result.wald_z);
    if !approximately_equal(
        result.one_sided_p_value,
        expected_p,
        WALD_P_VALUE_CHECK_TOLERANCE,
    ) {
        issues.push(PerceptualGlmmExecutionIssueV1::WaldPValueMismatch {
            endpoint: endpoint_name.into(),
        });
    }
    if result.confidence_interval_lower_log_odds > result.intercept_log_odds
        || result.confidence_interval_upper_log_odds < result.intercept_log_odds
        || result.confidence_interval_lower_log_odds
            > result.confidence_interval_upper_log_odds
    {
        issues.push(PerceptualGlmmExecutionIssueV1::InvalidConfidenceInterval {
            endpoint: endpoint_name.into(),
        });
    }
    let expected_or = result.intercept_log_odds.exp();
    if !approximately_equal(result.odds_ratio, expected_or, NUMERIC_RELATIVE_TOLERANCE) {
        issues.push(PerceptualGlmmExecutionIssueV1::OddsRatioMismatch {
            endpoint: endpoint_name.into(),
        });
    }
    if !result.converged {
        issues.push(PerceptualGlmmExecutionIssueV1::CompletedFitNotConverged {
            endpoint: endpoint_name.into(),
        });
    }
    if result.singular_fit {
        issues.push(PerceptualGlmmExecutionIssueV1::CompletedFitMarkedSingular {
            endpoint: endpoint_name.into(),
        });
    }
    if !result.hessian_positive_definite {
        issues.push(PerceptualGlmmExecutionIssueV1::CompletedFitInvalidHessian {
            endpoint: endpoint_name.into(),
        });
    }
    if result.fallback_model_executed {
        issues.push(PerceptualGlmmExecutionIssueV1::FallbackModelExecuted {
            endpoint: endpoint_name.into(),
        });
    }

    let decision_p = match role {
        RegisteredEndpointRoleV1::Primary => {
            if result.multiplicity_adjusted_p_value.is_some() {
                issues.push(PerceptualGlmmExecutionIssueV1::WrongPrimaryMultiplicityAdjustment);
            }
            result.one_sided_p_value
        }
        RegisteredEndpointRoleV1::KeySecondary => {
            let adjusted = result.multiplicity_adjusted_p_value;
            if analysis_spec.secondary_multiplicity
                != SecondaryMultiplicityPolicyV1::HolmWithinRegisteredSecondaryFamily
                || adjusted.is_none_or(|value| {
                    !value.is_finite()
                        || value < 0.0
                        || value > 1.0
                        || !approximately_equal(
                            value,
                            result.one_sided_p_value,
                            NUMERIC_RELATIVE_TOLERANCE,
                        )
                })
            {
                issues.push(
                    PerceptualGlmmExecutionIssueV1::WrongSecondaryMultiplicityAdjustment,
                );
            }
            adjusted.unwrap_or(result.one_sided_p_value)
        }
    };

    let expected_decision = if result.intercept_log_odds > 0.0
        && decision_p < analysis_spec.alpha
    {
        RegisteredNullDecisionV1::RejectedInRegisteredDirection
    } else {
        RegisteredNullDecisionV1::NotRejected
    };
    if result.decision != expected_decision {
        issues.push(PerceptualGlmmExecutionIssueV1::NullDecisionMismatch {
            endpoint: endpoint_name.into(),
        });
    }

    validate_endpoint_digests(
        endpoint_name,
        &result.raw_result_artifact_sha256,
        &result.stdout_sha256,
        &result.stderr_sha256,
        issues,
    );
}

fn validate_endpoint_digests(
    endpoint_name: &str,
    raw_result: &str,
    stdout: &str,
    stderr: &str,
    issues: &mut Vec<PerceptualGlmmExecutionIssueV1>,
) {
    for (field, digest) in [
        ("raw_result_artifact_sha256", raw_result),
        ("stdout_sha256", stdout),
        ("stderr_sha256", stderr),
    ] {
        if !is_sha256(digest) {
            issues.push(PerceptualGlmmExecutionIssueV1::InvalidDigest {
                field: format!("{endpoint_name}.{field}"),
            });
        }
    }
}

fn approximately_equal(found: f64, expected: f64, tolerance: f64) -> bool {
    if !found.is_finite() || !expected.is_finite() {
        return false;
    }
    let scale = 1.0f64.max(found.abs()).max(expected.abs());
    (found - expected).abs() <= tolerance * scale
}

/// Deterministic approximation to the upper tail of a standard normal.
/// Accuracy is sufficient only as an integrity cross-check of the registered
/// runner's reported one-sided Wald p-value; this is not a model fitter.
fn one_sided_standard_normal_survival(z: f64) -> f64 {
    if !z.is_finite() {
        return f64::NAN;
    }
    let x = z.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * x);
    let polynomial = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937
                    + t * (-1.821_255_978 + t * 1.330_274_429))));
    let density = 0.398_942_280_401_432_7 * (-0.5 * x * x).exp();
    let upper_for_positive = density * polynomial;
    if z >= 0.0 {
        upper_for_positive.clamp(0.0, 1.0)
    } else {
        (1.0 - upper_for_positive).clamp(0.0, 1.0)
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_normal_integrity_check_matches_known_values() {
        assert!(approximately_equal(
            one_sided_standard_normal_survival(0.0),
            0.5,
            WALD_P_VALUE_CHECK_TOLERANCE,
        ));
        assert!(approximately_equal(
            one_sided_standard_normal_survival(1.96),
            0.024_997_9,
            5.0e-6,
        ));
        assert!(approximately_equal(
            one_sided_standard_normal_survival(-1.96),
            0.975_002_1,
            5.0e-6,
        ));
    }

    #[test]
    fn decision_rule_requires_positive_effect_and_registered_p_threshold() {
        let beta: f64 = 0.4;
        let se: f64 = 0.2;
        let p = one_sided_standard_normal_survival(beta / se);
        assert!(p < 0.05);
        assert!(beta > 0.0);
        assert_eq!(
            if beta > 0.0 && p < 0.05 {
                RegisteredNullDecisionV1::RejectedInRegisteredDirection
            } else {
                RegisteredNullDecisionV1::NotRejected
            },
            RegisteredNullDecisionV1::RejectedInRegisteredDirection
        );
    }

    #[test]
    fn completed_and_inconclusive_are_disjoint_result_shapes() {
        let inconclusive = RegisteredEndpointGlmmExecutionV1::Inconclusive(
            InconclusiveRegisteredGlmmFitV1 {
                result_version: PERCEPTUAL_GLMM_ENDPOINT_RESULT_VERSION.into(),
                role: RegisteredEndpointRoleV1::Primary,
                task: PerceptualTaskV1::AbxDiscrimination,
                input_sha256: "a".repeat(64),
                observation_count: 16,
                participant_count: 2,
                item_count: 8,
                failure_kind: RegisteredGlmmFailureKindV1::FailedToConverge,
                diagnostic_code: "optimizer-max-iterations".into(),
                fallback_model_executed: false,
                raw_result_artifact_sha256: "b".repeat(64),
                stdout_sha256: "c".repeat(64),
                stderr_sha256: "d".repeat(64),
            },
        );
        assert!(matches!(
            inconclusive,
            RegisteredEndpointGlmmExecutionV1::Inconclusive(_)
        ));
    }
}
