// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1G1R: strict rederivation of the registered two-sided Wald interval.
//!
//! P1G1 already validates the registered model/input identity, convergence,
//! singularity/Hessian state, Wald statistic, one-sided p-value, odds ratio,
//! multiplicity handling, and null decision. Its v1 confidence-interval check
//! verifies only finiteness/order/containment. This additive validator requires
//! P1G1 to pass first, then independently derives the two-sided standard-normal
//! critical value from the frozen P1B confidence level and verifies both
//! reported interval endpoints as beta +/- z*SE.

use crate::evidence_digest::{
    perceptual_analysis_plan::FrozenPerceptualAnalysisSpecV1,
    perceptual_glmm_result::{
        CompletedRegisteredGlmmFitV1, FrozenPerceptualGlmmExecutionBundleV1,
        PerceptualGlmmExecutionIssueV1, RegisteredEndpointGlmmExecutionV1,
        validate_perceptual_glmm_execution_bundle,
    },
    perceptual_model_input::FrozenPerceptualModelInputBundleV1,
    perceptual_unblinding::FrozenPerceptualDerivedDatasetV1,
};
use serde::{Deserialize, Serialize};

pub const PERCEPTUAL_GLMM_STRICT_INTERVAL_VERSION: &str =
    "mel003-perceptual-glmm-strict-interval-v1";
pub const WALD_INTERVAL_RELATIVE_TOLERANCE: f64 = 1.0e-8;
pub const NORMAL_CRITICAL_BISECTION_ITERATIONS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualGlmmStrictIntervalIssueV1 {
    BaseContractRejected {
        issues: Vec<PerceptualGlmmExecutionIssueV1>,
    },
    InvalidConfidenceLevel,
    WaldIntervalMismatch { endpoint: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerceptualGlmmStrictIntervalReceiptV1 {
    pub receipt_version: String,
    pub glmm_execution_bundle_sha256: String,
    pub confidence_level: f64,
    pub two_sided_standard_normal_critical_value: f64,
    pub primary_interval_rederived: bool,
    pub key_secondary_interval_rederived: bool,
}

pub fn validate_strict_perceptual_glmm_execution_bundle(
    dataset: &FrozenPerceptualDerivedDatasetV1,
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
    model_inputs: &FrozenPerceptualModelInputBundleV1,
    bundle: &FrozenPerceptualGlmmExecutionBundleV1,
) -> Result<PerceptualGlmmStrictIntervalReceiptV1, Vec<PerceptualGlmmStrictIntervalIssueV1>> {
    let base_issues = validate_perceptual_glmm_execution_bundle(
        dataset,
        analysis_spec,
        model_inputs,
        bundle,
    );
    if !base_issues.is_empty() {
        return Err(vec![
            PerceptualGlmmStrictIntervalIssueV1::BaseContractRejected {
                issues: base_issues,
            },
        ]);
    }

    let critical = two_sided_standard_normal_critical_value(analysis_spec.confidence_level);
    if !critical.is_finite() || critical <= 0.0 {
        return Err(vec![
            PerceptualGlmmStrictIntervalIssueV1::InvalidConfidenceLevel,
        ]);
    }

    let mut issues = Vec::new();
    let primary_interval_rederived = validate_endpoint_interval(
        "primary",
        &bundle.primary,
        critical,
        &mut issues,
    );
    let key_secondary_interval_rederived = validate_endpoint_interval(
        "key_secondary",
        &bundle.key_secondary,
        critical,
        &mut issues,
    );

    if issues.is_empty() {
        Ok(PerceptualGlmmStrictIntervalReceiptV1 {
            receipt_version: PERCEPTUAL_GLMM_STRICT_INTERVAL_VERSION.into(),
            glmm_execution_bundle_sha256: bundle.bundle_sha256.clone(),
            confidence_level: analysis_spec.confidence_level,
            two_sided_standard_normal_critical_value: critical,
            primary_interval_rederived,
            key_secondary_interval_rederived,
        })
    } else {
        Err(issues)
    }
}

fn validate_endpoint_interval(
    endpoint: &str,
    execution: &RegisteredEndpointGlmmExecutionV1,
    critical: f64,
    issues: &mut Vec<PerceptualGlmmStrictIntervalIssueV1>,
) -> bool {
    match execution {
        RegisteredEndpointGlmmExecutionV1::Inconclusive(_) => true,
        RegisteredEndpointGlmmExecutionV1::Completed(result) => {
            if completed_wald_interval_matches(result, critical) {
                true
            } else {
                issues.push(
                    PerceptualGlmmStrictIntervalIssueV1::WaldIntervalMismatch {
                        endpoint: endpoint.into(),
                    },
                );
                false
            }
        }
    }
}

pub fn completed_wald_interval_matches(
    result: &CompletedRegisteredGlmmFitV1,
    two_sided_critical_value: f64,
) -> bool {
    if !two_sided_critical_value.is_finite() || two_sided_critical_value <= 0.0 {
        return false;
    }
    let half_width = two_sided_critical_value * result.standard_error;
    let expected_lower = result.intercept_log_odds - half_width;
    let expected_upper = result.intercept_log_odds + half_width;
    approximately_equal(
        result.confidence_interval_lower_log_odds,
        expected_lower,
        WALD_INTERVAL_RELATIVE_TOLERANCE,
    ) && approximately_equal(
        result.confidence_interval_upper_log_odds,
        expected_upper,
        WALD_INTERVAL_RELATIVE_TOLERANCE,
    )
}

/// Deterministically derives z* for a two-sided Wald interval at confidence c:
/// survival(z*) = (1-c)/2.
pub fn two_sided_standard_normal_critical_value(confidence_level: f64) -> f64 {
    if !confidence_level.is_finite()
        || confidence_level <= 0.5
        || confidence_level >= 1.0
    {
        return f64::NAN;
    }
    let target_upper_tail = (1.0 - confidence_level) * 0.5;
    let mut lower = 0.0;
    let mut upper = 12.0;
    for _ in 0..NORMAL_CRITICAL_BISECTION_ITERATIONS {
        let midpoint = lower + (upper - lower) * 0.5;
        if one_sided_standard_normal_survival(midpoint) > target_upper_tail {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    lower + (upper - lower) * 0.5
}

fn approximately_equal(found: f64, expected: f64, tolerance: f64) -> bool {
    if !found.is_finite() || !expected.is_finite() {
        return false;
    }
    let scale = 1.0f64.max(found.abs()).max(expected.abs());
    (found - expected).abs() <= tolerance * scale
}

/// Same deterministic approximation family used by P1G1 for its Wald
/// p-value integrity check. Kept local so this repair does not widen the public
/// surface of the frozen P1G1 module.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::{
        perceptual_glmm_result::{
            RegisteredNullDecisionV1, PERCEPTUAL_GLMM_ENDPOINT_RESULT_VERSION,
        },
        perceptual_model_input::RegisteredEndpointRoleV1,
        perceptual_study_protocol::PerceptualTaskV1,
    };

    fn completed_fit() -> CompletedRegisteredGlmmFitV1 {
        let beta: f64 = 0.4;
        let se: f64 = 0.2;
        let critical = two_sided_standard_normal_critical_value(0.95);
        CompletedRegisteredGlmmFitV1 {
            result_version: PERCEPTUAL_GLMM_ENDPOINT_RESULT_VERSION.into(),
            role: RegisteredEndpointRoleV1::Primary,
            task: PerceptualTaskV1::AbxDiscrimination,
            input_sha256: "a".repeat(64),
            observation_count: 384,
            participant_count: 48,
            item_count: 8,
            intercept_log_odds: beta,
            standard_error: se,
            wald_z: beta / se,
            one_sided_p_value: one_sided_standard_normal_survival(beta / se),
            multiplicity_adjusted_p_value: None,
            confidence_level: 0.95,
            confidence_interval_lower_log_odds: beta - critical * se,
            confidence_interval_upper_log_odds: beta + critical * se,
            odds_ratio: beta.exp(),
            participant_random_intercept_variance: 0.25,
            item_random_intercept_variance: 0.10,
            log_likelihood: -200.0,
            optimizer_iterations: 12,
            maximum_absolute_gradient: 1.0e-7,
            converged: true,
            singular_fit: false,
            hessian_positive_definite: true,
            fallback_model_executed: false,
            decision: RegisteredNullDecisionV1::RejectedInRegisteredDirection,
            raw_result_artifact_sha256: "b".repeat(64),
            stdout_sha256: "c".repeat(64),
            stderr_sha256: "d".repeat(64),
        }
    }

    #[test]
    fn ninety_five_percent_critical_value_matches_known_normal_quantile() {
        let critical = two_sided_standard_normal_critical_value(0.95);
        assert!((critical - 1.959_964).abs() < 5.0e-6);
    }

    #[test]
    fn exact_registered_wald_interval_matches() {
        let fit = completed_fit();
        let critical = two_sided_standard_normal_critical_value(fit.confidence_level);
        assert!(completed_wald_interval_matches(&fit, critical));
    }

    #[test]
    fn arbitrarily_widened_interval_is_rejected_even_if_it_contains_beta() {
        let mut fit = completed_fit();
        fit.confidence_interval_lower_log_odds -= 10.0;
        fit.confidence_interval_upper_log_odds += 10.0;
        let critical = two_sided_standard_normal_critical_value(fit.confidence_level);
        assert!(!completed_wald_interval_matches(&fit, critical));
    }

    #[test]
    fn shifted_same_width_interval_is_rejected() {
        let mut fit = completed_fit();
        fit.confidence_interval_lower_log_odds += 0.01;
        fit.confidence_interval_upper_log_odds += 0.01;
        let critical = two_sided_standard_normal_critical_value(fit.confidence_level);
        assert!(!completed_wald_interval_matches(&fit, critical));
    }
}
