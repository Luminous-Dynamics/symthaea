// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1BR: make perceptual power semantics and protocol binding load-bearing.
//!
//! P1B already freezes a robust simulation grid and chooses the smallest N whose
//! one-sided Wilson lower bound reaches target power in every registered
//! heterogeneity scenario. This layer closes two remaining semantic gaps:
//!
//! 1. the human-scale SESOI is explicitly **marginal**, not merely the
//!    conditional probability at zero random effects; and
//! 2. the concrete power-plan artifact is bound back into the protocol that may
//!    later authorize recruitment.
//!
//! The registered v1 generative model is a crossed Bernoulli/logit model with
//! independent Gaussian participant and item random intercepts. One participant
//! effect is reused across that participant's eight items and one item effect is
//! reused across all participants. For each heterogeneity scenario a fixed
//! intercept is deterministically calibrated so the population-average success
//! probability equals the frozen human SESOI.

use crate::evidence_digest::{
    canonical_json_sha256,
    perceptual_analysis_plan::{
        FrozenPerceptualAnalysisSpecV1, FrozenPerceptualPowerPlanV1,
        PowerHeterogeneityScenarioV1,
    },
    perceptual_study_protocol::FrozenPerceptualStudyProtocolV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PERCEPTUAL_POWER_QUALIFICATION_VERSION: &str =
    "mel003-perceptual-power-qualification-v1";
pub const PERCEPTUAL_POWER_PROTOCOL_BINDING_VERSION: &str =
    "mel003-perceptual-power-protocol-binding-v1";
pub const MARGINAL_INTEGRATION_BOUND_SD: f64 = 8.0;
pub const MARGINAL_INTEGRATION_INTERVALS: usize = 4096;
pub const MARGINAL_SESOI_TOLERANCE: f64 = 1.0e-6;
pub const CALIBRATED_INTERCEPT_TOLERANCE: f64 = 1.0e-8;
pub const CALIBRATION_BISECTION_ITERATIONS: usize = 160;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerSesoIScaleV1 {
    /// Population-average probability after integrating over participant and
    /// item heterogeneity. The same human-scale SESOI therefore has the same
    /// meaning in every registered variance scenario.
    MarginalPopulationAverageSuccessProbability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerGenerativeModelV1 {
    /// u_p ~ N(0, sigma_p), v_i ~ N(0, sigma_i), independently, with
    /// P(Y_pi=1 | u_p,v_i) = logistic(beta + u_p + v_i).
    CrossedIndependentGaussianRandomInterceptBernoulliLogit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerCalibrationMethodV1 {
    /// Deterministic bisection around a normalized composite-Simpson integral
    /// of the standard normal over +/- 8 SD. The omitted normal tail is far
    /// below the registered 1e-6 SESOI tolerance.
    BisectionNormalizedCompositeSimpsonPlusMinus8SdV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarginalPowerScenarioCalibrationV1 {
    pub scenario_id: String,
    pub calibrated_fixed_intercept_log_odds: f64,
    pub achieved_marginal_success_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerGridOutcomeAccountingV1 {
    pub scenario_id: String,
    pub completed_participants: usize,
    pub simulation_replicates: usize,
    pub primary_rejection_count: usize,
    pub primary_not_rejected_count: usize,
    /// Registered-model failures remain in the power denominator. They may not
    /// disappear from the simulation accounting or be replaced post hoc by a
    /// simpler model.
    pub primary_inconclusive_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualPowerQualificationV1 {
    pub qualification_version: String,
    pub analysis_spec_sha256: String,
    pub power_plan_sha256: String,
    pub sesoi_scale: PowerSesoIScaleV1,
    pub generative_model: PowerGenerativeModelV1,
    pub calibration_method: PowerCalibrationMethodV1,
    pub random_effects_independent: bool,
    pub participant_effect_reused_across_registered_items: bool,
    pub item_effect_reused_across_participants: bool,
    pub marginal_sesoi_tolerance: f64,
    pub scenario_calibrations: Vec<MarginalPowerScenarioCalibrationV1>,
    pub grid_outcomes: Vec<PowerGridOutcomeAccountingV1>,
    pub qualification_sha256: String,
}

#[derive(Serialize)]
struct PowerQualificationCommitment<'a> {
    qualification_version: &'a str,
    analysis_spec_sha256: &'a str,
    power_plan_sha256: &'a str,
    sesoi_scale: PowerSesoIScaleV1,
    generative_model: PowerGenerativeModelV1,
    calibration_method: PowerCalibrationMethodV1,
    random_effects_independent: bool,
    participant_effect_reused_across_registered_items: bool,
    item_effect_reused_across_participants: bool,
    marginal_sesoi_tolerance: f64,
    scenario_calibrations: &'a [MarginalPowerScenarioCalibrationV1],
    grid_outcomes: &'a [PowerGridOutcomeAccountingV1],
}

pub fn power_qualification_commitment(
    qualification: &FrozenPerceptualPowerQualificationV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&PowerQualificationCommitment {
        qualification_version: &qualification.qualification_version,
        analysis_spec_sha256: &qualification.analysis_spec_sha256,
        power_plan_sha256: &qualification.power_plan_sha256,
        sesoi_scale: qualification.sesoi_scale,
        generative_model: qualification.generative_model,
        calibration_method: qualification.calibration_method,
        random_effects_independent: qualification.random_effects_independent,
        participant_effect_reused_across_registered_items: qualification
            .participant_effect_reused_across_registered_items,
        item_effect_reused_across_participants: qualification
            .item_effect_reused_across_participants,
        marginal_sesoi_tolerance: qualification.marginal_sesoi_tolerance,
        scenario_calibrations: &qualification.scenario_calibrations,
        grid_outcomes: &qualification.grid_outcomes,
    })
}

pub fn seal_power_qualification(
    qualification: &mut FrozenPerceptualPowerQualificationV1,
) -> Result<(), serde_json::Error> {
    qualification
        .scenario_calibrations
        .sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    qualification.grid_outcomes.sort_by(|left, right| {
        left.scenario_id
            .cmp(&right.scenario_id)
            .then_with(|| left.completed_participants.cmp(&right.completed_participants))
    });
    qualification.qualification_sha256 = power_qualification_commitment(qualification)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualPowerQualificationIssueV1 {
    InvalidAnalysisSpec,
    InvalidPowerPlan,
    AnalysisSpecDigestMismatch,
    PowerPlanDigestMismatch,
    WrongVersion,
    WrongSesoIScale,
    WrongGenerativeModel,
    WrongCalibrationMethod,
    MissingCrossedReuseInvariant { field: String },
    WrongSesoITolerance,
    DuplicateScenarioCalibration { scenario_id: String },
    MissingScenarioCalibration { scenario_id: String },
    UnexpectedScenarioCalibration { scenario_id: String },
    InvalidScenarioCalibration { scenario_id: String },
    CalibratedInterceptMismatch { scenario_id: String },
    MarginalSesoIMismatch { scenario_id: String },
    DuplicateGridOutcome { scenario_id: String, participants: usize },
    MissingGridOutcome { scenario_id: String, participants: usize },
    UnexpectedGridOutcome { scenario_id: String, participants: usize },
    GridOutcomePlanMismatch { scenario_id: String, participants: usize },
    GridOutcomePartitionMismatch { scenario_id: String, participants: usize },
    InvalidDigest { field: String },
    QualificationDigestMismatch,
    SerializationFailed,
}

pub fn validate_power_qualification(
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
    power_plan: &FrozenPerceptualPowerPlanV1,
    qualification: &FrozenPerceptualPowerQualificationV1,
) -> Vec<PerceptualPowerQualificationIssueV1> {
    let mut issues = Vec::new();
    if !analysis_spec.validate().is_empty() {
        issues.push(PerceptualPowerQualificationIssueV1::InvalidAnalysisSpec);
    }
    if !power_plan.validate(analysis_spec).is_empty() {
        issues.push(PerceptualPowerQualificationIssueV1::InvalidPowerPlan);
    }

    let expected_analysis_digest = canonical_json_sha256(analysis_spec);
    match expected_analysis_digest {
        Ok(value) if value == qualification.analysis_spec_sha256 => {}
        Ok(_) => issues.push(PerceptualPowerQualificationIssueV1::AnalysisSpecDigestMismatch),
        Err(_) => issues.push(PerceptualPowerQualificationIssueV1::SerializationFailed),
    }
    let expected_plan_digest = power_plan.plan_sha256();
    match expected_plan_digest {
        Ok(value) if value == qualification.power_plan_sha256 => {}
        Ok(_) => issues.push(PerceptualPowerQualificationIssueV1::PowerPlanDigestMismatch),
        Err(_) => issues.push(PerceptualPowerQualificationIssueV1::SerializationFailed),
    }

    if qualification.qualification_version != PERCEPTUAL_POWER_QUALIFICATION_VERSION {
        issues.push(PerceptualPowerQualificationIssueV1::WrongVersion);
    }
    if qualification.sesoi_scale
        != PowerSesoIScaleV1::MarginalPopulationAverageSuccessProbability
    {
        issues.push(PerceptualPowerQualificationIssueV1::WrongSesoIScale);
    }
    if qualification.generative_model
        != PowerGenerativeModelV1::CrossedIndependentGaussianRandomInterceptBernoulliLogit
    {
        issues.push(PerceptualPowerQualificationIssueV1::WrongGenerativeModel);
    }
    if qualification.calibration_method
        != PowerCalibrationMethodV1::BisectionNormalizedCompositeSimpsonPlusMinus8SdV1
    {
        issues.push(PerceptualPowerQualificationIssueV1::WrongCalibrationMethod);
    }
    for (field, present) in [
        ("random_effects_independent", qualification.random_effects_independent),
        (
            "participant_effect_reused_across_registered_items",
            qualification.participant_effect_reused_across_registered_items,
        ),
        (
            "item_effect_reused_across_participants",
            qualification.item_effect_reused_across_participants,
        ),
    ] {
        if !present {
            issues.push(
                PerceptualPowerQualificationIssueV1::MissingCrossedReuseInvariant {
                    field: field.into(),
                },
            );
        }
    }
    if !approximately_equal(
        qualification.marginal_sesoi_tolerance,
        MARGINAL_SESOI_TOLERANCE,
        1.0e-12,
    ) {
        issues.push(PerceptualPowerQualificationIssueV1::WrongSesoITolerance);
    }

    validate_scenario_calibrations(power_plan, qualification, &mut issues);
    validate_grid_outcomes(power_plan, qualification, &mut issues);

    for (field, digest) in [
        ("analysis_spec_sha256", qualification.analysis_spec_sha256.as_str()),
        ("power_plan_sha256", qualification.power_plan_sha256.as_str()),
        ("qualification_sha256", qualification.qualification_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(PerceptualPowerQualificationIssueV1::InvalidDigest {
                field: field.into(),
            });
        }
    }
    match power_qualification_commitment(qualification) {
        Ok(value) if value == qualification.qualification_sha256 => {}
        Ok(_) => issues.push(PerceptualPowerQualificationIssueV1::QualificationDigestMismatch),
        Err(_) => issues.push(PerceptualPowerQualificationIssueV1::SerializationFailed),
    }
    issues
}

fn validate_scenario_calibrations(
    power_plan: &FrozenPerceptualPowerPlanV1,
    qualification: &FrozenPerceptualPowerQualificationV1,
    issues: &mut Vec<PerceptualPowerQualificationIssueV1>,
) {
    let registered_ids: BTreeSet<_> = power_plan
        .heterogeneity_scenarios
        .iter()
        .map(|scenario| scenario.scenario_id.as_str())
        .collect();
    let mut found = BTreeMap::new();
    for calibration in &qualification.scenario_calibrations {
        if found
            .insert(calibration.scenario_id.as_str(), calibration)
            .is_some()
        {
            issues.push(
                PerceptualPowerQualificationIssueV1::DuplicateScenarioCalibration {
                    scenario_id: calibration.scenario_id.clone(),
                },
            );
        }
        if !registered_ids.contains(calibration.scenario_id.as_str()) {
            issues.push(
                PerceptualPowerQualificationIssueV1::UnexpectedScenarioCalibration {
                    scenario_id: calibration.scenario_id.clone(),
                },
            );
        }
    }

    for scenario in &power_plan.heterogeneity_scenarios {
        let Some(calibration) = found.get(scenario.scenario_id.as_str()).copied() else {
            issues.push(
                PerceptualPowerQualificationIssueV1::MissingScenarioCalibration {
                    scenario_id: scenario.scenario_id.clone(),
                },
            );
            continue;
        };
        if !calibration.calibrated_fixed_intercept_log_odds.is_finite()
            || !calibration.achieved_marginal_success_probability.is_finite()
        {
            issues.push(
                PerceptualPowerQualificationIssueV1::InvalidScenarioCalibration {
                    scenario_id: scenario.scenario_id.clone(),
                },
            );
            continue;
        }

        let expected_beta = calibrated_marginal_fixed_intercept(
            power_plan.primary_sesoi_success_probability,
            scenario,
        );
        if !approximately_equal(
            calibration.calibrated_fixed_intercept_log_odds,
            expected_beta,
            CALIBRATED_INTERCEPT_TOLERANCE,
        ) {
            issues.push(
                PerceptualPowerQualificationIssueV1::CalibratedInterceptMismatch {
                    scenario_id: scenario.scenario_id.clone(),
                },
            );
        }
        let recomputed = marginal_success_probability(
            calibration.calibrated_fixed_intercept_log_odds,
            scenario.participant_intercept_sd,
            scenario.item_intercept_sd,
        );
        if !approximately_equal(
            calibration.achieved_marginal_success_probability,
            recomputed,
            MARGINAL_SESOI_TOLERANCE,
        ) || !approximately_equal(
            recomputed,
            power_plan.primary_sesoi_success_probability,
            MARGINAL_SESOI_TOLERANCE,
        ) {
            issues.push(PerceptualPowerQualificationIssueV1::MarginalSesoIMismatch {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
    }
}

fn validate_grid_outcomes(
    power_plan: &FrozenPerceptualPowerPlanV1,
    qualification: &FrozenPerceptualPowerQualificationV1,
    issues: &mut Vec<PerceptualPowerQualificationIssueV1>,
) {
    let registered_keys: BTreeSet<_> = power_plan
        .grid
        .iter()
        .map(|cell| (cell.scenario_id.as_str(), cell.completed_participants))
        .collect();
    let mut found = BTreeMap::new();
    for outcome in &qualification.grid_outcomes {
        let key = (outcome.scenario_id.as_str(), outcome.completed_participants);
        if found.insert(key, outcome).is_some() {
            issues.push(PerceptualPowerQualificationIssueV1::DuplicateGridOutcome {
                scenario_id: outcome.scenario_id.clone(),
                participants: outcome.completed_participants,
            });
        }
        if !registered_keys.contains(&key) {
            issues.push(PerceptualPowerQualificationIssueV1::UnexpectedGridOutcome {
                scenario_id: outcome.scenario_id.clone(),
                participants: outcome.completed_participants,
            });
        }
    }

    for cell in &power_plan.grid {
        let key = (cell.scenario_id.as_str(), cell.completed_participants);
        let Some(outcome) = found.get(&key).copied() else {
            issues.push(PerceptualPowerQualificationIssueV1::MissingGridOutcome {
                scenario_id: cell.scenario_id.clone(),
                participants: cell.completed_participants,
            });
            continue;
        };
        if outcome.simulation_replicates != cell.simulation_replicates
            || outcome.primary_rejection_count != cell.primary_rejection_count
        {
            issues.push(PerceptualPowerQualificationIssueV1::GridOutcomePlanMismatch {
                scenario_id: cell.scenario_id.clone(),
                participants: cell.completed_participants,
            });
        }
        let accounted = outcome
            .primary_rejection_count
            .checked_add(outcome.primary_not_rejected_count)
            .and_then(|value| value.checked_add(outcome.primary_inconclusive_count));
        if accounted != Some(outcome.simulation_replicates) {
            issues.push(
                PerceptualPowerQualificationIssueV1::GridOutcomePartitionMismatch {
                    scenario_id: cell.scenario_id.clone(),
                    participants: cell.completed_participants,
                },
            );
        }
    }
}

/// Population-average Bernoulli success probability for the registered crossed
/// random-intercept model. Because independent Gaussian participant and item
/// intercepts add to another Gaussian, their combined SD is `hypot(sigma_p,
/// sigma_i)`. The normalized Simpson integral avoids allowing the tiny +/-8 SD
/// truncation mass to perturb the requested human-scale probability.
pub fn marginal_success_probability(
    fixed_intercept_log_odds: f64,
    participant_intercept_sd: f64,
    item_intercept_sd: f64,
) -> f64 {
    if !fixed_intercept_log_odds.is_finite()
        || !participant_intercept_sd.is_finite()
        || !item_intercept_sd.is_finite()
        || participant_intercept_sd < 0.0
        || item_intercept_sd < 0.0
    {
        return f64::NAN;
    }
    let sigma = participant_intercept_sd.hypot(item_intercept_sd);
    if sigma == 0.0 {
        return logistic(fixed_intercept_log_odds);
    }

    let n = MARGINAL_INTEGRATION_INTERVALS;
    debug_assert!(n > 0 && n % 2 == 0);
    let lower = -MARGINAL_INTEGRATION_BOUND_SD;
    let upper = MARGINAL_INTEGRATION_BOUND_SD;
    let h = (upper - lower) / n as f64;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for index in 0..=n {
        let z = lower + h * index as f64;
        let weight = if index == 0 || index == n {
            1.0
        } else if index % 2 == 0 {
            2.0
        } else {
            4.0
        };
        let density = standard_normal_density(z);
        denominator += weight * density;
        numerator += weight * density * logistic(fixed_intercept_log_odds + sigma * z);
    }
    numerator / denominator
}

pub fn calibrated_marginal_fixed_intercept(
    target_success_probability: f64,
    scenario: &PowerHeterogeneityScenarioV1,
) -> f64 {
    if !target_success_probability.is_finite()
        || target_success_probability <= 0.0
        || target_success_probability >= 1.0
    {
        return f64::NAN;
    }
    let sigma = scenario
        .participant_intercept_sd
        .hypot(scenario.item_intercept_sd);
    if sigma == 0.0 {
        return logit(target_success_probability);
    }

    // +/-8 sigma plus a wide fixed margin brackets practically useful human
    // probabilities while remaining deterministic even for unusually large
    // registered nuisance variances.
    let bound = 40.0 + 8.0 * sigma;
    let mut lower = -bound;
    let mut upper = bound;
    for _ in 0..CALIBRATION_BISECTION_ITERATIONS {
        let midpoint = lower + (upper - lower) * 0.5;
        let probability = marginal_success_probability(
            midpoint,
            scenario.participant_intercept_sd,
            scenario.item_intercept_sd,
        );
        if probability < target_success_probability {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    lower + (upper - lower) * 0.5
}

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn logit(probability: f64) -> f64 {
    (probability / (1.0 - probability)).ln()
}

fn standard_normal_density(z: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * z * z).exp()
}

fn approximately_equal(found: f64, expected: f64, tolerance: f64) -> bool {
    if !found.is_finite() || !expected.is_finite() {
        return false;
    }
    let scale = 1.0f64.max(found.abs()).max(expected.abs());
    (found - expected).abs() <= tolerance * scale
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualPowerProtocolBindingV1 {
    pub binding_version: String,
    pub protocol_sha256: String,
    pub analysis_spec_sha256: String,
    pub power_plan_sha256: String,
    pub power_qualification_sha256: String,
    pub selected_completed_participants: usize,
    pub maximum_enrolled_participants: usize,
    pub binding_sha256: String,
}

#[derive(Serialize)]
struct PowerProtocolBindingCommitment<'a> {
    binding_version: &'a str,
    protocol_sha256: &'a str,
    analysis_spec_sha256: &'a str,
    power_plan_sha256: &'a str,
    power_qualification_sha256: &'a str,
    selected_completed_participants: usize,
    maximum_enrolled_participants: usize,
}

pub fn power_protocol_binding_commitment(
    binding: &FrozenPerceptualPowerProtocolBindingV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&PowerProtocolBindingCommitment {
        binding_version: &binding.binding_version,
        protocol_sha256: &binding.protocol_sha256,
        analysis_spec_sha256: &binding.analysis_spec_sha256,
        power_plan_sha256: &binding.power_plan_sha256,
        power_qualification_sha256: &binding.power_qualification_sha256,
        selected_completed_participants: binding.selected_completed_participants,
        maximum_enrolled_participants: binding.maximum_enrolled_participants,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualPowerProtocolBindingIssueV1 {
    InvalidProtocol,
    InvalidPowerQualification,
    WrongVersion,
    ProtocolDigestMismatch,
    AnalysisSpecDigestMismatch,
    PowerPlanDigestMismatch,
    PowerQualificationDigestMismatch,
    ProtocolPlanningArtifactMismatch,
    PlannedCompletedParticipantsMismatch,
    MaximumEnrollmentMismatch,
    OutcomeAdaptiveStoppingMismatch,
    StoredParticipantCountMismatch,
    InvalidDigest { field: String },
    BindingDigestMismatch,
    SerializationFailed,
}

pub fn build_power_protocol_binding(
    protocol: &FrozenPerceptualStudyProtocolV1,
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
    power_plan: &FrozenPerceptualPowerPlanV1,
    qualification: &FrozenPerceptualPowerQualificationV1,
) -> Result<
    FrozenPerceptualPowerProtocolBindingV1,
    Vec<PerceptualPowerProtocolBindingIssueV1>,
> {
    let mut binding = FrozenPerceptualPowerProtocolBindingV1 {
        binding_version: PERCEPTUAL_POWER_PROTOCOL_BINDING_VERSION.into(),
        protocol_sha256: canonical_json_sha256(protocol)
            .map_err(|_| vec![PerceptualPowerProtocolBindingIssueV1::SerializationFailed])?,
        analysis_spec_sha256: canonical_json_sha256(analysis_spec)
            .map_err(|_| vec![PerceptualPowerProtocolBindingIssueV1::SerializationFailed])?,
        power_plan_sha256: power_plan
            .plan_sha256()
            .map_err(|_| vec![PerceptualPowerProtocolBindingIssueV1::SerializationFailed])?,
        power_qualification_sha256: qualification.qualification_sha256.clone(),
        selected_completed_participants: power_plan.selected_completed_participants,
        maximum_enrolled_participants: power_plan.maximum_enrolled_participants,
        binding_sha256: String::new(),
    };
    binding.binding_sha256 = power_protocol_binding_commitment(&binding)
        .map_err(|_| vec![PerceptualPowerProtocolBindingIssueV1::SerializationFailed])?;
    let issues = validate_power_protocol_binding(
        protocol,
        analysis_spec,
        power_plan,
        qualification,
        &binding,
    );
    if issues.is_empty() {
        Ok(binding)
    } else {
        Err(issues)
    }
}

pub fn validate_power_protocol_binding(
    protocol: &FrozenPerceptualStudyProtocolV1,
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
    power_plan: &FrozenPerceptualPowerPlanV1,
    qualification: &FrozenPerceptualPowerQualificationV1,
    binding: &FrozenPerceptualPowerProtocolBindingV1,
) -> Vec<PerceptualPowerProtocolBindingIssueV1> {
    let mut issues = Vec::new();
    if !protocol.validate().is_empty() {
        issues.push(PerceptualPowerProtocolBindingIssueV1::InvalidProtocol);
    }
    if !validate_power_qualification(analysis_spec, power_plan, qualification).is_empty() {
        issues.push(PerceptualPowerProtocolBindingIssueV1::InvalidPowerQualification);
    }
    if binding.binding_version != PERCEPTUAL_POWER_PROTOCOL_BINDING_VERSION {
        issues.push(PerceptualPowerProtocolBindingIssueV1::WrongVersion);
    }

    let protocol_digest = canonical_json_sha256(protocol);
    match protocol_digest {
        Ok(ref value) if *value == binding.protocol_sha256 => {}
        Ok(_) => issues.push(PerceptualPowerProtocolBindingIssueV1::ProtocolDigestMismatch),
        Err(_) => issues.push(PerceptualPowerProtocolBindingIssueV1::SerializationFailed),
    }
    let analysis_digest = canonical_json_sha256(analysis_spec);
    match analysis_digest {
        Ok(ref value) if *value == binding.analysis_spec_sha256 => {}
        Ok(_) => issues.push(PerceptualPowerProtocolBindingIssueV1::AnalysisSpecDigestMismatch),
        Err(_) => issues.push(PerceptualPowerProtocolBindingIssueV1::SerializationFailed),
    }
    let plan_digest = power_plan.plan_sha256();
    match plan_digest {
        Ok(ref value) if *value == binding.power_plan_sha256 => {}
        Ok(_) => issues.push(PerceptualPowerProtocolBindingIssueV1::PowerPlanDigestMismatch),
        Err(_) => issues.push(PerceptualPowerProtocolBindingIssueV1::SerializationFailed),
    }
    if qualification.qualification_sha256 != binding.power_qualification_sha256 {
        issues.push(PerceptualPowerProtocolBindingIssueV1::PowerQualificationDigestMismatch);
    }

    if protocol.analysis_spec_sha256 != binding.analysis_spec_sha256
        || power_plan.analysis_spec_sha256 != binding.analysis_spec_sha256
    {
        issues.push(PerceptualPowerProtocolBindingIssueV1::AnalysisSpecDigestMismatch);
    }
    if protocol.sample_size.planning_artifact_sha256 != binding.power_plan_sha256 {
        issues.push(PerceptualPowerProtocolBindingIssueV1::ProtocolPlanningArtifactMismatch);
    }
    if protocol.sample_size.planned_completed_participants
        != power_plan.selected_completed_participants
    {
        issues.push(
            PerceptualPowerProtocolBindingIssueV1::PlannedCompletedParticipantsMismatch,
        );
    }
    if protocol.sample_size.maximum_enrolled_participants
        != power_plan.maximum_enrolled_participants
    {
        issues.push(PerceptualPowerProtocolBindingIssueV1::MaximumEnrollmentMismatch);
    }
    if protocol.sample_size.outcome_adaptive_stopping_allowed
        != power_plan.outcome_adaptive_stopping_allowed
    {
        issues.push(PerceptualPowerProtocolBindingIssueV1::OutcomeAdaptiveStoppingMismatch);
    }
    if binding.selected_completed_participants != power_plan.selected_completed_participants
        || binding.maximum_enrolled_participants != power_plan.maximum_enrolled_participants
    {
        issues.push(PerceptualPowerProtocolBindingIssueV1::StoredParticipantCountMismatch);
    }

    for (field, digest) in [
        ("protocol_sha256", binding.protocol_sha256.as_str()),
        ("analysis_spec_sha256", binding.analysis_spec_sha256.as_str()),
        ("power_plan_sha256", binding.power_plan_sha256.as_str()),
        (
            "power_qualification_sha256",
            binding.power_qualification_sha256.as_str(),
        ),
        ("binding_sha256", binding.binding_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(PerceptualPowerProtocolBindingIssueV1::InvalidDigest {
                field: field.into(),
            });
        }
    }
    match power_protocol_binding_commitment(binding) {
        Ok(value) if value == binding.binding_sha256 => {}
        Ok(_) => issues.push(PerceptualPowerProtocolBindingIssueV1::BindingDigestMismatch),
        Err(_) => issues.push(PerceptualPowerProtocolBindingIssueV1::SerializationFailed),
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::perceptual_analysis_plan::{
        BinaryLinkV1, ModelFailurePolicyV1, PerceptualAnalysisExecutionIdentityV1,
        PowerEffectBasisV1, PowerGridCellV1, PowerSelectionRuleV1,
        PowerSimulationRunnerIdentityV1, SecondaryConditioningPolicyV1,
        PERCEPTUAL_ANALYSIS_SPEC_VERSION, PERCEPTUAL_POWER_PLAN_VERSION,
    };
    use crate::evidence_digest::perceptual_study_protocol::{
        AnalysisPolicyV1, BlindingAndRandomizationPolicyV1,
        ExternalPerceptualPreregistrationV1, ForbiddenPerceptualClaimV1,
        LoudnessMatchingV1, Mel003AcousticSubjectBindingV1, MissingResponsePolicyV1,
        ParticipantPolicyV1, PerceptualEndpointRoleV1, PerceptualEndpointV1,
        PerceptualStudyItemV1, PerceptualTaskV1, PrimaryAnalysisModelV1,
        SampleSizePlanV1, SecondaryMultiplicityPolicyV1, StimulusExtentV1,
        StimulusPolicyV1, MEL003_C6F_BUNDLE_VERSION, MEL003_FIXED_SEEDS,
        PERCEPTUAL_STUDY_PROTOCOL_VERSION,
    };

    const DIGEST: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const COMMIT: &str = "cccccccccccccccccccccccccccccccccccccccc";

    fn analysis() -> FrozenPerceptualAnalysisSpecV1 {
        FrozenPerceptualAnalysisSpecV1 {
            spec_version: PERCEPTUAL_ANALYSIS_SPEC_VERSION.into(),
            primary_task: PerceptualTaskV1::AbxDiscrimination,
            key_secondary_task: PerceptualTaskV1::DirectionalRearticulation2Afc,
            primary_model: PrimaryAnalysisModelV1::CrossedParticipantItemLogisticRandomIntercepts,
            link: BinaryLinkV1::Logit,
            chance_probability: 0.5,
            alpha: 0.05,
            confidence_level: 0.95,
            participant_random_intercept_required: true,
            item_random_intercept_required: true,
            registered_item_seeds: MEL003_FIXED_SEEDS,
            primary_alternative_greater_than_chance: true,
            secondary_conditioning:
                SecondaryConditioningPolicyV1::IndependentTrialsNoConditioningOnPrimary,
            secondary_multiplicity:
                SecondaryMultiplicityPolicyV1::HolmWithinRegisteredSecondaryFamily,
            report_fixed_effect_estimate: true,
            report_odds_ratio: true,
            report_confidence_interval: true,
            report_participant_variance: true,
            report_item_variance: true,
            report_participant_summary: true,
            report_item_summary: true,
            model_failure_policy: ModelFailurePolicyV1::InconclusiveNoAutomaticSubstitution,
            execution: PerceptualAnalysisExecutionIdentityV1 {
                source_revision: "analysis-source-v1".into(),
                analysis_program_sha256: DIGEST.into(),
                environment_sha256: DIGEST.into(),
                model_engine: "qualified-glmm-runner".into(),
                model_engine_version: "v1".into(),
            },
        }
    }

    fn power_plan(analysis: &FrozenPerceptualAnalysisSpecV1) -> FrozenPerceptualPowerPlanV1 {
        FrozenPerceptualPowerPlanV1 {
            plan_version: PERCEPTUAL_POWER_PLAN_VERSION.into(),
            analysis_spec_sha256: analysis.spec_sha256().unwrap(),
            effect_basis: PowerEffectBasisV1::HumanPerceptualSesoI,
            primary_sesoi_success_probability: 0.60,
            item_count: MEL003_FIXED_SEEDS.len(),
            target_power: 0.80,
            alpha: analysis.alpha,
            selection_rule: PowerSelectionRuleV1::SmallestRobustNByOneSided95WilsonLowerBound,
            minimum_simulation_replicates: 5_000,
            heterogeneity_scenarios: vec![
                PowerHeterogeneityScenarioV1 {
                    scenario_id: "moderate".into(),
                    participant_intercept_sd: 0.5,
                    item_intercept_sd: 0.35,
                },
                PowerHeterogeneityScenarioV1 {
                    scenario_id: "high".into(),
                    participant_intercept_sd: 0.8,
                    item_intercept_sd: 0.55,
                },
            ],
            candidate_completed_participants: vec![32, 48, 64],
            grid: vec![
                PowerGridCellV1 { scenario_id: "moderate".into(), completed_participants: 32, simulation_replicates: 5_000, primary_rejection_count: 3_500 },
                PowerGridCellV1 { scenario_id: "moderate".into(), completed_participants: 48, simulation_replicates: 5_000, primary_rejection_count: 4_250 },
                PowerGridCellV1 { scenario_id: "moderate".into(), completed_participants: 64, simulation_replicates: 5_000, primary_rejection_count: 4_600 },
                PowerGridCellV1 { scenario_id: "high".into(), completed_participants: 32, simulation_replicates: 5_000, primary_rejection_count: 3_000 },
                PowerGridCellV1 { scenario_id: "high".into(), completed_participants: 48, simulation_replicates: 5_000, primary_rejection_count: 4_050 },
                PowerGridCellV1 { scenario_id: "high".into(), completed_participants: 64, simulation_replicates: 5_000, primary_rejection_count: 4_400 },
            ],
            selected_completed_participants: 48,
            anticipated_noncompletion_rate: 0.10,
            maximum_enrolled_participants: 54,
            outcome_adaptive_stopping_allowed: false,
            runner: PowerSimulationRunnerIdentityV1 {
                runner_version: "perceptual-power-sim-v1".into(),
                source_revision: "planning-source-v1".into(),
                binary_sha256: DIGEST.into(),
                environment_sha256: DIGEST.into(),
                rng_algorithm: "ChaCha20".into(),
                rng_seed: 7,
            },
        }
    }

    fn outcomes(plan: &FrozenPerceptualPowerPlanV1) -> Vec<PowerGridOutcomeAccountingV1> {
        plan.grid
            .iter()
            .map(|cell| {
                let inconclusive = 25usize;
                PowerGridOutcomeAccountingV1 {
                    scenario_id: cell.scenario_id.clone(),
                    completed_participants: cell.completed_participants,
                    simulation_replicates: cell.simulation_replicates,
                    primary_rejection_count: cell.primary_rejection_count,
                    primary_not_rejected_count: cell
                        .simulation_replicates
                        .saturating_sub(cell.primary_rejection_count + inconclusive),
                    primary_inconclusive_count: inconclusive,
                }
            })
            .collect()
    }

    fn qualification(
        analysis: &FrozenPerceptualAnalysisSpecV1,
        plan: &FrozenPerceptualPowerPlanV1,
    ) -> FrozenPerceptualPowerQualificationV1 {
        let scenario_calibrations = plan
            .heterogeneity_scenarios
            .iter()
            .map(|scenario| {
                let beta = calibrated_marginal_fixed_intercept(
                    plan.primary_sesoi_success_probability,
                    scenario,
                );
                MarginalPowerScenarioCalibrationV1 {
                    scenario_id: scenario.scenario_id.clone(),
                    calibrated_fixed_intercept_log_odds: beta,
                    achieved_marginal_success_probability: marginal_success_probability(
                        beta,
                        scenario.participant_intercept_sd,
                        scenario.item_intercept_sd,
                    ),
                }
            })
            .collect();
        let mut qualification = FrozenPerceptualPowerQualificationV1 {
            qualification_version: PERCEPTUAL_POWER_QUALIFICATION_VERSION.into(),
            analysis_spec_sha256: analysis.spec_sha256().unwrap(),
            power_plan_sha256: plan.plan_sha256().unwrap(),
            sesoi_scale: PowerSesoIScaleV1::MarginalPopulationAverageSuccessProbability,
            generative_model:
                PowerGenerativeModelV1::CrossedIndependentGaussianRandomInterceptBernoulliLogit,
            calibration_method:
                PowerCalibrationMethodV1::BisectionNormalizedCompositeSimpsonPlusMinus8SdV1,
            random_effects_independent: true,
            participant_effect_reused_across_registered_items: true,
            item_effect_reused_across_participants: true,
            marginal_sesoi_tolerance: MARGINAL_SESOI_TOLERANCE,
            scenario_calibrations,
            grid_outcomes: outcomes(plan),
            qualification_sha256: String::new(),
        };
        seal_power_qualification(&mut qualification).unwrap();
        qualification
    }

    fn protocol(
        analysis: &FrozenPerceptualAnalysisSpecV1,
        plan: &FrozenPerceptualPowerPlanV1,
    ) -> FrozenPerceptualStudyProtocolV1 {
        FrozenPerceptualStudyProtocolV1 {
            protocol_version: PERCEPTUAL_STUDY_PROTOCOL_VERSION.into(),
            acoustic_subject: Mel003AcousticSubjectBindingV1 {
                c6f_source_commit: COMMIT.into(),
                c6f_bundle_sha256: DIGEST.into(),
                c6f_bundle_version: MEL003_C6F_BUNDLE_VERSION.into(),
            },
            external_preregistration: ExternalPerceptualPreregistrationV1 {
                registry: "OSF".into(),
                record_id: "mel003-p1".into(),
                frozen_at_utc: "2026-09-19T00:00:00Z".into(),
                record_sha256: DIGEST.into(),
            },
            analysis_spec_sha256: analysis.spec_sha256().unwrap(),
            items: MEL003_FIXED_SEEDS
                .into_iter()
                .map(|seed| PerceptualStudyItemV1 {
                    item_id: format!("sonata-seed-{seed}"),
                    seed,
                })
                .collect(),
            endpoints: vec![
                PerceptualEndpointV1 {
                    task: PerceptualTaskV1::AbxDiscrimination,
                    role: PerceptualEndpointRoleV1::Primary,
                    chance_probability: 0.5,
                    estimand: "ABX correctness probability".into(),
                },
                PerceptualEndpointV1 {
                    task: PerceptualTaskV1::DirectionalRearticulation2Afc,
                    role: PerceptualEndpointRoleV1::KeySecondary,
                    chance_probability: 0.5,
                    estimand: "directional re-articulation selection probability".into(),
                },
            ],
            stimulus: StimulusPolicyV1 {
                extent: StimulusExtentV1::WholeFourBarSubject,
                synchronized_playhead_required: true,
                loudness_matching: LoudnessMatchingV1::PairwiseIntegratedLufsAttenuationOnly,
                maximum_attenuation_db: 6.0,
                preserve_pair_alignment: true,
                neutral_presentation_labels_required: true,
                disjoint_practice_material_required: true,
            },
            blinding: BlindingAndRandomizationPolicyV1 {
                randomization_commitment_sha256: DIGEST.into(),
                schedule_builder_version: "mel003-perceptual-schedule-builder-v1".into(),
                balance_ab_label_assignment: true,
                balance_abx_hidden_identity: true,
                balance_directional_left_right_assignment: true,
                reveal_correct_answers_during_scored_collection: false,
                arm_labelled_monitoring_during_collection: false,
                investigator_can_modify_schedule_after_first_response: false,
            },
            participants: ParticipantPolicyV1 {
                minimum_age_years: 18,
                informed_consent_required: true,
                pseudonymous_participant_tokens_required: true,
                raw_names_or_contact_details_in_study_dataset_allowed: false,
                stereo_playback_check_required: true,
                task_comprehension_practice_required: true,
                practice_feedback_allowed: true,
                scored_trial_feedback_allowed: false,
            },
            sample_size: SampleSizePlanV1 {
                planning_artifact_sha256: plan.plan_sha256().unwrap(),
                planned_completed_participants: plan.selected_completed_participants,
                maximum_enrolled_participants: plan.maximum_enrolled_participants,
                outcome_adaptive_stopping_allowed: plan.outcome_adaptive_stopping_allowed,
            },
            analysis: AnalysisPolicyV1 {
                primary_model: PrimaryAnalysisModelV1::CrossedParticipantItemLogisticRandomIntercepts,
                participant_grouping_factor_required: true,
                item_grouping_factor_required: true,
                primary_alternative_is_greater_than_chance: true,
                alpha: analysis.alpha,
                confidence_level: analysis.confidence_level,
                secondary_multiplicity: SecondaryMultiplicityPolicyV1::HolmWithinRegisteredSecondaryFamily,
                report_item_level_outcomes: true,
                report_participant_level_outcomes: true,
                report_random_effect_variance: true,
                missing_response_policy:
                    MissingResponsePolicyV1::RetainRawExcludeIncompleteSessionNoImputation,
            },
            forbidden_claims: vec![
                ForbiddenPerceptualClaimV1::Preference,
                ForbiddenPerceptualClaimV1::ArtisticQuality,
                ForbiddenPerceptualClaimV1::EmotionalImpact,
                ForbiddenPerceptualClaimV1::StyleIdentity,
                ForbiddenPerceptualClaimV1::CulturalAuthenticity,
                ForbiddenPerceptualClaimV1::HumanLikePerformance,
                ForbiddenPerceptualClaimV1::IndependentAcousticReplication,
                ForbiddenPerceptualClaimV1::CognitionProductAuthority,
                ForbiddenPerceptualClaimV1::GeneralizationBeyondRegisteredItemsAndEligiblePopulation,
            ],
            stimulus_pack_bound: false,
            participant_schedule_bound: false,
            collection_authorized: false,
            responses_present: false,
        }
    }

    #[test]
    fn zero_heterogeneity_reduces_to_logit_of_sesoi() {
        let scenario = PowerHeterogeneityScenarioV1 {
            scenario_id: "zero".into(),
            participant_intercept_sd: 0.0,
            item_intercept_sd: 0.0,
        };
        let beta = calibrated_marginal_fixed_intercept(0.60, &scenario);
        assert!(approximately_equal(beta, logit(0.60), 1.0e-12));
        assert!(approximately_equal(
            marginal_success_probability(beta, 0.0, 0.0),
            0.60,
            1.0e-12,
        ));
    }

    #[test]
    fn nonzero_heterogeneity_requires_marginal_recalibration() {
        let scenario = PowerHeterogeneityScenarioV1 {
            scenario_id: "heterogeneous".into(),
            participant_intercept_sd: 0.8,
            item_intercept_sd: 0.55,
        };
        let conditional_beta = logit(0.60);
        let calibrated = calibrated_marginal_fixed_intercept(0.60, &scenario);
        assert!(calibrated > conditional_beta);
        assert!(approximately_equal(
            marginal_success_probability(
                calibrated,
                scenario.participant_intercept_sd,
                scenario.item_intercept_sd,
            ),
            0.60,
            MARGINAL_SESOI_TOLERANCE,
        ));
    }

    #[test]
    fn qualification_requires_inconclusive_fits_to_remain_in_denominator() {
        let analysis = analysis();
        let plan = power_plan(&analysis);
        let mut qualification = qualification(&analysis, &plan);
        assert!(validate_power_qualification(&analysis, &plan, &qualification).is_empty());
        qualification.grid_outcomes[0].primary_inconclusive_count = 0;
        seal_power_qualification(&mut qualification).unwrap();
        assert!(validate_power_qualification(&analysis, &plan, &qualification)
            .iter()
            .any(|issue| matches!(
                issue,
                PerceptualPowerQualificationIssueV1::GridOutcomePartitionMismatch { .. }
            )));
    }

    #[test]
    fn protocol_binding_makes_selected_n_load_bearing() {
        let analysis = analysis();
        let plan = power_plan(&analysis);
        let qualification = qualification(&analysis, &plan);
        let protocol = protocol(&analysis, &plan);
        assert!(protocol.validate().is_empty());
        let binding = build_power_protocol_binding(
            &protocol,
            &analysis,
            &plan,
            &qualification,
        )
        .unwrap();
        assert!(validate_power_protocol_binding(
            &protocol,
            &analysis,
            &plan,
            &qualification,
            &binding,
        )
        .is_empty());

        let mut wrong_protocol = protocol.clone();
        wrong_protocol.sample_size.planned_completed_participants += 1;
        let issues = validate_power_protocol_binding(
            &wrong_protocol,
            &analysis,
            &plan,
            &qualification,
            &binding,
        );
        assert!(issues.iter().any(|issue| matches!(
            issue,
            PerceptualPowerProtocolBindingIssueV1::PlannedCompletedParticipantsMismatch
                | PerceptualPowerProtocolBindingIssueV1::ProtocolDigestMismatch
        )));
    }
}
