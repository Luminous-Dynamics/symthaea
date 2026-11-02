//! Exact policy-frontier optimization under noncompensatory confidence guardrails.
//!
//! This layer converts each global/local expected-value guardrail into a hard
//! linear inequality over the randomized policy weights. Policies without a
//! complete, estimable, sufficiently sampled bound for every requirement are
//! assigned q_i = 0 and carry explicit exclusion reasons. The existing
//! independent fixed-mixture verifier is run after optimization.
//!
//! V1 accepts only the same-target-law-by-design, independent-unit, linear
//! expectation contract already required by noncompensatory_guardrails. It does
//! not claim pathwise, tail-risk, quantile, CVaR, or nonlinear-ratio safety.

use super::noncompensatory_guardrails::{
    analyze_noncompensatory_guardrails, verify_noncompensatory_guardrail_report,
    GuardrailDirection, GuardrailEvidenceAvailability, GuardrailPolicyWeight,
    GuardrailRequirement, NoncompensatoryGuardrailReport,
    NoncompensatoryGuardrailRequest, GuardrailOverallDecision,
};
use super::policy_frontier_lp::{
    analyze_policy_frontier_with_constraints, FrontierAdditionalLinearConstraint,
    FrontierAlternativeMetrics, FrontierNullSize, FrontierPolicyWeight, PolicyFrontierError,
    PolicyFrontierPlan, PolicyFrontierReport,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

const REQUEST_SCHEMA: &str =
    "symthaea.architecture-engineering.guardrail-constrained-frontier-request";
const REQUEST_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str =
    "symthaea.architecture-engineering.guardrail-constrained-frontier-report";
const REPORT_VERSION: &str = "0.1.0";
const FEASIBILITY_TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardrailFrontierError {
    pub path: String,
    pub message: String,
}

impl GuardrailFrontierError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for GuardrailFrontierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for GuardrailFrontierError {}

impl From<PolicyFrontierError> for GuardrailFrontierError {
    fn from(error: PolicyFrontierError) -> Self {
        Self::new(error.path, error.message)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardrailConstrainedFrontierRequest {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub frontier_plan: PolicyFrontierPlan,
    /// Complete policy/requirement confidence evidence plus its target,
    /// sampling, roster, and simultaneous-family contracts. The embedded
    /// policy_weights identify the frozen policy family; their probabilities
    /// are not used as an optimizer starting point.
    pub guardrail_evidence: NoncompensatoryGuardrailRequest,
}

impl GuardrailConstrainedFrontierRequest {
    pub fn new(
        run_id: impl Into<String>,
        frontier_plan: PolicyFrontierPlan,
        guardrail_evidence: NoncompensatoryGuardrailRequest,
    ) -> Result<Self, GuardrailFrontierError> {
        let mut request = Self {
            schema: REQUEST_SCHEMA.into(),
            version: REQUEST_VERSION.into(),
            run_id: run_id.into(),
            frontier_plan,
            guardrail_evidence,
        };
        request.canonicalize()?;
        Ok(request)
    }

    pub fn canonicalize(&mut self) -> Result<(), GuardrailFrontierError> {
        if self.schema != REQUEST_SCHEMA || self.version != REQUEST_VERSION {
            return Err(GuardrailFrontierError::new(
                "request.schema",
                "unsupported guardrail-constrained frontier schema/version",
            ));
        }
        if self.run_id.trim().is_empty() {
            return Err(GuardrailFrontierError::new(
                "request.run_id",
                "run ID must be non-empty",
            ));
        }
        self.frontier_plan
            .canonicalize()
            .map_err(GuardrailFrontierError::from)?;
        // request_sha256 canonicalizes and validates the full source contract,
        // including the global/local roster, target relation, sampling contract,
        // and every supplied source receipt digest.
        self.guardrail_evidence
            .canonicalize()
            .map_err(|error| GuardrailFrontierError::new(error.path, error.message))?;

        let frontier_ids = self
            .frontier_plan
            .policies
            .iter()
            .map(|policy| policy.policy_id.as_str())
            .collect::<BTreeSet<_>>();
        let evidence_ids = self
            .guardrail_evidence
            .policy_weights
            .iter()
            .map(|row| row.policy_id.as_str())
            .collect::<BTreeSet<_>>();
        if frontier_ids != evidence_ids {
            return Err(GuardrailFrontierError::new(
                "request.guardrail_evidence.policy_weights",
                "guardrail evidence policy roster must exactly match the frozen frontier family",
            ));
        }
        self.guardrail_evidence
            .request_sha256()
            .map_err(|error| GuardrailFrontierError::new(error.path, error.message))?;
        Ok(())
    }

    pub fn request_sha256(&self) -> Result<String, GuardrailFrontierError> {
        let mut canonical = self.clone();
        canonical.canonicalize()?;
        digest(&canonical)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyEligibilityRecord {
    pub policy_id: String,
    pub eligible: bool,
    pub exclusion_reasons: Vec<String>,
    /// Digests of all supplied policy/requirement evidence rows. Missing rows
    /// are represented by exclusion_reasons and are never silently imputed.
    pub source_receipt_sha256: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardrailConstrainedRequirementResult {
    pub requirement: GuardrailRequirement,
    pub mixture_lower_confidence_bound: f64,
    pub mixture_upper_confidence_bound: f64,
    /// For UPPER, threshold - mixture upper bound. For LOWER,
    /// mixture lower bound - threshold. Nonnegative (up to tolerance) means
    /// the selected mixture satisfies this hard constraint.
    pub signed_slack: f64,
    pub active_at_tolerance: bool,
    pub contributing_policy_weights: Vec<FrontierPolicyWeight>,
    pub source_receipt_sha256: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardrailConstrainedFrontierReport {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub request_sha256: String,
    pub frontier_plan_sha256: String,
    pub guardrail_evidence_sha256: String,
    pub guardrail_roster_sha256: String,
    pub calibration_target_design_sha256: String,
    pub confirmatory_target_design_sha256: String,
    pub sampling_unit_contract_sha256: String,
    pub simultaneous_guardrail_failure_probability: f64,
    pub simultaneous_coverage_receipt_sha256: String,
    pub policy_eligibility: Vec<PolicyEligibilityRecord>,
    pub selected_policy_weights: Vec<FrontierPolicyWeight>,
    pub optimizer_report: PolicyFrontierReport,
    pub guardrail_results: Vec<GuardrailConstrainedRequirementResult>,
    /// Independent fixed-mixture check from noncompensatory_guardrails. Its
    /// verifier does not call the LP optimizer or its feasibility routines.
    pub independent_guardrail_report: NoncompensatoryGuardrailReport,
    pub active_constraint_ids: Vec<String>,
    pub report_sha256: String,
}

fn derive_policy_eligibility(
    request: &GuardrailConstrainedFrontierRequest,
) -> Vec<PolicyEligibilityRecord> {
    let evidence = &request.guardrail_evidence;
    let bound_map = evidence
        .policy_bounds
        .iter()
        .map(|row| ((row.policy_id.as_str(), row.requirement_id.as_str()), row))
        .collect::<BTreeMap<_, _>>();

    request
        .frontier_plan
        .policies
        .iter()
        .map(|policy| {
            let mut reasons = Vec::new();
            let mut receipts = Vec::new();
            for requirement in &evidence.requirements {
                match bound_map.get(&(policy.policy_id.as_str(), requirement.requirement_id.as_str())) {
                    None => reasons.push(format!(
                        "MISSING_BOUND:{}",
                        requirement.requirement_id
                    )),
                    Some(bound) => {
                        receipts.push(bound.source_receipt_sha256.clone());
                        match bound.availability {
                            GuardrailEvidenceAvailability::NotEstimable => reasons.push(format!(
                                "NOT_ESTIMABLE:{}:{}",
                                requirement.requirement_id,
                                bound.reason_code.as_deref().unwrap_or("UNSPECIFIED")
                            )),
                            GuardrailEvidenceAvailability::SimultaneousBoundsAvailable => {
                                if bound.sample_size < requirement.minimum_sample_size {
                                    reasons.push(format!(
                                        "BELOW_MINIMUM_SAMPLE_SIZE:{}:{}<{}",
                                        requirement.requirement_id,
                                        bound.sample_size,
                                        requirement.minimum_sample_size
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            reasons.sort();
            receipts.sort();
            receipts.dedup();
            PolicyEligibilityRecord {
                policy_id: policy.policy_id.clone(),
                eligible: reasons.is_empty(),
                exclusion_reasons: reasons,
                source_receipt_sha256: receipts,
            }
        })
        .collect()
}

fn build_guardrail_constraints(
    request: &GuardrailConstrainedFrontierRequest,
    eligibility: &[PolicyEligibilityRecord],
) -> Vec<FrontierAdditionalLinearConstraint> {
    let evidence = &request.guardrail_evidence;
    let policy_index = request
        .frontier_plan
        .policies
        .iter()
        .enumerate()
        .map(|(index, policy)| (policy.policy_id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let eligibility_map = eligibility
        .iter()
        .map(|row| (row.policy_id.as_str(), row.eligible))
        .collect::<BTreeMap<_, _>>();
    let bounds = evidence
        .policy_bounds
        .iter()
        .map(|row| ((row.policy_id.as_str(), row.requirement_id.as_str()), row))
        .collect::<BTreeMap<_, _>>();
    let policy_count = request.frontier_plan.policies.len();
    let mut constraints = Vec::new();

    // Explicitly remove any policy that lacks one or more applicable bounds,
    // has a NOT_ESTIMABLE cell, or is below a frozen minimum sample size.
    for policy in &request.frontier_plan.policies {
        if eligibility_map.get(policy.policy_id.as_str()) == Some(&false) {
            let index = policy_index[policy.policy_id.as_str()];
            let mut coefficients = vec![0.0; policy_count];
            coefficients[index] = 1.0; // q_i <= 0, combined with q_i >= 0
            constraints.push(FrontierAdditionalLinearConstraint {
                constraint_id: format!("EXCLUDE_POLICY:{}", policy.policy_id),
                coefficients,
                upper_bound: 0.0,
            });
        }
    }

    // Upper guards use simultaneous U_i; lower guards use -L_i. An ineligible
    // policy's coefficient is zero because the separate q_i=0 constraint
    // excludes it from every feasible mixture.
    for requirement in &evidence.requirements {
        let mut coefficients = vec![0.0; policy_count];
        for (index, policy) in request.frontier_plan.policies.iter().enumerate() {
            if eligibility_map.get(policy.policy_id.as_str()) != Some(&true) {
                continue;
            }
            let bound = bounds[&(policy.policy_id.as_str(), requirement.requirement_id.as_str())];
            let value = match requirement.direction {
                GuardrailDirection::UpperBound => bound.upper_confidence_bound.unwrap_or(f64::NAN),
                GuardrailDirection::LowerBound => -bound.lower_confidence_bound.unwrap_or(f64::NAN),
            };
            coefficients[index] = value;
        }
        let (upper_bound, suffix) = match requirement.direction {
            GuardrailDirection::UpperBound => (requirement.threshold, "UPPER"),
            GuardrailDirection::LowerBound => (-requirement.threshold, "LOWER"),
        };
        constraints.push(FrontierAdditionalLinearConstraint {
            constraint_id: format!("GUARDRAIL:{suffix}:{}", requirement.requirement_id),
            coefficients,
            upper_bound,
        });
    }
    constraints
}

fn selected_weights_with_zero_roster(
    request: &GuardrailConstrainedFrontierRequest,
    selected: &[FrontierPolicyWeight],
) -> Vec<GuardrailPolicyWeight> {
    let selected_map = selected
        .iter()
        .map(|row| (row.policy_id.as_str(), row.probability))
        .collect::<BTreeMap<_, _>>();
    request
        .frontier_plan
        .policies
        .iter()
        .map(|policy| GuardrailPolicyWeight {
            policy_id: policy.policy_id.clone(),
            probability: selected_map
                .get(policy.policy_id.as_str())
                .copied()
                .unwrap_or(0.0),
        })
        .collect()
}

fn requirement_results(
    request: &GuardrailConstrainedFrontierRequest,
    independent: &NoncompensatoryGuardrailReport,
    selected: &[FrontierPolicyWeight],
) -> Result<Vec<GuardrailConstrainedRequirementResult>, GuardrailFrontierError> {
    let selected_map = selected
        .iter()
        .map(|row| (row.policy_id.as_str(), row.probability))
        .collect::<BTreeMap<_, _>>();
    let result_map = independent
        .requirement_results
        .iter()
        .map(|row| (row.requirement_id.as_str(), row))
        .collect::<BTreeMap<_, _>>();

    let mut output = Vec::with_capacity(request.guardrail_evidence.requirements.len());
    for requirement in &request.guardrail_evidence.requirements {
        let result = result_map
            .get(requirement.requirement_id.as_str())
            .ok_or_else(|| GuardrailFrontierError::new(
                "report.guardrail_results",
                "independent verifier omitted a declared guardrail requirement",
            ))?;
        let lower = result.mixture_lower_confidence_bound.ok_or_else(|| {
            GuardrailFrontierError::new(
                "report.guardrail_results.mixture_lower_confidence_bound",
                "passing independent guardrail result must contain a lower mixture bound",
            )
        })?;
        let upper = result.mixture_upper_confidence_bound.ok_or_else(|| {
            GuardrailFrontierError::new(
                "report.guardrail_results.mixture_upper_confidence_bound",
                "passing independent guardrail result must contain an upper mixture bound",
            )
        })?;
        let signed_slack = match requirement.direction {
            GuardrailDirection::UpperBound => requirement.threshold - upper,
            GuardrailDirection::LowerBound => lower - requirement.threshold,
        };
        let receipts = result
            .contributing_policy_evidence
            .iter()
            .map(|row| row.source_receipt_sha256.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        output.push(GuardrailConstrainedRequirementResult {
            requirement: requirement.clone(),
            mixture_lower_confidence_bound: lower,
            mixture_upper_confidence_bound: upper,
            signed_slack,
            active_at_tolerance: signed_slack.abs() <= FEASIBILITY_TOLERANCE,
            contributing_policy_weights: selected
                .iter()
                .filter(|row| row.probability > FEASIBILITY_TOLERANCE)
                .filter(|row| selected_map.contains_key(row.policy_id.as_str()))
                .cloned()
                .collect(),
            source_receipt_sha256: receipts,
        });
    }
    Ok(output)
}

fn finish_report(
    mut report: GuardrailConstrainedFrontierReport,
) -> Result<GuardrailConstrainedFrontierReport, GuardrailFrontierError> {
    report.report_sha256.clear();
    report.report_sha256 = digest(&report)?;
    Ok(report)
}

pub fn analyze_guardrail_constrained_frontier(
    input: &GuardrailConstrainedFrontierRequest,
) -> Result<GuardrailConstrainedFrontierReport, GuardrailFrontierError> {
    let mut request = input.clone();
    request.canonicalize()?;
    let request_sha256 = request.request_sha256()?;
    let frontier_plan_sha256 = request
        .frontier_plan
        .plan_sha256()
        .map_err(GuardrailFrontierError::from)?;
    let guardrail_evidence_sha256 = request
        .guardrail_evidence
        .request_sha256()
        .map_err(|error| GuardrailFrontierError::new(error.path, error.message))?;

    let eligibility = derive_policy_eligibility(&request);
    let constraints = build_guardrail_constraints(&request, &eligibility);
    let optimizer_report = analyze_policy_frontier_with_constraints(
        &request.frontier_plan,
        &constraints,
    )
    .map_err(GuardrailFrontierError::from)?;

    // The post-selection checker receives the optimizer's exact mixture,
    // including zero-weight roster members, and independently reconstructs
    // all global/local mixture intervals without calling LP feasibility code.
    let mut final_evidence = request.guardrail_evidence.clone();
    final_evidence.policy_weights =
        selected_weights_with_zero_roster(&request, &optimizer_report.policy_weights);
    final_evidence
        .canonicalize()
        .map_err(|error| GuardrailFrontierError::new(error.path, error.message))?;
    let independent_guardrail_report = analyze_noncompensatory_guardrails(&final_evidence)
        .map_err(|error| GuardrailFrontierError::new(error.path, error.message))?;
    if independent_guardrail_report.overall_decision != GuardrailOverallDecision::Pass {
        return Err(GuardrailFrontierError::new(
            "independent_guardrail_report.overall_decision",
            "optimizer result did not clear the independent fixed-mixture guardrail verifier",
        ));
    }
    verify_noncompensatory_guardrail_report(
        &final_evidence,
        &independent_guardrail_report,
    )
    .map_err(|error| GuardrailFrontierError::new(error.path, error.message))?;

    let guardrail_results = requirement_results(
        &request,
        &independent_guardrail_report,
        &optimizer_report.policy_weights,
    )?;
    let mut active_constraint_ids = eligibility
        .iter()
        .filter(|row| !row.eligible)
        .map(|row| format!("EXCLUDE_POLICY:{}", row.policy_id))
        .collect::<Vec<_>>();
    active_constraint_ids.extend(guardrail_results
        .iter()
        .filter(|row| row.active_at_tolerance)
        .map(|row| {
            let direction = match row.requirement.direction {
                GuardrailDirection::UpperBound => "UPPER",
                GuardrailDirection::LowerBound => "LOWER",
            };
            format!("GUARDRAIL:{direction}:{}", row.requirement.requirement_id)
        }));
    active_constraint_ids.sort();

    finish_report(GuardrailConstrainedFrontierReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        run_id: request.run_id,
        request_sha256,
        frontier_plan_sha256,
        guardrail_evidence_sha256,
        guardrail_roster_sha256: request.guardrail_evidence.guardrail_roster_sha256,
        calibration_target_design_sha256: request.guardrail_evidence.calibration_target_design_sha256,
        confirmatory_target_design_sha256: request.guardrail_evidence.confirmatory_target_design_sha256,
        sampling_unit_contract_sha256: request.guardrail_evidence.sampling_unit_contract_sha256,
        simultaneous_guardrail_failure_probability: request
            .guardrail_evidence
            .simultaneous_guardrail_failure_probability,
        simultaneous_coverage_receipt_sha256: request
            .guardrail_evidence
            .simultaneous_coverage_receipt_sha256,
        policy_eligibility: eligibility,
        selected_policy_weights: optimizer_report.policy_weights.clone(),
        optimizer_report,
        guardrail_results,
        independent_guardrail_report,
        active_constraint_ids,
        report_sha256: String::new(),
    })
}

/// Feasibility/integrity verifier for a selected constrained-frontier report.
///
/// This checker deliberately does not call the optimizer or its internal
/// feasibility function. It reconstructs the selected mixture's null, cost,
/// utility, eligibility, and guardrail quantities from the source receipts.
/// It verifies feasibility and report integrity, not independent optimality;
/// optimality should additionally be compared to a distinct LP oracle.
pub fn verify_guardrail_constrained_frontier_report(
    input: &GuardrailConstrainedFrontierRequest,
    report: &GuardrailConstrainedFrontierReport,
) -> Result<(), GuardrailFrontierError> {
    let mut request = input.clone();
    request.canonicalize()?;
    let request_sha256 = request.request_sha256()?;
    let frontier_plan_sha256 = request
        .frontier_plan
        .plan_sha256()
        .map_err(GuardrailFrontierError::from)?;
    let guardrail_evidence_sha256 = request
        .guardrail_evidence
        .request_sha256()
        .map_err(|error| GuardrailFrontierError::new(error.path, error.message))?;

    if report.schema != REPORT_SCHEMA
        || report.version != REPORT_VERSION
        || report.run_id != request.run_id
        || report.request_sha256 != request_sha256
        || report.frontier_plan_sha256 != frontier_plan_sha256
        || report.guardrail_evidence_sha256 != guardrail_evidence_sha256
        || report.guardrail_roster_sha256 != request.guardrail_evidence.guardrail_roster_sha256
        || report.calibration_target_design_sha256
            != request.guardrail_evidence.calibration_target_design_sha256
        || report.confirmatory_target_design_sha256
            != request.guardrail_evidence.confirmatory_target_design_sha256
        || report.sampling_unit_contract_sha256
            != request.guardrail_evidence.sampling_unit_contract_sha256
        || report.simultaneous_guardrail_failure_probability
            != request.guardrail_evidence.simultaneous_guardrail_failure_probability
        || report.simultaneous_coverage_receipt_sha256
            != request.guardrail_evidence.simultaneous_coverage_receipt_sha256
    {
        return Err(GuardrailFrontierError::new(
            "report.provenance",
            "report identity or source-contract digests do not match the supplied request",
        ));
    }

    let expected_eligibility = derive_policy_eligibility(&request);
    if report.policy_eligibility != expected_eligibility {
        return Err(GuardrailFrontierError::new(
            "report.policy_eligibility",
            "policy eligibility/exclusion reasons differ from complete source evidence",
        ));
    }

    let selected_map = report
        .selected_policy_weights
        .iter()
        .map(|row| (row.policy_id.as_str(), row.probability))
        .collect::<BTreeMap<_, _>>();
    if report.selected_policy_weights != report.optimizer_report.policy_weights {
        return Err(GuardrailFrontierError::new(
            "report.selected_policy_weights",
            "selected mixture differs from the optimizer report",
        ));
    }
    let mut weight_sum = 0.0;
    let mut known_policies = BTreeSet::new();
    let eligibility_map = expected_eligibility
        .iter()
        .map(|row| (row.policy_id.as_str(), row.eligible))
        .collect::<BTreeMap<_, _>>();
    for weight in &report.selected_policy_weights {
        if !weight.probability.is_finite()
            || weight.probability <= 0.0
            || !known_policies.insert(weight.policy_id.as_str())
            || !eligibility_map.contains_key(weight.policy_id.as_str())
            || eligibility_map.get(weight.policy_id.as_str()) != Some(&true)
        {
            return Err(GuardrailFrontierError::new(
                "report.selected_policy_weights",
                "selected support contains a duplicate, nonpositive, unknown, or ineligible policy",
            ));
        }
        weight_sum += weight.probability;
    }
    if (weight_sum - 1.0).abs() > 1e-7 {
        return Err(GuardrailFrontierError::new(
            "report.selected_policy_weights",
            "selected policy probabilities do not sum to one",
        ));
    }

    // Independently reconstruct the original frontier quantities.
    let plan = &request.frontier_plan;
    if report.optimizer_report.plan_sha256 != frontier_plan_sha256 {
        return Err(GuardrailFrontierError::new(
            "report.optimizer_report.plan_sha256",
            "nested optimizer report is not bound to the supplied frontier plan",
        ));
    }
    let weights = plan
        .policies
        .iter()
        .map(|policy| selected_map.get(policy.policy_id.as_str()).copied().unwrap_or(0.0))
        .collect::<Vec<_>>();
    let mut expected_null_sizes = Vec::with_capacity(plan.policies[0].null_models.len());
    let mut worst_null_size = 0.0_f64;
    for null in &plan.policies[0].null_models {
        let size = plan.policies.iter().zip(&weights).map(|(policy, weight)| {
            let value = policy.null_models.iter()
                .find(|row| row.model_id == null.model_id)
                .map(|row| row.null_size)
                .unwrap_or(f64::NAN);
            weight * value
        }).sum::<f64>();
        if !size.is_finite() || size > plan.significance_level + FEASIBILITY_TOLERANCE {
            return Err(GuardrailFrontierError::new(
                "report.optimizer_report.null_model_sizes",
                "selected mixture violates a declared null Type-I constraint",
            ));
        }
        worst_null_size = worst_null_size.max(size);
        expected_null_sizes.push(FrontierNullSize {
            model_id: null.model_id.clone(),
            null_size: size,
        });
    }

    let mut worst_power = f64::INFINITY;
    let mut worst_utility = f64::INFINITY;
    let mut maximum_cost = 0.0_f64;
    let mut expected_alternatives = Vec::with_capacity(plan.policies[0].alternatives.len());
    for alternative in &plan.policies[0].alternatives {
        let mut power = 0.0;
        let mut cost = 0.0;
        let mut utility = 0.0;
        for (policy, weight) in plan.policies.iter().zip(&weights) {
            let row = policy.alternatives.iter()
                .find(|row| row.model_id == alternative.model_id)
                .ok_or_else(|| GuardrailFrontierError::new(
                    "report.optimizer_report.alternatives",
                    "frontier plan is missing a declared alternative row",
                ))?;
            power += weight * row.rejection_power;
            cost += weight * row.expected_evaluator_cost;
            utility += weight * row.decision_utility;
        }
        if !power.is_finite() || !cost.is_finite() || !utility.is_finite() {
            return Err(GuardrailFrontierError::new(
                "report.optimizer_report.alternatives",
                "reconstructed alternative metric is non-finite",
            ));
        }
        if plan.maximum_expected_cost.is_some_and(|cap| cost > cap + FEASIBILITY_TOLERANCE) {
            return Err(GuardrailFrontierError::new(
                "report.optimizer_report.alternatives",
                "selected mixture violates an evaluator-cost cap",
            ));
        }
        worst_power = worst_power.min(power);
        worst_utility = worst_utility.min(utility);
        maximum_cost = maximum_cost.max(cost);
        expected_alternatives.push(FrontierAlternativeMetrics {
            model_id: alternative.model_id.clone(),
            rejection_power: power,
            expected_evaluator_cost: cost,
            decision_utility: utility,
        });
    }
    let entropy = weights.iter().filter(|weight| **weight > 0.0)
        .map(|weight| -weight * weight.ln())
        .sum::<f64>();
    let null_rows_match = report.optimizer_report.null_model_sizes.len() == expected_null_sizes.len()
        && report.optimizer_report.null_model_sizes.iter().zip(&expected_null_sizes).all(|(actual, expected)| {
            actual.model_id == expected.model_id
                && (actual.null_size - expected.null_size).abs() <= 1e-7
        });
    let alternative_rows_match = report.optimizer_report.alternatives.len() == expected_alternatives.len()
        && report.optimizer_report.alternatives.iter().zip(&expected_alternatives).all(|(actual, expected)| {
            actual.model_id == expected.model_id
                && (actual.rejection_power - expected.rejection_power).abs() <= 1e-7
                && (actual.expected_evaluator_cost - expected.expected_evaluator_cost).abs() <= 1e-7
                && (actual.decision_utility - expected.decision_utility).abs() <= 1e-7
        });
    if (report.optimizer_report.worst_case_null_size - worst_null_size).abs() > 1e-7
        || (report.optimizer_report.worst_case_rejection_power - worst_power).abs() > 1e-7
        || (report.optimizer_report.worst_case_decision_utility - worst_utility).abs() > 1e-7
        || (report.optimizer_report.maximum_expected_evaluator_cost - maximum_cost).abs() > 1e-7
        || (report.optimizer_report.mixture_entropy_nats - entropy).abs() > 1e-7
        || (report.optimizer_report.remaining_alpha_budget
            - (plan.significance_level - worst_null_size)).abs() > 1e-7
        || report.optimizer_report.support_size != report.selected_policy_weights.len()
        || !null_rows_match
        || !alternative_rows_match
    {
        return Err(GuardrailFrontierError::new(
            "report.optimizer_report",
            "nested frontier metrics do not reconstruct from the selected mixture",
        ));
    }

    // Reconstruct every guardrail interval directly. This is deliberately
    // separate from the optimizer's additional-constraint feasibility path.
    let bounds = request.guardrail_evidence.policy_bounds.iter()
        .map(|row| ((row.policy_id.as_str(), row.requirement_id.as_str()), row))
        .collect::<BTreeMap<_, _>>();
    let mut expected_results = Vec::new();
    for requirement in &request.guardrail_evidence.requirements {
        let mut lower = 0.0;
        let mut upper = 0.0;
        let mut receipt_ids = BTreeSet::new();
        for selected in &report.selected_policy_weights {
            let bound = bounds.get(&(selected.policy_id.as_str(), requirement.requirement_id.as_str()))
                .ok_or_else(|| GuardrailFrontierError::new(
                    "report.guardrail_results",
                    "selected policy has no bound for a required guardrail",
                ))?;
            if bound.availability != GuardrailEvidenceAvailability::SimultaneousBoundsAvailable
                || bound.sample_size < requirement.minimum_sample_size
            {
                return Err(GuardrailFrontierError::new(
                    "report.guardrail_results",
                    "selected policy has non-estimable or below-minimum guardrail evidence",
                ));
            }
            lower += selected.probability * bound.lower_confidence_bound.unwrap_or(f64::NAN);
            upper += selected.probability * bound.upper_confidence_bound.unwrap_or(f64::NAN);
            receipt_ids.insert(bound.source_receipt_sha256.clone());
        }
        let slack = match requirement.direction {
            GuardrailDirection::UpperBound => requirement.threshold - upper,
            GuardrailDirection::LowerBound => lower - requirement.threshold,
        };
        if !lower.is_finite() || !upper.is_finite() || slack < -FEASIBILITY_TOLERANCE {
            return Err(GuardrailFrontierError::new(
                "report.guardrail_results",
                "selected mixture violates or cannot estimate a mandatory guardrail",
            ));
        }
        expected_results.push(GuardrailConstrainedRequirementResult {
            requirement: requirement.clone(),
            mixture_lower_confidence_bound: lower,
            mixture_upper_confidence_bound: upper,
            signed_slack: slack,
            active_at_tolerance: slack.abs() <= FEASIBILITY_TOLERANCE,
            contributing_policy_weights: report.selected_policy_weights.clone(),
            source_receipt_sha256: receipt_ids.into_iter().collect(),
        });
    }
    if report.guardrail_results != expected_results {
        return Err(GuardrailFrontierError::new(
            "report.guardrail_results",
            "stored guardrail mixture intervals differ from independent reconstruction",
        ));
    }
    let mut expected_active_ids = expected_eligibility
        .iter()
        .filter(|row| !row.eligible)
        .map(|row| format!("EXCLUDE_POLICY:{}", row.policy_id))
        .collect::<Vec<_>>();
    expected_active_ids.extend(expected_results
        .iter()
        .filter(|row| row.active_at_tolerance)
        .map(|row| {
            let direction = match row.requirement.direction {
                GuardrailDirection::UpperBound => "UPPER",
                GuardrailDirection::LowerBound => "LOWER",
            };
            format!("GUARDRAIL:{direction}:{}", row.requirement.requirement_id)
        }));
    expected_active_ids.sort();
    if report.active_constraint_ids != expected_active_ids {
        return Err(GuardrailFrontierError::new(
            "report.active_constraint_ids",
            "active/exclusion constraint roster differs from reconstructed guardrail constraints",
        ));
    }

    let mut final_evidence = request.guardrail_evidence.clone();
    final_evidence.policy_weights = selected_weights_with_zero_roster(
        &request,
        &report.selected_policy_weights,
    );
    final_evidence
        .canonicalize()
        .map_err(|error| GuardrailFrontierError::new(error.path, error.message))?;
    verify_noncompensatory_guardrail_report(
        &final_evidence,
        &report.independent_guardrail_report,
    )
    .map_err(|error| GuardrailFrontierError::new(error.path, error.message))?;
    if report.independent_guardrail_report.overall_decision != GuardrailOverallDecision::Pass {
        return Err(GuardrailFrontierError::new(
            "report.independent_guardrail_report",
            "independent fixed-mixture guardrail decision is not PASS",
        ));
    }

    let mut unhashed = report.clone();
    let stored_digest = unhashed.report_sha256.clone();
    unhashed.report_sha256.clear();
    let actual_digest = digest(&unhashed)?;
    if stored_digest != actual_digest {
        return Err(GuardrailFrontierError::new(
            "report.report_sha256",
            "report digest does not match canonical serialized report payload",
        ));
    }
    Ok(())
}

fn digest<T: Serialize>(value: &T) -> Result<String, GuardrailFrontierError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| GuardrailFrontierError::new("serialization", error.to_string()))?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_functional_confidence::CalibrationTargetRelation;
    use crate::noncompensatory_guardrails::{
        GuardrailEvidenceAvailability, GuardrailScope, GuardrailPolicyBound,
    };

    const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const HASH_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const HASH_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const HASH_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const HASH_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const UTILITY_HASH: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn policy(id: &str, utility: [f64; 2], cost: f64) -> super::super::policy_frontier_lp::FrontierPolicyReceipt {
        use super::super::policy_frontier_lp::{FrontierAlternativeReceipt, FrontierNullReceipt};
        super::super::policy_frontier_lp::FrontierPolicyReceipt {
            policy_id: id.into(),
            utility_contract_sha256: UTILITY_HASH.into(),
            null_models: vec![
                FrontierNullReceipt { model_id: "null-a".into(), null_size: 0.01 },
                FrontierNullReceipt { model_id: "null-b".into(), null_size: 0.02 },
            ],
            alternatives: vec![
                FrontierAlternativeReceipt {
                    model_id: "alt-a".into(),
                    rejection_power: utility[0],
                    expected_evaluator_cost: cost,
                    decision_utility: utility[0],
                },
                FrontierAlternativeReceipt {
                    model_id: "alt-b".into(),
                    rejection_power: utility[1],
                    expected_evaluator_cost: cost,
                    decision_utility: utility[1],
                },
            ],
        }
    }

    fn fixture() -> GuardrailConstrainedFrontierRequest {
        use super::super::noncompensatory_guardrails::{
            GuardrailRequirement, GuardrailPolicyBound,
        };
        let frontier = PolicyFrontierPlan::new(
            "guardrail-frontier-utility-trap",
            0.05,
            None,
            vec![
                policy("unsafe-high-utility", [0.90, 0.80], 2.0),
                policy("safe-lower-utility", [0.60, 0.50], 1.0),
            ],
        )
        .unwrap();
        let requirements = vec![
            GuardrailRequirement {
                requirement_id: "global-crash-rate".into(),
                guardrail_id: "crash-rate".into(),
                metric_contract_sha256: HASH_A.into(),
                scope: GuardrailScope::Global,
                stratum_id: None,
                model_id: "model-current".into(),
                direction: GuardrailDirection::UpperBound,
                threshold: 0.05,
                minimum_sample_size: 100,
            },
            GuardrailRequirement {
                requirement_id: "local-access-floor".into(),
                guardrail_id: "access-floor".into(),
                metric_contract_sha256: HASH_B.into(),
                scope: GuardrailScope::Local,
                stratum_id: Some("cohort-rural".into()),
                model_id: "model-current".into(),
                direction: GuardrailDirection::LowerBound,
                threshold: 0.70,
                minimum_sample_size: 100,
            },
        ];
        let mut bounds = Vec::new();
        for (policy_id, crash, access) in [
            ("unsafe-high-utility", (0.10, 0.12), (0.60, 0.70)),
            ("safe-lower-utility", (0.02, 0.03), (0.80, 0.90)),
        ] {
            bounds.push(GuardrailPolicyBound {
                policy_id: policy_id.into(),
                requirement_id: "global-crash-rate".into(),
                availability: GuardrailEvidenceAvailability::SimultaneousBoundsAvailable,
                lower_confidence_bound: Some(crash.0),
                upper_confidence_bound: Some(crash.1),
                sample_size: 1_000,
                source_receipt_sha256: HASH_C.into(),
                reason_code: None,
            });
            bounds.push(GuardrailPolicyBound {
                policy_id: policy_id.into(),
                requirement_id: "local-access-floor".into(),
                availability: GuardrailEvidenceAvailability::SimultaneousBoundsAvailable,
                lower_confidence_bound: Some(access.0),
                upper_confidence_bound: Some(access.1),
                sample_size: 1_000,
                source_receipt_sha256: HASH_D.into(),
                reason_code: None,
            });
        }
        let mut canonical_requirements = requirements.clone();
        canonical_requirements.sort_by(|a, b| a.requirement_id.cmp(&b.requirement_id));
        let roster_sha = digest(&canonical_requirements).unwrap();
        let evidence = NoncompensatoryGuardrailRequest::new(
            "guardrail-frontier-evidence",
            vec![
                GuardrailPolicyWeight { policy_id: "unsafe-high-utility".into(), probability: 0.5 },
                GuardrailPolicyWeight { policy_id: "safe-lower-utility".into(), probability: 0.5 },
            ],
            requirements,
            2,
            2,
            1,
            roster_sha,
            HASH_E,
            HASH_E,
            HASH_F,
            CalibrationTargetRelation::SameTargetLawByDesign,
            true,
            true,
            true,
            true,
            0.01,
            HASH_A,
            bounds,
        )
        .unwrap();
        GuardrailConstrainedFrontierRequest::new("utility-trap", frontier, evidence).unwrap()
    }

    #[test]
    fn unsafe_high_utility_policy_is_limited_inside_the_lp() {
        let request = fixture();
        let report = analyze_guardrail_constrained_frontier(&request).unwrap();
        let weights = report.selected_policy_weights.iter()
            .map(|row| (row.policy_id.as_str(), row.probability))
            .collect::<BTreeMap<_, _>>();
        let unsafe_weight = weights["unsafe-high-utility"];
        assert!((unsafe_weight - (2.0 / 9.0)).abs() < 1e-8);
        assert!(weights["safe-lower-utility"] > 0.0);
        let crash = report.guardrail_results.iter()
            .find(|row| row.requirement.requirement_id == "global-crash-rate")
            .unwrap();
        assert!((crash.mixture_upper_confidence_bound - 0.05).abs() < 1e-8);
        assert!(crash.active_at_tolerance);
        assert_eq!(report.independent_guardrail_report.overall_decision, GuardrailOverallDecision::Pass);
        verify_guardrail_constrained_frontier_report(&request, &report).unwrap();
    }

    #[test]
    fn nonestimable_policy_is_excluded_and_cannot_receive_positive_mass() {
        let mut request = fixture();
        let row = request.guardrail_evidence.policy_bounds.iter_mut()
            .find(|row| row.policy_id == "unsafe-high-utility" && row.requirement_id == "global-crash-rate")
            .unwrap();
        row.availability = GuardrailEvidenceAvailability::NotEstimable;
        row.lower_confidence_bound = None;
        row.upper_confidence_bound = None;
        row.reason_code = Some("INSUFFICIENT_STRATUM_SUPPORT".into());
        let report = analyze_guardrail_constrained_frontier(&request).unwrap();
        assert!(report.selected_policy_weights.iter().all(|row| row.policy_id != "unsafe-high-utility"));
        let excluded = report.policy_eligibility.iter()
            .find(|row| row.policy_id == "unsafe-high-utility").unwrap();
        assert!(!excluded.eligible);
        assert!(excluded.exclusion_reasons.iter().any(|reason| reason.starts_with("NOT_ESTIMABLE")));
        verify_guardrail_constrained_frontier_report(&request, &report).unwrap();
    }

    #[test]
    fn missing_policy_requirement_row_is_not_imputed_as_zero() {
        let mut request = fixture();
        request.guardrail_evidence.policy_bounds.retain(|row| {
            !(row.policy_id == "unsafe-high-utility" && row.requirement_id == "local-access-floor")
        });
        let report = analyze_guardrail_constrained_frontier(&request).unwrap();
        assert!(report.selected_policy_weights.iter().all(|row| row.policy_id != "unsafe-high-utility"));
        assert!(report.policy_eligibility.iter()
            .find(|row| row.policy_id == "unsafe-high-utility").unwrap()
            .exclusion_reasons.iter().any(|reason| reason.starts_with("MISSING_BOUND")));
        verify_guardrail_constrained_frontier_report(&request, &report).unwrap();
    }

    #[test]
    fn rejects_policy_family_expansion_after_calibration() {
        let mut request = fixture();
        request.frontier_plan.policies.push(policy("new-policy", [0.7, 0.7], 1.0));
        assert!(GuardrailConstrainedFrontierRequest::new(
            "expanded-family",
            request.frontier_plan,
            request.guardrail_evidence,
        ).is_err());
    }

    #[test]
    fn rejects_tampered_report_digest_and_guardrail_result() {
        let request = fixture();
        let report = analyze_guardrail_constrained_frontier(&request).unwrap();
        let mut tampered = report.clone();
        tampered.guardrail_results[0].signed_slack += 0.01;
        assert!(verify_guardrail_constrained_frontier_report(&request, &tampered).is_err());
        let mut altered = report.clone();
        altered.report_sha256 = HASH_A.into();
        assert!(verify_guardrail_constrained_frontier_report(&request, &altered).is_err());
    }

    #[test]
    fn a_linear_guardrail_boundary_is_inclusive_within_fixed_tolerance() {
        let request = fixture();
        let report = analyze_guardrail_constrained_frontier(&request).unwrap();
        let upper = report.guardrail_results.iter()
            .find(|row| row.requirement.direction == GuardrailDirection::UpperBound)
            .unwrap();
        assert!(upper.signed_slack >= -FEASIBILITY_TOLERANCE);
    }
}
