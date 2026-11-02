//! Exact finite LP over a bounded frontier of complete deterministic policies.
//!
//! Variables are q_i >= 0 (ex-ante probabilities over deterministic policies)
//! and z (worst-case alternative utility). The constraints are:
//!   sum_i q_i = 1
//!   sum_i q_i * size_i(null_j) <= alpha, for every declared null model j
//!   sum_i q_i * cost_i(alt_k) <= cap, for every alternative k (when capped)
//!   z <= sum_i q_i * utility_i(alt_k), for every alternative k
//!
//! Every basic feasible vertex is recovered by enumerating active-constraint
//! bases and deterministic partial-pivot Gaussian elimination. The frontier
//! and model-family sizes are deliberately bounded so this remains an exact
//! audit/reference solver rather than an unbounded general-purpose LP engine.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, error::Error, fmt};

const PLAN_SCHEMA: &str = "symthaea.architecture-engineering.policy-frontier-lp-plan";
const PLAN_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str = "symthaea.architecture-engineering.policy-frontier-lp-report";
const REPORT_VERSION: &str = "0.1.0";
const FEASIBILITY_TOLERANCE: f64 = 1e-9;
const PIVOT_TOLERANCE: f64 = 1e-11;
const OBJECTIVE_TOLERANCE: f64 = 1e-11;
const MAX_POLICIES: usize = 6;
const MAX_NULL_MODELS: usize = 4;
const MAX_ALTERNATIVE_MODELS: usize = 4;
const MAX_BASIS_COUNT: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyFrontierError {
    pub path: String,
    pub message: String,
}

impl PolicyFrontierError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}

impl fmt::Display for PolicyFrontierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for PolicyFrontierError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierNullReceipt {
    pub model_id: String,
    pub null_size: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierAlternativeReceipt {
    pub model_id: String,
    pub rejection_power: f64,
    pub expected_evaluator_cost: f64,
    pub decision_utility: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierPolicyReceipt {
    pub policy_id: String,
    /// Digest of the utility function, loss weights and cost convention.
    pub utility_contract_sha256: String,
    pub null_models: Vec<FrontierNullReceipt>,
    pub alternatives: Vec<FrontierAlternativeReceipt>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyFrontierPlan {
    pub schema: String,
    pub version: String,
    pub plan_id: String,
    pub significance_level: f64,
    pub maximum_expected_cost: Option<f64>,
    pub policies: Vec<FrontierPolicyReceipt>,
}

impl PolicyFrontierPlan {
    pub fn new(
        plan_id: impl Into<String>,
        significance_level: f64,
        maximum_expected_cost: Option<f64>,
        policies: Vec<FrontierPolicyReceipt>,
    ) -> Result<Self, PolicyFrontierError> {
        let mut plan = Self {
            schema: PLAN_SCHEMA.into(),
            version: PLAN_VERSION.into(),
            plan_id: plan_id.into(),
            significance_level,
            maximum_expected_cost,
            policies,
        };
        plan.canonicalize()?;
        Ok(plan)
    }

    pub fn canonicalize(&mut self) -> Result<(), PolicyFrontierError> {
        self.policies.sort_by(|a, b| a.policy_id.cmp(&b.policy_id));
        for policy in &mut self.policies {
            policy.null_models.sort_by(|a, b| a.model_id.cmp(&b.model_id));
            policy.alternatives.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        }
        self.validate()
    }

    pub fn validate(&self) -> Result<(), PolicyFrontierError> {
        if self.schema != PLAN_SCHEMA || self.version != PLAN_VERSION {
            return Err(PolicyFrontierError::new(
                "plan.schema",
                "unsupported policy-frontier LP schema/version",
            ));
        }
        if self.plan_id.trim().is_empty() {
            return Err(PolicyFrontierError::new("plan.plan_id", "plan ID must be non-empty"));
        }
        if !self.significance_level.is_finite()
            || !(0.0..1.0).contains(&self.significance_level)
        {
            return Err(PolicyFrontierError::new(
                "plan.significance_level",
                "alpha must be finite and strictly between zero and one",
            ));
        }
        if let Some(cap) = self.maximum_expected_cost {
            validate_nonnegative(cap, "plan.maximum_expected_cost")?;
        }
        if self.policies.len() < 2 || self.policies.len() > MAX_POLICIES {
            return Err(PolicyFrontierError::new(
                "plan.policies",
                format!("policy frontier must contain between two and {MAX_POLICIES} policies"),
            ));
        }

        let mut policy_ids = BTreeSet::new();
        let first = &self.policies[0];
        if first.null_models.is_empty() || first.null_models.len() > MAX_NULL_MODELS {
            return Err(PolicyFrontierError::new(
                "plan.policies[0].null_models",
                "null model count is empty or exceeds the supported ceiling",
            ));
        }
        if first.alternatives.len() < 2 || first.alternatives.len() > MAX_ALTERNATIVE_MODELS {
            return Err(PolicyFrontierError::new(
                "plan.policies[0].alternatives",
                "alternative model count is outside supported bounds",
            ));
        }

        let expected_nulls = first.null_models.iter()
            .map(|row| row.model_id.as_str()).collect::<BTreeSet<_>>();
        let expected_alternatives = first.alternatives.iter()
            .map(|row| row.model_id.as_str()).collect::<BTreeSet<_>>();
        validate_model_ids(&expected_nulls, "plan.policies[0].null_models")?;
        validate_model_ids(&expected_alternatives, "plan.policies[0].alternatives")?;
        validate_frontier_policy(first, 0, &expected_nulls, &expected_alternatives)?;

        let utility_hash = first.utility_contract_sha256.as_str();
        validate_hash(utility_hash, "plan.policies[0].utility_contract_sha256")?;
        policy_ids.insert(first.policy_id.as_str());

        for (index, policy) in self.policies.iter().enumerate().skip(1) {
            let path = format!("plan.policies[{index}]");
            if policy.policy_id.trim().is_empty() || !policy_ids.insert(policy.policy_id.as_str()) {
                return Err(PolicyFrontierError::new(
                    format!("{path}.policy_id"),
                    "policy IDs must be non-empty and unique",
                ));
            }
            validate_hash(
                &policy.utility_contract_sha256,
                &format!("{path}.utility_contract_sha256"),
            )?;
            if policy.utility_contract_sha256 != utility_hash {
                return Err(PolicyFrontierError::new(
                    format!("{path}.utility_contract_sha256"),
                    "all policy receipts must use the identical utility contract",
                ));
            }
            let null_ids = policy.null_models.iter()
                .map(|row| row.model_id.as_str()).collect::<BTreeSet<_>>();
            let alternative_ids = policy.alternatives.iter()
                .map(|row| row.model_id.as_str()).collect::<BTreeSet<_>>();
            if null_ids != expected_nulls {
                return Err(PolicyFrontierError::new(
                    format!("{path}.null_models"),
                    "policy receipt does not contain the identical declared null family",
                ));
            }
            if alternative_ids != expected_alternatives {
                return Err(PolicyFrontierError::new(
                    format!("{path}.alternatives"),
                    "policy receipt does not contain the identical declared alternative family",
                ));
            }
            validate_frontier_policy(policy, index, &expected_nulls, &expected_alternatives)?;
        }
        Ok(())
    }

    pub fn plan_sha256(&self) -> Result<String, PolicyFrontierError> {
        let mut canonical = self.clone();
        canonical.canonicalize()?;
        digest(&canonical)
    }
}

fn validate_frontier_policy(
    policy: &FrontierPolicyReceipt,
    index: usize,
    expected_nulls: &BTreeSet<&str>,
    expected_alternatives: &BTreeSet<&str>,
) -> Result<(), PolicyFrontierError> {
    let path = format!("plan.policies[{index}]");
    if policy.policy_id.trim().is_empty() {
        return Err(PolicyFrontierError::new(
            format!("{path}.policy_id"),
            "policy ID must be non-empty",
        ));
    }
    validate_hash(
        &policy.utility_contract_sha256,
        &format!("{path}.utility_contract_sha256"),
    )?;
    if policy.null_models.len() != expected_nulls.len() {
        return Err(PolicyFrontierError::new(
            format!("{path}.null_models"),
            "null model receipt count differs from the declared family",
        ));
    }
    let mut null_seen = BTreeSet::new();
    for (row_index, row) in policy.null_models.iter().enumerate() {
        if row.model_id.trim().is_empty()
            || !expected_nulls.contains(row.model_id.as_str())
            || !null_seen.insert(row.model_id.as_str())
        {
            return Err(PolicyFrontierError::new(
                format!("{path}.null_models[{row_index}].model_id"),
                "null receipt must reference one unique declared model",
            ));
        }
        validate_probability(
            row.null_size,
            &format!("{path}.null_models[{row_index}].null_size"),
        )?;
    }
    if policy.alternatives.len() != expected_alternatives.len() {
        return Err(PolicyFrontierError::new(
            format!("{path}.alternatives"),
            "alternative receipt count differs from the declared family",
        ));
    }
    let mut alternative_seen = BTreeSet::new();
    for (row_index, row) in policy.alternatives.iter().enumerate() {
        let row_path = format!("{path}.alternatives[{row_index}]");
        if row.model_id.trim().is_empty()
            || !expected_alternatives.contains(row.model_id.as_str())
            || !alternative_seen.insert(row.model_id.as_str())
        {
            return Err(PolicyFrontierError::new(
                format!("{row_path}.model_id"),
                "alternative receipt must reference one unique declared model",
            ));
        }
        validate_probability(
            row.rejection_power,
            &format!("{row_path}.rejection_power"),
        )?;
        validate_nonnegative(
            row.expected_evaluator_cost,
            &format!("{row_path}.expected_evaluator_cost"),
        )?;
        if !row.decision_utility.is_finite() {
            return Err(PolicyFrontierError::new(
                format!("{row_path}.decision_utility"),
                "decision utility must be finite",
            ));
        }
    }
    Ok(())
}

fn validate_model_ids(ids: &BTreeSet<&str>, path: &str) -> Result<(), PolicyFrontierError> {
    if ids.is_empty() || ids.iter().any(|id| id.trim().is_empty()) {
        return Err(PolicyFrontierError::new(
            path,
            "model IDs must be non-empty",
        ));
    }
    Ok(())
}

fn validate_hash(value: &str, path: &str) -> Result<(), PolicyFrontierError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PolicyFrontierError::new(
            path,
            "expected a 64-character hexadecimal SHA-256 utility-contract digest",
        ));
    }
    Ok(())
}

fn validate_probability(value: f64, path: &str) -> Result<(), PolicyFrontierError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(PolicyFrontierError::new(
            path,
            "value must be finite and lie in [0, 1]",
        ));
    }
    Ok(())
}

fn validate_nonnegative(value: f64, path: &str) -> Result<(), PolicyFrontierError> {
    if !value.is_finite() || value < 0.0 {
        return Err(PolicyFrontierError::new(
            path,
            "value must be finite and nonnegative",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct LinearConstraint {
    coefficients: Vec<f64>,
    rhs: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierPolicyWeight {
    pub policy_id: String,
    pub probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierNullSize {
    pub model_id: String,
    pub null_size: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierAlternativeMetrics {
    pub model_id: String,
    pub rejection_power: f64,
    pub expected_evaluator_cost: f64,
    pub decision_utility: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyFrontierReport {
    pub schema: String,
    pub version: String,
    pub plan_id: String,
    pub plan_sha256: String,
    pub enumerated_basis_count: usize,
    pub feasible_vertex_count: usize,
    pub support_size: usize,
    pub policy_weights: Vec<FrontierPolicyWeight>,
    pub null_model_sizes: Vec<FrontierNullSize>,
    pub alternatives: Vec<FrontierAlternativeMetrics>,
    pub worst_case_null_size: f64,
    pub worst_case_rejection_power: f64,
    pub worst_case_decision_utility: f64,
    pub maximum_expected_evaluator_cost: f64,
    pub mixture_entropy_nats: f64,
    pub remaining_alpha_budget: f64,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    plan_id: &'a str,
    plan_sha256: &'a str,
    enumerated_basis_count: usize,
    feasible_vertex_count: usize,
    support_size: usize,
    policy_weights: &'a [FrontierPolicyWeight],
    null_model_sizes: &'a [FrontierNullSize],
    alternatives: &'a [FrontierAlternativeMetrics],
    worst_case_null_size: f64,
    worst_case_rejection_power: f64,
    worst_case_decision_utility: f64,
    maximum_expected_evaluator_cost: f64,
    mixture_entropy_nats: f64,
    remaining_alpha_budget: f64,
}

#[derive(Debug, Clone)]
struct FrontierCandidate {
    weights: Vec<f64>,
    null_sizes: Vec<FrontierNullSize>,
    alternatives: Vec<FrontierAlternativeMetrics>,
    worst_null_size: f64,
    worst_power: f64,
    worst_utility: f64,
    maximum_cost: f64,
    entropy: f64,
}

/// An additional hard linear inequality over the randomized policy weights.
///
/// The represented relation is sum_i coefficients[i] * q_i <= upper_bound.
/// Coefficients are bound to the canonical policy order of the supplied plan.
/// These constraints are included in active-basis enumeration, not applied as
/// a post-hoc penalty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierAdditionalLinearConstraint {
    pub constraint_id: String,
    pub coefficients: Vec<f64>,
    pub upper_bound: f64,
}

fn validate_additional_constraints(
    plan: &PolicyFrontierPlan,
    additional: &[FrontierAdditionalLinearConstraint],
) -> Result<(), PolicyFrontierError> {
    let mut ids = BTreeSet::new();
    for (index, constraint) in additional.iter().enumerate() {
        let path = format!("additional_constraints[{index}]");
        if constraint.constraint_id.trim().is_empty()
            || !ids.insert(constraint.constraint_id.as_str())
        {
            return Err(PolicyFrontierError::new(
                format!("{path}.constraint_id"),
                "constraint IDs must be non-empty and unique",
            ));
        }
        if constraint.coefficients.len() != plan.policies.len()
            || constraint.coefficients.iter().any(|value| !value.is_finite())
            || !constraint.upper_bound.is_finite()
        {
            return Err(PolicyFrontierError::new(
                &path,
                "constraint coefficients must match the policy count and all values must be finite",
            ));
        }
    }
    Ok(())
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn solve_square_system(
    matrix: &[Vec<f64>],
    rhs: &[f64],
) -> Option<Vec<f64>> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut augmented = matrix.iter().zip(rhs).map(|(row, value)| {
        let mut result = row.clone();
        result.push(*value);
        result
    }).collect::<Vec<_>>();

    for column in 0..n {
        let pivot = (column..n).max_by(|left, right| {
            augmented[*left][column].abs().total_cmp(&augmented[*right][column].abs())
        })?;
        if augmented[pivot][column].abs() <= PIVOT_TOLERANCE {
            return None;
        }
        augmented.swap(column, pivot);
        let scale = augmented[column][column];
        for entry in column..=n {
            augmented[column][entry] /= scale;
        }
        for row in 0..n {
            if row == column {
                continue;
            }
            let factor = augmented[row][column];
            if factor.abs() <= PIVOT_TOLERANCE {
                continue;
            }
            for entry in column..=n {
                augmented[row][entry] -= factor * augmented[column][entry];
            }
        }
    }
    let solution = augmented.into_iter().map(|row| row[n]).collect::<Vec<_>>();
    solution.iter().all(|value| value.is_finite()).then_some(solution)
}

fn choose_count(n: usize, k: usize) -> Option<usize> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for index in 0..k {
        result = result.checked_mul(n - index)?;
        result /= index + 1;
        if result > MAX_BASIS_COUNT {
            return None;
        }
    }
    Some(result)
}

fn enumerate_combinations(
    total: usize,
    choose: usize,
    start: usize,
    current: &mut Vec<usize>,
    output: &mut Vec<Vec<usize>>,
) {
    if current.len() == choose {
        output.push(current.clone());
        return;
    }
    let remaining = choose - current.len();
    if total < remaining || start > total - remaining {
        return;
    }
    for index in start..=(total - remaining) {
        current.push(index);
        enumerate_combinations(total, choose, index + 1, current, output);
        current.pop();
    }
}

fn build_constraints(
    plan: &PolicyFrontierPlan,
) -> Result<Vec<LinearConstraint>, PolicyFrontierError> {
    let policy_count = plan.policies.len();
    let variable_count = policy_count + 1; // q[0..policy_count], z
    let mut constraints = Vec::new();

    for policy_index in 0..policy_count {
        let mut coefficients = vec![0.0; variable_count];
        coefficients[policy_index] = -1.0; // -q_i <= 0
        constraints.push(LinearConstraint { coefficients, rhs: 0.0 });
    }

    let null_ids = plan.policies[0].null_models.iter()
        .map(|row| row.model_id.as_str()).collect::<Vec<_>>();
    for null_id in null_ids {
        let mut coefficients = vec![0.0; variable_count];
        for (policy_index, policy) in plan.policies.iter().enumerate() {
            let row = policy.null_models.iter()
                .find(|row| row.model_id == null_id)
                .ok_or_else(|| PolicyFrontierError::new(
                    "plan.null_models",
                    format!("policy {} lacks null model {null_id}", policy.policy_id),
                ))?;
            coefficients[policy_index] = row.null_size;
        }
        constraints.push(LinearConstraint {
            coefficients,
            rhs: plan.significance_level,
        });
    }

    if let Some(cap) = plan.maximum_expected_cost {
        let alternative_ids = plan.policies[0].alternatives.iter()
            .map(|row| row.model_id.as_str()).collect::<Vec<_>>();
        for alternative_id in alternative_ids {
            let mut coefficients = vec![0.0; variable_count];
            for (policy_index, policy) in plan.policies.iter().enumerate() {
                let row = policy.alternatives.iter()
                    .find(|row| row.model_id == alternative_id)
                    .ok_or_else(|| PolicyFrontierError::new(
                        "plan.alternatives",
                        format!("policy {} lacks alternative {alternative_id}", policy.policy_id),
                    ))?;
                coefficients[policy_index] = row.expected_evaluator_cost;
            }
            constraints.push(LinearConstraint { coefficients, rhs: cap });
        }
    }

    let alternative_ids = plan.policies[0].alternatives.iter()
        .map(|row| row.model_id.as_str()).collect::<Vec<_>>();
    for alternative_id in alternative_ids {
        let mut coefficients = vec![0.0; variable_count];
        for (policy_index, policy) in plan.policies.iter().enumerate() {
            let row = policy.alternatives.iter()
                .find(|row| row.model_id == alternative_id)
                .ok_or_else(|| PolicyFrontierError::new(
                    "plan.alternatives",
                    format!("policy {} lacks alternative {alternative_id}", policy.policy_id),
                ))?;
            coefficients[policy_index] = -row.decision_utility;
        }
        coefficients[policy_count] = 1.0; // z - sum(q_i * utility_i,k) <= 0
        constraints.push(LinearConstraint { coefficients, rhs: 0.0 });
    }
    Ok(constraints)
}

fn evaluate_weights(
    plan: &PolicyFrontierPlan,
    weights: Vec<f64>,
) -> Result<Option<FrontierCandidate>, PolicyFrontierError> {
    if weights.len() != plan.policies.len()
        || weights.iter().any(|weight| !weight.is_finite() || *weight < -FEASIBILITY_TOLERANCE)
    {
        return Ok(None);
    }
    let sum = weights.iter().sum::<f64>();
    if !sum.is_finite() || (sum - 1.0).abs() > 1e-7 {
        return Ok(None);
    }
    let weights = weights.into_iter().map(|weight| weight.max(0.0) / sum).collect::<Vec<_>>();

    let null_ids = &plan.policies[0].null_models;
    let mut null_sizes = Vec::with_capacity(null_ids.len());
    let mut worst_null_size = 0.0_f64;
    for null in null_ids {
        let size = plan.policies.iter().zip(&weights).map(|(policy, weight)| {
            let row = policy.null_models.iter()
                .find(|row| row.model_id == null.model_id)
                .map(|row| row.null_size)
                .unwrap_or(f64::NAN);
            weight * row
        }).sum::<f64>();
        if !size.is_finite() || size > plan.significance_level + FEASIBILITY_TOLERANCE {
            return Ok(None);
        }
        worst_null_size = worst_null_size.max(size);
        null_sizes.push(FrontierNullSize { model_id: null.model_id.clone(), null_size: size });
    }

    let mut alternatives = Vec::new();
    let mut worst_power = f64::INFINITY;
    let mut worst_utility = f64::INFINITY;
    let mut maximum_cost = 0.0_f64;
    for alternative in &plan.policies[0].alternatives {
        let mut power = 0.0;
        let mut cost = 0.0;
        let mut utility = 0.0;
        for (policy, weight) in plan.policies.iter().zip(&weights) {
            let row = policy.alternatives.iter()
                .find(|row| row.model_id == alternative.model_id)
                .ok_or_else(|| PolicyFrontierError::new(
                    "evaluation.alternatives",
                    "policy alternative disappeared after validation",
                ))?;
            power += weight * row.rejection_power;
            cost += weight * row.expected_evaluator_cost;
            utility += weight * row.decision_utility;
        }
        if !power.is_finite() || !cost.is_finite() || !utility.is_finite() {
            return Err(PolicyFrontierError::new(
                "evaluation.alternatives",
                "mixed alternative metrics are non-finite",
            ));
        }
        if plan.maximum_expected_cost.is_some_and(|cap| cost > cap + FEASIBILITY_TOLERANCE) {
            return Ok(None);
        }
        worst_power = worst_power.min(power);
        worst_utility = worst_utility.min(utility);
        maximum_cost = maximum_cost.max(cost);
        alternatives.push(FrontierAlternativeMetrics {
            model_id: alternative.model_id.clone(),
            rejection_power: power,
            expected_evaluator_cost: cost,
            decision_utility: utility,
        });
    }

    let entropy = weights.iter().filter(|weight| **weight > 0.0)
        .map(|weight| -weight * weight.ln())
        .sum::<f64>();
    Ok(Some(FrontierCandidate {
        weights,
        null_sizes,
        alternatives,
        worst_null_size,
        worst_power,
        worst_utility,
        maximum_cost,
        entropy,
    }))
}

fn lexicographically_better(left: &[f64], right: &[f64]) -> bool {
    for (a, b) in left.iter().zip(right) {
        if a < b - OBJECTIVE_TOLERANCE {
            return true;
        }
        if a > b + OBJECTIVE_TOLERANCE {
            return false;
        }
    }
    false
}

fn better_frontier_candidate(left: &FrontierCandidate, right: &FrontierCandidate) -> bool {
    if left.worst_utility > right.worst_utility + OBJECTIVE_TOLERANCE {
        return true;
    }
    if (left.worst_utility - right.worst_utility).abs() > OBJECTIVE_TOLERANCE {
        return false;
    }
    if left.maximum_cost < right.maximum_cost - OBJECTIVE_TOLERANCE {
        return true;
    }
    if (left.maximum_cost - right.maximum_cost).abs() > OBJECTIVE_TOLERANCE {
        return false;
    }
    if left.worst_null_size < right.worst_null_size - OBJECTIVE_TOLERANCE {
        return true;
    }
    if (left.worst_null_size - right.worst_null_size).abs() > OBJECTIVE_TOLERANCE {
        return false;
    }
    lexicographically_better(&left.weights, &right.weights)
}

fn same_weights(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(a, b)| (a - b).abs() <= 1e-8)
}

pub fn analyze_policy_frontier(
    input: &PolicyFrontierPlan,
) -> Result<PolicyFrontierReport, PolicyFrontierError> {
    analyze_policy_frontier_with_constraints(input, &[])
}

/// Optimize the existing robust policy frontier with additional hard linear
/// constraints over policy-mixture probabilities. Additional inequalities
/// participate in exhaustive active-basis enumeration and feasibility checks.
/// They are not utility penalties and cannot be traded away for objective value.
pub fn analyze_policy_frontier_with_constraints(
    input: &PolicyFrontierPlan,
    additional_constraints: &[FrontierAdditionalLinearConstraint],
) -> Result<PolicyFrontierReport, PolicyFrontierError> {
    let mut plan = input.clone();
    plan.canonicalize()?;
    validate_additional_constraints(&plan, additional_constraints)?;
    let mut constraints = build_constraints(&plan)?;
    for constraint in additional_constraints {
        let mut coefficients = constraint.coefficients.clone();
        coefficients.push(0.0); // no coefficient on the utility hypograph variable z
        constraints.push(LinearConstraint {
            coefficients,
            rhs: constraint.upper_bound,
        });
    }
    let policy_count = plan.policies.len();
    let variable_count = policy_count + 1;
    let basis_count = choose_count(constraints.len(), policy_count).ok_or_else(|| {
        PolicyFrontierError::new(
            "plan.frontier",
            "active-constraint basis count exceeds the exact enumeration ceiling",
        )
    })?;

    let mut bases = Vec::with_capacity(basis_count);
    enumerate_combinations(
        constraints.len(),
        policy_count,
        0,
        &mut Vec::new(),
        &mut bases,
    );

    let equality = {
        let mut row = vec![1.0; variable_count];
        row[policy_count] = 0.0;
        row
    };
    let mut feasible_vertices: Vec<FrontierCandidate> = Vec::new();

    for basis in &bases {
        let mut matrix = Vec::with_capacity(variable_count);
        let mut rhs = Vec::with_capacity(variable_count);
        matrix.push(equality.clone());
        rhs.push(1.0);
        for constraint_index in basis {
            matrix.push(constraints[*constraint_index].coefficients.clone());
            rhs.push(constraints[*constraint_index].rhs);
        }
        let Some(solution) = solve_square_system(&matrix, &rhs) else {
            continue;
        };
        if solution.len() != variable_count {
            continue;
        }
        let weights = solution[..policy_count].to_vec();
        let Some(candidate) = evaluate_weights(&plan, weights)? else {
            continue;
        };
        if additional_constraints.iter().any(|constraint| {
            dot(&constraint.coefficients, &candidate.weights)
                > constraint.upper_bound + FEASIBILITY_TOLERANCE
        }) {
            continue;
        }
        // The LP basis must satisfy every utility hypograph inequality, and
        // z must touch the lower envelope. Reconstructing the lower envelope
        // independently prevents a malformed basis from overstating utility.
        let basis_utility = solution[policy_count];
        if !basis_utility.is_finite()
            || basis_utility > candidate.worst_utility + FEASIBILITY_TOLERANCE
            || (basis_utility - candidate.worst_utility).abs() > 1e-7
        {
            continue;
        }
        if !feasible_vertices.iter().any(|existing| same_weights(&existing.weights, &candidate.weights)) {
            feasible_vertices.push(candidate);
        }
    }

    if feasible_vertices.is_empty() {
        return Err(PolicyFrontierError::new(
            "evaluation.feasible_vertices",
            "no feasible randomized policy vertex satisfies all null and cost constraints",
        ));
    }
    let feasible_vertex_count = feasible_vertices.len();
    let best = feasible_vertices.into_iter().reduce(|incumbent, candidate| {
        if better_frontier_candidate(&candidate, &incumbent) {
            candidate
        } else {
            incumbent
        }
    }).ok_or_else(|| PolicyFrontierError::new(
        "evaluation.feasible_vertices",
        "feasible vertex set unexpectedly empty",
    ))?;

    let policy_weights = plan.policies.iter().zip(&best.weights)
        .filter(|(_, weight)| **weight > FEASIBILITY_TOLERANCE)
        .map(|(policy, weight)| FrontierPolicyWeight {
            policy_id: policy.policy_id.clone(),
            probability: *weight,
        })
        .collect::<Vec<_>>();
    let support_size = policy_weights.len();
    let plan_sha256 = plan.plan_sha256()?;
    let payload = ReportPayload {
        schema: REPORT_SCHEMA,
        version: REPORT_VERSION,
        plan_id: &plan.plan_id,
        plan_sha256: &plan_sha256,
        enumerated_basis_count: bases.len(),
        feasible_vertex_count,
        support_size,
        policy_weights: &policy_weights,
        null_model_sizes: &best.null_sizes,
        alternatives: &best.alternatives,
        worst_case_null_size: best.worst_null_size,
        worst_case_rejection_power: best.worst_power,
        worst_case_decision_utility: best.worst_utility,
        maximum_expected_evaluator_cost: best.maximum_cost,
        mixture_entropy_nats: best.entropy,
        remaining_alpha_budget: plan.significance_level - best.worst_null_size,
    };
    let report_sha256 = digest(&payload)?;
    Ok(PolicyFrontierReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        plan_id: plan.plan_id,
        plan_sha256,
        enumerated_basis_count: bases.len(),
        feasible_vertex_count,
        support_size,
        policy_weights,
        null_model_sizes: best.null_sizes,
        alternatives: best.alternatives,
        worst_case_null_size: best.worst_null_size,
        worst_case_rejection_power: best.worst_power,
        worst_case_decision_utility: best.worst_utility,
        maximum_expected_evaluator_cost: best.maximum_cost,
        mixture_entropy_nats: best.entropy,
        remaining_alpha_budget: plan.significance_level - best.worst_null_size,
        report_sha256,
    })
}

/// Verify the canonical payload digest of a policy-frontier report without
/// rerunning the optimizer. This checks report integrity, not LP optimality.
pub fn verify_policy_frontier_report_digest(
    report: &PolicyFrontierReport,
) -> Result<(), PolicyFrontierError> {
    let payload = ReportPayload {
        schema: &report.schema,
        version: &report.version,
        plan_id: &report.plan_id,
        plan_sha256: &report.plan_sha256,
        enumerated_basis_count: report.enumerated_basis_count,
        feasible_vertex_count: report.feasible_vertex_count,
        support_size: report.support_size,
        policy_weights: &report.policy_weights,
        null_model_sizes: &report.null_model_sizes,
        alternatives: &report.alternatives,
        worst_case_null_size: report.worst_case_null_size,
        worst_case_rejection_power: report.worst_case_rejection_power,
        worst_case_decision_utility: report.worst_case_decision_utility,
        maximum_expected_evaluator_cost: report.maximum_expected_evaluator_cost,
        mixture_entropy_nats: report.mixture_entropy_nats,
        remaining_alpha_budget: report.remaining_alpha_budget,
    };
    if digest(&payload)? != report.report_sha256 {
        return Err(PolicyFrontierError::new(
            "report.report_sha256",
            "policy-frontier report digest does not match its canonical payload",
        ));
    }
    Ok(())
}

pub fn verify_policy_frontier_report(
    plan: &PolicyFrontierPlan,
    report: &PolicyFrontierReport,
) -> Result<(), PolicyFrontierError> {
    if analyze_policy_frontier(plan)? != *report {
        return Err(PolicyFrontierError::new(
            "report",
            "stored policy-frontier report differs from exact LP reconstruction",
        ));
    }
    Ok(())
}

fn digest<T: Serialize>(value: &T) -> Result<String, PolicyFrontierError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        PolicyFrontierError::new("serialization", error.to_string())
    })?;
    Ok(Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const UTILITY_HASH: &str =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn policy(
        id: &str,
        cost: f64,
        utility_vector: [f64; 3],
    ) -> FrontierPolicyReceipt {
        let ids = ["alt-a", "alt-b", "alt-c"];
        FrontierPolicyReceipt {
            policy_id: id.into(),
            utility_contract_sha256: UTILITY_HASH.into(),
            null_models: vec![
                FrontierNullReceipt { model_id: "null-a".into(), null_size: 0.02 },
                FrontierNullReceipt { model_id: "null-b".into(), null_size: 0.03 },
            ],
            alternatives: ids.into_iter().zip(utility_vector).map(|(model_id, utility)| {
                FrontierAlternativeReceipt {
                    model_id: model_id.into(),
                    rejection_power: utility,
                    expected_evaluator_cost: cost,
                    decision_utility: utility,
                }
            }).collect(),
        }
    }

    fn fixture(cost_cap: Option<f64>) -> PolicyFrontierPlan {
        PolicyFrontierPlan::new(
            "three-policy-frontier-convexification",
            0.05,
            cost_cap,
            vec![
                policy("policy-a", 1.0, [1.0, 0.0, 0.0]),
                policy("policy-b", 2.0, [0.0, 1.0, 0.0]),
                policy("policy-c", 3.0, [0.0, 0.0, 1.0]),
            ],
        ).unwrap()
    }

    #[test]
    fn three_policy_frontier_finds_a_strict_three_way_optimum() {
        let plan = fixture(None);
        let report = analyze_policy_frontier(&plan).unwrap();
        assert!((report.worst_case_decision_utility - (1.0 / 3.0)).abs() < 1e-10);
        assert!((report.worst_case_rejection_power - (1.0 / 3.0)).abs() < 1e-10);
        assert_eq!(report.support_size, 3);
        assert!(report.policy_weights.iter().all(|row| (row.probability - 1.0 / 3.0).abs() < 1e-10));
        assert!(report.worst_case_null_size <= 0.05);
        verify_policy_frontier_report(&plan, &report).unwrap();
    }

    #[test]
    fn expected_cost_constraint_changes_the_frontier_solution() {
        let report = analyze_policy_frontier(&fixture(Some(1.6))).unwrap();
        assert_eq!(report.support_size, 3);
        assert!((report.worst_case_decision_utility - 0.2).abs() < 1e-10);
        assert!((report.maximum_expected_evaluator_cost - 1.6).abs() < 1e-10);
        let weights = report.policy_weights.iter()
            .map(|row| (row.policy_id.as_str(), row.probability))
            .collect::<std::collections::BTreeMap<_, _>>();
        assert!((weights["policy-a"] - 0.6).abs() < 1e-10);
        assert!((weights["policy-b"] - 0.2).abs() < 1e-10);
        assert!((weights["policy-c"] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn composite_null_family_is_enforced_per_model() {
        let mut plan = fixture(None);
        for policy in &mut plan.policies {
            policy.null_models[0].null_size = 0.01;
            policy.null_models[1].null_size = 0.06;
        }
        assert!(analyze_policy_frontier(&plan).is_err());
    }

    #[test]
    fn mismatched_policy_receipts_and_utility_contracts_fail_closed() {
        let mut missing = fixture(None);
        missing.policies[1].alternatives.pop();
        assert!(missing.validate().is_err());
        let mut hash = fixture(None);
        hash.policies[2].utility_contract_sha256 = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".into();
        assert!(hash.validate().is_err());
        let mut null_family = fixture(None);
        null_family.policies[1].null_models[0].model_id = "omitted-null".into();
        assert!(null_family.validate().is_err());
    }

    #[test]
    fn infeasible_cost_and_invalid_receipt_fail_closed() {
        assert!(analyze_policy_frontier(&fixture(Some(0.5))).is_err());
        let mut malformed = fixture(None);
        malformed.policies[0].alternatives[0].expected_evaluator_cost = f64::NAN;
        assert!(malformed.validate().is_err());
    }

    #[test]
    fn oversized_frontier_fails_closed() {
        let mut plan = fixture(None);
        for index in 0..4 {
            plan.policies.push(policy(
                &format!("extra-{index}"),
                1.0 + f64::from(index),
                [0.2, 0.2, 0.2],
            ));
        }
        assert!(plan.validate().is_err());
    }

    #[test]
    fn canonical_order_and_report_tampering_are_detected() {
        let plan = fixture(None);
        let report = analyze_policy_frontier(&plan).unwrap();
        let mut reordered = plan.clone();
        reordered.policies.reverse();
        for policy in &mut reordered.policies {
            policy.null_models.reverse();
            policy.alternatives.reverse();
        }
        assert_eq!(report, analyze_policy_frontier(&reordered).unwrap());
        let mut altered = report.clone();
        altered.worst_case_decision_utility += 0.01;
        assert!(verify_policy_frontier_report(&plan, &altered).is_err());
    }
}
