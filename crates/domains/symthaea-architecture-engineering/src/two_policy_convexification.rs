//! Exact two-policy randomized convexification for finite robust decision utility.
//!
//! A single random draw, made before observing data, selects complete policy A
//! with probability q or policy B with probability 1-q. Under every fixed
//! P0/P1 model, rejection size, power, expected cost and utility are linear in q.
//!
//! For a finite alternative family, worst-case utility is the lower envelope of
//! finitely many affine functions. Its maximum over the feasible q interval occurs
//! at an endpoint or at an intersection of two model-utility lines. Feasibility
//! boundaries add only linear roots. Enumerating these candidates is exact.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, error::Error, fmt};

const PLAN_SCHEMA: &str = "symthaea.architecture-engineering.two-policy-convexification-plan";
const PLAN_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str = "symthaea.architecture-engineering.two-policy-convexification-report";
const REPORT_VERSION: &str = "0.1.0";
const PROBABILITY_TOLERANCE: f64 = 1e-10;
const OBJECTIVE_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyMixtureError {
    pub path: String,
    pub message: String,
}

impl PolicyMixtureError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}

impl fmt::Display for PolicyMixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for PolicyMixtureError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyAlternativeReceipt {
    pub model_id: String,
    pub rejection_power: f64,
    pub expected_evaluator_cost: f64,
    pub decision_utility: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyNullModelReceipt {
    pub model_id: String,
    pub null_size: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeterministicPolicyReceipt {
    pub policy_id: String,
    /// Hash of the utility definition and weights used to calculate utilities.
    pub utility_contract_sha256: String,
    /// A separate type-I error size under every declared null model.
    pub null_models: Vec<PolicyNullModelReceipt>,
    pub alternatives: Vec<PolicyAlternativeReceipt>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TwoPolicyMixturePlan {
    pub schema: String,
    pub version: String,
    pub mixture_id: String,
    pub significance_level: f64,
    pub maximum_expected_cost: Option<f64>,
    pub policy_a: DeterministicPolicyReceipt,
    pub policy_b: DeterministicPolicyReceipt,
}

impl TwoPolicyMixturePlan {
    pub fn new(
        mixture_id: impl Into<String>,
        significance_level: f64,
        maximum_expected_cost: Option<f64>,
        policy_a: DeterministicPolicyReceipt,
        policy_b: DeterministicPolicyReceipt,
    ) -> Result<Self, PolicyMixtureError> {
        let mut plan = Self {
            schema: PLAN_SCHEMA.into(),
            version: PLAN_VERSION.into(),
            mixture_id: mixture_id.into(),
            significance_level,
            maximum_expected_cost,
            policy_a,
            policy_b,
        };
        plan.canonicalize()?;
        Ok(plan)
    }

    pub fn canonicalize(&mut self) -> Result<(), PolicyMixtureError> {
        self.policy_a.null_models.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        self.policy_b.null_models.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        self.policy_a.alternatives.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        self.policy_b.alternatives.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        self.validate()
    }

    pub fn validate(&self) -> Result<(), PolicyMixtureError> {
        if self.schema != PLAN_SCHEMA || self.version != PLAN_VERSION {
            return Err(PolicyMixtureError::new(
                "plan.schema",
                "unsupported two-policy mixture schema/version",
            ));
        }
        if self.mixture_id.trim().is_empty() {
            return Err(PolicyMixtureError::new("plan.mixture_id", "mixture ID must be non-empty"));
        }
        if !self.significance_level.is_finite()
            || !(0.0..1.0).contains(&self.significance_level)
        {
            return Err(PolicyMixtureError::new(
                "plan.significance_level",
                "alpha must be finite and strictly between zero and one",
            ));
        }
        if let Some(cap) = self.maximum_expected_cost {
            validate_nonnegative(cap, "plan.maximum_expected_cost")?;
        }
        if self.policy_a.policy_id.trim().is_empty()
            || self.policy_b.policy_id.trim().is_empty()
            || self.policy_a.policy_id == self.policy_b.policy_id
        {
            return Err(PolicyMixtureError::new(
                "plan.policy_ids",
                "policy IDs must be non-empty and distinct",
            ));
        }
        validate_contract_hash(
            &self.policy_a.utility_contract_sha256,
            "plan.policy_a.utility_contract_sha256",
        )?;
        validate_contract_hash(
            &self.policy_b.utility_contract_sha256,
            "plan.policy_b.utility_contract_sha256",
        )?;
        if self.policy_a.utility_contract_sha256 != self.policy_b.utility_contract_sha256 {
            return Err(PolicyMixtureError::new(
                "plan.utility_contract_sha256",
                "both policy receipts must use the same utility definition",
            ));
        }
        validate_null_receipts(&self.policy_a, "plan.policy_a")?;
        validate_null_receipts(&self.policy_b, "plan.policy_b")?;
        validate_alternative_receipts(&self.policy_a, "plan.policy_a")?;
        validate_alternative_receipts(&self.policy_b, "plan.policy_b")?;

        if self.policy_a.alternatives.len() < 2
            || self.policy_a.alternatives.len() != self.policy_b.alternatives.len()
        {
            return Err(PolicyMixtureError::new(
                "plan.alternatives",
                "both policies must contain the same family of at least two alternative models",
            ));
        }
        let models_a = self.policy_a.alternatives.iter()
            .map(|row| row.model_id.as_str()).collect::<BTreeSet<_>>();
        let models_b = self.policy_b.alternatives.iter()
            .map(|row| row.model_id.as_str()).collect::<BTreeSet<_>>();
        if models_a != models_b {
            return Err(PolicyMixtureError::new(
                "plan.alternatives",
                "policy receipts must contain identical alternative model IDs",
            ));
        }
        let nulls_a = self.policy_a.null_models.iter()
            .map(|row| row.model_id.as_str()).collect::<BTreeSet<_>>();
        let nulls_b = self.policy_b.null_models.iter()
            .map(|row| row.model_id.as_str()).collect::<BTreeSet<_>>();
        if nulls_a != nulls_b {
            return Err(PolicyMixtureError::new(
                "plan.null_models",
                "policy receipts must contain identical null model IDs",
            ));
        }
        Ok(())
    }

    pub fn plan_sha256(&self) -> Result<String, PolicyMixtureError> {
        let mut canonical = self.clone();
        canonical.canonicalize()?;
        digest(&canonical)
    }
}

fn validate_null_receipts(
    policy: &DeterministicPolicyReceipt,
    path: &str,
) -> Result<(), PolicyMixtureError> {
    if policy.null_models.is_empty() {
        return Err(PolicyMixtureError::new(
            format!("{path}.null_models"),
            "null model size vector cannot be empty",
        ));
    }
    let mut seen = BTreeSet::new();
    for (index, row) in policy.null_models.iter().enumerate() {
        let row_path = format!("{path}.null_models[{index}]");
        if row.model_id.trim().is_empty() || !seen.insert(row.model_id.as_str()) {
            return Err(PolicyMixtureError::new(
                format!("{row_path}.model_id"),
                "null model IDs must be non-empty and unique",
            ));
        }
        validate_probability(row.null_size, &format!("{row_path}.null_size"))?;
    }
    Ok(())
}

fn null_row<'a>(
    policy: &'a DeterministicPolicyReceipt,
    model_id: &str,
) -> Result<&'a PolicyNullModelReceipt, PolicyMixtureError> {
    policy.null_models.iter()
        .find(|row| row.model_id == model_id)
        .ok_or_else(|| PolicyMixtureError::new(
            "plan.null_models",
            format!("policy {} is missing null model {model_id}", policy.policy_id),
        ))
}

fn validate_alternative_receipts(
    policy: &DeterministicPolicyReceipt,
    path: &str,
) -> Result<(), PolicyMixtureError> {
    if policy.alternatives.is_empty() {
        return Err(PolicyMixtureError::new(
            format!("{path}.alternatives"),
            "alternative evaluation list cannot be empty",
        ));
    }
    let mut seen = BTreeSet::new();
    for (index, row) in policy.alternatives.iter().enumerate() {
        let row_path = format!("{path}.alternatives[{index}]");
        if row.model_id.trim().is_empty() || !seen.insert(row.model_id.as_str()) {
            return Err(PolicyMixtureError::new(
                format!("{row_path}.model_id"),
                "alternative model IDs must be non-empty and unique",
            ));
        }
        validate_probability(row.rejection_power, &format!("{row_path}.rejection_power"))?;
        validate_nonnegative(
            row.expected_evaluator_cost,
            &format!("{row_path}.expected_evaluator_cost"),
        )?;
        if !row.decision_utility.is_finite() {
            return Err(PolicyMixtureError::new(
                format!("{row_path}.decision_utility"),
                "decision utility must be finite",
            ));
        }
    }
    Ok(())
}

fn validate_contract_hash(value: &str, path: &str) -> Result<(), PolicyMixtureError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PolicyMixtureError::new(
            path,
            "utility contract must be a 64-character hexadecimal SHA-256 digest",
        ));
    }
    Ok(())
}

fn validate_probability(value: f64, path: &str) -> Result<(), PolicyMixtureError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(PolicyMixtureError::new(
            path,
            "value must be finite and lie in [0, 1]",
        ));
    }
    Ok(())
}

fn validate_nonnegative(value: f64, path: &str) -> Result<(), PolicyMixtureError> {
    if !value.is_finite() || value < 0.0 {
        return Err(PolicyMixtureError::new(
            path,
            "value must be finite and nonnegative",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MixedNullModelSize {
    pub model_id: String,
    pub null_size: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MixedAlternativeEvaluation {
    pub model_id: String,
    pub rejection_power: f64,
    pub expected_evaluator_cost: f64,
    pub decision_utility: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TwoPolicyMixtureReport {
    pub schema: String,
    pub version: String,
    pub mixture_id: String,
    pub plan_sha256: String,
    /// Probability of selecting policy A before any observations are received.
    pub policy_a_probability: f64,
    pub policy_b_probability: f64,
    pub mixture_entropy_nats: f64,
    pub candidate_probability_count: usize,
    pub feasible_probability_count: usize,
    pub worst_case_null_size: f64,
    pub null_model_sizes: Vec<MixedNullModelSize>,
    pub worst_case_rejection_power: f64,
    pub worst_case_decision_utility: f64,
    pub maximum_expected_evaluator_cost: f64,
    pub remaining_alpha_budget: f64,
    pub alternatives: Vec<MixedAlternativeEvaluation>,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    mixture_id: &'a str,
    plan_sha256: &'a str,
    policy_a_probability: f64,
    policy_b_probability: f64,
    mixture_entropy_nats: f64,
    candidate_probability_count: usize,
    feasible_probability_count: usize,
    worst_case_null_size: f64,
    null_model_sizes: &'a [MixedNullModelSize],
    worst_case_rejection_power: f64,
    worst_case_decision_utility: f64,
    maximum_expected_evaluator_cost: f64,
    remaining_alpha_budget: f64,
    alternatives: &'a [MixedAlternativeEvaluation],
}

#[derive(Debug, Clone)]
struct MixtureCandidate {
    q: f64,
    worst_null_size: f64,
    null_model_sizes: Vec<MixedNullModelSize>,
    worst_power: f64,
    worst_utility: f64,
    maximum_cost: f64,
    alternatives: Vec<MixedAlternativeEvaluation>,
}

fn policy_row<'a>(
    policy: &'a DeterministicPolicyReceipt,
    model_id: &str,
) -> Result<&'a PolicyAlternativeReceipt, PolicyMixtureError> {
    policy.alternatives.iter()
        .find(|row| row.model_id == model_id)
        .ok_or_else(|| PolicyMixtureError::new(
            "plan.alternatives",
            format!("policy {} is missing alternative {model_id}", policy.policy_id),
        ))
}

fn add_candidate(values: &mut Vec<f64>, q: f64) {
    if q.is_finite() && (-OBJECTIVE_TOLERANCE..=1.0 + OBJECTIVE_TOLERANCE).contains(&q) {
        values.push(q.clamp(0.0, 1.0));
    }
}

fn candidate_probabilities(
    plan: &TwoPolicyMixturePlan,
) -> Result<Vec<f64>, PolicyMixtureError> {
    let mut candidates = vec![0.0, 1.0];
    // The lower envelope of affine utility lines changes slope only where
    // two model-specific utilities intersect.
    for left in 0..plan.policy_a.alternatives.len() {
        for right in left + 1..plan.policy_a.alternatives.len() {
            let left_a = plan.policy_a.alternatives[left].decision_utility;
            let left_b = plan.policy_b.alternatives[left].decision_utility;
            let right_a = plan.policy_a.alternatives[right].decision_utility;
            let right_b = plan.policy_b.alternatives[right].decision_utility;
            let slope_difference = (left_a - left_b) - (right_a - right_b);
            if slope_difference.abs() > OBJECTIVE_TOLERANCE {
                add_candidate(
                    &mut candidates,
                    (right_b - left_b) / slope_difference,
                );
            }
        }
    }

    for null_model in &plan.policy_a.null_models {
        let size_a = null_model.null_size;
        let size_b = null_row(&plan.policy_b, &null_model.model_id)?.null_size;
        let null_delta = size_a - size_b;
        if null_delta.abs() > OBJECTIVE_TOLERANCE {
            add_candidate(
                &mut candidates,
                (plan.significance_level - size_b) / null_delta,
            );
        }
    }
    if let Some(cap) = plan.maximum_expected_cost {
        for model in &plan.policy_a.alternatives {
            let a = model.expected_evaluator_cost;
            let b = policy_row(&plan.policy_b, &model.model_id)?.expected_evaluator_cost;
            let delta = a - b;
            if delta.abs() > OBJECTIVE_TOLERANCE {
                add_candidate(&mut candidates, (cap - b) / delta);
            }
        }
    }

    candidates.sort_by(f64::total_cmp);
    candidates.dedup_by(|left, right| (*left - *right).abs() <= OBJECTIVE_TOLERANCE);
    Ok(candidates)
}

fn evaluate_mixture(
    plan: &TwoPolicyMixturePlan,
    q: f64,
) -> Result<MixtureCandidate, PolicyMixtureError> {
    let null_model_sizes = plan.policy_a.null_models.iter().map(|row_a| {
        let row_b = null_row(&plan.policy_b, &row_a.model_id)?;
        Ok(MixedNullModelSize {
            model_id: row_a.model_id.clone(),
            null_size: q * row_a.null_size + (1.0 - q) * row_b.null_size,
        })
    }).collect::<Result<Vec<_>, PolicyMixtureError>>()?;
    let worst_null_size = null_model_sizes.iter()
        .map(|row| row.null_size)
        .fold(0.0_f64, f64::max);
    let mut alternatives = Vec::with_capacity(plan.policy_a.alternatives.len());
    let mut worst_power = f64::INFINITY;
    let mut worst_utility = f64::INFINITY;
    let mut maximum_cost = 0.0_f64;
    for row_a in &plan.policy_a.alternatives {
        let row_b = policy_row(&plan.policy_b, &row_a.model_id)?;
        let power = q * row_a.rejection_power + (1.0 - q) * row_b.rejection_power;
        let cost = q * row_a.expected_evaluator_cost
            + (1.0 - q) * row_b.expected_evaluator_cost;
        let utility = q * row_a.decision_utility + (1.0 - q) * row_b.decision_utility;
        worst_power = worst_power.min(power);
        worst_utility = worst_utility.min(utility);
        maximum_cost = maximum_cost.max(cost);
        alternatives.push(MixedAlternativeEvaluation {
            model_id: row_a.model_id.clone(),
            rejection_power: power,
            expected_evaluator_cost: cost,
            decision_utility: utility,
        });
    }
    if !worst_null_size.is_finite() || !worst_power.is_finite()
        || !worst_utility.is_finite() || !maximum_cost.is_finite()
    {
        return Err(PolicyMixtureError::new(
            "evaluation",
            "mixture reconstruction produced a non-finite metric",
        ));
    }
    Ok(MixtureCandidate {
        q,
        worst_null_size,
        null_model_sizes,
        worst_power,
        worst_utility,
        maximum_cost,
        alternatives,
    })
}

fn better_candidate(candidate: &MixtureCandidate, incumbent: &MixtureCandidate) -> bool {
    if candidate.worst_utility > incumbent.worst_utility + OBJECTIVE_TOLERANCE {
        return true;
    }
    if (candidate.worst_utility - incumbent.worst_utility).abs() > OBJECTIVE_TOLERANCE {
        return false;
    }
    if candidate.maximum_cost < incumbent.maximum_cost - OBJECTIVE_TOLERANCE {
        return true;
    }
    if (candidate.maximum_cost - incumbent.maximum_cost).abs() > OBJECTIVE_TOLERANCE {
        return false;
    }
    if candidate.worst_null_size < incumbent.worst_null_size - OBJECTIVE_TOLERANCE {
        return true;
    }
    if (candidate.worst_null_size - incumbent.worst_null_size).abs() > OBJECTIVE_TOLERANCE {
        return false;
    }
    candidate.q < incumbent.q
}

pub fn analyze_two_policy_mixture(
    input: &TwoPolicyMixturePlan,
) -> Result<TwoPolicyMixtureReport, PolicyMixtureError> {
    let mut plan = input.clone();
    plan.canonicalize()?;
    let candidates = candidate_probabilities(&plan)?;
    let mut best: Option<MixtureCandidate> = None;
    let mut feasible_count = 0usize;

    for q in &candidates {
        let candidate = evaluate_mixture(&plan, *q)?;
        if candidate.worst_null_size > plan.significance_level + PROBABILITY_TOLERANCE {
            continue;
        }
        if plan.maximum_expected_cost.is_some_and(|cap| {
            candidate.maximum_cost > cap + OBJECTIVE_TOLERANCE
        }) {
            continue;
        }
        feasible_count += 1;
        if best.as_ref().is_none_or(|incumbent| better_candidate(&candidate, incumbent)) {
            best = Some(candidate);
        }
    }
    let best = best.ok_or_else(|| PolicyMixtureError::new(
        "evaluation.feasible_mixtures",
        "no policy mixture satisfies null-size and cost constraints",
    ))?;

    let q = best.q;
    let entropy = if q <= 0.0 || q >= 1.0 {
        0.0
    } else {
        -q * q.ln() - (1.0 - q) * (1.0 - q).ln()
    };
    let plan_sha256 = plan.plan_sha256()?;
    let payload = ReportPayload {
        schema: REPORT_SCHEMA,
        version: REPORT_VERSION,
        mixture_id: &plan.mixture_id,
        plan_sha256: &plan_sha256,
        policy_a_probability: q,
        policy_b_probability: 1.0 - q,
        mixture_entropy_nats: entropy,
        candidate_probability_count: candidates.len(),
        feasible_probability_count: feasible_count,
        worst_case_null_size: best.worst_null_size,
        null_model_sizes: &best.null_model_sizes,
        worst_case_rejection_power: best.worst_power,
        worst_case_decision_utility: best.worst_utility,
        maximum_expected_evaluator_cost: best.maximum_cost,
        remaining_alpha_budget: plan.significance_level - best.worst_null_size,
        alternatives: &best.alternatives,
    };
    let report_sha256 = digest(&payload)?;
    Ok(TwoPolicyMixtureReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        mixture_id: plan.mixture_id,
        plan_sha256,
        policy_a_probability: q,
        policy_b_probability: 1.0 - q,
        mixture_entropy_nats: entropy,
        candidate_probability_count: candidates.len(),
        feasible_probability_count: feasible_count,
        worst_case_null_size: best.worst_null_size,
        null_model_sizes: best.null_model_sizes,
        worst_case_rejection_power: best.worst_power,
        worst_case_decision_utility: best.worst_utility,
        maximum_expected_evaluator_cost: best.maximum_cost,
        remaining_alpha_budget: plan.significance_level - best.worst_null_size,
        alternatives: best.alternatives,
        report_sha256,
    })
}

pub fn verify_two_policy_mixture_report(
    plan: &TwoPolicyMixturePlan,
    report: &TwoPolicyMixtureReport,
) -> Result<(), PolicyMixtureError> {
    if analyze_two_policy_mixture(plan)? != *report {
        return Err(PolicyMixtureError::new(
            "report",
            "stored mixture report differs from exact breakpoint reconstruction",
        ));
    }
    Ok(())
}

fn digest<T: Serialize>(value: &T) -> Result<String, PolicyMixtureError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        PolicyMixtureError::new("serialization", error.to_string())
    })?;
    Ok(Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const UTILITY_HASH: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn receipt(
        policy_id: &str,
        null_models: [(&str, f64); 2],
        rows: [(&str, f64, f64, f64); 2],
    ) -> DeterministicPolicyReceipt {
        DeterministicPolicyReceipt {
            policy_id: policy_id.into(),
            utility_contract_sha256: UTILITY_HASH.into(),
            null_models: null_models.into_iter().map(|(model_id, null_size)| {
                PolicyNullModelReceipt { model_id: model_id.into(), null_size }
            }).collect(),
            alternatives: rows.into_iter().map(|(model_id, power, cost, utility)| {
                PolicyAlternativeReceipt {
                    model_id: model_id.into(),
                    rejection_power: power,
                    expected_evaluator_cost: cost,
                    decision_utility: utility,
                }
            }).collect(),
        }
    }

    fn crossing_fixture() -> TwoPolicyMixturePlan {
        let policy_a = receipt(
            "middle",
            [("null-a", 0.035), ("null-b", 0.02)],
            [
                ("alt-a", 0.66, 3.0, 5.62),
                ("alt-b", 0.54, 3.0, 4.18),
            ],
        );
        let policy_b = receipt(
            "deep",
            [("null-a", 0.0225), ("null-b", 0.03)],
            [
                ("alt-a", 0.576, 3.7, 4.542),
                ("alt-b", 0.615, 3.3, 5.05),
            ],
        );
        TwoPolicyMixturePlan::new(
            "two-policy-robust-convexification",
            0.05,
            Some(3.6),
            policy_a,
            policy_b,
        ).unwrap()
    }

    #[test]
    fn randomized_mixture_strictly_beats_both_deterministic_policies() {
        let plan = crossing_fixture();
        let report = analyze_two_policy_mixture(&plan).unwrap();
        let deterministic_best = 4.542_f64.max(4.18);
        assert!(report.worst_case_decision_utility > deterministic_best);
        assert!((report.policy_a_probability - 0.508 / 1.948).abs() < 1e-12);
        assert!((report.worst_case_decision_utility - 4.823).abs() < 1e-3);
        assert!((report.worst_case_null_size - 0.027392197125256674).abs() < 1e-12);
        assert_eq!(report.null_model_sizes.len(), 2);
        assert!(report.worst_case_null_size <= 0.05);
        assert!(report.maximum_expected_evaluator_cost <= 3.6);
        assert!(report.mixture_entropy_nats > 0.0);
        verify_two_policy_mixture_report(&plan, &report).unwrap();
    }

    #[test]
    fn size_and_cost_constraints_remove_infeasible_mixture_intervals() {
        let mut size_cap = crossing_fixture();
        size_cap.significance_level = 0.02;
        assert!(analyze_two_policy_mixture(&size_cap).is_err());

        let mut cost_cap = crossing_fixture();
        cost_cap.maximum_expected_cost = Some(3.2);
        let report = analyze_two_policy_mixture(&cost_cap).unwrap();
        assert!(report.maximum_expected_evaluator_cost <= 3.2 + 1e-12);
        assert!(report.policy_a_probability > 0.7);
    }

    #[test]
    fn incompatible_composite_null_size_constraints_have_no_feasible_mixture() {
        let mut plan = crossing_fixture();
        // Null A requires q <= 0.28, while null B requires q >= 0.40 at alpha=0.026.
        plan.significance_level = 0.026;
        assert!(analyze_two_policy_mixture(&plan).is_err());
    }

    #[test]
    fn mismatched_model_families_and_utility_contracts_fail_closed() {
        let mut models = crossing_fixture();
        models.policy_b.alternatives[1].model_id = "omitted-model".into();
        assert!(models.validate().is_err());

        let mut utility = crossing_fixture();
        utility.policy_b.utility_contract_sha256 = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".into();
        assert!(utility.validate().is_err());
    }

    #[test]
    fn malformed_receipts_and_no_mixture_feasibility_fail_closed() {
        let mut malformed = crossing_fixture();
        malformed.policy_a.alternatives[0].rejection_power = f64::NAN;
        assert!(malformed.validate().is_err());

        let mut no_cost = crossing_fixture();
        no_cost.maximum_expected_cost = Some(1.0);
        assert!(analyze_two_policy_mixture(&no_cost).is_err());
    }

    #[test]
    fn endpoint_and_non_crossing_cases_are_handled() {
        let mut no_cross = crossing_fixture();
        no_cross.policy_b.alternatives[0].decision_utility = 6.0;
        no_cross.policy_b.alternatives[1].decision_utility = 6.0;
        for null_model in &mut no_cross.policy_b.null_models {
            null_model.null_size = 0.01;
        }
        no_cross.maximum_expected_cost = None;
        let report = analyze_two_policy_mixture(&no_cross).unwrap();
        assert_eq!(report.policy_b_probability, 1.0);
        assert_eq!(report.mixture_entropy_nats, 0.0);
    }

    #[test]
    fn canonical_order_and_report_tampering_are_detected() {
        let plan = crossing_fixture();
        let report = analyze_two_policy_mixture(&plan).unwrap();
        let mut reordered = plan.clone();
        reordered.policy_a.alternatives.reverse();
        reordered.policy_b.alternatives.reverse();
        assert_eq!(report, analyze_two_policy_mixture(&reordered).unwrap());

        let mut altered = report.clone();
        altered.worst_case_null_size += 0.01;
        assert!(verify_two_policy_mixture_report(&plan, &altered).is_err());
    }
}
