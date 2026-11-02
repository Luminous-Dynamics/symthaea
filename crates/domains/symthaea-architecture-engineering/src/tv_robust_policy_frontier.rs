//! Total-variation robustness for finite policy-frontier optimization.
//!
//! The TV radii apply to the complete finite-horizon observable path law, not
//! independently to each time step. With TV(P,Q) = sup_A |P(A)-Q(A)|, an
//! expectation of a function with range [L,U] changes by at most
//! TV(P,Q) * (U-L).
//!
//! Statistical coverage failure for the simultaneous null-family envelope is
//! explicitly spent from alpha. A digest binds evidence identity; it does not
//! authenticate or prove that evidence.

use super::policy_frontier_lp::{
    analyze_policy_frontier, FrontierAlternativeReceipt, FrontierNullReceipt,
    PolicyFrontierError, PolicyFrontierPlan, PolicyFrontierReport,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, error::Error, fmt};

const ENVELOPE_SCHEMA: &str = "symthaea.architecture-engineering.tv-ambiguity-envelope";
const ENVELOPE_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str = "symthaea.architecture-engineering.tv-robust-frontier-report";
const REPORT_VERSION: &str = "0.1.0";
const TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TvRobustnessError {
    pub path: String,
    pub message: String,
}

impl TvRobustnessError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }

    fn from_frontier(error: PolicyFrontierError) -> Self {
        Self { path: error.path, message: error.message }
    }
}

impl fmt::Display for TvRobustnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for TvRobustnessError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TvNullPathEnvelope {
    pub model_id: String,
    /// TV radius for the complete finite-horizon observable path distribution.
    pub total_variation_radius: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TvAlternativePathEnvelope {
    pub model_id: String,
    /// TV radius for the complete finite-horizon observable path distribution.
    pub total_variation_radius: f64,
    /// Known support bound for the alternative's decision-utility random variable.
    pub utility_lower_bound: f64,
    pub utility_upper_bound: f64,
    /// Known support bound for evaluator cost, which must lie in [0, bound].
    pub evaluator_cost_upper_bound: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TvAmbiguityEnvelope {
    pub schema: String,
    pub version: String,
    pub envelope_id: String,
    /// SHA-256 of the canonical nominal PolicyFrontierPlan being robustified.
    pub source_plan_sha256: String,
    /// Simultaneous coverage-failure probability for every declared null radius.
    /// This is spent from the experiment's nominal type-I budget.
    pub null_family_coverage_failure_probability: f64,
    /// Simultaneous coverage-failure probability for all alternative radii and
    /// support bounds. It qualifies alternative performance bounds, not type-I.
    pub alternative_family_coverage_failure_probability: f64,
    /// The evidence artifact must cover the entire simultaneous family.
    pub null_coverage_receipt_sha256: String,
    pub alternative_coverage_receipt_sha256: String,
    /// Radius calibration must be independent of the confirmatory stream.
    pub calibration_independent_of_confirmatory_stream: bool,
    pub null_models: Vec<TvNullPathEnvelope>,
    pub alternatives: Vec<TvAlternativePathEnvelope>,
}

impl TvAmbiguityEnvelope {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        envelope_id: impl Into<String>,
        source_plan_sha256: impl Into<String>,
        null_family_coverage_failure_probability: f64,
        alternative_family_coverage_failure_probability: f64,
        null_coverage_receipt_sha256: impl Into<String>,
        alternative_coverage_receipt_sha256: impl Into<String>,
        calibration_independent_of_confirmatory_stream: bool,
        null_models: Vec<TvNullPathEnvelope>,
        alternatives: Vec<TvAlternativePathEnvelope>,
    ) -> Result<Self, TvRobustnessError> {
        let mut envelope = Self {
            schema: ENVELOPE_SCHEMA.into(),
            version: ENVELOPE_VERSION.into(),
            envelope_id: envelope_id.into(),
            source_plan_sha256: source_plan_sha256.into(),
            null_family_coverage_failure_probability,
            alternative_family_coverage_failure_probability,
            null_coverage_receipt_sha256: null_coverage_receipt_sha256.into(),
            alternative_coverage_receipt_sha256: alternative_coverage_receipt_sha256.into(),
            calibration_independent_of_confirmatory_stream,
            null_models,
            alternatives,
        };
        envelope.canonicalize()?;
        Ok(envelope)
    }

    pub fn canonicalize(&mut self) -> Result<(), TvRobustnessError> {
        self.null_models.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        self.alternatives.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        self.validate_structure()
    }

    fn validate_structure(&self) -> Result<(), TvRobustnessError> {
        if self.schema != ENVELOPE_SCHEMA || self.version != ENVELOPE_VERSION {
            return Err(TvRobustnessError::new(
                "envelope.schema",
                "unsupported TV ambiguity-envelope schema/version",
            ));
        }
        if self.envelope_id.trim().is_empty() {
            return Err(TvRobustnessError::new("envelope.envelope_id", "envelope ID must be non-empty"));
        }
        validate_hash(&self.source_plan_sha256, "envelope.source_plan_sha256")?;
        validate_hash(&self.null_coverage_receipt_sha256, "envelope.null_coverage_receipt_sha256")?;
        validate_hash(
            &self.alternative_coverage_receipt_sha256,
            "envelope.alternative_coverage_receipt_sha256",
        )?;
        validate_probability(
            self.null_family_coverage_failure_probability,
            "envelope.null_family_coverage_failure_probability",
        )?;
        validate_probability(
            self.alternative_family_coverage_failure_probability,
            "envelope.alternative_family_coverage_failure_probability",
        )?;
        if self.alternative_family_coverage_failure_probability >= 1.0 {
            return Err(TvRobustnessError::new(
                "envelope.alternative_family_coverage_failure_probability",
                "alternative-family coverage failure must be strictly smaller than one",
            ));
        }
        if !self.calibration_independent_of_confirmatory_stream {
            return Err(TvRobustnessError::new(
                "envelope.calibration_independent_of_confirmatory_stream",
                "coverage calibration must be independent of the confirmatory stream",
            ));
        }
        if self.null_models.is_empty() || self.alternatives.len() < 2 {
            return Err(TvRobustnessError::new(
                "envelope.models",
                "null family must be non-empty and alternative family must contain at least two models",
            ));
        }
        validate_unique_model_ids(
            self.null_models.iter().map(|row| row.model_id.as_str()),
            "envelope.null_models",
        )?;
        validate_unique_model_ids(
            self.alternatives.iter().map(|row| row.model_id.as_str()),
            "envelope.alternatives",
        )?;
        for (index, row) in self.null_models.iter().enumerate() {
            validate_probability(
                row.total_variation_radius,
                &format!("envelope.null_models[{index}].total_variation_radius"),
            )?;
        }
        for (index, row) in self.alternatives.iter().enumerate() {
            let path = format!("envelope.alternatives[{index}]");
            validate_probability(row.total_variation_radius, &format!("{path}.total_variation_radius"))?;
            if !row.utility_lower_bound.is_finite()
                || !row.utility_upper_bound.is_finite()
                || row.utility_lower_bound > row.utility_upper_bound
            {
                return Err(TvRobustnessError::new(
                    format!("{path}.utility_bounds"),
                    "utility support bounds must be finite and ordered",
                ));
            }
            if !row.evaluator_cost_upper_bound.is_finite() || row.evaluator_cost_upper_bound < 0.0 {
                return Err(TvRobustnessError::new(
                    format!("{path}.evaluator_cost_upper_bound"),
                    "evaluator-cost upper bound must be finite and nonnegative",
                ));
            }
        }
        Ok(())
    }

    pub fn envelope_sha256(&self) -> Result<String, TvRobustnessError> {
        let mut canonical = self.clone();
        canonical.canonicalize()?;
        digest(&canonical)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TvRobustPolicyFrontierReport {
    pub schema: String,
    pub version: String,
    pub envelope_id: String,
    pub source_plan_sha256: String,
    pub envelope_sha256: String,
    pub robustified_plan_sha256: String,
    pub nominal_significance_level: f64,
    pub conditional_significance_level: f64,
    pub null_family_coverage_failure_probability: f64,
    pub alternative_family_coverage_failure_probability: f64,
    pub worst_case_unconditional_type_i_upper_bound: f64,
    pub frontier: PolicyFrontierReport,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    envelope_id: &'a str,
    source_plan_sha256: &'a str,
    envelope_sha256: &'a str,
    robustified_plan_sha256: &'a str,
    nominal_significance_level: f64,
    conditional_significance_level: f64,
    null_family_coverage_failure_probability: f64,
    alternative_family_coverage_failure_probability: f64,
    worst_case_unconditional_type_i_upper_bound: f64,
    frontier: &'a PolicyFrontierReport,
}

fn validate_probability(value: f64, path: &str) -> Result<(), TvRobustnessError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(TvRobustnessError::new(path, "value must be finite and lie in [0, 1]"));
    }
    Ok(())
}

fn validate_hash(value: &str, path: &str) -> Result<(), TvRobustnessError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(TvRobustnessError::new(path, "expected a 64-character hexadecimal SHA-256 digest"));
    }
    Ok(())
}

fn validate_unique_model_ids<'a>(
    ids: impl Iterator<Item = &'a str>,
    path: &str,
) -> Result<(), TvRobustnessError> {
    let mut seen = BTreeSet::new();
    for (index, id) in ids.enumerate() {
        if id.trim().is_empty() || !seen.insert(id) {
            return Err(TvRobustnessError::new(
                format!("{path}[{index}].model_id"),
                "model IDs must be non-empty and unique",
            ));
        }
    }
    Ok(())
}

fn conditional_alpha(alpha: f64, coverage_failure: f64) -> Result<f64, TvRobustnessError> {
    if coverage_failure >= alpha {
        return Err(TvRobustnessError::new(
            "envelope.null_family_coverage_failure_probability",
            "null-radius coverage failure must be strictly smaller than nominal alpha",
        ));
    }
    let result = (alpha - coverage_failure) / (1.0 - coverage_failure);
    if !result.is_finite() || !(0.0..1.0).contains(&result) {
        return Err(TvRobustnessError::new(
            "envelope.conditional_significance_level",
            "conditional alpha calculation is invalid",
        ));
    }
    Ok(result)
}

fn robustify_plan(
    nominal: &PolicyFrontierPlan,
    envelope: &TvAmbiguityEnvelope,
) -> Result<PolicyFrontierPlan, TvRobustnessError> {
    let mut plan = nominal.clone();
    plan.canonicalize().map_err(TvRobustnessError::from_frontier)?;
    let source_digest = plan.plan_sha256().map_err(TvRobustnessError::from_frontier)?;
    if source_digest != envelope.source_plan_sha256 {
        return Err(TvRobustnessError::new(
            "envelope.source_plan_sha256",
            "ambiguity envelope is not bound to this canonical nominal plan",
        ));
    }
    let null_ids = plan.policies[0].null_models.iter()
        .map(|row| row.model_id.as_str()).collect::<BTreeSet<_>>();
    let envelope_null_ids = envelope.null_models.iter()
        .map(|row| row.model_id.as_str()).collect::<BTreeSet<_>>();
    if null_ids != envelope_null_ids {
        return Err(TvRobustnessError::new(
            "envelope.null_models",
            "TV envelope must cover exactly every declared null model",
        ));
    }
    let alternative_ids = plan.policies[0].alternatives.iter()
        .map(|row| row.model_id.as_str()).collect::<BTreeSet<_>>();
    let envelope_alternative_ids = envelope.alternatives.iter()
        .map(|row| row.model_id.as_str()).collect::<BTreeSet<_>>();
    if alternative_ids != envelope_alternative_ids {
        return Err(TvRobustnessError::new(
            "envelope.alternatives",
            "TV envelope must cover exactly every declared alternative model",
        ));
    }

    let original_alpha = plan.significance_level;
    plan.significance_level = conditional_alpha(
        original_alpha,
        envelope.null_family_coverage_failure_probability,
    )?;

    for policy in &mut plan.policies {
        for null_row in &mut policy.null_models {
            let radius = envelope.null_models.iter()
                .find(|item| item.model_id == null_row.model_id)
                .map(|item| item.total_variation_radius)
                .ok_or_else(|| TvRobustnessError::new(
                    "envelope.null_models",
                    "validated null envelope unexpectedly missing",
                ))?;
            null_row.null_size = (null_row.null_size + radius).min(1.0);
        }
        for alternative_row in &mut policy.alternatives {
            let bound = envelope.alternatives.iter()
                .find(|item| item.model_id == alternative_row.model_id)
                .ok_or_else(|| TvRobustnessError::new(
                    "envelope.alternatives",
                    "validated alternative envelope unexpectedly missing",
                ))?;
            if alternative_row.decision_utility < bound.utility_lower_bound - TOLERANCE
                || alternative_row.decision_utility > bound.utility_upper_bound + TOLERANCE
            {
                return Err(TvRobustnessError::new(
                    format!("policies.{}.alternatives.{}.decision_utility", policy.policy_id, alternative_row.model_id),
                    "nominal utility lies outside the declared support bounds",
                ));
            }
            if alternative_row.expected_evaluator_cost > bound.evaluator_cost_upper_bound + TOLERANCE {
                return Err(TvRobustnessError::new(
                    format!("policies.{}.alternatives.{}.expected_evaluator_cost", policy.policy_id, alternative_row.model_id),
                    "nominal evaluator cost exceeds the declared support bound",
                ));
            }
            let radius = bound.total_variation_radius;
            alternative_row.rejection_power = (alternative_row.rejection_power - radius).max(0.0);
            alternative_row.expected_evaluator_cost = (
                alternative_row.expected_evaluator_cost
                    + radius * bound.evaluator_cost_upper_bound
            ).min(bound.evaluator_cost_upper_bound);
            alternative_row.decision_utility = (
                alternative_row.decision_utility
                    - radius * (bound.utility_upper_bound - bound.utility_lower_bound)
            ).max(bound.utility_lower_bound);
        }
    }
    plan.canonicalize().map_err(TvRobustnessError::from_frontier)?;
    Ok(plan)
}

/// Optimize the supplied finite policy frontier after worst-case TV perturbation.
///
/// The null coverage failure probability is converted into conditional alpha
/// using delta + (1-delta)*alpha_cond = alpha. Alternative-family coverage
/// failure is reported separately because it qualifies power/utility/cost bounds,
/// not the null type-I guarantee.
pub fn analyze_tv_robust_policy_frontier(
    nominal: &PolicyFrontierPlan,
    input_envelope: &TvAmbiguityEnvelope,
) -> Result<TvRobustPolicyFrontierReport, TvRobustnessError> {
    let mut envelope = input_envelope.clone();
    envelope.canonicalize()?;
    let mut canonical_nominal = nominal.clone();
    canonical_nominal.canonicalize().map_err(TvRobustnessError::from_frontier)?;
    let source_plan_sha256 = canonical_nominal.plan_sha256()
        .map_err(TvRobustnessError::from_frontier)?;
    let robustified = robustify_plan(&canonical_nominal, &envelope)?;
    let frontier = analyze_policy_frontier(&robustified)
        .map_err(TvRobustnessError::from_frontier)?;
    let delta = envelope.null_family_coverage_failure_probability;
    let unconditional_type_i = delta + (1.0 - delta) * frontier.worst_case_null_size;
    if !unconditional_type_i.is_finite()
        || unconditional_type_i > canonical_nominal.significance_level + TOLERANCE
    {
        return Err(TvRobustnessError::new(
            "report.worst_case_unconditional_type_i_upper_bound",
            "reconstructed unconditional type-I upper bound exceeds nominal alpha",
        ));
    }
    let envelope_sha256 = envelope.envelope_sha256()?;
    let robustified_plan_sha256 = robustified.plan_sha256()
        .map_err(TvRobustnessError::from_frontier)?;
    let payload = ReportPayload {
        schema: REPORT_SCHEMA,
        version: REPORT_VERSION,
        envelope_id: &envelope.envelope_id,
        source_plan_sha256: &source_plan_sha256,
        envelope_sha256: &envelope_sha256,
        robustified_plan_sha256: &robustified_plan_sha256,
        nominal_significance_level: canonical_nominal.significance_level,
        conditional_significance_level: robustified.significance_level,
        null_family_coverage_failure_probability: delta,
        alternative_family_coverage_failure_probability: envelope.alternative_family_coverage_failure_probability,
        worst_case_unconditional_type_i_upper_bound: unconditional_type_i,
        frontier: &frontier,
    };
    let report_sha256 = digest(&payload)?;
    Ok(TvRobustPolicyFrontierReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        envelope_id: envelope.envelope_id,
        source_plan_sha256,
        envelope_sha256,
        robustified_plan_sha256,
        nominal_significance_level: canonical_nominal.significance_level,
        conditional_significance_level: robustified.significance_level,
        null_family_coverage_failure_probability: delta,
        alternative_family_coverage_failure_probability: envelope.alternative_family_coverage_failure_probability,
        worst_case_unconditional_type_i_upper_bound: unconditional_type_i,
        frontier,
        report_sha256,
    })
}

pub fn verify_tv_robust_policy_frontier_report(
    nominal: &PolicyFrontierPlan,
    envelope: &TvAmbiguityEnvelope,
    report: &TvRobustPolicyFrontierReport,
) -> Result<(), TvRobustnessError> {
    if analyze_tv_robust_policy_frontier(nominal, envelope)? != *report {
        return Err(TvRobustnessError::new(
            "report",
            "stored TV-robust frontier report differs from complete deterministic reconstruction",
        ));
    }
    Ok(())
}

fn digest<T: Serialize>(value: &T) -> Result<String, TvRobustnessError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| TvRobustnessError::new("serialization", error.to_string()))?;
    Ok(Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const UTILITY_HASH: &str =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    const RECEIPT_A: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const RECEIPT_B: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn policy(id: &str, null_a: f64, null_b: f64, alt_a: (f64, f64, f64), alt_b: (f64, f64, f64))
        -> super::super::policy_frontier_lp::FrontierPolicyReceipt
    {
        super::super::policy_frontier_lp::FrontierPolicyReceipt {
            policy_id: id.into(),
            utility_contract_sha256: UTILITY_HASH.into(),
            null_models: vec![
                FrontierNullReceipt { model_id: "null-a".into(), null_size: null_a },
                FrontierNullReceipt { model_id: "null-b".into(), null_size: null_b },
            ],
            alternatives: vec![
                FrontierAlternativeReceipt {
                    model_id: "alt-a".into(),
                    rejection_power: alt_a.0,
                    expected_evaluator_cost: alt_a.1,
                    decision_utility: alt_a.2,
                },
                FrontierAlternativeReceipt {
                    model_id: "alt-b".into(),
                    rejection_power: alt_b.0,
                    expected_evaluator_cost: alt_b.1,
                    decision_utility: alt_b.2,
                },
            ],
        }
    }

    fn nominal_plan(alpha: f64) -> PolicyFrontierPlan {
        PolicyFrontierPlan::new(
            "tv-robust-frontier-fixture",
            alpha,
            Some(9.0),
            vec![
                policy("policy-a", 0.01, 0.015, (0.85, 4.0, 8.0), (0.70, 6.0, 6.0)),
                policy("policy-b", 0.012, 0.020, (0.60, 2.0, 6.0), (0.92, 8.0, 9.0)),
            ],
        ).unwrap()
    }

    fn envelope(plan: &PolicyFrontierPlan) -> TvAmbiguityEnvelope {
        TvAmbiguityEnvelope::new(
            "fixture-tv-envelope",
            plan.plan_sha256().unwrap(),
            0.0025,
            0.01,
            RECEIPT_A,
            RECEIPT_B,
            true,
            vec![
                TvNullPathEnvelope { model_id: "null-a".into(), total_variation_radius: 0.01 },
                TvNullPathEnvelope { model_id: "null-b".into(), total_variation_radius: 0.015 },
            ],
            vec![
                TvAlternativePathEnvelope {
                    model_id: "alt-a".into(),
                    total_variation_radius: 0.05,
                    utility_lower_bound: 0.0,
                    utility_upper_bound: 10.0,
                    evaluator_cost_upper_bound: 10.0,
                },
                TvAlternativePathEnvelope {
                    model_id: "alt-b".into(),
                    total_variation_radius: 0.05,
                    utility_lower_bound: 0.0,
                    utility_upper_bound: 10.0,
                    evaluator_cost_upper_bound: 10.0,
                },
            ],
        ).unwrap()
    }

    #[test]
    fn exact_tv_bounds_are_applied_before_frontier_optimization() {
        let plan = nominal_plan(0.05);
        let uncertainty = envelope(&plan);
        let report = analyze_tv_robust_policy_frontier(&plan, &uncertainty).unwrap();
        assert!((report.conditional_significance_level - (0.05 - 0.0025) / 0.9975).abs() < 1e-12);
        let null_a = report.frontier.null_model_sizes.iter()
            .find(|row| row.model_id == "null-a").unwrap().null_size;
        let null_b = report.frontier.null_model_sizes.iter()
            .find(|row| row.model_id == "null-b").unwrap().null_size;
        assert!((0.020 - TOLERANCE..=0.022 + TOLERANCE).contains(&null_a));
        assert!((0.030 - TOLERANCE..=0.035 + TOLERANCE).contains(&null_b));
        assert!(report.frontier.alternatives.iter().all(|row| row.rejection_power >= 0.0));
        assert!(report.worst_case_unconditional_type_i_upper_bound <= 0.05 + TOLERANCE);
        verify_tv_robust_policy_frontier_report(&plan, &uncertainty, &report).unwrap();
    }

    #[test]
    fn null_coverage_failure_is_paid_from_the_type_i_budget() {
        let plan = nominal_plan(0.05);
        let mut uncertainty = envelope(&plan);
        uncertainty.null_family_coverage_failure_probability = 0.01;
        let report = analyze_tv_robust_policy_frontier(&plan, &uncertainty).unwrap();
        let expected = (0.05 - 0.01) / 0.99;
        assert!((report.conditional_significance_level - expected).abs() < 1e-12);
        let reconstructed = report.null_family_coverage_failure_probability
            + (1.0 - report.null_family_coverage_failure_probability)
                * report.frontier.worst_case_null_size;
        assert!((report.worst_case_unconditional_type_i_upper_bound - reconstructed).abs() < 1e-12);
        assert!(reconstructed <= 0.05 + TOLERANCE);
    }

    #[test]
    fn zero_radius_with_zero_coverage_failure_is_an_identity_control() {
        let plan = nominal_plan(0.05);
        let mut uncertainty = envelope(&plan);
        uncertainty.null_family_coverage_failure_probability = 0.0;
        uncertainty.alternative_family_coverage_failure_probability = 0.0;
        for row in &mut uncertainty.null_models {
            row.total_variation_radius = 0.0;
        }
        for row in &mut uncertainty.alternatives {
            row.total_variation_radius = 0.0;
        }
        let robust = analyze_tv_robust_policy_frontier(&plan, &uncertainty).unwrap();
        let nominal = analyze_policy_frontier(&plan).unwrap();
        assert_eq!(robust.frontier, nominal);
        assert_eq!(robust.conditional_significance_level, plan.significance_level);
        assert_eq!(robust.worst_case_unconditional_type_i_upper_bound, nominal.worst_case_null_size);
    }

    #[test]
    fn rejects_a_stale_nominal_plan_digest() {
        let plan = nominal_plan(0.05);
        let mut uncertainty = envelope(&plan);
        let changed = nominal_plan(0.04);
        assert!(analyze_tv_robust_policy_frontier(&changed, &uncertainty).is_err());
        uncertainty.source_plan_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into();
        assert!(analyze_tv_robust_policy_frontier(&plan, &uncertainty).is_err());
    }

    #[test]
    fn rejects_incomplete_model_families_and_dependent_calibration() {
        let plan = nominal_plan(0.05);
        let mut missing = envelope(&plan);
        missing.null_models.pop();
        assert!(analyze_tv_robust_policy_frontier(&plan, &missing).is_err());
        let mut dependent = envelope(&plan);
        dependent.calibration_independent_of_confirmatory_stream = false;
        assert!(analyze_tv_robust_policy_frontier(&plan, &dependent).is_err());
    }

    #[test]
    fn rejects_invalid_coverage_budgets_and_support_bounds() {
        let plan = nominal_plan(0.05);
        let mut exhausted = envelope(&plan);
        exhausted.null_family_coverage_failure_probability = 0.05;
        assert!(analyze_tv_robust_policy_frontier(&plan, &exhausted).is_err());
        let mut invalid = envelope(&plan);
        invalid.alternatives[0].evaluator_cost_upper_bound = 1.0;
        assert!(analyze_tv_robust_policy_frontier(&plan, &invalid).is_err());
        let mut reversed = envelope(&plan);
        reversed.alternatives[0].utility_lower_bound = 11.0;
        reversed.alternatives[0].utility_upper_bound = 10.0;
        assert!(analyze_tv_robust_policy_frontier(&plan, &reversed).is_err());
    }

    #[test]
    fn infeasible_tv_robustification_fails_closed() {
        let mut plan = nominal_plan(0.05);
        for policy in &mut plan.policies {
            for null in &mut policy.null_models {
                null.null_size = 0.049;
            }
        }
        let source_hash = plan.plan_sha256().unwrap();
        let mut uncertainty = envelope(&nominal_plan(0.05));
        uncertainty.source_plan_sha256 = source_hash;
        for null in &mut uncertainty.null_models {
            null.total_variation_radius = 0.01;
        }
        assert!(analyze_tv_robust_policy_frontier(&plan, &uncertainty).is_err());
    }

    #[test]
    fn report_tampering_is_detected_by_reconstruction() {
        let plan = nominal_plan(0.05);
        let uncertainty = envelope(&plan);
        let mut report = analyze_tv_robust_policy_frontier(&plan, &uncertainty).unwrap();
        report.worst_case_unconditional_type_i_upper_bound += 0.001;
        assert!(verify_tv_robust_policy_frontier_report(&plan, &uncertainty, &report).is_err());
    }
}
