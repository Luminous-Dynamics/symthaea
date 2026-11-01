//! Success-driven deployment quota sizing under an explicit Bernoulli yield contract.
//!
//! This computes the smallest attempt budget that reaches a requested number of verified
//! deployment successes with a specified probability, assuming independent attempts with a
//! stationary success probability no lower than a pre-qualified floor.
//!
//! This is a planning calculation, not a qualification engine: it pins the external floor
//! evidence by digest but does not inspect or independently validate that evidence.

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const FLOOR_SCHEMA: &str = "symthaea.nas.deployment-success-floor";
const PLAN_SCHEMA: &str = "symthaea.nas.deployment-quota-plan";
const VERSION: &str = "0.1.0";
const FLOOR_DOMAIN: &[u8] = b"symthaea.nas.deployment-success-floor.v0.1";
const PLAN_DOMAIN: &[u8] = b"symthaea.nas.deployment-quota-plan.v0.1";
const SHA256_HEX_LEN: usize = 64;
const MAX_ATTEMPTS: u64 = 1_000_000;
const MAX_QUOTA: u64 = 100_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentSuccessFloor {
    pub schema: String,
    pub version: String,
    pub benchmark_id: String,
    pub target_cohort_id: String,
    pub candidate_policy_sha256: String,
    pub evaluation_cohort_sha256: String,
    pub success_probability_lower_bound: f64,
    pub qualification_receipt_sha256: String,
    pub floor_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentQuotaPlan {
    pub schema: String,
    pub version: String,
    pub floor_sha256: String,
    pub benchmark_id: String,
    pub target_cohort_id: String,
    pub candidate_policy_sha256: String,
    pub requested_verified_successes: u64,
    pub maximum_failure_probability: f64,
    pub success_probability_lower_bound: f64,
    pub planned_attempts: u64,
    pub achieved_failure_probability_bound: f64,
    pub previous_attempts_failure_probability_bound: f64,
    pub plan_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentQuotaError {
    pub field: String,
    pub message: String,
}

impl DeploymentQuotaError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for DeploymentQuotaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for DeploymentQuotaError {}

/// Construct a content-addressed success-floor manifest.
///
/// This validates the floor's shape and probability range. It does not prove that the
/// held-out evaluation justifies the probability floor; an independent qualification
/// process must produce and preserve the referenced receipt before this may be used.
pub fn create_deployment_success_floor(
    benchmark_id: impl Into<String>,
    target_cohort_id: impl Into<String>,
    candidate_policy_sha256: impl Into<String>,
    evaluation_cohort_sha256: impl Into<String>,
    success_probability_lower_bound: f64,
    qualification_receipt_sha256: impl Into<String>,
) -> Result<DeploymentSuccessFloor, DeploymentQuotaError> {
    let mut floor = DeploymentSuccessFloor {
        schema: FLOOR_SCHEMA.into(),
        version: VERSION.into(),
        benchmark_id: benchmark_id.into(),
        target_cohort_id: target_cohort_id.into(),
        candidate_policy_sha256: candidate_policy_sha256.into(),
        evaluation_cohort_sha256: evaluation_cohort_sha256.into(),
        success_probability_lower_bound,
        qualification_receipt_sha256: qualification_receipt_sha256.into(),
        floor_sha256: String::new(),
    };
    validate_floor_fields(&floor)?;
    floor.floor_sha256 = compute_deployment_success_floor_sha256(&floor)?;
    Ok(floor)
}

pub fn compute_deployment_success_floor_sha256(
    floor: &DeploymentSuccessFloor,
) -> Result<String, DeploymentQuotaError> {
    validate_floor_fields(floor)?;
    let mut canonical = floor.clone();
    canonical.floor_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| DeploymentQuotaError::new("floor", error.to_string()))?;
    Ok(digest_bytes(FLOOR_DOMAIN, &bytes))
}

pub fn validate_deployment_success_floor(
    floor: &DeploymentSuccessFloor,
) -> Result<(), DeploymentQuotaError> {
    validate_floor_fields(floor)?;
    if !is_sha256(&floor.floor_sha256)
        || compute_deployment_success_floor_sha256(floor)? != floor.floor_sha256
    {
        return Err(DeploymentQuotaError::new(
            "floor_sha256",
            "does not match the canonical success-floor manifest",
        ));
    }
    Ok(())
}

fn validate_floor_fields(
    floor: &DeploymentSuccessFloor,
) -> Result<(), DeploymentQuotaError> {
    if floor.schema != FLOOR_SCHEMA || floor.version != VERSION {
        return Err(DeploymentQuotaError::new(
            "schema_version",
            "unsupported deployment success-floor schema/version",
        ));
    }
    for (field, value) in [
        ("benchmark_id", floor.benchmark_id.as_str()),
        ("target_cohort_id", floor.target_cohort_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(DeploymentQuotaError::new(field, "must be non-empty"));
        }
    }
    for (field, value) in [
        ("candidate_policy_sha256", floor.candidate_policy_sha256.as_str()),
        ("evaluation_cohort_sha256", floor.evaluation_cohort_sha256.as_str()),
        ("qualification_receipt_sha256", floor.qualification_receipt_sha256.as_str()),
    ] {
        require_sha256(field, value)?;
    }
    if !floor.success_probability_lower_bound.is_finite()
        || floor.success_probability_lower_bound <= 0.0
        || floor.success_probability_lower_bound > 1.0
    {
        return Err(DeploymentQuotaError::new(
            "success_probability_lower_bound",
            "must be finite and in (0, 1]",
        ));
    }
    Ok(())
}

/// Find the smallest n for which P(X < quota) <= delta under
/// X ~ Binomial(n, p_floor). Independence/stationarity are external premises.
pub fn plan_deployment_quota(
    floor: &DeploymentSuccessFloor,
    requested_verified_successes: u64,
    maximum_failure_probability: f64,
) -> Result<DeploymentQuotaPlan, DeploymentQuotaError> {
    validate_deployment_success_floor(floor)?;
    validate_quota_inputs(requested_verified_successes, maximum_failure_probability)?;

    let p = floor.success_probability_lower_bound;
    let attempts = minimum_attempts(requested_verified_successes, p, maximum_failure_probability)?;
    let achieved_failure = binomial_failure_probability(attempts, requested_verified_successes, p);
    let previous_failure = if attempts == requested_verified_successes {
        1.0
    } else {
        binomial_failure_probability(attempts - 1, requested_verified_successes, p)
    };

    let mut plan = DeploymentQuotaPlan {
        schema: PLAN_SCHEMA.into(),
        version: VERSION.into(),
        floor_sha256: floor.floor_sha256.clone(),
        benchmark_id: floor.benchmark_id.clone(),
        target_cohort_id: floor.target_cohort_id.clone(),
        candidate_policy_sha256: floor.candidate_policy_sha256.clone(),
        requested_verified_successes,
        maximum_failure_probability,
        success_probability_lower_bound: p,
        planned_attempts: attempts,
        achieved_failure_probability_bound: achieved_failure,
        previous_attempts_failure_probability_bound: previous_failure,
        plan_sha256: "0".repeat(SHA256_HEX_LEN),
    };
    plan.plan_sha256 = compute_deployment_quota_plan_sha256(&plan)?;
    Ok(plan)
}

pub fn compute_deployment_quota_plan_sha256(
    plan: &DeploymentQuotaPlan,
) -> Result<String, DeploymentQuotaError> {
    validate_plan_fields(plan)?;
    let mut canonical = plan.clone();
    canonical.plan_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| DeploymentQuotaError::new("plan", error.to_string()))?;
    Ok(digest_bytes(PLAN_DOMAIN, &bytes))
}

/// Recompute the minimum plan against the exact floor and reject caller-edited fields.
pub fn verify_deployment_quota_plan(
    floor: &DeploymentSuccessFloor,
    plan: &DeploymentQuotaPlan,
) -> Result<(), DeploymentQuotaError> {
    validate_plan_fields(plan)?;
    if plan.plan_sha256 != compute_deployment_quota_plan_sha256(plan)? {
        return Err(DeploymentQuotaError::new(
            "plan_sha256",
            "does not match the canonical quota plan",
        ));
    }
    let expected = plan_deployment_quota(
        floor,
        plan.requested_verified_successes,
        plan.maximum_failure_probability,
    )?;
    if &expected != plan {
        return Err(DeploymentQuotaError::new(
            "plan",
            "does not match the minimum attempt budget recomputed from the supplied floor",
        ));
    }
    Ok(())
}

fn validate_plan_fields(plan: &DeploymentQuotaPlan) -> Result<(), DeploymentQuotaError> {
    if plan.schema != PLAN_SCHEMA || plan.version != VERSION {
        return Err(DeploymentQuotaError::new(
            "schema_version",
            "unsupported deployment quota plan schema/version",
        ));
    }
    require_sha256("floor_sha256", &plan.floor_sha256)?;
    require_sha256("candidate_policy_sha256", &plan.candidate_policy_sha256)?;
    require_sha256("plan_sha256", &plan.plan_sha256)?;
    for (field, value) in [
        ("benchmark_id", plan.benchmark_id.as_str()),
        ("target_cohort_id", plan.target_cohort_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(DeploymentQuotaError::new(field, "must be non-empty"));
        }
    }
    validate_quota_inputs(plan.requested_verified_successes, plan.maximum_failure_probability)?;
    if !plan.success_probability_lower_bound.is_finite()
        || plan.success_probability_lower_bound <= 0.0
        || plan.success_probability_lower_bound > 1.0
    {
        return Err(DeploymentQuotaError::new(
            "success_probability_lower_bound",
            "must be finite and in (0, 1]",
        ));
    }
    if plan.planned_attempts < plan.requested_verified_successes
        || plan.planned_attempts > MAX_ATTEMPTS
    {
        return Err(DeploymentQuotaError::new(
            "planned_attempts",
            "must be between the requested quota and the supported attempt ceiling",
        ));
    }
    for (field, value) in [
        ("achieved_failure_probability_bound", plan.achieved_failure_probability_bound),
        ("previous_attempts_failure_probability_bound", plan.previous_attempts_failure_probability_bound),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(DeploymentQuotaError::new(field, "must be finite and within [0, 1]"));
        }
    }
    Ok(())
}

fn validate_quota_inputs(
    requested_verified_successes: u64,
    maximum_failure_probability: f64,
) -> Result<(), DeploymentQuotaError> {
    if requested_verified_successes == 0 || requested_verified_successes > MAX_QUOTA {
        return Err(DeploymentQuotaError::new(
            "requested_verified_successes",
            format!("must be in 1..={MAX_QUOTA}"),
        ));
    }
    if !maximum_failure_probability.is_finite()
        || maximum_failure_probability <= 0.0
        || maximum_failure_probability >= 1.0
    {
        return Err(DeploymentQuotaError::new(
            "maximum_failure_probability",
            "must be finite and in (0, 1)",
        ));
    }
    Ok(())
}

fn minimum_attempts(
    quota: u64,
    success_probability_lower_bound: f64,
    maximum_failure_probability: f64,
) -> Result<u64, DeploymentQuotaError> {
    if success_probability_lower_bound == 1.0 {
        return Ok(quota);
    }
    let mut lower = quota - 1;
    let mut upper = quota;
    while binomial_failure_probability(upper, quota, success_probability_lower_bound)
        > maximum_failure_probability
    {
        if upper >= MAX_ATTEMPTS {
            return Err(DeploymentQuotaError::new(
                "planned_attempts",
                format!("required attempts exceed the supported ceiling {MAX_ATTEMPTS}"),
            ));
        }
        upper = upper.saturating_mul(2).min(MAX_ATTEMPTS);
    }
    while lower + 1 < upper {
        let midpoint = lower + (upper - lower) / 2;
        if binomial_failure_probability(midpoint, quota, success_probability_lower_bound)
            <= maximum_failure_probability
        {
            upper = midpoint;
        } else {
            lower = midpoint;
        }
    }
    Ok(upper)
}

/// Stable log-space evaluation of P(Binomial(n,p) < quota), using adjacent log-PMF values.
fn binomial_failure_probability(n: u64, quota: u64, p: f64) -> f64 {
    if n < quota {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    let log_p = p.ln();
    let log_not_p = (-p).ln_1p();
    let mut log_mass = (n as f64) * log_not_p;
    let mut log_sum = f64::NEG_INFINITY;
    for k in 0..quota {
        log_sum = log_add_exp(log_sum, log_mass);
        if k + 1 < quota {
            log_mass += ((n - k) as f64).ln()
                - ((k + 1) as f64).ln()
                + log_p
                - log_not_p;
        }
    }
    log_sum.exp().clamp(0.0, 1.0)
}

fn log_add_exp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

fn require_sha256(field: &str, value: &str) -> Result<(), DeploymentQuotaError> {
    if !is_sha256(value) {
        return Err(DeploymentQuotaError::new(
            field,
            "must be a 64-character lowercase hexadecimal SHA-256 digest",
        ));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == SHA256_HEX_LEN
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    let mut encoded = String::with_capacity(SHA256_HEX_LEN);
    for byte in hasher.finalize() {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn floor(probability: f64) -> DeploymentSuccessFloor {
        create_deployment_success_floor(
            "nats-bench-mcu-v2",
            "nrf5340-required-target-v1",
            sha('a'),
            sha('b'),
            probability,
            sha('c'),
        ).unwrap()
    }

    #[test]
    fn known_case_one_success_half_yield_ninety_five_percent_probability() {
        let plan = plan_deployment_quota(&floor(0.5), 1, 0.05).unwrap();
        assert_eq!(plan.planned_attempts, 5);
        assert!((plan.achieved_failure_probability_bound - 0.03125).abs() < 1e-12);
        assert!((plan.previous_attempts_failure_probability_bound - 0.0625).abs() < 1e-12);
    }

    #[test]
    fn known_case_ten_successes_half_yield_ninety_five_percent_probability() {
        let plan = plan_deployment_quota(&floor(0.5), 10, 0.05).unwrap();
        assert_eq!(plan.planned_attempts, 28);
        assert!((plan.achieved_failure_probability_bound - 0.04357927665114407).abs() < 1e-12);
        assert!((plan.previous_attempts_failure_probability_bound - 0.061039060354232795).abs() < 1e-12);
    }

    #[test]
    fn known_case_ten_successes_seventy_percent_yield_ninety_five_percent_probability() {
        let plan = plan_deployment_quota(&floor(0.7), 10, 0.05).unwrap();
        assert_eq!(plan.planned_attempts, 19);
        assert!((plan.achieved_failure_probability_bound - 0.03255335688130099).abs() < 1e-12);
    }

    #[test]
    fn perfect_floor_requires_exactly_the_requested_quota() {
        let plan = plan_deployment_quota(&floor(1.0), 25, 0.01).unwrap();
        assert_eq!(plan.planned_attempts, 25);
        assert_eq!(plan.achieved_failure_probability_bound, 0.0);
        assert_eq!(plan.previous_attempts_failure_probability_bound, 1.0);
    }

    #[test]
    fn more_conservative_success_floor_never_reduces_attempt_budget() {
        let high = plan_deployment_quota(&floor(0.8), 25, 0.01).unwrap();
        let low = plan_deployment_quota(&floor(0.6), 25, 0.01).unwrap();
        assert!(low.planned_attempts >= high.planned_attempts);
    }

    #[test]
    fn stricter_quota_confidence_never_reduces_attempt_budget() {
        let relaxed = plan_deployment_quota(&floor(0.8), 25, 0.05).unwrap();
        let strict = plan_deployment_quota(&floor(0.8), 25, 0.001).unwrap();
        assert!(strict.planned_attempts >= relaxed.planned_attempts);
    }

    #[test]
    fn plan_is_bound_to_floor_and_recomputed_minimality() {
        let source_floor = floor(0.5);
        let plan = plan_deployment_quota(&source_floor, 10, 0.05).unwrap();
        verify_deployment_quota_plan(&source_floor, &plan).unwrap();

        let other_floor = floor(0.6);
        assert!(verify_deployment_quota_plan(&other_floor, &plan).is_err());

        let mut forged = plan.clone();
        forged.planned_attempts += 1;
        forged.plan_sha256 = compute_deployment_quota_plan_sha256(&forged).unwrap();
        assert!(verify_deployment_quota_plan(&source_floor, &forged).is_err());
    }

    #[test]
    fn rejects_floor_and_plan_tampering() {
        let source_floor = floor(0.5);
        let mut altered_floor = source_floor.clone();
        altered_floor.success_probability_lower_bound = 0.9;
        assert!(validate_deployment_success_floor(&altered_floor).is_err());

        let plan = plan_deployment_quota(&source_floor, 10, 0.05).unwrap();
        let mut altered_plan = plan.clone();
        altered_plan.maximum_failure_probability = 0.5;
        assert!(verify_deployment_quota_plan(&source_floor, &altered_plan).is_err());
    }

    #[test]
    fn rejects_invalid_quota_probability_and_digest_inputs() {
        assert!(plan_deployment_quota(&floor(0.5), 0, 0.05).is_err());
        assert!(plan_deployment_quota(&floor(0.5), 10, 1.0).is_err());
        assert!(plan_deployment_quota(&floor(0.5), 10, 0.0).is_err());
        assert!(floor(0.0).is_err());
    }

    #[test]
    fn serialized_floor_rejects_unknown_fields() {
        let encoded = serde_json::to_string(&floor(0.5)).unwrap();
        let injected = encoded.replacen(
            "\"floor_sha256\"",
            "\"unknown\":true,\"floor_sha256\"",
            1,
        );
        assert!(serde_json::from_str::<DeploymentSuccessFloor>(&injected).is_err());
    }

    #[test]
    fn serialized_plan_rejects_unknown_fields() {
        let plan = plan_deployment_quota(&floor(0.5), 10, 0.05).unwrap();
        let encoded = serde_json::to_string(&plan).unwrap();
        let injected = encoded.replacen(
            "\"plan_sha256\"",
            "\"unknown\":true,\"plan_sha256\"",
            1,
        );
        assert!(serde_json::from_str::<DeploymentQuotaPlan>(&injected).is_err());
    }
}
