//! Noncompensatory verification of simultaneous global/local guardrails.
//!
//! V1 consumes precomputed simultaneous one-sided confidence bounds and checks
//! whether a proposed randomized policy mixture clears every required guardrail.
//! It does not estimate the bounds or certify the completeness/authenticity of
//! the guardrail catalog. Utility is intentionally absent from this gate.

use super::decision_functional_confidence::CalibrationTargetRelation;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

const REQUEST_SCHEMA: &str = "symthaea.architecture-engineering.noncompensatory-guardrail-request";
const REQUEST_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str = "symthaea.architecture-engineering.noncompensatory-guardrail-report";
const REPORT_VERSION: &str = "0.1.0";
const GLOBAL_STRATUM: &str = "__GLOBAL__";
const TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardrailError {
    pub path: String,
    pub message: String,
}

impl GuardrailError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for GuardrailError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for GuardrailError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GuardrailScope {
    Global,
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GuardrailDirection {
    UpperBound,
    LowerBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GuardrailEvidenceAvailability {
    SimultaneousBoundsAvailable,
    NotEstimable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GuardrailCellDecision {
    Pass,
    ViolationEstablished,
    NotEstablished,
    MissingEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GuardrailOverallDecision {
    Pass,
    Blocked,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardrailPolicyWeight {
    pub policy_id: String,
    pub probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardrailRequirement {
    /// Unique row identity within the frozen applicable guardrail roster.
    pub requirement_id: String,
    /// Stable metric/guardrail identity; may occur in multiple models/strata.
    pub guardrail_id: String,
    pub metric_contract_sha256: String,
    pub scope: GuardrailScope,
    /// GLOBAL requires None; LOCAL requires a non-empty stratum ID.
    pub stratum_id: Option<String>,
    pub model_id: String,
    pub direction: GuardrailDirection,
    pub threshold: f64,
    pub minimum_sample_size: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardrailPolicyBound {
    pub policy_id: String,
    pub requirement_id: String,
    pub availability: GuardrailEvidenceAvailability,
    /// Both bounds are present only when simultaneous confidence evidence is available.
    pub lower_confidence_bound: Option<f64>,
    pub upper_confidence_bound: Option<f64>,
    pub sample_size: usize,
    pub source_receipt_sha256: String,
    /// Required only when availability is NOT_ESTIMABLE.
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoncompensatoryGuardrailRequest {
    pub schema: String,
    pub version: String,
    pub evaluation_id: String,
    pub policy_weights: Vec<GuardrailPolicyWeight>,
    pub requirements: Vec<GuardrailRequirement>,
    pub expected_policy_count: usize,
    pub expected_requirement_count: usize,
    pub expected_global_requirement_count: usize,
    /// Digest of the authoritative/applicable guardrail roster artifact.
    /// The digest binds bytes but does not prove catalog authority/completeness.
    pub guardrail_roster_sha256: String,
    pub calibration_target_design_sha256: String,
    pub confirmatory_target_design_sha256: String,
    pub target_relation: CalibrationTargetRelation,
    pub calibration_independent_of_confirmatory_stream: bool,
    /// V1 only handles expected guardrail functionals linear under the declared policy mixture.
    pub mixture_estimand_linear: bool,
    /// One simultaneous confidence failure probability for every policy × requirement bound.
    /// This is a guardrail false-safe budget, not the experiment's Type-I alpha.
    pub simultaneous_guardrail_failure_probability: f64,
    pub policy_bounds: Vec<GuardrailPolicyBound>,
}

impl NoncompensatoryGuardrailRequest {
    pub fn new(
        evaluation_id: impl Into<String>,
        policy_weights: Vec<GuardrailPolicyWeight>,
        requirements: Vec<GuardrailRequirement>,
        expected_policy_count: usize,
        expected_requirement_count: usize,
        expected_global_requirement_count: usize,
        guardrail_roster_sha256: impl Into<String>,
        calibration_target_design_sha256: impl Into<String>,
        confirmatory_target_design_sha256: impl Into<String>,
        target_relation: CalibrationTargetRelation,
        calibration_independent_of_confirmatory_stream: bool,
        mixture_estimand_linear: bool,
        simultaneous_guardrail_failure_probability: f64,
        policy_bounds: Vec<GuardrailPolicyBound>,
    ) -> Result<Self, GuardrailError> {
        let mut request = Self {
            schema: REQUEST_SCHEMA.into(),
            version: REQUEST_VERSION.into(),
            evaluation_id: evaluation_id.into(),
            policy_weights,
            requirements,
            expected_policy_count,
            expected_requirement_count,
            expected_global_requirement_count,
            guardrail_roster_sha256: guardrail_roster_sha256.into(),
            calibration_target_design_sha256: calibration_target_design_sha256.into(),
            confirmatory_target_design_sha256: confirmatory_target_design_sha256.into(),
            target_relation,
            calibration_independent_of_confirmatory_stream,
            mixture_estimand_linear,
            simultaneous_guardrail_failure_probability,
            policy_bounds,
        };
        request.canonicalize()?;
        Ok(request)
    }

    pub fn canonicalize(&mut self) -> Result<(), GuardrailError> {
        self.policy_weights
            .sort_by(|left, right| left.policy_id.cmp(&right.policy_id));
        self.requirements
            .sort_by(|left, right| left.requirement_id.cmp(&right.requirement_id));
        self.policy_bounds.sort_by(|left, right| {
            (&left.policy_id, &left.requirement_id)
                .cmp(&(&right.policy_id, &right.requirement_id))
        });
        self.validate_structure()
    }

    fn validate_structure(&self) -> Result<(), GuardrailError> {
        if self.schema != REQUEST_SCHEMA || self.version != REQUEST_VERSION {
            return Err(GuardrailError::new(
                "request.schema",
                "unsupported noncompensatory guardrail request schema/version",
            ));
        }
        if self.evaluation_id.trim().is_empty() {
            return Err(GuardrailError::new(
                "request.evaluation_id",
                "evaluation ID must be non-empty",
            ));
        }
        validate_hash(&self.guardrail_roster_sha256, "request.guardrail_roster_sha256")?;
        validate_hash(
            &self.calibration_target_design_sha256,
            "request.calibration_target_design_sha256",
        )?;
        validate_hash(
            &self.confirmatory_target_design_sha256,
            "request.confirmatory_target_design_sha256",
        )?;
        if self.target_relation != CalibrationTargetRelation::SameTargetLawByDesign {
            return Err(GuardrailError::new(
                "request.target_relation",
                "V1 accepts only SAME_TARGET_LAW_BY_DESIGN; transport and unknown relations are not supported",
            ));
        }
        if self.calibration_target_design_sha256 != self.confirmatory_target_design_sha256 {
            return Err(GuardrailError::new(
                "request.confirmatory_target_design_sha256",
                "calibration and confirmatory guardrail target-design identities differ",
            ));
        }
        if !self.calibration_independent_of_confirmatory_stream {
            return Err(GuardrailError::new(
                "request.calibration_independent_of_confirmatory_stream",
                "guardrail confidence calibration must be independent of confirmation",
            ));
        }
        if !self.mixture_estimand_linear {
            return Err(GuardrailError::new(
                "request.mixture_estimand_linear",
                "V1 only composes expected guardrail functionals that are linear under the declared policy mixture",
            ));
        }
        if !self.simultaneous_guardrail_failure_probability.is_finite()
            || !(0.0..1.0).contains(&self.simultaneous_guardrail_failure_probability)
        {
            return Err(GuardrailError::new(
                "request.simultaneous_guardrail_failure_probability",
                "guardrail confidence failure probability must lie strictly between zero and one",
            ));
        }
        if self.expected_policy_count == 0
            || self.policy_weights.len() != self.expected_policy_count
        {
            return Err(GuardrailError::new(
                "request.policy_weights",
                "policy roster count does not match the frozen expected policy count",
            ));
        }
        if self.expected_requirement_count == 0
            || self.requirements.len() != self.expected_requirement_count
        {
            return Err(GuardrailError::new(
                "request.requirements",
                "requirement count does not match the frozen applicable guardrail roster",
            ));
        }

        let mut policy_ids = BTreeSet::new();
        let mut probability_sum = 0.0;
        let mut positive_weight_seen = false;
        for (index, weight) in self.policy_weights.iter().enumerate() {
            if weight.policy_id.trim().is_empty() || !policy_ids.insert(weight.policy_id.as_str()) {
                return Err(GuardrailError::new(
                    format!("request.policy_weights[{index}].policy_id"),
                    "policy IDs must be non-empty and unique",
                ));
            }
            if !weight.probability.is_finite() || !(0.0..=1.0).contains(&weight.probability) {
                return Err(GuardrailError::new(
                    format!("request.policy_weights[{index}].probability"),
                    "policy probability must be finite and lie in [0, 1]",
                ));
            }
            positive_weight_seen |= weight.probability > 0.0;
            probability_sum += weight.probability;
        }
        if !positive_weight_seen || (probability_sum - 1.0).abs() > TOLERANCE {
            return Err(GuardrailError::new(
                "request.policy_weights",
                "mixture probabilities must sum to one and include positive support",
            ));
        }

        let mut requirement_ids = BTreeSet::new();
        let mut global_count = 0usize;
        let mut semantic_keys = BTreeSet::new();
        for (index, requirement) in self.requirements.iter().enumerate() {
            if requirement.requirement_id.trim().is_empty()
                || !requirement_ids.insert(requirement.requirement_id.as_str())
            {
                return Err(GuardrailError::new(
                    format!("request.requirements[{index}].requirement_id"),
                    "requirement IDs must be non-empty and unique",
                ));
            }
            if requirement.guardrail_id.trim().is_empty()
                || requirement.model_id.trim().is_empty()
            {
                return Err(GuardrailError::new(
                    format!("request.requirements[{index}]"),
                    "guardrail and model IDs must be non-empty",
                ));
            }
            validate_hash(
                &requirement.metric_contract_sha256,
                &format!("request.requirements[{index}].metric_contract_sha256"),
            )?;
            if !requirement.threshold.is_finite() || requirement.minimum_sample_size == 0 {
                return Err(GuardrailError::new(
                    format!("request.requirements[{index}]"),
                    "threshold must be finite and minimum sample size must be positive",
                ));
            }
            match (requirement.scope, requirement.stratum_id.as_deref()) {
                (GuardrailScope::Global, None) => global_count += 1,
                (GuardrailScope::Global, Some(_)) => {
                    return Err(GuardrailError::new(
                        format!("request.requirements[{index}].stratum_id"),
                        "GLOBAL guardrails must not name a local stratum",
                    ));
                }
                (GuardrailScope::Local, Some(stratum)) if !stratum.trim().is_empty() => {}
                (GuardrailScope::Local, _) => {
                    return Err(GuardrailError::new(
                        format!("request.requirements[{index}].stratum_id"),
                        "LOCAL guardrails require a non-empty stratum ID",
                    ));
                }
            }
            let semantic_key = (
                requirement.guardrail_id.as_str(),
                requirement.model_id.as_str(),
                requirement.scope as u8,
                requirement.stratum_id.as_deref().unwrap_or(GLOBAL_STRATUM),
            );
            if !semantic_keys.insert(semantic_key) {
                return Err(GuardrailError::new(
                    format!("request.requirements[{index}]"),
                    "duplicate guardrail/model/scope/stratum requirement",
                ));
            }
        }
        if global_count == 0 || global_count != self.expected_global_requirement_count {
            return Err(GuardrailError::new(
                "request.expected_global_requirement_count",
                "the complete required global-guardrail roster is missing or has the wrong count",
            ));
        }

        let expected_roster_digest = digest(&self.requirements)?;
        if expected_roster_digest != self.guardrail_roster_sha256 {
            return Err(GuardrailError::new(
                "request.guardrail_roster_sha256",
                "guardrail roster digest does not match the canonical applicable requirement list",
            ));
        }

        let mut bound_keys = BTreeSet::new();
        for (index, bound) in self.policy_bounds.iter().enumerate() {
            if !policy_ids.contains(bound.policy_id.as_str())
                || !requirement_ids.contains(bound.requirement_id.as_str())
            {
                return Err(GuardrailError::new(
                    format!("request.policy_bounds[{index}]"),
                    "bound evidence references an undeclared policy or requirement",
                ));
            }
            if !bound_keys.insert((bound.policy_id.as_str(), bound.requirement_id.as_str())) {
                return Err(GuardrailError::new(
                    format!("request.policy_bounds[{index}]"),
                    "duplicate policy/requirement evidence row",
                ));
            }
            validate_hash(
                &bound.source_receipt_sha256,
                &format!("request.policy_bounds[{index}].source_receipt_sha256"),
            )?;
            match bound.availability {
                GuardrailEvidenceAvailability::SimultaneousBoundsAvailable => {
                    let lower = bound.lower_confidence_bound.ok_or_else(|| {
                        GuardrailError::new(
                            format!("request.policy_bounds[{index}].lower_confidence_bound"),
                            "available evidence requires a lower confidence bound",
                        )
                    })?;
                    let upper = bound.upper_confidence_bound.ok_or_else(|| {
                        GuardrailError::new(
                            format!("request.policy_bounds[{index}].upper_confidence_bound"),
                            "available evidence requires an upper confidence bound",
                        )
                    })?;
                    if !lower.is_finite() || !upper.is_finite() || lower > upper {
                        return Err(GuardrailError::new(
                            format!("request.policy_bounds[{index}]"),
                            "confidence bounds must be finite and ordered",
                        ));
                    }
                    if bound.sample_size == 0 {
                        return Err(GuardrailError::new(
                            format!("request.policy_bounds[{index}].sample_size"),
                            "available bounds require a positive sample size",
                        ));
                    }
                    if bound.reason_code.is_some() {
                        return Err(GuardrailError::new(
                            format!("request.policy_bounds[{index}].reason_code"),
                            "available evidence must not carry a not-estimable reason code",
                        ));
                    }
                }
                GuardrailEvidenceAvailability::NotEstimable => {
                    if bound.lower_confidence_bound.is_some()
                        || bound.upper_confidence_bound.is_some()
                        || bound.reason_code.as_deref().is_none_or(str::is_empty)
                    {
                        return Err(GuardrailError::new(
                            format!("request.policy_bounds[{index}]"),
                            "NOT_ESTIMABLE requires no bounds and a non-empty reason code",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn request_sha256(&self) -> Result<String, GuardrailError> {
        let mut canonical = self.clone();
        canonical.canonicalize()?;
        digest(&canonical)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardrailRequirementResult {
    pub requirement_id: String,
    pub guardrail_id: String,
    pub metric_contract_sha256: String,
    pub scope: GuardrailScope,
    pub stratum_id: Option<String>,
    pub model_id: String,
    pub direction: GuardrailDirection,
    pub threshold: f64,
    pub mixture_lower_confidence_bound: Option<f64>,
    pub mixture_upper_confidence_bound: Option<f64>,
    pub decision: GuardrailCellDecision,
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoncompensatoryGuardrailReport {
    pub schema: String,
    pub version: String,
    pub evaluation_id: String,
    pub request_sha256: String,
    pub guardrail_roster_sha256: String,
    pub calibration_target_design_sha256: String,
    pub confirmatory_target_design_sha256: String,
    pub simultaneous_guardrail_failure_probability: f64,
    pub overall_decision: GuardrailOverallDecision,
    pub requirement_results: Vec<GuardrailRequirementResult>,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    evaluation_id: &'a str,
    request_sha256: &'a str,
    guardrail_roster_sha256: &'a str,
    calibration_target_design_sha256: &'a str,
    confirmatory_target_design_sha256: &'a str,
    simultaneous_guardrail_failure_probability: f64,
    overall_decision: GuardrailOverallDecision,
    requirement_results: &'a [GuardrailRequirementResult],
}

fn validate_hash(value: &str, path: &str) -> Result<(), GuardrailError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(GuardrailError::new(
            path,
            "expected a 64-character hexadecimal SHA-256 digest",
        ));
    }
    Ok(())
}

/// Verify a fixed randomized policy mixture against the complete applicable
/// global/local guardrail roster. Utility is intentionally not an input, so it
/// cannot compensate for a mandatory guardrail violation.
pub fn analyze_noncompensatory_guardrails(
    input: &NoncompensatoryGuardrailRequest,
) -> Result<NoncompensatoryGuardrailReport, GuardrailError> {
    let mut request = input.clone();
    request.canonicalize()?;

    let weight_map = request
        .policy_weights
        .iter()
        .map(|row| (row.policy_id.as_str(), row.probability))
        .collect::<BTreeMap<_, _>>();
    let bound_map = request
        .policy_bounds
        .iter()
        .map(|row| ((row.policy_id.as_str(), row.requirement_id.as_str()), row))
        .collect::<BTreeMap<_, _>>();

    let mut results = Vec::with_capacity(request.requirements.len());
    for requirement in &request.requirements {
        let mut lower_sum = 0.0;
        let mut upper_sum = 0.0;
        let mut unavailable_reason = None;
        for policy_id in weight_map.keys() {
            let probability = weight_map[policy_id];
            let Some(bound) = bound_map.get(&(*policy_id, requirement.requirement_id.as_str())) else {
                unavailable_reason.get_or_insert_with(|| "MISSING_POLICY_EVIDENCE".to_owned());
                continue;
            };
            match bound.availability {
                GuardrailEvidenceAvailability::NotEstimable => {
                    unavailable_reason.get_or_insert_with(|| {
                        bound.reason_code.clone().unwrap_or_else(|| "NOT_ESTIMABLE".into())
                    });
                }
                GuardrailEvidenceAvailability::SimultaneousBoundsAvailable => {
                    if probability > 0.0 {
                        lower_sum += probability * bound.lower_confidence_bound.unwrap_or_default();
                        upper_sum += probability * bound.upper_confidence_bound.unwrap_or_default();
                    }
                }
            }
        }

        let (decision, lower, upper, reason_code) = if let Some(reason) = unavailable_reason {
            (
                if reason == "MISSING_POLICY_EVIDENCE" {
                    GuardrailCellDecision::MissingEvidence
                } else {
                    GuardrailCellDecision::NotEstablished
                },
                None,
                None,
                Some(reason),
            )
        } else {
            match requirement.direction {
                GuardrailDirection::UpperBound if upper_sum <= requirement.threshold + TOLERANCE => (
                    GuardrailCellDecision::Pass,
                    Some(lower_sum),
                    Some(upper_sum),
                    None,
                ),
                GuardrailDirection::UpperBound if lower_sum > requirement.threshold + TOLERANCE => (
                    GuardrailCellDecision::ViolationEstablished,
                    Some(lower_sum),
                    Some(upper_sum),
                    Some("UPPER_GUARDRAIL_VIOLATION_ESTABLISHED".into()),
                ),
                GuardrailDirection::LowerBound if lower_sum >= requirement.threshold - TOLERANCE => (
                    GuardrailCellDecision::Pass,
                    Some(lower_sum),
                    Some(upper_sum),
                    None,
                ),
                GuardrailDirection::LowerBound if upper_sum < requirement.threshold - TOLERANCE => (
                    GuardrailCellDecision::ViolationEstablished,
                    Some(lower_sum),
                    Some(upper_sum),
                    Some("LOWER_GUARDRAIL_VIOLATION_ESTABLISHED".into()),
                ),
                _ => (
                    GuardrailCellDecision::NotEstablished,
                    Some(lower_sum),
                    Some(upper_sum),
                    Some("CONFIDENCE_REGION_STRADDLES_GUARDRAIL".into()),
                ),
            }
        };
        results.push(GuardrailRequirementResult {
            requirement_id: requirement.requirement_id.clone(),
            guardrail_id: requirement.guardrail_id.clone(),
            metric_contract_sha256: requirement.metric_contract_sha256.clone(),
            scope: requirement.scope,
            stratum_id: requirement.stratum_id.clone(),
            model_id: requirement.model_id.clone(),
            direction: requirement.direction,
            threshold: requirement.threshold,
            mixture_lower_confidence_bound: lower,
            mixture_upper_confidence_bound: upper,
            decision,
            reason_code,
        });
    }

    let overall_decision = if results
        .iter()
        .any(|row| row.decision == GuardrailCellDecision::ViolationEstablished)
    {
        GuardrailOverallDecision::Blocked
    } else if results.iter().any(|row| {
        matches!(
            row.decision,
            GuardrailCellDecision::NotEstablished | GuardrailCellDecision::MissingEvidence
        )
    }) {
        GuardrailOverallDecision::Indeterminate
    } else {
        GuardrailOverallDecision::Pass
    };

    let request_sha256 = request.request_sha256()?;
    let payload = ReportPayload {
        schema: REPORT_SCHEMA,
        version: REPORT_VERSION,
        evaluation_id: &request.evaluation_id,
        request_sha256: &request_sha256,
        guardrail_roster_sha256: &request.guardrail_roster_sha256,
        calibration_target_design_sha256: &request.calibration_target_design_sha256,
        confirmatory_target_design_sha256: &request.confirmatory_target_design_sha256,
        simultaneous_guardrail_failure_probability: request
            .simultaneous_guardrail_failure_probability,
        overall_decision,
        requirement_results: &results,
    };
    let report_sha256 = digest(&payload)?;
    Ok(NoncompensatoryGuardrailReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        evaluation_id: request.evaluation_id,
        request_sha256,
        guardrail_roster_sha256: request.guardrail_roster_sha256,
        calibration_target_design_sha256: request.calibration_target_design_sha256,
        confirmatory_target_design_sha256: request.confirmatory_target_design_sha256,
        simultaneous_guardrail_failure_probability: request
            .simultaneous_guardrail_failure_probability,
        overall_decision,
        requirement_results: results,
        report_sha256,
    })
}

pub fn verify_noncompensatory_guardrail_report(
    request: &NoncompensatoryGuardrailRequest,
    report: &NoncompensatoryGuardrailReport,
) -> Result<(), GuardrailError> {
    if analyze_noncompensatory_guardrails(request)? != *report {
        return Err(GuardrailError::new(
            "report",
            "stored guardrail report differs from complete deterministic reconstruction",
        ));
    }
    Ok(())
}

fn digest<T: Serialize>(value: &T) -> Result<String, GuardrailError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| GuardrailError::new("serialization", error.to_string()))?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const HASH_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const HASH_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn requirements() -> Vec<GuardrailRequirement> {
        vec![
            GuardrailRequirement {
                requirement_id: "global-crash-null".into(),
                guardrail_id: "crash-rate".into(),
                metric_contract_sha256: HASH_A.into(),
                scope: GuardrailScope::Global,
                stratum_id: None,
                model_id: "model-0".into(),
                direction: GuardrailDirection::UpperBound,
                threshold: 0.05,
                minimum_sample_size: 100,
            },
            GuardrailRequirement {
                requirement_id: "local-revenue-null".into(),
                guardrail_id: "revenue-floor".into(),
                metric_contract_sha256: HASH_B.into(),
                scope: GuardrailScope::Local,
                stratum_id: Some("segment-a".into()),
                model_id: "model-0".into(),
                direction: GuardrailDirection::LowerBound,
                threshold: 10.0,
                minimum_sample_size: 100,
            },
            GuardrailRequirement {
                requirement_id: "local-latency-null".into(),
                guardrail_id: "latency-cap".into(),
                metric_contract_sha256: HASH_C.into(),
                scope: GuardrailScope::Local,
                stratum_id: Some("segment-b".into()),
                model_id: "model-0".into(),
                direction: GuardrailDirection::UpperBound,
                threshold: 200.0,
                minimum_sample_size: 100,
            },
        ]
    }

    fn bound(
        policy_id: &str,
        requirement_id: &str,
        lower: f64,
        upper: f64,
    ) -> GuardrailPolicyBound {
        GuardrailPolicyBound {
            policy_id: policy_id.into(),
            requirement_id: requirement_id.into(),
            availability: GuardrailEvidenceAvailability::SimultaneousBoundsAvailable,
            lower_confidence_bound: Some(lower),
            upper_confidence_bound: Some(upper),
            sample_size: 1_000,
            source_receipt_sha256: HASH_D.into(),
            reason_code: None,
        }
    }

    fn request() -> NoncompensatoryGuardrailRequest {
        let reqs = requirements();
        let roster_sha = digest(&reqs).unwrap();
        let mut bounds = Vec::new();
        for policy_id in ["policy-a", "policy-b"] {
            bounds.push(bound(policy_id, "global-crash-null", 0.01, 0.02));
            bounds.push(bound(policy_id, "local-revenue-null", 10.5, 12.0));
            bounds.push(bound(policy_id, "local-latency-null", 100.0, 180.0));
        }
        NoncompensatoryGuardrailRequest::new(
            "guardrail-fixture",
            vec![
                GuardrailPolicyWeight {
                    policy_id: "policy-a".into(),
                    probability: 0.4,
                },
                GuardrailPolicyWeight {
                    policy_id: "policy-b".into(),
                    probability: 0.6,
                },
            ],
            reqs,
            2,
            3,
            1,
            roster_sha,
            HASH_A,
            HASH_A,
            CalibrationTargetRelation::SameTargetLawByDesign,
            true,
            true,
            0.01,
            bounds,
        )
        .unwrap()
    }

    #[test]
    fn passes_all_global_and_local_guardrails() {
        let request = request();
        let report = analyze_noncompensatory_guardrails(&request).unwrap();
        assert_eq!(report.overall_decision, GuardrailOverallDecision::Pass);
        assert!(report
            .requirement_results
            .iter()
            .all(|row| row.decision == GuardrailCellDecision::Pass));
        verify_noncompensatory_guardrail_report(&request, &report).unwrap();
    }

    #[test]
    fn blocks_when_upper_guardrail_violation_is_established() {
        let mut request = request();
        let row = request
            .policy_bounds
            .iter_mut()
            .find(|row| row.requirement_id == "global-crash-null" && row.policy_id == "policy-a")
            .unwrap();
        row.lower_confidence_bound = Some(0.06);
        row.upper_confidence_bound = Some(0.08);
        let row = request
            .policy_bounds
            .iter_mut()
            .find(|row| row.requirement_id == "global-crash-null" && row.policy_id == "policy-b")
            .unwrap();
        row.lower_confidence_bound = Some(0.07);
        row.upper_confidence_bound = Some(0.09);
        let report = analyze_noncompensatory_guardrails(&request).unwrap();
        assert_eq!(report.overall_decision, GuardrailOverallDecision::Blocked);
    }

    #[test]
    fn confidence_region_straddling_floor_is_indeterminate() {
        let mut request = request();
        let row = request
            .policy_bounds
            .iter_mut()
            .find(|row| row.requirement_id == "local-revenue-null" && row.policy_id == "policy-b")
            .unwrap();
        row.lower_confidence_bound = Some(8.0);
        row.upper_confidence_bound = Some(12.0);
        let report = analyze_noncompensatory_guardrails(&request).unwrap();
        assert_eq!(report.overall_decision, GuardrailOverallDecision::Indeterminate);
        assert_eq!(
            report.requirement_results.iter()
                .find(|row| row.requirement_id == "local-revenue-null")
                .unwrap().decision,
            GuardrailCellDecision::NotEstablished
        );
    }

    #[test]
    fn missing_policy_guardrail_evidence_is_indeterminate() {
        let mut request = request();
        request.policy_bounds.retain(|row| {
            !(row.policy_id == "policy-b" && row.requirement_id == "local-latency-null")
        });
        let report = analyze_noncompensatory_guardrails(&request).unwrap();
        assert_eq!(report.overall_decision, GuardrailOverallDecision::Indeterminate);
        assert_eq!(
            report.requirement_results.iter()
                .find(|row| row.requirement_id == "local-latency-null")
                .unwrap().decision,
            GuardrailCellDecision::MissingEvidence
        );
    }

    #[test]
    fn not_estimable_stratum_is_indeterminate() {
        let mut request = request();
        let row = request
            .policy_bounds
            .iter_mut()
            .find(|row| row.policy_id == "policy-a" && row.requirement_id == "local-revenue-null")
            .unwrap();
        row.availability = GuardrailEvidenceAvailability::NotEstimable;
        row.lower_confidence_bound = None;
        row.upper_confidence_bound = None;
        row.reason_code = Some("LOW_SAMPLE_COUNT".into());
        let report = analyze_noncompensatory_guardrails(&request).unwrap();
        assert_eq!(report.overall_decision, GuardrailOverallDecision::Indeterminate);
    }

    #[test]
    fn a_hard_guardrail_cannot_be_traded_for_utility() {
        // No utility value is accepted by the verifier API; therefore utility cannot compensate.
        let mut request = request();
        for row in &mut request.policy_bounds {
            if row.requirement_id == "global-crash-null" {
                row.lower_confidence_bound = Some(0.06);
                row.upper_confidence_bound = Some(0.08);
            }
        }
        let report = analyze_noncompensatory_guardrails(&request).unwrap();
        assert_eq!(report.overall_decision, GuardrailOverallDecision::Blocked);
    }

    #[test]
    fn rejects_incomplete_global_guardrail_roster() {
        let mut request = request();
        request.expected_global_requirement_count = 2;
        assert!(analyze_noncompensatory_guardrails(&request).is_err());
    }

    #[test]
    fn rejects_non_linear_mixture_estimands() {
        let mut request = request();
        request.mixture_estimand_linear = false;
        assert!(analyze_noncompensatory_guardrails(&request).is_err());
    }

    #[test]
    fn rejects_unequal_target_designs() {
        let mut request = request();
        request.confirmatory_target_design_sha256 = HASH_B.into();
        assert!(analyze_noncompensatory_guardrails(&request).is_err());
    }

    #[test]
    fn rejects_invalid_policy_mixture() {
        let mut request = request();
        request.policy_weights[0].probability = 0.2;
        assert!(analyze_noncompensatory_guardrails(&request).is_err());
    }

    #[test]
    fn rejects_invalid_confidence_bounds() {
        let mut request = request();
        let row = request.policy_bounds.iter_mut().find(|row| row.policy_id == "policy-a").unwrap();
        row.lower_confidence_bound = Some(f64::NAN);
        assert!(analyze_noncompensatory_guardrails(&request).is_err());
    }

    #[test]
    fn report_tampering_is_detected() {
        let request = request();
        let mut report = analyze_noncompensatory_guardrails(&request).unwrap();
        report.overall_decision = GuardrailOverallDecision::Blocked;
        assert!(verify_noncompensatory_guardrail_report(&request, &report).is_err());
    }
}
