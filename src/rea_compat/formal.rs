//! Conservative projection of REA v0 claim evaluations into the formal
//! ProviderObservation boundary.
//!
//! This is a typed mapping/export adapter, not a verifier. It recomputes the
//! complete claim-request digest and checks internal result/reference
//! consistency, but it does not authenticate the producer or independently
//! replay the derivation. The independent replay verifier remains a separate
//! qualification boundary.

use super::claims::{
    CLAIM_EVALUATION_SCHEMA, CLAIM_EVALUATOR_VERSION, CLAIM_SCHEMA, ClaimDisposition,
    ClaimError, ClaimEvaluation, ClaimPredicate, ClaimRequest, EvidenceAssessmentKind,
    validate_claim,
};
use super::{canonical_jcs_bytes, sha256_hex};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::fmt;

pub const FORMAL_PROVIDER_OBSERVATION_SCHEMA: &str =
    "luminous.rea.formal-provider-observation.v1";
pub const FORMAL_PROVIDER_OBSERVATION_RULE: &str =
    "rea-provider-observation-projection/v1";
pub const MAX_FORMAL_PROVIDER_OBSERVATION_JSON_BYTES: usize = 4 * 1024 * 1024;

const V0_OBSERVATION_RULE_ID: &str =
    "provider-reported-observation/exact-normalized-result/v0";
const V0_UNVERIFIED_RULE_ID: &str = "no-registered-predicate-rule/v0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormalEvidenceClass {
    ProviderObservation,
}

/// Semantic polarity at the formal boundary. This is deliberately not a
/// workflow pass/fail enum and contains no artifact-level truth verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormalSemanticPolarity {
    PositiveSupport,
    SemanticCounterevidence,
    NoSemanticConclusion,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormalClaimBinding {
    /// SHA-256 of RFC 8785 canonical bytes for the complete REA claim request.
    pub claim_request_sha256: String,
    /// Producer derivation commitment. This is not a producer signature.
    pub derivation_sha256: String,
    pub request_schema: String,
    pub result_schema: String,
    pub evaluator_version: String,
    pub evaluator_rule_id: String,
    pub claim_id: String,
    pub predicate: ClaimPredicate,
    pub subject_sha256: String,
    pub operation: String,
    pub platform: Option<String>,
    pub platform_version: Option<String>,
    pub environment_id: Option<String>,
    pub asserted_value: Value,
    pub source_predicate_type: Option<String>,
    /// Ordering is preserved exactly as committed by the request digest.
    pub requested_evidence_ids: Vec<String>,
    pub assumptions: Vec<String>,
    pub falsification_condition: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormalProviderObservation {
    pub schema: String,
    pub mapping_rule: String,
    pub evidence_class: FormalEvidenceClass,
    pub semantic_polarity: FormalSemanticPolarity,
    /// Preserve the original REA outcome; do not collapse it to a Boolean.
    pub rea_disposition: ClaimDisposition,
    pub binding: FormalClaimBinding,
    /// Full source result remains attached so per-premise assessments,
    /// contradictions, missing references and claim ceilings are not lost.
    pub rea_evaluation: ClaimEvaluation,
    pub claim_ceiling: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormalMappingErrorKind {
    InvalidEvaluation,
    UnsupportedPredicate,
    InvalidReferences,
    InvalidDigest,
    Canonicalization,
    InputTooLarge,
    InvalidJson,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormalMappingError {
    pub kind: FormalMappingErrorKind,
    pub message: String,
}

impl FormalMappingError {
    fn new(kind: FormalMappingErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for FormalMappingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for FormalMappingError {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_evidence_id(value: &str) -> bool {
    value
        .strip_prefix("ev_")
        .is_some_and(valid_sha256)
}

fn request_from_evaluation(
    evaluation: &ClaimEvaluation,
) -> Result<ClaimRequest, FormalMappingError> {
    let request = ClaimRequest {
        schema: evaluation.schema.clone(),
        claim_id: evaluation.claim_id.clone(),
        predicate: evaluation.predicate,
        scope: evaluation.scope.clone(),
        asserted_value: evaluation.asserted_value.clone(),
        source_predicate_type: evaluation.source_predicate_type.clone(),
        evidence_ids: evaluation.requested_evidence_ids.clone(),
        assumptions: evaluation.assumptions.clone(),
        falsification_condition: evaluation.falsification_condition.clone(),
    };
    validate_claim(&request).map_err(|error: ClaimError| {
        FormalMappingError::new(
            FormalMappingErrorKind::InvalidEvaluation,
            format!("embedded REA claim request is invalid: {error}"),
        )
    })?;
    Ok(request)
}

fn validate_reference_partition(
    evaluation: &ClaimEvaluation,
) -> Result<(), FormalMappingError> {
    let mut requested = HashSet::with_capacity(evaluation.requested_evidence_ids.len());
    for id in &evaluation.requested_evidence_ids {
        if !valid_evidence_id(id) || !requested.insert(id.as_str()) {
            return Err(FormalMappingError::new(
                FormalMappingErrorKind::InvalidReferences,
                "requested Evidence IDs must be valid, unique semantic IDs",
            ));
        }
    }

    let mut assessed = HashSet::with_capacity(evaluation.evidence_assessments.len());
    for assessment in &evaluation.evidence_assessments {
        if !requested.contains(assessment.evidence_id.as_str())
            || !assessed.insert(assessment.evidence_id.as_str())
        {
            return Err(FormalMappingError::new(
                FormalMappingErrorKind::InvalidReferences,
                "Evidence assessments must be unique and reference requested Evidence IDs",
            ));
        }
    }

    let mut missing = HashSet::with_capacity(evaluation.missing_evidence_ids.len());
    for id in &evaluation.missing_evidence_ids {
        if !valid_evidence_id(id)
            || !requested.contains(id.as_str())
            || !missing.insert(id.as_str())
            || assessed.contains(id.as_str())
        {
            return Err(FormalMappingError::new(
                FormalMappingErrorKind::InvalidReferences,
                "missing Evidence IDs must be unique requested IDs with no assessment",
            ));
        }
    }

    if assessed.union(&missing).count() != requested.len() {
        return Err(FormalMappingError::new(
            FormalMappingErrorKind::InvalidReferences,
            "every requested Evidence ID must have exactly one assessment or missing marker",
        ));
    }
    if evaluation.claim_ceiling.is_empty() {
        return Err(FormalMappingError::new(
            FormalMappingErrorKind::InvalidEvaluation,
            "REA claim ceiling must not be empty",
        ));
    }
    Ok(())
}

fn recompute_request_digest(
    request: &ClaimRequest,
) -> Result<String, FormalMappingError> {
    let value = serde_json::to_value(request).map_err(|error| {
        FormalMappingError::new(
            FormalMappingErrorKind::Canonicalization,
            format!("claim request serialization failed: {error}"),
        )
    })?;
    let canonical = canonical_jcs_bytes(&value).map_err(|error| {
        FormalMappingError::new(
            FormalMappingErrorKind::Canonicalization,
            format!("claim request JCS canonicalization failed: {error}"),
        )
    })?;
    Ok(sha256_hex(&canonical))
}

fn validate_evaluation(evaluation: &ClaimEvaluation) -> Result<FormalSemanticPolarity, FormalMappingError> {
    if evaluation.schema != CLAIM_EVALUATION_SCHEMA
        || evaluation.evaluator_version != CLAIM_EVALUATOR_VERSION
    {
        return Err(FormalMappingError::new(
            FormalMappingErrorKind::InvalidEvaluation,
            "unsupported REA claim result schema or evaluator version",
        ));
    }
    if evaluation.predicate != ClaimPredicate::ProviderReportedObservation {
        return Err(FormalMappingError::new(
            FormalMappingErrorKind::UnsupportedPredicate,
            "formal ProviderObservation mapping accepts only provider_reported_observation",
        ));
    }
    if !valid_sha256(&evaluation.derivation_sha256) {
        return Err(FormalMappingError::new(
            FormalMappingErrorKind::InvalidDigest,
            "REA derivation commitment must be lowercase SHA-256",
        ));
    }

    let expected_rule = if evaluation.assumptions.is_empty() {
        V0_OBSERVATION_RULE_ID
    } else {
        V0_UNVERIFIED_RULE_ID
    };
    if evaluation.rule_id != expected_rule {
        return Err(FormalMappingError::new(
            FormalMappingErrorKind::InvalidEvaluation,
            "REA evaluator rule ID is inconsistent with the v0 predicate/assumption contract",
        ));
    }

    let request = request_from_evaluation(evaluation)?;
    validate_reference_partition(evaluation)?;
    let actual_request_digest = recompute_request_digest(&request)?;
    if actual_request_digest != evaluation.claim_request_sha256 {
        return Err(FormalMappingError::new(
            FormalMappingErrorKind::InvalidDigest,
            "REA claim request digest does not match the complete embedded request",
        ));
    }

    let supports = evaluation
        .evidence_assessments
        .iter()
        .filter(|item| item.kind == EvidenceAssessmentKind::Supports)
        .count();
    let contradicts = evaluation
        .evidence_assessments
        .iter()
        .filter(|item| item.kind == EvidenceAssessmentKind::Contradicts)
        .count();
    let scope_unverifiable = evaluation
        .evidence_assessments
        .iter()
        .filter(|item| item.kind == EvidenceAssessmentKind::ScopeUnverifiable)
        .count();

    let polarity = match evaluation.disposition {
        ClaimDisposition::SupportedWithinScope
            if supports > 0
                && contradicts == 0
                && scope_unverifiable == 0
                && evaluation.missing_evidence_ids.is_empty()
                && evaluation.assumptions.is_empty() =>
        {
            FormalSemanticPolarity::PositiveSupport
        }
        ClaimDisposition::RefutedWithinScope
            if supports == 0
                && contradicts > 0
                && scope_unverifiable == 0
                && evaluation.missing_evidence_ids.is_empty()
                && evaluation.assumptions.is_empty() =>
        {
            FormalSemanticPolarity::SemanticCounterevidence
        }
        ClaimDisposition::Conflicted if supports > 0 && contradicts > 0 => {
            FormalSemanticPolarity::NoSemanticConclusion
        }
        ClaimDisposition::Unverified if !evaluation.assumptions.is_empty() => {
            FormalSemanticPolarity::NoSemanticConclusion
        }
        ClaimDisposition::SupportedWithinScope | ClaimDisposition::RefutedWithinScope => {
            return Err(FormalMappingError::new(
                FormalMappingErrorKind::InvalidEvaluation,
                "REA disposition is not justified by its per-premise assessments",
            ));
        }
        _ => FormalSemanticPolarity::NoSemanticConclusion,
    };
    Ok(polarity)
}

/// Project a validated v0 provider-observation evaluation into the formal
/// adapter envelope. The full REA result and exact request binding are retained.
///
/// This function does not authenticate the producer or independently replay
/// the derivation. Consumers that require independent replay must separately
/// require ReplayMatch from the REA replay verifier.
pub fn export_formal_provider_observation(
    evaluation: &ClaimEvaluation,
) -> Result<FormalProviderObservation, FormalMappingError> {
    let polarity = validate_evaluation(evaluation)?;
    let binding = FormalClaimBinding {
        claim_request_sha256: evaluation.claim_request_sha256.clone(),
        derivation_sha256: evaluation.derivation_sha256.clone(),
        request_schema: CLAIM_SCHEMA.to_owned(),
        result_schema: CLAIM_EVALUATION_SCHEMA.to_owned(),
        evaluator_version: evaluation.evaluator_version.clone(),
        evaluator_rule_id: evaluation.rule_id.clone(),
        claim_id: evaluation.claim_id.clone(),
        predicate: evaluation.predicate,
        subject_sha256: evaluation.scope.subject_sha256.clone(),
        operation: evaluation.scope.operation.clone(),
        platform: evaluation.scope.platform.clone(),
        platform_version: evaluation.scope.platform_version.clone(),
        environment_id: evaluation.scope.environment_id.clone(),
        asserted_value: evaluation.asserted_value.clone(),
        source_predicate_type: evaluation.source_predicate_type.clone(),
        requested_evidence_ids: evaluation.requested_evidence_ids.clone(),
        assumptions: evaluation.assumptions.clone(),
        falsification_condition: evaluation.falsification_condition.clone(),
    };

    Ok(FormalProviderObservation {
        schema: FORMAL_PROVIDER_OBSERVATION_SCHEMA.to_owned(),
        mapping_rule: FORMAL_PROVIDER_OBSERVATION_RULE.to_owned(),
        evidence_class: FormalEvidenceClass::ProviderObservation,
        semantic_polarity: polarity,
        rea_disposition: evaluation.disposition,
        binding,
        rea_evaluation: evaluation.clone(),
        claim_ceiling: vec![
            "This formal object represents only a provider-reported observation meta-claim.".to_owned(),
            "It does not prove the underlying artifact proposition, producer authenticity, signature validity, signer trust, package authorization, completeness or safety.".to_owned(),
            "The derivation SHA-256 is an integrity commitment, not a producer signature or trust anchor.".to_owned(),
            "Independent semantic replay is a separate required check and is not performed by this exporter.".to_owned(),
        ],
    })
}

/// Validate a typed formal envelope after deserialization or transport.
pub fn validate_formal_provider_observation(
    envelope: &FormalProviderObservation,
) -> Result<(), FormalMappingError> {
    if envelope.schema != FORMAL_PROVIDER_OBSERVATION_SCHEMA
        || envelope.mapping_rule != FORMAL_PROVIDER_OBSERVATION_RULE
        || envelope.evidence_class != FormalEvidenceClass::ProviderObservation
    {
        return Err(FormalMappingError::new(
            FormalMappingErrorKind::InvalidEvaluation,
            "unsupported formal provider-observation envelope schema, mapping rule or evidence class",
        ));
    }
    let expected = export_formal_provider_observation(&envelope.rea_evaluation)?;
    if &expected != envelope {
        return Err(FormalMappingError::new(
            FormalMappingErrorKind::InvalidEvaluation,
            "formal envelope projection does not exactly match its embedded REA evaluation",
        ));
    }
    Ok(())
}

/// Parse a formal envelope with a strict byte bound and closed, deny-unknown
/// structs. Raw duplicate-key rejection remains the responsibility of the
/// upstream REA request/result parser and replay verifier.
pub fn parse_formal_provider_observation_json(
    bytes: &[u8],
) -> Result<FormalProviderObservation, FormalMappingError> {
    if bytes.len() > MAX_FORMAL_PROVIDER_OBSERVATION_JSON_BYTES {
        return Err(FormalMappingError::new(
            FormalMappingErrorKind::InputTooLarge,
            "formal provider-observation envelope exceeds its byte limit",
        ));
    }
    let envelope: FormalProviderObservation = serde_json::from_slice(bytes).map_err(|error| {
        FormalMappingError::new(
            FormalMappingErrorKind::InvalidJson,
            format!("formal provider-observation envelope parsing failed: {error}"),
        )
    })?;
    validate_formal_provider_observation(&envelope)?;
    Ok(envelope)
}
