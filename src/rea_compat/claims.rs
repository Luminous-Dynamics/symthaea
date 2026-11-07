//! Conservative, replayable claim evaluation over verified REA Evidence.
//!
//! V0 deliberately proves only that a pinned Evidence record reports an exact
//! normalized value for an exact artifact/operation scope. It does not infer
//! artifact truth, receipt validity, authenticity, signature trust, completeness,
//! causal correctness, provider independence, or security-control compliance.

use super::{
    EVIDENCE_ID_PREFIX, VerifiedEvidence, canonical_jcs_bytes, sha256_hex, verify_evidence_value,
};
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize as SerdeDeserialize, Serialize};
use serde_json::{Map, Number, Value};
use std::collections::{HashMap, HashSet};
use std::fmt;

pub const CLAIM_SCHEMA: &str = "luminous.rea.claim.v0";
pub const CLAIM_EVALUATION_SCHEMA: &str = "luminous.rea.claim-evaluation.v0";
pub const CLAIM_EVALUATOR_VERSION: &str = "rea-claim-evaluator/v0";
pub const MAX_CLAIM_JSON_BYTES: usize = 1024 * 1024;
pub const MAX_CLAIM_EVIDENCE_RECORDS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, SerdeDeserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimPredicate {
    ProviderReportedObservation,
    MemberContentIntegrity,
    ArchiveContentIdentity,
    WholeArchiveCompleteness,
    ProducerAuthenticity,
    SignatureCryptographicValidity,
    SignerIdentity,
    SignerTrust,
    PackageAuthorization,
    PlatformAcceptance,
    ProviderCorroboration,
    CausalFinding,
    SecurityControlPass,
}

impl ClaimPredicate {
    fn has_v0_evaluator(self) -> bool {
        matches!(self, Self::ProviderReportedObservation)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, SerdeDeserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimDisposition {
    SupportedWithinScope,
    RefutedWithinScope,
    Underdetermined,
    Conflicted,
    Inapplicable,
    Unverified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimScope {
    pub subject_sha256: String,
    pub operation: String,
    #[serde(default)]
    pub platform: Option<String>,
    #[serde(default)]
    pub platform_version: Option<String>,
    #[serde(default)]
    pub environment_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimRequest {
    pub schema: String,
    pub claim_id: String,
    pub predicate: ClaimPredicate,
    pub scope: ClaimScope,
    pub asserted_value: Value,
    #[serde(default)]
    pub source_predicate_type: Option<String>,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
    #[serde(default)]
    pub assumptions: Vec<String>,
    pub falsification_condition: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, SerdeDeserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceAssessmentKind {
    Supports,
    Contradicts,
    Inapplicable,
    ScopeUnverifiable,
    NotEvaluated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceAssessment {
    pub evidence_id: String,
    pub kind: EvidenceAssessmentKind,
    pub provider_id: Option<String>,
    pub provider_name: Option<String>,
    pub provider_version: Option<String>,
    pub confidence: Option<String>,
    pub authority: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimEvaluation {
    pub schema: String,
    pub evaluator_version: String,
    pub rule_id: String,
    pub claim_id: String,
    pub claim_request_sha256: String,
    pub predicate: ClaimPredicate,
    pub disposition: ClaimDisposition,
    pub scope: ClaimScope,
    pub asserted_value: Value,
    pub source_predicate_type: Option<String>,
    pub requested_evidence_ids: Vec<String>,
    pub assumptions: Vec<String>,
    pub evidence_assessments: Vec<EvidenceAssessment>,
    pub missing_evidence_ids: Vec<String>,
    pub unmet_obligations: Vec<String>,
    pub falsification_condition: String,
    pub claim_ceiling: Vec<String>,
    pub derivation_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimErrorKind {
    InvalidJson,
    InputTooLarge,
    InvalidClaim,
    InvalidEvidence,
    DuplicateEvidence,
    UnexpectedEvidence,
    ResourceLimit,
    Canonicalization,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimError {
    pub kind: ClaimErrorKind,
    pub message: String,
}

impl ClaimError {
    fn new(kind: ClaimErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for ClaimError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for ClaimError {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_evidence_id(value: &str) -> bool {
    value
        .strip_prefix(EVIDENCE_ID_PREFIX)
        .is_some_and(valid_sha256)
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

pub fn validate_claim(claim: &ClaimRequest) -> Result<(), ClaimError> {
    let invalid = |message: &str| ClaimError::new(ClaimErrorKind::InvalidClaim, message);

    if claim.schema != CLAIM_SCHEMA {
        return Err(invalid("unsupported claim schema"));
    }
    if !nonempty(&claim.claim_id) {
        return Err(invalid("claim_id must be non-empty"));
    }
    if !valid_sha256(&claim.scope.subject_sha256) {
        return Err(invalid(
            "scope.subject_sha256 must be 64 lowercase hexadecimal characters",
        ));
    }
    if !nonempty(&claim.scope.operation) {
        return Err(invalid("scope.operation must be non-empty"));
    }
    for (name, value) in [
        ("scope.platform", claim.scope.platform.as_deref()),
        (
            "scope.platform_version",
            claim.scope.platform_version.as_deref(),
        ),
        (
            "scope.environment_id",
            claim.scope.environment_id.as_deref(),
        ),
    ] {
        if value.is_some_and(|value| !nonempty(value)) {
            return Err(invalid(&format!("{name} must be non-empty when present")));
        }
    }
    if !nonempty(&claim.falsification_condition) {
        return Err(invalid("falsification_condition must be non-empty"));
    }
    if claim
        .assumptions
        .iter()
        .any(|assumption| !nonempty(assumption))
    {
        return Err(invalid("assumptions must contain only non-empty strings"));
    }

    match claim.predicate {
        ClaimPredicate::ProviderReportedObservation => {
            if !claim.source_predicate_type.as_deref().is_some_and(nonempty) {
                return Err(invalid(
                    "provider_reported_observation requires source_predicate_type",
                ));
            }
        }
        _ if claim.source_predicate_type.is_some() => {
            return Err(invalid(
                "source_predicate_type is only valid for provider_reported_observation",
            ));
        }
        _ => {}
    }

    if claim.evidence_ids.len() > MAX_CLAIM_EVIDENCE_RECORDS {
        return Err(ClaimError::new(
            ClaimErrorKind::ResourceLimit,
            format!(
                "claim references {} Evidence records; limit is {}",
                claim.evidence_ids.len(),
                MAX_CLAIM_EVIDENCE_RECORDS
            ),
        ));
    }

    let mut seen = HashSet::with_capacity(claim.evidence_ids.len());
    for evidence_id in &claim.evidence_ids {
        if !valid_evidence_id(evidence_id) {
            return Err(invalid(
                "evidence_ids must contain ev_-prefixed lowercase SHA-256 identifiers",
            ));
        }
        if !seen.insert(evidence_id) {
            return Err(invalid("evidence_ids must not contain duplicates"));
        }
    }
    Ok(())
}

/// Parse a claim document while rejecting duplicate JSON property names at
/// every nesting level, including inside asserted_value.
pub fn parse_claim_json(input: &[u8]) -> Result<ClaimRequest, ClaimError> {
    if input.len() > MAX_CLAIM_JSON_BYTES {
        return Err(ClaimError::new(
            ClaimErrorKind::InputTooLarge,
            format!(
                "claim JSON is {} bytes; limit is {} bytes",
                input.len(),
                MAX_CLAIM_JSON_BYTES
            ),
        ));
    }

    let mut deserializer = serde_json::Deserializer::from_slice(input);
    let parsed = UniqueJsonValue::deserialize(&mut deserializer)
        .map_err(|error| ClaimError::new(ClaimErrorKind::InvalidJson, error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| ClaimError::new(ClaimErrorKind::InvalidJson, error.to_string()))?;

    let root = parsed.0.as_object().ok_or_else(|| {
        ClaimError::new(ClaimErrorKind::InvalidClaim, "claim must be a JSON object")
    })?;
    for required in [
        "schema",
        "claim_id",
        "predicate",
        "scope",
        "asserted_value",
        "source_predicate_type",
        "evidence_ids",
        "assumptions",
        "falsification_condition",
    ] {
        if !root.contains_key(required) {
            return Err(ClaimError::new(
                ClaimErrorKind::InvalidClaim,
                format!("missing required claim field: {required}"),
            ));
        }
    }

    let claim: ClaimRequest = serde_json::from_value(parsed.0)
        .map_err(|error| ClaimError::new(ClaimErrorKind::InvalidClaim, error.to_string()))?;
    validate_claim(&claim)?;
    Ok(claim)
}

struct UniqueJsonValue(Value);

impl<'de> Deserialize<'de> for UniqueJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueJsonVisitor)
    }
}

struct UniqueJsonVisitor;

impl<'de> Visitor<'de> for UniqueJsonVisitor {
    type Value = UniqueJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON with unique object property names")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(|number| UniqueJsonValue(Value::Number(number)))
            .ok_or_else(|| E::custom("non-finite numbers are not permitted"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::String(value)))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Null))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        UniqueJsonValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<UniqueJsonValue>()? {
            values.push(value.0);
        }
        Ok(UniqueJsonValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        let mut keys = HashSet::new();
        while let Some((key, value)) = object.next_entry::<String, UniqueJsonValue>()? {
            if !keys.insert(key.clone()) {
                return Err(de::Error::custom(format!(
                    "duplicate JSON object key '{key}'"
                )));
            }
            values.insert(key, value.0);
        }
        Ok(UniqueJsonValue(Value::Object(values)))
    }
}

fn optional_string(value: &Value, path: &[&str]) -> Option<String> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_str().map(str::to_owned)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScopeResult {
    Match,
    Inapplicable,
    Unverifiable,
}

fn compare_optional_scope(requested: Option<&str>, observed: Option<&str>) -> ScopeResult {
    match (requested, observed) {
        (Some(left), Some(right)) if left == right => ScopeResult::Match,
        (Some(_), Some(_)) => ScopeResult::Inapplicable,
        (None, None) => ScopeResult::Match,
        _ => ScopeResult::Unverifiable,
    }
}

fn assess_scope(claim: &ClaimRequest, evidence: &Value) -> (ScopeResult, String) {
    let observed_subject = optional_string(evidence, &["subject", "digest", "sha256"]);
    match observed_subject.as_deref() {
        Some(value) if value != claim.scope.subject_sha256 => {
            return (
                ScopeResult::Inapplicable,
                "subject digest does not match claim scope".to_owned(),
            );
        }
        None => {
            return (
                ScopeResult::Unverifiable,
                "Evidence has no subject digest to bind this claim".to_owned(),
            );
        }
        _ => {}
    }

    let observed_operation = optional_string(evidence, &["operation"]);
    if observed_operation.as_deref() != Some(claim.scope.operation.as_str()) {
        return (
            ScopeResult::Inapplicable,
            "operation does not match claim scope".to_owned(),
        );
    }

    if claim.scope.platform_version.is_some() {
        return (
            ScopeResult::Unverifiable,
            "REA Evidence v1 does not encode platform_version".to_owned(),
        );
    }

    let observed_platform = optional_string(evidence, &["environment", "platform"]);
    let observed_environment_id = optional_string(evidence, &["environment", "id"]);

    let platform = compare_optional_scope(
        claim.scope.platform.as_deref(),
        observed_platform.as_deref(),
    );
    if platform != ScopeResult::Match {
        return (
            platform,
            if platform == ScopeResult::Inapplicable {
                "platform does not match claim scope".to_owned()
            } else {
                "platform scope is absent or broader than the Evidence observation".to_owned()
            },
        );
    }

    let environment = compare_optional_scope(
        claim.scope.environment_id.as_deref(),
        observed_environment_id.as_deref(),
    );
    if environment != ScopeResult::Match {
        return (
            environment,
            if environment == ScopeResult::Inapplicable {
                "environment_id does not match claim scope".to_owned()
            } else {
                "environment identity is not recorded at the requested scope".to_owned()
            },
        );
    }

    (
        ScopeResult::Match,
        "exact subject, operation and represented environment scope match".to_owned(),
    )
}

struct ProviderFields {
    provider_id: Option<String>,
    provider_name: Option<String>,
    provider_version: Option<String>,
    confidence: Option<String>,
    authority: Option<String>,
}

fn provider_fields(evidence: &Value) -> ProviderFields {
    ProviderFields {
        provider_id: optional_string(evidence, &["provider", "id"]),
        provider_name: optional_string(evidence, &["provider", "name"]),
        provider_version: optional_string(evidence, &["provider", "version"]),
        confidence: optional_string(evidence, &["confidence"]),
        authority: optional_string(evidence, &["authority"]),
    }
}

fn canonical_json_equal(left: &Value, right: &Value) -> Result<bool, ClaimError> {
    let left_bytes = canonical_jcs_bytes(left)
        .map_err(|error| ClaimError::new(ClaimErrorKind::Canonicalization, error.to_string()))?;
    let right_bytes = canonical_jcs_bytes(right)
        .map_err(|error| ClaimError::new(ClaimErrorKind::Canonicalization, error.to_string()))?;
    Ok(left_bytes == right_bytes)
}

fn claim_ceiling(predicate: ClaimPredicate) -> Vec<String> {
    let mut ceiling = vec![
        "A supported result is scoped to the exact subject, operation, environment fields and Evidence IDs in this derivation.".to_owned(),
        "This evaluator verifies what a REA Evidence record reports; it does not prove the underlying real-world proposition is true.".to_owned(),
        "Evidence ID verification is a semantic commitment, not producer authentication or trusted-time evidence.".to_owned(),
    ];
    if predicate == ClaimPredicate::ProviderReportedObservation {
        ceiling.push(
            "Exact normalized_result equality establishes a recorded provider observation only; it does not establish artifact semantics, completeness, authenticity or safety.".to_owned(),
        );
    } else {
        ceiling.push(
            "No predicate-specific verifier is registered for this claim type in evaluator v0; no positive semantic conclusion is permitted.".to_owned(),
        );
    }
    ceiling
}

#[derive(Serialize)]
struct EvaluationBody<'a> {
    schema: &'a str,
    evaluator_version: &'a str,
    rule_id: &'a str,
    claim_id: &'a str,
    claim_request_sha256: &'a str,
    predicate: ClaimPredicate,
    disposition: ClaimDisposition,
    scope: &'a ClaimScope,
    asserted_value: &'a Value,
    source_predicate_type: &'a Option<String>,
    requested_evidence_ids: &'a [String],
    assumptions: &'a [String],
    evidence_assessments: &'a [EvidenceAssessment],
    missing_evidence_ids: &'a [String],
    unmet_obligations: &'a [String],
    falsification_condition: &'a str,
    claim_ceiling: &'a [String],
}

/// Evaluate one typed claim against REA Evidence records.
///
/// All supplied Evidence values are re-verified here; callers cannot make an
/// arbitrary VerifiedEvidence struct trustworthy by constructing it manually.
/// V0 only supports the narrow proposition that the selected Evidence records
/// report an exact normalized_result for an exact represented scope.
pub fn evaluate_claim(
    claim: &ClaimRequest,
    supplied_evidence: &[VerifiedEvidence],
) -> Result<ClaimEvaluation, ClaimError> {
    validate_claim(claim)?;
    if supplied_evidence.len() > claim.evidence_ids.len() {
        return Err(ClaimError::new(
            ClaimErrorKind::UnexpectedEvidence,
            "supplied Evidence set contains more records than the claim references",
        ));
    }

    let mut verified_by_id = HashMap::<String, &VerifiedEvidence>::new();
    for supplied in supplied_evidence {
        let verified = verify_evidence_value(&supplied.value).map_err(|error| {
            ClaimError::new(
                ClaimErrorKind::InvalidEvidence,
                format!(
                    "Evidence {} failed re-verification: {error}",
                    supplied.evidence_id
                ),
            )
        })?;
        if verified.evidence_id != supplied.evidence_id {
            return Err(ClaimError::new(
                ClaimErrorKind::InvalidEvidence,
                "supplied VerifiedEvidence identifier does not match its verified value",
            ));
        }
        let verified_id = verified.evidence_id.clone();
        drop(verified);
        if !claim.evidence_ids.contains(&verified_id) {
            return Err(ClaimError::new(
                ClaimErrorKind::UnexpectedEvidence,
                format!("supplied Evidence {verified_id} is not referenced by the claim"),
            ));
        }
        // Retain a reference to the caller's already-verified value rather
        // than a second full Evidence clone for the duration of evaluation.
        if verified_by_id.insert(verified_id, supplied).is_some() {
            return Err(ClaimError::new(
                ClaimErrorKind::DuplicateEvidence,
                "duplicate Evidence ID supplied to claim evaluator",
            ));
        }
    }

    let mut assessments = Vec::new();
    let mut missing = Vec::new();
    let mut unmet = Vec::new();
    let mut supporting = 0usize;
    let mut contradicting = 0usize;
    let mut inapplicable = 0usize;
    let mut unverifiable_scope = 0usize;

    if !claim.predicate.has_v0_evaluator() {
        unmet.push(format!(
            "no predicate-specific verifier is registered for {:?} in {}",
            claim.predicate, CLAIM_EVALUATOR_VERSION
        ));
    }
    if !claim.assumptions.is_empty() {
        unmet.push(
            "claim assumptions are recorded but evaluator v0 has no assumption verifier".to_owned(),
        );
    }

    for evidence_id in &claim.evidence_ids {
        let Some(evidence) = verified_by_id.get(evidence_id) else {
            missing.push(evidence_id.clone());
            continue;
        };

        let ProviderFields {
            provider_id,
            provider_name,
            provider_version,
            confidence,
            authority,
        } = provider_fields(&evidence.value);

        if !claim.predicate.has_v0_evaluator() || !claim.assumptions.is_empty() {
            assessments.push(EvidenceAssessment {
                evidence_id: evidence_id.clone(),
                kind: EvidenceAssessmentKind::NotEvaluated,
                provider_id,
                provider_name,
                provider_version,
                confidence,
                authority,
                reason: "no authorized inference rule evaluates this premise for the requested predicate".to_owned(),
            });
            continue;
        }

        let (scope_result, scope_reason) = assess_scope(claim, &evidence.value);
        match scope_result {
            ScopeResult::Inapplicable => {
                inapplicable += 1;
                assessments.push(EvidenceAssessment {
                    evidence_id: evidence_id.clone(),
                    kind: EvidenceAssessmentKind::Inapplicable,
                    provider_id,
                    provider_name,
                    provider_version,
                    confidence,
                    authority,
                    reason: scope_reason,
                });
                continue;
            }
            ScopeResult::Unverifiable => {
                unverifiable_scope += 1;
                assessments.push(EvidenceAssessment {
                    evidence_id: evidence_id.clone(),
                    kind: EvidenceAssessmentKind::ScopeUnverifiable,
                    provider_id,
                    provider_name,
                    provider_version,
                    confidence,
                    authority,
                    reason: scope_reason,
                });
                continue;
            }
            ScopeResult::Match => {}
        }

        let expected_predicate = claim
            .source_predicate_type
            .as_deref()
            .expect("validated provider observation predicate");
        let observed_predicate = optional_string(&evidence.value, &["predicate_type"]);
        if observed_predicate.as_deref() != Some(expected_predicate) {
            inapplicable += 1;
            assessments.push(EvidenceAssessment {
                evidence_id: evidence_id.clone(),
                kind: EvidenceAssessmentKind::Inapplicable,
                provider_id,
                provider_name,
                provider_version,
                confidence,
                authority,
                reason: "source predicate_type does not match the requested provider observation"
                    .to_owned(),
            });
            continue;
        }

        let observation_matches = evidence
            .value
            .get("normalized_result")
            .map(|observed| canonical_json_equal(observed, &claim.asserted_value))
            .transpose()?
            .unwrap_or(false);
        if observation_matches {
            supporting += 1;
            assessments.push(EvidenceAssessment {
                evidence_id: evidence_id.clone(),
                kind: EvidenceAssessmentKind::Supports,
                provider_id,
                provider_name,
                provider_version,
                confidence,
                authority,
                reason: "normalized_result matches asserted_value under RFC 8785 canonical JSON semantics".to_owned(),
            });
        } else {
            contradicting += 1;
            assessments.push(EvidenceAssessment {
                evidence_id: evidence_id.clone(),
                kind: EvidenceAssessmentKind::Contradicts,
                provider_id,
                provider_name,
                provider_version,
                confidence,
                authority,
                reason: "normalized_result differs from asserted_value".to_owned(),
            });
        }
    }

    let disposition = if !claim.predicate.has_v0_evaluator() || !claim.assumptions.is_empty() {
        ClaimDisposition::Unverified
    } else if supporting > 0 && contradicting > 0 {
        ClaimDisposition::Conflicted
    } else if unverifiable_scope > 0 || !missing.is_empty() {
        if supporting == 0
            && contradicting == 0
            && inapplicable > 0
            && unverifiable_scope == 0
            && missing.is_empty()
        {
            ClaimDisposition::Inapplicable
        } else {
            ClaimDisposition::Underdetermined
        }
    } else if supporting > 0 {
        ClaimDisposition::SupportedWithinScope
    } else if contradicting > 0 {
        ClaimDisposition::RefutedWithinScope
    } else if inapplicable > 0 {
        ClaimDisposition::Inapplicable
    } else {
        ClaimDisposition::Underdetermined
    };

    if claim.evidence_ids.is_empty() {
        unmet.push("no Evidence IDs were supplied".to_owned());
    }
    if unverifiable_scope > 0 {
        unmet.push(
            "one or more scope dimensions are not represented by the supplied REA Evidence records"
                .to_owned(),
        );
    }
    if !missing.is_empty() {
        unmet.push("one or more referenced Evidence IDs are missing".to_owned());
    }
    if supporting == 0 && contradicting == 0 && inapplicable == 0 && unverifiable_scope == 0 {
        unmet.push(
            "no applicable premise established or contradicted the asserted value".to_owned(),
        );
    }

    let rule_id = if claim.predicate.has_v0_evaluator() && claim.assumptions.is_empty() {
        "provider-reported-observation/exact-normalized-result/v0"
    } else {
        "no-registered-predicate-rule/v0"
    };

    let claim_value = serde_json::to_value(claim)
        .map_err(|error| ClaimError::new(ClaimErrorKind::Canonicalization, error.to_string()))?;
    let claim_bytes = canonical_jcs_bytes(&claim_value)
        .map_err(|error| ClaimError::new(ClaimErrorKind::Canonicalization, error.to_string()))?;
    let claim_request_sha256 = sha256_hex(&claim_bytes);

    let claim_ceiling = claim_ceiling(claim.predicate);
    let body = EvaluationBody {
        schema: CLAIM_EVALUATION_SCHEMA,
        evaluator_version: CLAIM_EVALUATOR_VERSION,
        rule_id,
        claim_id: &claim.claim_id,
        claim_request_sha256: &claim_request_sha256,
        predicate: claim.predicate,
        disposition,
        scope: &claim.scope,
        asserted_value: &claim.asserted_value,
        source_predicate_type: &claim.source_predicate_type,
        requested_evidence_ids: &claim.evidence_ids,
        assumptions: &claim.assumptions,
        evidence_assessments: &assessments,
        missing_evidence_ids: &missing,
        unmet_obligations: &unmet,
        falsification_condition: &claim.falsification_condition,
        claim_ceiling: &claim_ceiling,
    };
    let body_value = serde_json::to_value(body)
        .map_err(|error| ClaimError::new(ClaimErrorKind::Canonicalization, error.to_string()))?;
    let canonical = canonical_jcs_bytes(&body_value)
        .map_err(|error| ClaimError::new(ClaimErrorKind::Canonicalization, error.to_string()))?;
    let derivation_sha256 = sha256_hex(&canonical);

    Ok(ClaimEvaluation {
        schema: CLAIM_EVALUATION_SCHEMA.to_owned(),
        evaluator_version: CLAIM_EVALUATOR_VERSION.to_owned(),
        rule_id: rule_id.to_owned(),
        claim_id: claim.claim_id.clone(),
        claim_request_sha256,
        predicate: claim.predicate,
        disposition,
        scope: claim.scope.clone(),
        asserted_value: claim.asserted_value.clone(),
        source_predicate_type: claim.source_predicate_type.clone(),
        requested_evidence_ids: claim.evidence_ids.clone(),
        assumptions: claim.assumptions.clone(),
        evidence_assessments: assessments,
        missing_evidence_ids: missing,
        unmet_obligations: unmet,
        falsification_condition: claim.falsification_condition.clone(),
        claim_ceiling,
        derivation_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn verified_fixture() -> VerifiedEvidence {
        let value: Value = serde_json::from_str(include_str!(
            "../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
        ))
        .expect("REA Evidence fixture parses");
        verify_evidence_value(&value).expect("REA Evidence fixture verifies")
    }

    fn claim(predicate: ClaimPredicate, evidence_ids: Vec<String>) -> ClaimRequest {
        ClaimRequest {
            schema: CLAIM_SCHEMA.to_owned(),
            claim_id: "claim-0001".to_owned(),
            predicate,
            scope: ClaimScope {
                subject_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_owned(),
                operation: "inspect_artifact".to_owned(),
                platform: None,
                platform_version: None,
                environment_id: None,
            },
            asserted_value: json!({
                "entries": [{"path": "classes.dex", "size": 42}],
                "status": "complete"
            }),
            source_predicate_type: if predicate == ClaimPredicate::ProviderReportedObservation {
                Some("rea.analysis".to_owned())
            } else {
                None
            },
            evidence_ids,
            assumptions: Vec::new(),
            falsification_condition:
                "A same-scope verified Evidence record has a different normalized_result".to_owned(),
        }
    }

    #[test]
    fn exact_provider_report_is_supported_but_not_promoted_to_artifact_truth() {
        let evidence = verified_fixture();
        let claim = claim(
            ClaimPredicate::ProviderReportedObservation,
            vec![evidence.evidence_id.clone()],
        );
        let result = evaluate_claim(&claim, &[evidence]).unwrap();

        assert_eq!(result.disposition, ClaimDisposition::SupportedWithinScope);
        assert!(
            result
                .claim_ceiling
                .iter()
                .any(|line| line.contains("does not prove the underlying real-world proposition"))
        );
        assert_eq!(result.evidence_assessments.len(), 1);
        assert_eq!(
            result.evidence_assessments[0].kind,
            EvidenceAssessmentKind::Supports
        );
        assert!(valid_sha256(&result.derivation_sha256));
    }

    #[test]
    fn normalized_result_comparison_uses_jcs_number_semantics() {
        let evidence = verified_fixture();
        let mut numeric_claim = claim(
            ClaimPredicate::ProviderReportedObservation,
            vec![evidence.evidence_id.clone()],
        );
        numeric_claim.asserted_value["entries"][0]["size"] = json!(42.0);
        let result = evaluate_claim(&numeric_claim, std::slice::from_ref(&evidence)).unwrap();
        assert_eq!(result.disposition, ClaimDisposition::SupportedWithinScope);

        numeric_claim.asserted_value["entries"][0]["size"] = json!(42.0001);
        let changed = evaluate_claim(&numeric_claim, &[evidence]).unwrap();
        assert_eq!(changed.disposition, ClaimDisposition::RefutedWithinScope);
    }

    #[test]
    fn differing_same_scope_report_refutes_and_conflicting_reports_are_preserved() {
        let first = verified_fixture();
        let mut second_value = first.value.clone();
        second_value["provider"]["id"] = json!("independent-looking-provider");
        second_value["provider"]["name"] = json!("Another Provider");
        second_value["analysis_profile"]["provider"] = second_value["provider"].clone();
        second_value["analysis_profile"]["digest"] = json!(
            super::super::analysis_profile_digest(&second_value["analysis_profile"]).unwrap()
        );
        second_value["normalized_result"]["status"] = json!("partial");
        second_value["evidence_id"] =
            json!(super::super::compute_evidence_id(&second_value).unwrap());
        let second = verify_evidence_value(&second_value).unwrap();

        let mut contradicted = claim(
            ClaimPredicate::ProviderReportedObservation,
            vec![second.evidence_id.clone()],
        );
        contradicted.asserted_value["status"] = json!("complete");
        let result = evaluate_claim(&contradicted, std::slice::from_ref(&second)).unwrap();
        assert_eq!(result.disposition, ClaimDisposition::RefutedWithinScope);

        let conflict = claim(
            ClaimPredicate::ProviderReportedObservation,
            vec![first.evidence_id.clone(), second.evidence_id.clone()],
        );
        let result = evaluate_claim(&conflict, &[first, second]).unwrap();
        assert_eq!(result.disposition, ClaimDisposition::Conflicted);
        assert_eq!(
            result
                .evidence_assessments
                .iter()
                .filter(|item| item.kind == EvidenceAssessmentKind::Supports)
                .count(),
            1
        );
        assert_eq!(
            result
                .evidence_assessments
                .iter()
                .filter(|item| item.kind == EvidenceAssessmentKind::Contradicts)
                .count(),
            1
        );
    }

    #[test]
    fn wrong_subject_and_operation_are_not_applied_to_claim_scope() {
        let evidence = verified_fixture();
        let mut wrong_subject_claim = claim(
            ClaimPredicate::ProviderReportedObservation,
            vec![evidence.evidence_id.clone()],
        );
        wrong_subject_claim.scope.subject_sha256 =
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned();
        let result = evaluate_claim(&wrong_subject_claim, std::slice::from_ref(&evidence)).unwrap();
        assert_eq!(result.disposition, ClaimDisposition::Inapplicable);

        let mut wrong_operation_claim = claim(
            ClaimPredicate::ProviderReportedObservation,
            vec![evidence.evidence_id.clone()],
        );
        wrong_operation_claim.scope.operation = "verify_signature".to_owned();
        let result = evaluate_claim(&wrong_operation_claim, &[evidence]).unwrap();
        assert_eq!(result.disposition, ClaimDisposition::Inapplicable);
    }

    #[test]
    fn evidence_without_platform_version_cannot_support_version_scoped_claim() {
        let evidence = verified_fixture();
        let mut claim = claim(
            ClaimPredicate::ProviderReportedObservation,
            vec![evidence.evidence_id.clone()],
        );
        claim.scope.platform_version = Some("Android 17 / API 37".to_owned());
        let result = evaluate_claim(&claim, &[evidence]).unwrap();
        assert_eq!(result.disposition, ClaimDisposition::Underdetermined);
        assert!(
            result
                .evidence_assessments
                .iter()
                .any(|item| item.reason.contains("platform_version"))
        );
    }

    #[test]
    fn unimplemented_semantic_predicates_fail_closed() {
        let evidence = verified_fixture();
        let claim = claim(
            ClaimPredicate::ProducerAuthenticity,
            vec![evidence.evidence_id.clone()],
        );
        let result = evaluate_claim(&claim, &[evidence]).unwrap();
        assert_eq!(result.disposition, ClaimDisposition::Unverified);
        assert!(
            result
                .unmet_obligations
                .iter()
                .any(|item| item.contains("no predicate-specific verifier"))
        );
        assert!(
            result
                .evidence_assessments
                .iter()
                .all(|item| item.kind == EvidenceAssessmentKind::NotEvaluated)
        );
    }

    #[test]
    fn missing_evidence_and_assumptions_never_create_positive_support() {
        let missing_id = format!("ev_{}", "b".repeat(64));
        let mut claim = claim(
            ClaimPredicate::ProviderReportedObservation,
            vec![missing_id.clone()],
        );
        let empty = evaluate_claim(&claim, &[]).unwrap();
        assert_eq!(empty.disposition, ClaimDisposition::Underdetermined);
        assert_eq!(empty.missing_evidence_ids, vec![missing_id]);

        let evidence = verified_fixture();
        claim.evidence_ids = vec![evidence.evidence_id.clone()];
        claim
            .assumptions
            .push("the scanner database is complete".to_owned());
        let result = evaluate_claim(&claim, &[evidence]).unwrap();
        assert_eq!(result.disposition, ClaimDisposition::Unverified);
    }

    #[test]
    fn record_level_support_preserves_inferred_confidence_and_authority() {
        let original = verified_fixture();
        let mut value = original.value.clone();
        value["confidence"] = json!("inferred");
        value["authority"] = json!("analyst-inference");
        value["evidence_id"] = json!(super::super::compute_evidence_id(&value).unwrap());
        let evidence = verify_evidence_value(&value).unwrap();

        let claim = claim(
            ClaimPredicate::ProviderReportedObservation,
            vec![evidence.evidence_id.clone()],
        );
        let result = evaluate_claim(&claim, &[evidence]).unwrap();

        // This disposition qualifies only the exact contents of the record.
        // It does not upgrade an inferred analyst report to an observation
        // about artifact truth or to an independently trusted authority.
        assert_eq!(result.disposition, ClaimDisposition::SupportedWithinScope);
        assert_eq!(
            result.evidence_assessments[0].confidence.as_deref(),
            Some("inferred")
        );
        assert_eq!(
            result.evidence_assessments[0].authority.as_deref(),
            Some("analyst-inference")
        );
        assert!(
            result.claim_ceiling.iter().any(|line| {
                line.contains("does not prove the underlying real-world proposition")
            })
        );
    }

    #[test]
    fn forged_verified_evidence_metadata_is_rechecked() {
        let mut evidence = verified_fixture();
        evidence.evidence_id = format!("ev_{}", "f".repeat(64));
        let claim = claim(
            ClaimPredicate::ProviderReportedObservation,
            vec![evidence.evidence_id.clone()],
        );
        let error = evaluate_claim(&claim, &[evidence]).unwrap_err();
        assert_eq!(error.kind, ClaimErrorKind::InvalidEvidence);
    }

    #[test]
    fn raw_claim_parser_rejects_duplicate_keys_even_inside_asserted_value() {
        let raw = br#"{
            "schema":"luminous.rea.claim.v0",
            "schema":"luminous.rea.claim.v0",
            "claim_id":"c1",
            "predicate":"provider_reported_observation",
            "scope":{"subject_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","operation":"inspect_artifact"},
            "asserted_value":{"status":"complete","status":"partial"},
            "source_predicate_type":"rea.analysis",
            "evidence_ids":[],
            "assumptions":[],
            "falsification_condition":"a different normalized result"
        }"#;
        let error = parse_claim_json(raw).unwrap_err();
        assert_eq!(error.kind, ClaimErrorKind::InvalidJson);
        assert!(error.message.contains("duplicate"));
    }

    #[test]
    fn derivation_commitment_is_deterministic_and_changes_with_evidence_scope() {
        let evidence = verified_fixture();
        let claim = claim(
            ClaimPredicate::ProviderReportedObservation,
            vec![evidence.evidence_id.clone()],
        );
        let first = evaluate_claim(&claim, std::slice::from_ref(&evidence)).unwrap();
        let second = evaluate_claim(&claim, std::slice::from_ref(&evidence)).unwrap();
        assert_eq!(first.derivation_sha256, second.derivation_sha256);

        let mut changed = claim.clone();
        changed.scope.operation = "other_operation".to_owned();
        let different = evaluate_claim(&changed, std::slice::from_ref(&evidence)).unwrap();
        assert_ne!(first.derivation_sha256, different.derivation_sha256);

        let mut different_assertion = claim;
        different_assertion.asserted_value["status"] = json!("a different nonmatching value");
        let changed_assertion = evaluate_claim(&different_assertion, &[evidence]).unwrap();
        assert_ne!(first.derivation_sha256, changed_assertion.derivation_sha256);
    }
    #[test]
    fn unreferenced_evidence_is_rejected_instead_of_silently_ignored() {
        let evidence = verified_fixture();
        let claim = claim(ClaimPredicate::ProviderReportedObservation, Vec::new());
        let error = evaluate_claim(&claim, &[evidence]).unwrap_err();
        assert_eq!(error.kind, ClaimErrorKind::UnexpectedEvidence);
    }

    #[test]
    fn claim_evidence_count_is_bounded_before_evaluation() {
        let mut claim = claim(ClaimPredicate::ProviderReportedObservation, Vec::new());
        let id = format!("ev_{}", "a".repeat(64));
        claim.evidence_ids = vec![id; MAX_CLAIM_EVIDENCE_RECORDS + 1];
        let error = evaluate_claim(&claim, &[]).unwrap_err();
        assert_eq!(error.kind, ClaimErrorKind::ResourceLimit);
    }
}
