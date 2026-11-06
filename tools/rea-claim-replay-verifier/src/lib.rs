//! Independent replay path for REA claim-evaluation v0.
//!
//! This crate deliberately does not import or call the production evaluate_claim
//! function or its disposition/scope helpers. It independently parses the
//! serialized request and result, verifies every supplied Evidence record,
//! re-applies the single v0 rule, and recomputes the derivation digest.
//!
//! The shared canonicalization and Evidence identity primitives are RFC 8785 /
//! REA contract adapters. This crate does not claim independent cryptographic
//! primitive implementations or prove the underlying artifact proposition.

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize as SerdeDeserialize, Serialize};
use serde_json::{Map, Number, Value};
use std::collections::{HashMap, HashSet};
use std::fmt;
use symthaea_rea_compat::{
    EvidenceErrorKind, VerifiedEvidence, canonical_jcs_bytes, sha256_hex, verify_evidence_json,
};

pub const CLAIM_SCHEMA: &str = "luminous.rea.claim.v0";
pub const RESULT_SCHEMA: &str = "luminous.rea.claim-evaluation.v0";
pub const EVALUATOR_VERSION: &str = "rea-claim-evaluator/v0";
pub const MAX_CLAIM_JSON_BYTES: usize = 1024 * 1024;
pub const MAX_RESULT_JSON_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_TOTAL_EVIDENCE_JSON_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, SerdeDeserialize)]
#[serde(rename_all = "snake_case")]
pub enum Predicate {
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

impl Predicate {
    fn implemented_in_v0(self) -> bool {
        matches!(self, Self::ProviderReportedObservation)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, SerdeDeserialize)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    SupportedWithinScope,
    RefutedWithinScope,
    Underdetermined,
    Conflicted,
    Inapplicable,
    Unverified,
}

#[derive(Debug, Clone, PartialEq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct Scope {
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
    pub predicate: Predicate,
    pub scope: Scope,
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
pub enum AssessmentKind {
    Supports,
    Contradicts,
    Inapplicable,
    ScopeUnverifiable,
    NotEvaluated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct Assessment {
    pub evidence_id: String,
    pub kind: AssessmentKind,
    pub provider_id: Option<String>,
    pub provider_name: Option<String>,
    pub provider_version: Option<String>,
    pub confidence: Option<String>,
    pub authority: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimEvaluation {
    pub schema: String,
    pub evaluator_version: String,
    pub rule_id: String,
    pub claim_id: String,
    pub claim_request_sha256: String,
    pub predicate: Predicate,
    pub disposition: Disposition,
    pub scope: Scope,
    pub asserted_value: Value,
    pub source_predicate_type: Option<String>,
    pub requested_evidence_ids: Vec<String>,
    pub assumptions: Vec<String>,
    pub evidence_assessments: Vec<Assessment>,
    pub missing_evidence_ids: Vec<String>,
    pub unmet_obligations: Vec<String>,
    pub falsification_condition: String,
    pub claim_ceiling: Vec<String>,
    pub derivation_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayOutcome {
    ReplayMatch,
    ReplayMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayReceipt {
    pub outcome: ReplayOutcome,
    pub expected_disposition: Disposition,
    pub reported_disposition: Disposition,
    pub claim_request_sha256: String,
    pub recomputed_derivation_sha256: String,
    pub reported_derivation_sha256: String,
    pub evidence_count: usize,
    pub missing_evidence_count: usize,
    pub rule_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayErrorKind {
    InvalidJson,
    InputTooLarge,
    InvalidClaim,
    InvalidResult,
    InvalidEvidence,
    DuplicateEvidence,
    UnexpectedEvidence,
    Canonicalization,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayError {
    pub kind: ReplayErrorKind,
    pub message: String,
}

impl ReplayError {
    fn new(kind: ReplayErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for ReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for ReplayError {}

struct UniqueValue(Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueVisitor)
    }
}

struct UniqueVisitor;

impl<'de> Visitor<'de> for UniqueVisitor {
    type Value = UniqueValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON with unique property names at every nesting level")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(|number| UniqueValue(Value::Number(number)))
            .ok_or_else(|| E::custom("JSON numbers must be finite"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value)))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        UniqueValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(item) = sequence.next_element::<UniqueValue>()? {
            values.push(item.0);
        }
        Ok(UniqueValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        let mut seen = HashSet::new();
        while let Some((key, item)) = object.next_entry::<String, UniqueValue>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!(
                    "duplicate JSON object key '{key}'"
                )));
            }
            values.insert(key, item.0);
        }
        Ok(UniqueValue(Value::Object(values)))
    }
}

fn parse_unique_json(bytes: &[u8], max_bytes: usize, context: &str) -> Result<Value, ReplayError> {
    if bytes.len() > max_bytes {
        return Err(ReplayError::new(
            ReplayErrorKind::InputTooLarge,
            format!("{context} is {} bytes; limit is {max_bytes}", bytes.len()),
        ));
    }

    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let parsed = UniqueValue::deserialize(&mut deserializer).map_err(|error| {
        ReplayError::new(
            ReplayErrorKind::InvalidJson,
            format!("{context} parsing failed: {error}"),
        )
    })?;
    deserializer.end().map_err(|error| {
        ReplayError::new(
            ReplayErrorKind::InvalidJson,
            format!("{context} has trailing data: {error}"),
        )
    })?;
    Ok(parsed.0)
}

fn require_keys(value: &Value, keys: &[&str], context: &str) -> Result<(), ReplayError> {
    let object = value.as_object().ok_or_else(|| {
        ReplayError::new(
            ReplayErrorKind::InvalidJson,
            format!("{context} must be a JSON object"),
        )
    })?;
    for key in keys {
        if !object.contains_key(*key) {
            return Err(ReplayError::new(
                ReplayErrorKind::InvalidJson,
                format!("{context}.{key} is required"),
            ));
        }
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_evidence_id(value: &str) -> bool {
    value.strip_prefix("ev_").is_some_and(is_sha256)
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn parse_claim(bytes: &[u8]) -> Result<ClaimRequest, ReplayError> {
    let value = parse_unique_json(bytes, MAX_CLAIM_JSON_BYTES, "claim request")?;
    require_keys(
        &value,
        &[
            "schema",
            "claim_id",
            "predicate",
            "scope",
            "asserted_value",
            "source_predicate_type",
            "evidence_ids",
            "assumptions",
            "falsification_condition",
        ],
        "claim",
    )?;
    require_keys(
        value.get("scope").unwrap_or(&Value::Null),
        &["subject_sha256", "operation"],
        "claim.scope",
    )?;
    let claim: ClaimRequest = serde_json::from_value(value).map_err(|error| {
        ReplayError::new(
            ReplayErrorKind::InvalidClaim,
            format!("claim schema validation failed: {error}"),
        )
    })?;
    validate_claim(&claim)?;
    Ok(claim)
}

fn validate_claim(claim: &ClaimRequest) -> Result<(), ReplayError> {
    let invalid = |message: &str| ReplayError::new(ReplayErrorKind::InvalidClaim, message);
    if claim.schema != CLAIM_SCHEMA {
        return Err(invalid("unsupported claim request schema"));
    }
    if !nonempty(&claim.claim_id) {
        return Err(invalid("claim_id must be non-empty"));
    }
    if !is_sha256(&claim.scope.subject_sha256) {
        return Err(invalid("scope.subject_sha256 must be lowercase SHA-256"));
    }
    if !nonempty(&claim.scope.operation) {
        return Err(invalid("scope.operation must be non-empty"));
    }
    for (label, value) in [
        ("platform", claim.scope.platform.as_deref()),
        ("platform_version", claim.scope.platform_version.as_deref()),
        ("environment_id", claim.scope.environment_id.as_deref()),
    ] {
        if value.is_some_and(|value| !nonempty(value)) {
            return Err(invalid(&format!(
                "scope.{label} must be non-empty when present"
            )));
        }
    }
    if !nonempty(&claim.falsification_condition) {
        return Err(invalid("falsification_condition must be non-empty"));
    }
    if claim.assumptions.iter().any(|item| !nonempty(item)) {
        return Err(invalid("assumptions must contain non-empty strings"));
    }
    if claim.predicate == Predicate::ProviderReportedObservation {
        if !claim.source_predicate_type.as_deref().is_some_and(nonempty) {
            return Err(invalid(
                "provider_reported_observation requires source_predicate_type",
            ));
        }
    } else if claim.source_predicate_type.is_some() {
        return Err(invalid(
            "source_predicate_type is not valid for this predicate",
        ));
    }
    let mut seen = HashSet::new();
    for id in &claim.evidence_ids {
        if !is_evidence_id(id) {
            return Err(invalid("evidence_ids contains a malformed Evidence ID"));
        }
        if !seen.insert(id) {
            return Err(invalid("evidence_ids contains duplicate IDs"));
        }
    }
    Ok(())
}

fn parse_result(bytes: &[u8]) -> Result<(ClaimEvaluation, Value), ReplayError> {
    let value = parse_unique_json(bytes, MAX_RESULT_JSON_BYTES, "claim evaluation result")?;
    require_keys(
        &value,
        &[
            "schema",
            "evaluator_version",
            "rule_id",
            "claim_id",
            "claim_request_sha256",
            "predicate",
            "disposition",
            "scope",
            "asserted_value",
            "source_predicate_type",
            "requested_evidence_ids",
            "assumptions",
            "evidence_assessments",
            "missing_evidence_ids",
            "unmet_obligations",
            "falsification_condition",
            "claim_ceiling",
            "derivation_sha256",
        ],
        "result",
    )?;
    require_keys(
        value.get("scope").unwrap_or(&Value::Null),
        &[
            "subject_sha256",
            "operation",
            "platform",
            "platform_version",
            "environment_id",
        ],
        "result.scope",
    )?;
    let assessments = value
        .get("evidence_assessments")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ReplayError::new(
                ReplayErrorKind::InvalidResult,
                "result.evidence_assessments must be an array",
            )
        })?;
    for (index, assessment) in assessments.iter().enumerate() {
        require_keys(
            assessment,
            &[
                "evidence_id",
                "kind",
                "provider_id",
                "provider_name",
                "provider_version",
                "confidence",
                "authority",
                "reason",
            ],
            &format!("result.evidence_assessments[{index}]"),
        )?;
    }

    let result: ClaimEvaluation = serde_json::from_value(value).map_err(|error| {
        ReplayError::new(
            ReplayErrorKind::InvalidResult,
            format!("claim evaluation result schema validation failed: {error}"),
        )
    })?;
    if result.schema != RESULT_SCHEMA {
        return Err(ReplayError::new(
            ReplayErrorKind::InvalidResult,
            "unsupported claim evaluation result schema",
        ));
    }
    if result.evaluator_version != EVALUATOR_VERSION {
        return Err(ReplayError::new(
            ReplayErrorKind::InvalidResult,
            "unsupported evaluator version",
        ));
    }
    if !is_sha256(&result.claim_request_sha256) || !is_sha256(&result.derivation_sha256) {
        return Err(ReplayError::new(
            ReplayErrorKind::InvalidResult,
            "result digests must be lowercase SHA-256",
        ));
    }
    Ok((result, value))
}

pub fn replay_claim_evaluation(
    claim_json: &[u8],
    evidence_json: &[&[u8]],
    result_json: &[u8],
) -> Result<ReplayReceipt, ReplayError> {
    let claim = parse_claim(claim_json)?;
    let (reported, reported_value) = parse_result(result_json)?;

    let mut total_bytes = 0usize;
    let mut evidence_by_id = HashMap::<String, VerifiedEvidence>::new();
    for raw in evidence_json {
        total_bytes = total_bytes.checked_add(raw.len()).ok_or_else(|| {
            ReplayError::new(
                ReplayErrorKind::InputTooLarge,
                "total Evidence input size overflow",
            )
        })?;
        if total_bytes > MAX_TOTAL_EVIDENCE_JSON_BYTES {
            return Err(ReplayError::new(
                ReplayErrorKind::InputTooLarge,
                "total Evidence input exceeds replay verifier limit",
            ));
        }

        let evidence = verify_evidence_json(raw).map_err(|error| {
            let kind = match error.kind {
                EvidenceErrorKind::InputTooLarge => ReplayErrorKind::InputTooLarge,
                _ => ReplayErrorKind::InvalidEvidence,
            };
            ReplayError::new(
                kind,
                format!("REA Evidence verification failed: {error}"),
            )
        })?;
        if !claim.evidence_ids.contains(&evidence.evidence_id) {
            return Err(ReplayError::new(
                ReplayErrorKind::UnexpectedEvidence,
                format!(
                    "supplied Evidence {} is not referenced by the exact claim request",
                    evidence.evidence_id
                ),
            ));
        }
        if evidence_by_id
            .insert(evidence.evidence_id.clone(), evidence)
            .is_some()
        {
            return Err(ReplayError::new(
                ReplayErrorKind::DuplicateEvidence,
                "duplicate verified Evidence record supplied to replay",
            ));
        }
    }

    let expected = derive_reference_result(&claim, &evidence_by_id)?;
    let expected_value = serde_json::to_value(&expected).map_err(|error| {
        ReplayError::new(ReplayErrorKind::Canonicalization, error.to_string())
    })?;
    let expected_canonical = canonical_jcs_bytes(&expected_value).map_err(|error| {
        ReplayError::new(ReplayErrorKind::Canonicalization, error.to_string())
    })?;
    let reported_canonical = canonical_jcs_bytes(&reported_value).map_err(|error| {
        ReplayError::new(ReplayErrorKind::Canonicalization, error.to_string())
    })?;
    let outcome = if expected_canonical == reported_canonical {
        ReplayOutcome::ReplayMatch
    } else {
        ReplayOutcome::ReplayMismatch
    };

    Ok(ReplayReceipt {
        outcome,
        expected_disposition: expected.disposition,
        reported_disposition: reported.disposition,
        claim_request_sha256: expected.claim_request_sha256,
        recomputed_derivation_sha256: expected.derivation_sha256,
        reported_derivation_sha256: reported.derivation_sha256,
        evidence_count: evidence_by_id.len(),
        missing_evidence_count: expected.missing_evidence_ids.len(),
        rule_id: expected.rule_id,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScopeDisposition {
    Match,
    Inapplicable,
    Unverifiable,
}

fn compare_optional_scope(requested: Option<&str>, observed: Option<&str>) -> ScopeDisposition {
    match (requested, observed) {
        (Some(left), Some(right)) if left == right => ScopeDisposition::Match,
        (Some(_), Some(_)) => ScopeDisposition::Inapplicable,
        (None, None) => ScopeDisposition::Match,
        _ => ScopeDisposition::Unverifiable,
    }
}

fn string_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_str()
}

fn scope_assessment(claim: &ClaimRequest, evidence: &Value) -> (ScopeDisposition, String) {
    match string_at(evidence, &["subject", "digest", "sha256"]) {
        Some(digest) if digest == claim.scope.subject_sha256 => {}
        Some(_) => {
            return (
                ScopeDisposition::Inapplicable,
                "subject digest does not match claim scope".to_owned(),
            );
        }
        None => {
            return (
                ScopeDisposition::Unverifiable,
                "Evidence has no subject digest to bind this claim".to_owned(),
            );
        }
    }

    if string_at(evidence, &["operation"]) != Some(claim.scope.operation.as_str()) {
        return (
            ScopeDisposition::Inapplicable,
            "operation does not match claim scope".to_owned(),
        );
    }

    if claim.scope.platform_version.is_some() {
        return (
            ScopeDisposition::Unverifiable,
            "REA Evidence v1 does not encode platform_version".to_owned(),
        );
    }

    let platform = compare_optional_scope(
        claim.scope.platform.as_deref(),
        string_at(evidence, &["environment", "platform"]),
    );
    if platform != ScopeDisposition::Match {
        return (
            platform,
            match platform {
                ScopeDisposition::Inapplicable => "platform does not match claim scope",
                ScopeDisposition::Unverifiable => {
                    "platform scope is absent or broader than the Evidence observation"
                },
                ScopeDisposition::Match => unreachable!(),
            }
            .to_owned(),
        );
    }

    let environment = compare_optional_scope(
        claim.scope.environment_id.as_deref(),
        string_at(evidence, &["environment", "id"]),
    );
    if environment != ScopeDisposition::Match {
        return (
            environment,
            match environment {
                ScopeDisposition::Inapplicable => "environment_id does not match claim scope",
                ScopeDisposition::Unverifiable => {
                    "environment identity is not recorded at the requested scope"
                },
                ScopeDisposition::Match => unreachable!(),
            }
            .to_owned(),
        );
    }

    (
        ScopeDisposition::Match,
        "exact subject, operation and represented environment scope match".to_owned(),
    )
}

fn jcs_equal(left: &Value, right: &Value) -> Result<bool, ReplayError> {
    let left = canonical_jcs_bytes(left).map_err(|error| {
        ReplayError::new(
            ReplayErrorKind::Canonicalization,
            format!("left observation canonicalization failed: {error}"),
        )
    })?;
    let right = canonical_jcs_bytes(right).map_err(|error| {
        ReplayError::new(
            ReplayErrorKind::Canonicalization,
            format!("right observation canonicalization failed: {error}"),
        )
    })?;
    Ok(left == right)
}

fn metadata(
    value: &Value,
) -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
) {
    (
        string_at(value, &["provider", "id"]).map(str::to_owned),
        string_at(value, &["provider", "name"]).map(str::to_owned),
        string_at(value, &["provider", "version"]).map(str::to_owned),
        string_at(value, &["confidence"]).map(str::to_owned),
        string_at(value, &["authority"]).map(str::to_owned),
    )
}

fn assessment(
    evidence_id: &str,
    kind: AssessmentKind,
    evidence: &Value,
    reason: impl Into<String>,
) -> Assessment {
    let (provider_id, provider_name, provider_version, confidence, authority) = metadata(evidence);
    Assessment {
        evidence_id: evidence_id.to_owned(),
        kind,
        provider_id,
        provider_name,
        provider_version,
        confidence,
        authority,
        reason: reason.into(),
    }
}

fn claim_ceiling(predicate: Predicate) -> Vec<String> {
    let mut ceiling = vec![
        "A supported result is scoped to the exact subject, operation, environment fields and Evidence IDs in this derivation.".to_owned(),
        "This evaluator verifies what a REA Evidence record reports; it does not prove the underlying real-world proposition is true.".to_owned(),
        "Evidence ID verification is a semantic commitment, not producer authentication or trusted-time evidence.".to_owned(),
    ];
    if predicate == Predicate::ProviderReportedObservation {
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

fn derive_reference_result(
    claim: &ClaimRequest,
    evidence_by_id: &HashMap<String, VerifiedEvidence>,
) -> Result<ClaimEvaluation, ReplayError> {
    let mut assessments = Vec::new();
    let mut missing = Vec::new();
    let mut unmet = Vec::new();
    let mut supports = 0usize;
    let mut contradicts = 0usize;
    let mut inapplicable = 0usize;
    let mut unverifiable = 0usize;

    if !claim.predicate.implemented_in_v0() {
        unmet.push(format!(
            "no predicate-specific verifier is registered for {:?} in {}",
            claim.predicate, EVALUATOR_VERSION
        ));
    }
    if !claim.assumptions.is_empty() {
        unmet.push(
            "claim assumptions are recorded but evaluator v0 has no assumption verifier".to_owned(),
        );
    }

    for evidence_id in &claim.evidence_ids {
        let Some(record) = evidence_by_id.get(evidence_id) else {
            missing.push(evidence_id.clone());
            continue;
        };

        if !claim.predicate.implemented_in_v0() || !claim.assumptions.is_empty() {
            assessments.push(assessment(
                evidence_id,
                AssessmentKind::NotEvaluated,
                &record.value,
                "no authorized inference rule evaluates this premise for the requested predicate",
            ));
            continue;
        }

        let (scope, reason) = scope_assessment(claim, &record.value);
        match scope {
            ScopeDisposition::Inapplicable => {
                inapplicable += 1;
                assessments.push(assessment(
                    evidence_id,
                    AssessmentKind::Inapplicable,
                    &record.value,
                    reason,
                ));
                continue;
            }
            ScopeDisposition::Unverifiable => {
                unverifiable += 1;
                assessments.push(assessment(
                    evidence_id,
                    AssessmentKind::ScopeUnverifiable,
                    &record.value,
                    reason,
                ));
                continue;
            }
            ScopeDisposition::Match => {}
        }

        if string_at(&record.value, &["predicate_type"]) != claim.source_predicate_type.as_deref() {
            inapplicable += 1;
            assessments.push(assessment(
                evidence_id,
                AssessmentKind::Inapplicable,
                &record.value,
                "source predicate_type does not match the requested provider observation",
            ));
            continue;
        }

        let normalized = record
            .value
            .get("normalized_result")
            .unwrap_or(&Value::Null);
        if jcs_equal(normalized, &claim.asserted_value)? {
            supports += 1;
            assessments.push(assessment(
                evidence_id,
                AssessmentKind::Supports,
                &record.value,
                "normalized_result matches asserted_value under RFC 8785 canonical JSON semantics",
            ));
        } else {
            contradicts += 1;
            assessments.push(assessment(
                evidence_id,
                AssessmentKind::Contradicts,
                &record.value,
                "normalized_result differs from asserted_value",
            ));
        }
    }

    let disposition = if !claim.predicate.implemented_in_v0() || !claim.assumptions.is_empty() {
        Disposition::Unverified
    } else if supports > 0 && contradicts > 0 {
        Disposition::Conflicted
    } else if unverifiable > 0 || !missing.is_empty() {
        if supports == 0
            && contradicts == 0
            && inapplicable > 0
            && unverifiable == 0
            && missing.is_empty()
        {
            Disposition::Inapplicable
        } else {
            Disposition::Underdetermined
        }
    } else if supports > 0 {
        Disposition::SupportedWithinScope
    } else if contradicts > 0 {
        Disposition::RefutedWithinScope
    } else if inapplicable > 0 {
        Disposition::Inapplicable
    } else {
        Disposition::Underdetermined
    };

    if claim.evidence_ids.is_empty() {
        unmet.push("no Evidence IDs were supplied".to_owned());
    }
    if unverifiable > 0 {
        unmet.push(
            "one or more scope dimensions are not represented by the supplied REA Evidence records"
                .to_owned(),
        );
    }
    if !missing.is_empty() {
        unmet.push("one or more referenced Evidence IDs are missing".to_owned());
    }
    if supports == 0 && contradicts == 0 && inapplicable == 0 && unverifiable == 0 {
        unmet.push(
            "no applicable premise established or contradicted the asserted value".to_owned(),
        );
    }

    let rule_id = if claim.predicate.implemented_in_v0() && claim.assumptions.is_empty() {
        "provider-reported-observation/exact-normalized-result/v0"
    } else {
        "no-registered-predicate-rule/v0"
    };

    let claim_value = serde_json::to_value(claim)
        .map_err(|error| ReplayError::new(ReplayErrorKind::Canonicalization, error.to_string()))?;
    let claim_bytes = canonical_jcs_bytes(&claim_value)
        .map_err(|error| ReplayError::new(ReplayErrorKind::Canonicalization, error.to_string()))?;
    let claim_request_sha256 = sha256_hex(&claim_bytes);

    let ceiling = claim_ceiling(claim.predicate);
    let body = DerivationBody {
        schema: RESULT_SCHEMA,
        evaluator_version: EVALUATOR_VERSION,
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
        claim_ceiling: &ceiling,
    };
    let body_value = serde_json::to_value(body)
        .map_err(|error| ReplayError::new(ReplayErrorKind::Canonicalization, error.to_string()))?;
    let body_bytes = canonical_jcs_bytes(&body_value)
        .map_err(|error| ReplayError::new(ReplayErrorKind::Canonicalization, error.to_string()))?;
    let derivation_sha256 = sha256_hex(&body_bytes);

    Ok(ClaimEvaluation {
        schema: RESULT_SCHEMA.to_owned(),
        evaluator_version: EVALUATOR_VERSION.to_owned(),
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
        claim_ceiling: ceiling,
        derivation_sha256,
    })
}

#[derive(Serialize)]
struct DerivationBody<'a> {
    schema: &'a str,
    evaluator_version: &'a str,
    rule_id: &'a str,
    claim_id: &'a str,
    claim_request_sha256: &'a str,
    predicate: Predicate,
    disposition: Disposition,
    scope: &'a Scope,
    asserted_value: &'a Value,
    source_predicate_type: &'a Option<String>,
    requested_evidence_ids: &'a [String],
    assumptions: &'a [String],
    evidence_assessments: &'a [Assessment],
    missing_evidence_ids: &'a [String],
    unmet_obligations: &'a [String],
    falsification_condition: &'a str,
    claim_ceiling: &'a [String],
}
