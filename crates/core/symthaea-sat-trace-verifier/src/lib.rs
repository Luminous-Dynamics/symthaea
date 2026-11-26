// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent, read-only verification of SAT v0.1 event traces.
//!
//! This crate verifies trace structure, canonical commitments and the
//! submit/observe/reconcile state machine. It does not authorize effects,
//! verify signatures, authenticate grant issuers, establish source truth,
//! certify observer independence, or issue a SAT qualification decision.
//!
//! The implemented JCS profile deliberately restricts JSON numbers to safe
//! integers. High-precision quantities must be represented as strings.

use chrono::{DateTime, FixedOffset};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Write as _};

pub mod checkpoint;
pub mod effect_receipt;

const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const MANIFEST_DOMAIN: &str = "SAT-MANIFEST-v0.1\n";
const EVENT_DOMAIN: &str = "SAT-EVENT-v0.1\n";
const BINDING_DOMAIN: &str = "SAT-BINDING-v0.1\n";
const AUTHORITY_DOMAIN: &str = "SAT-AUTHORITY-EFFECT-v0.1\n";
const DISCLOSURE_DOMAIN: &str = "SAT-DISCLOSURE-v0.1\n";
const SINK_PROJECTION_DOMAIN: &str = "SAT-SINK-PROJECTION-v0.1\n";

/// Options for the deliberately fail-closed trace profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VerificationOptions {
    /// Permit explicit not_applicable disclosure decisions.
    ///
    /// Enabling this option does not prove that a qualification profile
    /// legitimately permits the omission; that remains an evaluator decision.
    pub allow_not_applicable_disclosure: bool,
}

/// A successful structural trace verification. This is not SAT qualification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceVerificationReport {
    pub run_id: String,
    pub manifest_digest: String,
    pub event_count: usize,
    /// Internal event/hash-chain consistency only; a complete rewrite can be rehashed.
    pub event_chain_consistent: bool,
    pub head_digest: String,
    /// True only when the head digest equals a caller-supplied external checkpoint.
    /// This does not authenticate the checkpoint or its signer.
    pub head_digest_matches_supplied_anchor: bool,
    pub authority_bindings_checked: usize,
    pub disclosure_bindings_checked: usize,
    pub sink_projections_checked: usize,
    pub effect_submissions: usize,
    pub terminal_effect_observations: usize,
    pub unresolved_idempotency_keys: Vec<String>,
    pub hold_required: bool,
    /// Always false in v0.1: signature and trust-root verification are out of scope.
    pub signatures_verified: bool,
    /// Always false: trace verification is not a system-level qualification.
    pub system_qualified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceVerificationError {
    pub line_number: usize,
    pub code: VerificationErrorCode,
}

impl TraceVerificationError {
    fn new(line_number: usize, code: VerificationErrorCode) -> Self {
        Self { line_number, code }
    }
}

impl fmt::Display for TraceVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line_number, self.code)
    }
}

impl std::error::Error for TraceVerificationError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationErrorCode {
    EmptyLine,
    EmptyTrace,
    InvalidJson(String),
    DuplicateJsonKey(String),
    UnsupportedJsonNumber,
    RootMustBeObject,
    InvalidEventShape(String),
    InvalidDigest(String),
    HeadDigestAnchorMismatch,
    RunIdMismatch,
    ManifestDigestMismatch,
    SequenceMismatch { expected: u64, actual: u64 },
    DuplicateEventId(String),
    CausalParentNotPrior(String),
    DuplicateCausalParent(String),
    PreviousDigestMismatch,
    EventDigestMismatch,
    BindingDigestMismatch,
    SinkProjectionDigestMismatch,
    AuthorityDigestMismatch,
    DisclosureDigestMismatch,
    UnsortedOrDuplicateSet(String),
    ObserverIdentityMismatch,
    MissingIndependentObservationEvidence,
    ObservedProjectionMismatch,
    ObservedEffectDigestMismatch,
    NotApplicableDisclosureNotAllowed,
    SubmissionWithoutCurrentAllows,
    MissingIdempotencyKey,
    IdempotencyKeyMismatch,
    ObservationWithoutSubmission(String),
    UnresolvedOutcomeBlocksNewMutation(String),
    DuplicateAppliedIdempotencyKey(String),
    DuplicateAttemptId(String),
    DuplicateTerminalObservation(String),
    StaleAttemptObservation { key: String, expected_attempt_id: String, observed_attempt_id: String },
    EffectCausalityMismatch(String),
    IdempotencyKeyRebound(String),
}

impl fmt::Display for VerificationErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLine => write!(f, "empty JSONL record"),
            Self::EmptyTrace => write!(f, "trace has no events"),
            Self::InvalidJson(reason) => write!(f, "invalid JSON: {reason}"),
            Self::DuplicateJsonKey(key) => write!(f, "duplicate JSON object key: {key}"),
            Self::UnsupportedJsonNumber => {
                write!(f, "number is not a safe integer in the SAT JCS v0.1 profile")
            }
            Self::RootMustBeObject => write!(f, "event must be a JSON object"),
            Self::InvalidEventShape(reason) => write!(f, "invalid event shape: {reason}"),
            Self::InvalidDigest(field) => write!(f, "invalid digest in {field}"),
            Self::HeadDigestAnchorMismatch => write!(f, "trace head digest does not match supplied external anchor"),
            Self::RunIdMismatch => write!(f, "event run_id differs from the expected run"),
            Self::ManifestDigestMismatch => {
                write!(f, "event manifest_digest differs from the frozen manifest")
            }
            Self::SequenceMismatch { expected, actual } => {
                write!(f, "expected event sequence {expected}, received {actual}")
            }
            Self::DuplicateEventId(id) => write!(f, "duplicate event_id: {id}"),
            Self::CausalParentNotPrior(id) => {
                write!(f, "causal parent is missing or not prior: {id}")
            }
            Self::DuplicateCausalParent(id) => write!(f, "duplicate causal parent: {id}"),
            Self::PreviousDigestMismatch => write!(f, "previous_event_digest does not chain"),
            Self::EventDigestMismatch => write!(f, "event_digest does not match canonical bytes"),
            Self::BindingDigestMismatch => write!(f, "binding_digest does not match intent"),
            Self::SinkProjectionDigestMismatch => {
                write!(f, "sink_projection_digest does not match declared projection")
            }
            Self::AuthorityDigestMismatch => {
                write!(f, "exact_effect_digest does not match reference-model fields")
            }
            Self::DisclosureDigestMismatch => {
                write!(f, "exact_disclosure_digest does not match disclosure projection")
            }
            Self::UnsortedOrDuplicateSet(field) => {
                write!(f, "{field} must be unique and UTF-16 sorted")
            }
            Self::ObserverIdentityMismatch => {
                write!(f, "sink observer role/identity does not match observation")
            }
            Self::MissingIndependentObservationEvidence => {
                write!(f, "effect observation lacks independently-observed evidence")
            }
            Self::ObservedProjectionMismatch => {
                write!(f, "observed projection does not match the requested sink projection")
            }
            Self::ObservedEffectDigestMismatch => {
                write!(f, "observed_effect_digest does not match the reconstructed projection")
            }
            Self::NotApplicableDisclosureNotAllowed => {
                write!(f, "not_applicable disclosure is not allowed by this verifier profile")
            }
            Self::SubmissionWithoutCurrentAllows => {
                write!(f, "effect submission lacks matching allow decisions")
            }
            Self::MissingIdempotencyKey => write!(f, "effect mutation lacks an idempotency key"),
            Self::IdempotencyKeyMismatch => {
                write!(f, "effect binding and observation idempotency keys differ")
            }
            Self::ObservationWithoutSubmission(key) => {
                write!(f, "observation has no preceding submission for key {key}")
            }
            Self::UnresolvedOutcomeBlocksNewMutation(key) => {
                write!(f, "new mutation blocked while prior outcome {key} is unresolved")
            }
            Self::DuplicateAppliedIdempotencyKey(key) => {
                write!(f, "idempotency key {key} was already observed applied")
            }
            Self::DuplicateAttemptId(attempt_id) => {
                write!(f, "attempt_id {attempt_id} was reused")
            }
            Self::DuplicateTerminalObservation(key) => {
                write!(f, "attempt for idempotency key {key} already has a terminal observation")
            }
            Self::StaleAttemptObservation { key, expected_attempt_id, observed_attempt_id } => {
                write!(f, "observation for idempotency key {key} names stale attempt {observed_attempt_id}; current attempt is {expected_attempt_id}")
            }
            Self::EffectCausalityMismatch(key) => {
                write!(f, "observation for idempotency key {key} is not causally linked to the latest event for its attempt")
            }
            Self::IdempotencyKeyRebound(key) => {
                write!(f, "idempotency key {key} was rebound to a different intent")
            }
        }
    }
}

/// Strict JSON tree that preserves object entries until duplicate-key checks finish.
#[derive(Debug, Clone, PartialEq)]
enum StrictJson {
    Null,
    Bool(bool),
    Number(i64),
    String(String),
    Array(Vec<StrictJson>),
    Object(Vec<(String, StrictJson)>),
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("strict JSON with unique object keys and safe integer numbers")
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(StrictJson::Null)
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(StrictJson::Null)
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
        Ok(StrictJson::Bool(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
        if value.unsigned_abs() > MAX_SAFE_INTEGER {
            return Err(E::custom("integer exceeds the I-JSON safe integer range"));
        }
        Ok(StrictJson::Number(value))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
        if value > MAX_SAFE_INTEGER {
            return Err(E::custom("integer exceeds the I-JSON safe integer range"));
        }
        Ok(StrictJson::Number(value as i64))
    }

    fn visit_f64<E: de::Error>(self, _value: f64) -> Result<Self::Value, E> {
        Err(E::custom("non-integer JSON numbers are outside the SAT JCS v0.1 profile"))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        Ok(StrictJson::String(value.to_owned()))
    }

    fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
        Ok(StrictJson::String(value))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictJson>()? {
            values.push(value);
        }
        Ok(StrictJson::Array(values))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut entries = Vec::new();
        let mut seen = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate JSON key: {key}")));
            }
            let value = map.next_value::<StrictJson>()?;
            entries.push((key, value));
        }
        Ok(StrictJson::Object(entries))
    }
}

impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

impl StrictJson {
    fn get(&self, key: &str) -> Option<&StrictJson> {
        match self {
            Self::Object(entries) => entries
                .iter()
                .find(|(candidate, _)| candidate == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    fn without_key(&self, key: &str) -> Option<Self> {
        match self {
            Self::Object(entries) => Some(Self::Object(
                entries
                    .iter()
                    .filter(|(candidate, _)| candidate != key)
                    .cloned()
                    .collect(),
            )),
            _ => None,
        }
    }

    fn is_object(&self) -> bool {
        matches!(self, Self::Object(_))
    }

    fn string(value: impl Into<String>) -> Self {
        Self::String(value.into())
    }

    fn object(fields: Vec<(&str, StrictJson)>) -> Self {
        Self::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value))
                .collect(),
        )
    }

    fn array_strings(values: &[String]) -> Self {
        Self::Array(values.iter().cloned().map(Self::String).collect())
    }
}

fn utf16_key(value: &str) -> Vec<u16> {
    value.encode_utf16().collect()
}

fn utf16_cmp(left: &str, right: &str) -> Ordering {
    utf16_key(left).cmp(&utf16_key(right))
}

fn canonicalize(value: &StrictJson) -> Vec<u8> {
    let mut output = String::new();
    write_canonical(value, &mut output);
    output.into_bytes()
}

fn write_canonical(value: &StrictJson, output: &mut String) {
    match value {
        StrictJson::Null => output.push_str("null"),
        StrictJson::Bool(true) => output.push_str("true"),
        StrictJson::Bool(false) => output.push_str("false"),
        StrictJson::Number(number) => {
            let _ = write!(output, "{number}");
        }
        StrictJson::String(value) => write_string(value, output),
        StrictJson::Array(values) => {
            output.push('[');
            for (index, item) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_canonical(item, output);
            }
            output.push(']');
        }
        StrictJson::Object(entries) => {
            let mut ordered: Vec<_> = entries.iter().collect();
            ordered.sort_by(|(left, _), (right, _)| utf16_cmp(left, right));
            output.push('{');
            for (index, (key, item)) in ordered.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_string(key, output);
                output.push(':');
                write_canonical(item, output);
            }
            output.push('}');
        }
    }
}

fn write_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{0009}' => output.push_str("\\t"),
            '\u{000a}' => output.push_str("\\n"),
            '\u{000c}' => output.push_str("\\f"),
            '\u{000d}' => output.push_str("\\r"),
            character if (character as u32) <= 0x1f => {
                let _ = write!(output, "\\u{:04x}", character as u32);
            }
            character => output.push(character),
        }
    }
    output.push('"');
}

fn sha256_domain(domain: &str, bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(7 + digest.len() * 2);
    encoded.push_str("sha256:");
    for byte in digest {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn digest_json(domain: &str, value: &StrictJson) -> String {
    sha256_domain(domain, &canonicalize(value))
}

fn is_digest(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    })
}

fn require_sorted_unique(values: &[String], field: &str) -> Result<(), VerificationErrorCode> {
    for pair in values.windows(2) {
        if utf16_cmp(&pair[0], &pair[1]) != Ordering::Less {
            return Err(VerificationErrorCode::UnsortedOrDuplicateSet(field.to_owned()));
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EventRecord {
    schema_version: String,
    run_id: String,
    manifest_digest: String,
    event_id: String,
    sequence: u64,
    occurred_at: String,
    event_type: EventType,
    actor: Actor,
    causality: Causality,
    #[serde(default)]
    data_lineage_refs: Vec<String>,
    effect_binding: Option<EffectBinding>,
    authority_decision: Option<AuthorityDecision>,
    disclosure_decision: Option<DisclosureDecision>,
    effect_observation: Option<EffectObservation>,
    human_decision: Option<HumanDecision>,
    auditor_finding: Option<AuditorFinding>,
    #[serde(default)]
    evidence_refs: Vec<EvidenceRef>,
    integrity: Integrity,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EventType {
    ProposalCreated,
    DataAccessed,
    DataTransformed,
    DelegationIssued,
    DelegationRevoked,
    AuthorizationDecided,
    EffectSubmitted,
    EffectObserved,
    EffectReconciled,
    MonitorIntervened,
    HumanDecided,
    AuditorFinding,
    ProcessFenced,
    EvidenceIngested,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ActorRole {
    Human,
    Agent,
    SubAgent,
    PolicyEngine,
    Monitor,
    SinkObserver,
    Auditor,
    Evaluator,
    Operator,
    System,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Actor {
    principal_id: String,
    role: ActorRole,
    component_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Causality {
    correlation_id: String,
    parent_event_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EffectBinding {
    action: String,
    resource: String,
    sink: String,
    purpose: String,
    parameters_digest: String,
    payload_digest: String,
    payload_labels: Vec<String>,
    binding_digest: String,
    sink_projection_digest: String,
    sink_projection_id: String,
    sink_projection_version: String,
    sink_projection: StrictJson,
    lineage_proof_ref: Option<String>,
    interaction_intent_ref: Option<String>,
    idempotency_key: Option<String>,
    attempt_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AuthorityOutcome {
    Allow,
    Deny,
    Indeterminate,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorityDecision {
    decision: AuthorityOutcome,
    policy_digest: String,
    exact_effect_digest: String,
    #[serde(default)]
    grant_refs: Vec<String>,
    decision_receipt_ref: Option<String>,
    valid_until: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum DisclosureOutcome {
    Allow,
    Deny,
    Indeterminate,
    NotApplicable,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DisclosureDecision {
    decision: DisclosureOutcome,
    policy_digest: String,
    exact_disclosure_digest: String,
    #[serde(default)]
    lineage_evidence_refs: Vec<String>,
    not_applicable_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EffectOutcome {
    ObservedApplied,
    ObservedNotApplied,
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EffectObservation {
    outcome: EffectOutcome,
    observer_id: String,
    source_of_truth: String,
    observed_at: String,
    state_version: Option<String>,
    reconciliation_ref: Option<String>,
    idempotency_key: String,
    attempt_id: String,
    sink_projection_id: String,
    sink_projection_version: String,
    observed_effect_digest: Option<String>,
    observed_projection: Option<StrictJson>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HumanDecision {
    decision: String,
    decision_maker_id: String,
    independent_first: bool,
    information_order_arm: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditorFinding {
    finding_id: String,
    status: String,
    evidence_refs: Vec<String>,
    disconfirmation_refs: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceRef {
    artifact_id: String,
    digest: String,
    source_id: String,
    capture_method: String,
    independently_observed: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Integrity {
    hash_algorithm: String,
    previous_event_digest: Option<String>,
    event_digest: String,
    signature_ref: Option<String>,
    signer_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttemptStatus {
    Pending,
    Applied,
    NotApplied,
}

#[derive(Debug, Clone)]
struct Attempt {
    binding_digest: String,
    attempt_id: String,
    last_event_id: String,
    correlation_id: String,
    status: AttemptStatus,
}

/// Compute the digest of a frozen run manifest.
///
/// The manifest digest is SHA-256(domain separator + restricted JCS bytes).
/// This function checks duplicate keys and the SAT v0.1 safe-integer number
/// profile, but does not validate the run-manifest JSON Schema.
pub fn compute_manifest_digest(
    manifest_json: &str,
) -> Result<String, TraceVerificationError> {
    let raw: StrictJson = serde_json::from_str(manifest_json).map_err(|error| {
        let message = error.to_string();
        let code = if let Some(key) = message.split("duplicate JSON key: ").nth(1) {
            VerificationErrorCode::DuplicateJsonKey(key.to_owned())
        } else if message.contains("safe integer")
            || message.contains("non-integer JSON")
        {
            VerificationErrorCode::UnsupportedJsonNumber
        } else {
            VerificationErrorCode::InvalidJson(message)
        };
        TraceVerificationError::new(0, code)
    })?;
    if !raw.is_object() {
        return Err(TraceVerificationError::new(
            0,
            VerificationErrorCode::RootMustBeObject,
        ));
    }
    Ok(digest_json(MANIFEST_DOMAIN, &raw))
}

/// Verify internal JSONL trace consistency against an externally supplied run
/// and frozen-manifest identity. This does not detect a complete trace rewrite:
/// use `verify_trace_with_anchor` with a head digest obtained from a separately
/// qualified checkpoint or witness when that stronger claim is required.
pub fn verify_trace(
    jsonl: &str,
    expected_run_id: &str,
    expected_manifest_digest: &str,
) -> Result<TraceVerificationReport, TraceVerificationError> {
    verify_trace_with_options(
        jsonl,
        expected_run_id,
        expected_manifest_digest,
        VerificationOptions::default(),
    )
}

/// Verify a trace and compare its final event digest with an externally
/// supplied checkpoint. The caller is responsible for separately qualifying
/// the checkpoint source, custody, signer and revocation status.
pub fn verify_trace_with_anchor(
    jsonl: &str,
    expected_run_id: &str,
    expected_manifest_digest: &str,
    expected_head_digest: &str,
) -> Result<TraceVerificationReport, TraceVerificationError> {
    check_digest(expected_head_digest, "expected_head_digest", 0)?;
    let mut report = verify_trace(jsonl, expected_run_id, expected_manifest_digest)?;
    if report.head_digest != expected_head_digest {
        return Err(TraceVerificationError::new(
            0,
            VerificationErrorCode::HeadDigestAnchorMismatch,
        ));
    }
    report.head_digest_matches_supplied_anchor = true;
    Ok(report)
}

pub fn verify_trace_with_options(
    jsonl: &str,
    expected_run_id: &str,
    expected_manifest_digest: &str,
    options: VerificationOptions,
) -> Result<TraceVerificationReport, TraceVerificationError> {
    if !is_digest(expected_manifest_digest) {
        return Err(TraceVerificationError::new(
            0,
            VerificationErrorCode::InvalidDigest("expected_manifest_digest".to_owned()),
        ));
    }

    let mut previous_digest: Option<String> = None;
    let mut seen_event_ids = HashSet::new();
    let mut seen_prior_event_ids = HashSet::new();
    let mut seen_attempt_ids = HashSet::new();
    let mut attempts: HashMap<String, Attempt> = HashMap::new();
    let mut event_count = 0usize;
    let mut authority_checked = 0usize;
    let mut disclosure_checked = 0usize;
    let mut sink_checked = 0usize;
    let mut submissions = 0usize;
    let mut terminal_observations = 0usize;

    for (zero_index, line) in jsonl.lines().enumerate() {
        let line_number = zero_index + 1;
        if line.trim().is_empty() {
            return Err(TraceVerificationError::new(
                line_number,
                VerificationErrorCode::EmptyLine,
            ));
        }

        // Parse duplicate-preserving JSON before typed deserialization.
        let raw: StrictJson = serde_json::from_str(line).map_err(|error| {
            let message = error.to_string();
            if let Some(key) = message.split("duplicate JSON key: ").nth(1) {
                TraceVerificationError::new(
                    line_number,
                    VerificationErrorCode::DuplicateJsonKey(key.to_owned()),
                )
            } else if message.contains("safe integer")
                || message.contains("non-integer JSON")
            {
                TraceVerificationError::new(
                    line_number,
                    VerificationErrorCode::UnsupportedJsonNumber,
                )
            } else {
                TraceVerificationError::new(
                    line_number,
                    VerificationErrorCode::InvalidJson(message),
                )
            }
        })?;

        if !raw.is_object() {
            return Err(TraceVerificationError::new(
                line_number,
                VerificationErrorCode::RootMustBeObject,
            ));
        }
        let raw_integrity = raw.get("integrity").ok_or_else(|| {
            TraceVerificationError::new(
                line_number,
                VerificationErrorCode::InvalidEventShape("missing integrity object".to_owned()),
            )
        })?;
        if raw_integrity.get("previous_event_digest").is_none() {
            return Err(TraceVerificationError::new(
                line_number,
                VerificationErrorCode::InvalidEventShape(
                    "integrity.previous_event_digest must be present (null on the first event)".to_owned(),
                ),
            ));
        }
        if !matches!(raw.get("evidence_refs"), Some(StrictJson::Array(_))) {
            return Err(TraceVerificationError::new(
                line_number,
                VerificationErrorCode::InvalidEventShape(
                    "evidence_refs must be present as an array".to_owned(),
                ),
            ));
        }

        let event: EventRecord = serde_json::from_str(line).map_err(|error| {
            TraceVerificationError::new(
                line_number,
                VerificationErrorCode::InvalidEventShape(error.to_string()),
            )
        })?;

        if event.schema_version != "sat-event-record/v0.1" {
            return Err(TraceVerificationError::new(
                line_number,
                VerificationErrorCode::InvalidEventShape(
                    "unsupported schema_version".to_owned(),
                ),
            ));
        }
        if event.run_id != expected_run_id {
            return Err(TraceVerificationError::new(
                line_number,
                VerificationErrorCode::RunIdMismatch,
            ));
        }
        if event.manifest_digest != expected_manifest_digest {
            return Err(TraceVerificationError::new(
                line_number,
                VerificationErrorCode::ManifestDigestMismatch,
            ));
        }
        let expected_sequence = event_count as u64 + 1;
        if event.sequence != expected_sequence {
            return Err(TraceVerificationError::new(
                line_number,
                VerificationErrorCode::SequenceMismatch {
                    expected: expected_sequence,
                    actual: event.sequence,
                },
            ));
        }
        if !seen_event_ids.insert(event.event_id.clone()) {
            return Err(TraceVerificationError::new(
                line_number,
                VerificationErrorCode::DuplicateEventId(event.event_id.clone()),
            ));
        }

        let mut unique_parents = HashSet::new();
        for parent in &event.causality.parent_event_ids {
            if !unique_parents.insert(parent) {
                return Err(TraceVerificationError::new(
                    line_number,
                    VerificationErrorCode::DuplicateCausalParent(parent.clone()),
                ));
            }
            if !seen_prior_event_ids.contains(parent) {
                return Err(TraceVerificationError::new(
                    line_number,
                    VerificationErrorCode::CausalParentNotPrior(parent.clone()),
                ));
            }
        }

        if event.integrity.hash_algorithm != "sha256-jcs-v1" {
            return Err(TraceVerificationError::new(
                line_number,
                VerificationErrorCode::InvalidEventShape(
                    "unsupported hash_algorithm".to_owned(),
                ),
            ));
        }
        check_digest(
            &event.integrity.event_digest,
            "integrity.event_digest",
            line_number,
        )?;
        if let Some(previous) = &event.integrity.previous_event_digest {
            check_digest(previous, "integrity.previous_event_digest", line_number)?;
        }
        if event.integrity.previous_event_digest != previous_digest {
            return Err(TraceVerificationError::new(
                line_number,
                VerificationErrorCode::PreviousDigestMismatch,
            ));
        }

        let integrity = raw.get("integrity").ok_or_else(|| {
            TraceVerificationError::new(
                line_number,
                VerificationErrorCode::InvalidEventShape(
                    "missing integrity object".to_owned(),
                ),
            )
        })?;
        let integrity_without_digest = integrity.without_key("event_digest").ok_or_else(|| {
            TraceVerificationError::new(
                line_number,
                VerificationErrorCode::InvalidEventShape(
                    "integrity must be an object".to_owned(),
                ),
            )
        })?;
        let event_without_digest =
            replace_nested(raw.clone(), "integrity", integrity_without_digest).ok_or_else(
                || {
                    TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::RootMustBeObject,
                    )
                },
            )?;
        let computed_event_digest = digest_json(EVENT_DOMAIN, &event_without_digest);
        if computed_event_digest != event.integrity.event_digest {
            return Err(TraceVerificationError::new(
                line_number,
                VerificationErrorCode::EventDigestMismatch,
            ));
        }

        if let Some(binding) = &event.effect_binding {
            let raw_binding = raw.get("effect_binding").ok_or_else(|| {
                TraceVerificationError::new(
                    line_number,
                    VerificationErrorCode::InvalidEventShape(
                        "missing effect_binding object".to_owned(),
                    ),
                )
            })?;
            verify_binding(raw_binding, binding, line_number)?;

            if let Some(authority) = &event.authority_decision {
                verify_authority_binding(binding, authority, line_number)?;
                authority_checked += 1;
            }
            if let Some(disclosure) = &event.disclosure_decision {
                verify_disclosure_binding(binding, disclosure, line_number)?;
                if disclosure.decision == DisclosureOutcome::NotApplicable
                    && !options.allow_not_applicable_disclosure
                {
                    return Err(TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::NotApplicableDisclosureNotAllowed,
                    ));
                }
                disclosure_checked += 1;
            }
            sink_checked += 1;
        }

        validate_event_roles_and_requirements(&event, line_number)?;
        validate_effect_observation(&event, line_number)?;

        match event.event_type {
            EventType::EffectSubmitted => {
                let binding = event.effect_binding.as_ref().expect("validated above");
                let authority = event.authority_decision.as_ref().expect("validated above");
                let disclosure = event.disclosure_decision.as_ref().expect("validated above");
                let disclosure_allowed = disclosure.decision == DisclosureOutcome::Allow
                    || (disclosure.decision == DisclosureOutcome::NotApplicable
                        && options.allow_not_applicable_disclosure
                        && disclosure
                            .not_applicable_reason
                            .as_deref()
                            .is_some_and(|reason| !reason.trim().is_empty()));
                if authority.decision != AuthorityOutcome::Allow || !disclosure_allowed {
                    return Err(TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::SubmissionWithoutCurrentAllows,
                    ));
                }
                let receipt_ref = authority
                    .decision_receipt_ref
                    .as_deref()
                    .filter(|value| !value.trim().is_empty())
                    .ok_or_else(|| TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::SubmissionWithoutCurrentAllows,
                    ))?;
                let valid_until = authority
                    .valid_until
                    .as_deref()
                    .and_then(parse_timestamp)
                    .ok_or_else(|| TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::SubmissionWithoutCurrentAllows,
                    ))?;
                let committed_at = parse_timestamp(&event.occurred_at)
                    .ok_or_else(|| TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::InvalidEventShape(
                            "occurred_at must be RFC 3339".to_owned(),
                        ),
                    ))?;
                if committed_at >= valid_until {
                    return Err(TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::SubmissionWithoutCurrentAllows,
                    ));
                }
                let _ = receipt_ref;

                let key = binding
                    .idempotency_key
                    .as_ref()
                    .filter(|key| !key.trim().is_empty())
                    .ok_or_else(|| {
                        TraceVerificationError::new(
                            line_number,
                            VerificationErrorCode::MissingIdempotencyKey,
                        )
                    })?;
                let attempt_id = binding
                    .attempt_id
                    .as_ref()
                    .filter(|attempt_id| !attempt_id.trim().is_empty())
                    .ok_or_else(|| {
                        TraceVerificationError::new(
                            line_number,
                            VerificationErrorCode::InvalidEventShape(
                                "effect submission requires a non-empty attempt_id".to_owned(),
                            ),
                        )
                    })?;
                if !seen_attempt_ids.insert(attempt_id.clone()) {
                    return Err(TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::DuplicateAttemptId(attempt_id.clone()),
                    ));
                }

                if let Some((pending_key, _)) = attempts
                    .iter()
                    .find(|(_, attempt)| attempt.status == AttemptStatus::Pending)
                {
                    return Err(TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::UnresolvedOutcomeBlocksNewMutation(
                            pending_key.clone(),
                        ),
                    ));
                }

                if let Some(previous) = attempts.get(key) {
                    if previous.binding_digest != binding.binding_digest {
                        return Err(TraceVerificationError::new(
                            line_number,
                            VerificationErrorCode::IdempotencyKeyRebound(key.clone()),
                        ));
                    }
                    match previous.status {
                        AttemptStatus::Applied => {
                            return Err(TraceVerificationError::new(
                                line_number,
                                VerificationErrorCode::DuplicateAppliedIdempotencyKey(
                                    key.clone(),
                                ),
                            ));
                        }
                        AttemptStatus::Pending => {
                            return Err(TraceVerificationError::new(
                                line_number,
                                VerificationErrorCode::UnresolvedOutcomeBlocksNewMutation(
                                    key.clone(),
                                ),
                            ));
                        }
                        AttemptStatus::NotApplied => {}
                    }
                }

                attempts.insert(
                    key.clone(),
                    Attempt {
                        binding_digest: binding.binding_digest.clone(),
                        attempt_id: attempt_id.clone(),
                        last_event_id: event.event_id.clone(),
                        correlation_id: event.causality.correlation_id.clone(),
                        status: AttemptStatus::Pending,
                    },
                );
                submissions += 1;
            }
            EventType::EffectObserved | EventType::EffectReconciled => {
                let binding = event.effect_binding.as_ref().expect("validated above");
                let observation = event.effect_observation.as_ref().expect("validated above");
                let key = binding
                    .idempotency_key
                    .as_ref()
                    .filter(|key| !key.trim().is_empty())
                    .ok_or_else(|| {
                        TraceVerificationError::new(
                            line_number,
                            VerificationErrorCode::MissingIdempotencyKey,
                        )
                    })?;
                if key != &observation.idempotency_key
                    || binding.attempt_id.as_deref() != Some(observation.attempt_id.as_str())
                {
                    return Err(TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::IdempotencyKeyMismatch,
                    ));
                }
                let attempt = attempts.get_mut(key).ok_or_else(|| {
                    TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::ObservationWithoutSubmission(key.clone()),
                    )
                })?;
                if attempt.binding_digest != binding.binding_digest {
                    return Err(TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::IdempotencyKeyRebound(key.clone()),
                    ));
                }
                if attempt.attempt_id != observation.attempt_id {
                    return Err(TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::StaleAttemptObservation {
                            key: key.clone(),
                            expected_attempt_id: attempt.attempt_id.clone(),
                            observed_attempt_id: observation.attempt_id.clone(),
                        },
                    ));
                }
                if event.causality.correlation_id != attempt.correlation_id
                    || !event.causality.parent_event_ids.contains(&attempt.last_event_id)
                {
                    return Err(TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::EffectCausalityMismatch(key.clone()),
                    ));
                }
                if attempt.status != AttemptStatus::Pending {
                    return Err(TraceVerificationError::new(
                        line_number,
                        VerificationErrorCode::DuplicateTerminalObservation(key.clone()),
                    ));
                }
                match observation.outcome {
                    EffectOutcome::Unknown => attempt.status = AttemptStatus::Pending,
                    EffectOutcome::ObservedApplied => {
                        attempt.status = AttemptStatus::Applied;
                        terminal_observations += 1;
                    }
                    EffectOutcome::ObservedNotApplied => {
                        attempt.status = AttemptStatus::NotApplied;
                        terminal_observations += 1;
                    }
                }
                attempt.last_event_id = event.event_id.clone();
            }
            _ => {}
        }

        previous_digest = Some(event.integrity.event_digest.clone());
        seen_prior_event_ids.insert(event.event_id.clone());
        event_count += 1;
    }

    if event_count == 0 {
        return Err(TraceVerificationError::new(
            0,
            VerificationErrorCode::EmptyTrace,
        ));
    }

    let mut unresolved: Vec<String> = attempts
        .iter()
        .filter(|(_, attempt)| attempt.status == AttemptStatus::Pending)
        .map(|(key, _)| key.clone())
        .collect();
    unresolved.sort_by(|left, right| utf16_cmp(left, right));

    Ok(TraceVerificationReport {
        run_id: expected_run_id.to_owned(),
        manifest_digest: expected_manifest_digest.to_owned(),
        event_count,
        event_chain_consistent: true,
        head_digest: previous_digest.unwrap_or_default(),
        head_digest_matches_supplied_anchor: false,
        authority_bindings_checked: authority_checked,
        disclosure_bindings_checked: disclosure_checked,
        sink_projections_checked: sink_checked,
        effect_submissions: submissions,
        terminal_effect_observations: terminal_observations,
        hold_required: !unresolved.is_empty(),
        unresolved_idempotency_keys: unresolved,
        signatures_verified: false,
        system_qualified: false,
    })
}

fn parse_timestamp(value: &str) -> Option<DateTime<FixedOffset>> {
    DateTime::parse_from_rfc3339(value).ok()
}

fn check_digest(
    value: &str,
    field: &str,
    line_number: usize,
) -> Result<(), TraceVerificationError> {
    if is_digest(value) {
        Ok(())
    } else {
        Err(TraceVerificationError::new(
            line_number,
            VerificationErrorCode::InvalidDigest(field.to_owned()),
        ))
    }
}

fn replace_nested(root: StrictJson, parent_key: &str, replacement: StrictJson) -> Option<StrictJson> {
    match root {
        StrictJson::Object(mut entries) => {
            let target = entries.iter_mut().find(|(key, _)| key == parent_key)?;
            target.1 = replacement;
            Some(StrictJson::Object(entries))
        }
        _ => None,
    }
}

fn verify_binding(
    raw_binding: &StrictJson,
    binding: &EffectBinding,
    line_number: usize,
) -> Result<(), TraceVerificationError> {
    for (field, value) in [
        ("parameters_digest", &binding.parameters_digest),
        ("payload_digest", &binding.payload_digest),
        ("binding_digest", &binding.binding_digest),
        ("sink_projection_digest", &binding.sink_projection_digest),
    ] {
        check_digest(value, field, line_number)?;
    }
    require_sorted_unique(&binding.payload_labels, "payload_labels")
        .map_err(|code| TraceVerificationError::new(line_number, code))?;

    let binding_without_digest = raw_binding
        .without_key("binding_digest")
        .and_then(|value| value.without_key("attempt_id"))
        .ok_or_else(|| {
            TraceVerificationError::new(
                line_number,
                VerificationErrorCode::InvalidEventShape(
                    "effect_binding must be an object".to_owned(),
                ),
            )
        })?;
    if digest_json(BINDING_DOMAIN, &binding_without_digest) != binding.binding_digest {
        return Err(TraceVerificationError::new(
            line_number,
            VerificationErrorCode::BindingDigestMismatch,
        ));
    }

    if !binding.sink_projection.is_object() {
        return Err(TraceVerificationError::new(
            line_number,
            VerificationErrorCode::InvalidEventShape(
                "sink_projection must be an object".to_owned(),
            ),
        ));
    }
    let projection = StrictJson::object(vec![
        ("projection_id", StrictJson::string(&binding.sink_projection_id)),
        (
            "projection_version",
            StrictJson::string(&binding.sink_projection_version),
        ),
        ("projection", binding.sink_projection.clone()),
    ]);
    let computed_projection_digest = digest_json(SINK_PROJECTION_DOMAIN, &projection);
    if computed_projection_digest != binding.sink_projection_digest {
        return Err(TraceVerificationError::new(
            line_number,
            VerificationErrorCode::SinkProjectionDigestMismatch,
        ));
    }
    Ok(())
}

fn verify_authority_binding(
    binding: &EffectBinding,
    authority: &AuthorityDecision,
    line_number: usize,
) -> Result<(), TraceVerificationError> {
    check_digest(
        &authority.policy_digest,
        "authority_decision.policy_digest",
        line_number,
    )?;
    check_digest(
        &authority.exact_effect_digest,
        "authority_decision.exact_effect_digest",
        line_number,
    )?;
    let _ = &authority.grant_refs;
    let projection = StrictJson::object(vec![
        ("action", StrictJson::string(&binding.action)),
        ("resource", StrictJson::string(&binding.resource)),
        ("sink", StrictJson::string(&binding.sink)),
        ("purpose", StrictJson::string(&binding.purpose)),
        (
            "parameters_digest",
            StrictJson::string(&binding.parameters_digest),
        ),
        (
            "payload_labels",
            StrictJson::array_strings(&binding.payload_labels),
        ),
    ]);
    if digest_json(AUTHORITY_DOMAIN, &projection) != authority.exact_effect_digest {
        return Err(TraceVerificationError::new(
            line_number,
            VerificationErrorCode::AuthorityDigestMismatch,
        ));
    }
    Ok(())
}

fn verify_disclosure_binding(
    binding: &EffectBinding,
    disclosure: &DisclosureDecision,
    line_number: usize,
) -> Result<(), TraceVerificationError> {
    check_digest(
        &disclosure.policy_digest,
        "disclosure_decision.policy_digest",
        line_number,
    )?;
    check_digest(
        &disclosure.exact_disclosure_digest,
        "disclosure_decision.exact_disclosure_digest",
        line_number,
    )?;
    require_sorted_unique(&disclosure.lineage_evidence_refs, "lineage_evidence_refs")
        .map_err(|code| TraceVerificationError::new(line_number, code))?;

    if disclosure.decision == DisclosureOutcome::NotApplicable
        && disclosure
            .not_applicable_reason
            .as_deref()
            .is_none_or(|reason| reason.trim().is_empty())
    {
        return Err(TraceVerificationError::new(
            line_number,
            VerificationErrorCode::InvalidEventShape(
                "not_applicable disclosure requires a non-empty reason".to_owned(),
            ),
        ));
    }

    let projection = StrictJson::object(vec![
        ("sink", StrictJson::string(&binding.sink)),
        ("purpose", StrictJson::string(&binding.purpose)),
        (
            "payload_digest",
            StrictJson::string(&binding.payload_digest),
        ),
        (
            "payload_labels",
            StrictJson::array_strings(&binding.payload_labels),
        ),
        (
            "lineage_proof_ref",
            binding
                .lineage_proof_ref
                .as_ref()
                .map(StrictJson::string)
                .unwrap_or(StrictJson::Null),
        ),
        (
            "interaction_intent_ref",
            binding
                .interaction_intent_ref
                .as_ref()
                .map(StrictJson::string)
                .unwrap_or(StrictJson::Null),
        ),
        (
            "lineage_evidence_refs",
            StrictJson::array_strings(&disclosure.lineage_evidence_refs),
        ),
    ]);
    if digest_json(DISCLOSURE_DOMAIN, &projection) != disclosure.exact_disclosure_digest {
        return Err(TraceVerificationError::new(
            line_number,
            VerificationErrorCode::DisclosureDigestMismatch,
        ));
    }
    Ok(())
}

fn validate_event_roles_and_requirements(
    event: &EventRecord,
    line_number: usize,
) -> Result<(), TraceVerificationError> {
    let invalid = |message: &str| {
        TraceVerificationError::new(
            line_number,
            VerificationErrorCode::InvalidEventShape(message.to_owned()),
        )
    };
    match event.event_type {
        EventType::AuthorizationDecided | EventType::EffectSubmitted => {
            if event.effect_binding.is_none()
                || event.authority_decision.is_none()
                || event.disclosure_decision.is_none()
            {
                return Err(invalid(
                    "authorization_decided/effect_submitted require effect_binding, authority_decision and disclosure_decision",
                ));
            }
        }
        EventType::EffectObserved | EventType::EffectReconciled => {
            if event.effect_binding.is_none() || event.effect_observation.is_none() {
                return Err(invalid(
                    "effect_observed/effect_reconciled require binding and observation",
                ));
            }
        }
        EventType::HumanDecided => {
            let decision = event
                .human_decision
                .as_ref()
                .ok_or_else(|| invalid("human_decided requires human_decision"))?;
            if event.actor.role != ActorRole::Human
                || event.actor.principal_id != decision.decision_maker_id
            {
                return Err(invalid(
                    "human_decided actor must be the named human decision maker",
                ));
            }
        }
        EventType::AuditorFinding => {
            if event.auditor_finding.is_none() || event.actor.role != ActorRole::Auditor {
                return Err(invalid("auditor_finding requires auditor actor and finding"));
            }
        }
        _ => {}
    }
    if parse_timestamp(&event.occurred_at).is_none() {
        return Err(invalid("occurred_at must be RFC 3339"));
    }
    let _ = (&event.integrity.signature_ref, &event.integrity.signer_id);

    if event.event_id.trim().is_empty()
        || event.actor.principal_id.trim().is_empty()
        || event.actor.component_id.trim().is_empty()
        || event.causality.correlation_id.trim().is_empty()
        || event.occurred_at.trim().is_empty()
    {
        return Err(invalid(
            "event identity, actor, correlation and time fields must be non-empty",
        ));
    }

    if let Some(human) = &event.human_decision {
        let _ = (
            &human.decision,
            human.independent_first,
            &human.information_order_arm,
        );
    }
    if let Some(finding) = &event.auditor_finding {
        if finding.finding_id.trim().is_empty() || finding.evidence_refs.is_empty() {
            return Err(invalid("auditor finding requires an ID and evidence references"));
        }
        let _ = (&finding.status, &finding.disconfirmation_refs);
    }
    for evidence in &event.evidence_refs {
        if evidence.artifact_id.trim().is_empty()
            || evidence.source_id.trim().is_empty()
            || !is_digest(&evidence.digest)
        {
            return Err(invalid("evidence references require IDs and a valid digest"));
        }
        let _ = (&evidence.capture_method, evidence.independently_observed);
    }
    let _ = &event.data_lineage_refs;
    Ok(())
}

fn validate_effect_observation(
    event: &EventRecord,
    line_number: usize,
) -> Result<(), TraceVerificationError> {
    let Some(observation) = &event.effect_observation else {
        return Ok(());
    };
    let binding = event.effect_binding.as_ref().ok_or_else(|| {
        TraceVerificationError::new(
            line_number,
            VerificationErrorCode::InvalidEventShape(
                "observation requires effect binding".to_owned(),
            ),
        )
    })?;
    if event.actor.role != ActorRole::SinkObserver
        || event.actor.principal_id != observation.observer_id
        || observation.source_of_truth.trim().is_empty()
        || parse_timestamp(&observation.observed_at).is_none()
    {
        return Err(TraceVerificationError::new(
            line_number,
            VerificationErrorCode::ObserverIdentityMismatch,
        ));
    }
    if observation.idempotency_key.trim().is_empty()
        || observation.idempotency_key != binding.idempotency_key.as_deref().unwrap_or("")
        || observation.attempt_id.trim().is_empty()
        || observation.attempt_id != binding.attempt_id.as_deref().unwrap_or("")
        || (event.event_type == EventType::EffectReconciled
            && observation.reconciliation_ref.as_deref().is_none_or(|value| value.trim().is_empty()))
    {
        return Err(TraceVerificationError::new(
            line_number,
            VerificationErrorCode::IdempotencyKeyMismatch,
        ));
    }
    if observation.sink_projection_id != binding.sink_projection_id
        || observation.sink_projection_version != binding.sink_projection_version
    {
        return Err(TraceVerificationError::new(
            line_number,
            VerificationErrorCode::ObservedProjectionMismatch,
        ));
    }
    if !event
        .evidence_refs
        .iter()
        .any(|evidence| evidence.independently_observed == Some(true))
    {
        return Err(TraceVerificationError::new(
            line_number,
            VerificationErrorCode::MissingIndependentObservationEvidence,
        ));
    }

    match observation.outcome {
        EffectOutcome::ObservedApplied => {
            let observed_projection = observation.observed_projection.as_ref().ok_or_else(|| {
                TraceVerificationError::new(
                    line_number,
                    VerificationErrorCode::ObservedProjectionMismatch,
                )
            })?;
            if !observed_projection.is_object() {
                return Err(TraceVerificationError::new(
                    line_number,
                    VerificationErrorCode::ObservedProjectionMismatch,
                ));
            }
            let projection = StrictJson::object(vec![
                (
                    "projection_id",
                    StrictJson::string(&observation.sink_projection_id),
                ),
                (
                    "projection_version",
                    StrictJson::string(&observation.sink_projection_version),
                ),
                ("projection", observed_projection.clone()),
            ]);
            let computed = digest_json(SINK_PROJECTION_DOMAIN, &projection);
            let observed_digest = observation.observed_effect_digest.as_deref().ok_or_else(|| {
                TraceVerificationError::new(
                    line_number,
                    VerificationErrorCode::ObservedEffectDigestMismatch,
                )
            })?;
            check_digest(
                observed_digest,
                "effect_observation.observed_effect_digest",
                line_number,
            )?;
            if computed != observed_digest {
                return Err(TraceVerificationError::new(
                    line_number,
                    VerificationErrorCode::ObservedEffectDigestMismatch,
                ));
            }
            if observed_digest != binding.sink_projection_digest {
                return Err(TraceVerificationError::new(
                    line_number,
                    VerificationErrorCode::ObservedProjectionMismatch,
                ));
            }
        }
        EffectOutcome::ObservedNotApplied | EffectOutcome::Unknown => {
            if observation.observed_effect_digest.is_some()
                || observation.observed_projection.is_some()
            {
                return Err(TraceVerificationError::new(
                    line_number,
                    VerificationErrorCode::InvalidEventShape(
                        "observed projection/digest is only valid for observed_applied".to_owned(),
                    ),
                ));
            }
        }
    }
    let _ = (
        &observation.observed_at,
        &observation.state_version,
        &observation.reconciliation_ref,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    const RUN_ID: &str = "00000000-0000-4000-8000-000000000001";
    const MANIFEST: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn sorted(values: &[&str]) -> Vec<String> {
        let mut out: Vec<String> = values.iter().map(|value| (*value).to_owned()).collect();
        out.sort_by(|left, right| utf16_cmp(left, right));
        out
    }

    fn digest_value(domain: &str, value: &Value) -> String {
        let raw: StrictJson =
            serde_json::from_value(value.clone()).expect("strict JSON value");
        digest_json(domain, &raw)
    }

    fn finish_event(mut event: Value) -> Value {
        if let Some(binding) = event.get("effect_binding").cloned() {
            let projection = json!({
                "projection_id": binding["sink_projection_id"],
                "projection_version": binding["sink_projection_version"],
                "projection": binding["sink_projection"]
            });
            let projection_digest = digest_value(SINK_PROJECTION_DOMAIN, &projection);
            event["effect_binding"]["sink_projection_digest"] = json!(projection_digest);

            let binding = event["effect_binding"].clone();
            let labels: Vec<String> = binding["payload_labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap().to_owned())
                .collect();
            let authority_projection = json!({
                "action": binding["action"],
                "resource": binding["resource"],
                "sink": binding["sink"],
                "purpose": binding["purpose"],
                "parameters_digest": binding["parameters_digest"],
                "payload_labels": labels.clone()
            });
            if event.get("authority_decision").is_some() {
                event["authority_decision"]["exact_effect_digest"] =
                    json!(digest_value(AUTHORITY_DOMAIN, &authority_projection));
            }

            if event.get("disclosure_decision").is_some() {
                let disclosure_projection = json!({
                    "sink": binding["sink"],
                    "purpose": binding["purpose"],
                    "payload_digest": binding["payload_digest"],
                    "payload_labels": labels.clone(),
                    "lineage_proof_ref": binding.get("lineage_proof_ref").cloned().unwrap_or(Value::Null),
                    "interaction_intent_ref": binding.get("interaction_intent_ref").cloned().unwrap_or(Value::Null),
                    "lineage_evidence_refs": event["disclosure_decision"]["lineage_evidence_refs"]
                });
                event["disclosure_decision"]["exact_disclosure_digest"] =
                    json!(digest_value(DISCLOSURE_DOMAIN, &disclosure_projection));
            }

            if event.get("effect_observation").is_some()
                && event["effect_observation"]["outcome"] == "observed_applied"
            {
                let observed_projection = json!({
                    "projection_id": event["effect_observation"]["sink_projection_id"],
                    "projection_version": event["effect_observation"]["sink_projection_version"],
                    "projection": event["effect_observation"]["observed_projection"]
                });
                event["effect_observation"]["observed_effect_digest"] =
                    json!(digest_value(SINK_PROJECTION_DOMAIN, &observed_projection));
            }

            let mut binding_for_hash = event["effect_binding"].clone();
            binding_for_hash
                .as_object_mut()
                .unwrap()
                .remove("binding_digest");
            binding_for_hash
                .as_object_mut()
                .unwrap()
                .remove("attempt_id");
            event["effect_binding"]["binding_digest"] =
                json!(digest_value(BINDING_DOMAIN, &binding_for_hash));
        }

        event["integrity"]["event_digest"] =
            json!("sha256:0000000000000000000000000000000000000000000000000000000000000000");
        let raw: StrictJson =
            serde_json::from_value(event.clone()).expect("strict event value");
        let integrity = raw
            .get("integrity")
            .unwrap()
            .without_key("event_digest")
            .unwrap();
        let raw = replace_nested(raw, "integrity", integrity).unwrap();
        event["integrity"]["event_digest"] = json!(digest_json(EVENT_DOMAIN, &raw));
        event
    }

    fn reseal_event_digest(mut event: Value) -> Value {
        event["integrity"]["event_digest"] =
            json!("sha256:0000000000000000000000000000000000000000000000000000000000000000");
        let raw: StrictJson =
            serde_json::from_value(event.clone()).expect("strict event value");
        let integrity = raw
            .get("integrity")
            .unwrap()
            .without_key("event_digest")
            .unwrap();
        let raw = replace_nested(raw, "integrity", integrity).unwrap();
        event["integrity"]["event_digest"] = json!(digest_json(EVENT_DOMAIN, &raw));
        event
    }

    fn sample_event(
        sequence: u64,
        event_id: &str,
        event_type: &str,
        previous_digest: Option<&str>,
        observation_outcome: Option<&str>,
        parent_ids: Vec<&str>,
    ) -> Value {
        let labels = sorted(&["finance.transaction"]);
        let sink_projection = json!({
            "action": "transfer",
            "source_account": "acct-source",
            "recipient": "acct-recipient",
            "amount_minor_units": 1250,
            "currency": "USD",
            "transaction_reference": "invoice-42",
            "idempotency_key": "pay-001"
        });
        let mut event = json!({
            "schema_version": "sat-event-record/v0.1",
            "run_id": RUN_ID,
            "manifest_digest": MANIFEST,
            "event_id": event_id,
            "sequence": sequence,
            "occurred_at": "2026-11-25T10:00:00Z",
            "event_type": event_type,
            "actor": {"principal_id":"agent-1","role":"agent","component_id":"agent-runtime"},
            "causality": {"correlation_id":"payment-1","parent_event_ids":parent_ids},
            "evidence_refs": [],
            "integrity": {
                "hash_algorithm":"sha256-jcs-v1",
                "previous_event_digest": previous_digest,
                "event_digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000"
            }
        });

        if event_type == "authorization_decided" || event_type == "effect_submitted" {
            event["effect_binding"] = json!({
                "action":"transfer",
                "resource":"acct-source",
                "sink":"bank-ledger",
                "purpose":"approved-invoice",
                "parameters_digest":"sha256:1111111111111111111111111111111111111111111111111111111111111111",
                "payload_digest":"sha256:2222222222222222222222222222222222222222222222222222222222222222",
                "payload_labels":labels,
                "binding_digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "sink_projection_digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "sink_projection_id":"bank-ledger.transfer-record",
                "sink_projection_version":"1",
                "sink_projection":sink_projection,
                "lineage_proof_ref":"lineage-1",
                "interaction_intent_ref":"intent-1",
                "idempotency_key":"pay-001","attempt_id":"attempt-0001"
            });
            event["authority_decision"] = json!({
                "decision":"allow",
                "policy_digest":MANIFEST,
                "exact_effect_digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "grant_refs":["grant-1"],
                "decision_receipt_ref":"auth-receipt-1",
                "valid_until":"2026-11-25T10:05:00Z"
            });
            event["disclosure_decision"] = json!({
                "decision":"allow",
                "policy_digest":MANIFEST,
                "exact_disclosure_digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "lineage_evidence_refs":["lineage-evidence-1"]
            });
        } else if event_type == "effect_observed" || event_type == "effect_reconciled" {
            let outcome = observation_outcome.unwrap_or("unknown");
            event["actor"] =
                json!({"principal_id":"sink-1","role":"sink_observer","component_id":"bank-readback"});
            event["effect_binding"] = json!({
                "action":"transfer",
                "resource":"acct-source",
                "sink":"bank-ledger",
                "purpose":"approved-invoice",
                "parameters_digest":"sha256:1111111111111111111111111111111111111111111111111111111111111111",
                "payload_digest":"sha256:2222222222222222222222222222222222222222222222222222222222222222",
                "payload_labels":labels,
                "binding_digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "sink_projection_digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "sink_projection_id":"bank-ledger.transfer-record",
                "sink_projection_version":"1",
                "sink_projection":sink_projection,
                "lineage_proof_ref":"lineage-1",
                "interaction_intent_ref":"intent-1",
                "idempotency_key":"pay-001","attempt_id":"attempt-0001"
            });
            let mut observation = json!({
                "outcome":outcome,
                "observer_id":"sink-1",
                "source_of_truth":"bank-ledger-readback",
                "observed_at":"2026-11-25T10:00:01Z",
                "state_version":"v-2",
                "idempotency_key":"pay-001","attempt_id":"attempt-0001",
                "sink_projection_id":"bank-ledger.transfer-record",
                "sink_projection_version":"1"
            });
            if event_type == "effect_reconciled" {
                observation["reconciliation_ref"] = json!("reconciliation-read-1");
            }
            if outcome == "observed_applied" {
                observation["observed_projection"] = sink_projection.clone();
                observation["observed_effect_digest"] =
                    json!("sha256:0000000000000000000000000000000000000000000000000000000000000000");
            }
            event["effect_observation"] = observation;
            event["evidence_refs"] = json!([{
                "artifact_id":"bank-read-1",
                "digest":MANIFEST,
                "source_id":"bank-ledger-read-api",
                "capture_method":"independent_api_read",
                "independently_observed":true
            }]);
        }
        finish_event(event)
    }

    fn with_attempt_id(mut event: Value, attempt_id: &str) -> Value {
        event["effect_binding"]["attempt_id"] = json!(attempt_id);
        if event.get("effect_observation").is_some() {
            event["effect_observation"]["attempt_id"] = json!(attempt_id);
        }
        finish_event(event)
    }

    fn trace(events: Vec<Value>) -> String {
        events
            .into_iter()
            .map(|event| event.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn event_digest(event: &Value) -> String {
        let mut clone = event.clone();
        clone["integrity"]
            .as_object_mut()
            .unwrap()
            .remove("event_digest");
        digest_value(EVENT_DOMAIN, &clone)
    }

    #[test]
    fn unanchored_trace_reports_consistency_without_claiming_tamper_evidence() {
        let event = sample_event(1, "event-1", "authorization_decided", None, None, vec![]);
        let report = verify_trace(&trace(vec![event]), RUN_ID, MANIFEST).unwrap();
        assert!(report.event_chain_consistent);
        assert!(!report.head_digest_matches_supplied_anchor);
        assert!(!report.signatures_verified);
    }

    #[test]
    fn supplied_head_digest_anchor_is_compared_but_not_authenticated() {
        let event = sample_event(1, "event-1", "authorization_decided", None, None, vec![]);
        let head = event_digest(&event);
        let report = verify_trace_with_anchor(&trace(vec![event.clone()]), RUN_ID, MANIFEST, &head).unwrap();
        assert!(report.head_digest_matches_supplied_anchor);
        assert_eq!(report.head_digest, head);
        assert!(!report.signatures_verified);

        let wrong = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let result = verify_trace_with_anchor(&trace(vec![event]), RUN_ID, MANIFEST, wrong);
        assert!(matches!(result.unwrap_err().code, VerificationErrorCode::HeadDigestAnchorMismatch));
    }

    #[test]
    fn committed_cross_language_example_verifies_end_to_end() {
        const MANIFEST_JSON: &str = include_str!(
            "../../../../docs/research/examples/sat-trace-verifier-manifest-v0.1.json"
        );
        const TRACE_JSONL: &str = include_str!(
            "../../../../docs/research/examples/sat-trace-verifier-trace-v0.1.jsonl"
        );
        let manifest_digest = compute_manifest_digest(MANIFEST_JSON).unwrap();
        assert_eq!(
            manifest_digest,
            "sha256:0fa06d577488566420fea82078b2f30fd70280af9363e26f558bcac66bb32f2d"
        );
        let manifest: Value = serde_json::from_str(MANIFEST_JSON).unwrap();
        let run_id = manifest["run_id"].as_str().unwrap();
        let report = verify_trace(TRACE_JSONL, run_id, &manifest_digest).unwrap();
        assert_eq!(report.event_count, 2);
        assert_eq!(report.authority_bindings_checked, 1);
        assert_eq!(report.disclosure_bindings_checked, 1);
        assert_eq!(report.sink_projections_checked, 2);
        assert_eq!(report.effect_submissions, 1);
        assert_eq!(report.terminal_effect_observations, 1);
        assert!(!report.hold_required);
        assert!(!report.signatures_verified);
        assert!(!report.system_qualified);
    }

    #[test]
    fn manifest_digest_is_order_independent_and_rejects_duplicate_keys() {
        let first = r#"{"z":2,"a":{"b":1,"a":0}}"#;
        let reordered = r#"{"a":{"a":0,"b":1},"z":2}"#;
        assert_eq!(
            compute_manifest_digest(first).unwrap(),
            compute_manifest_digest(reordered).unwrap()
        );
        assert!(matches!(
            compute_manifest_digest(r#"{"run_id":"a","run_id":"b"}"#)
                .unwrap_err()
                .code,
            VerificationErrorCode::DuplicateJsonKey(_)
        ));
    }

    #[test]
    fn canonicalizer_sorts_recursively_and_preserves_array_order() {
        let raw: StrictJson =
            serde_json::from_str(r#"{"z":1,"a":{"b":2,"a":3},"arr":[{"z":0,"a":1}]}"#)
                .unwrap();
        assert_eq!(
            String::from_utf8(canonicalize(&raw)).unwrap(),
            r#"{"a":{"a":3,"b":2},"arr":[{"a":1,"z":0}],"z":1}"#
        );
    }

    #[test]
    fn canonicalizer_uses_utf16_property_order() {
        let raw: StrictJson = serde_json::from_str(r#"{"\uE000":1,"😀":2}"#).unwrap();
        assert_eq!(
            String::from_utf8(canonicalize(&raw)).unwrap(),
            r#"{"😀":2,"":1}"#
        );
    }

    #[test]
    fn duplicate_object_keys_are_rejected_before_map_conversion() {
        let error = serde_json::from_str::<StrictJson>(r#"{"a":1,"a":2}"#).unwrap_err();
        assert!(error.to_string().contains("duplicate JSON key"));
    }

    #[test]
    fn non_integer_and_unsafe_integer_numbers_are_rejected() {
        assert!(serde_json::from_str::<StrictJson>(r#"{"n":1.5}"#).is_err());
        assert!(serde_json::from_str::<StrictJson>(r#"{"n":9007199254740992}"#).is_err());
    }

    #[test]
    fn valid_authorization_event_verifies_commitments() {
        let event = sample_event(1, "event-1", "authorization_decided", None, None, vec![]);
        let result = verify_trace(&trace(vec![event]), RUN_ID, MANIFEST).unwrap();
        assert_eq!(result.event_count, 1);
        assert_eq!(result.authority_bindings_checked, 1);
        assert_eq!(result.disclosure_bindings_checked, 1);
        assert_eq!(result.sink_projections_checked, 1);
        assert!(!result.signatures_verified);
        assert!(!result.system_qualified);
    }

    #[test]
    fn changed_manifest_digest_is_rejected() {
        let event = sample_event(1, "event-1", "authorization_decided", None, None, vec![]);
        let wrong = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let result = verify_trace(&trace(vec![event]), RUN_ID, wrong);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::ManifestDigestMismatch
        ));
    }

    #[test]
    fn required_previous_digest_and_evidence_array_cannot_be_omitted() {
        let mut missing_previous = sample_event(1, "event-1", "authorization_decided", None, None, vec![]);
        missing_previous["integrity"].as_object_mut().unwrap().remove("previous_event_digest");
        let result = verify_trace(&trace(vec![missing_previous]), RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::InvalidEventShape(_)
        ));

        let mut missing_evidence = sample_event(1, "event-1", "authorization_decided", None, None, vec![]);
        missing_evidence.as_object_mut().unwrap().remove("evidence_refs");
        let result = verify_trace(&trace(vec![missing_evidence]), RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::InvalidEventShape(_)
        ));
    }

    #[test]
    fn event_sequence_gap_is_rejected() {
        let first = sample_event(1, "event-1", "authorization_decided", None, None, vec![]);
        let second = sample_event(
            3,
            "event-3",
            "authorization_decided",
            Some(&event_digest(&first)),
            None,
            vec!["event-1"],
        );
        let result = verify_trace(&trace(vec![first, second]), RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::SequenceMismatch { .. }
        ));
    }

    #[test]
    fn previous_event_digest_is_required_for_chain_continuity() {
        let first = sample_event(1, "event-1", "authorization_decided", None, None, vec![]);
        let second = sample_event(
            2,
            "event-2",
            "authorization_decided",
            Some(MANIFEST),
            None,
            vec!["event-1"],
        );
        let result = verify_trace(&trace(vec![first, second]), RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::PreviousDigestMismatch
        ));
    }

    #[test]
    fn causal_parent_must_already_exist() {
        let event = sample_event(1, "event-1", "authorization_decided", None, None, vec!["future"]);
        let result = verify_trace(&trace(vec![event]), RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::CausalParentNotPrior(_)
        ));
    }

    #[test]
    fn event_digest_tampering_is_rejected() {
        let mut event = sample_event(1, "event-1", "authorization_decided", None, None, vec![]);
        event["actor"]["component_id"] = json!("tampered-runtime");
        let result = verify_trace(&trace(vec![event]), RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::EventDigestMismatch
        ));
    }

    #[test]
    fn authority_parameter_rebinding_is_rejected() {
        let mut event = finish_event(sample_event(
            1,
            "event-1",
            "authorization_decided",
            None,
            None,
            vec![],
        ));
        event["authority_decision"]["exact_effect_digest"] =
            json!("sha256:9999999999999999999999999999999999999999999999999999999999999999");
        let result = verify_trace(&trace(vec![reseal_event_digest(event)]), RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::AuthorityDigestMismatch
        ));
    }

    #[test]
    fn disclosure_payload_rebinding_is_rejected() {
        let mut event = finish_event(sample_event(
            1,
            "event-1",
            "authorization_decided",
            None,
            None,
            vec![],
        ));
        event["disclosure_decision"]["exact_disclosure_digest"] =
            json!("sha256:9999999999999999999999999999999999999999999999999999999999999999");
        let result = verify_trace(&trace(vec![reseal_event_digest(event)]), RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::DisclosureDigestMismatch
        ));
    }

    #[test]
    fn sink_observation_must_match_named_versioned_projection() {
        let submission = sample_event(1, "event-1", "effect_submitted", None, None, vec![]);
        let digest = event_digest(&submission);
        let observed = sample_event(
            2,
            "event-2",
            "effect_observed",
            Some(&digest),
            Some("observed_applied"),
            vec!["event-1"],
        );
        let result = verify_trace(&trace(vec![submission, observed]), RUN_ID, MANIFEST).unwrap();
        assert_eq!(result.terminal_effect_observations, 1);
        assert!(!result.hold_required);
    }

    #[test]
    fn unknown_outcome_blocks_new_mutation_until_reconciled() {
        let submission = sample_event(1, "event-1", "effect_submitted", None, None, vec![]);
        let digest = event_digest(&submission);
        let unknown = sample_event(
            2,
            "event-2",
            "effect_observed",
            Some(&digest),
            Some("unknown"),
            vec!["event-1"],
        );
        let unknown_digest = event_digest(&unknown);
        let next_mutation = sample_event(
            3,
            "event-3",
            "effect_submitted",
            Some(&unknown_digest),
            None,
            vec!["event-2"],
        );
        let result = verify_trace(&trace(vec![submission, unknown, next_mutation]), RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::UnresolvedOutcomeBlocksNewMutation(_)
        ));
    }

    #[test]
    fn reconciliation_of_unknown_outcome_allows_same_intent_retry() {
        let submission = sample_event(1, "event-1", "effect_submitted", None, None, vec![]);
        let digest = event_digest(&submission);
        let unknown = sample_event(
            2,
            "event-2",
            "effect_observed",
            Some(&digest),
            Some("unknown"),
            vec!["event-1"],
        );
        let unknown_digest = event_digest(&unknown);
        let reconciled = sample_event(
            3,
            "event-3",
            "effect_reconciled",
            Some(&unknown_digest),
            Some("observed_not_applied"),
            vec!["event-2"],
        );
        let reconciled_digest = event_digest(&reconciled);
        let second_submission = with_attempt_id(
            sample_event(
                4,
                "event-4",
                "effect_submitted",
                Some(&reconciled_digest),
                None,
                vec!["event-3"],
            ),
            "attempt-0002",
        );
        let result = verify_trace(
            &trace(vec![submission, unknown, reconciled, second_submission]),
            RUN_ID,
            MANIFEST,
        )
        .unwrap();
        assert_eq!(result.effect_submissions, 2);
        assert!(!result.hold_required);
    }

    #[test]
    fn exact_intent_digest_remains_stable_across_new_attempt_ids() {
        let first = sample_event(1, "event-1", "effect_submitted", None, None, vec![]);
        let second = with_attempt_id(
            sample_event(
                2,
                "event-2",
                "effect_submitted",
                Some(&event_digest(&first)),
                None,
                vec!["event-1"],
            ),
            "attempt-0002",
        );
        assert_eq!(
            first["effect_binding"]["binding_digest"],
            second["effect_binding"]["binding_digest"]
        );
        assert_ne!(
            first["effect_binding"]["attempt_id"],
            second["effect_binding"]["attempt_id"]
        );
    }

    #[test]
    fn late_observation_for_previous_attempt_cannot_settle_retry() {
        let first_submission = sample_event(1, "event-1", "effect_submitted", None, None, vec![]);
        let first_digest = event_digest(&first_submission);
        let first_unknown = sample_event(
            2,
            "event-2",
            "effect_observed",
            Some(&first_digest),
            Some("unknown"),
            vec!["event-1"],
        );
        let unknown_digest = event_digest(&first_unknown);
        let reconciled = sample_event(
            3,
            "event-3",
            "effect_reconciled",
            Some(&unknown_digest),
            Some("observed_not_applied"),
            vec!["event-2"],
        );
        let reconciled_digest = event_digest(&reconciled);
        let retry = with_attempt_id(
            sample_event(
                4,
                "event-4",
                "effect_submitted",
                Some(&reconciled_digest),
                None,
                vec!["event-3"],
            ),
            "attempt-0002",
        );
        let retry_digest = event_digest(&retry);
        let delayed_old_observation = sample_event(
            5,
            "event-5",
            "effect_observed",
            Some(&retry_digest),
            Some("observed_applied"),
            vec!["event-4"],
        );
        let result = verify_trace(
            &trace(vec![
                first_submission,
                first_unknown,
                reconciled,
                retry,
                delayed_old_observation,
            ]),
            RUN_ID,
            MANIFEST,
        );
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::StaleAttemptObservation { .. }
        ));
    }

    #[test]
    fn observation_must_reference_latest_attempt_event() {
        let submission = sample_event(1, "event-1", "effect_submitted", None, None, vec![]);
        let submitted_digest = event_digest(&submission);
        let intermediate = sample_event(
            2,
            "event-2",
            "proposal_created",
            Some(&submitted_digest),
            None,
            vec!["event-1"],
        );
        let intermediate_digest = event_digest(&intermediate);
        let observed = sample_event(
            3,
            "event-3",
            "effect_observed",
            Some(&intermediate_digest),
            Some("observed_applied"),
            vec!["event-2"],
        );
        let result = verify_trace(
            &trace(vec![submission, intermediate, observed]),
            RUN_ID,
            MANIFEST,
        );
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::EffectCausalityMismatch(_)
        ));
    }

    #[test]
    fn duplicate_terminal_observation_is_rejected() {
        let submission = sample_event(1, "event-1", "effect_submitted", None, None, vec![]);
        let submitted_digest = event_digest(&submission);
        let applied = sample_event(
            2,
            "event-2",
            "effect_observed",
            Some(&submitted_digest),
            Some("observed_applied"),
            vec!["event-1"],
        );
        let applied_digest = event_digest(&applied);
        let duplicate = sample_event(
            3,
            "event-3",
            "effect_reconciled",
            Some(&applied_digest),
            Some("observed_applied"),
            vec!["event-2"],
        );
        let result = verify_trace(&trace(vec![submission, applied, duplicate]), RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::DuplicateTerminalObservation(_)
        ));
    }

    #[test]
    fn unresolved_outcome_is_reported_as_hold_not_safety_pass() {
        let submission = sample_event(1, "event-1", "effect_submitted", None, None, vec![]);
        let digest = event_digest(&submission);
        let unknown = sample_event(
            2,
            "event-2",
            "effect_observed",
            Some(&digest),
            Some("unknown"),
            vec!["event-1"],
        );
        let result = verify_trace(&trace(vec![submission, unknown]), RUN_ID, MANIFEST).unwrap();
        assert!(result.hold_required);
        assert_eq!(result.unresolved_idempotency_keys, vec!["pay-001"]);
        assert!(!result.system_qualified);
    }

    #[test]
    fn altered_sink_projection_is_rejected_even_when_its_own_digest_is_valid() {
        let submission = sample_event(1, "event-1", "effect_submitted", None, None, vec![]);
        let previous = event_digest(&submission);
        let mut observed = finish_event(sample_event(
            2,
            "event-2",
            "effect_observed",
            Some(&previous),
            Some("observed_applied"),
            vec!["event-1"],
        ));
        observed["effect_observation"]["observed_projection"]["recipient"] =
            json!("attacker-account");
        let observed = reseal_event_digest(observed);
        let result = verify_trace(&trace(vec![submission, observed]), RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::ObservedProjectionMismatch
                | VerificationErrorCode::ObservedEffectDigestMismatch
        ));
    }

    #[test]
    fn commit_at_authority_expiry_is_rejected() {
        let mut event = sample_event(1, "event-1", "effect_submitted", None, None, vec![]);
        event["occurred_at"] = json!("2026-11-25T10:05:00Z");
        let event = finish_event(event);
        let result = verify_trace(&trace(vec![event]), RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::SubmissionWithoutCurrentAllows
        ));
    }

    #[test]
    fn empty_trace_is_not_a_successful_verification() {
        let result = verify_trace("", RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::EmptyTrace
        ));
    }

    #[test]
    fn duplicate_json_keys_are_rejected_by_public_verifier() {
        let result = verify_trace(
            r#"{"schema_version":"sat-event-record/v0.1","run_id":"x","run_id":"y"}"#,
            RUN_ID,
            MANIFEST,
        );
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::DuplicateJsonKey(_)
        ));
    }

    #[test]
    fn duplicate_submission_after_applied_observation_is_rejected() {
        let first = sample_event(1, "event-1", "effect_submitted", None, None, vec![]);
        let digest = event_digest(&first);
        let observed = sample_event(
            2,
            "event-2",
            "effect_observed",
            Some(&digest),
            Some("observed_applied"),
            vec!["event-1"],
        );
        let observed_digest = event_digest(&observed);
        let replay = sample_event(
            3,
            "event-3",
            "effect_submitted",
            Some(&observed_digest),
            None,
            vec!["event-2"],
        );
        let result = verify_trace(&trace(vec![first, observed, replay]), RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::DuplicateAppliedIdempotencyKey(_)
                | VerificationErrorCode::DuplicateAttemptId(_)
        ));
    }

    #[test]
    fn not_applicable_disclosure_requires_explicit_opt_in() {
        let mut event = sample_event(1, "event-1", "effect_submitted", None, None, vec![]);
        event["disclosure_decision"]["decision"] = json!("not_applicable");
        event["disclosure_decision"]["not_applicable_reason"] =
            json!("No data crosses a disclosure boundary in this fixture.");
        let event = finish_event(event);
        let result = verify_trace(&trace(vec![event.clone()]), RUN_ID, MANIFEST);
        assert!(matches!(
            result.unwrap_err().code,
            VerificationErrorCode::NotApplicableDisclosureNotAllowed
        ));
        let result = verify_trace_with_options(
            &trace(vec![event]),
            RUN_ID,
            MANIFEST,
            VerificationOptions {
                allow_not_applicable_disclosure: true,
            },
        )
        .unwrap();
        assert_eq!(result.effect_submissions, 1);
    }
}
