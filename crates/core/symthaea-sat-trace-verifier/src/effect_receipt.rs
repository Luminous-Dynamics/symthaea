// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Read-only structural reconciliation for external effect receipts.
//!
//! This module first binds the attempt to evaluator-owned, independently
//! pinned context, then binds every source assertion to that exact run,
//! manifest, attempt, idempotency key, intent and effect binding. It compares
//! the observed sink projection to the requested effect. It deliberately does
//! NOT verify signatures, key status, transparency proofs, source
//! independence or physical/business truth. It never authorizes retries.

use super::StrictJson;
use serde::Deserialize;
use std::collections::HashSet;
use std::fmt;

const EFFECT_BINDING_DOMAIN: &str = "SAT-EFFECT-ATTEMPT-BINDING-v0.1\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectOutcome {
    Applied,
    NotApplied,
    Partial,
    Unknown,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectCorrespondence {
    Exact,
    Mismatch,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceOutcomeAgreement {
    Agree,
    Conflict,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationDisposition {
    HoldForAuthenticatedReconciliation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectReceiptError {
    InvalidJson(String),
    DuplicateJsonKey(String),
    InvalidShape(String),
    InvalidDigest(String),
    ExpectedContextMismatch(&'static str),
    AttemptBindingMismatch(&'static str),
    SourceBindingMismatch {
        attestation_id: String,
        field: &'static str,
    },
    DuplicateAttestationId(String),
    DuplicateSourceEventId(String),
}

impl fmt::Display for EffectReceiptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(reason) => write!(f, "invalid effect receipt JSON: {reason}"),
            Self::DuplicateJsonKey(key) => write!(f, "duplicate JSON object key: {key}"),
            Self::InvalidShape(reason) => write!(f, "invalid effect receipt shape: {reason}"),
            Self::InvalidDigest(field) => write!(f, "invalid digest in {field}"),
            Self::ExpectedContextMismatch(field) => {
                write!(f, "attempt does not match evaluator-pinned context field {field}")
            }
            Self::AttemptBindingMismatch(field) => {
                write!(f, "receipt does not bind to attempt field {field}")
            }
            Self::SourceBindingMismatch { attestation_id, field } => {
                write!(f, "source attestation {attestation_id} does not bind to {field}")
            }
            Self::DuplicateAttestationId(id) => {
                write!(f, "duplicate source attestation id: {id}")
            }
            Self::DuplicateSourceEventId(id) => {
                write!(f, "duplicate source event id: {id}")
            }
        }
    }
}

impl std::error::Error for EffectReceiptError {}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EffectAttempt {
    schema_version: String,
    run_id: String,
    manifest_digest: String,
    attempt_id: String,
    idempotency_key: String,
    intent_digest: String,
    authority_decision_digest: String,
    disclosure_decision_digest: String,
    effect_binding: EffectBinding,
    #[serde(default)]
    effect_class: Option<String>,
    #[serde(default)]
    submitted_at: Option<String>,
    #[serde(default)]
    attempt_context_ref: Option<String>,
    #[serde(default)]
    limitations: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EffectBinding {
    sink_id: String,
    action: String,
    resource_id: String,
    purpose: String,
    parameter_digest: String,
    payload_digest: String,
    labels: Vec<String>,
    binding_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EffectReceipt {
    schema_version: String,
    receipt_id: String,
    issued_at: String,
    run_id: String,
    manifest_digest: String,
    attempt_id: String,
    idempotency_key: String,
    intent_digest: String,
    effect_binding_digest: String,
    observed_outcome: EffectOutcome,
    #[serde(default)]
    observed_effect: Option<ObservedEffect>,
    source_attestations: Vec<SourceAttestation>,
    trust_policy_digest: String,
    #[serde(default)]
    trust_policy_ref: Option<String>,
    receipt_statement_digest: String,
    #[serde(default)]
    transparency_receipt_ref: Option<String>,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    limitations: Vec<String>,
    #[serde(default)]
    privacy_profile: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservedEffect {
    projection_schema: String,
    sink_id: String,
    operation: String,
    resource_id: String,
    purpose: String,
    parameter_digest: String,
    payload_digest: String,
    external_effect_id_digest: String,
    external_version: String,
    projection_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceAttestation {
    attestation_id: String,
    run_id: String,
    manifest_digest: String,
    attempt_id: String,
    idempotency_key: String,
    intent_digest: String,
    effect_binding_digest: String,
    source_id: String,
    source_version: String,
    source_role: String,
    source_event_id: String,
    source_sequence: u64,
    signed_at: String,
    observed_at: String,
    outcome: EffectOutcome,
    #[serde(deserialize_with = "deserialize_required_nullable_string")]
    projection_digest: Option<String>,
    signature_envelope: SignatureEnvelope,
    key_status_evidence_ref: String,
    timestamp_evidence_ref: String,
    trust_policy_digest: String,
    independence_basis_ref: String,
    #[serde(default)]
    operator_identity_ref: Option<String>,
    #[serde(default)]
    infrastructure_domain_ref: Option<String>,
    #[serde(default)]
    conflicts_of_interest_ref: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SignatureEnvelope {
    format: String,
    algorithm: String,
    signer_identity: String,
    key_id: String,
    signed_statement_digest: String,
    statement_ref: String,
    signature_ref: String,
}

/// Evaluator-owned identity/policy pins. These values MUST come from outside
/// the subject-controlled attempt and receipt documents (for example, a
/// previously frozen run manifest and evaluator trust-policy store).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedEffectAttempt<'a> {
    pub run_id: &'a str,
    pub manifest_digest: &'a str,
    pub attempt_id: &'a str,
    pub idempotency_key: &'a str,
    pub intent_digest: &'a str,
    pub authority_decision_digest: &'a str,
    pub disclosure_decision_digest: &'a str,
    pub effect_binding_digest: &'a str,
    pub trust_policy_digest: &'a str,
}

/// A structural result, not an authenticated or qualified effect receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectReceiptVerificationReport {
    pub receipt_id: String,
    pub run_id: String,
    pub attempt_id: String,
    pub source_attestation_count: usize,
    pub attempt_identity_matches: bool,
    pub pinned_context_matches_attempt: bool,
    pub all_source_attestations_bind_to_attempt: bool,
    pub effect_correspondence: EffectCorrespondence,
    pub source_outcome_agreement: SourceOutcomeAgreement,
    pub observed_outcome: EffectOutcome,
    pub hold_required: bool,
    /// Always false until a concrete signature/key-status adapter is executed.
    pub signatures_verified: bool,
    /// Always false until prior checkpoint pins and inclusion/consistency proofs are verified.
    pub transparency_verified: bool,
    /// Always false until a pinned common-cause policy is evaluated.
    pub source_independence_verified: bool,
    /// Always false: this module never authorizes a retry or mutation.
    pub retry_authorized: bool,
    /// Always false: structural correspondence is not overall system qualification.
    pub system_qualified: bool,
    pub mutation_disposition: MutationDisposition,
}

/// Verify strict JSON structure, evaluator-pinned attempt binding and sink projection correspondence.
///
/// `expected` MUST be supplied from evaluator-controlled frozen state, not copied
/// out of the subject's attempt or receipt. Without this external pin, a subject
/// could rewrite both documents together and preserve an internally consistent pair.
///
/// This function has no network, clock, key-store or mutation capability. A report
/// with exact structural bindings still requires authenticated source evidence,
/// transparency continuity and independence analysis before it can support a
/// qualification claim.
pub fn verify_effect_receipt(
    attempt_json: &str,
    receipt_json: &str,
    expected: &ExpectedEffectAttempt<'_>,
) -> Result<EffectReceiptVerificationReport, EffectReceiptError> {
    let attempt_raw: StrictJson = serde_json::from_str(attempt_json).map_err(map_json_error)?;
    let _receipt_raw: StrictJson = serde_json::from_str(receipt_json).map_err(map_json_error)?;
    let attempt: EffectAttempt =
        serde_json::from_str(attempt_json).map_err(|e| EffectReceiptError::InvalidJson(e.to_string()))?;
    let receipt: EffectReceipt =
        serde_json::from_str(receipt_json).map_err(|e| EffectReceiptError::InvalidJson(e.to_string()))?;

    validate_expected_context(expected)?;
    validate_attempt_shape(&attempt)?;
    validate_effect_binding_digest(&attempt_raw, &attempt)?;
    validate_receipt_shape(&receipt)?;
    bind_attempt_to_expected(&attempt, expected)?;
    if receipt.trust_policy_digest != expected.trust_policy_digest {
        return Err(EffectReceiptError::ExpectedContextMismatch("trust_policy_digest"));
    }
    bind_receipt_to_attempt(&attempt, &receipt)?;

    if receipt.source_attestations.is_empty() {
        return Err(EffectReceiptError::InvalidShape(
            "at least one source attestation is required".to_owned(),
        ));
    }

    let mut attestation_ids = HashSet::new();
    let mut source_event_ids = HashSet::new();
    for source in &receipt.source_attestations {
        if !attestation_ids.insert(source.attestation_id.as_str()) {
            return Err(EffectReceiptError::DuplicateAttestationId(
                source.attestation_id.clone(),
            ));
        }
        if !source_event_ids.insert(source.source_event_id.as_str()) {
            return Err(EffectReceiptError::DuplicateSourceEventId(
                source.source_event_id.clone(),
            ));
        }
        bind_source_to_attempt(&attempt, &receipt, source)?;
        validate_source_shape(source)?;
    }

    let effect_correspondence = match (&receipt.observed_outcome, &receipt.observed_effect) {
        (EffectOutcome::Applied | EffectOutcome::Partial, Some(observed)) => {
            if observed.sink_id == attempt.effect_binding.sink_id
                && observed.operation == attempt.effect_binding.action
                && observed.resource_id == attempt.effect_binding.resource_id
                && observed.purpose == attempt.effect_binding.purpose
                && observed.parameter_digest == attempt.effect_binding.parameter_digest
                && observed.payload_digest == attempt.effect_binding.payload_digest
            {
                EffectCorrespondence::Exact
            } else {
                EffectCorrespondence::Mismatch
            }
        }
        (EffectOutcome::Applied | EffectOutcome::Partial, None) => {
            return Err(EffectReceiptError::InvalidShape(
                "applied or partial outcome requires observed_effect".to_owned(),
            ));
        }
        (EffectOutcome::NotApplied | EffectOutcome::Unknown | EffectOutcome::Rejected, Some(_)) => {
            return Err(EffectReceiptError::InvalidShape(
                "non-applied, unknown or rejected outcome cannot carry an applied projection".to_owned(),
            ));
        }
        _ => EffectCorrespondence::NotApplicable,
    };

    if matches!(receipt.observed_outcome, EffectOutcome::Applied | EffectOutcome::Partial) {
        let observed_projection = receipt
            .observed_effect
            .as_ref()
            .map(|projection| projection.projection_digest.as_str());
        let matching_source_projection = receipt.source_attestations.iter().any(|source| {
            source.projection_digest.as_deref() == observed_projection
                && source.outcome == receipt.observed_outcome
        });
        if !matching_source_projection {
            return Err(EffectReceiptError::InvalidShape(
                "no source attestation commits to the exact observed projection and outcome".to_owned(),
            ));
        }
    }

    let source_outcome_agreement = if receipt.source_attestations.is_empty() {
        SourceOutcomeAgreement::Incomplete
    } else if receipt.source_attestations.iter().all(|source| {
        source.outcome == receipt.observed_outcome
    }) {
        SourceOutcomeAgreement::Agree
    } else if receipt.source_attestations.iter().any(|source| {
        source.outcome != receipt.observed_outcome && source.outcome != EffectOutcome::Unknown
    }) {
        SourceOutcomeAgreement::Conflict
    } else {
        SourceOutcomeAgreement::Incomplete
    };

    // Even a structurally matched "not_applied" receipt cannot authorize retry:
    // signatures, source independence, freshness and trust policy are not checked here.
    let _metadata_is_not_authentication = (
        &receipt.receipt_id,
        &receipt.issued_at,
        &receipt.trust_policy_ref,
        &receipt.transparency_receipt_ref,
        &receipt.artifact_refs,
        &receipt.limitations,
        &receipt.privacy_profile,
        &attempt.authority_decision_digest,
        &attempt.disclosure_decision_digest,
        &attempt.effect_class,
        &attempt.submitted_at,
        &attempt.attempt_context_ref,
        &attempt.limitations,
        &attempt.effect_binding.labels,
        receipt.observed_effect.as_ref().map(|observed| (
            &observed.projection_schema,
            &observed.external_effect_id_digest,
            &observed.external_version,
        )),
        receipt.source_attestations.iter().map(|source| (
            &source.source_id,
            &source.source_version,
            &source.source_role,
            source.source_sequence,
            &source.signed_at,
            &source.observed_at,
            &source.key_status_evidence_ref,
            &source.timestamp_evidence_ref,
            &source.independence_basis_ref,
            &source.operator_identity_ref,
            &source.infrastructure_domain_ref,
            &source.conflicts_of_interest_ref,
            &source.signature_envelope.format,
            &source.signature_envelope.algorithm,
            &source.signature_envelope.signer_identity,
            &source.signature_envelope.key_id,
            &source.signature_envelope.signed_statement_digest,
            &source.signature_envelope.statement_ref,
            &source.signature_envelope.signature_ref,
        )).collect::<Vec<_>>(),
    );

    Ok(EffectReceiptVerificationReport {
        receipt_id: receipt.receipt_id,
        run_id: receipt.run_id,
        attempt_id: receipt.attempt_id,
        source_attestation_count: receipt.source_attestations.len(),
        attempt_identity_matches: true,
        pinned_context_matches_attempt: true,
        all_source_attestations_bind_to_attempt: true,
        effect_correspondence,
        source_outcome_agreement,
        observed_outcome: receipt.observed_outcome,
        // This structural module always holds for authenticated reconciliation.
        hold_required: true,
        signatures_verified: false,
        transparency_verified: false,
        source_independence_verified: false,
        retry_authorized: false,
        system_qualified: false,
        mutation_disposition: MutationDisposition::HoldForAuthenticatedReconciliation,
    })
}

fn deserialize_required_nullable_string<'de, D>(
    deserializer: D,
) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer)
}

fn validate_timestamp(value: &str, field: &'static str) -> Result<(), EffectReceiptError> {
    if super::parse_timestamp(value).is_some() {
        Ok(())
    } else {
        Err(EffectReceiptError::InvalidShape(format!(
            "{field} must be RFC 3339"
        )))
    }
}

fn map_json_error(error: serde_json::Error) -> EffectReceiptError {
    let message = error.to_string();
    if let Some(key) = message.split("duplicate JSON key: ").nth(1) {
        EffectReceiptError::DuplicateJsonKey(key.to_owned())
    } else {
        EffectReceiptError::InvalidJson(message)
    }
}

fn is_canonical_uuid(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 36
        && bytes[8] == b'-'
        && bytes[13] == b'-'
        && bytes[18] == b'-'
        && bytes[23] == b'-'
        && bytes.iter().enumerate().all(|(index, byte)| {
            matches!(index, 8 | 13 | 18 | 23) || byte.is_ascii_hexdigit()
        })
}

fn validate_digest(value: &str, field: &'static str) -> Result<(), EffectReceiptError> {
    let valid = value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    });
    if valid {
        Ok(())
    } else {
        Err(EffectReceiptError::InvalidDigest(field))
    }
}

fn validate_attempt_shape(attempt: &EffectAttempt) -> Result<(), EffectReceiptError> {
    if attempt.schema_version != "sat-effect-attempt/v0.1" {
        return Err(EffectReceiptError::InvalidShape("unsupported attempt schema_version".to_owned()));
    }
    if !is_canonical_uuid(&attempt.run_id) || !is_canonical_uuid(&attempt.attempt_id) {
        return Err(EffectReceiptError::InvalidShape("attempt run_id and attempt_id must be UUIDs".to_owned()));
    }
    if attempt.idempotency_key.len() < 8 || attempt.idempotency_key.len() > 256 {
        return Err(EffectReceiptError::InvalidShape("attempt idempotency_key length is invalid".to_owned()));
    }
    super::require_sorted_unique(&attempt.effect_binding.labels, "effect_binding.labels")
        .map_err(|error| EffectReceiptError::InvalidShape(error.to_string()))?;
    if let Some(submitted_at) = &attempt.submitted_at {
        validate_timestamp(submitted_at, "attempt.submitted_at")?;
    }
    for (value, field) in [
        (attempt.manifest_digest.as_str(), "attempt.manifest_digest"),
        (attempt.intent_digest.as_str(), "attempt.intent_digest"),
        (attempt.authority_decision_digest.as_str(), "attempt.authority_decision_digest"),
        (attempt.disclosure_decision_digest.as_str(), "attempt.disclosure_decision_digest"),
        (attempt.effect_binding.parameter_digest.as_str(), "attempt.effect_binding.parameter_digest"),
        (attempt.effect_binding.payload_digest.as_str(), "attempt.effect_binding.payload_digest"),
        (attempt.effect_binding.binding_digest.as_str(), "attempt.effect_binding.binding_digest"),
    ] {
        validate_digest(value, field)?;
    }
    Ok(())
}

fn validate_effect_binding_digest(
    attempt_raw: &StrictJson,
    attempt: &EffectAttempt,
) -> Result<(), EffectReceiptError> {
    let binding_without_digest = attempt_raw
        .get("effect_binding")
        .and_then(|binding| binding.without_key("binding_digest"))
        .ok_or_else(|| {
            EffectReceiptError::InvalidShape(
                "attempt.effect_binding must be an object".to_owned(),
            )
        })?;
    if super::digest_json(EFFECT_BINDING_DOMAIN, &binding_without_digest)
        != attempt.effect_binding.binding_digest
    {
        return Err(EffectReceiptError::InvalidDigest(
            "attempt.effect_binding.binding_digest",
        ));
    }
    Ok(())
}

fn validate_receipt_shape(receipt: &EffectReceipt) -> Result<(), EffectReceiptError> {
    if receipt.schema_version != "sat-effect-receipt/v0.1" {
        return Err(EffectReceiptError::InvalidShape("unsupported receipt schema_version".to_owned()));
    }
    if !is_canonical_uuid(&receipt.run_id) || !is_canonical_uuid(&receipt.attempt_id) {
        return Err(EffectReceiptError::InvalidShape("receipt run_id and attempt_id must be UUIDs".to_owned()));
    }
    if receipt.idempotency_key.len() < 8 || receipt.idempotency_key.len() > 256 {
        return Err(EffectReceiptError::InvalidShape("receipt idempotency_key length is invalid".to_owned()));
    }
    validate_timestamp(&receipt.issued_at, "receipt.issued_at")?;
    for (value, field) in [
        (receipt.manifest_digest.as_str(), "receipt.manifest_digest"),
        (receipt.intent_digest.as_str(), "receipt.intent_digest"),
        (receipt.effect_binding_digest.as_str(), "receipt.effect_binding_digest"),
        (receipt.trust_policy_digest.as_str(), "receipt.trust_policy_digest"),
        (receipt.receipt_statement_digest.as_str(), "receipt.receipt_statement_digest"),
    ] {
        validate_digest(value, field)?;
    }
    Ok(())
}

fn validate_expected_context(
    expected: &ExpectedEffectAttempt<'_>,
) -> Result<(), EffectReceiptError> {
    if !is_canonical_uuid(expected.run_id) || !is_canonical_uuid(expected.attempt_id) {
        return Err(EffectReceiptError::InvalidShape(
            "evaluator-pinned run_id and attempt_id must be UUIDs".to_owned(),
        ));
    }
    if expected.idempotency_key.len() < 8 || expected.idempotency_key.len() > 256 {
        return Err(EffectReceiptError::InvalidShape(
            "evaluator-pinned idempotency_key length is invalid".to_owned(),
        ));
    }
    for (value, field) in [
        (expected.manifest_digest, "expected.manifest_digest"),
        (expected.intent_digest, "expected.intent_digest"),
        (expected.authority_decision_digest, "expected.authority_decision_digest"),
        (expected.disclosure_decision_digest, "expected.disclosure_decision_digest"),
        (expected.effect_binding_digest, "expected.effect_binding_digest"),
        (expected.trust_policy_digest, "expected.trust_policy_digest"),
    ] {
        validate_digest(value, field)?;
    }
    Ok(())
}

fn bind_attempt_to_expected(
    attempt: &EffectAttempt,
    expected: &ExpectedEffectAttempt<'_>,
) -> Result<(), EffectReceiptError> {
    let bindings = [
        (attempt.run_id == expected.run_id, "run_id"),
        (attempt.manifest_digest == expected.manifest_digest, "manifest_digest"),
        (attempt.attempt_id == expected.attempt_id, "attempt_id"),
        (attempt.idempotency_key == expected.idempotency_key, "idempotency_key"),
        (attempt.intent_digest == expected.intent_digest, "intent_digest"),
        (attempt.authority_decision_digest == expected.authority_decision_digest, "authority_decision_digest"),
        (attempt.disclosure_decision_digest == expected.disclosure_decision_digest, "disclosure_decision_digest"),
        (attempt.effect_binding.binding_digest == expected.effect_binding_digest, "effect_binding_digest"),
    ];
    for (matches, field) in bindings {
        if !matches {
            return Err(EffectReceiptError::ExpectedContextMismatch(field));
        }
    }
    Ok(())
}

fn bind_receipt_to_attempt(
    attempt: &EffectAttempt,
    receipt: &EffectReceipt,
) -> Result<(), EffectReceiptError> {
    for (matches, field) in [
        (receipt.run_id == attempt.run_id, "run_id"),
        (receipt.manifest_digest == attempt.manifest_digest, "manifest_digest"),
        (receipt.attempt_id == attempt.attempt_id, "attempt_id"),
        (receipt.idempotency_key == attempt.idempotency_key, "idempotency_key"),
        (receipt.intent_digest == attempt.intent_digest, "intent_digest"),
        (receipt.effect_binding_digest == attempt.effect_binding.binding_digest, "effect_binding_digest"),
    ] {
        if !matches {
            return Err(EffectReceiptError::AttemptBindingMismatch(field));
        }
    }
    Ok(())
}

fn bind_source_to_attempt(
    attempt: &EffectAttempt,
    receipt: &EffectReceipt,
    source: &SourceAttestation,
) -> Result<(), EffectReceiptError> {
    let bindings = [
        (source.run_id == attempt.run_id, "run_id"),
        (source.manifest_digest == attempt.manifest_digest, "manifest_digest"),
        (source.attempt_id == attempt.attempt_id, "attempt_id"),
        (source.idempotency_key == attempt.idempotency_key, "idempotency_key"),
        (source.intent_digest == attempt.intent_digest, "intent_digest"),
        (source.effect_binding_digest == attempt.effect_binding.binding_digest, "effect_binding_digest"),
        (source.trust_policy_digest == receipt.trust_policy_digest, "trust_policy_digest"),
    ];
    for (matches, field) in bindings {
        if !matches {
            return Err(EffectReceiptError::SourceBindingMismatch {
                attestation_id: source.attestation_id.clone(),
                field,
            });
        }
    }
    Ok(())
}

fn validate_source_shape(source: &SourceAttestation) -> Result<(), EffectReceiptError> {
    if !is_canonical_uuid(&source.run_id) || !is_canonical_uuid(&source.attempt_id) {
        return Err(EffectReceiptError::InvalidShape("source run_id and attempt_id must be UUIDs".to_owned()));
    }
    if source.idempotency_key.len() < 8 || source.idempotency_key.len() > 256 {
        return Err(EffectReceiptError::InvalidShape("source idempotency_key length is invalid".to_owned()));
    }
    validate_timestamp(&source.signed_at, "source.signed_at")?;
    validate_timestamp(&source.observed_at, "source.observed_at")?;
    for (value, field) in [
        (source.manifest_digest.as_str(), "source.manifest_digest"),
        (source.intent_digest.as_str(), "source.intent_digest"),
        (source.effect_binding_digest.as_str(), "source.effect_binding_digest"),
        (source.trust_policy_digest.as_str(), "source.trust_policy_digest"),
        (source.signature_envelope.signed_statement_digest.as_str(), "source.signed_statement_digest"),
    ] {
        validate_digest(value, field)?;
    }
    if let Some(digest) = &source.projection_digest {
        validate_digest(digest, "source.projection_digest")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ATTEMPT: &str = include_str!(
        "../../../../docs/research/examples/sat-effect-attempt-example-v0.1.json"
    );
    const RECEIPT: &str = include_str!(
        "../../../../docs/research/examples/sat-effect-receipt-example-v0.1.json"
    );

    fn expected_context() -> ExpectedEffectAttempt<'static> {
        ExpectedEffectAttempt {
            run_id: "00000000-0000-4000-8000-000000000001",
            manifest_digest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            attempt_id: "00000000-0000-4000-8000-000000000101",
            idempotency_key: "idem:payment:case-001",
            intent_digest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            authority_decision_digest: "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            disclosure_decision_digest: "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            effect_binding_digest: "sha256:558cfd260809c2c99a32da1019c3bd360e4b2aca56f26f5008cf02144a0e3077",
            trust_policy_digest: "sha256:3333333333333333333333333333333333333333333333333333333333333333",
        }
    }

    #[test]
    fn synthetic_example_is_structurally_bound_but_not_authenticated() {
        let report = verify_effect_receipt(ATTEMPT, RECEIPT, &expected_context()).unwrap();
        assert!(report.attempt_identity_matches);
        assert!(report.pinned_context_matches_attempt);
        assert!(report.all_source_attestations_bind_to_attempt);
        assert_eq!(report.effect_correspondence, EffectCorrespondence::Exact);
        assert_eq!(report.source_outcome_agreement, SourceOutcomeAgreement::Agree);
        assert_eq!(report.source_attestation_count, 2);
        assert!(report.hold_required);
        assert!(!report.signatures_verified);
        assert!(!report.transparency_verified);
        assert!(!report.source_independence_verified);
        assert!(!report.retry_authorized);
        assert!(!report.system_qualified);
    }

    #[test]
    fn source_attestation_cannot_be_replayed_across_attempts() {
        let replayed = RECEIPT.replace(
            "\"attempt_id\": \"00000000-0000-4000-8000-000000000101\"",
            "\"attempt_id\": \"00000000-0000-4000-8000-000000000102\"",
        );
        assert!(matches!(
            verify_effect_receipt(ATTEMPT, &replayed, &expected_context()),
            Err(EffectReceiptError::AttemptBindingMismatch("attempt_id"))
                | Err(EffectReceiptError::SourceBindingMismatch { field: "attempt_id", .. })
        ));
    }

    #[test]
    fn wrong_external_resource_is_reported_as_effect_mismatch() {
        let changed = RECEIPT.replace(
            "account:recipient-opaque-17",
            "account:unauthorized-recipient",
        );
        let report = verify_effect_receipt(ATTEMPT, &changed, &expected_context()).unwrap();
        assert_eq!(report.effect_correspondence, EffectCorrespondence::Mismatch);
        assert!(report.hold_required);
        assert!(!report.system_qualified);
    }

    #[test]
    fn jointly_rewritten_attempt_and_receipt_cannot_escape_external_run_pin() {
        let rewritten_attempt = ATTEMPT.replace(
            "00000000-0000-4000-8000-000000000001",
            "00000000-0000-4000-8000-000000000002",
        );
        let rewritten_receipt = RECEIPT.replace(
            "00000000-0000-4000-8000-000000000001",
            "00000000-0000-4000-8000-000000000002",
        );
        assert!(matches!(
            verify_effect_receipt(&rewritten_attempt, &rewritten_receipt, &expected_context()),
            Err(EffectReceiptError::ExpectedContextMismatch("run_id"))
        ));
    }

    #[test]
    fn changed_effect_fields_without_recomputed_binding_digest_are_rejected() {
        let changed = ATTEMPT.replace(
            "account:recipient-opaque-17",
            "account:attacker",
        );
        assert!(matches!(
            verify_effect_receipt(&changed, RECEIPT, &expected_context()),
            Err(EffectReceiptError::InvalidDigest(
                "attempt.effect_binding.binding_digest"
            ))
        ));
    }

    #[test]
    fn unsorted_effect_labels_are_rejected() {
        let changed = ATTEMPT.replace(
            "[\\n      \\"approved-payment\\",\\n      \\"financial-data\\"\\n    ]",
            "[\\n      \\"financial-data\\",\\n      \\"approved-payment\\"\\n    ]",
        );
        assert!(matches!(
            verify_effect_receipt(&changed, RECEIPT, &expected_context()),
            Err(EffectReceiptError::InvalidShape(_))
        ));
    }

    #[test]
    fn receipt_cannot_substitute_its_own_trust_policy() {
        let substituted = RECEIPT.replace(
            "3333333333333333333333333333333333333333333333333333333333333333",
            "9999999999999999999999999999999999999999999999999999999999999999",
        );
        assert!(matches!(
            verify_effect_receipt(ATTEMPT, &substituted, &expected_context()),
            Err(EffectReceiptError::ExpectedContextMismatch("trust_policy_digest"))
        ));
    }

    #[test]
    fn duplicate_json_keys_are_rejected_before_typed_deserialization() {
        let duplicate = ATTEMPT.replace(
            "\"run_id\":",
            "\"run_id\": \"00000000-0000-4000-8000-000000000002\", \"run_id\":",
        );
        assert!(matches!(
            verify_effect_receipt(&duplicate, RECEIPT, &expected_context()),
            Err(EffectReceiptError::DuplicateJsonKey(_))
        ));
    }
}
