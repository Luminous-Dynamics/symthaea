// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Verifies exactly which memory-derived byte spans crossed the serialized
//! client/model request boundary. This is exposure evidence, not proof of
//! neural attention, semantic reliance, or causal influence.

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use ed25519_dalek::{Signature, VerifyingKey};
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const MACE_CONTEXT_EXPOSURE_SCHEMA_V1: &str = "mace_context_exposure_receipt_v0_1";
pub const MACE_CONTEXT_EXPOSURE_DOMAIN_V1: &[u8] = b"MACE-CONTEXT-EXPOSURE-V1\0";
pub const MACE_CONTEXT_EXPOSURE_MAX_RECEIPT_BYTES_V1: usize = 4 * 1024 * 1024;
pub const MACE_CONTEXT_EXPOSURE_MAX_BODY_BYTES_V1: usize = 32 * 1024 * 1024;
pub const MACE_CONTEXT_EXPOSURE_MAX_SPANS_V1: usize = 100_000;

/// Generic provider parameters are deliberately limited to static numeric
/// decoding/budget controls and boolean switches. Content-bearing or
/// provider-specific structured fields require a typed adapter.
pub const MACE_STATIC_PROVIDER_NUMBER_PARAMETER_KEYS_V1: &[&str] = &[
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "max_tokens",
    "max_completion_tokens",
    "seed",
    "n",
    "top_logprobs",
];
pub const MACE_STATIC_PROVIDER_BOOLEAN_PARAMETER_KEYS_V1: &[&str] = &[
    "stream",
    "logprobs",
    "parallel_tool_calls",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaceMemoryArmV1 {
    NoMemory, ExactScopeScan, ConventionalRetrieval, StaticHdc, HdcLtc,
    SemanticPlacebo, PreCorrectionSnapshot, PostCorrectionSnapshot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaceContextRoleV1 { System, Developer, User, Assistant, Tool }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaceExposureKindV1 {
    VerbatimSource, DeterministicDerivative, ModelGeneratedDerivative,
    CachedModelOutput, ToolOutputProjection,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaceAdmissionDenialReasonV1 {
    WrongScope, Unauthorized, UntrustedSource, InvalidLineage, Retracted,
    Revoked, Superseded, StaleAtUse, RightsRestricted, CommonCauseUnresolved,
    DigestMismatch, UnknownEvidence,
}

/// Commitment to an immutable encrypted request-body object. The resolver
/// must verify ciphertext/custody and decrypt under the pinned access policy.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaceRequestBodyArtifactV1 {
    pub artifact_id: String,
    pub plaintext_sha256: String,
    pub ciphertext_sha256: String,
    pub plaintext_bytes: u64,
    pub storage_profile_sha256: String,
    pub encryption_profile_sha256: String,
    pub access_policy_sha256: String,
    pub retention_policy_sha256: String,
    pub custody_receipt_sha256: String,
    pub immutable_object: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaceContextCaptureV1 {
    pub capture_point: String,
    pub request_media_type: String,
    pub provider_request_body_sha256: String,
    pub provider_request_body_bytes: u64,
    pub complete_provider_request_body: bool,
    pub complete_memory_origin_enumeration: bool,
    pub untracked_memory_origin_span_count: u64,
    pub request_body_artifact: MaceRequestBodyArtifactV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaceMemoryExposureSpanV1 {
    pub exposure_id: String,
    pub evidence_id: String,
    pub evidence_payload_sha256: String,
    pub exposed_span_sha256: String,
    pub span_start_byte: u64,
    pub span_end_byte: u64,
    pub context_role: MaceContextRoleV1,
    pub message_index: u64,
    pub exposure_kind: MaceExposureKindV1,
    pub parent_event_ids: Vec<String>,
    pub dependency_closure_sha256: String,
    pub transformation_receipt_sha256: String,
    pub source_receipt_sha256: String,
    pub authority_receipt_sha256: String,
    pub admissibility_receipt_sha256: String,
    pub lifecycle_revalidation_receipt_sha256: String,
    pub admission_status: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaceExposureAdmissionDenialV1 {
    pub evidence_id: String,
    pub reason: MaceAdmissionDenialReasonV1,
    pub admission_receipt_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaceExposureResponseBindingV1 {
    pub provider_response_id: String,
    pub provider_response_body_sha256: String,
    pub candidate_decision_trace_sha256: String,
    #[serde(default)]
    pub model_declared_support_event_ids: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaceExposureSignatureV1 {
    pub algorithm: String,
    pub key_id: String,
    pub signature_base64: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaceContextExposureReceiptV1 {
    pub schema_version: String,
    pub record_type: String,
    pub receipt_id: String,
    pub run_id: String,
    pub pair_block_id: String,
    pub world_cluster_id: String,
    pub query_id: String,
    pub arm: MaceMemoryArmV1,
    pub invocation_sequence: u64,
    pub decision_need_sha256: String,
    pub candidate_policy_sha256: String,
    pub source_registry_root_sha256: String,
    pub lineage_profile_sha256: String,
    pub lifecycle_profile_sha256: String,
    pub renderer_profile_sha256: String,
    pub transport_adapter_profile_sha256: String,
    pub provider_parameter_profile_sha256: String,
    pub model_revision: String,
    pub capture: MaceContextCaptureV1,
    pub exposures: Vec<MaceMemoryExposureSpanV1>,
    pub admission_denials: Vec<MaceExposureAdmissionDenialV1>,
    pub response_binding: MaceExposureResponseBindingV1,
    pub captured_at_monotonic_ns: u64,
    pub clock_profile_sha256: String,
    pub observer_profile_sha256: String,
    pub signature: MaceExposureSignatureV1,
}

/// Expected values must come from the independent run manifest/configuration,
/// not from the receipt or candidate process.
pub struct MaceExpectedExposureBindingV1<'a> {
    pub run_id: &'a str,
    pub pair_block_id: &'a str,
    pub world_cluster_id: &'a str,
    pub query_id: &'a str,
    pub arm: MaceMemoryArmV1,
    pub invocation_sequence: u64,
    pub decision_need_sha256: &'a str,
    pub candidate_policy_sha256: &'a str,
    pub source_registry_root_sha256: &'a str,
    pub lineage_profile_sha256: &'a str,
    pub lifecycle_profile_sha256: &'a str,
    pub renderer_profile_sha256: &'a str,
    pub transport_adapter_profile_sha256: &'a str,
    pub provider_parameter_profile_sha256: &'a str,
    pub model_revision: &'a str,
    pub clock_profile_sha256: &'a str,
    pub observer_profile_sha256: &'a str,
    pub storage_profile_sha256: &'a str,
    pub encryption_profile_sha256: &'a str,
    pub access_policy_sha256: &'a str,
    pub retention_policy_sha256: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaceContextExposureRejectV1 {
    ReceiptTooLarge, InvalidJson, WrongSchema, WrongRecordType, EmptyIdentifier,
    WrongRun, WrongPairBlock, WrongWorldCluster, WrongQuery, WrongArm,
    WrongInvocationSequence, WrongDecisionNeed, WrongCandidatePolicy,
    WrongSourceRegistry, WrongLineageProfile, WrongLifecycleProfile,
    WrongRendererProfile, WrongTransportProfile, WrongProviderParameterProfile, WrongModelRevision,
    WrongClockProfile, WrongObserverProfile, InvalidDigest, InvalidCapturePoint,
    InvalidMediaType, IncompleteRequestCapture, IncompleteOriginEnumeration,
    UntrackedOriginSpans, InvalidArtifactReference, ArtifactCustodyMismatch,
    ArtifactResolutionFailed, BodyTooLarge, BodyDigestMismatch,
    BodyLengthMismatch, TooManySpans, InvalidSpanRange, SpanDigestMismatch,
    OverlappingSpans, DuplicateExposureId, DuplicateParentId,
    InvalidAdmissionStatus, DuplicateDeniedEvidence, DeniedEvidenceExposed,
    InvalidResponseBinding, InvalidSignatureAlgorithm, InvalidSignatureEncoding,
    InvalidObserverKeyId, UnknownObserverKey, SignatureMismatch,
    InvalidRequestBody, UnclassifiedProviderParameterKey,
    ProviderParameterTypeMismatch, InvalidSpanEncoding, ExposureOutsideMessageContent,
    ExposureRoleMismatch,
    EvidenceAdmissionRejected, MemoryExposureInNoMemoryArm,
    ObserverSigningFailed, CanonicalizationFailed,
}

/// Fetches an immutable encrypted request-body object, verifies its ciphertext
/// and custody receipts, decrypts under the authorized policy, and returns the
/// exact plaintext bytes. This is an integration trait, not a new storage owner.
pub trait MaceRequestBodyArtifactResolverV1 {
    fn resolve_and_verify(
        &self,
        artifact: &MaceRequestBodyArtifactV1,
    ) -> Result<Vec<u8>, MaceContextExposureRejectV1>;
}

/// Resolves a key from the independently pinned observer roster/profile.
pub trait MaceExposureObserverKeyResolverV1 {
    fn resolve_key(
        &self,
        observer_profile_sha256: &str,
        key_id: &str,
    ) -> Result<VerifyingKey, MaceContextExposureRejectV1>;
}

/// Delegates canonical evidence-ID, lineage, source, authority, rights, scope,
/// estimand and lifecycle checks to the existing MACE owners.
pub trait MaceExposureAdmissionVerifierV1 {
    fn verify_admitted_exposure(
        &self,
        exposure: &MaceMemoryExposureSpanV1,
        expected: &MaceExpectedExposureBindingV1<'_>,
    ) -> Result<(), MaceContextExposureRejectV1>;
}

/// Issued only after receipt, artifact, byte ranges, signature and evidence
/// admission all verify. It intentionally does not retain raw request bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedMaceContextExposureReceiptV1 {
    receipt: MaceContextExposureReceiptV1,
}

impl VerifiedMaceContextExposureReceiptV1 {
    pub fn receipt(&self) -> &MaceContextExposureReceiptV1 { &self.receipt }
    pub fn request_body_sha256(&self) -> &str {
        &self.receipt.capture.provider_request_body_sha256
    }
    pub fn artifact_id(&self) -> &str {
        &self.receipt.capture.request_body_artifact.artifact_id
    }
    pub fn exposed_evidence_ids(&self) -> impl Iterator<Item = &str> {
        self.receipt.exposures.iter().map(|span| span.evidence_id.as_str())
    }
}

pub fn parse_and_verify_mace_context_exposure_receipt_v1(
    wire: &[u8],
    expected: &MaceExpectedExposureBindingV1<'_>,
    artifact_resolver: &impl MaceRequestBodyArtifactResolverV1,
    key_resolver: &impl MaceExposureObserverKeyResolverV1,
    admission_verifier: &impl MaceExposureAdmissionVerifierV1,
) -> Result<VerifiedMaceContextExposureReceiptV1, MaceContextExposureRejectV1> {
    if wire.len() > MACE_CONTEXT_EXPOSURE_MAX_RECEIPT_BYTES_V1 {
        return Err(MaceContextExposureRejectV1::ReceiptTooLarge);
    }
    // deny_unknown_fields plus typed deserialization rejects duplicate members.
    let receipt: MaceContextExposureReceiptV1 =
        serde_json::from_slice(wire).map_err(|_| MaceContextExposureRejectV1::InvalidJson)?;

    verify_receipt_bindings(&receipt, expected)?;
    verify_receipt_shapes(&receipt)?;
    // Authenticate the signed envelope before touching artifact storage or
    // invoking canonical evidence/authority lookups on attacker-supplied IDs.
    verify_observer_signature(&receipt, expected, key_resolver)?;

    let artifact = &receipt.capture.request_body_artifact;
    if artifact.storage_profile_sha256 != expected.storage_profile_sha256
        || artifact.encryption_profile_sha256 != expected.encryption_profile_sha256
        || artifact.access_policy_sha256 != expected.access_policy_sha256
        || artifact.retention_policy_sha256 != expected.retention_policy_sha256
    {
        return Err(MaceContextExposureRejectV1::ArtifactCustodyMismatch);
    }

    let body = artifact_resolver.resolve_and_verify(artifact)
        .map_err(|_| MaceContextExposureRejectV1::ArtifactResolutionFailed)?;
    if body.len() > MACE_CONTEXT_EXPOSURE_MAX_BODY_BYTES_V1 {
        return Err(MaceContextExposureRejectV1::BodyTooLarge);
    }
    if body.len() as u64 != receipt.capture.provider_request_body_bytes
        || body.len() as u64 != artifact.plaintext_bytes
    {
        return Err(MaceContextExposureRejectV1::BodyLengthMismatch);
    }
    let body_digest = sha256_hex(&body);
    if body_digest != receipt.capture.provider_request_body_sha256
        || body_digest != artifact.plaintext_sha256
    {
        return Err(MaceContextExposureRejectV1::BodyDigestMismatch);
    }

    verify_provider_request_contract(&receipt, &body, expected)?;
    verify_span_coverage(&receipt, &body)?;
    for exposure in &receipt.exposures {
        admission_verifier.verify_admitted_exposure(exposure, expected)
            .map_err(|_| MaceContextExposureRejectV1::EvidenceAdmissionRejected)?;
    }
    Ok(VerifiedMaceContextExposureReceiptV1 { receipt })
}


#[derive(Deserialize)]
struct CapturedProviderRequestV1<'a> {
    #[serde(borrow)]
    model: &'a str,
    #[serde(borrow)]
    messages: Vec<CapturedProviderMessageV1<'a>>,
    #[serde(flatten)]
    parameters: CapturedProviderParametersV1,
}

struct CapturedProviderParametersV1(BTreeMap<String, serde_json::Value>);

impl<'de> Deserialize<'de> for CapturedProviderParametersV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ParametersVisitor;

        impl<'de> Visitor<'de> for ParametersVisitor {
            type Value = CapturedProviderParametersV1;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a provider parameter object with unique keys")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut parameters = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
                    if parameters.contains_key(&key) {
                        return Err(de::Error::custom("duplicate provider parameter key"));
                    }
                    parameters.insert(key, value);
                }
                Ok(CapturedProviderParametersV1(parameters))
            }
        }

        deserializer.deserialize_map(ParametersVisitor)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapturedProviderMessageV1<'a> {
    #[serde(borrow)]
    role: &'a str,
    #[serde(borrow)]
    content: &'a RawValue,
}

fn context_role_name_v1(role: MaceContextRoleV1) -> &'static str {
    match role {
        MaceContextRoleV1::System => "system",
        MaceContextRoleV1::Developer => "developer",
        MaceContextRoleV1::User => "user",
        MaceContextRoleV1::Assistant => "assistant",
        MaceContextRoleV1::Tool => "tool",
    }
}

/// Independently parses the captured provider request. It recomputes the
/// provider-parameter profile from the exact body and proves each claimed
/// exposure range lies inside the indexed message's serialized content string.
fn verify_provider_request_contract(
    receipt: &MaceContextExposureReceiptV1,
    body: &[u8],
    expected: &MaceExpectedExposureBindingV1<'_>,
) -> Result<(), MaceContextExposureRejectV1> {
    let request: CapturedProviderRequestV1<'_> =
        serde_json::from_slice(body).map_err(|_| MaceContextExposureRejectV1::InvalidRequestBody)?;

    if request.model != expected.model_revision {
        return Err(MaceContextExposureRejectV1::WrongModelRevision);
    }

    for (key, value) in &request.parameters.0 {
        if MACE_STATIC_PROVIDER_NUMBER_PARAMETER_KEYS_V1.contains(&key.as_str()) {
            if !value.is_number() {
                return Err(MaceContextExposureRejectV1::ProviderParameterTypeMismatch);
            }
        } else if MACE_STATIC_PROVIDER_BOOLEAN_PARAMETER_KEYS_V1.contains(&key.as_str()) {
            if !value.is_boolean() {
                return Err(MaceContextExposureRejectV1::ProviderParameterTypeMismatch);
            }
        } else {
            return Err(MaceContextExposureRejectV1::UnclassifiedProviderParameterKey);
        }
    }
    let parameter_bytes = serde_json::to_vec(&request.parameters.0)
        .map_err(|_| MaceContextExposureRejectV1::InvalidRequestBody)?;
    let parameter_profile = sha256_hex(&parameter_bytes);
    if parameter_profile != expected.provider_parameter_profile_sha256
        || parameter_profile != receipt.provider_parameter_profile_sha256
    {
        return Err(MaceContextExposureRejectV1::WrongProviderParameterProfile);
    }

    if request.messages.is_empty() {
        return Err(MaceContextExposureRejectV1::InvalidRequestBody);
    }
    let mut content_ranges = Vec::with_capacity(request.messages.len());
    for message in &request.messages {
        if !matches!(message.role, "system" | "developer" | "user" | "assistant" | "tool") {
            return Err(MaceContextExposureRejectV1::InvalidRequestBody);
        }
        let _: String = serde_json::from_str(message.content.get())
            .map_err(|_| MaceContextExposureRejectV1::InvalidRequestBody)?;
        let raw = message.content.get().as_bytes();
        if raw.len() < 2 || raw.first() != Some(&b'"') || raw.last() != Some(&b'"') {
            return Err(MaceContextExposureRejectV1::InvalidRequestBody);
        }
        let body_base = body.as_ptr() as usize;
        let raw_base = raw.as_ptr() as usize;
        let raw_offset = raw_base
            .checked_sub(body_base)
            .ok_or(MaceContextExposureRejectV1::InvalidRequestBody)?;
        let content_start = raw_offset
            .checked_add(1)
            .ok_or(MaceContextExposureRejectV1::InvalidRequestBody)?;
        let content_end = raw_offset
            .checked_add(raw.len() - 1)
            .ok_or(MaceContextExposureRejectV1::InvalidRequestBody)?;
        if content_end > body.len() || content_start > content_end {
            return Err(MaceContextExposureRejectV1::InvalidRequestBody);
        }
        content_ranges.push((content_start as u64, content_end as u64, message.role));
    }

    for exposure in &receipt.exposures {
        let index = usize::try_from(exposure.message_index)
            .map_err(|_| MaceContextExposureRejectV1::InvalidRequestBody)?;
        let (content_start, content_end, actual_role) = content_ranges
            .get(index)
            .copied()
            .ok_or(MaceContextExposureRejectV1::ExposureOutsideMessageContent)?;
        if actual_role != context_role_name_v1(exposure.context_role) {
            return Err(MaceContextExposureRejectV1::ExposureRoleMismatch);
        }
        if exposure.span_start_byte < content_start
            || exposure.span_end_byte > content_end
            || exposure.span_start_byte >= exposure.span_end_byte
        {
            return Err(MaceContextExposureRejectV1::ExposureOutsideMessageContent);
        }
    }
    Ok(())
}

fn verify_receipt_bindings(
    receipt: &MaceContextExposureReceiptV1,
    expected: &MaceExpectedExposureBindingV1<'_>,
) -> Result<(), MaceContextExposureRejectV1> {
    if receipt.schema_version != MACE_CONTEXT_EXPOSURE_SCHEMA_V1 {
        return Err(MaceContextExposureRejectV1::WrongSchema);
    }
    if receipt.record_type != "model_context_exposure" {
        return Err(MaceContextExposureRejectV1::WrongRecordType);
    }
    for value in [
        receipt.receipt_id.as_str(), receipt.run_id.as_str(),
        receipt.pair_block_id.as_str(), receipt.world_cluster_id.as_str(),
        receipt.query_id.as_str(), receipt.capture.request_body_artifact.artifact_id.as_str(),
        receipt.response_binding.provider_response_id.as_str(),
    ] {
        validate_identifier(value, 256)?;
    }
    validate_identifier(&receipt.model_revision, 512)?;
    validate_identifier(&receipt.capture.request_media_type, 128)?;
    if let Some(request_id) = receipt.capture.provider_request_id.as_deref() {
        validate_identifier(request_id, 256)?;
    }
    if receipt.run_id != expected.run_id { return Err(MaceContextExposureRejectV1::WrongRun); }
    if receipt.pair_block_id != expected.pair_block_id { return Err(MaceContextExposureRejectV1::WrongPairBlock); }
    if receipt.world_cluster_id != expected.world_cluster_id { return Err(MaceContextExposureRejectV1::WrongWorldCluster); }
    if receipt.query_id != expected.query_id { return Err(MaceContextExposureRejectV1::WrongQuery); }
    if receipt.arm != expected.arm { return Err(MaceContextExposureRejectV1::WrongArm); }
    if receipt.invocation_sequence != expected.invocation_sequence { return Err(MaceContextExposureRejectV1::WrongInvocationSequence); }
    if receipt.decision_need_sha256 != expected.decision_need_sha256 { return Err(MaceContextExposureRejectV1::WrongDecisionNeed); }
    if receipt.candidate_policy_sha256 != expected.candidate_policy_sha256 { return Err(MaceContextExposureRejectV1::WrongCandidatePolicy); }
    if receipt.source_registry_root_sha256 != expected.source_registry_root_sha256 { return Err(MaceContextExposureRejectV1::WrongSourceRegistry); }
    if receipt.lineage_profile_sha256 != expected.lineage_profile_sha256 { return Err(MaceContextExposureRejectV1::WrongLineageProfile); }
    if receipt.lifecycle_profile_sha256 != expected.lifecycle_profile_sha256 { return Err(MaceContextExposureRejectV1::WrongLifecycleProfile); }
    if receipt.renderer_profile_sha256 != expected.renderer_profile_sha256 { return Err(MaceContextExposureRejectV1::WrongRendererProfile); }
    if receipt.transport_adapter_profile_sha256 != expected.transport_adapter_profile_sha256 { return Err(MaceContextExposureRejectV1::WrongTransportProfile); }
    if receipt.provider_parameter_profile_sha256 != expected.provider_parameter_profile_sha256 { return Err(MaceContextExposureRejectV1::WrongProviderParameterProfile); }
    if receipt.model_revision != expected.model_revision { return Err(MaceContextExposureRejectV1::WrongModelRevision); }
    if receipt.clock_profile_sha256 != expected.clock_profile_sha256 { return Err(MaceContextExposureRejectV1::WrongClockProfile); }
    if receipt.observer_profile_sha256 != expected.observer_profile_sha256 { return Err(MaceContextExposureRejectV1::WrongObserverProfile); }
    if receipt.capture.capture_point != "serialized_request_before_network_send" {
        return Err(MaceContextExposureRejectV1::InvalidCapturePoint);
    }
    if receipt.capture.request_media_type.trim().is_empty() {
        return Err(MaceContextExposureRejectV1::InvalidMediaType);
    }
    if !receipt.capture.complete_provider_request_body {
        return Err(MaceContextExposureRejectV1::IncompleteRequestCapture);
    }
    if !receipt.capture.complete_memory_origin_enumeration {
        return Err(MaceContextExposureRejectV1::IncompleteOriginEnumeration);
    }
    if receipt.capture.untracked_memory_origin_span_count != 0 {
        return Err(MaceContextExposureRejectV1::UntrackedOriginSpans);
    }
    if receipt.capture.provider_request_body_bytes == 0 {
        return Err(MaceContextExposureRejectV1::BodyLengthMismatch);
    }
    if !receipt.capture.request_body_artifact.immutable_object {
        return Err(MaceContextExposureRejectV1::InvalidArtifactReference);
    }
    if receipt.arm == MaceMemoryArmV1::NoMemory
        && (!receipt.exposures.is_empty() || !receipt.admission_denials.is_empty())
    {
        return Err(MaceContextExposureRejectV1::MemoryExposureInNoMemoryArm);
    }
    Ok(())
}

fn verify_receipt_shapes(
    receipt: &MaceContextExposureReceiptV1,
) -> Result<(), MaceContextExposureRejectV1> {
    for digest in [
        receipt.decision_need_sha256.as_str(), receipt.candidate_policy_sha256.as_str(),
        receipt.source_registry_root_sha256.as_str(), receipt.lineage_profile_sha256.as_str(),
        receipt.lifecycle_profile_sha256.as_str(), receipt.renderer_profile_sha256.as_str(),
        receipt.transport_adapter_profile_sha256.as_str(), receipt.provider_parameter_profile_sha256.as_str(), receipt.clock_profile_sha256.as_str(),
        receipt.observer_profile_sha256.as_str(), receipt.capture.provider_request_body_sha256.as_str(),
        receipt.capture.request_body_artifact.plaintext_sha256.as_str(),
        receipt.capture.request_body_artifact.ciphertext_sha256.as_str(),
        receipt.capture.request_body_artifact.storage_profile_sha256.as_str(),
        receipt.capture.request_body_artifact.encryption_profile_sha256.as_str(),
        receipt.capture.request_body_artifact.access_policy_sha256.as_str(),
        receipt.capture.request_body_artifact.retention_policy_sha256.as_str(),
        receipt.capture.request_body_artifact.custody_receipt_sha256.as_str(),
        receipt.response_binding.provider_response_body_sha256.as_str(),
        receipt.response_binding.candidate_decision_trace_sha256.as_str(),
    ] { validate_sha256(digest)?; }

    if receipt.exposures.len() > MACE_CONTEXT_EXPOSURE_MAX_SPANS_V1 {
        return Err(MaceContextExposureRejectV1::TooManySpans);
    }
    if receipt.admission_denials.len() > MACE_CONTEXT_EXPOSURE_MAX_SPANS_V1 {
        return Err(MaceContextExposureRejectV1::TooManySpans);
    }
    let mut exposure_ids = BTreeSet::new();
    for span in &receipt.exposures {
        validate_identifier(&span.exposure_id, 256)?;
        validate_identifier(&span.evidence_id, 256)?;
        if !exposure_ids.insert(span.exposure_id.as_str()) {
            return Err(MaceContextExposureRejectV1::DuplicateExposureId);
        }
        if span.admission_status != "admitted" {
            return Err(MaceContextExposureRejectV1::InvalidAdmissionStatus);
        }
        let mut parents = BTreeSet::new();
        for parent in &span.parent_event_ids {
            validate_identifier(parent, 256)?;
            if !parents.insert(parent) { return Err(MaceContextExposureRejectV1::DuplicateParentId); }
        }
        for digest in [
            span.evidence_payload_sha256.as_str(), span.exposed_span_sha256.as_str(),
            span.dependency_closure_sha256.as_str(), span.transformation_receipt_sha256.as_str(),
            span.source_receipt_sha256.as_str(), span.authority_receipt_sha256.as_str(),
            span.admissibility_receipt_sha256.as_str(), span.lifecycle_revalidation_receipt_sha256.as_str(),
        ] { validate_sha256(digest)?; }
        if span.span_end_byte <= span.span_start_byte {
            return Err(MaceContextExposureRejectV1::InvalidSpanRange);
        }
    }

    let mut denied = BTreeSet::new();
    for denial in &receipt.admission_denials {
        validate_identifier(&denial.evidence_id, 256)?;
        if !denied.insert(denial.evidence_id.as_str()) {
            return Err(MaceContextExposureRejectV1::DuplicateDeniedEvidence);
        }
        validate_sha256(&denial.admission_receipt_sha256)?;
    }
    if receipt.exposures.iter().any(|span| denied.contains(span.evidence_id.as_str())) {
        return Err(MaceContextExposureRejectV1::DeniedEvidenceExposed);
    }

    let mut model_support = BTreeSet::new();
    for id in &receipt.response_binding.model_declared_support_event_ids {
        validate_identifier(id, 256)?;
        if !model_support.insert(id) {
            return Err(MaceContextExposureRejectV1::InvalidResponseBinding);
        }
    }
    if receipt.signature.algorithm != "Ed25519" {
        return Err(MaceContextExposureRejectV1::InvalidSignatureAlgorithm);
    }
    validate_identifier(&receipt.signature.key_id, 256)
        .map_err(|_| MaceContextExposureRejectV1::InvalidObserverKeyId)?;
    Ok(())
}

fn verify_span_coverage(
    receipt: &MaceContextExposureReceiptV1,
    body: &[u8],
) -> Result<(), MaceContextExposureRejectV1> {
    let mut spans: Vec<&MaceMemoryExposureSpanV1> = receipt.exposures.iter().collect();
    spans.sort_by_key(|span| (span.span_start_byte, span.span_end_byte));
    let mut previous_end = 0u64;
    for (index, span) in spans.iter().enumerate() {
        let start = usize::try_from(span.span_start_byte)
            .map_err(|_| MaceContextExposureRejectV1::InvalidSpanRange)?;
        let end = usize::try_from(span.span_end_byte)
            .map_err(|_| MaceContextExposureRejectV1::InvalidSpanRange)?;
        if end > body.len() || start >= end {
            return Err(MaceContextExposureRejectV1::InvalidSpanRange);
        }
        if index > 0 && span.span_start_byte < previous_end {
            return Err(MaceContextExposureRejectV1::OverlappingSpans);
        }
        previous_end = span.span_end_byte;
        if sha256_hex(&body[start..end]) != span.exposed_span_sha256 {
            return Err(MaceContextExposureRejectV1::SpanDigestMismatch);
        }
        // A valid memory span must begin and end at complete UTF-8 / JSON
        // string-content boundaries. A digest over half of an escape sequence
        // or a partial multi-byte scalar is not a valid exposed text segment.
        let mut quoted_fragment = Vec::with_capacity(end - start + 2);
        quoted_fragment.push(b'"');
        quoted_fragment.extend_from_slice(&body[start..end]);
        quoted_fragment.push(b'"');
        serde_json::from_slice::<String>(&quoted_fragment)
            .map_err(|_| MaceContextExposureRejectV1::InvalidSpanEncoding)?;
    }
    Ok(())
}

fn verify_observer_signature(
    receipt: &MaceContextExposureReceiptV1,
    expected: &MaceExpectedExposureBindingV1<'_>,
    resolver: &impl MaceExposureObserverKeyResolverV1,
) -> Result<(), MaceContextExposureRejectV1> {
    let key = resolver.resolve_key(expected.observer_profile_sha256, &receipt.signature.key_id)
        .map_err(|_| MaceContextExposureRejectV1::UnknownObserverKey)?;
    if ed25519_spki_key_id_v1(&key) != receipt.signature.key_id {
        return Err(MaceContextExposureRejectV1::InvalidObserverKeyId);
    }
    let decoded = BASE64_STANDARD.decode(&receipt.signature.signature_base64)
        .map_err(|_| MaceContextExposureRejectV1::InvalidSignatureEncoding)?;
    if BASE64_STANDARD.encode(&decoded) != receipt.signature.signature_base64 {
        return Err(MaceContextExposureRejectV1::InvalidSignatureEncoding);
    }
    let signature = Signature::from_slice(&decoded)
        .map_err(|_| MaceContextExposureRejectV1::InvalidSignatureEncoding)?;
    let digest = exposure_signature_digest_v1(receipt)?;
    key.verify_strict(&digest, &signature)
        .map_err(|_| MaceContextExposureRejectV1::SignatureMismatch)
}


/// Signs a finalized receipt digest using an observer key whose custody is
/// provided by the deployment. Implementations may wrap a KMS/HSM; this trait
/// does not expose private key material to the MACE adapter.
pub trait MaceExposureReceiptSignerV1 {
    fn key_id(&self) -> &str;
    fn sign_digest(
        &self,
        digest: &[u8; 32],
    ) -> Result<[u8; 64], MaceContextExposureRejectV1>;
}

/// Adds the observer's Ed25519 signature to a fully populated receipt.
/// Request serialization, source-span mapping, artifact custody and response
/// binding must already have been completed by the qualified capture adapter.
pub fn sign_mace_context_exposure_receipt_v1(
    mut receipt: MaceContextExposureReceiptV1,
    signer: &impl MaceExposureReceiptSignerV1,
) -> Result<MaceContextExposureReceiptV1, MaceContextExposureRejectV1> {
    validate_identifier(signer.key_id(), 256)
        .map_err(|_| MaceContextExposureRejectV1::InvalidObserverKeyId)?;
    receipt.signature.algorithm = "Ed25519".to_owned();
    receipt.signature.key_id = signer.key_id().to_owned();
    receipt.signature.signature_base64.clear();
    let digest = exposure_signature_digest_v1(&receipt)?;
    let signature = signer
        .sign_digest(&digest)
        .map_err(|_| MaceContextExposureRejectV1::ObserverSigningFailed)?;
    receipt.signature.signature_base64 = BASE64_STANDARD.encode(signature);
    Ok(receipt)
}

/// SHA-256 over Ed25519 SubjectPublicKeyInfo DER, truncated to the first 16
/// bytes. This matches MON-005's witness-key-ID convention.
pub fn ed25519_spki_key_id_v1(key: &VerifyingKey) -> String {
    const PREFIX: [u8; 12] =
        [0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00];
    let mut der = Vec::with_capacity(44);
    der.extend_from_slice(&PREFIX);
    der.extend_from_slice(key.as_bytes());
    hex_lower(&Sha256::digest(der)[..16])
}

/// SHA-256("MACE-CONTEXT-EXPOSURE-V1\\0" || JCS(receipt_without_signature)).
pub fn exposure_signature_digest_v1(
    receipt: &MaceContextExposureReceiptV1,
) -> Result<[u8; 32], MaceContextExposureRejectV1> {
    let mut value = serde_json::to_value(receipt)
        .map_err(|_| MaceContextExposureRejectV1::CanonicalizationFailed)?;
    value.as_object_mut()
        .ok_or(MaceContextExposureRejectV1::CanonicalizationFailed)?
        .remove("signature");
    let canonical = serde_json_canonicalizer::to_vec(&value)
        .map_err(|_| MaceContextExposureRejectV1::CanonicalizationFailed)?;
    let mut hasher = Sha256::new();
    hasher.update(MACE_CONTEXT_EXPOSURE_DOMAIN_V1);
    hasher.update(canonical);
    Ok(hasher.finalize().into())
}

fn validate_identifier(value: &str, max_chars: usize) -> Result<(), MaceContextExposureRejectV1> {
    if value.trim().is_empty() || value.trim() != value || value.chars().count() > max_chars {
        return Err(MaceContextExposureRejectV1::EmptyIdentifier);
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), MaceContextExposureRejectV1> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(MaceContextExposureRejectV1::InvalidDigest);
    }
    Ok(())
}
fn sha256_hex(bytes: &[u8]) -> String { hex_lower(&Sha256::digest(bytes)) }
fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    const BODY: &[u8] = br#"{"model":"model-pinned-1","messages":[{"role":"user","content":"context"}]}"#;
    const D1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const D2: &str = "2222222222222222222222222222222222222222222222222222222222222222";
    const D3: &str = "3333333333333333333333333333333333333333333333333333333333333333";
    const D4: &str = "4444444444444444444444444444444444444444444444444444444444444444";
    const D5: &str = "5555555555555555555555555555555555555555555555555555555555555555";
    const D6: &str = "6666666666666666666666666666666666666666666666666666666666666666";
    const D7: &str = "7777777777777777777777777777777777777777777777777777777777777777";
    const D8: &str = "8888888888888888888888888888888888888888888888888888888888888888";
    const D9: &str = "9999999999999999999999999999999999999999999999999999999999999999";

    struct TestSigner<'a> { key: &'a SigningKey, key_id: String }
    impl MaceExposureReceiptSignerV1 for TestSigner<'_> {
        fn key_id(&self) -> &str { &self.key_id }
        fn sign_digest(&self, digest: &[u8; 32])
            -> Result<[u8; 64], MaceContextExposureRejectV1> {
            Ok(self.key.sign(digest).to_bytes())
        }
    }

    struct FakeBodyResolver { body: Vec<u8>, fail: bool }
    impl MaceRequestBodyArtifactResolverV1 for FakeBodyResolver {
        fn resolve_and_verify(&self, _: &MaceRequestBodyArtifactV1)
            -> Result<Vec<u8>, MaceContextExposureRejectV1> {
            if self.fail { Err(MaceContextExposureRejectV1::ArtifactResolutionFailed) }
            else { Ok(self.body.clone()) }
        }
    }
    struct FakeKeyResolver { profile: String, key: VerifyingKey }
    impl MaceExposureObserverKeyResolverV1 for FakeKeyResolver {
        fn resolve_key(&self, profile: &str, key_id: &str)
            -> Result<VerifyingKey, MaceContextExposureRejectV1> {
            if profile != self.profile || key_id != ed25519_spki_key_id_v1(&self.key) {
                Err(MaceContextExposureRejectV1::UnknownObserverKey)
            } else { Ok(self.key.clone()) }
        }
    }
    struct AcceptEvidence;
    impl MaceExposureAdmissionVerifierV1 for AcceptEvidence {
        fn verify_admitted_exposure(&self, _: &MaceMemoryExposureSpanV1, _: &MaceExpectedExposureBindingV1<'_>)
            -> Result<(), MaceContextExposureRejectV1> { Ok(()) }
    }
    struct RejectEvidence;
    impl MaceExposureAdmissionVerifierV1 for RejectEvidence {
        fn verify_admitted_exposure(&self, _: &MaceMemoryExposureSpanV1, _: &MaceExpectedExposureBindingV1<'_>)
            -> Result<(), MaceContextExposureRejectV1> {
            Err(MaceContextExposureRejectV1::EvidenceAdmissionRejected)
        }
    }
    fn hash(bytes: &[u8]) -> String { sha256_hex(bytes) }

    fn refresh_body_binding(
        receipt: &mut MaceContextExposureReceiptV1,
        body: &[u8],
        key: &SigningKey,
    ) {
        receipt.capture.provider_request_body_sha256 = hash(body);
        receipt.capture.provider_request_body_bytes = body.len() as u64;
        receipt.capture.request_body_artifact.plaintext_sha256 = hash(body);
        receipt.capture.request_body_artifact.plaintext_bytes = body.len() as u64;
        let digest = exposure_signature_digest_v1(receipt).unwrap();
        receipt.signature.signature_base64 = BASE64_STANDARD.encode(key.sign(&digest).to_bytes());
    }

    fn expected() -> MaceExpectedExposureBindingV1<'static> {
        expected_with_parameter_profile(
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        )
    }

    fn expected_with_parameter_profile(
        profile: &str,
    ) -> MaceExpectedExposureBindingV1<'_> {
        MaceExpectedExposureBindingV1 {
            run_id: "run-1", pair_block_id: "pair-1", world_cluster_id: "cluster-1",
            query_id: "query-1", arm: MaceMemoryArmV1::ConventionalRetrieval,
            invocation_sequence: 0, decision_need_sha256: D1, candidate_policy_sha256: D2,
            source_registry_root_sha256: D3, lineage_profile_sha256: D4,
            lifecycle_profile_sha256: D5, renderer_profile_sha256: D6,
            transport_adapter_profile_sha256: D7, provider_parameter_profile_sha256: profile,
            model_revision: "model-pinned-1", clock_profile_sha256: D8,
            observer_profile_sha256: D9, storage_profile_sha256: D1,
            encryption_profile_sha256: D2, access_policy_sha256: D3,
            retention_policy_sha256: D4,
        }
    }
    fn fixture() -> (MaceContextExposureReceiptV1, SigningKey, Vec<u8>) {
        let key = SigningKey::from_bytes(&[7u8; 32]);
        let body = BODY.to_vec();
        let needle = b"context";
        let start = body.windows(needle.len()).position(|w| w == needle).unwrap() as u64;
        let end = start + needle.len() as u64;
        let vk = key.verifying_key();
        let mut receipt = MaceContextExposureReceiptV1 {
            schema_version: MACE_CONTEXT_EXPOSURE_SCHEMA_V1.to_owned(),
            record_type: "model_context_exposure".to_owned(), receipt_id: "receipt-1".to_owned(),
            run_id: "run-1".to_owned(), pair_block_id: "pair-1".to_owned(),
            world_cluster_id: "cluster-1".to_owned(), query_id: "query-1".to_owned(),
            arm: MaceMemoryArmV1::ConventionalRetrieval, invocation_sequence: 0,
            decision_need_sha256: D1.to_owned(), candidate_policy_sha256: D2.to_owned(),
            source_registry_root_sha256: D3.to_owned(), lineage_profile_sha256: D4.to_owned(),
            lifecycle_profile_sha256: D5.to_owned(), renderer_profile_sha256: D6.to_owned(),
            transport_adapter_profile_sha256: D7.to_owned(), provider_parameter_profile_sha256: "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a".to_owned(), model_revision: "model-pinned-1".to_owned(),
            capture: MaceContextCaptureV1 {
                capture_point: "serialized_request_before_network_send".to_owned(),
                request_media_type: "application/json".to_owned(),
                provider_request_body_sha256: hash(&body), provider_request_body_bytes: body.len() as u64,
                complete_provider_request_body: true, complete_memory_origin_enumeration: true,
                untracked_memory_origin_span_count: 0,
                request_body_artifact: MaceRequestBodyArtifactV1 {
                    artifact_id: "encrypted://artifact/1".to_owned(), plaintext_sha256: hash(&body),
                    ciphertext_sha256: D8.to_owned(), plaintext_bytes: body.len() as u64,
                    storage_profile_sha256: D1.to_owned(), encryption_profile_sha256: D2.to_owned(),
                    access_policy_sha256: D3.to_owned(), retention_policy_sha256: D4.to_owned(),
                    custody_receipt_sha256: D5.to_owned(), immutable_object: true,
                },
                provider_request_id: Some("request-1".to_owned()),
            },
            exposures: vec![MaceMemoryExposureSpanV1 {
                exposure_id: "exposure-1".to_owned(), evidence_id: "evidence-1".to_owned(),
                evidence_payload_sha256: D1.to_owned(), exposed_span_sha256: hash(needle),
                span_start_byte: start, span_end_byte: end, context_role: MaceContextRoleV1::User,
                message_index: 0, exposure_kind: MaceExposureKindV1::VerbatimSource,
                parent_event_ids: vec!["source-1".to_owned()], dependency_closure_sha256: D2.to_owned(),
                transformation_receipt_sha256: D3.to_owned(), source_receipt_sha256: D4.to_owned(),
                authority_receipt_sha256: D5.to_owned(), admissibility_receipt_sha256: D6.to_owned(),
                lifecycle_revalidation_receipt_sha256: D7.to_owned(), admission_status: "admitted".to_owned(),
            }],
            admission_denials: vec![],
            response_binding: MaceExposureResponseBindingV1 {
                provider_response_id: "response-1".to_owned(), provider_response_body_sha256: D8.to_owned(),
                candidate_decision_trace_sha256: D9.to_owned(),
                model_declared_support_event_ids: vec!["evidence-1".to_owned()],
            },
            captured_at_monotonic_ns: 42, clock_profile_sha256: D8.to_owned(),
            observer_profile_sha256: D9.to_owned(),
            signature: MaceExposureSignatureV1 {
                algorithm: "Ed25519".to_owned(), key_id: ed25519_spki_key_id_v1(&vk),
                signature_base64: BASE64_STANDARD.encode([0u8; 64]),
            },
        };
        let digest = exposure_signature_digest_v1(&receipt).unwrap();
        receipt.signature.signature_base64 = BASE64_STANDARD.encode(key.sign(&digest).to_bytes());
        (receipt, key, body)
    }
    fn verify_fixture(
        receipt: &MaceContextExposureReceiptV1, body: Vec<u8>, key: &SigningKey,
        expected: &MaceExpectedExposureBindingV1<'_>, evidence: &impl MaceExposureAdmissionVerifierV1,
    ) -> Result<VerifiedMaceContextExposureReceiptV1, MaceContextExposureRejectV1> {
        let body_resolver = FakeBodyResolver { body, fail: false };
        let keys = FakeKeyResolver { profile: expected.observer_profile_sha256.to_owned(), key: key.verifying_key() };
        parse_and_verify_mace_context_exposure_receipt_v1(
            &serde_json::to_vec(receipt).unwrap(), expected, &body_resolver, &keys, evidence,
        )
    }

    #[test]
    fn signer_finalizes_receipt_using_domain_separated_digest() {
        let (mut receipt, key, body) = fixture();
        receipt.signature.signature_base64.clear();
        receipt.signature.key_id.clear();
        let signer = TestSigner {
            key: &key,
            key_id: ed25519_spki_key_id_v1(&key.verifying_key()),
        };
        let signed = sign_mace_context_exposure_receipt_v1(receipt, &signer).unwrap();
        assert_eq!(signed.signature.algorithm, "Ed25519");
        assert_eq!(signed.signature.key_id, signer.key_id);
        assert_eq!(
            verify_fixture(&signed, body, &key, &expected(), &AcceptEvidence)
                .unwrap().artifact_id(),
            "encrypted://artifact/1"
        );
    }

    #[test]
    fn verifies_signed_request_body_and_evidence_binding() {
        let (receipt, key, body) = fixture();
        let verified = verify_fixture(&receipt, body, &key, &expected(), &AcceptEvidence).unwrap();
        assert_eq!(verified.request_body_sha256(), receipt.capture.provider_request_body_sha256);
        assert_eq!(verified.artifact_id(), "encrypted://artifact/1");
        assert_eq!(verified.exposed_evidence_ids().collect::<Vec<_>>(), vec!["evidence-1"]);
    }
    #[test]
    fn rejects_wrong_run_binding() {
        let (receipt, key, body) = fixture();
        let mut e = expected(); e.run_id = "other-run";
        assert_eq!(verify_fixture(&receipt, body, &key, &e, &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::WrongRun);
    }
    #[test]
    fn rejects_changed_request_body() {
        let (receipt, key, mut body) = fixture(); body[0] = b'[';
        assert_eq!(verify_fixture(&receipt, body, &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::BodyDigestMismatch);
    }
    #[test]
    fn rejects_overlapping_spans() {
        let (mut receipt, key, body) = fixture();
        let mut second = receipt.exposures[0].clone();
        second.exposure_id = "exposure-2".to_owned(); second.span_start_byte += 1;
        second.exposed_span_sha256 = hash(&body[second.span_start_byte as usize..second.span_end_byte as usize]);
        receipt.exposures.push(second);
        let digest = exposure_signature_digest_v1(&receipt).unwrap();
        receipt.signature.signature_base64 = BASE64_STANDARD.encode(key.sign(&digest).to_bytes());
        assert_eq!(verify_fixture(&receipt, body, &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::OverlappingSpans);
    }
    #[test]
    fn rejects_untracked_memory_origin_bytes() {
        let (mut receipt, key, body) = fixture(); receipt.capture.untracked_memory_origin_span_count = 1;
        let digest = exposure_signature_digest_v1(&receipt).unwrap();
        receipt.signature.signature_base64 = BASE64_STANDARD.encode(key.sign(&digest).to_bytes());
        assert_eq!(verify_fixture(&receipt, body, &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::UntrackedOriginSpans);
    }
    #[test]
    fn rejects_invalid_signature() {
        let (mut receipt, key, body) = fixture();
        receipt.signature.signature_base64 = BASE64_STANDARD.encode([9u8; 64]);
        assert_eq!(verify_fixture(&receipt, body, &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::SignatureMismatch);
    }
    #[test]
    fn rejects_untrusted_observer_key_id() {
        let (mut receipt, key, body) = fixture();
        receipt.signature.key_id = "00000000000000000000000000000000".to_owned();
        let digest = exposure_signature_digest_v1(&receipt).unwrap();
        receipt.signature.signature_base64 = BASE64_STANDARD.encode(key.sign(&digest).to_bytes());
        assert_eq!(verify_fixture(&receipt, body, &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::UnknownObserverKey);
    }
    #[test]
    fn rejects_exposure_that_fails_canonical_admission() {
        let (receipt, key, body) = fixture();
        assert_eq!(verify_fixture(&receipt, body, &key, &expected(), &RejectEvidence).unwrap_err(),
            MaceContextExposureRejectV1::EvidenceAdmissionRejected);
    }
    #[test]
    fn rejects_evidence_both_denied_and_exposed() {
        let (mut receipt, key, body) = fixture();
        receipt.admission_denials.push(MaceExposureAdmissionDenialV1 {
            evidence_id: "evidence-1".to_owned(), reason: MaceAdmissionDenialReasonV1::Revoked,
            admission_receipt_sha256: D1.to_owned(),
        });
        let digest = exposure_signature_digest_v1(&receipt).unwrap();
        receipt.signature.signature_base64 = BASE64_STANDARD.encode(key.sign(&digest).to_bytes());
        assert_eq!(verify_fixture(&receipt, body, &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::DeniedEvidenceExposed);
    }
    #[test]
    fn rejects_artifact_profile_substitution() {
        let (mut receipt, key, body) = fixture();
        receipt.capture.request_body_artifact.storage_profile_sha256 = D9.to_owned();
        let digest = exposure_signature_digest_v1(&receipt).unwrap();
        receipt.signature.signature_base64 = BASE64_STANDARD.encode(key.sign(&digest).to_bytes());
        assert_eq!(verify_fixture(&receipt, body, &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::ArtifactCustodyMismatch);
    }
    #[test]
    fn rejects_artifact_unavailability() {
        let (receipt, key, body) = fixture();
        let resolver = FakeBodyResolver { body, fail: true };
        let keys = FakeKeyResolver { profile: D9.to_owned(), key: key.verifying_key() };
        assert_eq!(parse_and_verify_mace_context_exposure_receipt_v1(
            &serde_json::to_vec(&receipt).unwrap(), &expected(), &resolver, &keys, &AcceptEvidence
        ).unwrap_err(), MaceContextExposureRejectV1::ArtifactResolutionFailed);
    }
    #[test]
    fn rejects_memory_contamination_of_no_memory_arm() {
        let (mut receipt, key, body) = fixture(); receipt.arm = MaceMemoryArmV1::NoMemory;
        let digest = exposure_signature_digest_v1(&receipt).unwrap();
        receipt.signature.signature_base64 = BASE64_STANDARD.encode(key.sign(&digest).to_bytes());
        let mut e = expected(); e.arm = MaceMemoryArmV1::NoMemory;
        assert_eq!(verify_fixture(&receipt, body, &key, &e, &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::MemoryExposureInNoMemoryArm);
    }
    #[test]
    fn rejects_provider_parameter_profile_substitution() {
        let (mut receipt, key, body) = fixture();
        receipt.provider_parameter_profile_sha256 = D8.to_owned();
        let digest = exposure_signature_digest_v1(&receipt).unwrap();
        receipt.signature.signature_base64 = BASE64_STANDARD.encode(key.sign(&digest).to_bytes());
        assert_eq!(
            verify_fixture(&receipt, body, &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::WrongProviderParameterProfile
        );
    }

    #[test]
    fn recomputes_parameter_profile_from_exact_request_body() {
        let (mut receipt, key, _) = fixture();
        let body = br#"{"model":"model-pinned-1","messages":[{"role":"user","content":"context"}],"temperature":0.7}"#;
        refresh_body_binding(&mut receipt, body, &key);
        assert_eq!(
            verify_fixture(&receipt, body.to_vec(), &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::WrongProviderParameterProfile
        );
    }

    #[test]
    fn rejects_free_form_string_request_parameters() {
        let (mut receipt, key, _) = fixture();
        let body = br#"{"model":"model-pinned-1","messages":[{"role":"user","content":"context"}],"temperature":"hidden memory text"}"#;
        refresh_body_binding(&mut receipt, body, &key);
        assert_eq!(
            verify_fixture(&receipt, body.to_vec(), &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::ProviderParameterTypeMismatch
        );
    }

    #[test]
    fn rejects_nested_request_parameter_objects() {
        let (mut receipt, key, _) = fixture();
        let body = br#"{"model":"model-pinned-1","messages":[{"role":"user","content":"context"}],"temperature":{"nested":"hidden dynamic text"}}"#;
        refresh_body_binding(&mut receipt, body, &key);
        assert_eq!(
            verify_fixture(&receipt, body.to_vec(), &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::ProviderParameterTypeMismatch
        );
    }

    #[test]
    fn rejects_unclassified_provider_parameter_names() {
        let (mut receipt, key, _) = fixture();
        let body = br#"{"model":"model-pinned-1","messages":[{"role":"user","content":"context"}],"instructions":true}"#;
        refresh_body_binding(&mut receipt, body, &key);
        assert_eq!(
            verify_fixture(&receipt, body.to_vec(), &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::UnclassifiedProviderParameterKey
        );
    }

    #[test]
    fn accepts_recomputed_allowlisted_numeric_and_boolean_parameters() {
        let (mut receipt, key, _) = fixture();
        let body = br#"{"model":"model-pinned-1","messages":[{"role":"user","content":"context"}],"stream":true,"temperature":0.2}"#;
        let mut parameters = BTreeMap::new();
        parameters.insert("stream".to_owned(), serde_json::Value::Bool(true));
        parameters.insert("temperature".to_owned(), serde_json::json!(0.2));
        let profile = hash(&serde_json::to_vec(&parameters).unwrap());
        receipt.provider_parameter_profile_sha256 = profile.clone();
        refresh_body_binding(&mut receipt, body, &key);
        let bound = expected_with_parameter_profile(&profile);
        assert!(
            verify_fixture(&receipt, body.to_vec(), &key, &bound, &AcceptEvidence).is_ok()
        );
    }

    #[test]
    fn verifies_escaped_unicode_content_offsets_against_raw_request() {
        let (mut receipt, key, _) = fixture();
        let body = br#"{"model":"model-pinned-1","messages":[{"role":"user","content":"prefix 😀 \"quoted\" line\ncontext"}]}"#;
        let needle = b"context";
        let start = body.windows(needle.len()).position(|window| window == needle).unwrap() as u64;
        receipt.exposures[0].span_start_byte = start;
        receipt.exposures[0].span_end_byte = start + needle.len() as u64;
        receipt.exposures[0].exposed_span_sha256 = hash(needle);
        refresh_body_binding(&mut receipt, body, &key);
        assert!(
            verify_fixture(&receipt, body.to_vec(), &key, &expected(), &AcceptEvidence).is_ok()
        );
    }

    #[test]
    fn rejects_span_that_splits_a_json_escape_sequence() {
        let (mut receipt, key, _) = fixture();
        let body = br#"{"model":"model-pinned-1","messages":[{"role":"user","content":"prefix\\ncontext"}]}"#;
        let escape_start = body.windows(2).position(|window| window == b"\\n").unwrap() as u64;
        receipt.exposures[0].span_start_byte = escape_start;
        receipt.exposures[0].span_end_byte = escape_start + 1;
        receipt.exposures[0].exposed_span_sha256 = hash(b"\\");
        refresh_body_binding(&mut receipt, body, &key);
        assert_eq!(
            verify_fixture(&receipt, body.to_vec(), &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::InvalidSpanEncoding
        );
    }

    #[test]
    fn rejects_span_that_splits_a_utf8_scalar() {
        let (mut receipt, key, _) = fixture();
        let body = br#"{"model":"model-pinned-1","messages":[{"role":"user","content":"prefix 😀 context"}]}"#;
        let emoji_start = body.windows(4).position(|window| window == "😀".as_bytes()).unwrap() as u64;
        receipt.exposures[0].span_start_byte = emoji_start + 1;
        receipt.exposures[0].span_end_byte = emoji_start + 2;
        receipt.exposures[0].exposed_span_sha256 =
            hash(&body[(emoji_start + 1) as usize..(emoji_start + 2) as usize]);
        refresh_body_binding(&mut receipt, body, &key);
        assert_eq!(
            verify_fixture(&receipt, body.to_vec(), &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::InvalidSpanEncoding
        );
    }

    #[test]
    fn rejects_duplicate_top_level_model_members() {
        let (mut receipt, key, _) = fixture();
        let body = br#"{"model":"model-pinned-1","model":"model-pinned-1","messages":[{"role":"user","content":"context"}]}"#;
        refresh_body_binding(&mut receipt, body, &key);
        assert_eq!(
            verify_fixture(&receipt, body.to_vec(), &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::InvalidRequestBody
        );
    }

    #[test]
    fn rejects_duplicate_top_level_message_members() {
        let (mut receipt, key, _) = fixture();
        let body = br#"{"model":"model-pinned-1","messages":[{"role":"user","content":"context"}],"messages":[{"role":"user","content":"other"}]}"#;
        refresh_body_binding(&mut receipt, body, &key);
        assert_eq!(
            verify_fixture(&receipt, body.to_vec(), &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::InvalidRequestBody
        );
    }

    #[test]
    fn rejects_duplicate_provider_parameter_members() {
        let (mut receipt, key, _) = fixture();
        let body = br#"{"model":"model-pinned-1","messages":[{"role":"user","content":"context"}],"temperature":0.2,"temperature":0.9}"#;
        refresh_body_binding(&mut receipt, body, &key);
        assert_eq!(
            verify_fixture(&receipt, body.to_vec(), &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::InvalidRequestBody
        );
    }

    #[test]
    fn rejects_untracked_message_fields() {
        let (mut receipt, key, _) = fixture();
        let body = br#"{"model":"model-pinned-1","messages":[{"role":"user","content":"context","metadata":"hidden"}]}"#;
        refresh_body_binding(&mut receipt, body, &key);
        assert_eq!(
            verify_fixture(&receipt, body.to_vec(), &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::InvalidRequestBody
        );
    }

    #[test]
    fn rejects_model_selector_in_body_that_differs_from_pinned_revision() {
        let (mut receipt, key, _) = fixture();
        let body = br#"{"model":"model-pinned-2","messages":[{"role":"user","content":"context"}]}"#;
        refresh_body_binding(&mut receipt, body, &key);
        assert_eq!(
            verify_fixture(&receipt, body.to_vec(), &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::WrongModelRevision
        );
    }

    #[test]
    fn rejects_exposure_span_outside_the_indexed_message_content() {
        let (mut receipt, key, body) = fixture();
        let needle = b"model-pi";
        let start = body.windows(needle.len()).position(|w| w == needle).unwrap() as u64;
        receipt.exposures[0].span_start_byte = start;
        receipt.exposures[0].span_end_byte = start + needle.len() as u64;
        receipt.exposures[0].exposed_span_sha256 =
            hash(&body[start as usize..(start as usize + needle.len())]);
        let digest = exposure_signature_digest_v1(&receipt).unwrap();
        receipt.signature.signature_base64 = BASE64_STANDARD.encode(key.sign(&digest).to_bytes());
        assert_eq!(
            verify_fixture(&receipt, body, &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::ExposureOutsideMessageContent
        );
    }

    #[test]
    fn rejects_exposure_role_mismatch_with_request_message() {
        let (mut receipt, key, body) = fixture();
        receipt.exposures[0].context_role = MaceContextRoleV1::Developer;
        let digest = exposure_signature_digest_v1(&receipt).unwrap();
        receipt.signature.signature_base64 = BASE64_STANDARD.encode(key.sign(&digest).to_bytes());
        assert_eq!(
            verify_fixture(&receipt, body, &key, &expected(), &AcceptEvidence).unwrap_err(),
            MaceContextExposureRejectV1::ExposureRoleMismatch
        );
    }

    #[test]
    fn rejects_duplicate_and_unknown_fields() {
        let (receipt, _, _) = fixture();
        let json = serde_json::to_string(&receipt).unwrap();
        let duplicate = json.replacen(
            "\"record_type\":\"model_context_exposure\"",
            "\"record_type\":\"model_context_exposure\",\"record_type\":\"model_context_exposure\"", 1,
        );
        assert!(serde_json::from_str::<MaceContextExposureReceiptV1>(&duplicate).is_err());
        let unknown = json.replacen(
            "\"record_type\":\"model_context_exposure\"",
            "\"record_type\":\"model_context_exposure\",\"unrecognized\":true", 1,
        );
        assert!(serde_json::from_str::<MaceContextExposureReceiptV1>(&unknown).is_err());
    }
}
