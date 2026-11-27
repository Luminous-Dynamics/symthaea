// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Producer-side request serialization and exposure-span mapping for MACE.
//!
//! This is a provider-neutral deterministic capture seam. It serializes once,
//! derives byte spans while writing the exact JSON body, stores those bytes
//! through an immutable-artifact interface, sends that same byte slice through
//! a transport interface, and signs the resulting exposure receipt.
//!
//! The source map is only as trustworthy as the typed renderer input. This
//! module cannot discover arbitrary untagged memory hidden in plain strings.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use crate::mace_context_exposure_receipt::{
    sign_mace_context_exposure_receipt_v1,
    MaceContextCaptureV1,
    MaceContextExposureReceiptV1,
    MaceContextRoleV1,
    MaceExposureAdmissionDenialV1,
    MaceExposureKindV1,
    MaceExposureResponseBindingV1,
    MaceExposureSignatureV1,
    MaceExpectedExposureBindingV1,
    MaceMemoryArmV1,
    MaceMemoryExposureSpanV1,
    MaceRequestBodyArtifactV1,
    MaceExposureReceiptSignerV1,
    MaceExposureAdmissionVerifierV1,
    MACE_CONTEXT_EXPOSURE_MAX_BODY_BYTES_V1,
    MACE_CONTEXT_EXPOSURE_MAX_SPANS_V1,
};

pub const MACE_CONTEXT_EXPOSURE_CAPTURE_MAX_MESSAGES_V1: usize = 10_000;
pub const MACE_CONTEXT_EXPOSURE_CAPTURE_MAX_SEGMENTS_V1: usize = 100_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaceExposureSourceMetadataV1 {
    pub evidence_id: String,
    pub evidence_payload_sha256: String,
    pub exposure_kind: MaceExposureKindV1,
    pub parent_event_ids: Vec<String>,
    pub dependency_closure_sha256: String,
    pub transformation_receipt_sha256: String,
    pub source_receipt_sha256: String,
    pub authority_receipt_sha256: String,
    pub admissibility_receipt_sha256: String,
    pub lifecycle_revalidation_receipt_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaceRequestSegmentOriginV1 {
    NonMemory,
    Memory(MaceExposureSourceMetadataV1),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaceRequestSegmentV1 {
    pub text: String,
    pub origin: MaceRequestSegmentOriginV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaceRequestMessageV1 {
    pub role: MaceContextRoleV1,
    pub segments: Vec<MaceRequestSegmentV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaceProviderNeutralRequestV1 {
    /// Exact provider model selector sent in the serialized request. It must
    /// match the independently pinned model_revision in the run manifest.
    pub model: String,
    /// Additional provider request fields such as temperature, top_p,
    /// response_format and tool schemas. These are non-memory fields; memory-
    /// derived content must be expressed as typed message segments instead.
    /// BTreeMap gives deterministic key order. Reserved model/messages keys
    /// are rejected to prevent duplicate JSON members.
    pub request_parameters: BTreeMap<String, Value>,
    pub messages: Vec<MaceRequestMessageV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaceSerializedProviderRequestV1 {
    body: Vec<u8>,
    exposures: Vec<MaceMemoryExposureSpanV1>,
}

impl MaceSerializedProviderRequestV1 {
    pub fn body(&self) -> &[u8] { &self.body }
    pub fn exposures(&self) -> &[MaceMemoryExposureSpanV1] { &self.exposures }
    pub fn body_sha256(&self) -> String { sha256_hex(&self.body) }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaceExposureCaptureRejectV1 {
    InvalidIdentifier,
    InvalidDigest,
    EmptyModel,
    ModelRevisionMismatch,
    EmptyMessages,
    TooManyMessages,
    EmptySegmentList,
    EmptyMemorySegment,
    TooManySegments,
    TooManyExposures,
    MemoryInNoMemoryArm,
    RequestBodyTooLarge,
    ArtifactWriterFailed,
    ArtifactBindingMismatch,
    TransportFailed,
    TransportSentDifferentBytes,
    InvalidResponseBinding,
    ReceiptSigningFailed,
    InvalidCaptureContext,
    EvidenceAdmissionRejected,
}

pub struct MaceExposureCaptureContextV1<'a> {
    pub receipt_id: &'a str,
    pub captured_at_monotonic_ns: u64,
    pub candidate_decision_trace_sha256: &'a str,
    pub admission_denials: Vec<MaceExposureAdmissionDenialV1>,
    pub model_declared_support_event_ids: Vec<String>,
}

/// The writer must persist the exact body as an immutable encrypted object,
/// and return independently auditable ciphertext/custody/profile commitments.
/// The production implementation owns encryption, KMS/HSM and object-store
/// guarantees; this module does not implement those facilities.
pub trait MaceRequestBodyArtifactWriterV1 {
    fn store_immutable_encrypted(
        &self,
        plaintext_body: &[u8],
        expected: &MaceExpectedExposureBindingV1<'_>,
    ) -> Result<MaceRequestBodyArtifactV1, MaceExposureCaptureRejectV1>;
}

/// The transport must send the exact supplied byte slice without re-serializing
/// it. The fake transport echoes the received-body digest so tests can verify
/// capture/send identity. A real adapter must preserve the same invariant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaceProviderResponseV1 {
    pub provider_request_id: Option<String>,
    pub provider_response_id: String,
    pub provider_response_body: Vec<u8>,
    pub received_request_body_sha256: String,
}

pub trait MaceProviderTransportV1 {
    fn send_exact(
        &self,
        serialized_request_body: &[u8],
    ) -> Result<MaceProviderResponseV1, MaceExposureCaptureRejectV1>;
}

/// Serializes a provider-neutral request exactly once and constructs the
/// memory-origin byte map during serialization. Offsets are UTF-8 byte
/// offsets into the returned body, inclusive at start and exclusive at end.
pub fn serialize_mace_provider_request_with_exposure_map_v1(
    request: &MaceProviderNeutralRequestV1,
    arm: MaceMemoryArmV1,
) -> Result<MaceSerializedProviderRequestV1, MaceExposureCaptureRejectV1> {
    if request.model.trim().is_empty() {
        return Err(MaceExposureCaptureRejectV1::EmptyModel);
    }
    validate_identifier(&request.model)?;
    if request.messages.is_empty() {
        return Err(MaceExposureCaptureRejectV1::EmptyMessages);
    }
    if request.messages.len() > MACE_CONTEXT_EXPOSURE_CAPTURE_MAX_MESSAGES_V1 {
        return Err(MaceExposureCaptureRejectV1::TooManyMessages);
    }

    let mut body = Vec::new();
    let mut exposures = Vec::new();
    let mut segment_count = 0usize;

    body.extend_from_slice(br#"{"model":"#);
    append_json_string(&mut body, &request.model);
    body.extend_from_slice(br#","messages":["#);

    for (message_index, message) in request.messages.iter().enumerate() {
        if message.segments.is_empty() {
            return Err(MaceExposureCaptureRejectV1::EmptySegmentList);
        }
        if message_index > 0 {
            body.push(b',');
        }
        body.extend_from_slice(br#"{"role":"#);
        append_json_string(&mut body, context_role_name(message.role));
        body.extend_from_slice(br#","content":""#);

        for segment in &message.segments {
            segment_count = segment_count.saturating_add(1);
            if segment_count > MACE_CONTEXT_EXPOSURE_CAPTURE_MAX_SEGMENTS_V1 {
                return Err(MaceExposureCaptureRejectV1::TooManySegments);
            }
            if segment.text.is_empty() {
                if matches!(&segment.origin, MaceRequestSegmentOriginV1::Memory(_)) {
                    return Err(MaceExposureCaptureRejectV1::EmptyMemorySegment);
                }
                continue;
            }

            let start = body.len();
            append_json_string_content(&mut body, &segment.text);
            let end = body.len();

            if let MaceRequestSegmentOriginV1::Memory(metadata) = &segment.origin {
                if arm == MaceMemoryArmV1::NoMemory {
                    return Err(MaceExposureCaptureRejectV1::MemoryInNoMemoryArm);
                }
                validate_source_metadata(metadata)?;
                if exposures.len() >= MACE_CONTEXT_EXPOSURE_MAX_SPANS_V1 {
                    return Err(MaceExposureCaptureRejectV1::TooManyExposures);
                }
                let start_byte = u64::try_from(start)
                    .map_err(|_| MaceExposureCaptureRejectV1::RequestBodyTooLarge)?;
                let end_byte = u64::try_from(end)
                    .map_err(|_| MaceExposureCaptureRejectV1::RequestBodyTooLarge)?;
                let span_digest = sha256_hex(&body[start..end]);
                exposures.push(MaceMemoryExposureSpanV1 {
                    exposure_id: format!("exposure-{}", exposures.len()),
                    evidence_id: metadata.evidence_id.clone(),
                    evidence_payload_sha256: metadata.evidence_payload_sha256.clone(),
                    exposed_span_sha256: span_digest,
                    span_start_byte: start_byte,
                    span_end_byte: end_byte,
                    context_role: message.role,
                    message_index: message_index as u64,
                    exposure_kind: metadata.exposure_kind,
                    parent_event_ids: metadata.parent_event_ids.clone(),
                    dependency_closure_sha256: metadata.dependency_closure_sha256.clone(),
                    transformation_receipt_sha256: metadata.transformation_receipt_sha256.clone(),
                    source_receipt_sha256: metadata.source_receipt_sha256.clone(),
                    authority_receipt_sha256: metadata.authority_receipt_sha256.clone(),
                    admissibility_receipt_sha256: metadata.admissibility_receipt_sha256.clone(),
                    lifecycle_revalidation_receipt_sha256: metadata.lifecycle_revalidation_receipt_sha256.clone(),
                    admission_status: "admitted".to_owned(),
                });
            }
            if body.len() > MACE_CONTEXT_EXPOSURE_MAX_BODY_BYTES_V1 {
                return Err(MaceExposureCaptureRejectV1::RequestBodyTooLarge);
            }
        }
        body.extend_from_slice(br#""}"#);
    }
    body.extend_from_slice(b"]");
    for (key, value) in &request.request_parameters {
        validate_identifier(key)?;
        if key == "model" || key == "messages" {
            return Err(MaceExposureCaptureRejectV1::InvalidIdentifier);
        }
        body.push(b',');
        append_json_string(&mut body, key);
        body.push(b':');
        let encoded_value = serde_json::to_vec(value)
            .map_err(|_| MaceExposureCaptureRejectV1::InvalidCaptureContext)?;
        body.extend_from_slice(&encoded_value);
    }
    body.push(b'}');

    if body.len() > MACE_CONTEXT_EXPOSURE_MAX_BODY_BYTES_V1 {
        return Err(MaceExposureCaptureRejectV1::RequestBodyTooLarge);
    }

    // Spans are emitted in serializer order and are disjoint by construction.
    Ok(MaceSerializedProviderRequestV1 { body, exposures })
}

/// Serialize, retain, send and sign exactly one model invocation.
///
/// This function deliberately receives the independent expected binding used
/// by the consumer verifier. It does not construct or alter canonical evidence
/// authority. The response and candidate decision trace are bound before the
/// observer signature is issued.
pub fn capture_send_and_sign_mace_invocation_v1(
    request: &MaceProviderNeutralRequestV1,
    expected: &MaceExpectedExposureBindingV1<'_>,
    context: MaceExposureCaptureContextV1<'_>,
    admission_verifier: &impl MaceExposureAdmissionVerifierV1,
    artifact_writer: &impl MaceRequestBodyArtifactWriterV1,
    transport: &impl MaceProviderTransportV1,
    signer: &impl MaceExposureReceiptSignerV1,
) -> Result<(MaceContextExposureReceiptV1, MaceProviderResponseV1), MaceExposureCaptureRejectV1> {
    validate_identifier(context.receipt_id)?;
    if request.model != expected.model_revision {
        return Err(MaceExposureCaptureRejectV1::ModelRevisionMismatch);
    }
    validate_digest(context.candidate_decision_trace_sha256)?;
    if expected.arm == MaceMemoryArmV1::NoMemory && !context.admission_denials.is_empty() {
        return Err(MaceExposureCaptureRejectV1::MemoryInNoMemoryArm);
    }
    validate_digest(expected.clock_profile_sha256)?;
    validate_digest(expected.observer_profile_sha256)?;

    let serialized = serialize_mace_provider_request_with_exposure_map_v1(request, expected.arm)?;

    // Admission occurs before immutable storage or provider transport. Every
    // typed memory segment must carry a fresh, use-bound ingress receipt and
    // a current purpose/rights decision.
    for exposure in serialized.exposures() {
        admission_verifier
            .verify_admitted_exposure(exposure, expected)
            .map_err(|_| MaceExposureCaptureRejectV1::EvidenceAdmissionRejected)?;
    }

    let body_sha256 = serialized.body_sha256();
    let artifact = artifact_writer
        .store_immutable_encrypted(serialized.body(), expected)
        .map_err(|_| MaceExposureCaptureRejectV1::ArtifactWriterFailed)?;
    validate_artifact_binding(&artifact, serialized.body(), expected)?;

    // The same in-memory byte slice used to compute the digest and artifact
    // commitment is sent to the transport; no second serialization occurs.
    let response = transport.send_exact(serialized.body())
        .map_err(|_| MaceExposureCaptureRejectV1::TransportFailed)?;
    if response.received_request_body_sha256 != body_sha256 {
        return Err(MaceExposureCaptureRejectV1::TransportSentDifferentBytes);
    }
    validate_identifier(&response.provider_response_id)
        .map_err(|_| MaceExposureCaptureRejectV1::InvalidResponseBinding)?;
    validate_digest(&sha256_hex(&response.provider_response_body))
        .map_err(|_| MaceExposureCaptureRejectV1::InvalidResponseBinding)?;
    if let Some(request_id) = response.provider_request_id.as_deref() {
        validate_identifier(request_id)
            .map_err(|_| MaceExposureCaptureRejectV1::InvalidResponseBinding)?;
    }

    let receipt = MaceContextExposureReceiptV1 {
        schema_version: "mace_context_exposure_receipt_v0_1".to_owned(),
        record_type: "model_context_exposure".to_owned(),
        receipt_id: context.receipt_id.to_owned(),
        run_id: expected.run_id.to_owned(),
        pair_block_id: expected.pair_block_id.to_owned(),
        world_cluster_id: expected.world_cluster_id.to_owned(),
        query_id: expected.query_id.to_owned(),
        arm: expected.arm,
        invocation_sequence: expected.invocation_sequence,
        decision_need_sha256: expected.decision_need_sha256.to_owned(),
        candidate_policy_sha256: expected.candidate_policy_sha256.to_owned(),
        source_registry_root_sha256: expected.source_registry_root_sha256.to_owned(),
        lineage_profile_sha256: expected.lineage_profile_sha256.to_owned(),
        lifecycle_profile_sha256: expected.lifecycle_profile_sha256.to_owned(),
        renderer_profile_sha256: expected.renderer_profile_sha256.to_owned(),
        transport_adapter_profile_sha256: expected.transport_adapter_profile_sha256.to_owned(),
        model_revision: expected.model_revision.to_owned(),
        capture: MaceContextCaptureV1 {
            capture_point: "serialized_request_before_network_send".to_owned(),
            request_media_type: "application/json".to_owned(),
            provider_request_body_sha256: body_sha256,
            provider_request_body_bytes: serialized.body().len() as u64,
            complete_provider_request_body: true,
            complete_memory_origin_enumeration: true,
            untracked_memory_origin_span_count: 0,
            request_body_artifact: artifact,
            provider_request_id: response.provider_request_id.clone(),
        },
        exposures: serialized.exposures,
        admission_denials: context.admission_denials,
        response_binding: MaceExposureResponseBindingV1 {
            provider_response_id: response.provider_response_id.clone(),
            provider_response_body_sha256: sha256_hex(&response.provider_response_body),
            candidate_decision_trace_sha256: context.candidate_decision_trace_sha256.to_owned(),
            model_declared_support_event_ids: context.model_declared_support_event_ids,
        },
        captured_at_monotonic_ns: context.captured_at_monotonic_ns,
        clock_profile_sha256: expected.clock_profile_sha256.to_owned(),
        observer_profile_sha256: expected.observer_profile_sha256.to_owned(),
        signature: MaceExposureSignatureV1 {
            algorithm: "Ed25519".to_owned(),
            key_id: signer.key_id().to_owned(),
            signature_base64: String::new(),
        },
    };
    sign_mace_context_exposure_receipt_v1(receipt, signer)
        .map(|signed| (signed, response))
        .map_err(|_| MaceExposureCaptureRejectV1::ReceiptSigningFailed)
}

/// Append a complete JSON string, including quotes.
fn append_json_string(output: &mut Vec<u8>, value: &str) {
    output.push(b'"');
    append_json_string_content(output, value);
    output.push(b'"');
}

/// Append JSON string content with canonical JSON escapes. Non-control Unicode
/// scalar values are emitted as their UTF-8 bytes. Span offsets are measured
/// around this function's exact output, so quotes/backslashes/control escapes
/// are correctly included in each exposed span digest and byte range.
fn append_json_string_content(output: &mut Vec<u8>, value: &str) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for ch in value.chars() {
        match ch {
            '"' => output.extend_from_slice(br#"\""#),
            '\\' => output.extend_from_slice(br#"\\"#),
            '\u{08}' => output.extend_from_slice(br#"\b"#),
            '\u{0c}' => output.extend_from_slice(br#"\f"#),
            '\n' => output.extend_from_slice(br#"\n"#),
            '\r' => output.extend_from_slice(br#"\r"#),
            '\t' => output.extend_from_slice(br#"\t"#),
            control if (control as u32) < 0x20 => {
                let code = control as u8;
                output.extend_from_slice(br#"\u00"#);
                output.push(HEX[usize::from(code >> 4)]);
                output.push(HEX[usize::from(code & 0x0f)]);
            }
            other => {
                let mut buffer = [0u8; 4];
                output.extend_from_slice(other.encode_utf8(&mut buffer).as_bytes());
            }
        }
    }
}

fn context_role_name(role: MaceContextRoleV1) -> &'static str {
    match role {
        MaceContextRoleV1::System => "system",
        MaceContextRoleV1::Developer => "developer",
        MaceContextRoleV1::User => "user",
        MaceContextRoleV1::Assistant => "assistant",
        MaceContextRoleV1::Tool => "tool",
    }
}

fn validate_source_metadata(
    metadata: &MaceExposureSourceMetadataV1,
) -> Result<(), MaceExposureCaptureRejectV1> {
    validate_identifier(&metadata.evidence_id)?;
    for digest in [
        metadata.evidence_payload_sha256.as_str(),
        metadata.dependency_closure_sha256.as_str(),
        metadata.transformation_receipt_sha256.as_str(),
        metadata.source_receipt_sha256.as_str(),
        metadata.authority_receipt_sha256.as_str(),
        metadata.admissibility_receipt_sha256.as_str(),
        metadata.lifecycle_revalidation_receipt_sha256.as_str(),
    ] {
        validate_digest(digest)?;
    }
    let mut parents = std::collections::BTreeSet::new();
    for parent in &metadata.parent_event_ids {
        validate_identifier(parent)?;
        if !parents.insert(parent) {
            return Err(MaceExposureCaptureRejectV1::InvalidIdentifier);
        }
    }
    Ok(())
}

fn validate_artifact_binding(
    artifact: &MaceRequestBodyArtifactV1,
    body: &[u8],
    expected: &MaceExpectedExposureBindingV1<'_>,
) -> Result<(), MaceExposureCaptureRejectV1> {
    if !artifact.immutable_object
        || artifact.plaintext_bytes != body.len() as u64
        || artifact.plaintext_sha256 != sha256_hex(body)
        || artifact.storage_profile_sha256 != expected.storage_profile_sha256
        || artifact.encryption_profile_sha256 != expected.encryption_profile_sha256
        || artifact.access_policy_sha256 != expected.access_policy_sha256
        || artifact.retention_policy_sha256 != expected.retention_policy_sha256
    {
        return Err(MaceExposureCaptureRejectV1::ArtifactBindingMismatch);
    }
    for digest in [
        artifact.ciphertext_sha256.as_str(),
        artifact.custody_receipt_sha256.as_str(),
    ] {
        validate_digest(digest)?;
    }
    validate_identifier(&artifact.artifact_id)?;
    Ok(())
}

fn validate_identifier(value: &str) -> Result<(), MaceExposureCaptureRejectV1> {
    if value.is_empty() || value.trim() != value || value.chars().count() > 256 {
        return Err(MaceExposureCaptureRejectV1::InvalidIdentifier);
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), MaceExposureCaptureRejectV1> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(MaceExposureCaptureRejectV1::InvalidDigest);
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mace_context_exposure_receipt::{
        parse_and_verify_mace_context_exposure_receipt_v1,
        MaceExposureAdmissionVerifierV1,
        MaceExposureObserverKeyResolverV1,
        MaceRequestBodyArtifactResolverV1,
        MaceContextExposureRejectV1,
        MaceExpectedExposureBindingV1,
        VerifiedMaceContextExposureReceiptV1,
    };
    use ed25519_dalek::{SigningKey, VerifyingKey};

    const D1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const D2: &str = "2222222222222222222222222222222222222222222222222222222222222222";
    const D3: &str = "3333333333333333333333333333333333333333333333333333333333333333";
    const D4: &str = "4444444444444444444444444444444444444444444444444444444444444444";
    const D5: &str = "5555555555555555555555555555555555555555555555555555555555555555";
    const D6: &str = "6666666666666666666666666666666666666666666666666666666666666666";
    const D7: &str = "7777777777777777777777777777777777777777777777777777777777777777";
    const D8: &str = "8888888888888888888888888888888888888888888888888888888888888888";
    const D9: &str = "9999999999999999999999999999999999999999999999999999999999999999";

    struct TestArtifactStore;
    impl MaceRequestBodyArtifactWriterV1 for TestArtifactStore {
        fn store_immutable_encrypted(
            &self, body: &[u8], expected: &MaceExpectedExposureBindingV1<'_>,
        ) -> Result<MaceRequestBodyArtifactV1, MaceExposureCaptureRejectV1> {
            let mut ciphertext_preimage = b"TEST-ONLY-NOT-ENCRYPTED\0".to_vec();
            ciphertext_preimage.extend_from_slice(body);
            Ok(MaceRequestBodyArtifactV1 {
                artifact_id: "test-artifact-1".to_owned(),
                plaintext_sha256: sha256_hex(body),
                ciphertext_sha256: sha256_hex(&ciphertext_preimage),
                plaintext_bytes: body.len() as u64,
                storage_profile_sha256: expected.storage_profile_sha256.to_owned(),
                encryption_profile_sha256: expected.encryption_profile_sha256.to_owned(),
                access_policy_sha256: expected.access_policy_sha256.to_owned(),
                retention_policy_sha256: expected.retention_policy_sha256.to_owned(),
                custody_receipt_sha256: D9.to_owned(),
                immutable_object: true,
            })
        }
    }

    struct TestTransport;
    impl MaceProviderTransportV1 for TestTransport {
        fn send_exact(
            &self, body: &[u8],
        ) -> Result<MaceProviderResponseV1, MaceExposureCaptureRejectV1> {
            Ok(MaceProviderResponseV1 {
                provider_request_id: Some("request-1".to_owned()),
                provider_response_id: "response-1".to_owned(),
                provider_response_body: br#"{"ok":true}"#.to_vec(),
                received_request_body_sha256: sha256_hex(body),
            })
        }
    }

    struct TestSigner { key: SigningKey, id: String }
    impl MaceExposureReceiptSignerV1 for TestSigner {
        fn key_id(&self) -> &str { &self.id }
        fn sign_digest(&self, digest: &[u8; 32])
            -> Result<[u8; 64], MaceContextExposureRejectV1> {
            use ed25519_dalek::Signer;
            Ok(self.key.sign(digest).to_bytes())
        }
    }

    struct TestArtifactResolver { body: Vec<u8> }
    impl MaceRequestBodyArtifactResolverV1 for TestArtifactResolver {
        fn resolve_and_verify(&self, _: &MaceRequestBodyArtifactV1)
            -> Result<Vec<u8>, MaceContextExposureRejectV1> {
            Ok(self.body.clone())
        }
    }
    struct TestKeyResolver { key: VerifyingKey, profile: String }
    impl MaceExposureObserverKeyResolverV1 for TestKeyResolver {
        fn resolve_key(&self, profile: &str, key_id: &str)
            -> Result<VerifyingKey, MaceContextExposureRejectV1> {
            let expected_id = crate::mace_context_exposure_receipt::ed25519_spki_key_id_v1(&self.key);
            if profile != self.profile || key_id != expected_id {
                return Err(MaceContextExposureRejectV1::UnknownObserverKey);
            }
            Ok(self.key.clone())
        }
    }
    struct AcceptEvidence;
    impl MaceExposureAdmissionVerifierV1 for AcceptEvidence {
        fn verify_admitted_exposure(
            &self, _: &MaceMemoryExposureSpanV1, _: &MaceExpectedExposureBindingV1<'_>,
        ) -> Result<(), MaceContextExposureRejectV1> { Ok(()) }
    }

    fn expected() -> MaceExpectedExposureBindingV1<'static> {
        MaceExpectedExposureBindingV1 {
            run_id: "run-1", pair_block_id: "pair-1", world_cluster_id: "cluster-1",
            query_id: "query-1", arm: MaceMemoryArmV1::ConventionalRetrieval,
            invocation_sequence: 0, decision_need_sha256: D1, candidate_policy_sha256: D2,
            source_registry_root_sha256: D3, lineage_profile_sha256: D4,
            lifecycle_profile_sha256: D5, renderer_profile_sha256: D6,
            transport_adapter_profile_sha256: D7, model_revision: "model-v1",
            clock_profile_sha256: D8, observer_profile_sha256: D9,
            storage_profile_sha256: D1, encryption_profile_sha256: D2,
            access_policy_sha256: D3, retention_policy_sha256: D4,
            use_valid_at_ms: 100, use_known_at_ms: 200,
            use_purpose_code: "mace.context.exposure",
            use_consumer_policy_sha256: D5,
            use_scope: broca_mace_policy::Scope {
                regime: broca_mace_policy::Regime::Baseline,
                objective: broca_mace_policy::Objective::Quality,
            },
            require_complete_dependency_lineage: true,
        }
    }
    fn metadata(id: &str) -> MaceExposureSourceMetadataV1 {
        MaceExposureSourceMetadataV1 {
            evidence_id: id.to_owned(), evidence_payload_sha256: D1.to_owned(),
            exposure_kind: MaceExposureKindV1::VerbatimSource,
            parent_event_ids: vec![format!("source-{id}")],
            dependency_closure_sha256: D2.to_owned(),
            transformation_receipt_sha256: D3.to_owned(),
            source_receipt_sha256: D4.to_owned(), authority_receipt_sha256: D5.to_owned(),
            admissibility_receipt_sha256: D6.to_owned(), lifecycle_revalidation_receipt_sha256: D7.to_owned(),
        }
    }
    fn signer() -> TestSigner {
        let key = SigningKey::from_bytes(&[8u8; 32]);
        let id = crate::mace_context_exposure_receipt::ed25519_spki_key_id_v1(&key.verifying_key());
        TestSigner { key, id }
    }

    #[test]
    fn maps_json_escaped_unicode_to_exact_utf8_byte_offsets() {
        let request = MaceProviderNeutralRequestV1 {
            model: "test-model".to_owned(),
            request_parameters: BTreeMap::new(),
            messages: vec![MaceRequestMessageV1 {
                role: MaceContextRoleV1::User,
                segments: vec![
                    MaceRequestSegmentV1 { text: "prefix 😀 ".to_owned(), origin: MaceRequestSegmentOriginV1::NonMemory },
                    MaceRequestSegmentV1 { text: "quote \" slash \\ newline\n".to_owned(), origin: MaceRequestSegmentOriginV1::Memory(metadata("e1")) },
                    MaceRequestSegmentV1 { text: " suffix".to_owned(), origin: MaceRequestSegmentOriginV1::NonMemory },
                ],
            }],
        };
        let serialized = serialize_mace_provider_request_with_exposure_map_v1(
            &request, MaceMemoryArmV1::ConventionalRetrieval
        ).unwrap();
        let span = &serialized.exposures()[0];
        let slice = &serialized.body()[span.span_start_byte as usize..span.span_end_byte as usize];
        assert_eq!(sha256_hex(slice), span.exposed_span_sha256);
        assert!(std::str::from_utf8(slice).unwrap().contains("\\\""));
        assert!(std::str::from_utf8(slice).unwrap().contains("\\n"));
        assert!(std::str::from_utf8(serialized.body()).unwrap().contains("😀"));
    }

    #[test]
    fn repeated_identical_memory_text_keeps_distinct_source_offsets() {
        let request = MaceProviderNeutralRequestV1 {
            model: "test-model".to_owned(),
            request_parameters: BTreeMap::new(),
            messages: vec![MaceRequestMessageV1 {
                role: MaceContextRoleV1::Tool,
                segments: vec![
                    MaceRequestSegmentV1 { text: "same".to_owned(), origin: MaceRequestSegmentOriginV1::Memory(metadata("e1")) },
                    MaceRequestSegmentV1 { text: "|".to_owned(), origin: MaceRequestSegmentOriginV1::NonMemory },
                    MaceRequestSegmentV1 { text: "same".to_owned(), origin: MaceRequestSegmentOriginV1::Memory(metadata("e2")) },
                ],
            }],
        };
        let serialized = serialize_mace_provider_request_with_exposure_map_v1(
            &request, MaceMemoryArmV1::ConventionalRetrieval
        ).unwrap();
        assert_eq!(serialized.exposures().len(), 2);
        assert_ne!(serialized.exposures()[0].span_start_byte, serialized.exposures()[1].span_start_byte);
        assert_eq!(serialized.exposures()[0].evidence_id, "e1");
        assert_eq!(serialized.exposures()[1].evidence_id, "e2");
    }

    #[test]
    fn full_fake_transport_round_trip_uses_identical_sent_bytes() {
        let request = MaceProviderNeutralRequestV1 {
            model: "test-model".to_owned(),
            request_parameters: BTreeMap::new(),
            messages: vec![MaceRequestMessageV1 {
                role: MaceContextRoleV1::User,
                segments: vec![
                    MaceRequestSegmentV1 { text: "decision context: ".to_owned(), origin: MaceRequestSegmentOriginV1::NonMemory },
                    MaceRequestSegmentV1 { text: "source value".to_owned(), origin: MaceRequestSegmentOriginV1::Memory(metadata("e1")) },
                ],
            }],
        };
        let expected = expected();
        let signer = signer();
        let (receipt, response) = capture_send_and_sign_mace_invocation_v1(
            &request, &expected,
            MaceExposureCaptureContextV1 {
                receipt_id: "receipt-1", captured_at_monotonic_ns: 42,
                candidate_decision_trace_sha256: D8, admission_denials: vec![],
                model_declared_support_event_ids: vec!["e1".to_owned()],
            },
            &AcceptEvidence,
            &TestArtifactStore, &TestTransport, &signer,
        ).unwrap();
        assert_eq!(response.received_request_body_sha256, receipt.capture.provider_request_body_sha256);

        let body = {
            let serialized = serialize_mace_provider_request_with_exposure_map_v1(
                &request, MaceMemoryArmV1::ConventionalRetrieval
            ).unwrap();
            serialized.body().to_vec()
        };
        let body_resolver = TestArtifactResolver { body };
        let keys = TestKeyResolver {
            key: signer.key.verifying_key(),
            profile: expected.observer_profile_sha256.to_owned(),
        };
        let verified: VerifiedMaceContextExposureReceiptV1 =
            parse_and_verify_mace_context_exposure_receipt_v1(
                &serde_json::to_vec(&receipt).unwrap(), &expected,
                &body_resolver, &keys, &AcceptEvidence
            ).unwrap();
        assert_eq!(verified.exposed_evidence_ids().collect::<Vec<_>>(), vec!["e1"]);
    }

    #[test]
    fn serializes_non_memory_provider_parameters_in_deterministic_json() {
        let mut request_parameters = BTreeMap::new();
        request_parameters.insert("temperature".to_owned(), serde_json::json!(0.2));
        request_parameters.insert("top_p".to_owned(), serde_json::json!(0.7));
        let request = MaceProviderNeutralRequestV1 {
            model: "test-model".to_owned(),
            request_parameters,
            messages: vec![MaceRequestMessageV1 {
                role: MaceContextRoleV1::User,
                segments: vec![MaceRequestSegmentV1 {
                    text: "plain".to_owned(),
                    origin: MaceRequestSegmentOriginV1::NonMemory,
                }],
            }],
        };
        let serialized = serialize_mace_provider_request_with_exposure_map_v1(
            &request, MaceMemoryArmV1::ConventionalRetrieval
        ).unwrap();
        let parsed: Value = serde_json::from_slice(serialized.body()).unwrap();
        assert_eq!(parsed["model"], "test-model");
        assert_eq!(parsed["temperature"], serde_json::json!(0.2));
        assert_eq!(parsed["top_p"], serde_json::json!(0.7));
    }

    #[test]
    fn rejects_request_model_alias_not_equal_to_pinned_revision() {
        let request = MaceProviderNeutralRequestV1 {
            model: "un-pinned-alias".to_owned(),
            request_parameters: BTreeMap::new(),
            messages: vec![MaceRequestMessageV1 {
                role: MaceContextRoleV1::User,
                segments: vec![MaceRequestSegmentV1 {
                    text: "plain".to_owned(),
                    origin: MaceRequestSegmentOriginV1::NonMemory,
                }],
            }],
        };
        assert_eq!(
            capture_send_and_sign_mace_invocation_v1(
                &request, &expected(),
                MaceExposureCaptureContextV1 {
                    receipt_id: "receipt-1", captured_at_monotonic_ns: 42,
                    candidate_decision_trace_sha256: D8, admission_denials: vec![],
                    model_declared_support_event_ids: vec![],
                },
                &AcceptEvidence,
                &TestArtifactStore, &TestTransport, &signer(),
            ).unwrap_err(),
            MaceExposureCaptureRejectV1::EmptyModel
        );
    }

    #[test]
    fn true_no_memory_arm_rejects_typed_memory_segments() {
        let request = MaceProviderNeutralRequestV1 {
            model: "test-model".to_owned(),
            request_parameters: BTreeMap::new(),
            messages: vec![MaceRequestMessageV1 {
                role: MaceContextRoleV1::User,
                segments: vec![MaceRequestSegmentV1 {
                    text: "should not enter".to_owned(),
                    origin: MaceRequestSegmentOriginV1::Memory(metadata("e1")),
                }],
            }],
        };
        assert_eq!(
            serialize_mace_provider_request_with_exposure_map_v1(
                &request, MaceMemoryArmV1::NoMemory
            ).unwrap_err(),
            MaceExposureCaptureRejectV1::MemoryInNoMemoryArm
        );
    }

    #[test]
    fn artifact_profile_mismatch_is_rejected_before_transport() {
        struct WrongArtifactStore;
        impl MaceRequestBodyArtifactWriterV1 for WrongArtifactStore {
            fn store_immutable_encrypted(
                &self, body: &[u8], expected: &MaceExpectedExposureBindingV1<'_>,
            ) -> Result<MaceRequestBodyArtifactV1, MaceExposureCaptureRejectV1> {
                let mut a = TestArtifactStore.store_immutable_encrypted(body, expected)?;
                a.storage_profile_sha256 = D9.to_owned();
                Ok(a)
            }
        }
        let request = MaceProviderNeutralRequestV1 {
            model: "test-model".to_owned(),
            request_parameters: BTreeMap::new(),
            messages: vec![MaceRequestMessageV1 {
                role: MaceContextRoleV1::User,
                segments: vec![MaceRequestSegmentV1 { text: "plain".to_owned(), origin: MaceRequestSegmentOriginV1::NonMemory }],
            }],
        };
        assert_eq!(
            capture_send_and_sign_mace_invocation_v1(
                &request, &expected(),
                MaceExposureCaptureContextV1 {
                    receipt_id: "receipt-1", captured_at_monotonic_ns: 42,
                    candidate_decision_trace_sha256: D8, admission_denials: vec![],
                    model_declared_support_event_ids: vec![],
                },
                &AcceptEvidence,
                &WrongArtifactStore, &TestTransport, &signer()
            ).unwrap_err(),
            MaceExposureCaptureRejectV1::ArtifactBindingMismatch
        );
    }

    #[test]
    fn rejects_unrevalidated_memory_before_artifact_storage_or_transport() {
        struct RejectEvidence;
        impl MaceExposureAdmissionVerifierV1 for RejectEvidence {
            fn verify_admitted_exposure(
                &self,
                _: &MaceMemoryExposureSpanV1,
                _: &MaceExpectedExposureBindingV1<'_>,
            ) -> Result<(), MaceContextExposureRejectV1> {
                Err(MaceContextExposureRejectV1::EvidenceAdmissionRejected)
            }
        }
        struct PanicArtifactStore;
        impl MaceRequestBodyArtifactWriterV1 for PanicArtifactStore {
            fn store_immutable_encrypted(
                &self,
                _: &[u8],
                _: &MaceExpectedExposureBindingV1<'_>,
            ) -> Result<MaceRequestBodyArtifactV1, MaceExposureCaptureRejectV1> {
                panic!("artifact storage must not run before use-time admission");
            }
        }
        struct PanicTransport;
        impl MaceProviderTransportV1 for PanicTransport {
            fn send_exact(
                &self,
                _: &[u8],
            ) -> Result<MaceProviderResponseV1, MaceExposureCaptureRejectV1> {
                panic!("provider transport must not run before use-time admission");
            }
        }

        let request = MaceProviderNeutralRequestV1 {
            model: "model-v1".to_owned(),
            request_parameters: BTreeMap::new(),
            messages: vec![MaceRequestMessageV1 {
                role: MaceContextRoleV1::User,
                segments: vec![MaceRequestSegmentV1 {
                    text: "memory-derived source".to_owned(),
                    origin: MaceRequestSegmentOriginV1::Memory(metadata("e1")),
                }],
            }],
        };
        let result = capture_send_and_sign_mace_invocation_v1(
            &request,
            &expected(),
            MaceExposureCaptureContextV1 {
                receipt_id: "receipt-1",
                captured_at_monotonic_ns: 42,
                candidate_decision_trace_sha256: D8,
                admission_denials: vec![],
                model_declared_support_event_ids: vec!["e1".to_owned()],
            },
            &RejectEvidence,
            &PanicArtifactStore,
            &PanicTransport,
            &signer(),
        );
        assert_eq!(result, Err(MaceExposureCaptureRejectV1::EvidenceAdmissionRejected));
    }

}
