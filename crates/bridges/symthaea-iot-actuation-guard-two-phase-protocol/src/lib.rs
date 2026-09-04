// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Two-phase evidence-only wire contract for privileged cyber-physical actuation.
//!
//! The existing v1 guard request deliberately remains unchanged. Its one-shot frame
//! carries transport and interlock evidence together and is useful as a pre-semantic
//! evidence boundary, but that shape cannot prove that a hardware controller observed
//! a semantic head that the privileged guard persisted only after request admission.
//!
//! This protocol makes the stronger causal order representable without carrying any
//! portable authority token:
//!
//! ```text
//! phase 1: exact Xenia receipt + exact physical-effect payload
//!     -> privileged transport verification
//!     -> durable semantic reservation (outside this crate)
//!     -> SemanticReservationChallengeV1
//!
//! phase 2: challenge-bound device attestation result
//!     -> controller statement binds challenge + exact attestation-result digest
//!     -> controller evidence/signature
//!     -> bounded/canonical post-reservation evidence
//! ```
//!
//! Trust registries, semantic persistence, trusted time, verifier implementations and
//! HAL/device handles remain privileged-process state and never cross this wire.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_device_protocol::DeviceSemanticHead;
use symthaea_iot_final_gate::{
    MAX_INTERLOCK_COUNT, MAX_INTERLOCK_LABEL_BYTES, MAX_PHYSICAL_INTERLOCK_EVIDENCE_BYTES,
    MAX_PHYSICAL_INTERLOCK_REPORT_LIFETIME_MS,
};
use symthaea_iot_posture::{DeviceAttestationResultV1, MAX_ATTESTATION_SIGNATURE_BYTES};
use symthaea_iot_transport_receipt::{
    MAX_XENIA_PHYSICAL_EFFECT_PAYLOAD_BYTES, MAX_XENIA_RECEIPT_BYTES, TransportTrustHead,
};
use thiserror::Error;

/// Phase-1 admission wire schema. Version 2 deliberately excludes interlock evidence.
pub const ACTUATION_GUARD_ADMISSION_SCHEMA_VERSION: u16 = 2;
/// Post-reservation challenge schema.
pub const SEMANTIC_RESERVATION_CHALLENGE_SCHEMA_VERSION: u16 = 1;
/// Controller-signed post-reservation statement schema.
pub const POST_RESERVATION_INTERLOCK_STATEMENT_SCHEMA_VERSION: u16 = 1;
/// Phase-2 response wire schema.
pub const ACTUATION_GUARD_POST_RESERVATION_SCHEMA_VERSION: u16 = 2;
/// Short ceiling for one post-persistence challenge round trip.
pub const MAX_SEMANTIC_RESERVATION_CHALLENGE_LIFETIME_MS: u64 = 5_000;
/// Bound canonical serialized attestation-result bytes before decode.
pub const MAX_DEVICE_ATTESTATION_RESULT_BYTES: usize = 96 * 1024;
/// Bound canonical serialized post-reservation interlock report bytes.
pub const MAX_POST_RESERVATION_INTERLOCK_REPORT_BYTES: usize = 16 * 1024;
/// Hard phase-1 outer-frame ceiling, checked before deserialization.
pub const MAX_GUARD_ADMISSION_FRAME_BYTES: usize = 128 * 1024;
/// Hard phase-2 outer-frame ceiling, checked before deserialization.
pub const MAX_GUARD_POST_RESERVATION_FRAME_BYTES: usize = 160 * 1024;
/// Conservative device identifier bound for this IPC contract.
pub const MAX_GUARD_DEVICE_ID_BYTES: usize = 512;

const ADMISSION_DOMAIN: &[u8] = b"symthaea-iot-actuation-guard-admission-v2\0";
const CHALLENGE_DOMAIN: &[u8] = b"symthaea-iot-semantic-reservation-challenge-v1\0";
const DEVICE_ATTESTATION_OBJECT_DOMAIN: &[u8] =
    b"symthaea-iot-device-attestation-object-v1\0";
const INTERLOCK_STATEMENT_DOMAIN: &[u8] =
    b"symthaea-iot-post-reservation-interlock-statement-v1\0";
const INTERLOCK_REPORT_DOMAIN: &[u8] = b"symthaea-iot-post-reservation-interlock-report-v1\0";
const POST_RESERVATION_RESPONSE_DOMAIN: &[u8] =
    b"symthaea-iot-actuation-guard-post-reservation-v2\0";

/// Phase-1 request from an unprivileged caller.
///
/// No interlock report is accepted in this frame. A controller report that is intended
/// to authorize later physical composition must be generated only after the privileged
/// guard has persisted a semantic reservation and issued a challenge for that head.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationGuardAdmissionRequestV2 {
    pub schema_version: u16,
    pub raw_transport_receipt: Vec<u8>,
    pub raw_physical_effect_payload: Vec<u8>,
}

impl ActuationGuardAdmissionRequestV2 {
    pub fn validate_structure(&self) -> Result<(), TwoPhaseGuardProtocolError> {
        if self.schema_version != ACTUATION_GUARD_ADMISSION_SCHEMA_VERSION {
            return Err(TwoPhaseGuardProtocolError::UnsupportedAdmissionSchema);
        }
        if self.raw_transport_receipt.is_empty()
            || self.raw_transport_receipt.len() > MAX_XENIA_RECEIPT_BYTES
        {
            return Err(TwoPhaseGuardProtocolError::TransportReceiptSizeOutOfBounds);
        }
        if self.raw_physical_effect_payload.is_empty()
            || self.raw_physical_effect_payload.len() > MAX_XENIA_PHYSICAL_EFFECT_PAYLOAD_BYTES
        {
            return Err(TwoPhaseGuardProtocolError::PhysicalPayloadSizeOutOfBounds);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, TwoPhaseGuardProtocolError> {
        self.validate_structure()?;
        let bytes = bincode::serialize(self).map_err(TwoPhaseGuardProtocolError::Encoding)?;
        if bytes.len() > MAX_GUARD_ADMISSION_FRAME_BYTES {
            return Err(TwoPhaseGuardProtocolError::AdmissionFrameSizeOutOfBounds);
        }
        Ok(bytes)
    }

    /// Audit-only commitment to the exact canonical phase-1 frame.
    pub fn digest(&self) -> Result<Digest32, TwoPhaseGuardProtocolError> {
        Ok(digest_frame(ADMISSION_DOMAIN, &self.canonical_bytes()?))
    }
}

/// Bounded/canonical phase-1 evidence. This is not authority.
#[derive(Debug)]
pub struct DecodedGuardAdmissionV2 {
    request_digest: Digest32,
    raw_transport_receipt: Vec<u8>,
    raw_physical_effect_payload: Vec<u8>,
}

impl DecodedGuardAdmissionV2 {
    pub const fn request_digest(&self) -> Digest32 {
        self.request_digest
    }

    pub fn raw_transport_receipt(&self) -> &[u8] {
        &self.raw_transport_receipt
    }

    pub fn raw_physical_effect_payload(&self) -> &[u8] {
        &self.raw_physical_effect_payload
    }

    pub fn into_parts(self) -> (Vec<u8>, Vec<u8>, Digest32) {
        (
            self.raw_transport_receipt,
            self.raw_physical_effect_payload,
            self.request_digest,
        )
    }
}

/// Decode one exact phase-1 frame. Trailing bytes cannot smuggle policy, trust state,
/// runtime observations or a premature interlock report.
pub fn decode_canonical_guard_admission_v2(
    frame: &[u8],
) -> Result<DecodedGuardAdmissionV2, TwoPhaseGuardProtocolError> {
    if frame.is_empty() || frame.len() > MAX_GUARD_ADMISSION_FRAME_BYTES {
        return Err(TwoPhaseGuardProtocolError::AdmissionFrameSizeOutOfBounds);
    }
    let request: ActuationGuardAdmissionRequestV2 =
        bincode::deserialize(frame).map_err(TwoPhaseGuardProtocolError::Decoding)?;
    request.validate_structure()?;
    if request.canonical_bytes()? != frame {
        return Err(TwoPhaseGuardProtocolError::NonCanonicalAdmissionEncoding);
    }
    Ok(DecodedGuardAdmissionV2 {
        request_digest: digest_frame(ADMISSION_DOMAIN, frame),
        raw_transport_receipt: request.raw_transport_receipt,
        raw_physical_effect_payload: request.raw_physical_effect_payload,
    })
}

/// Privileged challenge emitted only after one exact semantic successor head has been
/// durably persisted by the guard's trusted persistence boundary.
///
/// `nonce` must be generated by the privileged deployment. This protocol crate has no
/// RNG and does not accept caller-controlled challenge creation as a trust claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticReservationChallengeV1 {
    pub schema_version: u16,
    pub nonce: [u8; 32],
    pub admission_request_digest: Digest32,
    pub envelope_digest: Digest32,
    pub transport_receipt_digest: Digest32,
    pub device: ResourceRef,
    pub transport_trust_head: TransportTrustHead,
    pub semantic_head: DeviceSemanticHead,
    pub persisted_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
}

impl SemanticReservationChallengeV1 {
    pub fn validate(&self) -> Result<(), TwoPhaseGuardProtocolError> {
        if self.schema_version != SEMANTIC_RESERVATION_CHALLENGE_SCHEMA_VERSION {
            return Err(TwoPhaseGuardProtocolError::UnsupportedChallengeSchema);
        }
        if self.nonce == [0; 32] {
            return Err(TwoPhaseGuardProtocolError::ZeroChallengeNonce);
        }
        if self.admission_request_digest == zero_digest()
            || self.envelope_digest == zero_digest()
            || self.transport_receipt_digest == zero_digest()
            || self.transport_trust_head.sequence == 0
            || self.transport_trust_head.digest == zero_digest()
            || self.semantic_head.generation == 0
            || self.semantic_head.digest == zero_digest()
        {
            return Err(TwoPhaseGuardProtocolError::ZeroChallengeSecurityCommitment);
        }
        if !valid_device(&self.device) {
            return Err(TwoPhaseGuardProtocolError::InvalidChallengeDevice);
        }
        if self.persisted_at_unix_ms == 0 {
            return Err(TwoPhaseGuardProtocolError::InvalidChallengeWindow);
        }
        let lifetime = self
            .expires_at_unix_ms
            .checked_sub(self.persisted_at_unix_ms)
            .ok_or(TwoPhaseGuardProtocolError::InvalidChallengeWindow)?;
        if lifetime == 0 || lifetime > MAX_SEMANTIC_RESERVATION_CHALLENGE_LIFETIME_MS {
            return Err(TwoPhaseGuardProtocolError::InvalidChallengeWindow);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, TwoPhaseGuardProtocolError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(CHALLENGE_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.nonce);
        update_digest(&mut h, self.admission_request_digest);
        update_digest(&mut h, self.envelope_digest);
        update_digest(&mut h, self.transport_receipt_digest);
        update_string(&mut h, &self.device.0);
        h.update(&self.transport_trust_head.sequence.to_be_bytes());
        update_digest(&mut h, self.transport_trust_head.digest);
        h.update(&self.semantic_head.generation.to_be_bytes());
        update_digest(&mut h, self.semantic_head.digest);
        h.update(&self.persisted_at_unix_ms.to_be_bytes());
        h.update(&self.expires_at_unix_ms.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    pub fn is_fresh_at(&self, now_unix_ms: u64) -> bool {
        now_unix_ms >= self.persisted_at_unix_ms && now_unix_ms < self.expires_at_unix_ms
    }
}

/// Exact domain-separated commitment to one canonical device-attestation result.
///
/// The post-reservation controller statement commits this value, so the controller
/// cannot produce its final statement before the exact challenge-bound device appraisal
/// object exists. Signature verification of that appraisal remains a later guard TCB
/// responsibility.
pub fn device_attestation_result_digest(
    result: &DeviceAttestationResultV1,
) -> Result<Digest32, TwoPhaseGuardProtocolError> {
    result
        .body
        .validate_structure()
        .map_err(|_| TwoPhaseGuardProtocolError::InvalidDeviceAttestationStructure)?;
    if result.signature.is_empty() || result.signature.len() > MAX_ATTESTATION_SIGNATURE_BYTES {
        return Err(TwoPhaseGuardProtocolError::DeviceAttestationSignatureSizeOutOfBounds);
    }
    let bytes = bincode::serialize(result)
        .map_err(|_| TwoPhaseGuardProtocolError::InvalidDeviceAttestationEncoding)?;
    if bytes.len() > MAX_DEVICE_ATTESTATION_RESULT_BYTES {
        return Err(TwoPhaseGuardProtocolError::DeviceAttestationSizeOutOfBounds);
    }
    Ok(digest_frame(DEVICE_ATTESTATION_OBJECT_DOMAIN, &bytes))
}

/// Exact controller statement signed after the semantic reservation challenge exists.
///
/// The signature/evidence commitment is intentionally *not* a field of this type. The
/// controller can therefore compute this digest before producing its signature; the
/// enclosing report commits the resulting raw evidence separately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostReservationInterlockStatementV1 {
    pub schema_version: u16,
    pub challenge_digest: Digest32,
    /// Exact canonical challenge-bound device-attestation result observed before the
    /// controller produces this statement.
    pub device_attestation_result_digest: Digest32,
    pub controller_id: String,
    pub device: ResourceRef,
    pub envelope_digest: Digest32,
    pub semantic_head: DeviceSemanticHead,
    pub transport_trust_head: TransportTrustHead,
    pub asserted_interlocks: BTreeSet<String>,
    pub checked_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
}

impl PostReservationInterlockStatementV1 {
    pub fn validate_structure(&self) -> Result<(), TwoPhaseGuardProtocolError> {
        if self.schema_version != POST_RESERVATION_INTERLOCK_STATEMENT_SCHEMA_VERSION {
            return Err(TwoPhaseGuardProtocolError::UnsupportedInterlockStatementSchema);
        }
        if self.challenge_digest == zero_digest()
            || self.device_attestation_result_digest == zero_digest()
            || self.envelope_digest == zero_digest()
            || self.semantic_head.generation == 0
            || self.semantic_head.digest == zero_digest()
            || self.transport_trust_head.sequence == 0
            || self.transport_trust_head.digest == zero_digest()
        {
            return Err(TwoPhaseGuardProtocolError::ZeroInterlockSecurityCommitment);
        }
        if !valid_label(&self.controller_id) || !valid_device(&self.device) {
            return Err(TwoPhaseGuardProtocolError::InvalidInterlockIdentity);
        }
        if self.asserted_interlocks.is_empty()
            || self.asserted_interlocks.len() > MAX_INTERLOCK_COUNT
            || self.asserted_interlocks.iter().any(|label| !valid_label(label))
        {
            return Err(TwoPhaseGuardProtocolError::InvalidInterlockSurface);
        }
        let lifetime = self
            .expires_at_unix_ms
            .checked_sub(self.checked_at_unix_ms)
            .ok_or(TwoPhaseGuardProtocolError::InvalidInterlockWindow)?;
        if lifetime == 0 || lifetime > MAX_PHYSICAL_INTERLOCK_REPORT_LIFETIME_MS {
            return Err(TwoPhaseGuardProtocolError::InvalidInterlockWindow);
        }
        Ok(())
    }

    /// Domain-separated digest authenticated by the controller evidence profile.
    pub fn digest(&self) -> Result<Digest32, TwoPhaseGuardProtocolError> {
        self.validate_structure()?;
        let mut h = blake3::Hasher::new();
        h.update(INTERLOCK_STATEMENT_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        update_digest(&mut h, self.challenge_digest);
        update_digest(&mut h, self.device_attestation_result_digest);
        update_string(&mut h, &self.controller_id);
        update_string(&mut h, &self.device.0);
        update_digest(&mut h, self.envelope_digest);
        h.update(&self.semantic_head.generation.to_be_bytes());
        update_digest(&mut h, self.semantic_head.digest);
        h.update(&self.transport_trust_head.sequence.to_be_bytes());
        update_digest(&mut h, self.transport_trust_head.digest);
        update_strings(&mut h, &self.asserted_interlocks);
        h.update(&self.checked_at_unix_ms.to_be_bytes());
        h.update(&self.expires_at_unix_ms.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }
}

/// Complete portable interlock evidence after the controller signature exists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostReservationInterlockReportV1 {
    pub statement: PostReservationInterlockStatementV1,
    pub evidence_digest: Digest32,
}

impl PostReservationInterlockReportV1 {
    pub fn validate_structure(&self) -> Result<(), TwoPhaseGuardProtocolError> {
        self.statement.validate_structure()?;
        if self.evidence_digest == zero_digest() {
            return Err(TwoPhaseGuardProtocolError::ZeroInterlockEvidenceCommitment);
        }
        Ok(())
    }

    /// Full audit/object commitment. Controller authentication uses `statement.digest()`;
    /// exact evidence bytes are independently bound by `evidence_digest`.
    pub fn full_digest(&self) -> Result<Digest32, TwoPhaseGuardProtocolError> {
        self.validate_structure()?;
        let mut h = blake3::Hasher::new();
        h.update(INTERLOCK_REPORT_DOMAIN);
        update_digest(&mut h, self.statement.digest()?);
        update_digest(&mut h, self.evidence_digest);
        Ok(Digest32(*h.finalize().as_bytes()))
    }
}

/// Phase-2 portable evidence returned after the privileged challenge was issued.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationGuardPostReservationResponseV2 {
    pub schema_version: u16,
    /// Canonical serialized `DeviceAttestationResultV1` whose body binds the exact
    /// reservation challenge digest.
    pub raw_device_attestation_result: Vec<u8>,
    /// Canonical serialized `PostReservationInterlockReportV1`.
    pub raw_interlock_report: Vec<u8>,
    /// Exact raw controller evidence/signature committed by the report.
    pub raw_interlock_evidence: Vec<u8>,
}

impl ActuationGuardPostReservationResponseV2 {
    pub fn validate_structure(&self) -> Result<(), TwoPhaseGuardProtocolError> {
        if self.schema_version != ACTUATION_GUARD_POST_RESERVATION_SCHEMA_VERSION {
            return Err(TwoPhaseGuardProtocolError::UnsupportedPostReservationSchema);
        }
        if self.raw_device_attestation_result.is_empty()
            || self.raw_device_attestation_result.len() > MAX_DEVICE_ATTESTATION_RESULT_BYTES
        {
            return Err(TwoPhaseGuardProtocolError::DeviceAttestationSizeOutOfBounds);
        }
        if self.raw_interlock_report.is_empty()
            || self.raw_interlock_report.len() > MAX_POST_RESERVATION_INTERLOCK_REPORT_BYTES
        {
            return Err(TwoPhaseGuardProtocolError::PostReservationReportSizeOutOfBounds);
        }
        if self.raw_interlock_evidence.is_empty()
            || self.raw_interlock_evidence.len() > MAX_PHYSICAL_INTERLOCK_EVIDENCE_BYTES
        {
            return Err(TwoPhaseGuardProtocolError::InterlockEvidenceSizeOutOfBounds);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, TwoPhaseGuardProtocolError> {
        self.validate_structure()?;
        let bytes = bincode::serialize(self).map_err(TwoPhaseGuardProtocolError::Encoding)?;
        if bytes.len() > MAX_GUARD_POST_RESERVATION_FRAME_BYTES {
            return Err(TwoPhaseGuardProtocolError::PostReservationFrameSizeOutOfBounds);
        }
        Ok(bytes)
    }

    pub fn digest(&self) -> Result<Digest32, TwoPhaseGuardProtocolError> {
        Ok(digest_frame(
            POST_RESERVATION_RESPONSE_DOMAIN,
            &self.canonical_bytes()?,
        ))
    }
}

/// Canonically parsed phase-2 evidence bound to one exact privileged challenge.
/// This remains portable evidence, not actuator authority.
#[derive(Debug)]
pub struct DecodedPostReservationEvidenceV2 {
    response_digest: Digest32,
    raw_device_attestation_result: Vec<u8>,
    device_attestation_result: DeviceAttestationResultV1,
    interlock_report: PostReservationInterlockReportV1,
    raw_interlock_evidence: Vec<u8>,
}

impl DecodedPostReservationEvidenceV2 {
    pub const fn response_digest(&self) -> Digest32 {
        self.response_digest
    }

    pub fn raw_device_attestation_result(&self) -> &[u8] {
        &self.raw_device_attestation_result
    }

    pub fn device_attestation_result(&self) -> &DeviceAttestationResultV1 {
        &self.device_attestation_result
    }

    pub fn interlock_report(&self) -> &PostReservationInterlockReportV1 {
        &self.interlock_report
    }

    pub fn raw_interlock_evidence(&self) -> &[u8] {
        &self.raw_interlock_evidence
    }

    pub fn into_parts(
        self,
    ) -> (
        Vec<u8>,
        DeviceAttestationResultV1,
        PostReservationInterlockReportV1,
        Vec<u8>,
        Digest32,
    ) {
        (
            self.raw_device_attestation_result,
            self.device_attestation_result,
            self.interlock_report,
            self.raw_interlock_evidence,
            self.response_digest,
        )
    }
}

/// Decode and cross-bind one phase-2 frame to the exact privileged challenge.
///
/// This function deliberately performs no signature verification. It establishes the
/// wire/correlation invariants that later fixed guard verifiers must consume:
///
/// - both independent evidence streams bind the exact challenge digest;
/// - both target the challenged device;
/// - controller statement binds the exact envelope, semantic head and transport head;
/// - controller statement additionally binds the exact canonical device-appraisal object;
/// - controller observation cannot predate semantic persistence;
/// - controller evidence lifetime remains inside the challenge window; and
/// - raw controller evidence matches its independent commitment.
pub fn decode_canonical_post_reservation_response_v2(
    frame: &[u8],
    challenge: &SemanticReservationChallengeV1,
) -> Result<DecodedPostReservationEvidenceV2, TwoPhaseGuardProtocolError> {
    challenge.validate()?;
    if frame.is_empty() || frame.len() > MAX_GUARD_POST_RESERVATION_FRAME_BYTES {
        return Err(TwoPhaseGuardProtocolError::PostReservationFrameSizeOutOfBounds);
    }
    let response: ActuationGuardPostReservationResponseV2 =
        bincode::deserialize(frame).map_err(TwoPhaseGuardProtocolError::Decoding)?;
    response.validate_structure()?;
    if response.canonical_bytes()? != frame {
        return Err(TwoPhaseGuardProtocolError::NonCanonicalPostReservationEncoding);
    }

    let attestation: DeviceAttestationResultV1 =
        bincode::deserialize(&response.raw_device_attestation_result)
            .map_err(|_| TwoPhaseGuardProtocolError::InvalidDeviceAttestationEncoding)?;
    attestation
        .body
        .validate_structure()
        .map_err(|_| TwoPhaseGuardProtocolError::InvalidDeviceAttestationStructure)?;
    if attestation.signature.is_empty() || attestation.signature.len() > MAX_ATTESTATION_SIGNATURE_BYTES {
        return Err(TwoPhaseGuardProtocolError::DeviceAttestationSignatureSizeOutOfBounds);
    }
    let canonical_attestation = bincode::serialize(&attestation)
        .map_err(|_| TwoPhaseGuardProtocolError::InvalidDeviceAttestationEncoding)?;
    if canonical_attestation != response.raw_device_attestation_result {
        return Err(TwoPhaseGuardProtocolError::NonCanonicalDeviceAttestationEncoding);
    }
    let attestation_result_digest =
        digest_frame(DEVICE_ATTESTATION_OBJECT_DOMAIN, &canonical_attestation);

    let report: PostReservationInterlockReportV1 =
        bincode::deserialize(&response.raw_interlock_report)
            .map_err(|_| TwoPhaseGuardProtocolError::InvalidPostReservationReportEncoding)?;
    report.validate_structure()?;
    let canonical_report = bincode::serialize(&report)
        .map_err(|_| TwoPhaseGuardProtocolError::InvalidPostReservationReportEncoding)?;
    if canonical_report != response.raw_interlock_report {
        return Err(TwoPhaseGuardProtocolError::NonCanonicalPostReservationReportEncoding);
    }

    if Digest32(*blake3::hash(&response.raw_interlock_evidence).as_bytes())
        != report.evidence_digest
    {
        return Err(TwoPhaseGuardProtocolError::InterlockEvidenceDigestMismatch);
    }

    let challenge_digest = challenge.digest()?;
    if attestation.body.challenge_digest != challenge_digest
        || report.statement.challenge_digest != challenge_digest
    {
        return Err(TwoPhaseGuardProtocolError::ChallengeBindingMismatch);
    }
    if report.statement.device_attestation_result_digest != attestation_result_digest {
        return Err(TwoPhaseGuardProtocolError::InterlockDeviceAttestationDigestMismatch);
    }
    if attestation.body.device != challenge.device || report.statement.device != challenge.device {
        return Err(TwoPhaseGuardProtocolError::ChallengeDeviceMismatch);
    }
    if report.statement.envelope_digest != challenge.envelope_digest {
        return Err(TwoPhaseGuardProtocolError::ChallengeEnvelopeMismatch);
    }
    if report.statement.semantic_head != challenge.semantic_head {
        return Err(TwoPhaseGuardProtocolError::ChallengeSemanticHeadMismatch);
    }
    if report.statement.transport_trust_head != challenge.transport_trust_head {
        return Err(TwoPhaseGuardProtocolError::ChallengeTransportHeadMismatch);
    }
    if report.statement.checked_at_unix_ms < challenge.persisted_at_unix_ms {
        return Err(TwoPhaseGuardProtocolError::InterlockPredatesSemanticPersistence);
    }
    if report.statement.expires_at_unix_ms > challenge.expires_at_unix_ms {
        return Err(TwoPhaseGuardProtocolError::InterlockOutlivesChallenge);
    }

    Ok(DecodedPostReservationEvidenceV2 {
        response_digest: digest_frame(POST_RESERVATION_RESPONSE_DOMAIN, frame),
        raw_device_attestation_result: response.raw_device_attestation_result,
        device_attestation_result: attestation,
        interlock_report: report,
        raw_interlock_evidence: response.raw_interlock_evidence,
    })
}

fn valid_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_INTERLOCK_LABEL_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn valid_device(device: &ResourceRef) -> bool {
    !device.0.is_empty()
        && device.0.len() <= MAX_GUARD_DEVICE_ID_BYTES
        && device.0.trim() == device.0
        && !device.0.chars().any(char::is_control)
}

fn digest_frame(domain: &[u8], frame: &[u8]) -> Digest32 {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    h.update(&(frame.len() as u64).to_be_bytes());
    h.update(frame);
    Digest32(*h.finalize().as_bytes())
}

fn update_string(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

fn update_strings(h: &mut blake3::Hasher, values: &BTreeSet<String>) {
    h.update(&(values.len() as u64).to_be_bytes());
    for value in values {
        update_string(h, value);
    }
}

fn update_digest(h: &mut blake3::Hasher, Digest32(value): Digest32) {
    h.update(&value);
}

const fn zero_digest() -> Digest32 {
    Digest32([0; 32])
}

#[derive(Debug, Error)]
pub enum TwoPhaseGuardProtocolError {
    #[error("unsupported two-phase admission schema")]
    UnsupportedAdmissionSchema,
    #[error("two-phase admission frame size is outside accepted bounds")]
    AdmissionFrameSizeOutOfBounds,
    #[error("guard transport receipt size is outside accepted bounds")]
    TransportReceiptSizeOutOfBounds,
    #[error("guard physical payload size is outside accepted bounds")]
    PhysicalPayloadSizeOutOfBounds,
    #[error("two-phase admission request is not canonically encoded")]
    NonCanonicalAdmissionEncoding,
    #[error("unsupported semantic-reservation challenge schema")]
    UnsupportedChallengeSchema,
    #[error("semantic-reservation challenge nonce is zero")]
    ZeroChallengeNonce,
    #[error("semantic-reservation challenge contains a zero security commitment")]
    ZeroChallengeSecurityCommitment,
    #[error("semantic-reservation challenge targets an invalid device")]
    InvalidChallengeDevice,
    #[error("semantic-reservation challenge validity window is invalid")]
    InvalidChallengeWindow,
    #[error("unsupported post-reservation interlock statement schema")]
    UnsupportedInterlockStatementSchema,
    #[error("post-reservation interlock statement contains a zero security commitment")]
    ZeroInterlockSecurityCommitment,
    #[error("post-reservation interlock statement identity is invalid")]
    InvalidInterlockIdentity,
    #[error("post-reservation interlock statement surface is invalid")]
    InvalidInterlockSurface,
    #[error("post-reservation interlock statement validity window is invalid")]
    InvalidInterlockWindow,
    #[error("post-reservation interlock report has a zero evidence commitment")]
    ZeroInterlockEvidenceCommitment,
    #[error("unsupported post-reservation response schema")]
    UnsupportedPostReservationSchema,
    #[error("device attestation result size is outside accepted bounds")]
    DeviceAttestationSizeOutOfBounds,
    #[error("device attestation signature size is outside accepted bounds")]
    DeviceAttestationSignatureSizeOutOfBounds,
    #[error("post-reservation interlock report size is outside accepted bounds")]
    PostReservationReportSizeOutOfBounds,
    #[error("post-reservation interlock evidence size is outside accepted bounds")]
    InterlockEvidenceSizeOutOfBounds,
    #[error("post-reservation frame size is outside accepted bounds")]
    PostReservationFrameSizeOutOfBounds,
    #[error("post-reservation response is not canonically encoded")]
    NonCanonicalPostReservationEncoding,
    #[error("device attestation result encoding is invalid")]
    InvalidDeviceAttestationEncoding,
    #[error("device attestation result structure is invalid")]
    InvalidDeviceAttestationStructure,
    #[error("device attestation result is not canonically encoded")]
    NonCanonicalDeviceAttestationEncoding,
    #[error("post-reservation interlock report encoding is invalid")]
    InvalidPostReservationReportEncoding,
    #[error("post-reservation interlock report is not canonically encoded")]
    NonCanonicalPostReservationReportEncoding,
    #[error("raw controller evidence does not match the interlock report")]
    InterlockEvidenceDigestMismatch,
    #[error("phase-2 evidence does not bind the exact semantic-reservation challenge")]
    ChallengeBindingMismatch,
    #[error("controller statement does not bind the exact device-attestation result")]
    InterlockDeviceAttestationDigestMismatch,
    #[error("phase-2 evidence targets another challenged device")]
    ChallengeDeviceMismatch,
    #[error("controller statement binds another physical envelope")]
    ChallengeEnvelopeMismatch,
    #[error("controller statement binds another semantic reservation head")]
    ChallengeSemanticHeadMismatch,
    #[error("controller statement binds another transport-trust generation")]
    ChallengeTransportHeadMismatch,
    #[error("controller observation predates durable semantic persistence")]
    InterlockPredatesSemanticPersistence,
    #[error("controller evidence outlives the privileged challenge")]
    InterlockOutlivesChallenge,
    #[error("two-phase protocol decode failed: {0}")]
    Decoding(#[source] bincode::Error),
    #[error("two-phase protocol encode failed: {0}")]
    Encoding(#[source] bincode::Error),
}
