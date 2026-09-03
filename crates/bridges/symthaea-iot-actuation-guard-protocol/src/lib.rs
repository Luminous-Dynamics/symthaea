// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-only IPC boundary for a privileged cyber-physical actuation guard.
//!
//! The process boundary deliberately transports *portable evidence*, never an opaque
//! in-process authority token. The unprivileged caller may submit only:
//!
//! - the exact raw Xenia authenticated-payload receipt;
//! - the exact raw physical-effect envelope bytes authenticated by that receipt;
//! - the exact raw physical-interlock report bytes; and
//! - the exact raw hardware evidence committed by that report.
//!
//! Trust registries/heads, device enforcement policy, durable semantic checkpoints,
//! local runtime observations, trusted time, verifier selection, and HAL/device handles
//! are intentionally absent from the wire type. Those must be owned by the privileged
//! guard process and combined with these bytes locally.
//!
//! Successfully decoding [`DecodedGuardEvidence`] proves only that the IPC frame is
//! bounded/canonical and that the raw hardware-evidence digest matches the canonical
//! interlock report. It grants **no actuator authority**.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_authority::Digest32;
use symthaea_iot_final_gate::{
    MAX_PHYSICAL_INTERLOCK_EVIDENCE_BYTES, PhysicalInterlockReportV1,
};
use symthaea_iot_transport_receipt::{
    MAX_XENIA_PHYSICAL_EFFECT_PAYLOAD_BYTES, MAX_XENIA_RECEIPT_BYTES,
};
use thiserror::Error;

/// Current guard request wire schema.
pub const ACTUATION_GUARD_REQUEST_SCHEMA_VERSION: u16 = 1;
/// Maximum canonical serialized physical-interlock report accepted over IPC.
pub const MAX_GUARD_INTERLOCK_REPORT_BYTES: usize = 16 * 1024;
/// Hard outer IPC-frame ceiling, checked before deserialization.
///
/// The four individual payload ceilings sum to less than this value, leaving bounded
/// room for bincode vector lengths and the schema field.
pub const MAX_GUARD_REQUEST_FRAME_BYTES: usize = 128 * 1024;

const ACTUATION_GUARD_REQUEST_DOMAIN: &[u8] = b"symthaea-iot-actuation-guard-request-v1\0";

/// Portable request accepted from an unprivileged caller.
///
/// Every field is raw evidence. There is intentionally no caller-selectable policy,
/// trust anchor, current time, local observation, verifier implementation, or HAL
/// configuration in this schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationGuardRequestV1 {
    /// Fail-closed schema version.
    pub schema_version: u16,
    /// Exact portable Xenia authenticated-payload receipt bytes.
    pub raw_transport_receipt: Vec<u8>,
    /// Exact physical-effect envelope bytes named by the Xenia receipt.
    pub raw_physical_effect_payload: Vec<u8>,
    /// Exact canonical serialized `PhysicalInterlockReportV1` bytes.
    pub raw_interlock_report: Vec<u8>,
    /// Exact hardware evidence bytes committed by the interlock report.
    pub raw_interlock_evidence: Vec<u8>,
}

impl ActuationGuardRequestV1 {
    /// Validate only bounded wire structure. No trust or authority decision occurs here.
    pub fn validate_structure(&self) -> Result<(), GuardProtocolError> {
        if self.schema_version != ACTUATION_GUARD_REQUEST_SCHEMA_VERSION {
            return Err(GuardProtocolError::UnsupportedSchema);
        }
        if self.raw_transport_receipt.is_empty()
            || self.raw_transport_receipt.len() > MAX_XENIA_RECEIPT_BYTES
        {
            return Err(GuardProtocolError::TransportReceiptSizeOutOfBounds);
        }
        if self.raw_physical_effect_payload.is_empty()
            || self.raw_physical_effect_payload.len() > MAX_XENIA_PHYSICAL_EFFECT_PAYLOAD_BYTES
        {
            return Err(GuardProtocolError::PhysicalPayloadSizeOutOfBounds);
        }
        if self.raw_interlock_report.is_empty()
            || self.raw_interlock_report.len() > MAX_GUARD_INTERLOCK_REPORT_BYTES
        {
            return Err(GuardProtocolError::InterlockReportSizeOutOfBounds);
        }
        if self.raw_interlock_evidence.is_empty()
            || self.raw_interlock_evidence.len() > MAX_PHYSICAL_INTERLOCK_EVIDENCE_BYTES
        {
            return Err(GuardProtocolError::InterlockEvidenceSizeOutOfBounds);
        }
        Ok(())
    }

    /// Canonical bincode-v1 frame bytes.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, GuardProtocolError> {
        self.validate_structure()?;
        let bytes = bincode::serialize(self).map_err(GuardProtocolError::Encoding)?;
        if bytes.len() > MAX_GUARD_REQUEST_FRAME_BYTES {
            return Err(GuardProtocolError::FrameSizeOutOfBounds);
        }
        Ok(bytes)
    }

    /// Audit-only commitment to the exact canonical IPC request bytes.
    ///
    /// This digest is correlation evidence, not authority.
    pub fn digest(&self) -> Result<Digest32, GuardProtocolError> {
        let bytes = self.canonical_bytes()?;
        Ok(digest_frame(&bytes))
    }
}

/// Bounded/canonical evidence decoded from one guard request.
///
/// This type is intentionally non-serializable and does not represent authorization.
/// The privileged guard must still verify transport trust/signatures, device semantics,
/// physical-interlock policy, controller-key trust, freshness, and the final JIT fence.
#[derive(Debug)]
pub struct DecodedGuardEvidence {
    request_digest: Digest32,
    raw_transport_receipt: Vec<u8>,
    raw_physical_effect_payload: Vec<u8>,
    interlock_report: PhysicalInterlockReportV1,
    raw_interlock_evidence: Vec<u8>,
}

impl DecodedGuardEvidence {
    /// Audit-only digest of the canonical outer request frame.
    pub const fn request_digest(&self) -> Digest32 {
        self.request_digest
    }

    /// Exact Xenia portable receipt bytes supplied across IPC.
    pub fn raw_transport_receipt(&self) -> &[u8] {
        &self.raw_transport_receipt
    }

    /// Exact authenticated physical-effect bytes supplied across IPC.
    pub fn raw_physical_effect_payload(&self) -> &[u8] {
        &self.raw_physical_effect_payload
    }

    /// Canonically decoded interlock report. The report is still evidence, not trust.
    pub fn interlock_report(&self) -> &PhysicalInterlockReportV1 {
        &self.interlock_report
    }

    /// Exact raw hardware evidence whose BLAKE3 digest matches the report commitment.
    pub fn raw_interlock_evidence(&self) -> &[u8] {
        &self.raw_interlock_evidence
    }

    /// After the guard has borrowed the transport receipt/payload for its local
    /// transport-verification stage, consume the remaining interlock evidence without
    /// cloning it. The returned request digest remains audit-only.
    pub fn into_interlock_parts(self) -> (PhysicalInterlockReportV1, Vec<u8>, Digest32) {
        (
            self.interlock_report,
            self.raw_interlock_evidence,
            self.request_digest,
        )
    }
}

/// Decode one complete IPC frame under strict pre-deserialization size limits.
///
/// The function requires canonical bincode encoding for both the outer request and the
/// embedded physical-interlock report. It also verifies the raw hardware-evidence
/// digest before returning any parsed evidence to the privileged guard core.
pub fn decode_canonical_guard_request(
    frame: &[u8],
) -> Result<DecodedGuardEvidence, GuardProtocolError> {
    if frame.is_empty() || frame.len() > MAX_GUARD_REQUEST_FRAME_BYTES {
        return Err(GuardProtocolError::FrameSizeOutOfBounds);
    }

    let request: ActuationGuardRequestV1 =
        bincode::deserialize(frame).map_err(GuardProtocolError::Decoding)?;
    request.validate_structure()?;

    let canonical_request = request.canonical_bytes()?;
    if canonical_request != frame {
        return Err(GuardProtocolError::NonCanonicalRequestEncoding);
    }

    let report: PhysicalInterlockReportV1 = bincode::deserialize(&request.raw_interlock_report)
        .map_err(|_| GuardProtocolError::InvalidInterlockReportEncoding)?;
    report
        .validate_structure()
        .map_err(|_| GuardProtocolError::InvalidInterlockReportStructure)?;
    let canonical_report = bincode::serialize(&report)
        .map_err(|_| GuardProtocolError::InvalidInterlockReportEncoding)?;
    if canonical_report != request.raw_interlock_report {
        return Err(GuardProtocolError::NonCanonicalInterlockReportEncoding);
    }

    let evidence_digest = Digest32(*blake3::hash(&request.raw_interlock_evidence).as_bytes());
    if evidence_digest != report.evidence_digest {
        return Err(GuardProtocolError::InterlockEvidenceDigestMismatch);
    }

    Ok(DecodedGuardEvidence {
        request_digest: digest_frame(frame),
        raw_transport_receipt: request.raw_transport_receipt,
        raw_physical_effect_payload: request.raw_physical_effect_payload,
        interlock_report: report,
        raw_interlock_evidence: request.raw_interlock_evidence,
    })
}

fn digest_frame(frame: &[u8]) -> Digest32 {
    let mut h = blake3::Hasher::new();
    h.update(ACTUATION_GUARD_REQUEST_DOMAIN);
    h.update(&(frame.len() as u64).to_be_bytes());
    h.update(frame);
    Digest32(*h.finalize().as_bytes())
}

/// Fail-closed parsing failures at the unprivileged-to-privileged IPC boundary.
#[derive(Debug, Error)]
pub enum GuardProtocolError {
    /// Unknown request schema.
    #[error("unsupported actuation guard request schema")]
    UnsupportedSchema,
    /// Whole IPC frame is empty or exceeds the pre-deserialization ceiling.
    #[error("actuation guard IPC frame size is outside accepted bounds")]
    FrameSizeOutOfBounds,
    /// Xenia receipt field is empty or oversized.
    #[error("guard transport receipt size is outside accepted bounds")]
    TransportReceiptSizeOutOfBounds,
    /// Physical-effect payload field is empty or oversized.
    #[error("guard physical payload size is outside accepted bounds")]
    PhysicalPayloadSizeOutOfBounds,
    /// Interlock report field is empty or oversized.
    #[error("guard interlock report size is outside accepted bounds")]
    InterlockReportSizeOutOfBounds,
    /// Raw hardware-evidence field is empty or oversized.
    #[error("guard interlock evidence size is outside accepted bounds")]
    InterlockEvidenceSizeOutOfBounds,
    /// Outer request could not be decoded.
    #[error("actuation guard request decode failed: {0}")]
    Decoding(#[source] bincode::Error),
    /// Outer request could not be encoded canonically.
    #[error("actuation guard request encode failed: {0}")]
    Encoding(#[source] bincode::Error),
    /// Outer request had an alternate/trailing encoding.
    #[error("actuation guard request is not canonically encoded")]
    NonCanonicalRequestEncoding,
    /// Embedded interlock report is not valid bincode-v1.
    #[error("physical interlock report encoding is invalid")]
    InvalidInterlockReportEncoding,
    /// Embedded interlock report failed its own bounded structural checks.
    #[error("physical interlock report structure is invalid")]
    InvalidInterlockReportStructure,
    /// Embedded report bytes were not the unique canonical encoding.
    #[error("physical interlock report is not canonically encoded")]
    NonCanonicalInterlockReportEncoding,
    /// Raw hardware evidence differs from the report's exact commitment.
    #[error("raw hardware evidence does not match the interlock report")]
    InterlockEvidenceDigestMismatch,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use symthaea_authority::ResourceRef;
    use symthaea_iot_device_protocol::DeviceSemanticHead;
    use symthaea_iot_final_gate::PHYSICAL_INTERLOCK_REPORT_SCHEMA_VERSION;
    use symthaea_iot_transport_receipt::TransportTrustHead;

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn report(evidence: &[u8]) -> PhysicalInterlockReportV1 {
        PhysicalInterlockReportV1 {
            schema_version: PHYSICAL_INTERLOCK_REPORT_SCHEMA_VERSION,
            controller_id: "safety-plc:field-a".into(),
            device: ResourceRef("iot:valve:72".into()),
            envelope_digest: d(7),
            device_head: DeviceSemanticHead {
                generation: 9,
                digest: d(8),
            },
            transport_trust_head: TransportTrustHead {
                sequence: 5,
                digest: d(9),
            },
            asserted_interlocks: BTreeSet::from([
                "pressure-safe".into(),
                "manual-stop-ready".into(),
            ]),
            checked_at_unix_ms: 10_000,
            expires_at_unix_ms: 11_000,
            evidence_digest: Digest32(*blake3::hash(evidence).as_bytes()),
        }
    }

    fn request(evidence: &[u8]) -> ActuationGuardRequestV1 {
        let report = report(evidence);
        ActuationGuardRequestV1 {
            schema_version: ACTUATION_GUARD_REQUEST_SCHEMA_VERSION,
            raw_transport_receipt: vec![0x11; 128],
            raw_physical_effect_payload: vec![0x22; 256],
            raw_interlock_report: bincode::serialize(&report).unwrap(),
            raw_interlock_evidence: evidence.to_vec(),
        }
    }

    #[test]
    fn exact_canonical_evidence_frame_decodes_without_minting_authority() {
        let request = request(b"controller-signature");
        let frame = request.canonical_bytes().unwrap();
        let decoded = decode_canonical_guard_request(&frame).unwrap();
        assert_eq!(decoded.request_digest(), request.digest().unwrap());
        assert_eq!(
            decoded.raw_transport_receipt(),
            request.raw_transport_receipt.as_slice()
        );
        assert_eq!(
            decoded.raw_physical_effect_payload(),
            request.raw_physical_effect_payload.as_slice()
        );
        assert_eq!(decoded.interlock_report().controller_id, "safety-plc:field-a");
    }

    #[test]
    fn hardware_evidence_substitution_is_rejected_at_ipc_boundary() {
        let mut request = request(b"controller-signature");
        request.raw_interlock_evidence = b"substituted-signature".to_vec();
        let frame = request.canonical_bytes().unwrap();
        assert!(matches!(
            decode_canonical_guard_request(&frame),
            Err(GuardProtocolError::InterlockEvidenceDigestMismatch)
        ));
    }

    #[test]
    fn unsupported_schema_fails_closed() {
        let mut request = request(b"controller-signature");
        request.schema_version = 2;
        let frame = bincode::serialize(&request).unwrap();
        assert!(matches!(
            decode_canonical_guard_request(&frame),
            Err(GuardProtocolError::UnsupportedSchema)
        ));
    }

    #[test]
    fn oversized_frame_is_rejected_before_deserialization() {
        let frame = vec![0u8; MAX_GUARD_REQUEST_FRAME_BYTES + 1];
        assert!(matches!(
            decode_canonical_guard_request(&frame),
            Err(GuardProtocolError::FrameSizeOutOfBounds)
        ));
    }

    #[test]
    fn oversized_individual_evidence_field_is_rejected() {
        let mut request = request(b"controller-signature");
        request.raw_interlock_evidence = vec![0x44; MAX_PHYSICAL_INTERLOCK_EVIDENCE_BYTES + 1];
        assert!(matches!(
            request.validate_structure(),
            Err(GuardProtocolError::InterlockEvidenceSizeOutOfBounds)
        ));
    }

    #[test]
    fn trailing_data_cannot_smuggle_caller_owned_policy_or_trust_state() {
        let request = request(b"controller-signature");
        let mut frame = request.canonical_bytes().unwrap();
        frame.extend_from_slice(b"untrusted-policy-or-head");
        assert!(decode_canonical_guard_request(&frame).is_err());
    }

    #[test]
    fn audit_digest_changes_when_any_portable_evidence_changes() {
        let a = request(b"controller-signature");
        let mut b = a.clone();
        b.raw_physical_effect_payload[0] ^= 1;
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }
}
