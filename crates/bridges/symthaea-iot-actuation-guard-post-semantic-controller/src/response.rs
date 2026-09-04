// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};
use symthaea_authority::Digest32;
use symthaea_iot_actuation_guard_two_phase_protocol::{
    MAX_POST_RESERVATION_INTERLOCK_REPORT_BYTES, PostReservationInterlockReportV1,
};

use crate::{
    MAX_POST_SEMANTIC_CONTROLLER_EVIDENCE_BYTES, MAX_POST_SEMANTIC_CONTROLLER_RESPONSE_BYTES,
    POST_SEMANTIC_CONTROLLER_RESPONSE_DOMAIN, POST_SEMANTIC_CONTROLLER_RESPONSE_SCHEMA_VERSION,
    PostSemanticControllerChallengeV1, PostSemanticControllerError, digest_frame,
};

/// Portable controller response produced only after the post-semantic challenge exists.
///
/// The previously authenticated device appraisal is not retransmitted. The controller
/// statement commits its exact object digest through the challenge/report binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostSemanticControllerResponseV1 {
    pub schema_version: u16,
    pub raw_interlock_report: Vec<u8>,
    pub raw_interlock_evidence: Vec<u8>,
}

impl PostSemanticControllerResponseV1 {
    pub fn validate_structure(&self) -> Result<(), PostSemanticControllerError> {
        if self.schema_version != POST_SEMANTIC_CONTROLLER_RESPONSE_SCHEMA_VERSION {
            return Err(PostSemanticControllerError::UnsupportedResponseSchema);
        }
        if self.raw_interlock_report.is_empty()
            || self.raw_interlock_report.len() > MAX_POST_RESERVATION_INTERLOCK_REPORT_BYTES
        {
            return Err(PostSemanticControllerError::ReportSizeOutOfBounds);
        }
        if self.raw_interlock_evidence.is_empty()
            || self.raw_interlock_evidence.len() > MAX_POST_SEMANTIC_CONTROLLER_EVIDENCE_BYTES
        {
            return Err(PostSemanticControllerError::EvidenceSizeOutOfBounds);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, PostSemanticControllerError> {
        self.validate_structure()?;
        let bytes = bincode::serialize(self).map_err(PostSemanticControllerError::Encoding)?;
        if bytes.len() > MAX_POST_SEMANTIC_CONTROLLER_RESPONSE_BYTES {
            return Err(PostSemanticControllerError::ResponseSizeOutOfBounds);
        }
        Ok(bytes)
    }

    pub fn digest(&self) -> Result<Digest32, PostSemanticControllerError> {
        Ok(digest_frame(
            POST_SEMANTIC_CONTROLLER_RESPONSE_DOMAIN,
            &self.canonical_bytes()?,
        ))
    }
}

/// Canonical, challenge-correlated controller evidence. This is still not actuator authority;
/// current controller-key trust and interlock policy remain a later fixed guard stage.
#[derive(Debug)]
pub struct DecodedPostSemanticControllerEvidence {
    report: PostReservationInterlockReportV1,
    raw_interlock_evidence: Vec<u8>,
    response_digest: Digest32,
}

impl DecodedPostSemanticControllerEvidence {
    pub fn report(&self) -> &PostReservationInterlockReportV1 {
        &self.report
    }

    pub fn raw_interlock_evidence(&self) -> &[u8] {
        &self.raw_interlock_evidence
    }

    pub const fn response_digest(&self) -> Digest32 {
        self.response_digest
    }

    pub fn into_parts(self) -> (PostReservationInterlockReportV1, Vec<u8>, Digest32) {
        (
            self.report,
            self.raw_interlock_evidence,
            self.response_digest,
        )
    }
}

pub fn decode_post_semantic_controller_response(
    frame: &[u8],
    challenge: &PostSemanticControllerChallengeV1,
) -> Result<DecodedPostSemanticControllerEvidence, PostSemanticControllerError> {
    challenge.validate()?;
    if frame.is_empty() || frame.len() > MAX_POST_SEMANTIC_CONTROLLER_RESPONSE_BYTES {
        return Err(PostSemanticControllerError::ResponseSizeOutOfBounds);
    }
    let response: PostSemanticControllerResponseV1 =
        bincode::deserialize(frame).map_err(PostSemanticControllerError::Decoding)?;
    response.validate_structure()?;
    if response.canonical_bytes()? != frame {
        return Err(PostSemanticControllerError::NonCanonicalResponseEncoding);
    }

    let report: PostReservationInterlockReportV1 =
        bincode::deserialize(&response.raw_interlock_report)
            .map_err(|_| PostSemanticControllerError::InvalidReportEncoding)?;
    report
        .validate_structure()
        .map_err(|_| PostSemanticControllerError::InvalidReportEncoding)?;
    let canonical_report =
        bincode::serialize(&report).map_err(|_| PostSemanticControllerError::InvalidReportEncoding)?;
    if canonical_report != response.raw_interlock_report {
        return Err(PostSemanticControllerError::NonCanonicalReportEncoding);
    }

    let evidence_digest = Digest32(*blake3::hash(&response.raw_interlock_evidence).as_bytes());
    if evidence_digest != report.evidence_digest {
        return Err(PostSemanticControllerError::EvidenceDigestMismatch);
    }

    let challenge_digest = challenge.digest()?;
    let statement = &report.statement;
    if statement.challenge_digest != challenge_digest {
        return Err(PostSemanticControllerError::ChallengeBindingMismatch);
    }
    if statement.device_attestation_result_digest != challenge.device_attestation_object_digest() {
        return Err(PostSemanticControllerError::DeviceRealityBindingMismatch);
    }
    if statement.device != *challenge.device() {
        return Err(PostSemanticControllerError::DeviceMismatch);
    }
    if statement.envelope_digest != challenge.envelope_digest() {
        return Err(PostSemanticControllerError::EnvelopeMismatch);
    }
    if statement.semantic_head != challenge.semantic_head() {
        return Err(PostSemanticControllerError::SemanticHeadMismatch);
    }
    if statement.transport_trust_head != challenge.transport_trust_head() {
        return Err(PostSemanticControllerError::TransportTrustMismatch);
    }
    if statement.checked_at_unix_ms < challenge.semantic_persisted_at_unix_ms()
        || statement.checked_at_unix_ms < challenge.issued_at_unix_ms()
    {
        return Err(PostSemanticControllerError::ControllerObservationPredatesChallenge);
    }
    if statement.expires_at_unix_ms > challenge.expires_at_unix_ms() {
        return Err(PostSemanticControllerError::ControllerReportOutlivesChallenge);
    }

    Ok(DecodedPostSemanticControllerEvidence {
        report,
        raw_interlock_evidence: response.raw_interlock_evidence,
        response_digest: digest_frame(POST_SEMANTIC_CONTROLLER_RESPONSE_DOMAIN, frame),
    })
}
