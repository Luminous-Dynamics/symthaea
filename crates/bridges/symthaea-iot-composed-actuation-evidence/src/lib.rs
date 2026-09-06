// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Non-authorizing composition of the exact evidence required before a live actuation fence.
//!
//! This crate consumes three independently produced affine proof objects:
//!
//! - fixed-current Xenia transport evidence;
//! - crash-durable admission/device-reality/semantic acceptance; and
//! - the post-semantic controller/interlock proof.
//!
//! Composition re-establishes their shared commitments and retains the exact proofs together, but
//! deliberately owns no current registry, current fence, relying-party clock, JIT lease, HAL
//! capability or actuator I/O. Evidence composition does not create power.

#![deny(unsafe_code)]

use symthaea_authority::Digest32;
use symthaea_iot_actuation_guard_interlock::VerifiedPostSemanticPhysicalInterlock;
use symthaea_iot_actuation_guard_semantic_persistence::PersistedSemanticAcceptance;
use symthaea_iot_transport_current_revalidation::RevalidatedXeniaTransport;
use thiserror::Error;

const COMPOSED_ACTUATION_EVIDENCE_DOMAIN: &[u8] = b"symthaea-iot-composed-actuation-evidence-v1\0";

/// Consume independently produced exact/durable proofs and prove that they describe one physical
/// actuation lineage. Success remains evidence only.
pub fn compose_actuation_evidence(
    transport: RevalidatedXeniaTransport,
    semantic: PersistedSemanticAcceptance,
    interlock: VerifiedPostSemanticPhysicalInterlock,
) -> Result<ComposedActuationEvidence, ComposedActuationEvidenceError> {
    let admission = semantic.admission_reservation();
    let reality = semantic.device_reality();
    let challenge = interlock.challenge();
    let controller_evidence = interlock.evidence();
    let report = controller_evidence.report();
    let statement = &report.statement;

    // Reconstruct both durable heads from the exact retained checkpoint objects. Composition must
    // not trust a cached head field when it can reproduce the durable commitment locally.
    let reconstructed_admission_head = admission
        .checkpoint()
        .head()
        .map_err(|_| ComposedActuationEvidenceError::AdmissionHeadReconstructionFailed)?;
    if reconstructed_admission_head != admission.head() {
        return Err(ComposedActuationEvidenceError::AdmissionHeadMismatch);
    }
    let reconstructed_semantic_head = semantic
        .checkpoint()
        .head()
        .map_err(|_| ComposedActuationEvidenceError::SemanticHeadReconstructionFailed)?;
    if reconstructed_semantic_head != semantic.device_head() {
        return Err(ComposedActuationEvidenceError::SemanticHeadMismatch);
    }

    transport
        .envelope()
        .validate_structure()
        .map_err(|_| ComposedActuationEvidenceError::TransportEnvelopeInvalid)?;
    let reconstructed_envelope_digest = transport
        .envelope()
        .digest()
        .map_err(|_| ComposedActuationEvidenceError::TransportEnvelopeInvalid)?;
    if reconstructed_envelope_digest != transport.envelope_digest() {
        return Err(ComposedActuationEvidenceError::TransportEnvelopeCommitmentMismatch);
    }

    // The transport and durable branches may come from separate affine verification objects, but
    // they must commit to the exact same authenticated physical-effect object and receipt lineage.
    if transport.envelope() != admission.envelope()
        || transport.envelope_digest() != admission.envelope_digest()
        || transport.envelope_digest() != semantic.envelope_digest()
        || transport.envelope_digest() != reality.envelope_digest()
    {
        return Err(ComposedActuationEvidenceError::PhysicalEnvelopeMismatch);
    }
    if transport.receipt_digest() != admission.transport_receipt_digest()
        || transport.receipt_digest() != reality.transport_receipt_digest()
    {
        return Err(ComposedActuationEvidenceError::TransportReceiptMismatch);
    }
    if transport.transport_trust_head() != admission.transport_trust_head()
        || transport.transport_trust_head() != reality.transport_trust_head()
    {
        return Err(ComposedActuationEvidenceError::TransportTrustHeadMismatch);
    }

    if reality.reservation_head() != admission.head() {
        return Err(ComposedActuationEvidenceError::DeviceRealityReservationMismatch);
    }
    if reality.config_digest() != admission.checkpoint().config_digest {
        return Err(ComposedActuationEvidenceError::DeviceRealityConfigMismatch);
    }
    if reality.attestation_object_digest() != semantic.device_attestation_object_digest() {
        return Err(ComposedActuationEvidenceError::DeviceRealityObjectMismatch);
    }

    challenge
        .validate()
        .map_err(|_| ComposedActuationEvidenceError::PostSemanticChallengeInvalid)?;
    if challenge.admission_reservation_head() != admission.head() {
        return Err(ComposedActuationEvidenceError::ChallengeAdmissionHeadMismatch);
    }
    if challenge.envelope_digest() != transport.envelope_digest() {
        return Err(ComposedActuationEvidenceError::ChallengeEnvelopeMismatch);
    }
    if challenge.transport_trust_head() != transport.transport_trust_head() {
        return Err(ComposedActuationEvidenceError::ChallengeTransportTrustMismatch);
    }
    if challenge.semantic_head() != semantic.device_head() {
        return Err(ComposedActuationEvidenceError::ChallengeSemanticHeadMismatch);
    }
    if challenge.device() != &admission.envelope().command.device {
        return Err(ComposedActuationEvidenceError::ChallengeDeviceMismatch);
    }
    if challenge.device_attestation_object_digest() != reality.attestation_object_digest() {
        return Err(ComposedActuationEvidenceError::ChallengeDeviceRealityMismatch);
    }
    if challenge.semantic_persisted_at_unix_ms() != semantic.semantic_persisted_at_unix_ms() {
        return Err(ComposedActuationEvidenceError::ChallengeSemanticPersistenceMismatch);
    }

    let device_reality_expires_at_unix_ms = reality
        .attestation_result()
        .body
        .expires_at_unix_s
        .checked_mul(1_000)
        .ok_or(ComposedActuationEvidenceError::TimeOverflow)?;
    if challenge.device_reality_expires_at_unix_ms() != device_reality_expires_at_unix_ms {
        return Err(ComposedActuationEvidenceError::ChallengeDeviceRealityExpiryMismatch);
    }
    let effect_deadline_unix_ms = admission
        .envelope()
        .send_not_after_unix_s
        .checked_mul(1_000)
        .ok_or(ComposedActuationEvidenceError::TimeOverflow)?;
    if challenge.effect_deadline_unix_ms() != effect_deadline_unix_ms {
        return Err(ComposedActuationEvidenceError::ChallengeEffectDeadlineMismatch);
    }

    report
        .validate_structure()
        .map_err(|_| ComposedActuationEvidenceError::ControllerStatementInvalid)?;
    let challenge_digest = challenge
        .digest()
        .map_err(|_| ComposedActuationEvidenceError::PostSemanticChallengeInvalid)?;
    if statement.challenge_digest != challenge_digest {
        return Err(ComposedActuationEvidenceError::ControllerChallengeMismatch);
    }
    if statement.device_attestation_result_digest != reality.attestation_object_digest() {
        return Err(ComposedActuationEvidenceError::ControllerDeviceRealityMismatch);
    }
    if statement.device != admission.envelope().command.device {
        return Err(ComposedActuationEvidenceError::ControllerDeviceMismatch);
    }
    if statement.envelope_digest != transport.envelope_digest() {
        return Err(ComposedActuationEvidenceError::ControllerEnvelopeMismatch);
    }
    if statement.semantic_head != semantic.device_head() {
        return Err(ComposedActuationEvidenceError::ControllerSemanticHeadMismatch);
    }
    if statement.transport_trust_head != transport.transport_trust_head() {
        return Err(ComposedActuationEvidenceError::ControllerTransportTrustMismatch);
    }

    let reconstructed_statement_digest = statement
        .digest()
        .map_err(|_| ComposedActuationEvidenceError::ControllerStatementInvalid)?;
    if reconstructed_statement_digest != interlock.statement_digest()
        || report.evidence_digest != interlock.evidence_digest()
        || controller_evidence.response_digest() != interlock.response_digest()
    {
        return Err(ComposedActuationEvidenceError::ControllerCommitmentMismatch);
    }

    let composition_digest = composition_digest(
        &transport,
        &semantic,
        &interlock,
        challenge_digest,
        effect_deadline_unix_ms,
        device_reality_expires_at_unix_ms,
    );

    Ok(ComposedActuationEvidence {
        transport,
        semantic,
        interlock,
        composition_digest,
    })
}

/// Affine correlation object retaining the exact three proof families for a later live fence.
///
/// It intentionally implements neither `Clone` nor serialization and confers no actuator/HAL
/// authority. A later privileged boundary must still obtain current state for every revocable root.
#[derive(Debug)]
pub struct ComposedActuationEvidence {
    transport: RevalidatedXeniaTransport,
    semantic: PersistedSemanticAcceptance,
    interlock: VerifiedPostSemanticPhysicalInterlock,
    composition_digest: Digest32,
}

impl ComposedActuationEvidence {
    pub fn transport(&self) -> &RevalidatedXeniaTransport {
        &self.transport
    }

    pub fn semantic_acceptance(&self) -> &PersistedSemanticAcceptance {
        &self.semantic
    }

    pub fn post_semantic_interlock(&self) -> &VerifiedPostSemanticPhysicalInterlock {
        &self.interlock
    }

    pub const fn composition_digest(&self) -> Digest32 {
        self.composition_digest
    }

    pub fn into_parts(
        self,
    ) -> (
        RevalidatedXeniaTransport,
        PersistedSemanticAcceptance,
        VerifiedPostSemanticPhysicalInterlock,
    ) {
        (self.transport, self.semantic, self.interlock)
    }
}

fn composition_digest(
    transport: &RevalidatedXeniaTransport,
    semantic: &PersistedSemanticAcceptance,
    interlock: &VerifiedPostSemanticPhysicalInterlock,
    challenge_digest: Digest32,
    effect_deadline_unix_ms: u64,
    device_reality_expires_at_unix_ms: u64,
) -> Digest32 {
    let admission = semantic.admission_reservation();
    let reality = semantic.device_reality();
    let admission_head = admission.head();
    let semantic_head = semantic.device_head();
    let device_reality_head = reality.trust_head();
    let interlock_head = interlock.interlock_trust_head();
    let transport_head = transport.transport_trust_head();

    let mut h = blake3::Hasher::new();
    h.update(COMPOSED_ACTUATION_EVIDENCE_DOMAIN);
    update_digest(&mut h, transport.exact_evidence_digest());
    update_digest(&mut h, transport.receipt_digest());
    update_digest(&mut h, transport.payload_digest());
    update_digest(&mut h, transport.envelope_digest());
    h.update(&transport_head.sequence.to_be_bytes());
    update_digest(&mut h, transport_head.digest);
    update_digest(&mut h, transport.transport_key_digest());
    h.update(&transport.valid_until_unix_ms().to_be_bytes());

    h.update(&admission_head.generation.to_be_bytes());
    update_digest(&mut h, admission_head.digest);
    h.update(&semantic_head.generation.to_be_bytes());
    update_digest(&mut h, semantic_head.digest);

    update_digest(&mut h, reality.attestation_object_digest());
    update_digest(&mut h, reality.result_digest());
    h.update(&device_reality_head.sequence.to_be_bytes());
    update_digest(&mut h, device_reality_head.digest);
    update_digest(&mut h, reality.policy_digest());
    update_digest(&mut h, reality.key_digest());

    update_digest(&mut h, challenge_digest);
    update_digest(&mut h, interlock.statement_digest());
    update_digest(&mut h, interlock.evidence_digest());
    update_digest(&mut h, interlock.response_digest());
    h.update(&interlock_head.sequence.to_be_bytes());
    update_digest(&mut h, interlock_head.digest);
    update_digest(&mut h, interlock.policy_digest());
    update_digest(&mut h, interlock.controller_key_digest());

    h.update(&semantic.semantic_persisted_at_unix_ms().to_be_bytes());
    h.update(&device_reality_expires_at_unix_ms.to_be_bytes());
    h.update(&effect_deadline_unix_ms.to_be_bytes());

    Digest32(*h.finalize().as_bytes())
}

fn update_digest(h: &mut blake3::Hasher, Digest32(bytes): Digest32) {
    h.update(&bytes);
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ComposedActuationEvidenceError {
    #[error("durable admission head could not be reconstructed")]
    AdmissionHeadReconstructionFailed,
    #[error("durable admission head differs from its retained proof")]
    AdmissionHeadMismatch,
    #[error("durable semantic head could not be reconstructed")]
    SemanticHeadReconstructionFailed,
    #[error("durable semantic head differs from its retained proof")]
    SemanticHeadMismatch,
    #[error("fixed-current transport envelope is structurally invalid")]
    TransportEnvelopeInvalid,
    #[error("fixed-current transport envelope digest does not reproduce its exact object")]
    TransportEnvelopeCommitmentMismatch,
    #[error("transport and durable semantic branches describe different physical envelopes")]
    PhysicalEnvelopeMismatch,
    #[error("transport and durable semantic branches bind different Xenia receipts")]
    TransportReceiptMismatch,
    #[error("transport and durable semantic branches bind different Xenia trust generations")]
    TransportTrustHeadMismatch,
    #[error("device-reality proof binds a different durable admission reservation")]
    DeviceRealityReservationMismatch,
    #[error("device-reality proof binds a different device enforcement configuration")]
    DeviceRealityConfigMismatch,
    #[error("semantic acceptance retained a different signed device appraisal object")]
    DeviceRealityObjectMismatch,
    #[error("post-semantic controller challenge is invalid")]
    PostSemanticChallengeInvalid,
    #[error("post-semantic challenge binds a different admission head")]
    ChallengeAdmissionHeadMismatch,
    #[error("post-semantic challenge binds a different physical envelope")]
    ChallengeEnvelopeMismatch,
    #[error("post-semantic challenge binds a different Xenia trust generation")]
    ChallengeTransportTrustMismatch,
    #[error("post-semantic challenge binds a different durable semantic head")]
    ChallengeSemanticHeadMismatch,
    #[error("post-semantic challenge targets a different device")]
    ChallengeDeviceMismatch,
    #[error("post-semantic challenge binds a different signed device appraisal")]
    ChallengeDeviceRealityMismatch,
    #[error("post-semantic challenge records a different semantic persistence event")]
    ChallengeSemanticPersistenceMismatch,
    #[error("post-semantic challenge records a different device-reality expiry")]
    ChallengeDeviceRealityExpiryMismatch,
    #[error("post-semantic challenge records a different physical-effect deadline")]
    ChallengeEffectDeadlineMismatch,
    #[error("controller statement/report is invalid")]
    ControllerStatementInvalid,
    #[error("controller statement does not bind the exact post-semantic challenge")]
    ControllerChallengeMismatch,
    #[error("controller statement does not bind the exact signed device appraisal")]
    ControllerDeviceRealityMismatch,
    #[error("controller statement targets a different device")]
    ControllerDeviceMismatch,
    #[error("controller statement binds a different physical envelope")]
    ControllerEnvelopeMismatch,
    #[error("controller statement binds a different durable semantic head")]
    ControllerSemanticHeadMismatch,
    #[error("controller statement binds a different Xenia transport-trust generation")]
    ControllerTransportTrustMismatch,
    #[error("controller statement/evidence/response commitments do not reproduce the proof")]
    ControllerCommitmentMismatch,
    #[error("protocol time conversion overflow")]
    TimeOverflow,
}
