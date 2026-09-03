// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Final fail-closed software gate before a cyber-physical adapter may consider I/O.
//!
//! This crate performs no actuator I/O and does not claim that a physical effect
//! occurred. It joins three independently established, opaque facts:
//!
//! - [`VerifiedTransportEnvelope`]: Xenia authenticated the exact canonical
//!   physical-effect bytes under current transport trust;
//! - [`SemanticallyAcceptedEffect`]: device-local semantic/replay state accepted the
//!   same envelope and its successor checkpoint was confirmed persisted;
//! - [`VerifiedPhysicalInterlock`]: a separately authenticated hardware evidence
//!   boundary reported the required physical interlocks for that exact envelope,
//!   device semantic head, and transport-trust head.
//!
//! The final [`FinalActuatorPermit`] is intentionally non-serializable and non-clone.
//! It is a local one-use type-state token, not a portable authorization certificate.
//! A concrete hardware adapter should require this token by type and consume it when
//! attempting device I/O.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_authority::DeviceCommand;
use symthaea_iot_device_protocol::{DeviceSemanticHead, SemanticallyAcceptedEffect};
use symthaea_iot_transport_receipt::{TransportTrustHead, VerifiedTransportEnvelope};
use thiserror::Error;

/// Current physical-interlock policy schema.
pub const PHYSICAL_INTERLOCK_POLICY_SCHEMA_VERSION: u16 = 1;
/// Current physical-interlock report schema.
pub const PHYSICAL_INTERLOCK_REPORT_SCHEMA_VERSION: u16 = 1;
/// Maximum lifetime of one accepted physical-interlock report.
pub const MAX_PHYSICAL_INTERLOCK_REPORT_LIFETIME_MS: u64 = 2_000;
/// Maximum raw hardware-evidence payload accepted before provider verification.
pub const MAX_PHYSICAL_INTERLOCK_EVIDENCE_BYTES: usize = 16 * 1024;
/// Final gate refuses transport evidence older than this even if the outer command
/// send window is longer.
pub const MAX_FINAL_TRANSPORT_TO_ACTUATION_MS: u64 = 2_000;
/// Bound deployment/controller/interlock labels.
pub const MAX_INTERLOCK_LABEL_BYTES: usize = 128;
/// Bound the exact required/asserted interlock surface.
pub const MAX_INTERLOCK_COUNT: usize = 64;

const PHYSICAL_INTERLOCK_POLICY_DOMAIN: &[u8] = b"symthaea-iot-physical-interlock-policy-v1\0";
const PHYSICAL_INTERLOCK_REPORT_DOMAIN: &[u8] = b"symthaea-iot-physical-interlock-report-v1\0";

/// Device-local policy defining which hardware controller and physical interlocks
/// must corroborate a consequential command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalInterlockPolicyV1 {
    /// Fail-closed schema version.
    pub schema_version: u16,
    /// Exact physical device protected by this policy.
    pub device: ResourceRef,
    /// Hardware-controller identities permitted to produce interlock evidence.
    pub allowed_controllers: BTreeSet<String>,
    /// Exact required interlock names. The report must assert this exact set.
    pub required_interlocks: BTreeSet<String>,
    /// Deployment-specific lifetime ceiling no larger than the global maximum.
    pub max_report_lifetime_ms: u64,
}

impl PhysicalInterlockPolicyV1 {
    /// Validate policy structure independent of any specific report.
    pub fn validate(&self) -> Result<(), FinalActuatorGateError> {
        if self.schema_version != PHYSICAL_INTERLOCK_POLICY_SCHEMA_VERSION {
            return Err(FinalActuatorGateError::UnsupportedInterlockPolicySchema);
        }
        if self.device.0.trim() != self.device.0 || self.device.0.is_empty() {
            return Err(FinalActuatorGateError::InvalidInterlockDevice);
        }
        if self.allowed_controllers.is_empty()
            || self.allowed_controllers.len() > MAX_INTERLOCK_COUNT
            || self.required_interlocks.is_empty()
            || self.required_interlocks.len() > MAX_INTERLOCK_COUNT
        {
            return Err(FinalActuatorGateError::InvalidInterlockPolicySurface);
        }
        if self
            .allowed_controllers
            .iter()
            .chain(self.required_interlocks.iter())
            .any(|label| !valid_label(label))
        {
            return Err(FinalActuatorGateError::InvalidInterlockLabel);
        }
        if self.max_report_lifetime_ms == 0
            || self.max_report_lifetime_ms > MAX_PHYSICAL_INTERLOCK_REPORT_LIFETIME_MS
        {
            return Err(FinalActuatorGateError::InvalidInterlockLifetime);
        }
        Ok(())
    }

    /// Domain-separated commitment to the complete local interlock policy.
    pub fn digest(&self) -> Result<Digest32, FinalActuatorGateError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(PHYSICAL_INTERLOCK_POLICY_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        update_string(&mut h, &self.device.0);
        update_strings(&mut h, &self.allowed_controllers);
        update_strings(&mut h, &self.required_interlocks);
        h.update(&self.max_report_lifetime_ms.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }
}

/// Bounded statement emitted by a hardware evidence boundary.
///
/// The report itself is not trusted merely because it is deserializable. A caller must
/// pass it through [`verify_physical_interlock`] with raw evidence and a concrete
/// [`HardwareInterlockEvidenceVerifier`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalInterlockReportV1 {
    /// Fail-closed schema version.
    pub schema_version: u16,
    /// Hardware controller that produced the evidence.
    pub controller_id: String,
    /// Exact physical device being interlocked.
    pub device: ResourceRef,
    /// Semantic digest of the exact physical-effect envelope.
    pub envelope_digest: Digest32,
    /// Durable device semantic generation observed by the hardware boundary.
    pub device_head: DeviceSemanticHead,
    /// Transport-trust generation associated with the authenticated command.
    pub transport_trust_head: TransportTrustHead,
    /// Exact set of physical interlocks asserted safe/closed/ready.
    pub asserted_interlocks: BTreeSet<String>,
    /// Trusted controller observation time in Unix milliseconds.
    pub checked_at_unix_ms: u64,
    /// Exclusive report expiry in Unix milliseconds.
    pub expires_at_unix_ms: u64,
    /// BLAKE3-256 commitment to the exact raw hardware evidence bytes.
    pub evidence_digest: Digest32,
}

impl PhysicalInterlockReportV1 {
    /// Validate bounded report structure independent of policy and provider trust.
    pub fn validate_structure(&self) -> Result<(), FinalActuatorGateError> {
        if self.schema_version != PHYSICAL_INTERLOCK_REPORT_SCHEMA_VERSION {
            return Err(FinalActuatorGateError::UnsupportedInterlockReportSchema);
        }
        if !valid_label(&self.controller_id)
            || self.device.0.trim() != self.device.0
            || self.device.0.is_empty()
        {
            return Err(FinalActuatorGateError::InvalidInterlockReportIdentity);
        }
        if self.asserted_interlocks.is_empty()
            || self.asserted_interlocks.len() > MAX_INTERLOCK_COUNT
            || self
                .asserted_interlocks
                .iter()
                .any(|label| !valid_label(label))
        {
            return Err(FinalActuatorGateError::InvalidInterlockReportSurface);
        }
        if self.envelope_digest == Digest32([0; 32])
            || self.device_head.generation == 0
            || self.device_head.digest == Digest32([0; 32])
            || self.transport_trust_head.sequence == 0
            || self.transport_trust_head.digest == Digest32([0; 32])
            || self.evidence_digest == Digest32([0; 32])
        {
            return Err(FinalActuatorGateError::ZeroInterlockSecurityCommitment);
        }
        let lifetime = self
            .expires_at_unix_ms
            .checked_sub(self.checked_at_unix_ms)
            .ok_or(FinalActuatorGateError::InvalidInterlockLifetime)?;
        if lifetime == 0 || lifetime > MAX_PHYSICAL_INTERLOCK_REPORT_LIFETIME_MS {
            return Err(FinalActuatorGateError::InvalidInterlockLifetime);
        }
        Ok(())
    }

    /// Domain-separated digest that a hardware verifier must authenticate.
    pub fn digest(&self) -> Result<Digest32, FinalActuatorGateError> {
        self.validate_structure()?;
        let mut h = blake3::Hasher::new();
        h.update(PHYSICAL_INTERLOCK_REPORT_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        update_string(&mut h, &self.controller_id);
        update_string(&mut h, &self.device.0);
        update_digest(&mut h, self.envelope_digest);
        h.update(&self.device_head.generation.to_be_bytes());
        update_digest(&mut h, self.device_head.digest);
        h.update(&self.transport_trust_head.sequence.to_be_bytes());
        update_digest(&mut h, self.transport_trust_head.digest);
        update_strings(&mut h, &self.asserted_interlocks);
        h.update(&self.checked_at_unix_ms.to_be_bytes());
        h.update(&self.expires_at_unix_ms.to_be_bytes());
        update_digest(&mut h, self.evidence_digest);
        Ok(Digest32(*h.finalize().as_bytes()))
    }
}

/// Provider boundary for device-specific hardware interlock evidence.
///
/// A production provider may verify a TPM/TEE quote, safety-PLC signature, secure
/// element assertion, or another reviewed hardware evidence format. It must
/// authenticate the exact report digest and raw evidence supplied here. Returning
/// `true` is the provider's cryptographic/hardware claim; this crate independently
/// performs policy, freshness, binding, and composition checks around it.
pub trait HardwareInterlockEvidenceVerifier {
    /// Verify raw hardware evidence for the exact controller/report commitment.
    fn verify_interlock_evidence(
        &self,
        controller_id: &str,
        report_digest: Digest32,
        raw_evidence: &[u8],
    ) -> bool;
}

/// Opaque local proof that a fresh hardware interlock report passed policy and its
/// configured evidence verifier.
#[derive(Debug)]
pub struct VerifiedPhysicalInterlock {
    controller_id: String,
    device: ResourceRef,
    envelope_digest: Digest32,
    device_head: DeviceSemanticHead,
    transport_trust_head: TransportTrustHead,
    report_digest: Digest32,
    evidence_digest: Digest32,
    checked_at_unix_ms: u64,
    expires_at_unix_ms: u64,
}

impl VerifiedPhysicalInterlock {
    /// Controller identity that produced the accepted hardware evidence.
    pub fn controller_id(&self) -> &str {
        &self.controller_id
    }

    /// Exact physical device bound by the evidence.
    pub fn device(&self) -> &ResourceRef {
        &self.device
    }

    /// Exact physical-effect envelope commitment.
    pub const fn envelope_digest(&self) -> Digest32 {
        self.envelope_digest
    }

    /// Device semantic generation bound by the hardware evidence.
    pub const fn device_head(&self) -> DeviceSemanticHead {
        self.device_head
    }

    /// Transport-trust generation bound by the hardware evidence.
    pub const fn transport_trust_head(&self) -> TransportTrustHead {
        self.transport_trust_head
    }

    /// Commitment to the exact interlock report.
    pub const fn report_digest(&self) -> Digest32 {
        self.report_digest
    }

    /// Commitment to the exact raw hardware evidence.
    pub const fn evidence_digest(&self) -> Digest32 {
        self.evidence_digest
    }

    /// Trusted hardware observation time.
    pub const fn checked_at_unix_ms(&self) -> u64 {
        self.checked_at_unix_ms
    }

    /// Exclusive hardware-evidence expiry.
    pub const fn expires_at_unix_ms(&self) -> u64 {
        self.expires_at_unix_ms
    }
}

/// Verify one interlock report under exact device-local policy and hardware evidence.
pub fn verify_physical_interlock(
    policy: &PhysicalInterlockPolicyV1,
    report: PhysicalInterlockReportV1,
    raw_evidence: &[u8],
    now_unix_ms: u64,
    verifier: &impl HardwareInterlockEvidenceVerifier,
) -> Result<VerifiedPhysicalInterlock, FinalActuatorGateError> {
    policy.validate()?;
    report.validate_structure()?;
    if report.device != policy.device {
        return Err(FinalActuatorGateError::InterlockDeviceMismatch);
    }
    if !policy.allowed_controllers.contains(&report.controller_id) {
        return Err(FinalActuatorGateError::InterlockControllerDenied);
    }
    if report.asserted_interlocks != policy.required_interlocks {
        return Err(FinalActuatorGateError::InterlockSetMismatch);
    }
    let lifetime = report.expires_at_unix_ms - report.checked_at_unix_ms;
    if lifetime > policy.max_report_lifetime_ms {
        return Err(FinalActuatorGateError::InterlockLifetimeExceedsPolicy);
    }
    if raw_evidence.is_empty() || raw_evidence.len() > MAX_PHYSICAL_INTERLOCK_EVIDENCE_BYTES {
        return Err(FinalActuatorGateError::InterlockEvidenceSizeOutOfBounds);
    }
    let evidence_digest = Digest32(*blake3::hash(raw_evidence).as_bytes());
    if evidence_digest != report.evidence_digest {
        return Err(FinalActuatorGateError::InterlockEvidenceDigestMismatch);
    }
    if now_unix_ms < report.checked_at_unix_ms || now_unix_ms >= report.expires_at_unix_ms {
        return Err(FinalActuatorGateError::InterlockReportNotFresh);
    }
    let report_digest = report.digest()?;
    if !verifier.verify_interlock_evidence(&report.controller_id, report_digest, raw_evidence) {
        return Err(FinalActuatorGateError::InterlockEvidenceVerificationFailed);
    }

    Ok(VerifiedPhysicalInterlock {
        controller_id: report.controller_id,
        device: report.device,
        envelope_digest: report.envelope_digest,
        device_head: report.device_head,
        transport_trust_head: report.transport_trust_head,
        report_digest,
        evidence_digest,
        checked_at_unix_ms: report.checked_at_unix_ms,
        expires_at_unix_ms: report.expires_at_unix_ms,
    })
}

/// Final local one-use type-state token for an actuator adapter.
///
/// It is intentionally neither `Clone` nor `Serialize`. The adapter should accept it
/// by value and perform exactly one I/O attempt for [`Self::command`].
#[derive(Debug)]
pub struct FinalActuatorPermit {
    command: DeviceCommand,
    envelope_digest: Digest32,
    device_head: DeviceSemanticHead,
    transport_trust_head: TransportTrustHead,
    transport_receipt_digest: Digest32,
    transport_session_evidence_digest: [u8; 32],
    transport_peer_identity_fingerprint: [u8; 32],
    interlock_controller_id: String,
    interlock_report_digest: Digest32,
    interlock_evidence_digest: Digest32,
    joined_at_unix_ms: u64,
    must_dispatch_by_unix_ms: u64,
}

impl FinalActuatorPermit {
    /// Exact physical command approved by all three independent boundaries.
    pub fn command(&self) -> &DeviceCommand {
        &self.command
    }

    /// Exact common physical-effect envelope commitment.
    pub const fn envelope_digest(&self) -> Digest32 {
        self.envelope_digest
    }

    /// Durable device semantic generation consumed by this join.
    pub const fn device_head(&self) -> DeviceSemanticHead {
        self.device_head
    }

    /// Anti-rollback transport-trust generation used by this join.
    pub const fn transport_trust_head(&self) -> TransportTrustHead {
        self.transport_trust_head
    }

    /// Xenia receipt-body commitment for audit correlation.
    pub const fn transport_receipt_digest(&self) -> Digest32 {
        self.transport_receipt_digest
    }

    /// Xenia opaque authenticated-session evidence commitment.
    pub const fn transport_session_evidence_digest(&self) -> [u8; 32] {
        self.transport_session_evidence_digest
    }

    /// Xenia authenticated remote peer fingerprint.
    pub const fn transport_peer_identity_fingerprint(&self) -> [u8; 32] {
        self.transport_peer_identity_fingerprint
    }

    /// Hardware controller whose interlock evidence participated in the join.
    pub fn interlock_controller_id(&self) -> &str {
        &self.interlock_controller_id
    }

    /// Exact hardware interlock report commitment.
    pub const fn interlock_report_digest(&self) -> Digest32 {
        self.interlock_report_digest
    }

    /// Exact raw hardware evidence commitment.
    pub const fn interlock_evidence_digest(&self) -> Digest32 {
        self.interlock_evidence_digest
    }

    /// Relying-party time at which all independent proof streams were joined.
    pub const fn joined_at_unix_ms(&self) -> u64 {
        self.joined_at_unix_ms
    }

    /// Inclusive latest millisecond at which a consuming adapter may attempt I/O.
    pub const fn must_dispatch_by_unix_ms(&self) -> u64 {
        self.must_dispatch_by_unix_ms
    }
}

/// Consume and join authenticated transport, durable device semantics, and verified
/// physical interlock evidence for the exact same command lineage.
pub fn join_final_actuator_gate(
    transport: VerifiedTransportEnvelope,
    semantic: SemanticallyAcceptedEffect,
    interlock: VerifiedPhysicalInterlock,
    now_unix_ms: u64,
) -> Result<FinalActuatorPermit, FinalActuatorGateError> {
    if transport.envelope_digest() != semantic.envelope_digest()
        || transport.envelope_digest() != interlock.envelope_digest
    {
        return Err(FinalActuatorGateError::EnvelopeCommitmentMismatch);
    }
    if &transport.envelope().command != semantic.command() {
        return Err(FinalActuatorGateError::CommandMismatch);
    }
    if transport.envelope().command.device != interlock.device {
        return Err(FinalActuatorGateError::InterlockDeviceMismatch);
    }
    if semantic.device_head() != interlock.device_head {
        return Err(FinalActuatorGateError::DeviceSemanticHeadMismatch);
    }
    if transport.trust_head() != interlock.transport_trust_head {
        return Err(FinalActuatorGateError::TransportTrustHeadMismatch);
    }
    if interlock.checked_at_unix_ms < transport.opened_at_unix_ms() {
        return Err(FinalActuatorGateError::InterlockPredatesAuthenticatedTransport);
    }
    if now_unix_ms < transport.opened_at_unix_ms() {
        return Err(FinalActuatorGateError::FinalJoinPredatesAuthenticatedTransport);
    }
    let transport_age = now_unix_ms
        .checked_sub(transport.opened_at_unix_ms())
        .ok_or(FinalActuatorGateError::FinalJoinPredatesAuthenticatedTransport)?;
    if transport_age > MAX_FINAL_TRANSPORT_TO_ACTUATION_MS {
        return Err(FinalActuatorGateError::AuthenticatedTransportTooOldForActuation);
    }
    if now_unix_ms < interlock.checked_at_unix_ms || now_unix_ms >= interlock.expires_at_unix_ms {
        return Err(FinalActuatorGateError::InterlockNoLongerFresh);
    }

    let send_not_after_unix_ms = transport
        .envelope()
        .send_not_after_unix_s
        .checked_mul(1_000)
        .ok_or(FinalActuatorGateError::PhysicalEnvelopeTimeOverflow)?;
    if now_unix_ms > send_not_after_unix_ms {
        return Err(FinalActuatorGateError::PhysicalEnvelopeSendDeadlineElapsed);
    }
    let transport_deadline = transport
        .opened_at_unix_ms()
        .checked_add(MAX_FINAL_TRANSPORT_TO_ACTUATION_MS)
        .ok_or(FinalActuatorGateError::TransportAgeDeadlineOverflow)?;
    let interlock_last_valid = interlock
        .expires_at_unix_ms
        .checked_sub(1)
        .ok_or(FinalActuatorGateError::InvalidInterlockLifetime)?;
    let must_dispatch_by_unix_ms = send_not_after_unix_ms
        .min(transport_deadline)
        .min(interlock_last_valid);
    if now_unix_ms > must_dispatch_by_unix_ms {
        return Err(FinalActuatorGateError::FinalDispatchWindowElapsed);
    }

    Ok(FinalActuatorPermit {
        command: transport.envelope().command.clone(),
        envelope_digest: transport.envelope_digest(),
        device_head: semantic.device_head(),
        transport_trust_head: transport.trust_head(),
        transport_receipt_digest: transport.receipt_digest(),
        transport_session_evidence_digest: transport.session_evidence_digest(),
        transport_peer_identity_fingerprint: transport.peer_identity_fingerprint(),
        interlock_controller_id: interlock.controller_id,
        interlock_report_digest: interlock.report_digest,
        interlock_evidence_digest: interlock.evidence_digest,
        joined_at_unix_ms: now_unix_ms,
        must_dispatch_by_unix_ms,
    })
}

fn valid_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_INTERLOCK_LABEL_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn update_string(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u32).to_be_bytes());
    h.update(value.as_bytes());
}

fn update_strings(h: &mut blake3::Hasher, values: &BTreeSet<String>) {
    h.update(&(values.len() as u32).to_be_bytes());
    for value in values {
        update_string(h, value);
    }
}

fn update_digest(h: &mut blake3::Hasher, Digest32(value): Digest32) {
    h.update(&value);
}

/// Fail-closed final-gate error vocabulary.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum FinalActuatorGateError {
    /// Unknown physical-interlock policy schema.
    #[error("unsupported physical-interlock policy schema")]
    UnsupportedInterlockPolicySchema,
    /// Unknown physical-interlock report schema.
    #[error("unsupported physical-interlock report schema")]
    UnsupportedInterlockReportSchema,
    /// Local interlock policy contains an invalid device identity.
    #[error("physical-interlock policy has an invalid device identity")]
    InvalidInterlockDevice,
    /// Policy has an empty or excessively large controller/interlock surface.
    #[error("physical-interlock policy surface is invalid")]
    InvalidInterlockPolicySurface,
    /// Controller/interlock label failed canonical bounds.
    #[error("physical-interlock label is invalid")]
    InvalidInterlockLabel,
    /// Hardware report contains malformed controller/device identity.
    #[error("physical-interlock report identity is invalid")]
    InvalidInterlockReportIdentity,
    /// Hardware report contains malformed asserted-interlock surface.
    #[error("physical-interlock report surface is invalid")]
    InvalidInterlockReportSurface,
    /// Interlock/report validity interval is malformed or exceeds global bounds.
    #[error("physical-interlock report lifetime is invalid")]
    InvalidInterlockLifetime,
    /// A required security commitment in the hardware report is zero.
    #[error("physical-interlock report contains a zero security commitment")]
    ZeroInterlockSecurityCommitment,
    /// Hardware report targets a different physical device.
    #[error("physical-interlock device does not match the command")]
    InterlockDeviceMismatch,
    /// Hardware controller is not permitted by local policy.
    #[error("physical-interlock controller is denied by local policy")]
    InterlockControllerDenied,
    /// Reported interlock set is not the exact required local set.
    #[error("physical-interlock report does not assert the exact required set")]
    InterlockSetMismatch,
    /// Report lifetime exceeds the stricter local policy ceiling.
    #[error("physical-interlock report lifetime exceeds local policy")]
    InterlockLifetimeExceedsPolicy,
    /// Raw hardware evidence is empty or too large.
    #[error("physical-interlock evidence size is outside accepted bounds")]
    InterlockEvidenceSizeOutOfBounds,
    /// Raw hardware evidence does not match the committed digest.
    #[error("physical-interlock raw evidence digest mismatch")]
    InterlockEvidenceDigestMismatch,
    /// Hardware report is not fresh at verification time.
    #[error("physical-interlock report is not fresh")]
    InterlockReportNotFresh,
    /// Configured hardware provider rejected the report/evidence pair.
    #[error("physical-interlock evidence verification failed")]
    InterlockEvidenceVerificationFailed,
    /// Transport, semantic, and interlock proof streams do not bind one envelope.
    #[error("final gate proof streams bind different physical envelopes")]
    EnvelopeCommitmentMismatch,
    /// Transport and semantic proof streams disagree on the command content.
    #[error("final gate transport and semantic commands differ")]
    CommandMismatch,
    /// Hardware report binds another durable device semantic generation.
    #[error("physical-interlock report binds another device semantic head")]
    DeviceSemanticHeadMismatch,
    /// Hardware report binds another transport-trust generation.
    #[error("physical-interlock report binds another transport-trust head")]
    TransportTrustHeadMismatch,
    /// Hardware observation predates authenticated Xenia receipt acceptance.
    #[error("physical-interlock observation predates authenticated transport acceptance")]
    InterlockPredatesAuthenticatedTransport,
    /// Relying-party clock is earlier than authenticated Xenia acceptance.
    #[error("final join time predates authenticated transport acceptance")]
    FinalJoinPredatesAuthenticatedTransport,
    /// Authenticated transport evidence is too old for the final physical join.
    #[error("authenticated transport evidence is too old for actuation")]
    AuthenticatedTransportTooOldForActuation,
    /// Previously verified hardware evidence is no longer fresh at join time.
    #[error("physical-interlock evidence is no longer fresh")]
    InterlockNoLongerFresh,
    /// Physical-envelope second-to-millisecond conversion overflowed.
    #[error("physical-effect envelope send deadline overflowed milliseconds")]
    PhysicalEnvelopeTimeOverflow,
    /// Physical envelope's host/device send deadline has elapsed.
    #[error("physical-effect envelope send deadline elapsed")]
    PhysicalEnvelopeSendDeadlineElapsed,
    /// Local transport-age deadline overflowed.
    #[error("final transport-age deadline overflowed")]
    TransportAgeDeadlineOverflow,
    /// The derived minimum final dispatch window has elapsed.
    #[error("final actuator dispatch window elapsed")]
    FinalDispatchWindowElapsed,
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use symthaea_action_checkpoint::CheckpointHead;
    use symthaea_authority::{Operation, PrincipalId, TaskId};
    use symthaea_iot_authority::{
        DEVICE_COMMAND_SCHEMA_VERSION, DeviceRuntimeState, InclusiveRangeI64,
        SAFETY_ENVELOPE_SCHEMA_VERSION, SafetyEnvelope,
    };
    use symthaea_iot_device_protocol::{
        DEVICE_ENFORCEMENT_CONFIG_SCHEMA_VERSION, DeviceEnforcementConfigV1,
        DeviceSemanticCheckpointV1, PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION,
        PhysicalEffectEnvelopeV1, prepare_semantic_acceptance,
    };
    use symthaea_iot_durable_runtime::DurableIoTHead;
    use symthaea_iot_policy::ActuationPolicyHead;
    use symthaea_iot_posture::VerifierTrustHead;
    use symthaea_iot_transport_receipt::{
        TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION, TransportAttestorKeyV1,
        TransportAttestorStatus, TransportTrustRegistry, TransportTrustSnapshotV1,
        XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA, XENIA_ED25519_SIGNATURE_LEN,
        XENIA_HYBRID_SIGNATURE_SUITE, XENIA_ML_DSA_65_PUBLIC_KEY_LEN,
        XENIA_ML_DSA_65_SIGNATURE_LEN, XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE,
        XeniaAuthenticatedPayloadReceiptBodyV1, XeniaAuthenticatedPayloadReceiptV1,
        XeniaReceiptPeerRoleV1, HybridReceiptSignatureVerifier,
        verify_xenia_transport_receipt,
    };

    struct TestReceiptVerifier;

    impl HybridReceiptSignatureVerifier for TestReceiptVerifier {
        fn verify_ed25519(
            &self,
            _public_key: &[u8; 32],
            digest: &[u8; 32],
            signature: &[u8; XENIA_ED25519_SIGNATURE_LEN],
        ) -> bool {
            signature[..32] == digest[..]
        }

        fn verify_ml_dsa_65(
            &self,
            _public_key: &[u8],
            digest: &[u8; 32],
            signature: &[u8; XENIA_ML_DSA_65_SIGNATURE_LEN],
        ) -> bool {
            signature[..32] == digest[..]
        }
    }

    struct AcceptHardwareEvidence;

    impl HardwareInterlockEvidenceVerifier for AcceptHardwareEvidence {
        fn verify_interlock_evidence(
            &self,
            _controller_id: &str,
            _report_digest: Digest32,
            raw_evidence: &[u8],
        ) -> bool {
            !raw_evidence.is_empty()
        }
    }

    struct RejectHardwareEvidence;

    impl HardwareInterlockEvidenceVerifier for RejectHardwareEvidence {
        fn verify_interlock_evidence(
            &self,
            _controller_id: &str,
            _report_digest: Digest32,
            _raw_evidence: &[u8],
        ) -> bool {
            false
        }
    }

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn safety() -> SafetyEnvelope {
        SafetyEnvelope {
            schema_version: SAFETY_ENVELOPE_SCHEMA_VERSION,
            policy_id: "device-local-safe-open".into(),
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            allowed_firmware: BTreeSet::from([d(7)]),
            parameter_ranges: BTreeMap::from([(
                "duration_ms".into(),
                InclusiveRangeI64 {
                    min: 1_000,
                    max: 120_000,
                },
            )]),
            required_observations: BTreeMap::from([(
                "pressure_x100".into(),
                InclusiveRangeI64 {
                    min: 100,
                    max: 350_000,
                },
            )]),
        }
    }

    fn config() -> DeviceEnforcementConfigV1 {
        DeviceEnforcementConfigV1 {
            schema_version: DEVICE_ENFORCEMENT_CONFIG_SCHEMA_VERSION,
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            exact_policy_digest: d(20),
            minimum_policy_registry_sequence: 5,
            safety: safety(),
            maximum_envelope_lifetime_s: 5,
        }
    }

    fn runtime() -> DeviceRuntimeState {
        DeviceRuntimeState {
            running_firmware: d(7),
            last_accepted_sequence: None,
            observations: BTreeMap::from([("pressure_x100".into(), 20_000)]),
        }
    }

    fn envelope(sequence: u64) -> PhysicalEffectEnvelopeV1 {
        PhysicalEffectEnvelopeV1 {
            schema_version: PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION,
            command: DeviceCommand {
                schema_version: DEVICE_COMMAND_SCHEMA_VERSION,
                command_id: format!("cmd-{sequence}"),
                actor: PrincipalId("agent:irrigation".into()),
                executor: PrincipalId("gateway:field-a".into()),
                task: Some(TaskId("irrigate:zone-7".into())),
                device: ResourceRef("iot:valve:72".into()),
                operation: Operation("valve.open".into()),
                expected_firmware: d(7),
                sequence,
                issued_at_unix_s: 100,
                expires_at_unix_s: 120,
                parameters: BTreeMap::from([("duration_ms".into(), 5_000)]),
            },
            proposal_digest: d(2),
            policy_digest: d(20),
            policy_registry_head: ActuationPolicyHead {
                sequence: 5,
                digest: d(4),
            },
            durable_host_head: DurableIoTHead {
                action_head: CheckpointHead {
                    sequence: 1,
                    digest: d(5),
                },
                digest: d(6),
            },
            posture_result_digest: d(8),
            posture_evidence_digest: d(9),
            posture_reference_values_digest: d(10),
            posture_appraisal_policy_digest: d(11),
            posture_challenge_digest: d(12),
            posture_verifier_trust_head: VerifierTrustHead {
                sequence: 1,
                digest: d(13),
            },
            posture_expires_at_unix_s: 120,
            host_preflight_at_unix_s: 110,
            send_not_after_unix_s: 115,
        }
    }

    fn semantic(envelope: PhysicalEffectEnvelopeV1) -> SemanticallyAcceptedEffect {
        let checkpoint = DeviceSemanticCheckpointV1::genesis(&config()).unwrap();
        let head = checkpoint.head().unwrap();
        let pending = prepare_semantic_acceptance(
            envelope,
            &config(),
            &runtime(),
            &checkpoint,
            head,
            112,
        )
        .unwrap();
        let expected = pending.expected_head();
        pending.confirm_persisted(expected).unwrap()
    }

    fn registry() -> TransportTrustRegistry {
        TransportTrustRegistry::genesis(TransportTrustSnapshotV1 {
            schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_ms: 90_000,
            expires_at_unix_ms: 130_000,
            previous_snapshot_digest: None,
            keys: vec![TransportAttestorKeyV1 {
                attestor_id: "xenia-gateway-a".into(),
                key_id: "transport-key-1".into(),
                ed25519_public_key: [0x21; 32],
                ml_dsa_public_key: vec![0x22; XENIA_ML_DSA_65_PUBLIC_KEY_LEN],
                status: TransportAttestorStatus::Active,
                not_before_unix_ms: 90_000,
                not_after_unix_ms: 130_000,
                max_receipt_lifetime_ms: 2_000,
                required_peer_role: XeniaReceiptPeerRoleV1::Viewer,
                allowed_peer_fingerprints: BTreeSet::from([[0x44; 32]]),
                require_input_control: true,
            }],
        })
        .unwrap()
    }

    fn transport(envelope: &PhysicalEffectEnvelopeV1) -> VerifiedTransportEnvelope {
        let raw_payload = bincode::serialize(envelope).unwrap();
        let body = XeniaAuthenticatedPayloadReceiptBodyV1 {
            schema: XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA.into(),
            attestor_id: "xenia-gateway-a".into(),
            key_id: "transport-key-1".into(),
            signature_algorithm: XENIA_HYBRID_SIGNATURE_SUITE.into(),
            session_evidence_digest: [0x31; 32],
            peer_role: XeniaReceiptPeerRoleV1::Viewer,
            peer_identity_fingerprint: [0x44; 32],
            transcript_hash: [0x45; 32],
            session_context_hash: [0x46; 32],
            telemetry_enabled: false,
            input_control_enabled: true,
            payload_type: XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE,
            payload_len: raw_payload.len() as u32,
            payload_digest: *blake3::hash(&raw_payload).as_bytes(),
            sealed_envelope_digest: [0x47; 32],
            opened_at_unix_ms: 112_000,
            expires_at_unix_ms: 114_000,
        };
        let digest = body.signing_digest().unwrap();
        let mut ed = [0u8; XENIA_ED25519_SIGNATURE_LEN];
        ed[..32].copy_from_slice(&digest);
        let mut pq = [0u8; XENIA_ML_DSA_65_SIGNATURE_LEN];
        pq[..32].copy_from_slice(&digest);
        let raw_receipt = bincode::serialize(&XeniaAuthenticatedPayloadReceiptV1 {
            body,
            ed25519_signature: ed,
            ml_dsa_signature: pq,
        })
        .unwrap();
        verify_xenia_transport_receipt(
            &registry(),
            &raw_receipt,
            &raw_payload,
            113_000,
            &TestReceiptVerifier,
        )
        .unwrap()
    }

    fn interlock_policy() -> PhysicalInterlockPolicyV1 {
        PhysicalInterlockPolicyV1 {
            schema_version: PHYSICAL_INTERLOCK_POLICY_SCHEMA_VERSION,
            device: ResourceRef("iot:valve:72".into()),
            allowed_controllers: BTreeSet::from(["safety-plc:field-a".into()]),
            required_interlocks: BTreeSet::from([
                "emergency-stop-clear".into(),
                "pressure-interlock-ready".into(),
            ]),
            max_report_lifetime_ms: 1_000,
        }
    }

    fn interlock(
        transport: &VerifiedTransportEnvelope,
        semantic: &SemanticallyAcceptedEffect,
        checked_at_unix_ms: u64,
        expires_at_unix_ms: u64,
    ) -> VerifiedPhysicalInterlock {
        let raw_evidence = b"authenticated-safety-plc-evidence";
        verify_physical_interlock(
            &interlock_policy(),
            PhysicalInterlockReportV1 {
                schema_version: PHYSICAL_INTERLOCK_REPORT_SCHEMA_VERSION,
                controller_id: "safety-plc:field-a".into(),
                device: transport.envelope().command.device.clone(),
                envelope_digest: transport.envelope_digest(),
                device_head: semantic.device_head(),
                transport_trust_head: transport.trust_head(),
                asserted_interlocks: interlock_policy().required_interlocks,
                checked_at_unix_ms,
                expires_at_unix_ms,
                evidence_digest: Digest32(*blake3::hash(raw_evidence).as_bytes()),
            },
            raw_evidence,
            checked_at_unix_ms,
            &AcceptHardwareEvidence,
        )
        .unwrap()
    }

    #[test]
    fn exact_three_way_join_mints_final_nonportable_permit() {
        let envelope = envelope(7);
        let transport = transport(&envelope);
        let semantic = semantic(envelope);
        let interlock = interlock(&transport, &semantic, 113_100, 113_900);

        let permit = join_final_actuator_gate(transport, semantic, interlock, 113_200).unwrap();
        assert_eq!(permit.command().sequence, 7);
        assert_eq!(permit.interlock_controller_id(), "safety-plc:field-a");
        assert!(permit.must_dispatch_by_unix_ms() >= 113_200);
    }

    #[test]
    fn valid_proofs_for_different_envelopes_cannot_be_combined() {
        let envelope_a = envelope(7);
        let envelope_b = envelope(8);
        let transport = transport(&envelope_a);
        let semantic = semantic(envelope_b);
        let interlock = interlock(&transport, &semantic, 113_100, 113_900);

        assert!(matches!(
            join_final_actuator_gate(transport, semantic, interlock, 113_200),
            Err(FinalActuatorGateError::EnvelopeCommitmentMismatch)
        ));
    }

    #[test]
    fn interlock_cannot_predate_authenticated_transport() {
        let envelope = envelope(7);
        let transport = transport(&envelope);
        let semantic = semantic(envelope);
        let interlock = interlock(&transport, &semantic, 111_900, 112_800);

        assert!(matches!(
            join_final_actuator_gate(transport, semantic, interlock, 112_100),
            Err(FinalActuatorGateError::InterlockPredatesAuthenticatedTransport)
        ));
    }

    #[test]
    fn missing_required_physical_interlock_fails_before_provider_trust() {
        let envelope = envelope(7);
        let transport = transport(&envelope);
        let semantic = semantic(envelope);
        let raw_evidence = b"authenticated-safety-plc-evidence";
        let mut asserted = interlock_policy().required_interlocks;
        asserted.remove("pressure-interlock-ready");
        let report = PhysicalInterlockReportV1 {
            schema_version: PHYSICAL_INTERLOCK_REPORT_SCHEMA_VERSION,
            controller_id: "safety-plc:field-a".into(),
            device: transport.envelope().command.device.clone(),
            envelope_digest: transport.envelope_digest(),
            device_head: semantic.device_head(),
            transport_trust_head: transport.trust_head(),
            asserted_interlocks: asserted,
            checked_at_unix_ms: 113_100,
            expires_at_unix_ms: 113_900,
            evidence_digest: Digest32(*blake3::hash(raw_evidence).as_bytes()),
        };

        assert!(matches!(
            verify_physical_interlock(
                &interlock_policy(),
                report,
                raw_evidence,
                113_200,
                &AcceptHardwareEvidence,
            ),
            Err(FinalActuatorGateError::InterlockSetMismatch)
        ));
    }

    #[test]
    fn hardware_provider_rejection_fails_closed() {
        let envelope = envelope(7);
        let transport = transport(&envelope);
        let semantic = semantic(envelope);
        let raw_evidence = b"authenticated-safety-plc-evidence";
        let report = PhysicalInterlockReportV1 {
            schema_version: PHYSICAL_INTERLOCK_REPORT_SCHEMA_VERSION,
            controller_id: "safety-plc:field-a".into(),
            device: transport.envelope().command.device.clone(),
            envelope_digest: transport.envelope_digest(),
            device_head: semantic.device_head(),
            transport_trust_head: transport.trust_head(),
            asserted_interlocks: interlock_policy().required_interlocks,
            checked_at_unix_ms: 113_100,
            expires_at_unix_ms: 113_900,
            evidence_digest: Digest32(*blake3::hash(raw_evidence).as_bytes()),
        };

        assert!(matches!(
            verify_physical_interlock(
                &interlock_policy(),
                report,
                raw_evidence,
                113_200,
                &RejectHardwareEvidence,
            ),
            Err(FinalActuatorGateError::InterlockEvidenceVerificationFailed)
        ));
    }

    #[test]
    fn transport_token_ages_out_before_outer_send_window() {
        let envelope = envelope(7);
        let transport = transport(&envelope);
        let semantic = semantic(envelope);
        let interlock = interlock(&transport, &semantic, 114_000, 114_900);

        assert!(matches!(
            join_final_actuator_gate(transport, semantic, interlock, 114_100),
            Err(FinalActuatorGateError::AuthenticatedTransportTooOldForActuation)
        ));
    }
}
