// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complete just-in-time revocation fencing before cyber-physical HAL access.
//!
//! A [`FinalActuatorPermit`] proves that transport authentication, durable device
//! semantics, and physical interlock evidence agreed at the final software join. Three
//! independently current state roots can still invalidate that permit before hardware
//! access:
//!
//! 1. Xenia transport-attestor trust can advance, revoke, rotate, or naturally expire;
//! 2. durable device semantic state can advance after a newer command is accepted; and
//! 3. hardware-controller trust can advance, revoke, rotate, or naturally expire.
//!
//! This crate checks all three roots at one relying-party time, re-binds the exact raw
//! Xenia receipt to the receipt digest already committed in the final permit, rechecks
//! the exact transport key and receipt lifetime, re-verifies the exact hardware
//! evidence under current controller-key trust, and then invokes the existing interlock
//! JIT fence.
//!
//! The resulting [`CompleteJitHalLease`] holds borrows of the exact current transport
//! registry and durable device checkpoint while its inner `JustInTimeHalLease` holds
//! the exact current interlock registry. This is ordinary Rust type-state, not a sandbox:
//! a production guard must still serialize state mutation and own exclusive HAL/device
//! access in a minimal privileged process.

#![deny(unsafe_code)]

use symthaea_authority::Digest32;
use symthaea_iot_authority::DeviceCommand;
use symthaea_iot_device_protocol::{DeviceSemanticCheckpointV1, DeviceSemanticHead};
use symthaea_iot_final_gate::FinalActuatorPermit;
use symthaea_iot_interlock_trust::{
    InterlockControllerEvidenceVerifier, InterlockTrustHead, InterlockTrustRegistry,
    upgrade_final_actuator_permit, verify_interlock_key_binding,
};
use symthaea_iot_jit_fence::{JustInTimeHalLease, fence_for_hal};
use symthaea_iot_transport_receipt::{
    MAX_XENIA_RECEIPT_BYTES, TransportAttestorStatus, TransportTrustHead,
    TransportTrustRegistry, XeniaAuthenticatedPayloadReceiptV1,
};
use thiserror::Error;

/// Exact guard-owned current state and independently retained anti-rollback anchors.
///
/// These values must be loaded/authenticated by the privileged guard itself. They are
/// deliberately separate from the unprivileged evidence request defined by
/// `symthaea-iot-actuation-guard-protocol`.
#[derive(Debug, Clone, Copy)]
pub struct GuardOwnedCurrentState<'a> {
    interlock_registry: &'a InterlockTrustRegistry,
    interlock_head: InterlockTrustHead,
    transport_registry: &'a TransportTrustRegistry,
    transport_head: TransportTrustHead,
    device_checkpoint: &'a DeviceSemanticCheckpointV1,
    device_head: DeviceSemanticHead,
}

impl<'a> GuardOwnedCurrentState<'a> {
    /// Bind the exact current registries/checkpoint to their independently retained heads.
    pub const fn new(
        interlock_registry: &'a InterlockTrustRegistry,
        interlock_head: InterlockTrustHead,
        transport_registry: &'a TransportTrustRegistry,
        transport_head: TransportTrustHead,
        device_checkpoint: &'a DeviceSemanticCheckpointV1,
        device_head: DeviceSemanticHead,
    ) -> Self {
        Self {
            interlock_registry,
            interlock_head,
            transport_registry,
            transport_head,
            device_checkpoint,
            device_head,
        }
    }
}

/// Final local HAL lease after every revocable current-state root was fenced.
///
/// This type is intentionally non-clone and non-serializable. The privileged guard
/// should consume it immediately for one hardware attempt and must not permit another
/// command to advance guard-owned semantic/trust state concurrently with that attempt.
#[derive(Debug)]
pub struct CompleteJitHalLease<'a> {
    inner: JustInTimeHalLease<'a>,
    _current_transport_registry: &'a TransportTrustRegistry,
    _current_device_checkpoint: &'a DeviceSemanticCheckpointV1,
    current_transport_head: TransportTrustHead,
    current_device_head: DeviceSemanticHead,
    transport_attestor_id: String,
    transport_key_id: String,
    transport_key_digest: Digest32,
    must_attempt_by_unix_ms: u64,
}

impl CompleteJitHalLease<'_> {
    /// Exact physical command whose complete revocation vector survived the JIT fence.
    pub fn command(&self) -> &DeviceCommand {
        self.inner.command()
    }

    /// Exact physical-effect envelope commitment.
    pub const fn envelope_digest(&self) -> Digest32 {
        self.inner.envelope_digest()
    }

    /// Current interlock-controller trust head held by the inner JIT lease.
    pub const fn current_interlock_trust_head(&self) -> InterlockTrustHead {
        self.inner.current_trust_head()
    }

    /// Current Xenia transport-attestor trust head.
    pub const fn current_transport_trust_head(&self) -> TransportTrustHead {
        self.current_transport_head
    }

    /// Current durable device semantic head.
    pub const fn current_device_head(&self) -> DeviceSemanticHead {
        self.current_device_head
    }

    /// Exact Xenia transport attestor identity rechecked at HAL handoff.
    pub fn transport_attestor_id(&self) -> &str {
        &self.transport_attestor_id
    }

    /// Exact Xenia transport key identity rechecked at HAL handoff.
    pub fn transport_key_id(&self) -> &str {
        &self.transport_key_id
    }

    /// Commitment to the current transport key/policy record.
    pub const fn transport_key_digest(&self) -> Digest32 {
        self.transport_key_digest
    }

    /// Local time at which the complete JIT fence ran.
    pub const fn fenced_at_unix_ms(&self) -> u64 {
        self.inner.fenced_at_unix_ms()
    }

    /// Inclusive latest millisecond at which the privileged guard may begin HAL I/O.
    pub const fn must_attempt_by_unix_ms(&self) -> u64 {
        self.must_attempt_by_unix_ms
    }
}

#[derive(Debug)]
struct CurrentTransportBinding {
    attestor_id: String,
    key_id: String,
    key_digest: Digest32,
    last_valid_unix_ms: u64,
}

/// Consume a final software permit and produce the final HAL lease only if every
/// revocable guard-owned state root is still exactly current.
///
/// `raw_transport_receipt` and `raw_interlock_evidence` must be the exact portable
/// evidence retained from the guard's bounded IPC request. The receipt body is rebound
/// to `permit.transport_receipt_digest()` before it is used to select a current key.
pub fn fence_final_permit_for_hal<'a>(
    permit: FinalActuatorPermit,
    current: GuardOwnedCurrentState<'a>,
    raw_transport_receipt: &[u8],
    raw_interlock_evidence: &[u8],
    now_unix_ms: u64,
    interlock_verifier: &impl InterlockControllerEvidenceVerifier,
) -> Result<CompleteJitHalLease<'a>, JitRevocationError> {
    validate_transport_heads(
        current.transport_registry.head(),
        current.transport_head,
        permit.transport_trust_head(),
    )?;

    let actual_device_head = current
        .device_checkpoint
        .head()
        .map_err(|_| JitRevocationError::CurrentDeviceCheckpointInvalid)?;
    validate_device_heads(actual_device_head, current.device_head, permit.device_head())?;

    let transport = validate_current_transport_binding(
        current.transport_registry,
        current.transport_head,
        permit.transport_receipt_digest(),
        raw_transport_receipt,
        now_unix_ms,
    )?;

    let binding = verify_interlock_key_binding(
        current.interlock_registry,
        permit.interlock_controller_id(),
        permit.interlock_report_digest(),
        permit.interlock_evidence_digest(),
        raw_interlock_evidence,
        permit.joined_at_unix_ms(),
        now_unix_ms,
        interlock_verifier,
    )
    .map_err(|_| JitRevocationError::CurrentInterlockBindingFailed)?;

    let trust_bound = upgrade_final_actuator_permit(permit, binding, now_unix_ms)
        .map_err(|_| JitRevocationError::FinalPermitTrustUpgradeFailed)?;
    let inner = fence_for_hal(
        trust_bound,
        current.interlock_registry,
        current.interlock_head,
        now_unix_ms,
    )
    .map_err(|_| JitRevocationError::InterlockJitFenceFailed)?;

    let must_attempt_by_unix_ms = inner
        .must_attempt_by_unix_ms()
        .min(transport.last_valid_unix_ms);
    if must_attempt_by_unix_ms < now_unix_ms {
        return Err(JitRevocationError::NoLiveCompleteHalWindow);
    }

    Ok(CompleteJitHalLease {
        inner,
        _current_transport_registry: current.transport_registry,
        _current_device_checkpoint: current.device_checkpoint,
        current_transport_head: current.transport_head,
        current_device_head: current.device_head,
        transport_attestor_id: transport.attestor_id,
        transport_key_id: transport.key_id,
        transport_key_digest: transport.key_digest,
        must_attempt_by_unix_ms,
    })
}

fn validate_transport_heads(
    registry_head: TransportTrustHead,
    externally_anchored_head: TransportTrustHead,
    permit_head: TransportTrustHead,
) -> Result<(), JitRevocationError> {
    if registry_head != externally_anchored_head {
        return Err(JitRevocationError::ExternallyAnchoredTransportHeadMismatch);
    }
    if permit_head != externally_anchored_head {
        return Err(JitRevocationError::TransportTrustAdvancedSincePermit);
    }
    Ok(())
}

fn validate_device_heads(
    checkpoint_head: DeviceSemanticHead,
    externally_anchored_head: DeviceSemanticHead,
    permit_head: DeviceSemanticHead,
) -> Result<(), JitRevocationError> {
    if checkpoint_head != externally_anchored_head {
        return Err(JitRevocationError::ExternallyAnchoredDeviceHeadMismatch);
    }
    if permit_head != externally_anchored_head {
        return Err(JitRevocationError::DeviceStateAdvancedSincePermit);
    }
    Ok(())
}

fn validate_current_transport_binding(
    registry: &TransportTrustRegistry,
    externally_anchored_head: TransportTrustHead,
    permit_receipt_digest: Digest32,
    raw_receipt: &[u8],
    now_unix_ms: u64,
) -> Result<CurrentTransportBinding, JitRevocationError> {
    if raw_receipt.is_empty() || raw_receipt.len() > MAX_XENIA_RECEIPT_BYTES {
        return Err(JitRevocationError::TransportReceiptSizeOutOfBounds);
    }
    if registry.head() != externally_anchored_head {
        return Err(JitRevocationError::ExternallyAnchoredTransportHeadMismatch);
    }

    let snapshot = registry.snapshot();
    if now_unix_ms < snapshot.issued_at_unix_ms || now_unix_ms >= snapshot.expires_at_unix_ms {
        return Err(JitRevocationError::CurrentTransportSnapshotNotFresh);
    }

    let receipt: XeniaAuthenticatedPayloadReceiptV1 = bincode::deserialize(raw_receipt)
        .map_err(|_| JitRevocationError::InvalidTransportReceiptEncoding)?;
    let canonical = bincode::serialize(&receipt)
        .map_err(|_| JitRevocationError::InvalidTransportReceiptEncoding)?;
    if canonical != raw_receipt {
        return Err(JitRevocationError::NonCanonicalTransportReceipt);
    }
    receipt
        .body
        .validate_structure()
        .map_err(|_| JitRevocationError::InvalidTransportReceiptStructure)?;
    let signing_digest = receipt
        .body
        .signing_digest()
        .map_err(|_| JitRevocationError::InvalidTransportReceiptStructure)?;
    if Digest32(signing_digest) != permit_receipt_digest {
        return Err(JitRevocationError::TransportReceiptDoesNotMatchPermit);
    }
    if now_unix_ms < receipt.body.opened_at_unix_ms
        || now_unix_ms >= receipt.body.expires_at_unix_ms
    {
        return Err(JitRevocationError::TransportReceiptNotFreshAtHal);
    }

    let key = snapshot
        .keys
        .iter()
        .find(|key| {
            key.attestor_id == receipt.body.attestor_id && key.key_id == receipt.body.key_id
        })
        .ok_or(JitRevocationError::CurrentTransportKeyMissing)?;
    if key.status != TransportAttestorStatus::Active
        || receipt.body.opened_at_unix_ms < key.not_before_unix_ms
        || receipt.body.opened_at_unix_ms >= key.not_after_unix_ms
        || now_unix_ms < key.not_before_unix_ms
        || now_unix_ms >= key.not_after_unix_ms
    {
        return Err(JitRevocationError::CurrentTransportKeyNotActive);
    }
    let receipt_lifetime = receipt
        .body
        .expires_at_unix_ms
        .checked_sub(receipt.body.opened_at_unix_ms)
        .ok_or(JitRevocationError::InvalidTransportReceiptStructure)?;
    if receipt_lifetime > key.max_receipt_lifetime_ms {
        return Err(JitRevocationError::TransportReceiptLifetimeExceedsCurrentKeyPolicy);
    }

    let key_digest = key
        .digest()
        .map_err(|_| JitRevocationError::CurrentTransportKeyInvalid)?;
    let snapshot_last_valid = snapshot
        .expires_at_unix_ms
        .checked_sub(1)
        .ok_or(JitRevocationError::TimeOverflow)?;
    let key_last_valid = key
        .not_after_unix_ms
        .checked_sub(1)
        .ok_or(JitRevocationError::TimeOverflow)?;
    let receipt_last_valid = receipt
        .body
        .expires_at_unix_ms
        .checked_sub(1)
        .ok_or(JitRevocationError::TimeOverflow)?;

    Ok(CurrentTransportBinding {
        attestor_id: key.attestor_id.clone(),
        key_id: key.key_id.clone(),
        key_digest,
        last_valid_unix_ms: snapshot_last_valid
            .min(key_last_valid)
            .min(receipt_last_valid),
    })
}

/// Fail-closed errors for the complete HAL-time revocation vector.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum JitRevocationError {
    /// Current transport registry differs from its independently retained head.
    #[error("current transport registry does not match externally anchored head")]
    ExternallyAnchoredTransportHeadMismatch,
    /// Transport trust changed after the final permit was created.
    #[error("transport trust advanced after final permit creation")]
    TransportTrustAdvancedSincePermit,
    /// Current durable device checkpoint could not produce a valid head.
    #[error("current durable device checkpoint is invalid")]
    CurrentDeviceCheckpointInvalid,
    /// Current device checkpoint differs from the independently retained device head.
    #[error("current device checkpoint does not match externally anchored device head")]
    ExternallyAnchoredDeviceHeadMismatch,
    /// Durable device semantic state advanced after the permit was created.
    #[error("durable device semantic state advanced after final permit creation")]
    DeviceStateAdvancedSincePermit,
    /// Raw Xenia receipt is empty or oversized.
    #[error("transport receipt size is outside accepted bounds at HAL fence")]
    TransportReceiptSizeOutOfBounds,
    /// Current transport trust snapshot is not fresh.
    #[error("current transport trust snapshot is not fresh at HAL fence")]
    CurrentTransportSnapshotNotFresh,
    /// Raw Xenia receipt is not decodable as the exact v1 wire shape.
    #[error("transport receipt encoding is invalid at HAL fence")]
    InvalidTransportReceiptEncoding,
    /// Raw Xenia receipt has an alternate/trailing encoding.
    #[error("transport receipt is not canonically encoded at HAL fence")]
    NonCanonicalTransportReceipt,
    /// Receipt body failed its bounded structural contract.
    #[error("transport receipt structure is invalid at HAL fence")]
    InvalidTransportReceiptStructure,
    /// Receipt body is not the exact body already committed into the permit.
    #[error("transport receipt does not match final permit commitment")]
    TransportReceiptDoesNotMatchPermit,
    /// Receipt is not fresh at HAL handoff.
    #[error("transport receipt expired or is from the future at HAL fence")]
    TransportReceiptNotFreshAtHal,
    /// Exact current attestor/key identity no longer exists.
    #[error("current transport attestor key is missing")]
    CurrentTransportKeyMissing,
    /// Exact transport key is retired, revoked, not yet valid, or expired.
    #[error("current transport attestor key is not active at HAL fence")]
    CurrentTransportKeyNotActive,
    /// Receipt lifetime violates the current unchanged key policy.
    #[error("transport receipt lifetime exceeds current key policy")]
    TransportReceiptLifetimeExceedsCurrentKeyPolicy,
    /// Current transport key record failed structural validation.
    #[error("current transport key record is invalid")]
    CurrentTransportKeyInvalid,
    /// Current controller-key proof could not be recreated from the permit commitments.
    #[error("current interlock controller binding failed at HAL fence")]
    CurrentInterlockBindingFailed,
    /// Final permit could not be upgraded under the current controller key.
    #[error("final actuator permit trust upgrade failed at HAL fence")]
    FinalPermitTrustUpgradeFailed,
    /// Existing interlock JIT fence rejected the current state.
    #[error("interlock just-in-time trust fence failed")]
    InterlockJitFenceFailed,
    /// Deadline arithmetic overflowed.
    #[error("complete HAL fence time arithmetic overflow")]
    TimeOverflow,
    /// No common live window remains across all current state roots.
    #[error("no live complete HAL attempt window remains")]
    NoLiveCompleteHalWindow,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use symthaea_iot_transport_receipt::{
        TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION, TransportAttestorKeyV1,
        TransportTrustSnapshotV1, XeniaAuthenticatedPayloadReceiptBodyV1,
        XeniaReceiptPeerRoleV1, XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA,
        XENIA_ED25519_SIGNATURE_LEN, XENIA_HYBRID_SIGNATURE_SUITE,
        XENIA_ML_DSA_65_PUBLIC_KEY_LEN, XENIA_ML_DSA_65_SIGNATURE_LEN,
        XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE,
    };

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn transport_key(not_after_unix_ms: u64) -> TransportAttestorKeyV1 {
        TransportAttestorKeyV1 {
            attestor_id: "xenia-transport:guard-a".into(),
            key_id: "receipt-key-1".into(),
            ed25519_public_key: [0x31; 32],
            ml_dsa_public_key: vec![0x41; XENIA_ML_DSA_65_PUBLIC_KEY_LEN],
            status: TransportAttestorStatus::Active,
            not_before_unix_ms: 100_000,
            not_after_unix_ms,
            max_receipt_lifetime_ms: 1_500,
            required_peer_role: XeniaReceiptPeerRoleV1::Viewer,
            allowed_peer_fingerprints: BTreeSet::from([[0x51; 32]]),
            require_input_control: true,
        }
    }

    fn transport_registry(not_after_unix_ms: u64) -> TransportTrustRegistry {
        TransportTrustRegistry::genesis(TransportTrustSnapshotV1 {
            schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_ms: 100_000,
            expires_at_unix_ms: 103_000,
            previous_snapshot_digest: None,
            keys: vec![transport_key(not_after_unix_ms)],
        })
        .unwrap()
    }

    fn receipt(expires_at_unix_ms: u64) -> XeniaAuthenticatedPayloadReceiptV1 {
        XeniaAuthenticatedPayloadReceiptV1 {
            body: XeniaAuthenticatedPayloadReceiptBodyV1 {
                schema: XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA.into(),
                attestor_id: "xenia-transport:guard-a".into(),
                key_id: "receipt-key-1".into(),
                signature_algorithm: XENIA_HYBRID_SIGNATURE_SUITE.into(),
                session_evidence_digest: [0x11; 32],
                peer_role: XeniaReceiptPeerRoleV1::Viewer,
                peer_identity_fingerprint: [0x51; 32],
                transcript_hash: [0x12; 32],
                session_context_hash: [0x13; 32],
                telemetry_enabled: true,
                input_control_enabled: true,
                payload_type: XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE,
                payload_len: 128,
                payload_digest: [0x14; 32],
                sealed_envelope_digest: [0x15; 32],
                opened_at_unix_ms: 100_100,
                expires_at_unix_ms,
            },
            ed25519_signature: [0x61; XENIA_ED25519_SIGNATURE_LEN],
            ml_dsa_signature: [0x62; XENIA_ML_DSA_65_SIGNATURE_LEN],
        }
    }

    #[test]
    fn exact_transport_and_device_heads_pass() {
        let transport = TransportTrustHead {
            sequence: 7,
            digest: d(7),
        };
        assert_eq!(validate_transport_heads(transport, transport, transport), Ok(()));

        let device = DeviceSemanticHead {
            generation: 9,
            digest: d(9),
        };
        assert_eq!(validate_device_heads(device, device, device), Ok(()));
    }

    #[test]
    fn transport_generation_change_invalidates_old_permit() {
        let old = TransportTrustHead {
            sequence: 7,
            digest: d(7),
        };
        let current = TransportTrustHead {
            sequence: 8,
            digest: d(8),
        };
        assert_eq!(
            validate_transport_heads(current, current, old),
            Err(JitRevocationError::TransportTrustAdvancedSincePermit)
        );
    }

    #[test]
    fn device_semantic_advance_invalidates_old_permit() {
        let old = DeviceSemanticHead {
            generation: 9,
            digest: d(9),
        };
        let current = DeviceSemanticHead {
            generation: 10,
            digest: d(10),
        };
        assert_eq!(
            validate_device_heads(current, current, old),
            Err(JitRevocationError::DeviceStateAdvancedSincePermit)
        );
    }

    #[test]
    fn exact_receipt_rebinds_to_current_live_transport_key() {
        let registry = transport_registry(102_000);
        let receipt = receipt(101_000);
        let raw = bincode::serialize(&receipt).unwrap();
        let expected = Digest32(receipt.body.signing_digest().unwrap());
        let binding = validate_current_transport_binding(
            &registry,
            registry.head(),
            expected,
            &raw,
            100_500,
        )
        .unwrap();
        assert_eq!(binding.attestor_id, "xenia-transport:guard-a");
        assert_eq!(binding.key_id, "receipt-key-1");
        assert_eq!(binding.last_valid_unix_ms, 100_999);
    }

    #[test]
    fn natural_transport_key_expiry_invalidates_receipt_without_head_change() {
        let registry = transport_registry(100_500);
        let receipt = receipt(101_000);
        let raw = bincode::serialize(&receipt).unwrap();
        let expected = Digest32(receipt.body.signing_digest().unwrap());
        assert_eq!(
            validate_current_transport_binding(
                &registry,
                registry.head(),
                expected,
                &raw,
                100_600,
            )
            .unwrap_err(),
            JitRevocationError::CurrentTransportKeyNotActive
        );
    }

    #[test]
    fn receipt_expiry_invalidates_permit_even_when_key_and_head_remain_live() {
        let registry = transport_registry(102_000);
        let receipt = receipt(100_500);
        let raw = bincode::serialize(&receipt).unwrap();
        let expected = Digest32(receipt.body.signing_digest().unwrap());
        assert_eq!(
            validate_current_transport_binding(
                &registry,
                registry.head(),
                expected,
                &raw,
                100_500,
            )
            .unwrap_err(),
            JitRevocationError::TransportReceiptNotFreshAtHal
        );
    }

    #[test]
    fn different_receipt_body_cannot_select_transport_key_for_permit() {
        let registry = transport_registry(102_000);
        let receipt = receipt(101_000);
        let raw = bincode::serialize(&receipt).unwrap();
        assert_eq!(
            validate_current_transport_binding(
                &registry,
                registry.head(),
                d(0xEE),
                &raw,
                100_500,
            )
            .unwrap_err(),
            JitRevocationError::TransportReceiptDoesNotMatchPermit
        );
    }

    #[test]
    fn trailing_receipt_bytes_are_rejected_before_key_selection() {
        let registry = transport_registry(102_000);
        let receipt = receipt(101_000);
        let expected = Digest32(receipt.body.signing_digest().unwrap());
        let mut raw = bincode::serialize(&receipt).unwrap();
        raw.extend_from_slice(b"alternate-receipt");
        assert!(validate_current_transport_binding(
            &registry,
            registry.head(),
            expected,
            &raw,
            100_500,
        )
        .is_err());
    }
}
