// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Anti-rollback controller-key trust for physical interlock evidence.
//!
//! `symthaea-iot-final-gate` deliberately leaves the concrete hardware evidence
//! provider outside its core type-state join. This crate removes ambient controller
//! key configuration from that boundary by giving interlock verifier keys an explicit,
//! sequence-numbered trust lineage and by re-verifying the exact report/evidence
//! commitments already carried by a [`FinalActuatorPermit`].
//!
//! The result, [`TrustBoundFinalActuatorPermit`], remains local, non-clone and
//! non-serializable. A product/HAL adapter can require this stronger token when its
//! hardware-interlock provider is backed by explicit public-key trust.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_authority::Digest32;
use symthaea_iot_final_gate::{
    FinalActuatorPermit, MAX_PHYSICAL_INTERLOCK_EVIDENCE_BYTES,
};
use thiserror::Error;

/// Current controller-trust snapshot schema.
pub const INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION: u16 = 1;
/// Maximum controller/key records in one trust snapshot.
pub const MAX_INTERLOCK_TRUST_KEYS: usize = 512;
/// Maximum opaque public-key encoding retained by the generic trust registry.
pub const MAX_INTERLOCK_PUBLIC_KEY_BYTES: usize = 8 * 1024;
/// Maximum controller/key/algorithm label length.
pub const MAX_INTERLOCK_TRUST_LABEL_BYTES: usize = 128;

const INTERLOCK_KEY_DOMAIN: &[u8] = b"symthaea-iot-interlock-controller-key-v1\0";
const INTERLOCK_TRUST_DOMAIN: &[u8] = b"symthaea-iot-interlock-trust-snapshot-v1\0";

/// Monotonic lifecycle state for one controller verifier key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterlockControllerKeyStatus {
    /// Key may verify current interlock evidence within its time window.
    Active,
    /// Key is retired and may never become active again.
    Retired,
    /// Key is revoked; revocation is terminal.
    Revoked,
}

impl InterlockControllerKeyStatus {
    fn tag(self) -> u8 {
        match self {
            Self::Active => 0,
            Self::Retired => 1,
            Self::Revoked => 2,
        }
    }

    fn transition_allowed(self, next: Self) -> bool {
        match self {
            Self::Active => true,
            Self::Retired => matches!(next, Self::Retired | Self::Revoked),
            Self::Revoked => next == Self::Revoked,
        }
    }
}

/// One explicitly trusted hardware-interlock controller verification key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterlockControllerKeyV1 {
    /// Controller identity that must match the final permit.
    pub controller_id: String,
    /// Stable key lifecycle identity.
    pub key_id: String,
    /// Provider-specific algorithm/profile label.
    pub algorithm: String,
    /// Exact opaque public-key bytes consumed by the concrete verifier.
    pub public_key: Vec<u8>,
    /// Monotonic lifecycle state.
    pub status: InterlockControllerKeyStatus,
    /// Earliest Unix millisecond at which this key may authenticate evidence.
    pub not_before_unix_ms: u64,
    /// Exclusive key expiry in Unix milliseconds.
    pub not_after_unix_ms: u64,
}

impl InterlockControllerKeyV1 {
    /// Validate one key record independent of current time.
    pub fn validate(&self) -> Result<(), InterlockTrustError> {
        if !valid_label(&self.controller_id)
            || !valid_label(&self.key_id)
            || !valid_label(&self.algorithm)
        {
            return Err(InterlockTrustError::InvalidKeyIdentity);
        }
        if self.public_key.is_empty()
            || self.public_key.len() > MAX_INTERLOCK_PUBLIC_KEY_BYTES
            || self.public_key.iter().all(|byte| *byte == 0)
        {
            return Err(InterlockTrustError::InvalidPublicKey);
        }
        if self.not_before_unix_ms >= self.not_after_unix_ms {
            return Err(InterlockTrustError::InvalidKeyWindow);
        }
        Ok(())
    }

    /// Domain-separated commitment to key identity/material/lifecycle.
    pub fn digest(&self) -> Result<Digest32, InterlockTrustError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(INTERLOCK_KEY_DOMAIN);
        update_string(&mut h, &self.controller_id);
        update_string(&mut h, &self.key_id);
        update_string(&mut h, &self.algorithm);
        h.update(&(self.public_key.len() as u32).to_be_bytes());
        h.update(&self.public_key);
        h.update(&[self.status.tag()]);
        h.update(&self.not_before_unix_ms.to_be_bytes());
        h.update(&self.not_after_unix_ms.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    fn immutable_identity_eq(&self, other: &Self) -> bool {
        self.controller_id == other.controller_id
            && self.key_id == other.key_id
            && self.algorithm == other.algorithm
            && self.public_key == other.public_key
            && self.not_before_unix_ms == other.not_before_unix_ms
    }

    fn active_at(&self, unix_ms: u64) -> bool {
        self.status == InterlockControllerKeyStatus::Active
            && unix_ms >= self.not_before_unix_ms
            && unix_ms < self.not_after_unix_ms
    }
}

/// Public hash-chained controller-key trust generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterlockTrustSnapshotV1 {
    /// Fail-closed schema version.
    pub schema_version: u16,
    /// Monotonic generation; genesis is one.
    pub sequence: u64,
    /// Snapshot issue time in Unix milliseconds.
    pub issued_at_unix_ms: u64,
    /// Exclusive snapshot expiry.
    pub expires_at_unix_ms: u64,
    /// Previous snapshot commitment; absent only for genesis.
    pub previous_snapshot_digest: Option<Digest32>,
    /// Complete persistent controller-key lifecycle table.
    pub keys: Vec<InterlockControllerKeyV1>,
}

impl InterlockTrustSnapshotV1 {
    /// Validate one public trust generation.
    pub fn validate(&self) -> Result<(), InterlockTrustError> {
        if self.schema_version != INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION {
            return Err(InterlockTrustError::UnsupportedTrustSchema);
        }
        if self.sequence == 0 {
            return Err(InterlockTrustError::TrustSequenceZero);
        }
        if self.issued_at_unix_ms >= self.expires_at_unix_ms {
            return Err(InterlockTrustError::InvalidTrustWindow);
        }
        if self.keys.is_empty() || self.keys.len() > MAX_INTERLOCK_TRUST_KEYS {
            return Err(InterlockTrustError::InvalidTrustKeyCount);
        }
        if self.sequence == 1 && self.previous_snapshot_digest.is_some() {
            return Err(InterlockTrustError::GenesisHasPredecessor);
        }
        if self.sequence > 1 && self.previous_snapshot_digest.is_none() {
            return Err(InterlockTrustError::SuccessorMissingPredecessor);
        }

        let mut identities = BTreeSet::new();
        let mut active_controllers = BTreeSet::new();
        for key in &self.keys {
            key.validate()?;
            if !identities.insert((key.controller_id.clone(), key.key_id.clone())) {
                return Err(InterlockTrustError::DuplicateKeyIdentity);
            }
            if key.status == InterlockControllerKeyStatus::Active
                && !active_controllers.insert(key.controller_id.clone())
            {
                return Err(InterlockTrustError::MultipleActiveKeysForController);
            }
        }
        Ok(())
    }

    /// Order-independent commitment to the complete snapshot.
    pub fn digest(&self) -> Result<Digest32, InterlockTrustError> {
        self.validate()?;
        let mut keys = self.keys.iter().collect::<Vec<_>>();
        keys.sort_by(|left, right| {
            (&left.controller_id, &left.key_id).cmp(&(&right.controller_id, &right.key_id))
        });
        let mut h = blake3::Hasher::new();
        h.update(INTERLOCK_TRUST_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.sequence.to_be_bytes());
        h.update(&self.issued_at_unix_ms.to_be_bytes());
        h.update(&self.expires_at_unix_ms.to_be_bytes());
        match self.previous_snapshot_digest {
            Some(Digest32(bytes)) => {
                h.update(&[1]);
                h.update(&bytes);
            }
            None => {
                h.update(&[0]);
            }
        }
        h.update(&(keys.len() as u32).to_be_bytes());
        for key in keys {
            let Digest32(bytes) = key.digest()?;
            h.update(&bytes);
        }
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    fn key_map(&self) -> BTreeMap<(&str, &str), &InterlockControllerKeyV1> {
        self.keys
            .iter()
            .map(|key| ((key.controller_id.as_str(), key.key_id.as_str()), key))
            .collect()
    }
}

/// Externally retainable anti-rollback anchor for interlock controller trust.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterlockTrustHead {
    /// Current generation.
    pub sequence: u64,
    /// Commitment to the exact snapshot.
    pub digest: Digest32,
}

/// Verified in-process controller-key trust state.
#[derive(Debug)]
pub struct InterlockTrustRegistry {
    snapshot: InterlockTrustSnapshotV1,
    head: InterlockTrustHead,
}

impl InterlockTrustRegistry {
    /// Accept generation one of a new controller trust lineage.
    pub fn genesis(snapshot: InterlockTrustSnapshotV1) -> Result<Self, InterlockTrustError> {
        snapshot.validate()?;
        if snapshot.sequence != 1 || snapshot.previous_snapshot_digest.is_some() {
            return Err(InterlockTrustError::NotGenesis);
        }
        let head = InterlockTrustHead {
            sequence: 1,
            digest: snapshot.digest()?,
        };
        Ok(Self { snapshot, head })
    }

    /// Verify and accept the immediate successor generation.
    pub fn successor(
        &self,
        snapshot: InterlockTrustSnapshotV1,
    ) -> Result<Self, InterlockTrustError> {
        snapshot.validate()?;
        let expected = self
            .head
            .sequence
            .checked_add(1)
            .ok_or(InterlockTrustError::TrustSequenceOverflow)?;
        if snapshot.sequence != expected {
            return Err(InterlockTrustError::TrustSequenceNotNext {
                expected,
                proposed: snapshot.sequence,
            });
        }
        if snapshot.previous_snapshot_digest != Some(self.head.digest) {
            return Err(InterlockTrustError::TrustPredecessorMismatch);
        }
        if snapshot.issued_at_unix_ms < self.snapshot.issued_at_unix_ms {
            return Err(InterlockTrustError::TrustIssuedAtRegressed);
        }
        validate_successor(&self.snapshot, &snapshot)?;
        let head = InterlockTrustHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        Ok(Self { snapshot, head })
    }

    /// Restore a persisted snapshot only against its separately retained trusted head.
    pub fn restore(
        snapshot: InterlockTrustSnapshotV1,
        trusted_head: InterlockTrustHead,
    ) -> Result<Self, InterlockTrustError> {
        snapshot.validate()?;
        let head = InterlockTrustHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        if head != trusted_head {
            return Err(InterlockTrustError::TrustedHeadMismatch);
        }
        Ok(Self { snapshot, head })
    }

    /// Current externally retainable trust head.
    pub const fn head(&self) -> InterlockTrustHead {
        self.head
    }

    /// Read-only public snapshot for persistence/audit.
    pub fn snapshot(&self) -> &InterlockTrustSnapshotV1 {
        &self.snapshot
    }

    fn active_key(
        &self,
        controller_id: &str,
        bound_at_unix_ms: u64,
        now_unix_ms: u64,
    ) -> Result<&InterlockControllerKeyV1, InterlockTrustError> {
        if now_unix_ms < self.snapshot.issued_at_unix_ms
            || now_unix_ms >= self.snapshot.expires_at_unix_ms
        {
            return Err(InterlockTrustError::TrustSnapshotNotFresh);
        }
        let key = self
            .snapshot
            .keys
            .iter()
            .find(|key| {
                key.controller_id == controller_id
                    && key.status == InterlockControllerKeyStatus::Active
            })
            .ok_or(InterlockTrustError::NoActiveControllerKey)?;
        if !key.active_at(bound_at_unix_ms) || !key.active_at(now_unix_ms) {
            return Err(InterlockTrustError::ControllerKeyNotActive);
        }
        Ok(key)
    }
}

fn validate_successor(
    previous: &InterlockTrustSnapshotV1,
    next: &InterlockTrustSnapshotV1,
) -> Result<(), InterlockTrustError> {
    let previous_map = previous.key_map();
    let next_map = next.key_map();
    for (identity, old) in previous_map {
        let new = next_map
            .get(&identity)
            .ok_or(InterlockTrustError::TrustedKeyDeleted)?;
        if !old.immutable_identity_eq(new) {
            return Err(InterlockTrustError::TrustedKeyMutated);
        }
        if !old.status.transition_allowed(new.status) {
            return Err(InterlockTrustError::TrustedKeyReactivated);
        }
        if new.not_after_unix_ms > old.not_after_unix_ms {
            return Err(InterlockTrustError::TrustedKeyExpiryExtended);
        }
    }
    Ok(())
}

/// Provider boundary for the concrete controller evidence/signature format.
pub trait InterlockControllerEvidenceVerifier {
    /// Verify that `raw_evidence` authenticates the exact report digest under the
    /// supplied current controller key.
    fn verify_controller_evidence(
        &self,
        controller_id: &str,
        key_id: &str,
        algorithm: &str,
        public_key: &[u8],
        report_digest: Digest32,
        raw_evidence: &[u8],
    ) -> bool;
}

/// Opaque proof that one exact interlock report/evidence pair was re-verified under
/// a current anti-rollback controller key generation.
#[derive(Debug)]
pub struct VerifiedInterlockKeyBinding {
    controller_id: String,
    key_id: String,
    key_digest: Digest32,
    report_digest: Digest32,
    evidence_digest: Digest32,
    trust_head: InterlockTrustHead,
    bound_at_unix_ms: u64,
    verified_at_unix_ms: u64,
}

impl VerifiedInterlockKeyBinding {
    /// Controller identity verified by current trust.
    pub fn controller_id(&self) -> &str {
        &self.controller_id
    }

    /// Exact current key identity.
    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    /// Commitment to the trusted controller key record.
    pub const fn key_digest(&self) -> Digest32 {
        self.key_digest
    }

    /// Exact interlock report commitment.
    pub const fn report_digest(&self) -> Digest32 {
        self.report_digest
    }

    /// Exact raw hardware evidence commitment.
    pub const fn evidence_digest(&self) -> Digest32 {
        self.evidence_digest
    }

    /// Anti-rollback controller trust generation used for verification.
    pub const fn trust_head(&self) -> InterlockTrustHead {
        self.trust_head
    }

    /// Final-permit join time this key verification was explicitly bound to.
    pub const fn bound_at_unix_ms(&self) -> u64 {
        self.bound_at_unix_ms
    }

    /// Relying-party time at which current key trust was verified.
    pub const fn verified_at_unix_ms(&self) -> u64 {
        self.verified_at_unix_ms
    }
}

/// Re-verify exact interlock report/evidence commitments under current controller-key
/// trust before upgrading a final actuator permit.
pub fn verify_interlock_key_binding(
    registry: &InterlockTrustRegistry,
    controller_id: &str,
    report_digest: Digest32,
    evidence_digest: Digest32,
    raw_evidence: &[u8],
    bound_at_unix_ms: u64,
    now_unix_ms: u64,
    verifier: &impl InterlockControllerEvidenceVerifier,
) -> Result<VerifiedInterlockKeyBinding, InterlockTrustError> {
    if !valid_label(controller_id) {
        return Err(InterlockTrustError::InvalidKeyIdentity);
    }
    if report_digest == Digest32([0; 32]) || evidence_digest == Digest32([0; 32]) {
        return Err(InterlockTrustError::ZeroEvidenceCommitment);
    }
    if now_unix_ms < bound_at_unix_ms {
        return Err(InterlockTrustError::VerificationPredatesBinding);
    }
    if raw_evidence.is_empty() || raw_evidence.len() > MAX_PHYSICAL_INTERLOCK_EVIDENCE_BYTES {
        return Err(InterlockTrustError::EvidenceSizeOutOfBounds);
    }
    if Digest32(*blake3::hash(raw_evidence).as_bytes()) != evidence_digest {
        return Err(InterlockTrustError::EvidenceDigestMismatch);
    }

    let key = registry.active_key(controller_id, bound_at_unix_ms, now_unix_ms)?;
    let key_digest = key.digest()?;
    if !verifier.verify_controller_evidence(
        &key.controller_id,
        &key.key_id,
        &key.algorithm,
        &key.public_key,
        report_digest,
        raw_evidence,
    ) {
        return Err(InterlockTrustError::ControllerEvidenceVerificationFailed);
    }

    Ok(VerifiedInterlockKeyBinding {
        controller_id: key.controller_id.clone(),
        key_id: key.key_id.clone(),
        key_digest,
        report_digest,
        evidence_digest,
        trust_head: registry.head(),
        bound_at_unix_ms,
        verified_at_unix_ms: now_unix_ms,
    })
}

/// Stronger local permit whose exact hardware report/evidence commitments have been
/// verified under an explicit anti-rollback controller key generation.
#[derive(Debug)]
pub struct TrustBoundFinalActuatorPermit {
    permit: FinalActuatorPermit,
    interlock_key_id: String,
    interlock_key_digest: Digest32,
    interlock_trust_head: InterlockTrustHead,
    trust_verified_at_unix_ms: u64,
}

impl TrustBoundFinalActuatorPermit {
    /// Underlying exact physical command.
    pub fn command(&self) -> &symthaea_iot_authority::DeviceCommand {
        self.permit.command()
    }

    /// Exact common physical-effect envelope commitment.
    pub const fn envelope_digest(&self) -> Digest32 {
        self.permit.envelope_digest()
    }

    /// Current controller key identity used for the explicit trust upgrade.
    pub fn interlock_key_id(&self) -> &str {
        &self.interlock_key_id
    }

    /// Commitment to the exact controller key record.
    pub const fn interlock_key_digest(&self) -> Digest32 {
        self.interlock_key_digest
    }

    /// Anti-rollback controller trust generation used by this stronger permit.
    pub const fn interlock_trust_head(&self) -> InterlockTrustHead {
        self.interlock_trust_head
    }

    /// Relying-party time at which controller-key trust was verified.
    pub const fn trust_verified_at_unix_ms(&self) -> u64 {
        self.trust_verified_at_unix_ms
    }

    /// Inclusive latest millisecond inherited from the underlying final permit.
    pub const fn must_dispatch_by_unix_ms(&self) -> u64 {
        self.permit.must_dispatch_by_unix_ms()
    }
}

/// Consume a final actuator permit and matching explicit controller-key proof.
pub fn upgrade_final_actuator_permit(
    permit: FinalActuatorPermit,
    binding: VerifiedInterlockKeyBinding,
    now_unix_ms: u64,
) -> Result<TrustBoundFinalActuatorPermit, InterlockTrustError> {
    if binding.controller_id != permit.interlock_controller_id() {
        return Err(InterlockTrustError::PermitControllerMismatch);
    }
    if binding.report_digest != permit.interlock_report_digest() {
        return Err(InterlockTrustError::PermitReportMismatch);
    }
    if binding.evidence_digest != permit.interlock_evidence_digest() {
        return Err(InterlockTrustError::PermitEvidenceMismatch);
    }
    if binding.bound_at_unix_ms != permit.joined_at_unix_ms() {
        return Err(InterlockTrustError::PermitBindingTimeMismatch);
    }
    if binding.verified_at_unix_ms < permit.joined_at_unix_ms()
        || now_unix_ms < binding.verified_at_unix_ms
    {
        return Err(InterlockTrustError::PermitTrustTimeInvalid);
    }
    if now_unix_ms > permit.must_dispatch_by_unix_ms() {
        return Err(InterlockTrustError::PermitDispatchWindowElapsed);
    }

    Ok(TrustBoundFinalActuatorPermit {
        permit,
        interlock_key_id: binding.key_id,
        interlock_key_digest: binding.key_digest,
        interlock_trust_head: binding.trust_head,
        trust_verified_at_unix_ms: binding.verified_at_unix_ms,
    })
}

fn valid_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_INTERLOCK_TRUST_LABEL_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn update_string(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u32).to_be_bytes());
    h.update(value.as_bytes());
}

/// Fail-closed controller-trust and final-permit upgrade errors.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum InterlockTrustError {
    /// Unknown trust snapshot schema.
    #[error("unsupported interlock controller trust schema")]
    UnsupportedTrustSchema,
    /// Key/controller/algorithm identity is malformed.
    #[error("interlock controller key identity is invalid")]
    InvalidKeyIdentity,
    /// Public-key encoding is empty, oversized, or all zero.
    #[error("interlock controller public key is invalid")]
    InvalidPublicKey,
    /// Key validity interval is malformed.
    #[error("interlock controller key validity window is invalid")]
    InvalidKeyWindow,
    /// Trust generation zero is invalid.
    #[error("interlock trust sequence zero is invalid")]
    TrustSequenceZero,
    /// Trust snapshot validity interval is malformed.
    #[error("interlock trust snapshot validity window is invalid")]
    InvalidTrustWindow,
    /// Trust snapshot contains an invalid number of keys.
    #[error("interlock trust snapshot key count is invalid")]
    InvalidTrustKeyCount,
    /// Genesis unexpectedly carries a predecessor.
    #[error("interlock trust genesis must not have a predecessor")]
    GenesisHasPredecessor,
    /// Non-genesis snapshot omitted its predecessor commitment.
    #[error("interlock trust successor is missing predecessor commitment")]
    SuccessorMissingPredecessor,
    /// Duplicate `(controller_id,key_id)` identity.
    #[error("duplicate interlock controller key identity")]
    DuplicateKeyIdentity,
    /// More than one key is marked active for one controller.
    #[error("multiple active interlock keys exist for one controller")]
    MultipleActiveKeysForController,
    /// Snapshot is not generation one.
    #[error("snapshot is not interlock trust genesis")]
    NotGenesis,
    /// Trust sequence overflowed.
    #[error("interlock trust sequence overflow")]
    TrustSequenceOverflow,
    /// Successor sequence is not immediate.
    #[error("interlock trust sequence is not next: expected {expected}, proposed {proposed}")]
    TrustSequenceNotNext { expected: u64, proposed: u64 },
    /// Successor predecessor commitment does not match current trusted head.
    #[error("interlock trust predecessor mismatch")]
    TrustPredecessorMismatch,
    /// Successor issue time regressed.
    #[error("interlock trust issue time regressed")]
    TrustIssuedAtRegressed,
    /// Persisted snapshot does not match independently retained head.
    #[error("interlock trusted-head mismatch")]
    TrustedHeadMismatch,
    /// Existing key identity disappeared from a successor generation.
    #[error("existing interlock controller key identity was deleted")]
    TrustedKeyDeleted,
    /// Existing key material/algorithm/start-time mutated under the same key ID.
    #[error("interlock controller key mutated under the same key id")]
    TrustedKeyMutated,
    /// Retired/revoked key attempted to become active again.
    #[error("interlock controller key lifecycle attempted reactivation")]
    TrustedKeyReactivated,
    /// Existing key expiry was extended under the same key ID.
    #[error("interlock controller key expiry was extended")]
    TrustedKeyExpiryExtended,
    /// Current trust snapshot is not fresh.
    #[error("interlock trust snapshot is not fresh")]
    TrustSnapshotNotFresh,
    /// No current active key exists for the requested controller.
    #[error("no active trusted key exists for interlock controller")]
    NoActiveControllerKey,
    /// Current controller key was not active at both required times.
    #[error("interlock controller key is not active at required times")]
    ControllerKeyNotActive,
    /// Report/evidence commitment is zero.
    #[error("interlock key binding contains a zero evidence commitment")]
    ZeroEvidenceCommitment,
    /// Key verification time predates the final-permit binding time.
    #[error("interlock key verification predates binding time")]
    VerificationPredatesBinding,
    /// Raw hardware evidence is empty or too large.
    #[error("interlock key evidence size is outside accepted bounds")]
    EvidenceSizeOutOfBounds,
    /// Raw evidence does not match the final permit's committed digest.
    #[error("interlock key evidence digest mismatch")]
    EvidenceDigestMismatch,
    /// Concrete key/evidence provider rejected the exact report/evidence pair.
    #[error("interlock controller evidence verification failed")]
    ControllerEvidenceVerificationFailed,
    /// Key binding belongs to another controller.
    #[error("interlock key binding controller does not match final permit")]
    PermitControllerMismatch,
    /// Key binding authenticates another report.
    #[error("interlock key binding report does not match final permit")]
    PermitReportMismatch,
    /// Key binding authenticates another raw hardware evidence blob.
    #[error("interlock key binding evidence does not match final permit")]
    PermitEvidenceMismatch,
    /// Key proof was not explicitly bound to the final permit join time.
    #[error("interlock key binding time does not match final permit")]
    PermitBindingTimeMismatch,
    /// Key trust verification time is inconsistent with final permit time.
    #[error("interlock key verification time is invalid for final permit")]
    PermitTrustTimeInvalid,
    /// Underlying final permit's dispatch window elapsed.
    #[error("final actuator permit dispatch window elapsed")]
    PermitDispatchWindowElapsed,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AcceptVerifier;

    impl InterlockControllerEvidenceVerifier for AcceptVerifier {
        fn verify_controller_evidence(
            &self,
            _controller_id: &str,
            _key_id: &str,
            _algorithm: &str,
            _public_key: &[u8],
            _report_digest: Digest32,
            raw_evidence: &[u8],
        ) -> bool {
            !raw_evidence.is_empty()
        }
    }

    struct RejectVerifier;

    impl InterlockControllerEvidenceVerifier for RejectVerifier {
        fn verify_controller_evidence(
            &self,
            _controller_id: &str,
            _key_id: &str,
            _algorithm: &str,
            _public_key: &[u8],
            _report_digest: Digest32,
            _raw_evidence: &[u8],
        ) -> bool {
            false
        }
    }

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn key(status: InterlockControllerKeyStatus) -> InterlockControllerKeyV1 {
        InterlockControllerKeyV1 {
            controller_id: "safety-plc:field-a".into(),
            key_id: "plc-key-1".into(),
            algorithm: "vendor-signature-v1".into(),
            public_key: vec![0x42; 64],
            status,
            not_before_unix_ms: 100_000,
            not_after_unix_ms: 130_000,
        }
    }

    fn registry() -> InterlockTrustRegistry {
        InterlockTrustRegistry::genesis(InterlockTrustSnapshotV1 {
            schema_version: INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_ms: 100_000,
            expires_at_unix_ms: 130_000,
            previous_snapshot_digest: None,
            keys: vec![key(InterlockControllerKeyStatus::Active)],
        })
        .unwrap()
    }

    #[test]
    fn exact_report_evidence_mints_current_key_binding() {
        let raw = b"controller-attestation";
        let evidence_digest = Digest32(*blake3::hash(raw).as_bytes());
        let binding = verify_interlock_key_binding(
            &registry(),
            "safety-plc:field-a",
            d(7),
            evidence_digest,
            raw,
            113_200,
            113_300,
            &AcceptVerifier,
        )
        .unwrap();
        assert_eq!(binding.key_id(), "plc-key-1");
        assert_eq!(binding.report_digest(), d(7));
        assert_eq!(binding.evidence_digest(), evidence_digest);
        assert_eq!(binding.trust_head().sequence, 1);
    }

    #[test]
    fn raw_evidence_substitution_is_rejected_before_provider() {
        let committed = b"controller-attestation";
        let altered = b"controller-attestation-altered";
        let evidence_digest = Digest32(*blake3::hash(committed).as_bytes());
        assert!(matches!(
            verify_interlock_key_binding(
                &registry(),
                "safety-plc:field-a",
                d(7),
                evidence_digest,
                altered,
                113_200,
                113_300,
                &AcceptVerifier,
            ),
            Err(InterlockTrustError::EvidenceDigestMismatch)
        ));
    }

    #[test]
    fn provider_rejection_fails_closed() {
        let raw = b"controller-attestation";
        let evidence_digest = Digest32(*blake3::hash(raw).as_bytes());
        assert!(matches!(
            verify_interlock_key_binding(
                &registry(),
                "safety-plc:field-a",
                d(7),
                evidence_digest,
                raw,
                113_200,
                113_300,
                &RejectVerifier,
            ),
            Err(InterlockTrustError::ControllerEvidenceVerificationFailed)
        ));
    }

    #[test]
    fn retired_key_cannot_reactivate() {
        let base = registry();
        let snapshot2 = InterlockTrustSnapshotV1 {
            schema_version: INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 2,
            issued_at_unix_ms: 101_000,
            expires_at_unix_ms: 130_000,
            previous_snapshot_digest: Some(base.head().digest),
            keys: vec![InterlockControllerKeyV1 {
                status: InterlockControllerKeyStatus::Retired,
                not_after_unix_ms: 125_000,
                ..key(InterlockControllerKeyStatus::Active)
            }],
        };
        let retired = base.successor(snapshot2).unwrap();
        let snapshot3 = InterlockTrustSnapshotV1 {
            schema_version: INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 3,
            issued_at_unix_ms: 102_000,
            expires_at_unix_ms: 130_000,
            previous_snapshot_digest: Some(retired.head().digest),
            keys: vec![InterlockControllerKeyV1 {
                status: InterlockControllerKeyStatus::Active,
                not_after_unix_ms: 125_000,
                ..key(InterlockControllerKeyStatus::Active)
            }],
        };
        assert!(matches!(
            retired.successor(snapshot3),
            Err(InterlockTrustError::TrustedKeyReactivated)
        ));
    }

    #[test]
    fn rotation_requires_new_key_identity() {
        let base = registry();
        let old = InterlockControllerKeyV1 {
            status: InterlockControllerKeyStatus::Retired,
            not_after_unix_ms: 120_000,
            ..key(InterlockControllerKeyStatus::Active)
        };
        let new = InterlockControllerKeyV1 {
            controller_id: "safety-plc:field-a".into(),
            key_id: "plc-key-2".into(),
            algorithm: "vendor-signature-v1".into(),
            public_key: vec![0x55; 64],
            status: InterlockControllerKeyStatus::Active,
            not_before_unix_ms: 110_000,
            not_after_unix_ms: 140_000,
        };
        let snapshot2 = InterlockTrustSnapshotV1 {
            schema_version: INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 2,
            issued_at_unix_ms: 110_000,
            expires_at_unix_ms: 135_000,
            previous_snapshot_digest: Some(base.head().digest),
            keys: vec![old, new],
        };
        let rotated = base.successor(snapshot2).unwrap();
        let raw = b"controller-attestation";
        let binding = verify_interlock_key_binding(
            &rotated,
            "safety-plc:field-a",
            d(9),
            Digest32(*blake3::hash(raw).as_bytes()),
            raw,
            113_000,
            113_500,
            &AcceptVerifier,
        )
        .unwrap();
        assert_eq!(binding.key_id(), "plc-key-2");
    }
}
