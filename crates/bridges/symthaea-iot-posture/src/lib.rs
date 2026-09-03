// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! RATS-shaped relying-party boundary for cyber-physical device posture.
//!
//! This crate deliberately does not parse raw EAT/CBOR/TPM evidence. In the RATS
//! model, an Attester produces Evidence, a Verifier appraises it against reference
//! values and evidence policy, and a Relying Party consumes the resulting
//! Attestation Result. This crate is that relying-party boundary.
//!
//! Raw/self-asserted [`DeviceRuntimeState`] must not enter the product actuation
//! path. Instead a bounded, challenge-bound, signed [`DeviceAttestationResultV1`]
//! is authenticated against an anti-rollback verifier-key registry. Only then does
//! this crate mint opaque [`VerifiedDevicePosture`].
//!
//! The signature provider remains an explicit TCB boundary in v0.1. A concrete
//! COSE/EAT/Xenia/TPM verifier can implement [`AttestationResultSignatureVerifier`]
//! without changing downstream actuation APIs.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_authority::DeviceRuntimeState;
use thiserror::Error;

pub const POSTURE_CHALLENGE_SCHEMA_VERSION: u16 = 1;
pub const DEVICE_ATTESTATION_RESULT_SCHEMA_VERSION: u16 = 1;
pub const VERIFIER_TRUST_SNAPSHOT_SCHEMA_VERSION: u16 = 1;
pub const MAX_POSTURE_CHALLENGE_LIFETIME_S: u64 = 300;
pub const MAX_VERIFIER_RESULT_LIFETIME_S: u64 = 900;
pub const MAX_POSTURE_OBSERVATIONS: usize = 256;
pub const MAX_POSTURE_OBSERVATION_NAME_BYTES: usize = 128;
pub const MAX_VERIFIER_ID_BYTES: usize = 256;
pub const MAX_KEY_ID_BYTES: usize = 256;
pub const MAX_ALGORITHM_NAME_BYTES: usize = 128;
pub const MAX_ATTESTATION_SIGNATURE_BYTES: usize = 64 * 1024;
pub const MAX_VERIFIER_KEYS: usize = 4096;

const CHALLENGE_DOMAIN: &[u8] = b"symthaea-iot-posture-challenge-v1\0";
const RESULT_DOMAIN: &[u8] = b"symthaea-iot-attestation-result-v1\0";
const RESULT_SIGNATURE_DOMAIN: &[u8] = b"symthaea-iot-attestation-result-signature-v1\0";
const TRUST_SNAPSHOT_DOMAIN: &[u8] = b"symthaea-iot-verifier-trust-snapshot-v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostureChallengeV1 {
    pub schema_version: u16,
    pub nonce: [u8; 32],
    pub device: ResourceRef,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
}

impl PostureChallengeV1 {
    pub fn validate(&self) -> Result<(), PostureError> {
        if self.schema_version != POSTURE_CHALLENGE_SCHEMA_VERSION {
            return Err(PostureError::UnsupportedChallengeSchema);
        }
        if self.nonce == [0; 32] {
            return Err(PostureError::ZeroChallengeNonce);
        }
        let lifetime = self
            .expires_at_unix_s
            .checked_sub(self.issued_at_unix_s)
            .ok_or(PostureError::InvalidChallengeWindow)?;
        if lifetime == 0 || lifetime > MAX_POSTURE_CHALLENGE_LIFETIME_S {
            return Err(PostureError::InvalidChallengeWindow);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, PostureError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(CHALLENGE_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.nonce);
        update_string(&mut h, &self.device.0);
        h.update(&self.issued_at_unix_s.to_be_bytes());
        h.update(&self.expires_at_unix_s.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceAttestationResultBodyV1 {
    pub schema_version: u16,
    pub verifier_id: String,
    pub key_id: String,
    pub algorithm: String,
    pub device: ResourceRef,
    pub challenge_digest: Digest32,
    pub appraised_at_unix_s: u64,
    pub expires_at_unix_s: u64,
    pub evidence_digest: Digest32,
    pub reference_values_digest: Digest32,
    pub appraisal_policy_digest: Digest32,
    pub running_firmware: Digest32,
    pub last_accepted_sequence: Option<u64>,
    pub observations: BTreeMap<String, i64>,
}

impl DeviceAttestationResultBodyV1 {
    pub fn validate_structure(&self) -> Result<(), PostureError> {
        if self.schema_version != DEVICE_ATTESTATION_RESULT_SCHEMA_VERSION {
            return Err(PostureError::UnsupportedResultSchema);
        }
        validate_identifier(&self.verifier_id, MAX_VERIFIER_ID_BYTES)
            .map_err(|_| PostureError::InvalidVerifierId)?;
        validate_identifier(&self.key_id, MAX_KEY_ID_BYTES)
            .map_err(|_| PostureError::InvalidKeyId)?;
        validate_identifier(&self.algorithm, MAX_ALGORITHM_NAME_BYTES)
            .map_err(|_| PostureError::InvalidAlgorithm)?;
        let lifetime = self
            .expires_at_unix_s
            .checked_sub(self.appraised_at_unix_s)
            .ok_or(PostureError::InvalidResultWindow)?;
        if lifetime == 0 || lifetime > MAX_VERIFIER_RESULT_LIFETIME_S {
            return Err(PostureError::InvalidResultWindow);
        }
        if self.evidence_digest == Digest32([0; 32])
            || self.reference_values_digest == Digest32([0; 32])
            || self.appraisal_policy_digest == Digest32([0; 32])
            || self.running_firmware == Digest32([0; 32])
        {
            return Err(PostureError::ZeroSecurityDigest);
        }
        if self.observations.len() > MAX_POSTURE_OBSERVATIONS {
            return Err(PostureError::TooManyObservations {
                actual: self.observations.len(),
                maximum: MAX_POSTURE_OBSERVATIONS,
            });
        }
        for name in self.observations.keys() {
            if name.is_empty()
                || name.trim() != name
                || name.len() > MAX_POSTURE_OBSERVATION_NAME_BYTES
            {
                return Err(PostureError::InvalidObservationName(name.clone()));
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, PostureError> {
        self.validate_structure()?;
        let mut h = blake3::Hasher::new();
        h.update(RESULT_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        update_string(&mut h, &self.verifier_id);
        update_string(&mut h, &self.key_id);
        update_string(&mut h, &self.algorithm);
        update_string(&mut h, &self.device.0);
        update_digest(&mut h, self.challenge_digest);
        h.update(&self.appraised_at_unix_s.to_be_bytes());
        h.update(&self.expires_at_unix_s.to_be_bytes());
        update_digest(&mut h, self.evidence_digest);
        update_digest(&mut h, self.reference_values_digest);
        update_digest(&mut h, self.appraisal_policy_digest);
        update_digest(&mut h, self.running_firmware);
        match self.last_accepted_sequence {
            Some(value) => {
                h.update(&[1]);
                h.update(&value.to_be_bytes());
            }
            None => {
                h.update(&[0]);
            }
        }
        h.update(&(self.observations.len() as u64).to_be_bytes());
        for (name, value) in &self.observations {
            update_string(&mut h, name);
            h.update(&value.to_be_bytes());
        }
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    pub fn signature_message(&self) -> Result<Vec<u8>, PostureError> {
        let Digest32(digest) = self.digest()?;
        let mut message = Vec::with_capacity(RESULT_SIGNATURE_DOMAIN.len() + digest.len());
        message.extend_from_slice(RESULT_SIGNATURE_DOMAIN);
        message.extend_from_slice(&digest);
        Ok(message)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceAttestationResultV1 {
    pub body: DeviceAttestationResultBodyV1,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerifierKeyStatus {
    Active,
    Retired,
    Revoked,
}

impl VerifierKeyStatus {
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierKeyTrustV1 {
    pub verifier_id: String,
    pub key_id: String,
    pub algorithm: String,
    pub not_before_unix_s: u64,
    pub not_after_unix_s: Option<u64>,
    pub max_result_lifetime_s: u64,
    pub status: VerifierKeyStatus,
}

impl VerifierKeyTrustV1 {
    fn validate(&self) -> Result<(), PostureError> {
        validate_identifier(&self.verifier_id, MAX_VERIFIER_ID_BYTES)
            .map_err(|_| PostureError::InvalidVerifierId)?;
        validate_identifier(&self.key_id, MAX_KEY_ID_BYTES)
            .map_err(|_| PostureError::InvalidKeyId)?;
        validate_identifier(&self.algorithm, MAX_ALGORITHM_NAME_BYTES)
            .map_err(|_| PostureError::InvalidAlgorithm)?;
        if self
            .not_after_unix_s
            .is_some_and(|end| end <= self.not_before_unix_s)
        {
            return Err(PostureError::InvalidVerifierKeyWindow);
        }
        if self.max_result_lifetime_s == 0
            || self.max_result_lifetime_s > MAX_VERIFIER_RESULT_LIFETIME_S
        {
            return Err(PostureError::InvalidVerifierResultLifetime);
        }
        Ok(())
    }

    fn identity(&self) -> (&str, &str, &str) {
        (&self.verifier_id, &self.key_id, &self.algorithm)
    }

    fn active_at(&self, now_unix_s: u64) -> bool {
        self.status == VerifierKeyStatus::Active
            && now_unix_s >= self.not_before_unix_s
            && self
                .not_after_unix_s
                .is_none_or(|end| now_unix_s < end)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierTrustSnapshotV1 {
    pub schema_version: u16,
    pub sequence: u64,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
    pub previous_snapshot_digest: Option<Digest32>,
    pub keys: Vec<VerifierKeyTrustV1>,
}

impl VerifierTrustSnapshotV1 {
    pub fn validate(&self) -> Result<(), PostureError> {
        if self.schema_version != VERIFIER_TRUST_SNAPSHOT_SCHEMA_VERSION {
            return Err(PostureError::UnsupportedTrustSnapshotSchema);
        }
        if self.sequence == 0 {
            return Err(PostureError::TrustSequenceZero);
        }
        if self.issued_at_unix_s >= self.expires_at_unix_s {
            return Err(PostureError::InvalidTrustSnapshotWindow);
        }
        if self.keys.is_empty() {
            return Err(PostureError::EmptyTrustSnapshot);
        }
        if self.keys.len() > MAX_VERIFIER_KEYS {
            return Err(PostureError::TooManyVerifierKeys {
                actual: self.keys.len(),
                maximum: MAX_VERIFIER_KEYS,
            });
        }
        if self.sequence == 1 && self.previous_snapshot_digest.is_some() {
            return Err(PostureError::GenesisTrustHasPredecessor);
        }
        if self.sequence > 1 && self.previous_snapshot_digest.is_none() {
            return Err(PostureError::MissingTrustPredecessor);
        }
        let mut identities = BTreeSet::new();
        for key in &self.keys {
            key.validate()?;
            let identity = (
                key.verifier_id.clone(),
                key.key_id.clone(),
                key.algorithm.clone(),
            );
            if !identities.insert(identity) {
                return Err(PostureError::DuplicateVerifierKey);
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, PostureError> {
        self.validate()?;
        let mut keys = self.keys.iter().collect::<Vec<_>>();
        keys.sort_by(|a, b| a.identity().cmp(&b.identity()));
        let mut h = blake3::Hasher::new();
        h.update(TRUST_SNAPSHOT_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.sequence.to_be_bytes());
        h.update(&self.issued_at_unix_s.to_be_bytes());
        h.update(&self.expires_at_unix_s.to_be_bytes());
        match self.previous_snapshot_digest {
            Some(value) => {
                h.update(&[1]);
                update_digest(&mut h, value);
            }
            None => {
                h.update(&[0]);
            }
        }
        h.update(&(keys.len() as u64).to_be_bytes());
        for key in keys {
            update_string(&mut h, &key.verifier_id);
            update_string(&mut h, &key.key_id);
            update_string(&mut h, &key.algorithm);
            h.update(&key.not_before_unix_s.to_be_bytes());
            match key.not_after_unix_s {
                Some(end) => {
                    h.update(&[1]);
                    h.update(&end.to_be_bytes());
                }
                None => h.update(&[0]),
            }
            h.update(&key.max_result_lifetime_s.to_be_bytes());
            h.update(&[key.status.tag()]);
        }
        Ok(Digest32(*h.finalize().as_bytes()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierTrustHead {
    pub sequence: u64,
    pub digest: Digest32,
}

#[derive(Debug)]
pub struct VerifierTrustRegistry {
    snapshot: VerifierTrustSnapshotV1,
    head: VerifierTrustHead,
}

impl VerifierTrustRegistry {
    pub fn genesis(snapshot: VerifierTrustSnapshotV1) -> Result<Self, PostureError> {
        snapshot.validate()?;
        if snapshot.sequence != 1 || snapshot.previous_snapshot_digest.is_some() {
            return Err(PostureError::NotGenesisTrustSnapshot);
        }
        let head = VerifierTrustHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        Ok(Self { snapshot, head })
    }

    pub fn successor(&self, snapshot: VerifierTrustSnapshotV1) -> Result<Self, PostureError> {
        snapshot.validate()?;
        let expected = self
            .head
            .sequence
            .checked_add(1)
            .ok_or(PostureError::TrustSequenceOverflow)?;
        if snapshot.sequence != expected {
            return Err(PostureError::TrustSequenceNotNext {
                expected,
                proposed: snapshot.sequence,
            });
        }
        if snapshot.previous_snapshot_digest != Some(self.head.digest) {
            return Err(PostureError::TrustPredecessorMismatch);
        }
        if snapshot.issued_at_unix_s < self.snapshot.issued_at_unix_s {
            return Err(PostureError::TrustIssuedAtRegressed);
        }
        validate_key_successor(&self.snapshot, &snapshot)?;
        let head = VerifierTrustHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        Ok(Self { snapshot, head })
    }

    pub fn restore(
        snapshot: VerifierTrustSnapshotV1,
        trusted_head: VerifierTrustHead,
    ) -> Result<Self, PostureError> {
        snapshot.validate()?;
        let head = VerifierTrustHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        if head != trusted_head {
            return Err(PostureError::TrustedVerifierHeadMismatch);
        }
        Ok(Self { snapshot, head })
    }

    pub const fn head(&self) -> VerifierTrustHead {
        self.head
    }

    pub fn snapshot(&self) -> &VerifierTrustSnapshotV1 {
        &self.snapshot
    }

    fn key_for(
        &self,
        body: &DeviceAttestationResultBodyV1,
        now_unix_s: u64,
    ) -> Result<&VerifierKeyTrustV1, PostureError> {
        if now_unix_s < self.snapshot.issued_at_unix_s
            || now_unix_s >= self.snapshot.expires_at_unix_s
        {
            return Err(PostureError::VerifierTrustSnapshotNotFresh);
        }
        let key = self
            .snapshot
            .keys
            .iter()
            .find(|key| {
                key.verifier_id == body.verifier_id
                    && key.key_id == body.key_id
                    && key.algorithm == body.algorithm
            })
            .ok_or(PostureError::VerifierKeyUnknown)?;
        if !key.active_at(now_unix_s) || !key.active_at(body.appraised_at_unix_s) {
            return Err(PostureError::VerifierKeyNotActive);
        }
        Ok(key)
    }
}

pub trait AttestationResultSignatureVerifier {
    fn verify(
        &self,
        algorithm: &str,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

/// Opaque relying-party posture token. It cannot be deserialized, constructed, or
/// cloned by downstream callers.
#[derive(Debug)]
pub struct VerifiedDevicePosture {
    device: ResourceRef,
    runtime: DeviceRuntimeState,
    result_digest: Digest32,
    evidence_digest: Digest32,
    reference_values_digest: Digest32,
    appraisal_policy_digest: Digest32,
    challenge_digest: Digest32,
    verifier_id: String,
    trust_head: VerifierTrustHead,
    appraised_at_unix_s: u64,
    expires_at_unix_s: u64,
}

impl VerifiedDevicePosture {
    pub fn device(&self) -> &ResourceRef {
        &self.device
    }

    pub fn runtime_state(&self) -> &DeviceRuntimeState {
        &self.runtime
    }

    pub fn result_digest(&self) -> Digest32 {
        self.result_digest
    }

    pub fn evidence_digest(&self) -> Digest32 {
        self.evidence_digest
    }

    pub fn reference_values_digest(&self) -> Digest32 {
        self.reference_values_digest
    }

    pub fn appraisal_policy_digest(&self) -> Digest32 {
        self.appraisal_policy_digest
    }

    pub fn challenge_digest(&self) -> Digest32 {
        self.challenge_digest
    }

    pub fn verifier_id(&self) -> &str {
        &self.verifier_id
    }

    pub const fn trust_head(&self) -> VerifierTrustHead {
        self.trust_head
    }

    pub const fn appraised_at_unix_s(&self) -> u64 {
        self.appraised_at_unix_s
    }

    pub const fn expires_at_unix_s(&self) -> u64 {
        self.expires_at_unix_s
    }

    pub fn is_fresh_at(&self, now_unix_s: u64) -> bool {
        now_unix_s >= self.appraised_at_unix_s && now_unix_s < self.expires_at_unix_s
    }
}

pub fn verify_device_posture(
    result: DeviceAttestationResultV1,
    challenge: &PostureChallengeV1,
    trust: &VerifierTrustRegistry,
    now_unix_s: u64,
    verifier: &dyn AttestationResultSignatureVerifier,
) -> Result<VerifiedDevicePosture, PostureError> {
    challenge.validate()?;
    let challenge_digest = challenge.digest()?;
    if now_unix_s < challenge.issued_at_unix_s || now_unix_s >= challenge.expires_at_unix_s {
        return Err(PostureError::ChallengeNotFresh);
    }
    result.body.validate_structure()?;
    if result.signature.is_empty() {
        return Err(PostureError::EmptyAttestationSignature);
    }
    if result.signature.len() > MAX_ATTESTATION_SIGNATURE_BYTES {
        return Err(PostureError::AttestationSignatureTooLarge {
            actual: result.signature.len(),
            maximum: MAX_ATTESTATION_SIGNATURE_BYTES,
        });
    }
    if result.body.device != challenge.device {
        return Err(PostureError::ChallengeDeviceMismatch);
    }
    if result.body.challenge_digest != challenge_digest {
        return Err(PostureError::ChallengeBindingMismatch);
    }
    if result.body.appraised_at_unix_s < challenge.issued_at_unix_s
        || result.body.appraised_at_unix_s > now_unix_s
        || result.body.expires_at_unix_s > challenge.expires_at_unix_s
        || now_unix_s >= result.body.expires_at_unix_s
    {
        return Err(PostureError::ResultNotFreshForChallenge);
    }

    let trusted_key = trust.key_for(&result.body, now_unix_s)?;
    let lifetime = result.body.expires_at_unix_s - result.body.appraised_at_unix_s;
    if lifetime > trusted_key.max_result_lifetime_s {
        return Err(PostureError::ResultLifetimeExceedsVerifierPolicy {
            proposed: lifetime,
            maximum: trusted_key.max_result_lifetime_s,
        });
    }

    let message = result.body.signature_message()?;
    let valid = verifier
        .verify(
            &result.body.algorithm,
            &result.body.key_id,
            &message,
            &result.signature,
        )
        .map_err(PostureError::SignatureProvider)?;
    if !valid {
        return Err(PostureError::InvalidAttestationSignature);
    }

    let result_digest = result.body.digest()?;
    let runtime = DeviceRuntimeState {
        running_firmware: result.body.running_firmware,
        last_accepted_sequence: result.body.last_accepted_sequence,
        observations: result.body.observations,
    };

    Ok(VerifiedDevicePosture {
        device: result.body.device,
        runtime,
        result_digest,
        evidence_digest: result.body.evidence_digest,
        reference_values_digest: result.body.reference_values_digest,
        appraisal_policy_digest: result.body.appraisal_policy_digest,
        challenge_digest,
        verifier_id: result.body.verifier_id,
        trust_head: trust.head(),
        appraised_at_unix_s: result.body.appraised_at_unix_s,
        expires_at_unix_s: result.body.expires_at_unix_s,
    })
}

fn validate_key_successor(
    previous: &VerifierTrustSnapshotV1,
    current: &VerifierTrustSnapshotV1,
) -> Result<(), PostureError> {
    let current_map = current
        .keys
        .iter()
        .map(|key| (key.identity(), key))
        .collect::<BTreeMap<_, _>>();
    for old in &previous.keys {
        let Some(new) = current_map.get(&old.identity()) else {
            return Err(PostureError::VerifierKeyDeleted);
        };
        if old.not_before_unix_s != new.not_before_unix_s
            || old.not_after_unix_s != new.not_after_unix_s
            || old.max_result_lifetime_s != new.max_result_lifetime_s
        {
            return Err(PostureError::VerifierKeyMetadataChanged);
        }
        if !old.status.transition_allowed(new.status) {
            return Err(PostureError::VerifierKeyLifecycleRollback);
        }
    }
    Ok(())
}

fn validate_identifier(value: &str, maximum: usize) -> Result<(), ()> {
    if value.is_empty() || value.trim() != value || value.len() > maximum {
        return Err(());
    }
    Ok(())
}

fn update_string(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

fn update_digest(h: &mut blake3::Hasher, Digest32(value): Digest32) {
    h.update(&value);
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PostureError {
    #[error("unsupported posture challenge schema")]
    UnsupportedChallengeSchema,
    #[error("challenge nonce must be non-zero")]
    ZeroChallengeNonce,
    #[error("invalid posture challenge window")]
    InvalidChallengeWindow,
    #[error("posture challenge is not fresh")]
    ChallengeNotFresh,
    #[error("unsupported device attestation result schema")]
    UnsupportedResultSchema,
    #[error("invalid verifier id")]
    InvalidVerifierId,
    #[error("invalid verifier key id")]
    InvalidKeyId,
    #[error("invalid verifier algorithm")]
    InvalidAlgorithm,
    #[error("invalid attestation-result validity window")]
    InvalidResultWindow,
    #[error("attestation result contains a zero security digest")]
    ZeroSecurityDigest,
    #[error("too many posture observations: {actual} > {maximum}")]
    TooManyObservations { actual: usize, maximum: usize },
    #[error("invalid posture observation name: {0}")]
    InvalidObservationName(String),
    #[error("empty attestation-result signature")]
    EmptyAttestationSignature,
    #[error("attestation-result signature too large: {actual} > {maximum}")]
    AttestationSignatureTooLarge { actual: usize, maximum: usize },
    #[error("attestation result is for another challenge device")]
    ChallengeDeviceMismatch,
    #[error("attestation result does not bind the relying-party challenge")]
    ChallengeBindingMismatch,
    #[error("attestation result is stale or outside the challenge window")]
    ResultNotFreshForChallenge,
    #[error("unsupported verifier trust snapshot schema")]
    UnsupportedTrustSnapshotSchema,
    #[error("verifier trust snapshot sequence is zero")]
    TrustSequenceZero,
    #[error("invalid verifier trust snapshot window")]
    InvalidTrustSnapshotWindow,
    #[error("empty verifier trust snapshot")]
    EmptyTrustSnapshot,
    #[error("too many verifier keys: {actual} > {maximum}")]
    TooManyVerifierKeys { actual: usize, maximum: usize },
    #[error("genesis verifier trust snapshot has a predecessor")]
    GenesisTrustHasPredecessor,
    #[error("non-genesis verifier trust snapshot is missing predecessor")]
    MissingTrustPredecessor,
    #[error("duplicate verifier key identity")]
    DuplicateVerifierKey,
    #[error("invalid verifier key validity window")]
    InvalidVerifierKeyWindow,
    #[error("invalid verifier maximum result lifetime")]
    InvalidVerifierResultLifetime,
    #[error("snapshot is not verifier-trust genesis")]
    NotGenesisTrustSnapshot,
    #[error("verifier trust sequence overflow")]
    TrustSequenceOverflow,
    #[error("verifier trust sequence is not next: expected {expected}, got {proposed}")]
    TrustSequenceNotNext { expected: u64, proposed: u64 },
    #[error("verifier trust predecessor mismatch")]
    TrustPredecessorMismatch,
    #[error("verifier trust issued-at regressed")]
    TrustIssuedAtRegressed,
    #[error("trusted verifier head mismatch")]
    TrustedVerifierHeadMismatch,
    #[error("verifier trust snapshot is not fresh")]
    VerifierTrustSnapshotNotFresh,
    #[error("verifier key is unknown")]
    VerifierKeyUnknown,
    #[error("verifier key is not active for this appraisal/current time")]
    VerifierKeyNotActive,
    #[error("verifier key was deleted from successor trust state")]
    VerifierKeyDeleted,
    #[error("verifier key immutable metadata changed")]
    VerifierKeyMetadataChanged,
    #[error("verifier key lifecycle rolled backward")]
    VerifierKeyLifecycleRollback,
    #[error("attestation-result lifetime {proposed}s exceeds verifier policy {maximum}s")]
    ResultLifetimeExceedsVerifierPolicy { proposed: u64, maximum: u64 },
    #[error("attestation signature provider failed: {0}")]
    SignatureProvider(String),
    #[error("invalid attestation-result signature")]
    InvalidAttestationSignature,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestVerifier;

    impl AttestationResultSignatureVerifier for TestVerifier {
        fn verify(
            &self,
            _algorithm: &str,
            _key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(blake3::hash(message).as_bytes() == signature)
        }
    }

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn challenge() -> PostureChallengeV1 {
        PostureChallengeV1 {
            schema_version: POSTURE_CHALLENGE_SCHEMA_VERSION,
            nonce: [0xA5; 32],
            device: ResourceRef("iot:valve:72".into()),
            issued_at_unix_s: 4_900,
            expires_at_unix_s: 5_100,
        }
    }

    fn key(status: VerifierKeyStatus) -> VerifierKeyTrustV1 {
        VerifierKeyTrustV1 {
            verifier_id: "verifier:fleet-a".into(),
            key_id: "key-1".into(),
            algorithm: "test-blake3".into(),
            not_before_unix_s: 1_000,
            not_after_unix_s: Some(9_000),
            max_result_lifetime_s: 120,
            status,
        }
    }

    fn trust() -> VerifierTrustRegistry {
        VerifierTrustRegistry::genesis(VerifierTrustSnapshotV1 {
            schema_version: VERIFIER_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_s: 1_000,
            expires_at_unix_s: 9_000,
            previous_snapshot_digest: None,
            keys: vec![key(VerifierKeyStatus::Active)],
        })
        .unwrap()
    }

    fn signed_result(challenge: &PostureChallengeV1) -> DeviceAttestationResultV1 {
        let body = DeviceAttestationResultBodyV1 {
            schema_version: DEVICE_ATTESTATION_RESULT_SCHEMA_VERSION,
            verifier_id: "verifier:fleet-a".into(),
            key_id: "key-1".into(),
            algorithm: "test-blake3".into(),
            device: challenge.device.clone(),
            challenge_digest: challenge.digest().unwrap(),
            appraised_at_unix_s: 4_995,
            expires_at_unix_s: 5_050,
            evidence_digest: digest(1),
            reference_values_digest: digest(2),
            appraisal_policy_digest: digest(3),
            running_firmware: digest(7),
            last_accepted_sequence: Some(42),
            observations: BTreeMap::from([("tank_pressure_kpa_x100".into(), 210_000)]),
        };
        let signature = blake3::hash(&body.signature_message().unwrap())
            .as_bytes()
            .to_vec();
        DeviceAttestationResultV1 { body, signature }
    }

    #[test]
    fn valid_challenge_bound_result_mints_opaque_posture() {
        let challenge = challenge();
        let posture = verify_device_posture(
            signed_result(&challenge),
            &challenge,
            &trust(),
            5_000,
            &TestVerifier,
        )
        .unwrap();
        assert_eq!(posture.device(), &challenge.device);
        assert_eq!(posture.runtime_state().running_firmware, digest(7));
        assert_eq!(posture.runtime_state().last_accepted_sequence, Some(42));
        assert!(posture.is_fresh_at(5_000));
    }

    #[test]
    fn another_challenge_cannot_reuse_result() {
        let challenge = challenge();
        let result = signed_result(&challenge);
        let mut other = challenge.clone();
        other.nonce = [0xB6; 32];
        assert_eq!(
            verify_device_posture(result, &other, &trust(), 5_000, &TestVerifier),
            Err(PostureError::ChallengeBindingMismatch)
        );
    }

    #[test]
    fn revoked_verifier_key_dominates_valid_signature() {
        let challenge = challenge();
        let revoked = VerifierTrustRegistry::genesis(VerifierTrustSnapshotV1 {
            schema_version: VERIFIER_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_s: 1_000,
            expires_at_unix_s: 9_000,
            previous_snapshot_digest: None,
            keys: vec![key(VerifierKeyStatus::Revoked)],
        })
        .unwrap();
        assert_eq!(
            verify_device_posture(
                signed_result(&challenge),
                &challenge,
                &revoked,
                5_000,
                &TestVerifier,
            ),
            Err(PostureError::VerifierKeyNotActive)
        );
    }

    #[test]
    fn invalid_signature_fails_closed() {
        let challenge = challenge();
        let mut result = signed_result(&challenge);
        result.signature[0] ^= 0xFF;
        assert_eq!(
            verify_device_posture(result, &challenge, &trust(), 5_000, &TestVerifier),
            Err(PostureError::InvalidAttestationSignature)
        );
    }

    #[test]
    fn result_cannot_outlive_challenge() {
        let challenge = challenge();
        let mut result = signed_result(&challenge);
        result.body.expires_at_unix_s = challenge.expires_at_unix_s + 1;
        result.signature = blake3::hash(&result.body.signature_message().unwrap())
            .as_bytes()
            .to_vec();
        assert_eq!(
            verify_device_posture(result, &challenge, &trust(), 5_000, &TestVerifier),
            Err(PostureError::ResultNotFreshForChallenge)
        );
    }

    #[test]
    fn verifier_key_deletion_is_rollback() {
        let first = trust();
        let second = VerifierTrustSnapshotV1 {
            schema_version: VERIFIER_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 2,
            issued_at_unix_s: 2_000,
            expires_at_unix_s: 9_000,
            previous_snapshot_digest: Some(first.head().digest),
            keys: vec![VerifierKeyTrustV1 {
                verifier_id: "verifier:other".into(),
                key_id: "key-2".into(),
                algorithm: "test-blake3".into(),
                not_before_unix_s: 1_000,
                not_after_unix_s: Some(9_000),
                max_result_lifetime_s: 120,
                status: VerifierKeyStatus::Active,
            }],
        };
        assert_eq!(first.successor(second), Err(PostureError::VerifierKeyDeleted));
    }

    #[test]
    fn revocation_is_sticky() {
        let first = VerifierTrustRegistry::genesis(VerifierTrustSnapshotV1 {
            schema_version: VERIFIER_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_s: 1_000,
            expires_at_unix_s: 9_000,
            previous_snapshot_digest: None,
            keys: vec![key(VerifierKeyStatus::Revoked)],
        })
        .unwrap();
        let mut reactivated = key(VerifierKeyStatus::Active);
        reactivated.not_before_unix_s = 1_000;
        let next = VerifierTrustSnapshotV1 {
            schema_version: VERIFIER_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 2,
            issued_at_unix_s: 2_000,
            expires_at_unix_s: 9_000,
            previous_snapshot_digest: Some(first.head().digest),
            keys: vec![reactivated],
        };
        assert_eq!(
            first.successor(next),
            Err(PostureError::VerifierKeyLifecycleRollback)
        );
    }
}
