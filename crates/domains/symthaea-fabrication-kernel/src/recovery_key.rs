// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Offline recovery-key sets and bounded break-glass activation ceremonies.
//!
//! Recovery authority is intentionally narrower than ordinary release or
//! machine authority. A recovery activation is one-time, incident-bound,
//! target-bound, short-lived, and requires an independently described key set
//! in addition to the generic threshold ceremony.

use crate::attestation::SignatureAlgorithm;
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::threshold::VerifiedThresholdCeremony;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const RECOVERY_KEY_SET_SCHEMA: &str = "symthaea.fabrication.recovery-key-set.v1";
pub const RECOVERY_ACTIVATION_SCHEMA: &str = "symthaea.fabrication.recovery-activation.v1";
pub const MAX_RECOVERY_PARTICIPANTS: usize = 32;
pub const MAX_RECOVERY_ID_BYTES: usize = 128;
pub const MAX_RECOVERY_REASON_BYTES: usize = 4 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecoveryScope {
    RestoreTrustSnapshot,
    UnlockGatewayState,
    ReissueMembership,
    RevokeCompromisedKeys,
    AuthorizeRollback,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryParticipant {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub custodian_id: String,
    pub region: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryKeySet {
    pub schema_version: String,
    pub key_set_id: String,
    pub generation: u64,
    pub valid_from_unix_s: u64,
    pub valid_until_unix_s: u64,
    pub minimum_distinct_signers: usize,
    pub minimum_distinct_regions: usize,
    pub allowed_scopes: BTreeSet<RecoveryScope>,
    pub participants: Vec<RecoveryParticipant>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryActivationRequest {
    pub schema_version: String,
    pub key_set_digest: Sha256Digest,
    pub scope: RecoveryScope,
    pub target_digest: Sha256Digest,
    pub incident_digest: Sha256Digest,
    pub nonce: Sha256Digest,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizedRecoveryActivation {
    pub request: RecoveryActivationRequest,
    pub request_digest: Sha256Digest,
    pub key_set_id: String,
    pub key_set_generation: u64,
    pub ceremony_digest: Sha256Digest,
    pub trust_snapshot_digest: Sha256Digest,
    pub signers: Vec<(SignatureAlgorithm, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryActivationPolicy {
    pub maximum_activation_lifetime_s: u64,
    pub require_algorithm_diversity: bool,
}

impl Default for RecoveryActivationPolicy {
    fn default() -> Self {
        Self {
            maximum_activation_lifetime_s: 30 * 60,
            require_algorithm_diversity: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryKeyError {
    UnsupportedSchema,
    InvalidKeySetId,
    GenerationZero,
    InvalidValidityWindow,
    EmptyParticipants,
    TooManyParticipants { actual: usize, maximum: usize },
    InvalidQuorum,
    EmptyScopes,
    InvalidScope,
    InvalidParticipant(String),
    DuplicateSigner(String),
    DuplicateCustodian(String),
    InvalidRequestWindow,
    ActivationTooLong,
    KeySetNotValid,
    KeySetDigestMismatch,
    ScopeNotAllowed,
    ZeroTargetDigest,
    ZeroIncidentDigest,
    ZeroNonce,
    InvalidReason,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    UnknownCeremonySigner(String),
    InsufficientSigners { actual: usize, required: usize },
    InsufficientRegions { actual: usize, required: usize },
    AlgorithmDiversityMissing,
    Encoding(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryActivationTracker {
    latest_key_set_generation: Option<u64>,
    consumed_nonces: BTreeSet<Sha256Digest>,
    accepted_requests: BTreeMap<Sha256Digest, Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryTrackingError {
    InvalidTrackerState,
    Encoding(String),
    KeySetGenerationRollback { latest: u64, proposed: u64 },
    NonceReplay,
    RequestSubstitution,
}

impl RecoveryKeySet {
    pub fn new(
        key_set_id: impl Into<String>,
        generation: u64,
        valid_from_unix_s: u64,
        valid_until_unix_s: u64,
        minimum_distinct_signers: usize,
        minimum_distinct_regions: usize,
        allowed_scopes: BTreeSet<RecoveryScope>,
        mut participants: Vec<RecoveryParticipant>,
    ) -> Result<Self, RecoveryKeyError> {
        participants.sort_by(|left, right| {
            (&left.algorithm, left.key_id.as_str()).cmp(&(&right.algorithm, right.key_id.as_str()))
        });
        let value = Self {
            schema_version: RECOVERY_KEY_SET_SCHEMA.into(),
            key_set_id: key_set_id.into(),
            generation,
            valid_from_unix_s,
            valid_until_unix_s,
            minimum_distinct_signers,
            minimum_distinct_regions,
            allowed_scopes,
            participants,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), RecoveryKeyError> {
        if self.schema_version != RECOVERY_KEY_SET_SCHEMA {
            return Err(RecoveryKeyError::UnsupportedSchema);
        }
        validate_token(&self.key_set_id).map_err(|_| RecoveryKeyError::InvalidKeySetId)?;
        if self.generation == 0 {
            return Err(RecoveryKeyError::GenerationZero);
        }
        if self.valid_from_unix_s >= self.valid_until_unix_s {
            return Err(RecoveryKeyError::InvalidValidityWindow);
        }
        if self.participants.is_empty() {
            return Err(RecoveryKeyError::EmptyParticipants);
        }
        if self.participants.len() > MAX_RECOVERY_PARTICIPANTS {
            return Err(RecoveryKeyError::TooManyParticipants {
                actual: self.participants.len(),
                maximum: MAX_RECOVERY_PARTICIPANTS,
            });
        }
        if self.minimum_distinct_signers == 0
            || self.minimum_distinct_signers > self.participants.len()
            || self.minimum_distinct_regions == 0
            || self.minimum_distinct_regions > self.participants.len()
        {
            return Err(RecoveryKeyError::InvalidQuorum);
        }
        if self.allowed_scopes.is_empty() {
            return Err(RecoveryKeyError::EmptyScopes);
        }
        for scope in &self.allowed_scopes {
            validate_scope(scope)?;
        }
        let mut signers = BTreeSet::new();
        let mut custodians = BTreeSet::new();
        let mut previous = None::<(SignatureAlgorithm, String)>;
        for participant in &self.participants {
            if !participant.algorithm.is_canonical()
                || validate_token(&participant.key_id).is_err()
                || validate_token(&participant.custodian_id).is_err()
                || validate_token(&participant.region).is_err()
            {
                return Err(RecoveryKeyError::InvalidParticipant(
                    participant.key_id.clone(),
                ));
            }
            let identity = (participant.algorithm.clone(), participant.key_id.clone());
            if !signers.insert(identity.clone()) {
                return Err(RecoveryKeyError::DuplicateSigner(
                    participant.key_id.clone(),
                ));
            }
            if !custodians.insert(participant.custodian_id.clone()) {
                return Err(RecoveryKeyError::DuplicateCustodian(
                    participant.custodian_id.clone(),
                ));
            }
            if previous.as_ref().is_some_and(|value| value >= &identity) {
                return Err(RecoveryKeyError::DuplicateSigner(
                    participant.key_id.clone(),
                ));
            }
            previous = Some(identity);
        }
        let regions = self
            .participants
            .iter()
            .map(|participant| participant.region.as_str())
            .collect::<BTreeSet<_>>();
        if regions.len() < self.minimum_distinct_regions {
            return Err(RecoveryKeyError::InvalidQuorum);
        }
        Ok(())
    }
}

impl RecoveryActivationRequest {
    pub fn validate(
        &self,
        key_set: &RecoveryKeySet,
        policy: &RecoveryActivationPolicy,
        now_unix_s: u64,
    ) -> Result<(), RecoveryKeyError> {
        if self.schema_version != RECOVERY_ACTIVATION_SCHEMA {
            return Err(RecoveryKeyError::UnsupportedSchema);
        }
        key_set.validate()?;
        if now_unix_s < key_set.valid_from_unix_s || now_unix_s >= key_set.valid_until_unix_s {
            return Err(RecoveryKeyError::KeySetNotValid);
        }
        if self.key_set_digest != digest_recovery_key_set(key_set)? {
            return Err(RecoveryKeyError::KeySetDigestMismatch);
        }
        validate_scope(&self.scope)?;
        if !key_set.allowed_scopes.contains(&self.scope) {
            return Err(RecoveryKeyError::ScopeNotAllowed);
        }
        if self.target_digest.0 == [0; 32] {
            return Err(RecoveryKeyError::ZeroTargetDigest);
        }
        if self.incident_digest.0 == [0; 32] {
            return Err(RecoveryKeyError::ZeroIncidentDigest);
        }
        if self.nonce.0 == [0; 32] {
            return Err(RecoveryKeyError::ZeroNonce);
        }
        if self.issued_at_unix_s > now_unix_s
            || now_unix_s >= self.expires_at_unix_s
            || self.issued_at_unix_s >= self.expires_at_unix_s
        {
            return Err(RecoveryKeyError::InvalidRequestWindow);
        }
        if self.expires_at_unix_s - self.issued_at_unix_s > policy.maximum_activation_lifetime_s {
            return Err(RecoveryKeyError::ActivationTooLong);
        }
        if self.reason.trim().is_empty()
            || self.reason != self.reason.trim()
            || self.reason.len() > MAX_RECOVERY_REASON_BYTES
            || self.reason.chars().any(char::is_control)
        {
            return Err(RecoveryKeyError::InvalidReason);
        }
        Ok(())
    }
}

impl RecoveryActivationTracker {
    pub fn validate(&self) -> Result<(), RecoveryTrackingError> {
        if self.latest_key_set_generation == Some(0)
            || self.consumed_nonces.iter().any(|nonce| nonce.0 == [0; 32])
            || self
                .accepted_requests
                .values()
                .any(|digest| digest.0 == [0; 32])
            || self.consumed_nonces != self.accepted_requests.keys().copied().collect()
            || (self.latest_key_set_generation.is_none() && !self.consumed_nonces.is_empty())
        {
            return Err(RecoveryTrackingError::InvalidTrackerState);
        }
        Ok(())
    }

    pub fn accept(
        &mut self,
        activation: &AuthorizedRecoveryActivation,
    ) -> Result<(), RecoveryTrackingError> {
        self.validate()?;
        if let Some(latest) = self.latest_key_set_generation {
            if activation.key_set_generation < latest {
                return Err(RecoveryTrackingError::KeySetGenerationRollback {
                    latest,
                    proposed: activation.key_set_generation,
                });
            }
        }
        if self.consumed_nonces.contains(&activation.request.nonce) {
            if self.accepted_requests.get(&activation.request.nonce)
                == Some(&activation.request_digest)
            {
                return Ok(());
            }
            return Err(RecoveryTrackingError::NonceReplay);
        }
        if self
            .accepted_requests
            .values()
            .any(|digest| *digest == activation.request_digest)
        {
            return Err(RecoveryTrackingError::RequestSubstitution);
        }
        self.latest_key_set_generation = Some(activation.key_set_generation);
        self.consumed_nonces.insert(activation.request.nonce);
        self.accepted_requests
            .insert(activation.request.nonce, activation.request_digest);
        Ok(())
    }

    pub fn latest_key_set_generation(&self) -> Option<u64> {
        self.latest_key_set_generation
    }
    pub fn consumed_nonces(&self) -> &BTreeSet<Sha256Digest> {
        &self.consumed_nonces
    }
}

pub fn digest_recovery_activation_tracker(
    tracker: &RecoveryActivationTracker,
) -> Result<Sha256Digest, RecoveryTrackingError> {
    tracker.validate()?;
    let bytes = serde_json::to_vec(tracker)
        .map_err(|error| RecoveryTrackingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.recovery-activation-tracker.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn digest_recovery_key_set(key_set: &RecoveryKeySet) -> Result<Sha256Digest, RecoveryKeyError> {
    key_set.validate()?;
    let bytes = serde_json::to_vec(key_set)
        .map_err(|error| RecoveryKeyError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.recovery-key-set-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn digest_recovery_activation_request(
    request: &RecoveryActivationRequest,
    key_set: &RecoveryKeySet,
    policy: &RecoveryActivationPolicy,
    now_unix_s: u64,
) -> Result<Sha256Digest, RecoveryKeyError> {
    request.validate(key_set, policy, now_unix_s)?;
    let bytes = serde_json::to_vec(request)
        .map_err(|error| RecoveryKeyError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.recovery-activation-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_recovery_activation(
    request: RecoveryActivationRequest,
    key_set: &RecoveryKeySet,
    policy: &RecoveryActivationPolicy,
    now_unix_s: u64,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedRecoveryActivation, RecoveryKeyError> {
    let request_digest = digest_recovery_activation_request(&request, key_set, policy, now_unix_s)?;
    if ceremony.purpose() != "recovery-key-activation" {
        return Err(RecoveryKeyError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != request_digest {
        return Err(RecoveryKeyError::CeremonyPayloadMismatch);
    }
    let participant_map = key_set
        .participants
        .iter()
        .map(|participant| {
            (
                (participant.algorithm.clone(), participant.key_id.clone()),
                participant,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut regions = BTreeSet::new();
    let mut algorithms = BTreeSet::new();
    for signer in ceremony.signers() {
        let Some(participant) = participant_map.get(signer) else {
            return Err(RecoveryKeyError::UnknownCeremonySigner(signer.1.clone()));
        };
        regions.insert(participant.region.clone());
        algorithms.insert(participant.algorithm.clone());
    }
    if ceremony.signers().len() < key_set.minimum_distinct_signers {
        return Err(RecoveryKeyError::InsufficientSigners {
            actual: ceremony.signers().len(),
            required: key_set.minimum_distinct_signers,
        });
    }
    if regions.len() < key_set.minimum_distinct_regions {
        return Err(RecoveryKeyError::InsufficientRegions {
            actual: regions.len(),
            required: key_set.minimum_distinct_regions,
        });
    }
    if policy.require_algorithm_diversity && algorithms.len() < 2 {
        return Err(RecoveryKeyError::AlgorithmDiversityMissing);
    }
    Ok(AuthorizedRecoveryActivation {
        request,
        request_digest,
        key_set_id: key_set.key_set_id.clone(),
        key_set_generation: key_set.generation,
        ceremony_digest: ceremony.ceremony_digest(),
        trust_snapshot_digest: ceremony.trust_snapshot_digest(),
        signers: ceremony.signers().to_vec(),
    })
}

fn validate_scope(scope: &RecoveryScope) -> Result<(), RecoveryKeyError> {
    if let RecoveryScope::Other(value) = scope {
        validate_token(value).map_err(|_| RecoveryKeyError::InvalidScope)?;
    }
    Ok(())
}

fn validate_token(value: &str) -> Result<(), ()> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_RECOVERY_ID_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    fn key_set() -> RecoveryKeySet {
        RecoveryKeySet::new(
            "offline-2026",
            3,
            10,
            1_000,
            2,
            2,
            BTreeSet::from([RecoveryScope::RestoreTrustSnapshot]),
            vec![
                RecoveryParticipant {
                    algorithm: SignatureAlgorithm::Ed25519,
                    key_id: "recovery-a".into(),
                    custodian_id: "custodian-a".into(),
                    region: "af-south".into(),
                },
                RecoveryParticipant {
                    algorithm: SignatureAlgorithm::MlDsa65,
                    key_id: "recovery-b".into(),
                    custodian_id: "custodian-b".into(),
                    region: "eu-west".into(),
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn request_is_exactly_keyset_incident_target_and_nonce_bound() {
        let set = key_set();
        let request = RecoveryActivationRequest {
            schema_version: RECOVERY_ACTIVATION_SCHEMA.into(),
            key_set_digest: digest_recovery_key_set(&set).unwrap(),
            scope: RecoveryScope::RestoreTrustSnapshot,
            target_digest: sha256(b"trust-7"),
            incident_digest: sha256(b"incident"),
            nonce: sha256(b"nonce"),
            issued_at_unix_s: 100,
            expires_at_unix_s: 200,
            reason: "restore the last certified trust snapshot".into(),
        };
        request
            .validate(&set, &RecoveryActivationPolicy::default(), 150)
            .unwrap();
        let mut changed = request.clone();
        changed.key_set_digest = sha256(b"other");
        assert_eq!(
            changed.validate(&set, &RecoveryActivationPolicy::default(), 150),
            Err(RecoveryKeyError::KeySetDigestMismatch)
        );
    }
}
