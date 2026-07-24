// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent transparency-head gossip and split-view detection.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{
    CheckpointPublicKeyId, CheckpointPublicSignature, CheckpointPublicSigningKey,
    CheckpointPublicVerificationError, CheckpointPublicVerifyingKey,
    CheckpointSignedTransparencyHead, CheckpointTransparencyConsistencyProof,
    CheckpointTransparencyLogId,
    MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES, MAX_CHECKPOINT_PUBLIC_SIGNERS,
};

pub const CHECKPOINT_TRANSPARENCY_GOSSIP_OBSERVER_SCHEMA: &str =
    "symthaea.checkpoint-transparency-gossip-observer.v1";
pub const CHECKPOINT_TRANSPARENCY_GOSSIP_POLICY_SCHEMA: &str =
    "symthaea.checkpoint-transparency-gossip-policy.v1";
pub const CHECKPOINT_TRANSPARENCY_GOSSIP_STATEMENT_SCHEMA: &str =
    "symthaea.checkpoint-transparency-gossip-statement.v1";
pub const CHECKPOINT_TRANSPARENCY_GOSSIP_BUNDLE_SCHEMA: &str =
    "symthaea.checkpoint-transparency-gossip-bundle.v1";
pub const CHECKPOINT_TRANSPARENCY_GOSSIP_SUMMARY_SCHEMA: &str =
    "symthaea.checkpoint-transparency-gossip-summary.v1";
pub const CHECKPOINT_TRANSPARENCY_SPLIT_VIEW_NEGATIVE_SCHEMA: &str =
    "symthaea.checkpoint-transparency-split-view-negative.v1";
pub const MAX_CHECKPOINT_TRANSPARENCY_GOSSIP_OBSERVATIONS: usize = 256;
pub const MAX_CHECKPOINT_TRANSPARENCY_GOSSIP_STATEMENT_AGE_SECONDS: u64 = 7 * 24 * 60 * 60;

const GOSSIP_POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea-checkpoint-transparency-gossip-policy-digest-v1\0";
const GOSSIP_STATEMENT_DIGEST_DOMAIN: &[u8] =
    b"symthaea-checkpoint-transparency-gossip-statement-digest-v1\0";
const GOSSIP_STATEMENT_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea-checkpoint-transparency-gossip-statement-signature-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointTransparencyOriginId(pub [u8; 16]);

impl CheckpointTransparencyOriginId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, CheckpointTransparencyGossipError> {
        if bytes == [0u8; 16] {
            return Err(CheckpointTransparencyGossipError::InvalidOrigin);
        }
        Ok(Self(bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointTransparencyGossipObserver {
    pub schema: String,
    pub observer_id: [u8; 16],
    pub organization_binding: [u8; 32],
    pub verifying_key: CheckpointPublicVerifyingKey,
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointTransparencyGossipObserver {
    pub fn validate(&self) -> Result<(), CheckpointTransparencyGossipError> {
        self.verifying_key.validate()?;
        if self.schema != CHECKPOINT_TRANSPARENCY_GOSSIP_OBSERVER_SCHEMA
            || self.observer_id == [0u8; 16]
            || self.organization_binding == [0u8; 32]
            || self.valid_from_unix_seconds == 0
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointTransparencyGossipError::InvalidObserver);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointTransparencyGossipPolicy {
    pub schema: String,
    pub policy_id: [u8; 16],
    pub log_id: CheckpointTransparencyLogId,
    pub transparency_authority_key_id: CheckpointPublicKeyId,
    pub observers: Vec<CheckpointTransparencyGossipObserver>,
    pub threshold: u16,
    pub minimum_organizations: u16,
    pub maximum_statement_age_seconds: u64,
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointTransparencyGossipPolicy {
    pub fn validate(&self) -> Result<(), CheckpointTransparencyGossipError> {
        if self.schema != CHECKPOINT_TRANSPARENCY_GOSSIP_POLICY_SCHEMA
            || self.policy_id == [0u8; 16]
            || self.log_id.0 == [0u8; 16]
            || self.transparency_authority_key_id.0 == [0u8; 16]
            || self.observers.len() < 2
            || self.observers.len() > MAX_CHECKPOINT_PUBLIC_SIGNERS
            || self.threshold < 2
            || usize::from(self.threshold) > self.observers.len()
            || self.minimum_organizations < 2
            || self.minimum_organizations > self.threshold
            || self.maximum_statement_age_seconds == 0
            || self.maximum_statement_age_seconds
                > MAX_CHECKPOINT_TRANSPARENCY_GOSSIP_STATEMENT_AGE_SECONDS
            || self.valid_from_unix_seconds == 0
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointTransparencyGossipError::InvalidPolicy);
        }
        let mut observer_ids = HashSet::with_capacity(self.observers.len());
        let mut key_ids = HashSet::with_capacity(self.observers.len());
        let mut key_bytes = HashSet::with_capacity(self.observers.len());
        let mut organizations = HashSet::with_capacity(self.observers.len());
        for observer in &self.observers {
            observer.validate()?;
            if observer.valid_from_unix_seconds > self.valid_from_unix_seconds
                || observer.valid_until_unix_seconds < self.valid_until_unix_seconds
                || !observer_ids.insert(observer.observer_id)
                || !key_ids.insert(observer.verifying_key.key_id)
                || !key_bytes.insert(observer.verifying_key.verifying_key_bytes)
            {
                return Err(CheckpointTransparencyGossipError::DuplicateObserver);
            }
            organizations.insert(observer.organization_binding);
        }
        if organizations.len() < usize::from(self.minimum_organizations) {
            return Err(CheckpointTransparencyGossipError::InvalidPolicy);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<[u8; 32], CheckpointTransparencyGossipError> {
        self.validate()?;
        gossip_digest(GOSSIP_POLICY_DIGEST_DOMAIN, self)
    }

    pub fn observer(
        &self,
        key_id: CheckpointPublicKeyId,
    ) -> Option<&CheckpointTransparencyGossipObserver> {
        self.observers
            .iter()
            .find(|observer| observer.verifying_key.key_id == key_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CheckpointTransparencyGossipStatementBody {
    policy_digest: [u8; 32],
    origin_id: CheckpointTransparencyOriginId,
    source_binding: [u8; 32],
    observer_key_id: CheckpointPublicKeyId,
    head_digest: [u8; 32],
    observed_at_unix_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointTransparencyGossipStatement {
    pub schema: String,
    pub policy_digest: [u8; 32],
    pub origin_id: CheckpointTransparencyOriginId,
    pub source_binding: [u8; 32],
    pub observer_key_id: CheckpointPublicKeyId,
    pub signed_head: CheckpointSignedTransparencyHead,
    pub observed_at_unix_seconds: u64,
    pub signature: CheckpointPublicSignature,
}

impl CheckpointTransparencyGossipStatement {
    pub fn sign(
        observer_signing_key: &CheckpointPublicSigningKey,
        policy: &CheckpointTransparencyGossipPolicy,
        origin_id: CheckpointTransparencyOriginId,
        source_binding: [u8; 32],
        signed_head: CheckpointSignedTransparencyHead,
        observed_at_unix_seconds: u64,
    ) -> Result<Self, CheckpointTransparencyGossipError> {
        let observer = policy
            .observer(observer_signing_key.key_id())
            .ok_or(CheckpointTransparencyGossipError::UnknownObserver)?;
        if source_binding == [0u8; 32]
            || observed_at_unix_seconds < observer.valid_from_unix_seconds
            || observed_at_unix_seconds > observer.valid_until_unix_seconds
        {
            return Err(CheckpointTransparencyGossipError::InvalidStatement);
        }
        let policy_digest = policy.digest()?;
        let head_digest = signed_head.head.digest()?;
        let body = CheckpointTransparencyGossipStatementBody {
            policy_digest,
            origin_id,
            source_binding,
            observer_key_id: observer_signing_key.key_id(),
            head_digest,
            observed_at_unix_seconds,
        };
        let body_digest = gossip_digest(GOSSIP_STATEMENT_DIGEST_DOMAIN, &body)?;
        Ok(Self {
            schema: CHECKPOINT_TRANSPARENCY_GOSSIP_STATEMENT_SCHEMA.to_owned(),
            policy_digest,
            origin_id,
            source_binding,
            observer_key_id: observer_signing_key.key_id(),
            signed_head,
            observed_at_unix_seconds,
            signature: observer_signing_key.sign(
                GOSSIP_STATEMENT_SIGNATURE_DOMAIN,
                &body_digest,
            )?,
        })
    }

    fn body_digest(&self) -> Result<[u8; 32], CheckpointTransparencyGossipError> {
        if self.schema != CHECKPOINT_TRANSPARENCY_GOSSIP_STATEMENT_SCHEMA
            || self.policy_digest == [0u8; 32]
            || self.origin_id.0 == [0u8; 16]
            || self.source_binding == [0u8; 32]
            || self.observer_key_id.0 == [0u8; 16]
            || self.observed_at_unix_seconds == 0
        {
            return Err(CheckpointTransparencyGossipError::InvalidStatement);
        }
        gossip_digest(
            GOSSIP_STATEMENT_DIGEST_DOMAIN,
            &CheckpointTransparencyGossipStatementBody {
                policy_digest: self.policy_digest,
                origin_id: self.origin_id,
                source_binding: self.source_binding,
                observer_key_id: self.observer_key_id,
                head_digest: self.signed_head.head.digest()?,
                observed_at_unix_seconds: self.observed_at_unix_seconds,
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointTransparencyGossipObservation {
    pub statement: CheckpointTransparencyGossipStatement,
    pub consistency_proof: Option<CheckpointTransparencyConsistencyProof>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointTransparencyGossipBundle {
    pub schema: String,
    pub anchor_head: CheckpointSignedTransparencyHead,
    pub observations: Vec<CheckpointTransparencyGossipObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointTransparencyGossipSummary {
    pub schema: String,
    pub anchor_head_digest: [u8; 32],
    pub valid_observations: usize,
    pub unique_origins: usize,
    pub unique_organizations: usize,
    pub consistency_paths_verified: usize,
    pub equal_size_heads_verified: usize,
}

impl CheckpointTransparencyGossipSummary {
    pub fn validate(&self) -> Result<(), CheckpointTransparencyGossipError> {
        if self.schema != CHECKPOINT_TRANSPARENCY_GOSSIP_SUMMARY_SCHEMA
            || self.anchor_head_digest == [0u8; 32]
            || self.valid_observations < 2
            || self.unique_origins < 2
            || self.unique_organizations < 2
        {
            return Err(CheckpointTransparencyGossipError::InvalidBundle);
        }
        Ok(())
    }
}

impl CheckpointTransparencyGossipBundle {
    pub fn verify(
        &self,
        policy: &CheckpointTransparencyGossipPolicy,
        transparency_authority_key: &CheckpointPublicVerifyingKey,
        verification_time_unix_seconds: u64,
    ) -> Result<CheckpointTransparencyGossipSummary, CheckpointTransparencyGossipError> {
        policy.validate()?;
        if self.schema != CHECKPOINT_TRANSPARENCY_GOSSIP_BUNDLE_SCHEMA
            || self.observations.is_empty()
            || self.observations.len() > MAX_CHECKPOINT_TRANSPARENCY_GOSSIP_OBSERVATIONS
            || policy.transparency_authority_key_id != transparency_authority_key.key_id
            || verification_time_unix_seconds < policy.valid_from_unix_seconds
            || verification_time_unix_seconds > policy.valid_until_unix_seconds
        {
            return Err(CheckpointTransparencyGossipError::InvalidBundle);
        }
        let anchor_head_digest = self.anchor_head.verify(transparency_authority_key)?;
        let anchor = &self.anchor_head.head;
        if anchor.log_id != policy.log_id {
            return Err(CheckpointTransparencyGossipError::InvalidBundle);
        }
        let policy_digest = policy.digest()?;
        let mut observers = HashSet::new();
        let mut origins = HashSet::new();
        let mut source_bindings = HashSet::new();
        let mut organizations = HashSet::new();
        let mut heads_by_size: HashMap<u64, [u8; 32]> = HashMap::new();
        heads_by_size.insert(anchor.entry_count, anchor_head_digest);
        let mut consistency_paths_verified = 0usize;
        let mut equal_size_heads_verified = 0usize;
        for observation in &self.observations {
            let statement = &observation.statement;
            let observer = policy
                .observer(statement.observer_key_id)
                .ok_or(CheckpointTransparencyGossipError::UnknownObserver)?;
            if statement.policy_digest != policy_digest
                || statement.observed_at_unix_seconds < observer.valid_from_unix_seconds
                || statement.observed_at_unix_seconds > observer.valid_until_unix_seconds
                || statement.observed_at_unix_seconds > verification_time_unix_seconds
                || verification_time_unix_seconds
                    .saturating_sub(statement.observed_at_unix_seconds)
                    > policy.maximum_statement_age_seconds
                || !observers.insert(statement.observer_key_id)
                || !origins.insert(statement.origin_id)
                || !source_bindings.insert(statement.source_binding)
            {
                return Err(CheckpointTransparencyGossipError::InvalidStatement);
            }
            let observed_head_digest = statement
                .signed_head
                .verify(transparency_authority_key)?;
            let observed_head = &statement.signed_head.head;
            if observed_head.log_id != policy.log_id
                || statement.observed_at_unix_seconds < observed_head.issued_at_unix_seconds
            {
                return Err(CheckpointTransparencyGossipError::InvalidStatement);
            }
            let body_digest = statement.body_digest()?;
            observer.verifying_key.verify(
                GOSSIP_STATEMENT_SIGNATURE_DOMAIN,
                &body_digest,
                &statement.signature,
            )?;
            if let Some(existing_digest) = heads_by_size.get(&observed_head.entry_count) {
                if *existing_digest != observed_head_digest {
                    return Err(CheckpointTransparencyGossipError::SplitViewDetected);
                }
            } else {
                heads_by_size.insert(observed_head.entry_count, observed_head_digest);
            }
            organizations.insert(observer.organization_binding);

            if observed_head.entry_count == anchor.entry_count {
                if observed_head != anchor || observation.consistency_proof.is_some() {
                    return Err(CheckpointTransparencyGossipError::InvalidConsistencyPath);
                }
                equal_size_heads_verified = equal_size_heads_verified.saturating_add(1);
            } else {
                let proof = observation
                    .consistency_proof
                    .as_ref()
                    .ok_or(CheckpointTransparencyGossipError::InvalidConsistencyPath)?;
                proof.verify()?;
                let valid_direction = if observed_head.entry_count < anchor.entry_count {
                    proof.prior_head == *observed_head && proof.current_head == *anchor
                } else {
                    proof.prior_head == *anchor && proof.current_head == *observed_head
                };
                if !valid_direction {
                    return Err(CheckpointTransparencyGossipError::InvalidConsistencyPath);
                }
                consistency_paths_verified = consistency_paths_verified.saturating_add(1);
            }
        }
        if observers.len() < usize::from(policy.threshold)
            || origins.len() < usize::from(policy.threshold)
            || organizations.len() < usize::from(policy.minimum_organizations)
        {
            return Err(CheckpointTransparencyGossipError::InsufficientObservations);
        }
        let summary = CheckpointTransparencyGossipSummary {
            schema: CHECKPOINT_TRANSPARENCY_GOSSIP_SUMMARY_SCHEMA.to_owned(),
            anchor_head_digest,
            valid_observations: observers.len(),
            unique_origins: origins.len(),
            unique_organizations: organizations.len(),
            consistency_paths_verified,
            equal_size_heads_verified,
        };
        summary.validate()?;
        Ok(summary)
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointTransparencySplitViewNegativeSummary {
    pub schema: String,
    pub exercised: bool,
    pub split_view_rejected: bool,
}

impl CheckpointTransparencySplitViewNegativeSummary {
    pub fn validate(&self) -> Result<(), CheckpointTransparencyGossipError> {
        if self.schema != CHECKPOINT_TRANSPARENCY_SPLIT_VIEW_NEGATIVE_SCHEMA
            || !self.exercised
            || !self.split_view_rejected
        {
            return Err(CheckpointTransparencyGossipError::SplitViewNotRejected);
        }
        Ok(())
    }
}

pub fn verify_transparency_split_view_negative(
    candidate: &CheckpointTransparencyGossipBundle,
    policy: &CheckpointTransparencyGossipPolicy,
    transparency_authority_key: &CheckpointPublicVerifyingKey,
    verification_time_unix_seconds: u64,
) -> Result<CheckpointTransparencySplitViewNegativeSummary, CheckpointTransparencyGossipError> {
    match candidate.verify(
        policy,
        transparency_authority_key,
        verification_time_unix_seconds,
    ) {
        Err(CheckpointTransparencyGossipError::SplitViewDetected) => {
            let summary = CheckpointTransparencySplitViewNegativeSummary {
                schema: CHECKPOINT_TRANSPARENCY_SPLIT_VIEW_NEGATIVE_SCHEMA.to_owned(),
                exercised: true,
                split_view_rejected: true,
            };
            summary.validate()?;
            Ok(summary)
        }
        Ok(_) => Err(CheckpointTransparencyGossipError::SplitViewNotRejected),
        Err(error) => Err(error),
    }
}

fn gossip_digest<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<[u8; 32], CheckpointTransparencyGossipError> {
    let encoded = postcard::to_stdvec(value)
        .map_err(|_| CheckpointTransparencyGossipError::Encoding)?;
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES {
        return Err(CheckpointTransparencyGossipError::TooLarge);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(*hasher.finalize().as_bytes())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointTransparencyGossipError {
    InvalidOrigin,
    InvalidObserver,
    DuplicateObserver,
    UnknownObserver,
    InvalidPolicy,
    InvalidStatement,
    InvalidBundle,
    InvalidConsistencyPath,
    SplitViewDetected,
    SplitViewNotRejected,
    InsufficientObservations,
    Encoding,
    TooLarge,
    PublicVerificationFailed,
}

impl std::fmt::Display for CheckpointTransparencyGossipError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidOrigin => "invalid transparency gossip origin",
            Self::InvalidObserver => "invalid transparency gossip observer",
            Self::DuplicateObserver => "duplicate transparency gossip observer",
            Self::UnknownObserver => "unknown transparency gossip observer",
            Self::InvalidPolicy => "invalid transparency gossip policy",
            Self::InvalidStatement => "invalid transparency gossip statement",
            Self::InvalidBundle => "invalid transparency gossip bundle",
            Self::InvalidConsistencyPath => "invalid transparency gossip consistency path",
            Self::SplitViewDetected => "transparency split view detected",
            Self::SplitViewNotRejected => "transparency split-view negative was not rejected",
            Self::InsufficientObservations => "insufficient independent gossip observations",
            Self::Encoding => "transparency gossip encoding failed",
            Self::TooLarge => "transparency gossip artifact exceeds its bound",
            Self::PublicVerificationFailed => "public transparency verification failed",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for CheckpointTransparencyGossipError {}

impl From<CheckpointPublicVerificationError> for CheckpointTransparencyGossipError {
    fn from(_: CheckpointPublicVerificationError) -> Self {
        Self::PublicVerificationFailed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CheckpointTransparencyEntryKind, CheckpointTransparencyLog,
    };

    fn signing_key(id: u8, seed: u8) -> CheckpointPublicSigningKey {
        CheckpointPublicSigningKey::from_seed(
            CheckpointPublicKeyId::new([id; 16]).unwrap(),
            [seed; 32],
        )
        .unwrap()
    }

    fn signed_head(
        authority: &CheckpointPublicSigningKey,
        artifact: u8,
    ) -> CheckpointSignedTransparencyHead {
        let log_id = CheckpointTransparencyLogId::new([0x71; 16]).unwrap();
        let mut log = CheckpointTransparencyLog::new(log_id);
        log.append(
            1,
            CheckpointTransparencyEntryKind::FederationPlan,
            [artifact; 32],
            [artifact.wrapping_add(1); 32],
            100,
        )
        .unwrap();
        CheckpointSignedTransparencyHead::sign(authority, log.head(110).unwrap()).unwrap()
    }

    fn policy(
        authority: &CheckpointPublicSigningKey,
        first: &CheckpointPublicSigningKey,
        second: &CheckpointPublicSigningKey,
    ) -> CheckpointTransparencyGossipPolicy {
        CheckpointTransparencyGossipPolicy {
            schema: CHECKPOINT_TRANSPARENCY_GOSSIP_POLICY_SCHEMA.to_owned(),
            policy_id: [0x72; 16],
            log_id: CheckpointTransparencyLogId::new([0x71; 16]).unwrap(),
            transparency_authority_key_id: authority.key_id(),
            observers: vec![
                CheckpointTransparencyGossipObserver {
                    schema: CHECKPOINT_TRANSPARENCY_GOSSIP_OBSERVER_SCHEMA.to_owned(),
                    observer_id: [0x81; 16],
                    organization_binding: [0x91; 32],
                    verifying_key: first.verifying_key(),
                    valid_from_unix_seconds: 100,
                    valid_until_unix_seconds: 500,
                },
                CheckpointTransparencyGossipObserver {
                    schema: CHECKPOINT_TRANSPARENCY_GOSSIP_OBSERVER_SCHEMA.to_owned(),
                    observer_id: [0x82; 16],
                    organization_binding: [0x92; 32],
                    verifying_key: second.verifying_key(),
                    valid_from_unix_seconds: 100,
                    valid_until_unix_seconds: 500,
                },
            ],
            threshold: 2,
            minimum_organizations: 2,
            maximum_statement_age_seconds: 100,
            valid_from_unix_seconds: 100,
            valid_until_unix_seconds: 500,
        }
    }

    #[test]
    fn independent_equal_size_observations_verify() {
        let authority = signing_key(1, 21);
        let first = signing_key(2, 22);
        let second = signing_key(3, 23);
        let policy = policy(&authority, &first, &second);
        let head = signed_head(&authority, 0x31);
        let bundle = CheckpointTransparencyGossipBundle {
            schema: CHECKPOINT_TRANSPARENCY_GOSSIP_BUNDLE_SCHEMA.to_owned(),
            anchor_head: head.clone(),
            observations: vec![
                CheckpointTransparencyGossipObservation {
                    statement: CheckpointTransparencyGossipStatement::sign(
                        &first,
                        &policy,
                        CheckpointTransparencyOriginId::new([0xa1; 16]).unwrap(),
                        [0xb1; 32],
                        head.clone(),
                        120,
                    )
                    .unwrap(),
                    consistency_proof: None,
                },
                CheckpointTransparencyGossipObservation {
                    statement: CheckpointTransparencyGossipStatement::sign(
                        &second,
                        &policy,
                        CheckpointTransparencyOriginId::new([0xa2; 16]).unwrap(),
                        [0xb2; 32],
                        head,
                        121,
                    )
                    .unwrap(),
                    consistency_proof: None,
                },
            ],
        };
        let summary = bundle.verify(&policy, &authority.verifying_key(), 130).unwrap();
        assert_eq!(summary.valid_observations, 2);
        assert_eq!(summary.equal_size_heads_verified, 2);
    }

    #[test]
    fn equal_size_fork_is_rejected() {
        let authority = signing_key(1, 31);
        let first = signing_key(2, 32);
        let second = signing_key(3, 33);
        let policy = policy(&authority, &first, &second);
        let anchor = signed_head(&authority, 0x41);
        let fork = signed_head(&authority, 0x42);
        let candidate = CheckpointTransparencyGossipBundle {
            schema: CHECKPOINT_TRANSPARENCY_GOSSIP_BUNDLE_SCHEMA.to_owned(),
            anchor_head: anchor.clone(),
            observations: vec![
                CheckpointTransparencyGossipObservation {
                    statement: CheckpointTransparencyGossipStatement::sign(
                        &first,
                        &policy,
                        CheckpointTransparencyOriginId::new([0xc1; 16]).unwrap(),
                        [0xd1; 32],
                        anchor,
                        120,
                    )
                    .unwrap(),
                    consistency_proof: None,
                },
                CheckpointTransparencyGossipObservation {
                    statement: CheckpointTransparencyGossipStatement::sign(
                        &second,
                        &policy,
                        CheckpointTransparencyOriginId::new([0xc2; 16]).unwrap(),
                        [0xd2; 32],
                        fork,
                        121,
                    )
                    .unwrap(),
                    consistency_proof: None,
                },
            ],
        };
        verify_transparency_split_view_negative(
            &candidate,
            &policy,
            &authority.verifying_key(),
            130,
        )
        .unwrap();
    }
}
