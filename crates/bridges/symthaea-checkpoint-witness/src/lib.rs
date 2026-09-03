// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh external checkpoint witnesses for Symthaea crash recovery.
//!
//! A signature over an Agency Kernel checkpoint head is not sufficient anti-
//! rollback evidence: an attacker can replay an old valid signature together
//! with an old local database. V1 therefore requires a fresh random challenge.
//! Independent witnesses answer that challenge with the head currently retained
//! in their own durable monotonic state.
//!
//! The witness service contract is intentionally stronger than "sign this head":
//! the client does not choose the returned head. Each service must load its own
//! latest retained head for the exact grant/domain, refuse rollback of that
//! state, and sign that head together with the fresh challenge. This verifier
//! then requires threshold agreement plus organizational/service diversity.
//!
//! Challenge freshness is derived from `VerifiedAuthorityTime`; caller wall
//! clock values are never accepted. This crate creates no execution authority.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use symthaea_action_checkpoint::CheckpointHead;
use symthaea_authority::Digest32;
use symthaea_authority_time::{AuthorityTimeError, VerifiedAuthorityTime};
use thiserror::Error;

pub const CHECKPOINT_WITNESS_SCHEMA_VERSION: u16 = 1;
pub const MAX_CHECKPOINT_WITNESSES: usize = 64;
pub const MAX_CHECKPOINT_STATEMENTS: usize = 128;
pub const MAX_CHECKPOINT_CHALLENGE_AGE_S: u64 = 60;

const POLICY_DOMAIN: &[u8] = b"symthaea.checkpoint-witness.policy.v1\0";
const CHALLENGE_DOMAIN: &[u8] = b"symthaea.checkpoint-witness.challenge.v1\0";
const STATEMENT_DOMAIN: &[u8] = b"symthaea.checkpoint-witness.statement.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CheckpointWitnessId(pub [u8; 16]);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedCheckpointWitnessV1 {
    pub witness_id: CheckpointWitnessId,
    pub verifying_key: [u8; 32],
    pub organization_binding: [u8; 32],
    pub service_binding: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointWitnessPolicyV1 {
    pub schema_version: u16,
    pub policy_id: [u8; 16],
    pub witnesses: Vec<TrustedCheckpointWitnessV1>,
    pub threshold: u16,
    pub minimum_organizations: u16,
    /// Maximum worst-case elapsed time from challenge creation to verification.
    pub maximum_challenge_age_s: u64,
}

impl CheckpointWitnessPolicyV1 {
    pub fn validate(&self) -> Result<(), CheckpointWitnessError> {
        if self.schema_version != CHECKPOINT_WITNESS_SCHEMA_VERSION
            || self.policy_id == [0; 16]
            || self.witnesses.len() < 2
            || self.witnesses.len() > MAX_CHECKPOINT_WITNESSES
            || self.threshold < 2
            || usize::from(self.threshold) > self.witnesses.len()
            || self.minimum_organizations < 2
            || self.minimum_organizations > self.threshold
            || self.maximum_challenge_age_s == 0
            || self.maximum_challenge_age_s > MAX_CHECKPOINT_CHALLENGE_AGE_S
        {
            return Err(CheckpointWitnessError::InvalidPolicy);
        }

        let mut ids = BTreeSet::new();
        let mut keys = BTreeSet::new();
        let mut services = BTreeSet::new();
        let mut organizations = BTreeSet::new();
        for witness in &self.witnesses {
            if witness.witness_id.0 == [0; 16]
                || witness.verifying_key == [0; 32]
                || witness.organization_binding == [0; 32]
                || witness.service_binding == [0; 32]
                || VerifyingKey::from_bytes(&witness.verifying_key).is_err()
                || !ids.insert(witness.witness_id)
                || !keys.insert(witness.verifying_key)
                || !services.insert(witness.service_binding)
            {
                return Err(CheckpointWitnessError::InvalidPolicy);
            }
            organizations.insert(witness.organization_binding);
        }
        if organizations.len() < usize::from(self.minimum_organizations) {
            return Err(CheckpointWitnessError::InvalidPolicy);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<[u8; 32], CheckpointWitnessError> {
        self.validate()?;
        let mut transcript = Transcript::new(POLICY_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.policy_id);
        transcript.u32(
            u32::try_from(self.witnesses.len()).map_err(|_| CheckpointWitnessError::Encoding)?,
        );
        for witness in &self.witnesses {
            transcript.fixed(&witness.witness_id.0);
            transcript.fixed(&witness.verifying_key);
            transcript.fixed(&witness.organization_binding);
            transcript.fixed(&witness.service_binding);
        }
        transcript.u16(self.threshold);
        transcript.u16(self.minimum_organizations);
        transcript.u64(self.maximum_challenge_age_s);
        Ok(transcript.finish())
    }

    fn witness(&self, id: CheckpointWitnessId) -> Option<&TrustedCheckpointWitnessV1> {
        self.witnesses.iter().find(|witness| witness.witness_id == id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointWitnessChallengeV1 {
    pub schema_version: u16,
    pub nonce: [u8; 32],
    pub grant_digest: Digest32,
    pub witness_policy_digest: [u8; 32],
    pub time_policy_digest: [u8; 32],
}

impl CheckpointWitnessChallengeV1 {
    pub fn digest(&self) -> [u8; 32] {
        let mut transcript = Transcript::new(CHALLENGE_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.nonce);
        transcript.fixed(&self.grant_digest.0);
        transcript.fixed(&self.witness_policy_digest);
        transcript.fixed(&self.time_policy_digest);
        transcript.finish()
    }
}

/// Local state for one fresh witness query.
///
/// `created_not_before_unix_s` is the earliest plausible creation time from the
/// trusted-time interval. Comparing the later *latest* plausible time against
/// it gives a worst-case challenge age, so uncertainty cannot lengthen the
/// freshness window.
#[derive(Debug)]
pub struct PendingCheckpointWitnessChallenge {
    wire: CheckpointWitnessChallengeV1,
    created_not_before_unix_s: u64,
}

impl PendingCheckpointWitnessChallenge {
    pub fn new(
        policy: &CheckpointWitnessPolicyV1,
        grant_digest: Digest32,
        authority_time: &VerifiedAuthorityTime,
    ) -> Result<Self, CheckpointWitnessError> {
        if grant_digest.0 == [0; 32] {
            return Err(CheckpointWitnessError::InvalidGrantDigest);
        }
        policy.validate()?;
        authority_time.require_subject(grant_digest.0)?;
        // Force the time fact through its freshness/boot-time check now.
        let _ = authority_time.conservative_now_unix_s()?;
        let (created_not_before_unix_s, _) = authority_time.interval_at_verification();
        let mut nonce = [0u8; 32];
        getrandom::getrandom(&mut nonce).map_err(|_| CheckpointWitnessError::RandomnessUnavailable)?;
        if nonce == [0; 32] {
            return Err(CheckpointWitnessError::RandomnessUnavailable);
        }
        Ok(Self {
            wire: CheckpointWitnessChallengeV1 {
                schema_version: CHECKPOINT_WITNESS_SCHEMA_VERSION,
                nonce,
                grant_digest,
                witness_policy_digest: policy.digest()?,
                time_policy_digest: authority_time.policy_digest(),
            },
            created_not_before_unix_s,
        })
    }

    pub fn wire(&self) -> CheckpointWitnessChallengeV1 {
        self.wire
    }
}

/// Fresh witness response. `witness_generation` is the witness service's own
/// monotonically increasing durable generation, not the Agency Kernel sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointWitnessStatementV1 {
    pub schema_version: u16,
    pub witness_id: CheckpointWitnessId,
    pub challenge_nonce: [u8; 32],
    pub grant_digest: Digest32,
    pub witness_policy_digest: [u8; 32],
    pub time_policy_digest: [u8; 32],
    pub checkpoint_sequence: u64,
    pub checkpoint_digest: Digest32,
    pub witness_generation: u64,
    pub signature: Vec<u8>,
}

impl CheckpointWitnessStatementV1 {
    pub fn canonical_message(&self) -> Result<Vec<u8>, CheckpointWitnessError> {
        if self.schema_version != CHECKPOINT_WITNESS_SCHEMA_VERSION
            || self.witness_id.0 == [0; 16]
            || self.challenge_nonce == [0; 32]
            || self.grant_digest.0 == [0; 32]
            || self.witness_policy_digest == [0; 32]
            || self.time_policy_digest == [0; 32]
            || self.checkpoint_digest.0 == [0; 32]
            || self.witness_generation == 0
        {
            return Err(CheckpointWitnessError::InvalidStatement);
        }
        let mut transcript = Transcript::new(STATEMENT_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.witness_id.0);
        transcript.fixed(&self.challenge_nonce);
        transcript.fixed(&self.grant_digest.0);
        transcript.fixed(&self.witness_policy_digest);
        transcript.fixed(&self.time_policy_digest);
        transcript.u64(self.checkpoint_sequence);
        transcript.fixed(&self.checkpoint_digest.0);
        transcript.u64(self.witness_generation);
        Ok(transcript.into_bytes())
    }

    pub fn checkpoint_head(&self) -> CheckpointHead {
        CheckpointHead {
            sequence: self.checkpoint_sequence,
            digest: self.checkpoint_digest,
        }
    }
}

/// Opaque proof that fresh, independent external witnesses agreed on the head
/// currently retained for one exact capability-grant lineage.
#[derive(Debug)]
pub struct VerifiedCheckpointHead {
    grant_digest: Digest32,
    head: CheckpointHead,
    witness_policy_digest: [u8; 32],
    time_policy_digest: [u8; 32],
    witness_count: u16,
    organization_count: u16,
}

impl VerifiedCheckpointHead {
    pub fn grant_digest(&self) -> Digest32 {
        self.grant_digest
    }

    pub fn head(&self) -> CheckpointHead {
        self.head
    }

    pub fn witness_policy_digest(&self) -> [u8; 32] {
        self.witness_policy_digest
    }

    pub fn time_policy_digest(&self) -> [u8; 32] {
        self.time_policy_digest
    }

    pub fn witness_count(&self) -> u16 {
        self.witness_count
    }

    pub fn organization_count(&self) -> u16 {
        self.organization_count
    }

    pub fn require_grant(&self, expected: Digest32) -> Result<(), CheckpointWitnessError> {
        if self.grant_digest != expected {
            return Err(CheckpointWitnessError::GrantMismatch);
        }
        Ok(())
    }
}

pub fn verify_checkpoint_witnesses_v1(
    policy: &CheckpointWitnessPolicyV1,
    challenge: PendingCheckpointWitnessChallenge,
    response_time: &VerifiedAuthorityTime,
    statements: &[CheckpointWitnessStatementV1],
) -> Result<VerifiedCheckpointHead, CheckpointWitnessError> {
    policy.validate()?;
    if statements.len() < usize::from(policy.threshold)
        || statements.len() > MAX_CHECKPOINT_STATEMENTS
    {
        return Err(CheckpointWitnessError::InsufficientStatements);
    }
    if challenge.wire.schema_version != CHECKPOINT_WITNESS_SCHEMA_VERSION
        || challenge.wire.witness_policy_digest != policy.digest()?
        || challenge.wire.grant_digest.0 == [0; 32]
        || challenge.wire.nonce == [0; 32]
    {
        return Err(CheckpointWitnessError::InvalidChallenge);
    }

    response_time.require_subject(challenge.wire.grant_digest.0)?;
    if response_time.policy_digest() != challenge.wire.time_policy_digest {
        return Err(CheckpointWitnessError::TimePolicyChangedDuringChallenge);
    }
    let response_not_after = response_time.conservative_now_unix_s()?;
    let worst_case_age = response_not_after
        .checked_sub(challenge.created_not_before_unix_s)
        .ok_or(CheckpointWitnessError::TimeMovedBackward)?;
    if worst_case_age > policy.maximum_challenge_age_s {
        return Err(CheckpointWitnessError::ChallengeExpired);
    }

    let mut ids = BTreeSet::new();
    let mut organizations = BTreeSet::new();
    let mut services = BTreeSet::new();
    let mut agreed_head: Option<CheckpointHead> = None;

    for statement in statements {
        let witness = policy
            .witness(statement.witness_id)
            .ok_or(CheckpointWitnessError::UnknownWitness)?;
        if statement.schema_version != CHECKPOINT_WITNESS_SCHEMA_VERSION
            || statement.challenge_nonce != challenge.wire.nonce
            || statement.grant_digest != challenge.wire.grant_digest
            || statement.witness_policy_digest != challenge.wire.witness_policy_digest
            || statement.time_policy_digest != challenge.wire.time_policy_digest
            || statement.checkpoint_digest.0 == [0; 32]
            || statement.witness_generation == 0
            || !ids.insert(statement.witness_id)
        {
            return Err(CheckpointWitnessError::InvalidStatement);
        }

        let signature_bytes: [u8; 64] = statement
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| CheckpointWitnessError::BadSignatureLength)?;
        let key = VerifyingKey::from_bytes(&witness.verifying_key)
            .map_err(|_| CheckpointWitnessError::InvalidPolicy)?;
        key.verify(
            &statement.canonical_message()?,
            &Signature::from_bytes(&signature_bytes),
        )
        .map_err(|_| CheckpointWitnessError::BadSignature)?;

        let head = statement.checkpoint_head();
        match agreed_head {
            None => agreed_head = Some(head),
            Some(expected) if expected == head => {}
            Some(_) => return Err(CheckpointWitnessError::CheckpointDisagreement),
        }
        organizations.insert(witness.organization_binding);
        services.insert(witness.service_binding);
    }

    if ids.len() < usize::from(policy.threshold)
        || organizations.len() < usize::from(policy.minimum_organizations)
        || services.len() < usize::from(policy.threshold)
    {
        return Err(CheckpointWitnessError::InsufficientDiversity);
    }

    Ok(VerifiedCheckpointHead {
        grant_digest: challenge.wire.grant_digest,
        head: agreed_head.ok_or(CheckpointWitnessError::InsufficientStatements)?,
        witness_policy_digest: challenge.wire.witness_policy_digest,
        time_policy_digest: challenge.wire.time_policy_digest,
        witness_count: u16::try_from(ids.len()).map_err(|_| CheckpointWitnessError::Encoding)?,
        organization_count: u16::try_from(organizations.len())
            .map_err(|_| CheckpointWitnessError::Encoding)?,
    })
}

struct Transcript {
    bytes: Vec<u8>,
}

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(256);
        bytes.extend_from_slice(&(domain.len() as u32).to_be_bytes());
        bytes.extend_from_slice(domain);
        Self { bytes }
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn fixed<const N: usize>(&mut self, value: &[u8; N]) {
        self.bytes.extend_from_slice(value);
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    fn finish(self) -> [u8; 32] {
        *blake3::hash(&self.bytes).as_bytes()
    }
}

#[derive(Debug, Error)]
pub enum CheckpointWitnessError {
    #[error("checkpoint witness policy is invalid")]
    InvalidPolicy,
    #[error("checkpoint witness grant digest is invalid")]
    InvalidGrantDigest,
    #[error("secure checkpoint challenge randomness is unavailable")]
    RandomnessUnavailable,
    #[error("checkpoint witness challenge is invalid")]
    InvalidChallenge,
    #[error("not enough checkpoint witness statements were supplied")]
    InsufficientStatements,
    #[error("checkpoint statement references an unknown witness")]
    UnknownWitness,
    #[error("checkpoint witness statement is malformed or does not bind the challenge")]
    InvalidStatement,
    #[error("checkpoint witness Ed25519 signature must be exactly 64 bytes")]
    BadSignatureLength,
    #[error("checkpoint witness signature verification failed")]
    BadSignature,
    #[error("checkpoint witnesses do not satisfy organizational/service diversity")]
    InsufficientDiversity,
    #[error("fresh checkpoint witnesses disagree on the retained Agency Kernel head")]
    CheckpointDisagreement,
    #[error("verified checkpoint witness belongs to a different grant")]
    GrantMismatch,
    #[error("trusted-time policy changed while checkpoint challenge was in flight")]
    TimePolicyChangedDuringChallenge,
    #[error("checkpoint challenge exceeded its maximum worst-case age")]
    ChallengeExpired,
    #[error("trusted time moved backward across checkpoint challenge")]
    TimeMovedBackward,
    #[error("trusted authority time failed: {0}")]
    AuthorityTime(#[from] AuthorityTimeError),
    #[error("checkpoint witness canonical encoding failed")]
    Encoding,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use symthaea_authority_time::{
        AUTHORITY_TIME_SCHEMA_VERSION, AuthorityTimeStatementV1, PendingAuthorityTimeChallenge,
        TimeAuthorityId, TrustedTimeAuthorityV1, TrustedTimePolicyV1,
        verify_authority_time_v1,
    };

    fn authority_time(grant_digest: Digest32, witnessed: u64) -> VerifiedAuthorityTime {
        let key_a = SigningKey::from_bytes(&[71; 32]);
        let key_b = SigningKey::from_bytes(&[72; 32]);
        let policy = TrustedTimePolicyV1 {
            schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
            policy_id: [73; 16],
            authorities: vec![
                TrustedTimeAuthorityV1 {
                    authority_id: TimeAuthorityId([1; 16]),
                    verifying_key: key_a.verifying_key().to_bytes(),
                    organization_binding: [81; 32],
                    service_binding: [91; 32],
                },
                TrustedTimeAuthorityV1 {
                    authority_id: TimeAuthorityId([2; 16]),
                    verifying_key: key_b.verifying_key().to_bytes(),
                    organization_binding: [82; 32],
                    service_binding: [92; 32],
                },
            ],
            threshold: 2,
            minimum_organizations: 2,
            maximum_uncertainty_s: 1,
            maximum_challenge_age_ns: 5_000_000_000,
            maximum_post_verification_age_ns: 5_000_000_000,
        };
        let pending = PendingAuthorityTimeChallenge::new(&policy, grant_digest.0).unwrap();
        let challenge = pending.wire();
        let sign = |authority_id: TimeAuthorityId, key: &SigningKey| {
            let mut statement = AuthorityTimeStatementV1 {
                schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
                authority_id,
                policy_digest: challenge.policy_digest,
                subject_digest: challenge.subject_digest,
                challenge_nonce: challenge.nonce,
                witnessed_unix_s: witnessed,
                uncertainty_s: 1,
                signature: Vec::new(),
            };
            statement.signature = key
                .sign(&statement.canonical_message().unwrap())
                .to_bytes()
                .to_vec();
            statement
        };
        verify_authority_time_v1(
            &policy,
            pending,
            &[
                sign(TimeAuthorityId([1; 16]), &key_a),
                sign(TimeAuthorityId([2; 16]), &key_b),
            ],
        )
        .unwrap()
    }

    fn witness_policy() -> (CheckpointWitnessPolicyV1, Vec<SigningKey>) {
        let key_a = SigningKey::from_bytes(&[11; 32]);
        let key_b = SigningKey::from_bytes(&[12; 32]);
        let key_c = SigningKey::from_bytes(&[13; 32]);
        (
            CheckpointWitnessPolicyV1 {
                schema_version: CHECKPOINT_WITNESS_SCHEMA_VERSION,
                policy_id: [14; 16],
                witnesses: vec![
                    TrustedCheckpointWitnessV1 {
                        witness_id: CheckpointWitnessId([1; 16]),
                        verifying_key: key_a.verifying_key().to_bytes(),
                        organization_binding: [21; 32],
                        service_binding: [31; 32],
                    },
                    TrustedCheckpointWitnessV1 {
                        witness_id: CheckpointWitnessId([2; 16]),
                        verifying_key: key_b.verifying_key().to_bytes(),
                        organization_binding: [22; 32],
                        service_binding: [32; 32],
                    },
                    TrustedCheckpointWitnessV1 {
                        witness_id: CheckpointWitnessId([3; 16]),
                        verifying_key: key_c.verifying_key().to_bytes(),
                        organization_binding: [23; 32],
                        service_binding: [33; 32],
                    },
                ],
                threshold: 2,
                minimum_organizations: 2,
                maximum_challenge_age_s: 10,
            },
            vec![key_a, key_b, key_c],
        )
    }

    fn statement(
        challenge: CheckpointWitnessChallengeV1,
        key: &SigningKey,
        witness_id: CheckpointWitnessId,
        head: CheckpointHead,
        generation: u64,
    ) -> CheckpointWitnessStatementV1 {
        let mut statement = CheckpointWitnessStatementV1 {
            schema_version: CHECKPOINT_WITNESS_SCHEMA_VERSION,
            witness_id,
            challenge_nonce: challenge.nonce,
            grant_digest: challenge.grant_digest,
            witness_policy_digest: challenge.witness_policy_digest,
            time_policy_digest: challenge.time_policy_digest,
            checkpoint_sequence: head.sequence,
            checkpoint_digest: head.digest,
            witness_generation: generation,
            signature: Vec::new(),
        };
        statement.signature = key
            .sign(&statement.canonical_message().unwrap())
            .to_bytes()
            .to_vec();
        statement
    }

    #[test]
    fn fresh_independent_witnesses_produce_opaque_verified_head() {
        let grant = Digest32([44; 32]);
        let time = authority_time(grant, 1_000);
        let (policy, keys) = witness_policy();
        let pending = PendingCheckpointWitnessChallenge::new(&policy, grant, &time).unwrap();
        let challenge = pending.wire();
        let head = CheckpointHead {
            sequence: 9,
            digest: Digest32([55; 32]),
        };
        let verified = verify_checkpoint_witnesses_v1(
            &policy,
            pending,
            &time,
            &[
                statement(challenge, &keys[0], CheckpointWitnessId([1; 16]), head, 20),
                statement(challenge, &keys[1], CheckpointWitnessId([2; 16]), head, 31),
            ],
        )
        .unwrap();
        assert_eq!(verified.head(), head);
        assert_eq!(verified.grant_digest(), grant);
        assert_eq!(verified.witness_count(), 2);
    }

    #[test]
    fn old_signed_statement_cannot_cross_a_new_challenge_nonce() {
        let grant = Digest32([44; 32]);
        let time = authority_time(grant, 1_000);
        let (policy, keys) = witness_policy();
        let first = PendingCheckpointWitnessChallenge::new(&policy, grant, &time).unwrap();
        let first_wire = first.wire();
        let head = CheckpointHead {
            sequence: 9,
            digest: Digest32([55; 32]),
        };
        let old = statement(first_wire, &keys[0], CheckpointWitnessId([1; 16]), head, 20);
        let fresh = PendingCheckpointWitnessChallenge::new(&policy, grant, &time).unwrap();
        let fresh_wire = fresh.wire();
        let second = statement(fresh_wire, &keys[1], CheckpointWitnessId([2; 16]), head, 31);
        assert!(matches!(
            verify_checkpoint_witnesses_v1(&policy, fresh, &time, &[old, second]),
            Err(CheckpointWitnessError::InvalidStatement)
        ));
    }

    #[test]
    fn fresh_witness_disagreement_contains_instead_of_selecting_a_head() {
        let grant = Digest32([44; 32]);
        let time = authority_time(grant, 1_000);
        let (policy, keys) = witness_policy();
        let pending = PendingCheckpointWitnessChallenge::new(&policy, grant, &time).unwrap();
        let challenge = pending.wire();
        let h1 = CheckpointHead {
            sequence: 9,
            digest: Digest32([55; 32]),
        };
        let h2 = CheckpointHead {
            sequence: 10,
            digest: Digest32([56; 32]),
        };
        assert!(matches!(
            verify_checkpoint_witnesses_v1(
                &policy,
                pending,
                &time,
                &[
                    statement(challenge, &keys[0], CheckpointWitnessId([1; 16]), h1, 20),
                    statement(challenge, &keys[1], CheckpointWitnessId([2; 16]), h2, 31),
                ],
            ),
            Err(CheckpointWitnessError::CheckpointDisagreement)
        ));
    }

    #[test]
    fn witness_for_another_grant_cannot_be_reused() {
        let grant = Digest32([44; 32]);
        let other = Digest32([45; 32]);
        let time = authority_time(grant, 1_000);
        let (policy, keys) = witness_policy();
        let pending = PendingCheckpointWitnessChallenge::new(&policy, grant, &time).unwrap();
        let challenge = pending.wire();
        let head = CheckpointHead {
            sequence: 9,
            digest: Digest32([55; 32]),
        };
        let mut wrong = statement(challenge, &keys[0], CheckpointWitnessId([1; 16]), head, 20);
        wrong.grant_digest = other;
        let valid = statement(challenge, &keys[1], CheckpointWitnessId([2; 16]), head, 31);
        assert!(matches!(
            verify_checkpoint_witnesses_v1(&policy, pending, &time, &[wrong, valid]),
            Err(CheckpointWitnessError::InvalidStatement)
        ));
    }
}
