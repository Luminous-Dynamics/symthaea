// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh, grant-scoped authority-state evidence for bounded Symthaea agency.
//!
//! A capability cannot be evaluated safely from a caller-provided epoch plus a
//! caller-provided list of revocations: those two values can come from different
//! moments, and either can be stale. V1 therefore authenticates them as one
//! indivisible state snapshot.
//!
//! Independent state witnesses answer a fresh challenge with the authority
//! epoch and the complete set of negative-authority facts relevant to one exact
//! grant at the source frontier they currently observe. The client does not ask
//! witnesses to sign a caller-selected snapshot. Threshold agreement,
//! organizational/service diversity, challenge freshness, and exact canonical
//! commitments are all required before an opaque [`VerifiedAuthorityState`] is
//! produced.
//!
//! The witness-service contract is intentionally strong: for the challenged
//! grant, every response must contain every currently applicable negative fact
//! from the witness's authoritative source. A witness that omits a revocation is
//! faulty. Threshold diversity limits dependence on any one faulty witness.
//!
//! This crate creates no capability and exposes no execution API.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use symthaea_authority::{
    AuthorityEpoch, CapabilityGrant, Digest32, NegativeAuthorityFact, PrincipalId, ResourceRef,
};
use symthaea_authority_time::{AuthorityTimeError, VerifiedAuthorityTime};
use thiserror::Error;

pub const AUTHORITY_STATE_SCHEMA_VERSION: u16 = 1;
pub const MAX_AUTHORITY_STATE_WITNESSES: usize = 64;
pub const MAX_AUTHORITY_STATE_STATEMENTS: usize = 128;
pub const MAX_NEGATIVE_FACTS_PER_GRANT: usize = 4096;
pub const MAX_AUTHORITY_IDENTIFIER_BYTES: usize = 1024;
pub const MAX_AUTHORITY_STATE_CHALLENGE_AGE_S: u64 = 60;
pub const MAX_AUTHORITY_STATE_POST_VERIFY_AGE_S: u64 = 60;

const POLICY_DOMAIN: &[u8] = b"symthaea.authority-state.policy.v1\0";
const CHALLENGE_DOMAIN: &[u8] = b"symthaea.authority-state.challenge.v1\0";
const STATEMENT_DOMAIN: &[u8] = b"symthaea.authority-state.statement.v1\0";
const SNAPSHOT_DOMAIN: &[u8] = b"symthaea.authority-state.snapshot.v1\0";
const NEGATIVE_FACT_DOMAIN: &[u8] = b"symthaea.authority-state.negative-fact.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthorityStateWitnessId(pub [u8; 16]);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedAuthorityStateWitnessV1 {
    pub witness_id: AuthorityStateWitnessId,
    pub verifying_key: [u8; 32],
    pub organization_binding: [u8; 32],
    pub service_binding: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityStatePolicyV1 {
    pub schema_version: u16,
    pub policy_id: [u8; 16],
    pub witnesses: Vec<TrustedAuthorityStateWitnessV1>,
    pub threshold: u16,
    pub minimum_organizations: u16,
    /// Maximum worst-case elapsed time from challenge creation to verification.
    pub maximum_challenge_age_s: u64,
    /// Maximum worst-case age of a verified state fact at point of use.
    pub maximum_post_verification_age_s: u64,
}

impl AuthorityStatePolicyV1 {
    pub fn validate(&self) -> Result<(), AuthorityStateError> {
        if self.schema_version != AUTHORITY_STATE_SCHEMA_VERSION
            || self.policy_id == [0; 16]
            || self.witnesses.len() < 2
            || self.witnesses.len() > MAX_AUTHORITY_STATE_WITNESSES
            || self.threshold < 2
            || usize::from(self.threshold) > self.witnesses.len()
            || self.minimum_organizations < 2
            || self.minimum_organizations > self.threshold
            || self.maximum_challenge_age_s == 0
            || self.maximum_challenge_age_s > MAX_AUTHORITY_STATE_CHALLENGE_AGE_S
            || self.maximum_post_verification_age_s == 0
            || self.maximum_post_verification_age_s > MAX_AUTHORITY_STATE_POST_VERIFY_AGE_S
        {
            return Err(AuthorityStateError::InvalidPolicy);
        }

        let mut ids = BTreeSet::new();
        let mut keys = BTreeSet::new();
        let mut organizations = BTreeSet::new();
        let mut services = BTreeSet::new();
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
                return Err(AuthorityStateError::InvalidPolicy);
            }
            organizations.insert(witness.organization_binding);
        }
        if organizations.len() < usize::from(self.minimum_organizations) {
            return Err(AuthorityStateError::InvalidPolicy);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<[u8; 32], AuthorityStateError> {
        self.validate()?;
        let mut transcript = Transcript::new(POLICY_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.policy_id);
        transcript.u32(
            u32::try_from(self.witnesses.len()).map_err(|_| AuthorityStateError::Encoding)?,
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
        transcript.u64(self.maximum_post_verification_age_s);
        Ok(transcript.finish())
    }

    fn witness(&self, id: AuthorityStateWitnessId) -> Option<&TrustedAuthorityStateWitnessV1> {
        self.witnesses.iter().find(|witness| witness.witness_id == id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityStateChallengeV1 {
    pub schema_version: u16,
    pub nonce: [u8; 32],
    pub grant_digest: Digest32,
    pub state_policy_digest: [u8; 32],
    pub time_policy_digest: [u8; 32],
}

impl AuthorityStateChallengeV1 {
    pub fn digest(&self) -> [u8; 32] {
        let mut transcript = Transcript::new(CHALLENGE_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.nonce);
        transcript.fixed(&self.grant_digest.0);
        transcript.fixed(&self.state_policy_digest);
        transcript.fixed(&self.time_policy_digest);
        transcript.finish()
    }
}

/// Local state for one fresh authority-state query.
///
/// The client does not include an epoch, revocation set, or source frontier in
/// the challenge. Witnesses must answer from their own current authoritative
/// state. This prevents a caller from pinning witnesses to an old-but-valid
/// snapshot.
#[derive(Debug)]
pub struct PendingAuthorityStateChallenge {
    wire: AuthorityStateChallengeV1,
    created_not_before_unix_s: u64,
}

impl PendingAuthorityStateChallenge {
    pub fn new(
        policy: &AuthorityStatePolicyV1,
        grant: &CapabilityGrant,
        authority_time: &VerifiedAuthorityTime,
    ) -> Result<Self, AuthorityStateError> {
        policy.validate()?;
        let grant_digest = grant.digest();
        authority_time.require_subject(grant_digest.0)?;
        // Force the supplied time fact through its boot-time freshness check.
        let _ = authority_time.conservative_now_unix_s()?;
        let (created_not_before_unix_s, _) = authority_time.interval_at_verification();

        let mut nonce = [0u8; 32];
        getrandom::getrandom(&mut nonce).map_err(|_| AuthorityStateError::RandomnessUnavailable)?;
        if nonce == [0; 32] {
            return Err(AuthorityStateError::RandomnessUnavailable);
        }

        Ok(Self {
            wire: AuthorityStateChallengeV1 {
                schema_version: AUTHORITY_STATE_SCHEMA_VERSION,
                nonce,
                grant_digest,
                state_policy_digest: policy.digest()?,
                time_policy_digest: authority_time.policy_digest(),
            },
            created_not_before_unix_s,
        })
    }

    pub fn wire(&self) -> AuthorityStateChallengeV1 {
        self.wire
    }
}

/// One witness's answer from its current grant-scoped authority state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityStateStatementV1 {
    pub schema_version: u16,
    pub witness_id: AuthorityStateWitnessId,
    pub challenge_nonce: [u8; 32],
    pub grant_digest: Digest32,
    pub state_policy_digest: [u8; 32],
    pub time_policy_digest: [u8; 32],
    /// Monotonic sequence of the authoritative source frontier (for example a
    /// Xenia ledger entry count).
    pub source_frontier_sequence: u64,
    /// Exact source frontier commitment (for example a Xenia ledger head hash).
    pub source_frontier_digest: Digest32,
    /// Monotonic authority-state snapshot sequence maintained by the source.
    pub state_sequence: u64,
    pub authority_epoch: AuthorityEpoch,
    /// Complete negative-authority fact set relevant to the exact grant.
    pub negative_facts: Vec<NegativeAuthorityFact>,
    /// Witness service's own durable monotonic generation.
    pub witness_generation: u64,
    pub signature: Vec<u8>,
}

impl AuthorityStateStatementV1 {
    pub fn snapshot_digest(&self) -> Result<Digest32, AuthorityStateError> {
        snapshot_digest_v1(
            self.grant_digest,
            self.source_frontier_sequence,
            self.source_frontier_digest,
            self.state_sequence,
            self.authority_epoch,
            &self.negative_facts,
        )
    }

    pub fn canonical_message(&self) -> Result<Vec<u8>, AuthorityStateError> {
        if self.schema_version != AUTHORITY_STATE_SCHEMA_VERSION
            || self.witness_id.0 == [0; 16]
            || self.challenge_nonce == [0; 32]
            || self.grant_digest.0 == [0; 32]
            || self.state_policy_digest == [0; 32]
            || self.time_policy_digest == [0; 32]
            || self.source_frontier_sequence == 0
            || self.source_frontier_digest.0 == [0; 32]
            || self.state_sequence == 0
            || self.witness_generation == 0
        {
            return Err(AuthorityStateError::InvalidStatement);
        }
        let snapshot_digest = self.snapshot_digest()?;
        let mut transcript = Transcript::new(STATEMENT_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.witness_id.0);
        transcript.fixed(&self.challenge_nonce);
        transcript.fixed(&self.grant_digest.0);
        transcript.fixed(&self.state_policy_digest);
        transcript.fixed(&self.time_policy_digest);
        transcript.fixed(&snapshot_digest.0);
        transcript.u64(self.witness_generation);
        Ok(transcript.into_bytes())
    }
}

/// Opaque proof that independent witnesses freshly agreed on one indivisible
/// epoch + negative-fact snapshot for an exact grant.
///
/// This type intentionally does not implement `Clone`. It is not authority by
/// itself; it is an authenticated environmental fact consumed by admission.
#[derive(Debug)]
pub struct VerifiedAuthorityState {
    grant_digest: Digest32,
    source_frontier_sequence: u64,
    source_frontier_digest: Digest32,
    state_sequence: u64,
    authority_epoch: AuthorityEpoch,
    negative_facts: Vec<NegativeAuthorityFact>,
    snapshot_digest: Digest32,
    state_policy_digest: [u8; 32],
    time_policy_digest: [u8; 32],
    verified_not_before_unix_s: u64,
    maximum_post_verification_age_s: u64,
    witness_count: u16,
    organization_count: u16,
}

impl VerifiedAuthorityState {
    pub fn grant_digest(&self) -> Digest32 {
        self.grant_digest
    }

    pub fn source_frontier(&self) -> (u64, Digest32) {
        (self.source_frontier_sequence, self.source_frontier_digest)
    }

    pub fn state_sequence(&self) -> u64 {
        self.state_sequence
    }

    pub fn authority_epoch(&self) -> AuthorityEpoch {
        self.authority_epoch
    }

    pub fn negative_facts(&self) -> &[NegativeAuthorityFact] {
        &self.negative_facts
    }

    pub fn snapshot_digest(&self) -> Digest32 {
        self.snapshot_digest
    }

    pub fn state_policy_digest(&self) -> [u8; 32] {
        self.state_policy_digest
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

    pub fn require_grant(&self, grant: &CapabilityGrant) -> Result<(), AuthorityStateError> {
        if self.grant_digest != grant.digest() {
            return Err(AuthorityStateError::GrantMismatch);
        }
        Ok(())
    }

    /// Require this state fact to remain within its short worst-case freshness
    /// window under a current trusted-time fact for the same grant and policy.
    pub fn ensure_fresh(
        &self,
        grant: &CapabilityGrant,
        authority_time: &VerifiedAuthorityTime,
    ) -> Result<(), AuthorityStateError> {
        self.require_grant(grant)?;
        authority_time.require_subject(self.grant_digest.0)?;
        if authority_time.policy_digest() != self.time_policy_digest {
            return Err(AuthorityStateError::TimePolicyChanged);
        }
        let current_not_after = authority_time.conservative_now_unix_s()?;
        let worst_case_age = current_not_after
            .checked_sub(self.verified_not_before_unix_s)
            .ok_or(AuthorityStateError::TimeMovedBackward)?;
        if worst_case_age > self.maximum_post_verification_age_s {
            return Err(AuthorityStateError::VerifiedStateStale);
        }
        Ok(())
    }
}

/// Verify fresh, threshold-authenticated authority state for one exact grant.
pub fn verify_authority_state_v1(
    policy: &AuthorityStatePolicyV1,
    grant: &CapabilityGrant,
    challenge: PendingAuthorityStateChallenge,
    response_time: &VerifiedAuthorityTime,
    statements: &[AuthorityStateStatementV1],
) -> Result<VerifiedAuthorityState, AuthorityStateError> {
    policy.validate()?;
    let grant_digest = grant.digest();
    if statements.len() < usize::from(policy.threshold)
        || statements.len() > MAX_AUTHORITY_STATE_STATEMENTS
    {
        return Err(AuthorityStateError::InsufficientStatements);
    }
    if challenge.wire.schema_version != AUTHORITY_STATE_SCHEMA_VERSION
        || challenge.wire.grant_digest != grant_digest
        || challenge.wire.state_policy_digest != policy.digest()?
        || challenge.wire.nonce == [0; 32]
    {
        return Err(AuthorityStateError::InvalidChallenge);
    }

    response_time.require_subject(grant_digest.0)?;
    if response_time.policy_digest() != challenge.wire.time_policy_digest {
        return Err(AuthorityStateError::TimePolicyChangedDuringChallenge);
    }
    let response_not_after = response_time.conservative_now_unix_s()?;
    let worst_case_challenge_age = response_not_after
        .checked_sub(challenge.created_not_before_unix_s)
        .ok_or(AuthorityStateError::TimeMovedBackward)?;
    if worst_case_challenge_age > policy.maximum_challenge_age_s {
        return Err(AuthorityStateError::ChallengeExpired);
    }

    let mut ids = BTreeSet::new();
    let mut organizations = BTreeSet::new();
    let mut services = BTreeSet::new();
    let mut agreed_snapshot: Option<Digest32> = None;
    let mut agreed_source_frontier: Option<(u64, Digest32)> = None;
    let mut agreed_state_sequence: Option<u64> = None;
    let mut agreed_epoch: Option<AuthorityEpoch> = None;
    let mut agreed_facts: Option<Vec<NegativeAuthorityFact>> = None;

    for statement in statements {
        let witness = policy
            .witness(statement.witness_id)
            .ok_or(AuthorityStateError::UnknownWitness)?;
        if statement.schema_version != AUTHORITY_STATE_SCHEMA_VERSION
            || statement.challenge_nonce != challenge.wire.nonce
            || statement.grant_digest != grant_digest
            || statement.state_policy_digest != challenge.wire.state_policy_digest
            || statement.time_policy_digest != challenge.wire.time_policy_digest
            || statement.source_frontier_sequence == 0
            || statement.source_frontier_digest.0 == [0; 32]
            || statement.state_sequence == 0
            || statement.witness_generation == 0
            || !ids.insert(statement.witness_id)
        {
            return Err(AuthorityStateError::InvalidStatement);
        }

        let canonical_facts = canonical_grant_facts(grant, &statement.negative_facts)?;
        let signature_bytes: [u8; 64] = statement
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| AuthorityStateError::BadSignatureLength)?;
        let key = VerifyingKey::from_bytes(&witness.verifying_key)
            .map_err(|_| AuthorityStateError::InvalidPolicy)?;
        key.verify(
            &statement.canonical_message()?,
            &Signature::from_bytes(&signature_bytes),
        )
        .map_err(|_| AuthorityStateError::BadSignature)?;

        let snapshot = statement.snapshot_digest()?;
        match agreed_snapshot {
            None => agreed_snapshot = Some(snapshot),
            Some(expected) if expected == snapshot => {}
            Some(_) => return Err(AuthorityStateError::StateDisagreement),
        }

        let source_frontier = (
            statement.source_frontier_sequence,
            statement.source_frontier_digest,
        );
        match agreed_source_frontier {
            None => agreed_source_frontier = Some(source_frontier),
            Some(expected) if expected == source_frontier => {}
            Some(_) => return Err(AuthorityStateError::StateDisagreement),
        }
        match agreed_state_sequence {
            None => agreed_state_sequence = Some(statement.state_sequence),
            Some(expected) if expected == statement.state_sequence => {}
            Some(_) => return Err(AuthorityStateError::StateDisagreement),
        }
        match agreed_epoch {
            None => agreed_epoch = Some(statement.authority_epoch),
            Some(expected) if expected == statement.authority_epoch => {}
            Some(_) => return Err(AuthorityStateError::StateDisagreement),
        }
        match &agreed_facts {
            None => agreed_facts = Some(canonical_facts),
            Some(expected) if expected == &canonical_facts => {}
            Some(_) => return Err(AuthorityStateError::StateDisagreement),
        }

        organizations.insert(witness.organization_binding);
        services.insert(witness.service_binding);
    }

    if ids.len() < usize::from(policy.threshold)
        || organizations.len() < usize::from(policy.minimum_organizations)
        || services.len() < usize::from(policy.threshold)
    {
        return Err(AuthorityStateError::InsufficientDiversity);
    }

    let (verified_not_before_unix_s, _) = response_time.interval_at_verification();
    let (source_frontier_sequence, source_frontier_digest) =
        agreed_source_frontier.ok_or(AuthorityStateError::InsufficientStatements)?;

    Ok(VerifiedAuthorityState {
        grant_digest,
        source_frontier_sequence,
        source_frontier_digest,
        state_sequence: agreed_state_sequence.ok_or(AuthorityStateError::InsufficientStatements)?,
        authority_epoch: agreed_epoch.ok_or(AuthorityStateError::InsufficientStatements)?,
        negative_facts: agreed_facts.ok_or(AuthorityStateError::InsufficientStatements)?,
        snapshot_digest: agreed_snapshot.ok_or(AuthorityStateError::InsufficientStatements)?,
        state_policy_digest: challenge.wire.state_policy_digest,
        time_policy_digest: challenge.wire.time_policy_digest,
        verified_not_before_unix_s,
        maximum_post_verification_age_s: policy.maximum_post_verification_age_s,
        witness_count: u16::try_from(ids.len()).map_err(|_| AuthorityStateError::Encoding)?,
        organization_count: u16::try_from(organizations.len())
            .map_err(|_| AuthorityStateError::Encoding)?,
    })
}

/// Domain-separated commitment to one negative-authority fact.
pub fn negative_fact_digest_v1(fact: &NegativeAuthorityFact) -> Result<Digest32, AuthorityStateError> {
    let mut transcript = Transcript::new(NEGATIVE_FACT_DOMAIN);
    transcript.u16(AUTHORITY_STATE_SCHEMA_VERSION);
    match fact {
        NegativeAuthorityFact::RevokeGrant { grant_digest } => {
            transcript.byte(1);
            transcript.fixed(&grant_digest.0);
        }
        NegativeAuthorityFact::TombstonePrincipal { principal } => {
            transcript.byte(2);
            transcript.string(&principal.0)?;
        }
        NegativeAuthorityFact::FreezeResource { resource } => {
            transcript.byte(3);
            transcript.string(&resource.0)?;
        }
        NegativeAuthorityFact::MinimumResourceEpoch {
            resource,
            minimum_epoch,
        } => {
            transcript.byte(4);
            transcript.string(&resource.0)?;
            transcript.u64(minimum_epoch.0);
        }
    }
    Ok(Digest32(transcript.finish()))
}

pub fn snapshot_digest_v1(
    grant_digest: Digest32,
    source_frontier_sequence: u64,
    source_frontier_digest: Digest32,
    state_sequence: u64,
    authority_epoch: AuthorityEpoch,
    negative_facts: &[NegativeAuthorityFact],
) -> Result<Digest32, AuthorityStateError> {
    if grant_digest.0 == [0; 32]
        || source_frontier_sequence == 0
        || source_frontier_digest.0 == [0; 32]
        || state_sequence == 0
    {
        return Err(AuthorityStateError::InvalidStatement);
    }
    let canonical = canonical_fact_set(negative_facts)?;
    let mut transcript = Transcript::new(SNAPSHOT_DOMAIN);
    transcript.u16(AUTHORITY_STATE_SCHEMA_VERSION);
    transcript.fixed(&grant_digest.0);
    transcript.u64(source_frontier_sequence);
    transcript.fixed(&source_frontier_digest.0);
    transcript.u64(state_sequence);
    transcript.u64(authority_epoch.0);
    transcript.u32(u32::try_from(canonical.len()).map_err(|_| AuthorityStateError::Encoding)?);
    for (digest, _) in canonical {
        transcript.fixed(&digest.0);
    }
    Ok(Digest32(transcript.finish()))
}

fn canonical_grant_facts(
    grant: &CapabilityGrant,
    facts: &[NegativeAuthorityFact],
) -> Result<Vec<NegativeAuthorityFact>, AuthorityStateError> {
    let canonical = canonical_fact_set(facts)?;
    for (_, fact) in &canonical {
        if !fact_relevant_to_grant(grant, fact) {
            return Err(AuthorityStateError::IrrelevantNegativeFact);
        }
    }
    Ok(canonical.into_iter().map(|(_, fact)| fact).collect())
}

fn canonical_fact_set(
    facts: &[NegativeAuthorityFact],
) -> Result<Vec<(Digest32, NegativeAuthorityFact)>, AuthorityStateError> {
    if facts.len() > MAX_NEGATIVE_FACTS_PER_GRANT {
        return Err(AuthorityStateError::TooManyNegativeFacts);
    }
    let mut by_digest = BTreeMap::new();
    for fact in facts {
        validate_fact_identifiers(fact)?;
        let digest = negative_fact_digest_v1(fact)?;
        if by_digest.insert(digest, fact.clone()).is_some() {
            return Err(AuthorityStateError::DuplicateNegativeFact);
        }
    }
    Ok(by_digest.into_iter().collect())
}

fn validate_fact_identifiers(fact: &NegativeAuthorityFact) -> Result<(), AuthorityStateError> {
    match fact {
        NegativeAuthorityFact::RevokeGrant { .. } => Ok(()),
        NegativeAuthorityFact::TombstonePrincipal { principal } => validate_principal(principal),
        NegativeAuthorityFact::FreezeResource { resource }
        | NegativeAuthorityFact::MinimumResourceEpoch { resource, .. } => validate_resource(resource),
    }
}

fn validate_principal(principal: &PrincipalId) -> Result<(), AuthorityStateError> {
    validate_identifier(&principal.0)
}

fn validate_resource(resource: &ResourceRef) -> Result<(), AuthorityStateError> {
    validate_identifier(&resource.0)
}

fn validate_identifier(value: &str) -> Result<(), AuthorityStateError> {
    if value.is_empty() || value.len() > MAX_AUTHORITY_IDENTIFIER_BYTES {
        return Err(AuthorityStateError::InvalidIdentifier);
    }
    Ok(())
}

fn fact_relevant_to_grant(grant: &CapabilityGrant, fact: &NegativeAuthorityFact) -> bool {
    match fact {
        NegativeAuthorityFact::RevokeGrant { grant_digest } => *grant_digest == grant.digest(),
        NegativeAuthorityFact::TombstonePrincipal { principal } => {
            *principal == grant.subject || grant.audience.as_ref() == Some(principal)
        }
        NegativeAuthorityFact::FreezeResource { resource } => grant.resources.contains(resource),
        NegativeAuthorityFact::MinimumResourceEpoch { resource, .. } => {
            grant.resources.contains(resource)
        }
    }
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

    fn byte(&mut self, value: u8) {
        self.bytes.push(value);
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

    fn string(&mut self, value: &str) -> Result<(), AuthorityStateError> {
        validate_identifier(value)?;
        let len = u32::try_from(value.len()).map_err(|_| AuthorityStateError::Encoding)?;
        self.u32(len);
        self.bytes.extend_from_slice(value.as_bytes());
        Ok(())
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    fn finish(self) -> [u8; 32] {
        *blake3::hash(&self.bytes).as_bytes()
    }
}

#[derive(Debug, Error)]
pub enum AuthorityStateError {
    #[error("authority-state witness policy is invalid")]
    InvalidPolicy,
    #[error("secure authority-state challenge randomness is unavailable")]
    RandomnessUnavailable,
    #[error("authority-state challenge is invalid")]
    InvalidChallenge,
    #[error("not enough authority-state witness statements were supplied")]
    InsufficientStatements,
    #[error("authority-state statement references an unknown witness")]
    UnknownWitness,
    #[error("authority-state witness statement is malformed or does not bind the challenge")]
    InvalidStatement,
    #[error("authority-state Ed25519 signature must be exactly 64 bytes")]
    BadSignatureLength,
    #[error("authority-state witness signature verification failed")]
    BadSignature,
    #[error("authority-state witnesses do not satisfy organizational/service diversity")]
    InsufficientDiversity,
    #[error("fresh authority-state witnesses disagree on epoch, revocations, or source frontier")]
    StateDisagreement,
    #[error("verified authority state belongs to a different capability grant")]
    GrantMismatch,
    #[error("trusted-time policy changed while authority-state challenge was in flight")]
    TimePolicyChangedDuringChallenge,
    #[error("trusted-time policy changed after authority-state verification")]
    TimePolicyChanged,
    #[error("authority-state challenge exceeded its maximum worst-case age")]
    ChallengeExpired,
    #[error("verified authority state exceeded its maximum worst-case age")]
    VerifiedStateStale,
    #[error("trusted time moved backward across authority-state verification")]
    TimeMovedBackward,
    #[error("negative-authority fact set exceeded the per-grant bound")]
    TooManyNegativeFacts,
    #[error("negative-authority fact set contains a duplicate commitment")]
    DuplicateNegativeFact,
    #[error("negative-authority fact is not relevant to the challenged grant")]
    IrrelevantNegativeFact,
    #[error("authority principal/resource identifier is empty or exceeds the canonical bound")]
    InvalidIdentifier,
    #[error("trusted authority time failed: {0}")]
    AuthorityTime(#[from] AuthorityTimeError),
    #[error("authority-state canonical encoding failed")]
    Encoding,
}
