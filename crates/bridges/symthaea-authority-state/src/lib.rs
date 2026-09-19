// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Threshold-authenticated current authority state for v0.2 capability records.
//!
//! This crate verifies *state facts*, not execution authority. A successful
//! verification proves that independent witnesses freshly agreed on one exact
//! grant-scoped source frontier, state sequence, current authority epoch,
//! current authority-context state, and complete relevant negative-authority facts.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use symthaea_authority::{
    AuthorityContextRef, AuthorityEpoch, AuthorityEvaluationInput, CapabilityGrant, Digest32,
    GrantUseState, GrantValidationError, NegativeAuthorityFact, PrincipalId, ResourceRef,
};
use symthaea_authority_time::{AuthorityTimeError, VerifiedAuthorityTime};
use thiserror::Error;

pub const AUTHORITY_STATE_SCHEMA_VERSION: u16 = 2;
pub const MAX_AUTHORITY_STATE_WITNESSES: usize = 64;
pub const MAX_AUTHORITY_STATE_STATEMENTS: usize = 128;
pub const MAX_NEGATIVE_FACTS_PER_GRANT: usize = 4096;
pub const MAX_AUTHORITY_IDENTIFIER_BYTES: usize = 1024;
pub const MAX_AUTHORITY_STATE_CHALLENGE_AGE_S: u64 = 60;
pub const MAX_AUTHORITY_STATE_POST_VERIFY_AGE_S: u64 = 60;

const POLICY_DOMAIN: &[u8] = b"symthaea.authority-state.policy.v2\0";
const STATEMENT_DOMAIN: &[u8] = b"symthaea.authority-state.statement.v2\0";
const SNAPSHOT_DOMAIN: &[u8] = b"symthaea.authority-state.snapshot.v2\0";
const NEGATIVE_FACT_DOMAIN: &[u8] = b"symthaea.authority-state.negative-fact.v2\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthorityStateWitnessId(pub [u8; 16]);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedAuthorityStateWitnessV2 {
    pub witness_id: AuthorityStateWitnessId,
    pub verifying_key: [u8; 32],
    pub organization_binding: [u8; 32],
    pub service_binding: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityStatePolicyV2 {
    pub schema_version: u16,
    pub policy_id: [u8; 16],
    pub witnesses: Vec<TrustedAuthorityStateWitnessV2>,
    pub threshold: u16,
    pub minimum_organizations: u16,
    pub maximum_challenge_age_s: u64,
    pub maximum_post_verification_age_s: u64,
}

impl AuthorityStatePolicyV2 {
    pub fn validate(&self) -> Result<(), AuthorityStateError> {
        if self.schema_version != AUTHORITY_STATE_SCHEMA_VERSION
            || self.policy_id == [0; 16]
            || self.witnesses.len() < 2
            || self.witnesses.len() > MAX_AUTHORITY_STATE_WITNESSES
            || self.threshold < 2
            || usize::from(self.threshold) > self.witnesses.len()
            || self.minimum_organizations < 2
            || self.minimum_organizations > self.threshold
            || !(1..=MAX_AUTHORITY_STATE_CHALLENGE_AGE_S)
                .contains(&self.maximum_challenge_age_s)
            || !(1..=MAX_AUTHORITY_STATE_POST_VERIFY_AGE_S)
                .contains(&self.maximum_post_verification_age_s)
        {
            return Err(AuthorityStateError::InvalidPolicy);
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
        let mut t = Transcript::new(POLICY_DOMAIN);
        t.u16(self.schema_version);
        t.fixed(&self.policy_id);
        t.u32(self.witnesses.len())?;
        for witness in &self.witnesses {
            t.fixed(&witness.witness_id.0);
            t.fixed(&witness.verifying_key);
            t.fixed(&witness.organization_binding);
            t.fixed(&witness.service_binding);
        }
        t.u16(self.threshold);
        t.u16(self.minimum_organizations);
        t.u64(self.maximum_challenge_age_s);
        t.u64(self.maximum_post_verification_age_s);
        Ok(t.finish())
    }

    fn witness(&self, id: AuthorityStateWitnessId) -> Option<&TrustedAuthorityStateWitnessV2> {
        self.witnesses.iter().find(|w| w.witness_id == id)
    }
}

/// Wire challenge. It binds the exact grant and policies but deliberately does
/// not contain the epoch, context state, negative facts, or source frontier
/// witnesses are expected to report from their own current source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityStateChallengeV2 {
    pub schema_version: u16,
    pub nonce: [u8; 32],
    pub grant_digest: Digest32,
    pub state_policy_digest: [u8; 32],
    pub time_policy_digest: [u8; 32],
}

#[derive(Debug)]
pub struct PendingAuthorityStateChallengeV2 {
    wire: AuthorityStateChallengeV2,
    created_not_before_unix_s: u64,
}

impl PendingAuthorityStateChallengeV2 {
    pub fn new(
        policy: &AuthorityStatePolicyV2,
        grant: &CapabilityGrant,
        time: &VerifiedAuthorityTime,
    ) -> Result<Self, AuthorityStateError> {
        policy.validate()?;
        grant.validate()?;
        let grant_digest = grant.digest();
        time.require_subject(grant_digest.0)?;
        let _ = time.conservative_now_unix_s()?;
        let (created_not_before_unix_s, _) = time.interval_at_verification();

        let mut nonce = [0; 32];
        getrandom::getrandom(&mut nonce).map_err(|_| AuthorityStateError::RandomnessUnavailable)?;
        if nonce == [0; 32] {
            return Err(AuthorityStateError::RandomnessUnavailable);
        }

        Ok(Self {
            wire: AuthorityStateChallengeV2 {
                schema_version: AUTHORITY_STATE_SCHEMA_VERSION,
                nonce,
                grant_digest,
                state_policy_digest: policy.digest()?,
                time_policy_digest: time.policy_digest(),
            },
            created_not_before_unix_s,
        })
    }

    pub fn wire(&self) -> AuthorityStateChallengeV2 {
        self.wire
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityStateStatementV2 {
    pub schema_version: u16,
    pub witness_id: AuthorityStateWitnessId,
    pub challenge_nonce: [u8; 32],
    pub grant_digest: Digest32,
    pub state_policy_digest: [u8; 32],
    pub time_policy_digest: [u8; 32],
    pub source_frontier_sequence: u64,
    pub source_frontier_digest: Digest32,
    pub state_sequence: u64,
    pub authority_epoch: AuthorityEpoch,
    /// Current domain authority context. `None` means the authoritative source
    /// currently exposes no active context for this grant's authority domain.
    pub authority_context: Option<AuthorityContextRef>,
    pub negative_facts: Vec<NegativeAuthorityFact>,
    pub witness_generation: u64,
    pub signature: Vec<u8>,
}

impl AuthorityStateStatementV2 {
    pub fn canonical_message(&self) -> Result<Vec<u8>, AuthorityStateError> {
        validate_statement_shape(self)?;
        let snapshot = snapshot_digest_v2(
            self.grant_digest,
            self.source_frontier_sequence,
            self.source_frontier_digest,
            self.state_sequence,
            self.authority_epoch,
            self.authority_context.as_ref(),
            &self.negative_facts,
        )?;

        let mut t = Transcript::new(STATEMENT_DOMAIN);
        t.u16(self.schema_version);
        t.fixed(&self.witness_id.0);
        t.fixed(&self.challenge_nonce);
        t.fixed(&self.grant_digest.0);
        t.fixed(&self.state_policy_digest);
        t.fixed(&self.time_policy_digest);
        t.fixed(&snapshot.0);
        t.u64(self.witness_generation);
        Ok(t.bytes)
    }
}

/// Opaque current-state proof. Not Clone, not Serde, and not execution authority.
#[derive(Debug)]
pub struct VerifiedAuthorityStateV2 {
    grant_digest: Digest32,
    source_frontier_sequence: u64,
    source_frontier_digest: Digest32,
    state_sequence: u64,
    authority_epoch: AuthorityEpoch,
    authority_context: Option<AuthorityContextRef>,
    negative_facts: Vec<NegativeAuthorityFact>,
    snapshot_digest: Digest32,
    state_policy_digest: [u8; 32],
    time_policy_digest: [u8; 32],
    verified_not_before_unix_s: u64,
    maximum_post_verification_age_s: u64,
    witness_count: u16,
    organization_count: u16,
}

impl VerifiedAuthorityStateV2 {
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

    pub fn authority_context(&self) -> Option<&AuthorityContextRef> {
        self.authority_context.as_ref()
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

    pub fn ensure_fresh(
        &self,
        grant: &CapabilityGrant,
        time: &VerifiedAuthorityTime,
    ) -> Result<(), AuthorityStateError> {
        grant.validate()?;
        if grant.digest() != self.grant_digest {
            return Err(AuthorityStateError::GrantMismatch);
        }
        time.require_subject(self.grant_digest.0)?;
        if time.policy_digest() != self.time_policy_digest {
            return Err(AuthorityStateError::TimePolicyChanged);
        }
        let age = time
            .conservative_now_unix_s()?
            .checked_sub(self.verified_not_before_unix_s)
            .ok_or(AuthorityStateError::TimeMovedBackward)?;
        if age > self.maximum_post_verification_age_s {
            return Err(AuthorityStateError::VerifiedStateStale);
        }
        Ok(())
    }

    /// Build pure evaluator input from verified current state and separately
    /// supplied crash-conservative use accounting. This still grants no effect.
    /// A verified absence of an active context fails closed rather than being
    /// converted into a placeholder context identity.
    pub fn evaluation_input(
        &self,
        grant: &CapabilityGrant,
        time: &VerifiedAuthorityTime,
        use_state: GrantUseState,
    ) -> Result<AuthorityEvaluationInput, AuthorityStateError> {
        self.ensure_fresh(grant, time)?;
        let current_authority_context = self
            .authority_context
            .clone()
            .ok_or(AuthorityStateError::NoCurrentAuthorityContext)?;
        Ok(AuthorityEvaluationInput {
            now_unix_s: time.conservative_now_unix_s()?,
            current_epoch: self.authority_epoch,
            current_authority_context,
            use_state,
        })
    }
}

pub fn verify_authority_state_v2(
    policy: &AuthorityStatePolicyV2,
    grant: &CapabilityGrant,
    challenge: PendingAuthorityStateChallengeV2,
    time: &VerifiedAuthorityTime,
    statements: &[AuthorityStateStatementV2],
) -> Result<VerifiedAuthorityStateV2, AuthorityStateError> {
    policy.validate()?;
    grant.validate()?;
    let grant_digest = grant.digest();
    if !(usize::from(policy.threshold)..=MAX_AUTHORITY_STATE_STATEMENTS)
        .contains(&statements.len())
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

    time.require_subject(grant_digest.0)?;
    if time.policy_digest() != challenge.wire.time_policy_digest {
        return Err(AuthorityStateError::TimePolicyChangedDuringChallenge);
    }
    let challenge_age = time
        .conservative_now_unix_s()?
        .checked_sub(challenge.created_not_before_unix_s)
        .ok_or(AuthorityStateError::TimeMovedBackward)?;
    if challenge_age > policy.maximum_challenge_age_s {
        return Err(AuthorityStateError::ChallengeExpired);
    }

    let mut ids = BTreeSet::new();
    let mut organizations = BTreeSet::new();
    let mut services = BTreeSet::new();
    let mut snapshot = None;
    let mut frontier = None;
    let mut state_sequence = None;
    let mut epoch = None;
    let mut context = None;
    let mut facts = None;

    for statement in statements {
        validate_statement_shape(statement)?;
        let witness = policy
            .witness(statement.witness_id)
            .ok_or(AuthorityStateError::UnknownWitness)?;
        if statement.challenge_nonce != challenge.wire.nonce
            || statement.grant_digest != grant_digest
            || statement.state_policy_digest != challenge.wire.state_policy_digest
            || statement.time_policy_digest != challenge.wire.time_policy_digest
            || !ids.insert(statement.witness_id)
        {
            return Err(AuthorityStateError::InvalidStatement);
        }

        let canonical_facts = canonical_grant_facts(grant, &statement.negative_facts)?;
        let signature: [u8; 64] = statement
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| AuthorityStateError::BadSignatureLength)?;
        VerifyingKey::from_bytes(&witness.verifying_key)
            .map_err(|_| AuthorityStateError::InvalidPolicy)?
            .verify(
                &statement.canonical_message()?,
                &Signature::from_bytes(&signature),
            )
            .map_err(|_| AuthorityStateError::BadSignature)?;

        agree(
            &mut snapshot,
            snapshot_digest_v2(
                statement.grant_digest,
                statement.source_frontier_sequence,
                statement.source_frontier_digest,
                statement.state_sequence,
                statement.authority_epoch,
                statement.authority_context.as_ref(),
                &statement.negative_facts,
            )?,
        )?;
        agree(
            &mut frontier,
            (
                statement.source_frontier_sequence,
                statement.source_frontier_digest,
            ),
        )?;
        agree(&mut state_sequence, statement.state_sequence)?;
        agree(&mut epoch, statement.authority_epoch)?;
        agree(&mut context, statement.authority_context.clone())?;
        agree(&mut facts, canonical_facts)?;
        organizations.insert(witness.organization_binding);
        services.insert(witness.service_binding);
    }

    if ids.len() < usize::from(policy.threshold)
        || organizations.len() < usize::from(policy.minimum_organizations)
        || services.len() < usize::from(policy.threshold)
    {
        return Err(AuthorityStateError::InsufficientDiversity);
    }

    let (verified_not_before_unix_s, _) = time.interval_at_verification();
    let (source_frontier_sequence, source_frontier_digest) =
        frontier.ok_or(AuthorityStateError::InsufficientStatements)?;

    Ok(VerifiedAuthorityStateV2 {
        grant_digest,
        source_frontier_sequence,
        source_frontier_digest,
        state_sequence: state_sequence.ok_or(AuthorityStateError::InsufficientStatements)?,
        authority_epoch: epoch.ok_or(AuthorityStateError::InsufficientStatements)?,
        authority_context: context.ok_or(AuthorityStateError::InsufficientStatements)?,
        negative_facts: facts.ok_or(AuthorityStateError::InsufficientStatements)?,
        snapshot_digest: snapshot.ok_or(AuthorityStateError::InsufficientStatements)?,
        state_policy_digest: challenge.wire.state_policy_digest,
        time_policy_digest: challenge.wire.time_policy_digest,
        verified_not_before_unix_s,
        maximum_post_verification_age_s: policy.maximum_post_verification_age_s,
        witness_count: u16::try_from(ids.len()).map_err(|_| AuthorityStateError::Encoding)?,
        organization_count: u16::try_from(organizations.len())
            .map_err(|_| AuthorityStateError::Encoding)?,
    })
}

pub fn snapshot_digest_v2(
    grant_digest: Digest32,
    source_frontier_sequence: u64,
    source_frontier_digest: Digest32,
    state_sequence: u64,
    authority_epoch: AuthorityEpoch,
    authority_context: Option<&AuthorityContextRef>,
    negative_facts: &[NegativeAuthorityFact],
) -> Result<Digest32, AuthorityStateError> {
    if grant_digest.0 == [0; 32]
        || source_frontier_sequence == 0
        || source_frontier_digest.0 == [0; 32]
        || state_sequence == 0
        || authority_epoch.0 == 0
    {
        return Err(AuthorityStateError::InvalidStatement);
    }
    if let Some(context) = authority_context {
        validate_context(context)?;
    }
    let canonical = canonical_fact_set(negative_facts)?;
    let mut t = Transcript::new(SNAPSHOT_DOMAIN);
    t.u16(AUTHORITY_STATE_SCHEMA_VERSION);
    t.fixed(&grant_digest.0);
    t.u64(source_frontier_sequence);
    t.fixed(&source_frontier_digest.0);
    t.u64(state_sequence);
    t.u64(authority_epoch.0);
    match authority_context {
        Some(context) => {
            t.byte(1);
            t.string(&context.namespace.0)?;
            t.fixed(&context.digest.0);
        }
        None => t.byte(0),
    }
    t.u32(canonical.len())?;
    for (digest, _) in canonical {
        t.fixed(&digest.0);
    }
    Ok(Digest32(t.finish()))
}

pub fn negative_fact_digest_v2(
    fact: &NegativeAuthorityFact,
) -> Result<Digest32, AuthorityStateError> {
    let mut t = Transcript::new(NEGATIVE_FACT_DOMAIN);
    t.u16(AUTHORITY_STATE_SCHEMA_VERSION);
    match fact {
        NegativeAuthorityFact::RevokeGrant { grant_digest } => {
            if grant_digest.0 == [0; 32] {
                return Err(AuthorityStateError::InvalidNegativeFact);
            }
            t.byte(1);
            t.fixed(&grant_digest.0);
        }
        NegativeAuthorityFact::RevokeContext { context } => {
            validate_context(context)?;
            t.byte(2);
            t.string(&context.namespace.0)?;
            t.fixed(&context.digest.0);
        }
        NegativeAuthorityFact::TombstonePrincipal { principal } => {
            validate_principal(principal)?;
            t.byte(3);
            t.string(&principal.0)?;
        }
        NegativeAuthorityFact::FreezeResource { resource } => {
            validate_resource(resource)?;
            t.byte(4);
            t.string(&resource.0)?;
        }
        NegativeAuthorityFact::MinimumResourceEpoch {
            resource,
            minimum_epoch,
        } => {
            validate_resource(resource)?;
            if minimum_epoch.0 == 0 {
                return Err(AuthorityStateError::InvalidNegativeFact);
            }
            t.byte(5);
            t.string(&resource.0)?;
            t.u64(minimum_epoch.0);
        }
    }
    Ok(Digest32(t.finish()))
}

fn validate_statement_shape(statement: &AuthorityStateStatementV2) -> Result<(), AuthorityStateError> {
    if statement.schema_version != AUTHORITY_STATE_SCHEMA_VERSION
        || statement.witness_id.0 == [0; 16]
        || statement.challenge_nonce == [0; 32]
        || statement.grant_digest.0 == [0; 32]
        || statement.state_policy_digest == [0; 32]
        || statement.time_policy_digest == [0; 32]
        || statement.source_frontier_sequence == 0
        || statement.source_frontier_digest.0 == [0; 32]
        || statement.state_sequence == 0
        || statement.authority_epoch.0 == 0
        || statement.witness_generation == 0
    {
        return Err(AuthorityStateError::InvalidStatement);
    }
    if let Some(context) = statement.authority_context.as_ref() {
        validate_context(context)?;
    }
    Ok(())
}

fn agree<T: PartialEq>(slot: &mut Option<T>, value: T) -> Result<(), AuthorityStateError> {
    match slot {
        None => {
            *slot = Some(value);
            Ok(())
        }
        Some(expected) if expected == &value => Ok(()),
        Some(_) => Err(AuthorityStateError::StateDisagreement),
    }
}

fn canonical_grant_facts(
    grant: &CapabilityGrant,
    facts: &[NegativeAuthorityFact],
) -> Result<Vec<NegativeAuthorityFact>, AuthorityStateError> {
    let canonical = canonical_fact_set(facts)?;
    if canonical
        .iter()
        .any(|(_, fact)| !fact_relevant_to_grant(grant, fact))
    {
        return Err(AuthorityStateError::IrrelevantNegativeFact);
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
        let digest = negative_fact_digest_v2(fact)?;
        if by_digest.insert(digest, fact.clone()).is_some() {
            return Err(AuthorityStateError::DuplicateNegativeFact);
        }
    }
    Ok(by_digest.into_iter().collect())
}

fn fact_relevant_to_grant(grant: &CapabilityGrant, fact: &NegativeAuthorityFact) -> bool {
    match fact {
        NegativeAuthorityFact::RevokeGrant { grant_digest } => *grant_digest == grant.digest(),
        NegativeAuthorityFact::RevokeContext { context } => context == &grant.authority_context,
        NegativeAuthorityFact::TombstonePrincipal { principal } => {
            principal == &grant.issuer
                || principal == &grant.subject
                || grant.audience.as_ref() == Some(principal)
        }
        NegativeAuthorityFact::FreezeResource { resource }
        | NegativeAuthorityFact::MinimumResourceEpoch { resource, .. } => {
            grant.resources.contains(resource)
        }
    }
}

fn validate_context(context: &AuthorityContextRef) -> Result<(), AuthorityStateError> {
    validate_identifier(&context.namespace.0)?;
    if context.digest.0 == [0; 32] {
        return Err(AuthorityStateError::InvalidAuthorityContext);
    }
    Ok(())
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

    fn u32(&mut self, value: usize) -> Result<(), AuthorityStateError> {
        let value = u32::try_from(value).map_err(|_| AuthorityStateError::Encoding)?;
        self.bytes.extend_from_slice(&value.to_be_bytes());
        Ok(())
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn fixed<const N: usize>(&mut self, value: &[u8; N]) {
        self.bytes.extend_from_slice(value);
    }

    fn string(&mut self, value: &str) -> Result<(), AuthorityStateError> {
        validate_identifier(value)?;
        self.u32(value.len())?;
        self.bytes.extend_from_slice(value.as_bytes());
        Ok(())
    }

    fn finish(self) -> [u8; 32] {
        *blake3::hash(&self.bytes).as_bytes()
    }
}

#[derive(Debug, Error)]
pub enum AuthorityStateError {
    #[error("authority-state witness policy is invalid")]
    InvalidPolicy,
    #[error("capability grant is invalid: {0}")]
    InvalidGrant(#[from] GrantValidationError),
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
    #[error("authority-state authority context is malformed")]
    InvalidAuthorityContext,
    #[error("verified current authority state reports no active authority context")]
    NoCurrentAuthorityContext,
    #[error("negative-authority fact is malformed")]
    InvalidNegativeFact,
    #[error("authority-state Ed25519 signature must be exactly 64 bytes")]
    BadSignatureLength,
    #[error("authority-state witness signature verification failed")]
    BadSignature,
    #[error("authority-state witnesses do not satisfy organizational/service diversity")]
    InsufficientDiversity,
    #[error("fresh authority-state witnesses disagree on frontier, sequence, epoch, context state, or negatives")]
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
    #[error("authority identifier is empty or exceeds the canonical bound")]
    InvalidIdentifier,
    #[error("trusted authority time failed: {0}")]
    AuthorityTime(#[from] AuthorityTimeError),
    #[error("authority-state canonical encoding failed")]
    Encoding,
}

#[cfg(test)]
mod tests;
