// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lineage-bound successor to verified authority state v0.2.
//!
//! V2 proves fresh threshold agreement on an exact source-frontier sequence and
//! digest, but does not identify the causal lineage in which that sequence has
//! meaning. V3 keeps the V2 witness/time policy semantics while assigning new
//! challenge, statement, and snapshot domains and binding an explicit source
//! lineage/incarnation into every signed authority-state snapshot.
//!
//! A lineage transition is deliberately *not* inferred here. Different lineages
//! are incomparable until a later migration/recovery theorem supplies an
//! explicit transition receipt.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use symthaea_authority::{
    AuthorityContextRef, AuthorityEpoch, CapabilityGrant, Digest32, GrantValidationError,
    NegativeAuthorityFact, PrincipalId, ResourceRef,
};
use symthaea_authority_state::{
    negative_fact_digest_v2, AuthorityStateError, AuthorityStatePolicyV2, AuthorityStateWitnessId,
    MAX_AUTHORITY_IDENTIFIER_BYTES, MAX_AUTHORITY_STATE_STATEMENTS, MAX_NEGATIVE_FACTS_PER_GRANT,
};
use symthaea_authority_time::{AuthorityTimeError, VerifiedAuthorityTime};
use thiserror::Error;

pub const AUTHORITY_STATE_LINEAGE_SCHEMA_VERSION: u16 = 3;

const CHALLENGE_DOMAIN_V3: &[u8] = b"symthaea.authority-state.challenge.v3\0";
const STATEMENT_DOMAIN_V3: &[u8] = b"symthaea.authority-state.statement.v3\0";
const SNAPSHOT_DOMAIN_V3: &[u8] = b"symthaea.authority-state.snapshot.v3\0";

/// Exact causal lineage/incarnation in which a source-frontier sequence is ordered.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthorityFrontierLineageV1 {
    pub namespace: String,
    pub incarnation: Digest32,
}

impl AuthorityFrontierLineageV1 {
    pub fn new(
        namespace: impl Into<String>,
        incarnation: Digest32,
    ) -> Result<Self, AuthorityStateLineageError> {
        let lineage = Self {
            namespace: namespace.into(),
            incarnation,
        };
        lineage.validate()?;
        Ok(lineage)
    }

    pub fn validate(&self) -> Result<(), AuthorityStateLineageError> {
        validate_identifier(&self.namespace)?;
        if self.incarnation.0 == [0; 32] {
            return Err(AuthorityStateLineageError::InvalidFrontierLineage);
        }
        Ok(())
    }
}

/// Fresh wire challenge for the lineage-bound V3 protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityStateChallengeV3 {
    pub schema_version: u16,
    pub nonce: [u8; 32],
    pub grant_digest: Digest32,
    pub state_policy_digest: [u8; 32],
    pub time_policy_digest: [u8; 32],
}

impl AuthorityStateChallengeV3 {
    pub fn digest(&self) -> Result<Digest32, AuthorityStateLineageError> {
        if self.schema_version != AUTHORITY_STATE_LINEAGE_SCHEMA_VERSION
            || self.nonce == [0; 32]
            || self.grant_digest.0 == [0; 32]
            || self.state_policy_digest == [0; 32]
            || self.time_policy_digest == [0; 32]
        {
            return Err(AuthorityStateLineageError::InvalidChallenge);
        }
        let mut transcript = Transcript::new(CHALLENGE_DOMAIN_V3);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.nonce);
        transcript.fixed(&self.grant_digest.0);
        transcript.fixed(&self.state_policy_digest);
        transcript.fixed(&self.time_policy_digest);
        Ok(Digest32(transcript.finish()))
    }
}

/// Local V3 challenge state. The local creation time is not caller-provided.
#[derive(Debug)]
pub struct PendingAuthorityStateChallengeV3 {
    wire: AuthorityStateChallengeV3,
    created_not_before_unix_s: u64,
}

impl PendingAuthorityStateChallengeV3 {
    pub fn new(
        policy: &AuthorityStatePolicyV2,
        grant: &CapabilityGrant,
        time: &VerifiedAuthorityTime,
    ) -> Result<Self, AuthorityStateLineageError> {
        policy.validate()?;
        grant.validate()?;
        let grant_digest = grant.digest();
        time.require_subject(grant_digest.0)?;
        let _ = time.conservative_now_unix_s()?;
        let (created_not_before_unix_s, _) = time.interval_at_verification();

        let mut nonce = [0u8; 32];
        getrandom::getrandom(&mut nonce)
            .map_err(|_| AuthorityStateLineageError::RandomnessUnavailable)?;
        if nonce == [0; 32] {
            return Err(AuthorityStateLineageError::RandomnessUnavailable);
        }

        let wire = AuthorityStateChallengeV3 {
            schema_version: AUTHORITY_STATE_LINEAGE_SCHEMA_VERSION,
            nonce,
            grant_digest,
            state_policy_digest: policy.digest()?,
            time_policy_digest: time.policy_digest(),
        };
        let _ = wire.digest()?;
        Ok(Self {
            wire,
            created_not_before_unix_s,
        })
    }

    pub fn wire(&self) -> AuthorityStateChallengeV3 {
        self.wire
    }
}

/// One witness's V3 statement. The lineage is signed, not inferred locally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityStateStatementV3 {
    pub schema_version: u16,
    pub witness_id: AuthorityStateWitnessId,
    pub challenge_nonce: [u8; 32],
    pub grant_digest: Digest32,
    pub state_policy_digest: [u8; 32],
    pub time_policy_digest: [u8; 32],
    pub source_frontier_lineage: AuthorityFrontierLineageV1,
    pub source_frontier_sequence: u64,
    pub source_frontier_digest: Digest32,
    pub state_sequence: u64,
    pub authority_epoch: AuthorityEpoch,
    pub authority_context: Option<AuthorityContextRef>,
    pub negative_facts: Vec<NegativeAuthorityFact>,
    pub witness_generation: u64,
    pub signature: Vec<u8>,
}

impl AuthorityStateStatementV3 {
    pub fn canonical_message(&self) -> Result<Vec<u8>, AuthorityStateLineageError> {
        validate_statement_shape(self)?;
        let snapshot = snapshot_digest_v3(
            self.grant_digest,
            &self.source_frontier_lineage,
            self.source_frontier_sequence,
            self.source_frontier_digest,
            self.state_sequence,
            self.authority_epoch,
            self.authority_context.as_ref(),
            &self.negative_facts,
        )?;

        let mut transcript = Transcript::new(STATEMENT_DOMAIN_V3);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.witness_id.0);
        transcript.fixed(&self.challenge_nonce);
        transcript.fixed(&self.grant_digest.0);
        transcript.fixed(&self.state_policy_digest);
        transcript.fixed(&self.time_policy_digest);
        transcript.fixed(&snapshot.0);
        transcript.u64(self.witness_generation);
        Ok(transcript.into_bytes())
    }
}

/// Opaque lineage-bound current authority-state proof.
///
/// Deliberately neither Clone nor Serde and still not execution authority.
#[derive(Debug)]
pub struct VerifiedAuthorityStateV3 {
    grant_digest: Digest32,
    source_frontier_lineage: AuthorityFrontierLineageV1,
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

impl VerifiedAuthorityStateV3 {
    pub fn grant_digest(&self) -> Digest32 {
        self.grant_digest
    }

    pub fn source_frontier_lineage(&self) -> &AuthorityFrontierLineageV1 {
        &self.source_frontier_lineage
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
    ) -> Result<(), AuthorityStateLineageError> {
        grant.validate()?;
        if grant.digest() != self.grant_digest {
            return Err(AuthorityStateLineageError::GrantMismatch);
        }
        time.require_subject(self.grant_digest.0)?;
        if time.policy_digest() != self.time_policy_digest {
            return Err(AuthorityStateLineageError::TimePolicyChanged);
        }
        let age = time
            .conservative_now_unix_s()?
            .checked_sub(self.verified_not_before_unix_s)
            .ok_or(AuthorityStateLineageError::TimeMovedBackward)?;
        if age > self.maximum_post_verification_age_s {
            return Err(AuthorityStateLineageError::VerifiedStateStale);
        }
        Ok(())
    }

    /// Compare this verified frontier with one previously bound reference.
    ///
    /// Different lineages are intentionally incomparable. This method never
    /// turns a larger sequence in a different lineage into evidence of progress.
    pub fn compare_source_frontier(
        &self,
        reference_lineage: &AuthorityFrontierLineageV1,
        reference_sequence: u64,
        reference_digest: Digest32,
    ) -> Result<AuthorityFrontierProgressV1, AuthorityFrontierOrderError> {
        reference_lineage
            .validate()
            .map_err(|_| AuthorityFrontierOrderError::InvalidReference)?;
        if reference_sequence == 0 || reference_digest.0 == [0; 32] {
            return Err(AuthorityFrontierOrderError::InvalidReference);
        }
        if &self.source_frontier_lineage != reference_lineage {
            return Err(AuthorityFrontierOrderError::DifferentLineage);
        }
        if self.source_frontier_sequence < reference_sequence {
            return Err(AuthorityFrontierOrderError::Rollback {
                reference: reference_sequence,
                current: self.source_frontier_sequence,
            });
        }
        if self.source_frontier_sequence == reference_sequence {
            if self.source_frontier_digest != reference_digest {
                return Err(AuthorityFrontierOrderError::Contradiction);
            }
            return Ok(AuthorityFrontierProgressV1::Same);
        }
        Ok(AuthorityFrontierProgressV1::Newer)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorityFrontierProgressV1 {
    Same,
    Newer,
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum AuthorityFrontierOrderError {
    #[error("frontier reference is malformed")]
    InvalidReference,
    #[error("authority frontiers belong to different lineages and are not directly comparable")]
    DifferentLineage,
    #[error("authority frontier sequence rolled back from {reference} to {current}")]
    Rollback { reference: u64, current: u64 },
    #[error("same authority frontier sequence names a different digest")]
    Contradiction,
}

pub fn verify_authority_state_v3(
    policy: &AuthorityStatePolicyV2,
    grant: &CapabilityGrant,
    challenge: PendingAuthorityStateChallengeV3,
    time: &VerifiedAuthorityTime,
    statements: &[AuthorityStateStatementV3],
) -> Result<VerifiedAuthorityStateV3, AuthorityStateLineageError> {
    policy.validate()?;
    grant.validate()?;
    let grant_digest = grant.digest();
    if !(usize::from(policy.threshold)..=MAX_AUTHORITY_STATE_STATEMENTS).contains(&statements.len()) {
        return Err(AuthorityStateLineageError::InsufficientStatements);
    }
    if challenge.wire.schema_version != AUTHORITY_STATE_LINEAGE_SCHEMA_VERSION
        || challenge.wire.grant_digest != grant_digest
        || challenge.wire.state_policy_digest != policy.digest()?
        || challenge.wire.nonce == [0; 32]
    {
        return Err(AuthorityStateLineageError::InvalidChallenge);
    }

    time.require_subject(grant_digest.0)?;
    if time.policy_digest() != challenge.wire.time_policy_digest {
        return Err(AuthorityStateLineageError::TimePolicyChangedDuringChallenge);
    }
    let challenge_age = time
        .conservative_now_unix_s()?
        .checked_sub(challenge.created_not_before_unix_s)
        .ok_or(AuthorityStateLineageError::TimeMovedBackward)?;
    if challenge_age > policy.maximum_challenge_age_s {
        return Err(AuthorityStateLineageError::ChallengeExpired);
    }

    let mut ids = BTreeSet::new();
    let mut organizations = BTreeSet::new();
    let mut services = BTreeSet::new();
    let mut snapshot = None;
    let mut lineage = None;
    let mut frontier = None;
    let mut state_sequence = None;
    let mut epoch = None;
    let mut context = None;
    let mut facts = None;

    for statement in statements {
        validate_statement_shape(statement)?;
        let witness = policy
            .witnesses
            .iter()
            .find(|witness| witness.witness_id == statement.witness_id)
            .ok_or(AuthorityStateLineageError::UnknownWitness)?;
        if statement.challenge_nonce != challenge.wire.nonce
            || statement.grant_digest != grant_digest
            || statement.state_policy_digest != challenge.wire.state_policy_digest
            || statement.time_policy_digest != challenge.wire.time_policy_digest
            || !ids.insert(statement.witness_id)
        {
            return Err(AuthorityStateLineageError::InvalidStatement);
        }

        let canonical_facts = canonical_grant_facts(grant, &statement.negative_facts)?;
        let signature_bytes: [u8; 64] = statement
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| AuthorityStateLineageError::BadSignatureLength)?;
        VerifyingKey::from_bytes(&witness.verifying_key)
            .map_err(|_| AuthorityStateLineageError::InvalidPolicy)?
            .verify(
                &statement.canonical_message()?,
                &Signature::from_bytes(&signature_bytes),
            )
            .map_err(|_| AuthorityStateLineageError::BadSignature)?;

        agree(
            &mut snapshot,
            snapshot_digest_v3(
                statement.grant_digest,
                &statement.source_frontier_lineage,
                statement.source_frontier_sequence,
                statement.source_frontier_digest,
                statement.state_sequence,
                statement.authority_epoch,
                statement.authority_context.as_ref(),
                &statement.negative_facts,
            )?,
        )?;
        agree(&mut lineage, statement.source_frontier_lineage.clone())?;
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
        return Err(AuthorityStateLineageError::InsufficientDiversity);
    }

    let (verified_not_before_unix_s, _) = time.interval_at_verification();
    let (source_frontier_sequence, source_frontier_digest) =
        frontier.ok_or(AuthorityStateLineageError::InsufficientStatements)?;

    Ok(VerifiedAuthorityStateV3 {
        grant_digest,
        source_frontier_lineage: lineage.ok_or(AuthorityStateLineageError::InsufficientStatements)?,
        source_frontier_sequence,
        source_frontier_digest,
        state_sequence: state_sequence.ok_or(AuthorityStateLineageError::InsufficientStatements)?,
        authority_epoch: epoch.ok_or(AuthorityStateLineageError::InsufficientStatements)?,
        authority_context: context.ok_or(AuthorityStateLineageError::InsufficientStatements)?,
        negative_facts: facts.ok_or(AuthorityStateLineageError::InsufficientStatements)?,
        snapshot_digest: snapshot.ok_or(AuthorityStateLineageError::InsufficientStatements)?,
        state_policy_digest: challenge.wire.state_policy_digest,
        time_policy_digest: challenge.wire.time_policy_digest,
        verified_not_before_unix_s,
        maximum_post_verification_age_s: policy.maximum_post_verification_age_s,
        witness_count: u16::try_from(ids.len()).map_err(|_| AuthorityStateLineageError::Encoding)?,
        organization_count: u16::try_from(organizations.len())
            .map_err(|_| AuthorityStateLineageError::Encoding)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn snapshot_digest_v3(
    grant_digest: Digest32,
    source_frontier_lineage: &AuthorityFrontierLineageV1,
    source_frontier_sequence: u64,
    source_frontier_digest: Digest32,
    state_sequence: u64,
    authority_epoch: AuthorityEpoch,
    authority_context: Option<&AuthorityContextRef>,
    negative_facts: &[NegativeAuthorityFact],
) -> Result<Digest32, AuthorityStateLineageError> {
    source_frontier_lineage.validate()?;
    if grant_digest.0 == [0; 32]
        || source_frontier_sequence == 0
        || source_frontier_digest.0 == [0; 32]
        || state_sequence == 0
        || authority_epoch.0 == 0
    {
        return Err(AuthorityStateLineageError::InvalidStatement);
    }
    if let Some(context) = authority_context {
        validate_context(context)?;
    }
    let canonical = canonical_fact_set(negative_facts)?;
    let mut transcript = Transcript::new(SNAPSHOT_DOMAIN_V3);
    transcript.u16(AUTHORITY_STATE_LINEAGE_SCHEMA_VERSION);
    transcript.fixed(&grant_digest.0);
    transcript.string(&source_frontier_lineage.namespace)?;
    transcript.fixed(&source_frontier_lineage.incarnation.0);
    transcript.u64(source_frontier_sequence);
    transcript.fixed(&source_frontier_digest.0);
    transcript.u64(state_sequence);
    transcript.u64(authority_epoch.0);
    match authority_context {
        Some(context) => {
            transcript.byte(1);
            transcript.string(&context.namespace.0)?;
            transcript.fixed(&context.digest.0);
        }
        None => transcript.byte(0),
    }
    transcript.u32(canonical.len())?;
    for (digest, _) in canonical {
        transcript.fixed(&digest.0);
    }
    Ok(Digest32(transcript.finish()))
}

fn validate_statement_shape(
    statement: &AuthorityStateStatementV3,
) -> Result<(), AuthorityStateLineageError> {
    statement.source_frontier_lineage.validate()?;
    if statement.schema_version != AUTHORITY_STATE_LINEAGE_SCHEMA_VERSION
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
        return Err(AuthorityStateLineageError::InvalidStatement);
    }
    if let Some(context) = statement.authority_context.as_ref() {
        validate_context(context)?;
    }
    Ok(())
}

fn agree<T: PartialEq>(
    slot: &mut Option<T>,
    value: T,
) -> Result<(), AuthorityStateLineageError> {
    match slot {
        None => {
            *slot = Some(value);
            Ok(())
        }
        Some(expected) if expected == &value => Ok(()),
        Some(_) => Err(AuthorityStateLineageError::StateDisagreement),
    }
}

fn canonical_grant_facts(
    grant: &CapabilityGrant,
    facts: &[NegativeAuthorityFact],
) -> Result<Vec<NegativeAuthorityFact>, AuthorityStateLineageError> {
    let canonical = canonical_fact_set(facts)?;
    if canonical
        .iter()
        .any(|(_, fact)| !fact_relevant_to_grant(grant, fact))
    {
        return Err(AuthorityStateLineageError::IrrelevantNegativeFact);
    }
    Ok(canonical.into_iter().map(|(_, fact)| fact).collect())
}

fn canonical_fact_set(
    facts: &[NegativeAuthorityFact],
) -> Result<Vec<(Digest32, NegativeAuthorityFact)>, AuthorityStateLineageError> {
    if facts.len() > MAX_NEGATIVE_FACTS_PER_GRANT {
        return Err(AuthorityStateLineageError::TooManyNegativeFacts);
    }
    let mut by_digest = BTreeMap::new();
    for fact in facts {
        let digest = negative_fact_digest_v2(fact)?;
        if by_digest.insert(digest, fact.clone()).is_some() {
            return Err(AuthorityStateLineageError::DuplicateNegativeFact);
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

fn validate_context(context: &AuthorityContextRef) -> Result<(), AuthorityStateLineageError> {
    validate_identifier(&context.namespace.0)?;
    if context.digest.0 == [0; 32] {
        return Err(AuthorityStateLineageError::InvalidAuthorityContext);
    }
    Ok(())
}

#[allow(dead_code)]
fn validate_principal(principal: &PrincipalId) -> Result<(), AuthorityStateLineageError> {
    validate_identifier(&principal.0)
}

#[allow(dead_code)]
fn validate_resource(resource: &ResourceRef) -> Result<(), AuthorityStateLineageError> {
    validate_identifier(&resource.0)
}

fn validate_identifier(value: &str) -> Result<(), AuthorityStateLineageError> {
    if value.is_empty() || value.len() > MAX_AUTHORITY_IDENTIFIER_BYTES {
        return Err(AuthorityStateLineageError::InvalidIdentifier);
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

    fn u32(&mut self, value: usize) -> Result<(), AuthorityStateLineageError> {
        let value = u32::try_from(value).map_err(|_| AuthorityStateLineageError::Encoding)?;
        self.bytes.extend_from_slice(&value.to_be_bytes());
        Ok(())
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn fixed<const N: usize>(&mut self, value: &[u8; N]) {
        self.bytes.extend_from_slice(value);
    }

    fn string(&mut self, value: &str) -> Result<(), AuthorityStateLineageError> {
        validate_identifier(value)?;
        self.u32(value.len())?;
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
pub enum AuthorityStateLineageError {
    #[error("authority-state V2 witness policy is invalid")]
    InvalidPolicy,
    #[error("capability grant is invalid: {0}")]
    InvalidGrant(#[from] GrantValidationError),
    #[error("secure authority-state challenge randomness is unavailable")]
    RandomnessUnavailable,
    #[error("lineage-bound authority-state challenge is invalid")]
    InvalidChallenge,
    #[error("authority frontier lineage/incarnation is malformed")]
    InvalidFrontierLineage,
    #[error("not enough lineage-bound authority-state statements were supplied")]
    InsufficientStatements,
    #[error("lineage-bound statement references an unknown witness")]
    UnknownWitness,
    #[error("lineage-bound authority-state statement is malformed or mismatched")]
    InvalidStatement,
    #[error("authority-state authority context is malformed")]
    InvalidAuthorityContext,
    #[error("authority-state witness signature must be exactly 64 bytes")]
    BadSignatureLength,
    #[error("authority-state witness signature verification failed")]
    BadSignature,
    #[error("authority-state witnesses do not satisfy organization/service diversity")]
    InsufficientDiversity,
    #[error("fresh authority-state witnesses disagree on lineage/frontier/state")]
    StateDisagreement,
    #[error("verified authority state belongs to a different capability grant")]
    GrantMismatch,
    #[error("trusted-time policy changed while V3 challenge was in flight")]
    TimePolicyChangedDuringChallenge,
    #[error("trusted-time policy changed after V3 verification")]
    TimePolicyChanged,
    #[error("lineage-bound authority-state challenge expired")]
    ChallengeExpired,
    #[error("verified lineage-bound authority state is stale")]
    VerifiedStateStale,
    #[error("trusted time moved backward across authority-state verification")]
    TimeMovedBackward,
    #[error("negative-authority fact set exceeded the inherited per-grant bound")]
    TooManyNegativeFacts,
    #[error("negative-authority fact set contains a duplicate commitment")]
    DuplicateNegativeFact,
    #[error("negative-authority fact is not relevant to the challenged grant")]
    IrrelevantNegativeFact,
    #[error("authority identifier is empty or exceeds the canonical bound")]
    InvalidIdentifier,
    #[error("trusted authority time failed: {0}")]
    AuthorityTime(#[from] AuthorityTimeError),
    #[error("V2 authority-state primitive failed: {0}")]
    AuthorityStateV2(#[from] AuthorityStateError),
    #[error("lineage-bound authority-state canonical encoding failed")]
    Encoding,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use symthaea_authority::{Operation, PurposeId};
    use symthaea_authority_state::{
        AuthorityStateStatementV2, TrustedAuthorityStateWitnessV2, AUTHORITY_STATE_SCHEMA_VERSION,
    };
    use symthaea_authority_time::{
        verify_authority_time_v1, AuthorityTimeStatementV1, PendingAuthorityTimeChallenge,
        TimeAuthorityId, TrustedTimeAuthorityV1, TrustedTimePolicyV1,
        AUTHORITY_TIME_SCHEMA_VERSION,
    };

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn lineage(byte: u8) -> AuthorityFrontierLineageV1 {
        AuthorityFrontierLineageV1::new("authority-ledger", digest(byte)).unwrap()
    }

    fn root_grant() -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "grant-lineage-v3",
            PrincipalId("issuer".into()),
            PrincipalId("robot-1".into()),
            PurposeId("goal-directed-actuation".into()),
            AuthorityEpoch(7),
            AuthorityContextRef::new("swarm-controller", digest(9)),
        );
        grant.resources.insert(ResourceRef("robot-1".into()));
        grant.operations.insert(Operation("move".into()));
        grant
    }

    fn verified_time(subject: [u8; 32]) -> VerifiedAuthorityTime {
        let key1 = SigningKey::from_bytes(&[1u8; 32]);
        let key2 = SigningKey::from_bytes(&[2u8; 32]);
        let policy = TrustedTimePolicyV1 {
            schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
            policy_id: [7; 16],
            authorities: vec![
                TrustedTimeAuthorityV1 {
                    authority_id: TimeAuthorityId([1; 16]),
                    verifying_key: key1.verifying_key().to_bytes(),
                    organization_binding: [11; 32],
                    service_binding: [21; 32],
                },
                TrustedTimeAuthorityV1 {
                    authority_id: TimeAuthorityId([2; 16]),
                    verifying_key: key2.verifying_key().to_bytes(),
                    organization_binding: [12; 32],
                    service_binding: [22; 32],
                },
            ],
            threshold: 2,
            minimum_organizations: 2,
            maximum_uncertainty_s: 2,
            maximum_challenge_age_ns: 60_000_000_000,
            maximum_post_verification_age_ns: 60_000_000_000,
        };
        let challenge = PendingAuthorityTimeChallenge::new(&policy, subject).unwrap();
        let wire = challenge.wire();
        let mut first = AuthorityTimeStatementV1 {
            schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
            authority_id: TimeAuthorityId([1; 16]),
            policy_digest: wire.policy_digest,
            subject_digest: wire.subject_digest,
            challenge_nonce: wire.nonce,
            witnessed_unix_s: 1_800_000_000,
            uncertainty_s: 1,
            signature: Vec::new(),
        };
        first.signature = key1
            .sign(&first.canonical_message().unwrap())
            .to_bytes()
            .to_vec();
        let mut second = AuthorityTimeStatementV1 {
            authority_id: TimeAuthorityId([2; 16]),
            ..first.clone()
        };
        second.signature = key2
            .sign(&second.canonical_message().unwrap())
            .to_bytes()
            .to_vec();
        verify_authority_time_v1(&policy, challenge, &[first, second]).unwrap()
    }

    fn state_policy() -> (AuthorityStatePolicyV2, SigningKey, SigningKey) {
        let key1 = SigningKey::from_bytes(&[3u8; 32]);
        let key2 = SigningKey::from_bytes(&[4u8; 32]);
        (
            AuthorityStatePolicyV2 {
                schema_version: AUTHORITY_STATE_SCHEMA_VERSION,
                policy_id: [9; 16],
                witnesses: vec![
                    TrustedAuthorityStateWitnessV2 {
                        witness_id: AuthorityStateWitnessId([3; 16]),
                        verifying_key: key1.verifying_key().to_bytes(),
                        organization_binding: [31; 32],
                        service_binding: [41; 32],
                    },
                    TrustedAuthorityStateWitnessV2 {
                        witness_id: AuthorityStateWitnessId([4; 16]),
                        verifying_key: key2.verifying_key().to_bytes(),
                        organization_binding: [32; 32],
                        service_binding: [42; 32],
                    },
                ],
                threshold: 2,
                minimum_organizations: 2,
                maximum_challenge_age_s: 60,
                maximum_post_verification_age_s: 60,
            },
            key1,
            key2,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn signed_v3(
        grant: &CapabilityGrant,
        wire: AuthorityStateChallengeV3,
        key: &SigningKey,
        witness_id: [u8; 16],
        source_lineage: AuthorityFrontierLineageV1,
        frontier_sequence: u64,
        frontier_digest: Digest32,
        state_sequence: u64,
    ) -> AuthorityStateStatementV3 {
        let mut statement = AuthorityStateStatementV3 {
            schema_version: AUTHORITY_STATE_LINEAGE_SCHEMA_VERSION,
            witness_id: AuthorityStateWitnessId(witness_id),
            challenge_nonce: wire.nonce,
            grant_digest: grant.digest(),
            state_policy_digest: wire.state_policy_digest,
            time_policy_digest: wire.time_policy_digest,
            source_frontier_lineage: source_lineage,
            source_frontier_sequence: frontier_sequence,
            source_frontier_digest: frontier_digest,
            state_sequence,
            authority_epoch: grant.authority_epoch,
            authority_context: Some(grant.authority_context.clone()),
            negative_facts: vec![],
            witness_generation: 1,
            signature: Vec::new(),
        };
        statement.signature = key
            .sign(&statement.canonical_message().unwrap())
            .to_bytes()
            .to_vec();
        statement
    }

    fn verified_v3(
        source_lineage: AuthorityFrontierLineageV1,
        frontier_sequence: u64,
        frontier_digest: Digest32,
        state_sequence: u64,
    ) -> VerifiedAuthorityStateV3 {
        let grant = root_grant();
        let time = verified_time(grant.digest().0);
        let (policy, key1, key2) = state_policy();
        let challenge = PendingAuthorityStateChallengeV3::new(&policy, &grant, &time).unwrap();
        let wire = challenge.wire();
        let statements = vec![
            signed_v3(
                &grant,
                wire,
                &key1,
                [3; 16],
                source_lineage.clone(),
                frontier_sequence,
                frontier_digest,
                state_sequence,
            ),
            signed_v3(
                &grant,
                wire,
                &key2,
                [4; 16],
                source_lineage,
                frontier_sequence,
                frontier_digest,
                state_sequence,
            ),
        ];
        verify_authority_state_v3(&policy, &grant, challenge, &time, &statements).unwrap()
    }

    #[test]
    fn snapshot_identity_binds_frontier_lineage() {
        let grant = root_grant();
        let a = snapshot_digest_v3(
            grant.digest(),
            &lineage(1),
            11,
            digest(55),
            12,
            grant.authority_epoch,
            Some(&grant.authority_context),
            &[],
        )
        .unwrap();
        let b = snapshot_digest_v3(
            grant.digest(),
            &lineage(2),
            11,
            digest(55),
            12,
            grant.authority_epoch,
            Some(&grant.authority_context),
            &[],
        )
        .unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn v2_and_v3_statement_domains_are_not_replay_equivalent() {
        let grant = root_grant();
        let time = verified_time(grant.digest().0);
        let (policy, _, _) = state_policy();
        let challenge_v3 = PendingAuthorityStateChallengeV3::new(&policy, &grant, &time).unwrap();
        let wire_v3 = challenge_v3.wire();
        let v3 = AuthorityStateStatementV3 {
            schema_version: AUTHORITY_STATE_LINEAGE_SCHEMA_VERSION,
            witness_id: AuthorityStateWitnessId([3; 16]),
            challenge_nonce: wire_v3.nonce,
            grant_digest: grant.digest(),
            state_policy_digest: wire_v3.state_policy_digest,
            time_policy_digest: wire_v3.time_policy_digest,
            source_frontier_lineage: lineage(1),
            source_frontier_sequence: 11,
            source_frontier_digest: digest(55),
            state_sequence: 12,
            authority_epoch: grant.authority_epoch,
            authority_context: Some(grant.authority_context.clone()),
            negative_facts: vec![],
            witness_generation: 1,
            signature: vec![],
        };
        let v2 = AuthorityStateStatementV2 {
            schema_version: AUTHORITY_STATE_SCHEMA_VERSION,
            witness_id: AuthorityStateWitnessId([3; 16]),
            challenge_nonce: wire_v3.nonce,
            grant_digest: grant.digest(),
            state_policy_digest: wire_v3.state_policy_digest,
            time_policy_digest: wire_v3.time_policy_digest,
            source_frontier_sequence: 11,
            source_frontier_digest: digest(55),
            state_sequence: 12,
            authority_epoch: grant.authority_epoch,
            authority_context: Some(grant.authority_context.clone()),
            negative_facts: vec![],
            witness_generation: 1,
            signature: vec![],
        };
        assert_ne!(v2.canonical_message().unwrap(), v3.canonical_message().unwrap());
    }

    #[test]
    fn witnesses_must_agree_on_exact_frontier_lineage() {
        let grant = root_grant();
        let time = verified_time(grant.digest().0);
        let (policy, key1, key2) = state_policy();
        let challenge = PendingAuthorityStateChallengeV3::new(&policy, &grant, &time).unwrap();
        let wire = challenge.wire();
        let statements = vec![
            signed_v3(&grant, wire, &key1, [3; 16], lineage(1), 11, digest(55), 12),
            signed_v3(&grant, wire, &key2, [4; 16], lineage(2), 11, digest(55), 12),
        ];
        assert!(matches!(
            verify_authority_state_v3(&policy, &grant, challenge, &time, &statements),
            Err(AuthorityStateLineageError::StateDisagreement)
        ));
    }

    #[test]
    fn same_lineage_higher_sequence_is_newer() {
        let current = verified_v3(lineage(1), 12, digest(56), 13);
        assert_eq!(
            current.compare_source_frontier(&lineage(1), 11, digest(55)).unwrap(),
            AuthorityFrontierProgressV1::Newer
        );
    }

    #[test]
    fn same_lineage_lower_sequence_is_rollback() {
        let current = verified_v3(lineage(1), 10, digest(54), 13);
        assert!(matches!(
            current.compare_source_frontier(&lineage(1), 11, digest(55)),
            Err(AuthorityFrontierOrderError::Rollback { .. })
        ));
    }

    #[test]
    fn same_lineage_equal_sequence_conflicting_digest_is_contradiction() {
        let current = verified_v3(lineage(1), 11, digest(56), 13);
        assert_eq!(
            current.compare_source_frontier(&lineage(1), 11, digest(55)),
            Err(AuthorityFrontierOrderError::Contradiction)
        );
    }

    #[test]
    fn different_lineage_is_incomparable_even_with_higher_sequence() {
        let current = verified_v3(lineage(2), 999, digest(99), 1000);
        assert_eq!(
            current.compare_source_frontier(&lineage(1), 11, digest(55)),
            Err(AuthorityFrontierOrderError::DifferentLineage)
        );
    }

    #[test]
    fn v3_preserves_negative_fact_count_bound() {
        let grant = root_grant();
        let facts = vec![
            NegativeAuthorityFact::RevokeGrant {
                grant_digest: grant.digest(),
            };
            MAX_NEGATIVE_FACTS_PER_GRANT + 1
        ];
        assert!(matches!(
            snapshot_digest_v3(
                grant.digest(),
                &lineage(1),
                11,
                digest(55),
                12,
                grant.authority_epoch,
                Some(&grant.authority_context),
                &facts,
            ),
            Err(AuthorityStateLineageError::TooManyNegativeFacts)
        ));
    }
}
