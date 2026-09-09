// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Challenge-bound, multi-authority time evidence for bounded Symthaea agency.
//!
//! This crate exists because a caller-supplied wall clock is not an authority
//! fact. A process that can choose `now_unix_s` can extend an otherwise expired
//! capability simply by moving time backwards.
//!
//! V1 therefore derives an admission-time upper bound without accepting wall
//! clock input from the caller:
//!
//! 1. generate a fresh random challenge after reading Linux `/proc/uptime`;
//! 2. require multiple independent authorities to sign the challenge, exact
//!    subject commitment, witnessed Unix time, and bounded uncertainty;
//! 3. verify signatures and policy diversity;
//! 4. conservatively add the full challenge round-trip duration to every signed
//!    upper bound;
//! 5. intersect all accepted intervals;
//! 6. retain only a short-lived verified fact whose age is advanced by Linux
//!    boot time, which includes suspend on the supported platform.
//!
//! Greater uncertainty can only move the conservative upper bound forward and
//! therefore expire authority sooner. Missing/invalid/stale evidence fails
//! closed. This crate creates no capability and exposes no execution API.

#![deny(unsafe_code)]

use std::collections::BTreeSet;
use std::fs;

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const AUTHORITY_TIME_SCHEMA_VERSION: u16 = 1;
pub const MAX_TIME_AUTHORITIES: usize = 64;
pub const MAX_TIME_STATEMENTS: usize = 128;
pub const MAX_TIME_UNCERTAINTY_S: u64 = 3_600;
pub const MAX_CHALLENGE_AGE_NS: u64 = 60_000_000_000;
pub const MAX_POST_VERIFICATION_AGE_NS: u64 = 60_000_000_000;

const POLICY_DOMAIN: &[u8] = b"symthaea.authority-time.policy.v1\0";
const CHALLENGE_DOMAIN: &[u8] = b"symthaea.authority-time.challenge.v1\0";
const STATEMENT_DOMAIN: &[u8] = b"symthaea.authority-time.statement.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TimeAuthorityId(pub [u8; 16]);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedTimeAuthorityV1 {
    pub authority_id: TimeAuthorityId,
    pub verifying_key: [u8; 32],
    /// Stable commitment to the organization controlling the signer.
    pub organization_binding: [u8; 32],
    /// Stable commitment to the independently operated time service.
    pub service_binding: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedTimePolicyV1 {
    pub schema_version: u16,
    pub policy_id: [u8; 16],
    pub authorities: Vec<TrustedTimeAuthorityV1>,
    pub threshold: u16,
    pub minimum_organizations: u16,
    pub maximum_uncertainty_s: u64,
    pub maximum_challenge_age_ns: u64,
    pub maximum_post_verification_age_ns: u64,
}

impl TrustedTimePolicyV1 {
    pub fn validate(&self) -> Result<(), AuthorityTimeError> {
        if self.schema_version != AUTHORITY_TIME_SCHEMA_VERSION
            || self.policy_id == [0; 16]
            || self.authorities.len() < 2
            || self.authorities.len() > MAX_TIME_AUTHORITIES
            || self.threshold < 2
            || usize::from(self.threshold) > self.authorities.len()
            || self.minimum_organizations < 2
            || self.minimum_organizations > self.threshold
            || self.maximum_uncertainty_s == 0
            || self.maximum_uncertainty_s > MAX_TIME_UNCERTAINTY_S
            || self.maximum_challenge_age_ns == 0
            || self.maximum_challenge_age_ns > MAX_CHALLENGE_AGE_NS
            || self.maximum_post_verification_age_ns == 0
            || self.maximum_post_verification_age_ns > MAX_POST_VERIFICATION_AGE_NS
        {
            return Err(AuthorityTimeError::InvalidPolicy);
        }

        let mut authority_ids = BTreeSet::new();
        let mut keys = BTreeSet::new();
        let mut organizations = BTreeSet::new();
        let mut services = BTreeSet::new();
        for authority in &self.authorities {
            if authority.authority_id.0 == [0; 16]
                || authority.verifying_key == [0; 32]
                || authority.organization_binding == [0; 32]
                || authority.service_binding == [0; 32]
                || VerifyingKey::from_bytes(&authority.verifying_key).is_err()
                || !authority_ids.insert(authority.authority_id)
                || !keys.insert(authority.verifying_key)
                || !services.insert(authority.service_binding)
            {
                return Err(AuthorityTimeError::InvalidPolicy);
            }
            organizations.insert(authority.organization_binding);
        }
        if organizations.len() < usize::from(self.minimum_organizations) {
            return Err(AuthorityTimeError::InvalidPolicy);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<[u8; 32], AuthorityTimeError> {
        self.validate()?;
        let mut transcript = Transcript::new(POLICY_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.policy_id);
        transcript.u32(
            u32::try_from(self.authorities.len()).map_err(|_| AuthorityTimeError::Encoding)?,
        );
        for authority in &self.authorities {
            transcript.fixed(&authority.authority_id.0);
            transcript.fixed(&authority.verifying_key);
            transcript.fixed(&authority.organization_binding);
            transcript.fixed(&authority.service_binding);
        }
        transcript.u16(self.threshold);
        transcript.u16(self.minimum_organizations);
        transcript.u64(self.maximum_uncertainty_s);
        transcript.u64(self.maximum_challenge_age_ns);
        transcript.u64(self.maximum_post_verification_age_ns);
        Ok(transcript.finish())
    }

    fn authority(&self, authority_id: TimeAuthorityId) -> Option<&TrustedTimeAuthorityV1> {
        self.authorities
            .iter()
            .find(|authority| authority.authority_id == authority_id)
    }
}

/// Wire-safe challenge sent to trusted time authorities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityTimeChallengeV1 {
    pub schema_version: u16,
    pub nonce: [u8; 32],
    /// Exact admission subject, normally the capability-grant commitment.
    pub subject_digest: [u8; 32],
    pub policy_digest: [u8; 32],
}

impl AuthorityTimeChallengeV1 {
    pub fn digest(&self) -> [u8; 32] {
        let mut transcript = Transcript::new(CHALLENGE_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.nonce);
        transcript.fixed(&self.subject_digest);
        transcript.fixed(&self.policy_digest);
        transcript.finish()
    }
}

/// Local challenge state. The boot-time send instant never leaves the process.
#[derive(Debug)]
pub struct PendingAuthorityTimeChallenge {
    wire: AuthorityTimeChallengeV1,
    sent_boottime_ns: u64,
}

impl PendingAuthorityTimeChallenge {
    pub fn new(
        policy: &TrustedTimePolicyV1,
        subject_digest: [u8; 32],
    ) -> Result<Self, AuthorityTimeError> {
        if subject_digest == [0; 32] {
            return Err(AuthorityTimeError::InvalidSubject);
        }
        let policy_digest = policy.digest()?;
        let sent_boottime_ns = linux_boottime_ns()?;
        let mut nonce = [0u8; 32];
        getrandom::getrandom(&mut nonce).map_err(|_| AuthorityTimeError::RandomnessUnavailable)?;
        if nonce == [0; 32] {
            return Err(AuthorityTimeError::RandomnessUnavailable);
        }
        Ok(Self {
            wire: AuthorityTimeChallengeV1 {
                schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
                nonce,
                subject_digest,
                policy_digest,
            },
            sent_boottime_ns,
        })
    }

    pub fn wire(&self) -> AuthorityTimeChallengeV1 {
        self.wire
    }
}

/// One authority's signed response to an exact fresh challenge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityTimeStatementV1 {
    pub schema_version: u16,
    pub authority_id: TimeAuthorityId,
    pub policy_digest: [u8; 32],
    pub subject_digest: [u8; 32],
    pub challenge_nonce: [u8; 32],
    pub witnessed_unix_s: u64,
    pub uncertainty_s: u64,
    /// Ed25519 signature bytes. V1 requires exactly 64 bytes at verification.
    pub signature: Vec<u8>,
}

impl AuthorityTimeStatementV1 {
    pub fn canonical_message(&self) -> Result<Vec<u8>, AuthorityTimeError> {
        if self.schema_version != AUTHORITY_TIME_SCHEMA_VERSION
            || self.authority_id.0 == [0; 16]
            || self.policy_digest == [0; 32]
            || self.subject_digest == [0; 32]
            || self.challenge_nonce == [0; 32]
            || self.witnessed_unix_s == 0
            || self.uncertainty_s == 0
        {
            return Err(AuthorityTimeError::InvalidStatement);
        }
        let mut transcript = Transcript::new(STATEMENT_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.authority_id.0);
        transcript.fixed(&self.policy_digest);
        transcript.fixed(&self.subject_digest);
        transcript.fixed(&self.challenge_nonce);
        transcript.u64(self.witnessed_unix_s);
        transcript.u64(self.uncertainty_s);
        Ok(transcript.into_bytes())
    }
}

/// Opaque short-lived fact that an exact subject was observed within a
/// cryptographically verified time interval.
///
/// There is intentionally no public constructor. Production callers obtain this
/// only through [`verify_authority_time_v1`].
#[derive(Debug)]
pub struct VerifiedAuthorityTime {
    subject_digest: [u8; 32],
    policy_digest: [u8; 32],
    consensus_not_before_unix_s: u64,
    consensus_not_after_at_verification_unix_s: u64,
    verified_boottime_ns: u64,
    maximum_post_verification_age_ns: u64,
    authority_count: u16,
    organization_count: u16,
}

impl VerifiedAuthorityTime {
    pub fn subject_digest(&self) -> [u8; 32] {
        self.subject_digest
    }

    pub fn policy_digest(&self) -> [u8; 32] {
        self.policy_digest
    }

    pub fn authority_count(&self) -> u16 {
        self.authority_count
    }

    pub fn organization_count(&self) -> u16 {
        self.organization_count
    }

    pub fn interval_at_verification(&self) -> (u64, u64) {
        (
            self.consensus_not_before_unix_s,
            self.consensus_not_after_at_verification_unix_s,
        )
    }

    pub fn require_subject(&self, expected: [u8; 32]) -> Result<(), AuthorityTimeError> {
        if self.subject_digest != expected {
            return Err(AuthorityTimeError::SubjectMismatch);
        }
        Ok(())
    }

    /// Conservative upper bound for the current Unix time.
    ///
    /// Linux boot time is re-read on every use. The full elapsed duration is
    /// rounded upward to seconds, so local passage of time cannot make a grant
    /// live for longer. Facts older than the policy's short post-verification
    /// window fail closed and require a new challenge.
    pub fn conservative_now_unix_s(&self) -> Result<u64, AuthorityTimeError> {
        let now_boottime_ns = linux_boottime_ns()?;
        let elapsed_ns = now_boottime_ns
            .checked_sub(self.verified_boottime_ns)
            .ok_or(AuthorityTimeError::BootTimeMovedBackward)?;
        if elapsed_ns > self.maximum_post_verification_age_ns {
            return Err(AuthorityTimeError::VerifiedTimeStale);
        }
        self.consensus_not_after_at_verification_unix_s
            .checked_add(ceil_ns_to_s(elapsed_ns)?)
            .ok_or(AuthorityTimeError::ArithmeticOverflow)
    }
}

/// Verify challenged time evidence without accepting a caller wall clock.
pub fn verify_authority_time_v1(
    policy: &TrustedTimePolicyV1,
    challenge: PendingAuthorityTimeChallenge,
    statements: &[AuthorityTimeStatementV1],
) -> Result<VerifiedAuthorityTime, AuthorityTimeError> {
    policy.validate()?;
    if statements.len() < usize::from(policy.threshold)
        || statements.len() > MAX_TIME_STATEMENTS
    {
        return Err(AuthorityTimeError::InsufficientStatements);
    }
    if challenge.wire.schema_version != AUTHORITY_TIME_SCHEMA_VERSION
        || challenge.wire.policy_digest != policy.digest()?
        || challenge.wire.subject_digest == [0; 32]
        || challenge.wire.nonce == [0; 32]
    {
        return Err(AuthorityTimeError::InvalidChallenge);
    }

    let received_boottime_ns = linux_boottime_ns()?;
    let challenge_age_ns = received_boottime_ns
        .checked_sub(challenge.sent_boottime_ns)
        .ok_or(AuthorityTimeError::BootTimeMovedBackward)?;
    if challenge_age_ns > policy.maximum_challenge_age_ns {
        return Err(AuthorityTimeError::ChallengeExpired);
    }
    let transport_upper_s = ceil_ns_to_s(challenge_age_ns)?;

    let mut authority_ids = BTreeSet::new();
    let mut organizations = BTreeSet::new();
    let mut services = BTreeSet::new();
    let mut consensus_not_before = 0u64;
    let mut consensus_not_after = u64::MAX;

    for statement in statements {
        let authority = policy
            .authority(statement.authority_id)
            .ok_or(AuthorityTimeError::UnknownAuthority)?;
        if statement.schema_version != AUTHORITY_TIME_SCHEMA_VERSION
            || statement.policy_digest != challenge.wire.policy_digest
            || statement.subject_digest != challenge.wire.subject_digest
            || statement.challenge_nonce != challenge.wire.nonce
            || statement.uncertainty_s == 0
            || statement.uncertainty_s > policy.maximum_uncertainty_s
            || !authority_ids.insert(statement.authority_id)
        {
            return Err(AuthorityTimeError::InvalidStatement);
        }

        let message = statement.canonical_message()?;
        let key = VerifyingKey::from_bytes(&authority.verifying_key)
            .map_err(|_| AuthorityTimeError::InvalidPolicy)?;
        let signature_bytes: [u8; 64] = statement
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| AuthorityTimeError::BadSignatureLength)?;
        key.verify(&message, &Signature::from_bytes(&signature_bytes))
            .map_err(|_| AuthorityTimeError::BadSignature)?;

        let lower = statement
            .witnessed_unix_s
            .saturating_sub(statement.uncertainty_s);
        let upper_at_verification = statement
            .witnessed_unix_s
            .checked_add(statement.uncertainty_s)
            .and_then(|value| value.checked_add(transport_upper_s))
            .ok_or(AuthorityTimeError::ArithmeticOverflow)?;
        consensus_not_before = consensus_not_before.max(lower);
        consensus_not_after = consensus_not_after.min(upper_at_verification);
        organizations.insert(authority.organization_binding);
        services.insert(authority.service_binding);
    }

    if authority_ids.len() < usize::from(policy.threshold)
        || organizations.len() < usize::from(policy.minimum_organizations)
        || services.len() < usize::from(policy.threshold)
    {
        return Err(AuthorityTimeError::InsufficientDiversity);
    }
    if consensus_not_after < consensus_not_before {
        return Err(AuthorityTimeError::NoTimeConsensus);
    }

    Ok(VerifiedAuthorityTime {
        subject_digest: challenge.wire.subject_digest,
        policy_digest: challenge.wire.policy_digest,
        consensus_not_before_unix_s: consensus_not_before,
        consensus_not_after_at_verification_unix_s: consensus_not_after,
        verified_boottime_ns: received_boottime_ns,
        maximum_post_verification_age_ns: policy.maximum_post_verification_age_ns,
        authority_count: u16::try_from(authority_ids.len())
            .map_err(|_| AuthorityTimeError::Encoding)?,
        organization_count: u16::try_from(organizations.len())
            .map_err(|_| AuthorityTimeError::Encoding)?,
    })
}

/// Linux boot time in nanoseconds, derived without unsafe code.
///
/// `/proc/uptime` is treated as kernel evidence under the Agency Kernel threat
/// model. A fully compromised kernel/hypervisor remains outside the guarantee.
/// Linux `/proc/uptime` includes suspended time, avoiding the authority lease
/// extension that occurs with clocks that stop while the machine sleeps.
fn linux_boottime_ns() -> Result<u64, AuthorityTimeError> {
    let text = fs::read_to_string("/proc/uptime").map_err(AuthorityTimeError::BootTimeRead)?;
    let token = text
        .split_whitespace()
        .next()
        .ok_or(AuthorityTimeError::MalformedBootTime)?;
    parse_decimal_seconds_ns(token)
}

fn parse_decimal_seconds_ns(value: &str) -> Result<u64, AuthorityTimeError> {
    let (whole, fractional) = value.split_once('.').unwrap_or((value, ""));
    if whole.is_empty()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fractional.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(AuthorityTimeError::MalformedBootTime);
    }
    let seconds: u64 = whole
        .parse()
        .map_err(|_| AuthorityTimeError::MalformedBootTime)?;
    let mut nanos = 0u64;
    let mut digits = 0u32;
    for byte in fractional.bytes().take(9) {
        nanos = nanos
            .checked_mul(10)
            .and_then(|value| value.checked_add(u64::from(byte - b'0')))
            .ok_or(AuthorityTimeError::ArithmeticOverflow)?;
        digits += 1;
    }
    while digits < 9 {
        nanos = nanos
            .checked_mul(10)
            .ok_or(AuthorityTimeError::ArithmeticOverflow)?;
        digits += 1;
    }
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanos))
        .ok_or(AuthorityTimeError::ArithmeticOverflow)
}

fn ceil_ns_to_s(ns: u64) -> Result<u64, AuthorityTimeError> {
    let whole = ns / 1_000_000_000;
    if ns % 1_000_000_000 == 0 {
        Ok(whole)
    } else {
        whole
            .checked_add(1)
            .ok_or(AuthorityTimeError::ArithmeticOverflow)
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
pub enum AuthorityTimeError {
    #[error("trusted time policy is invalid")]
    InvalidPolicy,
    #[error("authority-time subject commitment is invalid")]
    InvalidSubject,
    #[error("authority-time challenge is invalid")]
    InvalidChallenge,
    #[error("secure challenge randomness is unavailable")]
    RandomnessUnavailable,
    #[error("not enough signed time statements were supplied")]
    InsufficientStatements,
    #[error("time statement references an unknown authority")]
    UnknownAuthority,
    #[error("signed time statement is malformed or does not bind the challenge")]
    InvalidStatement,
    #[error("time statement Ed25519 signature must be exactly 64 bytes")]
    BadSignatureLength,
    #[error("time statement signature verification failed")]
    BadSignature,
    #[error("time authorities do not satisfy the configured organizational/service diversity")]
    InsufficientDiversity,
    #[error("challenged time statements have no overlapping conservative interval")]
    NoTimeConsensus,
    #[error("authority-time challenge exceeded its boot-time round-trip limit")]
    ChallengeExpired,
    #[error("verified authority time belongs to a different admission subject")]
    SubjectMismatch,
    #[error("verified authority-time fact exceeded its short post-verification lifetime")]
    VerifiedTimeStale,
    #[error("Linux boot-time clock moved backward")]
    BootTimeMovedBackward,
    #[error("failed to read Linux boot-time evidence: {0}")]
    BootTimeRead(#[source] std::io::Error),
    #[error("Linux boot-time evidence is malformed")]
    MalformedBootTime,
    #[error("authority-time arithmetic overflow")]
    ArithmeticOverflow,
    #[error("authority-time canonical encoding failed")]
    Encoding,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn authority(seed: u8, id: u8, org: u8, service: u8) -> (TrustedTimeAuthorityV1, SigningKey) {
        let signing_key = SigningKey::from_bytes(&[seed; 32]);
        (
            TrustedTimeAuthorityV1 {
                authority_id: TimeAuthorityId([id; 16]),
                verifying_key: signing_key.verifying_key().to_bytes(),
                organization_binding: [org; 32],
                service_binding: [service; 32],
            },
            signing_key,
        )
    }

    fn policy() -> (TrustedTimePolicyV1, Vec<SigningKey>) {
        let (a1, k1) = authority(1, 1, 11, 21);
        let (a2, k2) = authority(2, 2, 12, 22);
        let (a3, k3) = authority(3, 3, 13, 23);
        (
            TrustedTimePolicyV1 {
                schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
                policy_id: [9; 16],
                authorities: vec![a1, a2, a3],
                threshold: 2,
                minimum_organizations: 2,
                maximum_uncertainty_s: 5,
                maximum_challenge_age_ns: 5_000_000_000,
                maximum_post_verification_age_ns: 5_000_000_000,
            },
            vec![k1, k2, k3],
        )
    }

    fn pending(policy: &TrustedTimePolicyV1, subject: [u8; 32]) -> PendingAuthorityTimeChallenge {
        PendingAuthorityTimeChallenge {
            wire: AuthorityTimeChallengeV1 {
                schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
                nonce: [7; 32],
                subject_digest: subject,
                policy_digest: policy.digest().unwrap(),
            },
            sent_boottime_ns: linux_boottime_ns().unwrap(),
        }
    }

    fn statement(
        policy: &TrustedTimePolicyV1,
        challenge: AuthorityTimeChallengeV1,
        key: &SigningKey,
        authority_id: TimeAuthorityId,
        witnessed_unix_s: u64,
        uncertainty_s: u64,
    ) -> AuthorityTimeStatementV1 {
        let mut statement = AuthorityTimeStatementV1 {
            schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
            authority_id,
            policy_digest: policy.digest().unwrap(),
            subject_digest: challenge.subject_digest,
            challenge_nonce: challenge.nonce,
            witnessed_unix_s,
            uncertainty_s,
            signature: Vec::new(),
        };
        statement.signature = key
            .sign(&statement.canonical_message().unwrap())
            .to_bytes()
            .to_vec();
        statement
    }

    #[test]
    fn challenged_diverse_signers_produce_conservative_time_fact() {
        let (policy, keys) = policy();
        let subject = [44; 32];
        let pending = pending(&policy, subject);
        let wire = pending.wire();
        let statements = vec![
            statement(&policy, wire, &keys[0], TimeAuthorityId([1; 16]), 1_000, 2),
            statement(&policy, wire, &keys[1], TimeAuthorityId([2; 16]), 1_001, 2),
        ];
        let verified = verify_authority_time_v1(&policy, pending, &statements).unwrap();
        assert_eq!(verified.subject_digest(), subject);
        assert_eq!(verified.authority_count(), 2);
        assert_eq!(verified.organization_count(), 2);
        let (lower, upper) = verified.interval_at_verification();
        assert!(lower >= 999);
        assert!(upper >= lower);
        assert!(verified.conservative_now_unix_s().unwrap() >= upper);
    }

    #[test]
    fn wrong_challenge_nonce_cannot_replay_old_statement() {
        let (policy, keys) = policy();
        let pending = pending(&policy, [44; 32]);
        let wire = pending.wire();
        let mut first = statement(&policy, wire, &keys[0], TimeAuthorityId([1; 16]), 1_000, 2);
        first.challenge_nonce = [99; 32];
        let second = statement(&policy, wire, &keys[1], TimeAuthorityId([2; 16]), 1_001, 2);
        assert!(matches!(
            verify_authority_time_v1(&policy, pending, &[first, second]),
            Err(AuthorityTimeError::InvalidStatement)
        ));
    }

    #[test]
    fn bad_signature_fails_closed() {
        let (policy, keys) = policy();
        let pending = pending(&policy, [44; 32]);
        let wire = pending.wire();
        let mut first = statement(&policy, wire, &keys[0], TimeAuthorityId([1; 16]), 1_000, 2);
        first.signature[0] ^= 0x80;
        let second = statement(&policy, wire, &keys[1], TimeAuthorityId([2; 16]), 1_001, 2);
        assert!(matches!(
            verify_authority_time_v1(&policy, pending, &[first, second]),
            Err(AuthorityTimeError::BadSignature)
        ));
    }

    #[test]
    fn short_signature_fails_closed_before_crypto() {
        let (policy, keys) = policy();
        let pending = pending(&policy, [44; 32]);
        let wire = pending.wire();
        let mut first = statement(&policy, wire, &keys[0], TimeAuthorityId([1; 16]), 1_000, 2);
        first.signature.truncate(63);
        let second = statement(&policy, wire, &keys[1], TimeAuthorityId([2; 16]), 1_001, 2);
        assert!(matches!(
            verify_authority_time_v1(&policy, pending, &[first, second]),
            Err(AuthorityTimeError::BadSignatureLength)
        ));
    }

    #[test]
    fn duplicate_organization_cannot_satisfy_diversity() {
        let (mut policy, keys) = policy();
        policy.authorities[1].organization_binding = policy.authorities[0].organization_binding;
        policy.validate().unwrap();
        let pending = pending(&policy, [44; 32]);
        let wire = pending.wire();
        let statements = vec![
            statement(&policy, wire, &keys[0], TimeAuthorityId([1; 16]), 1_000, 2),
            statement(&policy, wire, &keys[1], TimeAuthorityId([2; 16]), 1_001, 2),
        ];
        assert!(matches!(
            verify_authority_time_v1(&policy, pending, &statements),
            Err(AuthorityTimeError::InsufficientDiversity)
        ));
    }

    #[test]
    fn non_overlapping_time_claims_do_not_create_a_clock() {
        let (policy, keys) = policy();
        let pending = pending(&policy, [44; 32]);
        let wire = pending.wire();
        let statements = vec![
            statement(&policy, wire, &keys[0], TimeAuthorityId([1; 16]), 1_000, 1),
            statement(&policy, wire, &keys[1], TimeAuthorityId([2; 16]), 2_000, 1),
        ];
        assert!(matches!(
            verify_authority_time_v1(&policy, pending, &statements),
            Err(AuthorityTimeError::NoTimeConsensus)
        ));
    }

    #[test]
    fn stale_challenge_fails_without_consulting_wall_clock() {
        let (policy, keys) = policy();
        let mut pending = pending(&policy, [44; 32]);
        pending.sent_boottime_ns = linux_boottime_ns()
            .unwrap()
            .saturating_sub(policy.maximum_challenge_age_ns + 1);
        let wire = pending.wire();
        let statements = vec![
            statement(&policy, wire, &keys[0], TimeAuthorityId([1; 16]), 1_000, 2),
            statement(&policy, wire, &keys[1], TimeAuthorityId([2; 16]), 1_001, 2),
        ];
        assert!(matches!(
            verify_authority_time_v1(&policy, pending, &statements),
            Err(AuthorityTimeError::ChallengeExpired)
        ));
    }

    #[test]
    fn subject_binding_prevents_cross_grant_time_reuse() {
        let (policy, keys) = policy();
        let pending = pending(&policy, [44; 32]);
        let wire = pending.wire();
        let statements = vec![
            statement(&policy, wire, &keys[0], TimeAuthorityId([1; 16]), 1_000, 2),
            statement(&policy, wire, &keys[1], TimeAuthorityId([2; 16]), 1_001, 2),
        ];
        let verified = verify_authority_time_v1(&policy, pending, &statements).unwrap();
        assert!(matches!(
            verified.require_subject([55; 32]),
            Err(AuthorityTimeError::SubjectMismatch)
        ));
    }

    #[test]
    fn decimal_uptime_parser_is_integer_only_and_precise() {
        assert_eq!(parse_decimal_seconds_ns("12").unwrap(), 12_000_000_000);
        assert_eq!(parse_decimal_seconds_ns("12.34").unwrap(), 12_340_000_000);
        assert_eq!(parse_decimal_seconds_ns("0.000000001").unwrap(), 1);
        assert!(parse_decimal_seconds_ns("12.bad").is_err());
    }

    #[test]
    fn ceil_duration_never_rounds_elapsed_time_down() {
        assert_eq!(ceil_ns_to_s(0).unwrap(), 0);
        assert_eq!(ceil_ns_to_s(1).unwrap(), 1);
        assert_eq!(ceil_ns_to_s(1_000_000_000).unwrap(), 1);
        assert_eq!(ceil_ns_to_s(1_000_000_001).unwrap(), 2);
    }
}
