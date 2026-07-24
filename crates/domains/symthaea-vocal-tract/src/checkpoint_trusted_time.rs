// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public, multi-authority time attestations for checkpoint promotion.
//!
//! These statements provide externally witnessed time bounds. They do not claim
//! secure hardware time unless the authority policy is provisioned from such a
//! service and its organization and key bindings are independently governed.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{
    CheckpointPublicSignature, CheckpointPublicSigningKey,
    CheckpointPublicVerificationError, CheckpointPublicVerifyingKey,
    MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES,
};

pub const CHECKPOINT_TRUSTED_TIME_AUTHORITY_SCHEMA: &str =
    "symthaea.checkpoint-trusted-time-authority.v1";
pub const CHECKPOINT_TRUSTED_TIME_POLICY_SCHEMA: &str =
    "symthaea.checkpoint-trusted-time-policy.v1";
pub const CHECKPOINT_TRUSTED_TIME_STATEMENT_SCHEMA: &str =
    "symthaea.checkpoint-trusted-time-statement.v1";
pub const CHECKPOINT_TRUSTED_TIME_BUNDLE_SCHEMA: &str =
    "symthaea.checkpoint-trusted-time-bundle.v1";
pub const CHECKPOINT_TRUSTED_TIME_SUMMARY_SCHEMA: &str =
    "symthaea.checkpoint-trusted-time-summary.v1";
pub const CHECKPOINT_TRUSTED_TIME_STALE_NEGATIVE_SCHEMA: &str =
    "symthaea.checkpoint-trusted-time-stale-negative.v1";

pub const MAX_CHECKPOINT_TRUSTED_TIME_AUTHORITIES: usize = 64;
pub const MAX_CHECKPOINT_TRUSTED_TIME_STATEMENTS: usize = 128;
pub const MAX_CHECKPOINT_TRUSTED_TIME_UNCERTAINTY_SECONDS: u64 = 3_600;
pub const MAX_CHECKPOINT_TRUSTED_TIME_STATEMENT_AGE_SECONDS: u64 = 86_400;

const TRUSTED_TIME_POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea-checkpoint-trusted-time-policy-digest-v1\0";
const TRUSTED_TIME_STATEMENT_BODY_DOMAIN: &[u8] =
    b"symthaea-checkpoint-trusted-time-statement-body-v1\0";
const TRUSTED_TIME_STATEMENT_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea-checkpoint-trusted-time-statement-signature-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointTrustedTimeAuthorityId(pub [u8; 16]);

impl CheckpointTrustedTimeAuthorityId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, CheckpointTrustedTimeError> {
        if bytes == [0u8; 16] {
            return Err(CheckpointTrustedTimeError::InvalidAuthority);
        }
        Ok(Self(bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointTrustedTimeAuthority {
    pub schema: String,
    pub authority_id: CheckpointTrustedTimeAuthorityId,
    pub verifying_key: CheckpointPublicVerifyingKey,
    pub organization_binding: [u8; 32],
    pub service_binding: [u8; 32],
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointTrustedTimeAuthority {
    pub fn validate(&self) -> Result<(), CheckpointTrustedTimeError> {
        self.verifying_key.validate()?;
        if self.schema != CHECKPOINT_TRUSTED_TIME_AUTHORITY_SCHEMA
            || self.authority_id.0 == [0u8; 16]
            || self.organization_binding == [0u8; 32]
            || self.service_binding == [0u8; 32]
            || self.valid_from_unix_seconds == 0
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointTrustedTimeError::InvalidAuthority);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointTrustedTimePolicy {
    pub schema: String,
    pub policy_id: [u8; 16],
    pub authorities: Vec<CheckpointTrustedTimeAuthority>,
    pub threshold: u16,
    pub minimum_organizations: u16,
    pub maximum_uncertainty_seconds: u64,
    pub maximum_statement_age_seconds: u64,
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointTrustedTimePolicy {
    pub fn validate(&self) -> Result<(), CheckpointTrustedTimeError> {
        if self.schema != CHECKPOINT_TRUSTED_TIME_POLICY_SCHEMA
            || self.policy_id == [0u8; 16]
            || self.authorities.len() < 2
            || self.authorities.len() > MAX_CHECKPOINT_TRUSTED_TIME_AUTHORITIES
            || self.threshold < 2
            || usize::from(self.threshold) > self.authorities.len()
            || self.minimum_organizations < 2
            || self.minimum_organizations > self.threshold
            || self.maximum_uncertainty_seconds == 0
            || self.maximum_uncertainty_seconds > MAX_CHECKPOINT_TRUSTED_TIME_UNCERTAINTY_SECONDS
            || self.maximum_statement_age_seconds == 0
            || self.maximum_statement_age_seconds > MAX_CHECKPOINT_TRUSTED_TIME_STATEMENT_AGE_SECONDS
            || self.valid_from_unix_seconds == 0
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointTrustedTimeError::InvalidPolicy);
        }
        let mut authority_ids = HashSet::with_capacity(self.authorities.len());
        let mut key_ids = HashSet::with_capacity(self.authorities.len());
        let mut organizations = HashSet::with_capacity(self.authorities.len());
        let mut services = HashSet::with_capacity(self.authorities.len());
        for authority in &self.authorities {
            authority.validate()?;
            if authority.valid_from_unix_seconds < self.valid_from_unix_seconds
                || authority.valid_until_unix_seconds > self.valid_until_unix_seconds
                || !authority_ids.insert(authority.authority_id)
                || !key_ids.insert(authority.verifying_key.key_id)
                || !services.insert(authority.service_binding)
            {
                return Err(CheckpointTrustedTimeError::InvalidPolicy);
            }
            organizations.insert(authority.organization_binding);
        }
        if organizations.len() < usize::from(self.minimum_organizations) {
            return Err(CheckpointTrustedTimeError::InvalidPolicy);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<[u8; 32], CheckpointTrustedTimeError> {
        self.validate()?;
        trusted_time_digest(TRUSTED_TIME_POLICY_DIGEST_DOMAIN, self)
    }

    pub fn authority(
        &self,
        authority_id: CheckpointTrustedTimeAuthorityId,
    ) -> Option<&CheckpointTrustedTimeAuthority> {
        self.authorities
            .iter()
            .find(|authority| authority.authority_id == authority_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CheckpointTrustedTimeStatementBody {
    policy_digest: [u8; 32],
    subject_digest: [u8; 32],
    authority_id: CheckpointTrustedTimeAuthorityId,
    witnessed_unix_seconds: u64,
    uncertainty_seconds: u64,
    issued_at_unix_seconds: u64,
    nonce: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointTrustedTimeStatement {
    pub schema: String,
    pub policy_digest: [u8; 32],
    pub subject_digest: [u8; 32],
    pub authority_id: CheckpointTrustedTimeAuthorityId,
    pub witnessed_unix_seconds: u64,
    pub uncertainty_seconds: u64,
    pub issued_at_unix_seconds: u64,
    pub nonce: [u8; 32],
    pub signature: CheckpointPublicSignature,
}

impl CheckpointTrustedTimeStatement {
    #[allow(clippy::too_many_arguments)]
    pub fn sign(
        signing_key: &CheckpointPublicSigningKey,
        policy: &CheckpointTrustedTimePolicy,
        authority_id: CheckpointTrustedTimeAuthorityId,
        subject_digest: [u8; 32],
        witnessed_unix_seconds: u64,
        uncertainty_seconds: u64,
        issued_at_unix_seconds: u64,
        nonce: [u8; 32],
    ) -> Result<Self, CheckpointTrustedTimeError> {
        let authority = policy
            .authority(authority_id)
            .ok_or(CheckpointTrustedTimeError::UnknownAuthority)?;
        if signing_key.key_id() != authority.verifying_key.key_id
            || subject_digest == [0u8; 32]
            || witnessed_unix_seconds == 0
            || uncertainty_seconds == 0
            || uncertainty_seconds > policy.maximum_uncertainty_seconds
            || issued_at_unix_seconds < authority.valid_from_unix_seconds
            || issued_at_unix_seconds > authority.valid_until_unix_seconds
            || nonce == [0u8; 32]
        {
            return Err(CheckpointTrustedTimeError::InvalidStatement);
        }
        let policy_digest = policy.digest()?;
        let body = CheckpointTrustedTimeStatementBody {
            policy_digest,
            subject_digest,
            authority_id,
            witnessed_unix_seconds,
            uncertainty_seconds,
            issued_at_unix_seconds,
            nonce,
        };
        let body_digest = trusted_time_digest(TRUSTED_TIME_STATEMENT_BODY_DOMAIN, &body)?;
        Ok(Self {
            schema: CHECKPOINT_TRUSTED_TIME_STATEMENT_SCHEMA.to_owned(),
            policy_digest,
            subject_digest,
            authority_id,
            witnessed_unix_seconds,
            uncertainty_seconds,
            issued_at_unix_seconds,
            nonce,
            signature: signing_key.sign(TRUSTED_TIME_STATEMENT_SIGNATURE_DOMAIN, &body_digest)?,
        })
    }

    fn body_digest(&self) -> Result<[u8; 32], CheckpointTrustedTimeError> {
        if self.schema != CHECKPOINT_TRUSTED_TIME_STATEMENT_SCHEMA
            || self.policy_digest == [0u8; 32]
            || self.subject_digest == [0u8; 32]
            || self.authority_id.0 == [0u8; 16]
            || self.witnessed_unix_seconds == 0
            || self.uncertainty_seconds == 0
            || self.issued_at_unix_seconds == 0
            || self.nonce == [0u8; 32]
        {
            return Err(CheckpointTrustedTimeError::InvalidStatement);
        }
        trusted_time_digest(
            TRUSTED_TIME_STATEMENT_BODY_DOMAIN,
            &CheckpointTrustedTimeStatementBody {
                policy_digest: self.policy_digest,
                subject_digest: self.subject_digest,
                authority_id: self.authority_id,
                witnessed_unix_seconds: self.witnessed_unix_seconds,
                uncertainty_seconds: self.uncertainty_seconds,
                issued_at_unix_seconds: self.issued_at_unix_seconds,
                nonce: self.nonce,
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointTrustedTimeBundle {
    pub schema: String,
    pub policy: CheckpointTrustedTimePolicy,
    pub subject_digest: [u8; 32],
    pub statements: Vec<CheckpointTrustedTimeStatement>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointTrustedTimeSummary {
    pub schema: String,
    pub subject_digest: [u8; 32],
    pub policy_digest: [u8; 32],
    pub valid_statements: usize,
    pub unique_organizations: usize,
    pub unique_services: usize,
    pub consensus_not_before_unix_seconds: u64,
    pub consensus_not_after_unix_seconds: u64,
}

impl CheckpointTrustedTimeSummary {
    pub fn validate(&self) -> Result<(), CheckpointTrustedTimeError> {
        if self.schema != CHECKPOINT_TRUSTED_TIME_SUMMARY_SCHEMA
            || self.subject_digest == [0u8; 32]
            || self.policy_digest == [0u8; 32]
            || self.valid_statements < 2
            || self.unique_organizations < 2
            || self.unique_services < 2
            || self.consensus_not_before_unix_seconds == 0
            || self.consensus_not_after_unix_seconds < self.consensus_not_before_unix_seconds
        {
            return Err(CheckpointTrustedTimeError::InvalidBundle);
        }
        Ok(())
    }
}

impl CheckpointTrustedTimeBundle {
    pub fn verify(
        &self,
        verification_time_unix_seconds: u64,
    ) -> Result<CheckpointTrustedTimeSummary, CheckpointTrustedTimeError> {
        self.policy.validate()?;
        if self.schema != CHECKPOINT_TRUSTED_TIME_BUNDLE_SCHEMA
            || self.subject_digest == [0u8; 32]
            || self.statements.is_empty()
            || self.statements.len() > MAX_CHECKPOINT_TRUSTED_TIME_STATEMENTS
            || verification_time_unix_seconds < self.policy.valid_from_unix_seconds
            || verification_time_unix_seconds > self.policy.valid_until_unix_seconds
        {
            return Err(CheckpointTrustedTimeError::InvalidBundle);
        }
        let policy_digest = self.policy.digest()?;
        let mut authorities = HashSet::new();
        let mut organizations = HashSet::new();
        let mut services = HashSet::new();
        let mut nonces = HashSet::new();
        let mut consensus_not_before = 0u64;
        let mut consensus_not_after = u64::MAX;
        for statement in &self.statements {
            let authority = self
                .policy
                .authority(statement.authority_id)
                .ok_or(CheckpointTrustedTimeError::UnknownAuthority)?;
            if statement.policy_digest != policy_digest
                || statement.subject_digest != self.subject_digest
                || statement.uncertainty_seconds > self.policy.maximum_uncertainty_seconds
                || statement.issued_at_unix_seconds > verification_time_unix_seconds
                || verification_time_unix_seconds.saturating_sub(statement.issued_at_unix_seconds)
                    > self.policy.maximum_statement_age_seconds
                || statement.issued_at_unix_seconds < authority.valid_from_unix_seconds
                || statement.issued_at_unix_seconds > authority.valid_until_unix_seconds
                || !authorities.insert(statement.authority_id)
                || !nonces.insert(statement.nonce)
            {
                return Err(CheckpointTrustedTimeError::InvalidStatement);
            }
            let body_digest = statement.body_digest()?;
            authority.verifying_key.verify(
                TRUSTED_TIME_STATEMENT_SIGNATURE_DOMAIN,
                &body_digest,
                &statement.signature,
            )?;
            let lower = statement
                .witnessed_unix_seconds
                .saturating_sub(statement.uncertainty_seconds);
            let upper = statement
                .witnessed_unix_seconds
                .saturating_add(statement.uncertainty_seconds);
            consensus_not_before = consensus_not_before.max(lower);
            consensus_not_after = consensus_not_after.min(upper);
            organizations.insert(authority.organization_binding);
            services.insert(authority.service_binding);
        }
        if authorities.len() < usize::from(self.policy.threshold)
            || organizations.len() < usize::from(self.policy.minimum_organizations)
            || services.len() < usize::from(self.policy.threshold)
            || consensus_not_after < consensus_not_before
            || verification_time_unix_seconds < consensus_not_before
            || verification_time_unix_seconds > consensus_not_after
        {
            return Err(CheckpointTrustedTimeError::NoTimeConsensus);
        }
        let summary = CheckpointTrustedTimeSummary {
            schema: CHECKPOINT_TRUSTED_TIME_SUMMARY_SCHEMA.to_owned(),
            subject_digest: self.subject_digest,
            policy_digest,
            valid_statements: authorities.len(),
            unique_organizations: organizations.len(),
            unique_services: services.len(),
            consensus_not_before_unix_seconds: consensus_not_before,
            consensus_not_after_unix_seconds: consensus_not_after,
        };
        summary.validate()?;
        Ok(summary)
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointTrustedTimeStaleNegativeSummary {
    pub schema: String,
    pub stale_candidate_rejected: bool,
}

impl CheckpointTrustedTimeStaleNegativeSummary {
    pub fn validate(&self) -> Result<(), CheckpointTrustedTimeError> {
        if self.schema != CHECKPOINT_TRUSTED_TIME_STALE_NEGATIVE_SCHEMA
            || !self.stale_candidate_rejected
        {
            return Err(CheckpointTrustedTimeError::StaleTimeNotRejected);
        }
        Ok(())
    }
}

pub fn verify_trusted_time_stale_negative(
    candidate: &CheckpointTrustedTimeBundle,
    verification_time_unix_seconds: u64,
) -> Result<CheckpointTrustedTimeStaleNegativeSummary, CheckpointTrustedTimeError> {
    candidate.policy.validate()?;
    let latest_issue_time = candidate
        .statements
        .iter()
        .map(|statement| statement.issued_at_unix_seconds)
        .max()
        .ok_or(CheckpointTrustedTimeError::InvalidBundle)?;
    let consensus_not_before = candidate
        .statements
        .iter()
        .map(|statement| {
            statement
                .witnessed_unix_seconds
                .saturating_sub(statement.uncertainty_seconds)
        })
        .max()
        .ok_or(CheckpointTrustedTimeError::InvalidBundle)?;
    let consensus_not_after = candidate
        .statements
        .iter()
        .map(|statement| {
            statement
                .witnessed_unix_seconds
                .saturating_add(statement.uncertainty_seconds)
        })
        .min()
        .ok_or(CheckpointTrustedTimeError::InvalidBundle)?;
    let fresh_verification_time = latest_issue_time.max(consensus_not_before);
    if fresh_verification_time > consensus_not_after {
        return Err(CheckpointTrustedTimeError::InvalidStaleCandidate);
    }
    candidate.verify(fresh_verification_time)?;
    if candidate.statements.iter().any(|statement| {
        verification_time_unix_seconds
            .saturating_sub(statement.issued_at_unix_seconds)
            <= candidate.policy.maximum_statement_age_seconds
    }) {
        return Err(CheckpointTrustedTimeError::InvalidStaleCandidate);
    }
    match candidate.verify(verification_time_unix_seconds) {
        Err(CheckpointTrustedTimeError::InvalidStatement) => {
            let summary = CheckpointTrustedTimeStaleNegativeSummary {
                schema: CHECKPOINT_TRUSTED_TIME_STALE_NEGATIVE_SCHEMA.to_owned(),
                stale_candidate_rejected: true,
            };
            summary.validate()?;
            Ok(summary)
        }
        Err(error) => Err(error),
        Ok(_) => Err(CheckpointTrustedTimeError::StaleTimeNotRejected),
    }
}

fn trusted_time_digest<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<[u8; 32], CheckpointTrustedTimeError> {
    let encoded = postcard::to_stdvec(value).map_err(|_| CheckpointTrustedTimeError::Encoding)?;
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES {
        return Err(CheckpointTrustedTimeError::TooLarge);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(*hasher.finalize().as_bytes())
}

#[derive(Debug)]
pub enum CheckpointTrustedTimeError {
    InvalidAuthority,
    InvalidPolicy,
    UnknownAuthority,
    InvalidStatement,
    NoTimeConsensus,
    InvalidBundle,
    Encoding,
    TooLarge,
    PublicVerification,
    StaleTimeNotRejected,
    InvalidStaleCandidate,
}

impl std::fmt::Display for CheckpointTrustedTimeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidAuthority => "invalid trusted-time authority",
            Self::InvalidPolicy => "invalid trusted-time policy",
            Self::UnknownAuthority => "unknown trusted-time authority",
            Self::InvalidStatement => "invalid trusted-time statement",
            Self::NoTimeConsensus => "trusted-time statements do not overlap",
            Self::InvalidBundle => "invalid trusted-time bundle",
            Self::Encoding => "trusted-time artifact encoding failed",
            Self::TooLarge => "trusted-time artifact exceeds its bound",
            Self::PublicVerification => "trusted-time signature verification failed",
            Self::StaleTimeNotRejected => "stale trusted-time evidence was not rejected",
            Self::InvalidStaleCandidate => "trusted-time stale candidate was not valid before expiry",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for CheckpointTrustedTimeError {}

impl From<CheckpointPublicVerificationError> for CheckpointTrustedTimeError {
    fn from(_: CheckpointPublicVerificationError) -> Self {
        Self::PublicVerification
    }
}
