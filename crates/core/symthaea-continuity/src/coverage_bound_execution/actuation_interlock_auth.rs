// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authentication for protocol-only actuation interlock decisions.
//!
//! This module deliberately does **not** prove backend/resource enforcement.
//! It proves only that one exact canonical interlock claim was authenticated by the
//! provisioned decision root for the exact enforcement-boundary profile/backend and
//! was also bound to the protocol's fresh challenge/predecessor chain.
//!
//! `BoundDecision != AuthenticatedDecision != ResourceEnforcement`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::actuation_interlock::{
    ActuationEnforcementBoundaryProfileId, ActuationEnforcementBoundaryProfileV1,
    ActuationInterlockClaimId, ActuationInterlockClaimV1, ActuationInterlockDispositionV1,
    ActuationInterlockError, ActuationInterlockSubjectId, ActuationInterlockSubjectV1,
    BoundActuationInterlockDecisionId, BoundActuationInterlockDecisionV1,
    canonical_actuation_interlock_claim_digest,
};
use crate::execution_capability::ExecutionBackendId;

pub const ACTUATION_INTERLOCK_AUTHENTICATION_PROFILE_SCHEMA_V1: &str =
    "symthaea-continuity-actuation-interlock-authentication-profile-v1";
pub const ACTUATION_INTERLOCK_AUTH_PURPOSE_V1: &str =
    "symthaea.continuity.actuation-interlock.authentication.v1";

const PROFILE_DOMAIN: &[u8] =
    b"symthaea.continuity.actuation-interlock-authentication-profile.v1\0";
const TRUSTED_ROOT_DOMAIN: &[u8] =
    b"symthaea.continuity.trusted-actuation-interlock-authentication-root.v1\0";
const AUTHENTICATED_DOMAIN: &[u8] =
    b"symthaea.continuity.authenticated-actuation-interlock-decision.v1\0";
const QUALIFIED_DOMAIN: &[u8] =
    b"symthaea.continuity.qualified-authenticated-actuation-interlock-decision.v1\0";
const MAX_TEXT_BYTES: usize = 1024;

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name {
            pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
        }
    };
}

digest_id!(ActuationInterlockAuthenticationProfileId);
digest_id!(TrustedActuationInterlockAuthenticationRootId);
digest_id!(AuthenticatedActuationInterlockDecisionId);
digest_id!(QualifiedAuthenticatedActuationInterlockDecisionId);

/// Serializable authentication configuration for one exact enforcement boundary.
/// Configuration is not trust; the non-Serde trusted root below must be provisioned
/// independently by an adapter/platform boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationInterlockAuthenticationProfileV1 {
    schema_version: String,
    profile_name: String,
    enforcement_profile_id: ActuationEnforcementBoundaryProfileId,
    backend_id: ExecutionBackendId,
    authentication_root_digest: [u8; 32],
    authentication_root_epoch: u64,
    authentication_implementation_digest: [u8; 32],
    profile_generation: u64,
    profile_id: ActuationInterlockAuthenticationProfileId,
}

impl ActuationInterlockAuthenticationProfileV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile_name: impl Into<String>,
        enforcement: &ActuationEnforcementBoundaryProfileV1,
        authentication_root_digest: [u8; 32],
        authentication_root_epoch: u64,
        authentication_implementation_digest: [u8; 32],
        profile_generation: u64,
    ) -> Result<Self, ActuationInterlockAuthenticationError> {
        enforcement.validate()?;
        let profile_name = checked_text(
            "actuation interlock authentication profile name",
            profile_name.into(),
        )?;
        if authentication_root_digest == [0; 32]
            || authentication_implementation_digest == [0; 32]
        {
            return Err(ActuationInterlockAuthenticationError::ZeroDigest);
        }
        if authentication_root_epoch == 0 || profile_generation == 0 {
            return Err(ActuationInterlockAuthenticationError::ZeroGeneration);
        }
        let profile_id = ActuationInterlockAuthenticationProfileId(hash_profile(
            &profile_name,
            enforcement.id(),
            enforcement.backend_id(),
            authentication_root_digest,
            authentication_root_epoch,
            authentication_implementation_digest,
            profile_generation,
        ));
        Ok(Self {
            schema_version: ACTUATION_INTERLOCK_AUTHENTICATION_PROFILE_SCHEMA_V1.to_owned(),
            profile_name,
            enforcement_profile_id: enforcement.id(),
            backend_id: enforcement.backend_id(),
            authentication_root_digest,
            authentication_root_epoch,
            authentication_implementation_digest,
            profile_generation,
            profile_id,
        })
    }

    pub fn validate(&self) -> Result<(), ActuationInterlockAuthenticationError> {
        if self.schema_version != ACTUATION_INTERLOCK_AUTHENTICATION_PROFILE_SCHEMA_V1 {
            return Err(ActuationInterlockAuthenticationError::UnsupportedProfileSchema(
                self.schema_version.clone(),
            ));
        }
        let canonical = checked_text(
            "actuation interlock authentication profile name",
            self.profile_name.clone(),
        )?;
        if canonical != self.profile_name {
            return Err(ActuationInterlockAuthenticationError::NonCanonicalText);
        }
        if self.authentication_root_digest == [0; 32]
            || self.authentication_implementation_digest == [0; 32]
        {
            return Err(ActuationInterlockAuthenticationError::ZeroDigest);
        }
        if self.authentication_root_epoch == 0 || self.profile_generation == 0 {
            return Err(ActuationInterlockAuthenticationError::ZeroGeneration);
        }
        let expected = ActuationInterlockAuthenticationProfileId(hash_profile(
            &self.profile_name,
            self.enforcement_profile_id,
            self.backend_id,
            self.authentication_root_digest,
            self.authentication_root_epoch,
            self.authentication_implementation_digest,
            self.profile_generation,
        ));
        if expected != self.profile_id {
            return Err(ActuationInterlockAuthenticationError::ProfileIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ActuationInterlockAuthenticationProfileId { self.profile_id }
    pub fn enforcement_profile_id(&self) -> ActuationEnforcementBoundaryProfileId {
        self.enforcement_profile_id
    }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
    pub fn authentication_root_epoch(&self) -> u64 { self.authentication_root_epoch }
}

/// Provisioned trust root. There is intentionally no public production constructor in
/// V1; a serializable profile must never be able to manufacture trust by itself.
#[derive(Debug)]
pub struct TrustedActuationInterlockAuthenticationRootV1 {
    root_id: TrustedActuationInterlockAuthenticationRootId,
    profile_id: ActuationInterlockAuthenticationProfileId,
    enforcement_profile_id: ActuationEnforcementBoundaryProfileId,
    backend_id: ExecutionBackendId,
    authentication_root_digest: [u8; 32],
    authentication_root_epoch: u64,
}

impl TrustedActuationInterlockAuthenticationRootV1 {
    #[cfg(test)]
    pub(crate) fn provision_for_test(
        profile: &ActuationInterlockAuthenticationProfileV1,
    ) -> Result<Self, ActuationInterlockAuthenticationError> {
        profile.validate()?;
        let root_id = TrustedActuationInterlockAuthenticationRootId(domain_hash_parts(
            TRUSTED_ROOT_DOMAIN,
            &[
                profile.id().as_bytes(),
                profile.enforcement_profile_id().as_bytes(),
                profile.backend_id().as_bytes(),
                &profile.authentication_root_digest,
                &profile.authentication_root_epoch.to_le_bytes(),
            ],
        ));
        Ok(Self {
            root_id,
            profile_id: profile.id(),
            enforcement_profile_id: profile.enforcement_profile_id(),
            backend_id: profile.backend_id(),
            authentication_root_digest: profile.authentication_root_digest,
            authentication_root_epoch: profile.authentication_root_epoch,
        })
    }

    pub fn id(&self) -> TrustedActuationInterlockAuthenticationRootId { self.root_id }
    pub fn profile_id(&self) -> ActuationInterlockAuthenticationProfileId { self.profile_id }
}

/// Authenticated canonical interlock claim. Still not resource enforcement.
#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedActuationInterlockDecisionV1 {
    claim: ActuationInterlockClaimV1,
    profile_id: ActuationInterlockAuthenticationProfileId,
    trusted_root_id: TrustedActuationInterlockAuthenticationRootId,
    canonical_claim_digest: [u8; 32],
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedActuationInterlockDecisionId,
}

impl AuthenticatedActuationInterlockDecisionV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: ActuationInterlockClaimV1,
        profile: &ActuationInterlockAuthenticationProfileV1,
        root: &TrustedActuationInterlockAuthenticationRootV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, ActuationInterlockAuthenticationError> {
        claim.validate()?;
        profile.validate()?;
        require_root_matches_profile(root, profile)?;
        if authentication_evidence_digest == [0; 32] {
            return Err(ActuationInterlockAuthenticationError::ZeroDigest);
        }
        let canonical_claim_digest = canonical_actuation_interlock_claim_digest(&claim)?;
        let evidence_id = AuthenticatedActuationInterlockDecisionId(domain_hash_parts(
            AUTHENTICATED_DOMAIN,
            &[
                claim.id().as_bytes(),
                profile.id().as_bytes(),
                root.id().as_bytes(),
                &canonical_claim_digest,
                &authentication_evidence_digest,
            ],
        ));
        Ok(Self {
            claim,
            profile_id: profile.id(),
            trusted_root_id: root.id(),
            canonical_claim_digest,
            authentication_evidence_digest,
            evidence_id,
        })
    }
}

/// Non-Serde evidence that an exact protocol decision was authenticated under the
/// exact provisioned boundary root. This object does not grant I/O authority and does
/// not claim the fence generation was the newest generation at the resource.
#[derive(Debug, Clone)]
pub struct QualifiedAuthenticatedActuationInterlockDecisionV1 {
    qualified_id: QualifiedAuthenticatedActuationInterlockDecisionId,
    subject_id: ActuationInterlockSubjectId,
    bound_decision_id: BoundActuationInterlockDecisionId,
    claim_id: ActuationInterlockClaimId,
    fencing_generation: u64,
    disposition: ActuationInterlockDispositionV1,
    enforcement_profile_id: ActuationEnforcementBoundaryProfileId,
    backend_id: ExecutionBackendId,
    authentication_profile_id: ActuationInterlockAuthenticationProfileId,
    trusted_root_id: TrustedActuationInterlockAuthenticationRootId,
    observed_at_unix_ms: u64,
}

impl QualifiedAuthenticatedActuationInterlockDecisionV1 {
    pub(crate) fn qualify(
        subject: &ActuationInterlockSubjectV1,
        bound: &BoundActuationInterlockDecisionV1,
        enforcement: &ActuationEnforcementBoundaryProfileV1,
        profile: &ActuationInterlockAuthenticationProfileV1,
        root: &TrustedActuationInterlockAuthenticationRootV1,
        authenticated: &AuthenticatedActuationInterlockDecisionV1,
    ) -> Result<Self, ActuationInterlockAuthenticationError> {
        subject.validate()?;
        enforcement.validate()?;
        profile.validate()?;
        authenticated.claim.validate()?;
        require_root_matches_profile(root, profile)?;

        if subject.enforcement_profile_id() != enforcement.id()
            || subject.backend_id() != enforcement.backend_id()
            || profile.enforcement_profile_id() != enforcement.id()
            || profile.backend_id() != enforcement.backend_id()
            || authenticated.profile_id != profile.id()
            || authenticated.trusted_root_id != root.id()
        {
            return Err(ActuationInterlockAuthenticationError::BoundaryContextMismatch);
        }
        if bound.subject_id() != subject.id()
            || bound.claim_id() != authenticated.claim.id()
            || bound.fencing_generation() != authenticated.claim.fencing_generation()
            || bound.disposition() != authenticated.claim.disposition()
        {
            return Err(ActuationInterlockAuthenticationError::DecisionContextMismatch);
        }
        let expected_digest = canonical_actuation_interlock_claim_digest(&authenticated.claim)?;
        if expected_digest != authenticated.canonical_claim_digest {
            return Err(ActuationInterlockAuthenticationError::CanonicalDigestMismatch);
        }

        let qualified_id = QualifiedAuthenticatedActuationInterlockDecisionId(domain_hash_parts(
            QUALIFIED_DOMAIN,
            &[
                subject.id().as_bytes(),
                bound.id().as_bytes(),
                authenticated.claim.id().as_bytes(),
                profile.id().as_bytes(),
                root.id().as_bytes(),
                authenticated.evidence_id.as_bytes(),
                &expected_digest,
            ],
        ));
        Ok(Self {
            qualified_id,
            subject_id: subject.id(),
            bound_decision_id: bound.id(),
            claim_id: authenticated.claim.id(),
            fencing_generation: authenticated.claim.fencing_generation(),
            disposition: authenticated.claim.disposition(),
            enforcement_profile_id: enforcement.id(),
            backend_id: enforcement.backend_id(),
            authentication_profile_id: profile.id(),
            trusted_root_id: root.id(),
            observed_at_unix_ms: authenticated.claim.observed_at_unix_ms(),
        })
    }

    pub fn id(&self) -> QualifiedAuthenticatedActuationInterlockDecisionId { self.qualified_id }
    pub fn subject_id(&self) -> ActuationInterlockSubjectId { self.subject_id }
    pub fn bound_decision_id(&self) -> BoundActuationInterlockDecisionId { self.bound_decision_id }
    pub fn claim_id(&self) -> ActuationInterlockClaimId { self.claim_id }
    pub fn fencing_generation(&self) -> u64 { self.fencing_generation }
    pub fn disposition(&self) -> ActuationInterlockDispositionV1 { self.disposition }
    pub fn enforcement_profile_id(&self) -> ActuationEnforcementBoundaryProfileId {
        self.enforcement_profile_id
    }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
    pub fn observed_at_unix_ms(&self) -> u64 { self.observed_at_unix_ms }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ActuationInterlockAuthenticationError {
    #[error(transparent)]
    Interlock(#[from] ActuationInterlockError),
    #[error("unsupported actuation interlock authentication profile schema: {0}")]
    UnsupportedProfileSchema(String),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds the text bound")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("actuation interlock authentication profile text is not canonical")]
    NonCanonicalText,
    #[error("actuation interlock authentication digest must be non-zero")]
    ZeroDigest,
    #[error("actuation interlock authentication generation must be non-zero")]
    ZeroGeneration,
    #[error("actuation interlock authentication profile identity mismatch")]
    ProfileIdentityMismatch,
    #[error("trusted actuation interlock root does not match authentication profile")]
    TrustedRootMismatch,
    #[error("actuation interlock authentication boundary/backend context mismatch")]
    BoundaryContextMismatch,
    #[error("authenticated actuation decision differs from bound protocol decision")]
    DecisionContextMismatch,
    #[error("authenticated actuation decision canonical digest mismatch")]
    CanonicalDigestMismatch,
}

fn require_root_matches_profile(
    root: &TrustedActuationInterlockAuthenticationRootV1,
    profile: &ActuationInterlockAuthenticationProfileV1,
) -> Result<(), ActuationInterlockAuthenticationError> {
    profile.validate()?;
    if root.profile_id != profile.id()
        || root.enforcement_profile_id != profile.enforcement_profile_id()
        || root.backend_id != profile.backend_id()
        || root.authentication_root_digest != profile.authentication_root_digest
        || root.authentication_root_epoch != profile.authentication_root_epoch
    {
        return Err(ActuationInterlockAuthenticationError::TrustedRootMismatch);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn hash_profile(
    profile_name: &str,
    enforcement_profile_id: ActuationEnforcementBoundaryProfileId,
    backend_id: ExecutionBackendId,
    authentication_root_digest: [u8; 32],
    authentication_root_epoch: u64,
    authentication_implementation_digest: [u8; 32],
    profile_generation: u64,
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(PROFILE_DOMAIN);
    hash_len_prefixed(&mut h, profile_name.as_bytes());
    h.update(enforcement_profile_id.as_bytes());
    h.update(backend_id.as_bytes());
    h.update(&authentication_root_digest);
    h.update(&authentication_root_epoch.to_le_bytes());
    h.update(&authentication_implementation_digest);
    h.update(&profile_generation.to_le_bytes());
    *h.finalize().as_bytes()
}

fn domain_hash_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    for part in parts {
        h.update(&((*part).len() as u64).to_le_bytes());
        h.update(part);
    }
    *h.finalize().as_bytes()
}

fn hash_len_prefixed(h: &mut blake3::Hasher, bytes: &[u8]) {
    h.update(&(bytes.len() as u64).to_le_bytes());
    h.update(bytes);
}

fn checked_text(
    field: &'static str,
    value: String,
) -> Result<String, ActuationInterlockAuthenticationError> {
    let value = value.trim().to_owned();
    if value.is_empty() {
        return Err(ActuationInterlockAuthenticationError::BlankText { field });
    }
    if value.len() > MAX_TEXT_BYTES {
        return Err(ActuationInterlockAuthenticationError::TextTooLong { field });
    }
    if value.chars().any(char::is_control) {
        return Err(ActuationInterlockAuthenticationError::ControlCharacters { field });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authentication_domains_are_distinct() {
        assert_ne!(PROFILE_DOMAIN, TRUSTED_ROOT_DOMAIN);
        assert_ne!(TRUSTED_ROOT_DOMAIN, AUTHENTICATED_DOMAIN);
        assert_ne!(AUTHENTICATED_DOMAIN, QUALIFIED_DOMAIN);
    }
}