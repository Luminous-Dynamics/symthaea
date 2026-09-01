// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dedicated final-resolution authority.
//!
//! Resolution is intentionally represented by an opaque public grant type rather
//! than by reusing execution or observation grants at the host boundary. The
//! implementation reuses the already-established authority-domain and epoch
//! machinery internally, while callers see a distinct [`ResolutionGrant`] and
//! [`ResolutionVerifier`]. This keeps final interpretation independently
//! attributable without duplicating trust-domain logic.
//!
//! A future lower-level capability marker can replace the private transport once
//! the resolution semantics are stable; that migration should not change this
//! public host-facing distinction.
//!
//! ```compile_fail
//! use symthaea_ai_assurance::{
//!     AuthorityDomain, Observe, PrincipalId, ResolutionGrant, Scope,
//! };
//!
//! fn requires_resolution_authority(_: ResolutionGrant) {}
//!
//! let observation = AuthorityDomain::new(PrincipalId::new());
//! let grant = observation.issue_bound_one_shot::<Observe>(
//!     PrincipalId::new(),
//!     Scope::new("workspace", ["symthaea"]).unwrap(),
//!     None,
//!     [0; 32],
//! );
//!
//! // Observation authority is a different public type and cannot authorize
//! // final interpretation.
//! requires_resolution_authority(grant);
//! ```

use crate::capability::{GrantMetadata, Observe, PrincipalId, Scope};
use crate::trusted::{
    AuthorityDomain, AuthorityDomainId, AuthorityEpoch, AuthorityVerifier, TrustError,
    TrustedBoundOneShotCapability,
};
use std::time::SystemTime;

/// Trusted policy authority that can mint exact one-shot final-resolution grants.
#[derive(Debug)]
pub struct ResolutionAuthorityDomain {
    inner: AuthorityDomain,
}

impl ResolutionAuthorityDomain {
    /// Create an independent final-resolution authority domain.
    pub fn new(principal: PrincipalId) -> Self {
        Self {
            inner: AuthorityDomain::new(principal),
        }
    }

    /// Authority domain identity preserved in final evidence.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.inner.domain_id()
    }

    /// Root resolver-policy principal.
    pub fn principal(&self) -> PrincipalId {
        self.inner.principal()
    }

    /// Create a verifier retained by the strict host runtime.
    pub fn verifier(&self) -> ResolutionVerifier {
        ResolutionVerifier {
            inner: self.inner.verifier(),
        }
    }

    /// Mint exact one-shot authority for one final-resolution binding.
    pub fn issue_bound_one_shot(
        &self,
        subject: PrincipalId,
        scope: Scope,
        expires_at: Option<SystemTime>,
        binding: [u8; 32],
    ) -> ResolutionGrant {
        ResolutionGrant {
            inner: self
                .inner
                .issue_bound_one_shot::<Observe>(subject, scope, expires_at, binding),
        }
    }

    /// Revoke all outstanding resolver grants from earlier epochs.
    pub fn revoke_all(&self) -> Result<AuthorityEpoch, TrustError> {
        self.inner.revoke_all()
    }
}

/// Host-retained trust anchor for final-resolution authority.
#[derive(Debug, Clone)]
pub struct ResolutionVerifier {
    inner: AuthorityVerifier,
}

impl ResolutionVerifier {
    /// Trusted resolver authority-domain identity.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.inner.domain_id()
    }

    /// Current resolver revocation epoch.
    pub fn current_epoch(&self) -> AuthorityEpoch {
        self.inner.current_epoch()
    }
}

/// Opaque one-shot final-resolution authority.
///
/// This type intentionally implements neither `Copy` nor `Clone`. Observation
/// grants cannot be passed directly to strict final-resolution APIs because the
/// public types are distinct even though they reuse the same qualified trust
/// transport internally.
#[derive(Debug)]
pub struct ResolutionGrant {
    inner: TrustedBoundOneShotCapability<Observe>,
}

impl ResolutionGrant {
    /// Resolver authority domain that minted this grant.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.inner.domain_id()
    }

    /// Resolver revocation epoch in which this grant was minted.
    pub fn epoch(&self) -> AuthorityEpoch {
        self.inner.epoch()
    }

    /// Immutable grant metadata.
    pub fn metadata(&self) -> &GrantMetadata {
        self.inner.metadata()
    }

    /// Exact observed-lineage + decision digest this grant authorizes.
    pub fn binding(&self) -> [u8; 32] {
        self.inner.binding()
    }

    /// Validate domain, epoch, and expiry against a host-retained resolver verifier.
    pub fn validate_with(
        &self,
        verifier: &ResolutionVerifier,
        now: SystemTime,
    ) -> Result<(), TrustError> {
        self.inner.validate_with(&verifier.inner, now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unrelated_resolution_domain_is_rejected() {
        let expected = ResolutionAuthorityDomain::new(PrincipalId::new());
        let wrong = ResolutionAuthorityDomain::new(PrincipalId::new());
        let subject = PrincipalId::new();
        let scope = Scope::new("workspace", ["symthaea"]).unwrap();
        let grant = wrong.issue_bound_one_shot(subject, scope, None, [7; 32]);

        assert!(
            grant
                .validate_with(&expected.verifier(), SystemTime::now())
                .is_err()
        );
    }

    #[test]
    fn resolver_epoch_revocation_invalidates_old_grants() {
        let resolver = ResolutionAuthorityDomain::new(PrincipalId::new());
        let subject = PrincipalId::new();
        let scope = Scope::new("workspace", ["symthaea"]).unwrap();
        let grant = resolver.issue_bound_one_shot(subject, scope, None, [3; 32]);
        resolver.revoke_all().unwrap();

        assert!(
            grant
                .validate_with(&resolver.verifier(), SystemTime::now())
                .is_err()
        );
    }
}
