// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Process-local authority scoping for Symthaea extension admissions.
//!
//! `ActiveAdmission` proves that an admission was current when activated and can
//! be rechecked later, but it is intentionally runtime-neutral. This crate adds a
//! second property for host orchestration: the admission must also belong to the
//! exact process-local authority instance trusted by the consumer.
//!
//! The scope identity is an unexported `Arc` allocation. It is not serialized,
//! hashed, named, or reconstructed from manifest/principal strings. A caller may
//! create a different authority instance, but admissions minted by that instance
//! are rejected by consumers bound to another [`AuthorityScope`].
//!
//! Multi-provider consumers can bundle scoped tokens into a [`ScopedAdmissionSet`].
//! The set preserves a contiguous `Vec<ActiveAdmission>` internally so existing
//! routing code can remain authority-agnostic. That slice is exposed only through
//! the matching [`AuthorityScope`] and only after every member passes a fresh
//! point-of-use currentness recheck.

#![deny(unsafe_code)]

use std::sync::Arc;
use symthaea_extension_admission::{
    ActiveAdmission, AdmissionCurrentnessSource, AdmissionProblem, AdmissionRecord,
};
use symthaea_extension_core::{ExtensionId, ExtensionManifest};
use thiserror::Error;

#[derive(Debug)]
struct AuthoritySeal;

/// Minting owner for one process-local extension authority instance.
///
/// Keep this value inside trusted host orchestration. Hand verification-only
/// [`AuthorityScope`] values to consumers that need to accept scoped admissions.
#[derive(Debug)]
pub struct AdmissionAuthority {
    seal: Arc<AuthoritySeal>,
}

impl AdmissionAuthority {
    /// Create a fresh authority instance. Each call produces a distinct scope.
    pub fn new() -> Self {
        Self {
            seal: Arc::new(AuthoritySeal),
        }
    }

    /// Obtain a verification-only handle for consumers owned by this authority.
    pub fn scope(&self) -> AuthorityScope {
        AuthorityScope {
            seal: Arc::clone(&self.seal),
        }
    }

    /// Activate an admission against live host currentness and bind the resulting
    /// token to this exact process-local authority instance.
    pub fn activate(
        &self,
        record: &AdmissionRecord,
        manifest: &ExtensionManifest,
        currentness: &dyn AdmissionCurrentnessSource,
    ) -> Result<ScopedAdmission, AdmissionProblem> {
        let admission = record.activate(manifest, currentness)?;
        Ok(ScopedAdmission {
            admission,
            seal: Arc::clone(&self.seal),
        })
    }
}

impl Default for AdmissionAuthority {
    fn default() -> Self {
        Self::new()
    }
}

/// Cloneable verification-only scope for one [`AdmissionAuthority`].
///
/// This handle can verify and recheck admissions from its authority instance but
/// cannot mint new [`ScopedAdmission`] values.
#[derive(Debug, Clone)]
pub struct AuthorityScope {
    seal: Arc<AuthoritySeal>,
}

impl AuthorityScope {
    /// Return true only when `admission` was minted by this exact authority
    /// instance. Equality is process-local pointer identity, not string identity.
    pub fn accepts(&self, admission: &ScopedAdmission) -> bool {
        Arc::ptr_eq(&self.seal, &admission.seal)
    }

    /// Return true only when `admissions` was bundled by this exact authority
    /// scope. The set's members remain hidden until a live recheck succeeds.
    pub fn accepts_set(&self, admissions: &ScopedAdmissionSet) -> bool {
        Arc::ptr_eq(&self.seal, &admissions.seal)
    }

    /// Borrow the underlying live admission only after scope identity is proven.
    pub fn admission<'a>(
        &self,
        admission: &'a ScopedAdmission,
    ) -> Result<&'a ActiveAdmission, AuthorityScopeError> {
        if !self.accepts(admission) {
            return Err(AuthorityScopeError::ForeignAuthority);
        }
        Ok(&admission.admission)
    }

    /// Verify authority-instance identity and then re-resolve live currentness at
    /// the actual use site.
    pub fn recheck<'a>(
        &self,
        admission: &'a ScopedAdmission,
        currentness: &dyn AdmissionCurrentnessSource,
    ) -> Result<&'a ActiveAdmission, ScopedAdmissionError> {
        let admission = self.admission(admission)?;
        admission.recheck_currentness(currentness)?;
        Ok(admission)
    }

    /// Combine multiple already-scoped admissions into one non-cloneable,
    /// process-local set without cloning or serializing authority.
    ///
    /// Every member must belong to this exact authority instance. The check is
    /// completed for the whole input before any inner admission is moved into the
    /// returned set. Duplicate extensions are intentionally preserved so the
    /// capability router can continue to fail closed on duplicate admissions.
    pub fn bundle(
        &self,
        admissions: Vec<ScopedAdmission>,
    ) -> Result<ScopedAdmissionSet, AuthorityScopeError> {
        if admissions.iter().any(|admission| !self.accepts(admission)) {
            return Err(AuthorityScopeError::ForeignAuthority);
        }

        Ok(ScopedAdmissionSet {
            admissions: admissions
                .into_iter()
                .map(|admission| admission.admission)
                .collect(),
            seal: Arc::clone(&self.seal),
        })
    }

    /// Verify this set belongs to the same host authority and recheck every
    /// admission against the authoritative live currentness source before
    /// exposing the contiguous slice used by routing.
    ///
    /// No partially checked slice is returned: the first stale/revoked member
    /// fails the entire operation.
    pub fn recheck_set<'a>(
        &self,
        admissions: &'a ScopedAdmissionSet,
        currentness: &dyn AdmissionCurrentnessSource,
    ) -> Result<&'a [ActiveAdmission], ScopedAdmissionSetError> {
        if !self.accepts_set(admissions) {
            return Err(AuthorityScopeError::ForeignAuthority.into());
        }

        for admission in &admissions.admissions {
            if let Err(problem) = admission.recheck_currentness(currentness) {
                return Err(ScopedAdmissionSetError::Currentness {
                    extension: admission.extension().clone(),
                    problem,
                });
            }
        }

        Ok(&admissions.admissions)
    }
}

/// Non-serializable active admission bound to one process-local authority.
///
/// The underlying `ActiveAdmission` is deliberately not exposed directly; a
/// consumer must present the matching [`AuthorityScope`] to borrow it.
#[derive(Debug)]
pub struct ScopedAdmission {
    admission: ActiveAdmission,
    seal: Arc<AuthoritySeal>,
}

impl ScopedAdmission {
    /// Convenience identity accessor that does not grant use authority.
    pub fn extension(&self) -> &ExtensionId {
        self.admission.extension()
    }
}

/// Non-serializable collection of admissions from one process-local authority.
///
/// This type intentionally does not implement `Clone` or Serde. Its inner slice
/// is not exposed directly; callers need the matching [`AuthorityScope`] and a
/// live [`AdmissionCurrentnessSource`] via [`AuthorityScope::recheck_set`].
#[derive(Debug)]
pub struct ScopedAdmissionSet {
    admissions: Vec<ActiveAdmission>,
    seal: Arc<AuthoritySeal>,
}

impl ScopedAdmissionSet {
    /// Number of scoped admissions. This is metadata, not use authority.
    pub fn len(&self) -> usize {
        self.admissions.len()
    }

    /// Whether the set contains no admissions.
    pub fn is_empty(&self) -> bool {
        self.admissions.is_empty()
    }

    /// Iterate provider identities without exposing the underlying admissions.
    pub fn extensions(&self) -> impl Iterator<Item = &ExtensionId> {
        self.admissions.iter().map(ActiveAdmission::extension)
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum AuthorityScopeError {
    #[error("admission belongs to a different process-local authority instance")]
    ForeignAuthority,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ScopedAdmissionError {
    #[error(transparent)]
    Scope(#[from] AuthorityScopeError),
    #[error("admission is no longer current: {0:?}")]
    Currentness(AdmissionProblem),
}

impl From<AdmissionProblem> for ScopedAdmissionError {
    fn from(value: AdmissionProblem) -> Self {
        Self::Currentness(value)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ScopedAdmissionSetError {
    #[error(transparent)]
    Scope(#[from] AuthorityScopeError),
    #[error("admission for {extension} is no longer current: {problem:?}")]
    Currentness {
        extension: ExtensionId,
        problem: AdmissionProblem,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_extension_admission::{
        AdmissionContext, AdmissionSubject, PrincipalId, Sha256Digest, TrustLevel,
    };
    use symthaea_extension_core::{
        AbiVersion, CapabilityDescriptor, CapabilityId, EffectClass, ExtensionKind, PermissionSet,
        ResourceBudget, RuntimeKind,
    };

    #[derive(Debug, Clone, Copy)]
    struct Currentness(Option<AdmissionContext>);

    impl AdmissionCurrentnessSource for Currentness {
        fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
            self.0
        }
    }

    fn manifest() -> ExtensionManifest {
        ExtensionManifest {
            id: ExtensionId::new("org.example.scoped"),
            name: "Scoped provider".into(),
            version: "1.0.0".into(),
            abi: AbiVersion::V1,
            kind: ExtensionKind::Simulation,
            runtime: RuntimeKind::Native,
            description: String::new(),
            provides: vec![CapabilityDescriptor {
                id: CapabilityId::new("engineering.simulation.circuit"),
                description: "test capability".into(),
                effect: EffectClass::Pure,
            }],
            requires: vec![],
            permissions: PermissionSet::default(),
            resources: ResourceBudget::default(),
        }
    }

    fn record() -> AdmissionRecord {
        AdmissionRecord::issue(
            ExtensionId::new("org.example.scoped"),
            "1.0.0",
            Sha256Digest::new([1; 32]),
            Sha256Digest::new([2; 32]),
            Sha256Digest::new([3; 32]),
            PrincipalId::new("local:extension-authority").unwrap(),
            Some(PrincipalId::new("did:example:publisher").unwrap()),
            TrustLevel::Trusted,
            vec![CapabilityId::new("engineering.simulation.circuit")],
            PermissionSet::default(),
            7,
            11,
        )
        .unwrap()
    }

    fn current() -> Currentness {
        Currentness(Some(AdmissionContext::active(7, 11)))
    }

    #[test]
    fn own_scope_accepts_and_foreign_scope_rejects_same_record() {
        let host = AdmissionAuthority::new();
        let foreign = AdmissionAuthority::new();
        let scoped = host.activate(&record(), &manifest(), &current()).unwrap();

        assert!(host.scope().accepts(&scoped));
        assert!(!foreign.scope().accepts(&scoped));
        assert_eq!(
            foreign.scope().admission(&scoped),
            Err(AuthorityScopeError::ForeignAuthority)
        );
    }

    #[test]
    fn separately_minted_admission_cannot_cross_authority_scope() {
        let host = AdmissionAuthority::new();
        let impostor = AdmissionAuthority::new();
        let host_scoped = host.activate(&record(), &manifest(), &current()).unwrap();
        let impostor_scoped = impostor.activate(&record(), &manifest(), &current()).unwrap();

        assert!(host.scope().admission(&host_scoped).is_ok());
        assert_eq!(
            host.scope().admission(&impostor_scoped),
            Err(AuthorityScopeError::ForeignAuthority)
        );
    }

    #[test]
    fn activation_still_fails_closed_when_currentness_is_unavailable() {
        let host = AdmissionAuthority::new();
        let result = host.activate(&record(), &manifest(), &Currentness(None));
        assert!(matches!(
            result,
            Err(AdmissionProblem::CurrentnessUnavailable)
        ));
    }

    #[test]
    fn scope_recheck_invalidates_revoked_token_at_use_time() {
        let host = AdmissionAuthority::new();
        let scope = host.scope();
        let scoped = host.activate(&record(), &manifest(), &current()).unwrap();
        let revoked = Currentness(Some(AdmissionContext::revoked(7, 11)));

        assert!(scope.recheck(&scoped, &current()).is_ok());
        assert_eq!(
            scope.recheck(&scoped, &revoked),
            Err(ScopedAdmissionError::Currentness(AdmissionProblem::Revoked))
        );
    }

    #[test]
    fn cloned_verification_scope_preserves_identity_without_minting() {
        let host = AdmissionAuthority::new();
        let scope = host.scope();
        let clone = scope.clone();
        let scoped = host.activate(&record(), &manifest(), &current()).unwrap();

        assert!(scope.accepts(&scoped));
        assert!(clone.accepts(&scoped));
    }

    #[test]
    fn scoped_set_exposes_contiguous_admissions_only_after_live_recheck() {
        let host = AdmissionAuthority::new();
        let scope = host.scope();
        let first = host.activate(&record(), &manifest(), &current()).unwrap();
        let second = host.activate(&record(), &manifest(), &current()).unwrap();
        let set = scope.bundle(vec![first, second]).unwrap();

        assert_eq!(set.len(), 2);
        assert!(scope.accepts_set(&set));
        let admissions = scope.recheck_set(&set, &current()).unwrap();
        assert_eq!(admissions.len(), 2);
        assert_eq!(admissions[0].extension(), &ExtensionId::new("org.example.scoped"));
    }

    #[test]
    fn foreign_member_rejects_whole_bundle_before_authority_is_stripped() {
        let host = AdmissionAuthority::new();
        let foreign = AdmissionAuthority::new();
        let own = host.activate(&record(), &manifest(), &current()).unwrap();
        let outsider = foreign.activate(&record(), &manifest(), &current()).unwrap();

        assert_eq!(
            host.scope().bundle(vec![own, outsider]).unwrap_err(),
            AuthorityScopeError::ForeignAuthority
        );
    }

    #[test]
    fn foreign_scope_cannot_open_scoped_set() {
        let host = AdmissionAuthority::new();
        let foreign = AdmissionAuthority::new();
        let scoped = host.activate(&record(), &manifest(), &current()).unwrap();
        let set = host.scope().bundle(vec![scoped]).unwrap();

        assert!(matches!(
            foreign.scope().recheck_set(&set, &current()),
            Err(ScopedAdmissionSetError::Scope(
                AuthorityScopeError::ForeignAuthority
            ))
        ));
    }

    #[test]
    fn one_revoked_member_fails_entire_set_without_partial_slice() {
        let host = AdmissionAuthority::new();
        let scope = host.scope();
        let scoped = host.activate(&record(), &manifest(), &current()).unwrap();
        let set = scope.bundle(vec![scoped]).unwrap();
        let revoked = Currentness(Some(AdmissionContext::revoked(7, 11)));

        assert!(matches!(
            scope.recheck_set(&set, &revoked),
            Err(ScopedAdmissionSetError::Currentness {
                problem: AdmissionProblem::Revoked,
                ..
            })
        ));
    }

    #[test]
    fn empty_scoped_set_is_valid_and_exposes_empty_slice() {
        let host = AdmissionAuthority::new();
        let scope = host.scope();
        let set = scope.bundle(Vec::new()).unwrap();

        assert!(set.is_empty());
        assert!(scope.recheck_set(&set, &current()).unwrap().is_empty());
    }
}
