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

#![deny(unsafe_code)]

use std::sync::Arc;
use symthaea_extension_admission::{
    ActiveAdmission, AdmissionCurrentnessSource, AdmissionProblem, AdmissionRecord,
};
use symthaea_extension_core::ExtensionManifest;
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
    pub fn extension(&self) -> &symthaea_extension_core::ExtensionId {
        self.admission.extension()
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

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_extension_admission::{
        AdmissionContext, AdmissionSubject, PrincipalId, Sha256Digest, TrustLevel,
    };
    use symthaea_extension_core::{
        AbiVersion, CapabilityDescriptor, CapabilityId, EffectClass, ExtensionId, ExtensionKind,
        PermissionSet, ResourceBudget, RuntimeKind,
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
        assert_eq!(result, Err(AdmissionProblem::CurrentnessUnavailable));
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
}
