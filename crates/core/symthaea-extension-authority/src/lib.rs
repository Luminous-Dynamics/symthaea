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
//! Checked access to one or many `ActiveAdmission` values is closure-scoped.
//! Public callers cannot obtain a raw active-admission borrow or slice and carry
//! it beyond the currentness checks bracketing one operation.

#![deny(unsafe_code)]

use std::sync::Arc;
use symthaea_extension_admission::{
    ActiveAdmission, AdmissionCurrentnessSource, AdmissionProblem, AdmissionRecord,
};
use symthaea_extension_core::{ExtensionId, ExtensionManifest};
use thiserror::Error;

#[derive(Debug)]
struct AuthoritySeal;

#[derive(Debug)]
pub struct AdmissionAuthority {
    seal: Arc<AuthoritySeal>,
}

impl AdmissionAuthority {
    pub fn new() -> Self {
        Self {
            seal: Arc::new(AuthoritySeal),
        }
    }

    pub fn scope(&self) -> AuthorityScope {
        AuthorityScope {
            seal: Arc::clone(&self.seal),
        }
    }

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

#[derive(Debug, Clone)]
pub struct AuthorityScope {
    seal: Arc<AuthoritySeal>,
}

impl AuthorityScope {
    pub fn accepts(&self, admission: &ScopedAdmission) -> bool {
        Arc::ptr_eq(&self.seal, &admission.seal)
    }

    pub fn accepts_set(&self, admissions: &ScopedAdmissionSet) -> bool {
        Arc::ptr_eq(&self.seal, &admissions.seal)
    }

    /// Verify scope/currentness, run one operation, then recheck currentness
    /// before releasing the result.
    ///
    /// A post-use failure means the callback already ran; callers must not assume
    /// retry is safe for side-effecting operations.
    ///
    /// ```compile_fail
    /// use symthaea_extension_admission::{ActiveAdmission, AdmissionCurrentnessSource};
    /// use symthaea_extension_authority::{AuthorityScope, ScopedAdmission};
    ///
    /// fn leak<'s>(
    ///     scope: &AuthorityScope,
    ///     admission: &'s ScopedAdmission,
    ///     currentness: &dyn AdmissionCurrentnessSource,
    /// ) -> &'s ActiveAdmission {
    ///     scope
    ///         .with_rechecked(admission, currentness, |checked| checked)
    ///         .unwrap()
    /// }
    /// ```
    pub fn with_rechecked<R, F>(
        &self,
        admission: &ScopedAdmission,
        currentness: &dyn AdmissionCurrentnessSource,
        operation: F,
    ) -> Result<R, ScopedAdmissionError>
    where
        F: for<'a> FnOnce(&'a ActiveAdmission) -> R,
    {
        if !self.accepts(admission) {
            return Err(AuthorityScopeError::ForeignAuthority.into());
        }
        admission.admission.recheck_currentness(currentness)?;
        let result = operation(&admission.admission);
        admission
            .admission
            .recheck_currentness(currentness)
            .map_err(ScopedAdmissionError::PostUseCurrentness)?;
        Ok(result)
    }

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

    /// Verify the whole set before use, run one callback over the contiguous
    /// admission slice, then recheck the whole set before releasing the result.
    ///
    /// This is the appropriate boundary for an operation that genuinely depends
    /// on every admission in the set, such as deterministic candidate routing.
    /// After routing selects one provider, use [`Self::with_rechecked_member`] to
    /// bracket execution with only the winning provider's authority.
    ///
    /// Currentness lookups are sequential unless the injected source itself
    /// supplies stronger snapshot semantics. This API guarantees all-or-nothing
    /// exposure and entry/exit bracketing, not continuous revocation or an atomic
    /// multi-subject trust-store snapshot.
    ///
    /// [`ScopedAdmissionSetError::PostUseCurrentness`] means the callback already
    /// ran and its effects, if any, may have occurred.
    ///
    /// ```compile_fail
    /// use symthaea_extension_admission::{ActiveAdmission, AdmissionCurrentnessSource};
    /// use symthaea_extension_authority::{AuthorityScope, ScopedAdmissionSet};
    ///
    /// fn leak<'s>(
    ///     scope: &AuthorityScope,
    ///     admissions: &'s ScopedAdmissionSet,
    ///     currentness: &dyn AdmissionCurrentnessSource,
    /// ) -> &'s [ActiveAdmission] {
    ///     scope
    ///         .with_rechecked_set(admissions, currentness, |checked| checked)
    ///         .unwrap()
    /// }
    /// ```
    pub fn with_rechecked_set<R, F>(
        &self,
        admissions: &ScopedAdmissionSet,
        currentness: &dyn AdmissionCurrentnessSource,
        operation: F,
    ) -> Result<R, ScopedAdmissionSetError>
    where
        F: for<'a> FnOnce(&'a [ActiveAdmission]) -> R,
    {
        if !self.accepts_set(admissions) {
            return Err(AuthorityScopeError::ForeignAuthority.into());
        }

        recheck_all(&admissions.admissions, currentness, false)?;
        let result = operation(&admissions.admissions);
        recheck_all(&admissions.admissions, currentness, true)?;
        Ok(result)
    }

    /// Run one operation under exactly one member of a same-scope admission set.
    ///
    /// The requested extension must occur exactly once. This deliberately fails
    /// closed on missing or duplicate members rather than relying on a previous
    /// routing call. The selected member is rechecked immediately before and
    /// after the callback, so unrelated candidate admissions cannot invalidate a
    /// result after routing has already selected a winner.
    pub fn with_rechecked_member<R, F>(
        &self,
        admissions: &ScopedAdmissionSet,
        extension: &ExtensionId,
        currentness: &dyn AdmissionCurrentnessSource,
        operation: F,
    ) -> Result<R, ScopedAdmissionSetError>
    where
        F: for<'a> FnOnce(&'a ActiveAdmission) -> R,
    {
        if !self.accepts_set(admissions) {
            return Err(AuthorityScopeError::ForeignAuthority.into());
        }

        let mut matches = admissions
            .admissions
            .iter()
            .filter(|admission| admission.extension() == extension);
        let admission = matches
            .next()
            .ok_or_else(|| ScopedAdmissionSetError::MemberMissing {
                extension: extension.clone(),
            })?;
        if matches.next().is_some() {
            return Err(ScopedAdmissionSetError::MemberAmbiguous {
                extension: extension.clone(),
            });
        }

        admission
            .recheck_currentness(currentness)
            .map_err(|problem| ScopedAdmissionSetError::Currentness {
                extension: extension.clone(),
                problem,
            })?;
        let result = operation(admission);
        admission
            .recheck_currentness(currentness)
            .map_err(|problem| ScopedAdmissionSetError::PostUseCurrentness {
                extension: extension.clone(),
                problem,
            })?;
        Ok(result)
    }
}

fn recheck_all(
    admissions: &[ActiveAdmission],
    currentness: &dyn AdmissionCurrentnessSource,
    post_use: bool,
) -> Result<(), ScopedAdmissionSetError> {
    for admission in admissions {
        if let Err(problem) = admission.recheck_currentness(currentness) {
            let extension = admission.extension().clone();
            return Err(if post_use {
                ScopedAdmissionSetError::PostUseCurrentness { extension, problem }
            } else {
                ScopedAdmissionSetError::Currentness { extension, problem }
            });
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct ScopedAdmission {
    admission: ActiveAdmission,
    seal: Arc<AuthoritySeal>,
}

impl ScopedAdmission {
    pub fn extension(&self) -> &ExtensionId {
        self.admission.extension()
    }
}

#[derive(Debug)]
pub struct ScopedAdmissionSet {
    admissions: Vec<ActiveAdmission>,
    seal: Arc<AuthoritySeal>,
}

impl ScopedAdmissionSet {
    pub fn len(&self) -> usize {
        self.admissions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.admissions.is_empty()
    }

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
    #[error("admission is not current before use: {0:?}")]
    Currentness(AdmissionProblem),
    #[error("admission changed after the operation began; the operation may already have executed: {0:?}")]
    PostUseCurrentness(AdmissionProblem),
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
    #[error("admission set does not contain {extension:?}")]
    MemberMissing { extension: ExtensionId },
    #[error("admission set contains multiple entries for {extension:?}")]
    MemberAmbiguous { extension: ExtensionId },
    #[error("admission for {extension:?} is not current before use: {problem:?}")]
    Currentness {
        extension: ExtensionId,
        problem: AdmissionProblem,
    },
    #[error("admission for {extension:?} changed after the operation began; the operation may already have executed: {problem:?}")]
    PostUseCurrentness {
        extension: ExtensionId,
        problem: AdmissionProblem,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
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

    #[derive(Debug)]
    struct RevokeAfterFirstCheck {
        calls: AtomicUsize,
    }

    impl AdmissionCurrentnessSource for RevokeAfterFirstCheck {
        fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            Some(if call == 0 {
                AdmissionContext::active(7, 11)
            } else {
                AdmissionContext::revoked(7, 11)
            })
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
        assert!(matches!(
            foreign.scope().with_rechecked(&scoped, &current(), |_| ()),
            Err(ScopedAdmissionError::Scope(AuthorityScopeError::ForeignAuthority))
        ));
    }

    #[test]
    fn activation_still_fails_closed_when_currentness_is_unavailable() {
        let host = AdmissionAuthority::new();
        assert!(matches!(
            host.activate(&record(), &manifest(), &Currentness(None)),
            Err(AdmissionProblem::CurrentnessUnavailable)
        ));
    }

    #[test]
    fn pre_use_revocation_prevents_operation() {
        let host = AdmissionAuthority::new();
        let scope = host.scope();
        let scoped = host.activate(&record(), &manifest(), &current()).unwrap();
        let revoked = Currentness(Some(AdmissionContext::revoked(7, 11)));
        let operations = AtomicUsize::new(0);
        assert_eq!(
            scope.with_rechecked(&scoped, &revoked, |_| {
                operations.fetch_add(1, Ordering::SeqCst);
            }),
            Err(ScopedAdmissionError::Currentness(AdmissionProblem::Revoked))
        );
        assert_eq!(operations.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn post_use_revocation_withholds_single_result() {
        let host = AdmissionAuthority::new();
        let scope = host.scope();
        let scoped = host.activate(&record(), &manifest(), &current()).unwrap();
        let source = RevokeAfterFirstCheck {
            calls: AtomicUsize::new(0),
        };
        let operations = AtomicUsize::new(0);
        let error = scope
            .with_rechecked(&scoped, &source, |_| {
                operations.fetch_add(1, Ordering::SeqCst);
                42_u64
            })
            .unwrap_err();
        assert_eq!(operations.load(Ordering::SeqCst), 1);
        assert_eq!(
            error,
            ScopedAdmissionError::PostUseCurrentness(AdmissionProblem::Revoked)
        );
    }

    #[test]
    fn scoped_set_exposes_admissions_only_inside_checked_operation() {
        let host = AdmissionAuthority::new();
        let scope = host.scope();
        let first = host.activate(&record(), &manifest(), &current()).unwrap();
        let second = host.activate(&record(), &manifest(), &current()).unwrap();
        let set = scope.bundle(vec![first, second]).unwrap();
        let seen = scope
            .with_rechecked_set(&set, &current(), |admissions| admissions.len())
            .unwrap();
        assert_eq!(seen, 2);
    }

    #[test]
    fn selected_member_requires_exactly_one_matching_admission() {
        let host = AdmissionAuthority::new();
        let scope = host.scope();
        let id = ExtensionId::new("org.example.scoped");
        let empty = scope.bundle(Vec::new()).unwrap();
        assert_eq!(
            scope.with_rechecked_member(&empty, &id, &current(), |_| ()),
            Err(ScopedAdmissionSetError::MemberMissing {
                extension: id.clone()
            })
        );

        let first = host.activate(&record(), &manifest(), &current()).unwrap();
        let second = host.activate(&record(), &manifest(), &current()).unwrap();
        let duplicate = scope.bundle(vec![first, second]).unwrap();
        assert_eq!(
            scope.with_rechecked_member(&duplicate, &id, &current(), |_| ()),
            Err(ScopedAdmissionSetError::MemberAmbiguous { extension: id })
        );
    }

    #[test]
    fn selected_member_is_bracketed_independently() {
        let host = AdmissionAuthority::new();
        let scope = host.scope();
        let id = ExtensionId::new("org.example.scoped");
        let scoped = host.activate(&record(), &manifest(), &current()).unwrap();
        let set = scope.bundle(vec![scoped]).unwrap();
        let source = RevokeAfterFirstCheck {
            calls: AtomicUsize::new(0),
        };
        let operations = AtomicUsize::new(0);
        let error = scope
            .with_rechecked_member(&set, &id, &source, |_| {
                operations.fetch_add(1, Ordering::SeqCst);
                42_u64
            })
            .unwrap_err();
        assert_eq!(operations.load(Ordering::SeqCst), 1);
        assert!(matches!(
            error,
            ScopedAdmissionSetError::PostUseCurrentness {
                problem: AdmissionProblem::Revoked,
                ..
            }
        ));
    }

    #[test]
    fn foreign_member_rejects_whole_bundle() {
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
    fn foreign_scope_cannot_use_scoped_set() {
        let host = AdmissionAuthority::new();
        let foreign = AdmissionAuthority::new();
        let scoped = host.activate(&record(), &manifest(), &current()).unwrap();
        let set = host.scope().bundle(vec![scoped]).unwrap();
        assert!(matches!(
            foreign.scope().with_rechecked_set(&set, &current(), |_| ()),
            Err(ScopedAdmissionSetError::Scope(AuthorityScopeError::ForeignAuthority))
        ));
    }

    #[test]
    fn pre_use_revoked_member_prevents_set_operation() {
        let host = AdmissionAuthority::new();
        let scope = host.scope();
        let scoped = host.activate(&record(), &manifest(), &current()).unwrap();
        let set = scope.bundle(vec![scoped]).unwrap();
        let revoked = Currentness(Some(AdmissionContext::revoked(7, 11)));
        let operations = AtomicUsize::new(0);
        assert!(matches!(
            scope.with_rechecked_set(&set, &revoked, |_| {
                operations.fetch_add(1, Ordering::SeqCst);
            }),
            Err(ScopedAdmissionSetError::Currentness {
                problem: AdmissionProblem::Revoked,
                ..
            })
        ));
        assert_eq!(operations.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn post_use_revoked_member_withholds_set_result() {
        let host = AdmissionAuthority::new();
        let scope = host.scope();
        let scoped = host.activate(&record(), &manifest(), &current()).unwrap();
        let set = scope.bundle(vec![scoped]).unwrap();
        let source = RevokeAfterFirstCheck {
            calls: AtomicUsize::new(0),
        };
        let operations = AtomicUsize::new(0);
        let error = scope
            .with_rechecked_set(&set, &source, |_| {
                operations.fetch_add(1, Ordering::SeqCst);
                42_u64
            })
            .unwrap_err();
        assert_eq!(operations.load(Ordering::SeqCst), 1);
        assert!(matches!(
            error,
            ScopedAdmissionSetError::PostUseCurrentness {
                problem: AdmissionProblem::Revoked,
                ..
            }
        ));
    }

    #[test]
    fn empty_scoped_set_is_valid() {
        let host = AdmissionAuthority::new();
        let scope = host.scope();
        let set = scope.bundle(Vec::new()).unwrap();
        assert!(set.is_empty());
        assert_eq!(
            scope.with_rechecked_set(&set, &current(), |admissions| admissions.len()),
            Ok(0)
        );
    }
}
