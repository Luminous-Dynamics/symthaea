// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed runtime assurance boundary for welfare-sensitive interventions.
//!
//! `AssuredInterventionPermit` is the positive capability emitted only after the full hardened
//! chain succeeds. The lower-level core `PolicyPass` remains a negative-policy result and must
//! not be treated as execution authority by production integrations.

#![deny(unsafe_code)]

#[cfg(test)]
#[path = "lib.rs"]
mod legacy_composition_tests;

use std::error::Error as StdError;

use symthaea_core::intervention_interlock::{
    BilateralInterventionInterlock, ExplicitConsentState, InterlockDecision, InterventionRequest,
};
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_fabrication_kernel::trust::{TrustSnapshot, digest_trust_snapshot};
use symthaea_welfare_authority::{
    SignedWelfareInterventionAuthority, WelfareAuthorityContextError,
    WelfareAuthorityPolicyManifest, WelfareAuthoritySignatureVerifier, WelfareAuthorityTracker,
    WelfareAuthorityUseError, digest_welfare_authority_policy_manifest, evaluate_once,
    verify_context_bound_welfare_authority,
};
use symthaea_welfare_consent::{
    LiveSubjectConsentUseError, SubjectConsentLedger, SubjectIdentityRegistry,
    SubjectIdentityStatus, bind_latest_live,
};
use thiserror::Error;

/// Non-clone, non-serializable capability proving that the full welfare-assurance chain passed.
///
/// The private fields intentionally prevent callers from manufacturing this token from a bare
/// `InterlockDecision::PolicyPass` or a textual authority/consent reference.
#[derive(Debug)]
pub struct AssuredInterventionPermit {
    request: InterventionRequest,
    subject_id: String,
    minted_at_unix_s: u64,
    not_after_unix_s: u64,
    consent_statement_digest: Option<Sha256Digest>,
    consent_identity_epoch: Option<u64>,
    authority_statement_digest: Sha256Digest,
    authority_policy_manifest_digest: Sha256Digest,
    authority_trust_snapshot_digest: Sha256Digest,
}

impl AssuredInterventionPermit {
    /// Subject/instance/lineage targeted by the permitted intervention.
    pub fn target_id(&self) -> &str {
        &self.request.target_id
    }

    /// Exact coarse intervention class that passed the assurance chain.
    pub fn action(&self) -> SubjectAffectingAction {
        self.request.action
    }

    /// Human/machine-readable rationale that was covered by consent and authority scope.
    pub fn rationale(&self) -> &str {
        &self.request.rationale
    }

    /// Whether the signed request asserted emergency containment.
    ///
    /// This is read-only audit/policy context. It does not make the permit more authoritative.
    pub fn is_emergency(&self) -> bool {
        self.request.emergency
    }

    /// Explicit consent state that was bound into the verified request.
    pub fn explicit_consent_state(&self) -> ExplicitConsentState {
        self.request.evidence.consent_state
    }

    /// Whether a welfare-review reference was bound into the verified request.
    ///
    /// Presence is useful to stricter domain adapters; the opaque reference itself remains an
    /// upstream evidence locator and is not reinterpreted here.
    pub fn has_welfare_review_reference(&self) -> bool {
        self.request.evidence.welfare_review_ref.is_some()
    }

    /// Whether an independent-review reference was bound into the verified request.
    pub fn has_independent_review_reference(&self) -> bool {
        self.request.evidence.independent_review_ref.is_some()
    }

    /// Time at which the full assurance permit was minted.
    pub fn minted_at_unix_s(&self) -> u64 {
        self.minted_at_unix_s
    }

    /// Earliest expiry across authority, trust, live consent and subject identity windows.
    pub fn not_after_unix_s(&self) -> u64 {
        self.not_after_unix_s
    }

    /// Digest of the active consent statement, when this intervention used explicit consent.
    pub fn consent_statement_digest(&self) -> Option<Sha256Digest> {
        self.consent_statement_digest
    }

    /// Subject identity epoch under which the active consent was signed.
    pub fn consent_identity_epoch(&self) -> Option<u64> {
        self.consent_identity_epoch
    }

    /// Digest of the verified operator-authority statement.
    pub fn authority_statement_digest(&self) -> Sha256Digest {
        self.authority_statement_digest
    }

    /// Digest of the signer policy committed by the authority signature.
    pub fn authority_policy_manifest_digest(&self) -> Sha256Digest {
        self.authority_policy_manifest_digest
    }

    /// Digest of the exact trust snapshot committed by the authority signature.
    pub fn authority_trust_snapshot_digest(&self) -> Sha256Digest {
        self.authority_trust_snapshot_digest
    }
}

/// Run the complete hardened chain exactly once and mint a typed permit only on `PolicyPass`.
///
/// Ordering is intentional:
///
/// 1. resolve live subject consent against the current subject identity;
/// 2. verify authority against the exact policy/trust context;
/// 3. require that authority to match the consent-bound request;
/// 4. consume the authority through replay fencing;
/// 5. run the bilateral interlock;
/// 6. emit an opaque permit only if the result is `PolicyPass`.
///
/// A legacy pre-populated `authority_ref` is rejected at this boundary even if the core policy
/// checker would otherwise accept a non-empty string for compatibility.
#[allow(clippy::too_many_arguments)]
pub fn authorize_intervention_once(
    subject_id: &str,
    consent_ledger: &SubjectConsentLedger,
    subject_registry: &SubjectIdentityRegistry,
    authority_nonce: &str,
    signed_authority: &SignedWelfareInterventionAuthority,
    authority_manifest: &WelfareAuthorityPolicyManifest,
    authority_trust_snapshot: &TrustSnapshot,
    authority_verifier: &dyn WelfareAuthoritySignatureVerifier,
    authority_tracker: &mut WelfareAuthorityTracker,
    request: &InterventionRequest,
    unix_s: u64,
) -> Result<AssuredInterventionPermit, AssuranceGateError> {
    if request.evidence.authority_ref.is_some() {
        return Err(AssuranceGateError::LegacyAuthorityReferencePresent);
    }

    let consent_bound =
        bind_latest_live(consent_ledger, subject_registry, subject_id, request, unix_s)?;

    let verified_authority = verify_context_bound_welfare_authority(
        authority_nonce,
        signed_authority,
        authority_manifest,
        authority_trust_snapshot,
        unix_s,
        authority_verifier,
    )?;

    let decision = evaluate_once(
        authority_tracker,
        &verified_authority,
        &BilateralInterventionInterlock,
        &consent_bound,
        unix_s,
    )?;

    if decision != InterlockDecision::PolicyPass {
        return Err(AssuranceGateError::PolicyRejected { decision });
    }

    // Reconstruct the exact fully-bound request for audit identity. This operation is pure;
    // replay consumption already happened above and is not repeated here.
    let fully_bound = verified_authority.bind_request(&consent_bound, unix_s)?;

    let active_consent = if fully_bound.evidence.consent_ref.is_some() {
        consent_ledger.latest_for(subject_id, &fully_bound.target_id, fully_bound.action)
    } else {
        None
    };
    let active_identity = active_consent.and_then(|_| subject_registry.binding(subject_id));

    let authority_policy_manifest_digest =
        digest_welfare_authority_policy_manifest(authority_manifest)?;

    let mut not_after_unix_s = verified_authority
        .statement()
        .expires_at_unix_s
        .min(authority_trust_snapshot.expires_at_unix_s);
    if let Some(consent) = active_consent {
        not_after_unix_s = not_after_unix_s.min(consent.statement().expires_at_unix_s);
    }
    if let Some(identity_expiry) = active_identity.and_then(|identity| identity.not_after_unix_s) {
        not_after_unix_s = not_after_unix_s.min(identity_expiry);
    }
    if not_after_unix_s <= unix_s {
        return Err(AssuranceGateError::NoLivePermitWindow);
    }

    Ok(AssuredInterventionPermit {
        request: fully_bound,
        subject_id: subject_id.to_string(),
        minted_at_unix_s: unix_s,
        not_after_unix_s,
        consent_statement_digest: active_consent.map(|consent| consent.statement_digest()),
        consent_identity_epoch: active_consent.map(|consent| consent.statement().identity_epoch),
        authority_statement_digest: verified_authority.statement_digest(),
        authority_policy_manifest_digest,
        authority_trust_snapshot_digest: verified_authority.trust_snapshot_digest(),
    })
}

/// Executor called only after the permit's live mutable contexts have been revalidated.
///
/// The permit is consumed by `execute_permit_once`, and the executor receives only a borrowed
/// capability during that call. This makes the intended safe integration path validate-and-act
/// rather than validate-now / actuate-arbitrarily-later.
pub trait AssuredInterventionExecutor {
    type Output;
    type Error: StdError + Send + Sync + 'static;

    fn execute(
        &mut self,
        permit: &AssuredInterventionPermit,
    ) -> Result<Self::Output, Self::Error>;
}

/// Revalidate all mutable contexts and immediately consume the permit through an executor.
///
/// This prevents a held permit from silently surviving later consent withdrawal, subject-key
/// rotation/revocation, authority-policy mutation, trust-snapshot rollover/revocation, or expiry.
#[allow(clippy::too_many_arguments)]
pub fn execute_permit_once<E: AssuredInterventionExecutor>(
    permit: AssuredInterventionPermit,
    consent_ledger: &SubjectConsentLedger,
    subject_registry: &SubjectIdentityRegistry,
    current_authority_manifest: &WelfareAuthorityPolicyManifest,
    current_trust_snapshot: &TrustSnapshot,
    unix_s: u64,
    executor: &mut E,
) -> Result<E::Output, PermitExecutionError<E::Error>> {
    if unix_s < permit.minted_at_unix_s {
        return Err(PermitExecutionError::TimeRegression {
            minted_at: permit.minted_at_unix_s,
            proposed: unix_s,
        });
    }
    if unix_s >= permit.not_after_unix_s {
        return Err(PermitExecutionError::Expired {
            not_after: permit.not_after_unix_s,
            proposed: unix_s,
        });
    }

    let current_policy_digest =
        digest_welfare_authority_policy_manifest(current_authority_manifest)?;
    if current_policy_digest != permit.authority_policy_manifest_digest {
        return Err(PermitExecutionError::AuthorityPolicyChanged);
    }

    let current_trust_digest = digest_trust_snapshot(current_trust_snapshot)
        .map_err(|error| PermitExecutionError::TrustSnapshotInvalid(format!("{error:?}")))?;
    if current_trust_digest != permit.authority_trust_snapshot_digest {
        return Err(PermitExecutionError::AuthorityTrustContextChanged);
    }
    if !current_trust_snapshot.is_fresh_at(unix_s) {
        return Err(PermitExecutionError::AuthorityTrustSnapshotStale);
    }

    if let Some(expected_consent_digest) = permit.consent_statement_digest {
        let consent = consent_ledger
            .latest_for(&permit.subject_id, &permit.request.target_id, permit.request.action)
            .ok_or(PermitExecutionError::ConsentChangedOrMissing)?;
        if consent.statement_digest() != expected_consent_digest {
            return Err(PermitExecutionError::ConsentChangedOrMissing);
        }
        let statement = consent.statement();
        if unix_s < statement.issued_at_unix_s || unix_s >= statement.expires_at_unix_s {
            return Err(PermitExecutionError::ConsentExpired);
        }

        let identity = subject_registry
            .binding(&permit.subject_id)
            .ok_or(PermitExecutionError::SubjectIdentityChanged)?;
        let (signer_algorithm, signer_key_id) = consent.signer();
        if identity.status != SubjectIdentityStatus::Active
            || identity.identity_epoch != statement.identity_epoch
            || !identity.active_at(unix_s)
            || &identity.algorithm != signer_algorithm
            || identity.key_id.as_str() != signer_key_id
        {
            return Err(PermitExecutionError::SubjectIdentityChanged);
        }
    }

    executor
        .execute(&permit)
        .map_err(PermitExecutionError::Executor)
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AssuranceGateError {
    /// Hardened callers must not pre-seed the compatibility string authority field.
    #[error("legacy authority_ref is not accepted at the assured-intervention boundary")]
    LegacyAuthorityReferencePresent,
    /// Live subject-consent resolution failed.
    #[error(transparent)]
    Consent(#[from] LiveSubjectConsentUseError),
    /// Authority policy/trust context verification failed.
    #[error(transparent)]
    AuthorityContext(#[from] WelfareAuthorityContextError),
    /// Exact authority binding, replay fencing, or structural interlock evaluation failed.
    #[error(transparent)]
    AuthorityUse(#[from] WelfareAuthorityUseError),
    /// The authority was valid, but bilateral consent/welfare policy did not pass.
    #[error("bilateral intervention policy rejected the request: {decision:?}")]
    PolicyRejected { decision: InterlockDecision },
    /// All assurance inputs passed individually but had no positive shared lifetime remaining.
    #[error("assurance inputs have no live permit window")]
    NoLivePermitWindow,
}

#[derive(Debug, Error)]
pub enum PermitExecutionError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("permit use time regressed before mint time: minted={minted_at}, proposed={proposed}")]
    TimeRegression { minted_at: u64, proposed: u64 },
    #[error("permit expired at {not_after}; proposed use={proposed}")]
    Expired { not_after: u64, proposed: u64 },
    #[error("authority policy changed after permit mint")]
    AuthorityPolicyChanged,
    #[error("authority trust snapshot changed after permit mint")]
    AuthorityTrustContextChanged,
    #[error("authority trust snapshot is no longer fresh")]
    AuthorityTrustSnapshotStale,
    #[error("trust snapshot invalid: {0}")]
    TrustSnapshotInvalid(String),
    #[error("consent was withdrawn, replaced, or removed after permit mint")]
    ConsentChangedOrMissing,
    #[error("consent expired after permit mint")]
    ConsentExpired,
    #[error("subject identity/key/epoch changed after permit mint")]
    SubjectIdentityChanged,
    #[error(transparent)]
    AuthorityContext(#[from] WelfareAuthorityContextError),
    #[error("assured intervention executor failed: {0}")]
    Executor(#[source] E),
}
