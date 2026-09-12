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

use symthaea_core::intervention_interlock::{
    BilateralInterventionInterlock, InterlockDecision, InterventionRequest,
};
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_fabrication_kernel::trust::TrustSnapshot;
use symthaea_welfare_authority::{
    SignedWelfareInterventionAuthority, WelfareAuthorityContextError,
    WelfareAuthorityPolicyManifest, WelfareAuthoritySignatureVerifier, WelfareAuthorityTracker,
    WelfareAuthorityUseError, digest_welfare_authority_policy_manifest, evaluate_once,
    verify_context_bound_welfare_authority,
};
use symthaea_welfare_consent::{
    LiveSubjectConsentUseError, SubjectConsentLedger, SubjectIdentityRegistry, bind_latest_live,
};
use thiserror::Error;

/// Non-clone, non-serializable capability proving that the full welfare-assurance chain passed.
///
/// The private fields intentionally prevent callers from manufacturing this token from a bare
/// `InterlockDecision::PolicyPass` or a textual authority/consent reference.
#[derive(Debug)]
pub struct AssuredInterventionPermit {
    request: InterventionRequest,
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

    let authority_policy_manifest_digest =
        digest_welfare_authority_policy_manifest(authority_manifest)?;

    Ok(AssuredInterventionPermit {
        request: fully_bound,
        consent_statement_digest: active_consent.map(|consent| consent.statement_digest()),
        consent_identity_epoch: active_consent.map(|consent| consent.statement().identity_epoch),
        authority_statement_digest: verified_authority.statement_digest(),
        authority_policy_manifest_digest,
        authority_trust_snapshot_digest: verified_authority.trust_snapshot_digest(),
    })
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
}
