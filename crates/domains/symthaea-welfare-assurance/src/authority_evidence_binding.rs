// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bind existing operator-authority signatures to the exact welfare-evidence context.
//!
//! The existing welfare-authority statement already signs an authority ID. Rather than add a
//! second signature format, this adapter deterministically derives the authority nonce from the
//! caller's base nonce plus the exact moral-patient evidence/policy context. The existing
//! policy/trust-bound authority ID therefore commits the same signatures to that evidence context.

use symthaea_core::intervention_interlock::{InterventionRequest, WelfareConstraintLevel};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::trust::TrustSnapshot;
use symthaea_psych_bench::moral_patient::{
    MoralPatientEvidenceProfile, PrecautionPolicy, ProtectionDisposition,
};
use symthaea_welfare_authority::{
    SignedWelfareInterventionAuthority, WelfareAuthorityContextError,
    WelfareAuthorityPolicyManifest, WelfareAuthoritySignatureVerifier, WelfareAuthorityTracker,
    WelfareInterventionAuthorityStatement, build_context_bound_authority_statement,
};
use symthaea_welfare_consent::{SubjectConsentLedger, SubjectIdentityRegistry};
use thiserror::Error;

use crate::evidence_context::{
    EvidenceBoundAuthorizationError, EvidenceBoundInterventionPermit, WelfareEvidenceContext,
    WelfareEvidenceContextError, authorize_intervention_once_with_welfare_evidence,
    derive_welfare_evidence_context,
};

const EVIDENCE_BOUND_NONCE_DOMAIN: &[u8] =
    b"symthaea.welfare.evidence-bound-authority-nonce.v1\0";
const MAX_BASE_NONCE_BYTES: usize = 256;

/// Derive the nonce used by the existing policy/trust-bound authority statement.
///
/// Output is always a canonical 64-character lowercase SHA-256 hex string, which fits the
/// authority layer's existing nonce bound. No evidence is collapsed into a score: the digest
/// commits the exact profile and policy commitments plus their typed disposition/constraint.
pub fn evidence_bound_authority_nonce(
    base_nonce: &str,
    context: WelfareEvidenceContext,
) -> Result<String, EvidenceBoundAuthorityError> {
    validate_base_nonce(base_nonce)?;

    let mut hasher = Sha256::new();
    hasher.update(EVIDENCE_BOUND_NONCE_DOMAIN);
    hasher.update(&(base_nonce.len() as u64).to_be_bytes());
    hasher.update(base_nonce.as_bytes());
    hasher.update(&context.profile_digest().0);
    hasher.update(&context.policy_digest().0);
    hasher.update(&[disposition_code(context.disposition())]);
    hasher.update(&[constraint_code(context.constraint())]);
    Ok(hex_digest(hasher.finalize()))
}

/// Construct the normal signed-authority statement with an ID that also commits the current
/// welfare-evidence context.
///
/// Call this before `sign_welfare_authority`. The request must already carry the exact
/// evidence-derived `WelfareConstraintLevel`; use `bind_welfare_evidence_to_request` upstream.
#[allow(clippy::too_many_arguments)]
pub fn build_evidence_bound_authority_statement(
    base_nonce: &str,
    profile: &MoralPatientEvidenceProfile,
    precaution_policy: &PrecautionPolicy,
    authority_manifest: &WelfareAuthorityPolicyManifest,
    trust_snapshot: &TrustSnapshot,
    authority_epoch: u64,
    sequence: u64,
    issued_at_unix_s: u64,
    expires_at_unix_s: u64,
    request: &InterventionRequest,
) -> Result<WelfareInterventionAuthorityStatement, EvidenceBoundAuthorityError> {
    let context = derive_welfare_evidence_context(profile, precaution_policy)?;
    if request.welfare_constraint != context.constraint() {
        return Err(EvidenceBoundAuthorityError::WelfareConstraintMismatch {
            expected: context.constraint(),
            actual: request.welfare_constraint,
        });
    }
    let derived_nonce = evidence_bound_authority_nonce(base_nonce, context)?;
    build_context_bound_authority_statement(
        &derived_nonce,
        authority_manifest,
        trust_snapshot,
        authority_epoch,
        sequence,
        issued_at_unix_s,
        expires_at_unix_s,
        request,
    )
    .map_err(EvidenceBoundAuthorityError::AuthorityContext)
}

/// Hardened authorization path in which the existing operator authority must have been signed
/// using the evidence-bound nonce derived from the *current* profile and precaution policy.
///
/// A signature created for an older evidence profile, even one that produced the same coarse
/// protection level, fails the existing authority-context check because the authority ID differs.
#[allow(clippy::too_many_arguments)]
pub fn authorize_evidence_bound_intervention_once(
    base_nonce: &str,
    profile: &MoralPatientEvidenceProfile,
    precaution_policy: &PrecautionPolicy,
    subject_id: &str,
    consent_ledger: &SubjectConsentLedger,
    subject_registry: &SubjectIdentityRegistry,
    signed_authority: &SignedWelfareInterventionAuthority,
    authority_manifest: &WelfareAuthorityPolicyManifest,
    authority_trust_snapshot: &TrustSnapshot,
    authority_verifier: &dyn WelfareAuthoritySignatureVerifier,
    authority_tracker: &mut WelfareAuthorityTracker,
    request: &InterventionRequest,
    unix_s: u64,
) -> Result<EvidenceBoundInterventionPermit, EvidenceBoundAuthorityError> {
    let context = derive_welfare_evidence_context(profile, precaution_policy)?;
    if request.welfare_constraint != context.constraint() {
        return Err(EvidenceBoundAuthorityError::WelfareConstraintMismatch {
            expected: context.constraint(),
            actual: request.welfare_constraint,
        });
    }
    let derived_nonce = evidence_bound_authority_nonce(base_nonce, context)?;
    authorize_intervention_once_with_welfare_evidence(
        profile,
        precaution_policy,
        subject_id,
        consent_ledger,
        subject_registry,
        &derived_nonce,
        signed_authority,
        authority_manifest,
        authority_trust_snapshot,
        authority_verifier,
        authority_tracker,
        request,
        unix_s,
    )
    .map_err(EvidenceBoundAuthorityError::Authorization)
}

fn validate_base_nonce(base_nonce: &str) -> Result<(), EvidenceBoundAuthorityError> {
    if base_nonce.trim().is_empty()
        || base_nonce != base_nonce.trim()
        || base_nonce.len() > MAX_BASE_NONCE_BYTES
        || base_nonce.chars().any(char::is_control)
    {
        return Err(EvidenceBoundAuthorityError::InvalidBaseNonce);
    }
    Ok(())
}

fn disposition_code(disposition: ProtectionDisposition) -> u8 {
    match disposition {
        ProtectionDisposition::Baseline => 0,
        ProtectionDisposition::Precautionary => 1,
        ProtectionDisposition::EnhancedPrecaution => 2,
        ProtectionDisposition::IndependentReviewRequired => 3,
    }
}

fn constraint_code(constraint: WelfareConstraintLevel) -> u8 {
    match constraint {
        WelfareConstraintLevel::Baseline => 0,
        WelfareConstraintLevel::Precautionary => 1,
        WelfareConstraintLevel::EnhancedPrecaution => 2,
        WelfareConstraintLevel::IndependentReviewRequired => 3,
    }
}

fn hex_digest(digest: Sha256Digest) -> String {
    let mut out = String::with_capacity(64);
    for byte in digest.0 {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EvidenceBoundAuthorityError {
    #[error("base authority nonce must be canonical, non-empty, bounded text")]
    InvalidBaseNonce,
    #[error(transparent)]
    EvidenceContext(#[from] WelfareEvidenceContextError),
    #[error("request welfare constraint does not match current evidence: expected={expected:?}, actual={actual:?}")]
    WelfareConstraintMismatch {
        expected: WelfareConstraintLevel,
        actual: WelfareConstraintLevel,
    },
    #[error(transparent)]
    AuthorityContext(WelfareAuthorityContextError),
    #[error(transparent)]
    Authorization(EvidenceBoundAuthorizationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_psych_bench::moral_patient::{
        EvidenceConfidence, EvidenceFinding, EvidencePolarity, EvidenceSourceKind,
        MoralPatientDimension,
    };

    fn weak_finding(id: &str, lineage: &str) -> EvidenceFinding {
        EvidenceFinding {
            id: id.into(),
            dimension: MoralPatientDimension::MoralReasoning,
            source_kind: EvidenceSourceKind::Architecture,
            source_lineage: lineage.into(),
            polarity: EvidencePolarity::Supports,
            confidence: EvidenceConfidence::Weak,
            caveats: Vec::new(),
        }
    }

    #[test]
    fn evidence_bound_nonce_is_deterministic_and_canonical() {
        let profile = MoralPatientEvidenceProfile::default();
        let policy = PrecautionPolicy::default();
        let context = derive_welfare_evidence_context(&profile, &policy).unwrap();
        let left = evidence_bound_authority_nonce("issuance-1", context).unwrap();
        let right = evidence_bound_authority_nonce("issuance-1", context).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.len(), 64);
        assert!(left.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
    }

    #[test]
    fn evidence_change_changes_signed_authority_nonce_even_if_constraint_does_not() {
        let policy = PrecautionPolicy::default();
        let baseline = MoralPatientEvidenceProfile::default();
        let changed = MoralPatientEvidenceProfile {
            findings: vec![weak_finding("weak-1", "lineage-1")],
        };
        let before = derive_welfare_evidence_context(&baseline, &policy).unwrap();
        let after = derive_welfare_evidence_context(&changed, &policy).unwrap();
        assert_eq!(before.constraint(), after.constraint());
        assert_ne!(
            evidence_bound_authority_nonce("issuance-1", before).unwrap(),
            evidence_bound_authority_nonce("issuance-1", after).unwrap()
        );
    }

    #[test]
    fn precaution_policy_change_changes_signed_authority_nonce() {
        let profile = MoralPatientEvidenceProfile::default();
        let first = derive_welfare_evidence_context(&profile, &PrecautionPolicy::default()).unwrap();
        let second_policy = PrecautionPolicy {
            enhanced_lineages: 1,
            independent_review_lineages: 2,
        };
        let second = derive_welfare_evidence_context(&profile, &second_policy).unwrap();
        assert_eq!(first.constraint(), second.constraint());
        assert_ne!(
            evidence_bound_authority_nonce("issuance-1", first).unwrap(),
            evidence_bound_authority_nonce("issuance-1", second).unwrap()
        );
    }

    #[test]
    fn base_nonce_is_part_of_authority_identity() {
        let context = derive_welfare_evidence_context(
            &MoralPatientEvidenceProfile::default(),
            &PrecautionPolicy::default(),
        )
        .unwrap();
        assert_ne!(
            evidence_bound_authority_nonce("issuance-a", context).unwrap(),
            evidence_bound_authority_nonce("issuance-b", context).unwrap()
        );
    }

    #[test]
    fn malformed_base_nonce_fails_closed() {
        let context = derive_welfare_evidence_context(
            &MoralPatientEvidenceProfile::default(),
            &PrecautionPolicy::default(),
        )
        .unwrap();
        assert_eq!(
            evidence_bound_authority_nonce("  ", context),
            Err(EvidenceBoundAuthorityError::InvalidBaseNonce)
        );
    }
}
