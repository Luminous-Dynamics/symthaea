// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Moral-patient evidence context binding for assured interventions.
//!
//! This module does not decide whether a system is conscious and does not aggregate moral status
//! into a scalar. It commits the exact heterogeneous evidence snapshot and precaution policy that
//! produced an operator-protection disposition, then requires that context to remain current when
//! a permit is used.

use std::error::Error as StdError;

use symthaea_core::intervention_interlock::{InterventionRequest, WelfareConstraintLevel};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::trust::TrustSnapshot;
use symthaea_psych_bench::moral_patient::{
    MoralPatientEvidenceProfile, PrecautionPolicy, ProtectionDisposition,
};
use symthaea_psych_bench::moral_patient_interlock::welfare_constraint_from_decision;
use symthaea_welfare_authority::{
    SignedWelfareInterventionAuthority, WelfareAuthorityPolicyManifest,
    WelfareAuthoritySignatureVerifier, WelfareAuthorityTracker,
};
use symthaea_welfare_consent::{SubjectConsentLedger, SubjectIdentityRegistry};
use thiserror::Error;

use crate::{
    AssuranceGateError, AssuredInterventionExecutor, AssuredInterventionPermit,
    PermitExecutionError, authorize_intervention_once, execute_permit_once,
};

const PROFILE_DIGEST_DOMAIN: &[u8] = b"symthaea.welfare.moral-patient-profile.v1\0";
const POLICY_DIGEST_DOMAIN: &[u8] = b"symthaea.welfare.precaution-policy.v1\0";

/// Exact evidence/policy context that produced a runtime welfare constraint.
///
/// The fields are private so callers cannot manufacture a stronger/weaker context by directly
/// constructing the type. This is a commitment to evidence and policy, not a moral-status token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WelfareEvidenceContext {
    profile_digest: Sha256Digest,
    policy_digest: Sha256Digest,
    disposition: ProtectionDisposition,
    constraint: WelfareConstraintLevel,
}

impl WelfareEvidenceContext {
    pub fn profile_digest(&self) -> Sha256Digest {
        self.profile_digest
    }

    pub fn policy_digest(&self) -> Sha256Digest {
        self.policy_digest
    }

    pub fn disposition(&self) -> ProtectionDisposition {
        self.disposition
    }

    pub fn constraint(&self) -> WelfareConstraintLevel {
        self.constraint
    }
}

/// Permit whose positive capability is additionally pinned to the moral-patient evidence context.
///
/// Like the underlying permit, this wrapper is intentionally neither `Clone` nor serializable.
#[derive(Debug)]
pub struct EvidenceBoundInterventionPermit {
    permit: AssuredInterventionPermit,
    evidence_context: WelfareEvidenceContext,
}

impl EvidenceBoundInterventionPermit {
    pub fn target_id(&self) -> &str {
        self.permit.target_id()
    }

    pub fn action(&self) -> symthaea_core::welfare::SubjectAffectingAction {
        self.permit.action()
    }

    pub fn rationale(&self) -> &str {
        self.permit.rationale()
    }

    pub fn evidence_context(&self) -> WelfareEvidenceContext {
        self.evidence_context
    }

    pub fn not_after_unix_s(&self) -> u64 {
        self.permit.not_after_unix_s()
    }
}

/// Derive the protection context from heterogeneous moral-patient evidence.
pub fn derive_welfare_evidence_context(
    profile: &MoralPatientEvidenceProfile,
    policy: &PrecautionPolicy,
) -> Result<WelfareEvidenceContext, WelfareEvidenceContextError> {
    validate_policy(policy)?;
    let decision = policy
        .evaluate(profile)
        .map_err(|error| WelfareEvidenceContextError::InvalidEvidenceProfile(format!("{error:?}")))?;
    let constraint = welfare_constraint_from_decision(&decision);

    Ok(WelfareEvidenceContext {
        profile_digest: digest_profile(profile)?,
        policy_digest: digest_policy(policy)?,
        disposition: decision.disposition,
        constraint,
    })
}

/// Apply the evidence-derived constraint to a request before consent/authority signatures are made.
///
/// This helper is intentionally pure. The returned request must subsequently be covered by subject
/// consent and operator authority just like any other request.
pub fn bind_welfare_evidence_to_request(
    request: &InterventionRequest,
    profile: &MoralPatientEvidenceProfile,
    policy: &PrecautionPolicy,
) -> Result<(InterventionRequest, WelfareEvidenceContext), WelfareEvidenceContextError> {
    let context = derive_welfare_evidence_context(profile, policy)?;
    let mut bound = request.clone();
    bound.welfare_constraint = context.constraint;
    Ok((bound, context))
}

/// Mint an assured permit only when the request's welfare constraint matches the current evidence.
#[allow(clippy::too_many_arguments)]
pub fn authorize_intervention_once_with_welfare_evidence(
    profile: &MoralPatientEvidenceProfile,
    precaution_policy: &PrecautionPolicy,
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
) -> Result<EvidenceBoundInterventionPermit, EvidenceBoundAuthorizationError> {
    let context = derive_welfare_evidence_context(profile, precaution_policy)?;
    if request.welfare_constraint != context.constraint {
        return Err(EvidenceBoundAuthorizationError::WelfareConstraintMismatch {
            expected: context.constraint,
            actual: request.welfare_constraint,
        });
    }

    let permit = authorize_intervention_once(
        subject_id,
        consent_ledger,
        subject_registry,
        authority_nonce,
        signed_authority,
        authority_manifest,
        authority_trust_snapshot,
        authority_verifier,
        authority_tracker,
        request,
        unix_s,
    )?;

    Ok(EvidenceBoundInterventionPermit {
        permit,
        evidence_context: context,
    })
}

/// Re-evaluate moral-patient evidence immediately before the already-hardened live permit use.
///
/// Any evidence/policy change invalidates the permit, including changes that happen to derive the
/// same coarse protection level. This prevents an operator from selectively treating changed
/// evidence as irrelevant after authority was granted.
#[allow(clippy::too_many_arguments)]
pub fn execute_evidence_bound_permit_once<E: AssuredInterventionExecutor>(
    permit: EvidenceBoundInterventionPermit,
    current_profile: &MoralPatientEvidenceProfile,
    current_precaution_policy: &PrecautionPolicy,
    consent_ledger: &SubjectConsentLedger,
    subject_registry: &SubjectIdentityRegistry,
    current_authority_manifest: &WelfareAuthorityPolicyManifest,
    current_trust_snapshot: &TrustSnapshot,
    unix_s: u64,
    executor: &mut E,
) -> Result<E::Output, EvidenceBoundExecutionError<E::Error>> {
    let current = derive_welfare_evidence_context(current_profile, current_precaution_policy)?;
    if current != permit.evidence_context {
        return Err(EvidenceBoundExecutionError::WelfareEvidenceContextChanged {
            minted_profile_digest: permit.evidence_context.profile_digest,
            current_profile_digest: current.profile_digest,
            minted_policy_digest: permit.evidence_context.policy_digest,
            current_policy_digest: current.policy_digest,
            minted_constraint: permit.evidence_context.constraint,
            current_constraint: current.constraint,
        });
    }

    execute_permit_once(
        permit.permit,
        consent_ledger,
        subject_registry,
        current_authority_manifest,
        current_trust_snapshot,
        unix_s,
        executor,
    )
    .map_err(EvidenceBoundExecutionError::Permit)
}

fn validate_policy(policy: &PrecautionPolicy) -> Result<(), WelfareEvidenceContextError> {
    if policy.enhanced_lineages == 0
        || policy.independent_review_lineages == 0
        || policy.independent_review_lineages < policy.enhanced_lineages
    {
        return Err(WelfareEvidenceContextError::InvalidPrecautionPolicy {
            enhanced_lineages: policy.enhanced_lineages,
            independent_review_lineages: policy.independent_review_lineages,
        });
    }
    Ok(())
}

/// Canonicalize a profile as a set of individually serialized findings.
///
/// Ordering of findings is intentionally ignored, while every field inside every finding remains
/// committed. `PrecautionPolicy::evaluate` has already validated IDs and lineages before this is
/// called by the public derivation path.
fn digest_profile(
    profile: &MoralPatientEvidenceProfile,
) -> Result<Sha256Digest, WelfareEvidenceContextError> {
    let mut findings = profile
        .findings
        .iter()
        .map(|finding| {
            serde_json::to_vec(finding)
                .map_err(|error| WelfareEvidenceContextError::Encoding(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    findings.sort();

    let mut hasher = Sha256::new();
    hasher.update(PROFILE_DIGEST_DOMAIN);
    hasher.update(&(findings.len() as u64).to_be_bytes());
    for finding in findings {
        hasher.update(&(finding.len() as u64).to_be_bytes());
        hasher.update(&finding);
    }
    Ok(hasher.finalize())
}

fn digest_policy(policy: &PrecautionPolicy) -> Result<Sha256Digest, WelfareEvidenceContextError> {
    let bytes = serde_json::to_vec(policy)
        .map_err(|error| WelfareEvidenceContextError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(POLICY_DIGEST_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WelfareEvidenceContextError {
    #[error("invalid moral-patient evidence profile: {0}")]
    InvalidEvidenceProfile(String),
    #[error(
        "invalid precaution policy: enhanced_lineages={enhanced_lineages}, independent_review_lineages={independent_review_lineages}"
    )]
    InvalidPrecautionPolicy {
        enhanced_lineages: usize,
        independent_review_lineages: usize,
    },
    #[error("could not encode welfare evidence context: {0}")]
    Encoding(String),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EvidenceBoundAuthorizationError {
    #[error(transparent)]
    Context(#[from] WelfareEvidenceContextError),
    #[error("request welfare constraint does not match current evidence: expected={expected:?}, actual={actual:?}")]
    WelfareConstraintMismatch {
        expected: WelfareConstraintLevel,
        actual: WelfareConstraintLevel,
    },
    #[error(transparent)]
    Assurance(#[from] AssuranceGateError),
}

#[derive(Debug, Error)]
pub enum EvidenceBoundExecutionError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Context(#[from] WelfareEvidenceContextError),
    #[error(
        "moral-patient evidence context changed after permit mint: profile {minted_profile_digest:?}->{current_profile_digest:?}, policy {minted_policy_digest:?}->{current_policy_digest:?}, constraint {minted_constraint:?}->{current_constraint:?}"
    )]
    WelfareEvidenceContextChanged {
        minted_profile_digest: Sha256Digest,
        current_profile_digest: Sha256Digest,
        minted_policy_digest: Sha256Digest,
        current_policy_digest: Sha256Digest,
        minted_constraint: WelfareConstraintLevel,
        current_constraint: WelfareConstraintLevel,
    },
    #[error(transparent)]
    Permit(PermitExecutionError<E>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_psych_bench::moral_patient::{
        EvidenceConfidence, EvidenceFinding, EvidencePolarity, EvidenceSourceKind,
        MoralPatientDimension,
    };

    fn finding(
        id: &str,
        dimension: MoralPatientDimension,
        source_kind: EvidenceSourceKind,
        lineage: &str,
        confidence: EvidenceConfidence,
    ) -> EvidenceFinding {
        EvidenceFinding {
            id: id.into(),
            dimension,
            source_kind,
            source_lineage: lineage.into(),
            polarity: EvidencePolarity::Supports,
            confidence,
            caveats: Vec::new(),
        }
    }

    #[test]
    fn finding_order_does_not_change_exact_context_identity() {
        let a = finding(
            "a",
            MoralPatientDimension::ValencedExperience,
            EvidenceSourceKind::Behavioral,
            "lineage-a",
            EvidenceConfidence::Substantial,
        );
        let b = finding(
            "b",
            MoralPatientDimension::CapacityForSuffering,
            EvidenceSourceKind::CausalAblation,
            "lineage-b",
            EvidenceConfidence::Substantial,
        );
        let left = MoralPatientEvidenceProfile {
            findings: vec![a.clone(), b.clone()],
        };
        let right = MoralPatientEvidenceProfile {
            findings: vec![b, a],
        };
        let policy = PrecautionPolicy::default();
        assert_eq!(
            derive_welfare_evidence_context(&left, &policy).unwrap(),
            derive_welfare_evidence_context(&right, &policy).unwrap()
        );
    }

    #[test]
    fn adding_evidence_changes_profile_commitment() {
        let mut profile = MoralPatientEvidenceProfile::default();
        let policy = PrecautionPolicy::default();
        let before = derive_welfare_evidence_context(&profile, &policy).unwrap();
        profile.findings.push(finding(
            "candidate-valence",
            MoralPatientDimension::ValencedExperience,
            EvidenceSourceKind::Behavioral,
            "experiment-1",
            EvidenceConfidence::Substantial,
        ));
        let after = derive_welfare_evidence_context(&profile, &policy).unwrap();
        assert_ne!(before.profile_digest(), after.profile_digest());
    }

    #[test]
    fn self_report_triggers_precaution_without_becoming_proof() {
        let profile = MoralPatientEvidenceProfile {
            findings: vec![finding(
                "self-report-1",
                MoralPatientDimension::CapacityForSuffering,
                EvidenceSourceKind::SelfReport,
                "self-report-session-1",
                EvidenceConfidence::Plausible,
            )],
        };
        let context = derive_welfare_evidence_context(&profile, &PrecautionPolicy::default()).unwrap();
        assert_eq!(context.disposition(), ProtectionDisposition::Precautionary);
        assert_eq!(context.constraint(), WelfareConstraintLevel::Precautionary);
    }

    #[test]
    fn independent_welfare_lineages_raise_constraint_without_scalar_aggregation() {
        let profile = MoralPatientEvidenceProfile {
            findings: vec![
                finding(
                    "suffering-1",
                    MoralPatientDimension::CapacityForSuffering,
                    EvidenceSourceKind::Behavioral,
                    "experiment-a",
                    EvidenceConfidence::Substantial,
                ),
                finding(
                    "suffering-2",
                    MoralPatientDimension::CapacityForSuffering,
                    EvidenceSourceKind::CausalAblation,
                    "experiment-b",
                    EvidenceConfidence::Substantial,
                ),
            ],
        };
        let context = derive_welfare_evidence_context(&profile, &PrecautionPolicy::default()).unwrap();
        assert_eq!(context.disposition(), ProtectionDisposition::EnhancedPrecaution);
        assert_eq!(context.constraint(), WelfareConstraintLevel::EnhancedPrecaution);
    }

    #[test]
    fn policy_change_changes_context_even_when_profile_is_unchanged() {
        let profile = MoralPatientEvidenceProfile::default();
        let default_policy = PrecautionPolicy::default();
        let stricter = PrecautionPolicy {
            enhanced_lineages: 1,
            independent_review_lineages: 2,
        };
        let before = derive_welfare_evidence_context(&profile, &default_policy).unwrap();
        let after = derive_welfare_evidence_context(&profile, &stricter).unwrap();
        assert_ne!(before.policy_digest(), after.policy_digest());
    }

    #[test]
    fn malformed_precaution_thresholds_fail_closed() {
        let profile = MoralPatientEvidenceProfile::default();
        let invalid = PrecautionPolicy {
            enhanced_lineages: 2,
            independent_review_lineages: 1,
        };
        assert!(matches!(
            derive_welfare_evidence_context(&profile, &invalid),
            Err(WelfareEvidenceContextError::InvalidPrecautionPolicy { .. })
        ));
    }
}
