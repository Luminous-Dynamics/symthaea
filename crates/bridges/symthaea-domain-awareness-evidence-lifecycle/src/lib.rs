// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scope independently verified domain-awareness candidate evidence to an exact
//! reviewed safety contract and applicability window.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_domain_awareness_evidence::{
    CandidateEvidence, EvidenceBindingError, IndependentVerification,
};
use symthaea_evidence_lifecycle::ScopedSafetyEvidenceReceipt;
use symthaea_formal_safety::SafetyCase;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceApplicability {
    pub valid_from_ms: u64,
    pub valid_until_ms: u64,
    /// Durable deployment/configuration references justifying applicability.
    pub applicability_refs: Vec<String>,
}

impl EvidenceApplicability {
    pub fn validate(&self) -> bool {
        self.valid_from_ms <= self.valid_until_ms
            && !self.applicability_refs.is_empty()
            && self
                .applicability_refs
                .iter()
                .all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopedEvidenceError {
    InvalidCandidate,
    InvalidVerification,
    InvalidApplicability,
    CandidateObligationMissingFromSafetyCase,
    ApplicabilityPredatesCandidateEvidence,
    ApplicabilityPredatesVerification,
    VerificationAfterValidityWindow,
    Verification(EvidenceBindingError),
}

impl From<EvidenceBindingError> for ScopedEvidenceError {
    fn from(value: EvidenceBindingError) -> Self {
        Self::Verification(value)
    }
}

/// Independently verify a candidate and bind the resulting receipt to the exact
/// safety contract/configuration for which it is being reviewed.
///
/// The validity window must begin no earlier than both the candidate artifact's
/// observation time and the independent verification time. This avoids claiming
/// deployment readiness for a period before the evidence or its verification
/// existed.
pub fn verify_and_scope_candidate(
    candidate: &CandidateEvidence,
    verification: &IndependentVerification,
    safety_case: &SafetyCase,
    applicability: &EvidenceApplicability,
) -> Result<ScopedSafetyEvidenceReceipt, ScopedEvidenceError> {
    if !candidate.validate() {
        return Err(ScopedEvidenceError::InvalidCandidate);
    }
    if !verification.validate() {
        return Err(ScopedEvidenceError::InvalidVerification);
    }
    if !applicability.validate() {
        return Err(ScopedEvidenceError::InvalidApplicability);
    }

    let obligation_key = candidate.obligation_key();
    let obligation_present = safety_case.obligations.iter().any(|obligation| {
        obligation.stable_key() == obligation_key
            && obligation.expected_evidence == candidate.evidence_kind()
    });
    if !obligation_present {
        return Err(ScopedEvidenceError::CandidateObligationMissingFromSafetyCase);
    }
    if applicability.valid_from_ms < candidate.observed_at_ms {
        return Err(ScopedEvidenceError::ApplicabilityPredatesCandidateEvidence);
    }
    if applicability.valid_from_ms < verification.verified_at_ms {
        return Err(ScopedEvidenceError::ApplicabilityPredatesVerification);
    }
    if verification.verified_at_ms > applicability.valid_until_ms {
        return Err(ScopedEvidenceError::VerificationAfterValidityWindow);
    }

    let receipt = candidate.verify(verification)?;
    Ok(ScopedSafetyEvidenceReceipt {
        receipt,
        contract_digest: safety_case.contract_digest(),
        valid_from_ms: applicability.valid_from_ms,
        valid_until_ms: applicability.valid_until_ms,
        applicability_refs: applicability.applicability_refs.clone(),
    })
}

/// This bridge only prepares evidence for lifecycle assessment.
pub const fn grants_physical_authority() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_domain_awareness_evidence::ArtifactBinding;
    use symthaea_formal_safety::{
        DomainAwarenessObligation, SafetyCaseTemplate, StrictSafetyCaseStatus,
        assess_strict_safety_case,
    };

    fn candidate() -> CandidateEvidence {
        CandidateEvidence {
            candidate_id: "DA-017:crucible:100".into(),
            obligation: DomainAwarenessObligation::NegativeOnlyStressEvidenceRequired,
            evidence_ref: "artifact:crucible".into(),
            evidence_digest: "blake3:abc".into(),
            observed_at_ms: 100,
            rationale: "negative-only deployment stress evidence".into(),
        }
    }

    fn verification(at: u64) -> IndependentVerification {
        IndependentVerification {
            receipt_id: "receipt-1".into(),
            verifier_ref: "verifier:independent-safety".into(),
            verified_at_ms: at,
        }
    }

    fn applicability(from: u64, until: u64) -> EvidenceApplicability {
        EvidenceApplicability {
            valid_from_ms: from,
            valid_until_ms: until,
            applicability_refs: vec!["deployment-config:harbor-v3".into()],
        }
    }

    #[test]
    fn candidate_can_be_scoped_to_exact_domain_awareness_contract() {
        let case = SafetyCase::from_template("harbor-a", SafetyCaseTemplate::DomainAwareness);
        let scoped = verify_and_scope_candidate(
            &candidate(),
            &verification(200),
            &case,
            &applicability(200, 10_000),
        )
        .unwrap();
        assert_eq!(scoped.contract_digest, case.contract_digest());
        assert_eq!(scoped.valid_from_ms, 200);
        assert_eq!(scoped.receipt.verifier_ref, "verifier:independent-safety");
        assert!(!grants_physical_authority());
    }

    #[test]
    fn unrelated_safety_case_cannot_accept_candidate_by_digest_string_alone() {
        let case = SafetyCase::from_template("bridge", SafetyCaseTemplate::CivilStructure);
        assert_eq!(
            verify_and_scope_candidate(
                &candidate(),
                &verification(200),
                &case,
                &applicability(200, 10_000),
            ),
            Err(ScopedEvidenceError::CandidateObligationMissingFromSafetyCase)
        );
    }

    #[test]
    fn applicability_cannot_begin_before_candidate_or_verification() {
        let case = SafetyCase::from_template("harbor-a", SafetyCaseTemplate::DomainAwareness);
        assert_eq!(
            verify_and_scope_candidate(
                &candidate(),
                &verification(200),
                &case,
                &applicability(99, 10_000),
            ),
            Err(ScopedEvidenceError::ApplicabilityPredatesCandidateEvidence)
        );
        assert_eq!(
            verify_and_scope_candidate(
                &candidate(),
                &verification(200),
                &case,
                &applicability(150, 10_000),
            ),
            Err(ScopedEvidenceError::ApplicabilityPredatesVerification)
        );
    }

    #[test]
    fn already_expired_at_verification_is_rejected() {
        let case = SafetyCase::from_template("harbor-a", SafetyCaseTemplate::DomainAwareness);
        assert_eq!(
            verify_and_scope_candidate(
                &candidate(),
                &verification(500),
                &case,
                &applicability(500, 499),
            ),
            Err(ScopedEvidenceError::InvalidApplicability)
        );
    }

    #[test]
    fn scoped_receipt_still_does_not_discharge_open_safety_case() {
        let case = SafetyCase::from_template("harbor-a", SafetyCaseTemplate::DomainAwareness);
        let scoped = verify_and_scope_candidate(
            &candidate(),
            &verification(200),
            &case,
            &applicability(200, 10_000),
        )
        .unwrap();
        let strict = assess_strict_safety_case(&case, &[scoped.receipt]);
        assert_eq!(strict.status, StrictSafetyCaseStatus::Blocked);
    }

    #[test]
    fn artifact_binding_type_is_not_itself_scope_authority() {
        let binding = ArtifactBinding {
            evidence_ref: "artifact:x".into(),
            evidence_digest: "blake3:x".into(),
        };
        assert!(binding.validate());
        assert!(!grants_physical_authority());
    }
}
