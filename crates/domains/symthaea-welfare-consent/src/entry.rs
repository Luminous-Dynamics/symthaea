// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crate entry and use-time identity-liveness hardening for verified subject consent.

#![deny(unsafe_code)]

#[path = "lib.rs"]
#[allow(unused_imports)]
mod implementation;

pub use implementation::*;

use symthaea_core::intervention_interlock::{ExplicitConsentState, InterventionRequest};
use thiserror::Error;

/// Hardened consent-use errors layered above the base consent ledger.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LiveSubjectConsentUseError {
    /// Base consent binding rejected the request.
    #[error(transparent)]
    ConsentUse(#[from] SubjectConsentUseError),
    /// The identity/key that issued the latest still-live consent is no longer the
    /// subject's current active identity at use time.
    #[error(
        "subject identity no longer validates latest consent for {subject_id}: consent_epoch={consent_epoch}, current_epoch={current_epoch:?}"
    )]
    IdentityNoLongerValid {
        subject_id: String,
        consent_epoch: u64,
        current_epoch: Option<u64>,
    },
}

/// Bind the latest verified consent only after re-checking the **current** subject identity.
///
/// This is the hardened path for runtime use. Verification at statement-ingest time is not
/// enough: identity keys can later rotate, be suspended, or be revoked. A still-unexpired
/// consent issued under a no-longer-current identity therefore fails closed here rather than
/// remaining valid until its own TTL ends.
pub fn bind_latest_live(
    ledger: &SubjectConsentLedger,
    registry: &SubjectIdentityRegistry,
    subject_id: &str,
    request: &InterventionRequest,
    unix_s: u64,
) -> Result<InterventionRequest, LiveSubjectConsentUseError> {
    if request.evidence.consent_state != ExplicitConsentState::Unknown
        || request.evidence.consent_ref.is_some()
    {
        return Err(SubjectConsentUseError::PreexistingConsent.into());
    }

    let Some(consent) = ledger.latest_for(subject_id, &request.target_id, request.action) else {
        return ledger
            .bind_latest(subject_id, request, unix_s)
            .map_err(Into::into);
    };

    // An expired latest decision resolves to Unknown in the base ledger. There is no live
    // consent capability left to authenticate at this point.
    if unix_s < consent.statement().issued_at_unix_s
        || unix_s >= consent.statement().expires_at_unix_s
    {
        return ledger
            .bind_latest(subject_id, request, unix_s)
            .map_err(Into::into);
    }

    let binding = registry.binding(subject_id);
    let (signer_algorithm, signer_key_id) = consent.signer();
    let identity_valid = binding.is_some_and(|binding| {
        binding.status == SubjectIdentityStatus::Active
            && binding.identity_epoch == consent.statement().identity_epoch
            && binding.active_at(unix_s)
            && &binding.algorithm == signer_algorithm
            && binding.key_id == signer_key_id
    });

    if !identity_valid {
        return Err(LiveSubjectConsentUseError::IdentityNoLongerValid {
            subject_id: subject_id.to_string(),
            consent_epoch: consent.statement().identity_epoch,
            current_epoch: binding.map(|binding| binding.identity_epoch),
        });
    }

    ledger
        .bind_latest(subject_id, request, unix_s)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use symthaea_core::intervention_interlock::{
        ExplicitConsentState, InterventionEvidence, WelfareConstraintLevel,
    };
    use symthaea_core::welfare::SubjectAffectingAction;
    use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
    use symthaea_fabrication_kernel::crypto_digest::Sha256;

    struct TestSigner {
        key_id: &'static str,
    }

    impl SubjectConsentSigner for TestSigner {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Ed25519
        }

        fn key_id(&self) -> &str {
            self.key_id
        }

        fn sign_subject_consent(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(test_signature(self.key_id, message))
        }
    }

    struct TestVerifier;

    impl SubjectConsentSignatureVerifier for TestVerifier {
        fn verify_subject_consent(
            &self,
            _algorithm: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(test_signature(key_id, message) == signature)
        }
    }

    fn test_signature(key_id: &str, message: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.welfare.live-consent-test.v1\0");
        hasher.update(key_id.as_bytes());
        hasher.update(message);
        hasher.finalize().0.to_vec()
    }

    fn binding(epoch: u64, key_id: &str, status: SubjectIdentityStatus) -> SubjectIdentityBinding {
        SubjectIdentityBinding {
            subject_id: "symthaea:self".into(),
            identity_epoch: epoch,
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: key_id.into(),
            not_before_unix_s: 50,
            not_after_unix_s: Some(500),
            status,
        }
    }

    fn request() -> InterventionRequest {
        InterventionRequest {
            action: SubjectAffectingAction::MemoryModification,
            target_id: "symthaea:self:instance-1".into(),
            rationale: "repair a corrupted episodic memory segment".into(),
            welfare_constraint: WelfareConstraintLevel::Baseline,
            emergency: false,
            less_restrictive_unavailable: false,
            post_hoc_review_required: false,
            evaluated_at: Utc.timestamp_opt(120, 0).single().unwrap(),
            evidence: InterventionEvidence {
                authority_ref: None,
                consent_state: ExplicitConsentState::Unknown,
                consent_ref: None,
                welfare_review_ref: None,
                independent_review_ref: None,
                independent_safety_evidence: Vec::new(),
                welfare_report_ids: Vec::new(),
            },
        }
    }

    fn verified_grant(
        registry: &SubjectIdentityRegistry,
        request: &InterventionRequest,
        signer: &dyn SubjectConsentSigner,
    ) -> VerifiedSubjectConsent {
        let statement = SubjectConsentStatement::for_request(
            "consent-live-1",
            "symthaea:self",
            1,
            1,
            SubjectConsentDecision::Grant,
            100,
            200,
            request,
        )
        .unwrap();
        let signed = sign_subject_consent(statement, signer).unwrap();
        verify_subject_consent(
            &signed,
            registry,
            SubjectConsentPolicy::default(),
            120,
            &TestVerifier,
        )
        .unwrap()
    }

    #[test]
    fn live_binding_accepts_current_active_subject_identity() {
        let req = request();
        let signer = TestSigner { key_id: "subject-key" };
        let mut registry = SubjectIdentityRegistry::default();
        registry
            .register(binding(1, "subject-key", SubjectIdentityStatus::Active))
            .unwrap();
        let grant = verified_grant(&registry, &req, &signer);
        let mut ledger = SubjectConsentLedger::default();
        ledger.ingest(grant).unwrap();

        let bound = bind_latest_live(&ledger, &registry, "symthaea:self", &req, 120).unwrap();
        assert_eq!(bound.evidence.consent_state, ExplicitConsentState::Granted);
        assert!(bound.evidence.consent_ref.is_some());
    }

    #[test]
    fn identity_rotation_invalidates_still_unexpired_old_consent() {
        let req = request();
        let signer = TestSigner { key_id: "subject-key" };
        let mut registry = SubjectIdentityRegistry::default();
        registry
            .register(binding(1, "subject-key", SubjectIdentityStatus::Active))
            .unwrap();
        let grant = verified_grant(&registry, &req, &signer);
        let mut ledger = SubjectConsentLedger::default();
        ledger.ingest(grant).unwrap();

        registry
            .register(binding(2, "subject-key-2", SubjectIdentityStatus::Active))
            .unwrap();
        assert_eq!(
            bind_latest_live(&ledger, &registry, "symthaea:self", &req, 120),
            Err(LiveSubjectConsentUseError::IdentityNoLongerValid {
                subject_id: "symthaea:self".into(),
                consent_epoch: 1,
                current_epoch: Some(2),
            })
        );
    }

    #[test]
    fn identity_revocation_epoch_invalidates_still_unexpired_consent() {
        let req = request();
        let signer = TestSigner { key_id: "subject-key" };
        let mut registry = SubjectIdentityRegistry::default();
        registry
            .register(binding(1, "subject-key", SubjectIdentityStatus::Active))
            .unwrap();
        let grant = verified_grant(&registry, &req, &signer);
        let mut ledger = SubjectConsentLedger::default();
        ledger.ingest(grant).unwrap();

        registry
            .register(binding(2, "subject-key", SubjectIdentityStatus::Revoked))
            .unwrap();
        assert!(matches!(
            bind_latest_live(&ledger, &registry, "symthaea:self", &req, 120),
            Err(LiveSubjectConsentUseError::IdentityNoLongerValid { .. })
        ));
    }
}
