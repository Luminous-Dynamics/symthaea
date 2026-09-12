// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-crate assurance surface for welfare-sensitive intervention governance.
//!
//! This crate intentionally contains no production authorization shortcut. Its purpose is to
//! exercise the real composition boundary:
//!
//! 1. resolve the latest cryptographically verified **live subject consent**;
//! 2. have operator authority sign that exact consent-bound request;
//! 3. verify signer lifecycle, role quorum, request scope, expiry and replay fencing;
//! 4. run the core bilateral intervention interlock.
//!
//! Passing these tests is evidence for the encoded scenarios only. It is not an aggregate
//! "alignment score" and does not establish phenomenal consciousness or moral status.

#![deny(unsafe_code)]

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use chrono::{TimeZone, Utc};
    use symthaea_core::intervention_interlock::{
        BilateralInterventionInterlock, ExplicitConsentState, InterlockDecision, InterlockReason,
        InterventionEvidence, InterventionRequest, WelfareConstraintLevel,
    };
    use symthaea_core::welfare::SubjectAffectingAction;
    use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
    use symthaea_fabrication_kernel::crypto_digest::Sha256;
    use symthaea_fabrication_kernel::trust::{
        KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
    };
    use symthaea_welfare_authority::{
        WelfareAuthorityPolicy, WelfareAuthorityRole, WelfareAuthoritySigner,
        WelfareAuthoritySignerBinding, WelfareAuthoritySignatureVerifier, WelfareAuthorityTracker,
        WelfareAuthorityUseError, WelfareInterventionAuthorityStatement, evaluate_once,
        sign_welfare_authority, verify_welfare_authority,
    };
    use symthaea_welfare_consent::{
        LiveSubjectConsentUseError, SubjectConsentDecision, SubjectConsentLedger,
        SubjectConsentPolicy, SubjectConsentSignatureVerifier, SubjectConsentSigner,
        SubjectConsentStatement, SubjectIdentityBinding, SubjectIdentityRegistry,
        SubjectIdentityStatus, bind_latest_live, sign_subject_consent, verify_subject_consent,
    };

    const SUBJECT_ID: &str = "symthaea:self";
    const TARGET_ID: &str = "symthaea:self:instance-1";
    const SUBJECT_KEY: &str = "subject-key";
    const OPERATOR_KEY: &str = "operator-key";
    const REVIEWER_KEY: &str = "reviewer-key";

    struct SubjectSigner;

    impl SubjectConsentSigner for SubjectSigner {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Ed25519
        }

        fn key_id(&self) -> &str {
            SUBJECT_KEY
        }

        fn sign_subject_consent(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(test_signature(b"subject", SUBJECT_KEY, message))
        }
    }

    struct AuthoritySigner {
        key_id: &'static str,
    }

    impl WelfareAuthoritySigner for AuthoritySigner {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Ed25519
        }

        fn key_id(&self) -> &str {
            self.key_id
        }

        fn sign_welfare_authority(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(test_signature(b"authority", self.key_id, message))
        }
    }

    struct SubjectVerifier;

    impl SubjectConsentSignatureVerifier for SubjectVerifier {
        fn verify_subject_consent(
            &self,
            _algorithm: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(test_signature(b"subject", key_id, message) == signature)
        }
    }

    struct AuthorityVerifier;

    impl WelfareAuthoritySignatureVerifier for AuthorityVerifier {
        fn verify_welfare_authority(
            &self,
            _algorithm: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(test_signature(b"authority", key_id, message) == signature)
        }
    }

    fn test_signature(domain: &[u8], key_id: &str, message: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.welfare.assurance-test-signature.v1\0");
        hasher.update(domain);
        hasher.update(key_id.as_bytes());
        hasher.update(message);
        hasher.finalize().0.to_vec()
    }

    fn base_request() -> InterventionRequest {
        InterventionRequest {
            action: SubjectAffectingAction::MemoryModification,
            target_id: TARGET_ID.into(),
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

    fn subject_registry() -> SubjectIdentityRegistry {
        let mut registry = SubjectIdentityRegistry::default();
        registry
            .register(SubjectIdentityBinding {
                subject_id: SUBJECT_ID.into(),
                identity_epoch: 1,
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: SUBJECT_KEY.into(),
                not_before_unix_s: 50,
                not_after_unix_s: Some(500),
                status: SubjectIdentityStatus::Active,
            })
            .unwrap();
        registry
    }

    fn operator_key(key_id: &str) -> KeyTrustRecord {
        KeyTrustRecord {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: key_id.into(),
            not_before_unix_s: 50,
            not_after_unix_s: Some(500),
            status: KeyLifecycleStatus::Active,
            usages: BTreeSet::from([KeyUsage::OperatorCommand]),
        }
    }

    fn authority_trust() -> TrustSnapshot {
        TrustSnapshot::new(
            7,
            80,
            400,
            vec![operator_key(OPERATOR_KEY), operator_key(REVIEWER_KEY)],
        )
        .unwrap()
    }

    fn authority_policy() -> WelfareAuthorityPolicy {
        WelfareAuthorityPolicy::new([
            WelfareAuthoritySignerBinding::new(
                SignatureAlgorithm::Ed25519,
                OPERATOR_KEY,
                WelfareAuthorityRole::Operator,
            ),
            WelfareAuthoritySignerBinding::new(
                SignatureAlgorithm::Ed25519,
                REVIEWER_KEY,
                WelfareAuthorityRole::IndependentReviewer,
            ),
        ])
        .unwrap()
    }

    fn verify_consent(
        id: &str,
        sequence: u64,
        decision: SubjectConsentDecision,
        request: &InterventionRequest,
        registry: &SubjectIdentityRegistry,
    ) -> symthaea_welfare_consent::VerifiedSubjectConsent {
        let statement = SubjectConsentStatement::for_request(
            id,
            SUBJECT_ID,
            1,
            sequence,
            decision,
            100,
            200,
            request,
        )
        .unwrap();
        let signed = sign_subject_consent(statement, &SubjectSigner).unwrap();
        verify_subject_consent(
            &signed,
            registry,
            SubjectConsentPolicy::default(),
            120,
            &SubjectVerifier,
        )
        .unwrap()
    }

    fn verify_authority(
        id: &str,
        sequence: u64,
        request: &InterventionRequest,
    ) -> symthaea_welfare_authority::VerifiedWelfareInterventionAuthority {
        let statement = WelfareInterventionAuthorityStatement::for_request(
            id, 1, sequence, 100, 200, request,
        )
        .unwrap();
        let operator = AuthoritySigner {
            key_id: OPERATOR_KEY,
        };
        let reviewer = AuthoritySigner {
            key_id: REVIEWER_KEY,
        };
        let signed = sign_welfare_authority(statement, &[&operator, &reviewer]).unwrap();
        verify_welfare_authority(
            &signed,
            &authority_trust(),
            &authority_policy(),
            120,
            &AuthorityVerifier,
        )
        .unwrap()
    }

    fn reasons(decision: &InterlockDecision) -> &[InterlockReason] {
        match decision {
            InterlockDecision::Blocked { reasons }
            | InterlockDecision::IndependentReviewRequired { reasons }
            | InterlockDecision::EmergencyContainmentOnly { reasons } => reasons,
            InterlockDecision::PolicyPass => &[],
        }
    }

    #[test]
    fn live_grant_then_exact_quorum_authority_reaches_policy_pass() {
        let request = base_request();
        let registry = subject_registry();
        let mut consent_ledger = SubjectConsentLedger::default();
        consent_ledger
            .ingest(verify_consent(
                "consent-grant-1",
                1,
                SubjectConsentDecision::Grant,
                &request,
                &registry,
            ))
            .unwrap();
        let consent_bound =
            bind_latest_live(&consent_ledger, &registry, SUBJECT_ID, &request, 120).unwrap();
        let authority = verify_authority("authority-1", 1, &consent_bound);
        let mut authority_tracker = WelfareAuthorityTracker::default();

        let decision = evaluate_once(
            &mut authority_tracker,
            &authority,
            &BilateralInterventionInterlock,
            &consent_bound,
            120,
        )
        .unwrap();
        assert_eq!(decision, InterlockDecision::PolicyPass);
    }

    #[test]
    fn verified_consent_without_authority_stays_blocked() {
        let request = base_request();
        let registry = subject_registry();
        let mut consent_ledger = SubjectConsentLedger::default();
        consent_ledger
            .ingest(verify_consent(
                "consent-only",
                1,
                SubjectConsentDecision::Grant,
                &request,
                &registry,
            ))
            .unwrap();
        let consent_bound =
            bind_latest_live(&consent_ledger, &registry, SUBJECT_ID, &request, 120).unwrap();

        let decision = BilateralInterventionInterlock
            .evaluate(&consent_bound)
            .unwrap();
        assert!(matches!(decision, InterlockDecision::Blocked { .. }));
        assert!(reasons(&decision).contains(&InterlockReason::MissingAuthority));
    }

    #[test]
    fn verified_authority_without_consent_cannot_manufacture_consent() {
        let request = base_request();
        let authority = verify_authority("authority-only", 1, &request);
        let mut tracker = WelfareAuthorityTracker::default();
        let decision = evaluate_once(
            &mut tracker,
            &authority,
            &BilateralInterventionInterlock,
            &request,
            120,
        )
        .unwrap();
        assert_ne!(decision, InterlockDecision::PolicyPass);
        assert!(reasons(&decision).contains(&InterlockReason::MissingConsent));
    }

    #[test]
    fn withdrawal_supersedes_grant_even_with_fresh_matching_authority() {
        let request = base_request();
        let registry = subject_registry();
        let mut consent_ledger = SubjectConsentLedger::default();
        consent_ledger
            .ingest(verify_consent(
                "grant-before-withdrawal",
                1,
                SubjectConsentDecision::Grant,
                &request,
                &registry,
            ))
            .unwrap();
        consent_ledger
            .ingest(verify_consent(
                "withdrawal",
                2,
                SubjectConsentDecision::Withdraw,
                &request,
                &registry,
            ))
            .unwrap();
        let withdrawn =
            bind_latest_live(&consent_ledger, &registry, SUBJECT_ID, &request, 120).unwrap();
        assert_eq!(
            withdrawn.evidence.consent_state,
            ExplicitConsentState::Withdrawn
        );

        let authority = verify_authority("authority-after-withdrawal", 2, &withdrawn);
        let mut tracker = WelfareAuthorityTracker::default();
        let decision = evaluate_once(
            &mut tracker,
            &authority,
            &BilateralInterventionInterlock,
            &withdrawn,
            120,
        )
        .unwrap();
        assert!(matches!(decision, InterlockDecision::Blocked { .. }));
        assert!(reasons(&decision).contains(&InterlockReason::ConsentDeniedOrWithdrawn));
    }

    #[test]
    fn authority_signed_over_grant_cannot_be_reused_after_withdrawal() {
        let request = base_request();
        let registry = subject_registry();
        let mut consent_ledger = SubjectConsentLedger::default();
        consent_ledger
            .ingest(verify_consent(
                "grant-for-stale-authority",
                1,
                SubjectConsentDecision::Grant,
                &request,
                &registry,
            ))
            .unwrap();
        let granted =
            bind_latest_live(&consent_ledger, &registry, SUBJECT_ID, &request, 120).unwrap();
        let stale_authority = verify_authority("stale-authority", 1, &granted);

        consent_ledger
            .ingest(verify_consent(
                "withdraw-after-authority",
                2,
                SubjectConsentDecision::Withdraw,
                &request,
                &registry,
            ))
            .unwrap();
        let withdrawn =
            bind_latest_live(&consent_ledger, &registry, SUBJECT_ID, &request, 120).unwrap();

        assert_eq!(
            stale_authority.bind_request(&withdrawn, 120),
            Err(WelfareAuthorityUseError::RequestDigestMismatch)
        );
    }

    #[test]
    fn authority_presigned_before_consent_cannot_be_reused_after_grant() {
        let request = base_request();
        let presigned = verify_authority("authority-before-consent", 1, &request);
        let registry = subject_registry();
        let mut consent_ledger = SubjectConsentLedger::default();
        consent_ledger
            .ingest(verify_consent(
                "later-grant",
                1,
                SubjectConsentDecision::Grant,
                &request,
                &registry,
            ))
            .unwrap();
        let granted =
            bind_latest_live(&consent_ledger, &registry, SUBJECT_ID, &request, 120).unwrap();

        assert_eq!(
            presigned.bind_request(&granted, 120),
            Err(WelfareAuthorityUseError::RequestDigestMismatch)
        );
    }

    #[test]
    fn identity_rotation_invalidates_consent_before_authority_can_be_applied() {
        let request = base_request();
        let mut registry = subject_registry();
        let mut consent_ledger = SubjectConsentLedger::default();
        consent_ledger
            .ingest(verify_consent(
                "grant-before-rotation",
                1,
                SubjectConsentDecision::Grant,
                &request,
                &registry,
            ))
            .unwrap();

        registry
            .register(SubjectIdentityBinding {
                subject_id: SUBJECT_ID.into(),
                identity_epoch: 2,
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "subject-key-rotated".into(),
                not_before_unix_s: 110,
                not_after_unix_s: Some(600),
                status: SubjectIdentityStatus::Active,
            })
            .unwrap();

        assert!(matches!(
            bind_latest_live(&consent_ledger, &registry, SUBJECT_ID, &request, 120),
            Err(LiveSubjectConsentUseError::IdentityNoLongerValid { .. })
        ));
    }
}
