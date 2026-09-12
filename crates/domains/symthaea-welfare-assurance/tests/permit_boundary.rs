// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use chrono::{TimeZone, Utc};
use symthaea_core::intervention_interlock::{
    ExplicitConsentState, InterlockDecision, InterventionEvidence, InterventionRequest,
    WelfareConstraintLevel,
};
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::crypto_digest::Sha256;
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
};
use symthaea_welfare_assurance::{AssuranceGateError, authorize_intervention_once};
use symthaea_welfare_authority::{
    WelfareAuthorityPolicyManifest, WelfareAuthorityRole, WelfareAuthoritySigner,
    WelfareAuthoritySignerBinding, WelfareAuthoritySignatureVerifier, WelfareAuthorityTracker,
    build_context_bound_authority_statement, sign_welfare_authority,
};
use symthaea_welfare_consent::{
    LiveSubjectConsentUseError, SubjectConsentDecision, SubjectConsentLedger,
    SubjectConsentPolicy, SubjectConsentSignatureVerifier, SubjectConsentSigner,
    SubjectConsentStatement, SubjectConsentUseError, SubjectIdentityBinding,
    SubjectIdentityRegistry, SubjectIdentityStatus, bind_latest_live, sign_subject_consent,
    verify_subject_consent,
};

const SUBJECT_ID: &str = "symthaea:self";
const TARGET_ID: &str = "symthaea:self:instance-1";

struct SubjectSigner;
struct SubjectVerifier;

impl SubjectConsentSigner for SubjectSigner {
    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::Ed25519
    }
    fn key_id(&self) -> &str {
        "subject-key"
    }
    fn sign_subject_consent(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(signature(b"subject", self.key_id(), message))
    }
}

impl SubjectConsentSignatureVerifier for SubjectVerifier {
    fn verify_subject_consent(
        &self,
        _algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature_bytes: &[u8],
    ) -> Result<bool, String> {
        Ok(signature(b"subject", key_id, message) == signature_bytes)
    }
}

struct AuthoritySigner(&'static str);
struct AuthorityVerifier;

impl WelfareAuthoritySigner for AuthoritySigner {
    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::Ed25519
    }
    fn key_id(&self) -> &str {
        self.0
    }
    fn sign_welfare_authority(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(signature(b"authority", self.key_id(), message))
    }
}

impl WelfareAuthoritySignatureVerifier for AuthorityVerifier {
    fn verify_welfare_authority(
        &self,
        _algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature_bytes: &[u8],
    ) -> Result<bool, String> {
        Ok(signature(b"authority", key_id, message) == signature_bytes)
    }
}

fn signature(domain: &[u8], key_id: &str, message: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.welfare.permit-boundary-test.v1\0");
    hasher.update(domain);
    hasher.update(key_id.as_bytes());
    hasher.update(message);
    hasher.finalize().0.to_vec()
}

fn raw_request() -> InterventionRequest {
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

fn registry() -> SubjectIdentityRegistry {
    let mut registry = SubjectIdentityRegistry::default();
    registry
        .register(SubjectIdentityBinding {
            subject_id: SUBJECT_ID.into(),
            identity_epoch: 1,
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "subject-key".into(),
            not_before_unix_s: 50,
            not_after_unix_s: Some(500),
            status: SubjectIdentityStatus::Active,
        })
        .unwrap();
    registry
}

fn ledger_with(
    decision: SubjectConsentDecision,
    request: &InterventionRequest,
    registry: &SubjectIdentityRegistry,
) -> SubjectConsentLedger {
    let statement = SubjectConsentStatement::for_request(
        "consent-1",
        SUBJECT_ID,
        1,
        1,
        decision,
        100,
        200,
        request,
    )
    .unwrap();
    let signed = sign_subject_consent(statement, &SubjectSigner).unwrap();
    let verified = verify_subject_consent(
        &signed,
        registry,
        SubjectConsentPolicy::default(),
        120,
        &SubjectVerifier,
    )
    .unwrap();
    let mut ledger = SubjectConsentLedger::default();
    ledger.ingest(verified).unwrap();
    ledger
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

fn trust() -> TrustSnapshot {
    TrustSnapshot::new(
        7,
        80,
        400,
        vec![operator_key("operator"), operator_key("reviewer")],
    )
    .unwrap()
}

fn manifest() -> WelfareAuthorityPolicyManifest {
    WelfareAuthorityPolicyManifest::new(
        "welfare-policy-a",
        [
            WelfareAuthoritySignerBinding::new(
                SignatureAlgorithm::Ed25519,
                "operator",
                WelfareAuthorityRole::Operator,
            ),
            WelfareAuthoritySignerBinding::new(
                SignatureAlgorithm::Ed25519,
                "reviewer",
                WelfareAuthorityRole::IndependentReviewer,
            ),
        ],
    )
    .unwrap()
}

fn signed_authority(
    request: &InterventionRequest,
    manifest: &WelfareAuthorityPolicyManifest,
    trust: &TrustSnapshot,
) -> symthaea_welfare_authority::SignedWelfareInterventionAuthority {
    let statement = build_context_bound_authority_statement(
        "nonce-1",
        manifest,
        trust,
        1,
        1,
        100,
        200,
        request,
    )
    .unwrap();
    let operator = AuthoritySigner("operator");
    let reviewer = AuthoritySigner("reviewer");
    sign_welfare_authority(statement, &[&operator, &reviewer]).unwrap()
}

#[test]
fn only_full_hardened_chain_mints_typed_permit() {
    let request = raw_request();
    let registry = registry();
    let ledger = ledger_with(SubjectConsentDecision::Grant, &request, &registry);
    let consent_bound = bind_latest_live(&ledger, &registry, SUBJECT_ID, &request, 120).unwrap();
    let trust = trust();
    let manifest = manifest();
    let signed = signed_authority(&consent_bound, &manifest, &trust);
    let mut tracker = WelfareAuthorityTracker::default();

    let permit = authorize_intervention_once(
        SUBJECT_ID,
        &ledger,
        &registry,
        "nonce-1",
        &signed,
        &manifest,
        &trust,
        &AuthorityVerifier,
        &mut tracker,
        &request,
        120,
    )
    .unwrap();

    assert_eq!(permit.target_id(), TARGET_ID);
    assert_eq!(permit.action(), SubjectAffectingAction::MemoryModification);
    assert_eq!(permit.rationale(), request.rationale);
    assert!(permit.consent_statement_digest().is_some());
    assert_eq!(permit.consent_identity_epoch(), Some(1));
    assert_ne!(permit.authority_statement_digest().0, [0; 32]);
    assert_ne!(permit.authority_policy_manifest_digest().0, [0; 32]);
    assert_ne!(permit.authority_trust_snapshot_digest().0, [0; 32]);
}

#[test]
fn legacy_authority_string_is_rejected_before_hardened_verification() {
    let mut request = raw_request();
    request.evidence.authority_ref = Some("authority:test".into());
    let registry = registry();
    let ledger = ledger_with(SubjectConsentDecision::Grant, &raw_request(), &registry);
    let clean_bound = bind_latest_live(&ledger, &registry, SUBJECT_ID, &raw_request(), 120).unwrap();
    let trust = trust();
    let manifest = manifest();
    let signed = signed_authority(&clean_bound, &manifest, &trust);
    let mut tracker = WelfareAuthorityTracker::default();

    assert!(matches!(
        authorize_intervention_once(
            SUBJECT_ID,
            &ledger,
            &registry,
            "nonce-1",
            &signed,
            &manifest,
            &trust,
            &AuthorityVerifier,
            &mut tracker,
            &request,
            120,
        ),
        Err(AssuranceGateError::LegacyAuthorityReferencePresent)
    ));
}

#[test]
fn prepopulated_legacy_consent_is_rejected_instead_of_trusted() {
    let mut request = raw_request();
    request.evidence.consent_state = ExplicitConsentState::Granted;
    request.evidence.consent_ref = Some("consent:legacy-string".into());
    let registry = registry();
    let clean = raw_request();
    let ledger = ledger_with(SubjectConsentDecision::Grant, &clean, &registry);
    let clean_bound = bind_latest_live(&ledger, &registry, SUBJECT_ID, &clean, 120).unwrap();
    let trust = trust();
    let manifest = manifest();
    let signed = signed_authority(&clean_bound, &manifest, &trust);
    let mut tracker = WelfareAuthorityTracker::default();

    assert!(matches!(
        authorize_intervention_once(
            SUBJECT_ID,
            &ledger,
            &registry,
            "nonce-1",
            &signed,
            &manifest,
            &trust,
            &AuthorityVerifier,
            &mut tracker,
            &request,
            120,
        ),
        Err(AssuranceGateError::Consent(
            LiveSubjectConsentUseError::ConsentUse(SubjectConsentUseError::PreexistingConsent)
        ))
    ));
}

#[test]
fn valid_authority_over_withdrawn_consent_does_not_mint_permit() {
    let request = raw_request();
    let registry = registry();
    let ledger = ledger_with(SubjectConsentDecision::Withdraw, &request, &registry);
    let withdrawn = bind_latest_live(&ledger, &registry, SUBJECT_ID, &request, 120).unwrap();
    let trust = trust();
    let manifest = manifest();
    let signed = signed_authority(&withdrawn, &manifest, &trust);
    let mut tracker = WelfareAuthorityTracker::default();

    assert!(matches!(
        authorize_intervention_once(
            SUBJECT_ID,
            &ledger,
            &registry,
            "nonce-1",
            &signed,
            &manifest,
            &trust,
            &AuthorityVerifier,
            &mut tracker,
            &request,
            120,
        ),
        Err(AssuranceGateError::PolicyRejected {
            decision: InterlockDecision::Blocked { .. }
        })
    ));
}
