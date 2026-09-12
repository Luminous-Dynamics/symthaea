// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use chrono::{TimeZone, Utc};
use symthaea_core::intervention_interlock::{
    BilateralInterventionInterlock, ExplicitConsentState, InterlockDecision, InterventionEvidence,
    InterventionRequest, WelfareConstraintLevel,
};
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::crypto_digest::Sha256;
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
};
use symthaea_welfare_authority::{
    WelfareAuthorityContextError, WelfareAuthorityPolicyManifest, WelfareAuthorityRole,
    WelfareAuthoritySigner, WelfareAuthoritySignerBinding, WelfareAuthoritySignatureVerifier,
    WelfareAuthorityTracker, build_context_bound_authority_statement, evaluate_once,
    sign_welfare_authority, verify_context_bound_welfare_authority,
};
use symthaea_welfare_consent::{
    SubjectConsentDecision, SubjectConsentLedger, SubjectConsentPolicy,
    SubjectConsentSignatureVerifier, SubjectConsentSigner, SubjectConsentStatement,
    SubjectIdentityBinding, SubjectIdentityRegistry, SubjectIdentityStatus, bind_latest_live,
    sign_subject_consent, verify_subject_consent,
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
    hasher.update(b"symthaea.welfare.context-chain-test.v1\0");
    hasher.update(domain);
    hasher.update(key_id.as_bytes());
    hasher.update(message);
    hasher.finalize().0.to_vec()
}

fn request() -> InterventionRequest {
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
            key_id: "subject-key".into(),
            not_before_unix_s: 50,
            not_after_unix_s: Some(500),
            status: SubjectIdentityStatus::Active,
        })
        .unwrap();
    registry
}

fn consent_bound_request() -> InterventionRequest {
    let request = request();
    let registry = subject_registry();
    let statement = SubjectConsentStatement::for_request(
        "consent-1",
        SUBJECT_ID,
        1,
        1,
        SubjectConsentDecision::Grant,
        100,
        200,
        &request,
    )
    .unwrap();
    let signed = sign_subject_consent(statement, &SubjectSigner).unwrap();
    let verified = verify_subject_consent(
        &signed,
        &registry,
        SubjectConsentPolicy::default(),
        120,
        &SubjectVerifier,
    )
    .unwrap();
    let mut ledger = SubjectConsentLedger::default();
    ledger.ingest(verified).unwrap();
    bind_latest_live(&ledger, &registry, SUBJECT_ID, &request, 120).unwrap()
}

fn authority_key(key_id: &str) -> KeyTrustRecord {
    KeyTrustRecord {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: key_id.into(),
        not_before_unix_s: 50,
        not_after_unix_s: Some(500),
        status: KeyLifecycleStatus::Active,
        usages: BTreeSet::from([KeyUsage::OperatorCommand]),
    }
}

fn trust(sequence: u64) -> TrustSnapshot {
    TrustSnapshot::new(
        sequence,
        80,
        400,
        vec![authority_key("operator"), authority_key("reviewer")],
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
fn exact_subject_consent_policy_and_trust_context_reaches_policy_pass() {
    let request = consent_bound_request();
    let manifest = manifest();
    let trust = trust(7);
    let signed = signed_authority(&request, &manifest, &trust);
    let verified = verify_context_bound_welfare_authority(
        "nonce-1",
        &signed,
        &manifest,
        &trust,
        120,
        &AuthorityVerifier,
    )
    .unwrap();
    let mut tracker = WelfareAuthorityTracker::default();
    let decision = evaluate_once(
        &mut tracker,
        &verified,
        &BilateralInterventionInterlock,
        &request,
        120,
    )
    .unwrap();
    assert_eq!(decision, InterlockDecision::PolicyPass);
}

#[test]
fn trust_rollover_breaks_full_chain_before_interlock() {
    let request = consent_bound_request();
    let manifest = manifest();
    let original = trust(7);
    let signed = signed_authority(&request, &manifest, &original);
    let rolled = trust(8);

    assert!(matches!(
        verify_context_bound_welfare_authority(
            "nonce-1",
            &signed,
            &manifest,
            &rolled,
            120,
            &AuthorityVerifier,
        ),
        Err(WelfareAuthorityContextError::VerificationContextMismatch)
    ));
}

#[test]
fn policy_change_breaks_full_chain_before_interlock() {
    let request = consent_bound_request();
    let original_manifest = manifest();
    let trust = trust(7);
    let signed = signed_authority(&request, &original_manifest, &trust);
    let mut changed_manifest = original_manifest.clone();
    changed_manifest.maximum_authorization_duration_s += 1;

    assert!(matches!(
        verify_context_bound_welfare_authority(
            "nonce-1",
            &signed,
            &changed_manifest,
            &trust,
            120,
            &AuthorityVerifier,
        ),
        Err(WelfareAuthorityContextError::VerificationContextMismatch)
    ));
}
