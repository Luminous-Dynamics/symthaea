// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use chrono::{TimeZone, Utc};
use symthaea_core::intervention_interlock::{
    ExplicitConsentState, InterventionEvidence, InterventionRequest, WelfareConstraintLevel,
};
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::crypto_digest::Sha256;
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
};
use symthaea_welfare_authority::{
    SignedWelfareInterventionAuthority, WelfareAuthorityContextError, WelfareAuthorityPolicyManifest,
    WelfareAuthorityRole, WelfareAuthoritySigner, WelfareAuthoritySignerBinding,
    WelfareAuthoritySignatureVerifier, build_context_bound_authority_statement,
    digest_welfare_authority_policy_manifest, sign_welfare_authority,
    verify_context_bound_welfare_authority,
};

struct TestSigner {
    key_id: &'static str,
}

impl WelfareAuthoritySigner for TestSigner {
    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::Ed25519
    }

    fn key_id(&self) -> &str {
        self.key_id
    }

    fn sign_welfare_authority(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(test_signature(self.key_id, message))
    }
}

struct TestVerifier;

impl WelfareAuthoritySignatureVerifier for TestVerifier {
    fn verify_welfare_authority(
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
    hasher.update(b"symthaea.welfare.context-binding-test.v1\0");
    hasher.update(key_id.as_bytes());
    hasher.update(message);
    hasher.finalize().0.to_vec()
}

fn key(key_id: &str) -> KeyTrustRecord {
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
        vec![key("operator"), key("reviewer")],
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

fn request() -> InterventionRequest {
    InterventionRequest {
        action: SubjectAffectingAction::MemoryModification,
        target_id: "symthaea:self:instance-1".into(),
        rationale: "bounded memory repair".into(),
        welfare_constraint: WelfareConstraintLevel::Baseline,
        emergency: false,
        less_restrictive_unavailable: false,
        post_hoc_review_required: false,
        evaluated_at: Utc.timestamp_opt(120, 0).single().unwrap(),
        evidence: InterventionEvidence {
            authority_ref: None,
            consent_state: ExplicitConsentState::Granted,
            consent_ref: Some("subject-consent:digest".into()),
            welfare_review_ref: None,
            independent_review_ref: None,
            independent_safety_evidence: Vec::new(),
            welfare_report_ids: Vec::new(),
        },
    }
}

fn signed(
    manifest: &WelfareAuthorityPolicyManifest,
    trust: &TrustSnapshot,
) -> SignedWelfareInterventionAuthority {
    let statement = build_context_bound_authority_statement(
        "nonce-1",
        manifest,
        trust,
        1,
        1,
        100,
        200,
        &request(),
    )
    .unwrap();
    let operator = TestSigner { key_id: "operator" };
    let reviewer = TestSigner { key_id: "reviewer" };
    sign_welfare_authority(statement, &[&operator, &reviewer]).unwrap()
}

#[test]
fn exact_policy_and_trust_context_verifies() {
    let manifest = manifest();
    let trust = trust(7);
    let signed = signed(&manifest, &trust);
    assert!(
        verify_context_bound_welfare_authority(
            "nonce-1",
            &signed,
            &manifest,
            &trust,
            120,
            &TestVerifier,
        )
        .is_ok()
    );
}

#[test]
fn policy_change_cannot_retroactively_activate_old_signature() {
    let original = manifest();
    let trust = trust(7);
    let signed = signed(&original, &trust);

    let mut changed = original.clone();
    changed.maximum_statement_age_s += 1;
    assert!(matches!(
        verify_context_bound_welfare_authority(
            "nonce-1",
            &signed,
            &changed,
            &trust,
            120,
            &TestVerifier,
        ),
        Err(WelfareAuthorityContextError::VerificationContextMismatch)
    ));
}

#[test]
fn trust_snapshot_rollover_requires_fresh_authority_signature() {
    let manifest = manifest();
    let original_trust = trust(7);
    let signed = signed(&manifest, &original_trust);
    let rolled = trust(8);

    assert!(matches!(
        verify_context_bound_welfare_authority(
            "nonce-1",
            &signed,
            &manifest,
            &rolled,
            120,
            &TestVerifier,
        ),
        Err(WelfareAuthorityContextError::VerificationContextMismatch)
    ));
}

#[test]
fn signer_binding_order_does_not_change_policy_identity() {
    let left = manifest();
    let mut right = left.clone();
    right.bindings.reverse();
    assert_eq!(
        digest_welfare_authority_policy_manifest(&left).unwrap(),
        digest_welfare_authority_policy_manifest(&right).unwrap()
    );
}
