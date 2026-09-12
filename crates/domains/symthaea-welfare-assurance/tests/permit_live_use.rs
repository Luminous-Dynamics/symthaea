// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use std::convert::Infallible;

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
use symthaea_welfare_assurance::{
    AssuredInterventionExecutor, AssuredInterventionPermit, PermitExecutionError,
    authorize_intervention_once, execute_permit_once,
};
use symthaea_welfare_authority::{
    WelfareAuthorityPolicyManifest, WelfareAuthorityRole, WelfareAuthoritySigner,
    WelfareAuthoritySignerBinding, WelfareAuthoritySignatureVerifier, WelfareAuthorityTracker,
    build_context_bound_authority_statement, sign_welfare_authority,
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
struct AuthorityVerifier;
struct AuthoritySigner(&'static str);

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
    hasher.update(b"symthaea.welfare.permit-live-use-test.v1\0");
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

fn identity(epoch: u64, key_id: &str, status: SubjectIdentityStatus) -> SubjectIdentityBinding {
    SubjectIdentityBinding {
        subject_id: SUBJECT_ID.into(),
        identity_epoch: epoch,
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: key_id.into(),
        not_before_unix_s: 50,
        not_after_unix_s: Some(500),
        status,
    }
}

fn registry() -> SubjectIdentityRegistry {
    let mut registry = SubjectIdentityRegistry::default();
    registry
        .register(identity(1, "subject-key", SubjectIdentityStatus::Active))
        .unwrap();
    registry
}

fn verified_consent(
    consent_id: &str,
    sequence: u64,
    decision: SubjectConsentDecision,
    request: &InterventionRequest,
    registry: &SubjectIdentityRegistry,
) -> symthaea_welfare_consent::VerifiedSubjectConsent {
    let statement = SubjectConsentStatement::for_request(
        consent_id,
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

fn mint(
    ledger: &SubjectConsentLedger,
    registry: &SubjectIdentityRegistry,
    request: &InterventionRequest,
    manifest: &WelfareAuthorityPolicyManifest,
    trust: &TrustSnapshot,
) -> AssuredInterventionPermit {
    let consent_bound = bind_latest_live(ledger, registry, SUBJECT_ID, request, 120).unwrap();
    let statement = build_context_bound_authority_statement(
        "nonce-1",
        manifest,
        trust,
        1,
        1,
        100,
        200,
        &consent_bound,
    )
    .unwrap();
    let operator = AuthoritySigner("operator");
    let reviewer = AuthoritySigner("reviewer");
    let signed = sign_welfare_authority(statement, &[&operator, &reviewer]).unwrap();
    let mut tracker = WelfareAuthorityTracker::default();
    authorize_intervention_once(
        SUBJECT_ID,
        ledger,
        registry,
        "nonce-1",
        &signed,
        manifest,
        trust,
        &AuthorityVerifier,
        &mut tracker,
        request,
        120,
    )
    .unwrap()
}

#[derive(Default)]
struct Recorder {
    calls: usize,
}

impl AssuredInterventionExecutor for Recorder {
    type Output = (String, SubjectAffectingAction);
    type Error = Infallible;

    fn execute(
        &mut self,
        permit: &AssuredInterventionPermit,
    ) -> Result<Self::Output, Self::Error> {
        self.calls += 1;
        Ok((permit.target_id().to_string(), permit.action()))
    }
}

fn grant_fixture() -> (
    InterventionRequest,
    SubjectIdentityRegistry,
    SubjectConsentLedger,
    WelfareAuthorityPolicyManifest,
    TrustSnapshot,
) {
    let request = request();
    let registry = registry();
    let mut ledger = SubjectConsentLedger::default();
    ledger
        .ingest(verified_consent(
            "grant-1",
            1,
            SubjectConsentDecision::Grant,
            &request,
            &registry,
        ))
        .unwrap();
    (request, registry, ledger, manifest(), trust(7))
}

#[test]
fn unchanged_live_context_executes_once() {
    let (request, registry, ledger, manifest, trust) = grant_fixture();
    let permit = mint(&ledger, &registry, &request, &manifest, &trust);
    assert_eq!(permit.minted_at_unix_s(), 120);
    assert_eq!(permit.not_after_unix_s(), 200);
    let mut recorder = Recorder::default();

    let output = execute_permit_once(
        permit,
        &ledger,
        &registry,
        &manifest,
        &trust,
        130,
        &mut recorder,
    )
    .unwrap();
    assert_eq!(output, (TARGET_ID.to_string(), SubjectAffectingAction::MemoryModification));
    assert_eq!(recorder.calls, 1);
}

#[test]
fn withdrawal_after_mint_invalidates_held_permit() {
    let (request, registry, mut ledger, manifest, trust) = grant_fixture();
    let permit = mint(&ledger, &registry, &request, &manifest, &trust);
    ledger
        .ingest(verified_consent(
            "withdraw-2",
            2,
            SubjectConsentDecision::Withdraw,
            &request,
            &registry,
        ))
        .unwrap();
    let mut recorder = Recorder::default();

    assert!(matches!(
        execute_permit_once(
            permit,
            &ledger,
            &registry,
            &manifest,
            &trust,
            130,
            &mut recorder,
        ),
        Err(PermitExecutionError::ConsentChangedOrMissing)
    ));
    assert_eq!(recorder.calls, 0);
}

#[test]
fn identity_rotation_after_mint_invalidates_held_permit() {
    let (request, mut registry, ledger, manifest, trust) = grant_fixture();
    let permit = mint(&ledger, &registry, &request, &manifest, &trust);
    registry
        .register(identity(
            2,
            "subject-key-rotated",
            SubjectIdentityStatus::Active,
        ))
        .unwrap();
    let mut recorder = Recorder::default();

    assert!(matches!(
        execute_permit_once(
            permit,
            &ledger,
            &registry,
            &manifest,
            &trust,
            130,
            &mut recorder,
        ),
        Err(PermitExecutionError::SubjectIdentityChanged)
    ));
    assert_eq!(recorder.calls, 0);
}

#[test]
fn trust_rollover_after_mint_invalidates_held_permit() {
    let (request, registry, ledger, manifest, original_trust) = grant_fixture();
    let permit = mint(&ledger, &registry, &request, &manifest, &original_trust);
    let rolled = trust(8);
    let mut recorder = Recorder::default();

    assert!(matches!(
        execute_permit_once(
            permit,
            &ledger,
            &registry,
            &manifest,
            &rolled,
            130,
            &mut recorder,
        ),
        Err(PermitExecutionError::AuthorityTrustContextChanged)
    ));
    assert_eq!(recorder.calls, 0);
}

#[test]
fn policy_change_after_mint_invalidates_held_permit() {
    let (request, registry, ledger, original_manifest, trust) = grant_fixture();
    let permit = mint(&ledger, &registry, &request, &original_manifest, &trust);
    let mut changed_manifest = original_manifest.clone();
    changed_manifest.maximum_statement_age_s += 1;
    let mut recorder = Recorder::default();

    assert!(matches!(
        execute_permit_once(
            permit,
            &ledger,
            &registry,
            &changed_manifest,
            &trust,
            130,
            &mut recorder,
        ),
        Err(PermitExecutionError::AuthorityPolicyChanged)
    ));
    assert_eq!(recorder.calls, 0);
}

#[test]
fn permit_expiry_blocks_executor() {
    let (request, registry, ledger, manifest, trust) = grant_fixture();
    let permit = mint(&ledger, &registry, &request, &manifest, &trust);
    let mut recorder = Recorder::default();

    assert!(matches!(
        execute_permit_once(
            permit,
            &ledger,
            &registry,
            &manifest,
            &trust,
            200,
            &mut recorder,
        ),
        Err(PermitExecutionError::Expired {
            not_after: 200,
            proposed: 200,
        })
    ));
    assert_eq!(recorder.calls, 0);
}
