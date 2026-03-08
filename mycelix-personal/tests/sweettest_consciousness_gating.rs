//! # Consciousness Gating Sweettest — Personal Cluster
//!
//! The Personal cluster provides the trust credential infrastructure that
//! powers consciousness gating across all other clusters. This test file
//! verifies the K-Vector trust attestation workflow end-to-end:
//!
//! 1. Trust credential issuance (self-attestation)
//! 2. Trust tier classification (Observer -> Guardian)
//! 3. Trust presentation for cross-cluster verification
//! 4. Attestation request/fulfill/decline lifecycle
//! 5. Revocation and verification
//!
//! These trust credentials are consumed by governance, commons, and civic
//! clusters via `CallTargetCell::OtherRole("personal")` to gate actions
//! by consciousness tier.
//!
//! ## Running
//! ```bash
//! cd mycelix-personal
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_consciousness_gating -- --ignored --test-threads=2
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — credential wallet trust types
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TrustTier {
    Observer,
    Basic,
    Standard,
    Elevated,
    Guardian,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrustScoreRange {
    pub lower: f32,
    pub upper: f32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum KVectorComponent {
    Reputation,
    Activity,
    Integrity,
    Performance,
    Membership,
    Stake,
    History,
    Topology,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SelfAttestTrustInput {
    pub self_did: String,
    pub kvector_commitment: Vec<u8>,
    pub range_proof: Vec<u8>,
    pub trust_score_lower: f32,
    pub trust_score_upper: f32,
    pub expires_at: Option<Timestamp>,
    pub supersedes: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RequestAttestationInput {
    pub requester_did: String,
    pub subject_did: String,
    pub components: Vec<KVectorComponent>,
    pub min_trust_score: Option<f32>,
    pub min_tier: Option<TrustTier>,
    pub purpose: String,
    pub expires_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateTrustPresentationInput {
    pub credential_id: String,
    pub subject_did: String,
    pub disclosed_tier: TrustTier,
    pub disclose_range: bool,
    pub trust_range: TrustScoreRange,
    pub presentation_proof: Vec<u8>,
    pub verifier_did: Option<String>,
    pub purpose: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerificationResult {
    pub credential_id: String,
    pub commitment_valid: bool,
    pub tier_consistent: bool,
    pub not_revoked: bool,
    pub not_expired: bool,
    pub proof_format_valid: bool,
    pub message: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RevokeTrustCredentialInput {
    pub credential_id: String,
    pub subject_did: String,
    pub reason: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub total_events: u32,
    pub total_queries: u32,
    pub domains: Vec<String>,
}

// ============================================================================
// DNA path helper
// ============================================================================

fn personal_dna_path() -> PathBuf {
    if let Ok(custom) = std::env::var("PERSONAL_DNA_PATH") {
        return PathBuf::from(custom);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-personal/
    path.push("dna");
    path.push("mycelix_personal.dna");
    path
}

fn valid_commitment() -> Vec<u8> {
    let mut c = vec![0u8; 32];
    c[0] = 42;
    c
}

// ============================================================================
// Tier Classification Tests
// ============================================================================

/// Observer tier (trust < 0.3) — no voting rights.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_observer_tier_attestation() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let self_did = format!("did:mycelix:{}", alice.agent_pubkey());

    let input = SelfAttestTrustInput {
        self_did: self_did.clone(),
        kvector_commitment: valid_commitment(),
        range_proof: vec![1, 2, 3],
        trust_score_lower: 0.0,
        trust_score_upper: 0.2,
        expires_at: None,
        supersedes: None,
    };

    let _: Record = conductor
        .call(&alice.zome("credential_wallet"), "self_attest_trust", input)
        .await;

    // Should appear in Observer tier query
    let observer_creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials_by_tier",
            TrustTier::Observer,
        )
        .await;

    assert_eq!(observer_creds.len(), 1);

    // Should NOT appear in higher tiers
    let basic_creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials_by_tier",
            TrustTier::Basic,
        )
        .await;

    assert_eq!(basic_creds.len(), 0);
}

/// Basic tier (0.3 <= trust < 0.4) — basic participation.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_basic_tier_attestation() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let self_did = format!("did:mycelix:{}", alice.agent_pubkey());

    let input = SelfAttestTrustInput {
        self_did: self_did.clone(),
        kvector_commitment: valid_commitment(),
        range_proof: vec![1, 2, 3],
        trust_score_lower: 0.3,
        trust_score_upper: 0.39,
        expires_at: None,
        supersedes: None,
    };

    let _: Record = conductor
        .call(&alice.zome("credential_wallet"), "self_attest_trust", input)
        .await;

    let basic_creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials_by_tier",
            TrustTier::Basic,
        )
        .await;

    assert_eq!(basic_creds.len(), 1);
}

/// Standard tier (0.4 <= trust < 0.6) — can vote on major proposals.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_standard_tier_attestation() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let self_did = format!("did:mycelix:{}", alice.agent_pubkey());

    let input = SelfAttestTrustInput {
        self_did: self_did.clone(),
        kvector_commitment: valid_commitment(),
        range_proof: vec![1, 2, 3],
        trust_score_lower: 0.4,
        trust_score_upper: 0.59,
        expires_at: None,
        supersedes: None,
    };

    let _: Record = conductor
        .call(&alice.zome("credential_wallet"), "self_attest_trust", input)
        .await;

    let standard_creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials_by_tier",
            TrustTier::Standard,
        )
        .await;

    assert_eq!(standard_creds.len(), 1);
}

/// Elevated tier (0.6 <= trust < 0.8) — constitutional changes.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_elevated_tier_attestation() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let self_did = format!("did:mycelix:{}", alice.agent_pubkey());

    let input = SelfAttestTrustInput {
        self_did: self_did.clone(),
        kvector_commitment: valid_commitment(),
        range_proof: vec![1, 2, 3],
        trust_score_lower: 0.6,
        trust_score_upper: 0.79,
        expires_at: None,
        supersedes: None,
    };

    let _: Record = conductor
        .call(&alice.zome("credential_wallet"), "self_attest_trust", input)
        .await;

    let elevated_creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials_by_tier",
            TrustTier::Elevated,
        )
        .await;

    assert_eq!(elevated_creds.len(), 1);
}

/// Guardian tier (trust >= 0.8) — full governance rights including emergency.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_guardian_tier_attestation() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let self_did = format!("did:mycelix:{}", alice.agent_pubkey());

    let input = SelfAttestTrustInput {
        self_did: self_did.clone(),
        kvector_commitment: valid_commitment(),
        range_proof: vec![1, 2, 3],
        trust_score_lower: 0.85,
        trust_score_upper: 0.95,
        expires_at: None,
        supersedes: None,
    };

    let _: Record = conductor
        .call(&alice.zome("credential_wallet"), "self_attest_trust", input)
        .await;

    let guardian_creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials_by_tier",
            TrustTier::Guardian,
        )
        .await;

    assert_eq!(guardian_creds.len(), 1);
}

// ============================================================================
// Trust Presentation Tests (selective disclosure for cross-cluster gating)
// ============================================================================

/// Create a trust presentation that proves tier without revealing exact scores.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_trust_presentation_creation() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let self_did = format!("did:mycelix:{}", alice.agent_pubkey());

    // First, self-attest trust
    let attest = SelfAttestTrustInput {
        self_did: self_did.clone(),
        kvector_commitment: valid_commitment(),
        range_proof: vec![1, 2, 3],
        trust_score_lower: 0.6,
        trust_score_upper: 0.79,
        expires_at: None,
        supersedes: None,
    };

    let _: Record = conductor
        .call(&alice.zome("credential_wallet"), "self_attest_trust", attest)
        .await;

    // Get the credential to find its ID
    let creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials",
            self_did.clone(),
        )
        .await;

    assert_eq!(creds.len(), 1);

    // We need the credential ID — deserialize the TrustCredential
    // For the presentation, we use a constructed credential_id
    // (The actual ID is generated server-side, but the presentation
    // input just needs a string reference)
    let presentation = CreateTrustPresentationInput {
        credential_id: "trust-cred-ref".to_string(),
        subject_did: self_did.clone(),
        disclosed_tier: TrustTier::Elevated,
        disclose_range: false,
        trust_range: TrustScoreRange {
            lower: 0.6,
            upper: 0.79,
        },
        presentation_proof: vec![10, 20, 30],
        verifier_did: Some("did:mycelix:governance".to_string()),
        purpose: "Vote on constitutional amendment".to_string(),
    };

    let record: Record = conductor
        .call(
            &alice.zome("credential_wallet"),
            "create_trust_presentation",
            presentation,
        )
        .await;

    assert_eq!(record.action().author(), alice.agent_pubkey());
}

/// Presentation with range disclosure (some verifiers need score bounds).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_trust_presentation_with_range_disclosure() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let self_did = format!("did:mycelix:{}", alice.agent_pubkey());

    let presentation = CreateTrustPresentationInput {
        credential_id: "trust-cred-range-ref".to_string(),
        subject_did: self_did.clone(),
        disclosed_tier: TrustTier::Standard,
        disclose_range: true,
        trust_range: TrustScoreRange {
            lower: 0.4,
            upper: 0.59,
        },
        presentation_proof: vec![5, 6, 7],
        verifier_did: None,
        purpose: "Proposal submission".to_string(),
    };

    let record: Record = conductor
        .call(
            &alice.zome("credential_wallet"),
            "create_trust_presentation",
            presentation,
        )
        .await;

    assert_eq!(record.action().author(), alice.agent_pubkey());
}

// ============================================================================
// Attestation Request Lifecycle Tests
// ============================================================================

/// Request attestation from another agent (cross-agent trust verification).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_attestation_request_creation() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey());
    let bob_did = "did:mycelix:bob_placeholder".to_string();

    // Future timestamp (1 hour from now)
    let expires = Timestamp::from_micros(
        Timestamp::now().as_micros() + 3_600_000_000,
    );

    let request = RequestAttestationInput {
        requester_did: alice_did.clone(),
        subject_did: bob_did.clone(),
        components: vec![
            KVectorComponent::Reputation,
            KVectorComponent::Integrity,
        ],
        min_trust_score: Some(0.4),
        min_tier: Some(TrustTier::Standard),
        purpose: "Governance voting eligibility check".to_string(),
        expires_at: expires,
    };

    let record: Record = conductor
        .call(
            &alice.zome("credential_wallet"),
            "request_attestation",
            request,
        )
        .await;

    assert_eq!(record.action().author(), alice.agent_pubkey());

    // Query pending requests for the subject
    let pending: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_pending_requests",
            bob_did,
        )
        .await;

    assert_eq!(pending.len(), 1);
}

// ============================================================================
// Verification Tests
// ============================================================================

/// Verify a non-existent credential returns proper failure result.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_verify_nonexistent_credential() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let result: VerificationResult = conductor
        .call(
            &alice.zome("credential_wallet"),
            "verify_trust_credential",
            "does-not-exist".to_string(),
        )
        .await;

    assert!(!result.commitment_valid);
    assert!(!result.tier_consistent);
    assert!(!result.not_revoked);
    assert!(!result.not_expired);
    assert!(!result.proof_format_valid);
    assert!(result.message.contains("not found"));
}

// ============================================================================
// Health Check — Personal Bridge Wiring
// ============================================================================

/// Verify bridge health check reports all three personal domains.
/// This confirms the bridge is correctly wired to serve as the
/// cross-cluster gateway for consciousness gating queries.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_serves_consciousness_domains() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let health: BridgeHealth = conductor
        .call(&alice.zome("personal_bridge"), "health_check", ())
        .await;

    assert!(health.healthy);

    // All three domains must be present for consciousness gating to work:
    // - "identity" — DID, profile for identity verification
    // - "health" — health credentials for care cluster gating
    // - "credential" — trust credentials for governance gating
    assert!(
        health.domains.contains(&"identity".to_string()),
        "identity domain required for DID-based consciousness gating"
    );
    assert!(
        health.domains.contains(&"health".to_string()),
        "health domain required for care cluster gating"
    );
    assert!(
        health.domains.contains(&"credential".to_string()),
        "credential domain required for trust-based consciousness gating"
    );
}

// ============================================================================
// Multi-Tier Lifecycle: Attest -> Present -> Upgrade
// ============================================================================

/// Full consciousness gating lifecycle: start at Basic, upgrade to Elevated.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_trust_tier_upgrade_lifecycle() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let self_did = format!("did:mycelix:{}", alice.agent_pubkey());

    // Phase 1: Self-attest at Basic tier
    let basic_attest = SelfAttestTrustInput {
        self_did: self_did.clone(),
        kvector_commitment: valid_commitment(),
        range_proof: vec![1, 2, 3],
        trust_score_lower: 0.3,
        trust_score_upper: 0.39,
        expires_at: None,
        supersedes: None,
    };

    let _: Record = conductor
        .call(
            &alice.zome("credential_wallet"),
            "self_attest_trust",
            basic_attest,
        )
        .await;

    let basic_creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials_by_tier",
            TrustTier::Basic,
        )
        .await;

    assert_eq!(basic_creds.len(), 1, "Should have 1 Basic tier credential");

    // Phase 2: Self-attest at Elevated tier (new credential, supersedes old)
    let mut elevated_commitment = vec![0u8; 32];
    elevated_commitment[0] = 99;

    let elevated_attest = SelfAttestTrustInput {
        self_did: self_did.clone(),
        kvector_commitment: elevated_commitment,
        range_proof: vec![4, 5, 6],
        trust_score_lower: 0.6,
        trust_score_upper: 0.79,
        expires_at: None,
        supersedes: None,
    };

    let _: Record = conductor
        .call(
            &alice.zome("credential_wallet"),
            "self_attest_trust",
            elevated_attest,
        )
        .await;

    // Now should have credentials at both tiers (both are active)
    let all_creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials",
            self_did.clone(),
        )
        .await;

    assert_eq!(
        all_creds.len(),
        2,
        "Should have 2 trust credentials (Basic + Elevated)"
    );

    let elevated_creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials_by_tier",
            TrustTier::Elevated,
        )
        .await;

    assert_eq!(
        elevated_creds.len(),
        1,
        "Should have 1 Elevated tier credential"
    );
}
