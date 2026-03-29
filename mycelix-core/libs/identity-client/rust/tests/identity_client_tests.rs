// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for Mycelix Identity Client
//!
//! These tests verify the identity client functionality including:
//! - DID resolution
//! - Credential verification
//! - Assurance level checking
//! - Guardian endorsement checking

use mycelix_identity_client::{
    AssuranceLevel, CredentialProof, FallbackMode, Guardian, GuardianType,
    IdentityClient, IdentityClientConfig, IdentityVerificationRequest,
    VerifiableCredential,
};
use std::collections::HashMap;

/// Test that the identity client can be created with default configuration
#[tokio::test]
async fn test_client_creation_defaults() {
    let client = IdentityClient::with_defaults();
    assert_eq!(client.conductor_url(), "ws://localhost:8888");

    let (did_cache, rev_cache) = client.cache_stats().await;
    assert_eq!(did_cache, 0);
    assert_eq!(rev_cache, 0);
}

/// Test client creation with custom conductor URL
#[tokio::test]
async fn test_client_creation_with_url() {
    let client = IdentityClient::with_conductor_url("ws://custom:9999");
    assert_eq!(client.conductor_url(), "ws://custom:9999");
}

/// Test client creation with full configuration
#[tokio::test]
async fn test_client_creation_with_config() {
    let config = IdentityClientConfig {
        conductor_url: "ws://localhost:8888".to_string(),
        enable_cache: false,
        cache_ttl_secs: 60,
        fallback_mode: FallbackMode::Silent,
        identity_role_name: "custom_identity".to_string(),
    };

    let client = IdentityClient::new(config);
    assert_eq!(client.conductor_url(), "ws://localhost:8888");
}

/// Test DID validation - invalid format should fail
#[tokio::test]
async fn test_invalid_did_format() {
    let client = IdentityClient::with_defaults();

    // Invalid DID format
    let result = client.resolve_did("invalid:did:format").await.unwrap();
    assert!(!result.success);
    assert!(result.error.is_some());
    assert!(result.error.unwrap().contains("Invalid DID format"));
}

/// Test DID validation - valid format (but unresolvable without hApp)
#[tokio::test]
async fn test_valid_did_format() {
    let config = IdentityClientConfig {
        conductor_url: "ws://localhost:8888".to_string(),
        enable_cache: true,
        cache_ttl_secs: 300,
        fallback_mode: FallbackMode::Silent, // Silent mode for testing
        identity_role_name: "identity".to_string(),
    };
    let client = IdentityClient::new(config);

    // Valid DID format (will fail resolution as hApp not available)
    let result = client.resolve_did("did:mycelix:abc123xyz").await.unwrap();
    // Since hApp is not available, should return false with graceful degradation
    assert!(!result.success);
}

/// Test assurance level comparison
#[tokio::test]
async fn test_assurance_level_comparison() {
    assert!(AssuranceLevel::E4 > AssuranceLevel::E3);
    assert!(AssuranceLevel::E3 > AssuranceLevel::E2);
    assert!(AssuranceLevel::E2 > AssuranceLevel::E1);
    assert!(AssuranceLevel::E1 > AssuranceLevel::E0);

    assert_eq!(AssuranceLevel::E0.value(), 0);
    assert_eq!(AssuranceLevel::E1.value(), 1);
    assert_eq!(AssuranceLevel::E2.value(), 2);
    assert_eq!(AssuranceLevel::E3.value(), 3);
    assert_eq!(AssuranceLevel::E4.value(), 4);
}

/// Test assurance level parsing from string
#[tokio::test]
async fn test_assurance_level_parsing() {
    assert_eq!(AssuranceLevel::from_str("E0"), Some(AssuranceLevel::E0));
    assert_eq!(AssuranceLevel::from_str("e1"), Some(AssuranceLevel::E1));
    assert_eq!(AssuranceLevel::from_str("E2"), Some(AssuranceLevel::E2));
    assert_eq!(AssuranceLevel::from_str("E3"), Some(AssuranceLevel::E3));
    assert_eq!(AssuranceLevel::from_str("E4"), Some(AssuranceLevel::E4));
    assert_eq!(AssuranceLevel::from_str("invalid"), None);
}

/// Test assurance level descriptions
#[tokio::test]
async fn test_assurance_level_descriptions() {
    assert_eq!(AssuranceLevel::E0.description(), "Unverified");
    assert_eq!(AssuranceLevel::E1.description(), "Email Verified");
    assert_eq!(AssuranceLevel::E2.description(), "Phone + Recovery");
    assert_eq!(AssuranceLevel::E3.description(), "Government ID");
    assert_eq!(AssuranceLevel::E4.description(), "Biometric + MFA");
}

/// Test credential verification with valid credential structure
#[tokio::test]
async fn test_credential_verification() {
    let config = IdentityClientConfig {
        conductor_url: "ws://localhost:8888".to_string(),
        enable_cache: true,
        cache_ttl_secs: 300,
        fallback_mode: FallbackMode::Silent,
        identity_role_name: "identity".to_string(),
    };
    let client = IdentityClient::new(config);

    let credential = VerifiableCredential {
        context: vec!["https://www.w3.org/2018/credentials/v1".to_string()],
        id: "urn:uuid:12345".to_string(),
        type_: vec!["VerifiableCredential".to_string(), "EmailCredential".to_string()],
        issuer: "did:mycelix:issuer123".to_string(),
        issuance_date: "2026-01-01T00:00:00Z".to_string(),
        expiration_date: Some("2027-01-01T00:00:00Z".to_string()),
        credential_subject: HashMap::from([
            ("email".to_string(), serde_json::json!("user@example.com")),
        ]),
        credential_status: None,
        proof: Some(CredentialProof {
            type_: "Ed25519Signature2020".to_string(),
            created: "2026-01-01T00:00:00Z".to_string(),
            verification_method: "did:mycelix:issuer123#key-1".to_string(),
            proof_purpose: "assertionMethod".to_string(),
            proof_value: "mock_signature".to_string(),
        }),
    };

    let result = client.verify_credential(&credential).await.unwrap();
    // Basic schema validation should pass
    assert!(result.checks.schema_valid);
    // Expiration check should pass (2027 is in the future)
    assert!(result.checks.not_expired);
}

/// Test get_assurance_level when hApp is unavailable
#[tokio::test]
async fn test_assurance_level_unavailable() {
    let config = IdentityClientConfig {
        conductor_url: "ws://localhost:8888".to_string(),
        enable_cache: true,
        cache_ttl_secs: 300,
        fallback_mode: FallbackMode::Silent,
        identity_role_name: "identity".to_string(),
    };
    let client = IdentityClient::new(config);

    let level = client.get_assurance_level("did:mycelix:test123").await.unwrap();
    // Should return E0 (default) when hApp unavailable
    assert_eq!(level, AssuranceLevel::E0);
}

/// Test meets_assurance_level comparison
#[tokio::test]
async fn test_meets_assurance_level() {
    let config = IdentityClientConfig {
        conductor_url: "ws://localhost:8888".to_string(),
        enable_cache: true,
        cache_ttl_secs: 300,
        fallback_mode: FallbackMode::Silent,
        identity_role_name: "identity".to_string(),
    };
    let client = IdentityClient::new(config);

    // When hApp unavailable, returns E0 which only meets E0 requirement
    let meets_e0 = client.meets_assurance_level("did:mycelix:test", AssuranceLevel::E0).await.unwrap();
    assert!(meets_e0);

    let meets_e1 = client.meets_assurance_level("did:mycelix:test", AssuranceLevel::E1).await.unwrap();
    assert!(!meets_e1);
}

/// Test identity verification request/response
#[tokio::test]
async fn test_identity_verification() {
    let config = IdentityClientConfig {
        conductor_url: "ws://localhost:8888".to_string(),
        enable_cache: true,
        cache_ttl_secs: 300,
        fallback_mode: FallbackMode::Silent,
        identity_role_name: "identity".to_string(),
    };
    let client = IdentityClient::new(config);

    let request = IdentityVerificationRequest {
        did: "did:mycelix:testuser".to_string(),
        min_assurance_level: Some(AssuranceLevel::E1),
        required_credentials: None,
        source_happ: "mycelix_mail".to_string(),
    };

    let response = client.verify_identity(request).await.unwrap();
    assert_eq!(response.did, "did:mycelix:testuser");
    // Without hApp, is_valid should be false
    assert!(!response.is_valid);
}

/// Test guardian endorsement check
#[tokio::test]
async fn test_guardian_endorsement() {
    let config = IdentityClientConfig {
        conductor_url: "ws://localhost:8888".to_string(),
        enable_cache: true,
        cache_ttl_secs: 300,
        fallback_mode: FallbackMode::Silent,
        identity_role_name: "identity".to_string(),
    };
    let client = IdentityClient::new(config);

    let guardians = client.check_guardian_endorsement("did:mycelix:test").await.unwrap();
    // Without hApp connection, returns empty list
    assert!(guardians.is_empty());
}

/// Test full guardian endorsement result
#[tokio::test]
async fn test_guardian_endorsement_result() {
    let config = IdentityClientConfig {
        conductor_url: "ws://localhost:8888".to_string(),
        enable_cache: true,
        cache_ttl_secs: 300,
        fallback_mode: FallbackMode::Silent,
        identity_role_name: "identity".to_string(),
    };
    let client = IdentityClient::new(config);

    let result = client.get_guardian_endorsement_result("did:mycelix:test").await.unwrap();
    assert_eq!(result.did, "did:mycelix:test");
    assert_eq!(result.total_guardians, 0);
    assert_eq!(result.endorsement_count, 0);
    assert!(!result.meets_recovery_threshold);
}

/// Test cache clearing
#[tokio::test]
async fn test_cache_clearing() {
    let client = IdentityClient::with_defaults();

    // Verify initial state
    let (did_count, rev_count) = client.cache_stats().await;
    assert_eq!(did_count, 0);
    assert_eq!(rev_count, 0);

    // Clear (should be a no-op but not fail)
    client.clear_cache().await;

    let (did_count, rev_count) = client.cache_stats().await;
    assert_eq!(did_count, 0);
    assert_eq!(rev_count, 0);
}

/// Test Guardian type defaults
#[tokio::test]
async fn test_guardian_defaults() {
    let guardian = Guardian::default();
    assert!(guardian.did.is_empty());
    assert!(guardian.name.is_none());
    assert_eq!(guardian.guardian_type, GuardianType::Social);
    assert!(!guardian.has_endorsed);
    assert!(guardian.endorsed_at.is_none());
    assert_eq!(guardian.trust_weight, 1.0);
}

/// Test cross-hApp reputation when unavailable
#[tokio::test]
async fn test_cross_happ_reputation() {
    let config = IdentityClientConfig {
        conductor_url: "ws://localhost:8888".to_string(),
        enable_cache: true,
        cache_ttl_secs: 300,
        fallback_mode: FallbackMode::Silent,
        identity_role_name: "identity".to_string(),
    };
    let client = IdentityClient::new(config);

    let reputation = client.get_cross_happ_reputation("did:mycelix:test").await.unwrap();
    assert_eq!(reputation.did, "did:mycelix:test");
    assert_eq!(reputation.aggregate_score, 0.5); // Default fallback score
    assert!(reputation.scores.is_empty());
}

/// Test connection status management
#[tokio::test]
async fn test_connection_status() {
    let client = IdentityClient::with_defaults();

    // Initially not connected (no real conductor)
    let connected = client.check_connection().await;
    assert!(!connected);

    // Set connection status manually
    client.set_connection_status(true).await;
    let connected = client.check_connection().await;
    assert!(connected);

    // Set back to disconnected
    client.set_connection_status(false).await;
    let connected = client.check_connection().await;
    assert!(!connected);
}
