// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Bridge Zome Unit Tests
//!
//! Comprehensive test coverage for cross-hApp content verification,
//! author reputation queries, content references, and media events.

use serde::{Deserialize, Serialize};

// =============================================================================
// Test Data Structures (mirrors integrity types for testing)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestContentVerificationRequest {
    pub id: String,
    pub content_hash: String,
    pub source_happ: String,
    pub verification_type: TestVerificationType,
    pub requested_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestVerificationType {
    AuthorVerification,
    FactCheck,
    SourceCheck,
    PlagiarismCheck,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestContentVerificationResult {
    pub id: String,
    pub content_hash: String,
    pub author_did: Option<String>,
    pub is_verified: bool,
    pub fact_check_score: Option<f64>,
    pub epistemic_classification: Option<TestEpistemicClassification>,
    pub sources: Vec<String>,
    pub verified_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestEpistemicClassification {
    pub empirical: f64,
    pub normative: f64,
    pub mythic: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestAuthorReputationQuery {
    pub id: String,
    pub author_did: String,
    pub source_happ: String,
    pub queried_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestAuthorReputationResult {
    pub id: String,
    pub author_did: String,
    pub publication_count: u32,
    pub average_quality_score: f64,
    pub fact_check_accuracy: f64,
    pub endorsement_count: u32,
    pub matl_score: f64,
    pub calculated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestContentReference {
    pub id: String,
    pub content_hash: String,
    pub source_happ: String,
    pub title: String,
    pub author_did: String,
    pub content_type: TestContentType,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestContentType {
    Article,
    Report,
    Analysis,
    Opinion,
    Review,
    Evidence,
    Claim,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestMediaBridgeEvent {
    pub id: String,
    pub event_type: TestMediaEventType,
    pub content_hash: Option<String>,
    pub author_did: Option<String>,
    pub payload: String,
    pub source_happ: String,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestMediaEventType {
    ContentPublished,
    ContentVerified,
    FactCheckCompleted,
    AuthorVerified,
    ContentDisputed,
    ContentRetracted,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_verification_request() -> TestContentVerificationRequest {
    TestContentVerificationRequest {
        id: "verify:mycelix-governance:hash123:1704067200000000".to_string(),
        content_hash: "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG".to_string(),
        source_happ: "mycelix-governance".to_string(),
        verification_type: TestVerificationType::AuthorVerification,
        requested_at: 1704067200000000,
    }
}

fn create_test_verification_result() -> TestContentVerificationResult {
    TestContentVerificationResult {
        id: "result:hash123:1704067200000000".to_string(),
        content_hash: "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG".to_string(),
        author_did: Some("did:mycelix:author123".to_string()),
        is_verified: true,
        fact_check_score: Some(0.85),
        epistemic_classification: Some(TestEpistemicClassification {
            empirical: 0.7,
            normative: 0.2,
            mythic: 0.1,
        }),
        sources: vec!["https://source1.com".to_string()],
        verified_at: 1704067200000000,
    }
}

fn create_test_reputation_query() -> TestAuthorReputationQuery {
    TestAuthorReputationQuery {
        id: "author:mycelix-governance:did:mycelix:author123:1704067200000000".to_string(),
        author_did: "did:mycelix:author123".to_string(),
        source_happ: "mycelix-governance".to_string(),
        queried_at: 1704067200000000,
    }
}

fn create_test_reputation_result() -> TestAuthorReputationResult {
    TestAuthorReputationResult {
        id: "rep:did:mycelix:author123:1704067200000000".to_string(),
        author_did: "did:mycelix:author123".to_string(),
        publication_count: 50,
        average_quality_score: 0.78,
        fact_check_accuracy: 0.92,
        endorsement_count: 200,
        matl_score: 0.65,
        calculated_at: 1704067200000000,
    }
}

fn create_test_content_reference() -> TestContentReference {
    TestContentReference {
        id: "ref:mycelix-governance:hash123:1704067200000000".to_string(),
        content_hash: "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG".to_string(),
        source_happ: "mycelix-governance".to_string(),
        title: "Proposal for Community Guidelines".to_string(),
        author_did: "did:mycelix:author123".to_string(),
        content_type: TestContentType::Report,
        created_at: 1704067200000000,
    }
}

fn create_test_media_event() -> TestMediaBridgeEvent {
    TestMediaBridgeEvent {
        id: "event:ContentPublished:1704067200000000".to_string(),
        event_type: TestMediaEventType::ContentPublished,
        content_hash: Some("QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG".to_string()),
        author_did: Some("did:mycelix:author123".to_string()),
        payload: "{}".to_string(),
        source_happ: "mycelix-media".to_string(),
        timestamp: 1704067200000000,
    }
}

fn validate_did(did: &str) -> bool {
    did.starts_with("did:mycelix:")
}

fn validate_content_hash(hash: &str) -> bool {
    !hash.is_empty()
}

fn validate_verification_request(request: &TestContentVerificationRequest) -> Result<(), String> {
    if !validate_content_hash(&request.content_hash) {
        return Err("Content hash required".to_string());
    }
    Ok(())
}

fn validate_reputation_query(query: &TestAuthorReputationQuery) -> Result<(), String> {
    if !validate_did(&query.author_did) {
        return Err("Author must have valid DID".to_string());
    }
    Ok(())
}

fn validate_reputation_result(result: &TestAuthorReputationResult) -> Result<(), String> {
    if result.average_quality_score < 0.0 || result.average_quality_score > 1.0 {
        return Err("Quality score must be 0.0-1.0".to_string());
    }
    Ok(())
}

fn validate_media_event(event: &TestMediaBridgeEvent) -> Result<(), String> {
    if event.source_happ.is_empty() {
        return Err("Source hApp required".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CONTENT VERIFICATION REQUEST TESTS
    // =========================================================================

    #[test]
    fn test_verification_request_creation() {
        let request = create_test_verification_request();
        assert!(!request.id.is_empty());
        assert!(!request.content_hash.is_empty());
        assert!(!request.source_happ.is_empty());
    }

    #[test]
    fn test_verification_request_requires_content_hash() {
        let mut request = create_test_verification_request();
        request.content_hash = "".to_string();
        let result = validate_verification_request(&request);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Content hash required");
    }

    #[test]
    fn test_verification_types_supported() {
        let types = vec![
            TestVerificationType::AuthorVerification,
            TestVerificationType::FactCheck,
            TestVerificationType::SourceCheck,
            TestVerificationType::PlagiarismCheck,
        ];
        for vtype in types {
            let mut request = create_test_verification_request();
            request.verification_type = vtype;
            assert!(validate_verification_request(&request).is_ok());
        }
    }

    #[test]
    fn test_verification_request_has_source_happ() {
        let request = create_test_verification_request();
        assert!(!request.source_happ.is_empty());
    }

    #[test]
    fn test_verification_request_timestamp() {
        let request = create_test_verification_request();
        assert!(request.requested_at > 0);
    }

    // =========================================================================
    // CONTENT VERIFICATION RESULT TESTS
    // =========================================================================

    #[test]
    fn test_verification_result_creation() {
        let result = create_test_verification_result();
        assert!(!result.id.is_empty());
        assert!(!result.content_hash.is_empty());
    }

    #[test]
    fn test_verification_result_verified() {
        let result = create_test_verification_result();
        assert!(result.is_verified);
    }

    #[test]
    fn test_verification_result_not_verified() {
        let mut result = create_test_verification_result();
        result.is_verified = false;
        assert!(!result.is_verified);
    }

    #[test]
    fn test_verification_result_author_optional() {
        let mut result = create_test_verification_result();
        result.author_did = None;
        assert!(result.author_did.is_none());
    }

    #[test]
    fn test_verification_result_fact_check_score_optional() {
        let mut result = create_test_verification_result();
        result.fact_check_score = None;
        assert!(result.fact_check_score.is_none());
    }

    #[test]
    fn test_verification_result_fact_check_score_range() {
        let result = create_test_verification_result();
        if let Some(score) = result.fact_check_score {
            assert!(score >= 0.0 && score <= 1.0);
        }
    }

    #[test]
    fn test_verification_result_epistemic_classification() {
        let result = create_test_verification_result();
        if let Some(ref ec) = result.epistemic_classification {
            assert!(ec.empirical >= 0.0 && ec.empirical <= 1.0);
            assert!(ec.normative >= 0.0 && ec.normative <= 1.0);
            assert!(ec.mythic >= 0.0 && ec.mythic <= 1.0);
        }
    }

    #[test]
    fn test_verification_result_sources() {
        let result = create_test_verification_result();
        assert!(!result.sources.is_empty());
    }

    #[test]
    fn test_verification_result_empty_sources() {
        let mut result = create_test_verification_result();
        result.sources = vec![];
        assert!(result.sources.is_empty());
    }

    // =========================================================================
    // AUTHOR REPUTATION QUERY TESTS
    // =========================================================================

    #[test]
    fn test_reputation_query_creation() {
        let query = create_test_reputation_query();
        assert!(!query.id.is_empty());
        assert!(!query.author_did.is_empty());
    }

    #[test]
    fn test_reputation_query_requires_valid_did() {
        let query = create_test_reputation_query();
        assert!(validate_did(&query.author_did));
    }

    #[test]
    fn test_reputation_query_invalid_did_rejected() {
        let mut query = create_test_reputation_query();
        query.author_did = "did:other:author".to_string();
        let result = validate_reputation_query(&query);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Author must have valid DID");
    }

    #[test]
    fn test_reputation_query_has_source_happ() {
        let query = create_test_reputation_query();
        assert!(!query.source_happ.is_empty());
    }

    // =========================================================================
    // AUTHOR REPUTATION RESULT TESTS
    // =========================================================================

    #[test]
    fn test_reputation_result_creation() {
        let result = create_test_reputation_result();
        assert!(!result.id.is_empty());
        assert!(!result.author_did.is_empty());
    }

    #[test]
    fn test_reputation_result_quality_score_valid_range() {
        let valid_scores = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        for score in valid_scores {
            let mut result = create_test_reputation_result();
            result.average_quality_score = score;
            assert!(validate_reputation_result(&result).is_ok());
        }
    }

    #[test]
    fn test_reputation_result_quality_score_negative_rejected() {
        let mut result = create_test_reputation_result();
        result.average_quality_score = -0.1;
        let validation = validate_reputation_result(&result);
        assert!(validation.is_err());
    }

    #[test]
    fn test_reputation_result_quality_score_over_1_rejected() {
        let mut result = create_test_reputation_result();
        result.average_quality_score = 1.5;
        let validation = validate_reputation_result(&result);
        assert!(validation.is_err());
    }

    #[test]
    fn test_reputation_result_components() {
        let result = create_test_reputation_result();
        assert!(result.publication_count > 0);
        assert!(result.average_quality_score >= 0.0 && result.average_quality_score <= 1.0);
        assert!(result.fact_check_accuracy >= 0.0 && result.fact_check_accuracy <= 1.0);
        assert!(result.endorsement_count > 0);
        assert!(result.matl_score >= 0.0 && result.matl_score <= 1.0);
    }

    #[test]
    fn test_reputation_result_matl_score() {
        // MATL (Mean Achievement Trust Level) from identity hApp
        let result = create_test_reputation_result();
        assert!(result.matl_score >= 0.0 && result.matl_score <= 1.0);
    }

    #[test]
    fn test_reputation_calculation() {
        fn calculate_reputation(
            quality_score: f64,
            fact_check_accuracy: f64,
            matl_score: f64,
        ) -> f64 {
            // Weighted average of reputation components
            (quality_score * 0.3 + fact_check_accuracy * 0.4 + matl_score * 0.3).min(1.0)
        }

        let reputation = calculate_reputation(0.8, 0.9, 0.7);
        assert!(reputation >= 0.0 && reputation <= 1.0);
        assert!((reputation - 0.81).abs() < 0.01);
    }

    // =========================================================================
    // CONTENT REFERENCE TESTS
    // =========================================================================

    #[test]
    fn test_content_reference_creation() {
        let reference = create_test_content_reference();
        assert!(!reference.id.is_empty());
        assert!(!reference.content_hash.is_empty());
        assert!(!reference.title.is_empty());
    }

    #[test]
    fn test_content_reference_has_source_happ() {
        let reference = create_test_content_reference();
        assert!(!reference.source_happ.is_empty());
    }

    #[test]
    fn test_content_reference_author_did() {
        let reference = create_test_content_reference();
        assert!(validate_did(&reference.author_did));
    }

    #[test]
    fn test_content_types_supported() {
        let types = vec![
            TestContentType::Article,
            TestContentType::Report,
            TestContentType::Analysis,
            TestContentType::Opinion,
            TestContentType::Review,
            TestContentType::Evidence,
            TestContentType::Claim,
        ];
        for ctype in types {
            let mut reference = create_test_content_reference();
            reference.content_type = ctype;
            assert!(!reference.id.is_empty());
        }
    }

    #[test]
    fn test_content_reference_timestamp() {
        let reference = create_test_content_reference();
        assert!(reference.created_at > 0);
    }

    // =========================================================================
    // MEDIA BRIDGE EVENT TESTS
    // =========================================================================

    #[test]
    fn test_media_event_creation() {
        let event = create_test_media_event();
        assert!(!event.id.is_empty());
        assert!(!event.source_happ.is_empty());
    }

    #[test]
    fn test_media_event_requires_source_happ() {
        let mut event = create_test_media_event();
        event.source_happ = "".to_string();
        let result = validate_media_event(&event);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Source hApp required");
    }

    #[test]
    fn test_media_event_types_supported() {
        let types = vec![
            TestMediaEventType::ContentPublished,
            TestMediaEventType::ContentVerified,
            TestMediaEventType::FactCheckCompleted,
            TestMediaEventType::AuthorVerified,
            TestMediaEventType::ContentDisputed,
            TestMediaEventType::ContentRetracted,
        ];
        for etype in types {
            let mut event = create_test_media_event();
            event.event_type = etype;
            assert!(validate_media_event(&event).is_ok());
        }
    }

    #[test]
    fn test_media_event_content_hash_optional() {
        let mut event = create_test_media_event();
        event.content_hash = None;
        assert!(validate_media_event(&event).is_ok());
    }

    #[test]
    fn test_media_event_author_did_optional() {
        let mut event = create_test_media_event();
        event.author_did = None;
        assert!(validate_media_event(&event).is_ok());
    }

    #[test]
    fn test_media_event_payload() {
        let event = create_test_media_event();
        assert!(!event.payload.is_empty());
    }

    #[test]
    fn test_media_event_timestamp() {
        let event = create_test_media_event();
        assert!(event.timestamp > 0);
    }

    // =========================================================================
    // CROSS-HAPP INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_content_by_happ_filtering() {
        let references = vec![
            create_test_content_reference(),
            {
                let mut r = create_test_content_reference();
                r.source_happ = "mycelix-identity".to_string();
                r
            },
            {
                let mut r = create_test_content_reference();
                r.source_happ = "mycelix-governance".to_string();
                r
            },
        ];

        let governance_content: Vec<&TestContentReference> = references
            .iter()
            .filter(|r| r.source_happ == "mycelix-governance")
            .collect();

        assert_eq!(governance_content.len(), 2);
    }

    #[test]
    fn test_recent_events_filtering() {
        let now = 1704153600000000i64;
        let one_day = 86400000000i64;

        let events = vec![
            create_test_media_event(),
            {
                let mut e = create_test_media_event();
                e.timestamp = now - one_day; // 1 day ago
                e
            },
            {
                let mut e = create_test_media_event();
                e.timestamp = now - (7 * one_day); // 1 week ago
                e
            },
        ];

        let recent: Vec<&TestMediaBridgeEvent> = events
            .iter()
            .filter(|e| e.timestamp > now - (2 * one_day))
            .collect();

        assert_eq!(recent.len(), 2);
    }

    // =========================================================================
    // VERIFICATION WORKFLOW TESTS
    // =========================================================================

    #[test]
    fn test_verification_workflow() {
        // Step 1: Create verification request
        let request = create_test_verification_request();
        assert!(!request.content_hash.is_empty());

        // Step 2: Create verification result
        let mut result = create_test_verification_result();
        result.content_hash = request.content_hash.clone();
        assert_eq!(request.content_hash, result.content_hash);

        // Step 3: Broadcast event
        let mut event = create_test_media_event();
        event.event_type = TestMediaEventType::ContentVerified;
        event.content_hash = Some(result.content_hash.clone());
        assert!(matches!(
            event.event_type,
            TestMediaEventType::ContentVerified
        ));
    }

    #[test]
    fn test_author_reputation_workflow() {
        // Step 1: Query reputation
        let query = create_test_reputation_query();
        assert!(!query.author_did.is_empty());

        // Step 2: Calculate and return result
        let mut result = create_test_reputation_result();
        result.author_did = query.author_did.clone();
        assert_eq!(query.author_did, result.author_did);

        // Step 3: Broadcast verification event if needed
        let mut event = create_test_media_event();
        event.event_type = TestMediaEventType::AuthorVerified;
        event.author_did = Some(result.author_did.clone());
        assert!(matches!(
            event.event_type,
            TestMediaEventType::AuthorVerified
        ));
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_verification_request_long_content_hash() {
        let mut request = create_test_verification_request();
        request.content_hash = "A".repeat(128);
        assert!(validate_verification_request(&request).is_ok());
    }

    #[test]
    fn test_content_reference_long_title() {
        let mut reference = create_test_content_reference();
        reference.title = "A".repeat(1000);
        assert!(!reference.title.is_empty());
    }

    #[test]
    fn test_media_event_large_payload() {
        let mut event = create_test_media_event();
        event.payload = serde_json::json!({
            "key1": "value1",
            "key2": "value2",
            "data": (0..100).collect::<Vec<i32>>(),
        })
        .to_string();
        assert!(validate_media_event(&event).is_ok());
    }

    #[test]
    fn test_many_content_references() {
        let references: Vec<TestContentReference> = (0..100)
            .map(|i| {
                let mut r = create_test_content_reference();
                r.id = format!("ref:{}", i);
                r.content_hash = format!("hash{}", i);
                r
            })
            .collect();

        assert_eq!(references.len(), 100);
    }

    #[test]
    fn test_reputation_result_zero_publications() {
        let mut result = create_test_reputation_result();
        result.publication_count = 0;
        result.average_quality_score = 0.5; // Neutral for new authors
        assert!(validate_reputation_result(&result).is_ok());
    }

    #[test]
    fn test_verification_result_many_sources() {
        let mut result = create_test_verification_result();
        result.sources = (0..50)
            .map(|i| format!("https://source{}.com", i))
            .collect();
        assert_eq!(result.sources.len(), 50);
    }

    #[test]
    fn test_different_happs_integration() {
        let happs = vec![
            "mycelix-identity",
            "mycelix-governance",
            "mycelix-energy",
            "mycelix-property",
            "mycelix-health",
        ];

        for happ in happs {
            let mut request = create_test_verification_request();
            request.source_happ = happ.to_string();
            assert!(validate_verification_request(&request).is_ok());
        }
    }

    #[test]
    fn test_content_dispute_event() {
        let mut event = create_test_media_event();
        event.event_type = TestMediaEventType::ContentDisputed;
        event.payload = serde_json::json!({
            "dispute_id": "dispute123",
            "reason": "Inaccurate information",
        })
        .to_string();
        assert!(matches!(
            event.event_type,
            TestMediaEventType::ContentDisputed
        ));
    }

    #[test]
    fn test_content_retraction_event() {
        let mut event = create_test_media_event();
        event.event_type = TestMediaEventType::ContentRetracted;
        event.payload = serde_json::json!({
            "reason": "Source retracted original claim",
            "retracted_at": 1704153600000000i64,
        })
        .to_string();
        assert!(matches!(
            event.event_type,
            TestMediaEventType::ContentRetracted
        ));
    }
}
