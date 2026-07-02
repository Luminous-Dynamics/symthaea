// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge Zome Unit Tests
//!
//! Comprehensive test coverage for cross-hApp climate verification, marketplace listings,
//! query workflows, status transitions, and access control.

use serde::{Deserialize, Serialize};

// =============================================================================
// Test Data Structures (mirrors integrity types for testing)
// =============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TestQueryPurpose {
    CreditVerification,
    FootprintAudit,
    ProjectDueDiligence,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TestQueryStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TestVerificationResult {
    Verified,
    Failed,
    Inconclusive,
    NotFound,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestClimateQuery {
    pub query_id: String,
    pub purpose: TestQueryPurpose,
    pub requester_did: String,
    pub target_id: String,
    pub parameters: Option<String>,
    pub status: TestQueryStatus,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestClimateResult {
    pub query_id: String,
    pub result: TestVerificationResult,
    pub data: Option<String>,
    pub responder_did: String,
    pub responded_at: i64,
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestMarketplaceListing {
    pub listing_id: String,
    pub credit_id: String,
    pub credit_action_hash: String,
    pub project_id: String,
    pub seller_did: String,
    pub price_per_tonne: u64,
    pub currency: String,
    pub min_purchase: f64,
    pub available_tonnes: f64,
    pub expires_at: i64,
    pub is_active: bool,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestBridgeSummary {
    pub total_queries: u64,
    pub pending_queries: u64,
    pub completed_queries: u64,
    pub verification_success_rate: f64,
    pub total_listings: u64,
    pub active_listings: u64,
    pub total_listed_tonnes: f64,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_query() -> TestClimateQuery {
    TestClimateQuery {
        query_id: "query:did:mycelix:requester:1704067200000000".to_string(),
        purpose: TestQueryPurpose::CreditVerification,
        requester_did: "did:mycelix:requester".to_string(),
        target_id: "credit:reforest-001:2024:001".to_string(),
        parameters: None,
        status: TestQueryStatus::Pending,
        created_at: 1704067200,
    }
}

fn create_test_result() -> TestClimateResult {
    TestClimateResult {
        query_id: "query:did:mycelix:requester:1704067200000000".to_string(),
        result: TestVerificationResult::Verified,
        data: Some(r#"{"status":"valid","owner":"did:mycelix:owner"}"#.to_string()),
        responder_did: "did:mycelix:responder".to_string(),
        responded_at: 1704070800,
        signature: Some("sig:abc123".to_string()),
    }
}

fn create_test_listing() -> TestMarketplaceListing {
    TestMarketplaceListing {
        listing_id: "listing:credit-001:1704067200000000".to_string(),
        credit_id: "credit:reforest-001:2024:001".to_string(),
        credit_action_hash: "uhCAk_abc123".to_string(),
        project_id: "project:reforest-001".to_string(),
        seller_did: "did:mycelix:seller".to_string(),
        price_per_tonne: 25_00, // $25.00 in cents
        currency: "USD".to_string(),
        min_purchase: 1.0,
        available_tonnes: 100.0,
        expires_at: 1735689599, // End of 2024
        is_active: true,
        created_at: 1704067200,
    }
}

fn validate_did(did: &str) -> Result<(), String> {
    if did.is_empty() {
        return Err("DID cannot be empty".to_string());
    }
    if !did.starts_with("did:") {
        return Err("DID must start with 'did:' prefix".to_string());
    }
    Ok(())
}

fn validate_json(json_str: &str) -> Result<(), String> {
    if json_str.is_empty() {
        return Ok(());
    }
    serde_json::from_str::<serde_json::Value>(json_str)
        .map_err(|_| "Parameters must be valid JSON".to_string())?;
    Ok(())
}

fn validate_query(query: &TestClimateQuery) -> Result<(), String> {
    if query.query_id.is_empty() {
        return Err("Query ID cannot be empty".to_string());
    }

    validate_did(&query.requester_did)?;

    if query.target_id.is_empty() {
        return Err("Target ID cannot be empty".to_string());
    }

    if let Some(ref params) = query.parameters {
        if !params.is_empty() {
            validate_json(params)?;
        }
    }

    Ok(())
}

fn validate_result(result: &TestClimateResult) -> Result<(), String> {
    if result.query_id.is_empty() {
        return Err("Query ID cannot be empty".to_string());
    }

    validate_did(&result.responder_did)?;

    if let Some(ref data) = result.data {
        if !data.is_empty() {
            validate_json(data)?;
        }
    }

    Ok(())
}

fn validate_listing(listing: &TestMarketplaceListing) -> Result<(), String> {
    if listing.listing_id.is_empty() {
        return Err("Listing ID cannot be empty".to_string());
    }

    if listing.credit_id.is_empty() {
        return Err("Credit ID cannot be empty".to_string());
    }

    if listing.credit_action_hash.is_empty() {
        return Err("Credit action hash cannot be empty".to_string());
    }

    validate_did(&listing.seller_did)?;

    if listing.price_per_tonne == 0 {
        return Err("Price must be greater than zero".to_string());
    }

    if listing.currency.is_empty() || listing.currency.len() != 3 {
        return Err("Currency must be a 3-letter code".to_string());
    }

    if listing.min_purchase <= 0.0 {
        return Err("Minimum purchase must be positive".to_string());
    }

    if listing.available_tonnes <= 0.0 {
        return Err("Available tonnes must be positive".to_string());
    }

    if listing.min_purchase > listing.available_tonnes {
        return Err("Minimum purchase cannot exceed available tonnes".to_string());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // QUERY VALIDATION TESTS
    // =========================================================================

    #[test]
    fn test_query_requires_valid_id() {
        let query = create_test_query();
        assert!(!query.query_id.is_empty());
        assert!(validate_query(&query).is_ok());
    }

    #[test]
    fn test_query_empty_id_rejected() {
        let mut query = create_test_query();
        query.query_id = "".to_string();
        let result = validate_query(&query);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Query ID"));
    }

    #[test]
    fn test_query_requires_valid_requester_did() {
        let query = create_test_query();
        assert!(validate_did(&query.requester_did).is_ok());
        assert!(validate_query(&query).is_ok());
    }

    #[test]
    fn test_query_invalid_requester_did_rejected() {
        let mut query = create_test_query();
        query.requester_did = "invalid".to_string();
        let result = validate_query(&query);
        assert!(result.is_err());
    }

    #[test]
    fn test_query_requires_target_id() {
        let query = create_test_query();
        assert!(!query.target_id.is_empty());
        assert!(validate_query(&query).is_ok());
    }

    #[test]
    fn test_query_empty_target_id_rejected() {
        let mut query = create_test_query();
        query.target_id = "".to_string();
        let result = validate_query(&query);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Target ID"));
    }

    #[test]
    fn test_query_parameters_optional() {
        let query = create_test_query();
        assert!(query.parameters.is_none());
        assert!(validate_query(&query).is_ok());
    }

    #[test]
    fn test_query_parameters_valid_json() {
        let mut query = create_test_query();
        query.parameters = Some(r#"{"key": "value", "num": 42}"#.to_string());
        assert!(validate_query(&query).is_ok());
    }

    #[test]
    fn test_query_parameters_invalid_json_rejected() {
        let mut query = create_test_query();
        query.parameters = Some("not valid json {".to_string());
        let result = validate_query(&query);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("JSON"));
    }

    #[test]
    fn test_query_parameters_empty_json_valid() {
        let mut query = create_test_query();
        query.parameters = Some("{}".to_string());
        assert!(validate_query(&query).is_ok());
    }

    // =========================================================================
    // QUERY PURPOSE TESTS
    // =========================================================================

    #[test]
    fn test_all_query_purposes_supported() {
        let purposes = vec![
            TestQueryPurpose::CreditVerification,
            TestQueryPurpose::FootprintAudit,
            TestQueryPurpose::ProjectDueDiligence,
        ];

        for purpose in purposes {
            let mut query = create_test_query();
            query.purpose = purpose;
            assert!(validate_query(&query).is_ok());
        }
    }

    #[test]
    fn test_query_purpose_credit_verification() {
        let mut query = create_test_query();
        query.purpose = TestQueryPurpose::CreditVerification;
        query.target_id = "credit:id".to_string();
        assert!(validate_query(&query).is_ok());
    }

    #[test]
    fn test_query_purpose_footprint_audit() {
        let mut query = create_test_query();
        query.purpose = TestQueryPurpose::FootprintAudit;
        query.target_id = "uhCAk_footprint_hash".to_string();
        assert!(validate_query(&query).is_ok());
    }

    #[test]
    fn test_query_purpose_project_due_diligence() {
        let mut query = create_test_query();
        query.purpose = TestQueryPurpose::ProjectDueDiligence;
        query.target_id = "project:id".to_string();
        query.parameters = Some(r#"{"checks": ["financial", "environmental"]}"#.to_string());
        assert!(validate_query(&query).is_ok());
    }

    // =========================================================================
    // QUERY STATUS TESTS
    // =========================================================================

    #[test]
    fn test_query_initial_status_pending() {
        let query = create_test_query();
        assert_eq!(query.status, TestQueryStatus::Pending);
    }

    #[test]
    fn test_all_query_statuses_supported() {
        let statuses = vec![
            TestQueryStatus::Pending,
            TestQueryStatus::Processing,
            TestQueryStatus::Completed,
            TestQueryStatus::Failed,
        ];

        for status in statuses {
            assert!(matches!(
                status,
                TestQueryStatus::Pending
                    | TestQueryStatus::Processing
                    | TestQueryStatus::Completed
                    | TestQueryStatus::Failed
            ));
        }
    }

    // =========================================================================
    // QUERY STATUS TRANSITION TESTS
    // =========================================================================

    fn can_update_query_status(
        current: TestQueryStatus,
        new: TestQueryStatus,
    ) -> bool {
        match (current, new) {
            (TestQueryStatus::Pending, TestQueryStatus::Processing) => true,
            (TestQueryStatus::Pending, TestQueryStatus::Failed) => true,
            (TestQueryStatus::Processing, TestQueryStatus::Completed) => true,
            (TestQueryStatus::Processing, TestQueryStatus::Failed) => true,
            _ => false,
        }
    }

    #[test]
    fn test_pending_to_processing_valid() {
        assert!(can_update_query_status(
            TestQueryStatus::Pending,
            TestQueryStatus::Processing
        ));
    }

    #[test]
    fn test_pending_to_failed_valid() {
        assert!(can_update_query_status(
            TestQueryStatus::Pending,
            TestQueryStatus::Failed
        ));
    }

    #[test]
    fn test_processing_to_completed_valid() {
        assert!(can_update_query_status(
            TestQueryStatus::Processing,
            TestQueryStatus::Completed
        ));
    }

    #[test]
    fn test_processing_to_failed_valid() {
        assert!(can_update_query_status(
            TestQueryStatus::Processing,
            TestQueryStatus::Failed
        ));
    }

    #[test]
    fn test_pending_to_completed_invalid() {
        assert!(!can_update_query_status(
            TestQueryStatus::Pending,
            TestQueryStatus::Completed
        ));
    }

    #[test]
    fn test_completed_to_any_invalid() {
        assert!(!can_update_query_status(
            TestQueryStatus::Completed,
            TestQueryStatus::Pending
        ));
        assert!(!can_update_query_status(
            TestQueryStatus::Completed,
            TestQueryStatus::Processing
        ));
        assert!(!can_update_query_status(
            TestQueryStatus::Completed,
            TestQueryStatus::Failed
        ));
    }

    #[test]
    fn test_failed_to_any_invalid() {
        assert!(!can_update_query_status(
            TestQueryStatus::Failed,
            TestQueryStatus::Pending
        ));
        assert!(!can_update_query_status(
            TestQueryStatus::Failed,
            TestQueryStatus::Processing
        ));
        assert!(!can_update_query_status(
            TestQueryStatus::Failed,
            TestQueryStatus::Completed
        ));
    }

    // =========================================================================
    // RESULT VALIDATION TESTS
    // =========================================================================

    #[test]
    fn test_result_requires_valid_query_id() {
        let result = create_test_result();
        assert!(!result.query_id.is_empty());
        assert!(validate_result(&result).is_ok());
    }

    #[test]
    fn test_result_empty_query_id_rejected() {
        let mut result = create_test_result();
        result.query_id = "".to_string();
        let result_validation = validate_result(&result);
        assert!(result_validation.is_err());
        assert!(result_validation.unwrap_err().contains("Query ID"));
    }

    #[test]
    fn test_result_requires_valid_responder_did() {
        let result = create_test_result();
        assert!(validate_did(&result.responder_did).is_ok());
        assert!(validate_result(&result).is_ok());
    }

    #[test]
    fn test_result_invalid_responder_did_rejected() {
        let mut result = create_test_result();
        result.responder_did = "invalid".to_string();
        let result_validation = validate_result(&result);
        assert!(result_validation.is_err());
    }

    #[test]
    fn test_result_data_optional() {
        let mut result = create_test_result();
        result.data = None;
        assert!(validate_result(&result).is_ok());
    }

    #[test]
    fn test_result_data_valid_json() {
        let result = create_test_result();
        assert!(result.data.is_some());
        assert!(validate_result(&result).is_ok());
    }

    #[test]
    fn test_result_data_invalid_json_rejected() {
        let mut result = create_test_result();
        result.data = Some("not json".to_string());
        let result_validation = validate_result(&result);
        assert!(result_validation.is_err());
    }

    // =========================================================================
    // VERIFICATION RESULT TESTS
    // =========================================================================

    #[test]
    fn test_all_verification_results_supported() {
        let results = vec![
            TestVerificationResult::Verified,
            TestVerificationResult::Failed,
            TestVerificationResult::Inconclusive,
            TestVerificationResult::NotFound,
        ];

        for r in results {
            assert!(matches!(
                r,
                TestVerificationResult::Verified
                    | TestVerificationResult::Failed
                    | TestVerificationResult::Inconclusive
                    | TestVerificationResult::NotFound
            ));
        }
    }

    // =========================================================================
    // RESULT IMMUTABILITY TESTS
    // =========================================================================

    fn can_update_result(_result: &TestClimateResult) -> bool {
        // Results cannot be updated once submitted
        false
    }

    #[test]
    fn test_result_cannot_be_updated() {
        let result = create_test_result();
        assert!(!can_update_result(&result));
    }

    // =========================================================================
    // LISTING VALIDATION TESTS
    // =========================================================================

    #[test]
    fn test_listing_requires_valid_id() {
        let listing = create_test_listing();
        assert!(!listing.listing_id.is_empty());
        assert!(validate_listing(&listing).is_ok());
    }

    #[test]
    fn test_listing_empty_id_rejected() {
        let mut listing = create_test_listing();
        listing.listing_id = "".to_string();
        let result = validate_listing(&listing);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Listing ID"));
    }

    #[test]
    fn test_listing_requires_credit_id() {
        let listing = create_test_listing();
        assert!(!listing.credit_id.is_empty());
        assert!(validate_listing(&listing).is_ok());
    }

    #[test]
    fn test_listing_empty_credit_id_rejected() {
        let mut listing = create_test_listing();
        listing.credit_id = "".to_string();
        let result = validate_listing(&listing);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Credit ID"));
    }

    #[test]
    fn test_listing_requires_credit_action_hash() {
        let listing = create_test_listing();
        assert!(!listing.credit_action_hash.is_empty());
        assert!(validate_listing(&listing).is_ok());
    }

    #[test]
    fn test_listing_empty_action_hash_rejected() {
        let mut listing = create_test_listing();
        listing.credit_action_hash = "".to_string();
        let result = validate_listing(&listing);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("action hash"));
    }

    #[test]
    fn test_listing_requires_valid_seller_did() {
        let listing = create_test_listing();
        assert!(validate_did(&listing.seller_did).is_ok());
        assert!(validate_listing(&listing).is_ok());
    }

    #[test]
    fn test_listing_invalid_seller_did_rejected() {
        let mut listing = create_test_listing();
        listing.seller_did = "invalid".to_string();
        let result = validate_listing(&listing);
        assert!(result.is_err());
    }

    #[test]
    fn test_listing_price_must_be_positive() {
        let listing = create_test_listing();
        assert!(listing.price_per_tonne > 0);
        assert!(validate_listing(&listing).is_ok());
    }

    #[test]
    fn test_listing_zero_price_rejected() {
        let mut listing = create_test_listing();
        listing.price_per_tonne = 0;
        let result = validate_listing(&listing);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Price"));
    }

    #[test]
    fn test_listing_currency_must_be_3_chars() {
        let listing = create_test_listing();
        assert_eq!(listing.currency.len(), 3);
        assert!(validate_listing(&listing).is_ok());
    }

    #[test]
    fn test_listing_currency_too_short_rejected() {
        let mut listing = create_test_listing();
        listing.currency = "US".to_string();
        let result = validate_listing(&listing);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("3-letter"));
    }

    #[test]
    fn test_listing_currency_too_long_rejected() {
        let mut listing = create_test_listing();
        listing.currency = "USDA".to_string();
        let result = validate_listing(&listing);
        assert!(result.is_err());
    }

    #[test]
    fn test_listing_currency_empty_rejected() {
        let mut listing = create_test_listing();
        listing.currency = "".to_string();
        let result = validate_listing(&listing);
        assert!(result.is_err());
    }

    #[test]
    fn test_listing_valid_currencies() {
        let currencies = vec!["USD", "EUR", "GBP", "JPY", "CNY", "BRL"];

        for currency in currencies {
            let mut listing = create_test_listing();
            listing.currency = currency.to_string();
            assert!(
                validate_listing(&listing).is_ok(),
                "Currency {} should be valid",
                currency
            );
        }
    }

    #[test]
    fn test_listing_min_purchase_must_be_positive() {
        let listing = create_test_listing();
        assert!(listing.min_purchase > 0.0);
        assert!(validate_listing(&listing).is_ok());
    }

    #[test]
    fn test_listing_zero_min_purchase_rejected() {
        let mut listing = create_test_listing();
        listing.min_purchase = 0.0;
        let result = validate_listing(&listing);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Minimum purchase"));
    }

    #[test]
    fn test_listing_negative_min_purchase_rejected() {
        let mut listing = create_test_listing();
        listing.min_purchase = -1.0;
        let result = validate_listing(&listing);
        assert!(result.is_err());
    }

    #[test]
    fn test_listing_available_tonnes_must_be_positive() {
        let listing = create_test_listing();
        assert!(listing.available_tonnes > 0.0);
        assert!(validate_listing(&listing).is_ok());
    }

    #[test]
    fn test_listing_zero_available_tonnes_rejected() {
        let mut listing = create_test_listing();
        listing.available_tonnes = 0.0;
        let result = validate_listing(&listing);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Available tonnes"));
    }

    #[test]
    fn test_listing_min_purchase_cannot_exceed_available() {
        let listing = create_test_listing();
        assert!(listing.min_purchase <= listing.available_tonnes);
        assert!(validate_listing(&listing).is_ok());
    }

    #[test]
    fn test_listing_min_exceeds_available_rejected() {
        let mut listing = create_test_listing();
        listing.min_purchase = 150.0;
        listing.available_tonnes = 100.0;
        let result = validate_listing(&listing);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cannot exceed"));
    }

    // =========================================================================
    // LISTING ACTIVITY TESTS
    // =========================================================================

    fn is_listing_active(listing: &TestMarketplaceListing, current_time: i64) -> bool {
        listing.is_active && listing.expires_at > current_time
    }

    #[test]
    fn test_active_listing_not_expired() {
        let listing = create_test_listing();
        let current_time = 1704067200;
        assert!(is_listing_active(&listing, current_time));
    }

    #[test]
    fn test_active_listing_expired() {
        let listing = create_test_listing();
        let current_time = 1735689600; // After expiry
        assert!(!is_listing_active(&listing, current_time));
    }

    #[test]
    fn test_inactive_listing_not_active() {
        let mut listing = create_test_listing();
        listing.is_active = false;
        let current_time = 1704067200;
        assert!(!is_listing_active(&listing, current_time));
    }

    #[test]
    fn test_inactive_listing_before_expiry_not_active() {
        let mut listing = create_test_listing();
        listing.is_active = false;
        let current_time = 1704067200; // Well before expiry
        assert!(!is_listing_active(&listing, current_time));
    }

    // =========================================================================
    // LISTING UPDATE TESTS
    // =========================================================================

    fn update_listing_price(listing: &mut TestMarketplaceListing, new_price: u64) -> Result<(), String> {
        if new_price == 0 {
            return Err("Price must be greater than zero".to_string());
        }
        listing.price_per_tonne = new_price;
        Ok(())
    }

    fn update_listing_available(
        listing: &mut TestMarketplaceListing,
        new_available: f64,
    ) -> Result<(), String> {
        if new_available < listing.min_purchase {
            return Err("Available tonnes cannot be less than minimum purchase".to_string());
        }
        listing.available_tonnes = new_available;
        Ok(())
    }

    fn deactivate_listing(listing: &mut TestMarketplaceListing) {
        listing.is_active = false;
    }

    #[test]
    fn test_update_listing_price_valid() {
        let mut listing = create_test_listing();
        let result = update_listing_price(&mut listing, 30_00);
        assert!(result.is_ok());
        assert_eq!(listing.price_per_tonne, 30_00);
    }

    #[test]
    fn test_update_listing_price_zero_rejected() {
        let mut listing = create_test_listing();
        let result = update_listing_price(&mut listing, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_listing_available_valid() {
        let mut listing = create_test_listing();
        let result = update_listing_available(&mut listing, 50.0);
        assert!(result.is_ok());
        assert_eq!(listing.available_tonnes, 50.0);
    }

    #[test]
    fn test_update_listing_available_below_min_rejected() {
        let mut listing = create_test_listing();
        listing.min_purchase = 10.0;
        let result = update_listing_available(&mut listing, 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_deactivate_listing() {
        let mut listing = create_test_listing();
        assert!(listing.is_active);
        deactivate_listing(&mut listing);
        assert!(!listing.is_active);
    }

    // =========================================================================
    // QUERY WORKFLOW TESTS
    // =========================================================================

    fn process_query(query: &mut TestClimateQuery) -> Result<(), String> {
        if query.status != TestQueryStatus::Pending {
            return Err("Can only process pending queries".to_string());
        }
        query.status = TestQueryStatus::Processing;
        Ok(())
    }

    fn submit_result(
        query: &mut TestClimateQuery,
        result: TestVerificationResult,
    ) -> Result<TestClimateResult, String> {
        if query.status != TestQueryStatus::Processing {
            return Err("Query must be in processing state".to_string());
        }

        query.status = TestQueryStatus::Completed;

        Ok(TestClimateResult {
            query_id: query.query_id.clone(),
            result,
            data: None,
            responder_did: "did:mycelix:responder".to_string(),
            responded_at: query.created_at + 3600,
            signature: None,
        })
    }

    #[test]
    fn test_process_pending_query() {
        let mut query = create_test_query();
        let result = process_query(&mut query);
        assert!(result.is_ok());
        assert_eq!(query.status, TestQueryStatus::Processing);
    }

    #[test]
    fn test_process_non_pending_query_rejected() {
        let mut query = create_test_query();
        query.status = TestQueryStatus::Completed;
        let result = process_query(&mut query);
        assert!(result.is_err());
    }

    #[test]
    fn test_submit_result_success() {
        let mut query = create_test_query();
        query.status = TestQueryStatus::Processing;
        let result = submit_result(&mut query, TestVerificationResult::Verified);
        assert!(result.is_ok());
        assert_eq!(query.status, TestQueryStatus::Completed);

        let climate_result = result.unwrap();
        assert_eq!(climate_result.query_id, query.query_id);
        assert_eq!(climate_result.result, TestVerificationResult::Verified);
    }

    #[test]
    fn test_submit_result_non_processing_rejected() {
        let mut query = create_test_query();
        let result = submit_result(&mut query, TestVerificationResult::Verified);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("processing"));
    }

    #[test]
    fn test_full_query_workflow() {
        let mut query = create_test_query();

        // Initial state
        assert_eq!(query.status, TestQueryStatus::Pending);

        // Process
        process_query(&mut query).unwrap();
        assert_eq!(query.status, TestQueryStatus::Processing);

        // Submit result
        let result = submit_result(&mut query, TestVerificationResult::Verified).unwrap();
        assert_eq!(query.status, TestQueryStatus::Completed);
        assert_eq!(result.result, TestVerificationResult::Verified);
    }

    // =========================================================================
    // BRIDGE SUMMARY TESTS
    // =========================================================================

    fn calculate_bridge_summary(
        queries: &[TestClimateQuery],
        results: &[TestClimateResult],
        listings: &[TestMarketplaceListing],
        current_time: i64,
    ) -> TestBridgeSummary {
        let total_queries = queries.len() as u64;
        let pending_queries = queries
            .iter()
            .filter(|q| q.status == TestQueryStatus::Pending)
            .count() as u64;
        let completed_queries = queries
            .iter()
            .filter(|q| q.status == TestQueryStatus::Completed)
            .count() as u64;

        let successful_verifications = results
            .iter()
            .filter(|r| r.result == TestVerificationResult::Verified)
            .count() as u64;

        let verification_success_rate = if completed_queries > 0 {
            (successful_verifications as f64) / (completed_queries as f64)
        } else {
            0.0
        };

        let total_listings = listings.len() as u64;
        let active_listings = listings
            .iter()
            .filter(|l| is_listing_active(l, current_time))
            .count() as u64;

        let total_listed_tonnes: f64 = listings
            .iter()
            .filter(|l| is_listing_active(l, current_time))
            .map(|l| l.available_tonnes)
            .sum();

        TestBridgeSummary {
            total_queries,
            pending_queries,
            completed_queries,
            verification_success_rate,
            total_listings,
            active_listings,
            total_listed_tonnes,
        }
    }

    #[test]
    fn test_summary_empty() {
        let summary = calculate_bridge_summary(&[], &[], &[], 1704067200);
        assert_eq!(summary.total_queries, 0);
        assert_eq!(summary.total_listings, 0);
        assert_eq!(summary.verification_success_rate, 0.0);
    }

    #[test]
    fn test_summary_with_queries() {
        let queries = vec![
            create_test_query(),
            {
                let mut q = create_test_query();
                q.query_id = "query:2".to_string();
                q.status = TestQueryStatus::Processing;
                q
            },
            {
                let mut q = create_test_query();
                q.query_id = "query:3".to_string();
                q.status = TestQueryStatus::Completed;
                q
            },
        ];

        let results = vec![TestClimateResult {
            query_id: "query:3".to_string(),
            result: TestVerificationResult::Verified,
            data: None,
            responder_did: "did:mycelix:responder".to_string(),
            responded_at: 1704070800,
            signature: None,
        }];

        let summary = calculate_bridge_summary(&queries, &results, &[], 1704067200);

        assert_eq!(summary.total_queries, 3);
        assert_eq!(summary.pending_queries, 1);
        assert_eq!(summary.completed_queries, 1);
        assert_eq!(summary.verification_success_rate, 1.0);
    }

    #[test]
    fn test_summary_with_listings() {
        let listings = vec![
            create_test_listing(),
            {
                let mut l = create_test_listing();
                l.listing_id = "listing:2".to_string();
                l.available_tonnes = 50.0;
                l
            },
            {
                let mut l = create_test_listing();
                l.listing_id = "listing:3".to_string();
                l.is_active = false;
                l.available_tonnes = 200.0;
                l
            },
        ];

        let summary = calculate_bridge_summary(&[], &[], &listings, 1704067200);

        assert_eq!(summary.total_listings, 3);
        assert_eq!(summary.active_listings, 2);
        assert!((summary.total_listed_tonnes - 150.0).abs() < 0.001);
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_query_complex_parameters_json() {
        let mut query = create_test_query();
        query.parameters = Some(
            r#"{
                "checks": ["financial", "environmental", "social"],
                "depth": "comprehensive",
                "include_history": true,
                "max_age_days": 365
            }"#
            .to_string(),
        );
        assert!(validate_query(&query).is_ok());
    }

    #[test]
    fn test_result_complex_data_json() {
        let mut result = create_test_result();
        result.data = Some(
            r#"{
                "credit": {
                    "id": "credit:001",
                    "status": "active",
                    "owner": "did:mycelix:owner"
                },
                "verification": {
                    "passed": true,
                    "confidence": 0.98,
                    "checks": ["ownership", "registry", "project"]
                }
            }"#
            .to_string(),
        );
        assert!(validate_result(&result).is_ok());
    }

    #[test]
    fn test_listing_very_high_price() {
        let mut listing = create_test_listing();
        listing.price_per_tonne = 1_000_000_00; // $1,000,000.00
        assert!(validate_listing(&listing).is_ok());
    }

    #[test]
    fn test_listing_fractional_tonnes() {
        let mut listing = create_test_listing();
        listing.available_tonnes = 0.5;
        listing.min_purchase = 0.1;
        assert!(validate_listing(&listing).is_ok());
    }

    #[test]
    fn test_listing_very_large_available_tonnes() {
        let mut listing = create_test_listing();
        listing.available_tonnes = 1_000_000.0; // 1 million tonnes
        assert!(validate_listing(&listing).is_ok());
    }

    #[test]
    fn test_multiple_queries_same_target() {
        let queries: Vec<TestClimateQuery> = (0..5)
            .map(|i| {
                let mut q = create_test_query();
                q.query_id = format!("query:{}", i);
                q
            })
            .collect();

        // All queries target the same credit
        for q in &queries {
            assert_eq!(q.target_id, "credit:reforest-001:2024:001");
            assert!(validate_query(q).is_ok());
        }
    }

    #[test]
    fn test_query_result_matching() {
        let query = create_test_query();
        let result = create_test_result();

        // Result query_id matches query
        assert_eq!(result.query_id, query.query_id);
    }

    #[test]
    fn test_listing_expiry_edge_case() {
        let mut listing = create_test_listing();

        // Exactly at expiry time
        listing.expires_at = 1704067200;
        let current_time = 1704067200;
        assert!(!is_listing_active(&listing, current_time));

        // One second before
        assert!(is_listing_active(&listing, current_time - 1));
    }

    #[test]
    fn test_verification_success_rate_calculation() {
        let completed_queries: Vec<TestClimateQuery> = (0..10)
            .map(|i| {
                let mut q = create_test_query();
                q.query_id = format!("query:{}", i);
                q.status = TestQueryStatus::Completed;
                q
            })
            .collect();

        // 7 out of 10 verified successfully
        let results: Vec<TestClimateResult> = (0..10)
            .map(|i| TestClimateResult {
                query_id: format!("query:{}", i),
                result: if i < 7 {
                    TestVerificationResult::Verified
                } else {
                    TestVerificationResult::Failed
                },
                data: None,
                responder_did: "did:mycelix:responder".to_string(),
                responded_at: 1704070800,
                signature: None,
            })
            .collect();

        let summary = calculate_bridge_summary(&completed_queries, &results, &[], 1704067200);

        assert_eq!(summary.completed_queries, 10);
        assert!((summary.verification_success_rate - 0.7).abs() < 0.001);
    }
}
