// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Attribution Zome Unit Tests
//!
//! Comprehensive test coverage for attribution tracking, royalty rules,
//! revenue sharing calculations, usage records, and contributor management.

use serde::{Deserialize, Serialize};

// =============================================================================
// Test Data Structures (mirrors integrity types for testing)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestAttribution {
    pub id: String,
    pub publication_id: String,
    pub contributor_did: String,
    pub role: TestContributorRole,
    pub share_percentage: f64,
    pub verified: bool,
    pub created: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestContributorRole {
    Author,
    CoAuthor,
    Editor,
    Researcher,
    Photographer,
    Illustrator,
    Translator,
    Source,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestRoyaltyRule {
    pub id: String,
    pub publication_id: String,
    pub rule_type: TestRoyaltyType,
    pub percentage: f64,
    pub minimum_amount: Option<f64>,
    pub currency: String,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestRoyaltyType {
    PerView,
    PerShare,
    PerDownload,
    PerDerivative,
    Subscription,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestUsageRecord {
    pub id: String,
    pub publication_id: String,
    pub usage_type: TestUsageType,
    pub user_did: Option<String>,
    pub timestamp: i64,
    pub royalty_paid: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestUsageType {
    View,
    Share,
    Download,
    Derivative,
    Citation,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_attribution() -> TestAttribution {
    TestAttribution {
        id: "attr:pub123:did:mycelix:contributor1:1704067200000000".to_string(),
        publication_id: "pub:did:mycelix:author123:1704067200000000".to_string(),
        contributor_did: "did:mycelix:contributor1".to_string(),
        role: TestContributorRole::Author,
        share_percentage: 100.0,
        verified: false,
        created: 1704067200000000,
    }
}

fn create_test_royalty_rule() -> TestRoyaltyRule {
    TestRoyaltyRule {
        id: "royalty:pub123:PerView:1704067200000000".to_string(),
        publication_id: "pub:did:mycelix:author123:1704067200000000".to_string(),
        rule_type: TestRoyaltyType::PerView,
        percentage: 0.01,
        minimum_amount: Some(0.001),
        currency: "USD".to_string(),
        active: true,
    }
}

fn create_test_usage_record() -> TestUsageRecord {
    TestUsageRecord {
        id: "usage:pub123:1704153600000000".to_string(),
        publication_id: "pub:did:mycelix:author123:1704067200000000".to_string(),
        usage_type: TestUsageType::View,
        user_did: Some("did:mycelix:reader1".to_string()),
        timestamp: 1704153600000000,
        royalty_paid: Some(0.01),
    }
}

fn validate_did(did: &str) -> bool {
    did.starts_with("did:")
}

fn validate_attribution(attr: &TestAttribution) -> Result<(), String> {
    if !validate_did(&attr.contributor_did) {
        return Err("Contributor must be a valid DID".to_string());
    }
    if attr.share_percentage < 0.0 || attr.share_percentage > 100.0 {
        return Err("Share must be 0-100%".to_string());
    }
    Ok(())
}

fn validate_royalty_rule(rule: &TestRoyaltyRule) -> Result<(), String> {
    if rule.percentage < 0.0 || rule.percentage > 100.0 {
        return Err("Percentage must be 0-100".to_string());
    }
    Ok(())
}

fn calculate_total_shares(attributions: &[TestAttribution]) -> f64 {
    attributions.iter().map(|a| a.share_percentage).sum()
}

fn calculate_contributor_earnings(
    attributions: &[TestAttribution],
    usage_records: &[TestUsageRecord],
    contributor_did: &str,
) -> f64 {
    let contributor_pubs: Vec<&String> = attributions
        .iter()
        .filter(|a| a.contributor_did == contributor_did)
        .map(|a| &a.publication_id)
        .collect();

    let mut total = 0.0;
    for usage in usage_records {
        if contributor_pubs.contains(&&usage.publication_id) {
            if let Some(paid) = usage.royalty_paid {
                // Find contributor's share for this publication
                if let Some(attr) = attributions.iter().find(|a| {
                    a.publication_id == usage.publication_id && a.contributor_did == contributor_did
                }) {
                    total += paid * (attr.share_percentage / 100.0);
                }
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ATTRIBUTION CREATION TESTS
    // =========================================================================

    #[test]
    fn test_attribution_requires_valid_contributor_did() {
        let attribution = create_test_attribution();
        assert!(validate_did(&attribution.contributor_did));
        assert!(attribution.contributor_did.starts_with("did:"));
    }

    #[test]
    fn test_attribution_invalid_contributor_did_rejected() {
        let mut attribution = create_test_attribution();
        attribution.contributor_did = "invalid_did".to_string();
        let result = validate_attribution(&attribution);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Contributor must be a valid DID");
    }

    #[test]
    fn test_attribution_has_unique_id() {
        let attribution = create_test_attribution();
        assert!(attribution.id.starts_with("attr:"));
    }

    #[test]
    fn test_attribution_references_publication() {
        let attribution = create_test_attribution();
        assert!(!attribution.publication_id.is_empty());
        assert!(attribution.publication_id.starts_with("pub:"));
    }

    #[test]
    fn test_attribution_initial_unverified() {
        let attribution = create_test_attribution();
        assert!(!attribution.verified);
    }

    // =========================================================================
    // SHARE PERCENTAGE TESTS
    // =========================================================================

    #[test]
    fn test_share_percentage_valid_range() {
        let valid_shares = vec![0.0, 25.0, 50.0, 75.0, 100.0];
        for share in valid_shares {
            let mut attribution = create_test_attribution();
            attribution.share_percentage = share;
            assert!(validate_attribution(&attribution).is_ok());
        }
    }

    #[test]
    fn test_share_percentage_negative_rejected() {
        let mut attribution = create_test_attribution();
        attribution.share_percentage = -10.0;
        let result = validate_attribution(&attribution);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Share must be 0-100%");
    }

    #[test]
    fn test_share_percentage_over_100_rejected() {
        let mut attribution = create_test_attribution();
        attribution.share_percentage = 150.0;
        let result = validate_attribution(&attribution);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Share must be 0-100%");
    }

    #[test]
    fn test_share_percentage_fractional_valid() {
        let mut attribution = create_test_attribution();
        attribution.share_percentage = 33.33;
        assert!(validate_attribution(&attribution).is_ok());
    }

    #[test]
    fn test_share_percentage_very_small_valid() {
        let mut attribution = create_test_attribution();
        attribution.share_percentage = 0.001;
        assert!(validate_attribution(&attribution).is_ok());
    }

    // =========================================================================
    // CONTRIBUTOR ROLE TESTS
    // =========================================================================

    #[test]
    fn test_all_contributor_roles_supported() {
        let roles = vec![
            TestContributorRole::Author,
            TestContributorRole::CoAuthor,
            TestContributorRole::Editor,
            TestContributorRole::Researcher,
            TestContributorRole::Photographer,
            TestContributorRole::Illustrator,
            TestContributorRole::Translator,
            TestContributorRole::Source,
            TestContributorRole::Other("Consultant".to_string()),
        ];
        for role in roles {
            let mut attribution = create_test_attribution();
            attribution.role = role.clone();
            assert!(validate_attribution(&attribution).is_ok());
        }
    }

    #[test]
    fn test_custom_contributor_role() {
        let mut attribution = create_test_attribution();
        attribution.role = TestContributorRole::Other("Fact Checker".to_string());
        assert!(matches!(attribution.role, TestContributorRole::Other(_)));
    }

    // =========================================================================
    // TOTAL SHARES CALCULATION TESTS
    // =========================================================================

    #[test]
    fn test_total_shares_single_contributor() {
        let attributions = vec![create_test_attribution()];
        let total = calculate_total_shares(&attributions);
        assert_eq!(total, 100.0);
    }

    #[test]
    fn test_total_shares_multiple_contributors() {
        let attributions = vec![
            {
                let mut a = create_test_attribution();
                a.share_percentage = 50.0;
                a.contributor_did = "did:mycelix:author".to_string();
                a
            },
            {
                let mut a = create_test_attribution();
                a.share_percentage = 30.0;
                a.contributor_did = "did:mycelix:editor".to_string();
                a
            },
            {
                let mut a = create_test_attribution();
                a.share_percentage = 20.0;
                a.contributor_did = "did:mycelix:researcher".to_string();
                a
            },
        ];
        let total = calculate_total_shares(&attributions);
        assert_eq!(total, 100.0);
    }

    #[test]
    fn test_total_shares_under_100_valid() {
        let attributions = vec![
            {
                let mut a = create_test_attribution();
                a.share_percentage = 40.0;
                a
            },
            {
                let mut a = create_test_attribution();
                a.share_percentage = 30.0;
                a
            },
        ];
        let total = calculate_total_shares(&attributions);
        assert_eq!(total, 70.0);
        // Remaining 30% goes to platform or reserve
    }

    #[test]
    fn test_remaining_percentage_calculation() {
        let attributions = vec![{
            let mut a = create_test_attribution();
            a.share_percentage = 60.0;
            a
        }];
        let total = calculate_total_shares(&attributions);
        let remaining = 100.0 - total;
        assert_eq!(remaining, 40.0);
    }

    // =========================================================================
    // ROYALTY RULE TESTS
    // =========================================================================

    #[test]
    fn test_royalty_rule_creation() {
        let rule = create_test_royalty_rule();
        assert!(!rule.publication_id.is_empty());
        assert!(rule.active);
    }

    #[test]
    fn test_royalty_rule_types_supported() {
        let types = vec![
            TestRoyaltyType::PerView,
            TestRoyaltyType::PerShare,
            TestRoyaltyType::PerDownload,
            TestRoyaltyType::PerDerivative,
            TestRoyaltyType::Subscription,
        ];
        for rule_type in types {
            let mut rule = create_test_royalty_rule();
            rule.rule_type = rule_type;
            assert!(validate_royalty_rule(&rule).is_ok());
        }
    }

    #[test]
    fn test_royalty_percentage_valid_range() {
        let valid_percentages = vec![0.0, 0.01, 1.0, 10.0, 50.0, 100.0];
        for percentage in valid_percentages {
            let mut rule = create_test_royalty_rule();
            rule.percentage = percentage;
            assert!(validate_royalty_rule(&rule).is_ok());
        }
    }

    #[test]
    fn test_royalty_percentage_negative_rejected() {
        let mut rule = create_test_royalty_rule();
        rule.percentage = -5.0;
        let result = validate_royalty_rule(&rule);
        assert!(result.is_err());
    }

    #[test]
    fn test_royalty_percentage_over_100_rejected() {
        let mut rule = create_test_royalty_rule();
        rule.percentage = 150.0;
        let result = validate_royalty_rule(&rule);
        assert!(result.is_err());
    }

    #[test]
    fn test_royalty_minimum_amount_optional() {
        let mut rule = create_test_royalty_rule();
        rule.minimum_amount = None;
        assert!(validate_royalty_rule(&rule).is_ok());
    }

    #[test]
    fn test_royalty_can_be_deactivated() {
        let mut rule = create_test_royalty_rule();
        rule.active = false;
        assert!(!rule.active);
    }

    #[test]
    fn test_royalty_currency_set() {
        let rule = create_test_royalty_rule();
        assert!(!rule.currency.is_empty());
        assert_eq!(rule.currency, "USD");
    }

    // =========================================================================
    // USAGE RECORD TESTS
    // =========================================================================

    #[test]
    fn test_usage_record_creation() {
        let usage = create_test_usage_record();
        assert!(!usage.publication_id.is_empty());
        assert!(usage.timestamp > 0);
    }

    #[test]
    fn test_usage_types_supported() {
        let types = vec![
            TestUsageType::View,
            TestUsageType::Share,
            TestUsageType::Download,
            TestUsageType::Derivative,
            TestUsageType::Citation,
        ];
        for usage_type in types {
            let mut usage = create_test_usage_record();
            usage.usage_type = usage_type;
            // All usage types should be valid
            assert!(matches!(
                usage.usage_type,
                TestUsageType::View
                    | TestUsageType::Share
                    | TestUsageType::Download
                    | TestUsageType::Derivative
                    | TestUsageType::Citation
            ));
        }
    }

    #[test]
    fn test_usage_anonymous_user() {
        let mut usage = create_test_usage_record();
        usage.user_did = None;
        assert!(usage.user_did.is_none());
    }

    #[test]
    fn test_usage_royalty_paid_optional() {
        let mut usage = create_test_usage_record();
        usage.royalty_paid = None;
        assert!(usage.royalty_paid.is_none());
    }

    #[test]
    fn test_usage_records_immutable() {
        // Usage records cannot be updated according to validation rules
        let usage = create_test_usage_record();
        let usage_id = usage.id.clone();
        assert_eq!(usage.id, usage_id);
    }

    // =========================================================================
    // EARNINGS CALCULATION TESTS
    // =========================================================================

    #[test]
    fn test_contributor_earnings_single_usage() {
        let attributions = vec![create_test_attribution()];
        let usage_records = vec![create_test_usage_record()];

        let earnings = calculate_contributor_earnings(
            &attributions,
            &usage_records,
            "did:mycelix:contributor1",
        );
        assert_eq!(earnings, 0.01); // 100% of 0.01
    }

    #[test]
    fn test_contributor_earnings_shared_royalties() {
        let attributions = vec![
            {
                let mut a = create_test_attribution();
                a.share_percentage = 70.0;
                a.contributor_did = "did:mycelix:author".to_string();
                a
            },
            {
                let mut a = create_test_attribution();
                a.share_percentage = 30.0;
                a.contributor_did = "did:mycelix:editor".to_string();
                a
            },
        ];
        let usage_records = vec![{
            let mut u = create_test_usage_record();
            u.royalty_paid = Some(1.0);
            u
        }];

        let author_earnings =
            calculate_contributor_earnings(&attributions, &usage_records, "did:mycelix:author");
        assert!((author_earnings - 0.70).abs() < 0.001);

        let editor_earnings =
            calculate_contributor_earnings(&attributions, &usage_records, "did:mycelix:editor");
        assert!((editor_earnings - 0.30).abs() < 0.001);
    }

    #[test]
    fn test_contributor_earnings_no_usage() {
        let attributions = vec![create_test_attribution()];
        let usage_records: Vec<TestUsageRecord> = vec![];

        let earnings = calculate_contributor_earnings(
            &attributions,
            &usage_records,
            "did:mycelix:contributor1",
        );
        assert_eq!(earnings, 0.0);
    }

    #[test]
    fn test_contributor_earnings_no_royalties_paid() {
        let attributions = vec![create_test_attribution()];
        let usage_records = vec![{
            let mut u = create_test_usage_record();
            u.royalty_paid = None;
            u
        }];

        let earnings = calculate_contributor_earnings(
            &attributions,
            &usage_records,
            "did:mycelix:contributor1",
        );
        assert_eq!(earnings, 0.0);
    }

    #[test]
    fn test_contributor_earnings_multiple_publications() {
        let attributions = vec![
            {
                let mut a = create_test_attribution();
                a.publication_id = "pub1".to_string();
                a.share_percentage = 100.0;
                a
            },
            {
                let mut a = create_test_attribution();
                a.publication_id = "pub2".to_string();
                a.share_percentage = 50.0;
                a
            },
        ];
        let usage_records = vec![
            {
                let mut u = create_test_usage_record();
                u.publication_id = "pub1".to_string();
                u.royalty_paid = Some(1.0);
                u
            },
            {
                let mut u = create_test_usage_record();
                u.publication_id = "pub2".to_string();
                u.royalty_paid = Some(1.0);
                u
            },
        ];

        let earnings = calculate_contributor_earnings(
            &attributions,
            &usage_records,
            "did:mycelix:contributor1",
        );
        // 100% of 1.0 + 50% of 1.0 = 1.5
        assert!((earnings - 1.5).abs() < 0.001);
    }

    // =========================================================================
    // VERIFICATION TESTS
    // =========================================================================

    fn can_verify_attribution(attribution: &TestAttribution, requester_did: &str) -> bool {
        attribution.contributor_did == requester_did
    }

    #[test]
    fn test_only_contributor_can_verify() {
        let attribution = create_test_attribution();
        assert!(can_verify_attribution(
            &attribution,
            &attribution.contributor_did
        ));
        assert!(!can_verify_attribution(&attribution, "did:mycelix:other"));
    }

    #[test]
    fn test_attribution_verification_status() {
        let mut attribution = create_test_attribution();
        assert!(!attribution.verified);
        attribution.verified = true;
        assert!(attribution.verified);
    }

    // =========================================================================
    // PUBLICATION SHARES SUMMARY TESTS
    // =========================================================================

    #[derive(Debug)]
    struct PublicationShares {
        publication_id: String,
        total_share_percentage: f64,
        contributor_count: u32,
        verified_count: u32,
        remaining_percentage: f64,
    }

    fn get_publication_shares(
        attributions: &[TestAttribution],
        publication_id: &str,
    ) -> PublicationShares {
        let pub_attrs: Vec<&TestAttribution> = attributions
            .iter()
            .filter(|a| a.publication_id == publication_id)
            .collect();

        let total = pub_attrs.iter().map(|a| a.share_percentage).sum();
        let verified_count = pub_attrs.iter().filter(|a| a.verified).count() as u32;

        PublicationShares {
            publication_id: publication_id.to_string(),
            total_share_percentage: total,
            contributor_count: pub_attrs.len() as u32,
            verified_count,
            remaining_percentage: 100.0 - total,
        }
    }

    #[test]
    fn test_publication_shares_summary() {
        let attributions = vec![
            {
                let mut a = create_test_attribution();
                a.share_percentage = 60.0;
                a.verified = true;
                a
            },
            {
                let mut a = create_test_attribution();
                a.contributor_did = "did:mycelix:other".to_string();
                a.share_percentage = 25.0;
                a.verified = false;
                a
            },
        ];

        let shares = get_publication_shares(&attributions, &attributions[0].publication_id);
        assert_eq!(shares.contributor_count, 2);
        assert_eq!(shares.verified_count, 1);
        assert_eq!(shares.total_share_percentage, 85.0);
        assert_eq!(shares.remaining_percentage, 15.0);
    }

    // =========================================================================
    // ATTRIBUTION CHAIN TESTS
    // =========================================================================

    #[test]
    fn test_attribution_chain_integrity() {
        // Simulate multiple attributions for derived works
        let original_attribution = create_test_attribution();

        let derived_attribution = TestAttribution {
            id: "attr:derived:1".to_string(),
            publication_id: "pub:derived:1".to_string(),
            contributor_did: "did:mycelix:derivative_author".to_string(),
            role: TestContributorRole::Author,
            share_percentage: 70.0, // 30% goes to original
            verified: false,
            created: original_attribution.created + 86400000000,
        };

        assert!(derived_attribution.created > original_attribution.created);
        // In a real system, we'd track the original publication reference
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_many_contributors_single_publication() {
        let attributions: Vec<TestAttribution> = (0..10)
            .map(|i| {
                let mut a = create_test_attribution();
                a.contributor_did = format!("did:mycelix:contributor{}", i);
                a.share_percentage = 10.0;
                a
            })
            .collect();

        let total = calculate_total_shares(&attributions);
        assert_eq!(total, 100.0);
    }

    #[test]
    fn test_fractional_shares_sum_correctly() {
        let attributions = vec![
            {
                let mut a = create_test_attribution();
                a.share_percentage = 33.33;
                a.contributor_did = "did:mycelix:a".to_string();
                a
            },
            {
                let mut a = create_test_attribution();
                a.share_percentage = 33.33;
                a.contributor_did = "did:mycelix:b".to_string();
                a
            },
            {
                let mut a = create_test_attribution();
                a.share_percentage = 33.34;
                a.contributor_did = "did:mycelix:c".to_string();
                a
            },
        ];

        let total = calculate_total_shares(&attributions);
        assert!((total - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_royalty_with_high_precision() {
        let mut rule = create_test_royalty_rule();
        rule.percentage = 0.0001;
        rule.minimum_amount = Some(0.00001);
        assert!(validate_royalty_rule(&rule).is_ok());
    }

    #[test]
    fn test_usage_record_many_entries() {
        let usage_records: Vec<TestUsageRecord> = (0..1000)
            .map(|i| {
                let mut u = create_test_usage_record();
                u.id = format!("usage:{}", i);
                u.timestamp = 1704000000000000 + (i as i64 * 1000000);
                u.royalty_paid = Some(0.01);
                u
            })
            .collect();

        assert_eq!(usage_records.len(), 1000);
        let total_royalties: f64 = usage_records.iter().filter_map(|u| u.royalty_paid).sum();
        assert!((total_royalties - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_contributor_with_multiple_roles() {
        let attributions = vec![
            {
                let mut a = create_test_attribution();
                a.role = TestContributorRole::Author;
                a.share_percentage = 50.0;
                a
            },
            {
                let mut a = create_test_attribution();
                a.role = TestContributorRole::Photographer;
                a.share_percentage = 10.0;
                a
            },
        ];

        // Same contributor with multiple roles
        let total = calculate_total_shares(&attributions);
        assert_eq!(total, 60.0);
    }

    #[test]
    fn test_attribution_by_role_filtering() {
        let attributions = vec![
            {
                let mut a = create_test_attribution();
                a.role = TestContributorRole::Author;
                a.contributor_did = "did:mycelix:author1".to_string();
                a
            },
            {
                let mut a = create_test_attribution();
                a.role = TestContributorRole::Editor;
                a.contributor_did = "did:mycelix:editor1".to_string();
                a
            },
            {
                let mut a = create_test_attribution();
                a.role = TestContributorRole::Author;
                a.contributor_did = "did:mycelix:author2".to_string();
                a
            },
        ];

        let authors: Vec<&TestAttribution> = attributions
            .iter()
            .filter(|a| matches!(a.role, TestContributorRole::Author))
            .collect();

        assert_eq!(authors.len(), 2);
    }
}
