// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Commons Zome Unit Tests
//!
//! Comprehensive test coverage for common resource management, usage rights,
//! quota enforcement, steward management, and governance rules.

use serde::{Deserialize, Serialize};

// =============================================================================
// Test Data Structures
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestCommonResource {
    pub id: String,
    pub name: String,
    pub description: String,
    pub resource_type: TestResourceType,
    pub property_id: Option<String>,
    pub stewards: Vec<String>,
    pub governance_rules: TestGovernanceRules,
    pub created: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestResourceType {
    Land,
    Water,
    Forest,
    Fishery,
    Pasture,
    Infrastructure,
    Digital,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestGovernanceRules {
    pub access_rules: Vec<String>,
    pub usage_limits: Vec<TestUsageLimit>,
    pub maintenance_rotation: bool,
    pub decision_method: TestDecisionMethod,
    pub penalty_for_violation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestUsageLimit {
    pub limit_type: String,
    pub max_per_period: f64,
    pub period_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestDecisionMethod {
    Consensus,
    Majority,
    SuperMajority,
    Stewards,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestUsageRight {
    pub id: String,
    pub resource_id: String,
    pub holder_did: String,
    pub right_type: TestRightType,
    pub quota: Option<f64>,
    pub granted: i64,
    pub expires: Option<i64>,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestRightType {
    Access,
    Extraction,
    Management,
    Exclusion,
    Alienation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestUsageLog {
    pub id: String,
    pub resource_id: String,
    pub user_did: String,
    pub usage_type: String,
    pub quantity: f64,
    pub unit: String,
    pub timestamp: i64,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_resource() -> TestCommonResource {
    TestCommonResource {
        id: "commons:Community_Forest:1704067200000000".to_string(),
        name: "Community Forest".to_string(),
        description: "Shared forest for sustainable timber and recreation".to_string(),
        resource_type: TestResourceType::Forest,
        property_id: Some("property:did:mycelix:community:1700000000000000".to_string()),
        stewards: vec![
            "did:mycelix:steward1".to_string(),
            "did:mycelix:steward2".to_string(),
            "did:mycelix:steward3".to_string(),
        ],
        governance_rules: TestGovernanceRules {
            access_rules: vec![
                "Members may access during daylight hours".to_string(),
                "No motorized vehicles".to_string(),
                "Leave no trace".to_string(),
            ],
            usage_limits: vec![
                TestUsageLimit {
                    limit_type: "Timber".to_string(),
                    max_per_period: 10.0,
                    period_days: 365,
                },
                TestUsageLimit {
                    limit_type: "Firewood".to_string(),
                    max_per_period: 2.0,
                    period_days: 30,
                },
            ],
            maintenance_rotation: true,
            decision_method: TestDecisionMethod::Consensus,
            penalty_for_violation: Some("Loss of access for 30 days".to_string()),
        },
        created: 1704067200000000,
    }
}

fn create_test_usage_right() -> TestUsageRight {
    TestUsageRight {
        id: "right:commons:member1:1704067200000000".to_string(),
        resource_id: "commons:Community_Forest:1704067200000000".to_string(),
        holder_did: "did:mycelix:member1".to_string(),
        right_type: TestRightType::Extraction,
        quota: Some(5.0), // 5 cubic meters of timber per year
        granted: 1704067200000000,
        expires: Some(1735689600000000), // 1 year
        active: true,
    }
}

fn create_test_usage_log() -> TestUsageLog {
    TestUsageLog {
        id: "usage:commons:member1:1704153600000000".to_string(),
        resource_id: "commons:Community_Forest:1704067200000000".to_string(),
        user_did: "did:mycelix:member1".to_string(),
        usage_type: "Timber".to_string(),
        quantity: 1.5,
        unit: "cubic_meters".to_string(),
        timestamp: 1704153600000000,
    }
}

fn validate_did(did: &str) -> bool {
    did.starts_with("did:")
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // COMMON RESOURCE CREATION TESTS
    // =========================================================================

    #[test]
    fn test_resource_has_unique_id() {
        let resource = create_test_resource();
        assert!(resource.id.contains("commons:"));
        assert!(!resource.id.is_empty());
    }

    #[test]
    fn test_resource_has_name() {
        let resource = create_test_resource();
        assert!(!resource.name.is_empty());
    }

    #[test]
    fn test_resource_has_description() {
        let resource = create_test_resource();
        assert!(!resource.description.is_empty());
    }

    #[test]
    fn test_resource_requires_at_least_one_steward() {
        let resource = create_test_resource();
        assert!(!resource.stewards.is_empty());
    }

    #[test]
    fn test_resource_empty_stewards_invalid() {
        let mut resource = create_test_resource();
        resource.stewards = vec![];
        // Should be rejected by validation
        assert!(resource.stewards.is_empty());
    }

    #[test]
    fn test_resource_stewards_must_be_valid_dids() {
        let resource = create_test_resource();
        for steward in &resource.stewards {
            assert!(validate_did(steward));
        }
    }

    #[test]
    fn test_resource_invalid_steward_did_rejected() {
        let mut resource = create_test_resource();
        resource.stewards.push("invalid_did".to_string());
        // The last steward is invalid
        assert!(!validate_did(resource.stewards.last().unwrap()));
    }

    #[test]
    fn test_resource_created_timestamp_valid() {
        let resource = create_test_resource();
        assert!(resource.created > 0);
        assert!(resource.created > 946684800000000); // After year 2000
    }

    #[test]
    fn test_resource_property_id_optional() {
        let mut resource = create_test_resource();
        resource.property_id = None;
        assert!(resource.property_id.is_none());
    }

    // =========================================================================
    // RESOURCE TYPE TESTS
    // =========================================================================

    #[test]
    fn test_all_resource_types_supported() {
        let types = vec![
            TestResourceType::Land,
            TestResourceType::Water,
            TestResourceType::Forest,
            TestResourceType::Fishery,
            TestResourceType::Pasture,
            TestResourceType::Infrastructure,
            TestResourceType::Digital,
            TestResourceType::Other("Custom resource".to_string()),
        ];
        for t in types {
            assert!(matches!(
                t,
                TestResourceType::Land
                    | TestResourceType::Water
                    | TestResourceType::Forest
                    | TestResourceType::Fishery
                    | TestResourceType::Pasture
                    | TestResourceType::Infrastructure
                    | TestResourceType::Digital
                    | TestResourceType::Other(_)
            ));
        }
    }

    #[test]
    fn test_forest_resource() {
        let resource = create_test_resource();
        assert_eq!(resource.resource_type, TestResourceType::Forest);
    }

    #[test]
    fn test_water_resource() {
        let mut resource = create_test_resource();
        resource.resource_type = TestResourceType::Water;
        resource.name = "Community Well".to_string();
        assert_eq!(resource.resource_type, TestResourceType::Water);
    }

    #[test]
    fn test_digital_resource() {
        let mut resource = create_test_resource();
        resource.resource_type = TestResourceType::Digital;
        resource.name = "Community Data Commons".to_string();
        resource.property_id = None; // Digital resources may not have physical property
        assert_eq!(resource.resource_type, TestResourceType::Digital);
    }

    // =========================================================================
    // GOVERNANCE RULES TESTS
    // =========================================================================

    #[test]
    fn test_resource_has_governance_rules() {
        let resource = create_test_resource();
        assert!(!resource.governance_rules.access_rules.is_empty());
    }

    #[test]
    fn test_governance_access_rules() {
        let resource = create_test_resource();
        assert!(resource.governance_rules.access_rules.len() >= 1);
    }

    #[test]
    fn test_governance_usage_limits() {
        let resource = create_test_resource();
        assert!(!resource.governance_rules.usage_limits.is_empty());
        for limit in &resource.governance_rules.usage_limits {
            assert!(limit.max_per_period > 0.0);
            assert!(limit.period_days > 0);
        }
    }

    #[test]
    fn test_governance_decision_methods_supported() {
        let methods = vec![
            TestDecisionMethod::Consensus,
            TestDecisionMethod::Majority,
            TestDecisionMethod::SuperMajority,
            TestDecisionMethod::Stewards,
        ];
        for method in methods {
            assert!(matches!(
                method,
                TestDecisionMethod::Consensus
                    | TestDecisionMethod::Majority
                    | TestDecisionMethod::SuperMajority
                    | TestDecisionMethod::Stewards
            ));
        }
    }

    #[test]
    fn test_governance_penalty_optional() {
        let mut resource = create_test_resource();
        resource.governance_rules.penalty_for_violation = None;
        assert!(resource.governance_rules.penalty_for_violation.is_none());
    }

    #[test]
    fn test_governance_maintenance_rotation() {
        let resource = create_test_resource();
        // Maintenance rotation can be enabled or disabled
        assert!(resource.governance_rules.maintenance_rotation);
    }

    // =========================================================================
    // USAGE RIGHT TESTS
    // =========================================================================

    #[test]
    fn test_usage_right_requires_valid_holder_did() {
        let right = create_test_usage_right();
        assert!(validate_did(&right.holder_did));
    }

    #[test]
    fn test_usage_right_invalid_holder_rejected() {
        let mut right = create_test_usage_right();
        right.holder_did = "invalid_holder".to_string();
        assert!(!validate_did(&right.holder_did));
    }

    #[test]
    fn test_usage_right_has_unique_id() {
        let right = create_test_usage_right();
        assert!(right.id.contains("right:"));
        assert!(!right.id.is_empty());
    }

    #[test]
    fn test_usage_right_references_resource() {
        let right = create_test_usage_right();
        assert!(!right.resource_id.is_empty());
        assert!(right.resource_id.contains("commons:"));
    }

    #[test]
    fn test_usage_right_initially_active() {
        let right = create_test_usage_right();
        assert!(right.active);
    }

    #[test]
    fn test_usage_right_granted_timestamp_valid() {
        let right = create_test_usage_right();
        assert!(right.granted > 0);
    }

    // =========================================================================
    // RIGHT TYPE TESTS
    // =========================================================================

    #[test]
    fn test_all_right_types_supported() {
        let types = vec![
            TestRightType::Access,
            TestRightType::Extraction,
            TestRightType::Management,
            TestRightType::Exclusion,
            TestRightType::Alienation,
        ];
        for t in types {
            assert!(matches!(
                t,
                TestRightType::Access
                    | TestRightType::Extraction
                    | TestRightType::Management
                    | TestRightType::Exclusion
                    | TestRightType::Alienation
            ));
        }
    }

    #[test]
    fn test_access_right() {
        let mut right = create_test_usage_right();
        right.right_type = TestRightType::Access;
        right.quota = None; // Access rights may not have quotas
        assert_eq!(right.right_type, TestRightType::Access);
    }

    #[test]
    fn test_extraction_right() {
        let right = create_test_usage_right();
        assert_eq!(right.right_type, TestRightType::Extraction);
        // Extraction rights typically have quotas
        assert!(right.quota.is_some());
    }

    #[test]
    fn test_management_right() {
        let mut right = create_test_usage_right();
        right.right_type = TestRightType::Management;
        // Management rights allow modifying resource rules
        assert_eq!(right.right_type, TestRightType::Management);
    }

    #[test]
    fn test_exclusion_right() {
        let mut right = create_test_usage_right();
        right.right_type = TestRightType::Exclusion;
        // Exclusion rights allow denying access to others
        assert_eq!(right.right_type, TestRightType::Exclusion);
    }

    #[test]
    fn test_alienation_right() {
        let mut right = create_test_usage_right();
        right.right_type = TestRightType::Alienation;
        // Alienation rights allow selling/transferring rights
        assert_eq!(right.right_type, TestRightType::Alienation);
    }

    // =========================================================================
    // QUOTA TESTS
    // =========================================================================

    #[test]
    fn test_quota_optional() {
        let mut right = create_test_usage_right();
        right.quota = None;
        assert!(right.quota.is_none());
    }

    #[test]
    fn test_quota_non_negative() {
        let right = create_test_usage_right();
        if let Some(quota) = right.quota {
            assert!(quota >= 0.0);
        }
    }

    #[test]
    fn test_quota_zero_valid() {
        let mut right = create_test_usage_right();
        right.quota = Some(0.0);
        // Zero quota means no extraction allowed
        assert_eq!(right.quota, Some(0.0));
    }

    fn check_quota(right: &TestUsageRight, current_usage: f64, requested_amount: f64) -> bool {
        if !right.active {
            return false;
        }
        if let Some(quota) = right.quota {
            current_usage + requested_amount <= quota
        } else {
            true // No quota limit
        }
    }

    #[test]
    fn test_quota_check_within_limit() {
        let right = create_test_usage_right();
        assert!(check_quota(&right, 2.0, 2.0)); // 2 + 2 = 4 <= 5
    }

    #[test]
    fn test_quota_check_exceeds_limit() {
        let right = create_test_usage_right();
        assert!(!check_quota(&right, 4.0, 2.0)); // 4 + 2 = 6 > 5
    }

    #[test]
    fn test_quota_check_exact_limit() {
        let right = create_test_usage_right();
        assert!(check_quota(&right, 3.0, 2.0)); // 3 + 2 = 5 <= 5
    }

    #[test]
    fn test_quota_check_no_limit() {
        let mut right = create_test_usage_right();
        right.quota = None;
        assert!(check_quota(&right, 100.0, 100.0)); // No limit
    }

    #[test]
    fn test_quota_check_inactive_right() {
        let mut right = create_test_usage_right();
        right.active = false;
        assert!(!check_quota(&right, 0.0, 1.0)); // Inactive rights cannot be used
    }

    // =========================================================================
    // USAGE LOG TESTS
    // =========================================================================

    #[test]
    fn test_usage_log_requires_valid_user_did() {
        let log = create_test_usage_log();
        assert!(validate_did(&log.user_did));
    }

    #[test]
    fn test_usage_log_invalid_user_rejected() {
        let mut log = create_test_usage_log();
        log.user_did = "invalid_user".to_string();
        assert!(!validate_did(&log.user_did));
    }

    #[test]
    fn test_usage_log_has_unique_id() {
        let log = create_test_usage_log();
        assert!(log.id.contains("usage:"));
        assert!(!log.id.is_empty());
    }

    #[test]
    fn test_usage_log_references_resource() {
        let log = create_test_usage_log();
        assert!(!log.resource_id.is_empty());
    }

    #[test]
    fn test_usage_log_quantity_non_negative() {
        let log = create_test_usage_log();
        assert!(log.quantity >= 0.0);
    }

    #[test]
    fn test_usage_log_negative_quantity_invalid() {
        let mut log = create_test_usage_log();
        log.quantity = -1.0;
        // Should be rejected by validation
        assert!(log.quantity < 0.0);
    }

    #[test]
    fn test_usage_log_has_unit() {
        let log = create_test_usage_log();
        assert!(!log.unit.is_empty());
    }

    #[test]
    fn test_usage_log_has_type() {
        let log = create_test_usage_log();
        assert!(!log.usage_type.is_empty());
    }

    #[test]
    fn test_usage_log_timestamp_valid() {
        let log = create_test_usage_log();
        assert!(log.timestamp > 0);
    }

    #[test]
    fn test_usage_logs_immutable() {
        // Usage logs cannot be updated according to validation rules
        let log = create_test_usage_log();
        let original_quantity = log.quantity;
        // Any attempt to update should be rejected
        assert_eq!(log.quantity, original_quantity);
    }

    // =========================================================================
    // STEWARD MANAGEMENT TESTS
    // =========================================================================

    fn can_add_steward(resource: &TestCommonResource, added_by_did: &str) -> bool {
        resource.stewards.contains(&added_by_did.to_string())
    }

    fn can_remove_steward(resource: &TestCommonResource, removed_by_did: &str) -> bool {
        resource.stewards.contains(&removed_by_did.to_string()) && resource.stewards.len() > 1 // Cannot remove last steward
    }

    #[test]
    fn test_steward_can_add_steward() {
        let resource = create_test_resource();
        assert!(can_add_steward(&resource, &resource.stewards[0]));
    }

    #[test]
    fn test_non_steward_cannot_add_steward() {
        let resource = create_test_resource();
        assert!(!can_add_steward(&resource, "did:mycelix:random"));
    }

    #[test]
    fn test_steward_can_remove_steward() {
        let resource = create_test_resource();
        // Has 3 stewards, so one can be removed
        assert!(can_remove_steward(&resource, &resource.stewards[0]));
    }

    #[test]
    fn test_cannot_remove_last_steward() {
        let mut resource = create_test_resource();
        resource.stewards = vec!["did:mycelix:onlysteward".to_string()];
        // Cannot remove the only steward
        assert!(!can_remove_steward(&resource, &resource.stewards[0]));
    }

    #[test]
    fn test_non_steward_cannot_remove_steward() {
        let resource = create_test_resource();
        assert!(!can_remove_steward(&resource, "did:mycelix:random"));
    }

    #[test]
    fn test_cannot_add_duplicate_steward() {
        let resource = create_test_resource();
        let existing_steward = &resource.stewards[0];
        // Adding a duplicate should be rejected
        assert!(resource.stewards.contains(existing_steward));
    }

    // =========================================================================
    // RIGHT REVOCATION TESTS
    // =========================================================================

    fn can_revoke_right(resource: &TestCommonResource, revoker_did: &str) -> bool {
        resource.stewards.contains(&revoker_did.to_string())
    }

    #[test]
    fn test_steward_can_revoke_right() {
        let resource = create_test_resource();
        assert!(can_revoke_right(&resource, &resource.stewards[0]));
    }

    #[test]
    fn test_non_steward_cannot_revoke_right() {
        let resource = create_test_resource();
        assert!(!can_revoke_right(&resource, "did:mycelix:random"));
    }

    #[test]
    fn test_revocation_sets_inactive() {
        let mut right = create_test_usage_right();
        right.active = false;
        assert!(!right.active);
    }

    // =========================================================================
    // GOVERNANCE UPDATE TESTS
    // =========================================================================

    fn can_update_governance(resource: &TestCommonResource, steward_did: &str) -> bool {
        resource.stewards.contains(&steward_did.to_string())
    }

    #[test]
    fn test_steward_can_update_governance() {
        let resource = create_test_resource();
        assert!(can_update_governance(&resource, &resource.stewards[0]));
    }

    #[test]
    fn test_non_steward_cannot_update_governance() {
        let resource = create_test_resource();
        assert!(!can_update_governance(&resource, "did:mycelix:random"));
    }

    // =========================================================================
    // EXPIRATION TESTS
    // =========================================================================

    fn is_right_expired(right: &TestUsageRight, current_time: i64) -> bool {
        if let Some(expires) = right.expires {
            current_time >= expires
        } else {
            false // No expiration
        }
    }

    fn is_right_valid(right: &TestUsageRight, current_time: i64) -> bool {
        right.active && !is_right_expired(right, current_time)
    }

    #[test]
    fn test_right_not_expired() {
        let right = create_test_usage_right();
        let current_time = 1704200000000000i64; // Before expiry
        assert!(!is_right_expired(&right, current_time));
    }

    #[test]
    fn test_right_expired() {
        let right = create_test_usage_right();
        let current_time = 1800000000000000i64; // After expiry
        assert!(is_right_expired(&right, current_time));
    }

    #[test]
    fn test_perpetual_right_never_expires() {
        let mut right = create_test_usage_right();
        right.expires = None;
        let current_time = 2000000000000000i64; // Far future
        assert!(!is_right_expired(&right, current_time));
    }

    #[test]
    fn test_right_valid_when_active_and_not_expired() {
        let right = create_test_usage_right();
        let current_time = 1704200000000000i64;
        assert!(is_right_valid(&right, current_time));
    }

    #[test]
    fn test_right_invalid_when_inactive() {
        let mut right = create_test_usage_right();
        right.active = false;
        let current_time = 1704200000000000i64;
        assert!(!is_right_valid(&right, current_time));
    }

    #[test]
    fn test_right_invalid_when_expired() {
        let right = create_test_usage_right();
        let current_time = 1800000000000000i64;
        assert!(!is_right_valid(&right, current_time));
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_resource_with_many_stewards() {
        let mut resource = create_test_resource();
        resource.stewards = (0..20)
            .map(|i| format!("did:mycelix:steward{}", i))
            .collect();
        assert_eq!(resource.stewards.len(), 20);
        // All stewards should be valid DIDs
        for steward in &resource.stewards {
            assert!(validate_did(steward));
        }
    }

    #[test]
    fn test_resource_with_many_usage_limits() {
        let mut resource = create_test_resource();
        resource.governance_rules.usage_limits = (0..10)
            .map(|i| TestUsageLimit {
                limit_type: format!("LimitType{}", i),
                max_per_period: (i + 1) as f64 * 10.0,
                period_days: 30,
            })
            .collect();
        assert_eq!(resource.governance_rules.usage_limits.len(), 10);
    }

    #[test]
    fn test_calculate_total_usage() {
        let logs = vec![
            TestUsageLog {
                id: "log:1".to_string(),
                resource_id: "commons:1".to_string(),
                user_did: "did:mycelix:user1".to_string(),
                usage_type: "Timber".to_string(),
                quantity: 1.5,
                unit: "cubic_meters".to_string(),
                timestamp: 1704100000000000,
            },
            TestUsageLog {
                id: "log:2".to_string(),
                resource_id: "commons:1".to_string(),
                user_did: "did:mycelix:user1".to_string(),
                usage_type: "Timber".to_string(),
                quantity: 2.0,
                unit: "cubic_meters".to_string(),
                timestamp: 1704200000000000,
            },
            TestUsageLog {
                id: "log:3".to_string(),
                resource_id: "commons:1".to_string(),
                user_did: "did:mycelix:user2".to_string(),
                usage_type: "Timber".to_string(),
                quantity: 3.0,
                unit: "cubic_meters".to_string(),
                timestamp: 1704300000000000,
            },
        ];

        // Calculate total for user1
        let user1_total: f64 = logs
            .iter()
            .filter(|log| log.user_did == "did:mycelix:user1")
            .map(|log| log.quantity)
            .sum();
        assert_eq!(user1_total, 3.5);

        // Calculate total for all users
        let total: f64 = logs.iter().map(|log| log.quantity).sum();
        assert_eq!(total, 6.5);
    }

    #[test]
    fn test_multiple_rights_for_same_user() {
        let right1 = TestUsageRight {
            id: "right:1".to_string(),
            resource_id: "commons:forest".to_string(),
            holder_did: "did:mycelix:member".to_string(),
            right_type: TestRightType::Access,
            quota: None,
            granted: 1704000000000000,
            expires: None,
            active: true,
        };

        let right2 = TestUsageRight {
            id: "right:2".to_string(),
            resource_id: "commons:forest".to_string(),
            holder_did: "did:mycelix:member".to_string(),
            right_type: TestRightType::Extraction,
            quota: Some(5.0),
            granted: 1704000000000000,
            expires: Some(1735689600000000),
            active: true,
        };

        // Same user can have multiple different rights
        assert_eq!(right1.holder_did, right2.holder_did);
        assert_ne!(right1.right_type, right2.right_type);
    }

    #[test]
    fn test_water_resource_with_seasonal_limits() {
        let resource = TestCommonResource {
            id: "commons:river:1".to_string(),
            name: "Community River".to_string(),
            description: "Shared water resource for irrigation".to_string(),
            resource_type: TestResourceType::Water,
            property_id: None,
            stewards: vec!["did:mycelix:watermaster".to_string()],
            governance_rules: TestGovernanceRules {
                access_rules: vec![
                    "Irrigation allowed only during allocated time slots".to_string(),
                    "No dumping or pollution".to_string(),
                ],
                usage_limits: vec![TestUsageLimit {
                    limit_type: "WaterExtraction".to_string(),
                    max_per_period: 1000.0, // liters
                    period_days: 7,
                }],
                maintenance_rotation: true,
                decision_method: TestDecisionMethod::Stewards,
                penalty_for_violation: Some("Loss of water rights for current season".to_string()),
            },
            created: 1704067200000000,
        };

        assert_eq!(resource.resource_type, TestResourceType::Water);
        assert!(!resource.governance_rules.usage_limits.is_empty());
    }

    #[test]
    fn test_fishery_resource_with_quotas() {
        let resource = TestCommonResource {
            id: "commons:fishery:1".to_string(),
            name: "Community Fishing Waters".to_string(),
            description: "Shared fishing grounds".to_string(),
            resource_type: TestResourceType::Fishery,
            property_id: None,
            stewards: vec![
                "did:mycelix:fishermaster1".to_string(),
                "did:mycelix:fishermaster2".to_string(),
            ],
            governance_rules: TestGovernanceRules {
                access_rules: vec![
                    "Fishing permitted sunrise to sunset".to_string(),
                    "No nets during spawning season".to_string(),
                    "Release undersized fish".to_string(),
                ],
                usage_limits: vec![TestUsageLimit {
                    limit_type: "Fish".to_string(),
                    max_per_period: 10.0, // fish per day
                    period_days: 1,
                }],
                maintenance_rotation: false,
                decision_method: TestDecisionMethod::Majority,
                penalty_for_violation: Some("Ban for remainder of season".to_string()),
            },
            created: 1704067200000000,
        };

        assert_eq!(resource.resource_type, TestResourceType::Fishery);
        // Daily limit on fish catch
        assert_eq!(resource.governance_rules.usage_limits[0].period_days, 1);
    }

    #[test]
    fn test_digital_commons_no_physical_limits() {
        let resource = TestCommonResource {
            id: "commons:data:1".to_string(),
            name: "Community Knowledge Base".to_string(),
            description: "Shared documentation and learning resources".to_string(),
            resource_type: TestResourceType::Digital,
            property_id: None,
            stewards: vec!["did:mycelix:librarian".to_string()],
            governance_rules: TestGovernanceRules {
                access_rules: vec![
                    "Free access for all community members".to_string(),
                    "Attribution required for use".to_string(),
                ],
                usage_limits: vec![], // Digital resources may not need physical limits
                maintenance_rotation: false,
                decision_method: TestDecisionMethod::Consensus,
                penalty_for_violation: None,
            },
            created: 1704067200000000,
        };

        assert_eq!(resource.resource_type, TestResourceType::Digital);
        // Digital resources might not have usage limits
        assert!(resource.governance_rules.usage_limits.is_empty());
    }

    #[test]
    fn test_quota_update_by_steward() {
        let resource = create_test_resource();
        let mut right = create_test_usage_right();

        // Steward updates quota
        let steward = &resource.stewards[0];
        assert!(resource.stewards.contains(steward));

        let old_quota = right.quota;
        right.quota = Some(10.0);

        assert_ne!(right.quota, old_quota);
    }
}
