// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Carbon Zome Unit Tests
//!
//! Comprehensive test coverage for carbon footprint tracking, carbon credit management,
//! status transitions, verification workflows, and access control.

use serde::{Deserialize, Serialize};

// =============================================================================
// Test Data Structures (mirrors integrity types for testing)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Copy)]
pub enum TestCreditStatus {
    Active,
    Transferred,
    Retired,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestCarbonFootprint {
    pub entity_did: String,
    pub period_start: i64,
    pub period_end: i64,
    pub scope1: f64,
    pub scope2: f64,
    pub scope3: f64,
    pub methodology: String,
    pub verified_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestCarbonCredit {
    pub id: String,
    pub project_id: String,
    pub vintage_year: u32,
    pub tonnes_co2e: f64,
    pub status: TestCreditStatus,
    pub owner_did: String,
    pub retired_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestCreditSummary {
    pub total_credits: u64,
    pub total_tonnes: f64,
    pub active_tonnes: f64,
    pub retired_tonnes: f64,
    pub transferred_count: u64,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_footprint() -> TestCarbonFootprint {
    TestCarbonFootprint {
        entity_did: "did:mycelix:org123".to_string(),
        period_start: 1704067200, // 2024-01-01
        period_end: 1735689599,   // 2024-12-31
        scope1: 150.5,
        scope2: 75.2,
        scope3: 320.8,
        methodology: "GHG Protocol".to_string(),
        verified_by: None,
    }
}

fn create_test_credit() -> TestCarbonCredit {
    TestCarbonCredit {
        id: "credit:reforest-001:2024:001".to_string(),
        project_id: "project:reforest-001".to_string(),
        vintage_year: 2024,
        tonnes_co2e: 10.0,
        status: TestCreditStatus::Active,
        owner_did: "did:mycelix:holder123".to_string(),
        retired_at: None,
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

fn validate_emissions(scope1: f64, scope2: f64, scope3: f64) -> Result<(), String> {
    if scope1 < 0.0 {
        return Err("Scope 1 emissions cannot be negative".to_string());
    }
    if scope2 < 0.0 {
        return Err("Scope 2 emissions cannot be negative".to_string());
    }
    if scope3 < 0.0 {
        return Err("Scope 3 emissions cannot be negative".to_string());
    }
    Ok(())
}

fn validate_footprint(footprint: &TestCarbonFootprint) -> Result<(), String> {
    validate_did(&footprint.entity_did)?;

    if let Some(ref verifier) = footprint.verified_by {
        validate_did(verifier)?;
    }

    validate_emissions(footprint.scope1, footprint.scope2, footprint.scope3)?;

    if footprint.period_start >= footprint.period_end {
        return Err("Period start must be before period end".to_string());
    }

    if footprint.methodology.is_empty() {
        return Err("Methodology cannot be empty".to_string());
    }

    Ok(())
}

fn validate_credit(credit: &TestCarbonCredit) -> Result<(), String> {
    if credit.id.is_empty() {
        return Err("Credit ID cannot be empty".to_string());
    }

    if credit.project_id.is_empty() {
        return Err("Project ID cannot be empty".to_string());
    }

    validate_did(&credit.owner_did)?;

    if credit.tonnes_co2e <= 0.0 {
        return Err("Credit tonnes must be positive".to_string());
    }

    if credit.vintage_year < 1990 || credit.vintage_year > 2100 {
        return Err("Vintage year must be between 1990 and 2100".to_string());
    }

    // Validate retired_at matches status
    match credit.status {
        TestCreditStatus::Retired => {
            if credit.retired_at.is_none() {
                return Err("Retired credits must have retired_at timestamp".to_string());
            }
        }
        _ => {
            if credit.retired_at.is_some() {
                return Err("Non-retired credits cannot have retired_at timestamp".to_string());
            }
        }
    }

    Ok(())
}

fn calculate_total_emissions(footprint: &TestCarbonFootprint) -> f64 {
    footprint.scope1 + footprint.scope2 + footprint.scope3
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CARBON FOOTPRINT VALIDATION TESTS
    // =========================================================================

    #[test]
    fn test_footprint_requires_valid_entity_did() {
        let footprint = create_test_footprint();
        assert!(validate_did(&footprint.entity_did).is_ok());
        assert!(footprint.entity_did.starts_with("did:"));
    }

    #[test]
    fn test_footprint_empty_did_rejected() {
        let mut footprint = create_test_footprint();
        footprint.entity_did = "".to_string();
        let result = validate_footprint(&footprint);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "DID cannot be empty");
    }

    #[test]
    fn test_footprint_invalid_did_format_rejected() {
        let mut footprint = create_test_footprint();
        footprint.entity_did = "invalid_did".to_string();
        let result = validate_footprint(&footprint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("did:"));
    }

    #[test]
    fn test_footprint_valid_emissions() {
        let footprint = create_test_footprint();
        assert!(validate_emissions(footprint.scope1, footprint.scope2, footprint.scope3).is_ok());
        assert!(footprint.scope1 >= 0.0);
        assert!(footprint.scope2 >= 0.0);
        assert!(footprint.scope3 >= 0.0);
    }

    #[test]
    fn test_footprint_negative_scope1_rejected() {
        let mut footprint = create_test_footprint();
        footprint.scope1 = -10.0;
        let result = validate_footprint(&footprint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Scope 1"));
    }

    #[test]
    fn test_footprint_negative_scope2_rejected() {
        let mut footprint = create_test_footprint();
        footprint.scope2 = -5.0;
        let result = validate_footprint(&footprint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Scope 2"));
    }

    #[test]
    fn test_footprint_negative_scope3_rejected() {
        let mut footprint = create_test_footprint();
        footprint.scope3 = -100.0;
        let result = validate_footprint(&footprint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Scope 3"));
    }

    #[test]
    fn test_footprint_zero_emissions_valid() {
        let mut footprint = create_test_footprint();
        footprint.scope1 = 0.0;
        footprint.scope2 = 0.0;
        footprint.scope3 = 0.0;
        assert!(validate_footprint(&footprint).is_ok());
    }

    #[test]
    fn test_footprint_period_start_before_end() {
        let footprint = create_test_footprint();
        assert!(footprint.period_start < footprint.period_end);
        assert!(validate_footprint(&footprint).is_ok());
    }

    #[test]
    fn test_footprint_period_start_equals_end_rejected() {
        let mut footprint = create_test_footprint();
        footprint.period_end = footprint.period_start;
        let result = validate_footprint(&footprint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Period start"));
    }

    #[test]
    fn test_footprint_period_start_after_end_rejected() {
        let mut footprint = create_test_footprint();
        footprint.period_start = footprint.period_end + 1000;
        let result = validate_footprint(&footprint);
        assert!(result.is_err());
    }

    #[test]
    fn test_footprint_requires_methodology() {
        let footprint = create_test_footprint();
        assert!(!footprint.methodology.is_empty());
        assert!(validate_footprint(&footprint).is_ok());
    }

    #[test]
    fn test_footprint_empty_methodology_rejected() {
        let mut footprint = create_test_footprint();
        footprint.methodology = "".to_string();
        let result = validate_footprint(&footprint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Methodology"));
    }

    #[test]
    fn test_footprint_valid_methodologies() {
        let valid_methodologies = vec![
            "GHG Protocol",
            "ISO 14064",
            "PAS 2050",
            "Custom Methodology v1.0",
        ];
        for methodology in valid_methodologies {
            let mut footprint = create_test_footprint();
            footprint.methodology = methodology.to_string();
            assert!(validate_footprint(&footprint).is_ok());
        }
    }

    #[test]
    fn test_footprint_verified_by_optional() {
        let footprint = create_test_footprint();
        assert!(footprint.verified_by.is_none());
        assert!(validate_footprint(&footprint).is_ok());
    }

    #[test]
    fn test_footprint_verified_by_valid_did() {
        let mut footprint = create_test_footprint();
        footprint.verified_by = Some("did:mycelix:verifier".to_string());
        assert!(validate_footprint(&footprint).is_ok());
    }

    #[test]
    fn test_footprint_verified_by_invalid_did_rejected() {
        let mut footprint = create_test_footprint();
        footprint.verified_by = Some("invalid_verifier".to_string());
        let result = validate_footprint(&footprint);
        assert!(result.is_err());
    }

    // =========================================================================
    // TOTAL EMISSIONS CALCULATION TESTS
    // =========================================================================

    #[test]
    fn test_total_emissions_calculation() {
        let footprint = create_test_footprint();
        let total = calculate_total_emissions(&footprint);
        assert!((total - 546.5).abs() < 0.001);
        assert_eq!(total, footprint.scope1 + footprint.scope2 + footprint.scope3);
    }

    #[test]
    fn test_total_emissions_zero_when_all_zero() {
        let mut footprint = create_test_footprint();
        footprint.scope1 = 0.0;
        footprint.scope2 = 0.0;
        footprint.scope3 = 0.0;
        let total = calculate_total_emissions(&footprint);
        assert_eq!(total, 0.0);
    }

    #[test]
    fn test_total_emissions_with_only_scope1() {
        let mut footprint = create_test_footprint();
        footprint.scope1 = 100.0;
        footprint.scope2 = 0.0;
        footprint.scope3 = 0.0;
        let total = calculate_total_emissions(&footprint);
        assert_eq!(total, 100.0);
    }

    // =========================================================================
    // CARBON CREDIT VALIDATION TESTS
    // =========================================================================

    #[test]
    fn test_credit_requires_valid_id() {
        let credit = create_test_credit();
        assert!(!credit.id.is_empty());
        assert!(validate_credit(&credit).is_ok());
    }

    #[test]
    fn test_credit_empty_id_rejected() {
        let mut credit = create_test_credit();
        credit.id = "".to_string();
        let result = validate_credit(&credit);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Credit ID"));
    }

    #[test]
    fn test_credit_requires_project_id() {
        let credit = create_test_credit();
        assert!(!credit.project_id.is_empty());
        assert!(validate_credit(&credit).is_ok());
    }

    #[test]
    fn test_credit_empty_project_id_rejected() {
        let mut credit = create_test_credit();
        credit.project_id = "".to_string();
        let result = validate_credit(&credit);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Project ID"));
    }

    #[test]
    fn test_credit_requires_valid_owner_did() {
        let credit = create_test_credit();
        assert!(validate_did(&credit.owner_did).is_ok());
        assert!(validate_credit(&credit).is_ok());
    }

    #[test]
    fn test_credit_invalid_owner_did_rejected() {
        let mut credit = create_test_credit();
        credit.owner_did = "invalid_owner".to_string();
        let result = validate_credit(&credit);
        assert!(result.is_err());
    }

    #[test]
    fn test_credit_tonnes_must_be_positive() {
        let credit = create_test_credit();
        assert!(credit.tonnes_co2e > 0.0);
        assert!(validate_credit(&credit).is_ok());
    }

    #[test]
    fn test_credit_zero_tonnes_rejected() {
        let mut credit = create_test_credit();
        credit.tonnes_co2e = 0.0;
        let result = validate_credit(&credit);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("tonnes must be positive"));
    }

    #[test]
    fn test_credit_negative_tonnes_rejected() {
        let mut credit = create_test_credit();
        credit.tonnes_co2e = -5.0;
        let result = validate_credit(&credit);
        assert!(result.is_err());
    }

    #[test]
    fn test_credit_vintage_year_valid_range() {
        let credit = create_test_credit();
        assert!(credit.vintage_year >= 1990 && credit.vintage_year <= 2100);
        assert!(validate_credit(&credit).is_ok());
    }

    #[test]
    fn test_credit_vintage_year_too_early_rejected() {
        let mut credit = create_test_credit();
        credit.vintage_year = 1989;
        let result = validate_credit(&credit);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Vintage year"));
    }

    #[test]
    fn test_credit_vintage_year_too_late_rejected() {
        let mut credit = create_test_credit();
        credit.vintage_year = 2101;
        let result = validate_credit(&credit);
        assert!(result.is_err());
    }

    #[test]
    fn test_credit_vintage_year_boundary_1990_valid() {
        let mut credit = create_test_credit();
        credit.vintage_year = 1990;
        assert!(validate_credit(&credit).is_ok());
    }

    #[test]
    fn test_credit_vintage_year_boundary_2100_valid() {
        let mut credit = create_test_credit();
        credit.vintage_year = 2100;
        assert!(validate_credit(&credit).is_ok());
    }

    // =========================================================================
    // CREDIT STATUS TESTS
    // =========================================================================

    #[test]
    fn test_credit_initial_status_active() {
        let credit = create_test_credit();
        assert_eq!(credit.status, TestCreditStatus::Active);
    }

    #[test]
    fn test_credit_all_statuses_supported() {
        let statuses = vec![
            TestCreditStatus::Active,
            TestCreditStatus::Transferred,
            TestCreditStatus::Retired,
        ];
        for status in statuses {
            assert!(matches!(status, TestCreditStatus::Active |
                TestCreditStatus::Transferred | TestCreditStatus::Retired));
        }
    }

    #[test]
    fn test_active_credit_no_retired_at() {
        let credit = create_test_credit();
        assert_eq!(credit.status, TestCreditStatus::Active);
        assert!(credit.retired_at.is_none());
        assert!(validate_credit(&credit).is_ok());
    }

    #[test]
    fn test_active_credit_with_retired_at_rejected() {
        let mut credit = create_test_credit();
        credit.status = TestCreditStatus::Active;
        credit.retired_at = Some(1704067200);
        let result = validate_credit(&credit);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Non-retired credits cannot have retired_at"));
    }

    #[test]
    fn test_retired_credit_requires_retired_at() {
        let mut credit = create_test_credit();
        credit.status = TestCreditStatus::Retired;
        credit.retired_at = Some(1704067200);
        assert!(validate_credit(&credit).is_ok());
    }

    #[test]
    fn test_retired_credit_without_retired_at_rejected() {
        let mut credit = create_test_credit();
        credit.status = TestCreditStatus::Retired;
        credit.retired_at = None;
        let result = validate_credit(&credit);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Retired credits must have retired_at"));
    }

    #[test]
    fn test_transferred_credit_no_retired_at() {
        let mut credit = create_test_credit();
        credit.status = TestCreditStatus::Transferred;
        credit.retired_at = None;
        assert!(validate_credit(&credit).is_ok());
    }

    // =========================================================================
    // CREDIT STATUS TRANSITION TESTS
    // =========================================================================

    fn can_transfer_credit(credit: &TestCarbonCredit) -> bool {
        credit.status == TestCreditStatus::Active
    }

    fn can_retire_credit(credit: &TestCarbonCredit) -> bool {
        credit.status == TestCreditStatus::Active
    }

    #[test]
    fn test_can_transfer_active_credit() {
        let credit = create_test_credit();
        assert_eq!(credit.status, TestCreditStatus::Active);
        assert!(can_transfer_credit(&credit));
    }

    #[test]
    fn test_cannot_transfer_retired_credit() {
        let mut credit = create_test_credit();
        credit.status = TestCreditStatus::Retired;
        credit.retired_at = Some(1704067200);
        assert!(!can_transfer_credit(&credit));
    }

    #[test]
    fn test_cannot_transfer_already_transferred_credit() {
        let mut credit = create_test_credit();
        credit.status = TestCreditStatus::Transferred;
        assert!(!can_transfer_credit(&credit));
    }

    #[test]
    fn test_can_retire_active_credit() {
        let credit = create_test_credit();
        assert!(can_retire_credit(&credit));
    }

    #[test]
    fn test_cannot_retire_already_retired_credit() {
        let mut credit = create_test_credit();
        credit.status = TestCreditStatus::Retired;
        credit.retired_at = Some(1704067200);
        assert!(!can_retire_credit(&credit));
    }

    #[test]
    fn test_cannot_retire_transferred_credit() {
        let mut credit = create_test_credit();
        credit.status = TestCreditStatus::Transferred;
        assert!(!can_retire_credit(&credit));
    }

    // =========================================================================
    // CREDIT TRANSFER TESTS
    // =========================================================================

    fn transfer_credit(credit: &mut TestCarbonCredit, new_owner: &str) -> Result<(), String> {
        if !can_transfer_credit(credit) {
            return Err("Only active credits can be transferred".to_string());
        }
        validate_did(new_owner)?;
        if credit.owner_did == new_owner {
            return Err("Cannot transfer to current owner".to_string());
        }
        credit.owner_did = new_owner.to_string();
        // Note: Status remains Active after transfer (credit is re-activated for new owner)
        Ok(())
    }

    #[test]
    fn test_transfer_credit_valid() {
        let mut credit = create_test_credit();
        let result = transfer_credit(&mut credit, "did:mycelix:newowner");
        assert!(result.is_ok());
        assert_eq!(credit.owner_did, "did:mycelix:newowner");
    }

    #[test]
    fn test_transfer_credit_invalid_new_owner_rejected() {
        let mut credit = create_test_credit();
        let result = transfer_credit(&mut credit, "invalid_owner");
        assert!(result.is_err());
    }

    #[test]
    fn test_transfer_credit_to_self_rejected() {
        let mut credit = create_test_credit();
        let current_owner = credit.owner_did.clone();
        let result = transfer_credit(&mut credit, &current_owner);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Cannot transfer to current owner"));
    }

    #[test]
    fn test_transfer_retired_credit_rejected() {
        let mut credit = create_test_credit();
        credit.status = TestCreditStatus::Retired;
        credit.retired_at = Some(1704067200);
        let result = transfer_credit(&mut credit, "did:mycelix:newowner");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Only active credits"));
    }

    // =========================================================================
    // CREDIT RETIREMENT TESTS
    // =========================================================================

    fn retire_credit(credit: &mut TestCarbonCredit, retired_at: i64) -> Result<(), String> {
        if !can_retire_credit(credit) {
            return Err("Only active credits can be retired".to_string());
        }
        credit.status = TestCreditStatus::Retired;
        credit.retired_at = Some(retired_at);
        Ok(())
    }

    #[test]
    fn test_retire_credit_valid() {
        let mut credit = create_test_credit();
        let result = retire_credit(&mut credit, 1704067200);
        assert!(result.is_ok());
        assert_eq!(credit.status, TestCreditStatus::Retired);
        assert_eq!(credit.retired_at, Some(1704067200));
    }

    #[test]
    fn test_retire_already_retired_credit_rejected() {
        let mut credit = create_test_credit();
        credit.status = TestCreditStatus::Retired;
        credit.retired_at = Some(1704000000);
        let result = retire_credit(&mut credit, 1704067200);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Only active credits"));
    }

    #[test]
    fn test_retirement_is_permanent() {
        let mut credit = create_test_credit();
        retire_credit(&mut credit, 1704067200).unwrap();
        // Cannot transfer after retirement
        assert!(!can_transfer_credit(&credit));
        // Cannot retire again
        assert!(!can_retire_credit(&credit));
    }

    // =========================================================================
    // CREDIT SUMMARY TESTS
    // =========================================================================

    fn calculate_credit_summary(credits: &[TestCarbonCredit]) -> TestCreditSummary {
        let mut summary = TestCreditSummary {
            total_credits: 0,
            total_tonnes: 0.0,
            active_tonnes: 0.0,
            retired_tonnes: 0.0,
            transferred_count: 0,
        };

        for credit in credits {
            summary.total_credits += 1;
            summary.total_tonnes += credit.tonnes_co2e;

            match credit.status {
                TestCreditStatus::Active => summary.active_tonnes += credit.tonnes_co2e,
                TestCreditStatus::Retired => summary.retired_tonnes += credit.tonnes_co2e,
                TestCreditStatus::Transferred => summary.transferred_count += 1,
            }
        }

        summary
    }

    #[test]
    fn test_credit_summary_empty() {
        let credits: Vec<TestCarbonCredit> = vec![];
        let summary = calculate_credit_summary(&credits);
        assert_eq!(summary.total_credits, 0);
        assert_eq!(summary.total_tonnes, 0.0);
    }

    #[test]
    fn test_credit_summary_single_active() {
        let credits = vec![create_test_credit()];
        let summary = calculate_credit_summary(&credits);
        assert_eq!(summary.total_credits, 1);
        assert_eq!(summary.total_tonnes, 10.0);
        assert_eq!(summary.active_tonnes, 10.0);
        assert_eq!(summary.retired_tonnes, 0.0);
        assert_eq!(summary.transferred_count, 0);
    }

    #[test]
    fn test_credit_summary_mixed_statuses() {
        let mut credits = vec![
            create_test_credit(), // Active, 10 tonnes
            {
                let mut c = create_test_credit();
                c.id = "credit:2".to_string();
                c.tonnes_co2e = 20.0;
                c.status = TestCreditStatus::Retired;
                c.retired_at = Some(1704067200);
                c
            },
            {
                let mut c = create_test_credit();
                c.id = "credit:3".to_string();
                c.tonnes_co2e = 15.0;
                c.status = TestCreditStatus::Transferred;
                c
            },
        ];

        let summary = calculate_credit_summary(&credits);
        assert_eq!(summary.total_credits, 3);
        assert!((summary.total_tonnes - 45.0).abs() < 0.001);
        assert!((summary.active_tonnes - 10.0).abs() < 0.001);
        assert!((summary.retired_tonnes - 20.0).abs() < 0.001);
        assert_eq!(summary.transferred_count, 1);
    }

    // =========================================================================
    // FOOTPRINT VERIFICATION WORKFLOW TESTS
    // =========================================================================

    fn verify_footprint(footprint: &mut TestCarbonFootprint, verifier_did: &str) -> Result<(), String> {
        validate_did(verifier_did)?;
        if footprint.verified_by.is_some() {
            return Err("Footprint already verified".to_string());
        }
        footprint.verified_by = Some(verifier_did.to_string());
        Ok(())
    }

    #[test]
    fn test_verify_footprint_valid() {
        let mut footprint = create_test_footprint();
        let result = verify_footprint(&mut footprint, "did:mycelix:verifier");
        assert!(result.is_ok());
        assert_eq!(footprint.verified_by, Some("did:mycelix:verifier".to_string()));
    }

    #[test]
    fn test_verify_footprint_invalid_verifier_rejected() {
        let mut footprint = create_test_footprint();
        let result = verify_footprint(&mut footprint, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_already_verified_footprint_rejected() {
        let mut footprint = create_test_footprint();
        footprint.verified_by = Some("did:mycelix:first_verifier".to_string());
        let result = verify_footprint(&mut footprint, "did:mycelix:second_verifier");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already verified"));
    }

    // =========================================================================
    // OFFSET CALCULATION TESTS
    // =========================================================================

    fn calculate_offset_needed(footprint: &TestCarbonFootprint) -> f64 {
        calculate_total_emissions(footprint)
    }

    fn calculate_remaining_offset(footprint: &TestCarbonFootprint, retired_tonnes: f64) -> f64 {
        let needed = calculate_offset_needed(footprint);
        if retired_tonnes >= needed {
            0.0
        } else {
            needed - retired_tonnes
        }
    }

    fn is_carbon_neutral(footprint: &TestCarbonFootprint, retired_tonnes: f64) -> bool {
        retired_tonnes >= calculate_offset_needed(footprint)
    }

    #[test]
    fn test_offset_needed_calculation() {
        let footprint = create_test_footprint();
        let needed = calculate_offset_needed(&footprint);
        assert!((needed - 546.5).abs() < 0.001);
    }

    #[test]
    fn test_remaining_offset_partial() {
        let footprint = create_test_footprint();
        let remaining = calculate_remaining_offset(&footprint, 200.0);
        assert!((remaining - 346.5).abs() < 0.001);
    }

    #[test]
    fn test_remaining_offset_fully_offset() {
        let footprint = create_test_footprint();
        let remaining = calculate_remaining_offset(&footprint, 600.0);
        assert_eq!(remaining, 0.0);
    }

    #[test]
    fn test_is_carbon_neutral_true() {
        let footprint = create_test_footprint();
        assert!(is_carbon_neutral(&footprint, 546.5));
        assert!(is_carbon_neutral(&footprint, 1000.0));
    }

    #[test]
    fn test_is_carbon_neutral_false() {
        let footprint = create_test_footprint();
        assert!(!is_carbon_neutral(&footprint, 500.0));
        assert!(!is_carbon_neutral(&footprint, 0.0));
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_footprint_very_large_emissions() {
        let mut footprint = create_test_footprint();
        footprint.scope1 = 1_000_000.0;
        footprint.scope2 = 500_000.0;
        footprint.scope3 = 2_000_000.0;
        assert!(validate_footprint(&footprint).is_ok());
        let total = calculate_total_emissions(&footprint);
        assert_eq!(total, 3_500_000.0);
    }

    #[test]
    fn test_footprint_very_small_emissions() {
        let mut footprint = create_test_footprint();
        footprint.scope1 = 0.001;
        footprint.scope2 = 0.002;
        footprint.scope3 = 0.003;
        assert!(validate_footprint(&footprint).is_ok());
        let total = calculate_total_emissions(&footprint);
        assert!((total - 0.006).abs() < 0.0001);
    }

    #[test]
    fn test_credit_fractional_tonnes() {
        let mut credit = create_test_credit();
        credit.tonnes_co2e = 0.5; // Half tonne
        assert!(validate_credit(&credit).is_ok());
    }

    #[test]
    fn test_credit_very_large_tonnes() {
        let mut credit = create_test_credit();
        credit.tonnes_co2e = 1_000_000.0;
        assert!(validate_credit(&credit).is_ok());
    }

    #[test]
    fn test_multiple_footprints_same_entity() {
        let footprints = (0..12)
            .map(|month| {
                let mut f = create_test_footprint();
                f.period_start = 1704067200 + (month * 30 * 24 * 3600); // Approximate monthly
                f.period_end = f.period_start + (30 * 24 * 3600);
                f.scope1 = 12.5; // Monthly emissions
                f.scope2 = 6.3;
                f.scope3 = 26.7;
                f
            })
            .collect::<Vec<_>>();

        // All should be valid
        for f in &footprints {
            assert!(validate_footprint(f).is_ok());
        }

        // Total annual emissions
        let total: f64 = footprints.iter().map(|f| calculate_total_emissions(f)).sum();
        assert!((total - 546.0).abs() < 0.1); // 12 * 45.5 = 546
    }

    #[test]
    fn test_credit_transfer_chain() {
        let mut credit = create_test_credit();
        let owners = vec![
            "did:mycelix:owner1",
            "did:mycelix:owner2",
            "did:mycelix:owner3",
            "did:mycelix:owner4",
        ];

        credit.owner_did = owners[0].to_string();

        // Transfer through chain
        for owner in &owners[1..] {
            let result = transfer_credit(&mut credit, owner);
            assert!(result.is_ok());
            assert_eq!(credit.owner_did, *owner);
        }

        assert_eq!(credit.owner_did, "did:mycelix:owner4");
    }

    #[test]
    fn test_multiple_credits_from_same_project() {
        let credits: Vec<TestCarbonCredit> = (0..100)
            .map(|i| TestCarbonCredit {
                id: format!("credit:project-001:2024:{:03}", i),
                project_id: "project:project-001".to_string(),
                vintage_year: 2024,
                tonnes_co2e: 10.0,
                status: TestCreditStatus::Active,
                owner_did: "did:mycelix:holder".to_string(),
                retired_at: None,
            })
            .collect();

        // All valid
        for c in &credits {
            assert!(validate_credit(c).is_ok());
        }

        // Summary
        let summary = calculate_credit_summary(&credits);
        assert_eq!(summary.total_credits, 100);
        assert_eq!(summary.total_tonnes, 1000.0);
        assert_eq!(summary.active_tonnes, 1000.0);
    }

    #[test]
    fn test_credit_vintage_years_span_decades() {
        let vintage_years: Vec<u32> = vec![1990, 2000, 2010, 2020, 2024, 2050, 2100];

        for year in vintage_years {
            let mut credit = create_test_credit();
            credit.vintage_year = year;
            assert!(validate_credit(&credit).is_ok(), "Year {} should be valid", year);
        }
    }
}
