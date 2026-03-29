// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Projects Zome Unit Tests
//!
//! Comprehensive test coverage for climate project registration, status transitions,
//! milestone tracking, verification workflows, and geographic validation.

use serde::{Deserialize, Serialize};

// =============================================================================
// Test Data Structures (mirrors integrity types for testing)
// =============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TestProjectType {
    Reforestation,
    RenewableEnergy,
    MethaneCapture,
    OceanRestoration,
    DirectAirCapture,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TestProjectStatus {
    Proposed,
    Verified,
    Active,
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestLocation {
    pub country_code: String,
    pub region: Option<String>,
    pub latitude: f64,
    pub longitude: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestClimateProject {
    pub id: String,
    pub name: String,
    pub project_type: TestProjectType,
    pub location: TestLocation,
    pub expected_credits: f64,
    pub start_date: i64,
    pub verifier_did: Option<String>,
    pub status: TestProjectStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestProjectMilestone {
    pub project_id: String,
    pub title: String,
    pub description: String,
    pub target_date: i64,
    pub completed_at: Option<i64>,
    pub credits_issued: Option<f64>,
    pub verified_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestProjectsSummary {
    pub total_projects: u64,
    pub proposed_count: u64,
    pub verified_count: u64,
    pub active_count: u64,
    pub completed_count: u64,
    pub total_expected_credits: f64,
    pub reforestation_count: u64,
    pub renewable_energy_count: u64,
    pub methane_capture_count: u64,
    pub ocean_restoration_count: u64,
    pub direct_air_capture_count: u64,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_location() -> TestLocation {
    TestLocation {
        country_code: "BR".to_string(),
        region: Some("Amazon".to_string()),
        latitude: -3.4653,
        longitude: -62.2159,
    }
}

fn create_test_project() -> TestClimateProject {
    TestClimateProject {
        id: "project:reforest-amazon-001".to_string(),
        name: "Amazon Reforestation Initiative".to_string(),
        project_type: TestProjectType::Reforestation,
        location: create_test_location(),
        expected_credits: 50000.0,
        start_date: 1704067200,
        verifier_did: None,
        status: TestProjectStatus::Proposed,
    }
}

fn create_test_milestone() -> TestProjectMilestone {
    TestProjectMilestone {
        project_id: "project:reforest-amazon-001".to_string(),
        title: "Phase 1: Initial Planting".to_string(),
        description: "Plant 10,000 native tree seedlings".to_string(),
        target_date: 1717200000, // June 2024
        completed_at: None,
        credits_issued: None,
        verified_by: None,
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

fn validate_location(location: &TestLocation) -> Result<(), String> {
    if location.country_code.len() != 2 {
        return Err("Country code must be 2 characters (ISO 3166-1 alpha-2)".to_string());
    }

    if location.latitude < -90.0 || location.latitude > 90.0 {
        return Err("Latitude must be between -90 and 90".to_string());
    }

    if location.longitude < -180.0 || location.longitude > 180.0 {
        return Err("Longitude must be between -180 and 180".to_string());
    }

    Ok(())
}

fn validate_project(project: &TestClimateProject) -> Result<(), String> {
    if project.id.is_empty() {
        return Err("Project ID cannot be empty".to_string());
    }

    if project.name.is_empty() {
        return Err("Project name cannot be empty".to_string());
    }

    validate_location(&project.location)?;

    if project.expected_credits < 0.0 {
        return Err("Expected credits cannot be negative".to_string());
    }

    if let Some(ref verifier) = project.verifier_did {
        validate_did(verifier)?;
    }

    Ok(())
}

fn validate_milestone(milestone: &TestProjectMilestone) -> Result<(), String> {
    if milestone.project_id.is_empty() {
        return Err("Project ID cannot be empty".to_string());
    }

    if milestone.title.is_empty() {
        return Err("Milestone title cannot be empty".to_string());
    }

    if let Some(credits) = milestone.credits_issued {
        if credits < 0.0 {
            return Err("Credits issued cannot be negative".to_string());
        }
    }

    if let Some(ref verifier) = milestone.verified_by {
        validate_did(verifier)?;
    }

    if let Some(completed) = milestone.completed_at {
        if completed < milestone.target_date - 31536000 {
            return Err("Completion date seems unreasonably early".to_string());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // LOCATION VALIDATION TESTS
    // =========================================================================

    #[test]
    fn test_location_valid_country_code() {
        let location = create_test_location();
        assert_eq!(location.country_code.len(), 2);
        assert!(validate_location(&location).is_ok());
    }

    #[test]
    fn test_location_country_code_too_short_rejected() {
        let mut location = create_test_location();
        location.country_code = "B".to_string();
        let result = validate_location(&location);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("2 characters"));
    }

    #[test]
    fn test_location_country_code_too_long_rejected() {
        let mut location = create_test_location();
        location.country_code = "BRA".to_string();
        let result = validate_location(&location);
        assert!(result.is_err());
    }

    #[test]
    fn test_location_valid_latitude_range() {
        let location = create_test_location();
        assert!(location.latitude >= -90.0 && location.latitude <= 90.0);
        assert!(validate_location(&location).is_ok());
    }

    #[test]
    fn test_location_latitude_too_low_rejected() {
        let mut location = create_test_location();
        location.latitude = -91.0;
        let result = validate_location(&location);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Latitude"));
    }

    #[test]
    fn test_location_latitude_too_high_rejected() {
        let mut location = create_test_location();
        location.latitude = 91.0;
        let result = validate_location(&location);
        assert!(result.is_err());
    }

    #[test]
    fn test_location_latitude_boundary_values() {
        let mut location = create_test_location();

        location.latitude = -90.0;
        assert!(validate_location(&location).is_ok());

        location.latitude = 90.0;
        assert!(validate_location(&location).is_ok());

        location.latitude = 0.0;
        assert!(validate_location(&location).is_ok());
    }

    #[test]
    fn test_location_valid_longitude_range() {
        let location = create_test_location();
        assert!(location.longitude >= -180.0 && location.longitude <= 180.0);
        assert!(validate_location(&location).is_ok());
    }

    #[test]
    fn test_location_longitude_too_low_rejected() {
        let mut location = create_test_location();
        location.longitude = -181.0;
        let result = validate_location(&location);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Longitude"));
    }

    #[test]
    fn test_location_longitude_too_high_rejected() {
        let mut location = create_test_location();
        location.longitude = 181.0;
        let result = validate_location(&location);
        assert!(result.is_err());
    }

    #[test]
    fn test_location_longitude_boundary_values() {
        let mut location = create_test_location();

        location.longitude = -180.0;
        assert!(validate_location(&location).is_ok());

        location.longitude = 180.0;
        assert!(validate_location(&location).is_ok());

        location.longitude = 0.0;
        assert!(validate_location(&location).is_ok());
    }

    #[test]
    fn test_location_region_optional() {
        let mut location = create_test_location();
        location.region = None;
        assert!(validate_location(&location).is_ok());
    }

    // =========================================================================
    // PROJECT VALIDATION TESTS
    // =========================================================================

    #[test]
    fn test_project_requires_valid_id() {
        let project = create_test_project();
        assert!(!project.id.is_empty());
        assert!(validate_project(&project).is_ok());
    }

    #[test]
    fn test_project_empty_id_rejected() {
        let mut project = create_test_project();
        project.id = "".to_string();
        let result = validate_project(&project);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Project ID"));
    }

    #[test]
    fn test_project_requires_name() {
        let project = create_test_project();
        assert!(!project.name.is_empty());
        assert!(validate_project(&project).is_ok());
    }

    #[test]
    fn test_project_empty_name_rejected() {
        let mut project = create_test_project();
        project.name = "".to_string();
        let result = validate_project(&project);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Project name"));
    }

    #[test]
    fn test_project_expected_credits_non_negative() {
        let project = create_test_project();
        assert!(project.expected_credits >= 0.0);
        assert!(validate_project(&project).is_ok());
    }

    #[test]
    fn test_project_negative_expected_credits_rejected() {
        let mut project = create_test_project();
        project.expected_credits = -1000.0;
        let result = validate_project(&project);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Expected credits"));
    }

    #[test]
    fn test_project_zero_expected_credits_valid() {
        let mut project = create_test_project();
        project.expected_credits = 0.0;
        assert!(validate_project(&project).is_ok());
    }

    #[test]
    fn test_project_verifier_optional() {
        let project = create_test_project();
        assert!(project.verifier_did.is_none());
        assert!(validate_project(&project).is_ok());
    }

    #[test]
    fn test_project_verifier_valid_did() {
        let mut project = create_test_project();
        project.verifier_did = Some("did:mycelix:verifier".to_string());
        assert!(validate_project(&project).is_ok());
    }

    #[test]
    fn test_project_verifier_invalid_did_rejected() {
        let mut project = create_test_project();
        project.verifier_did = Some("invalid_verifier".to_string());
        let result = validate_project(&project);
        assert!(result.is_err());
    }

    // =========================================================================
    // PROJECT TYPE TESTS
    // =========================================================================

    #[test]
    fn test_all_project_types_supported() {
        let types = vec![
            TestProjectType::Reforestation,
            TestProjectType::RenewableEnergy,
            TestProjectType::MethaneCapture,
            TestProjectType::OceanRestoration,
            TestProjectType::DirectAirCapture,
        ];

        for pt in types {
            let mut project = create_test_project();
            project.project_type = pt;
            assert!(validate_project(&project).is_ok());
        }
    }

    #[test]
    fn test_project_type_reforestation() {
        let project = create_test_project();
        assert_eq!(project.project_type, TestProjectType::Reforestation);
    }

    // =========================================================================
    // PROJECT STATUS TESTS
    // =========================================================================

    #[test]
    fn test_project_initial_status_proposed() {
        let project = create_test_project();
        assert_eq!(project.status, TestProjectStatus::Proposed);
    }

    #[test]
    fn test_all_project_statuses_supported() {
        let statuses = vec![
            TestProjectStatus::Proposed,
            TestProjectStatus::Verified,
            TestProjectStatus::Active,
            TestProjectStatus::Completed,
        ];

        for status in statuses {
            assert!(matches!(
                status,
                TestProjectStatus::Proposed
                    | TestProjectStatus::Verified
                    | TestProjectStatus::Active
                    | TestProjectStatus::Completed
            ));
        }
    }

    // =========================================================================
    // PROJECT STATUS TRANSITION TESTS
    // =========================================================================

    fn can_verify_project(project: &TestClimateProject) -> bool {
        project.status == TestProjectStatus::Proposed
    }

    fn can_activate_project(project: &TestClimateProject) -> bool {
        project.status == TestProjectStatus::Verified
    }

    fn can_complete_project(project: &TestClimateProject) -> bool {
        project.status == TestProjectStatus::Active
    }

    #[test]
    fn test_can_verify_proposed_project() {
        let project = create_test_project();
        assert_eq!(project.status, TestProjectStatus::Proposed);
        assert!(can_verify_project(&project));
    }

    #[test]
    fn test_cannot_verify_verified_project() {
        let mut project = create_test_project();
        project.status = TestProjectStatus::Verified;
        assert!(!can_verify_project(&project));
    }

    #[test]
    fn test_cannot_verify_active_project() {
        let mut project = create_test_project();
        project.status = TestProjectStatus::Active;
        assert!(!can_verify_project(&project));
    }

    #[test]
    fn test_cannot_verify_completed_project() {
        let mut project = create_test_project();
        project.status = TestProjectStatus::Completed;
        assert!(!can_verify_project(&project));
    }

    #[test]
    fn test_can_activate_verified_project() {
        let mut project = create_test_project();
        project.status = TestProjectStatus::Verified;
        assert!(can_activate_project(&project));
    }

    #[test]
    fn test_cannot_activate_proposed_project() {
        let project = create_test_project();
        assert!(!can_activate_project(&project));
    }

    #[test]
    fn test_cannot_activate_active_project() {
        let mut project = create_test_project();
        project.status = TestProjectStatus::Active;
        assert!(!can_activate_project(&project));
    }

    #[test]
    fn test_cannot_activate_completed_project() {
        let mut project = create_test_project();
        project.status = TestProjectStatus::Completed;
        assert!(!can_activate_project(&project));
    }

    #[test]
    fn test_can_complete_active_project() {
        let mut project = create_test_project();
        project.status = TestProjectStatus::Active;
        assert!(can_complete_project(&project));
    }

    #[test]
    fn test_cannot_complete_proposed_project() {
        let project = create_test_project();
        assert!(!can_complete_project(&project));
    }

    #[test]
    fn test_cannot_complete_verified_project() {
        let mut project = create_test_project();
        project.status = TestProjectStatus::Verified;
        assert!(!can_complete_project(&project));
    }

    #[test]
    fn test_cannot_complete_already_completed_project() {
        let mut project = create_test_project();
        project.status = TestProjectStatus::Completed;
        assert!(!can_complete_project(&project));
    }

    // =========================================================================
    // PROJECT VERIFICATION WORKFLOW TESTS
    // =========================================================================

    fn verify_project(project: &mut TestClimateProject, verifier_did: &str) -> Result<(), String> {
        if !can_verify_project(project) {
            return Err("Only proposed projects can be verified".to_string());
        }
        validate_did(verifier_did)?;
        project.verifier_did = Some(verifier_did.to_string());
        project.status = TestProjectStatus::Verified;
        Ok(())
    }

    fn activate_project(project: &mut TestClimateProject) -> Result<(), String> {
        if !can_activate_project(project) {
            return Err("Only verified projects can be activated".to_string());
        }
        project.status = TestProjectStatus::Active;
        Ok(())
    }

    fn complete_project(project: &mut TestClimateProject) -> Result<(), String> {
        if !can_complete_project(project) {
            return Err("Only active projects can be completed".to_string());
        }
        project.status = TestProjectStatus::Completed;
        Ok(())
    }

    #[test]
    fn test_verify_project_success() {
        let mut project = create_test_project();
        let result = verify_project(&mut project, "did:mycelix:verifier");
        assert!(result.is_ok());
        assert_eq!(project.status, TestProjectStatus::Verified);
        assert_eq!(
            project.verifier_did,
            Some("did:mycelix:verifier".to_string())
        );
    }

    #[test]
    fn test_verify_project_invalid_did_rejected() {
        let mut project = create_test_project();
        let result = verify_project(&mut project, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_non_proposed_project_rejected() {
        let mut project = create_test_project();
        project.status = TestProjectStatus::Active;
        let result = verify_project(&mut project, "did:mycelix:verifier");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Only proposed"));
    }

    #[test]
    fn test_activate_verified_project_success() {
        let mut project = create_test_project();
        project.status = TestProjectStatus::Verified;
        let result = activate_project(&mut project);
        assert!(result.is_ok());
        assert_eq!(project.status, TestProjectStatus::Active);
    }

    #[test]
    fn test_activate_non_verified_project_rejected() {
        let mut project = create_test_project();
        let result = activate_project(&mut project);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Only verified"));
    }

    #[test]
    fn test_complete_active_project_success() {
        let mut project = create_test_project();
        project.status = TestProjectStatus::Active;
        let result = complete_project(&mut project);
        assert!(result.is_ok());
        assert_eq!(project.status, TestProjectStatus::Completed);
    }

    #[test]
    fn test_complete_non_active_project_rejected() {
        let mut project = create_test_project();
        project.status = TestProjectStatus::Verified;
        let result = complete_project(&mut project);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Only active"));
    }

    #[test]
    fn test_full_project_lifecycle() {
        let mut project = create_test_project();

        // Initial state
        assert_eq!(project.status, TestProjectStatus::Proposed);

        // Verify
        verify_project(&mut project, "did:mycelix:verifier").unwrap();
        assert_eq!(project.status, TestProjectStatus::Verified);

        // Activate
        activate_project(&mut project).unwrap();
        assert_eq!(project.status, TestProjectStatus::Active);

        // Complete
        complete_project(&mut project).unwrap();
        assert_eq!(project.status, TestProjectStatus::Completed);

        // Cannot go back
        assert!(!can_verify_project(&project));
        assert!(!can_activate_project(&project));
        assert!(!can_complete_project(&project));
    }

    // =========================================================================
    // MILESTONE VALIDATION TESTS
    // =========================================================================

    #[test]
    fn test_milestone_requires_project_id() {
        let milestone = create_test_milestone();
        assert!(!milestone.project_id.is_empty());
        assert!(validate_milestone(&milestone).is_ok());
    }

    #[test]
    fn test_milestone_empty_project_id_rejected() {
        let mut milestone = create_test_milestone();
        milestone.project_id = "".to_string();
        let result = validate_milestone(&milestone);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Project ID"));
    }

    #[test]
    fn test_milestone_requires_title() {
        let milestone = create_test_milestone();
        assert!(!milestone.title.is_empty());
        assert!(validate_milestone(&milestone).is_ok());
    }

    #[test]
    fn test_milestone_empty_title_rejected() {
        let mut milestone = create_test_milestone();
        milestone.title = "".to_string();
        let result = validate_milestone(&milestone);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("title"));
    }

    #[test]
    fn test_milestone_credits_issued_optional() {
        let milestone = create_test_milestone();
        assert!(milestone.credits_issued.is_none());
        assert!(validate_milestone(&milestone).is_ok());
    }

    #[test]
    fn test_milestone_credits_issued_non_negative() {
        let mut milestone = create_test_milestone();
        milestone.credits_issued = Some(1000.0);
        assert!(validate_milestone(&milestone).is_ok());
    }

    #[test]
    fn test_milestone_negative_credits_rejected() {
        let mut milestone = create_test_milestone();
        milestone.credits_issued = Some(-100.0);
        let result = validate_milestone(&milestone);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Credits issued"));
    }

    #[test]
    fn test_milestone_zero_credits_valid() {
        let mut milestone = create_test_milestone();
        milestone.credits_issued = Some(0.0);
        assert!(validate_milestone(&milestone).is_ok());
    }

    #[test]
    fn test_milestone_verified_by_optional() {
        let milestone = create_test_milestone();
        assert!(milestone.verified_by.is_none());
        assert!(validate_milestone(&milestone).is_ok());
    }

    #[test]
    fn test_milestone_verified_by_valid_did() {
        let mut milestone = create_test_milestone();
        milestone.verified_by = Some("did:mycelix:verifier".to_string());
        assert!(validate_milestone(&milestone).is_ok());
    }

    #[test]
    fn test_milestone_verified_by_invalid_did_rejected() {
        let mut milestone = create_test_milestone();
        milestone.verified_by = Some("invalid".to_string());
        let result = validate_milestone(&milestone);
        assert!(result.is_err());
    }

    #[test]
    fn test_milestone_completed_at_optional() {
        let milestone = create_test_milestone();
        assert!(milestone.completed_at.is_none());
        assert!(validate_milestone(&milestone).is_ok());
    }

    #[test]
    fn test_milestone_completed_at_reasonable() {
        let mut milestone = create_test_milestone();
        milestone.completed_at = Some(milestone.target_date - 30 * 24 * 3600); // 30 days early
        assert!(validate_milestone(&milestone).is_ok());
    }

    #[test]
    fn test_milestone_completed_at_too_early_rejected() {
        let mut milestone = create_test_milestone();
        // More than 1 year before target
        milestone.completed_at = Some(milestone.target_date - 400 * 24 * 3600);
        let result = validate_milestone(&milestone);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unreasonably early"));
    }

    // =========================================================================
    // MILESTONE COMPLETION WORKFLOW TESTS
    // =========================================================================

    fn complete_milestone(
        milestone: &mut TestProjectMilestone,
        verifier_did: &str,
        credits_issued: Option<f64>,
        completed_at: i64,
    ) -> Result<(), String> {
        if milestone.completed_at.is_some() {
            return Err("Milestone is already completed".to_string());
        }
        validate_did(verifier_did)?;
        if let Some(credits) = credits_issued {
            if credits < 0.0 {
                return Err("Credits cannot be negative".to_string());
            }
        }
        milestone.completed_at = Some(completed_at);
        milestone.verified_by = Some(verifier_did.to_string());
        milestone.credits_issued = credits_issued;
        Ok(())
    }

    #[test]
    fn test_complete_milestone_success() {
        let mut milestone = create_test_milestone();
        let result = complete_milestone(&mut milestone, "did:mycelix:verifier", Some(5000.0), 1717200000);
        assert!(result.is_ok());
        assert!(milestone.completed_at.is_some());
        assert_eq!(milestone.credits_issued, Some(5000.0));
        assert_eq!(
            milestone.verified_by,
            Some("did:mycelix:verifier".to_string())
        );
    }

    #[test]
    fn test_complete_milestone_no_credits() {
        let mut milestone = create_test_milestone();
        let result = complete_milestone(&mut milestone, "did:mycelix:verifier", None, 1717200000);
        assert!(result.is_ok());
        assert!(milestone.credits_issued.is_none());
    }

    #[test]
    fn test_complete_already_completed_milestone_rejected() {
        let mut milestone = create_test_milestone();
        milestone.completed_at = Some(1717100000);
        let result = complete_milestone(&mut milestone, "did:mycelix:verifier", Some(1000.0), 1717200000);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already completed"));
    }

    #[test]
    fn test_complete_milestone_invalid_verifier_rejected() {
        let mut milestone = create_test_milestone();
        let result = complete_milestone(&mut milestone, "invalid", Some(1000.0), 1717200000);
        assert!(result.is_err());
    }

    // =========================================================================
    // PROJECT SUMMARY TESTS
    // =========================================================================

    fn calculate_project_summary(projects: &[TestClimateProject]) -> TestProjectsSummary {
        let mut summary = TestProjectsSummary {
            total_projects: 0,
            proposed_count: 0,
            verified_count: 0,
            active_count: 0,
            completed_count: 0,
            total_expected_credits: 0.0,
            reforestation_count: 0,
            renewable_energy_count: 0,
            methane_capture_count: 0,
            ocean_restoration_count: 0,
            direct_air_capture_count: 0,
        };

        for project in projects {
            summary.total_projects += 1;
            summary.total_expected_credits += project.expected_credits;

            match project.status {
                TestProjectStatus::Proposed => summary.proposed_count += 1,
                TestProjectStatus::Verified => summary.verified_count += 1,
                TestProjectStatus::Active => summary.active_count += 1,
                TestProjectStatus::Completed => summary.completed_count += 1,
            }

            match project.project_type {
                TestProjectType::Reforestation => summary.reforestation_count += 1,
                TestProjectType::RenewableEnergy => summary.renewable_energy_count += 1,
                TestProjectType::MethaneCapture => summary.methane_capture_count += 1,
                TestProjectType::OceanRestoration => summary.ocean_restoration_count += 1,
                TestProjectType::DirectAirCapture => summary.direct_air_capture_count += 1,
            }
        }

        summary
    }

    #[test]
    fn test_summary_empty_projects() {
        let projects: Vec<TestClimateProject> = vec![];
        let summary = calculate_project_summary(&projects);
        assert_eq!(summary.total_projects, 0);
        assert_eq!(summary.total_expected_credits, 0.0);
    }

    #[test]
    fn test_summary_single_project() {
        let projects = vec![create_test_project()];
        let summary = calculate_project_summary(&projects);
        assert_eq!(summary.total_projects, 1);
        assert_eq!(summary.proposed_count, 1);
        assert_eq!(summary.reforestation_count, 1);
        assert_eq!(summary.total_expected_credits, 50000.0);
    }

    #[test]
    fn test_summary_mixed_projects() {
        let mut projects = vec![];

        // Add different project types and statuses
        projects.push(create_test_project()); // Reforestation, Proposed, 50000

        let mut p2 = create_test_project();
        p2.id = "project:2".to_string();
        p2.project_type = TestProjectType::RenewableEnergy;
        p2.status = TestProjectStatus::Active;
        p2.expected_credits = 30000.0;
        projects.push(p2);

        let mut p3 = create_test_project();
        p3.id = "project:3".to_string();
        p3.project_type = TestProjectType::DirectAirCapture;
        p3.status = TestProjectStatus::Completed;
        p3.expected_credits = 10000.0;
        projects.push(p3);

        let summary = calculate_project_summary(&projects);

        assert_eq!(summary.total_projects, 3);
        assert_eq!(summary.proposed_count, 1);
        assert_eq!(summary.active_count, 1);
        assert_eq!(summary.completed_count, 1);
        assert_eq!(summary.reforestation_count, 1);
        assert_eq!(summary.renewable_energy_count, 1);
        assert_eq!(summary.direct_air_capture_count, 1);
        assert!((summary.total_expected_credits - 90000.0).abs() < 0.001);
    }

    // =========================================================================
    // GEOGRAPHIC DISTRIBUTION TESTS
    // =========================================================================

    #[test]
    fn test_projects_various_locations() {
        let locations = vec![
            ("BR", -3.4653, -62.2159),   // Brazil Amazon
            ("US", 37.7749, -122.4194),  // San Francisco
            ("KE", -1.2921, 36.8219),    // Kenya
            ("AU", -33.8688, 151.2093),  // Sydney
            ("NO", 60.4720, 8.4689),     // Norway
        ];

        for (country, lat, lon) in locations {
            let mut project = create_test_project();
            project.location = TestLocation {
                country_code: country.to_string(),
                region: None,
                latitude: lat,
                longitude: lon,
            };
            assert!(
                validate_project(&project).is_ok(),
                "Project in {} should be valid",
                country
            );
        }
    }

    #[test]
    fn test_polar_locations_valid() {
        let mut project = create_test_project();

        // Arctic
        project.location.latitude = 89.9;
        project.location.longitude = 0.0;
        assert!(validate_project(&project).is_ok());

        // Antarctic
        project.location.latitude = -89.9;
        project.location.longitude = 0.0;
        assert!(validate_project(&project).is_ok());
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_project_very_large_expected_credits() {
        let mut project = create_test_project();
        project.expected_credits = 10_000_000.0; // 10 million tonnes
        assert!(validate_project(&project).is_ok());
    }

    #[test]
    fn test_project_very_small_expected_credits() {
        let mut project = create_test_project();
        project.expected_credits = 0.001; // 1 kg
        assert!(validate_project(&project).is_ok());
    }

    #[test]
    fn test_multiple_milestones_same_project() {
        let milestones: Vec<TestProjectMilestone> = (0..10)
            .map(|i| TestProjectMilestone {
                project_id: "project:multi-milestone".to_string(),
                title: format!("Phase {}", i + 1),
                description: format!("Phase {} description", i + 1),
                target_date: 1704067200 + (i as i64 * 90 * 24 * 3600), // Quarterly
                completed_at: None,
                credits_issued: None,
                verified_by: None,
            })
            .collect();

        for m in &milestones {
            assert!(validate_milestone(m).is_ok());
        }
    }

    #[test]
    fn test_project_long_name() {
        let mut project = create_test_project();
        project.name = "A".repeat(500); // Very long name
        assert!(validate_project(&project).is_ok());
    }

    #[test]
    fn test_milestone_long_description() {
        let mut milestone = create_test_milestone();
        milestone.description = "B".repeat(10000); // Very long description
        assert!(validate_milestone(&milestone).is_ok());
    }

    #[test]
    fn test_projects_all_types_all_statuses() {
        let types = vec![
            TestProjectType::Reforestation,
            TestProjectType::RenewableEnergy,
            TestProjectType::MethaneCapture,
            TestProjectType::OceanRestoration,
            TestProjectType::DirectAirCapture,
        ];

        let statuses = vec![
            TestProjectStatus::Proposed,
            TestProjectStatus::Verified,
            TestProjectStatus::Active,
            TestProjectStatus::Completed,
        ];

        for pt in &types {
            for status in &statuses {
                let mut project = create_test_project();
                project.project_type = *pt;
                project.status = *status;
                if *status != TestProjectStatus::Proposed {
                    project.verifier_did = Some("did:mycelix:verifier".to_string());
                }
                assert!(
                    validate_project(&project).is_ok(),
                    "Project {:?}/{:?} should be valid",
                    pt,
                    status
                );
            }
        }
    }

    #[test]
    fn test_project_credits_accumulation_over_milestones() {
        let mut milestones = vec![
            create_test_milestone(),
            {
                let mut m = create_test_milestone();
                m.title = "Phase 2".to_string();
                m.target_date += 90 * 24 * 3600;
                m
            },
            {
                let mut m = create_test_milestone();
                m.title = "Phase 3".to_string();
                m.target_date += 180 * 24 * 3600;
                m
            },
        ];

        // Complete milestones with credits
        let credits_per_milestone = vec![5000.0, 7500.0, 12500.0];
        let mut total_credits = 0.0;

        for (i, m) in milestones.iter_mut().enumerate() {
            let credits = credits_per_milestone[i];
            complete_milestone(
                m,
                "did:mycelix:verifier",
                Some(credits),
                m.target_date,
            )
            .unwrap();
            total_credits += credits;
        }

        assert_eq!(total_credits, 25000.0);

        // Verify all milestones completed
        for m in &milestones {
            assert!(m.completed_at.is_some());
            assert!(m.credits_issued.is_some());
            assert!(m.verified_by.is_some());
        }
    }
}
