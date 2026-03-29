// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-Zome Integration Tests
//!
//! Integration tests covering interactions between zomes:
//! - Projects -> Carbon credit issuance flow
//! - Carbon -> Bridge marketplace listing flow
//! - Bridge -> Carbon verification flow
//! - Full lifecycle: Project -> Credit -> Marketplace -> Verification

use serde::{Deserialize, Serialize};

// =============================================================================
// Test Data Structures (minimal for integration testing)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockClimateProject {
    pub id: String,
    pub name: String,
    pub status: String,
    pub expected_credits: f64,
    pub verifier_did: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockMilestone {
    pub project_id: String,
    pub title: String,
    pub credits_issued: Option<f64>,
    pub completed_at: Option<i64>,
    pub verified_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockCarbonCredit {
    pub id: String,
    pub project_id: String,
    pub vintage_year: u32,
    pub tonnes_co2e: f64,
    pub status: String,
    pub owner_did: String,
    pub retired_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockCarbonFootprint {
    pub entity_did: String,
    pub total_emissions: f64,
    pub verified_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockMarketplaceListing {
    pub listing_id: String,
    pub credit_id: String,
    pub seller_did: String,
    pub price_per_tonne: u64,
    pub available_tonnes: f64,
    pub is_active: bool,
    pub expires_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockVerificationQuery {
    pub query_id: String,
    pub purpose: String,
    pub target_id: String,
    pub status: String,
    pub result: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockBridgeEvent {
    pub event_type: String,
    pub subject_id: String,
    pub payload: String,
    pub timestamp: i64,
}

// =============================================================================
// Mock State Management (simulates DHT state)
// =============================================================================

struct MockDHT {
    projects: Vec<MockClimateProject>,
    milestones: Vec<MockMilestone>,
    credits: Vec<MockCarbonCredit>,
    footprints: Vec<MockCarbonFootprint>,
    listings: Vec<MockMarketplaceListing>,
    queries: Vec<MockVerificationQuery>,
    events: Vec<MockBridgeEvent>,
    next_credit_id: u32,
}

impl MockDHT {
    fn new() -> Self {
        Self {
            projects: vec![],
            milestones: vec![],
            credits: vec![],
            footprints: vec![],
            listings: vec![],
            queries: vec![],
            events: vec![],
            next_credit_id: 1,
        }
    }

    // =========================================================================
    // Project Operations
    // =========================================================================

    fn create_project(&mut self, project: MockClimateProject) {
        self.events.push(MockBridgeEvent {
            event_type: "ProjectCreated".to_string(),
            subject_id: project.id.clone(),
            payload: format!(r#"{{"name":"{}","expected_credits":{}}}"#, project.name, project.expected_credits),
            timestamp: 1704067200000000,
        });
        self.projects.push(project);
    }

    fn get_project(&self, project_id: &str) -> Option<&MockClimateProject> {
        self.projects.iter().find(|p| p.id == project_id)
    }

    fn get_project_mut(&mut self, project_id: &str) -> Option<&mut MockClimateProject> {
        self.projects.iter_mut().find(|p| p.id == project_id)
    }

    fn verify_project(&mut self, project_id: &str, verifier_did: &str) -> bool {
        if let Some(project) = self.get_project_mut(project_id) {
            if project.status == "Proposed" {
                project.status = "Verified".to_string();
                project.verifier_did = Some(verifier_did.to_string());
                self.events.push(MockBridgeEvent {
                    event_type: "ProjectVerified".to_string(),
                    subject_id: project_id.to_string(),
                    payload: format!(r#"{{"verifier_did":"{}"}}"#, verifier_did),
                    timestamp: 1704153600000000,
                });
                return true;
            }
        }
        false
    }

    fn activate_project(&mut self, project_id: &str) -> bool {
        if let Some(project) = self.get_project_mut(project_id) {
            if project.status == "Verified" {
                project.status = "Active".to_string();
                self.events.push(MockBridgeEvent {
                    event_type: "ProjectActivated".to_string(),
                    subject_id: project_id.to_string(),
                    payload: "{}".to_string(),
                    timestamp: 1704240000000000,
                });
                return true;
            }
        }
        false
    }

    // =========================================================================
    // Milestone Operations
    // =========================================================================

    fn create_milestone(&mut self, milestone: MockMilestone) {
        self.milestones.push(milestone);
    }

    fn complete_milestone(
        &mut self,
        project_id: &str,
        milestone_title: &str,
        verifier_did: &str,
        credits_to_issue: f64,
        completed_at: i64,
    ) -> bool {
        // Verify project is active
        if let Some(project) = self.get_project(project_id) {
            if project.status != "Active" {
                return false;
            }
        } else {
            return false;
        }

        // Find and complete milestone
        if let Some(milestone) = self.milestones.iter_mut().find(|m| {
            m.project_id == project_id && m.title == milestone_title
        }) {
            if milestone.completed_at.is_some() {
                return false; // Already completed
            }

            milestone.completed_at = Some(completed_at);
            milestone.verified_by = Some(verifier_did.to_string());
            milestone.credits_issued = Some(credits_to_issue);

            // Issue credits
            self.issue_credits_for_milestone(project_id, credits_to_issue, verifier_did);

            self.events.push(MockBridgeEvent {
                event_type: "MilestoneCompleted".to_string(),
                subject_id: project_id.to_string(),
                payload: format!(
                    r#"{{"milestone":"{}","credits_issued":{}}}"#,
                    milestone_title, credits_to_issue
                ),
                timestamp: completed_at,
            });

            return true;
        }

        false
    }

    fn issue_credits_for_milestone(
        &mut self,
        project_id: &str,
        tonnes: f64,
        _verifier_did: &str,
    ) {
        // Issue credits in 10 tonne increments
        let num_credits = (tonnes / 10.0).ceil() as u32;
        let tonnes_per_credit = tonnes / num_credits as f64;

        for _ in 0..num_credits {
            let credit = MockCarbonCredit {
                id: format!("credit:{}:{:04}", project_id, self.next_credit_id),
                project_id: project_id.to_string(),
                vintage_year: 2024,
                tonnes_co2e: tonnes_per_credit,
                status: "Active".to_string(),
                owner_did: "did:mycelix:project_owner".to_string(), // Initial owner is project owner
                retired_at: None,
            };
            self.next_credit_id += 1;

            self.events.push(MockBridgeEvent {
                event_type: "CreditIssued".to_string(),
                subject_id: credit.id.clone(),
                payload: format!(
                    r#"{{"project_id":"{}","tonnes":{}}}"#,
                    project_id, tonnes_per_credit
                ),
                timestamp: 1704326400000000,
            });

            self.credits.push(credit);
        }
    }

    // =========================================================================
    // Carbon Credit Operations
    // =========================================================================

    fn get_credit(&self, credit_id: &str) -> Option<&MockCarbonCredit> {
        self.credits.iter().find(|c| c.id == credit_id)
    }

    fn get_credit_mut(&mut self, credit_id: &str) -> Option<&mut MockCarbonCredit> {
        self.credits.iter_mut().find(|c| c.id == credit_id)
    }

    fn get_credits_by_project(&self, project_id: &str) -> Vec<&MockCarbonCredit> {
        self.credits.iter().filter(|c| c.project_id == project_id).collect()
    }

    fn get_credits_by_owner(&self, owner_did: &str) -> Vec<&MockCarbonCredit> {
        self.credits.iter().filter(|c| c.owner_did == owner_did).collect()
    }

    fn transfer_credit(&mut self, credit_id: &str, new_owner_did: &str) -> bool {
        if let Some(credit) = self.get_credit_mut(credit_id) {
            if credit.status != "Active" {
                return false;
            }

            let old_owner = credit.owner_did.clone();
            credit.owner_did = new_owner_did.to_string();

            self.events.push(MockBridgeEvent {
                event_type: "CreditTransferred".to_string(),
                subject_id: credit_id.to_string(),
                payload: format!(
                    r#"{{"from":"{}","to":"{}"}}"#,
                    old_owner, new_owner_did
                ),
                timestamp: 1704412800000000,
            });

            return true;
        }
        false
    }

    fn retire_credit(&mut self, credit_id: &str, retired_at: i64) -> bool {
        // First check status and get tonnes
        let tonnes = {
            if let Some(credit) = self.credits.iter().find(|c| c.id == credit_id) {
                if credit.status != "Active" {
                    return false;
                }
                credit.tonnes_co2e
            } else {
                return false;
            }
        };

        // Now mutate
        if let Some(credit) = self.credits.iter_mut().find(|c| c.id == credit_id) {
            credit.status = "Retired".to_string();
            credit.retired_at = Some(retired_at);
        }

        self.events.push(MockBridgeEvent {
            event_type: "CreditRetired".to_string(),
            subject_id: credit_id.to_string(),
            payload: format!(r#"{{"tonnes":{}}}"#, tonnes),
            timestamp: retired_at,
        });

        true
    }

    // =========================================================================
    // Carbon Footprint Operations
    // =========================================================================

    fn create_footprint(&mut self, footprint: MockCarbonFootprint) {
        self.footprints.push(footprint);
    }

    fn get_footprints_by_entity(&self, entity_did: &str) -> Vec<&MockCarbonFootprint> {
        self.footprints.iter().filter(|f| f.entity_did == entity_did).collect()
    }

    fn get_total_emissions(&self, entity_did: &str) -> f64 {
        self.footprints
            .iter()
            .filter(|f| f.entity_did == entity_did)
            .map(|f| f.total_emissions)
            .sum()
    }

    // =========================================================================
    // Marketplace Operations
    // =========================================================================

    fn create_listing(&mut self, listing: MockMarketplaceListing) -> bool {
        // Verify credit exists and is active
        if let Some(credit) = self.get_credit(&listing.credit_id) {
            if credit.status != "Active" {
                return false;
            }
            if credit.owner_did != listing.seller_did {
                return false; // Only owner can list
            }
        } else {
            return false;
        }

        self.events.push(MockBridgeEvent {
            event_type: "ListingCreated".to_string(),
            subject_id: listing.listing_id.clone(),
            payload: format!(
                r#"{{"credit_id":"{}","price":{}}}"#,
                listing.credit_id, listing.price_per_tonne
            ),
            timestamp: 1704499200000000,
        });

        self.listings.push(listing);
        true
    }

    fn get_listing(&self, listing_id: &str) -> Option<&MockMarketplaceListing> {
        self.listings.iter().find(|l| l.listing_id == listing_id)
    }

    fn get_active_listings(&self, current_time: i64) -> Vec<&MockMarketplaceListing> {
        self.listings
            .iter()
            .filter(|l| l.is_active && l.expires_at > current_time)
            .collect()
    }

    fn deactivate_listing(&mut self, listing_id: &str) -> bool {
        if let Some(listing) = self.listings.iter_mut().find(|l| l.listing_id == listing_id) {
            listing.is_active = false;
            return true;
        }
        false
    }

    // =========================================================================
    // Verification Query Operations
    // =========================================================================

    fn create_verification_query(&mut self, query: MockVerificationQuery) {
        self.events.push(MockBridgeEvent {
            event_type: "VerificationQueryCreated".to_string(),
            subject_id: query.query_id.clone(),
            payload: format!(
                r#"{{"purpose":"{}","target":"{}"}}"#,
                query.purpose, query.target_id
            ),
            timestamp: 1704585600000000,
        });
        self.queries.push(query);
    }

    fn verify_credit(&mut self, query_id: &str, credit_id: &str) -> bool {
        // First get credit status
        let credit_status = self.credits.iter()
            .find(|c| c.id == credit_id)
            .map(|c| c.status.clone());

        // Then update query
        if let Some(query) = self.queries.iter_mut().find(|q| q.query_id == query_id) {
            match credit_status.as_deref() {
                Some("Active") => {
                    query.status = "Completed".to_string();
                    query.result = Some("Verified".to_string());
                    return true;
                }
                Some("Retired") => {
                    query.status = "Completed".to_string();
                    query.result = Some("Failed".to_string());
                    return true;
                }
                _ => {
                    query.status = "Completed".to_string();
                    query.result = Some("NotFound".to_string());
                }
            }
        }
        false
    }

    fn verify_project_due_diligence(&mut self, query_id: &str, project_id: &str) -> bool {
        // First get project status
        let project_status = self.projects.iter()
            .find(|p| p.id == project_id)
            .map(|p| p.status.clone());

        // Then update query
        if let Some(query) = self.queries.iter_mut().find(|q| q.query_id == query_id) {
            match project_status.as_deref() {
                Some("Active") | Some("Completed") => {
                    query.status = "Completed".to_string();
                    query.result = Some("Verified".to_string());
                    return true;
                }
                Some(_) => {
                    query.status = "Completed".to_string();
                    query.result = Some("Inconclusive".to_string());
                    return true;
                }
                None => {
                    query.status = "Completed".to_string();
                    query.result = Some("NotFound".to_string());
                }
            }
        }
        false
    }

    // =========================================================================
    // Event Tracking
    // =========================================================================

    fn get_events_for_subject(&self, subject_id: &str) -> Vec<&MockBridgeEvent> {
        self.events.iter().filter(|e| e.subject_id == subject_id).collect()
    }

    fn get_events_by_type(&self, event_type: &str) -> Vec<&MockBridgeEvent> {
        self.events.iter().filter(|e| e.event_type == event_type).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PROJECT -> CREDIT ISSUANCE INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_project_lifecycle_issues_credits_on_milestone() {
        let mut dht = MockDHT::new();

        // 1. Create project
        let project = MockClimateProject {
            id: "project:reforest-001".to_string(),
            name: "Amazon Reforestation".to_string(),
            status: "Proposed".to_string(),
            expected_credits: 50000.0,
            verifier_did: None,
        };
        dht.create_project(project);

        // 2. Create milestones
        dht.create_milestone(MockMilestone {
            project_id: "project:reforest-001".to_string(),
            title: "Phase 1".to_string(),
            credits_issued: None,
            completed_at: None,
            verified_by: None,
        });

        // 3. Verify and activate project
        assert!(dht.verify_project("project:reforest-001", "did:mycelix:verifier"));
        assert!(dht.activate_project("project:reforest-001"));

        // 4. Complete milestone - should issue credits
        assert!(dht.complete_milestone(
            "project:reforest-001",
            "Phase 1",
            "did:mycelix:verifier",
            100.0, // 100 tonnes
            1704326400000000
        ));

        // 5. Verify credits were issued
        let credits = dht.get_credits_by_project("project:reforest-001");
        assert!(!credits.is_empty());

        let total_issued: f64 = credits.iter().map(|c| c.tonnes_co2e).sum();
        assert!((total_issued - 100.0).abs() < 0.001);

        // 6. Verify events
        let events = dht.get_events_by_type("CreditIssued");
        assert!(!events.is_empty());
    }

    #[test]
    fn test_credits_not_issued_for_inactive_project() {
        let mut dht = MockDHT::new();

        // Create project but don't activate
        let project = MockClimateProject {
            id: "project:inactive".to_string(),
            name: "Inactive Project".to_string(),
            status: "Proposed".to_string(),
            expected_credits: 10000.0,
            verifier_did: None,
        };
        dht.create_project(project);

        dht.create_milestone(MockMilestone {
            project_id: "project:inactive".to_string(),
            title: "Phase 1".to_string(),
            credits_issued: None,
            completed_at: None,
            verified_by: None,
        });

        // Try to complete milestone
        let result = dht.complete_milestone(
            "project:inactive",
            "Phase 1",
            "did:mycelix:verifier",
            100.0,
            1704326400000000
        );

        assert!(!result);

        // No credits should be issued
        let credits = dht.get_credits_by_project("project:inactive");
        assert!(credits.is_empty());
    }

    #[test]
    fn test_multiple_milestones_cumulative_credits() {
        let mut dht = MockDHT::new();

        // Setup project
        let project = MockClimateProject {
            id: "project:multi".to_string(),
            name: "Multi-Milestone Project".to_string(),
            status: "Proposed".to_string(),
            expected_credits: 50000.0,
            verifier_did: None,
        };
        dht.create_project(project);

        // Create multiple milestones
        for i in 1..=3 {
            dht.create_milestone(MockMilestone {
                project_id: "project:multi".to_string(),
                title: format!("Phase {}", i),
                credits_issued: None,
                completed_at: None,
                verified_by: None,
            });
        }

        // Activate project
        dht.verify_project("project:multi", "did:mycelix:verifier");
        dht.activate_project("project:multi");

        // Complete milestones
        let credits_per_phase = vec![1000.0, 1500.0, 2500.0];
        for (i, credits) in credits_per_phase.iter().enumerate() {
            dht.complete_milestone(
                "project:multi",
                &format!("Phase {}", i + 1),
                "did:mycelix:verifier",
                *credits,
                1704326400000000 + (i as i64 * 86400000000),
            );
        }

        // Verify total credits
        let all_credits = dht.get_credits_by_project("project:multi");
        let total: f64 = all_credits.iter().map(|c| c.tonnes_co2e).sum();
        assert!((total - 5000.0).abs() < 0.001);
    }

    // =========================================================================
    // CARBON -> MARKETPLACE INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_credit_can_be_listed_on_marketplace() {
        let mut dht = MockDHT::new();

        // Create a credit directly
        dht.credits.push(MockCarbonCredit {
            id: "credit:test:001".to_string(),
            project_id: "project:test".to_string(),
            vintage_year: 2024,
            tonnes_co2e: 10.0,
            status: "Active".to_string(),
            owner_did: "did:mycelix:seller".to_string(),
            retired_at: None,
        });

        // Create listing
        let listing = MockMarketplaceListing {
            listing_id: "listing:001".to_string(),
            credit_id: "credit:test:001".to_string(),
            seller_did: "did:mycelix:seller".to_string(),
            price_per_tonne: 25_00,
            available_tonnes: 10.0,
            is_active: true,
            expires_at: 1735689599,
        };

        assert!(dht.create_listing(listing));

        // Verify listing exists
        let listing = dht.get_listing("listing:001");
        assert!(listing.is_some());

        // Verify event
        let events = dht.get_events_by_type("ListingCreated");
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_retired_credit_cannot_be_listed() {
        let mut dht = MockDHT::new();

        // Create a retired credit
        dht.credits.push(MockCarbonCredit {
            id: "credit:retired:001".to_string(),
            project_id: "project:test".to_string(),
            vintage_year: 2024,
            tonnes_co2e: 10.0,
            status: "Retired".to_string(),
            owner_did: "did:mycelix:seller".to_string(),
            retired_at: Some(1704326400),
        });

        // Try to list
        let listing = MockMarketplaceListing {
            listing_id: "listing:retired".to_string(),
            credit_id: "credit:retired:001".to_string(),
            seller_did: "did:mycelix:seller".to_string(),
            price_per_tonne: 25_00,
            available_tonnes: 10.0,
            is_active: true,
            expires_at: 1735689599,
        };

        assert!(!dht.create_listing(listing));
    }

    #[test]
    fn test_non_owner_cannot_list_credit() {
        let mut dht = MockDHT::new();

        // Create a credit owned by someone else
        dht.credits.push(MockCarbonCredit {
            id: "credit:other:001".to_string(),
            project_id: "project:test".to_string(),
            vintage_year: 2024,
            tonnes_co2e: 10.0,
            status: "Active".to_string(),
            owner_did: "did:mycelix:actual_owner".to_string(),
            retired_at: None,
        });

        // Try to list as different user
        let listing = MockMarketplaceListing {
            listing_id: "listing:unauthorized".to_string(),
            credit_id: "credit:other:001".to_string(),
            seller_did: "did:mycelix:not_owner".to_string(),
            price_per_tonne: 25_00,
            available_tonnes: 10.0,
            is_active: true,
            expires_at: 1735689599,
        };

        assert!(!dht.create_listing(listing));
    }

    #[test]
    fn test_credit_transfer_after_marketplace_sale() {
        let mut dht = MockDHT::new();

        // Create credit and list
        dht.credits.push(MockCarbonCredit {
            id: "credit:forsale:001".to_string(),
            project_id: "project:test".to_string(),
            vintage_year: 2024,
            tonnes_co2e: 10.0,
            status: "Active".to_string(),
            owner_did: "did:mycelix:seller".to_string(),
            retired_at: None,
        });

        let listing = MockMarketplaceListing {
            listing_id: "listing:forsale".to_string(),
            credit_id: "credit:forsale:001".to_string(),
            seller_did: "did:mycelix:seller".to_string(),
            price_per_tonne: 25_00,
            available_tonnes: 10.0,
            is_active: true,
            expires_at: 1735689599,
        };
        dht.create_listing(listing);

        // Simulate sale - transfer credit
        assert!(dht.transfer_credit("credit:forsale:001", "did:mycelix:buyer"));

        // Deactivate listing
        assert!(dht.deactivate_listing("listing:forsale"));

        // Verify ownership changed
        let credit = dht.get_credit("credit:forsale:001").unwrap();
        assert_eq!(credit.owner_did, "did:mycelix:buyer");

        // Verify listing inactive
        let listing = dht.get_listing("listing:forsale").unwrap();
        assert!(!listing.is_active);

        // Verify events
        let transfer_events = dht.get_events_by_type("CreditTransferred");
        assert_eq!(transfer_events.len(), 1);
    }

    // =========================================================================
    // BRIDGE -> CARBON VERIFICATION INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_verify_active_credit() {
        let mut dht = MockDHT::new();

        // Create active credit
        dht.credits.push(MockCarbonCredit {
            id: "credit:verify:001".to_string(),
            project_id: "project:test".to_string(),
            vintage_year: 2024,
            tonnes_co2e: 10.0,
            status: "Active".to_string(),
            owner_did: "did:mycelix:owner".to_string(),
            retired_at: None,
        });

        // Create verification query
        let query = MockVerificationQuery {
            query_id: "query:verify:001".to_string(),
            purpose: "CreditVerification".to_string(),
            target_id: "credit:verify:001".to_string(),
            status: "Pending".to_string(),
            result: None,
        };
        dht.create_verification_query(query);

        // Verify credit
        assert!(dht.verify_credit("query:verify:001", "credit:verify:001"));

        // Check result
        let query = dht.queries.iter().find(|q| q.query_id == "query:verify:001").unwrap();
        assert_eq!(query.status, "Completed");
        assert_eq!(query.result, Some("Verified".to_string()));
    }

    #[test]
    fn test_verify_retired_credit_fails() {
        let mut dht = MockDHT::new();

        // Create retired credit
        dht.credits.push(MockCarbonCredit {
            id: "credit:retired:verify".to_string(),
            project_id: "project:test".to_string(),
            vintage_year: 2024,
            tonnes_co2e: 10.0,
            status: "Retired".to_string(),
            owner_did: "did:mycelix:owner".to_string(),
            retired_at: Some(1704326400),
        });

        // Create verification query
        let query = MockVerificationQuery {
            query_id: "query:retired".to_string(),
            purpose: "CreditVerification".to_string(),
            target_id: "credit:retired:verify".to_string(),
            status: "Pending".to_string(),
            result: None,
        };
        dht.create_verification_query(query);

        // Verify credit
        dht.verify_credit("query:retired", "credit:retired:verify");

        // Check result
        let query = dht.queries.iter().find(|q| q.query_id == "query:retired").unwrap();
        assert_eq!(query.result, Some("Failed".to_string()));
    }

    #[test]
    fn test_verify_nonexistent_credit() {
        let mut dht = MockDHT::new();

        // Create verification query for nonexistent credit
        let query = MockVerificationQuery {
            query_id: "query:notfound".to_string(),
            purpose: "CreditVerification".to_string(),
            target_id: "credit:doesnt:exist".to_string(),
            status: "Pending".to_string(),
            result: None,
        };
        dht.create_verification_query(query);

        // Verify credit
        dht.verify_credit("query:notfound", "credit:doesnt:exist");

        // Check result
        let query = dht.queries.iter().find(|q| q.query_id == "query:notfound").unwrap();
        assert_eq!(query.result, Some("NotFound".to_string()));
    }

    #[test]
    fn test_project_due_diligence_active_project() {
        let mut dht = MockDHT::new();

        // Create and activate project
        let project = MockClimateProject {
            id: "project:duediligence".to_string(),
            name: "Due Diligence Test".to_string(),
            status: "Proposed".to_string(),
            expected_credits: 10000.0,
            verifier_did: None,
        };
        dht.create_project(project);
        dht.verify_project("project:duediligence", "did:mycelix:verifier");
        dht.activate_project("project:duediligence");

        // Create due diligence query
        let query = MockVerificationQuery {
            query_id: "query:dd".to_string(),
            purpose: "ProjectDueDiligence".to_string(),
            target_id: "project:duediligence".to_string(),
            status: "Pending".to_string(),
            result: None,
        };
        dht.create_verification_query(query);

        // Verify
        assert!(dht.verify_project_due_diligence("query:dd", "project:duediligence"));

        // Check result
        let query = dht.queries.iter().find(|q| q.query_id == "query:dd").unwrap();
        assert_eq!(query.result, Some("Verified".to_string()));
    }

    // =========================================================================
    // FULL LIFECYCLE INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_full_climate_lifecycle() {
        let mut dht = MockDHT::new();

        // 1. Create and verify project
        let project = MockClimateProject {
            id: "project:lifecycle".to_string(),
            name: "Full Lifecycle Test".to_string(),
            status: "Proposed".to_string(),
            expected_credits: 100.0,
            verifier_did: None,
        };
        dht.create_project(project);
        dht.verify_project("project:lifecycle", "did:mycelix:verifier");
        dht.activate_project("project:lifecycle");

        // 2. Create and complete milestone
        dht.create_milestone(MockMilestone {
            project_id: "project:lifecycle".to_string(),
            title: "First Milestone".to_string(),
            credits_issued: None,
            completed_at: None,
            verified_by: None,
        });
        dht.complete_milestone(
            "project:lifecycle",
            "First Milestone",
            "did:mycelix:verifier",
            50.0,
            1704326400000000,
        );

        // 3. Get issued credits
        let credits = dht.get_credits_by_project("project:lifecycle");
        assert!(!credits.is_empty());
        let first_credit_id = credits[0].id.clone();
        let first_credit_tonnes = credits[0].tonnes_co2e;

        // 4. Transfer credit ownership
        dht.transfer_credit(&first_credit_id, "did:mycelix:seller");

        // 5. List on marketplace
        let listing = MockMarketplaceListing {
            listing_id: "listing:lifecycle".to_string(),
            credit_id: first_credit_id.clone(),
            seller_did: "did:mycelix:seller".to_string(),
            price_per_tonne: 30_00,
            available_tonnes: first_credit_tonnes,
            is_active: true,
            expires_at: 1735689599,
        };
        assert!(dht.create_listing(listing));

        // 6. Verify credit via bridge
        let query = MockVerificationQuery {
            query_id: "query:lifecycle".to_string(),
            purpose: "CreditVerification".to_string(),
            target_id: first_credit_id.clone(),
            status: "Pending".to_string(),
            result: None,
        };
        dht.create_verification_query(query);
        dht.verify_credit("query:lifecycle", &first_credit_id);

        let query = dht.queries.iter().find(|q| q.query_id == "query:lifecycle").unwrap();
        assert_eq!(query.result, Some("Verified".to_string()));

        // 7. Simulate sale and transfer
        dht.transfer_credit(&first_credit_id, "did:mycelix:buyer");
        dht.deactivate_listing("listing:lifecycle");

        // 8. Buyer retires credit for offsetting
        dht.retire_credit(&first_credit_id, 1704499200000000);

        // 9. Verify full event chain
        let events = dht.get_events_for_subject(&first_credit_id);
        let event_types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();

        assert!(event_types.contains(&"CreditIssued"));
        assert!(event_types.contains(&"CreditTransferred"));
        assert!(event_types.contains(&"CreditRetired"));
    }

    #[test]
    fn test_offsetting_workflow() {
        let mut dht = MockDHT::new();

        // 1. Create organization footprint
        dht.create_footprint(MockCarbonFootprint {
            entity_did: "did:mycelix:organization".to_string(),
            total_emissions: 500.0, // 500 tonnes CO2
            verified_by: Some("did:mycelix:auditor".to_string()),
        });

        // 2. Setup project with credits
        let project = MockClimateProject {
            id: "project:offset".to_string(),
            name: "Offset Project".to_string(),
            status: "Proposed".to_string(),
            expected_credits: 1000.0,
            verifier_did: None,
        };
        dht.create_project(project);
        dht.verify_project("project:offset", "did:mycelix:verifier");
        dht.activate_project("project:offset");

        dht.create_milestone(MockMilestone {
            project_id: "project:offset".to_string(),
            title: "Credit Generation".to_string(),
            credits_issued: None,
            completed_at: None,
            verified_by: None,
        });
        dht.complete_milestone(
            "project:offset",
            "Credit Generation",
            "did:mycelix:verifier",
            600.0, // Issue 600 tonnes worth of credits
            1704326400000000,
        );

        // 3. Get credits and transfer to organization
        let credit_ids: Vec<String> = dht.get_credits_by_project("project:offset")
            .iter()
            .map(|c| c.id.clone())
            .collect();
        for credit_id in &credit_ids {
            dht.transfer_credit(credit_id, "did:mycelix:organization");
        }

        // 4. Calculate offset needed
        let emissions = dht.get_total_emissions("did:mycelix:organization");
        assert_eq!(emissions, 500.0);

        // 5. Retire credits to offset
        let org_credit_data: Vec<(String, f64)> = dht.get_credits_by_owner("did:mycelix:organization")
            .iter()
            .map(|c| (c.id.clone(), c.tonnes_co2e))
            .collect();
        let mut retired_tonnes = 0.0;

        for (credit_id, tonnes) in org_credit_data {
            if retired_tonnes >= 500.0 {
                break;
            }
            dht.retire_credit(&credit_id, 1704585600000000);
            retired_tonnes += tonnes;
        }

        // 6. Verify organization is carbon neutral
        assert!(retired_tonnes >= emissions);

        // 7. Check retired credits
        let retirement_events = dht.get_events_by_type("CreditRetired");
        assert!(!retirement_events.is_empty());
    }

    // =========================================================================
    // EVENT TRACKING TESTS
    // =========================================================================

    #[test]
    fn test_event_chronology() {
        let mut dht = MockDHT::new();

        // Create project
        let project = MockClimateProject {
            id: "project:events".to_string(),
            name: "Event Test".to_string(),
            status: "Proposed".to_string(),
            expected_credits: 100.0,
            verifier_did: None,
        };
        dht.create_project(project);

        // Verify
        dht.verify_project("project:events", "did:mycelix:verifier");

        // Activate
        dht.activate_project("project:events");

        // Get all events for project
        let events = dht.get_events_for_subject("project:events");

        // Verify chronological order
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_type, "ProjectCreated");
        assert_eq!(events[1].event_type, "ProjectVerified");
        assert_eq!(events[2].event_type, "ProjectActivated");

        // Verify timestamps increase
        assert!(events[1].timestamp > events[0].timestamp);
        assert!(events[2].timestamp > events[1].timestamp);
    }

    #[test]
    fn test_all_event_types_present() {
        let mut dht = MockDHT::new();

        // Full workflow
        let project = MockClimateProject {
            id: "project:allevents".to_string(),
            name: "All Events Test".to_string(),
            status: "Proposed".to_string(),
            expected_credits: 100.0,
            verifier_did: None,
        };
        dht.create_project(project);
        dht.verify_project("project:allevents", "did:mycelix:verifier");
        dht.activate_project("project:allevents");

        dht.create_milestone(MockMilestone {
            project_id: "project:allevents".to_string(),
            title: "Milestone".to_string(),
            credits_issued: None,
            completed_at: None,
            verified_by: None,
        });
        dht.complete_milestone(
            "project:allevents",
            "Milestone",
            "did:mycelix:verifier",
            50.0,
            1704326400000000,
        );

        let credits = dht.get_credits_by_project("project:allevents");
        let credit_id = credits[0].id.clone();

        dht.transfer_credit(&credit_id, "did:mycelix:seller");

        let listing = MockMarketplaceListing {
            listing_id: "listing:allevents".to_string(),
            credit_id: credit_id.clone(),
            seller_did: "did:mycelix:seller".to_string(),
            price_per_tonne: 25_00,
            available_tonnes: 10.0,
            is_active: true,
            expires_at: 1735689599,
        };
        dht.create_listing(listing);

        let query = MockVerificationQuery {
            query_id: "query:allevents".to_string(),
            purpose: "CreditVerification".to_string(),
            target_id: credit_id.clone(),
            status: "Pending".to_string(),
            result: None,
        };
        dht.create_verification_query(query);

        // Collect all event types
        let event_types: Vec<&str> = dht.events.iter().map(|e| e.event_type.as_str()).collect();

        // Verify all expected events exist
        assert!(event_types.contains(&"ProjectCreated"));
        assert!(event_types.contains(&"ProjectVerified"));
        assert!(event_types.contains(&"ProjectActivated"));
        assert!(event_types.contains(&"MilestoneCompleted"));
        assert!(event_types.contains(&"CreditIssued"));
        assert!(event_types.contains(&"CreditTransferred"));
        assert!(event_types.contains(&"ListingCreated"));
        assert!(event_types.contains(&"VerificationQueryCreated"));
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_concurrent_operations_same_credit() {
        let mut dht = MockDHT::new();

        // Create credit
        dht.credits.push(MockCarbonCredit {
            id: "credit:concurrent".to_string(),
            project_id: "project:test".to_string(),
            vintage_year: 2024,
            tonnes_co2e: 10.0,
            status: "Active".to_string(),
            owner_did: "did:mycelix:owner".to_string(),
            retired_at: None,
        });

        // List credit
        let listing = MockMarketplaceListing {
            listing_id: "listing:concurrent".to_string(),
            credit_id: "credit:concurrent".to_string(),
            seller_did: "did:mycelix:owner".to_string(),
            price_per_tonne: 25_00,
            available_tonnes: 10.0,
            is_active: true,
            expires_at: 1735689599,
        };
        dht.create_listing(listing);

        // Create verification query
        let query = MockVerificationQuery {
            query_id: "query:concurrent".to_string(),
            purpose: "CreditVerification".to_string(),
            target_id: "credit:concurrent".to_string(),
            status: "Pending".to_string(),
            result: None,
        };
        dht.create_verification_query(query);

        // Verification shows active
        dht.verify_credit("query:concurrent", "credit:concurrent");
        let query = dht.queries.iter().find(|q| q.query_id == "query:concurrent").unwrap();
        assert_eq!(query.result, Some("Verified".to_string()));

        // Transfer happens
        dht.transfer_credit("credit:concurrent", "did:mycelix:buyer");

        // Buyer retires
        dht.retire_credit("credit:concurrent", 1704585600000000);

        // Old listing should be deactivated (simulating proper workflow)
        dht.deactivate_listing("listing:concurrent");

        // Verify final state
        let credit = dht.get_credit("credit:concurrent").unwrap();
        assert_eq!(credit.status, "Retired");
        assert_eq!(credit.owner_did, "did:mycelix:buyer");

        let listing = dht.get_listing("listing:concurrent").unwrap();
        assert!(!listing.is_active);
    }

    #[test]
    fn test_large_scale_credit_issuance() {
        let mut dht = MockDHT::new();

        // Create large project
        let project = MockClimateProject {
            id: "project:large".to_string(),
            name: "Large Scale Project".to_string(),
            status: "Proposed".to_string(),
            expected_credits: 1_000_000.0,
            verifier_did: None,
        };
        dht.create_project(project);
        dht.verify_project("project:large", "did:mycelix:verifier");
        dht.activate_project("project:large");

        // Create milestone with large credit issuance
        dht.create_milestone(MockMilestone {
            project_id: "project:large".to_string(),
            title: "Large Issuance".to_string(),
            credits_issued: None,
            completed_at: None,
            verified_by: None,
        });
        dht.complete_milestone(
            "project:large",
            "Large Issuance",
            "did:mycelix:verifier",
            10000.0, // 10,000 tonnes
            1704326400000000,
        );

        // Verify credits were issued
        let credits = dht.get_credits_by_project("project:large");
        let total: f64 = credits.iter().map(|c| c.tonnes_co2e).sum();
        assert!((total - 10000.0).abs() < 0.001);
    }
}
