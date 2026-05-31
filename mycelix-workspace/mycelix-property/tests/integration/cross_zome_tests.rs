// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Cross-Zome Integration Tests
//!
//! Integration tests covering interactions between zomes:
//! - Transfer -> Registry ownership transfer flow
//! - Disputes -> Justice escalation
//! - Bridge event broadcasting across operations

use serde::{Deserialize, Serialize};

// =============================================================================
// Test Data Structures (minimal for integration testing)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockProperty {
    pub id: String,
    pub owner_did: String,
    pub last_transfer: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockTitleDeed {
    pub id: String,
    pub property_id: String,
    pub owner_did: String,
    pub deed_type: String,
    pub previous_deed_id: Option<String>,
    pub encumbrances: Vec<MockEncumbrance>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockEncumbrance {
    pub encumbrance_type: String,
    pub holder_did: String,
    pub amount: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockTransfer {
    pub id: String,
    pub property_id: String,
    pub from_did: String,
    pub to_did: String,
    pub status: String,
    pub conditions: Vec<MockCondition>,
    pub completed: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockCondition {
    pub condition_type: String,
    pub satisfied: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockDispute {
    pub id: String,
    pub property_id: String,
    pub claimant_did: String,
    pub respondent_did: String,
    pub status: String,
    pub justice_case_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MockBridgeEvent {
    pub event_type: String,
    pub property_id: String,
    pub subject_did: String,
    pub payload: String,
    pub timestamp: i64,
}

// =============================================================================
// Mock State Management (simulates DHT state)
// =============================================================================

struct MockDHT {
    properties: Vec<MockProperty>,
    deeds: Vec<MockTitleDeed>,
    transfers: Vec<MockTransfer>,
    disputes: Vec<MockDispute>,
    events: Vec<MockBridgeEvent>,
}

impl MockDHT {
    fn new() -> Self {
        Self {
            properties: vec![],
            deeds: vec![],
            transfers: vec![],
            disputes: vec![],
            events: vec![],
        }
    }

    // Registry operations
    fn register_property(&mut self, property: MockProperty, deed: MockTitleDeed) {
        self.properties.push(property.clone());
        self.deeds.push(deed);
        self.events.push(MockBridgeEvent {
            event_type: "PropertyRegistered".to_string(),
            property_id: property.id.clone(),
            subject_did: property.owner_did.clone(),
            payload: format!(r#"{{"owner_did":"{}"}}"#, property.owner_did),
            timestamp: 1704067200000000,
        });
    }

    fn get_property(&self, property_id: &str) -> Option<&MockProperty> {
        self.properties.iter().find(|p| p.id == property_id)
    }

    fn get_deed(&self, property_id: &str) -> Option<&MockTitleDeed> {
        self.deeds.iter().find(|d| d.property_id == property_id)
    }

    fn update_property_owner(&mut self, property_id: &str, new_owner: &str, transfer_time: i64) {
        if let Some(property) = self.properties.iter_mut().find(|p| p.id == property_id) {
            let old_owner = property.owner_did.clone();
            property.owner_did = new_owner.to_string();
            property.last_transfer = Some(transfer_time);

            // Emit bridge event
            self.events.push(MockBridgeEvent {
                event_type: "OwnershipTransferred".to_string(),
                property_id: property_id.to_string(),
                subject_did: new_owner.to_string(),
                payload: format!(r#"{{"from_did":"{}","to_did":"{}"}}"#, old_owner, new_owner),
                timestamp: transfer_time,
            });
        }
    }

    fn create_transfer_deed(
        &mut self,
        property_id: &str,
        new_owner: &str,
        previous_deed_id: &str,
        carry_encumbrances: Vec<MockEncumbrance>,
    ) -> MockTitleDeed {
        let deed = MockTitleDeed {
            id: format!("deed:{}:transfer:{}", property_id, 1704200000000000i64),
            property_id: property_id.to_string(),
            owner_did: new_owner.to_string(),
            deed_type: "Transfer".to_string(),
            previous_deed_id: Some(previous_deed_id.to_string()),
            encumbrances: carry_encumbrances,
        };
        self.deeds.push(deed.clone());
        deed
    }

    // Transfer operations
    fn initiate_transfer(&mut self, transfer: MockTransfer) {
        self.transfers.push(transfer);
    }

    fn get_transfer(&self, transfer_id: &str) -> Option<&MockTransfer> {
        self.transfers.iter().find(|t| t.id == transfer_id)
    }

    fn satisfy_transfer_condition(&mut self, transfer_id: &str, condition_index: usize) {
        if let Some(transfer) = self.transfers.iter_mut().find(|t| t.id == transfer_id) {
            if let Some(condition) = transfer.conditions.get_mut(condition_index) {
                condition.satisfied = true;
            }
        }
    }

    fn complete_transfer(&mut self, transfer_id: &str, completion_time: i64) -> bool {
        // First, extract all the data we need while doing read-only operations
        let transfer_data = {
            let transfer = self.transfers.iter().find(|t| t.id == transfer_id);
            match transfer {
                Some(t) => {
                    // Check all conditions satisfied
                    if !t.conditions.iter().all(|c| c.satisfied) {
                        return false;
                    }
                    Some((t.property_id.clone(), t.to_did.clone()))
                }
                None => None,
            }
        };

        let (property_id, to_did) = match transfer_data {
            Some(data) => data,
            None => return false,
        };

        // Get current deed for encumbrances
        let encumbrances = self
            .get_deed(&property_id)
            .map(|d| d.encumbrances.clone())
            .unwrap_or_default();

        // Get previous deed ID
        let previous_deed_id = self
            .get_deed(&property_id)
            .map(|d| d.id.clone())
            .unwrap_or_default();

        // Update property owner
        self.update_property_owner(&property_id, &to_did, completion_time);

        // Create new deed
        self.create_transfer_deed(&property_id, &to_did, &previous_deed_id, encumbrances);

        // Update transfer status
        if let Some(transfer) = self.transfers.iter_mut().find(|t| t.id == transfer_id) {
            transfer.status = "Completed".to_string();
            transfer.completed = Some(completion_time);
        }

        true
    }

    // Dispute operations
    fn file_dispute(&mut self, dispute: MockDispute) {
        self.events.push(MockBridgeEvent {
            event_type: "DisputeFiled".to_string(),
            property_id: dispute.property_id.clone(),
            subject_did: dispute.claimant_did.clone(),
            payload: format!(
                r#"{{"claimant":"{}","respondent":"{}"}}"#,
                dispute.claimant_did, dispute.respondent_did
            ),
            timestamp: 1704067200000000,
        });
        self.disputes.push(dispute);
    }

    fn get_dispute(&self, dispute_id: &str) -> Option<&MockDispute> {
        self.disputes.iter().find(|d| d.id == dispute_id)
    }

    fn escalate_to_justice(&mut self, dispute_id: &str, justice_case_id: &str) {
        if let Some(dispute) = self.disputes.iter_mut().find(|d| d.id == dispute_id) {
            dispute.status = "Arbitration".to_string();
            dispute.justice_case_id = Some(justice_case_id.to_string());
        }
    }

    fn get_events_for_property(&self, property_id: &str) -> Vec<&MockBridgeEvent> {
        self.events
            .iter()
            .filter(|e| e.property_id == property_id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TRANSFER -> REGISTRY INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_complete_transfer_updates_registry() {
        let mut dht = MockDHT::new();

        // 1. Register initial property
        let property = MockProperty {
            id: "property:1".to_string(),
            owner_did: "did:mycelix:seller".to_string(),
            last_transfer: None,
        };
        let deed = MockTitleDeed {
            id: "deed:1:original".to_string(),
            property_id: "property:1".to_string(),
            owner_did: "did:mycelix:seller".to_string(),
            deed_type: "Original".to_string(),
            previous_deed_id: None,
            encumbrances: vec![],
        };
        dht.register_property(property, deed);

        // 2. Initiate transfer
        let transfer = MockTransfer {
            id: "transfer:1".to_string(),
            property_id: "property:1".to_string(),
            from_did: "did:mycelix:seller".to_string(),
            to_did: "did:mycelix:buyer".to_string(),
            status: "Initiated".to_string(),
            conditions: vec![MockCondition {
                condition_type: "PaymentReceived".to_string(),
                satisfied: false,
            }],
            completed: None,
        };
        dht.initiate_transfer(transfer);

        // 3. Satisfy conditions
        dht.satisfy_transfer_condition("transfer:1", 0);

        // 4. Complete transfer (calls registry to update ownership)
        let completion_time = 1704200000000000i64;
        let success = dht.complete_transfer("transfer:1", completion_time);
        assert!(success);

        // 5. Verify registry updated
        let updated_property = dht.get_property("property:1").unwrap();
        assert_eq!(updated_property.owner_did, "did:mycelix:buyer");
        assert_eq!(updated_property.last_transfer, Some(completion_time));

        // 6. Verify new deed created
        let deeds: Vec<_> = dht
            .deeds
            .iter()
            .filter(|d| d.property_id == "property:1")
            .collect();
        assert_eq!(deeds.len(), 2); // Original + Transfer deed
        let transfer_deed = deeds.iter().find(|d| d.deed_type == "Transfer").unwrap();
        assert_eq!(transfer_deed.owner_did, "did:mycelix:buyer");
        assert!(transfer_deed.previous_deed_id.is_some());

        // 7. Verify transfer completed
        let completed_transfer = dht.get_transfer("transfer:1").unwrap();
        assert_eq!(completed_transfer.status, "Completed");
        assert!(completed_transfer.completed.is_some());
    }

    #[test]
    fn test_transfer_carries_forward_encumbrances() {
        let mut dht = MockDHT::new();

        // 1. Register property with encumbrance
        let property = MockProperty {
            id: "property:enc".to_string(),
            owner_did: "did:mycelix:seller".to_string(),
            last_transfer: None,
        };
        let deed = MockTitleDeed {
            id: "deed:enc:original".to_string(),
            property_id: "property:enc".to_string(),
            owner_did: "did:mycelix:seller".to_string(),
            deed_type: "Original".to_string(),
            previous_deed_id: None,
            encumbrances: vec![MockEncumbrance {
                encumbrance_type: "Mortgage".to_string(),
                holder_did: "did:mycelix:bank".to_string(),
                amount: Some(200000.0),
            }],
        };
        dht.register_property(property, deed);

        // 2. Initiate and complete transfer
        let transfer = MockTransfer {
            id: "transfer:enc".to_string(),
            property_id: "property:enc".to_string(),
            from_did: "did:mycelix:seller".to_string(),
            to_did: "did:mycelix:buyer".to_string(),
            status: "Initiated".to_string(),
            conditions: vec![], // No conditions for simplicity
            completed: None,
        };
        dht.initiate_transfer(transfer);
        dht.complete_transfer("transfer:enc", 1704200000000000);

        // 3. Verify encumbrances carried forward to new deed
        let new_deed = dht
            .deeds
            .iter()
            .filter(|d| d.property_id == "property:enc" && d.deed_type == "Transfer")
            .last()
            .unwrap();
        assert_eq!(new_deed.encumbrances.len(), 1);
        assert_eq!(new_deed.encumbrances[0].encumbrance_type, "Mortgage");
        assert_eq!(new_deed.encumbrances[0].holder_did, "did:mycelix:bank");
    }

    #[test]
    fn test_transfer_fails_with_unsatisfied_conditions() {
        let mut dht = MockDHT::new();

        // 1. Register property
        let property = MockProperty {
            id: "property:cond".to_string(),
            owner_did: "did:mycelix:seller".to_string(),
            last_transfer: None,
        };
        let deed = MockTitleDeed {
            id: "deed:cond:original".to_string(),
            property_id: "property:cond".to_string(),
            owner_did: "did:mycelix:seller".to_string(),
            deed_type: "Original".to_string(),
            previous_deed_id: None,
            encumbrances: vec![],
        };
        dht.register_property(property, deed);

        // 2. Initiate transfer with unsatisfied condition
        let transfer = MockTransfer {
            id: "transfer:cond".to_string(),
            property_id: "property:cond".to_string(),
            from_did: "did:mycelix:seller".to_string(),
            to_did: "did:mycelix:buyer".to_string(),
            status: "Initiated".to_string(),
            conditions: vec![MockCondition {
                condition_type: "PaymentReceived".to_string(),
                satisfied: false,
            }],
            completed: None,
        };
        dht.initiate_transfer(transfer);

        // 3. Try to complete without satisfying conditions
        let success = dht.complete_transfer("transfer:cond", 1704200000000000);
        assert!(!success);

        // 4. Verify property not transferred
        let property = dht.get_property("property:cond").unwrap();
        assert_eq!(property.owner_did, "did:mycelix:seller");
    }

    #[test]
    fn test_deed_chain_integrity() {
        let mut dht = MockDHT::new();

        // 1. Register original property
        let property = MockProperty {
            id: "property:chain".to_string(),
            owner_did: "did:mycelix:owner1".to_string(),
            last_transfer: None,
        };
        let original_deed = MockTitleDeed {
            id: "deed:chain:original".to_string(),
            property_id: "property:chain".to_string(),
            owner_did: "did:mycelix:owner1".to_string(),
            deed_type: "Original".to_string(),
            previous_deed_id: None,
            encumbrances: vec![],
        };
        dht.register_property(property, original_deed);

        // 2. First transfer
        let transfer1 = MockTransfer {
            id: "transfer:chain:1".to_string(),
            property_id: "property:chain".to_string(),
            from_did: "did:mycelix:owner1".to_string(),
            to_did: "did:mycelix:owner2".to_string(),
            status: "Initiated".to_string(),
            conditions: vec![],
            completed: None,
        };
        dht.initiate_transfer(transfer1);
        dht.complete_transfer("transfer:chain:1", 1704100000000000);

        // 3. Second transfer
        let transfer2 = MockTransfer {
            id: "transfer:chain:2".to_string(),
            property_id: "property:chain".to_string(),
            from_did: "did:mycelix:owner2".to_string(),
            to_did: "did:mycelix:owner3".to_string(),
            status: "Initiated".to_string(),
            conditions: vec![],
            completed: None,
        };
        dht.initiate_transfer(transfer2);
        dht.complete_transfer("transfer:chain:2", 1704200000000000);

        // 4. Verify deed chain
        let deeds: Vec<_> = dht
            .deeds
            .iter()
            .filter(|d| d.property_id == "property:chain")
            .collect();
        assert_eq!(deeds.len(), 3);

        // Original deed has no previous
        let original = deeds.iter().find(|d| d.deed_type == "Original").unwrap();
        assert!(original.previous_deed_id.is_none());

        // Transfer deeds have previous deed references
        let transfer_deeds: Vec<_> = deeds.iter().filter(|d| d.deed_type == "Transfer").collect();
        assert_eq!(transfer_deeds.len(), 2);
        for td in transfer_deeds {
            assert!(td.previous_deed_id.is_some());
        }
    }

    // =========================================================================
    // DISPUTE -> JUSTICE INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_dispute_escalation_to_justice() {
        let mut dht = MockDHT::new();

        // 1. File dispute
        let dispute = MockDispute {
            id: "dispute:1".to_string(),
            property_id: "property:disputed".to_string(),
            claimant_did: "did:mycelix:claimant".to_string(),
            respondent_did: "did:mycelix:respondent".to_string(),
            status: "Filed".to_string(),
            justice_case_id: None,
        };
        dht.file_dispute(dispute);

        // 2. Verify initial state
        let filed_dispute = dht.get_dispute("dispute:1").unwrap();
        assert_eq!(filed_dispute.status, "Filed");
        assert!(filed_dispute.justice_case_id.is_none());

        // 3. Escalate to justice
        dht.escalate_to_justice("dispute:1", "JUSTICE-CASE-001");

        // 4. Verify escalation
        let escalated_dispute = dht.get_dispute("dispute:1").unwrap();
        assert_eq!(escalated_dispute.status, "Arbitration");
        assert_eq!(
            escalated_dispute.justice_case_id,
            Some("JUSTICE-CASE-001".to_string())
        );
    }

    #[test]
    fn test_dispute_creates_bridge_event() {
        let mut dht = MockDHT::new();

        // 1. File dispute
        let dispute = MockDispute {
            id: "dispute:event".to_string(),
            property_id: "property:event".to_string(),
            claimant_did: "did:mycelix:claimant".to_string(),
            respondent_did: "did:mycelix:respondent".to_string(),
            status: "Filed".to_string(),
            justice_case_id: None,
        };
        dht.file_dispute(dispute);

        // 2. Verify bridge event created
        let events = dht.get_events_for_property("property:event");
        assert!(!events.is_empty());
        let dispute_event = events
            .iter()
            .find(|e| e.event_type == "DisputeFiled")
            .unwrap();
        assert!(dispute_event.payload.contains("claimant"));
        assert!(dispute_event.payload.contains("respondent"));
    }

    // =========================================================================
    // BRIDGE EVENT BROADCASTING TESTS
    // =========================================================================

    #[test]
    fn test_property_registration_broadcasts_event() {
        let mut dht = MockDHT::new();

        // 1. Register property
        let property = MockProperty {
            id: "property:broadcast".to_string(),
            owner_did: "did:mycelix:owner".to_string(),
            last_transfer: None,
        };
        let deed = MockTitleDeed {
            id: "deed:broadcast:original".to_string(),
            property_id: "property:broadcast".to_string(),
            owner_did: "did:mycelix:owner".to_string(),
            deed_type: "Original".to_string(),
            previous_deed_id: None,
            encumbrances: vec![],
        };
        dht.register_property(property, deed);

        // 2. Verify event
        let events = dht.get_events_for_property("property:broadcast");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "PropertyRegistered");
        assert_eq!(events[0].subject_did, "did:mycelix:owner");
    }

    #[test]
    fn test_ownership_transfer_broadcasts_event() {
        let mut dht = MockDHT::new();

        // 1. Setup
        let property = MockProperty {
            id: "property:transfer:event".to_string(),
            owner_did: "did:mycelix:seller".to_string(),
            last_transfer: None,
        };
        let deed = MockTitleDeed {
            id: "deed:transfer:event:original".to_string(),
            property_id: "property:transfer:event".to_string(),
            owner_did: "did:mycelix:seller".to_string(),
            deed_type: "Original".to_string(),
            previous_deed_id: None,
            encumbrances: vec![],
        };
        dht.register_property(property, deed);

        let transfer = MockTransfer {
            id: "transfer:event".to_string(),
            property_id: "property:transfer:event".to_string(),
            from_did: "did:mycelix:seller".to_string(),
            to_did: "did:mycelix:buyer".to_string(),
            status: "Initiated".to_string(),
            conditions: vec![],
            completed: None,
        };
        dht.initiate_transfer(transfer);

        // 2. Complete transfer
        dht.complete_transfer("transfer:event", 1704200000000000);

        // 3. Verify events
        let events = dht.get_events_for_property("property:transfer:event");
        assert!(events.len() >= 2); // Registration + Transfer

        let transfer_event = events
            .iter()
            .find(|e| e.event_type == "OwnershipTransferred")
            .unwrap();
        assert!(transfer_event.payload.contains("from_did"));
        assert!(transfer_event.payload.contains("to_did"));
        assert!(transfer_event.payload.contains("did:mycelix:seller"));
        assert!(transfer_event.payload.contains("did:mycelix:buyer"));
    }

    #[test]
    fn test_full_property_lifecycle_events() {
        let mut dht = MockDHT::new();

        // 1. Register
        let property = MockProperty {
            id: "property:lifecycle".to_string(),
            owner_did: "did:mycelix:original".to_string(),
            last_transfer: None,
        };
        let deed = MockTitleDeed {
            id: "deed:lifecycle:original".to_string(),
            property_id: "property:lifecycle".to_string(),
            owner_did: "did:mycelix:original".to_string(),
            deed_type: "Original".to_string(),
            previous_deed_id: None,
            encumbrances: vec![],
        };
        dht.register_property(property, deed);

        // 2. Transfer
        let transfer = MockTransfer {
            id: "transfer:lifecycle".to_string(),
            property_id: "property:lifecycle".to_string(),
            from_did: "did:mycelix:original".to_string(),
            to_did: "did:mycelix:new".to_string(),
            status: "Initiated".to_string(),
            conditions: vec![],
            completed: None,
        };
        dht.initiate_transfer(transfer);
        dht.complete_transfer("transfer:lifecycle", 1704100000000000);

        // 3. Dispute
        let dispute = MockDispute {
            id: "dispute:lifecycle".to_string(),
            property_id: "property:lifecycle".to_string(),
            claimant_did: "did:mycelix:neighbor".to_string(),
            respondent_did: "did:mycelix:new".to_string(),
            status: "Filed".to_string(),
            justice_case_id: None,
        };
        dht.file_dispute(dispute);

        // 4. Verify complete event history
        let events = dht.get_events_for_property("property:lifecycle");
        assert_eq!(events.len(), 3);

        let event_types: Vec<_> = events.iter().map(|e| e.event_type.as_str()).collect();
        assert!(event_types.contains(&"PropertyRegistered"));
        assert!(event_types.contains(&"OwnershipTransferred"));
        assert!(event_types.contains(&"DisputeFiled"));
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_multiple_transfers_same_property() {
        let mut dht = MockDHT::new();

        // Setup
        let property = MockProperty {
            id: "property:multi".to_string(),
            owner_did: "did:mycelix:owner1".to_string(),
            last_transfer: None,
        };
        let deed = MockTitleDeed {
            id: "deed:multi:original".to_string(),
            property_id: "property:multi".to_string(),
            owner_did: "did:mycelix:owner1".to_string(),
            deed_type: "Original".to_string(),
            previous_deed_id: None,
            encumbrances: vec![],
        };
        dht.register_property(property, deed);

        // Multiple sequential transfers
        let owners = vec!["owner1", "owner2", "owner3", "owner4"];
        for i in 0..(owners.len() - 1) {
            let transfer = MockTransfer {
                id: format!("transfer:multi:{}", i),
                property_id: "property:multi".to_string(),
                from_did: format!("did:mycelix:{}", owners[i]),
                to_did: format!("did:mycelix:{}", owners[i + 1]),
                status: "Initiated".to_string(),
                conditions: vec![],
                completed: None,
            };
            dht.initiate_transfer(transfer);
            dht.complete_transfer(
                &format!("transfer:multi:{}", i),
                1704000000000000 + (i as i64) * 100000000,
            );
        }

        // Verify final owner
        let property = dht.get_property("property:multi").unwrap();
        assert_eq!(property.owner_did, "did:mycelix:owner4");

        // Verify deed chain length
        let deeds: Vec<_> = dht
            .deeds
            .iter()
            .filter(|d| d.property_id == "property:multi")
            .collect();
        assert_eq!(deeds.len(), 4); // 1 original + 3 transfers

        // Verify events
        let events = dht.get_events_for_property("property:multi");
        let transfer_events: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == "OwnershipTransferred")
            .collect();
        assert_eq!(transfer_events.len(), 3);
    }

    #[test]
    fn test_concurrent_dispute_and_transfer() {
        let mut dht = MockDHT::new();

        // Setup
        let property = MockProperty {
            id: "property:concurrent".to_string(),
            owner_did: "did:mycelix:owner".to_string(),
            last_transfer: None,
        };
        let deed = MockTitleDeed {
            id: "deed:concurrent:original".to_string(),
            property_id: "property:concurrent".to_string(),
            owner_did: "did:mycelix:owner".to_string(),
            deed_type: "Original".to_string(),
            previous_deed_id: None,
            encumbrances: vec![],
        };
        dht.register_property(property, deed);

        // File dispute
        let dispute = MockDispute {
            id: "dispute:concurrent".to_string(),
            property_id: "property:concurrent".to_string(),
            claimant_did: "did:mycelix:claimant".to_string(),
            respondent_did: "did:mycelix:owner".to_string(),
            status: "Filed".to_string(),
            justice_case_id: None,
        };
        dht.file_dispute(dispute);

        // Initiate transfer while dispute active
        let transfer = MockTransfer {
            id: "transfer:concurrent".to_string(),
            property_id: "property:concurrent".to_string(),
            from_did: "did:mycelix:owner".to_string(),
            to_did: "did:mycelix:buyer".to_string(),
            status: "Initiated".to_string(),
            conditions: vec![],
            completed: None,
        };
        dht.initiate_transfer(transfer);

        // Both dispute and transfer exist for same property
        let dispute = dht.get_dispute("dispute:concurrent").unwrap();
        let transfer = dht.get_transfer("transfer:concurrent").unwrap();
        assert_eq!(dispute.property_id, transfer.property_id);
        // In real implementation, transfer might be blocked until dispute resolved
    }

    #[test]
    fn test_encumbrance_persistence_across_multiple_transfers() {
        let mut dht = MockDHT::new();

        // Setup with encumbrance
        let property = MockProperty {
            id: "property:enc:persist".to_string(),
            owner_did: "did:mycelix:owner1".to_string(),
            last_transfer: None,
        };
        let deed = MockTitleDeed {
            id: "deed:enc:persist:original".to_string(),
            property_id: "property:enc:persist".to_string(),
            owner_did: "did:mycelix:owner1".to_string(),
            deed_type: "Original".to_string(),
            previous_deed_id: None,
            encumbrances: vec![MockEncumbrance {
                encumbrance_type: "Mortgage".to_string(),
                holder_did: "did:mycelix:bank".to_string(),
                amount: Some(300000.0),
            }],
        };
        dht.register_property(property, deed);

        // Two transfers
        for i in 1..=2 {
            let transfer = MockTransfer {
                id: format!("transfer:enc:persist:{}", i),
                property_id: "property:enc:persist".to_string(),
                from_did: format!("did:mycelix:owner{}", i),
                to_did: format!("did:mycelix:owner{}", i + 1),
                status: "Initiated".to_string(),
                conditions: vec![],
                completed: None,
            };
            dht.initiate_transfer(transfer);
            dht.complete_transfer(
                &format!("transfer:enc:persist:{}", i),
                1704000000000000 + (i as i64) * 100000000,
            );
        }

        // Verify encumbrance still present on latest deed
        let latest_deed = dht
            .deeds
            .iter()
            .filter(|d| d.property_id == "property:enc:persist")
            .last()
            .unwrap();
        assert_eq!(latest_deed.encumbrances.len(), 1);
        assert_eq!(latest_deed.encumbrances[0].encumbrance_type, "Mortgage");
    }
}
