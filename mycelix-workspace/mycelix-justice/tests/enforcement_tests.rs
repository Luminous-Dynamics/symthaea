// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Enforcement Tests
//!
//! Tests for enforcement action creation, execution, status tracking,
//! and cross-hApp enforcement in the mycelix-justice system.

#[macro_use]
extern crate lazy_static;

#[cfg(test)]
mod enforcement_creation_tests {
    use super::*;

    // ==========================================
    // Enforcement Creation Tests
    // ==========================================

    #[test]
    fn test_create_enforcement_action() {
        let enforcement = create_test_enforcement(
            "enforcement-001",
            "decision-001",
            0, // remedy index
            "did:mycelix:system_enforcer",
        );

        let result = simulate_create_enforcement(enforcement);
        assert!(result.is_ok(), "Creating valid enforcement should succeed");
    }

    #[test]
    fn test_enforcement_creates_decision_link() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let decision_enforcements = simulate_get_decision_enforcement("decision-001");
        assert!(
            decision_enforcements
                .iter()
                .any(|e| e.id == "enforcement-001")
        );
    }

    #[test]
    fn test_enforcement_initial_status() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");

        let result = simulate_create_enforcement(enforcement).unwrap();
        assert_eq!(result.status, EnforcementStatus::Pending);
    }

    #[test]
    fn test_create_multiple_enforcements_for_decision() {
        // A decision may have multiple remedies requiring separate enforcement
        let enforcement1 =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        let enforcement2 =
            create_test_enforcement("enforcement-002", "decision-001", 1, "did:mycelix:enforcer");
        let enforcement3 =
            create_test_enforcement("enforcement-003", "decision-001", 2, "did:mycelix:enforcer");

        simulate_create_enforcement(enforcement1).unwrap();
        simulate_create_enforcement(enforcement2).unwrap();
        simulate_create_enforcement(enforcement3).unwrap();

        let decision_enforcements = simulate_get_decision_enforcement("decision-001");
        assert_eq!(decision_enforcements.len(), 3);
    }
}

#[cfg(test)]
mod enforcement_status_tests {
    use super::*;

    // ==========================================
    // Enforcement Status Transition Tests
    // ==========================================

    #[test]
    fn test_status_pending_to_in_progress() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let result =
            simulate_update_enforcement_status("enforcement-001", EnforcementStatus::InProgress);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, EnforcementStatus::InProgress);
    }

    #[test]
    fn test_status_in_progress_to_partially_completed() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();
        simulate_update_enforcement_status("enforcement-001", EnforcementStatus::InProgress)
            .unwrap();

        let result = simulate_update_enforcement_status(
            "enforcement-001",
            EnforcementStatus::PartiallyCompleted,
        );
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap().status,
            EnforcementStatus::PartiallyCompleted
        );
    }

    #[test]
    fn test_status_to_completed() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();
        simulate_update_enforcement_status("enforcement-001", EnforcementStatus::InProgress)
            .unwrap();

        let result =
            simulate_update_enforcement_status("enforcement-001", EnforcementStatus::Completed);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, EnforcementStatus::Completed);
    }

    #[test]
    fn test_status_completed_sets_timestamp() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let result = simulate_complete_enforcement("enforcement-001", None);
        assert!(result.is_ok());

        let completed = result.unwrap();
        assert!(completed.completed_at.is_some());
    }

    #[test]
    fn test_status_to_failed() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();
        simulate_update_enforcement_status("enforcement-001", EnforcementStatus::InProgress)
            .unwrap();

        let result =
            simulate_update_enforcement_status("enforcement-001", EnforcementStatus::Failed);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, EnforcementStatus::Failed);
    }

    #[test]
    fn test_status_to_contested() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let result =
            simulate_update_enforcement_status("enforcement-001", EnforcementStatus::Contested);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, EnforcementStatus::Contested);
    }

    #[test]
    fn test_get_pending_enforcements() {
        simulate_create_enforcement(create_test_enforcement(
            "e-001",
            "d-001",
            0,
            "did:mycelix:e",
        ))
        .unwrap();
        simulate_create_enforcement(create_test_enforcement(
            "e-002",
            "d-002",
            0,
            "did:mycelix:e",
        ))
        .unwrap();
        simulate_create_enforcement(create_test_enforcement(
            "e-003",
            "d-003",
            0,
            "did:mycelix:e",
        ))
        .unwrap();

        simulate_update_enforcement_status("e-002", EnforcementStatus::InProgress).unwrap();
        simulate_update_enforcement_status("e-003", EnforcementStatus::Completed).unwrap();

        let pending = simulate_get_pending_enforcements();
        // Should include pending and in_progress
        assert!(pending.iter().any(|e| e.id == "e-001"));
        assert!(pending.iter().any(|e| e.id == "e-002"));
        assert!(!pending.iter().any(|e| e.id == "e-003"));
    }

    #[test]
    fn test_get_enforcements_by_status() {
        simulate_create_enforcement(create_test_enforcement(
            "e-001",
            "d-001",
            0,
            "did:mycelix:e",
        ))
        .unwrap();
        simulate_create_enforcement(create_test_enforcement(
            "e-002",
            "d-002",
            0,
            "did:mycelix:e",
        ))
        .unwrap();
        simulate_update_enforcement_status("e-002", EnforcementStatus::Completed).unwrap();

        let pending = simulate_get_enforcements_by_status(EnforcementStatus::Pending);
        let completed = simulate_get_enforcements_by_status(EnforcementStatus::Completed);

        assert!(pending.iter().any(|e| e.id == "e-001"));
        assert!(completed.iter().any(|e| e.id == "e-002"));
    }
}

#[cfg(test)]
mod enforcement_action_tests {
    use super::*;

    // ==========================================
    // Enforcement Action Recording Tests
    // ==========================================

    #[test]
    fn test_record_enforcement_action() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let action = EnforcementAction {
            action_type: EnforcementActionType::Notification,
            target_happ: None,
            target_entry: None,
            result: "Notification sent to respondent".to_string(),
        };

        let result = simulate_record_action("enforcement-001", action);
        assert!(result.is_ok());
        assert!(!result.unwrap().actions.is_empty());
    }

    #[test]
    fn test_record_multiple_actions() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        simulate_record_action(
            "enforcement-001",
            EnforcementAction {
                action_type: EnforcementActionType::Notification,
                target_happ: None,
                target_entry: None,
                result: "Initial notification".to_string(),
            },
        )
        .unwrap();

        simulate_record_action(
            "enforcement-001",
            EnforcementAction {
                action_type: EnforcementActionType::FundsTransfer,
                target_happ: Some("mycelix-wallet".to_string()),
                target_entry: Some("txn-123".to_string()),
                result: "1000 HBAR transferred".to_string(),
            },
        )
        .unwrap();

        let updated = simulate_get_enforcement("enforcement-001").unwrap();
        assert_eq!(updated.actions.len(), 2);
    }

    #[test]
    fn test_record_funds_transfer_action() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let action = EnforcementAction {
            action_type: EnforcementActionType::FundsTransfer,
            target_happ: Some("mycelix-wallet".to_string()),
            target_entry: Some("txn-abc123".to_string()),
            result: "1500 HBAR transferred from did:mycelix:bob to did:mycelix:alice".to_string(),
        };

        let result = simulate_record_action("enforcement-001", action);
        assert!(result.is_ok());
    }

    #[test]
    fn test_record_asset_freeze_action() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let action = EnforcementAction {
            action_type: EnforcementActionType::AssetFreeze,
            target_happ: Some("mycelix-marketplace".to_string()),
            target_entry: Some("asset-789".to_string()),
            result: "Asset frozen pending resolution".to_string(),
        };

        let result = simulate_record_action("enforcement-001", action);
        assert!(result.is_ok());
    }

    #[test]
    fn test_record_reputation_update_action() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let action = EnforcementAction {
            action_type: EnforcementActionType::ReputationUpdate,
            target_happ: Some("mycelix-trust".to_string()),
            target_entry: None,
            result: "MATL score reduced by 50 points for did:mycelix:bob".to_string(),
        };

        let result = simulate_record_action("enforcement-001", action);
        assert!(result.is_ok());
    }

    #[test]
    fn test_record_access_revocation_action() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let action = EnforcementAction {
            action_type: EnforcementActionType::AccessRevocation,
            target_happ: Some("mycelix-governance".to_string()),
            target_entry: None,
            result: "Voting rights suspended for 90 days".to_string(),
        };

        let result = simulate_record_action("enforcement-001", action);
        assert!(result.is_ok());
    }

    #[test]
    fn test_record_manual_required_action() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let action = EnforcementAction {
            action_type: EnforcementActionType::ManualRequired,
            target_happ: None,
            target_entry: None,
            result: "Public apology must be posted to community forum".to_string(),
        };

        let result = simulate_record_action("enforcement-001", action);
        assert!(result.is_ok());
    }
}

#[cfg(test)]
mod enforcement_completion_tests {
    use super::*;

    // ==========================================
    // Enforcement Completion Tests
    // ==========================================

    #[test]
    fn test_complete_enforcement() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let result = simulate_complete_enforcement("enforcement-001", None);
        assert!(result.is_ok());

        let completed = result.unwrap();
        assert_eq!(completed.status, EnforcementStatus::Completed);
        assert!(completed.completed_at.is_some());
    }

    #[test]
    fn test_complete_enforcement_with_final_action() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let final_action = EnforcementAction {
            action_type: EnforcementActionType::Notification,
            target_happ: None,
            target_entry: None,
            result: "Final completion notification sent to all parties".to_string(),
        };

        let result = simulate_complete_enforcement("enforcement-001", Some(final_action));
        assert!(result.is_ok());

        let completed = result.unwrap();
        assert!(
            completed
                .actions
                .iter()
                .any(|a| a.result.contains("Final completion"))
        );
    }

    #[test]
    fn test_mark_enforcement_failed() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let result = simulate_mark_enforcement_failed(
            "enforcement-001",
            "Unable to locate respondent's assets for seizure",
        );
        assert!(result.is_ok());

        let failed = result.unwrap();
        assert_eq!(failed.status, EnforcementStatus::Failed);
        assert!(
            failed
                .actions
                .iter()
                .any(|a| a.action_type == EnforcementActionType::ManualRequired)
        );
    }

    #[test]
    fn test_completed_enforcement_cannot_add_actions() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();
        simulate_complete_enforcement("enforcement-001", None).unwrap();

        // Attempting to add action to completed enforcement should fail
        let action = EnforcementAction {
            action_type: EnforcementActionType::Notification,
            target_happ: None,
            target_entry: None,
            result: "Late action".to_string(),
        };

        let result = simulate_record_action("enforcement-001", action);
        assert!(
            result.is_err(),
            "Should not be able to add actions to completed enforcement"
        );
    }
}

#[cfg(test)]
mod cross_happ_enforcement_tests {
    use super::*;

    // ==========================================
    // Cross-hApp Enforcement Tests
    // ==========================================

    #[test]
    fn test_execute_cross_happ_action() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let result = simulate_execute_cross_happ_action(
            "enforcement-001",
            "mycelix-marketplace",
            Some("listing-12345".to_string()),
            "Listing suspended pending dispute resolution",
        );

        assert!(result.is_ok());
        let updated = result.unwrap();
        assert!(
            updated
                .actions
                .iter()
                .any(|a| a.action_type == EnforcementActionType::CrossHappAction)
        );
    }

    #[test]
    fn test_cross_happ_to_wallet() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let result = simulate_execute_cross_happ_action(
            "enforcement-001",
            "mycelix-wallet",
            Some("escrow-release-xyz".to_string()),
            "Escrow funds released to complainant",
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_cross_happ_to_trust() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let result = simulate_execute_cross_happ_action(
            "enforcement-001",
            "mycelix-trust",
            None,
            "MATL penalty of -100 applied to did:mycelix:bob",
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_cross_happ_to_governance() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let result = simulate_execute_cross_happ_action(
            "enforcement-001",
            "mycelix-governance",
            Some("membership-did:mycelix:bob".to_string()),
            "Governance participation suspended for 180 days",
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_cross_happ_to_credentials() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let result = simulate_execute_cross_happ_action(
            "enforcement-001",
            "mycelix-credentials",
            Some("credential-abc123".to_string()),
            "Professional credential revoked",
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_multiple_cross_happ_actions() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        // Execute actions on multiple hApps
        simulate_execute_cross_happ_action(
            "enforcement-001",
            "mycelix-wallet",
            None,
            "Funds seized",
        )
        .unwrap();
        simulate_execute_cross_happ_action(
            "enforcement-001",
            "mycelix-trust",
            None,
            "Reputation penalty applied",
        )
        .unwrap();
        simulate_execute_cross_happ_action(
            "enforcement-001",
            "mycelix-marketplace",
            None,
            "Account suspended",
        )
        .unwrap();

        let updated = simulate_get_enforcement("enforcement-001").unwrap();
        let cross_happ_actions: Vec<_> = updated
            .actions
            .iter()
            .filter(|a| a.action_type == EnforcementActionType::CrossHappAction)
            .collect();

        assert_eq!(cross_happ_actions.len(), 3);
    }
}

#[cfg(test)]
mod enforcement_query_tests {
    use super::*;

    // ==========================================
    // Enforcement Query Tests
    // ==========================================

    #[test]
    fn test_get_enforcement_by_id() {
        let enforcement =
            create_test_enforcement("enforcement-001", "decision-001", 0, "did:mycelix:enforcer");
        simulate_create_enforcement(enforcement).unwrap();

        let result = simulate_get_enforcement("enforcement-001");
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, "enforcement-001");
    }

    #[test]
    fn test_get_nonexistent_enforcement() {
        let result = simulate_get_enforcement("nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_get_decision_enforcement() {
        simulate_create_enforcement(create_test_enforcement(
            "e-001",
            "decision-001",
            0,
            "did:mycelix:e",
        ))
        .unwrap();
        simulate_create_enforcement(create_test_enforcement(
            "e-002",
            "decision-001",
            1,
            "did:mycelix:e",
        ))
        .unwrap();
        simulate_create_enforcement(create_test_enforcement(
            "e-003",
            "decision-002",
            0,
            "did:mycelix:e",
        ))
        .unwrap();

        let decision_1_enforcements = simulate_get_decision_enforcement("decision-001");
        let decision_2_enforcements = simulate_get_decision_enforcement("decision-002");

        assert_eq!(decision_1_enforcements.len(), 2);
        assert_eq!(decision_2_enforcements.len(), 1);
    }
}

// ==========================================
// Test Helper Types and Functions
// ==========================================

#[derive(Clone, Debug, PartialEq)]
enum EnforcementStatus {
    Pending,
    InProgress,
    PartiallyCompleted,
    Completed,
    Failed,
    Contested,
}

#[derive(Clone, Debug, PartialEq)]
enum EnforcementActionType {
    FundsTransfer,
    AssetFreeze,
    ReputationUpdate,
    AccessRevocation,
    Notification,
    ManualRequired,
    CrossHappAction,
}

#[derive(Clone, Debug)]
struct EnforcementAction {
    action_type: EnforcementActionType,
    target_happ: Option<String>,
    target_entry: Option<String>,
    result: String,
}

#[derive(Clone, Debug)]
struct TestEnforcement {
    id: String,
    decision_id: String,
    remedy_index: u32,
    enforcer: String,
    status: EnforcementStatus,
    actions: Vec<EnforcementAction>,
    completed_at: Option<MockTimestamp>,
}

#[derive(Clone, Debug)]
struct MockTimestamp {
    micros: i64,
}

impl MockTimestamp {
    fn now() -> Self {
        MockTimestamp {
            micros: 1706000000000000,
        }
    }
}

// In-memory store
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref ENFORCEMENT_STORE: Mutex<HashMap<String, TestEnforcement>> = Mutex::new(HashMap::new());
}

fn create_test_enforcement(
    id: &str,
    decision_id: &str,
    remedy_index: u32,
    enforcer: &str,
) -> TestEnforcement {
    TestEnforcement {
        id: id.to_string(),
        decision_id: decision_id.to_string(),
        remedy_index,
        enforcer: enforcer.to_string(),
        status: EnforcementStatus::Pending,
        actions: vec![],
        completed_at: None,
    }
}

fn simulate_create_enforcement(enforcement: TestEnforcement) -> Result<TestEnforcement, String> {
    if !enforcement.enforcer.starts_with("did:") {
        return Err("Enforcer must be a DID".to_string());
    }
    ENFORCEMENT_STORE
        .lock()
        .unwrap()
        .insert(enforcement.id.clone(), enforcement.clone());
    Ok(enforcement)
}

fn simulate_get_enforcement(id: &str) -> Option<TestEnforcement> {
    ENFORCEMENT_STORE.lock().unwrap().get(id).cloned()
}

fn simulate_get_decision_enforcement(decision_id: &str) -> Vec<TestEnforcement> {
    ENFORCEMENT_STORE
        .lock()
        .unwrap()
        .values()
        .filter(|e| e.decision_id == decision_id)
        .cloned()
        .collect()
}

fn simulate_update_enforcement_status(
    id: &str,
    new_status: EnforcementStatus,
) -> Result<TestEnforcement, String> {
    let mut store = ENFORCEMENT_STORE.lock().unwrap();
    if let Some(enforcement) = store.get_mut(id) {
        enforcement.status = new_status;
        if enforcement.status == EnforcementStatus::Completed {
            enforcement.completed_at = Some(MockTimestamp::now());
        }
        Ok(enforcement.clone())
    } else {
        Err("Enforcement not found".to_string())
    }
}

fn simulate_record_action(id: &str, action: EnforcementAction) -> Result<TestEnforcement, String> {
    let mut store = ENFORCEMENT_STORE.lock().unwrap();
    if let Some(enforcement) = store.get_mut(id) {
        if enforcement.status == EnforcementStatus::Completed {
            return Err("Cannot add actions to completed enforcement".to_string());
        }
        enforcement.actions.push(action);
        Ok(enforcement.clone())
    } else {
        Err("Enforcement not found".to_string())
    }
}

fn simulate_complete_enforcement(
    id: &str,
    final_action: Option<EnforcementAction>,
) -> Result<TestEnforcement, String> {
    let mut store = ENFORCEMENT_STORE.lock().unwrap();
    if let Some(enforcement) = store.get_mut(id) {
        if let Some(action) = final_action {
            enforcement.actions.push(action);
        }
        enforcement.status = EnforcementStatus::Completed;
        enforcement.completed_at = Some(MockTimestamp::now());
        Ok(enforcement.clone())
    } else {
        Err("Enforcement not found".to_string())
    }
}

fn simulate_mark_enforcement_failed(id: &str, reason: &str) -> Result<TestEnforcement, String> {
    let mut store = ENFORCEMENT_STORE.lock().unwrap();
    if let Some(enforcement) = store.get_mut(id) {
        enforcement.status = EnforcementStatus::Failed;
        enforcement.actions.push(EnforcementAction {
            action_type: EnforcementActionType::ManualRequired,
            target_happ: None,
            target_entry: None,
            result: reason.to_string(),
        });
        Ok(enforcement.clone())
    } else {
        Err("Enforcement not found".to_string())
    }
}

fn simulate_get_pending_enforcements() -> Vec<TestEnforcement> {
    ENFORCEMENT_STORE
        .lock()
        .unwrap()
        .values()
        .filter(|e| {
            e.status == EnforcementStatus::Pending || e.status == EnforcementStatus::InProgress
        })
        .cloned()
        .collect()
}

fn simulate_get_enforcements_by_status(status: EnforcementStatus) -> Vec<TestEnforcement> {
    ENFORCEMENT_STORE
        .lock()
        .unwrap()
        .values()
        .filter(|e| e.status == status)
        .cloned()
        .collect()
}

fn simulate_execute_cross_happ_action(
    id: &str,
    target_happ: &str,
    target_entry: Option<String>,
    result: &str,
) -> Result<TestEnforcement, String> {
    let action = EnforcementAction {
        action_type: EnforcementActionType::CrossHappAction,
        target_happ: Some(target_happ.to_string()),
        target_entry,
        result: result.to_string(),
    };
    simulate_record_action(id, action)
}
