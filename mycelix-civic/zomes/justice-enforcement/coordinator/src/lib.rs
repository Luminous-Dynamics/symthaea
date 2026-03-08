//! Enforcement Coordinator Zome
//!
//! Manages the execution of decisions, remedy enforcement,
//! and cross-hApp coordination for dispute resolution outcomes.

use hdk::prelude::*;
use justice_enforcement_integrity::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_constitutional, requirement_for_voting,
    GovernanceEligibility, GovernanceRequirement,
};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("civic_bridge", requirement, action_name)
}

/// Input for verifying a case exists before enforcement
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerifyCaseForEnforcementInput {
    pub case_id: String,
}

/// Result of case verification for enforcement
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct CaseVerificationResult {
    pub case_found: bool,
    pub case_id: Option<String>,
    pub phase: Option<String>,
    pub status: Option<String>,
    pub error: Option<String>,
}

/// Create an enforcement action

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

#[hdk_extern]
pub fn create_enforcement(enforcement: Enforcement) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_voting(), "create_enforcement")?;
    let action_hash = create_entry(&EntryTypes::Enforcement(enforcement.clone()))?;
    let record = get_latest_record(action_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not get created enforcement".into())
    ))?;

    // Link from decision
    let decision_path = Path::from(format!("decisions/{}/enforcement", enforcement.decision_id));
    create_link(
        decision_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::DecisionToEnforcement,
        (),
    )?;

    // Index by status
    let status_path = Path::from(format!("enforcement/status/{:?}", enforcement.status));
    create_link(
        status_path.path_entry_hash()?,
        action_hash,
        LinkTypes::AllCases,
        (),
    )?;

    Ok(record)
}

/// Get enforcement for a decision
#[hdk_extern]
pub fn get_decision_enforcement(decision_id: String) -> ExternResult<Vec<Record>> {
    let decision_path = Path::from(format!("decisions/{}/enforcement", decision_id));
    let links = get_links(
        LinkQuery::try_new(
            decision_path.path_entry_hash()?,
            LinkTypes::DecisionToEnforcement,
        )?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(action_hash)? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

/// Record an enforcement action taken
#[hdk_extern]
pub fn record_action(input: RecordActionInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_voting(), "record_action")?;
    let record = get(input.enforcement_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Enforcement not found".into())
    ))?;

    let mut enforcement: Enforcement = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid enforcement entry".into()
        )))?;

    enforcement.actions.push(input.action);

    let action_hash = update_entry(input.enforcement_hash, &enforcement)?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated enforcement".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordActionInput {
    pub enforcement_hash: ActionHash,
    pub action: EnforcementAction,
}

/// Update enforcement status
#[hdk_extern]
pub fn update_enforcement_status(input: UpdateEnforcementStatusInput) -> ExternResult<Record> {
    let _eligibility =
        require_consciousness(&requirement_for_voting(), "update_enforcement_status")?;
    let record = get(input.enforcement_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Enforcement not found".into())
    ))?;

    let mut enforcement: Enforcement = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid enforcement entry".into()
        )))?;

    let old_status = enforcement.status.clone();
    enforcement.status = input.new_status.clone();

    // If completed, set completed timestamp
    if matches!(input.new_status, EnforcementStatus::Completed) {
        enforcement.completed_at = Some(sys_time()?);
    }

    let action_hash = update_entry(input.enforcement_hash.clone(), &enforcement)?;

    // Update status index
    let old_status_path = Path::from(format!("enforcement/status/{:?}", old_status));
    let old_links = get_links(
        LinkQuery::try_new(old_status_path.path_entry_hash()?, LinkTypes::AllCases)?,
        GetStrategy::default(),
    )?;
    for link in old_links {
        if link.target == input.enforcement_hash.clone().into() {
            delete_link(link.create_link_hash, GetOptions::default())?;
        }
    }

    let new_status_path = Path::from(format!("enforcement/status/{:?}", input.new_status));
    create_link(
        new_status_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AllCases,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated enforcement".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateEnforcementStatusInput {
    pub enforcement_hash: ActionHash,
    pub new_status: EnforcementStatus,
}

/// Complete enforcement
#[hdk_extern]
pub fn complete_enforcement(input: CompleteEnforcementInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_voting(), "complete_enforcement")?;
    let record = get(input.enforcement_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Enforcement not found".into())
    ))?;

    let mut enforcement: Enforcement = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid enforcement entry".into()
        )))?;

    enforcement.status = EnforcementStatus::Completed;
    enforcement.completed_at = Some(sys_time()?);

    // Record final action
    if let Some(final_action) = input.final_action {
        enforcement.actions.push(final_action);
    }

    let action_hash = update_entry(input.enforcement_hash, &enforcement)?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated enforcement".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteEnforcementInput {
    pub enforcement_hash: ActionHash,
    pub final_action: Option<EnforcementAction>,
}

/// Mark enforcement as failed
#[hdk_extern]
pub fn mark_enforcement_failed(input: FailedEnforcementInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_voting(), "mark_enforcement_failed")?;
    let record = get(input.enforcement_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Enforcement not found".into())
    ))?;

    let mut enforcement: Enforcement = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid enforcement entry".into()
        )))?;

    enforcement.status = EnforcementStatus::Failed;

    // Record the failure reason
    let failure_action = EnforcementAction {
        action_type: EnforcementActionType::ManualRequired,
        target_happ: None,
        target_entry: None,
        executed_at: sys_time()?,
        result: input.reason,
    };
    enforcement.actions.push(failure_action);

    let action_hash = update_entry(input.enforcement_hash, &enforcement)?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated enforcement".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FailedEnforcementInput {
    pub enforcement_hash: ActionHash,
    pub reason: String,
}

/// Get pending enforcements
#[hdk_extern]
pub fn get_pending_enforcements(_: ()) -> ExternResult<Vec<Record>> {
    let pending_path = Path::from("enforcement/status/Pending");
    let links = get_links(
        LinkQuery::try_new(pending_path.path_entry_hash()?, LinkTypes::AllCases)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(action_hash)? {
                records.push(record);
            }
        }
    }

    // Also get in-progress
    let in_progress_path = Path::from("enforcement/status/InProgress");
    let ip_links = get_links(
        LinkQuery::try_new(in_progress_path.path_entry_hash()?, LinkTypes::AllCases)?,
        GetStrategy::default(),
    )?;

    for link in ip_links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(action_hash)? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

/// Get enforcements by status
#[hdk_extern]
pub fn get_enforcements_by_status(status: EnforcementStatus) -> ExternResult<Vec<Record>> {
    let status_path = Path::from(format!("enforcement/status/{:?}", status));
    let links = get_links(
        LinkQuery::try_new(status_path.path_entry_hash()?, LinkTypes::AllCases)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(action_hash)? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

/// Execute cross-hApp enforcement action
#[hdk_extern]
pub fn execute_cross_happ_action(input: CrossHappActionInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(
        &requirement_for_constitutional(),
        "execute_cross_happ_action",
    )?;
    let record = get(input.enforcement_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Enforcement not found".into())
    ))?;

    let mut enforcement: Enforcement = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid enforcement entry".into()
        )))?;

    // Record the cross-hApp action
    let action = EnforcementAction {
        action_type: EnforcementActionType::CrossHappAction,
        target_happ: Some(input.target_happ),
        target_entry: input.target_entry,
        executed_at: sys_time()?,
        result: input.result,
    };
    enforcement.actions.push(action);

    let action_hash = update_entry(input.enforcement_hash, &enforcement)?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated enforcement".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CrossHappActionInput {
    pub enforcement_hash: ActionHash,
    pub target_happ: String,
    pub target_entry: Option<String>,
    pub result: String,
}

// =============================================================================
// CROSS-DOMAIN: justice-enforcement → justice-cases
// =============================================================================

/// Verify that a case exists and has a decision before creating enforcement.
///
/// Cross-domain call: justice-enforcement → justice-cases via CallTargetCell::Local.
/// Ensures enforcement actions are only created for real, decided cases.
#[hdk_extern]
pub fn verify_case_for_enforcement(
    input: VerifyCaseForEnforcementInput,
) -> ExternResult<CaseVerificationResult> {
    // Call justice_cases to get all cases and find the matching one
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("justice_cases"),
        FunctionName::from("get_all_cases"),
        None,
        (),
    );

    let cases: Vec<Record> = match &response {
        Ok(ZomeCallResponse::Ok(extern_io)) => extern_io
            .decode()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?,
        Ok(other) => {
            return Ok(CaseVerificationResult {
                case_found: false,
                case_id: None,
                phase: None,
                status: None,
                error: Some(format!(
                    "Unexpected response from justice_cases: {:?}",
                    other
                )),
            });
        }
        Err(e) => {
            return Ok(CaseVerificationResult {
                case_found: false,
                case_id: None,
                phase: None,
                status: None,
                error: Some(format!("Failed to call justice_cases: {:?}", e)),
            });
        }
    };

    // Search through cases for matching case_id
    // Cases use a Case struct with a `case_id` or similar field
    // We check by looking at case entries for the matching ID
    for record in &cases {
        if let Some(entry) = record.entry().as_option() {
            // Try to decode as a generic JSON to extract case_id
            let bytes: SerializedBytes = SerializedBytes::try_from(entry.clone()).map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Serialize error: {:?}", e)))
            })?;
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes.bytes()) {
                if let Some(id) = value.get("case_id").and_then(|v| v.as_str()) {
                    if id == input.case_id {
                        let phase = value
                            .get("phase")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let status = value
                            .get("status")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());

                        return Ok(CaseVerificationResult {
                            case_found: true,
                            case_id: Some(input.case_id),
                            phase,
                            status,
                            error: None,
                        });
                    }
                }
            }
        }
    }

    Ok(CaseVerificationResult {
        case_found: false,
        case_id: None,
        phase: None,
        status: None,
        error: Some(format!(
            "Case '{}' not found in justice system",
            input.case_id
        )),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a fake ActionHash for tests
    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xAB; 36])
    }

    fn fake_enforcement_action() -> EnforcementAction {
        EnforcementAction {
            action_type: EnforcementActionType::Notification,
            target_happ: Some("mycelix-commons".into()),
            target_entry: Some("entry-abc".into()),
            executed_at: Timestamp::from_micros(1_000_000),
            result: "Notification sent successfully".into(),
        }
    }

    // ========================================================================
    // Existing CaseVerificationResult + VerifyCaseForEnforcementInput tests
    // ========================================================================

    #[test]
    fn case_verification_result_found_serde() {
        let r = CaseVerificationResult {
            case_found: true,
            case_id: Some("CASE-42".into()),
            phase: Some("Decision".into()),
            status: Some("Active".into()),
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: CaseVerificationResult = serde_json::from_str(&json).unwrap();
        assert!(r2.case_found);
        assert_eq!(r2.case_id.as_deref(), Some("CASE-42"));
        assert_eq!(r2.phase.as_deref(), Some("Decision"));
        assert_eq!(r2.status.as_deref(), Some("Active"));
        assert!(r2.error.is_none());
    }

    #[test]
    fn case_verification_result_not_found_serde() {
        let r = CaseVerificationResult {
            case_found: false,
            case_id: None,
            phase: None,
            status: None,
            error: Some("Case 'BOGUS' not found in justice system".into()),
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: CaseVerificationResult = serde_json::from_str(&json).unwrap();
        assert!(!r2.case_found);
        assert!(r2.case_id.is_none());
        assert!(r2.error.as_ref().unwrap().contains("BOGUS"));
    }

    #[test]
    fn verify_case_input_serde() {
        let input = VerifyCaseForEnforcementInput {
            case_id: "CASE-99".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: VerifyCaseForEnforcementInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.case_id, "CASE-99");
    }

    #[test]
    fn case_verification_all_none_fields() {
        let r = CaseVerificationResult {
            case_found: false,
            case_id: None,
            phase: None,
            status: None,
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: CaseVerificationResult = serde_json::from_str(&json).unwrap();
        assert!(!r2.case_found);
        assert!(r2.case_id.is_none());
        assert!(r2.phase.is_none());
        assert!(r2.status.is_none());
        assert!(r2.error.is_none());
    }

    // ========================================================================
    // RecordActionInput serde tests
    // ========================================================================

    #[test]
    fn record_action_input_serde_roundtrip() {
        let input = RecordActionInput {
            enforcement_hash: fake_action_hash(),
            action: fake_enforcement_action(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: RecordActionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.enforcement_hash, fake_action_hash());
        assert_eq!(input2.action.result, "Notification sent successfully");
        assert_eq!(
            input2.action.target_happ.as_deref(),
            Some("mycelix-commons")
        );
    }

    #[test]
    fn record_action_input_minimal_action() {
        let input = RecordActionInput {
            enforcement_hash: fake_action_hash(),
            action: EnforcementAction {
                action_type: EnforcementActionType::ManualRequired,
                target_happ: None,
                target_entry: None,
                executed_at: Timestamp::from_micros(0),
                result: "Manual intervention needed".into(),
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: RecordActionInput = serde_json::from_str(&json).unwrap();
        assert!(input2.action.target_happ.is_none());
        assert!(input2.action.target_entry.is_none());
    }

    // ========================================================================
    // UpdateEnforcementStatusInput serde tests
    // ========================================================================

    #[test]
    fn update_enforcement_status_input_serde_pending() {
        let input = UpdateEnforcementStatusInput {
            enforcement_hash: fake_action_hash(),
            new_status: EnforcementStatus::Pending,
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: UpdateEnforcementStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.enforcement_hash, fake_action_hash());
    }

    #[test]
    fn update_enforcement_status_input_serde_all_statuses() {
        let statuses = vec![
            EnforcementStatus::Pending,
            EnforcementStatus::InProgress,
            EnforcementStatus::PartiallyCompleted,
            EnforcementStatus::Completed,
            EnforcementStatus::Failed,
            EnforcementStatus::Contested,
        ];
        for status in statuses {
            let input = UpdateEnforcementStatusInput {
                enforcement_hash: fake_action_hash(),
                new_status: status.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let input2: UpdateEnforcementStatusInput = serde_json::from_str(&json).unwrap();
            assert_eq!(input2.enforcement_hash, fake_action_hash());
            // Verify the JSON contains the status name
            let json_val: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert!(
                json_val.get("new_status").is_some(),
                "Missing new_status for {:?}",
                status
            );
        }
    }

    // ========================================================================
    // CompleteEnforcementInput serde tests
    // ========================================================================

    #[test]
    fn complete_enforcement_input_serde_with_final_action() {
        let input = CompleteEnforcementInput {
            enforcement_hash: fake_action_hash(),
            final_action: Some(fake_enforcement_action()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: CompleteEnforcementInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.enforcement_hash, fake_action_hash());
        assert!(input2.final_action.is_some());
        assert_eq!(
            input2.final_action.unwrap().result,
            "Notification sent successfully"
        );
    }

    #[test]
    fn complete_enforcement_input_serde_no_final_action() {
        let input = CompleteEnforcementInput {
            enforcement_hash: fake_action_hash(),
            final_action: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: CompleteEnforcementInput = serde_json::from_str(&json).unwrap();
        assert!(input2.final_action.is_none());
    }

    // ========================================================================
    // FailedEnforcementInput serde tests
    // ========================================================================

    #[test]
    fn failed_enforcement_input_serde_roundtrip() {
        let input = FailedEnforcementInput {
            enforcement_hash: fake_action_hash(),
            reason: "Target account no longer exists".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: FailedEnforcementInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.enforcement_hash, fake_action_hash());
        assert_eq!(input2.reason, "Target account no longer exists");
    }

    #[test]
    fn failed_enforcement_input_serde_empty_reason() {
        let input = FailedEnforcementInput {
            enforcement_hash: fake_action_hash(),
            reason: "".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: FailedEnforcementInput = serde_json::from_str(&json).unwrap();
        assert!(input2.reason.is_empty());
    }

    // ========================================================================
    // CrossHappActionInput serde tests
    // ========================================================================

    #[test]
    fn cross_happ_action_input_serde_roundtrip() {
        let input = CrossHappActionInput {
            enforcement_hash: fake_action_hash(),
            target_happ: "mycelix-commons".into(),
            target_entry: Some("property-entry-42".into()),
            result: "Property transfer initiated".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: CrossHappActionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.target_happ, "mycelix-commons");
        assert_eq!(input2.target_entry.as_deref(), Some("property-entry-42"));
        assert_eq!(input2.result, "Property transfer initiated");
    }

    #[test]
    fn cross_happ_action_input_serde_no_target_entry() {
        let input = CrossHappActionInput {
            enforcement_hash: fake_action_hash(),
            target_happ: "mycelix-civic".into(),
            target_entry: None,
            result: "Broadcast notification".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: CrossHappActionInput = serde_json::from_str(&json).unwrap();
        assert!(input2.target_entry.is_none());
    }

    // ========================================================================
    // EnforcementAction serde tests
    // ========================================================================

    #[test]
    fn enforcement_action_serde_all_action_types() {
        let action_types = vec![
            EnforcementActionType::FundsTransfer,
            EnforcementActionType::AssetFreeze,
            EnforcementActionType::ReputationUpdate,
            EnforcementActionType::AccessRevocation,
            EnforcementActionType::Notification,
            EnforcementActionType::ManualRequired,
            EnforcementActionType::CrossHappAction,
        ];
        for at in action_types {
            let action = EnforcementAction {
                action_type: at.clone(),
                target_happ: None,
                target_entry: None,
                executed_at: Timestamp::from_micros(500),
                result: "ok".into(),
            };
            let json = serde_json::to_string(&action).unwrap();
            let action2: EnforcementAction = serde_json::from_str(&json).unwrap();
            assert_eq!(action2.action_type, at);
            assert_eq!(action2.result, "ok");
        }
    }

    #[test]
    fn enforcement_status_serde_all_variants() {
        let statuses = vec![
            EnforcementStatus::Pending,
            EnforcementStatus::InProgress,
            EnforcementStatus::PartiallyCompleted,
            EnforcementStatus::Completed,
            EnforcementStatus::Failed,
            EnforcementStatus::Contested,
        ];
        for s in statuses {
            let json = serde_json::to_string(&s).unwrap();
            let s2: EnforcementStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, s2);
        }
    }

    // ========================================================================
    // Additional edge-case and boundary tests
    // ========================================================================

    #[test]
    fn case_verification_result_equality() {
        let a = CaseVerificationResult {
            case_found: true,
            case_id: Some("CASE-1".into()),
            phase: Some("Enforcement".into()),
            status: Some("Active".into()),
            error: None,
        };
        let b = CaseVerificationResult {
            case_found: true,
            case_id: Some("CASE-1".into()),
            phase: Some("Enforcement".into()),
            status: Some("Active".into()),
            error: None,
        };
        assert_eq!(a, b);

        let c = CaseVerificationResult {
            case_found: false,
            case_id: Some("CASE-1".into()),
            phase: Some("Enforcement".into()),
            status: Some("Active".into()),
            error: None,
        };
        assert_ne!(a, c);
    }

    #[test]
    fn case_verification_result_with_phase_no_status() {
        let r = CaseVerificationResult {
            case_found: true,
            case_id: Some("CASE-PARTIAL".into()),
            phase: Some("Arbitration".into()),
            status: None,
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: CaseVerificationResult = serde_json::from_str(&json).unwrap();
        assert!(r2.case_found);
        assert_eq!(r2.phase.as_deref(), Some("Arbitration"));
        assert!(r2.status.is_none());
    }

    #[test]
    fn enforcement_action_all_optional_fields_present() {
        let action = EnforcementAction {
            action_type: EnforcementActionType::FundsTransfer,
            target_happ: Some("mycelix-commons".into()),
            target_entry: Some("transfer-entry-789".into()),
            executed_at: Timestamp::from_micros(999_999),
            result: "Transfer of 500 tokens completed".into(),
        };
        let json = serde_json::to_string(&action).unwrap();
        let action2: EnforcementAction = serde_json::from_str(&json).unwrap();
        assert_eq!(action2.action_type, EnforcementActionType::FundsTransfer);
        assert_eq!(action2.target_happ.as_deref(), Some("mycelix-commons"));
        assert_eq!(action2.target_entry.as_deref(), Some("transfer-entry-789"));
        assert_eq!(action2.result, "Transfer of 500 tokens completed");
    }

    #[test]
    fn failed_enforcement_input_unicode_reason() {
        let input = FailedEnforcementInput {
            enforcement_hash: fake_action_hash(),
            reason: "Ziel-Konto wurde geschlossen".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: FailedEnforcementInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.reason, "Ziel-Konto wurde geschlossen");
    }

    #[test]
    fn cross_happ_action_input_long_result() {
        let long_result = "x".repeat(10_000);
        let input = CrossHappActionInput {
            enforcement_hash: fake_action_hash(),
            target_happ: "mycelix-economic".into(),
            target_entry: Some("entry-long".into()),
            result: long_result.clone(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: CrossHappActionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.result.len(), 10_000);
    }

    #[test]
    fn complete_enforcement_input_with_cross_happ_final_action() {
        let input = CompleteEnforcementInput {
            enforcement_hash: fake_action_hash(),
            final_action: Some(EnforcementAction {
                action_type: EnforcementActionType::CrossHappAction,
                target_happ: Some("mycelix-civic".into()),
                target_entry: Some("media-ban-xyz".into()),
                executed_at: Timestamp::from_micros(2_000_000),
                result: "Media access revoked for respondent".into(),
            }),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: CompleteEnforcementInput = serde_json::from_str(&json).unwrap();
        let action = input2.final_action.unwrap();
        assert_eq!(action.action_type, EnforcementActionType::CrossHappAction);
        assert_eq!(action.target_happ.as_deref(), Some("mycelix-civic"));
        assert_eq!(action.target_entry.as_deref(), Some("media-ban-xyz"));
    }

    #[test]
    fn verify_case_input_empty_case_id() {
        let input = VerifyCaseForEnforcementInput { case_id: "".into() };
        let json = serde_json::to_string(&input).unwrap();
        let input2: VerifyCaseForEnforcementInput = serde_json::from_str(&json).unwrap();
        assert!(input2.case_id.is_empty());
    }

    #[test]
    fn enforcement_action_type_serde_all_variants() {
        let types = vec![
            EnforcementActionType::FundsTransfer,
            EnforcementActionType::AssetFreeze,
            EnforcementActionType::ReputationUpdate,
            EnforcementActionType::AccessRevocation,
            EnforcementActionType::Notification,
            EnforcementActionType::ManualRequired,
            EnforcementActionType::CrossHappAction,
        ];
        for t in types {
            let json = serde_json::to_string(&t).unwrap();
            let t2: EnforcementActionType = serde_json::from_str(&json).unwrap();
            assert_eq!(t, t2, "Failed for {:?}", t);
        }
    }
}
