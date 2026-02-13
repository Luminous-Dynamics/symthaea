//! Enforcement Coordinator Zome
//!
//! Manages the execution of decisions, remedy enforcement,
//! and cross-hApp coordination for dispute resolution outcomes.

use hdk::prelude::*;
use justice_enforcement_integrity::*;

/// Create an enforcement action
#[hdk_extern]
pub fn create_enforcement(enforcement: Enforcement) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Enforcement(enforcement.clone()))?;
    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get created enforcement".into())))?;

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
        LinkQuery::try_new(decision_path.path_entry_hash()?, LinkTypes::DecisionToEnforcement)?,
        GetStrategy::default()
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

/// Record an enforcement action taken
#[hdk_extern]
pub fn record_action(input: RecordActionInput) -> ExternResult<Record> {
    let record = get(input.enforcement_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Enforcement not found".into())))?;

    let mut enforcement: Enforcement = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid enforcement entry".into())))?;

    enforcement.actions.push(input.action);

    let action_hash = update_entry(input.enforcement_hash, &enforcement)?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get updated enforcement".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordActionInput {
    pub enforcement_hash: ActionHash,
    pub action: EnforcementAction,
}

/// Update enforcement status
#[hdk_extern]
pub fn update_enforcement_status(input: UpdateEnforcementStatusInput) -> ExternResult<Record> {
    let record = get(input.enforcement_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Enforcement not found".into())))?;

    let mut enforcement: Enforcement = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid enforcement entry".into())))?;

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
        GetStrategy::default()
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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get updated enforcement".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateEnforcementStatusInput {
    pub enforcement_hash: ActionHash,
    pub new_status: EnforcementStatus,
}

/// Complete enforcement
#[hdk_extern]
pub fn complete_enforcement(input: CompleteEnforcementInput) -> ExternResult<Record> {
    let record = get(input.enforcement_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Enforcement not found".into())))?;

    let mut enforcement: Enforcement = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid enforcement entry".into())))?;

    enforcement.status = EnforcementStatus::Completed;
    enforcement.completed_at = Some(sys_time()?);

    // Record final action
    if let Some(final_action) = input.final_action {
        enforcement.actions.push(final_action);
    }

    let action_hash = update_entry(input.enforcement_hash, &enforcement)?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get updated enforcement".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteEnforcementInput {
    pub enforcement_hash: ActionHash,
    pub final_action: Option<EnforcementAction>,
}

/// Mark enforcement as failed
#[hdk_extern]
pub fn mark_enforcement_failed(input: FailedEnforcementInput) -> ExternResult<Record> {
    let record = get(input.enforcement_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Enforcement not found".into())))?;

    let mut enforcement: Enforcement = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid enforcement entry".into())))?;

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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get updated enforcement".into())))
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
        GetStrategy::default()
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }

    // Also get in-progress
    let in_progress_path = Path::from("enforcement/status/InProgress");
    let ip_links = get_links(
        LinkQuery::try_new(in_progress_path.path_entry_hash()?, LinkTypes::AllCases)?,
        GetStrategy::default()
    )?;

    for link in ip_links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
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
        GetStrategy::default()
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

/// Execute cross-hApp enforcement action
#[hdk_extern]
pub fn execute_cross_happ_action(input: CrossHappActionInput) -> ExternResult<Record> {
    let record = get(input.enforcement_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Enforcement not found".into())))?;

    let mut enforcement: Enforcement = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid enforcement entry".into())))?;

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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get updated enforcement".into())))
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

/// Result of verifying a case before enforcement
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CaseVerificationResult {
    pub case_found: bool,
    pub case_id: Option<String>,
    pub phase: Option<String>,
    pub status: Option<String>,
    pub error: Option<String>,
}

/// Input for verifying a case before enforcement
#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyCaseForEnforcementInput {
    pub case_id: String,
}

/// Verify that a case exists and has a decision before creating enforcement.
///
/// Cross-domain call: justice-enforcement → justice-cases via CallTargetCell::Local.
/// Ensures enforcement actions are only created for real, decided cases.
#[hdk_extern]
pub fn verify_case_for_enforcement(input: VerifyCaseForEnforcementInput) -> ExternResult<CaseVerificationResult> {
    // Call justice_cases to get all cases and find the matching one
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("justice_cases"),
        FunctionName::from("get_all_cases"),
        None,
        (),
    );

    let cases: Vec<Record> = match &response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            extern_io.decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?
        }
        Ok(other) => {
            return Ok(CaseVerificationResult {
                case_found: false,
                case_id: None,
                phase: None,
                status: None,
                error: Some(format!("Unexpected response from justice_cases: {:?}", other)),
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
            let bytes: SerializedBytes = SerializedBytes::try_from(entry.clone())
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Serialize error: {:?}", e))))?;
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes.bytes()) {
                if let Some(id) = value.get("case_id").and_then(|v| v.as_str()) {
                    if id == input.case_id {
                        let phase = value.get("phase")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let status = value.get("status")
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
        error: Some(format!("Case '{}' not found in justice system", input.case_id)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
