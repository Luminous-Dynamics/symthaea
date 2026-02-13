//! Arbitration Coordinator Zome
//!
//! Manages arbitration panels, deliberations, decisions, and appeals.
//! Implements Tier 2 (arbitration) and Tier 3 (appeal) of the justice system.

use hdk::prelude::*;
use justice_arbitration_integrity::*;

/// Create an arbitration panel for a case
#[hdk_extern]
pub fn create_arbitration(arbitration: Arbitration) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Arbitration(arbitration.clone()))?;
    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get created arbitration".into())))?;

    // Link from case
    let case_path = Path::from(format!("cases/{}/arbitration", arbitration.case_id));
    create_link(
        case_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::CaseToArbitration,
        (),
    )?;

    // Link from each arbitrator
    for arb in &arbitration.arbitrators {
        let arb_path = Path::from(format!("arbitrators/{}/cases", arb.did));
        create_link(
            arb_path.path_entry_hash()?,
            action_hash.clone(),
            LinkTypes::ArbitratorToCases,
            (),
        )?;
    }

    Ok(record)
}

/// Get arbitration for a case
#[hdk_extern]
pub fn get_case_arbitration(case_id: String) -> ExternResult<Option<Record>> {
    let case_path = Path::from(format!("cases/{}/arbitration", case_id));
    let links = get_links(
        LinkQuery::try_new(case_path.path_entry_hash()?, LinkTypes::CaseToArbitration)?,
        GetStrategy::default()
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            return get(action_hash, GetOptions::default());
        }
    }

    Ok(None)
}

/// Update arbitration status
#[hdk_extern]
pub fn update_arbitration_status(input: UpdateArbStatusInput) -> ExternResult<Record> {
    let record = get(input.arbitration_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Arbitration not found".into())))?;

    let mut arb: Arbitration = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid arbitration entry".into())))?;

    arb.status = input.new_status;

    let action_hash = update_entry(input.arbitration_hash, &arb)?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get updated arbitration".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateArbStatusInput {
    pub arbitration_hash: ActionHash,
    pub new_status: ArbitrationStatus,
}

/// Record an arbitrator's acceptance or recusal
#[hdk_extern]
pub fn record_arbitrator_response(input: ArbitratorResponseInput) -> ExternResult<Record> {
    let record = get(input.arbitration_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Arbitration not found".into())))?;

    let mut arb: Arbitration = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid arbitration entry".into())))?;

    // Find and update the arbitrator
    for a in &mut arb.arbitrators {
        if a.did == input.arbitrator_did {
            a.accepted = input.accepted;
            a.recused = input.recused;
            a.recusal_reason = input.recusal_reason.clone();
        }
    }

    let action_hash = update_entry(input.arbitration_hash, &arb)?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get updated arbitration".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ArbitratorResponseInput {
    pub arbitration_hash: ActionHash,
    pub arbitrator_did: String,
    pub accepted: bool,
    pub recused: bool,
    pub recusal_reason: Option<String>,
}

/// Render a decision
#[hdk_extern]
pub fn render_decision(decision: Decision) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Decision(decision.clone()))?;
    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get created decision".into())))?;

    // Link from case
    let case_path = Path::from(format!("cases/{}/decisions", decision.case_id));
    create_link(
        case_path.path_entry_hash()?,
        action_hash,
        LinkTypes::CaseToDecisions,
        (),
    )?;

    Ok(record)
}

/// Get decisions for a case
#[hdk_extern]
pub fn get_case_decisions(case_id: String) -> ExternResult<Vec<Record>> {
    let case_path = Path::from(format!("cases/{}/decisions", case_id));
    let links = get_links(
        LinkQuery::try_new(case_path.path_entry_hash()?, LinkTypes::CaseToDecisions)?,
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

/// File an appeal
#[hdk_extern]
pub fn file_appeal(appeal: Appeal) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Appeal(appeal.clone()))?;
    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get created appeal".into())))?;

    // Link from decision
    let decision_path = Path::from(format!("decisions/{}/appeals", appeal.decision_id));
    create_link(
        decision_path.path_entry_hash()?,
        action_hash,
        LinkTypes::DecisionToAppeals,
        (),
    )?;

    Ok(record)
}

/// Get appeals for a decision
#[hdk_extern]
pub fn get_decision_appeals(decision_id: String) -> ExternResult<Vec<Record>> {
    let decision_path = Path::from(format!("decisions/{}/appeals", decision_id));
    let links = get_links(
        LinkQuery::try_new(decision_path.path_entry_hash()?, LinkTypes::DecisionToAppeals)?,
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

/// Update appeal status
#[hdk_extern]
pub fn update_appeal_status(input: UpdateAppealStatusInput) -> ExternResult<Record> {
    let record = get(input.appeal_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Appeal not found".into())))?;

    let mut appeal: Appeal = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid appeal entry".into())))?;

    appeal.status = input.new_status;

    let action_hash = update_entry(input.appeal_hash, &appeal)?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get updated appeal".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateAppealStatusInput {
    pub appeal_hash: ActionHash,
    pub new_status: AppealStatus,
}

/// Finalize a decision (no more appeals allowed)
#[hdk_extern]
pub fn finalize_decision(input: FinalizeDecisionInput) -> ExternResult<Record> {
    let record = get(input.decision_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Decision not found".into())))?;

    let mut decision: Decision = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid decision entry".into())))?;

    decision.finalized = true;

    let action_hash = update_entry(input.decision_hash, &decision)?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get updated decision".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FinalizeDecisionInput {
    pub decision_hash: ActionHash,
}

/// Get cases for an arbitrator
#[hdk_extern]
pub fn get_arbitrator_cases(arbitrator_did: String) -> ExternResult<Vec<Record>> {
    let arb_path = Path::from(format!("arbitrators/{}/cases", arbitrator_did));
    let links = get_links(
        LinkQuery::try_new(arb_path.path_entry_hash()?, LinkTypes::ArbitratorToCases)?,
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

// ============================================================================
// Cross-domain: Check emergency context for justice cases
// ============================================================================

/// Wire-compatible copy of emergency Disaster for deserialization.
#[derive(Serialize, Deserialize, Debug, Clone, SerializedBytes)]
struct LocalDisaster {
    pub id: String,
    pub disaster_type: LocalDisasterType,
    pub title: String,
    pub description: String,
    pub severity: LocalSeverityLevel,
    pub declared_by: AgentPubKey,
    pub declared_at: Timestamp,
    pub affected_area: LocalAffectedArea,
    pub status: LocalDisasterStatus,
    pub estimated_affected: u32,
    pub coordination_lead: Option<AgentPubKey>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum LocalDisasterType {
    Hurricane, Earthquake, Wildfire, Flood, Tornado,
    Pandemic, Industrial, MassCasualty, CyberAttack,
    Infrastructure, Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum LocalSeverityLevel { Level1, Level2, Level3, Level4, Level5 }

#[derive(Serialize, Deserialize, Debug, Clone)]
struct LocalAffectedArea {
    pub center_lat: f64,
    pub center_lon: f64,
    pub radius_km: f32,
    pub boundary: Option<Vec<(f64, f64)>>,
    pub zones: Vec<LocalOperationalZone>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct LocalOperationalZone {
    pub id: String,
    pub name: String,
    pub boundary: Vec<(f64, f64)>,
    pub priority: LocalZonePriority,
    pub status: LocalZoneStatus,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum LocalZonePriority { Critical, High, Medium, Low }

#[derive(Serialize, Deserialize, Debug, Clone)]
enum LocalZoneStatus { Unassessed, Active, Cleared, Hazardous, Evacuated }

#[derive(Serialize, Deserialize, Debug, Clone)]
enum LocalDisasterStatus { Declared, Active, Recovery, Closed }

#[derive(Serialize, Deserialize, Debug)]
pub struct CheckEmergencyContextInput {
    /// Case ID for correlation.
    pub case_id: String,
    /// Optional keyword to match against disaster titles/descriptions.
    pub keyword: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EmergencyContextResult {
    pub has_active_emergencies: bool,
    pub active_emergency_count: u32,
    pub matching_disaster: Option<String>,
    pub matching_severity: Option<String>,
    pub recommendation: Option<String>,
    pub error: Option<String>,
}

/// Check if there are active emergencies relevant to a justice case.
///
/// Cross-domain call: justice-arbitration queries emergency_incidents
/// via `call(CallTargetCell::Local, ...)` to determine if the case
/// context involves an active emergency. This adjusts urgency and
/// may trigger expedited procedures.
#[hdk_extern]
pub fn check_emergency_context_for_case(input: CheckEmergencyContextInput) -> ExternResult<EmergencyContextResult> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("emergency_incidents"),
        FunctionName::from("get_active_disasters"),
        None,
        (),
    );

    match &response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            let records: Vec<Record> = extern_io.decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?;

            let active_count = records.len() as u32;

            // Search for keyword match if provided
            if let Some(ref keyword) = input.keyword {
                let kw_lower = keyword.to_lowercase();
                for record in &records {
                    if let Some(disaster) = record
                        .entry()
                        .to_app_option::<LocalDisaster>()
                        .ok()
                        .flatten()
                    {
                        if disaster.title.to_lowercase().contains(&kw_lower)
                            || disaster.description.to_lowercase().contains(&kw_lower)
                        {
                            return Ok(EmergencyContextResult {
                                has_active_emergencies: true,
                                active_emergency_count: active_count,
                                matching_disaster: Some(disaster.title),
                                matching_severity: Some(format!("{:?}", disaster.severity)),
                                recommendation: Some(
                                    "Case relates to active emergency — consider expedited arbitration procedures".to_string()
                                ),
                                error: None,
                            });
                        }
                    }
                }
            }

            Ok(EmergencyContextResult {
                has_active_emergencies: active_count > 0,
                active_emergency_count: active_count,
                matching_disaster: None,
                matching_severity: None,
                recommendation: if active_count > 0 {
                    Some(format!(
                        "{} active emergency(ies) — verify case '{}' is not emergency-related",
                        active_count, input.case_id
                    ))
                } else {
                    None
                },
                error: None,
            })
        }
        Ok(ZomeCallResponse::NetworkError(err)) => Ok(EmergencyContextResult {
            has_active_emergencies: false,
            active_emergency_count: 0,
            matching_disaster: None,
            matching_severity: None,
            recommendation: None,
            error: Some(format!("Network error: {}", err)),
        }),
        _ => Ok(EmergencyContextResult {
            has_active_emergencies: false,
            active_emergency_count: 0,
            matching_disaster: None,
            matching_severity: None,
            recommendation: None,
            error: Some("Failed to query emergency incidents".into()),
        }),
    }
}
