//! Hearth Emergency Coordinator Zome
//!
//! Provides CRUD operations for emergency plans, alerts, and safety check-ins.

use hdk::prelude::*;
use hearth_emergency_integrity::*;
use hearth_types::*;

// ============================================================================
// Input Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEmergencyPlanInput {
    pub hearth_hash: ActionHash,
    pub contacts: Vec<EmergencyContact>,
    pub meeting_points: Vec<String>,
    pub medical_info_hashes: Vec<ActionHash>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdatePlanInput {
    pub plan_hash: ActionHash,
    pub input: CreateEmergencyPlanInput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaiseAlertInput {
    pub hearth_hash: ActionHash,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub location_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckInInput {
    pub alert_hash: ActionHash,
    pub status: SafetyStatus,
    pub location_hint: Option<String>,
}

// ============================================================================
// Extern Functions
// ============================================================================

/// Create a new emergency plan for a hearth.
/// Links the plan from the hearth via HearthToPlans.
#[hdk_extern]
pub fn create_emergency_plan(input: CreateEmergencyPlanInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let plan = EmergencyPlan {
        hearth_hash: input.hearth_hash.clone(),
        contacts: input.contacts,
        meeting_points: input.meeting_points,
        medical_info_hashes: input.medical_info_hashes,
        last_reviewed: now,
    };

    let plan_hash = create_entry(&EntryTypes::EmergencyPlan(plan))?;

    create_link(
        input.hearth_hash,
        plan_hash.clone(),
        LinkTypes::HearthToPlans,
        (),
    )?;

    let record = get(plan_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the newly created EmergencyPlan".into())
    ))?;

    Ok(record)
}

/// Update an existing emergency plan.
#[hdk_extern]
pub fn update_emergency_plan(input: UpdatePlanInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let plan = EmergencyPlan {
        hearth_hash: input.input.hearth_hash,
        contacts: input.input.contacts,
        meeting_points: input.input.meeting_points,
        medical_info_hashes: input.input.medical_info_hashes,
        last_reviewed: now,
    };

    let updated_hash = update_entry(input.plan_hash, &plan)?;

    let record = get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the updated EmergencyPlan".into())
    ))?;

    Ok(record)
}

/// Raise an emergency alert for a hearth.
/// Links the alert from the hearth and emits a HearthSignal::EmergencyAlert.
#[hdk_extern]
pub fn raise_alert(input: RaiseAlertInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent = agent_info()?.agent_initial_pubkey;

    let alert = EmergencyAlert {
        hearth_hash: input.hearth_hash.clone(),
        alert_type: input.alert_type,
        severity: input.severity.clone(),
        message: input.message.clone(),
        reporter: agent,
        location_hint: input.location_hint,
        created_at: now,
        resolved_at: None,
    };

    let alert_hash = create_entry(&EntryTypes::EmergencyAlert(alert))?;

    create_link(
        input.hearth_hash,
        alert_hash.clone(),
        LinkTypes::HearthToAlerts,
        (),
    )?;

    // Emit real-time signal so connected clients are immediately notified
    let signal = HearthSignal::EmergencyAlert {
        alert_hash: alert_hash.clone(),
        severity: input.severity,
        message: input.message,
    };
    emit_signal(&signal)?;

    let record = get(alert_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the newly created EmergencyAlert".into())
    ))?;

    Ok(record)
}

/// Check in during an emergency alert.
/// Links the check-in from the alert and from the agent.
#[hdk_extern]
pub fn check_in(input: CheckInInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent = agent_info()?.agent_initial_pubkey;

    // Retrieve the alert to get the hearth_hash
    let alert_record = get(input.alert_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Alert not found".into())))?;
    let alert: EmergencyAlert = alert_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize alert: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Alert entry is missing".into()
        )))?;

    let checkin = SafetyCheckIn {
        hearth_hash: alert.hearth_hash,
        alert_hash: input.alert_hash.clone(),
        member: agent.clone(),
        status: input.status,
        location_hint: input.location_hint,
        checked_in_at: now,
    };

    let checkin_hash = create_entry(&EntryTypes::SafetyCheckIn(checkin))?;

    create_link(
        input.alert_hash,
        checkin_hash.clone(),
        LinkTypes::AlertToCheckIns,
        (),
    )?;

    create_link(agent, checkin_hash.clone(), LinkTypes::AgentToCheckIns, ())?;

    let record = get(checkin_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the newly created SafetyCheckIn".into())
    ))?;

    Ok(record)
}

/// Resolve an active alert by setting resolved_at to now.
#[hdk_extern]
pub fn resolve_alert(alert_hash: ActionHash) -> ExternResult<Record> {
    let now = sys_time()?;

    let existing = get(alert_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Alert not found".into())))?;
    let mut alert: EmergencyAlert = existing
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize alert: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Alert entry is missing".into()
        )))?;

    alert.resolved_at = Some(now);

    let updated_hash = update_entry(alert_hash, &alert)?;

    let record = get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the updated EmergencyAlert".into())
    ))?;

    Ok(record)
}

/// Get all active (unresolved) alerts for a hearth.
#[hdk_extern]
pub fn get_active_alerts(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToAlerts)?,
        GetStrategy::default(),
    )?;

    let mut active_alerts = Vec::new();
    for link in links {
        let target = link
            .target
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Link target is not an ActionHash".into()
            )))?;

        if let Some(record) = get(target, GetOptions::default())? {
            let alert: EmergencyAlert = record
                .entry()
                .to_app_option()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Failed to deserialize alert: {e}"
                    )))
                })?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Alert entry is missing".into()
                )))?;

            if alert.resolved_at.is_none() {
                active_alerts.push(record);
            }
        }
    }

    Ok(active_alerts)
}

/// Get all check-ins for a specific alert.
#[hdk_extern]
pub fn get_alert_checkins(alert_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(alert_hash, LinkTypes::AlertToCheckIns)?,
        GetStrategy::default(),
    )?;

    let mut checkins = Vec::new();
    for link in links {
        let target = link
            .target
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Link target is not an ActionHash".into()
            )))?;

        if let Some(record) = get(target, GetOptions::default())? {
            checkins.push(record);
        }
    }

    Ok(checkins)
}

/// Get the emergency plan for a hearth (returns the most recent one).
#[hdk_extern]
pub fn get_emergency_plan(hearth_hash: ActionHash) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToPlans)?,
        GetStrategy::default(),
    )?;

    // Return the most recently linked plan
    if let Some(link) = links.last() {
        let target =
            link.target
                .clone()
                .into_action_hash()
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Link target is not an ActionHash".into()
                )))?;

        let record = get(target, GetOptions::default())?;
        Ok(record)
    } else {
        Ok(None)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Input Type Serde ----

    #[test]
    fn create_plan_input_serde_roundtrip() {
        let input = CreateEmergencyPlanInput {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            contacts: vec![EmergencyContact {
                name: "Alice".into(),
                phone: "555-1234".into(),
                relationship: "neighbor".into(),
                priority_order: 1,
            }],
            meeting_points: vec!["Front yard".into()],
            medical_info_hashes: vec![],
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateEmergencyPlanInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.contacts.len(), 1);
        assert_eq!(back.meeting_points.len(), 1);
    }

    #[test]
    fn raise_alert_input_serde_roundtrip() {
        let input = RaiseAlertInput {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            alert_type: AlertType::Fire,
            severity: AlertSeverity::Critical,
            message: "Fire!".into(),
            location_hint: Some("Kitchen".into()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: RaiseAlertInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.message, "Fire!");
    }

    #[test]
    fn checkin_input_serde_roundtrip() {
        let input = CheckInInput {
            alert_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            status: SafetyStatus::Safe,
            location_hint: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CheckInInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.status, SafetyStatus::Safe);
    }

    #[test]
    fn update_plan_input_serde_roundtrip() {
        let input = UpdatePlanInput {
            plan_hash: ActionHash::from_raw_36(vec![0xACu8; 36]),
            input: CreateEmergencyPlanInput {
                hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
                contacts: vec![],
                meeting_points: vec![],
                medical_info_hashes: vec![],
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: UpdatePlanInput = serde_json::from_str(&json).unwrap();
        assert!(back.input.contacts.is_empty());
    }

    #[test]
    fn raise_alert_input_all_types() {
        let types = vec![
            AlertType::Medical,
            AlertType::Natural,
            AlertType::Security,
            AlertType::Missing,
            AlertType::Fire,
            AlertType::Custom("Flood".into()),
        ];
        for at in types {
            let input = RaiseAlertInput {
                hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
                alert_type: at,
                severity: AlertSeverity::Low,
                message: "Test".into(),
                location_hint: None,
            };
            let json = serde_json::to_string(&input).unwrap();
            let _back: RaiseAlertInput = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn checkin_input_all_statuses() {
        let statuses = vec![
            SafetyStatus::Safe,
            SafetyStatus::NeedHelp,
            SafetyStatus::NoResponse,
        ];
        for status in statuses {
            let input = CheckInInput {
                alert_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
                status,
                location_hint: None,
            };
            let json = serde_json::to_string(&input).unwrap();
            let _back: CheckInInput = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn create_plan_input_with_medical_hashes() {
        let input = CreateEmergencyPlanInput {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            contacts: vec![],
            meeting_points: vec![],
            medical_info_hashes: vec![
                ActionHash::from_raw_36(vec![0x01u8; 36]),
                ActionHash::from_raw_36(vec![0x02u8; 36]),
            ],
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateEmergencyPlanInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.medical_info_hashes.len(), 2);
    }

    #[test]
    fn raise_alert_input_with_location() {
        let input = RaiseAlertInput {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            alert_type: AlertType::Medical,
            severity: AlertSeverity::High,
            message: "Medical emergency".into(),
            location_hint: Some("Second floor bedroom".into()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: RaiseAlertInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.location_hint.unwrap(), "Second floor bedroom");
    }
}
