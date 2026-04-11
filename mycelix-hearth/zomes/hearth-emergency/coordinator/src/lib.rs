// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hearth Emergency Coordinator Zome
//!
//! Provides CRUD operations for emergency plans, alerts, and safety check-ins.

use hdk::prelude::*;
use hearth_coordinator_common::{decode_zome_response, get_latest_record, require_membership};
use hearth_emergency_integrity::*;
use hearth_types::*;
use mycelix_bridge_common::{
    civic_requirement_basic, civic_requirement_proposal, GovernanceEligibility,
};

// ============================================================================
// Consciousness Gating
// ============================================================================


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
// Helpers
// ============================================================================

/// Check whether an alert can be resolved (resolved_at must be None).
fn is_alert_resolvable(alert: &EmergencyAlert) -> bool {
    alert.resolved_at.is_none()
}

/// Check whether an alert severity is critical (Medical or Fire).
#[allow(dead_code)]
fn is_severity_critical(severity: &AlertSeverity) -> bool {
    matches!(severity, AlertSeverity::Critical)
}

/// Check whether an alert type is inherently life-threatening (Medical or Fire).
#[allow(dead_code)]
fn is_alert_type_life_threatening(alert_type: &AlertType) -> bool {
    matches!(alert_type, AlertType::Medical | AlertType::Fire)
}

// ============================================================================
// Extern Functions
// ============================================================================

/// Create a new emergency plan for a hearth.
/// Links the plan from the hearth via HearthToPlans.
#[hdk_extern]
pub fn create_emergency_plan(input: CreateEmergencyPlanInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_proposal(), "create_emergency_plan")?;
    require_membership(&input.hearth_hash)?;
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
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_proposal(), "update_emergency_plan")?;
    require_membership(&input.input.hearth_hash)?;
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
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_basic(), "raise_alert")?;
    require_membership(&input.hearth_hash)?;
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
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_basic(), "check_in")?;
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

    require_membership(&alert.hearth_hash)?;

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
///
/// Auth: only the original reporter or a guardian can resolve an alert.
/// Status: only unresolved alerts (resolved_at is None) can be resolved.
#[hdk_extern]
pub fn resolve_alert(alert_hash: ActionHash) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_proposal(), "resolve_alert")?;
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

    // 1. Status check: cannot resolve an already-resolved alert
    if !is_alert_resolvable(&alert) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Alert is already resolved".into()
        )));
    }

    // 2. Auth: caller must be the original reporter OR a guardian
    let caller = agent_info()?.agent_initial_pubkey;
    if caller != alert.reporter {
        let caller_role: Option<MemberRole> = decode_zome_response(
            call(
                CallTargetCell::Local,
                ZomeName::new("hearth_kinship"),
                FunctionName::new("get_caller_role"),
                None,
                alert.hearth_hash.clone(),
            )?,
            "get_caller_role",
        )?;

        let role = caller_role.ok_or(wasm_error!(WasmErrorInner::Guest(
            "You are not an active member of this hearth".into()
        )))?;

        if !role.is_guardian() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only the original reporter or a guardian can resolve an alert".into()
            )));
        }
    }

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

        if let Some(record) = get_latest_record(target)? {
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

        if let Some(record) = get_latest_record(target)? {
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

        let record = get_latest_record(target)?;
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

    // ====================================================================
    // Pure helper: is_alert_resolvable
    // ====================================================================

    fn make_unresolved_alert() -> EmergencyAlert {
        EmergencyAlert {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            alert_type: AlertType::Medical,
            severity: AlertSeverity::High,
            message: "Test alert".into(),
            reporter: AgentPubKey::from_raw_36(vec![0xAAu8; 36]),
            location_hint: None,
            created_at: Timestamp::from_micros(1_000_000),
            resolved_at: None,
        }
    }

    fn make_resolved_alert() -> EmergencyAlert {
        EmergencyAlert {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            alert_type: AlertType::Medical,
            severity: AlertSeverity::High,
            message: "Test alert".into(),
            reporter: AgentPubKey::from_raw_36(vec![0xAAu8; 36]),
            location_hint: None,
            created_at: Timestamp::from_micros(1_000_000),
            resolved_at: Some(Timestamp::from_micros(2_000_000)),
        }
    }

    #[test]
    fn alert_resolvable_when_unresolved() {
        let alert = make_unresolved_alert();
        assert!(is_alert_resolvable(&alert));
    }

    #[test]
    fn alert_not_resolvable_when_already_resolved() {
        let alert = make_resolved_alert();
        assert!(!is_alert_resolvable(&alert));
    }

    #[test]
    fn alert_resolvable_all_types_unresolved() {
        let types = vec![
            AlertType::Medical,
            AlertType::Natural,
            AlertType::Security,
            AlertType::Missing,
            AlertType::Fire,
            AlertType::Custom("Flood".into()),
        ];
        for at in types {
            let mut alert = make_unresolved_alert();
            alert.alert_type = at;
            assert!(
                is_alert_resolvable(&alert),
                "unresolved alert should be resolvable regardless of type"
            );
        }
    }

    #[test]
    fn alert_not_resolvable_all_types_resolved() {
        let types = vec![
            AlertType::Medical,
            AlertType::Natural,
            AlertType::Security,
            AlertType::Missing,
            AlertType::Fire,
            AlertType::Custom("Flood".into()),
        ];
        for at in types {
            let mut alert = make_resolved_alert();
            alert.alert_type = at;
            assert!(
                !is_alert_resolvable(&alert),
                "resolved alert should not be resolvable regardless of type"
            );
        }
    }

    // ====================================================================
    // Pure helper: is_severity_critical
    // ====================================================================

    #[test]
    fn severity_critical_is_critical() {
        assert!(is_severity_critical(&AlertSeverity::Critical));
    }

    #[test]
    fn severity_high_is_not_critical() {
        assert!(!is_severity_critical(&AlertSeverity::High));
    }

    #[test]
    fn severity_medium_is_not_critical() {
        assert!(!is_severity_critical(&AlertSeverity::Medium));
    }

    #[test]
    fn severity_low_is_not_critical() {
        assert!(!is_severity_critical(&AlertSeverity::Low));
    }

    #[test]
    fn severity_critical_exhaustive() {
        let severities = vec![
            (AlertSeverity::Low, false),
            (AlertSeverity::Medium, false),
            (AlertSeverity::High, false),
            (AlertSeverity::Critical, true),
        ];
        for (severity, expected) in severities {
            assert_eq!(
                is_severity_critical(&severity),
                expected,
                "mismatch for {:?}",
                severity
            );
        }
    }

    // ====================================================================
    // Pure helper: is_alert_type_life_threatening
    // ====================================================================

    #[test]
    fn medical_is_life_threatening() {
        assert!(is_alert_type_life_threatening(&AlertType::Medical));
    }

    #[test]
    fn fire_is_life_threatening() {
        assert!(is_alert_type_life_threatening(&AlertType::Fire));
    }

    #[test]
    fn natural_is_not_life_threatening() {
        assert!(!is_alert_type_life_threatening(&AlertType::Natural));
    }

    #[test]
    fn security_is_not_life_threatening() {
        assert!(!is_alert_type_life_threatening(&AlertType::Security));
    }

    #[test]
    fn missing_is_not_life_threatening() {
        assert!(!is_alert_type_life_threatening(&AlertType::Missing));
    }

    #[test]
    fn custom_is_not_life_threatening() {
        assert!(!is_alert_type_life_threatening(&AlertType::Custom(
            "Flood".into()
        )));
    }

    // ====================================================================
    // Scenario tests: alert lifecycle
    // ====================================================================

    #[test]
    fn scenario_new_alert_is_resolvable() {
        // A freshly raised alert (resolved_at = None) should be resolvable
        let alert = make_unresolved_alert();
        assert!(is_alert_resolvable(&alert));
    }

    #[test]
    fn scenario_resolved_alert_cannot_be_re_resolved() {
        // Once resolved, alert should not be resolvable again (idempotency guard)
        let alert = make_resolved_alert();
        assert!(!is_alert_resolvable(&alert));
    }

    #[test]
    fn scenario_critical_medical_alert_lifecycle() {
        // Medical + Critical: life-threatening and critical severity
        let alert = EmergencyAlert {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            alert_type: AlertType::Medical,
            severity: AlertSeverity::Critical,
            message: "Heart attack".into(),
            reporter: AgentPubKey::from_raw_36(vec![0xAAu8; 36]),
            location_hint: Some("Living room".into()),
            created_at: Timestamp::from_micros(1_000_000),
            resolved_at: None,
        };

        assert!(is_alert_resolvable(&alert));
        assert!(is_severity_critical(&alert.severity));
        assert!(is_alert_type_life_threatening(&alert.alert_type));

        // After resolution
        let mut resolved = alert;
        resolved.resolved_at = Some(Timestamp::from_micros(2_000_000));
        assert!(!is_alert_resolvable(&resolved));
    }

    #[test]
    fn scenario_low_severity_security_alert() {
        // Security + Low: not life-threatening, not critical
        let alert = EmergencyAlert {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            alert_type: AlertType::Security,
            severity: AlertSeverity::Low,
            message: "Suspicious car parked outside".into(),
            reporter: AgentPubKey::from_raw_36(vec![0xAAu8; 36]),
            location_hint: None,
            created_at: Timestamp::from_micros(1_000_000),
            resolved_at: None,
        };

        assert!(is_alert_resolvable(&alert));
        assert!(!is_severity_critical(&alert.severity));
        assert!(!is_alert_type_life_threatening(&alert.alert_type));
    }

    #[test]
    fn scenario_fire_high_severity() {
        // Fire + High: life-threatening but not Critical severity
        let alert = EmergencyAlert {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            alert_type: AlertType::Fire,
            severity: AlertSeverity::High,
            message: "Smoke in the garage".into(),
            reporter: AgentPubKey::from_raw_36(vec![0xAAu8; 36]),
            location_hint: Some("Garage".into()),
            created_at: Timestamp::from_micros(1_000_000),
            resolved_at: None,
        };

        assert!(is_alert_resolvable(&alert));
        assert!(!is_severity_critical(&alert.severity));
        assert!(is_alert_type_life_threatening(&alert.alert_type));
    }

    #[test]
    fn scenario_guardian_auth_roles() {
        // Verify which roles count as guardian (used for auth fallback in resolve)
        assert!(MemberRole::Founder.is_guardian());
        assert!(MemberRole::Elder.is_guardian());
        assert!(MemberRole::Adult.is_guardian());
        assert!(!MemberRole::Youth.is_guardian());
        assert!(!MemberRole::Child.is_guardian());
        assert!(!MemberRole::Guest.is_guardian());
        assert!(!MemberRole::Ancestor.is_guardian());
    }

    #[test]
    fn scenario_all_severity_levels_with_medical_type() {
        // Medical alerts at all severity levels are life-threatening
        let severities = vec![
            AlertSeverity::Low,
            AlertSeverity::Medium,
            AlertSeverity::High,
            AlertSeverity::Critical,
        ];
        for sev in severities {
            let alert = EmergencyAlert {
                hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
                alert_type: AlertType::Medical,
                severity: sev.clone(),
                message: "Medical issue".into(),
                reporter: AgentPubKey::from_raw_36(vec![0xAAu8; 36]),
                location_hint: None,
                created_at: Timestamp::from_micros(1_000_000),
                resolved_at: None,
            };
            assert!(is_alert_type_life_threatening(&alert.alert_type));
            assert!(is_alert_resolvable(&alert));
        }
    }

    #[test]
    fn scenario_resolved_at_blocks_regardless_of_severity() {
        // Even a Critical alert, once resolved, cannot be re-resolved
        let alert = EmergencyAlert {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            alert_type: AlertType::Fire,
            severity: AlertSeverity::Critical,
            message: "House fire".into(),
            reporter: AgentPubKey::from_raw_36(vec![0xAAu8; 36]),
            location_hint: None,
            created_at: Timestamp::from_micros(1_000_000),
            resolved_at: Some(Timestamp::from_micros(2_000_000)),
        };
        assert!(!is_alert_resolvable(&alert));
        assert!(is_severity_critical(&alert.severity));
        assert!(is_alert_type_life_threatening(&alert.alert_type));
    }
}
