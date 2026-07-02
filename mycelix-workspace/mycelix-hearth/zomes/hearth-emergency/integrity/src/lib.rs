// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hearth Emergency Integrity Zome
//!
//! Defines entry types and validation for emergency plans, alerts,
//! and safety check-ins within a hearth.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};
use hearth_types::*;

// ============================================================================
// Entry Types
// ============================================================================

/// An emergency preparedness plan for a hearth.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EmergencyPlan {
    /// The hearth this plan belongs to.
    pub hearth_hash: ActionHash,
    /// Emergency contacts for the hearth.
    pub contacts: Vec<EmergencyContact>,
    /// Designated meeting points in case of emergency.
    pub meeting_points: Vec<String>,
    /// Hashes of medical information entries (stored elsewhere for privacy).
    pub medical_info_hashes: Vec<ActionHash>,
    /// When this plan was last reviewed by the hearth.
    pub last_reviewed: Timestamp,
}

/// An emergency alert raised within a hearth.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EmergencyAlert {
    /// The hearth this alert belongs to.
    pub hearth_hash: ActionHash,
    /// Category of emergency.
    pub alert_type: AlertType,
    /// Severity level.
    pub severity: AlertSeverity,
    /// Human-readable description of the emergency.
    pub message: String,
    /// Agent who reported the emergency.
    pub reporter: AgentPubKey,
    /// Optional location hint (address, room, landmark).
    pub location_hint: Option<String>,
    /// When the alert was created.
    pub created_at: Timestamp,
    /// When the alert was resolved (None if still active).
    pub resolved_at: Option<Timestamp>,
}

/// A safety check-in from a member in response to an alert.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SafetyCheckIn {
    /// The hearth this check-in belongs to.
    pub hearth_hash: ActionHash,
    /// The alert this check-in responds to.
    pub alert_hash: ActionHash,
    /// The member checking in.
    pub member: AgentPubKey,
    /// Reported safety status.
    pub status: SafetyStatus,
    /// Optional location hint.
    pub location_hint: Option<String>,
    /// When the member checked in.
    pub checked_in_at: Timestamp,
}

// ============================================================================
// Helper Types (not DHT entries)
// ============================================================================

/// Contact information for emergency situations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmergencyContact {
    /// Contact's name.
    pub name: String,
    /// Contact's phone number.
    pub phone: String,
    /// Relationship to the hearth (e.g. "neighbor", "family doctor").
    pub relationship: String,
    /// Priority ordering (lower = higher priority).
    pub priority_order: u32,
}

// ============================================================================
// Entry & Link Type Registration
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    EmergencyPlan(EmergencyPlan),
    EmergencyAlert(EmergencyAlert),
    SafetyCheckIn(SafetyCheckIn),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Hearth -> EmergencyPlan
    HearthToPlans,
    /// Hearth -> EmergencyAlert
    HearthToAlerts,
    /// EmergencyAlert -> SafetyCheckIn
    AlertToCheckIns,
    /// AgentPubKey -> SafetyCheckIn
    AgentToCheckIns,
}

// ============================================================================
// Genesis + Validation
// ============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry {
            app_entry,
            action: _,
        }) => match app_entry {
            EntryTypes::EmergencyPlan(plan) => validate_plan(&plan),
            EntryTypes::EmergencyAlert(alert) => validate_alert(&alert),
            EntryTypes::SafetyCheckIn(checkin) => validate_checkin(&checkin),
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry {
            app_entry,
            original_action_hash,
            ..
        }) => match app_entry {
            EntryTypes::EmergencyPlan(plan) => {
                validate_plan(&plan)?;
                validate_plan_immutable_fields(&plan, &original_action_hash)
            }
            EntryTypes::EmergencyAlert(alert) => {
                validate_alert(&alert)?;
                validate_alert_immutable_fields(&alert, &original_action_hash)
            }
            EntryTypes::SafetyCheckIn(_) => {
                // INVARIANT: SafetyCheckIn immutability — check-ins are point-in-time
                // records and cannot be modified after creation.
                Ok(ValidateCallbackResult::Invalid(
                    "SafetyCheckIn cannot be updated once created".into(),
                ))
            }
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterCreateLink {
            link_type: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds 512 bytes".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDeleteLink { tag, action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            let result = check_link_author_match(
                original_action.action().author(),
                &action.author,
            );
            if result != ValidateCallbackResult::Valid {
                return Ok(result);
            }
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds 512 bytes".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Invalid(
            "Emergency entries cannot be deleted once created".into(),
        )),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

pub fn validate_plan(plan: &EmergencyPlan) -> ExternResult<ValidateCallbackResult> {
    if plan.contacts.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Emergency plan contacts must be <= 20".into(),
        ));
    }
    if plan.meeting_points.len() > 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Emergency plan meeting_points must be <= 10".into(),
        ));
    }
    // Validate individual contacts
    for contact in &plan.contacts {
        if contact.name.is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Emergency contact name cannot be empty".into(),
            ));
        }
        if contact.name.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Emergency contact name must be <= 256 characters".into(),
            ));
        }
        if contact.phone.is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Emergency contact phone cannot be empty".into(),
            ));
        }
        if contact.phone.len() > 64 {
            return Ok(ValidateCallbackResult::Invalid(
                "Emergency contact phone must be <= 64 characters".into(),
            ));
        }
    }
    // Validate meeting points
    for mp in &plan.meeting_points {
        if mp.is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Meeting point cannot be empty".into(),
            ));
        }
        if mp.len() > 1024 {
            return Ok(ValidateCallbackResult::Invalid(
                "Meeting point must be <= 1024 characters".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_alert(alert: &EmergencyAlert) -> ExternResult<ValidateCallbackResult> {
    if alert.message.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Alert message cannot be empty".into(),
        ));
    }
    if alert.message.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Alert message must be <= 4096 characters".into(),
        ));
    }
    // Validate optional location_hint
    if let Some(ref hint) = alert.location_hint {
        if hint.len() > 1024 {
            return Ok(ValidateCallbackResult::Invalid(
                "Alert location_hint must be <= 1024 characters".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_checkin(checkin: &SafetyCheckIn) -> ExternResult<ValidateCallbackResult> {
    // Validate optional location_hint
    if let Some(ref hint) = checkin.location_hint {
        if hint.len() > 1024 {
            return Ok(ValidateCallbackResult::Invalid(
                "Check-in location_hint must be <= 1024 characters".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Immutable Field Validation
// ============================================================================

fn validate_plan_immutable_fields(
    new: &EmergencyPlan,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original: EmergencyPlan = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original EmergencyPlan: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original EmergencyPlan entry is missing".into()
        )))?;
    if new.hearth_hash != original.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on an EmergencyPlan".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_alert_immutable_fields(
    new: &EmergencyAlert,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original: EmergencyAlert = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original EmergencyAlert: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original EmergencyAlert entry is missing".into()
        )))?;
    if new.hearth_hash != original.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on an EmergencyAlert".into(),
        ));
    }
    if new.alert_type != original.alert_type {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change alert_type on an EmergencyAlert".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helper Constructors ----

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xAAu8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xABu8; 36])
    }

    fn fake_timestamp() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn make_contact(name: &str, phone: &str) -> EmergencyContact {
        EmergencyContact {
            name: name.into(),
            phone: phone.into(),
            relationship: "neighbor".into(),
            priority_order: 1,
        }
    }

    fn make_plan(contacts: Vec<EmergencyContact>, meeting_points: Vec<String>) -> EmergencyPlan {
        EmergencyPlan {
            hearth_hash: fake_action_hash(),
            contacts,
            meeting_points,
            medical_info_hashes: vec![],
            last_reviewed: fake_timestamp(),
        }
    }

    fn make_alert(message: &str) -> EmergencyAlert {
        EmergencyAlert {
            hearth_hash: fake_action_hash(),
            alert_type: AlertType::Medical,
            severity: AlertSeverity::High,
            message: message.into(),
            reporter: fake_agent(),
            location_hint: None,
            created_at: fake_timestamp(),
            resolved_at: None,
        }
    }

    fn make_checkin() -> SafetyCheckIn {
        SafetyCheckIn {
            hearth_hash: fake_action_hash(),
            alert_hash: fake_action_hash(),
            member: fake_agent(),
            status: SafetyStatus::Safe,
            location_hint: None,
            checked_in_at: fake_timestamp(),
        }
    }

    // ---- EmergencyPlan Validation ----

    #[test]
    fn valid_plan_passes() {
        let plan = make_plan(
            vec![make_contact("Alice", "555-1234")],
            vec!["Front yard".into()],
        );
        assert!(matches!(
            validate_plan(&plan).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn plan_empty_contacts_passes() {
        let plan = make_plan(vec![], vec![]);
        assert!(matches!(
            validate_plan(&plan).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn plan_max_20_contacts_passes() {
        let contacts: Vec<EmergencyContact> = (0..20)
            .map(|i| make_contact(&format!("Contact {i}"), &format!("555-{i:04}")))
            .collect();
        let plan = make_plan(contacts, vec![]);
        assert!(matches!(
            validate_plan(&plan).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn plan_21_contacts_rejected() {
        let contacts: Vec<EmergencyContact> = (0..21)
            .map(|i| make_contact(&format!("Contact {i}"), &format!("555-{i:04}")))
            .collect();
        let plan = make_plan(contacts, vec![]);
        match validate_plan(&plan).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 20")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn plan_max_10_meeting_points_passes() {
        let mps: Vec<String> = (0..10).map(|i| format!("Point {i}")).collect();
        let plan = make_plan(vec![], mps);
        assert!(matches!(
            validate_plan(&plan).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn plan_11_meeting_points_rejected() {
        let mps: Vec<String> = (0..11).map(|i| format!("Point {i}")).collect();
        let plan = make_plan(vec![], mps);
        match validate_plan(&plan).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 10")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn plan_empty_contact_name_rejected() {
        let plan = make_plan(vec![make_contact("", "555-1234")], vec![]);
        match validate_plan(&plan).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("name cannot be empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn plan_empty_contact_phone_rejected() {
        let plan = make_plan(vec![make_contact("Alice", "")], vec![]);
        match validate_plan(&plan).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("phone cannot be empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn plan_empty_meeting_point_rejected() {
        let plan = make_plan(vec![], vec!["".into()]);
        match validate_plan(&plan).unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Meeting point cannot be empty"))
            }
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- EmergencyAlert Validation ----

    #[test]
    fn valid_alert_passes() {
        let alert = make_alert("Fire in the kitchen!");
        assert!(matches!(
            validate_alert(&alert).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn alert_empty_message_rejected() {
        let alert = make_alert("");
        match validate_alert(&alert).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("cannot be empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn alert_message_at_max_passes() {
        let alert = make_alert(&"a".repeat(4096));
        assert!(matches!(
            validate_alert(&alert).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn alert_message_exceeds_max_rejected() {
        let alert = make_alert(&"a".repeat(4097));
        match validate_alert(&alert).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 4096")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn alert_with_location_hint_passes() {
        let mut alert = make_alert("Water leak");
        alert.location_hint = Some("Basement near the boiler".into());
        assert!(matches!(
            validate_alert(&alert).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn alert_location_hint_too_long_rejected() {
        let mut alert = make_alert("Water leak");
        alert.location_hint = Some("x".repeat(1025));
        match validate_alert(&alert).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("location_hint")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- SafetyCheckIn Validation ----

    #[test]
    fn valid_checkin_passes() {
        let checkin = make_checkin();
        assert!(matches!(
            validate_checkin(&checkin).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn checkin_with_location_passes() {
        let mut checkin = make_checkin();
        checkin.location_hint = Some("At the park".into());
        assert!(matches!(
            validate_checkin(&checkin).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn checkin_location_too_long_rejected() {
        let mut checkin = make_checkin();
        checkin.location_hint = Some("x".repeat(1025));
        match validate_checkin(&checkin).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("location_hint")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- Serde Roundtrips ----

    #[test]
    fn emergency_plan_serde_roundtrip() {
        let plan = make_plan(
            vec![make_contact("Bob", "555-9999")],
            vec!["Front door".into()],
        );
        let json = serde_json::to_string(&plan).unwrap();
        let back: EmergencyPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(back, plan);
    }

    #[test]
    fn emergency_alert_serde_roundtrip() {
        let alert = make_alert("Tornado warning");
        let json = serde_json::to_string(&alert).unwrap();
        let back: EmergencyAlert = serde_json::from_str(&json).unwrap();
        assert_eq!(back, alert);
    }

    #[test]
    fn safety_checkin_serde_roundtrip() {
        let checkin = make_checkin();
        let json = serde_json::to_string(&checkin).unwrap();
        let back: SafetyCheckIn = serde_json::from_str(&json).unwrap();
        assert_eq!(back, checkin);
    }

    #[test]
    fn emergency_contact_serde_roundtrip() {
        let contact = make_contact("Dr. Smith", "555-0000");
        let json = serde_json::to_string(&contact).unwrap();
        let back: EmergencyContact = serde_json::from_str(&json).unwrap();
        assert_eq!(back, contact);
    }

    // ---- Entry / Link Type Enums ----

    #[test]
    fn entry_types_all_variants_exist() {
        let _plan = UnitEntryTypes::EmergencyPlan;
        let _alert = UnitEntryTypes::EmergencyAlert;
        let _checkin = UnitEntryTypes::SafetyCheckIn;
    }

    #[test]
    fn link_types_all_variants_exist() {
        let _plans = LinkTypes::HearthToPlans;
        let _alerts = LinkTypes::HearthToAlerts;
        let _checkins = LinkTypes::AlertToCheckIns;
        let _agent_checkins = LinkTypes::AgentToCheckIns;
    }

    // ---- All AlertType variants ----

    #[test]
    fn alert_all_types_valid() {
        let types = vec![
            AlertType::Medical,
            AlertType::Natural,
            AlertType::Security,
            AlertType::Missing,
            AlertType::Fire,
            AlertType::Custom("Gas leak".into()),
        ];
        for at in types {
            let mut alert = make_alert("Test alert");
            alert.alert_type = at;
            assert!(matches!(
                validate_alert(&alert).unwrap(),
                ValidateCallbackResult::Valid
            ));
        }
    }

    // ---- All AlertSeverity variants ----

    #[test]
    fn alert_all_severities_valid() {
        let severities = vec![
            AlertSeverity::Low,
            AlertSeverity::Medium,
            AlertSeverity::High,
            AlertSeverity::Critical,
        ];
        for sev in severities {
            let mut alert = make_alert("Test alert");
            alert.severity = sev;
            assert!(matches!(
                validate_alert(&alert).unwrap(),
                ValidateCallbackResult::Valid
            ));
        }
    }

    // ---- All SafetyStatus variants ----

    #[test]
    fn checkin_all_statuses_valid() {
        let statuses = vec![
            SafetyStatus::Safe,
            SafetyStatus::NeedHelp,
            SafetyStatus::NoResponse,
        ];
        for status in statuses {
            let mut checkin = make_checkin();
            checkin.status = status;
            assert!(matches!(
                validate_checkin(&checkin).unwrap(),
                ValidateCallbackResult::Valid
            ));
        }
    }

    // ---- Immutable Field Pure Equality Tests ----

    #[test]
    fn plan_immutable_field_hearth_hash_difference_detected() {
        let a = make_plan(vec![make_contact("Alice", "555-1234")], vec![]);
        let mut b = a.clone();
        b.hearth_hash = ActionHash::from_raw_36(vec![0xCDu8; 36]);
        assert_ne!(a.hearth_hash, b.hearth_hash);
    }

    #[test]
    fn alert_immutable_field_hearth_hash_difference_detected() {
        let a = make_alert("Test alert");
        let mut b = a.clone();
        b.hearth_hash = ActionHash::from_raw_36(vec![0xCDu8; 36]);
        assert_ne!(a.hearth_hash, b.hearth_hash);
    }

    #[test]
    fn alert_immutable_field_alert_type_difference_detected() {
        let a = make_alert("Test alert");
        let mut b = a.clone();
        b.alert_type = AlertType::Fire;
        assert_ne!(a.alert_type, b.alert_type);
    }

    #[test]
    fn delete_guard_message_content() {
        let msg = "Emergency entries cannot be deleted once created";
        assert!(msg.contains("cannot be deleted"));
    }

    #[test]
    fn checkin_immutability_all_fields_stable() {
        let a = make_checkin();
        let b = a.clone();
        assert_eq!(a.hearth_hash, b.hearth_hash);
        assert_eq!(a.alert_hash, b.alert_hash);
        assert_eq!(a.member, b.member);
        assert_eq!(a.status, b.status);
        assert_eq!(a.checked_in_at, b.checked_in_at);
    }
}
