//! Incidents Integrity Zome
//! Defines entry types and validation for disaster incidents

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A declared disaster incident
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Disaster {
    pub id: String,
    pub disaster_type: DisasterType,
    pub title: String,
    pub description: String,
    pub severity: SeverityLevel,
    pub declared_by: AgentPubKey,
    pub declared_at: Timestamp,
    pub affected_area: AffectedArea,
    pub status: DisasterStatus,
    pub estimated_affected: u32,
    pub coordination_lead: Option<AgentPubKey>,
}

/// Types of disasters
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisasterType {
    Hurricane,
    Earthquake,
    Wildfire,
    Flood,
    Tornado,
    Pandemic,
    Industrial,
    MassCasualty,
    CyberAttack,
    Infrastructure,
    Other(String),
}

/// Severity levels (FEMA-aligned)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SeverityLevel {
    Level1,
    Level2,
    Level3,
    Level4,
    Level5,
}

/// Status of a disaster
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisasterStatus {
    Declared,
    Active,
    Recovery,
    Closed,
}

/// Geographic area affected by the disaster
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AffectedArea {
    pub center_lat: f64,
    pub center_lon: f64,
    pub radius_km: f32,
    pub boundary: Option<Vec<(f64, f64)>>,
    pub zones: Vec<OperationalZone>,
}

/// An operational zone within the affected area
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct OperationalZone {
    pub id: String,
    pub name: String,
    pub boundary: Vec<(f64, f64)>,
    pub priority: ZonePriority,
    pub status: ZoneStatus,
}

/// Priority level for operational zones
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ZonePriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Status of an operational zone
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ZoneStatus {
    Unassessed,
    Active,
    Cleared,
    Hazardous,
    Evacuated,
}

/// An update to an incident
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct IncidentUpdate {
    pub disaster_hash: ActionHash,
    pub author: AgentPubKey,
    pub timestamp: Timestamp,
    pub update_type: UpdateType,
    pub content: String,
}

/// Types of incident updates
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UpdateType {
    StatusChange,
    SeverityChange,
    AreaExpansion,
    AreaContraction,
    CasualtyReport,
    ResourceUpdate,
    WeatherUpdate,
    InfrastructureUpdate,
    EvacuationOrder,
    AllClear,
    General,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Disaster(Disaster),
    IncidentUpdate(IncidentUpdate),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllDisasters,
    ActiveDisasters,
    DisasterByType,
    DisasterToUpdate,
    AgentToDisaster,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Disaster(disaster) => validate_create_disaster(action, disaster),
                EntryTypes::IncidentUpdate(update) => validate_create_update(action, update),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Disaster(disaster) => validate_update_disaster(disaster),
                EntryTypes::IncidentUpdate(_) => Ok(ValidateCallbackResult::Invalid(
                    "Incident updates are immutable".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AllDisasters => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ActiveDisasters => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DisasterByType => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DisasterToUpdate => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToDisaster => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::ActiveDisasters => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_disaster(
    _action: Create,
    disaster: Disaster,
) -> ExternResult<ValidateCallbackResult> {
    if disaster.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Disaster ID cannot be empty".into(),
        ));
    }
    if disaster.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Disaster title cannot be empty".into(),
        ));
    }
    if disaster.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Disaster description cannot be empty".into(),
        ));
    }
    if disaster.affected_area.radius_km <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Affected area radius must be positive".into(),
        ));
    }
    if disaster.affected_area.center_lat < -90.0 || disaster.affected_area.center_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if disaster.affected_area.center_lon < -180.0 || disaster.affected_area.center_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_disaster(disaster: Disaster) -> ExternResult<ValidateCallbackResult> {
    if disaster.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Disaster ID cannot be empty".into(),
        ));
    }
    if disaster.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Disaster title cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_update(
    _action: Create,
    update: IncidentUpdate,
) -> ExternResult<ValidateCallbackResult> {
    if update.content.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Update content cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test helpers
    fn fake_create() -> Create {
        Create {
            author: fake_agent_pub_key(),
            timestamp: fake_timestamp(),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            weight: Default::default(),
        }
    }

    fn fake_agent_pub_key() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0; 36])
    }

    fn fake_timestamp() -> Timestamp {
        Timestamp::from_micros(0)
    }

    fn is_valid(result: ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    // Factory functions for valid defaults
    fn valid_disaster() -> Disaster {
        Disaster {
            id: "disaster-001".into(),
            disaster_type: DisasterType::Hurricane,
            title: "Hurricane Test".into(),
            description: "A test hurricane disaster".into(),
            severity: SeverityLevel::Level3,
            declared_by: fake_agent_pub_key(),
            declared_at: fake_timestamp(),
            affected_area: valid_affected_area(),
            status: DisasterStatus::Active,
            estimated_affected: 1000,
            coordination_lead: None,
        }
    }

    fn valid_affected_area() -> AffectedArea {
        AffectedArea {
            center_lat: 35.0,
            center_lon: -80.0,
            radius_km: 50.0,
            boundary: None,
            zones: vec![],
        }
    }

    fn valid_operational_zone() -> OperationalZone {
        OperationalZone {
            id: "zone-001".into(),
            name: "Red Zone".into(),
            boundary: vec![(35.0, -80.0), (35.1, -80.1), (35.0, -80.2)],
            priority: ZonePriority::Critical,
            status: ZoneStatus::Active,
        }
    }

    fn valid_incident_update() -> IncidentUpdate {
        IncidentUpdate {
            disaster_hash: fake_action_hash(),
            author: fake_agent_pub_key(),
            timestamp: fake_timestamp(),
            update_type: UpdateType::StatusChange,
            content: "Status updated to active".into(),
        }
    }

    // Anchor tests
    #[test]
    fn test_anchor_serde_roundtrip() {
        let anchor = Anchor("all_disasters".into());
        let serialized = serde_json::to_string(&anchor).unwrap();
        let deserialized: Anchor = serde_json::from_str(&serialized).unwrap();
        assert_eq!(anchor, deserialized);
    }

    // DisasterType tests
    #[test]
    fn test_disaster_type_serde_all_variants() {
        let variants = vec![
            DisasterType::Hurricane,
            DisasterType::Earthquake,
            DisasterType::Wildfire,
            DisasterType::Flood,
            DisasterType::Tornado,
            DisasterType::Pandemic,
            DisasterType::Industrial,
            DisasterType::MassCasualty,
            DisasterType::CyberAttack,
            DisasterType::Infrastructure,
            DisasterType::Other("Custom".into()),
        ];

        for variant in variants {
            let serialized = serde_json::to_string(&variant).unwrap();
            let deserialized: DisasterType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    // SeverityLevel tests
    #[test]
    fn test_severity_level_serde_all_variants() {
        let variants = vec![
            SeverityLevel::Level1,
            SeverityLevel::Level2,
            SeverityLevel::Level3,
            SeverityLevel::Level4,
            SeverityLevel::Level5,
        ];

        for variant in variants {
            let serialized = serde_json::to_string(&variant).unwrap();
            let deserialized: SeverityLevel = serde_json::from_str(&serialized).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    // DisasterStatus tests
    #[test]
    fn test_disaster_status_serde_all_variants() {
        let variants = vec![
            DisasterStatus::Declared,
            DisasterStatus::Active,
            DisasterStatus::Recovery,
            DisasterStatus::Closed,
        ];

        for variant in variants {
            let serialized = serde_json::to_string(&variant).unwrap();
            let deserialized: DisasterStatus = serde_json::from_str(&serialized).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    // ZonePriority tests
    #[test]
    fn test_zone_priority_serde_all_variants() {
        let variants = vec![
            ZonePriority::Critical,
            ZonePriority::High,
            ZonePriority::Medium,
            ZonePriority::Low,
        ];

        for variant in variants {
            let serialized = serde_json::to_string(&variant).unwrap();
            let deserialized: ZonePriority = serde_json::from_str(&serialized).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    // ZoneStatus tests
    #[test]
    fn test_zone_status_serde_all_variants() {
        let variants = vec![
            ZoneStatus::Unassessed,
            ZoneStatus::Active,
            ZoneStatus::Cleared,
            ZoneStatus::Hazardous,
            ZoneStatus::Evacuated,
        ];

        for variant in variants {
            let serialized = serde_json::to_string(&variant).unwrap();
            let deserialized: ZoneStatus = serde_json::from_str(&serialized).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    // UpdateType tests
    #[test]
    fn test_update_type_serde_all_variants() {
        let variants = vec![
            UpdateType::StatusChange,
            UpdateType::SeverityChange,
            UpdateType::AreaExpansion,
            UpdateType::AreaContraction,
            UpdateType::CasualtyReport,
            UpdateType::ResourceUpdate,
            UpdateType::WeatherUpdate,
            UpdateType::InfrastructureUpdate,
            UpdateType::EvacuationOrder,
            UpdateType::AllClear,
            UpdateType::General,
        ];

        for variant in variants {
            let serialized = serde_json::to_string(&variant).unwrap();
            let deserialized: UpdateType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    // AffectedArea tests
    #[test]
    fn test_affected_area_serde_roundtrip() {
        let area = AffectedArea {
            center_lat: 35.5,
            center_lon: -80.5,
            radius_km: 100.5,
            boundary: Some(vec![(35.0, -80.0), (36.0, -81.0)]),
            zones: vec![valid_operational_zone()],
        };

        let serialized = serde_json::to_string(&area).unwrap();
        let deserialized: AffectedArea = serde_json::from_str(&serialized).unwrap();
        assert_eq!(area, deserialized);
    }

    // OperationalZone tests
    #[test]
    fn test_operational_zone_serde_roundtrip() {
        let zone = valid_operational_zone();
        let serialized = serde_json::to_string(&zone).unwrap();
        let deserialized: OperationalZone = serde_json::from_str(&serialized).unwrap();
        assert_eq!(zone, deserialized);
    }

    // Disaster tests
    #[test]
    fn test_disaster_serde_roundtrip() {
        let disaster = valid_disaster();
        let serialized = serde_json::to_string(&disaster).unwrap();
        let deserialized: Disaster = serde_json::from_str(&serialized).unwrap();
        assert_eq!(disaster, deserialized);
    }

    #[test]
    fn test_disaster_with_coordination_lead() {
        let mut disaster = valid_disaster();
        disaster.coordination_lead = Some(fake_agent_pub_key());
        let serialized = serde_json::to_string(&disaster).unwrap();
        let deserialized: Disaster = serde_json::from_str(&serialized).unwrap();
        assert_eq!(disaster, deserialized);
    }

    // IncidentUpdate tests
    #[test]
    fn test_incident_update_serde_roundtrip() {
        let update = valid_incident_update();
        let serialized = serde_json::to_string(&update).unwrap();
        let deserialized: IncidentUpdate = serde_json::from_str(&serialized).unwrap();
        assert_eq!(update, deserialized);
    }

    // validate_create_disaster tests
    #[test]
    fn test_validate_create_disaster_valid() {
        let disaster = valid_disaster();
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_disaster_empty_id() {
        let mut disaster = valid_disaster();
        disaster.id = "".into();
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_create_disaster_empty_title() {
        let mut disaster = valid_disaster();
        disaster.title = "".into();
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_create_disaster_empty_description() {
        let mut disaster = valid_disaster();
        disaster.description = "".into();
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_create_disaster_zero_radius() {
        let mut disaster = valid_disaster();
        disaster.affected_area.radius_km = 0.0;
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_create_disaster_negative_radius() {
        let mut disaster = valid_disaster();
        disaster.affected_area.radius_km = -10.0;
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_create_disaster_latitude_below_min() {
        let mut disaster = valid_disaster();
        disaster.affected_area.center_lat = -91.0;
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_create_disaster_latitude_above_max() {
        let mut disaster = valid_disaster();
        disaster.affected_area.center_lat = 91.0;
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_create_disaster_latitude_at_min() {
        let mut disaster = valid_disaster();
        disaster.affected_area.center_lat = -90.0;
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_disaster_latitude_at_max() {
        let mut disaster = valid_disaster();
        disaster.affected_area.center_lat = 90.0;
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_disaster_longitude_below_min() {
        let mut disaster = valid_disaster();
        disaster.affected_area.center_lon = -181.0;
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_create_disaster_longitude_above_max() {
        let mut disaster = valid_disaster();
        disaster.affected_area.center_lon = 181.0;
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_create_disaster_longitude_at_min() {
        let mut disaster = valid_disaster();
        disaster.affected_area.center_lon = -180.0;
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_disaster_longitude_at_max() {
        let mut disaster = valid_disaster();
        disaster.affected_area.center_lon = 180.0;
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_disaster_with_boundary() {
        let mut disaster = valid_disaster();
        disaster.affected_area.boundary = Some(vec![
            (35.0, -80.0),
            (35.5, -80.5),
            (36.0, -81.0),
        ]);
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_disaster_with_zones() {
        let mut disaster = valid_disaster();
        disaster.affected_area.zones = vec![
            valid_operational_zone(),
            OperationalZone {
                id: "zone-002".into(),
                name: "Green Zone".into(),
                boundary: vec![(36.0, -81.0)],
                priority: ZonePriority::Low,
                status: ZoneStatus::Cleared,
            },
        ];
        let result = validate_create_disaster(fake_create(), disaster);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_disaster_all_disaster_types() {
        let disaster_types = vec![
            DisasterType::Hurricane,
            DisasterType::Earthquake,
            DisasterType::Wildfire,
            DisasterType::Flood,
            DisasterType::Tornado,
            DisasterType::Pandemic,
            DisasterType::Industrial,
            DisasterType::MassCasualty,
            DisasterType::CyberAttack,
            DisasterType::Infrastructure,
            DisasterType::Other("Custom".into()),
        ];

        for disaster_type in disaster_types {
            let mut disaster = valid_disaster();
            disaster.disaster_type = disaster_type;
            let result = validate_create_disaster(fake_create(), disaster);
            assert!(is_valid(result));
        }
    }

    #[test]
    fn test_validate_create_disaster_all_severity_levels() {
        let severity_levels = vec![
            SeverityLevel::Level1,
            SeverityLevel::Level2,
            SeverityLevel::Level3,
            SeverityLevel::Level4,
            SeverityLevel::Level5,
        ];

        for severity in severity_levels {
            let mut disaster = valid_disaster();
            disaster.severity = severity;
            let result = validate_create_disaster(fake_create(), disaster);
            assert!(is_valid(result));
        }
    }

    #[test]
    fn test_validate_create_disaster_all_statuses() {
        let statuses = vec![
            DisasterStatus::Declared,
            DisasterStatus::Active,
            DisasterStatus::Recovery,
            DisasterStatus::Closed,
        ];

        for status in statuses {
            let mut disaster = valid_disaster();
            disaster.status = status;
            let result = validate_create_disaster(fake_create(), disaster);
            assert!(is_valid(result));
        }
    }

    // validate_update_disaster tests
    #[test]
    fn test_validate_update_disaster_valid() {
        let disaster = valid_disaster();
        let result = validate_update_disaster(disaster);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_update_disaster_empty_id() {
        let mut disaster = valid_disaster();
        disaster.id = "".into();
        let result = validate_update_disaster(disaster);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_update_disaster_empty_title() {
        let mut disaster = valid_disaster();
        disaster.title = "".into();
        let result = validate_update_disaster(disaster);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_update_disaster_allows_empty_description() {
        let mut disaster = valid_disaster();
        disaster.description = "".into();
        let result = validate_update_disaster(disaster);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_update_disaster_allows_invalid_coordinates() {
        let mut disaster = valid_disaster();
        disaster.affected_area.center_lat = 100.0;
        disaster.affected_area.center_lon = 200.0;
        let result = validate_update_disaster(disaster);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_update_disaster_allows_zero_radius() {
        let mut disaster = valid_disaster();
        disaster.affected_area.radius_km = 0.0;
        let result = validate_update_disaster(disaster);
        assert!(is_valid(result));
    }

    // validate_create_update tests
    #[test]
    fn test_validate_create_update_valid() {
        let update = valid_incident_update();
        let result = validate_create_update(fake_create(), update);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_update_empty_content() {
        let mut update = valid_incident_update();
        update.content = "".into();
        let result = validate_create_update(fake_create(), update);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_create_update_all_types() {
        let update_types = vec![
            UpdateType::StatusChange,
            UpdateType::SeverityChange,
            UpdateType::AreaExpansion,
            UpdateType::AreaContraction,
            UpdateType::CasualtyReport,
            UpdateType::ResourceUpdate,
            UpdateType::WeatherUpdate,
            UpdateType::InfrastructureUpdate,
            UpdateType::EvacuationOrder,
            UpdateType::AllClear,
            UpdateType::General,
        ];

        for update_type in update_types {
            let mut update = valid_incident_update();
            update.update_type = update_type;
            let result = validate_create_update(fake_create(), update);
            assert!(is_valid(result));
        }
    }

    #[test]
    fn test_validate_create_update_long_content() {
        let mut update = valid_incident_update();
        update.content = "a".repeat(10000);
        let result = validate_create_update(fake_create(), update);
        assert!(is_valid(result));
    }
}
