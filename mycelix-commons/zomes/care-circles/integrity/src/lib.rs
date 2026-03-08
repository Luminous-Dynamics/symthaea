//! Circles Integrity Zome
//! Defines entry types and validation for care circles and membership.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Type of care circle
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CircleType {
    Neighborhood,
    Workplace,
    Faith,
    Family,
    School,
    Custom(String),
}

impl CircleType {
    pub fn anchor_key(&self) -> String {
        match self {
            CircleType::Neighborhood => "neighborhood".to_string(),
            CircleType::Workplace => "workplace".to_string(),
            CircleType::Faith => "faith".to_string(),
            CircleType::Family => "family".to_string(),
            CircleType::School => "school".to_string(),
            CircleType::Custom(s) => format!("custom_{}", s.to_lowercase().replace(' ', "_")),
        }
    }
}

/// Role within a care circle
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MemberRole {
    Organizer,
    Member,
    Observer,
}

/// A care circle - a group of people who coordinate mutual aid
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CareCircle {
    /// Circle name
    pub name: String,
    /// Description of the circle's purpose
    pub description: String,
    /// Location or area the circle serves
    pub location: String,
    /// Maximum number of members allowed
    pub max_members: u32,
    /// Agent who created the circle
    pub created_by: AgentPubKey,
    /// Type of circle
    pub circle_type: CircleType,
    /// Whether the circle is currently active
    pub active: bool,
    /// When the circle was created
    pub created_at: Timestamp,
}

/// Membership record linking an agent to a circle
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CircleMembership {
    /// Hash of the CareCircle entry
    pub circle_hash: ActionHash,
    /// The member agent
    pub member: AgentPubKey,
    /// Role in the circle
    pub role: MemberRole,
    /// When the member joined
    pub joined_at: Timestamp,
    /// Whether the membership is currently active
    pub active: bool,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    CareCircle(CareCircle),
    CircleMembership(CircleMembership),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All circles anchor
    AllCircles,
    /// Circle type anchor to circles of that type
    TypeToCircle,
    /// Circle to its memberships
    CircleToMembership,
    /// Agent to their memberships
    AgentToMembership,
    /// Agent to circles they created
    AgentToCreatedCircle,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CareCircle(circle) => validate_create_circle(action, circle),
                EntryTypes::CircleMembership(membership) => {
                    validate_create_membership(action, membership)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CareCircle(circle) => validate_update_circle(circle),
                EntryTypes::CircleMembership(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => match link_type {
            LinkTypes::AllCircles => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllCircles link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TypeToCircle => {
                // Type tags may carry serialized CircleType data
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TypeToCircle link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CircleToMembership => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "CircleToMembership link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToMembership => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToMembership link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToCreatedCircle => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToCreatedCircle link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => match link_type {
            LinkTypes::AllCircles => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllCircles delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TypeToCircle => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TypeToCircle delete link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CircleToMembership => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "CircleToMembership delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToMembership => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToMembership delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToCreatedCircle => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToCreatedCircle delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_circle_type_custom(circle_type: &CircleType) -> Result<(), String> {
    if let CircleType::Custom(s) = circle_type {
        if s.trim().is_empty() {
            return Err("Custom circle type label cannot be empty".to_string());
        }
        if s.len() > 128 {
            return Err("Custom circle type label must be 128 characters or fewer".to_string());
        }
    }
    Ok(())
}

fn validate_create_circle(
    _action: Create,
    circle: CareCircle,
) -> ExternResult<ValidateCallbackResult> {
    if circle.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle name cannot be empty".into(),
        ));
    }
    if circle.name.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle name must be 128 characters or fewer".into(),
        ));
    }
    if circle.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle description cannot be empty".into(),
        ));
    }
    if circle.description.len() > 2048 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle description must be 2048 characters or fewer".into(),
        ));
    }
    if circle.max_members < 2 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle must allow at least 2 members".into(),
        ));
    }
    if circle.max_members > 500 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle cannot have more than 500 members".into(),
        ));
    }
    if circle.location.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Location cannot be empty".into(),
        ));
    }
    if circle.location.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Location must be 512 characters or fewer".into(),
        ));
    }
    if let Err(msg) = validate_circle_type_custom(&circle.circle_type) {
        return Ok(ValidateCallbackResult::Invalid(msg));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_circle(circle: CareCircle) -> ExternResult<ValidateCallbackResult> {
    if circle.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle name cannot be empty".into(),
        ));
    }
    if circle.name.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle name must be 128 characters or fewer".into(),
        ));
    }
    if circle.description.len() > 2048 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle description must be 2048 characters or fewer".into(),
        ));
    }
    if circle.location.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Location must be 512 characters or fewer".into(),
        ));
    }
    if circle.max_members < 2 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle must allow at least 2 members".into(),
        ));
    }
    if circle.max_members > 500 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle cannot have more than 500 members".into(),
        ));
    }
    if let Err(msg) = validate_circle_type_custom(&circle.circle_type) {
        return Ok(ValidateCallbackResult::Invalid(msg));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_membership(
    _action: Create,
    _membership: CircleMembership,
) -> ExternResult<ValidateCallbackResult> {
    // Membership validation is primarily handled at the coordinator level
    // (checking circle exists, member count, etc.)
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdi::prelude::*;

    // Factory functions

    fn valid_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xdb; 36])
    }

    fn valid_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xab; 36])
    }

    fn valid_timestamp() -> Timestamp {
        Timestamp::from_micros(1000000)
    }

    fn valid_create_action() -> Create {
        Create {
            author: valid_agent(),
            timestamp: valid_timestamp(),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0xac; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0xad; 36]),
            weight: Default::default(),
        }
    }

    fn valid_circle() -> CareCircle {
        CareCircle {
            name: "Community Care Circle".to_string(),
            description: "A local neighborhood care circle for mutual aid and support".to_string(),
            location: "Downtown Portland, OR".to_string(),
            max_members: 20,
            created_by: valid_agent(),
            circle_type: CircleType::Neighborhood,
            active: true,
            created_at: valid_timestamp(),
        }
    }

    fn valid_membership() -> CircleMembership {
        CircleMembership {
            circle_hash: valid_action_hash(),
            member: valid_agent(),
            role: MemberRole::Member,
            joined_at: valid_timestamp(),
            active: true,
        }
    }

    // CircleType tests

    #[test]
    fn test_circle_type_serde_neighborhood() {
        let ct = CircleType::Neighborhood;
        let serialized = serde_json::to_string(&ct).unwrap();
        let deserialized: CircleType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(ct, deserialized);
    }

    #[test]
    fn test_circle_type_serde_workplace() {
        let ct = CircleType::Workplace;
        let serialized = serde_json::to_string(&ct).unwrap();
        let deserialized: CircleType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(ct, deserialized);
    }

    #[test]
    fn test_circle_type_serde_faith() {
        let ct = CircleType::Faith;
        let serialized = serde_json::to_string(&ct).unwrap();
        let deserialized: CircleType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(ct, deserialized);
    }

    #[test]
    fn test_circle_type_serde_family() {
        let ct = CircleType::Family;
        let serialized = serde_json::to_string(&ct).unwrap();
        let deserialized: CircleType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(ct, deserialized);
    }

    #[test]
    fn test_circle_type_serde_school() {
        let ct = CircleType::School;
        let serialized = serde_json::to_string(&ct).unwrap();
        let deserialized: CircleType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(ct, deserialized);
    }

    #[test]
    fn test_circle_type_serde_custom() {
        let ct = CircleType::Custom("Hobby Group".to_string());
        let serialized = serde_json::to_string(&ct).unwrap();
        let deserialized: CircleType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(ct, deserialized);
    }

    #[test]
    fn test_circle_type_anchor_key_neighborhood() {
        assert_eq!(CircleType::Neighborhood.anchor_key(), "neighborhood");
    }

    #[test]
    fn test_circle_type_anchor_key_workplace() {
        assert_eq!(CircleType::Workplace.anchor_key(), "workplace");
    }

    #[test]
    fn test_circle_type_anchor_key_faith() {
        assert_eq!(CircleType::Faith.anchor_key(), "faith");
    }

    #[test]
    fn test_circle_type_anchor_key_family() {
        assert_eq!(CircleType::Family.anchor_key(), "family");
    }

    #[test]
    fn test_circle_type_anchor_key_school() {
        assert_eq!(CircleType::School.anchor_key(), "school");
    }

    #[test]
    fn test_circle_type_anchor_key_custom_simple() {
        assert_eq!(
            CircleType::Custom("Hobby".to_string()).anchor_key(),
            "custom_hobby"
        );
    }

    #[test]
    fn test_circle_type_anchor_key_custom_with_spaces() {
        assert_eq!(
            CircleType::Custom("Book Club".to_string()).anchor_key(),
            "custom_book_club"
        );
    }

    #[test]
    fn test_circle_type_anchor_key_custom_mixed_case() {
        assert_eq!(
            CircleType::Custom("Gaming Group".to_string()).anchor_key(),
            "custom_gaming_group"
        );
    }

    // MemberRole tests

    #[test]
    fn test_member_role_serde_organizer() {
        let role = MemberRole::Organizer;
        let serialized = serde_json::to_string(&role).unwrap();
        let deserialized: MemberRole = serde_json::from_str(&serialized).unwrap();
        assert_eq!(role, deserialized);
    }

    #[test]
    fn test_member_role_serde_member() {
        let role = MemberRole::Member;
        let serialized = serde_json::to_string(&role).unwrap();
        let deserialized: MemberRole = serde_json::from_str(&serialized).unwrap();
        assert_eq!(role, deserialized);
    }

    #[test]
    fn test_member_role_serde_observer() {
        let role = MemberRole::Observer;
        let serialized = serde_json::to_string(&role).unwrap();
        let deserialized: MemberRole = serde_json::from_str(&serialized).unwrap();
        assert_eq!(role, deserialized);
    }

    // validate_create_circle tests

    #[test]
    fn test_create_circle_valid() {
        let circle = valid_circle();
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_circle_name_empty() {
        let mut circle = valid_circle();
        circle.name = "".to_string();
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle name cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_create_circle_name_at_max_length() {
        let mut circle = valid_circle();
        circle.name = "a".repeat(128);
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_circle_name_exceeds_max_length() {
        let mut circle = valid_circle();
        circle.name = "a".repeat(129);
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle name must be 128 characters or fewer");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_create_circle_description_empty() {
        let mut circle = valid_circle();
        circle.description = "".to_string();
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle description cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_create_circle_description_at_max_length() {
        let mut circle = valid_circle();
        circle.description = "a".repeat(2048);
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_circle_description_exceeds_max_length() {
        let mut circle = valid_circle();
        circle.description = "a".repeat(2049);
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle description must be 2048 characters or fewer");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_create_circle_max_members_at_minimum() {
        let mut circle = valid_circle();
        circle.max_members = 2;
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_circle_max_members_below_minimum() {
        let mut circle = valid_circle();
        circle.max_members = 1;
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle must allow at least 2 members");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_create_circle_max_members_at_maximum() {
        let mut circle = valid_circle();
        circle.max_members = 500;
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_circle_max_members_exceeds_maximum() {
        let mut circle = valid_circle();
        circle.max_members = 501;
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle cannot have more than 500 members");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_create_circle_location_empty() {
        let mut circle = valid_circle();
        circle.location = "".to_string();
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Location cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_create_circle_location_at_max_length() {
        let mut circle = valid_circle();
        circle.location = "a".repeat(512);
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_circle_location_exceeds_max_length() {
        let mut circle = valid_circle();
        circle.location = "a".repeat(513);
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Location must be 512 characters or fewer");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_create_circle_all_circle_types() {
        let types = vec![
            CircleType::Neighborhood,
            CircleType::Workplace,
            CircleType::Faith,
            CircleType::Family,
            CircleType::School,
            CircleType::Custom("Test".to_string()),
        ];

        for circle_type in types {
            let mut circle = valid_circle();
            circle.circle_type = circle_type;
            let action = valid_create_action();
            let result = validate_create_circle(action, circle).unwrap();
            assert_eq!(result, ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_create_circle_active_true() {
        let mut circle = valid_circle();
        circle.active = true;
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_circle_active_false() {
        let mut circle = valid_circle();
        circle.active = false;
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // validate_update_circle tests

    #[test]
    fn test_update_circle_valid() {
        let circle = valid_circle();
        let result = validate_update_circle(circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_circle_name_empty() {
        let mut circle = valid_circle();
        circle.name = "".to_string();
        let result = validate_update_circle(circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle name cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_update_circle_max_members_at_minimum() {
        let mut circle = valid_circle();
        circle.max_members = 2;
        let result = validate_update_circle(circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_circle_max_members_below_minimum() {
        let mut circle = valid_circle();
        circle.max_members = 1;
        let result = validate_update_circle(circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle must allow at least 2 members");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_update_circle_name_at_max_length() {
        let mut circle = valid_circle();
        circle.name = "a".repeat(128);
        let result = validate_update_circle(circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_circle_name_over_max_length_rejected() {
        let mut circle = valid_circle();
        circle.name = "a".repeat(129);
        let result = validate_update_circle(circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle name must be 128 characters or fewer");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_update_circle_description_at_max_length() {
        let mut circle = valid_circle();
        circle.description = "d".repeat(2048);
        let result = validate_update_circle(circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_circle_description_over_max_length_rejected() {
        let mut circle = valid_circle();
        circle.description = "d".repeat(2049);
        let result = validate_update_circle(circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle description must be 2048 characters or fewer");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_update_circle_location_at_max_length() {
        let mut circle = valid_circle();
        circle.location = "l".repeat(512);
        let result = validate_update_circle(circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_circle_location_over_max_length_rejected() {
        let mut circle = valid_circle();
        circle.location = "l".repeat(513);
        let result = validate_update_circle(circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Location must be 512 characters or fewer");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_update_circle_max_members_at_maximum() {
        let mut circle = valid_circle();
        circle.max_members = 500;
        let result = validate_update_circle(circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_circle_max_members_over_maximum_rejected() {
        let mut circle = valid_circle();
        circle.max_members = 501;
        let result = validate_update_circle(circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle cannot have more than 500 members");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_update_circle_allows_empty_description() {
        // Update validation allows empty description (may be clearing it)
        let mut circle = valid_circle();
        circle.description = "".to_string();
        let result = validate_update_circle(circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_circle_allows_empty_location() {
        // Update validation allows empty location (may be clearing it)
        let mut circle = valid_circle();
        circle.location = "".to_string();
        let result = validate_update_circle(circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // validate_create_membership tests

    #[test]
    fn test_create_membership_valid() {
        let membership = valid_membership();
        let action = valid_create_action();
        let result = validate_create_membership(action, membership).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_membership_all_roles() {
        let roles = vec![
            MemberRole::Organizer,
            MemberRole::Member,
            MemberRole::Observer,
        ];

        for role in roles {
            let mut membership = valid_membership();
            membership.role = role;
            let action = valid_create_action();
            let result = validate_create_membership(action, membership).unwrap();
            assert_eq!(result, ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_create_membership_active_true() {
        let mut membership = valid_membership();
        membership.active = true;
        let action = valid_create_action();
        let result = validate_create_membership(action, membership).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_membership_active_false() {
        let mut membership = valid_membership();
        membership.active = false;
        let action = valid_create_action();
        let result = validate_create_membership(action, membership).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // Entry type serde tests

    #[test]
    fn test_care_circle_serde_roundtrip() {
        let circle = valid_circle();
        let serialized = serde_json::to_string(&circle).unwrap();
        let deserialized: CareCircle = serde_json::from_str(&serialized).unwrap();
        assert_eq!(circle, deserialized);
    }

    #[test]
    fn test_circle_membership_serde_roundtrip() {
        let membership = valid_membership();
        let serialized = serde_json::to_string(&membership).unwrap();
        let deserialized: CircleMembership = serde_json::from_str(&serialized).unwrap();
        assert_eq!(membership, deserialized);
    }

    #[test]
    fn test_anchor_serde_roundtrip() {
        let anchor = crate::Anchor("test_anchor".to_string());
        let serialized = serde_json::to_string(&anchor).unwrap();
        let deserialized: crate::Anchor = serde_json::from_str(&serialized).unwrap();
        assert_eq!(anchor, deserialized);
    }

    // Boundary value tests

    #[test]
    fn test_create_circle_name_one_char() {
        let mut circle = valid_circle();
        circle.name = "A".to_string();
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_circle_description_one_char() {
        let mut circle = valid_circle();
        circle.description = "A".to_string();
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_circle_location_one_char() {
        let mut circle = valid_circle();
        circle.location = "X".to_string();
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_circle_max_members_typical_values() {
        let values = vec![3, 10, 50, 100, 250, 499];
        for max_members in values {
            let mut circle = valid_circle();
            circle.max_members = max_members;
            let action = valid_create_action();
            let result = validate_create_circle(action, circle).unwrap();
            assert_eq!(result, ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_create_circle_unicode_in_name() {
        let mut circle = valid_circle();
        circle.name = "Círculo de Cuidado 🌟".to_string();
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_circle_unicode_in_description() {
        let mut circle = valid_circle();
        circle.description = "A circle for mutual aid and support 💙🤝".to_string();
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_circle_unicode_in_location() {
        let mut circle = valid_circle();
        circle.location = "東京、日本".to_string();
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_circle_type_custom_empty_string() {
        let ct = CircleType::Custom("".to_string());
        assert_eq!(ct.anchor_key(), "custom_");
    }

    #[test]
    fn test_circle_type_custom_multiple_spaces() {
        let ct = CircleType::Custom("Multiple   Spaces   Here".to_string());
        assert_eq!(ct.anchor_key(), "custom_multiple___spaces___here");
    }

    #[test]
    fn test_create_circle_max_members_zero() {
        let mut circle = valid_circle();
        circle.max_members = 0;
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle must allow at least 2 members");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_update_circle_max_members_zero() {
        let mut circle = valid_circle();
        circle.max_members = 0;
        let result = validate_update_circle(circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle must allow at least 2 members");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    // ── Link tag validation tests ───────────────────────────────────────

    fn assert_valid_result(result: ExternResult<ValidateCallbackResult>) {
        match result {
            Ok(ValidateCallbackResult::Valid) => {}
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                panic!("Expected Valid, got Invalid: {msg}")
            }
            other => panic!("Expected Valid, got {other:?}"),
        }
    }

    fn assert_invalid_result(result: ExternResult<ValidateCallbackResult>, expected_substr: &str) {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains(expected_substr),
                    "Expected Invalid message containing '{expected_substr}', got: '{msg}'"
                );
            }
            Ok(ValidateCallbackResult::Valid) => {
                panic!("Expected Invalid containing '{expected_substr}', got Valid")
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    /// Helper to simulate the create link tag validation logic (same as in validate()).
    fn validate_create_link_tag(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::AllCircles => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllCircles link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TypeToCircle => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TypeToCircle link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CircleToMembership => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "CircleToMembership link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToMembership => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToMembership link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToCreatedCircle => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToCreatedCircle link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    fn validate_delete_link_tag(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::AllCircles => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllCircles delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TypeToCircle => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TypeToCircle delete link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CircleToMembership => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "CircleToMembership delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToMembership => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToMembership delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToCreatedCircle => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToCreatedCircle delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    // -- AllCircles link tag tests --

    #[test]
    fn test_link_all_circles_valid_tag() {
        assert_valid_result(validate_create_link_tag(
            LinkTypes::AllCircles,
            vec![0u8; 64],
        ));
    }

    #[test]
    fn test_link_all_circles_empty_tag() {
        assert_valid_result(validate_create_link_tag(LinkTypes::AllCircles, vec![]));
    }

    #[test]
    fn test_link_all_circles_tag_at_limit() {
        assert_valid_result(validate_create_link_tag(
            LinkTypes::AllCircles,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_link_all_circles_tag_too_long_rejected() {
        assert_invalid_result(
            validate_create_link_tag(LinkTypes::AllCircles, vec![0u8; 257]),
            "AllCircles link tag too long",
        );
    }

    // -- TypeToCircle link tag tests (512 byte limit) --

    #[test]
    fn test_link_type_to_circle_valid_tag() {
        assert_valid_result(validate_create_link_tag(
            LinkTypes::TypeToCircle,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_link_type_to_circle_tag_at_limit() {
        assert_valid_result(validate_create_link_tag(
            LinkTypes::TypeToCircle,
            vec![0u8; 512],
        ));
    }

    #[test]
    fn test_link_type_to_circle_tag_too_long_rejected() {
        assert_invalid_result(
            validate_create_link_tag(LinkTypes::TypeToCircle, vec![0u8; 513]),
            "TypeToCircle link tag too long",
        );
    }

    // -- CircleToMembership link tag tests --

    #[test]
    fn test_link_circle_to_membership_valid_tag() {
        assert_valid_result(validate_create_link_tag(
            LinkTypes::CircleToMembership,
            vec![0u8; 100],
        ));
    }

    #[test]
    fn test_link_circle_to_membership_tag_at_limit() {
        assert_valid_result(validate_create_link_tag(
            LinkTypes::CircleToMembership,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_link_circle_to_membership_tag_too_long_rejected() {
        assert_invalid_result(
            validate_create_link_tag(LinkTypes::CircleToMembership, vec![0u8; 257]),
            "CircleToMembership link tag too long",
        );
    }

    // -- AgentToMembership link tag tests --

    #[test]
    fn test_link_agent_to_membership_valid_tag() {
        assert_valid_result(validate_create_link_tag(
            LinkTypes::AgentToMembership,
            vec![0u8; 128],
        ));
    }

    #[test]
    fn test_link_agent_to_membership_tag_too_long_rejected() {
        assert_invalid_result(
            validate_create_link_tag(LinkTypes::AgentToMembership, vec![0u8; 257]),
            "AgentToMembership link tag too long",
        );
    }

    // -- AgentToCreatedCircle link tag tests --

    #[test]
    fn test_link_agent_to_created_circle_valid_tag() {
        assert_valid_result(validate_create_link_tag(
            LinkTypes::AgentToCreatedCircle,
            vec![0u8; 128],
        ));
    }

    #[test]
    fn test_link_agent_to_created_circle_tag_at_limit() {
        assert_valid_result(validate_create_link_tag(
            LinkTypes::AgentToCreatedCircle,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_link_agent_to_created_circle_tag_too_long_rejected() {
        assert_invalid_result(
            validate_create_link_tag(LinkTypes::AgentToCreatedCircle, vec![0u8; 257]),
            "AgentToCreatedCircle link tag too long",
        );
    }

    // -- Delete link tag tests --

    #[test]
    fn test_delete_link_all_circles_valid_tag() {
        assert_valid_result(validate_delete_link_tag(
            LinkTypes::AllCircles,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_delete_link_all_circles_tag_too_long_rejected() {
        assert_invalid_result(
            validate_delete_link_tag(LinkTypes::AllCircles, vec![0u8; 257]),
            "AllCircles delete link tag too long",
        );
    }

    #[test]
    fn test_delete_link_type_to_circle_valid_tag() {
        assert_valid_result(validate_delete_link_tag(
            LinkTypes::TypeToCircle,
            vec![0u8; 512],
        ));
    }

    #[test]
    fn test_delete_link_type_to_circle_tag_too_long_rejected() {
        assert_invalid_result(
            validate_delete_link_tag(LinkTypes::TypeToCircle, vec![0u8; 513]),
            "TypeToCircle delete link tag too long",
        );
    }

    #[test]
    fn test_delete_link_circle_to_membership_tag_too_long_rejected() {
        assert_invalid_result(
            validate_delete_link_tag(LinkTypes::CircleToMembership, vec![0u8; 257]),
            "CircleToMembership delete link tag too long",
        );
    }

    #[test]
    fn test_delete_link_agent_to_membership_tag_too_long_rejected() {
        assert_invalid_result(
            validate_delete_link_tag(LinkTypes::AgentToMembership, vec![0u8; 257]),
            "AgentToMembership delete link tag too long",
        );
    }

    #[test]
    fn test_delete_link_agent_to_created_circle_tag_too_long_rejected() {
        assert_invalid_result(
            validate_delete_link_tag(LinkTypes::AgentToCreatedCircle, vec![0u8; 257]),
            "AgentToCreatedCircle delete link tag too long",
        );
    }

    // ── CircleType::Custom string length validation tests ──────────────

    #[test]
    fn test_create_circle_custom_type_at_limit() {
        let mut circle = valid_circle();
        circle.circle_type = CircleType::Custom("a".repeat(128));
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_circle_custom_type_too_long() {
        let mut circle = valid_circle();
        circle.circle_type = CircleType::Custom("a".repeat(129));
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(
                    msg,
                    "Custom circle type label must be 128 characters or fewer"
                );
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_create_circle_custom_type_empty() {
        let mut circle = valid_circle();
        circle.circle_type = CircleType::Custom("".to_string());
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Custom circle type label cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_create_circle_custom_type_whitespace_only() {
        let mut circle = valid_circle();
        circle.circle_type = CircleType::Custom("   ".to_string());
        let action = valid_create_action();
        let result = validate_create_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Custom circle type label cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_update_circle_custom_type_at_limit() {
        let mut circle = valid_circle();
        circle.circle_type = CircleType::Custom("a".repeat(128));
        let result = validate_update_circle(circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_circle_custom_type_too_long() {
        let mut circle = valid_circle();
        circle.circle_type = CircleType::Custom("a".repeat(129));
        let result = validate_update_circle(circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(
                    msg,
                    "Custom circle type label must be 128 characters or fewer"
                );
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_update_circle_custom_type_empty() {
        let mut circle = valid_circle();
        circle.circle_type = CircleType::Custom("".to_string());
        let result = validate_update_circle(circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Custom circle type label cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }
}
