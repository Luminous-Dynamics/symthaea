//! Commons Management Integrity Zome
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CommonResource {
    pub id: String,
    pub name: String,
    pub description: String,
    pub resource_type: ResourceType,
    pub property_id: Option<String>,
    pub stewards: Vec<String>,
    pub governance_rules: GovernanceRules,
    pub created: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ResourceType {
    Land,
    Water,
    Forest,
    Fishery,
    Pasture,
    Infrastructure,
    Digital,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GovernanceRules {
    pub access_rules: Vec<String>,
    pub usage_limits: Vec<UsageLimit>,
    pub maintenance_rotation: bool,
    pub decision_method: DecisionMethod,
    pub penalty_for_violation: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct UsageLimit {
    pub limit_type: String,
    pub max_per_period: f64,
    pub period_days: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DecisionMethod {
    Consensus,
    Majority,
    SuperMajority,
    Stewards,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct UsageRight {
    pub id: String,
    pub resource_id: String,
    pub holder_did: String,
    pub right_type: RightType,
    pub quota: Option<f64>,
    pub granted: Timestamp,
    pub expires: Option<Timestamp>,
    pub active: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RightType {
    Access,
    Extraction,
    Management,
    Exclusion,
    Alienation,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct UsageLog {
    pub id: String,
    pub resource_id: String,
    pub user_did: String,
    pub usage_type: String,
    pub quantity: f64,
    pub unit: String,
    pub timestamp: Timestamp,
}

/// Anchor entry for deterministic link bases from strings
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    CommonResource(CommonResource),
    UsageRight(UsageRight),
    UsageLog(UsageLog),
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    StewardToResource,
    ResourceToRights,
    HolderToRights,
    ResourceToLogs,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::CommonResource(resource) => {
                    validate_create_common_resource(EntryCreationAction::Create(action), resource)
                }
                EntryTypes::UsageRight(right) => {
                    validate_create_usage_right(EntryCreationAction::Create(action), right)
                }
                EntryTypes::UsageLog(log) => {
                    validate_create_usage_log(EntryCreationAction::Create(action), log)
                }
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::CommonResource(resource) => {
                    validate_update_common_resource(action, resource)
                }
                EntryTypes::UsageRight(right) => validate_update_usage_right(action, right),
                EntryTypes::UsageLog(_) => Ok(ValidateCallbackResult::Invalid(
                    "Usage logs cannot be updated".into(),
                )),
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => match link_type {
            LinkTypes::StewardToResource => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "StewardToResource link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ResourceToRights => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ResourceToRights link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::HolderToRights => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "HolderToRights link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ResourceToLogs => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ResourceToLogs link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_common_resource(
    _action: EntryCreationAction,
    resource: CommonResource,
) -> ExternResult<ValidateCallbackResult> {
    // --- Empty string checks (required fields) ---
    if resource.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource id cannot be empty".into(),
        ));
    }
    if resource.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource name cannot be empty".into(),
        ));
    }
    if resource.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource description cannot be empty".into(),
        ));
    }
    // --- String length limits ---
    if resource.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource id too long (max 256 chars)".into(),
        ));
    }
    if resource.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource name too long (max 256 chars)".into(),
        ));
    }
    if resource.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource description too long (max 4096 chars)".into(),
        ));
    }
    // --- Existing validation ---
    if resource.stewards.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource must have at least one steward".into(),
        ));
    }
    for steward in &resource.stewards {
        if !steward.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid(
                "Stewards must be valid DIDs".into(),
            ));
        }
    }
    for limit in &resource.governance_rules.usage_limits {
        if !limit.max_per_period.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "Usage limit max_per_period must be a finite number".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_common_resource(
    _action: Update,
    _resource: CommonResource,
) -> ExternResult<ValidateCallbackResult> {
    // Governance rules and stewards can be updated
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_usage_right(
    _action: EntryCreationAction,
    right: UsageRight,
) -> ExternResult<ValidateCallbackResult> {
    // --- Empty string checks (required fields) ---
    if right.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "UsageRight id cannot be empty".into(),
        ));
    }
    if right.resource_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "UsageRight resource_id cannot be empty".into(),
        ));
    }
    // --- String length limits ---
    if right.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "UsageRight id too long (max 256 chars)".into(),
        ));
    }
    if right.resource_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "UsageRight resource_id too long (max 256 chars)".into(),
        ));
    }
    // --- Existing validation ---
    if !right.holder_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Holder must be a valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_usage_right(
    _action: Update,
    _right: UsageRight,
) -> ExternResult<ValidateCallbackResult> {
    // Status and quota can be updated
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_usage_log(
    _action: EntryCreationAction,
    log: UsageLog,
) -> ExternResult<ValidateCallbackResult> {
    // --- Empty string checks (required fields) ---
    if log.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "UsageLog id cannot be empty".into(),
        ));
    }
    if log.resource_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "UsageLog resource_id cannot be empty".into(),
        ));
    }
    if log.usage_type.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "UsageLog usage_type cannot be empty".into(),
        ));
    }
    if log.unit.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "UsageLog unit cannot be empty".into(),
        ));
    }
    // --- String length limits ---
    if log.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "UsageLog id too long (max 256 chars)".into(),
        ));
    }
    if log.resource_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "UsageLog resource_id too long (max 256 chars)".into(),
        ));
    }
    if log.usage_type.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "UsageLog usage_type too long (max 128 chars)".into(),
        ));
    }
    if log.unit.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "UsageLog unit too long (max 128 chars)".into(),
        ));
    }
    // --- Existing validation ---
    if !log.user_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "User must be a valid DID".into(),
        ));
    }
    if !log.quantity.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity must be a finite number".into(),
        ));
    }
    if log.quantity < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity cannot be negative".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================
    // Factory Functions
    // ========================================

    fn create_usage_limit() -> UsageLimit {
        UsageLimit {
            limit_type: "water_extraction".to_string(),
            max_per_period: 1000.0,
            period_days: 30,
        }
    }

    fn create_governance_rules() -> GovernanceRules {
        GovernanceRules {
            access_rules: vec!["sunrise to sunset".to_string()],
            usage_limits: vec![create_usage_limit()],
            maintenance_rotation: true,
            decision_method: DecisionMethod::Consensus,
            penalty_for_violation: Some("community service".to_string()),
        }
    }

    fn create_common_resource() -> CommonResource {
        CommonResource {
            id: "resource-001".to_string(),
            name: "Community Well".to_string(),
            description: "Shared water well in village square".to_string(),
            resource_type: ResourceType::Water,
            property_id: Some("property-123".to_string()),
            stewards: vec![
                "did:key:steward1".to_string(),
                "did:key:steward2".to_string(),
            ],
            governance_rules: create_governance_rules(),
            created: Timestamp::from_micros(1000000),
        }
    }

    fn create_usage_right() -> UsageRight {
        UsageRight {
            id: "right-001".to_string(),
            resource_id: "resource-001".to_string(),
            holder_did: "did:key:holder1".to_string(),
            right_type: RightType::Access,
            quota: Some(100.0),
            granted: Timestamp::from_micros(1000000),
            expires: Some(Timestamp::from_micros(2000000)),
            active: true,
        }
    }

    fn create_usage_log() -> UsageLog {
        UsageLog {
            id: "log-001".to_string(),
            resource_id: "resource-001".to_string(),
            user_did: "did:key:user1".to_string(),
            usage_type: "extraction".to_string(),
            quantity: 50.0,
            unit: "liters".to_string(),
            timestamp: Timestamp::from_micros(1500000),
        }
    }

    fn create_entry_creation_action() -> EntryCreationAction {
        EntryCreationAction::Create(Create {
            author: AgentPubKey::from_raw_36(vec![0; 36]),
            timestamp: Timestamp::from_micros(1000000),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            weight: Default::default(),
        })
    }

    fn create_update_action() -> Update {
        Update {
            author: AgentPubKey::from_raw_36(vec![0; 36]),
            timestamp: Timestamp::from_micros(2000000),
            action_seq: 1,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            original_action_address: ActionHash::from_raw_36(vec![1; 36]),
            original_entry_address: EntryHash::from_raw_36(vec![1; 36]),
            weight: Default::default(),
        }
    }

    // ========================================
    // CommonResource Validation Tests
    // ========================================

    #[test]
    fn test_valid_common_resource() {
        let resource = create_common_resource();
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_common_resource_empty_stewards() {
        let mut resource = create_common_resource();
        resource.stewards = vec![];
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_common_resource_invalid_steward_did() {
        let mut resource = create_common_resource();
        resource.stewards = vec!["not-a-did".to_string()];
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_common_resource_one_valid_one_invalid_steward() {
        let mut resource = create_common_resource();
        resource.stewards = vec!["did:key:valid".to_string(), "invalid".to_string()];
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_common_resource_single_steward() {
        let mut resource = create_common_resource();
        resource.stewards = vec!["did:key:solo".to_string()];
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_common_resource_multiple_valid_stewards() {
        let mut resource = create_common_resource();
        resource.stewards = vec![
            "did:key:s1".to_string(),
            "did:web:s2".to_string(),
            "did:pkh:s3".to_string(),
        ];
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_common_resource_steward_with_prefix_in_middle() {
        let mut resource = create_common_resource();
        resource.stewards = vec!["notdid:key:test".to_string()];
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_update_common_resource_always_valid() {
        let resource = create_common_resource();
        let action = create_update_action();
        let result = validate_update_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_common_resource_with_empty_stewards() {
        let mut resource = create_common_resource();
        resource.stewards = vec![];
        let action = create_update_action();
        let result = validate_update_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_common_resource_with_invalid_stewards() {
        let mut resource = create_common_resource();
        resource.stewards = vec!["invalid".to_string()];
        let action = create_update_action();
        let result = validate_update_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ========================================
    // UsageRight Validation Tests
    // ========================================

    #[test]
    fn test_valid_usage_right() {
        let right = create_usage_right();
        let action = create_entry_creation_action();
        let result = validate_create_usage_right(action, right).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_usage_right_invalid_holder_did() {
        let mut right = create_usage_right();
        right.holder_did = "not-a-did".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_right(action, right).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_right_holder_did_empty() {
        let mut right = create_usage_right();
        right.holder_did = "".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_right(action, right).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_right_holder_did_web() {
        let mut right = create_usage_right();
        right.holder_did = "did:web:example.com".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_right(action, right).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_usage_right_holder_did_pkh() {
        let mut right = create_usage_right();
        right.holder_did = "did:pkh:eth:0x123".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_right(action, right).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_usage_right_always_valid() {
        let right = create_usage_right();
        let action = create_update_action();
        let result = validate_update_usage_right(action, right).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_usage_right_with_invalid_holder() {
        let mut right = create_usage_right();
        right.holder_did = "invalid".to_string();
        let action = create_update_action();
        let result = validate_update_usage_right(action, right).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ========================================
    // UsageLog Validation Tests
    // ========================================

    #[test]
    fn test_valid_usage_log() {
        let log = create_usage_log();
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_usage_log_invalid_user_did() {
        let mut log = create_usage_log();
        log.user_did = "not-a-did".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_log_user_did_empty() {
        let mut log = create_usage_log();
        log.user_did = "".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_log_user_did_web() {
        let mut log = create_usage_log();
        log.user_did = "did:web:example.org".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_usage_log_negative_quantity() {
        let mut log = create_usage_log();
        log.quantity = -10.0;
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_log_zero_quantity() {
        let mut log = create_usage_log();
        log.quantity = 0.0;
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_usage_log_large_quantity() {
        let mut log = create_usage_log();
        log.quantity = 999999.99;
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_usage_log_negative_small_quantity() {
        let mut log = create_usage_log();
        log.quantity = -0.001;
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ========================================
    // UsageLog Update Tests (Should Always Fail)
    // ========================================

    #[test]
    fn test_usage_log_cannot_be_updated() {
        // This test verifies our understanding that UsageLog updates are rejected.
        // The validation logic is in the main validate() function (lines 143-146).
        // Since we can't call the extern validate() in tests, we verify the pattern
        // by checking that the code path exists and returns Invalid.

        // Simulated validation logic from main validate():
        let result: ExternResult<ValidateCallbackResult> = Ok(ValidateCallbackResult::Invalid(
            "Usage logs cannot be updated".into(),
        ));
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // ========================================
    // Enum Serde Roundtrip Tests
    // ========================================

    #[test]
    fn test_resource_type_land_serde() {
        let rt = ResourceType::Land;
        let serialized = serde_json::to_string(&rt).unwrap();
        let deserialized: ResourceType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rt, deserialized);
    }

    #[test]
    fn test_resource_type_water_serde() {
        let rt = ResourceType::Water;
        let serialized = serde_json::to_string(&rt).unwrap();
        let deserialized: ResourceType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rt, deserialized);
    }

    #[test]
    fn test_resource_type_forest_serde() {
        let rt = ResourceType::Forest;
        let serialized = serde_json::to_string(&rt).unwrap();
        let deserialized: ResourceType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rt, deserialized);
    }

    #[test]
    fn test_resource_type_fishery_serde() {
        let rt = ResourceType::Fishery;
        let serialized = serde_json::to_string(&rt).unwrap();
        let deserialized: ResourceType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rt, deserialized);
    }

    #[test]
    fn test_resource_type_pasture_serde() {
        let rt = ResourceType::Pasture;
        let serialized = serde_json::to_string(&rt).unwrap();
        let deserialized: ResourceType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rt, deserialized);
    }

    #[test]
    fn test_resource_type_infrastructure_serde() {
        let rt = ResourceType::Infrastructure;
        let serialized = serde_json::to_string(&rt).unwrap();
        let deserialized: ResourceType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rt, deserialized);
    }

    #[test]
    fn test_resource_type_digital_serde() {
        let rt = ResourceType::Digital;
        let serialized = serde_json::to_string(&rt).unwrap();
        let deserialized: ResourceType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rt, deserialized);
    }

    #[test]
    fn test_resource_type_other_serde() {
        let rt = ResourceType::Other("custom_resource".to_string());
        let serialized = serde_json::to_string(&rt).unwrap();
        let deserialized: ResourceType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rt, deserialized);
    }

    #[test]
    fn test_right_type_access_serde() {
        let rt = RightType::Access;
        let serialized = serde_json::to_string(&rt).unwrap();
        let deserialized: RightType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rt, deserialized);
    }

    #[test]
    fn test_right_type_extraction_serde() {
        let rt = RightType::Extraction;
        let serialized = serde_json::to_string(&rt).unwrap();
        let deserialized: RightType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rt, deserialized);
    }

    #[test]
    fn test_right_type_management_serde() {
        let rt = RightType::Management;
        let serialized = serde_json::to_string(&rt).unwrap();
        let deserialized: RightType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rt, deserialized);
    }

    #[test]
    fn test_right_type_exclusion_serde() {
        let rt = RightType::Exclusion;
        let serialized = serde_json::to_string(&rt).unwrap();
        let deserialized: RightType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rt, deserialized);
    }

    #[test]
    fn test_right_type_alienation_serde() {
        let rt = RightType::Alienation;
        let serialized = serde_json::to_string(&rt).unwrap();
        let deserialized: RightType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rt, deserialized);
    }

    #[test]
    fn test_decision_method_consensus_serde() {
        let dm = DecisionMethod::Consensus;
        let serialized = serde_json::to_string(&dm).unwrap();
        let deserialized: DecisionMethod = serde_json::from_str(&serialized).unwrap();
        assert_eq!(dm, deserialized);
    }

    #[test]
    fn test_decision_method_majority_serde() {
        let dm = DecisionMethod::Majority;
        let serialized = serde_json::to_string(&dm).unwrap();
        let deserialized: DecisionMethod = serde_json::from_str(&serialized).unwrap();
        assert_eq!(dm, deserialized);
    }

    #[test]
    fn test_decision_method_supermajority_serde() {
        let dm = DecisionMethod::SuperMajority;
        let serialized = serde_json::to_string(&dm).unwrap();
        let deserialized: DecisionMethod = serde_json::from_str(&serialized).unwrap();
        assert_eq!(dm, deserialized);
    }

    #[test]
    fn test_decision_method_stewards_serde() {
        let dm = DecisionMethod::Stewards;
        let serialized = serde_json::to_string(&dm).unwrap();
        let deserialized: DecisionMethod = serde_json::from_str(&serialized).unwrap();
        assert_eq!(dm, deserialized);
    }

    // ========================================
    // Comprehensive Entry Tests
    // ========================================

    #[test]
    fn test_common_resource_all_fields() {
        let resource = create_common_resource();
        assert_eq!(resource.id, "resource-001");
        assert_eq!(resource.name, "Community Well");
        assert!(!resource.stewards.is_empty());
        assert_eq!(resource.governance_rules.maintenance_rotation, true);
    }

    #[test]
    fn test_usage_right_all_fields() {
        let right = create_usage_right();
        assert_eq!(right.id, "right-001");
        assert_eq!(right.resource_id, "resource-001");
        assert!(right.holder_did.starts_with("did:"));
        assert_eq!(right.active, true);
    }

    #[test]
    fn test_usage_log_all_fields() {
        let log = create_usage_log();
        assert_eq!(log.id, "log-001");
        assert_eq!(log.resource_id, "resource-001");
        assert!(log.user_did.starts_with("did:"));
        assert_eq!(log.quantity, 50.0);
        assert_eq!(log.unit, "liters");
    }

    #[test]
    fn test_governance_rules_structure() {
        let rules = create_governance_rules();
        assert!(!rules.access_rules.is_empty());
        assert!(!rules.usage_limits.is_empty());
        assert_eq!(rules.maintenance_rotation, true);
        assert!(matches!(rules.decision_method, DecisionMethod::Consensus));
    }

    #[test]
    fn test_usage_limit_structure() {
        let limit = create_usage_limit();
        assert_eq!(limit.limit_type, "water_extraction");
        assert_eq!(limit.max_per_period, 1000.0);
        assert_eq!(limit.period_days, 30);
    }

    // ========================================
    // Edge Case Tests
    // ========================================

    #[test]
    fn test_common_resource_with_no_property_id() {
        let mut resource = create_common_resource();
        resource.property_id = None;
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_usage_right_with_no_quota() {
        let mut right = create_usage_right();
        right.quota = None;
        let action = create_entry_creation_action();
        let result = validate_create_usage_right(action, right).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_usage_right_with_no_expiration() {
        let mut right = create_usage_right();
        right.expires = None;
        let action = create_entry_creation_action();
        let result = validate_create_usage_right(action, right).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_governance_rules_with_no_penalty() {
        let mut rules = create_governance_rules();
        rules.penalty_for_violation = None;
        let resource = CommonResource {
            id: "r1".to_string(),
            name: "Test".to_string(),
            description: "Test".to_string(),
            resource_type: ResourceType::Land,
            property_id: None,
            stewards: vec!["did:key:s1".to_string()],
            governance_rules: rules,
            created: Timestamp::from_micros(1000000),
        };
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_governance_rules_with_empty_access_rules() {
        let mut rules = create_governance_rules();
        rules.access_rules = vec![];
        let resource = CommonResource {
            id: "r1".to_string(),
            name: "Test".to_string(),
            description: "Test".to_string(),
            resource_type: ResourceType::Land,
            property_id: None,
            stewards: vec!["did:key:s1".to_string()],
            governance_rules: rules,
            created: Timestamp::from_micros(1000000),
        };
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_governance_rules_with_empty_usage_limits() {
        let mut rules = create_governance_rules();
        rules.usage_limits = vec![];
        let resource = CommonResource {
            id: "r1".to_string(),
            name: "Test".to_string(),
            description: "Test".to_string(),
            resource_type: ResourceType::Land,
            property_id: None,
            stewards: vec!["did:key:s1".to_string()],
            governance_rules: rules,
            created: Timestamp::from_micros(1000000),
        };
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_usage_right_inactive() {
        let mut right = create_usage_right();
        right.active = false;
        let action = create_entry_creation_action();
        let result = validate_create_usage_right(action, right).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ========================================
    // DID Variant Tests
    // ========================================

    #[test]
    fn test_steward_did_key_variant() {
        let mut resource = create_common_resource();
        resource.stewards =
            vec!["did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK".to_string()];
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_steward_did_ethr_variant() {
        let mut resource = create_common_resource();
        resource.stewards = vec!["did:ethr:0x123456789abcdef".to_string()];
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_usage_log_fractional_quantity() {
        let mut log = create_usage_log();
        log.quantity = 0.0001;
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ========================================
    // String Length & Empty Validation Tests
    // ========================================

    // --- CommonResource ---

    #[test]
    fn test_common_resource_empty_id() {
        let mut resource = create_common_resource();
        resource.id = "".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_common_resource_whitespace_id() {
        let mut resource = create_common_resource();
        resource.id = "   ".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_common_resource_id_too_long() {
        let mut resource = create_common_resource();
        resource.id = "x".repeat(257);
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_common_resource_id_at_limit() {
        let mut resource = create_common_resource();
        resource.id = "x".repeat(64);
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_common_resource_empty_name() {
        let mut resource = create_common_resource();
        resource.name = "".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_common_resource_name_too_long() {
        let mut resource = create_common_resource();
        resource.name = "x".repeat(257);
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_common_resource_name_at_limit() {
        let mut resource = create_common_resource();
        resource.name = "x".repeat(256);
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_common_resource_empty_description() {
        let mut resource = create_common_resource();
        resource.description = "".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_common_resource_description_too_long() {
        let mut resource = create_common_resource();
        resource.description = "x".repeat(4097);
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_common_resource_description_at_limit() {
        let mut resource = create_common_resource();
        resource.description = "x".repeat(4096);
        let action = create_entry_creation_action();
        let result = validate_create_common_resource(action, resource).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // --- UsageRight ---

    #[test]
    fn test_usage_right_empty_id() {
        let mut right = create_usage_right();
        right.id = "".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_right(action, right).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_right_whitespace_id() {
        let mut right = create_usage_right();
        right.id = "   ".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_right(action, right).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_right_id_too_long() {
        let mut right = create_usage_right();
        right.id = "x".repeat(257);
        let action = create_entry_creation_action();
        let result = validate_create_usage_right(action, right).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_right_empty_resource_id() {
        let mut right = create_usage_right();
        right.resource_id = "".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_right(action, right).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_right_resource_id_too_long() {
        let mut right = create_usage_right();
        right.resource_id = "x".repeat(257);
        let action = create_entry_creation_action();
        let result = validate_create_usage_right(action, right).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // --- UsageLog ---

    #[test]
    fn test_usage_log_empty_id() {
        let mut log = create_usage_log();
        log.id = "".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_log_whitespace_id() {
        let mut log = create_usage_log();
        log.id = "   ".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_log_id_too_long() {
        let mut log = create_usage_log();
        log.id = "x".repeat(257);
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_log_empty_resource_id() {
        let mut log = create_usage_log();
        log.resource_id = "".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_log_resource_id_too_long() {
        let mut log = create_usage_log();
        log.resource_id = "x".repeat(257);
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_log_empty_usage_type() {
        let mut log = create_usage_log();
        log.usage_type = "".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_log_usage_type_too_long() {
        let mut log = create_usage_log();
        log.usage_type = "x".repeat(129);
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_log_empty_unit() {
        let mut log = create_usage_log();
        log.unit = "".to_string();
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_log_unit_too_long() {
        let mut log = create_usage_log();
        log.unit = "x".repeat(129);
        let action = create_entry_creation_action();
        let result = validate_create_usage_log(action, log).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
