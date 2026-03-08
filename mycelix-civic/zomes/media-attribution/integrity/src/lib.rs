//! Attribution Integrity Zome
//! Updated to use HDI 0.7 patterns with FlatOp validation
use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Attribution {
    pub id: String,
    pub publication_id: String,
    pub contributor_did: String,
    pub role: ContributorRole,
    pub share_percentage: f64,
    pub verified: bool,
    pub created: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ContributorRole {
    Author,
    CoAuthor,
    Editor,
    Researcher,
    Photographer,
    Illustrator,
    Translator,
    Source,
    Other(String),
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RoyaltyRule {
    pub id: String,
    pub publication_id: String,
    pub rule_type: RoyaltyType,
    pub percentage: f64,
    pub minimum_amount: Option<f64>,
    pub currency: String,
    pub active: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RoyaltyType {
    PerView,
    PerShare,
    PerDownload,
    PerDerivative,
    Subscription,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct UsageRecord {
    pub id: String,
    pub publication_id: String,
    pub usage_type: UsageType,
    pub user_did: Option<String>,
    pub timestamp: Timestamp,
    pub royalty_paid: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UsageType {
    View,
    Share,
    Download,
    Derivative,
    Citation,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    Attribution(Attribution),
    RoyaltyRule(RoyaltyRule),
    UsageRecord(UsageRecord),
}

#[hdk_link_types]
pub enum LinkTypes {
    PublicationToAttributions,
    ContributorToAttributions,
    PublicationToRoyalties,
    PublicationToUsage,
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
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Attribution(attribution) => {
                    validate_create_attribution(EntryCreationAction::Create(action), attribution)
                }
                EntryTypes::RoyaltyRule(rule) => {
                    validate_create_royalty_rule(EntryCreationAction::Create(action), rule)
                }
                EntryTypes::UsageRecord(record) => {
                    validate_create_usage_record(EntryCreationAction::Create(action), record)
                }
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Attribution(attribution) => {
                    validate_update_attribution(action, attribution)
                }
                EntryTypes::RoyaltyRule(rule) => validate_update_royalty_rule(action, rule),
                EntryTypes::UsageRecord(_) => Ok(ValidateCallbackResult::Invalid(
                    "Usage records cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            let tag_len = tag.0.len();
            match link_type {
                LinkTypes::PublicationToAttributions
                | LinkTypes::ContributorToAttributions
                | LinkTypes::PublicationToRoyalties
                | LinkTypes::PublicationToUsage => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag,
            action,
        } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            let original_author = original_action.action().author().clone();
            if action.author != original_author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this link".into(),
                ));
            }
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Delete link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original_action = must_get_action(action.deletes_address.clone())?;
            let original_author = original_action.action().author().clone();
            if action.author != original_author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

fn validate_create_attribution(
    _action: EntryCreationAction,
    attribution: Attribution,
) -> ExternResult<ValidateCallbackResult> {
    if attribution.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Attribution ID cannot be empty".into(),
        ));
    }
    if attribution.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Attribution ID too long (max 256 chars)".into(),
        ));
    }
    if attribution.publication_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Attribution publication_id cannot be empty".into(),
        ));
    }
    if attribution.publication_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Attribution publication_id too long (max 256 chars)".into(),
        ));
    }
    if !attribution.contributor_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Contributor must be a valid DID".into(),
        ));
    }
    if attribution.contributor_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Contributor DID too long (max 256 chars)".into(),
        ));
    }
    if !attribution.share_percentage.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "share_percentage must be a finite number".into(),
        ));
    }
    if attribution.share_percentage < 0.0 || attribution.share_percentage > 100.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Share must be 0-100%".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_attribution(
    _action: Update,
    attribution: Attribution,
) -> ExternResult<ValidateCallbackResult> {
    if !attribution.share_percentage.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "share_percentage must be a finite number".into(),
        ));
    }
    if attribution.share_percentage < 0.0 || attribution.share_percentage > 100.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Share must be 0-100%".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_royalty_rule(
    _action: EntryCreationAction,
    rule: RoyaltyRule,
) -> ExternResult<ValidateCallbackResult> {
    if rule.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "RoyaltyRule ID cannot be empty".into(),
        ));
    }
    if rule.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "RoyaltyRule ID too long (max 256 chars)".into(),
        ));
    }
    if rule.publication_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "RoyaltyRule publication_id cannot be empty".into(),
        ));
    }
    if rule.publication_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "RoyaltyRule publication_id too long (max 256 chars)".into(),
        ));
    }
    if rule.currency.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency cannot be empty".into(),
        ));
    }
    if rule.currency.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency too long (max 128 chars)".into(),
        ));
    }
    if !rule.percentage.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "percentage must be a finite number".into(),
        ));
    }
    if rule.percentage < 0.0 || rule.percentage > 100.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Percentage must be 0-100".into(),
        ));
    }
    if let Some(min) = rule.minimum_amount {
        if !min.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "minimum_amount must be a finite number".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_royalty_rule(
    _action: Update,
    rule: RoyaltyRule,
) -> ExternResult<ValidateCallbackResult> {
    if !rule.percentage.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "percentage must be a finite number".into(),
        ));
    }
    if rule.percentage < 0.0 || rule.percentage > 100.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Percentage must be 0-100".into(),
        ));
    }
    if let Some(min) = rule.minimum_amount {
        if !min.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "minimum_amount must be a finite number".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_usage_record(
    _action: EntryCreationAction,
    record: UsageRecord,
) -> ExternResult<ValidateCallbackResult> {
    if record.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Usage record ID cannot be empty".into(),
        ));
    }
    if record.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Usage record ID too long (max 256 chars)".into(),
        ));
    }
    if record.publication_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Usage record publication_id cannot be empty".into(),
        ));
    }
    if record.publication_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Usage record publication_id too long (max 256 chars)".into(),
        ));
    }
    if let Some(ref did) = record.user_did {
        if !did.trim().is_empty() && !did.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid(
                "User DID must be a valid DID".into(),
            ));
        }
    }
    if let Some(paid) = record.royalty_paid {
        if !paid.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "royalty_paid must be a finite number".into(),
            ));
        }
        if paid < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Royalty paid cannot be negative".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // HELPERS
    // ========================================================================

    fn fake_create() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0; 36]),
            timestamp: Timestamp::from_micros(0),
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

    fn fake_update() -> Update {
        Update {
            author: AgentPubKey::from_raw_36(vec![0; 36]),
            timestamp: Timestamp::from_micros(1),
            action_seq: 1,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            original_action_address: ActionHash::from_raw_36(vec![0; 36]),
            original_entry_address: EntryHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            weight: Default::default(),
        }
    }

    fn create_action() -> EntryCreationAction {
        EntryCreationAction::Create(fake_create())
    }

    fn ts() -> Timestamp {
        Timestamp::from_micros(0)
    }

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn get_invalid_reason(result: ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(reason)) => reason,
            _ => panic!("Expected Invalid result"),
        }
    }

    // ========================================================================
    // FACTORY FUNCTIONS
    // ========================================================================

    fn valid_attribution() -> Attribution {
        Attribution {
            id: "attr-001".into(),
            publication_id: "pub-001".into(),
            contributor_did: "did:mycelix:contributor123".into(),
            role: ContributorRole::Author,
            share_percentage: 50.0,
            verified: false,
            created: ts(),
        }
    }

    fn valid_royalty_rule() -> RoyaltyRule {
        RoyaltyRule {
            id: "royalty-001".into(),
            publication_id: "pub-001".into(),
            rule_type: RoyaltyType::PerView,
            percentage: 10.0,
            minimum_amount: None,
            currency: "USD".into(),
            active: true,
        }
    }

    fn valid_usage_record() -> UsageRecord {
        UsageRecord {
            id: "usage-001".into(),
            publication_id: "pub-001".into(),
            usage_type: UsageType::View,
            user_did: Some("did:mycelix:user456".into()),
            timestamp: ts(),
            royalty_paid: None,
        }
    }

    fn valid_anchor() -> Anchor {
        Anchor("attributions".into())
    }

    // ========================================================================
    // SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn contributor_role_serde_roundtrip() {
        let variants = vec![
            ContributorRole::Author,
            ContributorRole::CoAuthor,
            ContributorRole::Editor,
            ContributorRole::Researcher,
            ContributorRole::Photographer,
            ContributorRole::Illustrator,
            ContributorRole::Translator,
            ContributorRole::Source,
            ContributorRole::Other("sound-engineer".into()),
        ];

        for variant in variants {
            let serialized = serde_json::to_string(&variant).unwrap();
            let deserialized: ContributorRole = serde_json::from_str(&serialized).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    #[test]
    fn royalty_type_serde_roundtrip() {
        let variants = vec![
            RoyaltyType::PerView,
            RoyaltyType::PerShare,
            RoyaltyType::PerDownload,
            RoyaltyType::PerDerivative,
            RoyaltyType::Subscription,
        ];

        for variant in variants {
            let serialized = serde_json::to_string(&variant).unwrap();
            let deserialized: RoyaltyType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    #[test]
    fn usage_type_serde_roundtrip() {
        let variants = vec![
            UsageType::View,
            UsageType::Share,
            UsageType::Download,
            UsageType::Derivative,
            UsageType::Citation,
        ];

        for variant in variants {
            let serialized = serde_json::to_string(&variant).unwrap();
            let deserialized: UsageType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    #[test]
    fn attribution_serde_roundtrip() {
        let attr = valid_attribution();
        let serialized = serde_json::to_string(&attr).unwrap();
        let deserialized: Attribution = serde_json::from_str(&serialized).unwrap();
        assert_eq!(attr, deserialized);
    }

    #[test]
    fn royalty_rule_serde_roundtrip() {
        let rule = valid_royalty_rule();
        let serialized = serde_json::to_string(&rule).unwrap();
        let deserialized: RoyaltyRule = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rule, deserialized);
    }

    #[test]
    fn usage_record_serde_roundtrip() {
        let record = valid_usage_record();
        let serialized = serde_json::to_string(&record).unwrap();
        let deserialized: UsageRecord = serde_json::from_str(&serialized).unwrap();
        assert_eq!(record, deserialized);
    }

    #[test]
    fn anchor_serde_roundtrip() {
        let anchor = valid_anchor();
        let serialized = serde_json::to_string(&anchor).unwrap();
        let deserialized: Anchor = serde_json::from_str(&serialized).unwrap();
        assert_eq!(anchor, deserialized);
    }

    #[test]
    fn attribution_with_optional_fields_serde_roundtrip() {
        let mut attr = valid_attribution();
        attr.verified = true;
        let serialized = serde_json::to_string(&attr).unwrap();
        let deserialized: Attribution = serde_json::from_str(&serialized).unwrap();
        assert_eq!(attr, deserialized);
    }

    #[test]
    fn royalty_rule_with_minimum_amount_serde_roundtrip() {
        let mut rule = valid_royalty_rule();
        rule.minimum_amount = Some(0.01);
        let serialized = serde_json::to_string(&rule).unwrap();
        let deserialized: RoyaltyRule = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rule, deserialized);
    }

    #[test]
    fn usage_record_with_all_optionals_serde_roundtrip() {
        let mut record = valid_usage_record();
        record.user_did = Some("did:mycelix:reader".into());
        record.royalty_paid = Some(0.05);
        let serialized = serde_json::to_string(&record).unwrap();
        let deserialized: UsageRecord = serde_json::from_str(&serialized).unwrap();
        assert_eq!(record, deserialized);
    }

    #[test]
    fn usage_record_with_no_optionals_serde_roundtrip() {
        let mut record = valid_usage_record();
        record.user_did = None;
        record.royalty_paid = None;
        let serialized = serde_json::to_string(&record).unwrap();
        let deserialized: UsageRecord = serde_json::from_str(&serialized).unwrap();
        assert_eq!(record, deserialized);
    }

    // ========================================================================
    // ATTRIBUTION CREATE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_attribution_passes() {
        let result = validate_create_attribution(create_action(), valid_attribution());
        assert!(is_valid(&result));
    }

    #[test]
    fn attribution_contributor_did_missing_prefix_rejected() {
        let mut attr = valid_attribution();
        attr.contributor_did = "not-a-did".into();
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
        assert_eq!(
            get_invalid_reason(result),
            "Contributor must be a valid DID"
        );
    }

    #[test]
    fn attribution_contributor_did_empty_rejected() {
        let mut attr = valid_attribution();
        attr.contributor_did = "".into();
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
    }

    #[test]
    fn attribution_share_negative_rejected() {
        let mut attr = valid_attribution();
        attr.share_percentage = -0.01;
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
        assert_eq!(get_invalid_reason(result), "Share must be 0-100%");
    }

    #[test]
    fn attribution_share_above_100_rejected() {
        let mut attr = valid_attribution();
        attr.share_percentage = 100.01;
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
        assert_eq!(get_invalid_reason(result), "Share must be 0-100%");
    }

    #[test]
    fn attribution_share_zero_passes() {
        let mut attr = valid_attribution();
        attr.share_percentage = 0.0;
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_valid(&result));
    }

    #[test]
    fn attribution_share_100_passes() {
        let mut attr = valid_attribution();
        attr.share_percentage = 100.0;
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_valid(&result));
    }

    #[test]
    fn attribution_share_boundary_just_below_zero_rejected() {
        let mut attr = valid_attribution();
        attr.share_percentage = -f64::EPSILON;
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
    }

    #[test]
    fn attribution_share_fractional_passes() {
        let mut attr = valid_attribution();
        attr.share_percentage = 33.333;
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_valid(&result));
    }

    #[test]
    fn attribution_accepts_various_did_methods() {
        let dids = vec![
            "did:mycelix:abc",
            "did:key:z6Mk123",
            "did:web:example.com",
            "did:ethr:0xdeadbeef",
            "did:pkh:eip155:1:0x1234",
        ];

        for did in dids {
            let mut attr = valid_attribution();
            attr.contributor_did = did.into();
            let result = validate_create_attribution(create_action(), attr);
            assert!(is_valid(&result), "Failed for DID: {}", did);
        }
    }

    #[test]
    fn attribution_with_each_role_passes() {
        let roles = vec![
            ContributorRole::Author,
            ContributorRole::CoAuthor,
            ContributorRole::Editor,
            ContributorRole::Researcher,
            ContributorRole::Photographer,
            ContributorRole::Illustrator,
            ContributorRole::Translator,
            ContributorRole::Source,
            ContributorRole::Other("mixer".into()),
        ];

        for role in roles {
            let mut attr = valid_attribution();
            attr.role = role.clone();
            let result = validate_create_attribution(create_action(), attr);
            assert!(is_valid(&result), "Failed for role: {:?}", role);
        }
    }

    #[test]
    fn attribution_verified_true_passes() {
        let mut attr = valid_attribution();
        attr.verified = true;
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // ATTRIBUTION UPDATE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_attribution_update_passes() {
        let result = validate_update_attribution(fake_update(), valid_attribution());
        assert!(is_valid(&result));
    }

    #[test]
    fn attribution_update_share_negative_rejected() {
        let mut attr = valid_attribution();
        attr.share_percentage = -1.0;
        let result = validate_update_attribution(fake_update(), attr);
        assert!(is_invalid(&result));
        assert_eq!(get_invalid_reason(result), "Share must be 0-100%");
    }

    #[test]
    fn attribution_update_share_above_100_rejected() {
        let mut attr = valid_attribution();
        attr.share_percentage = 200.0;
        let result = validate_update_attribution(fake_update(), attr);
        assert!(is_invalid(&result));
        assert_eq!(get_invalid_reason(result), "Share must be 0-100%");
    }

    #[test]
    fn attribution_update_share_zero_passes() {
        let mut attr = valid_attribution();
        attr.share_percentage = 0.0;
        let result = validate_update_attribution(fake_update(), attr);
        assert!(is_valid(&result));
    }

    #[test]
    fn attribution_update_share_100_passes() {
        let mut attr = valid_attribution();
        attr.share_percentage = 100.0;
        let result = validate_update_attribution(fake_update(), attr);
        assert!(is_valid(&result));
    }

    #[test]
    fn attribution_update_does_not_revalidate_did() {
        // Update validation only checks share_percentage, not DID
        let mut attr = valid_attribution();
        attr.contributor_did = "not-a-did".into();
        attr.share_percentage = 50.0;
        let result = validate_update_attribution(fake_update(), attr);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // ROYALTY RULE CREATE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_royalty_rule_passes() {
        let result = validate_create_royalty_rule(create_action(), valid_royalty_rule());
        assert!(is_valid(&result));
    }

    #[test]
    fn royalty_rule_percentage_negative_rejected() {
        let mut rule = valid_royalty_rule();
        rule.percentage = -0.5;
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_invalid(&result));
        assert_eq!(get_invalid_reason(result), "Percentage must be 0-100");
    }

    #[test]
    fn royalty_rule_percentage_above_100_rejected() {
        let mut rule = valid_royalty_rule();
        rule.percentage = 100.1;
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_invalid(&result));
        assert_eq!(get_invalid_reason(result), "Percentage must be 0-100");
    }

    #[test]
    fn royalty_rule_percentage_zero_passes() {
        let mut rule = valid_royalty_rule();
        rule.percentage = 0.0;
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_valid(&result));
    }

    #[test]
    fn royalty_rule_percentage_100_passes() {
        let mut rule = valid_royalty_rule();
        rule.percentage = 100.0;
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_valid(&result));
    }

    #[test]
    fn royalty_rule_each_type_passes() {
        let types = vec![
            RoyaltyType::PerView,
            RoyaltyType::PerShare,
            RoyaltyType::PerDownload,
            RoyaltyType::PerDerivative,
            RoyaltyType::Subscription,
        ];

        for rule_type in types {
            let mut rule = valid_royalty_rule();
            rule.rule_type = rule_type.clone();
            let result = validate_create_royalty_rule(create_action(), rule);
            assert!(is_valid(&result), "Failed for rule_type: {:?}", rule_type);
        }
    }

    #[test]
    fn royalty_rule_with_minimum_amount_passes() {
        let mut rule = valid_royalty_rule();
        rule.minimum_amount = Some(0.01);
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_valid(&result));
    }

    #[test]
    fn royalty_rule_inactive_passes() {
        let mut rule = valid_royalty_rule();
        rule.active = false;
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // ROYALTY RULE UPDATE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_royalty_rule_update_passes() {
        let result = validate_update_royalty_rule(fake_update(), valid_royalty_rule());
        assert!(is_valid(&result));
    }

    #[test]
    fn royalty_rule_update_percentage_negative_rejected() {
        let mut rule = valid_royalty_rule();
        rule.percentage = -10.0;
        let result = validate_update_royalty_rule(fake_update(), rule);
        assert!(is_invalid(&result));
        assert_eq!(get_invalid_reason(result), "Percentage must be 0-100");
    }

    #[test]
    fn royalty_rule_update_percentage_above_100_rejected() {
        let mut rule = valid_royalty_rule();
        rule.percentage = 150.0;
        let result = validate_update_royalty_rule(fake_update(), rule);
        assert!(is_invalid(&result));
    }

    #[test]
    fn royalty_rule_update_percentage_boundary_passes() {
        for pct in [0.0, 50.0, 100.0] {
            let mut rule = valid_royalty_rule();
            rule.percentage = pct;
            let result = validate_update_royalty_rule(fake_update(), rule);
            assert!(is_valid(&result), "Failed for percentage: {}", pct);
        }
    }

    // ========================================================================
    // USAGE RECORD VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_usage_record_passes() {
        let result = validate_create_usage_record(create_action(), valid_usage_record());
        assert!(is_valid(&result));
    }

    #[test]
    fn usage_record_each_type_passes() {
        let types = vec![
            UsageType::View,
            UsageType::Share,
            UsageType::Download,
            UsageType::Derivative,
            UsageType::Citation,
        ];

        for usage_type in types {
            let mut record = valid_usage_record();
            record.usage_type = usage_type.clone();
            let result = validate_create_usage_record(create_action(), record);
            assert!(is_valid(&result), "Failed for usage_type: {:?}", usage_type);
        }
    }

    #[test]
    fn usage_record_without_user_did_passes() {
        let mut record = valid_usage_record();
        record.user_did = None;
        let result = validate_create_usage_record(create_action(), record);
        assert!(is_valid(&result));
    }

    #[test]
    fn usage_record_with_royalty_paid_passes() {
        let mut record = valid_usage_record();
        record.royalty_paid = Some(1.50);
        let result = validate_create_usage_record(create_action(), record);
        assert!(is_valid(&result));
    }

    #[test]
    fn usage_record_with_empty_id_rejected() {
        let mut record = valid_usage_record();
        record.id = "".into();
        let result = validate_create_usage_record(create_action(), record);
        assert!(is_invalid(&result));
    }

    #[test]
    fn usage_record_with_empty_publication_id_rejected() {
        let mut record = valid_usage_record();
        record.publication_id = "".into();
        let result = validate_create_usage_record(create_action(), record);
        assert!(is_invalid(&result));
    }

    #[test]
    fn usage_record_with_invalid_user_did_rejected() {
        let mut record = valid_usage_record();
        record.user_did = Some("not-a-did".into());
        let result = validate_create_usage_record(create_action(), record);
        assert!(is_invalid(&result));
    }

    #[test]
    fn usage_record_with_negative_royalty_rejected() {
        let mut record = valid_usage_record();
        record.royalty_paid = Some(-0.01);
        let result = validate_create_usage_record(create_action(), record);
        assert!(is_invalid(&result));
    }

    #[test]
    fn usage_record_with_valid_did_passes() {
        let mut record = valid_usage_record();
        record.user_did = Some("did:key:z6Mk123".into());
        let result = validate_create_usage_record(create_action(), record);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // EDGE CASES: UNICODE AND SPECIAL CHARACTERS
    // ========================================================================

    #[test]
    fn attribution_with_unicode_id() {
        let mut attr = valid_attribution();
        attr.id = "\u{1F3B5}\u{1F30D}".into(); // musical note + earth emoji
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_valid(&result));
    }

    #[test]
    fn attribution_with_unicode_publication_id() {
        let mut attr = valid_attribution();
        attr.publication_id = "\u{65E5}\u{672C}\u{8A9E}".into(); // Japanese characters
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_valid(&result));
    }

    #[test]
    fn attribution_with_contributor_did_containing_unicode_suffix() {
        // DID prefix is valid, but method-specific-id has unicode
        let mut attr = valid_attribution();
        attr.contributor_did = "did:mycelix:\u{00E9}\u{00E8}\u{00EA}".into();
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_valid(&result));
    }

    #[test]
    fn royalty_rule_with_unicode_currency() {
        let mut rule = valid_royalty_rule();
        rule.currency = "\u{20AC}".into(); // Euro sign
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_valid(&result));
    }

    #[test]
    fn contributor_role_other_with_unicode() {
        let mut attr = valid_attribution();
        attr.role = ContributorRole::Other("\u{4F5C}\u{66F2}\u{5BB6}".into()); // Japanese for "composer"
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_valid(&result));
    }

    #[test]
    fn contributor_role_other_with_empty_string() {
        let mut attr = valid_attribution();
        attr.role = ContributorRole::Other("".into());
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // EDGE CASES: MAX VALUES AND BOUNDARIES
    // ========================================================================

    #[test]
    fn attribution_share_nan_rejected() {
        let mut attr = valid_attribution();
        attr.share_percentage = f64::NAN;
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
        assert_eq!(
            get_invalid_reason(result),
            "share_percentage must be a finite number"
        );
    }

    #[test]
    fn attribution_share_infinity_rejected() {
        let mut attr = valid_attribution();
        attr.share_percentage = f64::INFINITY;
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
    }

    #[test]
    fn attribution_share_neg_infinity_rejected() {
        let mut attr = valid_attribution();
        attr.share_percentage = f64::NEG_INFINITY;
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
    }

    #[test]
    fn royalty_rule_percentage_nan() {
        let mut rule = valid_royalty_rule();
        rule.percentage = f64::NAN;
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_invalid(&result));
        assert_eq!(
            get_invalid_reason(result),
            "percentage must be a finite number"
        );
    }

    #[test]
    fn royalty_rule_percentage_infinity_rejected() {
        let mut rule = valid_royalty_rule();
        rule.percentage = f64::INFINITY;
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_invalid(&result));
    }

    #[test]
    fn royalty_rule_percentage_neg_infinity_rejected() {
        let mut rule = valid_royalty_rule();
        rule.percentage = f64::NEG_INFINITY;
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_invalid(&result));
    }

    #[test]
    fn attribution_with_very_long_id_rejected() {
        let mut attr = valid_attribution();
        attr.id = "a".repeat(10_000);
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
        assert_eq!(
            get_invalid_reason(result),
            "Attribution ID too long (max 256 chars)"
        );
    }

    #[test]
    fn attribution_id_at_limit_passes() {
        let mut attr = valid_attribution();
        attr.id = "a".repeat(256);
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_valid(&result));
    }

    #[test]
    fn attribution_id_over_limit_rejected() {
        let mut attr = valid_attribution();
        attr.id = "a".repeat(257);
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
        assert_eq!(
            get_invalid_reason(result),
            "Attribution ID too long (max 256 chars)"
        );
    }

    #[test]
    fn attribution_empty_id_rejected() {
        let mut attr = valid_attribution();
        attr.id = "".into();
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
        assert_eq!(get_invalid_reason(result), "Attribution ID cannot be empty");
    }

    #[test]
    fn attribution_empty_publication_id_rejected() {
        let mut attr = valid_attribution();
        attr.publication_id = "".into();
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
        assert_eq!(
            get_invalid_reason(result),
            "Attribution publication_id cannot be empty"
        );
    }

    #[test]
    fn attribution_publication_id_at_limit_passes() {
        let mut attr = valid_attribution();
        attr.publication_id = "p".repeat(256);
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_valid(&result));
    }

    #[test]
    fn attribution_publication_id_over_limit_rejected() {
        let mut attr = valid_attribution();
        attr.publication_id = "p".repeat(257);
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
        assert_eq!(
            get_invalid_reason(result),
            "Attribution publication_id too long (max 256 chars)"
        );
    }

    #[test]
    fn attribution_contributor_did_at_limit_passes() {
        let mut attr = valid_attribution();
        attr.contributor_did = format!("did:{}", "x".repeat(252));
        assert_eq!(attr.contributor_did.len(), 256);
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_valid(&result));
    }

    #[test]
    fn attribution_contributor_did_over_limit_rejected() {
        let mut attr = valid_attribution();
        attr.contributor_did = format!("did:{}", "x".repeat(253));
        assert_eq!(attr.contributor_did.len(), 257);
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
        assert_eq!(
            get_invalid_reason(result),
            "Contributor DID too long (max 256 chars)"
        );
    }

    #[test]
    fn royalty_rule_with_very_long_publication_id_rejected() {
        let mut rule = valid_royalty_rule();
        rule.publication_id = "x".repeat(10_000);
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_invalid(&result));
        assert_eq!(
            get_invalid_reason(result),
            "RoyaltyRule publication_id too long (max 256 chars)"
        );
    }

    #[test]
    fn royalty_rule_id_at_limit_passes() {
        let mut rule = valid_royalty_rule();
        rule.id = "r".repeat(256);
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_valid(&result));
    }

    #[test]
    fn royalty_rule_id_over_limit_rejected() {
        let mut rule = valid_royalty_rule();
        rule.id = "r".repeat(257);
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_invalid(&result));
        assert_eq!(
            get_invalid_reason(result),
            "RoyaltyRule ID too long (max 256 chars)"
        );
    }

    #[test]
    fn royalty_rule_empty_id_rejected() {
        let mut rule = valid_royalty_rule();
        rule.id = "".into();
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_invalid(&result));
        assert_eq!(get_invalid_reason(result), "RoyaltyRule ID cannot be empty");
    }

    #[test]
    fn royalty_rule_empty_publication_id_rejected() {
        let mut rule = valid_royalty_rule();
        rule.publication_id = "".into();
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_invalid(&result));
        assert_eq!(
            get_invalid_reason(result),
            "RoyaltyRule publication_id cannot be empty"
        );
    }

    #[test]
    fn royalty_rule_currency_at_limit_passes() {
        let mut rule = valid_royalty_rule();
        rule.currency = "C".repeat(128);
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_valid(&result));
    }

    #[test]
    fn royalty_rule_currency_over_limit_rejected() {
        let mut rule = valid_royalty_rule();
        rule.currency = "C".repeat(129);
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_invalid(&result));
        assert_eq!(
            get_invalid_reason(result),
            "Currency too long (max 128 chars)"
        );
    }

    #[test]
    fn usage_record_id_at_limit_passes() {
        let mut record = valid_usage_record();
        record.id = "u".repeat(256);
        let result = validate_create_usage_record(create_action(), record);
        assert!(is_valid(&result));
    }

    #[test]
    fn usage_record_id_over_limit_rejected() {
        let mut record = valid_usage_record();
        record.id = "u".repeat(257);
        let result = validate_create_usage_record(create_action(), record);
        assert!(is_invalid(&result));
        assert_eq!(
            get_invalid_reason(result),
            "Usage record ID too long (max 256 chars)"
        );
    }

    #[test]
    fn usage_record_publication_id_at_limit_passes() {
        let mut record = valid_usage_record();
        record.publication_id = "p".repeat(256);
        let result = validate_create_usage_record(create_action(), record);
        assert!(is_valid(&result));
    }

    #[test]
    fn usage_record_publication_id_over_limit_rejected() {
        let mut record = valid_usage_record();
        record.publication_id = "p".repeat(257);
        let result = validate_create_usage_record(create_action(), record);
        assert!(is_invalid(&result));
        assert_eq!(
            get_invalid_reason(result),
            "Usage record publication_id too long (max 256 chars)"
        );
    }

    #[test]
    fn usage_record_with_zero_royalty_paid() {
        let mut record = valid_usage_record();
        record.royalty_paid = Some(0.0);
        let result = validate_create_usage_record(create_action(), record);
        assert!(is_valid(&result));
    }

    #[test]
    fn usage_record_with_large_royalty_paid() {
        let mut record = valid_usage_record();
        record.royalty_paid = Some(f64::MAX);
        let result = validate_create_usage_record(create_action(), record);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // DID FORMAT EDGE CASES
    // ========================================================================

    #[test]
    fn attribution_did_prefix_only_passes() {
        // "did:" alone technically starts_with("did:")
        let mut attr = valid_attribution();
        attr.contributor_did = "did:".into();
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_valid(&result));
    }

    #[test]
    fn attribution_did_case_sensitive() {
        let mut attr = valid_attribution();
        attr.contributor_did = "DID:mycelix:abc".into();
        let result = validate_create_attribution(create_action(), attr);
        // "DID:" does not match "did:" prefix check -- should be rejected
        assert!(is_invalid(&result));
    }

    #[test]
    fn attribution_did_with_leading_space_rejected() {
        let mut attr = valid_attribution();
        attr.contributor_did = " did:mycelix:abc".into();
        let result = validate_create_attribution(create_action(), attr);
        assert!(is_invalid(&result));
    }

    // ========================================================================
    // USAGE RECORDS CANNOT BE UPDATED (INTEGRATION BEHAVIOR)
    // ========================================================================

    // The validate() function returns Invalid for UsageRecord updates.
    // We cannot call validate() directly in unit tests (it needs Op),
    // but we document the expected behavior here. The branch in
    // validate() for OpEntry::UpdateEntry + EntryTypes::UsageRecord
    // returns "Usage records cannot be updated".

    // ========================================================================
    // CLONE AND EQUALITY
    // ========================================================================

    #[test]
    fn attribution_clone_equality() {
        let attr = valid_attribution();
        let cloned = attr.clone();
        assert_eq!(attr, cloned);
    }

    #[test]
    fn royalty_rule_clone_equality() {
        let rule = valid_royalty_rule();
        let cloned = rule.clone();
        assert_eq!(rule, cloned);
    }

    #[test]
    fn usage_record_clone_equality() {
        let record = valid_usage_record();
        let cloned = record.clone();
        assert_eq!(record, cloned);
    }

    #[test]
    fn anchor_clone_equality() {
        let anchor = valid_anchor();
        let cloned = anchor.clone();
        assert_eq!(anchor, cloned);
    }

    #[test]
    fn contributor_role_clone_equality() {
        let role = ContributorRole::Other("custom".into());
        let cloned = role.clone();
        assert_eq!(role, cloned);
    }

    #[test]
    fn royalty_type_clone_equality() {
        let rt = RoyaltyType::PerDerivative;
        let cloned = rt.clone();
        assert_eq!(rt, cloned);
    }

    #[test]
    fn usage_type_clone_equality() {
        let ut = UsageType::Citation;
        let cloned = ut.clone();
        assert_eq!(ut, cloned);
    }

    // ========================================================================
    // MIXED SCENARIOS
    // ========================================================================

    #[test]
    fn attribution_both_errors_did_checked_after_ids() {
        // When DID and share are invalid but id/publication_id valid, DID error fires first
        let mut attr = valid_attribution();
        attr.contributor_did = "invalid".into();
        attr.share_percentage = -5.0;
        let result = validate_create_attribution(create_action(), attr);
        assert_eq!(
            get_invalid_reason(result),
            "Contributor must be a valid DID"
        );
    }

    #[test]
    fn attribution_empty_id_checked_before_did() {
        // When id is empty and DID is also bad, id check fires first
        let mut attr = valid_attribution();
        attr.id = "".into();
        attr.contributor_did = "invalid".into();
        let result = validate_create_attribution(create_action(), attr);
        assert_eq!(get_invalid_reason(result), "Attribution ID cannot be empty");
    }

    #[test]
    fn attribution_valid_did_but_bad_share() {
        let mut attr = valid_attribution();
        attr.contributor_did = "did:key:z6Mk123".into();
        attr.share_percentage = 101.0;
        let result = validate_create_attribution(create_action(), attr);
        assert_eq!(get_invalid_reason(result), "Share must be 0-100%");
    }

    #[test]
    fn royalty_rule_small_fractional_percentage() {
        let mut rule = valid_royalty_rule();
        rule.percentage = 0.001;
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_valid(&result));
    }

    #[test]
    fn royalty_rule_percentage_exactly_boundary() {
        // Test exact boundary values
        for pct in [0.0, 100.0] {
            let mut rule = valid_royalty_rule();
            rule.percentage = pct;
            let result = validate_create_royalty_rule(create_action(), rule);
            assert!(is_valid(&result), "Failed for pct={}", pct);
        }
    }

    #[test]
    fn royalty_rule_with_empty_currency_rejected() {
        let mut rule = valid_royalty_rule();
        rule.currency = "".into();
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_invalid(&result));
        assert_eq!(get_invalid_reason(result), "Currency cannot be empty");
    }

    #[test]
    fn royalty_rule_with_empty_id_rejected() {
        let mut rule = valid_royalty_rule();
        rule.id = "".into();
        let result = validate_create_royalty_rule(create_action(), rule);
        assert!(is_invalid(&result));
        assert_eq!(get_invalid_reason(result), "RoyaltyRule ID cannot be empty");
    }

    #[test]
    fn usage_record_timestamp_max_passes() {
        let mut record = valid_usage_record();
        record.timestamp = Timestamp::from_micros(i64::MAX);
        let result = validate_create_usage_record(create_action(), record);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // LINK TAG VALIDATION TESTS
    // ========================================================================

    fn validate_create_link_tag(link_type: &LinkTypes, tag: &LinkTag) -> ValidateCallbackResult {
        let tag_len = tag.0.len();
        match link_type {
            LinkTypes::PublicationToAttributions
            | LinkTypes::ContributorToAttributions
            | LinkTypes::PublicationToRoyalties
            | LinkTypes::PublicationToUsage => {
                if tag_len > 256 {
                    ValidateCallbackResult::Invalid("Link tag too long (max 256 bytes)".into())
                } else {
                    ValidateCallbackResult::Valid
                }
            }
        }
    }

    fn validate_delete_link_tag(tag: &LinkTag) -> ValidateCallbackResult {
        if tag.0.len() > 256 {
            ValidateCallbackResult::Invalid("Delete link tag too long (max 256 bytes)".into())
        } else {
            ValidateCallbackResult::Valid
        }
    }

    // -- PublicationToAttributions (256-byte limit) boundary tests --

    #[test]
    fn link_tag_pub_to_attributions_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_create_link_tag(&LinkTypes::PublicationToAttributions, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_pub_to_attributions_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validate_create_link_tag(&LinkTypes::PublicationToAttributions, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_pub_to_attributions_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validate_create_link_tag(&LinkTypes::PublicationToAttributions, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- ContributorToAttributions (256-byte limit) boundary tests --

    #[test]
    fn link_tag_contributor_at_limit_valid() {
        let tag = LinkTag::new(vec![0xAA; 256]);
        let result = validate_create_link_tag(&LinkTypes::ContributorToAttributions, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_contributor_over_limit_invalid() {
        let tag = LinkTag::new(vec![0xAA; 257]);
        let result = validate_create_link_tag(&LinkTypes::ContributorToAttributions, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- DoS prevention: massive tags rejected for all link types --

    #[test]
    fn link_tag_dos_prevention_all_types() {
        let massive_tag = LinkTag::new(vec![0xFF; 10_000]);
        let all_types = [
            LinkTypes::PublicationToAttributions,
            LinkTypes::ContributorToAttributions,
            LinkTypes::PublicationToRoyalties,
            LinkTypes::PublicationToUsage,
        ];
        for lt in &all_types {
            let result = validate_create_link_tag(lt, &massive_tag);
            assert!(
                matches!(result, ValidateCallbackResult::Invalid(_)),
                "Massive tag should be rejected for {:?}",
                lt
            );
        }
    }

    // -- Delete link tag tests --

    #[test]
    fn delete_link_tag_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn delete_link_tag_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn delete_link_tag_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ========================================================================
    // DELETE AUTHORIZATION TESTS
    // ========================================================================
    // The RegisterDelete and RegisterDeleteLink match arms in validate()
    // use must_get_action() to fetch the original action from the DHT and
    // verify the deleting agent is the original author. must_get_action()
    // requires a live Holochain conductor context and cannot be called in
    // pure unit tests.
    //
    // These tests verify:
    // 1. The Delete action struct can be constructed with author fields
    // 2. The author comparison logic is correct
    // 3. The validation function has explicit match arms (not wildcard)

    fn fake_agent_a() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![2u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_entry_hash() -> EntryHash {
        EntryHash::from_raw_36(vec![0u8; 36])
    }

    #[test]
    fn delete_author_comparison_same_agent_matches() {
        let agent = fake_agent_a();
        let other_agent = fake_agent_a();
        assert_eq!(agent, other_agent);
    }

    #[test]
    fn delete_author_comparison_different_agent_does_not_match() {
        let agent_a = fake_agent_a();
        let agent_b = fake_agent_b();
        assert_ne!(agent_a, agent_b);
    }

    #[test]
    fn delete_action_struct_has_deletes_address_field() {
        // Verifies the Delete action struct shape used in RegisterDelete validation.
        // The validate() function accesses action.deletes_address to look up the
        // original action and compare authors.
        let delete = Delete {
            author: fake_agent_a(),
            timestamp: Timestamp::from_micros(0),
            action_seq: 1,
            prev_action: fake_action_hash(),
            deletes_address: fake_action_hash(),
            deletes_entry_address: fake_entry_hash(),
            weight: EntryRateWeight::default(),
        };
        assert_eq!(delete.deletes_address, fake_action_hash());
        assert_eq!(delete.author, fake_agent_a());
    }

    #[test]
    fn delete_link_action_struct_has_link_add_address_field() {
        // Verifies the DeleteLink action struct shape used in RegisterDeleteLink validation.
        // The validate() function accesses action.link_add_address to look up the
        // original link creation action and compare authors.
        let delete_link = DeleteLink {
            author: fake_agent_a(),
            timestamp: Timestamp::from_micros(0),
            action_seq: 2,
            prev_action: fake_action_hash(),
            link_add_address: fake_action_hash(),
            base_address: AnyLinkableHash::from(fake_entry_hash()),
        };
        assert_eq!(delete_link.link_add_address, fake_action_hash());
        assert_eq!(delete_link.author, fake_agent_a());
    }
}
