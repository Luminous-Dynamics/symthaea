//! Credentials Integrity Zome
//! Defines entry types and validation for care provider credentials and references.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Type of care credential
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CredentialType {
    FirstAid,
    CPR,
    ChildcareTraining,
    ElderCare,
    MentalHealthFirstAid,
    BackgroundCheck,
    DrivingLicense,
    SpecialNeeds,
    Other(String),
}

impl CredentialType {
    pub fn anchor_key(&self) -> String {
        match self {
            CredentialType::FirstAid => "firstaid".to_string(),
            CredentialType::CPR => "cpr".to_string(),
            CredentialType::ChildcareTraining => "childcare_training".to_string(),
            CredentialType::ElderCare => "eldercare".to_string(),
            CredentialType::MentalHealthFirstAid => "mhfa".to_string(),
            CredentialType::BackgroundCheck => "background_check".to_string(),
            CredentialType::DrivingLicense => "driving".to_string(),
            CredentialType::SpecialNeeds => "special_needs".to_string(),
            CredentialType::Other(s) => format!("other_{}", s.to_lowercase().replace(' ', "_")),
        }
    }
}

/// A verifiable care credential held by a provider
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CareCredential {
    /// The agent holding this credential
    pub holder: AgentPubKey,
    /// Type of credential
    pub credential_type: CredentialType,
    /// Who issued the credential (agent key or external identifier)
    pub issuer: String,
    /// When the credential was issued
    pub issued_at: Timestamp,
    /// When the credential expires (if applicable)
    pub expires_at: Option<Timestamp>,
    /// Whether this credential has been verified by a trusted party
    pub verified: bool,
    /// Additional metadata (JSON)
    pub metadata: String,
}

/// A reference/testimonial from a care recipient about a provider
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CareReference {
    /// The care provider being referenced
    pub provider: AgentPubKey,
    /// The recipient giving the reference
    pub from_recipient: AgentPubKey,
    /// Rating (1-5)
    pub rating: u8,
    /// Written comment
    pub comment: String,
    /// Type of care that was provided
    pub care_type: String,
    /// When the reference was created
    pub created_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    CareCredential(CareCredential),
    CareReference(CareReference),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Agent to their credentials
    AgentToCredential,
    /// Credential type to credentials of that type
    TypeToCredential,
    /// Agent to references they have received
    AgentToReference,
    /// Agent to references they have given
    AgentGivenReferences,
    /// All verified credentials
    AllVerifiedCredentials,
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
                EntryTypes::CareCredential(cred) => validate_create_credential(action, cred),
                EntryTypes::CareReference(reference) => {
                    validate_create_reference(action, reference)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CareCredential(cred) => validate_update_credential(cred),
                EntryTypes::CareReference(reference) => validate_update_reference(reference),
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
            LinkTypes::AgentToCredential => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToCredential link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TypeToCredential => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TypeToCredential link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToReference => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToReference link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentGivenReferences => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentGivenReferences link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllVerifiedCredentials => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllVerifiedCredentials link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_credential_type_other(credential_type: &CredentialType) -> Result<(), String> {
    if let CredentialType::Other(s) = credential_type {
        if s.trim().is_empty() {
            return Err("Custom credential type label cannot be empty".to_string());
        }
        if s.len() > 128 {
            return Err("Custom credential type label must be 128 characters or fewer".to_string());
        }
    }
    Ok(())
}

fn validate_create_credential(
    _action: Create,
    cred: CareCredential,
) -> ExternResult<ValidateCallbackResult> {
    if cred.issuer.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer cannot be empty".into(),
        ));
    }
    if cred.issuer.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer must be 512 characters or fewer".into(),
        ));
    }
    if cred.metadata.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Metadata must be 4096 characters or fewer".into(),
        ));
    }
    // If metadata is non-empty, validate it is valid JSON
    if !cred.metadata.trim().is_empty()
        && serde_json::from_str::<serde_json::Value>(&cred.metadata).is_err()
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Metadata must be valid JSON".into(),
        ));
    }
    // If expires_at is set, it must be after issued_at
    if let Some(expires) = cred.expires_at {
        if expires <= cred.issued_at {
            return Ok(ValidateCallbackResult::Invalid(
                "Expiry must be after issuance".into(),
            ));
        }
    }
    if let Err(msg) = validate_credential_type_other(&cred.credential_type) {
        return Ok(ValidateCallbackResult::Invalid(msg));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_credential(cred: CareCredential) -> ExternResult<ValidateCallbackResult> {
    if cred.issuer.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer cannot be empty".into(),
        ));
    }
    if cred.issuer.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer must be 512 characters or fewer".into(),
        ));
    }
    if cred.metadata.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Metadata must be 4096 characters or fewer".into(),
        ));
    }
    if !cred.metadata.trim().is_empty()
        && serde_json::from_str::<serde_json::Value>(&cred.metadata).is_err()
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Metadata must be valid JSON".into(),
        ));
    }
    if let Some(expires) = cred.expires_at {
        if expires <= cred.issued_at {
            return Ok(ValidateCallbackResult::Invalid(
                "Expiry must be after issuance".into(),
            ));
        }
    }
    if let Err(msg) = validate_credential_type_other(&cred.credential_type) {
        return Ok(ValidateCallbackResult::Invalid(msg));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_reference(
    _action: Create,
    reference: CareReference,
) -> ExternResult<ValidateCallbackResult> {
    if reference.rating < 1 || reference.rating > 5 {
        return Ok(ValidateCallbackResult::Invalid(
            "Rating must be between 1 and 5".into(),
        ));
    }
    if reference.comment.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Comment cannot be empty".into(),
        ));
    }
    if reference.comment.len() > 2048 {
        return Ok(ValidateCallbackResult::Invalid(
            "Comment must be 2048 characters or fewer".into(),
        ));
    }
    if reference.care_type.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Care type cannot be empty".into(),
        ));
    }
    if reference.care_type.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Care type must be 128 characters or fewer".into(),
        ));
    }
    if reference.provider == reference.from_recipient {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot write a reference for yourself".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_reference(reference: CareReference) -> ExternResult<ValidateCallbackResult> {
    if reference.rating < 1 || reference.rating > 5 {
        return Ok(ValidateCallbackResult::Invalid(
            "Rating must be between 1 and 5".into(),
        ));
    }
    if reference.comment.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Comment cannot be empty".into(),
        ));
    }
    if reference.comment.len() > 2048 {
        return Ok(ValidateCallbackResult::Invalid(
            "Comment must be 2048 characters or fewer".into(),
        ));
    }
    if reference.care_type.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Care type cannot be empty".into(),
        ));
    }
    if reference.care_type.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Care type must be 128 characters or fewer".into(),
        ));
    }
    if reference.provider == reference.from_recipient {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot write a reference for yourself".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // Factory Functions
    // ============================================================================

    fn valid_agent_key() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xdb; 36])
    }

    fn valid_agent_key_2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xac; 36])
    }

    fn valid_timestamp() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn later_timestamp() -> Timestamp {
        Timestamp::from_micros(2_000_000)
    }

    fn valid_create_action() -> Create {
        Create {
            author: valid_agent_key(),
            timestamp: valid_timestamp(),
            action_seq: 0,
            prev_action: holo_hash::ActionHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: holo_hash::EntryHash::from_raw_36(vec![0; 36]),
            weight: Default::default(),
        }
    }

    fn valid_credential() -> CareCredential {
        CareCredential {
            holder: valid_agent_key(),
            credential_type: CredentialType::FirstAid,
            issuer: "Red Cross Chapter 42".to_string(),
            issued_at: valid_timestamp(),
            expires_at: Some(later_timestamp()),
            verified: false,
            metadata: "{}".to_string(),
        }
    }

    fn valid_reference() -> CareReference {
        CareReference {
            provider: valid_agent_key(),
            from_recipient: valid_agent_key_2(),
            rating: 5,
            comment: "Excellent care provided with professionalism and compassion.".to_string(),
            care_type: "Eldercare".to_string(),
            created_at: valid_timestamp(),
        }
    }

    // ============================================================================
    // CredentialType Tests
    // ============================================================================

    #[test]
    fn test_credential_type_anchor_key_first_aid() {
        assert_eq!(CredentialType::FirstAid.anchor_key(), "firstaid");
    }

    #[test]
    fn test_credential_type_anchor_key_cpr() {
        assert_eq!(CredentialType::CPR.anchor_key(), "cpr");
    }

    #[test]
    fn test_credential_type_anchor_key_childcare_training() {
        assert_eq!(
            CredentialType::ChildcareTraining.anchor_key(),
            "childcare_training"
        );
    }

    #[test]
    fn test_credential_type_anchor_key_elder_care() {
        assert_eq!(CredentialType::ElderCare.anchor_key(), "eldercare");
    }

    #[test]
    fn test_credential_type_anchor_key_mental_health_first_aid() {
        assert_eq!(CredentialType::MentalHealthFirstAid.anchor_key(), "mhfa");
    }

    #[test]
    fn test_credential_type_anchor_key_background_check() {
        assert_eq!(
            CredentialType::BackgroundCheck.anchor_key(),
            "background_check"
        );
    }

    #[test]
    fn test_credential_type_anchor_key_driving_license() {
        assert_eq!(CredentialType::DrivingLicense.anchor_key(), "driving");
    }

    #[test]
    fn test_credential_type_anchor_key_special_needs() {
        assert_eq!(CredentialType::SpecialNeeds.anchor_key(), "special_needs");
    }

    #[test]
    fn test_credential_type_anchor_key_other() {
        assert_eq!(
            CredentialType::Other("Massage Therapy".to_string()).anchor_key(),
            "other_massage_therapy"
        );
    }

    #[test]
    fn test_credential_type_anchor_key_other_with_spaces() {
        assert_eq!(
            CredentialType::Other("Advanced Life Support".to_string()).anchor_key(),
            "other_advanced_life_support"
        );
    }

    #[test]
    fn test_credential_type_serde_first_aid() {
        let cred_type = CredentialType::FirstAid;
        let json = serde_json::to_string(&cred_type).unwrap();
        let deserialized: CredentialType = serde_json::from_str(&json).unwrap();
        assert_eq!(cred_type, deserialized);
    }

    #[test]
    fn test_credential_type_serde_cpr() {
        let cred_type = CredentialType::CPR;
        let json = serde_json::to_string(&cred_type).unwrap();
        let deserialized: CredentialType = serde_json::from_str(&json).unwrap();
        assert_eq!(cred_type, deserialized);
    }

    #[test]
    fn test_credential_type_serde_childcare() {
        let cred_type = CredentialType::ChildcareTraining;
        let json = serde_json::to_string(&cred_type).unwrap();
        let deserialized: CredentialType = serde_json::from_str(&json).unwrap();
        assert_eq!(cred_type, deserialized);
    }

    #[test]
    fn test_credential_type_serde_elder_care() {
        let cred_type = CredentialType::ElderCare;
        let json = serde_json::to_string(&cred_type).unwrap();
        let deserialized: CredentialType = serde_json::from_str(&json).unwrap();
        assert_eq!(cred_type, deserialized);
    }

    #[test]
    fn test_credential_type_serde_mhfa() {
        let cred_type = CredentialType::MentalHealthFirstAid;
        let json = serde_json::to_string(&cred_type).unwrap();
        let deserialized: CredentialType = serde_json::from_str(&json).unwrap();
        assert_eq!(cred_type, deserialized);
    }

    #[test]
    fn test_credential_type_serde_background_check() {
        let cred_type = CredentialType::BackgroundCheck;
        let json = serde_json::to_string(&cred_type).unwrap();
        let deserialized: CredentialType = serde_json::from_str(&json).unwrap();
        assert_eq!(cred_type, deserialized);
    }

    #[test]
    fn test_credential_type_serde_driving_license() {
        let cred_type = CredentialType::DrivingLicense;
        let json = serde_json::to_string(&cred_type).unwrap();
        let deserialized: CredentialType = serde_json::from_str(&json).unwrap();
        assert_eq!(cred_type, deserialized);
    }

    #[test]
    fn test_credential_type_serde_special_needs() {
        let cred_type = CredentialType::SpecialNeeds;
        let json = serde_json::to_string(&cred_type).unwrap();
        let deserialized: CredentialType = serde_json::from_str(&json).unwrap();
        assert_eq!(cred_type, deserialized);
    }

    #[test]
    fn test_credential_type_serde_other() {
        let cred_type = CredentialType::Other("Acupuncture License".to_string());
        let json = serde_json::to_string(&cred_type).unwrap();
        let deserialized: CredentialType = serde_json::from_str(&json).unwrap();
        assert_eq!(cred_type, deserialized);
    }

    // ============================================================================
    // CareCredential Validation Tests
    // ============================================================================

    #[test]
    fn test_validate_credential_valid() {
        let cred = valid_credential();
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_credential_empty_issuer() {
        let mut cred = valid_credential();
        cred.issuer = "".to_string();
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Issuer cannot be empty");
        }
    }

    #[test]
    fn test_validate_credential_issuer_too_long() {
        let mut cred = valid_credential();
        cred.issuer = "a".repeat(513);
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Issuer must be 512 characters or fewer");
        }
    }

    #[test]
    fn test_validate_credential_issuer_exactly_512() {
        let mut cred = valid_credential();
        cred.issuer = "a".repeat(512);
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_credential_metadata_too_long() {
        let mut cred = valid_credential();
        cred.metadata = "a".repeat(4097);
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Metadata must be 4096 characters or fewer");
        }
    }

    #[test]
    fn test_validate_credential_metadata_exactly_4096() {
        let mut cred = valid_credential();
        // Valid JSON that's exactly 4096 chars
        let value = "x".repeat(4096 - 2);
        cred.metadata = format!("\"{}\"", value);
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_credential_metadata_empty_string() {
        let mut cred = valid_credential();
        cred.metadata = "".to_string();
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_credential_metadata_valid_json_object() {
        let mut cred = valid_credential();
        cred.metadata = r#"{"license_number":"12345","state":"CA"}"#.to_string();
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_credential_metadata_valid_json_array() {
        let mut cred = valid_credential();
        cred.metadata = r#"["certification1","certification2"]"#.to_string();
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_credential_metadata_valid_json_string() {
        let mut cred = valid_credential();
        cred.metadata = r#""simple string""#.to_string();
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_credential_metadata_valid_json_number() {
        let mut cred = valid_credential();
        cred.metadata = "12345".to_string();
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_credential_metadata_invalid_json() {
        let mut cred = valid_credential();
        cred.metadata = "{invalid json}".to_string();
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Metadata must be valid JSON");
        }
    }

    #[test]
    fn test_validate_credential_metadata_invalid_json_unclosed_brace() {
        let mut cred = valid_credential();
        cred.metadata = r#"{"key":"value""#.to_string();
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Metadata must be valid JSON");
        }
    }

    #[test]
    fn test_validate_credential_metadata_plain_text_not_json() {
        let mut cred = valid_credential();
        cred.metadata = "This is just plain text".to_string();
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Metadata must be valid JSON");
        }
    }

    #[test]
    fn test_validate_credential_expires_at_after_issued_at() {
        let mut cred = valid_credential();
        cred.issued_at = valid_timestamp();
        cred.expires_at = Some(later_timestamp());
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_credential_expires_at_before_issued_at() {
        let mut cred = valid_credential();
        cred.issued_at = later_timestamp();
        cred.expires_at = Some(valid_timestamp());
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Expiry must be after issuance");
        }
    }

    #[test]
    fn test_validate_credential_expires_at_equal_to_issued_at() {
        let mut cred = valid_credential();
        cred.issued_at = valid_timestamp();
        cred.expires_at = Some(valid_timestamp());
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Expiry must be after issuance");
        }
    }

    #[test]
    fn test_validate_credential_no_expiry() {
        let mut cred = valid_credential();
        cred.expires_at = None;
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_credential_verified_true() {
        let mut cred = valid_credential();
        cred.verified = true;
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_credential_verified_false() {
        let mut cred = valid_credential();
        cred.verified = false;
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_credential_all_credential_types() {
        let types = vec![
            CredentialType::FirstAid,
            CredentialType::CPR,
            CredentialType::ChildcareTraining,
            CredentialType::ElderCare,
            CredentialType::MentalHealthFirstAid,
            CredentialType::BackgroundCheck,
            CredentialType::DrivingLicense,
            CredentialType::SpecialNeeds,
            CredentialType::Other("Custom".to_string()),
        ];

        for cred_type in types {
            let mut cred = valid_credential();
            cred.credential_type = cred_type;
            let result = validate_create_credential(valid_create_action(), cred).unwrap();
            assert_eq!(result, ValidateCallbackResult::Valid);
        }
    }

    // ============================================================================
    // CareReference Validation Tests
    // ============================================================================

    #[test]
    fn test_validate_reference_valid() {
        let reference = valid_reference();
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_reference_rating_1() {
        let mut reference = valid_reference();
        reference.rating = 1;
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_reference_rating_2() {
        let mut reference = valid_reference();
        reference.rating = 2;
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_reference_rating_3() {
        let mut reference = valid_reference();
        reference.rating = 3;
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_reference_rating_4() {
        let mut reference = valid_reference();
        reference.rating = 4;
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_reference_rating_5() {
        let mut reference = valid_reference();
        reference.rating = 5;
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_reference_rating_0() {
        let mut reference = valid_reference();
        reference.rating = 0;
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Rating must be between 1 and 5");
        }
    }

    #[test]
    fn test_validate_reference_rating_6() {
        let mut reference = valid_reference();
        reference.rating = 6;
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Rating must be between 1 and 5");
        }
    }

    #[test]
    fn test_validate_reference_rating_255() {
        let mut reference = valid_reference();
        reference.rating = 255;
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Rating must be between 1 and 5");
        }
    }

    #[test]
    fn test_validate_reference_comment_empty() {
        let mut reference = valid_reference();
        reference.comment = "".to_string();
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Comment cannot be empty");
        }
    }

    #[test]
    fn test_validate_reference_comment_too_long() {
        let mut reference = valid_reference();
        reference.comment = "a".repeat(2049);
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Comment must be 2048 characters or fewer");
        }
    }

    #[test]
    fn test_validate_reference_comment_exactly_2048() {
        let mut reference = valid_reference();
        reference.comment = "a".repeat(2048);
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_reference_care_type_empty() {
        let mut reference = valid_reference();
        reference.care_type = "".to_string();
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Care type cannot be empty");
        }
    }

    #[test]
    fn test_validate_reference_care_type_too_long() {
        let mut reference = valid_reference();
        reference.care_type = "a".repeat(129);
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Care type must be 128 characters or fewer");
        }
    }

    #[test]
    fn test_validate_reference_care_type_exactly_128() {
        let mut reference = valid_reference();
        reference.care_type = "a".repeat(128);
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_reference_self_reference() {
        let mut reference = valid_reference();
        reference.provider = valid_agent_key();
        reference.from_recipient = valid_agent_key();
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Cannot write a reference for yourself");
        }
    }

    #[test]
    fn test_validate_reference_different_agents() {
        let mut reference = valid_reference();
        reference.provider = valid_agent_key();
        reference.from_recipient = valid_agent_key_2();
        let result = validate_create_reference(valid_create_action(), reference).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ============================================================================
    // Entry Struct Tests
    // ============================================================================

    #[test]
    fn test_care_credential_clone() {
        let cred = valid_credential();
        let cloned = cred.clone();
        assert_eq!(cred, cloned);
    }

    #[test]
    fn test_care_reference_clone() {
        let reference = valid_reference();
        let cloned = reference.clone();
        assert_eq!(reference, cloned);
    }

    #[test]
    fn test_anchor_clone() {
        let anchor = Anchor("test_anchor".to_string());
        let cloned = anchor.clone();
        assert_eq!(anchor, cloned);
    }

    // ============================================================================
    // Link Tag Validation Tests
    // ============================================================================

    fn validate_create_link_tag(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::AgentToCredential => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToCredential link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TypeToCredential => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TypeToCredential link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToReference => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToReference link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentGivenReferences => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentGivenReferences link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllVerifiedCredentials => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllVerifiedCredentials link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    #[test]
    fn test_link_tag_agent_to_credential_at_limit() {
        let result =
            validate_create_link_tag(LinkTypes::AgentToCredential, vec![0u8; 256]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_agent_to_credential_over_limit() {
        let result =
            validate_create_link_tag(LinkTypes::AgentToCredential, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_type_to_credential_at_limit() {
        let result = validate_create_link_tag(LinkTypes::TypeToCredential, vec![0u8; 512]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_type_to_credential_over_limit() {
        let result = validate_create_link_tag(LinkTypes::TypeToCredential, vec![0u8; 513]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_all_verified_credentials_at_limit() {
        let result =
            validate_create_link_tag(LinkTypes::AllVerifiedCredentials, vec![0u8; 256]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_all_verified_credentials_over_limit() {
        let result =
            validate_create_link_tag(LinkTypes::AllVerifiedCredentials, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_empty_tag_valid() {
        let result = validate_create_link_tag(LinkTypes::AgentToCredential, vec![]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_credential_type_equality() {
        assert_eq!(CredentialType::FirstAid, CredentialType::FirstAid);
        assert_ne!(CredentialType::FirstAid, CredentialType::CPR);
        assert_eq!(
            CredentialType::Other("Test".to_string()),
            CredentialType::Other("Test".to_string())
        );
        assert_ne!(
            CredentialType::Other("Test1".to_string()),
            CredentialType::Other("Test2".to_string())
        );
    }

    // ============================================================================
    // CredentialType::Other string length validation tests
    // ============================================================================

    #[test]
    fn test_validate_credential_other_type_at_limit() {
        let mut cred = valid_credential();
        cred.credential_type = CredentialType::Other("a".repeat(128));
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_credential_other_type_too_long() {
        let mut cred = valid_credential();
        cred.credential_type = CredentialType::Other("a".repeat(129));
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(
                msg,
                "Custom credential type label must be 128 characters or fewer"
            );
        }
    }

    #[test]
    fn test_validate_credential_other_type_empty() {
        let mut cred = valid_credential();
        cred.credential_type = CredentialType::Other("".to_string());
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Custom credential type label cannot be empty");
        }
    }

    #[test]
    fn test_validate_credential_other_type_whitespace_only() {
        let mut cred = valid_credential();
        cred.credential_type = CredentialType::Other("   ".to_string());
        let result = validate_create_credential(valid_create_action(), cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Custom credential type label cannot be empty");
        }
    }

    // ============================================================================
    // validate_update_credential tests
    // ============================================================================

    #[test]
    fn test_update_credential_valid() {
        let cred = valid_credential();
        let result = validate_update_credential(cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_credential_empty_issuer() {
        let mut cred = valid_credential();
        cred.issuer = "".to_string();
        let result = validate_update_credential(cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Issuer cannot be empty");
        }
    }

    #[test]
    fn test_update_credential_issuer_at_limit() {
        let mut cred = valid_credential();
        cred.issuer = "a".repeat(512);
        let result = validate_update_credential(cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_credential_issuer_too_long() {
        let mut cred = valid_credential();
        cred.issuer = "a".repeat(513);
        let result = validate_update_credential(cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Issuer must be 512 characters or fewer");
        }
    }

    #[test]
    fn test_update_credential_metadata_at_limit() {
        let mut cred = valid_credential();
        let value = "x".repeat(4096 - 2);
        cred.metadata = format!("\"{}\"", value);
        let result = validate_update_credential(cred).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_credential_metadata_too_long() {
        let mut cred = valid_credential();
        cred.metadata = "a".repeat(4097);
        let result = validate_update_credential(cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Metadata must be 4096 characters or fewer");
        }
    }

    #[test]
    fn test_update_credential_metadata_invalid_json() {
        let mut cred = valid_credential();
        cred.metadata = "{bad json}".to_string();
        let result = validate_update_credential(cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Metadata must be valid JSON");
        }
    }

    #[test]
    fn test_update_credential_expires_before_issued() {
        let mut cred = valid_credential();
        cred.issued_at = later_timestamp();
        cred.expires_at = Some(valid_timestamp());
        let result = validate_update_credential(cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Expiry must be after issuance");
        }
    }

    #[test]
    fn test_update_credential_other_type_too_long() {
        let mut cred = valid_credential();
        cred.credential_type = CredentialType::Other("a".repeat(129));
        let result = validate_update_credential(cred).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(
                msg,
                "Custom credential type label must be 128 characters or fewer"
            );
        }
    }

    // ============================================================================
    // validate_update_reference tests
    // ============================================================================

    #[test]
    fn test_update_reference_valid() {
        let reference = valid_reference();
        let result = validate_update_reference(reference).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_reference_rating_zero() {
        let mut reference = valid_reference();
        reference.rating = 0;
        let result = validate_update_reference(reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Rating must be between 1 and 5");
        }
    }

    #[test]
    fn test_update_reference_rating_six() {
        let mut reference = valid_reference();
        reference.rating = 6;
        let result = validate_update_reference(reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Rating must be between 1 and 5");
        }
    }

    #[test]
    fn test_update_reference_comment_empty() {
        let mut reference = valid_reference();
        reference.comment = "".to_string();
        let result = validate_update_reference(reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Comment cannot be empty");
        }
    }

    #[test]
    fn test_update_reference_comment_at_limit() {
        let mut reference = valid_reference();
        reference.comment = "a".repeat(2048);
        let result = validate_update_reference(reference).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_reference_comment_too_long() {
        let mut reference = valid_reference();
        reference.comment = "a".repeat(2049);
        let result = validate_update_reference(reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Comment must be 2048 characters or fewer");
        }
    }

    #[test]
    fn test_update_reference_care_type_empty() {
        let mut reference = valid_reference();
        reference.care_type = "".to_string();
        let result = validate_update_reference(reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Care type cannot be empty");
        }
    }

    #[test]
    fn test_update_reference_care_type_at_limit() {
        let mut reference = valid_reference();
        reference.care_type = "a".repeat(128);
        let result = validate_update_reference(reference).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_reference_care_type_too_long() {
        let mut reference = valid_reference();
        reference.care_type = "a".repeat(129);
        let result = validate_update_reference(reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Care type must be 128 characters or fewer");
        }
    }

    #[test]
    fn test_update_reference_self_reference() {
        let mut reference = valid_reference();
        reference.provider = valid_agent_key();
        reference.from_recipient = valid_agent_key();
        let result = validate_update_reference(reference).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
        if let ValidateCallbackResult::Invalid(msg) = result {
            assert_eq!(msg, "Cannot write a reference for yourself");
        }
    }
}
