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
                EntryTypes::CareReference(reference) => validate_create_reference(action, reference),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CareCredential(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CareReference(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
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

fn validate_create_credential(_action: Create, cred: CareCredential) -> ExternResult<ValidateCallbackResult> {
    if cred.issuer.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Issuer cannot be empty".into()));
    }
    if cred.issuer.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid("Issuer must be 512 characters or fewer".into()));
    }
    if cred.metadata.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid("Metadata must be 4096 characters or fewer".into()));
    }
    // If metadata is non-empty, validate it is valid JSON
    if !cred.metadata.is_empty() {
        if serde_json::from_str::<serde_json::Value>(&cred.metadata).is_err() {
            return Ok(ValidateCallbackResult::Invalid("Metadata must be valid JSON".into()));
        }
    }
    // If expires_at is set, it must be after issued_at
    if let Some(expires) = cred.expires_at {
        if expires <= cred.issued_at {
            return Ok(ValidateCallbackResult::Invalid("Expiry must be after issuance".into()));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_reference(_action: Create, reference: CareReference) -> ExternResult<ValidateCallbackResult> {
    if reference.rating < 1 || reference.rating > 5 {
        return Ok(ValidateCallbackResult::Invalid("Rating must be between 1 and 5".into()));
    }
    if reference.comment.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Comment cannot be empty".into()));
    }
    if reference.comment.len() > 2048 {
        return Ok(ValidateCallbackResult::Invalid("Comment must be 2048 characters or fewer".into()));
    }
    if reference.care_type.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Care type cannot be empty".into()));
    }
    if reference.care_type.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid("Care type must be 128 characters or fewer".into()));
    }
    if reference.provider == reference.from_recipient {
        return Ok(ValidateCallbackResult::Invalid("Cannot write a reference for yourself".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}
