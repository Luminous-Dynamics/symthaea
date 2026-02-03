//! Constitution Integrity Zome
//! Defines entry types and validation for charter and amendments
//!
//! Updated to use HDI 0.7 patterns

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// The constitutional charter
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Charter {
    /// Charter identifier
    pub id: String,
    /// Charter version
    pub version: u32,
    /// Preamble
    pub preamble: String,
    /// Articles (JSON array)
    pub articles: String,
    /// Fundamental rights
    pub rights: Vec<String>,
    /// Amendment process
    pub amendment_process: String,
    /// Adoption timestamp
    pub adopted: Timestamp,
    /// Last amendment timestamp
    pub last_amended: Option<Timestamp>,
}

/// A constitutional amendment
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Amendment {
    /// Amendment identifier
    pub id: String,
    /// Charter version this amends
    pub charter_version: u32,
    /// Amendment type
    pub amendment_type: AmendmentType,
    /// Article being amended (if applicable)
    pub article: Option<String>,
    /// Original text
    pub original_text: Option<String>,
    /// New text
    pub new_text: String,
    /// Rationale
    pub rationale: String,
    /// Proposer's DID
    pub proposer: String,
    /// Linked proposal ID
    pub proposal_id: String,
    /// Amendment status
    pub status: AmendmentStatus,
    /// Creation timestamp
    pub created: Timestamp,
    /// Ratification timestamp
    pub ratified: Option<Timestamp>,
}

/// Types of amendments
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AmendmentType {
    /// Add new article
    AddArticle,
    /// Modify existing article
    ModifyArticle,
    /// Remove article
    RemoveArticle,
    /// Add right
    AddRight,
    /// Modify right
    ModifyRight,
    /// Remove right
    RemoveRight,
    /// Modify preamble
    ModifyPreamble,
    /// Modify amendment process itself
    ModifyProcess,
}

/// Status of an amendment
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AmendmentStatus {
    /// Draft stage
    Draft,
    /// Under deliberation
    Deliberation,
    /// Voting in progress
    Voting,
    /// Ratified and in effect
    Ratified,
    /// Rejected by vote
    Rejected,
    /// Withdrawn by proposer
    Withdrawn,
}

/// Governance parameter
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GovernanceParameter {
    /// Parameter name
    pub name: String,
    /// Current value (JSON)
    pub value: String,
    /// Value type
    pub value_type: ParameterType,
    /// Description
    pub description: String,
    /// Minimum value (if applicable)
    pub min_value: Option<String>,
    /// Maximum value (if applicable)
    pub max_value: Option<String>,
    /// Last update timestamp
    pub updated: Timestamp,
    /// Proposal that last changed this
    pub changed_by_proposal: Option<String>,
}

/// Types of parameters
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ParameterType {
    Integer,
    Float,
    Percentage,
    Duration,
    Boolean,
    String,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Charter(Charter),
    Amendment(Amendment),
    GovernanceParameter(GovernanceParameter),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Current charter
    CurrentCharter,
    /// Charter version history
    CharterHistory,
    /// Charter to amendments
    CharterToAmendment,
    /// Parameter name to parameter
    ParameterIndex,
}

/// HDI 0.7 single validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Charter(charter) => validate_create_charter(action, charter),
                EntryTypes::Amendment(amendment) => validate_create_amendment(action, amendment),
                EntryTypes::GovernanceParameter(param) => validate_create_parameter(action, param),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Charter(charter) => {
                    validate_update_charter(action, charter, original_action_hash)
                }
                EntryTypes::Amendment(amendment) => validate_update_amendment(action, amendment),
                EntryTypes::GovernanceParameter(param) => validate_update_parameter(action, param),
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
            LinkTypes::CurrentCharter => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CharterHistory => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CharterToAmendment => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ParameterIndex => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::CurrentCharter => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate charter creation
fn validate_create_charter(
    _action: Create,
    charter: Charter,
) -> ExternResult<ValidateCallbackResult> {
    // Validate preamble not empty
    if charter.preamble.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Charter must have a preamble".into(),
        ));
    }

    // Validate articles is valid JSON
    if serde_json::from_str::<serde_json::Value>(&charter.articles).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "Articles must be valid JSON".into(),
        ));
    }

    // Validate at least one right
    if charter.rights.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Charter must define fundamental rights".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate charter update
fn validate_update_charter(
    _action: Update,
    _charter: Charter,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    // Charter updates require constitutional amendment process
    // Additional validation could check that an amendment was ratified
    Ok(ValidateCallbackResult::Valid)
}

/// Validate amendment creation
fn validate_create_amendment(
    _action: Create,
    amendment: Amendment,
) -> ExternResult<ValidateCallbackResult> {
    // Validate proposer is a DID
    if !amendment.proposer.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposer must be a valid DID".into(),
        ));
    }

    // Validate new text not empty
    if amendment.new_text.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Amendment must have new text".into(),
        ));
    }

    // Validate rationale provided
    if amendment.rationale.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Amendment must have rationale".into(),
        ));
    }

    // Validate modification amendments have original text
    match amendment.amendment_type {
        AmendmentType::ModifyArticle
        | AmendmentType::ModifyRight
        | AmendmentType::ModifyPreamble => {
            if amendment.original_text.is_none() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Modification amendments must include original text".into(),
                ));
            }
        }
        _ => {}
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate amendment update
fn validate_update_amendment(
    _action: Update,
    _amendment: Amendment,
) -> ExternResult<ValidateCallbackResult> {
    // Amendments can be updated (status changes during ratification process)
    Ok(ValidateCallbackResult::Valid)
}

/// Validate governance parameter creation
fn validate_create_parameter(
    _action: Create,
    param: GovernanceParameter,
) -> ExternResult<ValidateCallbackResult> {
    // Validate name not empty
    if param.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Parameter name cannot be empty".into(),
        ));
    }

    // Validate value is valid JSON
    if serde_json::from_str::<serde_json::Value>(&param.value).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "Parameter value must be valid JSON".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate governance parameter update
fn validate_update_parameter(
    _action: Update,
    param: GovernanceParameter,
) -> ExternResult<ValidateCallbackResult> {
    // Validate value is valid JSON
    if serde_json::from_str::<serde_json::Value>(&param.value).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "Parameter value must be valid JSON".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
