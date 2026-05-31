// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Credential Schema Integrity Zome
//! Defines entry types and validation for W3C-compatible credential schemas
//!
//! Updated to use HDI 0.7 patterns with FlatOp validation

use hdi::prelude::*;

/// Credential Schema definition
/// Follows W3C Verifiable Credentials Data Model with Mycelix extensions
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CredentialSchema {
    /// Unique schema identifier (e.g., "mycelix:schema:education:degree:v1")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Detailed description
    pub description: String,
    /// Schema version (semver)
    pub version: String,
    /// Schema author's DID
    pub author: String,
    /// JSON Schema for credential subject
    pub schema: String,
    /// Required fields for this credential type
    pub required_fields: Vec<String>,
    /// Optional fields
    pub optional_fields: Vec<String>,
    /// Credential type (maps to VC @type)
    pub credential_type: Vec<String>,
    /// Expiration policy (seconds, 0 = never)
    pub default_expiration: u64,
    /// Whether credentials of this type can be revoked
    pub revocable: bool,
    /// Whether this schema is active
    pub active: bool,
    /// Creation timestamp
    pub created: Timestamp,
    /// Last update timestamp
    pub updated: Timestamp,
}

/// Standard credential categories in the Mycelix ecosystem
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SchemaCategory {
    /// Educational achievements (degrees, certificates, courses)
    Education,
    /// Employment and professional credentials
    Employment,
    /// Identity verification (KYC, age, residence)
    Identity,
    /// Skills and competencies
    Skills,
    /// Governance participation (voting, proposals)
    Governance,
    /// Financial credentials (credit, lending)
    Financial,
    /// Energy sector credentials (operator, investor)
    Energy,
    /// Custom application-specific schemas
    Custom(String),
}

/// Schema endorsement by trusted issuers
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SchemaEndorsement {
    /// Schema being endorsed
    pub schema_id: String,
    /// Endorser's DID
    pub endorser: String,
    /// Trust level (0.0 to 1.0)
    pub trust_level: f64,
    /// Optional comment
    pub comment: Option<String>,
    /// Endorsement timestamp
    pub endorsed_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    CredentialSchema(CredentialSchema),
    SchemaEndorsement(SchemaEndorsement),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Author to their schemas
    AuthorToSchema,
    /// Category to schemas
    CategoryToSchema,
    /// Schema to endorsements
    SchemaToEndorsement,
    /// Schema version history
    SchemaHistory,
}

/// Genesis self-check - called when app is installed
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern matching
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::CredentialSchema(schema) => {
                    validate_create_credential_schema(EntryCreationAction::Create(action), schema)
                }
                EntryTypes::SchemaEndorsement(endorsement) => validate_create_schema_endorsement(
                    EntryCreationAction::Create(action),
                    endorsement,
                ),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::CredentialSchema(schema) => {
                    validate_update_credential_schema(action, schema, original_action_hash)
                }
                EntryTypes::SchemaEndorsement(_) => Ok(ValidateCallbackResult::Invalid(
                    "Schema endorsements cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::AuthorToSchema => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CategoryToSchema => Ok(ValidateCallbackResult::Valid),
            LinkTypes::SchemaToEndorsement => Ok(ValidateCallbackResult::Valid),
            LinkTypes::SchemaHistory => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate schema creation
fn validate_create_credential_schema(
    _action: EntryCreationAction,
    schema: CredentialSchema,
) -> ExternResult<ValidateCallbackResult> {
    // Validate schema ID format
    if !schema.id.starts_with("mycelix:schema:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Schema ID must start with 'mycelix:schema:'".into(),
        ));
    }

    // Validate version format (basic semver)
    if !schema.version.contains('.') {
        return Ok(ValidateCallbackResult::Invalid(
            "Version must be semver format (e.g., 1.0.0)".into(),
        ));
    }

    // Validate JSON Schema is valid JSON
    if serde_json::from_str::<serde_json::Value>(&schema.schema).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "Schema must be valid JSON".into(),
        ));
    }

    // Validate at least one credential type
    if schema.credential_type.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Must specify at least one credential type".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate schema update
fn validate_update_credential_schema(
    _action: Update,
    schema: CredentialSchema,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    // Basic validation - cannot change schema ID or author
    // Note: Full validation would require fetching original entry
    // For now, we validate the updated entry is valid

    if !schema.id.starts_with("mycelix:schema:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Schema ID must start with 'mycelix:schema:'".into(),
        ));
    }

    if serde_json::from_str::<serde_json::Value>(&schema.schema).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "Schema must be valid JSON".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate endorsement creation
fn validate_create_schema_endorsement(
    _action: EntryCreationAction,
    endorsement: SchemaEndorsement,
) -> ExternResult<ValidateCallbackResult> {
    // Validate trust level range
    if !(0.0..=1.0).contains(&endorsement.trust_level) {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust level must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate endorser has DID format
    if !endorsement.endorser.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Endorser must be a valid DID".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
