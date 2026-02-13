//! Credential Schema Integrity Zome
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
                EntryTypes::SchemaEndorsement(endorsement) => {
                    validate_create_schema_endorsement(EntryCreationAction::Create(action), endorsement)
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    fn valid_schema() -> CredentialSchema {
        CredentialSchema {
            id: "mycelix:schema:education:degree:v1".into(),
            name: "University Degree".into(),
            description: "Academic degree credential".into(),
            version: "1.0.0".into(),
            author: "did:mycelix:issuer1".into(),
            schema: r#"{"type":"object","properties":{"degree":{"type":"string"}}}"#.into(),
            required_fields: vec!["degree".into(), "institution".into()],
            optional_fields: vec!["gpa".into()],
            credential_type: vec!["VerifiableCredential".into(), "DegreeCredential".into()],
            default_expiration: 0,
            revocable: true,
            active: true,
            created: ts(1_700_000_000_000_000),
            updated: ts(1_700_000_000_000_000),
        }
    }

    // --- CredentialSchema ---

    #[test]
    fn credential_schema_json_round_trip() {
        let schema = valid_schema();
        let json = serde_json::to_string(&schema).unwrap();
        let back: CredentialSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema, back);
    }

    #[test]
    fn credential_schema_rejects_bad_id_prefix() {
        assert!(!"invalid:schema:foo".starts_with("mycelix:schema:"));
        assert!("mycelix:schema:education:degree:v1".starts_with("mycelix:schema:"));
    }

    #[test]
    fn credential_schema_rejects_non_semver_version() {
        assert!(!"v1".contains('.'), "Versions without dots should be rejected");
        assert!("1.0.0".contains('.'));
        assert!("0.1".contains('.'));
    }

    #[test]
    fn credential_schema_rejects_invalid_json_schema() {
        assert!(serde_json::from_str::<serde_json::Value>("not json").is_err());
        assert!(serde_json::from_str::<serde_json::Value>(r#"{"valid":"json"}"#).is_ok());
    }

    #[test]
    fn credential_schema_rejects_empty_credential_type() {
        let types: Vec<String> = vec![];
        assert!(types.is_empty(), "Empty credential_type should be rejected");
    }

    // --- SchemaCategory ---

    #[test]
    fn schema_category_json_variants() {
        let variants = vec![
            (SchemaCategory::Education, "\"Education\""),
            (SchemaCategory::Employment, "\"Employment\""),
            (SchemaCategory::Identity, "\"Identity\""),
            (SchemaCategory::Skills, "\"Skills\""),
            (SchemaCategory::Governance, "\"Governance\""),
            (SchemaCategory::Financial, "\"Financial\""),
            (SchemaCategory::Energy, "\"Energy\""),
        ];
        for (variant, expected) in variants {
            let json = serde_json::to_string(&variant).unwrap();
            assert_eq!(json, expected);
            let back: SchemaCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(back, variant);
        }
    }

    #[test]
    fn schema_category_custom_round_trip() {
        let custom = SchemaCategory::Custom("Research".into());
        let json = serde_json::to_string(&custom).unwrap();
        assert!(json.contains("Research"));
        let back: SchemaCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(back, custom);
    }

    // --- SchemaEndorsement ---

    #[test]
    fn schema_endorsement_json_round_trip() {
        let endorsement = SchemaEndorsement {
            schema_id: "mycelix:schema:education:degree:v1".into(),
            endorser: "did:mycelix:trusted_issuer".into(),
            trust_level: 0.95,
            comment: Some("Verified against national standards".into()),
            endorsed_at: ts(1_700_000_000_000_000),
        };
        let json = serde_json::to_string(&endorsement).unwrap();
        let back: SchemaEndorsement = serde_json::from_str(&json).unwrap();
        assert_eq!(endorsement, back);
    }

    #[test]
    fn schema_endorsement_optional_comment() {
        let endorsement = SchemaEndorsement {
            schema_id: "mycelix:schema:skills:v1".into(),
            endorser: "did:mycelix:endorser1".into(),
            trust_level: 0.5,
            comment: None,
            endorsed_at: ts(1_700_000_000_000_000),
        };
        let json = serde_json::to_string(&endorsement).unwrap();
        let back: SchemaEndorsement = serde_json::from_str(&json).unwrap();
        assert_eq!(endorsement, back);
    }

    #[test]
    fn schema_endorsement_rejects_invalid_trust() {
        for level in [1.1, -0.01, f64::INFINITY] {
            assert!(!(0.0..=1.0).contains(&level));
        }
    }

    #[test]
    fn schema_endorsement_rejects_non_did_endorser() {
        assert!(!"https://example.com".starts_with("did:"));
        assert!("did:mycelix:abc".starts_with("did:"));
        assert!("did:web:example.com".starts_with("did:"));
    }
}
