//! Credential Schema Coordinator Zome
//! Business logic for managing credential schemas
//!
//! Updated to use HDK 0.6 patterns

use hdk::prelude::*;
use credential_schema_integrity::*;

/// Create a deterministic entry hash from a string identifier
/// This is used for link bases when we need to link from string IDs
fn string_to_entry_hash(s: &str) -> EntryHash {
    EntryHash::from_raw_36(
        holo_hash::blake2b_256(s.as_bytes())
            .into_iter()
            .chain([0u8; 4])
            .collect::<Vec<u8>>(),
    )
}

/// Create a new credential schema
#[hdk_extern]
pub fn create_schema(schema: CredentialSchema) -> ExternResult<Record> {
    // Capability guard: only the claimed author can create schemas
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);
    if schema.author != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the claimed author can create schemas".into()
        )));
    }

    // Input validation
    if schema.id.is_empty() || schema.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Schema ID must be 1-256 characters".into())));
    }
    if schema.name.is_empty() || schema.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Schema name must be 1-256 characters".into())));
    }
    if schema.description.is_empty() || schema.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Schema description must be 1-4096 characters".into())));
    }
    if schema.version.is_empty() || schema.version.len() > 64 {
        return Err(wasm_error!(WasmErrorInner::Guest("Version must be 1-64 characters".into())));
    }
    if schema.author.is_empty() || schema.author.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Author must be 1-256 characters".into())));
    }
    if schema.schema.is_empty() || schema.schema.len() > 65536 {
        return Err(wasm_error!(WasmErrorInner::Guest("Schema JSON must be 1-65536 characters".into())));
    }
    if schema.required_fields.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest("Required fields must not exceed 100 entries".into())));
    }
    if schema.optional_fields.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest("Optional fields must not exceed 100 entries".into())));
    }
    if schema.credential_type.is_empty() || schema.credential_type.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest("Credential type must have 1-100 entries".into())));
    }

    let action_hash = create_entry(&EntryTypes::CredentialSchema(schema.clone()))?;

    // Link author to schema
    let author_hash = string_to_entry_hash(&schema.author);
    create_link(
        author_hash,
        action_hash.clone(),
        LinkTypes::AuthorToSchema,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find created schema".into()
        )))
}

/// Get schema by ID
#[hdk_extern]
pub fn get_schema(schema_id: String) -> ExternResult<Option<Record>> {
    // Search for schema by ID in the DHT
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CredentialSchema,
        )?))
        .include_entries(true);

    let records = query(filter)?;

    for record in records {
        if let Some(schema) = record
            .entry()
            .to_app_option::<CredentialSchema>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if schema.id == schema_id {
                return Ok(Some(record));
            }
        }
    }

    Ok(None)
}

/// List all schemas by author
#[hdk_extern]
pub fn get_schemas_by_author(author_did: String) -> ExternResult<Vec<Record>> {
    let author_hash = string_to_entry_hash(&author_did);
    let links = get_links(
        LinkQuery::try_new(author_hash, LinkTypes::AuthorToSchema)?,
        GetStrategy::default(),
    )?;

    let mut schemas = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target).map_err(|_| {
            wasm_error!(WasmErrorInner::Guest("Invalid link target".into()))
        })?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            schemas.push(record);
        }
    }

    Ok(schemas)
}

/// Update a schema (creates new version)
#[hdk_extern]
pub fn update_schema(input: UpdateSchemaInput) -> ExternResult<Record> {
    // Input validation
    if input.schema_id.is_empty() || input.schema_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Schema ID must be 1-256 characters".into())));
    }
    if let Some(ref name) = input.name {
        if name.is_empty() || name.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest("Name must be 1-256 characters".into())));
        }
    }
    if let Some(ref description) = input.description {
        if description.is_empty() || description.len() > 4096 {
            return Err(wasm_error!(WasmErrorInner::Guest("Description must be 1-4096 characters".into())));
        }
    }
    if let Some(ref version) = input.version {
        if version.is_empty() || version.len() > 64 {
            return Err(wasm_error!(WasmErrorInner::Guest("Version must be 1-64 characters".into())));
        }
    }
    if let Some(ref schema) = input.schema {
        if schema.is_empty() || schema.len() > 65536 {
            return Err(wasm_error!(WasmErrorInner::Guest("Schema JSON must be 1-65536 characters".into())));
        }
    }
    if let Some(ref fields) = input.required_fields {
        if fields.len() > 100 {
            return Err(wasm_error!(WasmErrorInner::Guest("Required fields must not exceed 100 entries".into())));
        }
    }
    if let Some(ref fields) = input.optional_fields {
        if fields.len() > 100 {
            return Err(wasm_error!(WasmErrorInner::Guest("Optional fields must not exceed 100 entries".into())));
        }
    }

    let current_record = get_schema(input.schema_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Schema not found".into())))?;

    let current_schema: CredentialSchema = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid schema entry".into())))?;

    // Capability guard: only the schema author can update it
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);
    if current_schema.author != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the schema author can update it".into()
        )));
    }

    let now = sys_time()?;

    let updated_schema = CredentialSchema {
        id: current_schema.id,
        name: input.name.unwrap_or(current_schema.name),
        description: input.description.unwrap_or(current_schema.description),
        version: input.version.unwrap_or(current_schema.version),
        author: current_schema.author,
        schema: input.schema.unwrap_or(current_schema.schema),
        required_fields: input.required_fields.unwrap_or(current_schema.required_fields),
        optional_fields: input.optional_fields.unwrap_or(current_schema.optional_fields),
        credential_type: current_schema.credential_type,
        default_expiration: input.default_expiration.unwrap_or(current_schema.default_expiration),
        revocable: current_schema.revocable,
        active: input.active.unwrap_or(current_schema.active),
        created: current_schema.created,
        updated: now,
    };

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::CredentialSchema(updated_schema),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find updated schema".into()
        )))
}

/// Input for updating a schema
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateSchemaInput {
    pub schema_id: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub version: Option<String>,
    pub schema: Option<String>,
    pub required_fields: Option<Vec<String>>,
    pub optional_fields: Option<Vec<String>>,
    pub default_expiration: Option<u64>,
    pub active: Option<bool>,
}

/// Endorse a schema
#[hdk_extern]
pub fn endorse_schema(input: EndorseSchemaInput) -> ExternResult<Record> {
    // Capability guard: only the claimed endorser can create endorsements
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);
    if input.endorser_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the claimed endorser can create endorsements".into()
        )));
    }

    // Input validation
    if input.schema_id.is_empty() || input.schema_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Schema ID must be 1-256 characters".into())));
    }
    if input.endorser_did.is_empty() || input.endorser_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Endorser DID must be 1-256 characters".into())));
    }
    if !(0.0..=1.0).contains(&input.trust_level) {
        return Err(wasm_error!(WasmErrorInner::Guest("Trust level must be between 0.0 and 1.0".into())));
    }
    if let Some(ref comment) = input.comment {
        if comment.len() > 4096 {
            return Err(wasm_error!(WasmErrorInner::Guest("Comment must not exceed 4096 characters".into())));
        }
    }

    let now = sys_time()?;

    let endorsement = SchemaEndorsement {
        schema_id: input.schema_id.clone(),
        endorser: input.endorser_did,
        trust_level: input.trust_level,
        comment: input.comment,
        endorsed_at: now,
    };

    let action_hash = create_entry(&EntryTypes::SchemaEndorsement(endorsement))?;

    // Link schema to endorsement
    let schema_record = get_schema(input.schema_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Schema not found".into())))?;

    create_link(
        schema_record.action_address().clone(),
        action_hash.clone(),
        LinkTypes::SchemaToEndorsement,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find endorsement".into()
        )))
}

/// Input for endorsing a schema
#[derive(Serialize, Deserialize, Debug)]
pub struct EndorseSchemaInput {
    pub schema_id: String,
    pub endorser_did: String,
    pub trust_level: f64,
    pub comment: Option<String>,
}

/// Get all endorsements for a schema
#[hdk_extern]
pub fn get_schema_endorsements(schema_id: String) -> ExternResult<Vec<Record>> {
    let schema_record = get_schema(schema_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Schema not found".into())))?;

    let links = get_links(
        LinkQuery::try_new(
            schema_record.action_address().clone(),
            LinkTypes::SchemaToEndorsement,
        )?,
        GetStrategy::default(),
    )?;

    let mut endorsements = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target).map_err(|_| {
            wasm_error!(WasmErrorInner::Guest("Invalid link target".into()))
        })?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            endorsements.push(record);
        }
    }

    Ok(endorsements)
}

/// List all active schemas
#[hdk_extern]
pub fn list_active_schemas(_: ()) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CredentialSchema,
        )?))
        .include_entries(true);

    let records = query(filter)?;

    let active_schemas: Vec<Record> = records
        .into_iter()
        .filter(|record| {
            record
                .entry()
                .to_app_option::<CredentialSchema>()
                .ok()
                .flatten()
                .map(|s| s.active)
                .unwrap_or(false)
        })
        .collect();

    Ok(active_schemas)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- string_to_entry_hash ---

    #[test]
    fn string_to_entry_hash_deterministic() {
        let h1 = string_to_entry_hash("did:mycelix:alice");
        let h2 = string_to_entry_hash("did:mycelix:alice");
        assert_eq!(h1, h2);
    }

    #[test]
    fn string_to_entry_hash_different_inputs_differ() {
        let h1 = string_to_entry_hash("schema:edu:v1");
        let h2 = string_to_entry_hash("schema:edu:v2");
        assert_ne!(h1, h2);
    }

    #[test]
    fn string_to_entry_hash_empty_string() {
        // Should not panic
        let _ = string_to_entry_hash("");
    }

    // --- UpdateSchemaInput serde ---

    #[test]
    fn update_schema_input_minimal_json() {
        let json = r#"{"schema_id":"schema:test:v1"}"#;
        let input: UpdateSchemaInput = serde_json::from_str(json).unwrap();
        assert_eq!(input.schema_id, "schema:test:v1");
        assert!(input.name.is_none());
        assert!(input.version.is_none());
        assert!(input.active.is_none());
    }

    #[test]
    fn update_schema_input_full_round_trip() {
        let input = UpdateSchemaInput {
            schema_id: "schema:test:v1".into(),
            name: Some("Test Schema".into()),
            description: Some("A test".into()),
            version: Some("2.0.0".into()),
            schema: Some("{}".into()),
            required_fields: Some(vec!["name".into()]),
            optional_fields: Some(vec!["email".into()]),
            default_expiration: Some(86400),
            active: Some(true),
        };
        let json = serde_json::to_string(&input).unwrap();
        let restored: UpdateSchemaInput = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.schema_id, "schema:test:v1");
        assert_eq!(restored.name.as_deref(), Some("Test Schema"));
        assert_eq!(restored.default_expiration, Some(86400));
    }

    // --- EndorseSchemaInput serde ---

    #[test]
    fn endorse_schema_input_serde() {
        let input = EndorseSchemaInput {
            schema_id: "schema:edu:v1".into(),
            endorser_did: "did:mycelix:bob".into(),
            trust_level: 0.85,
            comment: Some("Looks good".into()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let restored: EndorseSchemaInput = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.endorser_did, "did:mycelix:bob");
        assert!((restored.trust_level - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn endorse_schema_input_no_comment() {
        let json = r#"{"schema_id":"s","endorser_did":"d","trust_level":0.5}"#;
        let input: EndorseSchemaInput = serde_json::from_str(json).unwrap();
        assert!(input.comment.is_none());
    }

    // --- SchemaCategory serde ---

    #[test]
    fn schema_category_all_variants_round_trip() {
        let categories = vec![
            SchemaCategory::Education,
            SchemaCategory::Employment,
            SchemaCategory::Identity,
            SchemaCategory::Skills,
            SchemaCategory::Governance,
            SchemaCategory::Financial,
            SchemaCategory::Energy,
            SchemaCategory::Custom("MyApp".into()),
        ];
        for cat in categories {
            let json = serde_json::to_string(&cat).unwrap();
            let restored: SchemaCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, cat);
        }
    }
}
