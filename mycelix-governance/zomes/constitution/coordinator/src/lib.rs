//! Constitution Coordinator Zome
//! Business logic for charter and amendments
//!
//! Updated to use HDK 0.6 patterns

use hdk::prelude::*;
use constitution_integrity::*;

/// Helper to get or create an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Create or update the charter
#[hdk_extern]
pub fn create_charter(charter: Charter) -> ExternResult<Record> {
    // Input validation
    if charter.id.is_empty() || charter.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Charter ID must be 1-256 characters".into())));
    }
    if charter.preamble.is_empty() || charter.preamble.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Preamble must be 1-4096 characters".into())));
    }

    let action_hash = create_entry(&EntryTypes::Charter(charter))?;

    // Create anchor and link as current charter
    let anchor_entry = Anchor("current_charter".to_string());
    create_entry(&EntryTypes::Anchor(anchor_entry))?;

    create_link(
        anchor_hash("current_charter")?,
        action_hash.clone(),
        LinkTypes::CurrentCharter,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find charter".into()
        )))
}

/// Get the current charter
#[hdk_extern]
pub fn get_current_charter(_: ()) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("current_charter")?, LinkTypes::CurrentCharter)?,
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest_link {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}

/// Propose a constitutional amendment
#[hdk_extern]
pub fn propose_amendment(input: ProposeAmendmentInput) -> ExternResult<Record> {
    // Input validation
    if input.new_text.is_empty() || input.new_text.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("New text must be 1-4096 characters".into())));
    }
    if input.rationale.is_empty() || input.rationale.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Rationale must be 1-4096 characters".into())));
    }
    if input.proposer_did.is_empty() || input.proposer_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Proposer DID must be 1-256 characters".into())));
    }
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Proposal ID must be 1-256 characters".into())));
    }
    if let Some(ref article) = input.article {
        if article.is_empty() || article.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest("Article must be 1-256 characters".into())));
        }
    }
    if let Some(ref original_text) = input.original_text {
        if original_text.len() > 4096 {
            return Err(wasm_error!(WasmErrorInner::Guest("Original text must be at most 4096 characters".into())));
        }
    }

    let now = sys_time()?;
    let amendment_id = format!("amendment:{}:{}", input.proposal_id, now.as_micros());

    // Get current charter version
    let charter_version = if let Some(record) = get_current_charter(())? {
        if let Some(charter) = record
            .entry()
            .to_app_option::<Charter>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            charter.version
        } else {
            1
        }
    } else {
        1
    };

    let amendment = Amendment {
        id: amendment_id,
        charter_version,
        amendment_type: input.amendment_type,
        article: input.article,
        original_text: input.original_text,
        new_text: input.new_text,
        rationale: input.rationale,
        proposer: input.proposer_did.clone(),
        proposal_id: input.proposal_id.clone(),
        status: AmendmentStatus::Draft,
        created: now,
        ratified: None,
    };

    let action_hash = create_entry(&EntryTypes::Amendment(amendment))?;

    // Create anchor and link charter to amendment
    let version_anchor = format!("charter_v{}", charter_version);
    let anchor_entry = Anchor(version_anchor.clone());
    create_entry(&EntryTypes::Anchor(anchor_entry))?;

    create_link(
        anchor_hash(&version_anchor)?,
        action_hash.clone(),
        LinkTypes::CharterToAmendment,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find amendment".into()
        )))
}

/// Input for proposing an amendment
#[derive(Serialize, Deserialize, Debug)]
pub struct ProposeAmendmentInput {
    pub amendment_type: AmendmentType,
    pub article: Option<String>,
    pub original_text: Option<String>,
    pub new_text: String,
    pub rationale: String,
    pub proposer_did: String,
    pub proposal_id: String,
}

/// Ratify an amendment
#[hdk_extern]
pub fn ratify_amendment(amendment_id: String) -> ExternResult<Record> {
    // Find the amendment
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Amendment,
        )?))
        .include_entries(true);

    let records = query(filter)?;

    let mut amendment_record: Option<Record> = None;
    for record in records {
        if let Some(amend) = record
            .entry()
            .to_app_option::<Amendment>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if amend.id == amendment_id {
                amendment_record = Some(record);
                break;
            }
        }
    }

    let current_record = amendment_record.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Amendment not found".into()
    )))?;

    let current_amendment: Amendment = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid amendment entry".into()
        )))?;

    let now = sys_time()?;

    let ratified_amendment = Amendment {
        id: current_amendment.id.clone(),
        charter_version: current_amendment.charter_version,
        amendment_type: current_amendment.amendment_type.clone(),
        article: current_amendment.article.clone(),
        original_text: current_amendment.original_text.clone(),
        new_text: current_amendment.new_text.clone(),
        rationale: current_amendment.rationale.clone(),
        proposer: current_amendment.proposer.clone(),
        proposal_id: current_amendment.proposal_id.clone(),
        status: AmendmentStatus::Ratified,
        created: current_amendment.created,
        ratified: Some(now),
    };

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::Amendment(ratified_amendment),
    )?;

    // Apply amendment to charter
    apply_amendment_to_charter(&current_amendment)?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find ratified amendment".into()
        )))
}

/// Apply an amendment to the charter
fn apply_amendment_to_charter(amendment: &Amendment) -> ExternResult<()> {
    // Get current charter
    let current_charter = get_current_charter(())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No charter found".into())))?;

    let charter: Charter = current_charter
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid charter entry".into()
        )))?;

    let now = sys_time()?;

    // Create new charter version with amendment applied
    let new_charter = match amendment.amendment_type {
        AmendmentType::ModifyPreamble => Charter {
            id: charter.id,
            version: charter.version + 1,
            preamble: amendment.new_text.clone(),
            articles: charter.articles,
            rights: charter.rights,
            amendment_process: charter.amendment_process,
            adopted: charter.adopted,
            last_amended: Some(now),
        },
        _ => {
            // For other types, would need more complex logic
            // For now, just increment version
            Charter {
                id: charter.id,
                version: charter.version + 1,
                preamble: charter.preamble,
                articles: charter.articles, // Would modify based on amendment
                rights: charter.rights,
                amendment_process: charter.amendment_process,
                adopted: charter.adopted,
                last_amended: Some(now),
            }
        }
    };

    // Create new charter version
    create_charter(new_charter)?;

    Ok(())
}

/// Set a governance parameter
#[hdk_extern]
pub fn set_parameter(input: SetParameterInput) -> ExternResult<Record> {
    // Input validation
    if input.name.is_empty() || input.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Parameter name must be 1-256 characters".into())));
    }
    if input.value.is_empty() || input.value.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Parameter value must be 1-4096 characters".into())));
    }
    if input.description.is_empty() || input.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Description must be 1-4096 characters".into())));
    }

    let now = sys_time()?;

    let param = GovernanceParameter {
        name: input.name.clone(),
        value: input.value,
        value_type: input.value_type,
        description: input.description,
        min_value: input.min_value,
        max_value: input.max_value,
        updated: now,
        changed_by_proposal: input.proposal_id,
    };

    let action_hash = create_entry(&EntryTypes::GovernanceParameter(param))?;

    // Create anchor and link to parameter index
    let anchor_entry = Anchor(format!("param:{}", input.name));
    create_entry(&EntryTypes::Anchor(anchor_entry))?;

    create_link(
        anchor_hash(&format!("param:{}", input.name))?,
        action_hash.clone(),
        LinkTypes::ParameterIndex,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find parameter".into()
        )))
}

/// Input for setting a parameter
#[derive(Serialize, Deserialize, Debug)]
pub struct SetParameterInput {
    pub name: String,
    pub value: String,
    pub value_type: ParameterType,
    pub description: String,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
    pub proposal_id: Option<String>,
}

/// Get a governance parameter
#[hdk_extern]
pub fn get_parameter(name: String) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&format!("param:{}", name))?, LinkTypes::ParameterIndex)?,
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest_link {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}

/// List all governance parameters
#[hdk_extern]
pub fn list_parameters(_: ()) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::GovernanceParameter,
        )?))
        .include_entries(true);

    query(filter)
}
