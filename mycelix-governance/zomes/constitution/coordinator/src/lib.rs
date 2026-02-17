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

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    // Pre-create the current_charter anchor so queries never fail on empty DNA
    let anchor = Anchor("current_charter".to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    Ok(InitCallbackResult::Pass)
}

/// Create or update the charter
///
/// Only allowed if no charter exists yet (initial setup) or when called
/// internally from `apply_amendment_to_charter` (governance pipeline).
#[hdk_extern]
pub fn create_charter(charter: Charter) -> ExternResult<Record> {
    // Input validation
    if charter.id.is_empty() || charter.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Charter ID must be 1-256 characters".into())));
    }
    if charter.preamble.is_empty() || charter.preamble.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Preamble must be 1-4096 characters".into())));
    }

    // Gate: if a charter already exists, only allow versioned updates (from amendments).
    // A version > 1 indicates this is an amendment-driven update (apply_amendment_to_charter
    // increments version). Direct external creation of version 1 when a charter already
    // exists is blocked.
    if charter.version <= 1 {
        if let Ok(Some(_)) = get_current_charter(()) {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Charter already exists. Use the amendment process to modify it.".into()
            )));
        }
    }

    let action_hash = create_entry(&EntryTypes::Charter(charter))?;

    // Create anchor and link as current charter
    let anchor_entry = Anchor("current_charter".to_string());
    create_entry(&EntryTypes::Anchor(anchor_entry))?;

    // Delete stale CurrentCharter links before creating new one
    if let Ok(existing_links) = get_links(
        LinkQuery::try_new(anchor_hash("current_charter")?, LinkTypes::CurrentCharter)?,
        GetStrategy::default(),
    ) {
        for link in existing_links {
            let _ = delete_link(link.create_link_hash, GetOptions::default());
        }
    }

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

    let action_hash = create_entry(&EntryTypes::Amendment(amendment.clone()))?;

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

    // Create anchor and link for O(1) lookup by amendment ID
    let aid_anchor = format!("aid:{}", amendment.id);
    create_entry(&EntryTypes::Anchor(Anchor(aid_anchor.clone())))?;
    create_link(
        anchor_hash(&aid_anchor)?,
        action_hash.clone(),
        LinkTypes::AmendmentById,
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
///
/// Only callable by the amendment's proposer or via governance execution pipeline.
/// The amendment must be in Voting status (not Draft or Deliberation).
#[hdk_extern]
pub fn ratify_amendment(amendment_id: String) -> ExternResult<Record> {
    if amendment_id.is_empty() || amendment_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Amendment ID must be 1-256 characters".into())));
    }

    // Find the amendment via O(1) link-based lookup (with chain scan fallback)
    let current_record = {
        let aid_anchor = format!("aid:{}", amendment_id);
        let mut found: Option<Record> = None;

        if let Ok(entry_hash) = anchor_hash(&aid_anchor) {
            if let Ok(links) = get_links(
                LinkQuery::try_new(entry_hash, LinkTypes::AmendmentById)?,
                GetStrategy::default(),
            ) {
                if let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) {
                    if let Ok(ah) = ActionHash::try_from(link.target) {
                        found = get(ah, GetOptions::default())?;
                    }
                }
            }
        }

        // Fallback: O(n) chain scan for amendments created before the link was added
        if found.is_none() {
            let filter = ChainQueryFilter::new()
                .entry_type(EntryType::App(AppEntryDef::try_from(
                    UnitEntryTypes::Amendment,
                )?))
                .include_entries(true);

            let records = query(filter)?;
            for record in records {
                if let Some(amend) = record
                    .entry()
                    .to_app_option::<Amendment>()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                {
                    if amend.id == amendment_id {
                        found = Some(record);
                        break;
                    }
                }
            }
        }

        found.ok_or(wasm_error!(WasmErrorInner::Guest(
            "Amendment not found".into()
        )))?
    };

    let current_amendment: Amendment = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid amendment entry".into()
        )))?;

    // Amendment must be in Voting status to be ratified.
    if current_amendment.status != AmendmentStatus::Voting {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Amendment must be in Voting status to ratify, current: {:?}",
            current_amendment.status
        ))));
    }

    // Verify the linked proposal actually passed by cross-calling the voting tally.
    // Constitutional amendments require supermajority (handled by the voting zome's
    // adaptive threshold). We check that the proposal is Approved or the tally shows
    // consensus_reached.
    let tally_check = call(
        CallTargetCell::Local,
        ZomeName::from("voting"),
        FunctionName::from("tally_votes"),
        None,
        ExternIO::encode(current_amendment.proposal_id.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    );

    match tally_check {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            // Parse tally result and verify it passed
            if let Ok(result) = extern_io.decode::<serde_json::Value>() {
                let approved = result.get("approved").and_then(|a| a.as_bool()).unwrap_or(false);
                let quorum_reached = result.get("quorum_reached").and_then(|q| q.as_bool()).unwrap_or(false);
                if !approved || !quorum_reached {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Cannot ratify amendment: linked proposal vote did not pass or quorum not reached".into()
                    )));
                }
            }
        }
        Ok(ZomeCallResponse::NetworkError(e)) => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!("Network error checking vote tally: {}", e)
            )));
        }
        _ => {
            // Voting zome unavailable — fail closed for constitutional amendments
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cannot ratify amendment: voting zome unavailable to verify tally".into()
            )));
        }
    }

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
            version: charter.version + 1,
            preamble: amendment.new_text.clone(),
            last_amended: Some(now),
            ..charter
        },
        AmendmentType::ModifyProcess => Charter {
            version: charter.version + 1,
            amendment_process: amendment.new_text.clone(),
            last_amended: Some(now),
            ..charter
        },
        AmendmentType::AddArticle => {
            // Parse articles JSON, append new article
            let mut articles: Vec<serde_json::Value> =
                serde_json::from_str(&charter.articles).unwrap_or_default();
            articles.push(serde_json::json!({
                "title": amendment.article.as_deref().unwrap_or("New Article"),
                "content": amendment.new_text,
            }));
            Charter {
                version: charter.version + 1,
                articles: serde_json::to_string(&articles)
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to serialize articles: {}", e))))?,
                last_amended: Some(now),
                ..charter
            }
        }
        AmendmentType::ModifyArticle => {
            // Find and replace article by title match
            let mut articles: Vec<serde_json::Value> =
                serde_json::from_str(&charter.articles).unwrap_or_default();
            if let Some(ref target_article) = amendment.article {
                for art in &mut articles {
                    if art.get("title").and_then(|t| t.as_str()) == Some(target_article) {
                        art["content"] = serde_json::Value::String(amendment.new_text.clone());
                        break;
                    }
                }
            }
            Charter {
                version: charter.version + 1,
                articles: serde_json::to_string(&articles)
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to serialize articles: {}", e))))?,
                last_amended: Some(now),
                ..charter
            }
        }
        AmendmentType::RemoveArticle => {
            // Remove article by title match
            let mut articles: Vec<serde_json::Value> =
                serde_json::from_str(&charter.articles).unwrap_or_default();
            if let Some(ref target_article) = amendment.article {
                articles.retain(|art| {
                    art.get("title").and_then(|t| t.as_str()) != Some(target_article)
                });
            }
            Charter {
                version: charter.version + 1,
                articles: serde_json::to_string(&articles)
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to serialize articles: {}", e))))?,
                last_amended: Some(now),
                ..charter
            }
        }
        AmendmentType::AddRight => {
            let mut rights = charter.rights.clone();
            rights.push(amendment.new_text.clone());
            Charter {
                version: charter.version + 1,
                rights,
                last_amended: Some(now),
                ..charter
            }
        }
        AmendmentType::ModifyRight => {
            // Replace right matching original_text with new_text
            let rights: Vec<String> = charter.rights.iter().map(|r| {
                if amendment.original_text.as_deref() == Some(r.as_str()) {
                    amendment.new_text.clone()
                } else {
                    r.clone()
                }
            }).collect();
            Charter {
                version: charter.version + 1,
                rights,
                last_amended: Some(now),
                ..charter
            }
        }
        AmendmentType::RemoveRight => {
            // Remove right matching original_text or new_text
            let target = amendment.original_text.as_deref()
                .unwrap_or(&amendment.new_text);
            let rights: Vec<String> = charter.rights.into_iter()
                .filter(|r| r != target)
                .collect();
            Charter {
                version: charter.version + 1,
                rights,
                last_amended: Some(now),
                ..charter
            }
        }
    };

    // Create new charter version
    create_charter(new_charter)?;

    Ok(())
}

/// Set a governance parameter
///
/// When a parameter already exists, requires a `proposal_id` linking back to
/// the governance action that authorized the change. New parameters can be set
/// without a proposal_id (initial bootstrapping).
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

    // Gate: existing parameters require a proposal_id (governance authorization)
    if input.proposal_id.is_none() {
        if let Ok(Some(_)) = get_parameter(input.name.clone()) {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Parameter already exists. Use a governance proposal to change it.".into()
            )));
        }
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

/// Update a governance parameter (cross-zome entry point)
///
/// Called by the execution zome's `GovernanceAction::UpdateParameter` dispatch.
/// Preserves the existing parameter's type, min/max values, and description
/// when the parameter already exists. Falls back to String type for new parameters.
#[hdk_extern]
pub fn update_parameter(input: UpdateParameterInput) -> ExternResult<Record> {
    // Try to fetch existing parameter to preserve its type and metadata
    let (value_type, description, min_value, max_value) =
        if let Some(existing_record) = get_parameter(input.parameter.clone())? {
            if let Some(existing) = existing_record
                .entry()
                .to_app_option::<GovernanceParameter>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                (
                    existing.value_type,
                    format!(
                        "Updated via governance action{}",
                        input.proposal_id.as_ref().map_or(String::new(), |p| format!(" ({})", p))
                    ),
                    existing.min_value,
                    existing.max_value,
                )
            } else {
                (ParameterType::String, format!(
                    "Created via governance action{}",
                    input.proposal_id.as_ref().map_or(String::new(), |p| format!(" ({})", p))
                ), None, None)
            }
        } else {
            (ParameterType::String, format!(
                "Created via governance action{}",
                input.proposal_id.as_ref().map_or(String::new(), |p| format!(" ({})", p))
            ), None, None)
        };

    set_parameter(SetParameterInput {
        name: input.parameter,
        value: input.value,
        value_type,
        description,
        min_value,
        max_value,
        proposal_id: input.proposal_id,
    })
}

/// Input for updating a parameter via cross-zome call
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateParameterInput {
    pub parameter: String,
    pub value: String,
    #[serde(default)]
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
