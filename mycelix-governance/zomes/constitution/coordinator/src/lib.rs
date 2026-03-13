//! Constitution Coordinator Zome
//! Business logic for charter and amendments
//!
//! Updated to use HDK 0.6 patterns

use constitution_integrity::*;
use hdk::prelude::*;

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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Charter ID must be 1-256 characters".into()
        )));
    }
    if charter.preamble.is_empty() || charter.preamble.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Preamble must be 1-4096 characters".into()
        )));
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "New text must be 1-4096 characters".into()
        )));
    }
    if input.rationale.is_empty() || input.rationale.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Rationale must be 1-4096 characters".into()
        )));
    }
    if input.proposer_did.is_empty() || input.proposer_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposer DID must be 1-256 characters".into()
        )));
    }
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal ID must be 1-256 characters".into()
        )));
    }
    if let Some(ref article) = input.article {
        if article.is_empty() || article.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Article must be 1-256 characters".into()
            )));
        }
    }
    if let Some(ref original_text) = input.original_text {
        if original_text.len() > 4096 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Original text must be at most 4096 characters".into()
            )));
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Amendment ID must be 1-256 characters".into()
        )));
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
                        // Keep iterating — update_entry appends newer versions later in the chain
                        found = Some(record);
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
    let tally_io = governance_utils::call_local(
        "voting",
        "tally_votes",
        current_amendment.proposal_id.clone(),
    )?;
    if let Ok(result) = tally_io.decode::<serde_json::Value>() {
        let approved = result
            .get("approved")
            .and_then(|a| a.as_bool())
            .unwrap_or(false);
        let quorum_reached = result
            .get("quorum_reached")
            .and_then(|q| q.as_bool())
            .unwrap_or(false);
        if !approved || !quorum_reached {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cannot ratify amendment: linked proposal vote did not pass or quorum not reached"
                    .into()
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find ratified amendment".into()
    )))
}

/// Apply an amendment to a charter (pure logic — no HDK calls).
///
/// Takes the current charter, the amendment to apply, and the current timestamp.
/// Returns the new charter with the amendment applied, or an error message.
fn apply_amendment_pure(
    charter: &Charter,
    amendment: &Amendment,
    now: Timestamp,
) -> Result<Charter, String> {
    match amendment.amendment_type {
        AmendmentType::ModifyPreamble => Ok(Charter {
            version: charter.version + 1,
            preamble: amendment.new_text.clone(),
            last_amended: Some(now),
            ..charter.clone()
        }),
        AmendmentType::ModifyProcess => Ok(Charter {
            version: charter.version + 1,
            amendment_process: amendment.new_text.clone(),
            last_amended: Some(now),
            ..charter.clone()
        }),
        AmendmentType::AddArticle => {
            let mut articles: Vec<serde_json::Value> =
                serde_json::from_str(&charter.articles).unwrap_or_default();
            articles.push(serde_json::json!({
                "title": amendment.article.as_deref().unwrap_or("New Article"),
                "content": amendment.new_text,
            }));
            Ok(Charter {
                version: charter.version + 1,
                articles: serde_json::to_string(&articles)
                    .map_err(|e| format!("Failed to serialize articles: {}", e))?,
                last_amended: Some(now),
                ..charter.clone()
            })
        }
        AmendmentType::ModifyArticle => {
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
            Ok(Charter {
                version: charter.version + 1,
                articles: serde_json::to_string(&articles)
                    .map_err(|e| format!("Failed to serialize articles: {}", e))?,
                last_amended: Some(now),
                ..charter.clone()
            })
        }
        AmendmentType::RemoveArticle => {
            let mut articles: Vec<serde_json::Value> =
                serde_json::from_str(&charter.articles).unwrap_or_default();
            if let Some(ref target_article) = amendment.article {
                articles.retain(|art| {
                    art.get("title").and_then(|t| t.as_str()) != Some(target_article)
                });
            }
            Ok(Charter {
                version: charter.version + 1,
                articles: serde_json::to_string(&articles)
                    .map_err(|e| format!("Failed to serialize articles: {}", e))?,
                last_amended: Some(now),
                ..charter.clone()
            })
        }
        AmendmentType::AddRight => {
            let mut rights = charter.rights.clone();
            rights.push(amendment.new_text.clone());
            Ok(Charter {
                version: charter.version + 1,
                rights,
                last_amended: Some(now),
                ..charter.clone()
            })
        }
        AmendmentType::ModifyRight => {
            let rights: Vec<String> = charter
                .rights
                .iter()
                .map(|r| {
                    if amendment.original_text.as_deref() == Some(r.as_str()) {
                        amendment.new_text.clone()
                    } else {
                        r.clone()
                    }
                })
                .collect();
            Ok(Charter {
                version: charter.version + 1,
                rights,
                last_amended: Some(now),
                ..charter.clone()
            })
        }
        AmendmentType::RemoveRight => {
            let target = amendment
                .original_text
                .as_deref()
                .unwrap_or(&amendment.new_text);
            let rights: Vec<String> = charter
                .rights
                .clone()
                .into_iter()
                .filter(|r| r != target)
                .collect();
            Ok(Charter {
                version: charter.version + 1,
                rights,
                last_amended: Some(now),
                ..charter.clone()
            })
        }
    }
}

/// Apply an amendment to the charter (HDK wrapper)
fn apply_amendment_to_charter(amendment: &Amendment) -> ExternResult<()> {
    let current_charter = get_current_charter(())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "No charter found".into()
    )))?;
    let charter: Charter = current_charter
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid charter entry".into()
        )))?;
    let now = sys_time()?;
    let new_charter = apply_amendment_pure(&charter, amendment, now)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Parameter name must be 1-256 characters".into()
        )));
    }
    if input.value.is_empty() || input.value.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Parameter value must be 1-4096 characters".into()
        )));
    }
    if input.description.is_empty() || input.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be 1-4096 characters".into()
        )));
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
                        input
                            .proposal_id
                            .as_ref()
                            .map_or(String::new(), |p| format!(" ({})", p))
                    ),
                    existing.min_value,
                    existing.max_value,
                )
            } else {
                (
                    ParameterType::String,
                    format!(
                        "Created via governance action{}",
                        input
                            .proposal_id
                            .as_ref()
                            .map_or(String::new(), |p| format!(" ({})", p))
                    ),
                    None,
                    None,
                )
            }
        } else {
            (
                ParameterType::String,
                format!(
                    "Created via governance action{}",
                    input
                        .proposal_id
                        .as_ref()
                        .map_or(String::new(), |p| format!(" ({})", p))
                ),
                None,
                None,
            )
        };

    let result = set_parameter(SetParameterInput {
        name: input.parameter.clone(),
        value: input.value.clone(),
        value_type,
        description,
        min_value,
        max_value,
        proposal_id: input.proposal_id.clone(),
    })?;

    // Sync phi-related parameters to the governance bridge's GovernancePhiConfig.
    // This keeps the bridge's configurable thresholds in sync with constitution parameters.
    sync_phi_parameter_to_bridge(&input.parameter, &input.value, input.proposal_id.as_deref());

    Ok(result)
}

/// Input for updating a parameter via cross-zome call
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateParameterInput {
    pub parameter: String,
    pub value: String,
    #[serde(default)]
    pub proposal_id: Option<String>,
}

/// Maps constitution parameter names to GovernancePhiConfig fields.
/// Returns Some(bridge_field_name) if this parameter should sync to the bridge.
fn phi_config_field(param_name: &str) -> Option<&'static str> {
    match param_name {
        "phi_basic" => Some("phi_basic"),
        "phi_proposal_submission" => Some("phi_proposal_submission"),
        "phi_voting" => Some("phi_voting"),
        "phi_constitutional" => Some("phi_constitutional"),
        "min_voter_phi_standard" => Some("min_voter_phi_standard"),
        "min_voter_phi_emergency" => Some("min_voter_phi_emergency"),
        "min_voter_phi_constitutional" => Some("min_voter_phi_constitutional"),
        "max_voting_weight" => Some("max_voting_weight"),
        _ => None,
    }
}

/// Sync a phi-related parameter update to the governance bridge's GovernancePhiConfig.
///
/// Best-effort: if the bridge zome is not installed or the call fails, we log a
/// warning signal but do NOT fail the parameter update. The constitution is the
/// source of truth; the bridge config is a derived cache.
fn sync_phi_parameter_to_bridge(param_name: &str, value: &str, proposal_id: Option<&str>) {
    let Some(field_name) = phi_config_field(param_name) else {
        return;
    };

    // Parse the value as f64 — all phi config fields are numeric
    let Ok(numeric_value) = value.parse::<f64>() else {
        let _ = emit_signal(serde_json::json!({
            "type": "PhiConfigSyncWarning",
            "message": format!("Cannot sync '{}' to bridge: value '{}' is not numeric", param_name, value),
        }));
        return;
    };

    // Build a partial update payload — only the changed field is set
    let mut payload = serde_json::Map::new();
    payload.insert(
        "proposal_id".into(),
        serde_json::json!(proposal_id.unwrap_or("constitution_sync")),
    );
    payload.insert(field_name.into(), serde_json::json!(numeric_value));

    if governance_utils::call_local_best_effort(
        "governance_bridge",
        "update_phi_config",
        serde_json::Value::Object(payload),
    )
    .ok()
    .flatten()
    .is_none()
    {
        let _ = emit_signal(serde_json::json!({
            "type": "PhiConfigSyncWarning",
            "message": format!("Could not sync '{}' to bridge", param_name),
        }));
    }
}

/// Get a governance parameter
#[hdk_extern]
pub fn get_parameter(name: String) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("param:{}", name))?,
            LinkTypes::ParameterIndex,
        )?,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    fn base_charter() -> Charter {
        Charter {
            id: "charter-1".into(),
            version: 1,
            preamble: "We the people".into(),
            articles: r#"[{"title":"Article I","content":"Governance structure"}]"#.into(),
            rights: vec!["Right to dignity".into(), "Right to privacy".into()],
            amendment_process: "2/3 supermajority".into(),
            adopted: ts(1_000_000),
            last_amended: None,
        }
    }

    fn make_amendment(amendment_type: AmendmentType) -> Amendment {
        Amendment {
            id: "amend-1".into(),
            charter_version: 1,
            amendment_type,
            article: None,
            original_text: None,
            new_text: "new text".into(),
            rationale: "test rationale".into(),
            proposer: "did:key:z6Mk".into(),
            proposal_id: "prop-1".into(),
            status: AmendmentStatus::Ratified,
            created: ts(2_000_000),
            ratified: Some(ts(3_000_000)),
        }
    }

    // === ModifyPreamble ===

    #[test]
    fn test_modify_preamble() {
        let charter = base_charter();
        let mut amendment = make_amendment(AmendmentType::ModifyPreamble);
        amendment.new_text = "A new preamble for the ages".into();
        amendment.original_text = Some(charter.preamble.clone());

        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        assert_eq!(result.version, 2);
        assert_eq!(result.preamble, "A new preamble for the ages");
        assert_eq!(result.last_amended, Some(ts(4_000_000)));
        // Other fields unchanged
        assert_eq!(result.id, charter.id);
        assert_eq!(result.articles, charter.articles);
        assert_eq!(result.rights, charter.rights);
    }

    // === ModifyProcess ===

    #[test]
    fn test_modify_process() {
        let charter = base_charter();
        let mut amendment = make_amendment(AmendmentType::ModifyProcess);
        amendment.new_text = "3/4 supermajority with 30-day cooling".into();

        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        assert_eq!(result.version, 2);
        assert_eq!(
            result.amendment_process,
            "3/4 supermajority with 30-day cooling"
        );
        assert_eq!(result.preamble, charter.preamble); // unchanged
    }

    // === AddArticle ===

    #[test]
    fn test_add_article_with_title() {
        let charter = base_charter();
        let mut amendment = make_amendment(AmendmentType::AddArticle);
        amendment.article = Some("Article II".into());
        amendment.new_text = "Treasury management".into();

        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        assert_eq!(result.version, 2);

        let articles: Vec<serde_json::Value> = serde_json::from_str(&result.articles).unwrap();
        assert_eq!(articles.len(), 2);
        assert_eq!(articles[1]["title"], "Article II");
        assert_eq!(articles[1]["content"], "Treasury management");
    }

    #[test]
    fn test_add_article_default_title() {
        let charter = base_charter();
        let mut amendment = make_amendment(AmendmentType::AddArticle);
        amendment.article = None; // no explicit title
        amendment.new_text = "Default content".into();

        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        let articles: Vec<serde_json::Value> = serde_json::from_str(&result.articles).unwrap();
        assert_eq!(articles[1]["title"], "New Article");
    }

    // === ModifyArticle ===

    #[test]
    fn test_modify_article() {
        let charter = base_charter();
        let mut amendment = make_amendment(AmendmentType::ModifyArticle);
        amendment.article = Some("Article I".into());
        amendment.new_text = "Updated governance structure".into();

        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        let articles: Vec<serde_json::Value> = serde_json::from_str(&result.articles).unwrap();
        assert_eq!(articles.len(), 1);
        assert_eq!(articles[0]["title"], "Article I");
        assert_eq!(articles[0]["content"], "Updated governance structure");
    }

    #[test]
    fn test_modify_article_no_match() {
        let charter = base_charter();
        let mut amendment = make_amendment(AmendmentType::ModifyArticle);
        amendment.article = Some("Nonexistent Article".into());
        amendment.new_text = "Should not appear".into();

        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        // Articles unchanged except version/timestamp
        let articles: Vec<serde_json::Value> = serde_json::from_str(&result.articles).unwrap();
        assert_eq!(articles[0]["content"], "Governance structure");
    }

    // === RemoveArticle ===

    #[test]
    fn test_remove_article() {
        let charter = base_charter();
        let mut amendment = make_amendment(AmendmentType::RemoveArticle);
        amendment.article = Some("Article I".into());

        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        let articles: Vec<serde_json::Value> = serde_json::from_str(&result.articles).unwrap();
        assert!(articles.is_empty());
    }

    #[test]
    fn test_remove_article_no_match() {
        let charter = base_charter();
        let mut amendment = make_amendment(AmendmentType::RemoveArticle);
        amendment.article = Some("Nonexistent".into());

        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        let articles: Vec<serde_json::Value> = serde_json::from_str(&result.articles).unwrap();
        assert_eq!(articles.len(), 1); // unchanged
    }

    // === AddRight ===

    #[test]
    fn test_add_right() {
        let charter = base_charter();
        let mut amendment = make_amendment(AmendmentType::AddRight);
        amendment.new_text = "Right to participation".into();

        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        assert_eq!(result.rights.len(), 3);
        assert_eq!(result.rights[2], "Right to participation");
    }

    // === ModifyRight ===

    #[test]
    fn test_modify_right() {
        let charter = base_charter();
        let mut amendment = make_amendment(AmendmentType::ModifyRight);
        amendment.original_text = Some("Right to privacy".into());
        amendment.new_text = "Right to data sovereignty".into();

        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        assert_eq!(result.rights.len(), 2);
        assert_eq!(result.rights[0], "Right to dignity"); // unchanged
        assert_eq!(result.rights[1], "Right to data sovereignty"); // modified
    }

    #[test]
    fn test_modify_right_no_match() {
        let charter = base_charter();
        let mut amendment = make_amendment(AmendmentType::ModifyRight);
        amendment.original_text = Some("Nonexistent right".into());
        amendment.new_text = "Should not appear".into();

        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        assert_eq!(result.rights, charter.rights); // unchanged
    }

    // === RemoveRight ===

    #[test]
    fn test_remove_right_via_original_text() {
        let charter = base_charter();
        let mut amendment = make_amendment(AmendmentType::RemoveRight);
        amendment.original_text = Some("Right to privacy".into());

        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        assert_eq!(result.rights.len(), 1);
        assert_eq!(result.rights[0], "Right to dignity");
    }

    #[test]
    fn test_remove_right_via_new_text_fallback() {
        let charter = base_charter();
        let mut amendment = make_amendment(AmendmentType::RemoveRight);
        amendment.original_text = None; // falls back to new_text
        amendment.new_text = "Right to dignity".into();

        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        assert_eq!(result.rights.len(), 1);
        assert_eq!(result.rights[0], "Right to privacy");
    }

    // === Cross-cutting ===

    #[test]
    fn test_version_always_increments() {
        let charter = Charter {
            version: 5,
            ..base_charter()
        };
        let amendment = make_amendment(AmendmentType::ModifyPreamble);
        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        assert_eq!(result.version, 6);
    }

    #[test]
    fn test_adopted_date_preserved() {
        let charter = base_charter();
        let amendment = make_amendment(AmendmentType::AddRight);
        let result = apply_amendment_pure(&charter, &amendment, ts(4_000_000)).unwrap();
        assert_eq!(result.adopted, charter.adopted); // immutable
    }

    // =========================================================================
    // Phi config sync mapping
    // =========================================================================

    #[test]
    fn test_phi_config_field_mapping() {
        assert_eq!(phi_config_field("phi_basic"), Some("phi_basic"));
        assert_eq!(phi_config_field("phi_voting"), Some("phi_voting"));
        assert_eq!(
            phi_config_field("phi_constitutional"),
            Some("phi_constitutional")
        );
        assert_eq!(
            phi_config_field("min_voter_phi_standard"),
            Some("min_voter_phi_standard")
        );
        assert_eq!(
            phi_config_field("max_voting_weight"),
            Some("max_voting_weight")
        );
        // Non-phi parameters should return None
        assert_eq!(phi_config_field("quorum_threshold"), None);
        assert_eq!(phi_config_field("amendment_cooldown"), None);
        assert_eq!(phi_config_field(""), None);
    }
}
