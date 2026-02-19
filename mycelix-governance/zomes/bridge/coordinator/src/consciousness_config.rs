use super::*;

// =============================================================================
// CONFIGURABLE CONSCIOUSNESS PARAMETERS
// =============================================================================

const CONSCIOUSNESS_CONFIG_ANCHOR: &str = "consciousness_config";

/// Get the current GovernanceConsciousnessConfig from DHT, or return defaults.
///
/// This is the single source of truth for all consciousness gate thresholds at runtime.
/// The hardcoded values in GovernanceActionType and AdaptiveThreshold serve
/// as fallback defaults when no config has been bootstrapped.
pub fn get_current_consciousness_config() -> ExternResult<GovernanceConsciousnessConfig> {
    let anchor = match anchor_hash(CONSCIOUSNESS_CONFIG_ANCHOR) {
        Ok(h) => h,
        Err(_) => return Ok(GovernanceConsciousnessConfig::defaults(sys_time()?)),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ConsciousnessConfigIndex)?,
        GetStrategy::default(),
    )?;

    // Get the most recent config
    if let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid config link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(config) = record
                .entry()
                .to_app_option::<GovernanceConsciousnessConfig>()
                .ok()
                .flatten()
            {
                return Ok(config);
            }
        }
    }

    // No config found — return hardcoded defaults
    Ok(GovernanceConsciousnessConfig::defaults(sys_time()?))
}

/// Get the dynamic consciousness gate threshold for an action type.
///
/// Reads from GovernanceConsciousnessConfig if available, falls back to hardcoded.
pub fn get_dynamic_consciousness_gate(action_type: &GovernanceActionType) -> ExternResult<f64> {
    let config = get_current_consciousness_config()?;
    Ok(config.consciousness_gate_for(action_type))
}

/// Get the dynamic min voter consciousness for a proposal type.
pub fn get_dynamic_min_voter_consciousness(proposal_type: &ProposalType) -> ExternResult<f64> {
    let config = get_current_consciousness_config()?;
    Ok(config.min_voter_consciousness_for(proposal_type))
}

/// Bootstrap the default consciousness configuration.
///
/// Creates the initial GovernanceConsciousnessConfig with hardcoded defaults.
/// Can only be called once (subsequent calls are no-ops).
#[hdk_extern]
pub fn bootstrap_consciousness_config(_: ()) -> ExternResult<Record> {
    let anchor = anchor_hash(CONSCIOUSNESS_CONFIG_ANCHOR)?;

    // Check if config already exists
    let links = get_links(
        LinkQuery::try_new(anchor.clone(), LinkTypes::ConsciousnessConfigIndex)?,
        GetStrategy::default(),
    )?;
    if !links.is_empty() {
        // Config already exists — return the latest
        let link = links.into_iter().max_by_key(|l| l.timestamp).unwrap();
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default())?
            .ok_or(wasm_error!(WasmErrorInner::Guest("Config not found".into())));
    }

    let now = sys_time()?;
    let config = GovernanceConsciousnessConfig::defaults(now);

    let action_hash = create_entry(&EntryTypes::GovernanceConsciousnessConfig(config))?;

    create_entry(&EntryTypes::Anchor(Anchor(CONSCIOUSNESS_CONFIG_ANCHOR.to_string())))?;
    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::ConsciousnessConfigIndex,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Config not found after creation".into())))
}

/// Update consciousness configuration via governance proposal.
///
/// Requires a `proposal_id` linking back to the governance action that
/// authorized the change. All values are validated for range and monotonicity.
#[hdk_extern]
pub fn update_consciousness_config(input: UpdateConsciousnessConfigInput) -> ExternResult<Record> {
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "proposal_id is required and must be 1-256 characters".into()
        )));
    }

    let now = sys_time()?;
    let mut config = get_current_consciousness_config()?;

    // Apply only the fields that are provided
    if let Some(v) = input.consciousness_gate_basic { config.consciousness_gate_basic = v; }
    if let Some(v) = input.consciousness_gate_proposal { config.consciousness_gate_proposal = v; }
    if let Some(v) = input.consciousness_gate_voting { config.consciousness_gate_voting = v; }
    if let Some(v) = input.consciousness_gate_constitutional { config.consciousness_gate_constitutional = v; }
    if let Some(v) = input.min_voter_consciousness_standard { config.min_voter_consciousness_standard = v; }
    if let Some(v) = input.min_voter_consciousness_emergency { config.min_voter_consciousness_emergency = v; }
    if let Some(v) = input.min_voter_consciousness_constitutional { config.min_voter_consciousness_constitutional = v; }
    if let Some(v) = input.max_voting_weight { config.max_voting_weight = v; }

    config.updated_at = now;
    config.changed_by_proposal = Some(input.proposal_id);

    // Validate (range + monotonicity) — this is also checked by integrity validation
    check_consciousness_config(&config)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

    let action_hash = create_entry(&EntryTypes::GovernanceConsciousnessConfig(config))?;

    let anchor = anchor_hash(CONSCIOUSNESS_CONFIG_ANCHOR)?;
    create_entry(&EntryTypes::Anchor(Anchor(CONSCIOUSNESS_CONFIG_ANCHOR.to_string())))?;
    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::ConsciousnessConfigIndex,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Config not found after update".into())))
}

/// Get the current consciousness configuration
#[hdk_extern]
pub fn get_consciousness_config(_: ()) -> ExternResult<GovernanceConsciousnessConfig> {
    get_current_consciousness_config()
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateConsciousnessConfigInput {
    /// Governance proposal that authorized this change (required)
    pub proposal_id: String,
    pub consciousness_gate_basic: Option<f64>,
    pub consciousness_gate_proposal: Option<f64>,
    pub consciousness_gate_voting: Option<f64>,
    pub consciousness_gate_constitutional: Option<f64>,
    pub min_voter_consciousness_standard: Option<f64>,
    pub min_voter_consciousness_emergency: Option<f64>,
    pub min_voter_consciousness_constitutional: Option<f64>,
    pub max_voting_weight: Option<f64>,
}
