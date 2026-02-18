use super::*;

// =============================================================================
// CONFIGURABLE PHI PARAMETERS
// =============================================================================

const PHI_CONFIG_ANCHOR: &str = "phi_config";

/// Get the current GovernancePhiConfig from DHT, or return defaults.
///
/// This is the single source of truth for all Phi thresholds at runtime.
/// The hardcoded values in GovernanceActionType and AdaptiveThreshold serve
/// as fallback defaults when no config has been bootstrapped.
pub fn get_current_phi_config() -> ExternResult<GovernancePhiConfig> {
    let anchor = match anchor_hash(PHI_CONFIG_ANCHOR) {
        Ok(h) => h,
        Err(_) => return Ok(GovernancePhiConfig::defaults(sys_time()?)),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::PhiConfigIndex)?,
        GetStrategy::default(),
    )?;

    // Get the most recent config
    if let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid config link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(config) = record
                .entry()
                .to_app_option::<GovernancePhiConfig>()
                .ok()
                .flatten()
            {
                return Ok(config);
            }
        }
    }

    // No config found — return hardcoded defaults
    Ok(GovernancePhiConfig::defaults(sys_time()?))
}

/// Get the dynamic Phi threshold for an action type.
///
/// Reads from GovernancePhiConfig if available, falls back to hardcoded.
pub fn get_dynamic_phi_threshold(action_type: &GovernanceActionType) -> ExternResult<f64> {
    let config = get_current_phi_config()?;
    Ok(config.phi_threshold_for(action_type))
}

/// Get the dynamic min voter Phi for a proposal type.
pub fn get_dynamic_min_voter_phi(proposal_type: &ProposalType) -> ExternResult<f64> {
    let config = get_current_phi_config()?;
    Ok(config.min_voter_phi_for(proposal_type))
}

/// Bootstrap the default Phi configuration.
///
/// Creates the initial GovernancePhiConfig with hardcoded defaults.
/// Can only be called once (subsequent calls are no-ops).
#[hdk_extern]
pub fn bootstrap_phi_config(_: ()) -> ExternResult<Record> {
    let anchor = anchor_hash(PHI_CONFIG_ANCHOR)?;

    // Check if config already exists
    let links = get_links(
        LinkQuery::try_new(anchor.clone(), LinkTypes::PhiConfigIndex)?,
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
    let config = GovernancePhiConfig::defaults(now);

    let action_hash = create_entry(&EntryTypes::GovernancePhiConfig(config))?;

    create_entry(&EntryTypes::Anchor(Anchor(PHI_CONFIG_ANCHOR.to_string())))?;
    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::PhiConfigIndex,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Config not found after creation".into())))
}

/// Update Phi configuration via governance proposal.
///
/// Requires a `proposal_id` linking back to the governance action that
/// authorized the change. All values are validated for range and monotonicity.
#[hdk_extern]
pub fn update_phi_config(input: UpdatePhiConfigInput) -> ExternResult<Record> {
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "proposal_id is required and must be 1-256 characters".into()
        )));
    }

    let now = sys_time()?;
    let mut config = get_current_phi_config()?;

    // Apply only the fields that are provided
    if let Some(v) = input.phi_basic { config.phi_basic = v; }
    if let Some(v) = input.phi_proposal_submission { config.phi_proposal_submission = v; }
    if let Some(v) = input.phi_voting { config.phi_voting = v; }
    if let Some(v) = input.phi_constitutional { config.phi_constitutional = v; }
    if let Some(v) = input.min_voter_phi_standard { config.min_voter_phi_standard = v; }
    if let Some(v) = input.min_voter_phi_emergency { config.min_voter_phi_emergency = v; }
    if let Some(v) = input.min_voter_phi_constitutional { config.min_voter_phi_constitutional = v; }

    config.updated_at = now;
    config.changed_by_proposal = Some(input.proposal_id);

    // Validate (range + monotonicity) — this is also checked by integrity validation
    check_phi_config(&config)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

    let action_hash = create_entry(&EntryTypes::GovernancePhiConfig(config))?;

    let anchor = anchor_hash(PHI_CONFIG_ANCHOR)?;
    create_entry(&EntryTypes::Anchor(Anchor(PHI_CONFIG_ANCHOR.to_string())))?;
    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::PhiConfigIndex,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Config not found after update".into())))
}

/// Get the current Phi configuration
#[hdk_extern]
pub fn get_phi_config(_: ()) -> ExternResult<GovernancePhiConfig> {
    get_current_phi_config()
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdatePhiConfigInput {
    /// Governance proposal that authorized this change (required)
    pub proposal_id: String,
    pub phi_basic: Option<f64>,
    pub phi_proposal_submission: Option<f64>,
    pub phi_voting: Option<f64>,
    pub phi_constitutional: Option<f64>,
    pub min_voter_phi_standard: Option<f64>,
    pub min_voter_phi_emergency: Option<f64>,
    pub min_voter_phi_constitutional: Option<f64>,
}
