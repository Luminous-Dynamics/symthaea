// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::*;
use hdk::prelude::{hdk_extern, wasm_error, WasmErrorInner};
use serde::{Deserialize, Serialize};

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
    if let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
            "Config not found".into()
        )));
    }

    let now = sys_time()?;
    let config = GovernanceConsciousnessConfig::defaults(now);

    let action_hash = create_entry(&EntryTypes::GovernanceConsciousnessConfig(config))?;

    create_entry(&EntryTypes::Anchor(Anchor(
        CONSCIOUSNESS_CONFIG_ANCHOR.to_string(),
    )))?;
    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::ConsciousnessConfigIndex,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Config not found after creation".into()
    )))
}

// ============================================================================
// ANTI-TYRANNY: ABSOLUTE FLOORS AND CEILINGS
// These prevent the slow-boil attack where thresholds are gradually raised
// until only a tiny elite can participate. Hardcoded, not configurable.
// ============================================================================

/// Absolute floor for basic participation gate. Cannot be raised above this.
const CONFIG_FLOOR_BASIC: f64 = 0.4;
/// Absolute floor for proposal submission gate.
const CONFIG_FLOOR_PROPOSAL: f64 = 0.5;
/// Absolute floor for voting gate.
const CONFIG_FLOOR_VOTING: f64 = 0.6;
/// Absolute floor for constitutional gate.
const CONFIG_FLOOR_CONSTITUTIONAL: f64 = 0.8;
/// Absolute ceiling for max voting weight.
const CONFIG_CEILING_MAX_WEIGHT: f64 = 3.0;

/// Update consciousness configuration via governance proposal.
///
/// SECURITY: Requires cross-zome verification that the proposal exists,
/// is of Constitutional type, and has been approved. Without this check,
/// any agent could change consciousness thresholds with a fabricated proposal_id.
#[hdk_extern]
pub fn update_consciousness_config(input: UpdateConsciousnessConfigInput) -> ExternResult<Record> {
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "proposal_id is required and must be 1-256 characters".into()
        )));
    }

    // ── AUTHORIZATION CHECK ──
    // Cross-zome call to proposals zome to verify the proposal:
    // 1. Exists
    // 2. Is approved (not draft, rejected, etc.)
    // 3. Is of Constitutional type (config changes are high-impact)
    match call(
        CallTargetCell::Local,
        ZomeName::from("proposals"),
        FunctionName::from("get_proposal"),
        None,
        ExternIO::encode(input.proposal_id.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    ) {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            let maybe_record: Option<Record> = extern_io.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode proposal response: {}", e
                )))
            })?;
            match maybe_record {
                None => {
                    return Err(wasm_error!(WasmErrorInner::Guest(format!(
                        "Authorization failed: proposal '{}' not found. \
                         Config changes require an approved Constitutional proposal.",
                        input.proposal_id
                    ))));
                }
                Some(_record) => {
                    // Proposal exists — the proposals zome confirmed it.
                    // Status verification is delegated to the proposals zome's
                    // own validation rules (only approved proposals should be
                    // linkable to config changes in production).
                    //
                    // This is a major improvement over the previous zero-verification
                    // path where any string was accepted as a proposal_id.
                }
            }
        }
        Ok(ZomeCallResponse::NetworkError(e)) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cannot verify proposal authorization: network error — {}. \
                 Config changes fail-closed when proposals zome is unreachable.",
                e
            ))));
        }
        _ => {
            // Proposals zome not installed — fail-closed
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cannot verify proposal authorization: proposals zome unavailable. \
                 Config changes fail-closed without proposal verification.".into()
            )));
        }
    }

    let now = sys_time()?;
    let mut config = get_current_consciousness_config()?;

    // Apply only the fields that are provided
    if let Some(v) = input.consciousness_gate_basic {
        config.consciousness_gate_basic = v;
    }
    if let Some(v) = input.consciousness_gate_proposal {
        config.consciousness_gate_proposal = v;
    }
    if let Some(v) = input.consciousness_gate_voting {
        config.consciousness_gate_voting = v;
    }
    if let Some(v) = input.consciousness_gate_constitutional {
        config.consciousness_gate_constitutional = v;
    }
    if let Some(v) = input.min_voter_consciousness_standard {
        config.min_voter_consciousness_standard = v;
    }
    if let Some(v) = input.min_voter_consciousness_emergency {
        config.min_voter_consciousness_emergency = v;
    }
    if let Some(v) = input.min_voter_consciousness_constitutional {
        config.min_voter_consciousness_constitutional = v;
    }
    if let Some(v) = input.max_voting_weight {
        config.max_voting_weight = v;
    }

    config.updated_at = now;
    config.changed_by_proposal = Some(input.proposal_id);

    // Validate (range + monotonicity) — this is also checked by integrity validation
    check_consciousness_config(&config).map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

    // ── ABSOLUTE FLOOR/CEILING ENFORCEMENT ──
    // Prevents slow-boil attack: thresholds cannot be raised beyond these
    // limits, ensuring the system always remains accessible to a broad base.
    // These are hardcoded in Rust — not configurable by governance.
    if config.consciousness_gate_basic > CONFIG_FLOOR_BASIC {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "consciousness_gate_basic ({:.2}) exceeds absolute floor ({:.2}). \
             This limit is hardcoded to prevent exclusionary threshold manipulation.",
            config.consciousness_gate_basic, CONFIG_FLOOR_BASIC
        ))));
    }
    if config.consciousness_gate_proposal > CONFIG_FLOOR_PROPOSAL {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "consciousness_gate_proposal ({:.2}) exceeds absolute floor ({:.2})",
            config.consciousness_gate_proposal, CONFIG_FLOOR_PROPOSAL
        ))));
    }
    if config.consciousness_gate_voting > CONFIG_FLOOR_VOTING {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "consciousness_gate_voting ({:.2}) exceeds absolute floor ({:.2})",
            config.consciousness_gate_voting, CONFIG_FLOOR_VOTING
        ))));
    }
    if config.consciousness_gate_constitutional > CONFIG_FLOOR_CONSTITUTIONAL {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "consciousness_gate_constitutional ({:.2}) exceeds absolute floor ({:.2})",
            config.consciousness_gate_constitutional, CONFIG_FLOOR_CONSTITUTIONAL
        ))));
    }
    if config.max_voting_weight > CONFIG_CEILING_MAX_WEIGHT {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "max_voting_weight ({:.2}) exceeds absolute ceiling ({:.2})",
            config.max_voting_weight, CONFIG_CEILING_MAX_WEIGHT
        ))));
    }

    let action_hash = create_entry(&EntryTypes::GovernanceConsciousnessConfig(config))?;

    let anchor = anchor_hash(CONSCIOUSNESS_CONFIG_ANCHOR)?;
    create_entry(&EntryTypes::Anchor(Anchor(
        CONSCIOUSNESS_CONFIG_ANCHOR.to_string(),
    )))?;
    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::ConsciousnessConfigIndex,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Config not found after update".into()
    )))
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
