// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Agents Coordinator Zome
//!
//! Manages agent registration and model updates for federated learning.
//! Includes security controls: input validation, authorization, rate limiting.

use hdk::prelude::*;

// =============================================================================
// SECURITY: Constants and Configuration
// =============================================================================

/// Maximum length for agent IDs (prevents DoS via large IDs)
const MAX_AGENT_ID_LENGTH: usize = 256;

/// Maximum capabilities per agent
const MAX_CAPABILITIES: usize = 50;

/// Maximum weight vector size (10 million elements)
const MAX_WEIGHTS_SIZE: usize = 10_000_000;

/// Maximum registrations per minute per caller
const MAX_REGISTRATIONS_PER_MINUTE: u32 = 5;

/// Maximum updates per minute per agent
const MAX_UPDATES_PER_MINUTE: u32 = 60;

/// Minimum reputation score
const MIN_REPUTATION: f64 = 0.0;

/// Maximum reputation score
const MAX_REPUTATION: f64 = 1.0;

// =============================================================================
// Entry Types
// =============================================================================

/// Agent registration entry
#[hdk_entry_helper]
#[derive(Clone)]
pub struct AgentRegistration {
    pub agent_id: String,
    pub capabilities: Vec<String>,
    pub reputation_score: f64,
    pub registered_at: i64,
    pub registrant: String,
    pub active: bool,
}

/// Model update entry for federated learning
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ModelUpdate {
    pub agent_id: String,
    pub round_id: u32,
    pub weights_hash: String,  // Store hash, not full weights
    pub weights_size: usize,
    pub timestamp: i64,
    pub submitter: String,
}

/// Rate limit tracking (stored in memory, not DHT)
#[derive(Debug, Clone)]
struct RateLimitEntry {
    count: u32,
    window_start: i64,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    AgentRegistration(AgentRegistration),
    ModelUpdate(ModelUpdate),
}

// =============================================================================
// Link Types
// =============================================================================

#[hdk_link_types]
pub enum LinkTypes {
    AgentLink,
    ModelUpdateLink,
    AgentToUpdates,
}

// =============================================================================
// Input Validation
// =============================================================================

/// Validate agent ID format and length
fn validate_agent_id(agent_id: &str) -> ExternResult<()> {
    let trimmed = agent_id.trim();

    if trimmed.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Agent ID cannot be empty".to_string()
        )));
    }

    if trimmed.len() > MAX_AGENT_ID_LENGTH {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Agent ID exceeds maximum length of {} characters", MAX_AGENT_ID_LENGTH)
        )));
    }

    // Only allow alphanumeric, hyphens, underscores, colons (for DID format)
    if !trimmed.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == ':') {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Agent ID contains invalid characters. Only alphanumeric, hyphens, underscores, and colons allowed.".to_string()
        )));
    }

    Ok(())
}

/// Validate capabilities list
fn validate_capabilities(capabilities: &[String]) -> ExternResult<()> {
    if capabilities.len() > MAX_CAPABILITIES {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Too many capabilities: {} (max {})", capabilities.len(), MAX_CAPABILITIES)
        )));
    }

    for cap in capabilities {
        if cap.trim().is_empty() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Capability cannot be empty".to_string()
            )));
        }
        if cap.len() > 100 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Capability name too long (max 100 characters)".to_string()
            )));
        }
    }

    Ok(())
}

/// Validate reputation score
fn validate_reputation(score: f64) -> ExternResult<()> {
    if !score.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Reputation score must be a finite number".to_string()
        )));
    }

    if score < MIN_REPUTATION || score > MAX_REPUTATION {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Reputation score must be between {} and {}", MIN_REPUTATION, MAX_REPUTATION)
        )));
    }

    Ok(())
}

/// Validate weights for model update
fn validate_weights(weights: &[f32]) -> ExternResult<()> {
    if weights.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Weights cannot be empty".to_string()
        )));
    }

    if weights.len() > MAX_WEIGHTS_SIZE {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Weights too large: {} (max {})", weights.len(), MAX_WEIGHTS_SIZE)
        )));
    }

    // Check for NaN/Infinity
    for (i, &val) in weights.iter().enumerate() {
        if !val.is_finite() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!("Weight at index {} is not finite: {}", i, val)
            )));
        }
    }

    Ok(())
}

// =============================================================================
// Rate Limiting
// =============================================================================

/// Check rate limit for an action
fn check_rate_limit(action: &str, max_per_minute: u32) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey.to_string();
    let now = sys_time()?.0 as i64 / 1_000_000;
    let window_start = now - 60;

    // Use path-based rate limiting
    let rate_path = Path::from(format!("rate_limit.{}.{}", action, &agent[..16.min(agent.len())]));
    let rate_hash = match rate_path.clone().typed(LinkTypes::AgentLink) {
        Ok(typed) => typed.path_entry_hash()?,
        Err(_) => return Ok(()), // No rate limit path yet
    };

    let links = get_links(
        GetLinksInputBuilder::try_new(rate_hash, LinkTypes::AgentLink)?.build()
    )?;

    let recent_count = links.iter()
        .filter(|l| l.timestamp.0 as i64 / 1_000_000 > window_start)
        .count() as u32;

    if recent_count >= max_per_minute {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Rate limit exceeded: {} per minute maximum for {}", max_per_minute, action)
        )));
    }

    Ok(())
}

/// Record rate limit action
fn record_rate_limit_action(action: &str) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey.to_string();

    let rate_path = Path::from(format!("rate_limit.{}.{}", action, &agent[..16.min(agent.len())]));
    let typed = rate_path.typed(LinkTypes::AgentLink)?;
    typed.ensure()?;
    let rate_hash = typed.path_entry_hash()?;

    // Create a self-link as a rate limit marker
    create_link(
        rate_hash.clone(),
        rate_hash,
        LinkTypes::AgentLink,
        vec![],
    )?;

    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute SHA256 hash of weights
fn hash_weights(weights: &[f32]) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    for w in weights {
        hasher.update(w.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Create anchor hash
fn anchor_hash(type_name: &str, text: &str) -> ExternResult<EntryHash> {
    let path = Path::from(format!("{}.{}", type_name, text));
    let typed = path.typed(LinkTypes::AgentLink)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

// =============================================================================
// Public API
// =============================================================================

/// Input for agent registration
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RegisterAgentInput {
    pub agent_id: String,
    pub capabilities: Vec<String>,
    pub initial_reputation: Option<f64>,
}

/// Register a new agent
///
/// # Security
/// - Validates agent_id format and length
/// - Validates capabilities list
/// - Rate limited to prevent spam
/// - Records registrant for accountability
#[hdk_extern]
pub fn register_agent(input: RegisterAgentInput) -> ExternResult<ActionHash> {
    // Input validation
    validate_agent_id(&input.agent_id)?;
    validate_capabilities(&input.capabilities)?;

    let reputation = input.initial_reputation.unwrap_or(0.5);
    validate_reputation(reputation)?;

    // Rate limiting
    check_rate_limit("register", MAX_REGISTRATIONS_PER_MINUTE)?;

    let agent_info = agent_info()?;
    let now = sys_time()?.0 as i64 / 1_000_000;

    let registration = AgentRegistration {
        agent_id: input.agent_id.clone(),
        capabilities: input.capabilities,
        reputation_score: reputation,
        registered_at: now,
        registrant: agent_info.agent_initial_pubkey.to_string(),
        active: true,
    };

    let action_hash = create_entry(EntryTypes::AgentRegistration(registration))?;

    // Link from agents anchor
    let anchor = anchor_hash("agents", "all")?;
    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::AgentLink,
        input.agent_id.as_bytes().to_vec(),
    )?;

    // Record rate limit action
    record_rate_limit_action("register")?;

    Ok(action_hash)
}

/// Get all registered agents
#[hdk_extern]
pub fn get_all_agents(_: ()) -> ExternResult<Vec<AgentRegistration>> {
    let anchor = anchor_hash("agents", "all")?;
    let links = get_links(
        GetLinksInputBuilder::try_new(anchor, LinkTypes::AgentLink)?.build()
    )?;

    let mut agents = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(reg) = AgentRegistration::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if reg.active {
                            agents.push(reg);
                        }
                    }
                }
            }
        }
    }

    Ok(agents)
}

/// Get agent by ID
#[hdk_extern]
pub fn get_agent(agent_id: String) -> ExternResult<Option<AgentRegistration>> {
    validate_agent_id(&agent_id)?;

    let agents = get_all_agents(())?;
    Ok(agents.into_iter().find(|a| a.agent_id == agent_id))
}

/// Input for model update submission
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubmitUpdateInput {
    pub agent_id: String,
    pub round_id: u32,
    pub weights: Vec<f32>,
}

/// Submit a model update
///
/// # Security
/// - Validates agent exists and is active
/// - Validates weights (size, finite values)
/// - Rate limited per agent
/// - Stores hash instead of full weights
#[hdk_extern]
pub fn submit_model_update(input: SubmitUpdateInput) -> ExternResult<ActionHash> {
    // Validate input
    validate_agent_id(&input.agent_id)?;
    validate_weights(&input.weights)?;

    // Verify agent exists and is active
    let agent = get_agent(input.agent_id.clone())?;
    if agent.is_none() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Agent not found: {}", input.agent_id)
        )));
    }
    let agent = agent.expect("agent existence verified by is_none check above");
    if !agent.active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Agent is not active: {}", input.agent_id)
        )));
    }

    // Rate limiting
    check_rate_limit(&format!("update.{}", input.agent_id), MAX_UPDATES_PER_MINUTE)?;

    let agent_info = agent_info()?;
    let now = sys_time()?.0 as i64 / 1_000_000;

    // Store hash instead of full weights (privacy + storage efficiency)
    let weights_hash = hash_weights(&input.weights);

    let update = ModelUpdate {
        agent_id: input.agent_id.clone(),
        round_id: input.round_id,
        weights_hash,
        weights_size: input.weights.len(),
        timestamp: now,
        submitter: agent_info.agent_initial_pubkey.to_string(),
    };

    let action_hash = create_entry(EntryTypes::ModelUpdate(update))?;

    // Link from round anchor
    let round_anchor = anchor_hash("training_rounds", &input.round_id.to_string())?;
    create_link(
        round_anchor,
        action_hash.clone(),
        LinkTypes::ModelUpdateLink,
        input.agent_id.as_bytes().to_vec(),
    )?;

    // Link from agent to their updates
    let agent_anchor = anchor_hash("agent_updates", &input.agent_id)?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToUpdates,
        input.round_id.to_le_bytes().to_vec(),
    )?;

    // Record rate limit action
    record_rate_limit_action(&format!("update.{}", input.agent_id))?;

    Ok(action_hash)
}

/// Get all updates for a training round
#[hdk_extern]
pub fn get_round_updates(round_id: u32) -> ExternResult<Vec<ModelUpdate>> {
    let round_anchor = anchor_hash("training_rounds", &round_id.to_string())?;
    let links = get_links(
        GetLinksInputBuilder::try_new(round_anchor, LinkTypes::ModelUpdateLink)?.build()
    )?;

    let mut updates = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(update) = ModelUpdate::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        updates.push(update);
                    }
                }
            }
        }
    }

    Ok(updates)
}

/// Get all updates for a specific agent
#[hdk_extern]
pub fn get_agent_updates(agent_id: String) -> ExternResult<Vec<ModelUpdate>> {
    validate_agent_id(&agent_id)?;

    let agent_anchor = anchor_hash("agent_updates", &agent_id)?;
    let links = get_links(
        GetLinksInputBuilder::try_new(agent_anchor, LinkTypes::AgentToUpdates)?.build()
    )?;

    let mut updates = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(update) = ModelUpdate::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        updates.push(update);
                    }
                }
            }
        }
    }

    // Sort by round_id
    updates.sort_by_key(|u| u.round_id);

    Ok(updates)
}

/// Deactivate an agent (only registrant can do this)
#[hdk_extern]
pub fn deactivate_agent(agent_id: String) -> ExternResult<bool> {
    validate_agent_id(&agent_id)?;

    let caller = agent_info()?.agent_initial_pubkey.to_string();

    // Find the agent registration
    let anchor = anchor_hash("agents", "all")?;
    let links = get_links(
        GetLinksInputBuilder::try_new(anchor, LinkTypes::AgentLink)?.build()
    )?;

    for link in links {
        let tag_str = String::from_utf8_lossy(link.tag.as_ref());
        if tag_str == agent_id {
            if let Some(action_hash) = link.target.into_action_hash() {
                if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                    if let Some(Entry::App(bytes)) = record.entry().as_option() {
                        if let Ok(mut reg) = AgentRegistration::try_from(
                            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                        ) {
                            // Only registrant can deactivate
                            if reg.registrant != caller {
                                return Err(wasm_error!(WasmErrorInner::Guest(
                                    "Only the registrant can deactivate an agent".to_string()
                                )));
                            }

                            reg.active = false;
                            update_entry(action_hash, EntryTypes::AgentRegistration(reg))?;
                            return Ok(true);
                        }
                    }
                }
            }
        }
    }

    Ok(false)
}
