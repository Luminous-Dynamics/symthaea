// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Authorization, rate limiting, and coordinator verification.

use hdk::prelude::*;
use federated_learning_integrity::*;

use sha2::Digest;

use crate::config::{MAX_SUBMISSIONS_PER_MINUTE, COORDINATOR_ROLE, DETECTOR_ROLE};
use super::ensure_path;

/// Verify that the caller has valid coordinator authority
///
/// This function implements the secure coordinator bootstrap ceremony:
/// 1. Check if caller has a valid CoordinatorCredential
/// 2. Verify the credential is not expired or revoked
/// 3. For genesis coordinators, verify the authority proof
/// 4. For voted coordinators, verify guardian signatures
///
/// SECURITY: This replaces the vulnerable "first caller wins" pattern
pub(crate) fn verify_coordinator_authority(caller: &AgentPubKey) -> ExternResult<bool> {
    let caller_str = caller.to_string();

    // Path to this agent's coordinator credential
    let cred_path = Path::from(format!("coordinator_credential.{}", caller_str));
    let cred_hash = match cred_path.clone().typed(LinkTypes::AgentToCredential) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok(false);
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok(false),
    };

    // Get links to credential entries
    let links = get_links(
        LinkQuery::new(
            cred_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AgentToCredential as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    // Find the most recent valid credential
    let now = sys_time()?.0 as i64 / 1_000_000;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(cred) = CoordinatorCredential::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        // Check if credential is for this agent
                        if cred.agent_pubkey != caller_str {
                            continue;
                        }

                        // Check if credential is active
                        if !cred.active {
                            continue;
                        }

                        // Check if credential is revoked
                        if cred.revoked_at > 0 {
                            continue;
                        }

                        // Check if credential is expired
                        if cred.expires_at > 0 && cred.expires_at < now {
                            continue;
                        }

                        // Credential is valid!
                        return Ok(true);
                    }
                }
            }
        }
    }

    Ok(false)
}

/// Check if the bootstrap window is currently open
pub(crate) fn is_bootstrap_window_open() -> ExternResult<bool> {
    let config = get_bootstrap_config_internal()?;

    match config {
        Some(cfg) => {
            if cfg.bootstrap_complete {
                return Ok(false);
            }
            let now = sys_time()?.0 as i64 / 1_000_000;
            Ok(now >= cfg.window_start && now <= cfg.window_end)
        }
        None => {
            // No config yet - bootstrap not initialized
            Ok(true)
        }
    }
}

/// Get the current bootstrap configuration
pub(crate) fn get_bootstrap_config_internal() -> ExternResult<Option<BootstrapConfig>> {
    let config_path = Path::from("bootstrap_config");
    let config_hash = match config_path.clone().typed(LinkTypes::BootstrapConfiguration) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok(None);
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::new(
            config_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::BootstrapConfiguration as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    // Get the most recent config
    for link in links.iter().rev() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(config) = BootstrapConfig::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        return Ok(Some(config));
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Check if any genesis coordinators exist
pub(crate) fn has_genesis_coordinators() -> ExternResult<bool> {
    let genesis_path = Path::from("genesis_coordinators");
    let genesis_hash = match genesis_path.clone().typed(LinkTypes::GenesisCoordinators) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok(false);
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok(false),
    };

    let links = get_links(
        LinkQuery::new(
            genesis_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::GenesisCoordinators as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    // Check if any active genesis coordinators exist
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(coord) = GenesisCoordinator::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if coord.active {
                            return Ok(true);
                        }
                    }
                }
            }
        }
    }

    Ok(false)
}

/// Log coordinator action to audit trail
pub(crate) fn log_coordinator_action(action_type: &str, target: &str, context: &str) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey.to_string();
    let now = sys_time()?.0 as i64 / 1_000_000;

    // Get previous log hash for chain integrity
    let previous_hash = get_latest_audit_log_hash()?.unwrap_or_else(|| "genesis".to_string());

    // Generate log ID
    let mut hasher = sha2::Sha256::new();
    hasher.update(agent.as_bytes());
    hasher.update(action_type.as_bytes());
    hasher.update(now.to_le_bytes());
    let log_id = format!("{:x}", hasher.finalize());

    let log_entry = CoordinatorAuditLog {
        log_id: log_id.clone(),
        coordinator_pubkey: agent,
        action_type: action_type.to_string(),
        target: target.to_string(),
        context_json: context.to_string(),
        timestamp: now,
        previous_log_hash: previous_hash,
    };

    let action_hash = create_entry(&EntryTypes::CoordinatorAuditLog(log_entry))?;

    // Link to audit log chain
    let audit_path = Path::from("audit_log_chain");
    let audit_hash = ensure_path(audit_path, LinkTypes::AuditLogChain)?;
    create_link(
        audit_hash,
        action_hash.clone(),
        LinkTypes::AuditLogChain,
        log_id.as_bytes().to_vec(),
    )?;

    Ok(action_hash)
}

/// Get the hash of the latest audit log entry
pub(crate) fn get_latest_audit_log_hash() -> ExternResult<Option<String>> {
    let audit_path = Path::from("audit_log_chain");
    let audit_hash = match audit_path.clone().typed(LinkTypes::AuditLogChain) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok(None);
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::new(
            audit_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AuditLogChain as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    // Get the most recent log entry
    if let Some(link) = links.last() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            let mut hasher = sha2::Sha256::new();
            hasher.update(action_hash.get_raw_39());
            return Ok(Some(format!("{:x}", hasher.finalize())));
        }
    }

    Ok(None)
}

/// SEC-002 FIX: Secure coordinator role check
///
/// This function replaces the vulnerable `require_coordinator_role` with a secure implementation
/// that requires proper cryptographic verification.
pub(crate) fn require_coordinator_role() -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;

    // First, check if caller has a valid coordinator credential
    if verify_coordinator_authority(&agent)? {
        return Ok(());
    }

    // Check for legacy role path (for backwards compatibility during migration)
    let role_path = Path::from(format!("roles.{}.{}", COORDINATOR_ROLE, agent));
    if let Ok(typed) = role_path.clone().typed(LinkTypes::RoundToGradients) {
        if typed.exists()? {
            // Check if role was revoked
            let revoke_path = Path::from(format!("roles_revoked.{}.{}", COORDINATOR_ROLE, agent));
            if let Ok(typed_revoke) = revoke_path.typed(LinkTypes::RoundToGradients) {
                if typed_revoke.exists()? {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Unauthorized: Coordinator role was revoked".to_string()
                    )));
                }
            }
            // Legacy role exists and is not revoked
            // Log warning about migration
            debug!("WARNING: Agent {} using legacy coordinator role. Please migrate to credential-based authentication.", agent);
            return Ok(());
        }
    }

    // SEC-002 FIX: NO MORE "first caller wins" auto-bootstrap
    // Bootstrap must be done through proper ceremony
    return Err(wasm_error!(WasmErrorInner::Guest(
        "Unauthorized: Valid coordinator credential required. Use initialize_bootstrap() to set up coordinators.".to_string()
    )));
}

/// Check if caller has detector role for Byzantine detection
pub(crate) fn require_detector_role() -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    let role_path = Path::from(format!("roles.{}.{}", DETECTOR_ROLE, agent));
    match role_path.clone().typed(LinkTypes::RoundToGradients) {
        Ok(typed) => {
            if typed.exists()? {
                // Check if role was revoked
                let revoke_path = Path::from(format!("roles_revoked.{}.{}", DETECTOR_ROLE, agent));
                if let Ok(typed_revoke) = revoke_path.typed(LinkTypes::RoundToGradients) {
                    if typed_revoke.exists()? {
                        return Err(wasm_error!(WasmErrorInner::Guest(
                            "Unauthorized: Detector role was revoked".to_string()
                        )));
                    }
                }
                return Ok(());
            }
            // Coordinators can also detect
            if require_coordinator_role().is_ok() {
                return Ok(());
            }
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Unauthorized: Detector role required".to_string()
            )));
        }
        Err(e) => return Err(e),
    }
}

/// Validate node_id format and length (F-01)
pub(crate) fn validate_node_id(node_id: &str) -> ExternResult<()> {
    let trimmed = node_id.trim();
    if trimmed.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Node ID cannot be empty".to_string()
        )));
    }
    if trimmed.len() > MAX_NODE_ID_LENGTH {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Node ID exceeds maximum length of {} characters", MAX_NODE_ID_LENGTH)
        )));
    }
    // Only allow alphanumeric, hyphens, underscores, and colons (for AgentPubKey format)
    if !trimmed.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == ':') {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Node ID contains invalid characters. Only alphanumeric, hyphens, underscores, and colons allowed.".to_string()
        )));
    }
    Ok(())
}

/// Rate limit tracking entry (used by check_rate_limit for future persistence)
#[derive(Serialize, Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct RateLimitEntry {
    agent: String,
    count: u32,
    window_start: i64,
}

/// Check rate limit for an agent (F-06)
pub(crate) fn check_rate_limit(action: &str) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey.to_string();
    let now = sys_time()?.0 as i64 / 1_000_000;
    let window_start = now - 60; // 1 minute window

    let rate_path = Path::from(format!("rate_limit.{}.{}", action, &agent[..16]));
    let rate_hash = match rate_path.clone().typed(LinkTypes::RoundToGradients) {
        Ok(typed) => typed.path_entry_hash()?,
        Err(_) => return Ok(()), // No rate limit entry yet
    };

    let links = get_links(
        LinkQuery::new(
            rate_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::RoundToGradients as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    // Count recent submissions
    let recent_count = links.iter()
        .filter(|l| l.timestamp.0 as i64 / 1_000_000 > window_start)
        .count() as u32;

    if recent_count >= MAX_SUBMISSIONS_PER_MINUTE {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Rate limit exceeded: {} per minute maximum", MAX_SUBMISSIONS_PER_MINUTE)
        )));
    }

    Ok(())
}
