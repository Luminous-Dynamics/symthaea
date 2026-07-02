// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Model versioning, configuration registration, and version validation.

use hdk::prelude::*;
use federated_learning_integrity::*;

use crate::auth::require_coordinator_role;
use super::ensure_path;

// =============================================================================
// MODEL VERSION VALIDATION
// =============================================================================

/// Validate that a submitted gradient's architecture_hash matches the round's ModelConfig.
/// Returns Ok(()) if:
/// - No architecture_hash was provided (backwards compatible)
/// - No ModelConfig is registered for this round (no enforcement)
/// - The hashes match
/// Returns Err if both are present and they don't match.
#[allow(dead_code)] // Designed for future wiring into submit_gradient validation path
pub(crate) fn validate_model_version(round: u32, submitted_hash: &Option<String>) -> ExternResult<()> {
    let submitted = match submitted_hash {
        Some(h) if !h.is_empty() => h,
        _ => return Ok(()), // No hash provided — skip validation (backwards compatible)
    };

    // Look up the registered ModelConfig for this round
    let config_path = Path::from(format!("model_config.round.{}", round));
    let config_hash = match config_path.path_entry_hash() {
        Ok(h) => h,
        Err(_) => return Ok(()), // No config registered — skip validation
    };

    let links = get_links(
        LinkQuery::try_new(config_hash, LinkTypes::RoundToModelConfig)?,
        GetStrategy::default(),
    )?;

    // Get the latest config for this round
    let config = if let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid ModelConfig link target".into())))?;

        match get(action_hash, GetOptions::default())? {
            Some(record) => record
                .entry()
                .to_app_option::<ModelConfig>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
            None => return Ok(()), // Record not found — skip
        }
    } else {
        return Ok(()); // No config registered for this round
    };

    if let Some(config) = config {
        if config.architecture_hash != *submitted {
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!(
                    "Model version mismatch for round {}: expected architecture_hash '{}', got '{}'",
                    round, config.architecture_hash, submitted
                )
            )));
        }
    }

    Ok(())
}

/// Check that all gradients in a round were computed against the same model config.
/// Used during aggregation to prevent mixing gradients from different model versions.
/// Returns Ok(architecture_hash) if consistent, or Ok(None) if no config registered.
#[allow(dead_code)] // Designed for future wiring into aggregation validation
pub(crate) fn validate_round_model_consistency(round: u32) -> ExternResult<Option<String>> {
    let config_path = Path::from(format!("model_config.round.{}", round));
    let config_hash = match config_path.path_entry_hash() {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::try_new(config_hash, LinkTypes::RoundToModelConfig)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid ModelConfig link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(config) = record
                .entry()
                .to_app_option::<ModelConfig>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                return Ok(Some(config.architecture_hash));
            }
        }
    }

    Ok(None)
}

// =============================================================================
// MODEL CONFIGURATION API
// =============================================================================

/// Input for registering a model configuration
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterModelConfigInput {
    pub round: u32,
    pub architecture_hash: String,
    pub hyperparameters_json: String,
    pub dimension: u32,
    pub serialization_format: String,
    pub description: String,
}

/// Register a model configuration for a training round
/// SECURITY (F-03): Requires coordinator role
#[hdk_extern]
pub fn register_model_config(input: RegisterModelConfigInput) -> ExternResult<ActionHash> {
    require_coordinator_role()?;

    let now = sys_time()?;
    let agent = agent_info()?.agent_initial_pubkey;

    let config = ModelConfig {
        round: input.round,
        architecture_hash: input.architecture_hash,
        hyperparameters_json: input.hyperparameters_json,
        dimension: input.dimension,
        serialization_format: input.serialization_format,
        description: input.description,
        created_at: (now.as_micros() / 1_000_000) as i64,
        registered_by: format!("{}", agent),
    };

    let action_hash = create_entry(&EntryTypes::ModelConfig(config))?;

    // Link from round path for lookup by round
    let round_path = Path::from(format!("model_config.round.{}", input.round));
    let round_hash = ensure_path(round_path, LinkTypes::RoundToModelConfig)?;
    create_link(round_hash, action_hash.clone(), LinkTypes::RoundToModelConfig, ())?;

    // Link from registry for listing all configs
    let registry_path = Path::from("model_config.registry");
    let registry_hash = ensure_path(registry_path, LinkTypes::ModelConfigRegistry)?;
    create_link(registry_hash, action_hash.clone(), LinkTypes::ModelConfigRegistry, ())?;

    Ok(action_hash)
}

/// Get model configuration for a specific round
#[hdk_extern]
pub fn get_model_config(round: u32) -> ExternResult<Option<ModelConfig>> {
    let round_path = Path::from(format!("model_config.round.{}", round));
    let round_hash = match round_path.path_entry_hash() {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::try_new(round_hash, LinkTypes::RoundToModelConfig)?,
        GetStrategy::default(),
    )?;

    // Get the latest config for this round
    if let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            return record
                .entry()
                .to_app_option::<ModelConfig>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())));
        }
    }

    Ok(None)
}

/// List all registered model configurations
#[hdk_extern]
pub fn list_model_configs(_: ()) -> ExternResult<Vec<ModelConfig>> {
    let registry_path = Path::from("model_config.registry");
    let registry_hash = match registry_path.path_entry_hash() {
        Ok(h) => h,
        Err(_) => return Ok(Vec::new()),
    };

    let links = get_links(
        LinkQuery::try_new(registry_hash, LinkTypes::ModelConfigRegistry)?,
        GetStrategy::default(),
    )?;

    let mut configs = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(config) = record
                .entry()
                .to_app_option::<ModelConfig>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                configs.push(config);
            }
        }
    }

    // Sort by round number
    configs.sort_by_key(|c| c.round);
    Ok(configs)
}
