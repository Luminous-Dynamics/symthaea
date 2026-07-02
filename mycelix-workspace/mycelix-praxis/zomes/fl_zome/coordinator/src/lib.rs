// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Federated Learning Coordinator Zome
//!
//! Updated for Type 1 Civilization Substrate (Praxis-0.2)
//!
//! This zome coordinates byzantine-tolerant federated learning rounds.
//! It manages round lifecycle, participant registration, and aggregation triggers.

use fl_integrity::*;
use hdk::prelude::HdkPathExt;
use hdk::prelude::*;
use mycelix_zome_helpers as _;
use praxis_core::{ModelHash, RoundId, RoundState};

#[hdk_extern]
pub fn create_round(input: CreateRoundInput) -> ExternResult<ActionHash> {
    // Trust tier gate: requires Participant tier (score >= 0.3)
    mycelix_bridge_common::gate_civic(
        "edunet_bridge",
        &mycelix_bridge_common::civic_requirement_basic(),
        "create_round",
    )?;

    let now = sys_time()?;
    let round = FlRound {
        round_id: input.round_id.clone(),
        model_id: input.model_id.clone(),
        base_model_hash: input.parent_model_hash.clone(),
        state: RoundState::Active,
        min_participants: input.min_participants,
        max_participants: input.max_participants.unwrap_or(100),
        started_at: now.as_micros() / 1000,
        completed_at: None,
        clip_norm: input.privacy_params.clip_norm,
        privacy_epsilon: input.privacy_params.epsilon,
        privacy_delta: Some(0.0),
        current_participants: 0,
        aggregation_method: input.aggregation_method,
        aggregated_model_hash: None,
    };

    let action_hash = create_entry(EntryTypes::FlRound(round))?;

    // Link from model anchor to round
    let model_anchor = Path::from(format!(
        "model_rounds.{}",
        hex_encode(&input.parent_model_hash.0)
    ));
    let model_entry_hash = ensure_path(model_anchor, LinkTypes::ModelToRounds)?;
    create_link(
        model_entry_hash,
        action_hash.clone(),
        LinkTypes::ModelToRounds,
        (),
    )?;

    Ok(action_hash)
}

/// Get a round by its action hash
#[hdk_extern]
pub fn get_round(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Get all rounds for a specific model
#[hdk_extern]
pub fn get_model_rounds(model_hash: ModelHash) -> ExternResult<Vec<Record>> {
    let model_anchor = Path::from(format!("model_rounds.{}", hex_encode(&model_hash.0)));
    let model_entry_hash = ensure_path(model_anchor, LinkTypes::ModelToRounds)?;

    // Use Holochain 0.6 LinkQuery API
    let links = get_links(
        LinkQuery::new(
            model_entry_hash,
            LinkTypes::ModelToRounds.try_into_filter()?,
        ),
        GetStrategy::Local,
    )?;

    let mut rounds = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                rounds.push(record);
            }
        }
    }

    Ok(rounds)
}

/// Update an existing round (e.g., change state, add participants)
#[hdk_extern]
pub fn update_round(input: UpdateRoundInput) -> ExternResult<ActionHash> {
    let updated_action_hash = update_entry(
        input.original_action_hash,
        &EntryTypes::FlRound(input.updated_round),
    )?;
    Ok(updated_action_hash)
}
/// Submit a local model update with Proof of Gradient Quality (PoGQ)
#[hdk_extern]
pub fn submit_update(input: SubmitUpdateInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    // ── PROOF-IN-ZOME VERIFICATION ──
    // Instead of computing quality here, we verify the proof provided by the client.
    // This allows heavy HDC/ML math to happen on the host machine.
    verify_pogq_proof(&input.grad_commitment, &input.quality_proof).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "PoGQ Verification Failed: {}",
            e
        )))
    })?;

    let update = FlUpdate {
        round_id: input.round_id.clone(),
        model_id: input.model_id,
        parent_model_hash: input.parent_model_hash,
        grad_commitment: input.grad_commitment,
        clipped_l2_norm: input.clipped_l2_norm,
        local_val_loss: input.local_val_loss,
        sample_count: input.sample_count,
        timestamp: (sys_time()?.as_micros() / 1_000_000) as i64,
    };

    let action_hash = create_entry(EntryTypes::FlUpdate(update))?;

    // Link from round to update
    let round_anchor = Path::from(format!("round_updates.{}", input.round_id.0));
    let round_entry_hash = ensure_path(round_anchor, LinkTypes::RoundToUpdates)?;
    create_link(
        round_entry_hash,
        action_hash.clone(),
        LinkTypes::RoundToUpdates,
        (),
    )?;

    // Link from agent to their update (provenance)
    create_link(agent, action_hash.clone(), LinkTypes::RoundToUpdates, ())?;

    Ok(action_hash)
}

/// Lightweight verification of the Gradient Quality Proof.
/// Offloads the actual HDC math to the client.
fn verify_pogq_proof(commitment: &[u8], proof: &Vec<u8>) -> Result<(), String> {
    // 1. Verify commitment matches proof metadata
    // 2. Perform lightweight cryptographic check (e.g. Schnorr or basic ZKP verifier)
    if proof.is_empty() {
        return Err("Missing cryptographic proof".into());
    }

    // In production, this would call a compiled ZK verifier or HDC check
    // For now, we verify the commitment integrity.
    let hash = blake3::hash(commitment);
    if &proof[0..32] != hash.as_bytes() {
        return Err("Commitment mismatch in proof".into());
    }

    Ok(())
}

#[hdk_extern]
pub fn get_round_updates(round_id: RoundId) -> ExternResult<Vec<Record>> {
    let round_anchor = Path::from(format!("round_updates.{}", round_id.0));
    let round_entry_hash = ensure_path(round_anchor, LinkTypes::RoundToUpdates)?;

    let links = get_links(
        LinkQuery::new(
            round_entry_hash,
            LinkTypes::RoundToUpdates.try_into_filter()?,
        ),
        GetStrategy::Local,
    )?;

    let mut updates = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                updates.push(record);
            }
        }
    }

    Ok(updates)
}

/// Utility: Hash model gradients for PoGQ verification
pub fn compute_gradients_hash(gradients: &serde_json::Value) -> ExternResult<ModelHash> {
    // Serialize gradients and hash them
    let serialized = serde_json::to_string(gradients)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    let hash = blake3::hash(serialized.as_bytes());
    Ok(ModelHash(*hash.as_bytes()))
}

// ============================================================================
// Input/Output structures
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateRoundInput {
    pub round_id: RoundId,
    pub model_id: String,
    pub parent_model_hash: ModelHash,
    pub round_num: u32,
    pub min_participants: u32,
    pub max_participants: Option<u32>,
    pub deadline: i64,
    pub aggregation_method: String,
    pub privacy_params: praxis_core::PrivacyParams,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateRoundInput {
    pub original_action_hash: ActionHash,
    pub updated_round: FlRound,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitUpdateInput {
    pub round_id: RoundId,
    pub model_id: String,
    pub parent_model_hash: ModelHash,
    pub grad_commitment: Vec<u8>,
    pub quality_proof: Vec<u8>,
    pub clipped_l2_norm: f32,
    pub local_val_loss: f32,
    pub sample_count: u32,
}

// ============================================================================
// Helpers
// ============================================================================

fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.clone().typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

// ============== Skill Intelligence ==============

#[derive(Serialize, Deserialize, Debug)]
pub struct MasterySignalInput {
    pub skill: String,
    pub bucket: String,
}

#[hdk_extern]
pub fn contribute_mastery_signal(input: MasterySignalInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let signal_data = serde_json::json!({
        "skill": input.skill,
        "bucket": input.bucket,
        "agent": agent.to_string(),
    });

    let signal_bytes = serde_json::to_vec(&signal_data)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Serialize: {}", e))))?;

    let now = sys_time()?.as_micros() / 1_000_000;
    let update = FlUpdate {
        round_id: praxis_core::RoundId(format!("skill-intel-{}", input.skill)),
        model_id: "skill-intelligence-v1".to_string(),
        parent_model_hash: praxis_core::ModelHash([0u8; 32]),
        grad_commitment: signal_bytes,
        clipped_l2_norm: 0.0,
        local_val_loss: 0.0,
        sample_count: 1,
        timestamp: now as i64,
    };

    let hash = create_entry(EntryTypes::FlUpdate(update))?;

    let skill_anchor = ensure_path(
        Path::from(format!("skill_intel.{}", input.skill.to_lowercase())),
        LinkTypes::RoundToUpdates,
    )?;
    create_link(
        skill_anchor,
        hash.clone(),
        LinkTypes::RoundToUpdates,
        vec![],
    )?;

    Ok(hash)
}
