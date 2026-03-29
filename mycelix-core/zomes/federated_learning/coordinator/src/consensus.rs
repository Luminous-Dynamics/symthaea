// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Commit-reveal aggregation consensus for decentralized FL.
//!
//! Validators independently compute aggregations and participate in
//! commit-reveal consensus to verify aggregation correctness.

use hdk::prelude::*;
use federated_learning_integrity::*;

use mycelix_fl::pipeline::DecentralizedPipeline;

use crate::auth::*;
use crate::signals::Signal;
use crate::bridge::call_bridge_record_reputation_sync;
use super::ensure_path;
use crate::scheduling::get_active_round_schedule_internal;
use crate::pipeline::get_or_create_reputation;

/// Input for validator registration
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterValidatorInput {
    /// K-Vector trust profile as JSON (10 dimensions)
    pub kvector_json: String,
    /// Composite trust score from K-Vector
    pub trust_score: f32,
    /// Ed25519 public key for signing commitments (base64)
    pub signing_key: String,
}

/// Result of validator registration
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterValidatorResult {
    pub success: bool,
    pub action_hash: Option<ActionHash>,
    pub validator_count: usize,
}

/// Register as an aggregation validator
///
/// Any agent with sufficient K-Vector trust can register as a validator.
/// Validators independently compute aggregations and participate in
/// commit-reveal consensus to verify aggregation correctness.
#[hdk_extern]
pub fn register_as_validator(input: RegisterValidatorInput) -> ExternResult<RegisterValidatorResult> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_str = agent.to_string();
    let now = sys_time()?.0 as i64 / 1_000_000;

    // SECURITY [H-07]: Check for duplicate registration before creating a new one.
    // Prevents an agent from registering multiple times to gain extra weight.
    let existing_validators = get_active_validators_internal()?;
    if existing_validators.iter().any(|v| v.agent_pubkey == agent_str) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Already registered as a validator. Deregister first to re-register.".to_string()
        )));
    }

    // SECURITY [H-07]: Self-reported trust_score is ignored.
    // Look up on-chain reputation to prevent any agent from claiming trust_score: 1.0.
    let on_chain_rep = get_or_create_reputation(&agent_str)?;
    let on_chain_trust = on_chain_rep.reputation_score;

    if on_chain_trust < federated_learning_integrity::MIN_VALIDATOR_REPUTATION {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "On-chain reputation ({:.3}) below minimum validator threshold ({:.3})",
                on_chain_trust,
                federated_learning_integrity::MIN_VALIDATOR_REPUTATION
            )
        )));
    }

    // Validate K-Vector JSON
    if serde_json::from_str::<serde_json::Value>(&input.kvector_json).is_err() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "K-Vector must be valid JSON".to_string()
        )));
    }

    // Create validator registration entry using on-chain trust, not self-reported
    let registration = ValidatorRegistration {
        agent_pubkey: agent_str.clone(),
        reputation_score: on_chain_trust,
        trust_score: on_chain_trust,
        signing_key: input.signing_key,
        kvector_json: input.kvector_json,
        registered_at: now,
        active: true,
        rounds_participated: 0,
        consensus_matches: 0,
    };

    let action_hash = create_entry(&EntryTypes::ValidatorRegistration(registration))?;

    // Link to validator index
    let validators_path = Path::from("validators");
    let validators_hash = ensure_path(validators_path, LinkTypes::ValidatorRegistrations)?;
    create_link(
        validators_hash,
        action_hash.clone(),
        LinkTypes::ValidatorRegistrations,
        agent_str.as_bytes().to_vec(),
    )?;

    // Link from agent to their validator registration
    let agent_path = Path::from(format!("agent_validator.{}", agent_str));
    let agent_hash = ensure_path(agent_path, LinkTypes::AgentToValidator)?;
    create_link(
        agent_hash,
        action_hash.clone(),
        LinkTypes::AgentToValidator,
        vec![],
    )?;

    // Count active validators
    let count = get_active_validators_internal()?.len();

    Ok(RegisterValidatorResult {
        success: true,
        action_hash: Some(action_hash),
        validator_count: count,
    })
}

/// Get all active validators
#[hdk_extern]
pub fn get_active_validators(_: ()) -> ExternResult<Vec<ValidatorRegistration>> {
    get_active_validators_internal()
}

/// Internal helper to get active validators
pub(crate) fn get_active_validators_internal() -> ExternResult<Vec<ValidatorRegistration>> {
    let validators_path = Path::from("validators");
    let validators_hash = match validators_path.clone().typed(LinkTypes::ValidatorRegistrations) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok(vec![]);
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok(vec![]),
    };

    let links = get_links(
        LinkQuery::new(
            validators_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::ValidatorRegistrations as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut validators = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(reg) = ValidatorRegistration::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if reg.active {
                            validators.push(reg);
                        }
                    }
                }
            }
        }
    }

    Ok(validators)
}

/// Input for submitting an aggregation commitment
#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitCommitmentInput {
    /// FL round number
    pub round: u64,
    /// SHA-256(aggregated_HV16 || method || round)
    pub commitment_hash: String,
    /// Aggregation method used
    pub method: String,
    /// Number of gradients included
    pub gradient_count: u32,
    /// Number excluded by Byzantine detection
    pub excluded_count: u32,
}

/// Submit an aggregation commitment (commit phase)
///
/// After a round's gradient collection deadline, validators independently
/// run the full pipeline and commit a hash of their result. This prevents
/// validators from copying each other.
#[hdk_extern]
pub fn submit_aggregation_commitment(input: SubmitCommitmentInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_str = agent.to_string();
    let now = sys_time()?.0 as i64 / 1_000_000;

    // Verify caller is a registered validator
    let validators = get_active_validators_internal()?;
    let validator = validators.iter().find(|v| v.agent_pubkey == agent_str);
    let trust_score = match validator {
        Some(v) => v.trust_score,
        None => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Must be a registered validator to submit commitments".to_string()
            )));
        }
    };

    // Check round schedule to verify we're in commit phase
    if let Some(schedule) = get_active_round_schedule_internal()? {
        let round_end = schedule.round_start_time + schedule.round_duration_secs as i64;
        let commit_end = round_end + schedule.commit_window_secs as i64;

        if now < round_end {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Gradient collection phase still active — commit not allowed yet".to_string()
            )));
        }

        if now > commit_end {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Commit window has closed".to_string()
            )));
        }
    }

    // Check for duplicate commitment from this agent for this round
    let existing = get_round_commitments_internal(input.round)?;
    if existing.iter().any(|c| c.aggregator == agent_str) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Already submitted a commitment for this round".to_string()
        )));
    }

    let commit_round = input.round;
    let commit_hash_copy = input.commitment_hash.clone();

    let commitment = AggregationCommitment {
        round: input.round,
        aggregator: agent_str,
        commitment_hash: input.commitment_hash,
        method: input.method,
        gradient_count: input.gradient_count,
        excluded_count: input.excluded_count,
        committed_at: now,
        aggregator_trust_score: trust_score,
    };

    let action_hash = create_entry(&EntryTypes::AggregationCommitment(commitment))?;

    // Link to round
    let round_path = Path::from(format!("round_commitments.{}", commit_round));
    let round_hash = ensure_path(round_path, LinkTypes::RoundToCommitments)?;
    create_link(
        round_hash,
        action_hash.clone(),
        LinkTypes::RoundToCommitments,
        vec![],
    )?;

    // Emit gossip signal: this validator's commitment is ready
    emit_signal(Signal::CommitReady {
        round: commit_round,
        validator_id: agent.to_string(),
        commitment_hash: commit_hash_copy,
        source: Some(agent.to_string()),
        signature: None,
    })?;

    Ok(action_hash)
}

/// Get commitments for a round
#[hdk_extern]
pub fn get_round_commitments(round: u64) -> ExternResult<Vec<AggregationCommitment>> {
    get_round_commitments_internal(round)
}

pub(crate) fn get_round_commitments_internal(round: u64) -> ExternResult<Vec<AggregationCommitment>> {
    let round_path = Path::from(format!("round_commitments.{}", round));
    let round_hash = match round_path.clone().typed(LinkTypes::RoundToCommitments) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok(vec![]);
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok(vec![]),
    };

    let links = get_links(
        LinkQuery::new(
            round_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::RoundToCommitments as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut commitments = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(commit) = AggregationCommitment::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        commitments.push(commit);
                    }
                }
            }
        }
    }

    Ok(commitments)
}

/// Input for revealing aggregation result
#[derive(Serialize, Deserialize, Debug)]
pub struct RevealAggregationInput {
    /// FL round number
    pub round: u64,
    /// The actual aggregated HV16 data (typically 2KB)
    pub result_data: Vec<u8>,
    /// Byzantine detection summary as JSON
    pub detection_summary_json: String,
    /// Shapley values as JSON array of [node_id, value]
    pub shapley_values_json: String,
}

/// Reveal aggregation result (reveal phase)
///
/// After the commit deadline, validators reveal their actual aggregation results.
/// The SHA-256 hash of result_data must match the earlier commitment_hash.
#[hdk_extern]
pub fn reveal_aggregation(input: RevealAggregationInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_str = agent.to_string();
    let now = sys_time()?.0 as i64 / 1_000_000;

    // Verify caller is a registered validator
    let validators = get_active_validators_internal()?;
    if !validators.iter().any(|v| v.agent_pubkey == agent_str) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Must be a registered validator to reveal aggregation".to_string()
        )));
    }

    // Find the commitment for this round from this agent
    let commitments = get_round_commitments_internal(input.round)?;
    let commitment = commitments.iter().find(|c| c.aggregator == agent_str);
    let commitment = match commitment {
        Some(c) => c,
        None => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "No commitment found for this round — must commit before revealing".to_string()
            )));
        }
    };

    // Compute SHA-256 of result data using the SAME function as the pipeline
    let result_hash = DecentralizedPipeline::commitment_hash(
        &input.result_data,
        &commitment.method,
        input.round,
    );

    // Verify hash matches commitment
    if result_hash != commitment.commitment_hash {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Reveal hash {} does not match commitment hash {}",
                result_hash, commitment.commitment_hash
            )
        )));
    }

    // Check round schedule to verify we're in reveal phase
    if let Some(schedule) = get_active_round_schedule_internal()? {
        let round_end = schedule.round_start_time + schedule.round_duration_secs as i64;
        let commit_end = round_end + schedule.commit_window_secs as i64;
        let reveal_end = commit_end + schedule.reveal_window_secs as i64;

        if now < commit_end {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Commit phase still active — reveal not allowed yet".to_string()
            )));
        }

        if now > reveal_end {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Reveal window has closed".to_string()
            )));
        }
    }

    let reveal = AggregationReveal {
        round: input.round,
        aggregator: agent_str,
        result_data: input.result_data,
        result_hash,
        detection_summary_json: input.detection_summary_json,
        shapley_values_json: input.shapley_values_json,
        revealed_at: now,
    };

    let action_hash = create_entry(&EntryTypes::AggregationReveal(reveal))?;

    // Link to round
    let round_path = Path::from(format!("round_reveals.{}", input.round));
    let round_hash = ensure_path(round_path, LinkTypes::RoundToReveals)?;
    create_link(
        round_hash,
        action_hash.clone(),
        LinkTypes::RoundToReveals,
        vec![],
    )?;

    Ok(action_hash)
}

/// Get reveals for a round
#[hdk_extern]
pub fn get_round_reveals(round: u64) -> ExternResult<Vec<AggregationReveal>> {
    get_round_reveals_internal(round)
}

pub(crate) fn get_round_reveals_internal(round: u64) -> ExternResult<Vec<AggregationReveal>> {
    let round_path = Path::from(format!("round_reveals.{}", round));
    let round_hash = match round_path.clone().typed(LinkTypes::RoundToReveals) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok(vec![]);
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok(vec![]),
    };

    let links = get_links(
        LinkQuery::new(
            round_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::RoundToReveals as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut reveals = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(reveal) = AggregationReveal::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        reveals.push(reveal);
                    }
                }
            }
        }
    }

    Ok(reveals)
}

/// Finalize round consensus
///
/// Any node can call this after the reveal phase. It tallies validator reveals,
/// groups by result hash, computes reputation^2-weighted agreement, and if
/// >2/3 agrees on a hash, creates a ConsensusResult.
///
/// Disagreeing validators lose reputation (handled via Bridge reputation sync).
#[hdk_extern]
pub fn finalize_round_consensus(round: u64) -> ExternResult<Option<ConsensusResult>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_str = agent.to_string();
    let now = sys_time()?.0 as i64 / 1_000_000;

    // Check if consensus already exists for this round
    if let Some(existing) = get_round_consensus_internal(round)? {
        return Ok(Some(existing));
    }

    // Get all reveals for this round
    let reveals = get_round_reveals_internal(round)?;
    if reveals.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "No reveals found for this round".to_string()
        )));
    }

    // Get commitments to look up trust scores
    let commitments = get_round_commitments_internal(round)?;

    // Group reveals by result hash and compute reputation^2-weighted agreement
    let mut hash_groups: std::collections::HashMap<String, Vec<(String, f64)>> =
        std::collections::HashMap::new();
    let mut total_weight: f64 = 0.0;

    for reveal in &reveals {
        // Find the commitment for this aggregator to get trust score
        let trust_score = commitments.iter()
            .find(|c| c.aggregator == reveal.aggregator)
            .map(|c| c.aggregator_trust_score as f64)
            .unwrap_or(0.0);

        // Reputation^2 weighting
        let weight = trust_score * trust_score;
        total_weight += weight;

        hash_groups
            .entry(reveal.result_hash.clone())
            .or_default()
            .push((reveal.aggregator.clone(), weight));
    }

    // Find the hash with the most weight
    let mut best_hash = String::new();
    let mut best_weight: f64 = 0.0;
    let mut best_validators: Vec<String> = vec![];

    for (hash, voters) in &hash_groups {
        let group_weight: f64 = voters.iter().map(|(_, w)| w).sum();
        if group_weight > best_weight {
            best_hash = hash.clone();
            best_weight = group_weight;
            best_validators = voters.iter().map(|(v, _)| v.clone()).collect();
        }
    }

    // Check quorum (>2/3 weighted agreement)
    if total_weight <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "No valid validators with positive trust scores".to_string()
        )));
    }

    let quorum_ratio = best_weight / total_weight;
    if quorum_ratio < federated_learning_integrity::CONSENSUS_QUORUM_THRESHOLD {
        // No consensus yet -- return None (may need more reveals)
        return Ok(None);
    }

    // Find the method used (from the first agreeing reveal)
    let method = reveals.iter()
        .find(|r| r.result_hash == best_hash)
        .and_then(|r| {
            commitments.iter()
                .find(|c| c.aggregator == r.aggregator)
                .map(|c| c.method.clone())
        })
        .unwrap_or_else(|| "unknown".to_string());

    // Create consensus result
    let consensus = ConsensusResult {
        round,
        agreed_hash: best_hash,
        agreeing_validators_json: serde_json::to_string(&best_validators)
            .unwrap_or_else(|_| "[]".to_string()),
        total_weight,
        consensus_weight: best_weight,
        method,
        finalized_at: now,
        finalized_by: agent_str.clone(),
    };

    let action_hash = create_entry(&EntryTypes::ConsensusResult(consensus.clone()))?;

    // Link to round
    let round_path = Path::from(format!("round_consensus.{}", round));
    let round_hash = ensure_path(round_path, LinkTypes::RoundToConsensus)?;
    create_link(
        round_hash,
        action_hash.clone(),
        LinkTypes::RoundToConsensus,
        vec![],
    )?;

    // Penalize disagreeing validators via reputation
    for reveal in &reveals {
        if !best_validators.contains(&reveal.aggregator) {
            // Disagreeing validator -- record as Byzantine behavior
            let _ = call_bridge_record_reputation_sync(
                &reveal.aggregator,
                -0.1, // Reputation penalty for disagreement
                0,    // positive_interactions
                1,    // negative_interactions
                Some(format!("Aggregation consensus disagreement round {}", round)),
            );
        }
    }

    // Log the finalization
    let _ = log_coordinator_action(
        "finalize_consensus",
        &format!("round_{}", round),
        &format!("{{\"quorum_ratio\":{:.3},\"validators\":{},\"total_weight\":{:.3}}}",
            quorum_ratio, best_validators.len(), total_weight),
    );

    // Emit gossip signal: consensus reached for this round
    emit_signal(Signal::ConsensusReached {
        round,
        agreed_hash: consensus.agreed_hash.clone(),
        validator_count: best_validators.len() as u32,
        consensus_weight: best_weight,
        source: Some(agent_str),
        signature: None,
    })?;

    Ok(Some(consensus))
}

/// Get consensus result for a round
#[hdk_extern]
pub fn get_round_consensus(round: u64) -> ExternResult<Option<ConsensusResult>> {
    get_round_consensus_internal(round)
}

pub(crate) fn get_round_consensus_internal(round: u64) -> ExternResult<Option<ConsensusResult>> {
    let round_path = Path::from(format!("round_consensus.{}", round));
    let round_hash = match round_path.clone().typed(LinkTypes::RoundToConsensus) {
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
            round_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::RoundToConsensus as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(consensus) = ConsensusResult::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        return Ok(Some(consensus));
                    }
                }
            }
        }
    }

    Ok(None)
}
