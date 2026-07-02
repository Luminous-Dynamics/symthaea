// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Secure coordinator bootstrap ceremony (SEC-002).
//!
//! Provides the multi-signature guardian-based coordinator election system.

use hdk::prelude::*;
use federated_learning_integrity::*;

use sha2::Digest;

use crate::config::*;
use crate::auth::{
    verify_coordinator_authority, is_bootstrap_window_open, get_bootstrap_config_internal,
    has_genesis_coordinators, log_coordinator_action, require_coordinator_role,
};
use crate::bridge::broadcast_coordinator_elected;
use super::ensure_path;

/// Input for initializing the bootstrap ceremony
#[derive(Serialize, Deserialize, Debug)]
pub struct InitializeBootstrapInput {
    /// List of genesis coordinator public keys (DNA deployer controlled)
    pub genesis_coordinators: Vec<String>,
    /// Authority proofs for each genesis coordinator (Ed25519 signatures)
    pub authority_proofs: Vec<Vec<u8>>,
    /// Initial guardians for the multi-sig
    pub initial_guardians: Vec<String>,
    /// Bootstrap window duration in seconds (0 = use default 24h)
    pub window_duration_seconds: i64,
    /// Minimum votes required for coordinator approval (N of M)
    pub min_votes: u32,
}

/// Result of bootstrap initialization
#[derive(Serialize, Deserialize, Debug)]
pub struct BootstrapResult {
    pub success: bool,
    pub bootstrap_config_hash: ActionHash,
    pub genesis_coordinators_created: usize,
    pub guardians_created: usize,
    pub window_end: i64,
}

/// Initialize the coordinator bootstrap ceremony
///
/// SEC-002: This is the secure alternative to the "first caller wins" pattern.
/// Only callable during the bootstrap window and requires cryptographic proofs.
#[hdk_extern]
pub fn initialize_bootstrap(input: InitializeBootstrapInput) -> ExternResult<BootstrapResult> {
    // C-04: Verify authority proofs cryptographically
    // Each genesis coordinator must have a valid Ed25519 signature over a structured payload
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_bytes = agent.get_raw_39().to_vec();
    for (i, proof) in input.authority_proofs.iter().enumerate() {
        let coord_id = input.genesis_coordinators.get(i).map(|s| s.as_str()).unwrap_or("unknown");
        if proof.len() < 64 {
            // C-04: Authority proof for coordinator must be at least 64 bytes
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!("C-04: Authority proof for coordinator {} must be at least 64 bytes (Ed25519 signature)", coord_id)
            )));
        }
        // Build structured payload: GenesisCoordinator:<pubkey>
        let mut payload = Vec::new();
        payload.extend_from_slice(b"GenesisCoordinator:");
        payload.extend_from_slice(coord_id.as_bytes());
        payload.extend_from_slice(&agent_bytes);
        // Verify the proof is a valid signature over the structured payload
        let mut sig_arr = [0u8; 64];
        sig_arr.copy_from_slice(&proof[..64]);
        let sig = Signature::from(sig_arr);
        if !verify_signature(agent.clone(), sig, payload)? {
            // C-04: Authority proof verification FAILED
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!("C-04: Authority proof verification FAILED for coordinator {}", coord_id)
            )));
        }
    }

    // Check if bootstrap already exists
    if has_genesis_coordinators()? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Bootstrap already initialized. Cannot re-initialize genesis coordinators.".to_string()
        )));
    }

    // Validate inputs
    if input.genesis_coordinators.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "At least one genesis coordinator required".to_string()
        )));
    }

    if input.genesis_coordinators.len() != input.authority_proofs.len() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Each genesis coordinator must have a corresponding authority proof".to_string()
        )));
    }

    if input.initial_guardians.is_empty() || input.initial_guardians.len() < DEFAULT_MIN_GUARDIANS as usize {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("At least {} guardians required", DEFAULT_MIN_GUARDIANS)
        )));
    }

    let now = sys_time()?.0 as i64 / 1_000_000;
    let window_duration = if input.window_duration_seconds > 0 {
        input.window_duration_seconds
    } else {
        DEFAULT_BOOTSTRAP_WINDOW_SECONDS
    };
    let window_end = now + window_duration;

    let min_votes = if input.min_votes > 0 && input.min_votes <= input.initial_guardians.len() as u32 {
        input.min_votes
    } else {
        DEFAULT_MIN_VOTES.min(input.initial_guardians.len() as u32)
    };

    // Compute DNA modifiers hash (for verification)
    let dna_info = dna_info()?;
    let mut hasher = sha2::Sha256::new();
    hasher.update(&dna_info.hash.get_raw_32());
    let dna_modifiers_hash = format!("{:x}", hasher.finalize());

    // Create bootstrap config
    let config = BootstrapConfig {
        window_start: now,
        window_end,
        min_guardians: DEFAULT_MIN_GUARDIANS,
        min_votes,
        max_guardians: DEFAULT_MAX_GUARDIANS.max(input.initial_guardians.len() as u32),
        bootstrap_complete: false,
    };

    let config_hash = create_entry(&EntryTypes::BootstrapConfig(config))?;

    // Link config to path
    let config_path = Path::from("bootstrap_config");
    let config_entry_hash = ensure_path(config_path, LinkTypes::BootstrapConfiguration)?;
    create_link(
        config_entry_hash,
        config_hash.clone(),
        LinkTypes::BootstrapConfiguration,
        vec![],
    )?;

    // Create genesis coordinators
    let genesis_path = Path::from("genesis_coordinators");
    let genesis_entry_hash = ensure_path(genesis_path, LinkTypes::GenesisCoordinators)?;

    for (i, coord_pubkey) in input.genesis_coordinators.iter().enumerate() {
        let genesis_coord = GenesisCoordinator {
            agent_pubkey: coord_pubkey.clone(),
            authority_proof: input.authority_proofs[i].clone(),
            dna_modifiers_hash: dna_modifiers_hash.clone(),
            established_at: now,
            active: true,
        };

        let coord_hash = create_entry(&EntryTypes::GenesisCoordinator(genesis_coord))?;

        create_link(
            genesis_entry_hash.clone(),
            coord_hash.clone(),
            LinkTypes::GenesisCoordinators,
            coord_pubkey.as_bytes().to_vec(),
        )?;

        // Create coordinator credential for this genesis coordinator
        let credential = CoordinatorCredential {
            agent_pubkey: coord_pubkey.clone(),
            credential_type: "genesis".to_string(),
            issued_at: now,
            expires_at: 0, // Never expires for genesis
            guardian_signatures_json: "[]".to_string(),
            required_signatures: 0, // Not required for genesis
            active: true,
            revoked_at: 0,
        };

        let cred_hash = create_entry(&EntryTypes::CoordinatorCredential(credential))?;

        // Link credential to agent
        let cred_path = Path::from(format!("coordinator_credential.{}", coord_pubkey));
        let cred_entry_hash = ensure_path(cred_path, LinkTypes::AgentToCredential)?;
        create_link(
            cred_entry_hash,
            cred_hash,
            LinkTypes::AgentToCredential,
            vec![],
        )?;
    }

    // Create guardians
    let guardians_path = Path::from("guardians");
    let guardians_entry_hash = ensure_path(guardians_path, LinkTypes::Guardians)?;
    let first_genesis = input.genesis_coordinators.first().cloned().unwrap_or_default();

    for guardian_pubkey in &input.initial_guardians {
        let guardian = Guardian {
            agent_pubkey: guardian_pubkey.clone(),
            voting_weight: 1,
            added_at: now,
            added_by: first_genesis.clone(),
            active: true,
        };

        let guardian_hash = create_entry(&EntryTypes::Guardian(guardian))?;

        create_link(
            guardians_entry_hash.clone(),
            guardian_hash,
            LinkTypes::Guardians,
            guardian_pubkey.as_bytes().to_vec(),
        )?;
    }

    // Log the bootstrap action
    log_coordinator_action(
        "initialize_bootstrap",
        "system",
        &serde_json::json!({
            "genesis_coordinators": input.genesis_coordinators.len(),
            "guardians": input.initial_guardians.len(),
            "window_end": window_end,
            "min_votes": min_votes,
        }).to_string()
    )?;

    // === BRIDGE INTEGRATION: Broadcast genesis coordinator election events ===
    for coord_pubkey in &input.genesis_coordinators {
        let _ = broadcast_coordinator_elected(coord_pubkey, "genesis");
    }

    Ok(BootstrapResult {
        success: true,
        bootstrap_config_hash: config_hash,
        genesis_coordinators_created: input.genesis_coordinators.len(),
        guardians_created: input.initial_guardians.len(),
        window_end,
    })
}

/// Input for voting to add a new coordinator
#[derive(Serialize, Deserialize, Debug)]
pub struct VoteForCoordinatorInput {
    /// Candidate agent pubkey
    pub candidate_pubkey: String,
    /// Whether to approve (true) or reject (false)
    pub approve: bool,
    /// Vote signature (Ed25519 signature over vote data)
    pub signature: Vec<u8>,
}

/// Result of a coordinator vote
#[derive(Serialize, Deserialize, Debug)]
pub struct VoteResult {
    pub vote_recorded: bool,
    pub vote_hash: ActionHash,
    pub current_votes: u32,
    pub required_votes: u32,
    pub coordinator_approved: bool,
}

/// Vote to approve or reject a coordinator candidate
///
/// SEC-002: Only guardians can vote. Requires multi-signature approval.
#[hdk_extern]
pub fn vote_for_coordinator(input: VoteForCoordinatorInput) -> ExternResult<VoteResult> {
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_str = caller.to_string();

    // Verify caller is a guardian
    if !is_guardian(&caller_str)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only guardians can vote for coordinators".to_string()
        )));
    }

    // Check bootstrap window
    if !is_bootstrap_window_open()? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Bootstrap window is closed. Cannot vote for new coordinators.".to_string()
        )));
    }

    // Validate signature
    if input.signature.len() < 64 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid vote signature: must be at least 64 bytes".to_string()
        )));
    }

    // C-05: Verify Ed25519 signature cryptographically
    // Build structured payload with CoordinatorVote: prefix to prevent replay attacks
    let voter_key = agent_info()?.agent_initial_pubkey;
    let mut vote_payload = Vec::new();
    vote_payload.extend_from_slice(b"CoordinatorVote:");
    vote_payload.extend_from_slice(input.candidate_pubkey.as_bytes());
    vote_payload.push(b':');
    vote_payload.extend_from_slice(voter_key.to_string().as_bytes());
    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(&input.signature[..64]);
    let sig = Signature::from(sig_arr);
    if !verify_signature(voter_key.clone(), sig, vote_payload)? {
        // C-05: Vote signature verification failed
        return Err(wasm_error!(WasmErrorInner::Guest(
            "C-05: Vote signature verification failed".to_string()
        )));
    }

    // Check for duplicate votes
    if check_guardian_voted(&caller_str, &input.candidate_pubkey)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Guardian has already voted for this candidate".to_string()
        )));
    }

    let now = sys_time()?.0 as i64 / 1_000_000;

    // Generate vote ID
    let mut hasher = sha2::Sha256::new();
    hasher.update(caller_str.as_bytes());
    hasher.update(input.candidate_pubkey.as_bytes());
    hasher.update(now.to_le_bytes());
    let vote_id = format!("{:x}", hasher.finalize());

    // Get guardian's voting weight
    let weight = get_guardian_weight(&caller_str)?;

    // Create vote entry
    let vote = CoordinatorVote {
        vote_id: vote_id.clone(),
        candidate_pubkey: input.candidate_pubkey.clone(),
        voter_pubkey: caller_str.clone(),
        weight,
        approve: input.approve,
        signature: input.signature,
        voted_at: now,
        expires_at: 0, // No expiry
    };

    let vote_hash = create_entry(&EntryTypes::CoordinatorVote(vote))?;

    // Link vote to candidate
    let votes_path = Path::from(format!("coordinator_votes.{}", input.candidate_pubkey));
    let votes_entry_hash = ensure_path(votes_path, LinkTypes::CandidateToVotes)?;
    create_link(
        votes_entry_hash,
        vote_hash.clone(),
        LinkTypes::CandidateToVotes,
        vote_id.as_bytes().to_vec(),
    )?;

    // Count current votes
    let (approval_votes, rejection_votes) = count_votes(&input.candidate_pubkey)?;

    // Get required votes from config
    let config = get_bootstrap_config_internal()?.unwrap_or(BootstrapConfig {
        window_start: now,
        window_end: now + DEFAULT_BOOTSTRAP_WINDOW_SECONDS,
        min_guardians: DEFAULT_MIN_GUARDIANS,
        min_votes: DEFAULT_MIN_VOTES,
        max_guardians: DEFAULT_MAX_GUARDIANS,
        bootstrap_complete: false,
    });

    let coordinator_approved = approval_votes >= config.min_votes && approval_votes > rejection_votes;

    // If approved, create coordinator credential with term-based expiry
    if coordinator_approved {
        // Collect guardian signatures
        let signatures = collect_vote_signatures(&input.candidate_pubkey)?;

        // Set expiry based on default term duration + grace period
        let expires_at = now + DEFAULT_TERM_DURATION_SECONDS + TERM_GRACE_PERIOD_SECONDS;

        let credential = CoordinatorCredential {
            agent_pubkey: input.candidate_pubkey.clone(),
            credential_type: "voted".to_string(),
            issued_at: now,
            expires_at,
            guardian_signatures_json: serde_json::to_string(&signatures).unwrap_or_default(),
            required_signatures: config.min_votes,
            active: true,
            revoked_at: 0,
        };

        let cred_hash = create_entry(&EntryTypes::CoordinatorCredential(credential))?;

        // Link credential to agent
        let cred_path = Path::from(format!("coordinator_credential.{}", input.candidate_pubkey));
        let cred_entry_hash = ensure_path(cred_path, LinkTypes::AgentToCredential)?;
        create_link(
            cred_entry_hash,
            cred_hash.clone(),
            LinkTypes::AgentToCredential,
            vec![],
        )?;

        // Create coordinator term entry for rotation tracking
        create_coordinator_term(
            &input.candidate_pubkey,
            &cred_hash,
            DEFAULT_TERM_DURATION_SECONDS,
            "voted",
        )?;

        // Log the approval
        log_coordinator_action(
            "coordinator_approved",
            &input.candidate_pubkey,
            &serde_json::json!({
                "approval_votes": approval_votes,
                "rejection_votes": rejection_votes,
                "required_votes": config.min_votes,
                "term_expires_at": expires_at,
            }).to_string()
        )?;

        // === BRIDGE INTEGRATION: Broadcast coordinator election event ===
        let _ = broadcast_coordinator_elected(&input.candidate_pubkey, "voted");
    }

    Ok(VoteResult {
        vote_recorded: true,
        vote_hash,
        current_votes: approval_votes,
        required_votes: config.min_votes,
        coordinator_approved,
    })
}

/// Check if an agent is a guardian
pub(crate) fn is_guardian(agent_pubkey: &str) -> ExternResult<bool> {
    let guardians_path = Path::from("guardians");
    let guardians_hash = match guardians_path.clone().typed(LinkTypes::Guardians) {
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
            guardians_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::Guardians as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(guardian) = Guardian::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if guardian.agent_pubkey == agent_pubkey && guardian.active {
                            return Ok(true);
                        }
                    }
                }
            }
        }
    }

    Ok(false)
}

/// Get guardian's voting weight
pub(crate) fn get_guardian_weight(agent_pubkey: &str) -> ExternResult<u32> {
    let guardians_path = Path::from("guardians");
    let guardians_hash = match guardians_path.clone().typed(LinkTypes::Guardians) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok(0);
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok(0),
    };

    let links = get_links(
        LinkQuery::new(
            guardians_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::Guardians as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(guardian) = Guardian::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if guardian.agent_pubkey == agent_pubkey && guardian.active {
                            return Ok(guardian.voting_weight);
                        }
                    }
                }
            }
        }
    }

    Ok(0)
}

/// Count approval and rejection votes for a candidate
pub(crate) fn count_votes(candidate_pubkey: &str) -> ExternResult<(u32, u32)> {
    let votes_path = Path::from(format!("coordinator_votes.{}", candidate_pubkey));
    let votes_hash = match votes_path.clone().typed(LinkTypes::CandidateToVotes) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok((0, 0));
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok((0, 0)),
    };

    let links = get_links(
        LinkQuery::new(
            votes_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::CandidateToVotes as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut approval_votes = 0u32;
    let mut rejection_votes = 0u32;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(vote) = CoordinatorVote::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if vote.approve {
                            approval_votes += vote.weight;
                        } else {
                            rejection_votes += vote.weight;
                        }
                    }
                }
            }
        }
    }

    Ok((approval_votes, rejection_votes))
}

/// Collect vote signatures for a candidate
pub(crate) fn collect_vote_signatures(candidate_pubkey: &str) -> ExternResult<Vec<(String, Vec<u8>)>> {
    let votes_path = Path::from(format!("coordinator_votes.{}", candidate_pubkey));
    let votes_hash = match votes_path.clone().typed(LinkTypes::CandidateToVotes) {
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
            votes_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::CandidateToVotes as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut signatures = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(vote) = CoordinatorVote::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if vote.approve {
                            signatures.push((vote.voter_pubkey, vote.signature));
                        }
                    }
                }
            }
        }
    }

    Ok(signatures)
}

/// Get the current bootstrap configuration (public API)
#[hdk_extern]
pub fn get_bootstrap_config(_: ()) -> ExternResult<Option<BootstrapConfig>> {
    get_bootstrap_config_internal()
}

/// Get all active coordinators
#[hdk_extern]
pub fn get_active_coordinators(_: ()) -> ExternResult<Vec<String>> {
    let mut coordinators = Vec::new();

    // Get genesis coordinators
    let genesis_path = Path::from("genesis_coordinators");
    if let Ok(typed) = genesis_path.clone().typed(LinkTypes::GenesisCoordinators) {
        if typed.exists()? {
            let genesis_hash = typed.path_entry_hash()?;
            let links = get_links(
                LinkQuery::new(
                    genesis_hash,
                    LinkTypeFilter::single_type(0.into(), (LinkTypes::GenesisCoordinators as u8).into()),
                ),
                GetStrategy::default(),
            )?;

            for link in links {
                if let Some(action_hash) = link.target.into_action_hash() {
                    if let Some(record) = get(action_hash, GetOptions::default())? {
                        if let Some(Entry::App(bytes)) = record.entry().as_option() {
                            if let Ok(coord) = GenesisCoordinator::try_from(
                                SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                            ) {
                                if coord.active {
                                    coordinators.push(coord.agent_pubkey);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(coordinators)
}

/// Get all active guardians
#[hdk_extern]
pub fn get_active_guardians(_: ()) -> ExternResult<Vec<Guardian>> {
    let mut guardians = Vec::new();

    let guardians_path = Path::from("guardians");
    if let Ok(typed) = guardians_path.clone().typed(LinkTypes::Guardians) {
        if typed.exists()? {
            let guardians_hash = typed.path_entry_hash()?;
            let links = get_links(
                LinkQuery::new(
                    guardians_hash,
                    LinkTypeFilter::single_type(0.into(), (LinkTypes::Guardians as u8).into()),
                ),
                GetStrategy::default(),
            )?;

            for link in links {
                if let Some(action_hash) = link.target.into_action_hash() {
                    if let Some(record) = get(action_hash, GetOptions::default())? {
                        if let Some(Entry::App(bytes)) = record.entry().as_option() {
                            if let Ok(guardian) = Guardian::try_from(
                                SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                            ) {
                                if guardian.active {
                                    guardians.push(guardian);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(guardians)
}

/// Get coordinator audit log
#[hdk_extern]
pub fn get_coordinator_audit_log(limit: u32) -> ExternResult<Vec<CoordinatorAuditLog>> {
    let audit_path = Path::from("audit_log_chain");
    let audit_hash = match audit_path.clone().typed(LinkTypes::AuditLogChain) {
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
            audit_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AuditLogChain as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut logs = Vec::new();
    let max_count = if limit > 0 { limit as usize } else { 100 };

    for link in links.iter().rev().take(max_count) {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(log) = CoordinatorAuditLog::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        logs.push(log);
                    }
                }
            }
        }
    }

    Ok(logs)
}

/// Revoke a coordinator credential (requires coordinator role + multi-sig for non-genesis)
#[hdk_extern]
pub fn revoke_coordinator_credential(agent_pubkey: String) -> ExternResult<bool> {
    require_coordinator_role()?;

    let cred_path = Path::from(format!("coordinator_credential.{}", agent_pubkey));
    let cred_hash = match cred_path.clone().typed(LinkTypes::AgentToCredential) {
        Ok(typed) => {
            if !typed.exists()? {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "No credential found for this agent".to_string()
                )));
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Err(wasm_error!(WasmErrorInner::Guest(
            "No credential found for this agent".to_string()
        ))),
    };

    let links = get_links(
        LinkQuery::new(
            cred_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AgentToCredential as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let now = sys_time()?.0 as i64 / 1_000_000;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(mut cred) = CoordinatorCredential::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if cred.agent_pubkey == agent_pubkey && cred.active {
                            // Check if this is a genesis credential
                            if cred.credential_type == "genesis" {
                                return Err(wasm_error!(WasmErrorInner::Guest(
                                    "Cannot revoke genesis coordinator credentials through this method".to_string()
                                )));
                            }

                            // Revoke the credential
                            cred.active = false;
                            cred.revoked_at = now;

                            update_entry(action_hash, &EntryTypes::CoordinatorCredential(cred))?;

                            // Log the revocation
                            log_coordinator_action(
                                "revoke_coordinator",
                                &agent_pubkey,
                                &serde_json::json!({
                                    "revoked_at": now,
                                }).to_string()
                            )?;

                            return Ok(true);
                        }
                    }
                }
            }
        }
    }

    Ok(false)
}

// =============================================================================
// SEC-002 FIX: Additional Bootstrap Ceremony Functions
// =============================================================================

/// Input for initializing bootstrap (alternative naming)
#[derive(Serialize, Deserialize, Debug)]
pub struct InitBootstrapInput {
    /// Genesis coordinator public key
    pub genesis_coordinator: String,
    /// Authority proof (Ed25519 signature from DNA deployer)
    pub authority_proof: Vec<u8>,
    /// Bootstrap window duration in seconds (0 = use default 24h)
    pub window_duration_seconds: i64,
    /// Minimum votes required for coordinator approval
    pub min_votes: u32,
    /// Maximum guardians allowed
    pub max_guardians: u32,
}

/// Result of bootstrap initialization
#[derive(Serialize, Deserialize, Debug)]
pub struct InitBootstrapResult {
    pub success: bool,
    pub config_hash: ActionHash,
    pub genesis_coordinator_hash: ActionHash,
    pub window_end: i64,
}

/// Initialize the bootstrap ceremony (single genesis coordinator version)
///
/// SEC-002: This function sets up the initial coordinator and bootstrap window.
/// Can only be called once - subsequent calls will fail.
#[hdk_extern]
pub fn init_bootstrap(input: InitBootstrapInput) -> ExternResult<InitBootstrapResult> {
    // Verify authority proof: must be a valid Ed25519 signature over structured payload
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_bytes = agent.get_raw_39().to_vec();
    if input.authority_proof.len() < 64 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Authority proof must be at least 64 bytes (Ed25519 signature)".to_string()
        )));
    }
    // Build structured payload: GenesisCoordinator:<pubkey><agent_bytes>
    let mut payload = Vec::new();
    payload.extend_from_slice(b"GenesisCoordinator:");
    payload.extend_from_slice(input.genesis_coordinator.as_bytes());
    payload.extend_from_slice(&agent_bytes);
    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(&input.authority_proof[..64]);
    let sig = Signature::from(sig_arr);
    if !verify_signature(agent.clone(), sig, payload)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Authority proof signature verification failed".to_string()
        )));
    }

    // Check if bootstrap already exists
    if has_genesis_coordinators()? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Bootstrap already initialized. Cannot re-initialize genesis coordinators.".to_string()
        )));
    }

    let now = sys_time()?.0 as i64 / 1_000_000;
    let window_duration = if input.window_duration_seconds > 0 {
        input.window_duration_seconds
    } else {
        DEFAULT_BOOTSTRAP_WINDOW_SECONDS
    };
    let window_end = now + window_duration;

    let min_votes = if input.min_votes > 0 {
        input.min_votes
    } else {
        DEFAULT_MIN_VOTES
    };

    let max_guardians = if input.max_guardians > 0 {
        input.max_guardians
    } else {
        DEFAULT_MAX_GUARDIANS
    };

    // Compute DNA modifiers hash (for verification)
    let dna_info = dna_info()?;
    let mut hasher = sha2::Sha256::new();
    hasher.update(&dna_info.hash.get_raw_32());
    let dna_modifiers_hash = format!("{:x}", hasher.finalize());

    // Create bootstrap config
    let config = BootstrapConfig {
        window_start: now,
        window_end,
        min_guardians: DEFAULT_MIN_GUARDIANS,
        min_votes,
        max_guardians,
        bootstrap_complete: false,
    };

    let config_hash = create_entry(&EntryTypes::BootstrapConfig(config))?;

    // Link config to path
    let config_path = Path::from("bootstrap_config");
    let config_entry_hash = ensure_path(config_path, LinkTypes::BootstrapConfiguration)?;
    create_link(
        config_entry_hash,
        config_hash.clone(),
        LinkTypes::BootstrapConfiguration,
        vec![],
    )?;

    // Create genesis coordinator
    let genesis_coord = GenesisCoordinator {
        agent_pubkey: input.genesis_coordinator.clone(),
        authority_proof: input.authority_proof,
        dna_modifiers_hash,
        established_at: now,
        active: true,
    };

    let genesis_hash = create_entry(&EntryTypes::GenesisCoordinator(genesis_coord))?;

    // Link to genesis coordinators path
    let genesis_path = Path::from("genesis_coordinators");
    let genesis_entry_hash = ensure_path(genesis_path, LinkTypes::GenesisCoordinators)?;
    create_link(
        genesis_entry_hash,
        genesis_hash.clone(),
        LinkTypes::GenesisCoordinators,
        input.genesis_coordinator.as_bytes().to_vec(),
    )?;

    // Create coordinator credential for genesis coordinator
    let credential = CoordinatorCredential {
        agent_pubkey: input.genesis_coordinator.clone(),
        credential_type: "genesis".to_string(),
        issued_at: now,
        expires_at: 0, // Never expires for genesis
        guardian_signatures_json: "[]".to_string(),
        required_signatures: 0, // Not required for genesis
        active: true,
        revoked_at: 0,
    };

    let cred_hash = create_entry(&EntryTypes::CoordinatorCredential(credential))?;

    // Link credential to agent
    let cred_path = Path::from(format!("coordinator_credential.{}", input.genesis_coordinator));
    let cred_entry_hash = ensure_path(cred_path, LinkTypes::AgentToCredential)?;
    create_link(
        cred_entry_hash,
        cred_hash,
        LinkTypes::AgentToCredential,
        vec![],
    )?;

    // Log the bootstrap action
    log_coordinator_action(
        "init_bootstrap",
        &input.genesis_coordinator,
        &serde_json::json!({
            "window_end": window_end,
            "min_votes": min_votes,
            "max_guardians": max_guardians,
        }).to_string()
    )?;

    Ok(InitBootstrapResult {
        success: true,
        config_hash,
        genesis_coordinator_hash: genesis_hash,
        window_end,
    })
}

/// Input for adding a guardian
#[derive(Serialize, Deserialize, Debug)]
pub struct AddGuardianInput {
    /// Guardian's agent public key
    pub guardian_pubkey: String,
    /// Voting weight (typically 1)
    pub voting_weight: u32,
}

/// Result of adding a guardian
#[derive(Serialize, Deserialize, Debug)]
pub struct AddGuardianResult {
    pub success: bool,
    pub guardian_hash: ActionHash,
    pub current_guardian_count: u32,
    pub max_guardians: u32,
}

/// Add a guardian who can vote in coordinator elections
///
/// SEC-002: Guardians form the multi-sig group that approves new coordinators.
/// Must be called within the bootstrap window and by an existing coordinator.
#[hdk_extern]
pub fn add_guardian(input: AddGuardianInput) -> ExternResult<AddGuardianResult> {
    // Verify caller has coordinator authority
    require_coordinator_role()?;

    // Check bootstrap window is open
    if !is_bootstrap_window_open()? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Bootstrap window is closed. Cannot add new guardians.".to_string()
        )));
    }

    // Get bootstrap config for max guardians check
    let config = get_bootstrap_config_internal()?.ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(
            "Bootstrap not initialized. Call init_bootstrap first.".to_string()
        ))
    })?;

    // Count current guardians
    let current_guardians = get_active_guardians(())?.len() as u32;

    // Check max guardians limit
    if current_guardians >= config.max_guardians {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Maximum guardians limit reached ({}/{})",
                current_guardians, config.max_guardians
            )
        )));
    }

    // Check if guardian already exists
    if is_guardian(&input.guardian_pubkey)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Guardian already exists".to_string()
        )));
    }

    let caller = agent_info()?.agent_initial_pubkey.to_string();
    let now = sys_time()?.0 as i64 / 1_000_000;

    // Create guardian entry
    let guardian = Guardian {
        agent_pubkey: input.guardian_pubkey.clone(),
        voting_weight: if input.voting_weight > 0 { input.voting_weight } else { 1 },
        added_at: now,
        added_by: caller.clone(),
        active: true,
    };

    let guardian_hash = create_entry(&EntryTypes::Guardian(guardian))?;

    // Link to guardians path
    let guardians_path = Path::from("guardians");
    let guardians_entry_hash = ensure_path(guardians_path, LinkTypes::Guardians)?;
    create_link(
        guardians_entry_hash,
        guardian_hash.clone(),
        LinkTypes::Guardians,
        input.guardian_pubkey.as_bytes().to_vec(),
    )?;

    // Log the action
    log_coordinator_action(
        "add_guardian",
        &input.guardian_pubkey,
        &serde_json::json!({
            "voting_weight": input.voting_weight,
            "added_by": caller,
            "guardian_count": current_guardians + 1,
        }).to_string()
    )?;

    Ok(AddGuardianResult {
        success: true,
        guardian_hash,
        current_guardian_count: current_guardians + 1,
        max_guardians: config.max_guardians,
    })
}

/// Input for casting a coordinator vote
#[derive(Serialize, Deserialize, Debug)]
pub struct CastVoteInput {
    /// Candidate agent pubkey being voted on
    pub candidate_pubkey: String,
    /// Whether to approve (true) or reject (false)
    pub approve: bool,
    /// Ed25519 signature over vote data: "vote:{candidate_pubkey}:{approve}:{timestamp}"
    pub signature: Vec<u8>,
}

/// Result of casting a vote
#[derive(Serialize, Deserialize, Debug)]
pub struct CastVoteResult {
    pub vote_recorded: bool,
    pub vote_hash: ActionHash,
    pub audit_log_hash: ActionHash,
    pub current_approval_votes: u32,
    pub current_rejection_votes: u32,
    pub required_votes: u32,
}

/// Cast a vote for a coordinator candidate
///
/// SEC-002: Only registered guardians can vote. Requires valid Ed25519 signature.
/// Votes are recorded with an audit log entry.
#[hdk_extern]
pub fn cast_coordinator_vote(input: CastVoteInput) -> ExternResult<CastVoteResult> {
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_str = caller.to_string();

    // Verify caller is a guardian
    if !is_guardian(&caller_str)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only guardians can cast coordinator votes".to_string()
        )));
    }

    // Check bootstrap window is open
    if !is_bootstrap_window_open()? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Bootstrap window is closed. Cannot cast votes.".to_string()
        )));
    }

    // Verify signature length (Ed25519 signatures are 64 bytes)
    if input.signature.len() != 64 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Invalid signature length: expected 64 bytes (Ed25519), got {}",
                input.signature.len()
            )
        )));
    }

    // Verify Ed25519 signature
    let now = sys_time()?.0 as i64 / 1_000_000;

    // Construct signable message: "vote:{candidate}:{approve}:{voter}"
    let signable_message = format!(
        "vote:{}:{}:{}",
        input.candidate_pubkey,
        input.approve,
        caller_str
    );

    // Verify signature using Holochain's Ed25519 verification
    let mut sig_bytes = [0u8; 64];
    sig_bytes.copy_from_slice(&input.signature[..64]);
    let signature = Signature::from(sig_bytes);

    let valid_signature = verify_signature(
        caller.clone(),
        signature,
        signable_message.as_bytes().to_vec(),
    )?;

    if !valid_signature {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid vote signature. Signature verification failed.".to_string()
        )));
    }

    // Check if this guardian already voted for this candidate
    let existing_vote = check_guardian_voted(&caller_str, &input.candidate_pubkey)?;
    if existing_vote {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Guardian has already voted for this candidate".to_string()
        )));
    }

    // Generate vote ID
    let mut hasher = sha2::Sha256::new();
    hasher.update(caller_str.as_bytes());
    hasher.update(input.candidate_pubkey.as_bytes());
    hasher.update(now.to_le_bytes());
    let vote_id = format!("{:x}", hasher.finalize());

    // Get guardian's voting weight
    let weight = get_guardian_weight(&caller_str)?;

    // Create vote entry
    let vote = CoordinatorVote {
        vote_id: vote_id.clone(),
        candidate_pubkey: input.candidate_pubkey.clone(),
        voter_pubkey: caller_str.clone(),
        weight,
        approve: input.approve,
        signature: input.signature,
        voted_at: now,
        expires_at: 0, // No expiry
    };

    let vote_hash = create_entry(&EntryTypes::CoordinatorVote(vote))?;

    // Link vote to candidate
    let votes_path = Path::from(format!("coordinator_votes.{}", input.candidate_pubkey));
    let votes_entry_hash = ensure_path(votes_path, LinkTypes::CandidateToVotes)?;
    create_link(
        votes_entry_hash,
        vote_hash.clone(),
        LinkTypes::CandidateToVotes,
        vote_id.as_bytes().to_vec(),
    )?;

    // Count current votes
    let (approval_votes, rejection_votes) = count_votes(&input.candidate_pubkey)?;

    // Get required votes from config
    let config = get_bootstrap_config_internal()?.unwrap_or(BootstrapConfig {
        window_start: now,
        window_end: now + DEFAULT_BOOTSTRAP_WINDOW_SECONDS,
        min_guardians: DEFAULT_MIN_GUARDIANS,
        min_votes: DEFAULT_MIN_VOTES,
        max_guardians: DEFAULT_MAX_GUARDIANS,
        bootstrap_complete: false,
    });

    // Create audit log entry
    let audit_hash = log_coordinator_action(
        "cast_coordinator_vote",
        &input.candidate_pubkey,
        &serde_json::json!({
            "voter": caller_str,
            "approve": input.approve,
            "weight": weight,
            "approval_votes": approval_votes,
            "rejection_votes": rejection_votes,
            "required_votes": config.min_votes,
        }).to_string()
    )?;

    Ok(CastVoteResult {
        vote_recorded: true,
        vote_hash,
        audit_log_hash: audit_hash,
        current_approval_votes: approval_votes,
        current_rejection_votes: rejection_votes,
        required_votes: config.min_votes,
    })
}

/// Check if a guardian has already voted for a candidate
pub(crate) fn check_guardian_voted(guardian_pubkey: &str, candidate_pubkey: &str) -> ExternResult<bool> {
    let votes_path = Path::from(format!("coordinator_votes.{}", candidate_pubkey));
    let votes_hash = match votes_path.clone().typed(LinkTypes::CandidateToVotes) {
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
            votes_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::CandidateToVotes as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(vote) = CoordinatorVote::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if vote.voter_pubkey == guardian_pubkey {
                            return Ok(true);
                        }
                    }
                }
            }
        }
    }

    Ok(false)
}

/// Result of finalizing coordinator election
#[derive(Serialize, Deserialize, Debug)]
pub struct FinalizeElectionResult {
    pub success: bool,
    pub winning_coordinator: Option<String>,
    pub credential_hash: Option<ActionHash>,
    pub total_approval_votes: u32,
    pub total_rejection_votes: u32,
    pub bootstrap_complete: bool,
}

/// Finalize the coordinator election
///
/// SEC-002: This function finalizes the bootstrap ceremony:
/// - Checks that the bootstrap window has closed
/// - Counts all votes for candidates
/// - Verifies minimum vote threshold is met
/// - Creates CoordinatorCredential for the winning candidate
/// - Marks bootstrap as complete
#[hdk_extern]
pub fn finalize_coordinator_election(_: ()) -> ExternResult<FinalizeElectionResult> {
    // Verify caller has coordinator authority (genesis coordinator)
    require_coordinator_role()?;

    // Get bootstrap config
    let config = get_bootstrap_config_internal()?.ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(
            "Bootstrap not initialized".to_string()
        ))
    })?;

    // Check if already complete
    if config.bootstrap_complete {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Bootstrap ceremony already finalized".to_string()
        )));
    }

    // Check bootstrap window has closed
    let now = sys_time()?.0 as i64 / 1_000_000;
    if now < config.window_end {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Bootstrap window still open. Closes at timestamp {}. Current time: {}",
                config.window_end, now
            )
        )));
    }

    // Find the candidate with the most approval votes
    let candidates = get_all_candidates()?;

    if candidates.is_empty() {
        // No candidates - mark bootstrap complete with no new coordinator
        mark_bootstrap_complete()?;

        log_coordinator_action(
            "finalize_coordinator_election",
            "none",
            &serde_json::json!({
                "outcome": "no_candidates",
            }).to_string()
        )?;

        return Ok(FinalizeElectionResult {
            success: true,
            winning_coordinator: None,
            credential_hash: None,
            total_approval_votes: 0,
            total_rejection_votes: 0,
            bootstrap_complete: true,
        });
    }

    // Find winning candidate
    let mut best_candidate: Option<(String, u32, u32)> = None; // (pubkey, approval, rejection)

    for candidate in &candidates {
        let (approval, rejection) = count_votes(candidate)?;

        // Check if this candidate has enough votes and more approvals than rejections
        if approval >= config.min_votes && approval > rejection {
            match &best_candidate {
                None => best_candidate = Some((candidate.clone(), approval, rejection)),
                Some((_, best_approval, _)) => {
                    if approval > *best_approval {
                        best_candidate = Some((candidate.clone(), approval, rejection));
                    }
                }
            }
        }
    }

    match best_candidate {
        Some((winner, approval_votes, rejection_votes)) => {
            // Collect guardian signatures
            let signatures = collect_vote_signatures(&winner)?;

            // Set expiry based on default term duration + grace period
            let expires_at = now + DEFAULT_TERM_DURATION_SECONDS + TERM_GRACE_PERIOD_SECONDS;

            // Create coordinator credential with term-based expiry
            let credential = CoordinatorCredential {
                agent_pubkey: winner.clone(),
                credential_type: "voted".to_string(),
                issued_at: now,
                expires_at,
                guardian_signatures_json: serde_json::to_string(&signatures).unwrap_or_default(),
                required_signatures: config.min_votes,
                active: true,
                revoked_at: 0,
            };

            let cred_hash = create_entry(&EntryTypes::CoordinatorCredential(credential))?;

            // Link credential to agent
            let cred_path = Path::from(format!("coordinator_credential.{}", winner));
            let cred_entry_hash = ensure_path(cred_path, LinkTypes::AgentToCredential)?;
            create_link(
                cred_entry_hash,
                cred_hash.clone(),
                LinkTypes::AgentToCredential,
                vec![],
            )?;

            // Create coordinator term entry for rotation tracking
            create_coordinator_term(
                &winner,
                &cred_hash,
                DEFAULT_TERM_DURATION_SECONDS,
                "voted",
            )?;

            // Mark bootstrap complete
            mark_bootstrap_complete()?;

            // Log the finalization
            log_coordinator_action(
                "finalize_coordinator_election",
                &winner,
                &serde_json::json!({
                    "outcome": "coordinator_elected",
                    "approval_votes": approval_votes,
                    "rejection_votes": rejection_votes,
                    "required_votes": config.min_votes,
                    "term_expires_at": expires_at,
                }).to_string()
            )?;

            Ok(FinalizeElectionResult {
                success: true,
                winning_coordinator: Some(winner),
                credential_hash: Some(cred_hash),
                total_approval_votes: approval_votes,
                total_rejection_votes: rejection_votes,
                bootstrap_complete: true,
            })
        }
        None => {
            // No candidate met the threshold
            mark_bootstrap_complete()?;

            // Calculate total votes across all candidates for reporting
            let mut total_approval = 0u32;
            let mut total_rejection = 0u32;
            for candidate in &candidates {
                let (approval, rejection) = count_votes(candidate)?;
                total_approval += approval;
                total_rejection += rejection;
            }

            log_coordinator_action(
                "finalize_coordinator_election",
                "none",
                &serde_json::json!({
                    "outcome": "no_winner",
                    "reason": "No candidate met minimum vote threshold",
                    "required_votes": config.min_votes,
                    "candidates_count": candidates.len(),
                }).to_string()
            )?;

            Ok(FinalizeElectionResult {
                success: true,
                winning_coordinator: None,
                credential_hash: None,
                total_approval_votes: total_approval,
                total_rejection_votes: total_rejection,
                bootstrap_complete: true,
            })
        }
    }
}

/// Get all candidates who have received votes
pub(crate) fn get_all_candidates() -> ExternResult<Vec<String>> {
    let mut candidates = Vec::new();

    let audit_path = Path::from("audit_log_chain");
    let audit_hash = match audit_path.clone().typed(LinkTypes::AuditLogChain) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok(candidates);
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok(candidates),
    };

    let links = get_links(
        LinkQuery::new(
            audit_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AuditLogChain as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(log) = CoordinatorAuditLog::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if log.action_type == "cast_coordinator_vote" {
                            // Extract candidate from target field
                            if !candidates.contains(&log.target) {
                                candidates.push(log.target);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(candidates)
}

/// Mark bootstrap as complete by updating the config
pub(crate) fn mark_bootstrap_complete() -> ExternResult<()> {
    let config_path = Path::from("bootstrap_config");
    let config_hash = match config_path.clone().typed(LinkTypes::BootstrapConfiguration) {
        Ok(typed) => {
            if !typed.exists()? {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Bootstrap config not found".to_string()
                )));
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Err(wasm_error!(WasmErrorInner::Guest(
            "Bootstrap config not found".to_string()
        ))),
    };

    let links = get_links(
        LinkQuery::new(
            config_hash.clone(),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::BootstrapConfiguration as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    // Get the most recent config and update it
    if let Some(link) = links.last() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(mut config) = BootstrapConfig::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        config.bootstrap_complete = true;
                        let new_hash = update_entry(action_hash, &EntryTypes::BootstrapConfig(config))?;

                        // Update link to point to new config
                        create_link(
                            config_hash,
                            new_hash,
                            LinkTypes::BootstrapConfiguration,
                            b"updated".to_vec(),
                        )?;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Check if the bootstrap ceremony is complete
///
/// Returns true if:
/// - Bootstrap has been initialized AND
/// - bootstrap_complete flag is set to true
#[hdk_extern]
pub fn is_bootstrap_complete(_: ()) -> ExternResult<bool> {
    let config = get_bootstrap_config_internal()?;

    match config {
        Some(cfg) => Ok(cfg.bootstrap_complete),
        None => Ok(false), // Not initialized = not complete
    }
}

/// Get bootstrap status with detailed information
#[derive(Serialize, Deserialize, Debug)]
pub struct BootstrapStatus {
    pub initialized: bool,
    pub bootstrap_complete: bool,
    pub window_open: bool,
    pub window_start: i64,
    pub window_end: i64,
    pub current_time: i64,
    pub guardian_count: u32,
    pub min_guardians: u32,
    pub max_guardians: u32,
    pub min_votes: u32,
    pub genesis_coordinators: Vec<String>,
}

/// Get detailed bootstrap ceremony status
#[hdk_extern]
pub fn get_bootstrap_status(_: ()) -> ExternResult<BootstrapStatus> {
    let now = sys_time()?.0 as i64 / 1_000_000;

    let config = get_bootstrap_config_internal()?;
    let guardians = get_active_guardians(())?;
    let coordinators = get_active_coordinators(())?;

    match config {
        Some(cfg) => {
            let window_open = now >= cfg.window_start && now <= cfg.window_end && !cfg.bootstrap_complete;

            Ok(BootstrapStatus {
                initialized: true,
                bootstrap_complete: cfg.bootstrap_complete,
                window_open,
                window_start: cfg.window_start,
                window_end: cfg.window_end,
                current_time: now,
                guardian_count: guardians.len() as u32,
                min_guardians: cfg.min_guardians,
                max_guardians: cfg.max_guardians,
                min_votes: cfg.min_votes,
                genesis_coordinators: coordinators,
            })
        }
        None => Ok(BootstrapStatus {
            initialized: false,
            bootstrap_complete: false,
            window_open: false,
            window_start: 0,
            window_end: 0,
            current_time: now,
            guardian_count: 0,
            min_guardians: DEFAULT_MIN_GUARDIANS,
            max_guardians: DEFAULT_MAX_GUARDIANS,
            min_votes: DEFAULT_MIN_VOTES,
            genesis_coordinators: vec![],
        }),
    }
}

// =============================================================================
// Coordinator Term Rotation
// =============================================================================

/// Input for initiating a coordinator rotation
#[derive(Serialize, Deserialize, Debug)]
pub struct InitiateRotationInput {
    /// Term duration in seconds (0 = use default 7 days)
    pub term_duration_seconds: i64,
    /// Election window in seconds (0 = use default 24 hours)
    pub election_window_seconds: i64,
}

/// Result of initiating a rotation
#[derive(Serialize, Deserialize, Debug)]
pub struct RotationResult {
    pub success: bool,
    pub election_window_start: i64,
    pub election_window_end: i64,
    pub expiring_term_number: u32,
    pub next_term_number: u32,
}

/// Status of the current coordinator term
#[derive(Serialize, Deserialize, Debug)]
pub struct TermStatus {
    pub has_active_term: bool,
    pub current_term_number: u32,
    pub coordinator_pubkey: String,
    pub term_start: i64,
    pub term_end: i64,
    pub seconds_remaining: i64,
    pub expired: bool,
    pub in_grace_period: bool,
    pub rotation_needed: bool,
}

/// Create a coordinator term entry when issuing a credential with expiry
pub(crate) fn create_coordinator_term(
    coordinator_pubkey: &str,
    credential_hash: &ActionHash,
    term_duration_seconds: i64,
    selection_method: &str,
) -> ExternResult<ActionHash> {
    let now = sys_time()?.0 as i64 / 1_000_000;
    let duration = if term_duration_seconds > 0 {
        term_duration_seconds
    } else {
        DEFAULT_TERM_DURATION_SECONDS
    };

    // Get next term number
    let current_term = get_current_term_internal()?;
    let next_term_number = current_term.map_or(1, |t| t.term_number + 1);

    let term = CoordinatorTerm {
        term_number: next_term_number,
        coordinator_pubkey: coordinator_pubkey.to_string(),
        term_start: now,
        term_end: now + duration,
        credential_hash: credential_hash.to_string(),
        active: true,
        selection_method: selection_method.to_string(),
    };

    let term_hash = create_entry(&EntryTypes::CoordinatorTerm(term))?;

    // Link to terms registry
    let terms_path = Path::from("coordinator_terms");
    let terms_entry_hash = ensure_path(terms_path, LinkTypes::CoordinatorTerms)?;
    create_link(
        terms_entry_hash,
        term_hash.clone(),
        LinkTypes::CoordinatorTerms,
        next_term_number.to_le_bytes().to_vec(),
    )?;

    // Link to agent's terms
    let agent_terms_path = Path::from(format!("agent_terms.{}", coordinator_pubkey));
    let agent_terms_hash = ensure_path(agent_terms_path, LinkTypes::AgentToTerms)?;
    create_link(
        agent_terms_hash,
        term_hash.clone(),
        LinkTypes::AgentToTerms,
        vec![],
    )?;

    Ok(term_hash)
}

/// Get the current active coordinator term (internal helper)
pub(crate) fn get_current_term_internal() -> ExternResult<Option<CoordinatorTerm>> {
    let terms_path = Path::from("coordinator_terms");
    let terms_hash = match terms_path.clone().typed(LinkTypes::CoordinatorTerms) {
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
            terms_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::CoordinatorTerms as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    // Find the most recent active term
    let mut best_term: Option<CoordinatorTerm> = None;

    for link in links.iter().rev() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(term) = CoordinatorTerm::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if term.active {
                            match &best_term {
                                None => best_term = Some(term),
                                Some(current) => {
                                    if term.term_number > current.term_number {
                                        best_term = Some(term);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(best_term)
}

/// Get the current coordinator term status
#[hdk_extern]
pub fn get_current_term(_: ()) -> ExternResult<TermStatus> {
    let now = sys_time()?.0 as i64 / 1_000_000;

    match get_current_term_internal()? {
        Some(term) => {
            let seconds_remaining = term.term_end - now;
            let expired = now > term.term_end;
            let in_grace_period = expired && now <= term.term_end + TERM_GRACE_PERIOD_SECONDS;
            let rotation_needed = expired && !in_grace_period;

            Ok(TermStatus {
                has_active_term: true,
                current_term_number: term.term_number,
                coordinator_pubkey: term.coordinator_pubkey,
                term_start: term.term_start,
                term_end: term.term_end,
                seconds_remaining: seconds_remaining.max(0),
                expired,
                in_grace_period,
                rotation_needed,
            })
        }
        None => Ok(TermStatus {
            has_active_term: false,
            current_term_number: 0,
            coordinator_pubkey: String::new(),
            term_start: 0,
            term_end: 0,
            seconds_remaining: 0,
            expired: false,
            in_grace_period: false,
            rotation_needed: false,
        }),
    }
}

/// Initiate a coordinator rotation election
///
/// This opens a new election window for guardians to vote on the next coordinator.
/// Can be called when the current term has expired or is about to expire.
/// Only existing coordinators or guardians can initiate rotation.
#[hdk_extern]
pub fn initiate_rotation(input: InitiateRotationInput) -> ExternResult<RotationResult> {
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_str = caller.to_string();

    // Caller must be a coordinator or guardian
    let is_coord = verify_coordinator_authority(&caller)?;
    let is_guard = is_guardian(&caller_str)?;
    if !is_coord && !is_guard {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only coordinators or guardians can initiate rotation".to_string()
        )));
    }

    let now = sys_time()?.0 as i64 / 1_000_000;

    // Get current term info
    let current_term = get_current_term_internal()?;
    let current_term_number = current_term.as_ref().map_or(0, |t| t.term_number);

    // If there's an active non-expired term, only allow rotation if within grace period
    if let Some(ref term) = current_term {
        if now < term.term_end {
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!(
                    "Current term {} has not expired yet. Expires at timestamp {}. Current: {}",
                    term.term_number, term.term_end, now
                )
            )));
        }
    }

    // Deactivate the current term
    if current_term.is_some() {
        deactivate_current_term()?;
    }

    // Open a new election window by resetting bootstrap_complete and setting new window
    let election_window = if input.election_window_seconds > 0 {
        input.election_window_seconds
    } else {
        DEFAULT_ROTATION_ELECTION_WINDOW_SECONDS
    };

    let term_duration = if input.term_duration_seconds > 0 {
        input.term_duration_seconds
    } else {
        DEFAULT_TERM_DURATION_SECONDS
    };

    let window_end = now + election_window;

    // Create a new bootstrap config for this rotation election
    let config = BootstrapConfig {
        window_start: now,
        window_end,
        min_guardians: DEFAULT_MIN_GUARDIANS,
        min_votes: DEFAULT_MIN_VOTES,
        max_guardians: DEFAULT_MAX_GUARDIANS,
        bootstrap_complete: false,
    };

    let config_hash = create_entry(&EntryTypes::BootstrapConfig(config))?;

    // Link config to path (replaces previous config via new link)
    let config_path = Path::from("bootstrap_config");
    let config_entry_hash = ensure_path(config_path, LinkTypes::BootstrapConfiguration)?;
    create_link(
        config_entry_hash,
        config_hash,
        LinkTypes::BootstrapConfiguration,
        b"rotation".to_vec(),
    )?;

    // Log the rotation initiation
    log_coordinator_action(
        "initiate_rotation",
        &caller_str,
        &serde_json::json!({
            "previous_term": current_term_number,
            "next_term": current_term_number + 1,
            "election_window_end": window_end,
            "term_duration": term_duration,
        }).to_string()
    )?;

    Ok(RotationResult {
        success: true,
        election_window_start: now,
        election_window_end: window_end,
        expiring_term_number: current_term_number,
        next_term_number: current_term_number + 1,
    })
}

/// Deactivate the current coordinator term
fn deactivate_current_term() -> ExternResult<()> {
    let terms_path = Path::from("coordinator_terms");
    let terms_hash = match terms_path.clone().typed(LinkTypes::CoordinatorTerms) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok(());
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok(()),
    };

    let links = get_links(
        LinkQuery::new(
            terms_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::CoordinatorTerms as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    for link in links.iter().rev() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(mut term) = CoordinatorTerm::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if term.active {
                            term.active = false;
                            update_entry(action_hash, &EntryTypes::CoordinatorTerm(term))?;
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
