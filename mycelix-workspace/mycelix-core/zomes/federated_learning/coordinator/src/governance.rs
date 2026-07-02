// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Byzantine detection consensus and governance.
//!
//! Validators submit Byzantine detection votes, and RB-BFT consensus
//! tallies votes weighted by reputation^2.

use hdk::prelude::*;
use federated_learning_integrity::*;

use crate::bridge::call_bridge_record_reputation_sync;
use crate::consensus::get_active_validators_internal;
use super::ensure_path;

/// Input for submitting a Byzantine detection vote
#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitByzantineVoteInput {
    /// FL round number
    pub round: u64,
    /// Flagged nodes with confidence (JSON: [[node_id, confidence], ...])
    pub flagged_nodes_json: String,
    /// Detection layers used (JSON: ["L1", "L2", ...])
    pub detection_layers_json: String,
    /// SHA-256 hash of detection evidence
    pub evidence_hash: String,
}

/// Submit a Byzantine detection vote
///
/// Each validator runs the 9-layer defense stack independently and submits
/// a vote listing nodes they flagged. RB-BFT consensus tallies votes
/// weighted by reputation^2.
#[hdk_extern]
pub fn submit_byzantine_vote(input: SubmitByzantineVoteInput) -> ExternResult<ActionHash> {
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
                "Must be a registered validator to submit Byzantine votes".to_string()
            )));
        }
    };

    let vote = ByzantineVote {
        round: input.round,
        voter: agent_str,
        voter_reputation: trust_score,
        voter_trust_score: trust_score,
        flagged_nodes_json: input.flagged_nodes_json,
        detection_layers_json: input.detection_layers_json,
        evidence_hash: input.evidence_hash,
        voted_at: now,
    };

    let action_hash = create_entry(&EntryTypes::ByzantineVote(vote))?;

    // Link to round
    let round_path = Path::from(format!("round_byzantine_votes.{}", input.round));
    let round_hash = ensure_path(round_path, LinkTypes::RoundToByzantineVotes)?;
    create_link(
        round_hash,
        action_hash.clone(),
        LinkTypes::RoundToByzantineVotes,
        vec![],
    )?;

    Ok(action_hash)
}

/// Get Byzantine votes for a round
#[hdk_extern]
pub fn get_byzantine_votes(round: u64) -> ExternResult<Vec<ByzantineVote>> {
    let round_path = Path::from(format!("round_byzantine_votes.{}", round));
    let round_hash = match round_path.clone().typed(LinkTypes::RoundToByzantineVotes) {
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
            LinkTypeFilter::single_type(0.into(), (LinkTypes::RoundToByzantineVotes as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut votes = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(vote) = ByzantineVote::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        votes.push(vote);
                    }
                }
            }
        }
    }

    Ok(votes)
}

/// Byzantine conviction result
#[derive(Serialize, Deserialize, Debug)]
pub struct ByzantineConvictionResult {
    /// Nodes convicted (>2/3 weighted agreement)
    pub convicted_nodes: Vec<String>,
    /// Conviction confidence per node
    pub confidence_scores: Vec<(String, f64)>,
    /// Whether quorum was reached
    pub quorum_reached: bool,
    /// Total voter weight
    pub total_weight: f64,
}

/// Tally Byzantine votes for a round using reputation^2 weighted consensus
///
/// Groups votes by flagged node, computes weighted agreement,
/// and returns nodes convicted by >2/3 weighted agreement.
#[hdk_extern]
pub fn tally_byzantine_votes(round: u64) -> ExternResult<ByzantineConvictionResult> {
    let votes = get_byzantine_votes(round)?;

    if votes.is_empty() {
        return Ok(ByzantineConvictionResult {
            convicted_nodes: vec![],
            confidence_scores: vec![],
            quorum_reached: false,
            total_weight: 0.0,
        });
    }

    // Compute total voter weight (reputation^2)
    let total_weight: f64 = votes.iter()
        .map(|v| (v.voter_trust_score as f64).powi(2))
        .sum();

    // Parse flagged nodes and aggregate by node
    let mut node_flags: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    for vote in &votes {
        let voter_weight = (vote.voter_trust_score as f64).powi(2);

        // Parse flagged nodes JSON: [[node_id, confidence], ...]
        if let Ok(flagged) = serde_json::from_str::<Vec<(String, f64)>>(&vote.flagged_nodes_json) {
            for (node_id, confidence) in flagged {
                let weighted_flag = voter_weight * confidence;
                *node_flags.entry(node_id).or_insert(0.0) += weighted_flag;
            }
        }
    }

    // Convict nodes with >2/3 weighted agreement
    let quorum = total_weight * federated_learning_integrity::CONSENSUS_QUORUM_THRESHOLD;
    let mut convicted_nodes = Vec::new();
    let mut confidence_scores = Vec::new();

    for (node_id, weighted_flags) in &node_flags {
        let conviction_ratio = weighted_flags / total_weight;
        confidence_scores.push((node_id.clone(), conviction_ratio));

        if *weighted_flags >= quorum {
            convicted_nodes.push(node_id.clone());

            // Apply reputation penalty via Bridge
            let _ = call_bridge_record_reputation_sync(
                node_id,
                -0.2, // Reputation penalty for Byzantine conviction
                0,    // positive_interactions
                1,    // negative_interactions
                Some(format!("Byzantine conviction in round {} (confidence: {:.2})", round, conviction_ratio)),
            );
        }
    }

    Ok(ByzantineConvictionResult {
        convicted_nodes,
        confidence_scores,
        quorum_reached: !votes.is_empty(),
        total_weight,
    })
}

/// Summary of Byzantine consensus for a round
#[derive(Serialize, Deserialize, Debug)]
pub struct ByzantineConsensusSummary {
    /// FL round number
    pub round: u64,
    /// Nodes convicted by >2/3 weighted agreement
    pub convicted_nodes: Vec<String>,
    /// Per-node conviction ratios (node_id, weighted_ratio)
    pub conviction_ratios: Vec<(String, f64)>,
    /// Total number of Byzantine votes cast
    pub total_votes: usize,
    /// Total reputation^2 weight of all voters
    pub total_weight: f64,
    /// Whether quorum was reached (at least one vote exists)
    pub quorum_reached: bool,
}

/// Get Byzantine consensus summary for a round (read-only query)
///
/// Fetches all ByzantineVote entries for the given round via path links,
/// tallies votes weighted by voter_trust_score^2, and returns which nodes
/// were convicted (>2/3 weighted agreement), total votes, and total weight.
#[hdk_extern]
pub fn get_byzantine_consensus(round: u64) -> ExternResult<ByzantineConsensusSummary> {
    let votes = get_byzantine_votes(round)?;

    if votes.is_empty() {
        return Ok(ByzantineConsensusSummary {
            round,
            convicted_nodes: vec![],
            conviction_ratios: vec![],
            total_votes: 0,
            total_weight: 0.0,
            quorum_reached: false,
        });
    }

    // Compute total voter weight (reputation^2)
    let total_weight: f64 = votes.iter()
        .map(|v| (v.voter_trust_score as f64).powi(2))
        .sum();

    // Parse flagged nodes and aggregate weighted flags by node
    let mut node_flags: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    for vote in &votes {
        let voter_weight = (vote.voter_trust_score as f64).powi(2);

        // Parse flagged nodes JSON: [[node_id, confidence], ...]
        if let Ok(flagged) = serde_json::from_str::<Vec<(String, f64)>>(&vote.flagged_nodes_json) {
            for (node_id, confidence) in flagged {
                let weighted_flag = voter_weight * confidence;
                *node_flags.entry(node_id).or_insert(0.0) += weighted_flag;
            }
        }
    }

    // Convict nodes with >2/3 weighted agreement
    let quorum = total_weight * federated_learning_integrity::CONSENSUS_QUORUM_THRESHOLD;
    let mut convicted_nodes = Vec::new();
    let mut conviction_ratios = Vec::new();

    for (node_id, weighted_flags) in &node_flags {
        let conviction_ratio = weighted_flags / total_weight;
        conviction_ratios.push((node_id.clone(), conviction_ratio));

        if *weighted_flags >= quorum {
            convicted_nodes.push(node_id.clone());
        }
    }

    Ok(ByzantineConsensusSummary {
        round,
        convicted_nodes,
        conviction_ratios,
        total_votes: votes.len(),
        total_weight,
        quorum_reached: !votes.is_empty(),
    })
}
