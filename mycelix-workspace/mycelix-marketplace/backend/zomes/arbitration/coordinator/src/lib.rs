// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use arbitration_integrity::*;
use hdk::prelude::*;
use mycelix_common::{error_handling, link_queries, remote_calls, time};

/// File a dispute for a transaction
///
/// This creates a dispute entry and assigns arbitrators with high MATL scores.
/// Only buyer or seller can file.
#[hdk_extern]
pub fn file_dispute(input: FileDisputeInput) -> ExternResult<DisputeOutput> {
    let agent_info = agent_info()?;
    let filer = agent_info.agent_initial_pubkey.clone();

    // Get transaction details (from transactions zome)
    // Use shared utility for remote calls
    let transaction: TransactionInfo = remote_calls::call_zome(
        "transactions",
        "get_transaction",
        input.transaction_hash.clone(),
    )?;

    // Verify filer is buyer or seller
    if filer != transaction.buyer && filer != transaction.seller {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only buyer or seller can file dispute".into()
        )));
    }

    // Create dispute entry
    let dispute = Dispute {
        transaction_hash: input.transaction_hash.clone(),
        filed_by: filer,
        buyer: transaction.buyer.clone(),
        seller: transaction.seller.clone(),
        reason: input.reason,
        evidence_cids: input.evidence_cids,
        status: DisputeStatus::Filed,
        arbitrators: Vec::new(), // Will be assigned in next step
        created_at: time::now()?,
        updated_at: time::now()?,
    };

    let action_hash = create_entry(&EntryTypes::Dispute(dispute.clone()))?;

    // Create links
    create_link(
        input.transaction_hash.clone(),
        action_hash.clone(),
        LinkTypes::TransactionToDispute,
        (),
    )?;

    create_link(
        dispute.filed_by.clone(),
        action_hash.clone(),
        LinkTypes::AgentToFiledDisputes,
        (),
    )?;

    create_link(
        action_hash.clone(),
        action_hash.clone(),
        LinkTypes::AllDisputes,
        (),
    )?;

    // Assign arbitrators with high MATL scores
    let dispute_with_arbitrators = assign_arbitrators_internal(action_hash.clone(), dispute)?;

    // Emit monitoring metric
    monitoring::emit_metric(
        monitoring::MetricType::ArbitrationInitiated,
        1.0,
        Some(dispute_with_arbitrators.filed_by.clone()),
        Some(format!(
            "buyer:{:?},seller:{:?}",
            dispute_with_arbitrators.buyer, dispute_with_arbitrators.seller
        )),
    )?;

    Ok(DisputeOutput {
        dispute_hash: action_hash,
        dispute: dispute_with_arbitrators,
    })
}

/// Minimum MATL composite score required to be eligible for arbitrator
/// assignment. Matches the threshold documented on `submit_arbitration_vote`
/// and exercised by `test_arbitrator_selection_criteria`.
const ARBITRATOR_MIN_MATL_SCORE: f64 = 0.7;

/// Number of arbitrators assigned per dispute.
const ARBITRATOR_COUNT: usize = 3;

/// Register the calling agent as available for arbitrator assignment.
///
/// This is the arbitrator pool: agents opt in, and `assign_arbitrators_internal`
/// draws candidates from this pool (filtered by live MATL score and dispute-party
/// exclusion) instead of having no source of arbitrators at all.
#[hdk_extern]
pub fn register_as_arbitrator(_: ()) -> ExternResult<ActionHash> {
    let agent_info = agent_info()?;
    let agent = agent_info.agent_initial_pubkey;

    let registration = ArbitratorRegistration {
        agent: agent.clone(),
        registered_at: time::now()?,
    };

    let action_hash = create_entry(&EntryTypes::ArbitratorRegistration(registration))?;

    // Anchor link so the pool can be enumerated (mirrors the AllDisputes pattern
    // already used in this zome for browsing).
    create_link(
        arbitrator_pool_anchor()?,
        action_hash.clone(),
        LinkTypes::AllArbitrators,
        (),
    )?;

    Ok(action_hash)
}

/// Deterministic anchor hash for the arbitrator pool.
fn arbitrator_pool_anchor() -> ExternResult<EntryHash> {
    let path: Path = (&Anchor {
        anchor_type: "arbitrator_pool".to_string(),
        anchor_text: None,
    })
        .into();
    path.path_entry_hash()
}

/// Assign arbitrators to a dispute (internal helper)
///
/// Selects agents from the self-registered arbitrator pool (`register_as_arbitrator`)
/// with a live MATL score > `ARBITRATOR_MIN_MATL_SCORE`, excluding buyer, seller,
/// and filer. MATL scores are re-fetched live (not trusted from the registration
/// entry) since reputation drifts over time.
fn assign_arbitrators_internal(
    dispute_hash: ActionHash,
    mut dispute: Dispute,
) -> ExternResult<Dispute> {
    let registrations: Vec<ArbitratorRegistration> =
        link_queries::get_linked_entries(arbitrator_pool_anchor()?, LinkTypes::AllArbitrators)?;

    let mut eligible_arbitrators: Vec<AgentPubKey> = Vec::new();

    for registration in registrations {
        let candidate = registration.agent;

        // Exclude parties to the dispute
        if candidate == dispute.buyer
            || candidate == dispute.seller
            || candidate == dispute.filed_by
        {
            continue;
        }

        // Avoid duplicate assignment if an agent registered more than once
        if eligible_arbitrators.contains(&candidate) {
            continue;
        }

        // Live MATL score check (reputation zome is the source of truth).
        // get_agent_matl_score returns Option<MatlScore>; a new agent with no
        // score yet is treated as neutral (0.5), which is below the threshold
        // and therefore ineligible until they earn reputation.
        let matl_score: f64 =
            remote_calls::call_zome::<_, Option<reputation_integrity::MatlScore>>(
                "reputation",
                "get_agent_matl_score",
                candidate.clone(),
            )
            .ok()
            .flatten()
            .map(|score| score.composite)
            .unwrap_or(0.5);

        if matl_score > ARBITRATOR_MIN_MATL_SCORE {
            eligible_arbitrators.push(candidate);
        }

        if eligible_arbitrators.len() >= ARBITRATOR_COUNT {
            break;
        }
    }

    dispute.arbitrators = eligible_arbitrators.clone();
    dispute.status = DisputeStatus::UnderReview;
    dispute.updated_at = time::now()?;

    // Update the dispute entry
    update_entry(dispute_hash.clone(), &dispute)?;

    // Create links for arbitrators
    for arbitrator in eligible_arbitrators {
        create_link(
            arbitrator,
            dispute_hash.clone(),
            LinkTypes::AgentToArbitrationOpportunities,
            (),
        )?;
    }

    Ok(dispute)
}

/// Submit an arbitration vote
///
/// Arbitrators cast their vote with reasoning.
/// Vote is weighted by the arbitrator's MATL score.
#[hdk_extern]
pub fn submit_arbitration_vote(
    input: SubmitArbitrationVoteInput,
) -> ExternResult<ArbitrationVoteOutput> {
    let agent_info = agent_info()?;
    let arbitrator = agent_info.agent_initial_pubkey.clone();

    // Get dispute
    let dispute: Dispute = get_entry_from_hash(input.dispute_hash.clone())?;

    // Verify caller is an assigned arbitrator
    if !dispute.arbitrators.contains(&arbitrator) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Not an assigned arbitrator for this dispute".into()
        )));
    }

    // Get arbitrator's MATL score
    // Use shared utility for remote calls
    let matl_score: f64 =
        remote_calls::call_zome("reputation", "get_agent_matl_score", arbitrator.clone())?;

    // Create vote entry
    let vote = ArbitrationVote {
        dispute_hash: input.dispute_hash.clone(),
        arbitrator: arbitrator.clone(),
        favor_buyer: input.favor_buyer,
        reasoning: input.reasoning,
        arbitrator_matl_score: matl_score,
        voted_at: time::now()?,
    };

    let action_hash = create_entry(&EntryTypes::ArbitrationVote(vote.clone()))?;

    // Create link from dispute to vote
    create_link(
        input.dispute_hash.clone(),
        action_hash.clone(),
        LinkTypes::DisputeToVotes,
        (),
    )?;

    // Check if all arbitrators have voted
    let all_votes = get_dispute_votes(input.dispute_hash.clone())?;

    if all_votes.len() == dispute.arbitrators.len() {
        // All votes collected, update status to Voting
        let mut updated_dispute = dispute;
        updated_dispute.status = DisputeStatus::Voting;
        updated_dispute.updated_at = time::now()?;
        update_entry(input.dispute_hash, &updated_dispute)?;
    }

    Ok(ArbitrationVoteOutput {
        vote_hash: action_hash,
        vote,
    })
}

/// Finalize arbitration using MRC (Mutual Reputation Consensus)
///
/// This implements the weighted voting algorithm:
/// weighted_decision = Σ(vote * arbitrator_matl_score) / Σ(arbitrator_matl_scores)
///
/// If weighted_decision > 0.66 (66% threshold), buyer wins.
/// Otherwise, seller wins.
#[hdk_extern]
pub fn finalize_arbitration(dispute_hash: ActionHash) -> ExternResult<ArbitrationResultOutput> {
    // Get dispute
    let dispute: Dispute = get_entry_from_hash(dispute_hash.clone())?;

    // Verify status is Voting
    if dispute.status != DisputeStatus::Voting {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot finalize dispute with status {:?}",
            dispute.status
        ))));
    }

    // Get all votes
    let votes = get_dispute_votes(dispute_hash.clone())?;

    // Verify all arbitrators have voted
    if votes.len() != dispute.arbitrators.len() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Not all arbitrators have voted yet".into()
        )));
    }

    // Calculate weighted vote using MRC algorithm
    let (weighted_vote, _total_weight) = calculate_weighted_vote(&votes)?;

    // Determine winner (>0.66 = buyer wins)
    const THRESHOLD: f64 = 0.66;
    let buyer_wins = weighted_vote > THRESHOLD;

    let (winner, loser, status) = if buyer_wins {
        (
            dispute.buyer.clone(),
            dispute.seller.clone(),
            DisputeStatus::ResolvedBuyer,
        )
    } else {
        (
            dispute.seller.clone(),
            dispute.buyer.clone(),
            DisputeStatus::ResolvedSeller,
        )
    };

    // Get transaction value for compensation calculation
    let transaction: TransactionInfo = remote_calls::call_zome(
        "transactions",
        "get_transaction",
        dispute.transaction_hash.clone(),
    )?;

    // Calculate compensation based on transaction value and dispute outcome
    // Compensation formula:
    // - Base: transaction value
    // - Strong consensus (>0.85 or <0.15): full compensation
    // - Moderate consensus (0.66-0.85 or 0.15-0.34): 75% compensation
    // - Weak consensus (near threshold): 50% compensation
    let consensus_strength = if weighted_vote > 0.5 {
        weighted_vote
    } else {
        1.0 - weighted_vote
    };

    let compensation_multiplier = if consensus_strength >= 0.85 {
        1.0 // Strong consensus: full compensation
    } else if consensus_strength >= 0.75 {
        0.75 // Moderate consensus: 75% compensation
    } else {
        0.50 // Weak consensus: 50% compensation
    };

    let compensation_cents =
        Some((transaction.transaction_value_cents as f64 * compensation_multiplier) as u64);

    // Create result entry
    let result = ArbitrationResult {
        dispute_hash: dispute_hash.clone(),
        winner: winner.clone(),
        loser: loser.clone(),
        weighted_vote,
        total_votes: votes.len() as u32,
        compensation_cents,
        summary: format!(
            "Resolved in favor of {} with weighted vote of {:.2}. Compensation: {} cents ({}% based on consensus strength)",
            if buyer_wins { "buyer" } else { "seller" },
            weighted_vote,
            compensation_cents.unwrap_or(0),
            (compensation_multiplier * 100.0) as u32
        ),
        finalized_at: time::now()?,
    };

    let result_hash = create_entry(&EntryTypes::ArbitrationResult(result.clone()))?;

    // Create link from dispute to result
    create_link(
        dispute_hash.clone(),
        result_hash.clone(),
        LinkTypes::DisputeToResult,
        (),
    )?;

    // Update dispute status
    let mut updated_dispute = dispute;
    updated_dispute.status = status;
    updated_dispute.updated_at = time::now()?;
    update_entry(dispute_hash, &updated_dispute)?;

    // Update MATL scores based on outcome
    // Winner gets positive feedback, loser gets negative
    // Use shared utility for remote calls
    remote_calls::call_zome_void(
        "reputation",
        "update_matl_score",
        UpdateMatlInput {
            agent: loser,
            successful: false,
            transaction_value_cents: transaction.transaction_value_cents,
        },
    )?;

    Ok(ArbitrationResultOutput {
        result_hash,
        result,
    })
}

/// Get arbitration opportunities for the current user
///
/// Returns disputes where the user is an assigned arbitrator
/// and hasn't voted yet.
#[hdk_extern]
pub fn get_arbitration_opportunities(_: ()) -> ExternResult<DisputesResponse> {
    let agent_info = agent_info()?;
    let agent = agent_info.agent_initial_pubkey;

    // Use shared utility for get_links
    let links =
        link_queries::get_links_local(agent.clone(), LinkTypes::AgentToArbitrationOpportunities)?;

    let mut disputes = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            let dispute: Dispute = get_entry_from_hash(action_hash.clone())?;

            // Check if user has already voted
            let votes = get_dispute_votes(action_hash.clone())?;
            let already_voted = votes.iter().any(|v| v.vote.arbitrator == agent);

            if !already_voted {
                disputes.push(DisputeOutput {
                    dispute_hash: action_hash,
                    dispute,
                });
            }
        }
    }

    Ok(DisputesResponse { disputes })
}

/// Get a dispute by hash
#[hdk_extern]
pub fn get_dispute(dispute_hash: ActionHash) -> ExternResult<Option<DisputeOutput>> {
    match get(dispute_hash.clone(), GetOptions::default())? {
        Some(record) => {
            // Use shared utility for deserialization
            let dispute: Dispute = error_handling::deserialize_entry(&record)?;

            Ok(Some(DisputeOutput {
                dispute_hash,
                dispute,
            }))
        }
        None => Ok(None),
    }
}

// ===== Helper Functions =====

/// Calculate weighted vote using MRC algorithm
///
/// Formula: Σ(vote * matl_score) / Σ(matl_scores)
/// Returns (weighted_vote, total_weight)
fn calculate_weighted_vote(votes: &[ArbitrationVoteOutput]) -> ExternResult<(f64, f64)> {
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for vote_output in votes {
        let vote_value = if vote_output.vote.favor_buyer {
            1.0
        } else {
            0.0
        };

        weighted_sum += vote_value * vote_output.vote.arbitrator_matl_score;
        total_weight += vote_output.vote.arbitrator_matl_score;
    }

    if total_weight == 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "No valid votes (total weight is zero)".into()
        )));
    }

    let weighted_vote = weighted_sum / total_weight;

    Ok((weighted_vote, total_weight))
}

/// Get all votes for a dispute
fn get_dispute_votes(dispute_hash: ActionHash) -> ExternResult<Vec<ArbitrationVoteOutput>> {
    // Use shared utility for get_links
    let links = link_queries::get_links_local(dispute_hash, LinkTypes::DisputeToVotes)?;

    let mut votes = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                // Use shared utility for deserialization
                let vote: ArbitrationVote = error_handling::deserialize_entry(&record)?;

                votes.push(ArbitrationVoteOutput {
                    vote_hash: action_hash,
                    vote,
                });
            }
        }
    }

    Ok(votes)
}

/// Get entry from action hash (helper)
fn get_entry_from_hash<T: TryFrom<SerializedBytes, Error = SerializedBytesError>>(
    hash: ActionHash,
) -> ExternResult<T> {
    let record = get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Entry not found".into())))?;

    // Use shared utility for deserialization
    error_handling::deserialize_entry(&record)
}

// ===== Input/Output Types =====

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FileDisputeInput {
    pub transaction_hash: ActionHash,
    pub reason: String,
    pub evidence_cids: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DisputeOutput {
    pub dispute_hash: ActionHash,
    pub dispute: Dispute,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DisputesResponse {
    pub disputes: Vec<DisputeOutput>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubmitArbitrationVoteInput {
    pub dispute_hash: ActionHash,
    pub favor_buyer: bool,
    pub reasoning: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ArbitrationVoteOutput {
    pub vote_hash: ActionHash,
    pub vote: ArbitrationVote,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ArbitrationResultOutput {
    pub result_hash: ActionHash,
    pub result: ArbitrationResult,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TransactionInfo {
    pub buyer: AgentPubKey,
    pub seller: AgentPubKey,
    pub transaction_value_cents: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateMatlInput {
    pub agent: AgentPubKey,
    pub successful: bool,
    pub transaction_value_cents: u64,
}

// ===== Tests =====
#[cfg(test)]
mod tests;
