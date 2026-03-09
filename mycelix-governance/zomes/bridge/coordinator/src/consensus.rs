use super::*;

// =============================================================================
// WEIGHTED CONSENSUS COORDINATOR FUNCTIONS
// =============================================================================

/// Register as a consensus participant
///
/// Creates K-Vector and FederatedReputation entries for the calling agent,
/// establishing them as a participant in weighted consensus voting.
#[hdk_extern]
pub fn register_consensus_participant(_: ()) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Create initial K-Vector
    let k_vector = KVector::new_participant(agent_did.clone(), now);
    let k_vector_hash = create_entry(&EntryTypes::KVector(k_vector.clone()))?;

    // Link agent to K-Vector
    let agent_anchor = format!("agent:{}", agent_did);
    create_entry(&EntryTypes::Anchor(Anchor(agent_anchor.clone())))?;
    create_link(
        anchor_hash(&agent_anchor)?,
        k_vector_hash.clone(),
        LinkTypes::AgentToKVector,
        (),
    )?;

    // Create initial FederatedReputation
    let fed_rep = FederatedReputation::new_participant(agent_did.clone(), now);
    let fed_rep_hash = create_entry(&EntryTypes::FederatedReputation(fed_rep.clone()))?;

    // Link agent to FederatedReputation
    create_link(
        anchor_hash(&agent_anchor)?,
        fed_rep_hash.clone(),
        LinkTypes::AgentToFederatedRep,
        (),
    )?;

    // Get latest consciousness snapshot (if any) for initial phi
    let phi = match get_latest_agent_snapshot(&agent_did)? {
        Some((_, snapshot)) => snapshot.consciousness_level,
        None => 0.3, // Default for new participants
    };

    // Create ConsensusParticipant
    let participant = ConsensusParticipant {
        agent_did: agent_did.clone(),
        k_vector_id: format!("kvec:{}:{}", agent_did, now.as_micros()),
        federated_rep_id: Some(format!("fedrep:{}:{}", agent_did, now.as_micros())),
        reputation: k_vector.k_r,
        voting_weight: k_vector.voting_weight(),
        matl_score: 0.5, // Initial MATL score
        phi,
        federated_score: fed_rep.aggregated_score,
        is_active: true,
        rounds_participated: 0,
        successful_votes: 0,
        slashing_events: 0,
        last_slashing_at: None,
        streak_count: 0,
        registered_at: now,
        last_active_at: now,
    };

    let participant_hash = create_entry(&EntryTypes::ConsensusParticipant(participant))?;

    // Link agent to participant
    create_link(
        anchor_hash(&agent_anchor)?,
        participant_hash.clone(),
        LinkTypes::AgentToParticipant,
        (),
    )?;

    // Link to active participants
    create_entry(&EntryTypes::Anchor(Anchor(
        "active_participants".to_string(),
    )))?;
    create_link(
        anchor_hash("active_participants")?,
        participant_hash.clone(),
        LinkTypes::ActiveParticipants,
        (),
    )?;

    get(participant_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Participant not found".into()
    )))
}

/// Calculate holistic vote weight for a proposal
///
/// Computes the full holistic weight: Reputation² × (0.7 + 0.3 × Φ) × (1 + 0.2 × HarmonicAlignment)
#[hdk_extern]
pub fn calculate_holistic_vote_weight(
    input: CalculateWeightInput,
) -> ExternResult<HolisticVotingWeight> {
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Get participant's current state
    let participant = get_agent_participant(&agent_did)?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Not registered as participant".into())
    ))?;

    // Get latest consciousness Φ
    let phi = match get_latest_agent_snapshot(&agent_did)? {
        Some((_, snapshot)) => snapshot.consciousness_level,
        None => participant.phi,
    };

    // Calculate weight with harmonic alignment
    let weight = HolisticVotingWeight::calculate(
        participant.reputation,
        phi,
        input.harmonic_alignment.unwrap_or(0.0),
    );

    Ok(weight)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CalculateWeightInput {
    pub harmonic_alignment: Option<f64>,
}

/// Cast a weighted vote on a proposal
///
/// Creates a WeightedVote entry with full holistic weight calculation.
/// Verifies consciousness gate before allowing the vote.
#[hdk_extern]
pub fn cast_weighted_vote(input: CastWeightedVoteInput) -> ExternResult<WeightedVoteResult> {
    check_weighted_vote_input(&input).map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Get participant's current state
    let participant = get_agent_participant(&agent_did)?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Not registered as participant".into())
    ))?;

    // Check if participant can vote
    if !participant.can_vote() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Participant cannot vote (inactive or low reputation)".into()
        )));
    }

    // Get adaptive threshold for proposal type
    let threshold = AdaptiveThreshold::for_proposal_type(&input.proposal_type);

    // Get latest consciousness Φ
    let phi = match get_latest_agent_snapshot(&agent_did)? {
        Some((_, snapshot)) => snapshot.consciousness_level,
        None => participant.phi,
    };

    // Verify consciousness gate using dynamic (configurable) min voter consciousness
    let dynamic_min_consciousness = get_dynamic_min_voter_consciousness(&input.proposal_type)?;
    if phi < dynamic_min_consciousness {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Consciousness ({:.2}) below gate ({:.2}) for {:?} proposals",
            phi, dynamic_min_consciousness, input.proposal_type
        ))));
    }

    // Check cooldown
    if participant.in_cooldown(now) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Participant in cooldown period after slashing".into()
        )));
    }

    // Calculate holistic weight
    let holistic_weight = HolisticVotingWeight::calculate(
        participant.effective_reputation(now),
        phi,
        input.harmonic_alignment.unwrap_or(0.0),
    );

    // Create the vote
    let vote = WeightedVote {
        id: format!(
            "vote:{}:{}:{}",
            agent_did,
            input.proposal_id,
            now.as_micros()
        ),
        proposal_id: input.proposal_id.clone(),
        round: input.round,
        voter_did: agent_did.clone(),
        decision: input.decision,
        reputation: participant.effective_reputation(now),
        weight: holistic_weight.final_weight,
        phi,
        reason: input.reason,
        voted_at: now,
        signature: format!("sig:{}:{}", agent_did, now.as_micros()),
    };

    let vote_hash = create_entry(&EntryTypes::WeightedVote(vote.clone()))?;

    // Link vote to proposal round
    let round_anchor = format!("round:{}:{}", input.proposal_id, input.round);
    create_entry(&EntryTypes::Anchor(Anchor(round_anchor.clone())))?;
    create_link(
        anchor_hash(&round_anchor)?,
        vote_hash.clone(),
        LinkTypes::RoundToVotes,
        (),
    )?;

    // Link vote to agent
    let agent_anchor = format!("agent:{}", agent_did);
    create_link(
        anchor_hash(&agent_anchor)?,
        vote_hash,
        LinkTypes::AgentToVotes,
        (),
    )?;

    Ok(WeightedVoteResult {
        vote_id: vote.id,
        weight: holistic_weight.final_weight,
        weight_breakdown: holistic_weight.calculation_breakdown,
        decision: input.decision,
        phi_at_vote: phi,
        proposal_type: input.proposal_type,
        threshold_required: threshold.base_threshold,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CastWeightedVoteInput {
    pub proposal_id: String,
    pub proposal_type: ProposalType,
    pub round: u64,
    pub decision: VoteDecision,
    pub harmonic_alignment: Option<f64>,
    pub reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WeightedVoteResult {
    pub vote_id: String,
    pub weight: f64,
    pub weight_breakdown: String,
    pub decision: VoteDecision,
    pub phi_at_vote: f64,
    pub proposal_type: ProposalType,
    pub threshold_required: f64,
}

/// Get adaptive threshold for a proposal type
#[hdk_extern]
pub fn get_adaptive_threshold(proposal_type: ProposalType) -> ExternResult<AdaptiveThreshold> {
    Ok(AdaptiveThreshold::for_proposal_type(&proposal_type))
}

/// Get participant's current status including streak, cooldown, and effective reputation
#[hdk_extern]
pub fn get_participant_status(_: ()) -> ExternResult<ParticipantStatus> {
    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    let participant = get_agent_participant(&agent_did)?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Not registered as participant".into())
    ))?;

    // Get latest consciousness Φ
    let current_phi = match get_latest_agent_snapshot(&agent_did)? {
        Some((_, snapshot)) => snapshot.consciousness_level,
        None => participant.phi,
    };

    let effective_rep = participant.effective_reputation(now);
    let in_cooldown = participant.in_cooldown(now);
    let streak_bonus = participant.streak_bonus();

    // Check voting eligibility using dynamic (configurable) thresholds
    let consciousness_config = get_current_consciousness_config()?;
    let can_vote_standard = participant.can_vote()
        && current_phi >= consciousness_config.min_voter_consciousness_for(&ProposalType::Standard);

    let can_vote_emergency = participant.can_vote()
        && current_phi
            >= consciousness_config.min_voter_consciousness_for(&ProposalType::Emergency);

    let can_vote_constitutional = participant.can_vote()
        && current_phi
            >= consciousness_config.min_voter_consciousness_for(&ProposalType::Constitutional);

    Ok(ParticipantStatus {
        agent_did,
        is_active: participant.is_active,
        base_reputation: participant.reputation,
        effective_reputation: effective_rep,
        streak_count: participant.streak_count,
        streak_bonus,
        in_cooldown,
        current_phi,
        federated_score: participant.federated_score,
        rounds_participated: participant.rounds_participated,
        successful_votes: participant.successful_votes,
        success_rate: participant.success_rate(),
        slashing_events: participant.slashing_events,
        can_vote_standard,
        can_vote_emergency,
        can_vote_constitutional,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ParticipantStatus {
    pub agent_did: String,
    pub is_active: bool,
    pub base_reputation: f64,
    pub effective_reputation: f64,
    pub streak_count: u64,
    pub streak_bonus: f64,
    pub in_cooldown: bool,
    pub current_phi: f64,
    pub federated_score: f64,
    pub rounds_participated: u64,
    pub successful_votes: u64,
    pub success_rate: f64,
    pub slashing_events: u64,
    pub can_vote_standard: bool,
    pub can_vote_emergency: bool,
    pub can_vote_constitutional: bool,
}

/// Update federated reputation signals from external modules
///
/// ## Domain Boundary Enforcement
///
/// All input values are clamped to valid ranges [0.0, 1.0] for scores.
/// Per Commons Charter v1.0 Article II:
/// - Finance domain signals (stake, payments, escrow) can contribute at most 5% to aggregated score
/// - This is enforced in FederatedReputation.calculate_aggregated(), not here
/// - All normalized values are clamped to prevent manipulation
#[hdk_extern]
pub fn update_federated_reputation(input: UpdateFederatedReputationInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Get current federated reputation
    let (current_record, mut fed_rep) =
        get_agent_federated_reputation(&agent_did)?.ok_or(wasm_error!(WasmErrorInner::Guest(
            "No federated reputation found".into()
        )))?;

    // Update only the fields that are provided, with domain boundary clamping
    // All f64 scores are clamped to [0.0, 1.0] to prevent manipulation

    // Identity domain signals
    if let Some(v) = input.identity_verification {
        fed_rep.identity_verification = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.credential_count {
        fed_rep.credential_count = v.min(100); // Cap at 100 credentials
    }
    if let Some(v) = input.credential_quality {
        fed_rep.credential_quality = v.clamp(0.0, 1.0);
    }

    // Knowledge domain signals
    if let Some(v) = input.epistemic_contributions {
        fed_rep.epistemic_contributions = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.factcheck_accuracy {
        fed_rep.factcheck_accuracy = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.dark_spots_resolved {
        fed_rep.dark_spots_resolved = v.min(1000); // Cap at 1000
    }

    // Finance domain signals (these contribute max 5% to final score)
    if let Some(v) = input.stake_weight {
        fed_rep.stake_weight = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.payment_reliability {
        fed_rep.payment_reliability = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.escrow_completion_rate {
        fed_rep.escrow_completion_rate = v.clamp(0.0, 1.0);
    }

    // FL domain signals
    if let Some(v) = input.pogq_score {
        fed_rep.pogq_score = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.fl_contributions {
        fed_rep.fl_contributions = v.min(10000); // Cap at 10000
    }
    if let Some(v) = input.byzantine_clean_rate {
        fed_rep.byzantine_clean_rate = v.clamp(0.0, 1.0);
    }

    // Governance domain signals
    if let Some(v) = input.voting_participation {
        fed_rep.voting_participation = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.proposal_success_rate {
        fed_rep.proposal_success_rate = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.consensus_alignment {
        fed_rep.consensus_alignment = v.clamp(0.0, 1.0);
    }

    // Recalculate aggregated score (domain boundaries enforced in calculate_aggregated)
    fed_rep.refresh_aggregation(now);

    // Update the entry
    let updated_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::FederatedReputation(fed_rep),
    )?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Updated reputation not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct UpdateFederatedReputationInput {
    pub identity_verification: Option<f64>,
    pub credential_count: Option<u64>,
    pub credential_quality: Option<f64>,
    pub epistemic_contributions: Option<f64>,
    pub factcheck_accuracy: Option<f64>,
    pub dark_spots_resolved: Option<u64>,
    pub stake_weight: Option<f64>,
    pub payment_reliability: Option<f64>,
    pub escrow_completion_rate: Option<f64>,
    pub pogq_score: Option<f64>,
    pub fl_contributions: Option<u64>,
    pub byzantine_clean_rate: Option<f64>,
    pub voting_participation: Option<f64>,
    pub proposal_success_rate: Option<f64>,
    pub consensus_alignment: Option<f64>,
}

/// Get votes for a consensus round
#[hdk_extern]
pub fn get_round_votes(input: GetRoundVotesInput) -> ExternResult<Vec<Record>> {
    let round_anchor = format!("round:{}:{}", input.proposal_id, input.round);
    let anchor = match anchor_hash(&round_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(vec![]),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::RoundToVotes)?,
        GetStrategy::default(),
    )?;

    let mut votes = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            votes.push(record);
        }
    }

    Ok(votes)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetRoundVotesInput {
    pub proposal_id: String,
    pub round: u64,
}

/// Calculate consensus round result
#[hdk_extern]
pub fn calculate_round_result(input: CalculateRoundResultInput) -> ExternResult<RoundResult> {
    let votes = get_round_votes(GetRoundVotesInput {
        proposal_id: input.proposal_id.clone(),
        round: input.round,
    })?;

    let threshold = AdaptiveThreshold::for_proposal_type(&input.proposal_type);

    let mut total_weight = 0.0;
    let mut weighted_approvals = 0.0;
    let mut weighted_rejections = 0.0;
    let mut vote_count = 0u64;

    for record in &votes {
        if let Some(vote) = record
            .entry()
            .to_app_option::<WeightedVote>()
            .ok()
            .flatten()
        {
            total_weight += vote.weight;
            vote_count += 1;

            match vote.decision {
                VoteDecision::Approve => weighted_approvals += vote.weight,
                VoteDecision::Reject => weighted_rejections += vote.weight,
                VoteDecision::Abstain => {} // Abstains add to total but not to either side
            }
        }
    }

    let required_threshold = threshold.calculate_threshold(total_weight);
    let consensus_reached = weighted_approvals > required_threshold;
    let rejected = weighted_rejections > required_threshold;

    let approval_percentage = if total_weight > 0.0 {
        (weighted_approvals / total_weight) * 100.0
    } else {
        0.0
    };

    let quorum_met = threshold.quorum_met(vote_count, input.eligible_voters);

    let result = if !quorum_met {
        "quorum_not_met"
    } else if consensus_reached {
        "approved"
    } else if rejected {
        "rejected"
    } else {
        "pending"
    };

    Ok(RoundResult {
        proposal_id: input.proposal_id,
        round: input.round,
        proposal_type: input.proposal_type,
        total_weight,
        weighted_approvals,
        weighted_rejections,
        vote_count,
        required_threshold,
        approval_percentage,
        quorum_met,
        consensus_reached,
        rejected,
        result: result.to_string(),
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CalculateRoundResultInput {
    pub proposal_id: String,
    pub round: u64,
    pub proposal_type: ProposalType,
    pub eligible_voters: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RoundResult {
    pub proposal_id: String,
    pub round: u64,
    pub proposal_type: ProposalType,
    pub total_weight: f64,
    pub weighted_approvals: f64,
    pub weighted_rejections: f64,
    pub vote_count: u64,
    pub required_threshold: f64,
    pub approval_percentage: f64,
    pub quorum_met: bool,
    pub consensus_reached: bool,
    pub rejected: bool,
    pub result: String,
}
