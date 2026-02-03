//! Traffic Control Coordinator Zome
//!
//! Functions for automated space traffic negotiation.

use hdk::prelude::*;
use traffic_control_integrity::*;
use mycelix_space_shared::SpaceTimestamp;

/// Initiate a negotiation session
#[hdk_extern]
pub fn initiate_negotiation(input: InitiateNegotiationInput) -> ExternResult<ActionHash> {
    let session = NegotiationSession {
        session_id: input.session_id,
        conjunction_id: input.conjunction_id,
        primary_operator: input.primary_operator,
        secondary_operator: input.secondary_operator,
        primary_norad_id: input.primary_norad_id,
        secondary_norad_id: input.secondary_norad_id,
        tca: input.tca,
        status: SessionStatus::Pending,
        deadline: input.deadline,
        created_at: SpaceTimestamp::now(),
    };

    create_entry(&EntryTypes::NegotiationSession(session))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InitiateNegotiationInput {
    pub session_id: String,
    pub conjunction_id: String,
    pub primary_operator: AgentPubKey,
    pub secondary_operator: AgentPubKey,
    pub primary_norad_id: u32,
    pub secondary_norad_id: u32,
    pub tca: SpaceTimestamp,
    pub deadline: SpaceTimestamp,
}

/// Submit negotiation position
#[hdk_extern]
pub fn submit_position(input: SubmitPositionInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let position = NegotiationPosition {
        session_id: input.session_id,
        operator: agent,
        norad_id: input.norad_id,
        maneuver_capability: input.maneuver_capability,
        preferences: input.preferences,
        submitted_at: SpaceTimestamp::now(),
    };

    create_entry(&EntryTypes::NegotiationPosition(position))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmitPositionInput {
    pub session_id: String,
    pub norad_id: u32,
    pub maneuver_capability: ManeuverCapability,
    pub preferences: OperatorPreferences,
}

/// Submit a maneuver proposal
#[hdk_extern]
pub fn submit_proposal(input: SubmitProposalInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let proposal = ManeuverProposal {
        session_id: input.session_id,
        proposer: agent,
        maneuvering_object: input.maneuvering_object,
        burn_time: input.burn_time,
        delta_v_ms: input.delta_v_ms,
        direction: input.direction,
        resulting_miss_km: input.resulting_miss_km,
        resulting_pc: input.resulting_pc,
        cost_estimate: input.cost_estimate,
        status: ProposalStatus::Pending,
        created_at: SpaceTimestamp::now(),
    };

    create_entry(&EntryTypes::ManeuverProposal(proposal))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmitProposalInput {
    pub session_id: String,
    pub maneuvering_object: u32,
    pub burn_time: SpaceTimestamp,
    pub delta_v_ms: f64,
    pub direction: [f64; 3],
    pub resulting_miss_km: f64,
    pub resulting_pc: f64,
    pub cost_estimate: Option<CostEstimate>,
}

/// Accept a proposal (requires both operators)
#[hdk_extern]
pub fn accept_proposal(input: AcceptProposalInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Check if there's already a partial agreement
    // In a full implementation, we'd update the existing agreement

    let agreement = NegotiationAgreement {
        session_id: input.session_id,
        accepted_proposal: input.proposal_hash,
        primary_signature: Some(agent.clone()),
        secondary_signature: None,  // Other party needs to sign
        agreed_at: SpaceTimestamp::now(),
        execution_deadline: input.execution_deadline,
    };

    create_entry(&EntryTypes::NegotiationAgreement(agreement))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcceptProposalInput {
    pub session_id: String,
    pub proposal_hash: ActionHash,
    pub execution_deadline: SpaceTimestamp,
}
