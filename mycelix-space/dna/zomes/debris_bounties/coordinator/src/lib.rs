//! Debris Bounties Coordinator Zome
//!
//! Functions for the Kessler Cleanup Market.

use hdk::prelude::*;
use debris_bounties_integrity::*;
use mycelix_space_shared::SpaceTimestamp;

/// Create a new debris bounty
#[hdk_extern]
pub fn create_bounty(input: CreateBountyInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let bounty = DebrisBounty {
        bounty_id: input.bounty_id,
        debris_norad_id: input.debris_norad_id,
        justification: input.justification,
        amount: input.amount,
        currency: input.currency,
        expires_at: input.expires_at,
        status: BountyStatus::Open,
        creator: agent,
        created_at: SpaceTimestamp::now(),
        requirements: input.requirements,
    };

    create_entry(&EntryTypes::DebrisBounty(bounty))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateBountyInput {
    pub bounty_id: String,
    pub debris_norad_id: u32,
    pub justification: String,
    pub amount: u64,
    pub currency: String,
    pub expires_at: Option<SpaceTimestamp>,
    pub requirements: RemovalRequirements,
}

/// Contribute to an existing bounty
#[hdk_extern]
pub fn contribute_to_bounty(input: ContributeInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let contribution = BountyContribution {
        bounty_id: input.bounty_id,
        amount: input.amount,
        currency: input.currency,
        contributor: agent,
        message: input.message,
        contributed_at: SpaceTimestamp::now(),
    };

    create_entry(&EntryTypes::BountyContribution(contribution))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContributeInput {
    pub bounty_id: String,
    pub amount: u64,
    pub currency: String,
    pub message: Option<String>,
}

/// Claim a bounty (announce intent to remove debris)
#[hdk_extern]
pub fn claim_bounty(input: ClaimBountyInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let claim = RemovalClaim {
        bounty_id: input.bounty_id,
        claimer: agent,
        organization: input.organization,
        method: input.method,
        estimated_completion: input.estimated_completion,
        mission_plan: input.mission_plan,
        status: ClaimStatus::Pending,
        claimed_at: SpaceTimestamp::now(),
    };

    create_entry(&EntryTypes::RemovalClaim(claim))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClaimBountyInput {
    pub bounty_id: String,
    pub organization: String,
    pub method: RemovalMethod,
    pub estimated_completion: SpaceTimestamp,
    pub mission_plan: String,
}

/// Submit verification of debris removal
#[hdk_extern]
pub fn submit_verification(input: SubmitVerificationInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let verification = RemovalVerification {
        claim_id: input.claim_id,
        verifier: agent,
        verified: input.verified,
        evidence: input.evidence,
        verified_at: SpaceTimestamp::now(),
    };

    create_entry(&EntryTypes::RemovalVerification(verification))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmitVerificationInput {
    pub claim_id: ActionHash,
    pub verified: bool,
    pub evidence: VerificationEvidence,
}
