//! Debris Bounties Coordinator Zome
//!
//! Functions for the Kessler Cleanup Market.
//! Includes a state machine for bounty lifecycle management.

use hdk::prelude::*;
use debris_bounties_integrity::*;
use mycelix_space_shared::SpaceTimestamp;

// =============================================================================
// State Machine
// =============================================================================

/// Validate that a bounty status transition is legal.
///
/// ```text
/// Open -> Claimed | Expired | Cancelled
/// Claimed -> InProgress | Open (release)
/// InProgress -> PendingVerification
/// PendingVerification -> Completed | InProgress (verification failed)
/// Terminal: Completed, Expired, Cancelled
/// ```
fn is_valid_transition(from: &BountyStatus, to: &BountyStatus) -> bool {
    matches!(
        (from, to),
        (BountyStatus::Open, BountyStatus::Claimed)
        | (BountyStatus::Open, BountyStatus::Expired)
        | (BountyStatus::Open, BountyStatus::Cancelled)
        | (BountyStatus::Claimed, BountyStatus::InProgress)
        | (BountyStatus::Claimed, BountyStatus::Open) // release claim
        | (BountyStatus::InProgress, BountyStatus::PendingVerification)
        | (BountyStatus::PendingVerification, BountyStatus::Completed)
        | (BountyStatus::PendingVerification, BountyStatus::InProgress) // verification failed
    )
}

// =============================================================================
// Anchor helpers
// =============================================================================

/// Anchor for all bounties targeting a given debris NORAD ID.
fn anchor_for_object_bounties(norad_id: u32) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("bounties_for_debris.{}", norad_id));
    let typed = path.typed(LinkTypes::ObjectBounties)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Anchor for all currently active (non-terminal) bounties.
fn anchor_for_active_bounties() -> ExternResult<AnyLinkableHash> {
    let path = Path::from("active_bounties");
    let typed = path.typed(LinkTypes::ActiveBounties)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Anchor for all contributions to a specific bounty.
fn anchor_for_bounty_contributions(bounty_hash: &ActionHash) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("contributions_for.{}", bounty_hash));
    let typed = path.typed(LinkTypes::BountyContributions)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Anchor for all claims on a specific bounty.
fn anchor_for_bounty_claims(bounty_hash: &ActionHash) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("claims_for.{}", bounty_hash));
    let typed = path.typed(LinkTypes::BountyClaims)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Anchor for all bounties a given agent has contributed to.
fn anchor_for_contributor(agent: &AgentPubKey) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("contributor.{}", agent));
    let typed = path.typed(LinkTypes::ContributorBounties)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

// =============================================================================
// Write operations
// =============================================================================

/// Create a new debris bounty
#[hdk_extern]
pub fn create_bounty(input: CreateBountyInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let bounty = DebrisBounty {
        bounty_id: input.bounty_id.clone(),
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

    let action_hash = create_entry(&EntryTypes::DebrisBounty(bounty))?;

    // Link to debris object index
    let obj_anchor = anchor_for_object_bounties(input.debris_norad_id)?;
    create_link(
        obj_anchor,
        action_hash.clone(),
        LinkTypes::ObjectBounties,
        LinkTag::new(format!("bounty:{}", input.bounty_id)),
    )?;

    // Link to active bounties index
    let active_anchor = anchor_for_active_bounties()?;
    create_link(
        active_anchor,
        action_hash.clone(),
        LinkTypes::ActiveBounties,
        LinkTag::new(format!("active:{}", input.bounty_id)),
    )?;

    Ok(action_hash)
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
        bounty_id: input.bounty_id.clone(),
        amount: input.amount,
        currency: input.currency,
        contributor: agent.clone(),
        message: input.message,
        contributed_at: SpaceTimestamp::now(),
    };

    let action_hash = create_entry(&EntryTypes::BountyContribution(contribution))?;

    // Link to bounty's contributions anchor
    let contrib_anchor = anchor_for_bounty_contributions(&input.bounty_hash)?;
    create_link(
        contrib_anchor,
        action_hash.clone(),
        LinkTypes::BountyContributions,
        LinkTag::new(format!("contrib:{}", input.bounty_id)),
    )?;

    // Link to contributor's bounty index
    let contributor_anchor = anchor_for_contributor(&agent)?;
    let bounty_linkable: AnyLinkableHash = input.bounty_hash.clone().into();
    create_link(
        contributor_anchor,
        bounty_linkable,
        LinkTypes::ContributorBounties,
        LinkTag::new(format!("contributed:{}", input.bounty_id)),
    )?;

    Ok(action_hash)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContributeInput {
    pub bounty_hash: ActionHash,
    pub bounty_id: String,
    pub amount: u64,
    pub currency: String,
    pub message: Option<String>,
}

/// Claim a bounty (announce intent to remove debris).
/// Validates that the bounty is currently Open, then transitions to Claimed.
#[hdk_extern]
pub fn claim_bounty(input: ClaimBountyInput) -> ExternResult<ActionHash> {
    // Fetch the bounty and validate its status
    let record = get(input.bounty_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Bounty not found".to_string())))?;

    let mut bounty: DebrisBounty = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to deserialize bounty: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Entry is not a DebrisBounty".to_string())))?;

    if !is_valid_transition(&bounty.status, &BountyStatus::Claimed) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Cannot claim bounty in {:?} status", bounty.status)
        )));
    }

    // Update bounty status to Claimed
    bounty.status = BountyStatus::Claimed;
    update_entry(input.bounty_hash.clone(), &bounty)?;

    // Create the removal claim entry
    let agent = agent_info()?.agent_initial_pubkey;
    let claim = RemovalClaim {
        bounty_id: bounty.bounty_id.clone(),
        claimer: agent,
        organization: input.organization,
        method: input.method,
        estimated_completion: input.estimated_completion,
        mission_plan: input.mission_plan,
        status: ClaimStatus::Pending,
        claimed_at: SpaceTimestamp::now(),
    };

    let claim_hash = create_entry(&EntryTypes::RemovalClaim(claim))?;

    // Link claim to bounty's claims anchor
    let claims_anchor = anchor_for_bounty_claims(&input.bounty_hash)?;
    create_link(
        claims_anchor,
        claim_hash.clone(),
        LinkTypes::BountyClaims,
        LinkTag::new(format!("claim:{}", bounty.bounty_id)),
    )?;

    Ok(claim_hash)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClaimBountyInput {
    pub bounty_hash: ActionHash,
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

// =============================================================================
// Status management
// =============================================================================

/// Update a bounty's status with state machine validation.
/// Manages the ActiveBounties link: removes it when transitioning to a terminal state.
#[hdk_extern]
pub fn update_bounty_status(input: UpdateBountyStatusInput) -> ExternResult<ActionHash> {
    let record = get(input.bounty_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Bounty not found".to_string())))?;

    let mut bounty: DebrisBounty = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to deserialize: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Entry is not a DebrisBounty".to_string())))?;

    if !is_valid_transition(&bounty.status, &input.new_status) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid transition: {:?} -> {:?}", bounty.status, input.new_status)
        )));
    }

    bounty.status = input.new_status.clone();
    let action_hash = update_entry(input.bounty_hash.clone(), &bounty)?;

    // If transitioning to a terminal state, remove from active bounties
    let is_terminal = matches!(
        input.new_status,
        BountyStatus::Completed | BountyStatus::Expired | BountyStatus::Cancelled
    );

    if is_terminal {
        let active_anchor = anchor_for_active_bounties()?;
        let links = get_links(
            LinkQuery::try_new(active_anchor, LinkTypes::ActiveBounties)?,
            GetStrategy::Network,
        )?;
        for link in links {
            if link.target.into_action_hash().as_ref() == Some(&input.bounty_hash) {
                delete_link(link.create_link_hash, GetOptions::default())?;
            }
        }
    }

    Ok(action_hash)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateBountyStatusInput {
    pub bounty_hash: ActionHash,
    pub new_status: BountyStatus,
}

// =============================================================================
// Query operations
// =============================================================================

/// Get all bounties targeting a given debris NORAD ID
#[hdk_extern]
pub fn get_bounties_for_debris(norad_id: u32) -> ExternResult<Vec<DebrisBounty>> {
    let anchor = anchor_for_object_bounties(norad_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ObjectBounties)?,
        GetStrategy::Network,
    )?;

    let mut bounties = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(bounty) = record
            .entry()
            .to_app_option::<DebrisBounty>()
            .ok()
            .flatten()
        {
            bounties.push(bounty);
        }
    }

    Ok(bounties)
}

/// Get all currently active (non-terminal) bounties
#[hdk_extern]
pub fn get_active_bounties(_: ()) -> ExternResult<Vec<DebrisBounty>> {
    let anchor = anchor_for_active_bounties()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ActiveBounties)?,
        GetStrategy::Network,
    )?;

    let mut bounties = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(bounty) = record
            .entry()
            .to_app_option::<DebrisBounty>()
            .ok()
            .flatten()
        {
            bounties.push(bounty);
        }
    }

    Ok(bounties)
}

/// Get all contributions for a bounty
#[hdk_extern]
pub fn get_contributions(bounty_hash: ActionHash) -> ExternResult<Vec<BountyContribution>> {
    let anchor = anchor_for_bounty_contributions(&bounty_hash)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::BountyContributions)?,
        GetStrategy::Network,
    )?;

    let mut contributions = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(contrib) = record
            .entry()
            .to_app_option::<BountyContribution>()
            .ok()
            .flatten()
        {
            contributions.push(contrib);
        }
    }

    Ok(contributions)
}

/// Get all claims on a bounty
#[hdk_extern]
pub fn get_claims(bounty_hash: ActionHash) -> ExternResult<Vec<RemovalClaim>> {
    let anchor = anchor_for_bounty_claims(&bounty_hash)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::BountyClaims)?,
        GetStrategy::Network,
    )?;

    let mut claims = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(claim) = record
            .entry()
            .to_app_option::<RemovalClaim>()
            .ok()
            .flatten()
        {
            claims.push(claim);
        }
    }

    Ok(claims)
}
