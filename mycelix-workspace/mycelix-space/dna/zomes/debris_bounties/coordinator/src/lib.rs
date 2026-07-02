// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Debris Bounties Coordinator Zome
//!
//! Functions for the Kessler Cleanup Market.
//! Includes a state machine for bounty lifecycle management.

use debris_bounties_integrity::*;
use hdk::prelude::*;
use mycelix_space_shared::{
    PaginatedResponse, PaginationParams, SpaceError, SpaceErrorCode, SpaceTimestamp,
    gate_space_operation, requirement_for_bounty_claim, requirement_for_bounty_creation,
    requirement_for_bounty_verification, requirement_for_risk_update, validate_string_field,
};

// =============================================================================
// Signal types
// =============================================================================

/// Signal types emitted by the debris bounties zome
#[derive(Clone, Debug, Serialize, Deserialize, SerializedBytes)]
pub enum BountySignal {
    /// A new bounty was created
    BountyCreated {
        bounty_id: String,
        debris_norad_id: u32,
        amount: u64,
        /// Whether the target debris has tracking data in orbital_objects
        has_tracking_data: bool,
    },
    /// Someone contributed to a bounty
    BountyContributed {
        bounty_id: String,
        amount: u64,
        contributor: AgentPubKey,
    },
    /// A bounty was claimed by a removal service
    BountyClaimed {
        bounty_id: String,
        organization: String,
        method: RemovalMethod,
    },
    /// A bounty's status changed
    BountyStatusChanged {
        bounty_id: String,
        old_status: BountyStatus,
        new_status: BountyStatus,
    },
    /// Removal verification submitted
    VerificationSubmitted {
        claim_id: ActionHash,
        verified: bool,
    },
}

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

/// Anchor for all verifications of a specific claim.
fn anchor_for_claim_verifications(claim_hash: &ActionHash) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("verifications_for.{}", claim_hash));
    let typed = path.typed(LinkTypes::ClaimVerifications)?;
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

/// Create a new debris bounty.
///
/// Cross-zome checks whether the debris NORAD ID has tracking data in
/// orbital_objects. The bounty is always created regardless, but the
/// `has_tracking_data` field in the emitted signal indicates whether the
/// target debris is being tracked (useful for UI warnings).
#[hdk_extern]
pub fn create_bounty(input: CreateBountyInput) -> ExternResult<ActionHash> {
    gate_space_operation(&requirement_for_bounty_creation(), "create_bounty")?;

    // --- Input validation ---
    validate_string_field(&input.bounty_id, "bounty_id", 256).map_err(|e| e.into_wasm_error())?;
    validate_string_field(&input.justification, "justification", 2048)
        .map_err(|e| e.into_wasm_error())?;
    validate_string_field(&input.currency, "currency", 16).map_err(|e| e.into_wasm_error())?;

    if input.amount == 0 {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidBountyAmount,
            "Bounty amount must be greater than zero",
        )
        .into_wasm_error());
    }
    if input.debris_norad_id == 0 || input.debris_norad_id > 999999 {
        return Err(
            SpaceError::new(SpaceErrorCode::InvalidInput, "NORAD ID must be 1-999999")
                .into_wasm_error(),
        );
    }
    if input.requirements.verification_threshold == 0 {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            "verification_threshold must be at least 1",
        )
        .into_wasm_error());
    }

    // Cross-zome check: does the debris object have tracking data?
    let has_tracking = match call(
        CallTargetCell::Local,
        ZomeName::new("orbital_objects_coordinator"),
        FunctionName::new("get_latest_tles"),
        None,
        vec![input.debris_norad_id],
    ) {
        Ok(ZomeCallResponse::Ok(bytes)) => {
            // Non-empty response means tracking data exists.
            // We check byte length rather than deserializing the foreign type.
            bytes.into_vec().len() > 4 // msgpack empty array is ~1 byte
        }
        _ => false,
    };

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

    // Emit signal (includes cross-zome tracking status)
    emit_signal(BountySignal::BountyCreated {
        bounty_id: input.bounty_id,
        debris_norad_id: input.debris_norad_id,
        amount: input.amount,
        has_tracking_data: has_tracking,
    })?;

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

/// Contribute additional funds to an existing bounty.
///
/// Links the contribution to both the bounty's contributions anchor and the
/// contributor's personal bounty index. Emits a `BountyContributed` signal.
#[hdk_extern]
pub fn contribute_to_bounty(input: ContributeInput) -> ExternResult<ActionHash> {
    gate_space_operation(&requirement_for_bounty_claim(), "contribute_to_bounty")?;

    // --- Input validation ---
    validate_string_field(&input.bounty_id, "bounty_id", 256).map_err(|e| e.into_wasm_error())?;
    validate_string_field(&input.currency, "currency", 16).map_err(|e| e.into_wasm_error())?;
    if input.amount == 0 {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidBountyAmount,
            "Contribution amount must be greater than zero",
        )
        .into_wasm_error());
    }
    if let Some(ref msg) = input.message {
        if msg.len() > 1024 {
            return Err(SpaceError::new(
                SpaceErrorCode::InvalidInput,
                "Message exceeds 1024 characters",
            )
            .into_wasm_error());
        }
    }

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

    // Emit signal
    emit_signal(BountySignal::BountyContributed {
        bounty_id: input.bounty_id,
        amount: input.amount,
        contributor: agent,
    })?;

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
    gate_space_operation(&requirement_for_bounty_claim(), "claim_bounty")?;

    // --- Input validation ---
    validate_string_field(&input.organization, "organization", 256)
        .map_err(|e| e.into_wasm_error())?;
    validate_string_field(&input.mission_plan, "mission_plan", 4096)
        .map_err(|e| e.into_wasm_error())?;

    // Fetch the bounty and validate its status
    let record = get(input.bounty_hash.clone(), GetOptions::default())?.ok_or(
        SpaceError::new(SpaceErrorCode::BountyNotFound, "Bounty not found")
            .with_context(format!("hash: {}", input.bounty_hash))
            .into_wasm_error(),
    )?;

    let mut bounty: DebrisBounty = record
        .entry()
        .to_app_option()
        .map_err(|e| {
            SpaceError::new(
                SpaceErrorCode::InvalidInput,
                format!("Failed to deserialize bounty: {:?}", e),
            )
            .into_wasm_error()
        })?
        .ok_or(
            SpaceError::new(SpaceErrorCode::InvalidInput, "Entry is not a DebrisBounty")
                .into_wasm_error(),
        )?;

    if !is_valid_transition(&bounty.status, &BountyStatus::Claimed) {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidStateTransition,
            format!("Cannot claim bounty in {:?} status", bounty.status),
        )
        .with_context(format!("expected: Open, got: {:?}", bounty.status))
        .into_wasm_error());
    }

    // Update bounty status to Claimed
    bounty.status = BountyStatus::Claimed;
    update_entry(input.bounty_hash.clone(), &bounty)?;

    // Create the removal claim entry
    let agent = agent_info()?.agent_initial_pubkey;
    let signal_org = input.organization.clone();
    let signal_method = input.method.clone();
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

    // Emit signal
    emit_signal(BountySignal::BountyClaimed {
        bounty_id: bounty.bounty_id,
        organization: signal_org,
        method: signal_method,
    })?;

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

/// Submit verification evidence for a debris removal claim.
///
/// Verifiers independently attest whether the debris has been removed.
/// When the number of positive verifications meets the bounty's
/// `verification_threshold`, the bounty can transition to Completed.
/// Emits a `VerificationSubmitted` signal.
#[hdk_extern]
pub fn submit_verification(input: SubmitVerificationInput) -> ExternResult<ActionHash> {
    gate_space_operation(
        &requirement_for_bounty_verification(),
        "submit_verification",
    )?;

    let agent = agent_info()?.agent_initial_pubkey;

    let signal_claim_id = input.claim_id.clone();
    let verification = RemovalVerification {
        claim_id: input.claim_id,
        verifier: agent,
        verified: input.verified,
        evidence: input.evidence,
        verified_at: SpaceTimestamp::now(),
    };

    let action_hash = create_entry(&EntryTypes::RemovalVerification(verification))?;

    // Link verification to claim's verifications anchor
    let verif_anchor = anchor_for_claim_verifications(&signal_claim_id)?;
    create_link(
        verif_anchor,
        action_hash.clone(),
        LinkTypes::ClaimVerifications,
        LinkTag::new(format!("verif:{}", signal_claim_id)),
    )?;

    // Emit signal
    emit_signal(BountySignal::VerificationSubmitted {
        claim_id: signal_claim_id,
        verified: input.verified,
    })?;

    Ok(action_hash)
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
    gate_space_operation(&requirement_for_risk_update(), "update_bounty_status")?;

    let record = get(input.bounty_hash.clone(), GetOptions::default())?.ok_or(
        SpaceError::new(SpaceErrorCode::BountyNotFound, "Bounty not found")
            .with_context(format!("hash: {}", input.bounty_hash))
            .into_wasm_error(),
    )?;

    let mut bounty: DebrisBounty = record
        .entry()
        .to_app_option()
        .map_err(|e| {
            SpaceError::new(
                SpaceErrorCode::InvalidInput,
                format!("Failed to deserialize: {:?}", e),
            )
            .into_wasm_error()
        })?
        .ok_or(
            SpaceError::new(SpaceErrorCode::InvalidInput, "Entry is not a DebrisBounty")
                .into_wasm_error(),
        )?;

    if !is_valid_transition(&bounty.status, &input.new_status) {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidStateTransition,
            format!(
                "Invalid transition: {:?} -> {:?}",
                bounty.status, input.new_status
            ),
        )
        .with_context(format!(
            "from: {:?}, to: {:?}",
            bounty.status, input.new_status
        ))
        .into_wasm_error());
    }

    // --- Authorization: only creator can cancel their own bounty ---
    if input.new_status == BountyStatus::Cancelled {
        let agent = agent_info()?.agent_initial_pubkey;
        if agent != bounty.creator {
            return Err(SpaceError::new(
                SpaceErrorCode::Unauthorized,
                "Only the bounty creator can cancel a bounty",
            )
            .into_wasm_error());
        }
    }

    let old_status = bounty.status.clone();
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

    // Emit signal
    emit_signal(BountySignal::BountyStatusChanged {
        bounty_id: bounty.bounty_id,
        old_status,
        new_status: input.new_status,
    })?;

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

/// Get all bounties targeting a given debris NORAD ID.
///
/// Queries the `bounties_for_debris.{norad_id}` anchor. Returns bounties
/// in all states (including terminal).
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

/// Get all currently active (non-terminal) bounties on the network.
///
/// Bounties are removed from this index when they reach a terminal state
/// (Completed, Expired, Cancelled).
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

/// Get all contributions for a bounty by its `ActionHash`.
///
/// Queries the `contributions_for.{hash}` anchor.
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

/// Get all removal claims on a bounty by its `ActionHash`.
///
/// Queries the `claims_for.{hash}` anchor.
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

/// Get all verifications for a specific removal claim.
///
/// Queries the `verifications_for.{hash}` anchor.
#[hdk_extern]
pub fn get_verifications_for_claim(
    claim_hash: ActionHash,
) -> ExternResult<Vec<RemovalVerification>> {
    let anchor = anchor_for_claim_verifications(&claim_hash)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ClaimVerifications)?,
        GetStrategy::Network,
    )?;

    let mut verifications = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(verif) = record
            .entry()
            .to_app_option::<RemovalVerification>()
            .ok()
            .flatten()
        {
            verifications.push(verif);
        }
    }

    Ok(verifications)
}

// =============================================================================
// Paginated query operations
// =============================================================================

/// Paginated input for debris bounty queries
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaginatedDebrisQuery {
    pub norad_id: u32,
    #[serde(default)]
    pub pagination: PaginationParams,
}

/// Get bounties for a debris object with pagination
#[hdk_extern]
pub fn get_bounties_for_debris_paginated(
    input: PaginatedDebrisQuery,
) -> ExternResult<PaginatedResponse<DebrisBounty>> {
    let anchor = anchor_for_object_bounties(input.norad_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ObjectBounties)?,
        GetStrategy::Network,
    )?;
    resolve_links_paginated::<DebrisBounty>(links, &input.pagination)
}

/// Get active bounties with pagination
#[hdk_extern]
pub fn get_active_bounties_paginated(
    pagination: PaginationParams,
) -> ExternResult<PaginatedResponse<DebrisBounty>> {
    let anchor = anchor_for_active_bounties()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ActiveBounties)?,
        GetStrategy::Network,
    )?;
    resolve_links_paginated::<DebrisBounty>(links, &pagination)
}

/// Resolve a set of links into a paginated response, only fetching entries in the requested page.
fn resolve_links_paginated<T: TryFrom<SerializedBytes, Error = SerializedBytesError>>(
    links: Vec<Link>,
    params: &PaginationParams,
) -> ExternResult<PaginatedResponse<T>> {
    let total = links.len() as u32;
    let offset = params.effective_offset();
    let limit = params.effective_limit();

    let page_links = links
        .into_iter()
        .skip(offset)
        .take(limit)
        .collect::<Vec<_>>();

    let mut items = Vec::with_capacity(page_links.len());
    for link in page_links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(item) = record.entry().to_app_option::<T>().ok().flatten() {
            items.push(item);
        }
    }

    let effective_offset = offset as u32;
    let effective_limit = limit as u32;
    Ok(PaginatedResponse {
        has_more: effective_offset + effective_limit < total,
        items,
        total,
        offset: effective_offset,
        limit: effective_limit,
    })
}
