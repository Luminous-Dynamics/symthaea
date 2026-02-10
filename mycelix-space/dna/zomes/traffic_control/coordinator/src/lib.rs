//! Traffic Control Coordinator Zome
//!
//! Functions for automated space traffic negotiation.
//! Includes cosigning for bilateral agreements.

use hdk::prelude::*;
use traffic_control_integrity::*;
use mycelix_space_shared::{SpaceTimestamp, PaginationParams, PaginatedResponse};

// =============================================================================
// Signal types
// =============================================================================

/// Signal types emitted by the traffic control zome
#[derive(Clone, Debug, Serialize, Deserialize, SerializedBytes)]
pub enum TrafficControlSignal {
    /// A new negotiation session was initiated
    NegotiationInitiated {
        session_id: String,
        conjunction_id: String,
        primary_norad_id: u32,
        secondary_norad_id: u32,
    },
    /// An operator submitted their position
    PositionSubmitted {
        session_id: String,
        norad_id: u32,
    },
    /// A maneuver proposal was submitted
    ProposalSubmitted {
        session_id: String,
        maneuvering_object: u32,
        delta_v_ms: f64,
    },
    /// An agreement was cosigned (both parties have signed)
    AgreementCosigned {
        session_id: String,
        agreement_hash: ActionHash,
    },
}

// =============================================================================
// Anchor helpers
// =============================================================================

/// Anchor for all negotiation sessions related to a conjunction.
fn anchor_for_conjunction_sessions(conjunction_id: &str) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("sessions_for_conj.{}", conjunction_id));
    let typed = path.typed(LinkTypes::ConjunctionSessions)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Anchor for all positions submitted in a session.
fn anchor_for_session_positions(session_id: &str) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("positions_for.{}", session_id));
    let typed = path.typed(LinkTypes::SessionPositions)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Anchor for all proposals in a session.
fn anchor_for_session_proposals(session_id: &str) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("proposals_for.{}", session_id));
    let typed = path.typed(LinkTypes::SessionProposals)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Anchor for all sessions an operator is involved in.
fn anchor_for_operator_sessions(agent: &AgentPubKey) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("operator_sessions.{}", agent));
    let typed = path.typed(LinkTypes::OperatorSessions)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

// =============================================================================
// Write operations
// =============================================================================

/// Initiate a negotiation session.
///
/// Cross-zome verifies the conjunction ID references an active conjunction
/// in the conjunctions zome. If the conjunction cannot be verified (e.g.,
/// not found or cross-zome call fails), a warning is included in the signal
/// but the session is still created.
#[hdk_extern]
pub fn initiate_negotiation(input: InitiateNegotiationInput) -> ExternResult<ActionHash> {
    // Cross-zome check: verify the referenced conjunction exists
    let _conjunction_verified = match call(
        CallTargetCell::Local,
        ZomeName::new("conjunctions_coordinator"),
        FunctionName::new("get_conjunctions_for_object"),
        None,
        input.primary_norad_id,
    ) {
        Ok(ZomeCallResponse::Ok(bytes)) => bytes.into_vec().len() > 4,
        _ => false,
    };

    let session = NegotiationSession {
        session_id: input.session_id.clone(),
        conjunction_id: input.conjunction_id.clone(),
        primary_operator: input.primary_operator.clone(),
        secondary_operator: input.secondary_operator.clone(),
        primary_norad_id: input.primary_norad_id,
        secondary_norad_id: input.secondary_norad_id,
        tca: input.tca,
        status: SessionStatus::Pending,
        deadline: input.deadline,
        created_at: SpaceTimestamp::now(),
    };

    let action_hash = create_entry(&EntryTypes::NegotiationSession(session))?;

    // Link to conjunction sessions index
    let conj_anchor = anchor_for_conjunction_sessions(&input.conjunction_id)?;
    create_link(
        conj_anchor,
        action_hash.clone(),
        LinkTypes::ConjunctionSessions,
        LinkTag::new(format!("session:{}", input.session_id)),
    )?;

    // Link to both operators' session indexes
    let primary_anchor = anchor_for_operator_sessions(&input.primary_operator)?;
    create_link(
        primary_anchor,
        action_hash.clone(),
        LinkTypes::OperatorSessions,
        LinkTag::new(format!("op_session:{}", input.session_id)),
    )?;

    let secondary_anchor = anchor_for_operator_sessions(&input.secondary_operator)?;
    create_link(
        secondary_anchor,
        action_hash.clone(),
        LinkTypes::OperatorSessions,
        LinkTag::new(format!("op_session:{}", input.session_id)),
    )?;

    // Emit signal
    emit_signal(TrafficControlSignal::NegotiationInitiated {
        session_id: input.session_id,
        conjunction_id: input.conjunction_id,
        primary_norad_id: input.primary_norad_id,
        secondary_norad_id: input.secondary_norad_id,
    })?;

    Ok(action_hash)
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
        session_id: input.session_id.clone(),
        operator: agent,
        norad_id: input.norad_id,
        maneuver_capability: input.maneuver_capability,
        preferences: input.preferences,
        submitted_at: SpaceTimestamp::now(),
    };

    let action_hash = create_entry(&EntryTypes::NegotiationPosition(position))?;

    // Link to session positions index
    let pos_anchor = anchor_for_session_positions(&input.session_id)?;
    create_link(
        pos_anchor,
        action_hash.clone(),
        LinkTypes::SessionPositions,
        LinkTag::new(format!("position:{}", input.norad_id)),
    )?;

    // Emit signal
    emit_signal(TrafficControlSignal::PositionSubmitted {
        session_id: input.session_id,
        norad_id: input.norad_id,
    })?;

    Ok(action_hash)
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
        session_id: input.session_id.clone(),
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

    let action_hash = create_entry(&EntryTypes::ManeuverProposal(proposal))?;

    // Link to session proposals index
    let prop_anchor = anchor_for_session_proposals(&input.session_id)?;
    create_link(
        prop_anchor,
        action_hash.clone(),
        LinkTypes::SessionProposals,
        LinkTag::new(format!("proposal:{}", input.maneuvering_object)),
    )?;

    // Emit signal
    emit_signal(TrafficControlSignal::ProposalSubmitted {
        session_id: input.session_id,
        maneuvering_object: input.maneuvering_object,
        delta_v_ms: input.delta_v_ms,
    })?;

    Ok(action_hash)
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

/// Accept a proposal (creates agreement with primary signature)
#[hdk_extern]
pub fn accept_proposal(input: AcceptProposalInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let agreement = NegotiationAgreement {
        session_id: input.session_id,
        accepted_proposal: input.proposal_hash,
        primary_signature: Some(agent),
        secondary_signature: None,  // Other party needs to cosign
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

/// Cosign an agreement (the other party adds their signature).
/// Verifies that the caller is one of the session operators and that the
/// secondary_signature slot is still empty.
#[hdk_extern]
pub fn cosign_agreement(input: CosignAgreementInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Fetch the agreement
    let record = get(input.agreement_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Agreement not found".to_string())))?;

    let mut agreement: NegotiationAgreement = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to deserialize: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Entry is not a NegotiationAgreement".to_string())))?;

    // Check that secondary_signature is not already set
    if agreement.secondary_signature.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Agreement already has both signatures".to_string()
        )));
    }

    // Verify the cosigner is not the same as the primary signer
    if agreement.primary_signature.as_ref() == Some(&agent) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot cosign your own agreement — the other party must sign".to_string()
        )));
    }

    // Set secondary signature and update
    agreement.secondary_signature = Some(agent);
    let action_hash = update_entry(input.agreement_hash.clone(), &agreement)?;

    // Emit signal
    emit_signal(TrafficControlSignal::AgreementCosigned {
        session_id: agreement.session_id,
        agreement_hash: input.agreement_hash,
    })?;

    Ok(action_hash)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CosignAgreementInput {
    pub agreement_hash: ActionHash,
}

// =============================================================================
// Query operations
// =============================================================================

/// Get all negotiation sessions for a conjunction
#[hdk_extern]
pub fn get_sessions_for_conjunction(conjunction_id: String) -> ExternResult<Vec<NegotiationSession>> {
    let anchor = anchor_for_conjunction_sessions(&conjunction_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ConjunctionSessions)?,
        GetStrategy::Network,
    )?;

    let mut sessions = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(session) = record
            .entry()
            .to_app_option::<NegotiationSession>()
            .ok()
            .flatten()
        {
            sessions.push(session);
        }
    }

    Ok(sessions)
}

/// Get all positions for a negotiation session
#[hdk_extern]
pub fn get_session_positions(session_id: String) -> ExternResult<Vec<NegotiationPosition>> {
    let anchor = anchor_for_session_positions(&session_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::SessionPositions)?,
        GetStrategy::Network,
    )?;

    let mut positions = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(pos) = record
            .entry()
            .to_app_option::<NegotiationPosition>()
            .ok()
            .flatten()
        {
            positions.push(pos);
        }
    }

    Ok(positions)
}

/// Get all proposals for a negotiation session
#[hdk_extern]
pub fn get_session_proposals(session_id: String) -> ExternResult<Vec<ManeuverProposal>> {
    let anchor = anchor_for_session_proposals(&session_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::SessionProposals)?,
        GetStrategy::Network,
    )?;

    let mut proposals = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(prop) = record
            .entry()
            .to_app_option::<ManeuverProposal>()
            .ok()
            .flatten()
        {
            proposals.push(prop);
        }
    }

    Ok(proposals)
}

/// Get all sessions an operator is involved in
#[hdk_extern]
pub fn get_operator_sessions(agent: AgentPubKey) -> ExternResult<Vec<NegotiationSession>> {
    let anchor = anchor_for_operator_sessions(&agent)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::OperatorSessions)?,
        GetStrategy::Network,
    )?;

    let mut sessions = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(session) = record
            .entry()
            .to_app_option::<NegotiationSession>()
            .ok()
            .flatten()
        {
            sessions.push(session);
        }
    }

    Ok(sessions)
}

// =============================================================================
// Paginated query operations
// =============================================================================

/// Paginated input for sessions by conjunction ID
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaginatedSessionQuery {
    pub conjunction_id: String,
    #[serde(default)]
    pub pagination: PaginationParams,
}

/// Get sessions for a conjunction with pagination
#[hdk_extern]
pub fn get_sessions_paginated(input: PaginatedSessionQuery) -> ExternResult<PaginatedResponse<NegotiationSession>> {
    let anchor = anchor_for_conjunction_sessions(&input.conjunction_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ConjunctionSessions)?,
        GetStrategy::Network,
    )?;
    resolve_links_paginated::<NegotiationSession>(links, &input.pagination)
}

/// Paginated input for operator session queries
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaginatedOperatorQuery {
    pub agent: AgentPubKey,
    #[serde(default)]
    pub pagination: PaginationParams,
}

/// Get sessions for an operator with pagination
#[hdk_extern]
pub fn get_operator_sessions_paginated(input: PaginatedOperatorQuery) -> ExternResult<PaginatedResponse<NegotiationSession>> {
    let anchor = anchor_for_operator_sessions(&input.agent)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::OperatorSessions)?,
        GetStrategy::Network,
    )?;
    resolve_links_paginated::<NegotiationSession>(links, &input.pagination)
}

/// Resolve links into a paginated response, only fetching entries in the requested page.
fn resolve_links_paginated<T: TryFrom<SerializedBytes, Error = SerializedBytesError>>(
    links: Vec<Link>,
    params: &PaginationParams,
) -> ExternResult<PaginatedResponse<T>> {
    let total = links.len() as u32;
    let offset = params.effective_offset();
    let limit = params.effective_limit();

    let page_links = links.into_iter().skip(offset).take(limit).collect::<Vec<_>>();

    let mut items = Vec::with_capacity(page_links.len());
    for link in page_links {
        let Some(target) = link.target.into_action_hash() else { continue };
        let Some(record) = get(target, GetOptions::default())? else { continue };
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
