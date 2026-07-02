// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Traffic Control Coordinator Zome
//!
//! Functions for automated space traffic negotiation.
//! Includes cosigning for bilateral agreements.

use hdk::prelude::*;
use mycelix_space_shared::{
    PaginatedResponse, PaginationParams, SpaceError, SpaceErrorCode, SpaceTimestamp,
    gate_space_operation, requirement_for_agreement_signing, requirement_for_negotiation,
    requirement_for_observation, validate_string_field,
};
use orbital_mechanics::conjunction::ConjunctionAnalyzer;
use orbital_mechanics::lambert::solve_lambert;
use orbital_mechanics::propagator::Propagator;
use orbital_mechanics::tle::TwoLineElement;
use traffic_control_integrity::*;

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
    PositionSubmitted { session_id: String, norad_id: u32 },
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

/// Anchor for bilateral agreements in a session (uniqueness constraint).
fn anchor_for_session_agreements(session_id: &str) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("agreements_for_session.{}", session_id));
    let typed = path.typed(LinkTypes::SessionAgreements)?;
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
// Authorization helpers
// =============================================================================

/// Verify that the given agent is a participant (primary or secondary operator)
/// in the named session. Returns the session entry on success.
fn verify_session_participant(
    session_id: &str,
    agent: &AgentPubKey,
) -> ExternResult<NegotiationSession> {
    let conj_anchor_fallback = anchor_for_conjunction_sessions(session_id)?;
    // Search for the session by iterating all sessions linked from the session_id anchor
    // We use the operator_sessions anchor for the calling agent instead (more efficient)
    let op_anchor = anchor_for_operator_sessions(agent)?;
    let links = get_links(
        LinkQuery::try_new(op_anchor, LinkTypes::OperatorSessions)?,
        GetStrategy::Network,
    )?;

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
            if session.session_id == session_id {
                return Ok(session);
            }
        }
    }

    // Fallback: also check conjunction sessions anchor (in case operator_sessions link is missing)
    let conj_links = get_links(
        LinkQuery::try_new(conj_anchor_fallback, LinkTypes::ConjunctionSessions)?,
        GetStrategy::Network,
    )?;

    for link in conj_links {
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
            if session.session_id == session_id
                && (session.primary_operator == *agent || session.secondary_operator == *agent)
            {
                return Ok(session);
            }
        }
    }

    Err(SpaceError::new(
        SpaceErrorCode::Unauthorized,
        "You are not a participant in this negotiation session",
    )
    .with_context(format!("session: {}", session_id))
    .into_wasm_error())
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
    gate_space_operation(&requirement_for_negotiation(), "initiate_negotiation")?;

    // --- Input validation ---
    validate_string_field(&input.session_id, "session_id", 256).map_err(|e| e.into_wasm_error())?;
    validate_string_field(&input.conjunction_id, "conjunction_id", 256)
        .map_err(|e| e.into_wasm_error())?;

    if input.primary_norad_id == input.secondary_norad_id {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            "Primary and secondary NORAD IDs must differ",
        )
        .into_wasm_error());
    }
    if input.primary_operator == input.secondary_operator {
        return Err(SpaceError::new(
            SpaceErrorCode::SameSignerError,
            "Primary and secondary operators must be different agents",
        )
        .into_wasm_error());
    }

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

/// Submit an operator's negotiation position for a session.
///
/// Records maneuver capability and preferences, then links to the
/// session's positions anchor for query via `get_session_positions()`.
/// Emits a `PositionSubmitted` signal.
#[hdk_extern]
pub fn submit_position(input: SubmitPositionInput) -> ExternResult<ActionHash> {
    gate_space_operation(&requirement_for_observation(), "submit_position")?;

    // --- Input validation ---
    validate_string_field(&input.session_id, "session_id", 256).map_err(|e| e.into_wasm_error())?;

    if let Some(dv) = input.maneuver_capability.max_delta_v_ms {
        if dv < 0.0 || dv.is_nan() {
            return Err(SpaceError::new(
                SpaceErrorCode::InvalidProposal,
                "max_delta_v_ms must be non-negative",
            )
            .into_wasm_error());
        }
    }
    if let Some(fuel) = input.maneuver_capability.fuel_percentage {
        if !(0.0..=100.0).contains(&fuel) || fuel.is_nan() {
            return Err(SpaceError::new(
                SpaceErrorCode::InvalidProposal,
                "fuel_percentage must be 0-100",
            )
            .into_wasm_error());
        }
    }

    let agent = agent_info()?.agent_initial_pubkey;

    // --- Authorization: only session participants can submit positions ---
    verify_session_participant(&input.session_id, &agent)?;

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

/// Submit a maneuver proposal for a negotiation session.
///
/// Includes the proposed burn parameters, resulting miss distance and Pc,
/// and an optional cost estimate. Linked to the session's proposals anchor.
/// Emits a `ProposalSubmitted` signal.
#[hdk_extern]
pub fn submit_proposal(input: SubmitProposalInput) -> ExternResult<ActionHash> {
    gate_space_operation(&requirement_for_negotiation(), "submit_proposal")?;

    // --- Input validation ---
    validate_string_field(&input.session_id, "session_id", 256).map_err(|e| e.into_wasm_error())?;

    if input.delta_v_ms < 0.0 || input.delta_v_ms.is_nan() {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidProposal,
            "delta_v_ms must be non-negative",
        )
        .into_wasm_error());
    }
    if input.resulting_miss_km < 0.0 || input.resulting_miss_km.is_nan() {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidProposal,
            "resulting_miss_km must be non-negative",
        )
        .into_wasm_error());
    }
    if input.resulting_pc < 0.0 || input.resulting_pc > 1.0 || input.resulting_pc.is_nan() {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidProbability,
            "resulting_pc must be 0-1",
        )
        .into_wasm_error());
    }

    let agent = agent_info()?.agent_initial_pubkey;

    // --- Authorization: only session participants can submit proposals ---
    let session = verify_session_participant(&input.session_id, &agent)?;

    // --- Server-side Pc verification via SGP4 + Alfano ---
    // Fetch TLEs for both objects and independently compute collision probability.
    // If the caller-provided resulting_pc differs by more than 10× from our
    // computation, reject the proposal as implausible.
    if let Some(server_pc) = verify_pc_from_tles(
        session.primary_norad_id,
        session.secondary_norad_id,
        &input.burn_time,
    ) {
        // Allow a 10× tolerance band to account for different propagation epochs
        // and covariance assumptions
        let ratio = if server_pc > 1e-15 && input.resulting_pc > 1e-15 {
            (input.resulting_pc / server_pc).max(server_pc / input.resulting_pc)
        } else {
            1.0 // If either is effectively zero, skip ratio check
        };

        if ratio > 10.0 {
            return Err(SpaceError::new(
                SpaceErrorCode::InvalidProbability,
                "Claimed resulting_pc differs from server-computed Pc by more than 10×",
            )
            .with_context(format!(
                "claimed: {:.2e}, server: {:.2e}, ratio: {:.1}×",
                input.resulting_pc, server_pc, ratio
            ))
            .into_wasm_error());
        }
    }
    // If TLEs are unavailable the verification is best-effort — allow the proposal.

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

/// Accept a proposal, creating a `NegotiationAgreement` with the caller's signature.
///
/// The agreement starts with only the primary signature. The other party
/// must call `cosign_agreement()` to complete bilateral signing.
///
/// Guards:
/// - Only one agreement per session (uniqueness via SessionAgreements anchor)
/// - Execution deadline must be in the future
/// - Caller must be a session participant
#[hdk_extern]
pub fn accept_proposal(input: AcceptProposalInput) -> ExternResult<ActionHash> {
    gate_space_operation(&requirement_for_negotiation(), "accept_proposal")?;

    let agent = agent_info()?.agent_initial_pubkey;

    // --- Authorization: only session participants can accept proposals ---
    verify_session_participant(&input.session_id, &agent)?;

    // --- Deadline check: execution_deadline must be in the future ---
    let now = SpaceTimestamp::now();
    if now.micros >= input.execution_deadline.micros {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            "Execution deadline has already passed",
        )
        .with_context(format!(
            "now: {}, deadline: {}",
            now.micros, input.execution_deadline.micros
        ))
        .into_wasm_error());
    }

    // --- Uniqueness: only one agreement per session ---
    let agr_anchor = anchor_for_session_agreements(&input.session_id)?;
    let existing_links = get_links(
        LinkQuery::try_new(agr_anchor.clone(), LinkTypes::SessionAgreements)?,
        GetStrategy::Network,
    )?;
    if !existing_links.is_empty() {
        return Err(SpaceError::new(
            SpaceErrorCode::AlreadySigned,
            "An agreement already exists for this session",
        )
        .with_context(format!("session: {}", input.session_id))
        .into_wasm_error());
    }

    let agreement = NegotiationAgreement {
        session_id: input.session_id.clone(),
        accepted_proposal: input.proposal_hash,
        primary_signature: Some(agent),
        secondary_signature: None, // Other party needs to cosign
        agreed_at: SpaceTimestamp::now(),
        execution_deadline: input.execution_deadline,
    };

    let action_hash = create_entry(&EntryTypes::NegotiationAgreement(agreement))?;

    // Link agreement to session anchor for uniqueness tracking
    create_link(
        agr_anchor,
        action_hash.clone(),
        LinkTypes::SessionAgreements,
        LinkTag::new(format!("agreement:{}", input.session_id)),
    )?;

    Ok(action_hash)
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
    gate_space_operation(&requirement_for_agreement_signing(), "cosign_agreement")?;

    let agent = agent_info()?.agent_initial_pubkey;

    // Fetch the agreement
    let record = get(input.agreement_hash.clone(), GetOptions::default())?.ok_or(
        SpaceError::new(SpaceErrorCode::NotFound, "Agreement not found")
            .with_context(format!("hash: {}", input.agreement_hash))
            .into_wasm_error(),
    )?;

    let mut agreement: NegotiationAgreement = record
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
            SpaceError::new(
                SpaceErrorCode::InvalidInput,
                "Entry is not a NegotiationAgreement",
            )
            .into_wasm_error(),
        )?;

    // --- Deadline check: execution_deadline must not have passed ---
    let now = SpaceTimestamp::now();
    if now.micros >= agreement.execution_deadline.micros {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            "Execution deadline has passed — agreement can no longer be cosigned",
        )
        .with_context(format!(
            "now: {}, deadline: {}",
            now.micros, agreement.execution_deadline.micros
        ))
        .into_wasm_error());
    }

    // Check that secondary_signature is not already set
    if agreement.secondary_signature.is_some() {
        return Err(SpaceError::new(
            SpaceErrorCode::AlreadySigned,
            "Agreement already has both signatures",
        )
        .into_wasm_error());
    }

    // Verify the cosigner is not the same as the primary signer
    if agreement.primary_signature.as_ref() == Some(&agent) {
        return Err(SpaceError::new(
            SpaceErrorCode::SameSignerError,
            "Cannot cosign your own agreement — the other party must sign",
        )
        .into_wasm_error());
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
// Lambert-based avoidance planning
// =============================================================================

/// Input for avoidance option generation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AvoidanceInput {
    /// Session this is for
    pub session_id: String,
    /// Primary object TLE (line1\nline2)
    pub primary_tle_line1: String,
    pub primary_tle_line2: String,
    /// Secondary object TLE (for miss distance computation)
    pub secondary_tle_line1: String,
    pub secondary_tle_line2: String,
    /// Time of closest approach
    pub tca: SpaceTimestamp,
    /// Maximum allowable ΔV (m/s)
    pub max_delta_v_ms: f64,
    /// Hours before TCA to perform the burn
    pub lead_time_hours: f64,
}

/// Anchor for avoidance options linked to a session.
fn anchor_for_session_avoidance(session_id: &str) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("avoidance_for.{}", session_id));
    let typed = path.typed(LinkTypes::SessionAvoidanceOptions)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Generate Lambert-based avoidance maneuver options for a conjunction.
///
/// Propagates both objects to the burn time (TCA - lead_time), generates
/// 4 displacement targets (prograde/retrograde/radial/normal ±5 km), solves
/// Lambert for each, and returns feasible options sorted by ΔV.
#[hdk_extern]
pub fn generate_avoidance_options(input: AvoidanceInput) -> ExternResult<Vec<AvoidanceOption>> {
    validate_string_field(&input.session_id, "session_id", 256).map_err(|e| e.into_wasm_error())?;

    if input.max_delta_v_ms <= 0.0 || !input.max_delta_v_ms.is_finite() {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            "max_delta_v_ms must be positive",
        )
        .into_wasm_error());
    }
    if input.lead_time_hours <= 0.0 || !input.lead_time_hours.is_finite() {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            "lead_time_hours must be positive",
        )
        .into_wasm_error());
    }

    // Parse TLEs
    let primary_tle =
        TwoLineElement::parse_lines(None, &input.primary_tle_line1, &input.primary_tle_line2)
            .map_err(|e| {
                SpaceError::new(
                    SpaceErrorCode::TleParseError,
                    format!("Primary TLE parse error: {:?}", e),
                )
                .into_wasm_error()
            })?;
    let secondary_tle =
        TwoLineElement::parse_lines(None, &input.secondary_tle_line1, &input.secondary_tle_line2)
            .map_err(|e| {
            SpaceError::new(
                SpaceErrorCode::TleParseError,
                format!("Secondary TLE parse error: {:?}", e),
            )
            .into_wasm_error()
        })?;

    // Create propagators
    let primary_prop = Propagator::from_tle(&primary_tle).map_err(|e| {
        SpaceError::new(
            SpaceErrorCode::TleParseError,
            format!("Primary propagator error: {:?}", e),
        )
        .into_wasm_error()
    })?;
    let secondary_prop = Propagator::from_tle(&secondary_tle).map_err(|e| {
        SpaceError::new(
            SpaceErrorCode::TleParseError,
            format!("Secondary propagator error: {:?}", e),
        )
        .into_wasm_error()
    })?;

    // Convert SpaceTimestamp to DateTime
    let tca_secs = input.tca.micros / 1_000_000;
    let tca_nsecs = ((input.tca.micros % 1_000_000) * 1000) as u32;
    let tca_dt = chrono::DateTime::from_timestamp(tca_secs, tca_nsecs).ok_or_else(|| {
        SpaceError::new(SpaceErrorCode::InvalidInput, "Invalid TCA timestamp").into_wasm_error()
    })?;

    // Burn time = TCA - lead_time
    let lead_secs = (input.lead_time_hours * 3600.0) as i64;
    let burn_dt = tca_dt - chrono::Duration::seconds(lead_secs);
    let tof_secs = input.lead_time_hours * 3600.0;

    // Propagate primary to burn time and TCA
    let p_burn = primary_prop.propagate_to(burn_dt).map_err(|e| {
        SpaceError::new(
            SpaceErrorCode::InvalidInput,
            format!("Primary propagation error: {:?}", e),
        )
        .into_wasm_error()
    })?;
    let p_tca = primary_prop.propagate_to(tca_dt).map_err(|e| {
        SpaceError::new(
            SpaceErrorCode::InvalidInput,
            format!("Primary propagation to TCA error: {:?}", e),
        )
        .into_wasm_error()
    })?;

    // Propagate secondary to TCA (for miss distance computation)
    let s_tca = secondary_prop.propagate_to(tca_dt).map_err(|e| {
        SpaceError::new(
            SpaceErrorCode::InvalidInput,
            format!("Secondary propagation error: {:?}", e),
        )
        .into_wasm_error()
    })?;

    let r_burn = nalgebra::Vector3::new(p_burn.state.x, p_burn.state.y, p_burn.state.z);
    let v_burn = nalgebra::Vector3::new(p_burn.state.vx, p_burn.state.vy, p_burn.state.vz);
    let r_tca = nalgebra::Vector3::new(p_tca.state.x, p_tca.state.y, p_tca.state.z);
    let r_sec_tca = nalgebra::Vector3::new(s_tca.state.x, s_tca.state.y, s_tca.state.z);

    let mu = orbital_mechanics::coordinates::wgs84::MU;

    // Generate displacement targets at TCA: ±5 km in each direction
    let displacement_km = 5.0;
    let r_tca_mag = r_tca.norm();

    // Compute orbital frame unit vectors at TCA
    let r_hat = r_tca / r_tca_mag;
    let v_hat_tca = nalgebra::Vector3::new(p_tca.state.vx, p_tca.state.vy, p_tca.state.vz);
    let v_norm = v_hat_tca.norm();
    let prograde = if v_norm > 1e-10 {
        v_hat_tca / v_norm
    } else {
        r_hat
    };
    let normal = r_tca.cross(&v_hat_tca);
    let n_mag = normal.norm();
    let normal = if n_mag > 1e-10 {
        normal / n_mag
    } else {
        nalgebra::Vector3::new(0.0, 0.0, 1.0)
    };
    let radial = r_hat;

    let directions = [
        ("prograde", prograde * displacement_km),
        ("retrograde", -prograde * displacement_km),
        ("radial", radial * displacement_km),
        ("normal", normal * displacement_km),
    ];

    let max_dv_kms = input.max_delta_v_ms / 1000.0;

    let burn_epoch_micros = input.tca.micros - (lead_secs * 1_000_000);
    let burn_ts = SpaceTimestamp {
        micros: burn_epoch_micros,
    };

    let mut options = Vec::new();

    for (label, offset) in &directions {
        let r_target = r_tca + offset;

        // Solve Lambert: from r_burn at burn_time to r_target at TCA
        let sol = match solve_lambert(&r_burn, &r_target, tof_secs, mu, false) {
            Ok(s) => s,
            Err(_) => continue, // Skip if Lambert fails for this geometry
        };

        // ΔV = Lambert departure velocity - current velocity at burn
        let dv = sol.v1 - v_burn;
        let dv_mag = dv.norm();

        // Skip if exceeds fuel budget
        if dv_mag > max_dv_kms {
            continue;
        }

        // Compute new miss distance at TCA (post-maneuver position = r_target by construction)
        let miss_km = (r_target - r_sec_tca).norm();

        // Estimate post-maneuver Pc using Alfano
        let analyzer = ConjunctionAnalyzer::new();
        let post_p = orbital_mechanics::state::OrbitalState::new(
            0,
            tca_dt,
            orbital_mechanics::state::StateVector::from_vectors(r_target, sol.v2),
            orbital_mechanics::state::DataSource::OperatorEphemeris,
        );
        let post_s = orbital_mechanics::state::OrbitalState::new(
            0,
            tca_dt,
            s_tca.state.clone(),
            orbital_mechanics::state::DataSource::OperatorEphemeris,
        );
        let assessment = analyzer.assess(&post_p, &post_s);
        let resulting_pc = assessment.collision_probability.pc;

        let option = AvoidanceOption {
            session_id: input.session_id.clone(),
            maneuver_time: burn_ts.clone(),
            delta_v: vec![dv.x, dv.y, dv.z],
            delta_v_magnitude_ms: dv_mag * 1000.0,
            resulting_miss_distance_km: miss_km,
            resulting_pc,
            transfer_type: label.to_string(),
        };

        // Store on DHT
        let action_hash = create_entry(&EntryTypes::AvoidanceOption(option.clone()))?;
        let anchor = anchor_for_session_avoidance(&input.session_id)?;
        create_link(
            anchor,
            action_hash,
            LinkTypes::SessionAvoidanceOptions,
            LinkTag::new(format!("avoidance:{}:{}", input.session_id, label)),
        )?;

        options.push(option);
    }

    // Sort by ΔV magnitude (lowest first)
    options.sort_by(|a, b| {
        a.delta_v_magnitude_ms
            .partial_cmp(&b.delta_v_magnitude_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(options)
}

/// Get all avoidance options generated for a session.
#[hdk_extern]
pub fn get_avoidance_options(session_id: String) -> ExternResult<Vec<AvoidanceOption>> {
    let anchor = anchor_for_session_avoidance(&session_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::SessionAvoidanceOptions)?,
        GetStrategy::Network,
    )?;

    let mut options = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(opt) = record
            .entry()
            .to_app_option::<AvoidanceOption>()
            .ok()
            .flatten()
        {
            options.push(opt);
        }
    }

    // Sort by ΔV
    options.sort_by(|a, b| {
        a.delta_v_magnitude_ms
            .partial_cmp(&b.delta_v_magnitude_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(options)
}

// =============================================================================
// Fusion-aware avoidance planning
// =============================================================================

/// Input for fusion-aware avoidance generation.
///
/// Instead of requiring raw TLE strings, this takes NORAD IDs and fetches
/// the best available state from the observations zome (fused multi-sensor
/// estimate if available, with TLE fallback).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FusionAvoidanceInput {
    /// Session this is for
    pub session_id: String,
    /// Primary object NORAD ID
    pub primary_norad_id: u32,
    /// Secondary object NORAD ID
    pub secondary_norad_id: u32,
    /// Time of closest approach
    pub tca: SpaceTimestamp,
    /// Maximum allowable ΔV (m/s)
    pub max_delta_v_ms: f64,
    /// Hours before TCA to perform the burn
    pub lead_time_hours: f64,
    /// Fallback TLEs if fusion unavailable (line1, line2)
    pub primary_tle_fallback: Option<(String, String)>,
    pub secondary_tle_fallback: Option<(String, String)>,
}

/// Cross-zome response for fused state estimate.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct FusedEstimateResponse {
    pub norad_id: u32,
    pub epoch: SpaceTimestamp,
    pub state_vector: Vec<f64>,
    pub covariance_diagonal: Vec<f64>,
    pub fused_quality: f64,
}

/// Try to fetch a fused state estimate from the observations zome.
fn try_fetch_fused_state(norad_id: u32) -> Option<FusedEstimateResponse> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("observations"),
        FunctionName::new("get_fused_state"),
        None,
        norad_id,
    )
    .ok()?;

    let bytes = match response {
        ZomeCallResponse::Ok(bytes) => bytes,
        _ => return None,
    };

    bytes
        .decode::<Option<FusedEstimateResponse>>()
        .ok()
        .flatten()
}

/// Generate avoidance options using fused multi-sensor state estimates.
///
/// Attempts to use high-fidelity fused estimates from the observations zome.
/// Falls back to TLE-based propagation if fusion is unavailable or stale.
/// Returns options annotated with which state source was used.
#[hdk_extern]
pub fn generate_avoidance_from_fusion(
    input: FusionAvoidanceInput,
) -> ExternResult<Vec<AvoidanceOption>> {
    validate_string_field(&input.session_id, "session_id", 256).map_err(|e| e.into_wasm_error())?;

    if input.max_delta_v_ms <= 0.0 || !input.max_delta_v_ms.is_finite() {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            "max_delta_v_ms must be positive",
        )
        .into_wasm_error());
    }
    if input.lead_time_hours <= 0.0 || !input.lead_time_hours.is_finite() {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            "lead_time_hours must be positive",
        )
        .into_wasm_error());
    }

    // Try fusion for both objects
    let primary_fused = try_fetch_fused_state(input.primary_norad_id);
    let secondary_fused = try_fetch_fused_state(input.secondary_norad_id);

    // If we have fused states for both, check freshness relative to TCA
    let tca_micros = input.tca.micros;
    let max_age_micros: i64 = 24 * 3600 * 1_000_000; // 24 hours

    let use_fused = match (&primary_fused, &secondary_fused) {
        (Some(pf), Some(sf)) => {
            let p_age = (tca_micros - pf.epoch.micros).abs();
            let s_age = (tca_micros - sf.epoch.micros).abs();
            p_age < max_age_micros
                && s_age < max_age_micros
                && pf.fused_quality > 0.3
                && sf.fused_quality > 0.3
        }
        _ => false,
    };

    if use_fused {
        let pf = primary_fused.unwrap();
        let sf = secondary_fused.unwrap();

        if pf.state_vector.len() < 6 || sf.state_vector.len() < 6 {
            return Err(SpaceError::new(
                SpaceErrorCode::InvalidInput,
                "Fused state vector must have 6 elements",
            )
            .into_wasm_error());
        }

        // Use fused state vectors directly (already at epoch, close to TCA)
        let r_primary =
            nalgebra::Vector3::new(pf.state_vector[0], pf.state_vector[1], pf.state_vector[2]);
        let v_primary =
            nalgebra::Vector3::new(pf.state_vector[3], pf.state_vector[4], pf.state_vector[5]);
        let r_secondary =
            nalgebra::Vector3::new(sf.state_vector[0], sf.state_vector[1], sf.state_vector[2]);

        let mu = orbital_mechanics::coordinates::wgs84::MU;
        let displacement_km = 5.0;
        let tof_secs = input.lead_time_hours * 3600.0;
        let max_dv_kms = input.max_delta_v_ms / 1000.0;
        let lead_secs = (input.lead_time_hours * 3600.0) as i64;
        let burn_epoch_micros = input.tca.micros - (lead_secs * 1_000_000);
        let burn_ts = SpaceTimestamp {
            micros: burn_epoch_micros,
        };

        // Orbital frame at primary position
        let r_mag = r_primary.norm();
        let r_hat = r_primary / r_mag;
        let v_norm = v_primary.norm();
        let prograde = if v_norm > 1e-10 {
            v_primary / v_norm
        } else {
            r_hat
        };
        let normal = r_primary.cross(&v_primary);
        let n_mag = normal.norm();
        let normal = if n_mag > 1e-10 {
            normal / n_mag
        } else {
            nalgebra::Vector3::new(0.0, 0.0, 1.0)
        };
        let radial = r_hat;

        let directions = [
            ("prograde", prograde * displacement_km),
            ("retrograde", -prograde * displacement_km),
            ("radial", radial * displacement_km),
            ("normal", normal * displacement_km),
        ];

        let mut options = Vec::new();

        for (label, offset) in &directions {
            let r_target = r_primary + offset;

            let sol = match solve_lambert(&r_primary, &r_target, tof_secs, mu, false) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let dv = sol.v1 - v_primary;
            let dv_mag = dv.norm();

            if dv_mag > max_dv_kms {
                continue;
            }

            let miss_km = (r_target - r_secondary).norm();

            let option = AvoidanceOption {
                session_id: input.session_id.clone(),
                maneuver_time: burn_ts.clone(),
                delta_v: vec![dv.x, dv.y, dv.z],
                delta_v_magnitude_ms: dv_mag * 1000.0,
                resulting_miss_distance_km: miss_km,
                resulting_pc: 0.0, // Conservative: fusion path skips Alfano (no TLE)
                transfer_type: format!("{} (fused)", label),
            };

            let action_hash = create_entry(&EntryTypes::AvoidanceOption(option.clone()))?;
            let anchor = anchor_for_session_avoidance(&input.session_id)?;
            create_link(
                anchor,
                action_hash,
                LinkTypes::SessionAvoidanceOptions,
                LinkTag::new(format!("avoidance:{}:{}", input.session_id, label)),
            )?;

            options.push(option);
        }

        options.sort_by(|a, b| {
            a.delta_v_magnitude_ms
                .partial_cmp(&b.delta_v_magnitude_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        return Ok(options);
    }

    // Fallback to TLE-based avoidance
    let (p_line1, p_line2) = input.primary_tle_fallback.ok_or_else(|| {
        SpaceError::new(
            SpaceErrorCode::InvalidInput,
            "No fused state available and no TLE fallback provided for primary",
        )
        .into_wasm_error()
    })?;
    let (s_line1, s_line2) = input.secondary_tle_fallback.ok_or_else(|| {
        SpaceError::new(
            SpaceErrorCode::InvalidInput,
            "No fused state available and no TLE fallback provided for secondary",
        )
        .into_wasm_error()
    })?;

    // Delegate to TLE-based function
    generate_avoidance_options(AvoidanceInput {
        session_id: input.session_id,
        primary_tle_line1: p_line1,
        primary_tle_line2: p_line2,
        secondary_tle_line1: s_line1,
        secondary_tle_line2: s_line2,
        tca: input.tca,
        max_delta_v_ms: input.max_delta_v_ms,
        lead_time_hours: input.lead_time_hours,
    })
}

// =============================================================================
// Query operations
// =============================================================================

/// Get all negotiation sessions for a conjunction.
///
/// Queries the `sessions_for_conj.{conjunction_id}` anchor.
#[hdk_extern]
pub fn get_sessions_for_conjunction(
    conjunction_id: String,
) -> ExternResult<Vec<NegotiationSession>> {
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

/// Get all positions submitted in a negotiation session.
///
/// Queries the `positions_for.{session_id}` anchor.
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

/// Get all maneuver proposals for a negotiation session.
///
/// Queries the `proposals_for.{session_id}` anchor.
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

/// Get all negotiation sessions an operator is involved in (as primary or secondary).
///
/// Queries the `operator_sessions.{agent}` anchor.
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
pub fn get_sessions_paginated(
    input: PaginatedSessionQuery,
) -> ExternResult<PaginatedResponse<NegotiationSession>> {
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
pub fn get_operator_sessions_paginated(
    input: PaginatedOperatorQuery,
) -> ExternResult<PaginatedResponse<NegotiationSession>> {
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

// =============================================================================
// SGP4 + Alfano verification helpers
// =============================================================================

/// Mirror of orbital_objects TleLinesResponse for cross-zome deserialization.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct TleLinesResponse {
    pub norad_id: u32,
    pub line1: String,
    pub line2: String,
}

/// Attempt to independently compute collision probability for two objects at a
/// given time using SGP4 propagation and Alfano 2D Pc.
///
/// Returns `None` if TLEs are unavailable or propagation fails (best-effort).
fn verify_pc_from_tles(
    primary_norad_id: u32,
    secondary_norad_id: u32,
    burn_time: &SpaceTimestamp,
) -> Option<f64> {
    // Cross-zome call to orbital_objects for TLEs
    let all_ids: Vec<u32> = vec![primary_norad_id, secondary_norad_id];

    let response = call(
        CallTargetCell::Local,
        ZomeName::new("orbital_objects"),
        FunctionName::new("get_latest_tles"),
        None,
        all_ids,
    )
    .ok()?;

    let response_bytes = match response {
        ZomeCallResponse::Ok(bytes) => bytes,
        _ => return None,
    };

    let tle_lines: Vec<TleLinesResponse> = response_bytes.decode().ok()?;

    // Find TLEs for both objects
    let primary_tle_data = tle_lines.iter().find(|t| t.norad_id == primary_norad_id)?;
    let secondary_tle_data = tle_lines
        .iter()
        .find(|t| t.norad_id == secondary_norad_id)?;

    // Parse TLEs
    let primary_tle =
        TwoLineElement::parse_lines(None, &primary_tle_data.line1, &primary_tle_data.line2).ok()?;
    let secondary_tle =
        TwoLineElement::parse_lines(None, &secondary_tle_data.line1, &secondary_tle_data.line2)
            .ok()?;

    // Create propagators
    let primary_prop = Propagator::from_tle(&primary_tle).ok()?;
    let secondary_prop = Propagator::from_tle(&secondary_tle).ok()?;

    // Convert SpaceTimestamp (microseconds since Unix epoch) to chrono DateTime
    let secs = burn_time.micros / 1_000_000;
    let nsecs = ((burn_time.micros % 1_000_000) * 1000) as u32;
    let target_time = chrono::DateTime::from_timestamp(secs, nsecs)?;

    // Propagate both objects to the burn time
    let p_state = primary_prop.propagate_to(target_time).ok()?;
    let s_state = secondary_prop.propagate_to(target_time).ok()?;

    // Run Alfano 2D collision probability assessment
    let analyzer = ConjunctionAnalyzer::new();
    let assessment = analyzer.assess(&p_state, &s_state);

    Some(assessment.collision_probability.pc)
}

// =============================================================================
// Multi-party Negotiation (extends bilateral to N operators)
// =============================================================================

/// Anchor for conjunction proposals related to a conjunction ID.
fn anchor_for_conjunction_proposals(conjunction_id: &str) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("mp_proposals_for_conj.{}", conjunction_id));
    let typed = path.typed(LinkTypes::ConjunctionProposals)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Anchor for votes on a proposal.
fn anchor_for_proposal_votes(proposal_id: &str) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("votes_for_proposal.{}", proposal_id));
    let typed = path.typed(LinkTypes::ProposalToVotes)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Anchor for agreements on a proposal.
fn anchor_for_proposal_agreements(proposal_id: &str) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("agreements_for_proposal.{}", proposal_id));
    let typed = path.typed(LinkTypes::ProposalAgreements)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Anchor for proposals an operator is part of.
fn anchor_for_operator_proposals(agent: &AgentPubKey) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("operator_mp_proposals.{}", agent));
    let typed = path.typed(LinkTypes::OperatorProposals)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

// --- Input/Output types ---

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateProposalInput {
    pub proposal_id: String,
    pub conjunction_id: String,
    pub affected_operators: Vec<AgentPubKey>,
    pub tca: i64,
    pub primary_norad_id: u32,
    pub secondary_norad_ids: Vec<u32>,
    pub maneuver_options: Vec<ManeuverOption>,
    pub voting_deadline: i64,
    pub quorum_threshold: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoteInput {
    pub proposal_hash: ActionHash,
    pub proposal_id: String,
    pub preferred_option_index: u32,
    pub vote_weight: f64,
    pub justification: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposalTally {
    pub proposal_id: String,
    pub total_votes: usize,
    pub total_weight: f64,
    pub quorum_met: bool,
    pub option_tallies: Vec<OptionTally>,
    pub winner: Option<u32>,
    pub status: MultiPartyProposalStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptionTally {
    pub option_index: u32,
    pub vote_count: usize,
    pub total_weight: f64,
}

// --- Signal extensions ---

/// Extended signal types for multi-party negotiation.
///
/// These are emitted alongside the existing `TrafficControlSignal` variants.
#[derive(Clone, Debug, Serialize, Deserialize, SerializedBytes)]
pub enum MultiPartySignal {
    /// A new multi-party proposal was created
    ProposalCreated {
        proposal_id: String,
        conjunction_id: String,
        affected_operators_count: usize,
        voting_deadline: i64,
    },
    /// An operator cast a vote on a proposal
    VoteCast {
        proposal_id: String,
        voter: AgentPubKey,
        option_index: u32,
        vote_weight: f64,
    },
    /// Vote tally completed for a proposal
    TallyCompleted {
        proposal_id: String,
        quorum_met: bool,
        winning_option: Option<u32>,
        total_votes: usize,
    },
    /// Multi-party agreement created after successful tally
    AgreementCreated {
        proposal_id: String,
        conjunction_id: String,
        approved_option_index: u32,
        approving_operators_count: usize,
    },
}

// --- Coordinator functions ---

/// Create a multi-party conjunction proposal with pre-computed maneuver options.
///
/// All affected operators are notified via signal. Requires Citizen tier
/// (coordination authority) via consciousness gating.
#[hdk_extern]
pub fn create_conjunction_proposal(input: CreateProposalInput) -> ExternResult<ActionHash> {
    gate_space_operation(
        &requirement_for_negotiation(),
        "create_conjunction_proposal",
    )?;

    // --- Input validation ---
    validate_string_field(&input.proposal_id, "proposal_id", 256)
        .map_err(|e| e.into_wasm_error())?;
    validate_string_field(&input.conjunction_id, "conjunction_id", 256)
        .map_err(|e| e.into_wasm_error())?;

    if input.affected_operators.len() < 2 {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            format!(
                "At least 2 affected operators required, got {}",
                input.affected_operators.len()
            ),
        )
        .into_wasm_error());
    }

    if input.affected_operators.len() > 64 {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            format!(
                "Too many affected operators: {} (max 64)",
                input.affected_operators.len()
            ),
        )
        .into_wasm_error());
    }

    if input.maneuver_options.len() < 2 {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            format!(
                "At least 2 maneuver options required, got {}",
                input.maneuver_options.len()
            ),
        )
        .into_wasm_error());
    }

    if input.maneuver_options.len() > 32 {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            format!(
                "Too many maneuver options: {} (max 32)",
                input.maneuver_options.len()
            ),
        )
        .into_wasm_error());
    }

    if !input.quorum_threshold.is_finite()
        || input.quorum_threshold < 0.5
        || input.quorum_threshold > 1.0
    {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            format!(
                "quorum_threshold must be between 0.5 and 1.0, got {}",
                input.quorum_threshold
            ),
        )
        .into_wasm_error());
    }

    // Voting deadline must be in the future
    let now_micros = SpaceTimestamp::now().micros;
    if input.voting_deadline <= now_micros {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            "voting_deadline must be in the future",
        )
        .into_wasm_error());
    }

    // Validate NORAD IDs
    if input.primary_norad_id == 0 || input.primary_norad_id > 999999 {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidNoradId,
            format!("Invalid primary NORAD ID: {}", input.primary_norad_id),
        )
        .into_wasm_error());
    }
    for &norad in &input.secondary_norad_ids {
        if norad == 0 || norad > 999999 {
            return Err(SpaceError::new(
                SpaceErrorCode::InvalidNoradId,
                format!("Invalid secondary NORAD ID: {}", norad),
            )
            .into_wasm_error());
        }
    }
    if input.secondary_norad_ids.contains(&input.primary_norad_id) {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            "Primary NORAD ID must not appear in secondary list",
        )
        .into_wasm_error());
    }

    // Validate maneuver options
    for opt in &input.maneuver_options {
        if !opt.delta_v_ms.is_finite() || opt.delta_v_ms < 0.0 {
            return Err(SpaceError::new(
                SpaceErrorCode::InvalidProposal,
                format!(
                    "delta_v_ms must be non-negative and finite in option {}, got {}",
                    opt.option_index, opt.delta_v_ms
                ),
            )
            .into_wasm_error());
        }
        if !opt.post_maneuver_pc.is_finite()
            || opt.post_maneuver_pc < 0.0
            || opt.post_maneuver_pc > 1.0
        {
            return Err(SpaceError::new(
                SpaceErrorCode::InvalidProbability,
                format!(
                    "post_maneuver_pc must be between 0 and 1 in option {}, got {}",
                    opt.option_index, opt.post_maneuver_pc
                ),
            )
            .into_wasm_error());
        }
    }

    let agent = agent_info()?.agent_initial_pubkey;

    let proposal = ConjunctionProposal {
        proposal_id: input.proposal_id.clone(),
        conjunction_id: input.conjunction_id.clone(),
        affected_operators: input.affected_operators.clone(),
        tca: input.tca,
        primary_norad_id: input.primary_norad_id,
        secondary_norad_ids: input.secondary_norad_ids,
        maneuver_options: input.maneuver_options,
        voting_deadline: input.voting_deadline,
        status: MultiPartyProposalStatus::Voting,
        quorum_threshold: input.quorum_threshold,
        created_at: now_micros,
        created_by: agent,
    };

    let action_hash = create_entry(&EntryTypes::ConjunctionProposal(proposal))?;

    // Link: conjunction → proposal
    let conj_anchor = anchor_for_conjunction_proposals(&input.conjunction_id)?;
    create_link(
        conj_anchor,
        action_hash.clone(),
        LinkTypes::ConjunctionProposals,
        LinkTag::new(format!("mp_proposal:{}", input.proposal_id)),
    )?;

    // Link: each operator → proposal
    for operator in &input.affected_operators {
        let op_anchor = anchor_for_operator_proposals(operator)?;
        create_link(
            op_anchor,
            action_hash.clone(),
            LinkTypes::OperatorProposals,
            LinkTag::new(format!("mp_proposal:{}", input.proposal_id)),
        )?;
    }

    // Emit signal
    emit_signal(MultiPartySignal::ProposalCreated {
        proposal_id: input.proposal_id,
        conjunction_id: input.conjunction_id,
        affected_operators_count: input.affected_operators.len(),
        voting_deadline: input.voting_deadline,
    })?;

    Ok(action_hash)
}

/// Cast a weighted vote on a conjunction proposal.
///
/// The voter must be one of the affected operators. Double voting is prevented.
#[hdk_extern]
pub fn vote_on_proposal(input: VoteInput) -> ExternResult<ActionHash> {
    gate_space_operation(&requirement_for_observation(), "vote_on_proposal")?;

    // --- Input validation ---
    validate_string_field(&input.proposal_id, "proposal_id", 256)
        .map_err(|e| e.into_wasm_error())?;

    if !input.vote_weight.is_finite() || input.vote_weight < 0.0 || input.vote_weight > 1.0 {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            format!(
                "vote_weight must be between 0.0 and 1.0, got {}",
                input.vote_weight
            ),
        )
        .into_wasm_error());
    }

    if let Some(ref j) = input.justification {
        if j.len() > 2048 {
            return Err(SpaceError::new(
                SpaceErrorCode::InvalidInput,
                format!("Justification too long: {} chars (max 2048)", j.len()),
            )
            .into_wasm_error());
        }
    }

    let agent = agent_info()?.agent_initial_pubkey;

    // Fetch the proposal
    let record = get(input.proposal_hash.clone(), GetOptions::default())?.ok_or(
        SpaceError::new(SpaceErrorCode::NotFound, "Proposal not found")
            .with_context(format!("hash: {}", input.proposal_hash))
            .into_wasm_error(),
    )?;

    let proposal: ConjunctionProposal = record
        .entry()
        .to_app_option()
        .map_err(|e| {
            SpaceError::new(
                SpaceErrorCode::InvalidInput,
                format!("Failed to deserialize proposal: {:?}", e),
            )
            .into_wasm_error()
        })?
        .ok_or(
            SpaceError::new(
                SpaceErrorCode::InvalidInput,
                "Entry is not a ConjunctionProposal",
            )
            .into_wasm_error(),
        )?;

    // Verify proposal_id matches
    if proposal.proposal_id != input.proposal_id {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            "proposal_id does not match the proposal at the given hash",
        )
        .into_wasm_error());
    }

    // Verify proposal is in Voting status
    if proposal.status != MultiPartyProposalStatus::Voting {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidStateTransition,
            format!(
                "Proposal is not in Voting status (current: {:?})",
                proposal.status
            ),
        )
        .into_wasm_error());
    }

    // Verify voting deadline not passed
    let now_micros = SpaceTimestamp::now().micros;
    if now_micros > proposal.voting_deadline {
        return Err(
            SpaceError::new(SpaceErrorCode::InvalidInput, "Voting deadline has passed")
                .into_wasm_error(),
        );
    }

    // Verify voter is an affected operator
    if !proposal.affected_operators.contains(&agent) {
        return Err(SpaceError::new(
            SpaceErrorCode::Unauthorized,
            "You are not an affected operator for this proposal",
        )
        .into_wasm_error());
    }

    // Verify option_index is valid
    if !proposal
        .maneuver_options
        .iter()
        .any(|o| o.option_index == input.preferred_option_index)
    {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            format!(
                "Invalid option_index: {} (not found in proposal maneuver options)",
                input.preferred_option_index
            ),
        )
        .into_wasm_error());
    }

    // Prevent double voting — check existing votes
    let votes_anchor = anchor_for_proposal_votes(&input.proposal_id)?;
    let existing_links = get_links(
        LinkQuery::try_new(votes_anchor.clone(), LinkTypes::ProposalToVotes)?,
        GetStrategy::Network,
    )?;

    for link in &existing_links {
        let Some(target) = link.target.clone().into_action_hash() else {
            continue;
        };
        let Some(rec) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(existing_vote) = rec.entry().to_app_option::<OperatorVote>().ok().flatten() {
            if existing_vote.voter == agent && existing_vote.proposal_id == input.proposal_id {
                return Err(SpaceError::new(
                    SpaceErrorCode::AlreadySigned,
                    "You have already voted on this proposal",
                )
                .into_wasm_error());
            }
        }
    }

    let vote = OperatorVote {
        proposal_id: input.proposal_id.clone(),
        voter: agent.clone(),
        preferred_option_index: input.preferred_option_index,
        vote_weight: input.vote_weight,
        justification: input.justification,
        voted_at: now_micros,
    };

    let vote_hash = create_entry(&EntryTypes::OperatorVote(vote))?;

    // Link: proposal → vote
    create_link(
        votes_anchor,
        vote_hash.clone(),
        LinkTypes::ProposalToVotes,
        LinkTag::new(format!("vote:{}:{}", input.proposal_id, agent)),
    )?;

    // Emit signal
    emit_signal(MultiPartySignal::VoteCast {
        proposal_id: input.proposal_id.clone(),
        voter: agent,
        option_index: input.preferred_option_index,
        vote_weight: input.vote_weight,
    })?;

    // Check if quorum reached — auto-tally if all operators have voted
    let vote_count = existing_links.len() + 1; // +1 for the vote we just created
    if vote_count >= proposal.affected_operators.len() {
        // All operators voted; tally will be definitive
        let _ = tally_proposal(input.proposal_hash);
    }

    Ok(vote_hash)
}

/// Tally votes on a proposal and determine outcome.
///
/// Called automatically when all operators have voted, or can be called
/// manually after the voting deadline to finalize.
#[hdk_extern]
pub fn tally_proposal(proposal_hash: ActionHash) -> ExternResult<ProposalTally> {
    // Fetch the proposal
    let record = get(proposal_hash.clone(), GetOptions::default())?.ok_or(
        SpaceError::new(SpaceErrorCode::NotFound, "Proposal not found")
            .with_context(format!("hash: {}", proposal_hash))
            .into_wasm_error(),
    )?;

    let proposal: ConjunctionProposal = record
        .entry()
        .to_app_option()
        .map_err(|e| {
            SpaceError::new(
                SpaceErrorCode::InvalidInput,
                format!("Failed to deserialize proposal: {:?}", e),
            )
            .into_wasm_error()
        })?
        .ok_or(
            SpaceError::new(
                SpaceErrorCode::InvalidInput,
                "Entry is not a ConjunctionProposal",
            )
            .into_wasm_error(),
        )?;

    // Only tally proposals that are still in Voting status
    if proposal.status != MultiPartyProposalStatus::Voting {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidStateTransition,
            format!(
                "Proposal is not in Voting status (current: {:?})",
                proposal.status
            ),
        )
        .into_wasm_error());
    }

    // Fetch all votes
    let votes_anchor = anchor_for_proposal_votes(&proposal.proposal_id)?;
    let vote_links = get_links(
        LinkQuery::try_new(votes_anchor, LinkTypes::ProposalToVotes)?,
        GetStrategy::Network,
    )?;

    let mut votes: Vec<OperatorVote> = Vec::new();
    for link in vote_links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(rec) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(vote) = rec.entry().to_app_option::<OperatorVote>().ok().flatten() {
            if vote.proposal_id == proposal.proposal_id {
                votes.push(vote);
            }
        }
    }

    // Build option tallies
    let mut option_weight_map: std::collections::HashMap<u32, (usize, f64)> =
        std::collections::HashMap::new();
    let mut total_weight = 0.0;

    for vote in &votes {
        let entry = option_weight_map
            .entry(vote.preferred_option_index)
            .or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += vote.vote_weight;
        total_weight += vote.vote_weight;
    }

    let mut option_tallies: Vec<OptionTally> = option_weight_map
        .iter()
        .map(|(&option_index, &(vote_count, weight))| OptionTally {
            option_index,
            vote_count,
            total_weight: weight,
        })
        .collect();

    // Sort by total weight descending
    option_tallies.sort_by(|a, b| {
        b.total_weight
            .partial_cmp(&a.total_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Determine winner (highest weighted vote total)
    let winner = option_tallies.first().map(|t| t.option_index);

    // Check quorum: total_weight / max_possible_weight >= threshold
    // max_possible_weight = number of affected operators (each can contribute up to 1.0)
    let max_possible_weight = proposal.affected_operators.len() as f64;
    let quorum_met = max_possible_weight > 0.0
        && (total_weight / max_possible_weight) >= proposal.quorum_threshold;

    // Determine final status
    let now_micros = SpaceTimestamp::now().micros;
    let deadline_passed = now_micros > proposal.voting_deadline;

    let final_status = if quorum_met {
        MultiPartyProposalStatus::Approved
    } else if deadline_passed {
        MultiPartyProposalStatus::Expired
    } else {
        // Not enough votes yet and deadline hasn't passed — stay in Voting
        MultiPartyProposalStatus::Voting
    };

    // If finalized (Approved or Expired), update proposal status
    if final_status != MultiPartyProposalStatus::Voting {
        // Validate state transition using the state machine
        if !proposal.status.is_valid_transition(&final_status) {
            return Err(SpaceError::new(
                SpaceErrorCode::InvalidStateTransition,
                format!(
                    "Invalid proposal status transition: {:?} -> {:?}",
                    proposal.status, final_status
                ),
            )
            .into_wasm_error());
        }

        let mut updated_proposal = proposal.clone();
        updated_proposal.status = final_status.clone();
        update_entry(proposal_hash.clone(), &updated_proposal)?;

        // If approved, create a MultiPartyAgreement
        if final_status == MultiPartyProposalStatus::Approved {
            if let Some(winning_index) = winner {
                // Collect operators who voted for the winning option
                let approving_operators: Vec<AgentPubKey> = votes
                    .iter()
                    .filter(|v| v.preferred_option_index == winning_index)
                    .map(|v| v.voter.clone())
                    .collect();

                let winning_weight = option_tallies
                    .iter()
                    .find(|t| t.option_index == winning_index)
                    .map(|t| t.total_weight)
                    .unwrap_or(0.0);

                // Default execution deadline: 24 hours from now
                let execution_deadline = now_micros + (24 * 3600 * 1_000_000);

                let agreement = MultiPartyAgreement {
                    proposal_id: proposal.proposal_id.clone(),
                    conjunction_id: proposal.conjunction_id.clone(),
                    approved_option_index: winning_index,
                    approving_operators: approving_operators.clone(),
                    total_vote_weight: winning_weight,
                    quorum_met: true,
                    execution_deadline,
                    created_at: now_micros,
                };

                let agr_hash = create_entry(&EntryTypes::MultiPartyAgreement(agreement))?;

                // Link: proposal → agreement
                let agr_anchor = anchor_for_proposal_agreements(&proposal.proposal_id)?;
                create_link(
                    agr_anchor,
                    agr_hash,
                    LinkTypes::ProposalAgreements,
                    LinkTag::new(format!("agreement:{}", proposal.proposal_id)),
                )?;

                emit_signal(MultiPartySignal::AgreementCreated {
                    proposal_id: proposal.proposal_id.clone(),
                    conjunction_id: proposal.conjunction_id.clone(),
                    approved_option_index: winning_index,
                    approving_operators_count: approving_operators.len(),
                })?;
            }
        }
    }

    // Emit tally signal
    emit_signal(MultiPartySignal::TallyCompleted {
        proposal_id: proposal.proposal_id.clone(),
        quorum_met,
        winning_option: winner,
        total_votes: votes.len(),
    })?;

    Ok(ProposalTally {
        proposal_id: proposal.proposal_id,
        total_votes: votes.len(),
        total_weight,
        quorum_met,
        option_tallies,
        winner,
        status: final_status,
    })
}

/// Get all multi-party proposals for a conjunction.
#[hdk_extern]
pub fn get_proposals_for_conjunction(
    conjunction_id: String,
) -> ExternResult<Vec<ConjunctionProposal>> {
    let anchor = anchor_for_conjunction_proposals(&conjunction_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ConjunctionProposals)?,
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
        if let Some(cp) = record
            .entry()
            .to_app_option::<ConjunctionProposal>()
            .ok()
            .flatten()
        {
            proposals.push(cp);
        }
    }

    Ok(proposals)
}

/// Get all votes for a multi-party proposal.
#[hdk_extern]
pub fn get_votes_for_proposal(proposal_id: String) -> ExternResult<Vec<OperatorVote>> {
    let anchor = anchor_for_proposal_votes(&proposal_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ProposalToVotes)?,
        GetStrategy::Network,
    )?;

    let mut votes = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(vote) = record
            .entry()
            .to_app_option::<OperatorVote>()
            .ok()
            .flatten()
        {
            if vote.proposal_id == proposal_id {
                votes.push(vote);
            }
        }
    }

    Ok(votes)
}

/// Get all multi-party proposals an operator is part of.
#[hdk_extern]
pub fn get_operator_proposals(operator: AgentPubKey) -> ExternResult<Vec<ConjunctionProposal>> {
    let anchor = anchor_for_operator_proposals(&operator)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::OperatorProposals)?,
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
        if let Some(cp) = record
            .entry()
            .to_app_option::<ConjunctionProposal>()
            .ok()
            .flatten()
        {
            proposals.push(cp);
        }
    }

    Ok(proposals)
}

/// Get the multi-party agreement for a proposal (if approved).
#[hdk_extern]
pub fn get_proposal_agreement(proposal_id: String) -> ExternResult<Option<MultiPartyAgreement>> {
    let anchor = anchor_for_proposal_agreements(&proposal_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ProposalAgreements)?,
        GetStrategy::Network,
    )?;

    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(mpa) = record
            .entry()
            .to_app_option::<MultiPartyAgreement>()
            .ok()
            .flatten()
        {
            if mpa.proposal_id == proposal_id {
                return Ok(Some(mpa));
            }
        }
    }

    Ok(None)
}
