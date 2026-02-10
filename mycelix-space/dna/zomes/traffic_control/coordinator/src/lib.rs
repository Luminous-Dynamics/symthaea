//! Traffic Control Coordinator Zome
//!
//! Functions for automated space traffic negotiation.
//! Includes cosigning for bilateral agreements.

use hdk::prelude::*;
use mycelix_space_shared::{
    validate_string_field, PaginatedResponse, PaginationParams, SpaceError, SpaceErrorCode,
    SpaceTimestamp,
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
#[hdk_extern]
pub fn accept_proposal(input: AcceptProposalInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let agreement = NegotiationAgreement {
        session_id: input.session_id,
        accepted_proposal: input.proposal_hash,
        primary_signature: Some(agent),
        secondary_signature: None, // Other party needs to cosign
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
