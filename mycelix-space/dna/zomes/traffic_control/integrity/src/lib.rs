//! Traffic Control Integrity Zome
//!
//! Implements automated space traffic coordination through
//! AI-mediated negotiation between operators. When a conjunction
//! is detected, the involved operators' AI agents negotiate
//! who should maneuver and how.
//!
//! # The Negotiation Protocol
//!
//! 1. **Conjunction Detected**: System identifies close approach
//! 2. **Negotiation Initiated**: Both operators receive notification
//! 3. **Position Exchange**: Each side shares their constraints/preferences
//! 4. **Proposal Generation**: AI generates maneuver options
//! 5. **Agreement**: Both sides sign off on solution
//! 6. **Execution**: Chosen operator executes maneuver
//! 7. **Confirmation**: Network verifies new orbits are safe

use hdi::prelude::*;
use mycelix_space_shared::SpaceTimestamp;

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// Negotiation session for a conjunction
    NegotiationSession(NegotiationSession),

    /// Operator's position in negotiation
    NegotiationPosition(NegotiationPosition),

    /// Proposed solution
    ManeuverProposal(ManeuverProposal),

    /// Agreement on a solution
    NegotiationAgreement(NegotiationAgreement),

    /// Lambert-computed avoidance maneuver option
    AvoidanceOption(AvoidanceOption),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Sessions for a conjunction
    ConjunctionSessions,
    /// Positions in a session
    SessionPositions,
    /// Proposals in a session
    SessionProposals,
    /// Active sessions for an operator
    OperatorSessions,
    /// Avoidance options for a session
    SessionAvoidanceOptions,
}

/// A negotiation session between operators
#[hdk_entry_helper]
#[derive(Clone)]
pub struct NegotiationSession {
    /// Unique session ID
    pub session_id: String,

    /// Related conjunction event ID
    pub conjunction_id: String,

    /// Primary object operator
    pub primary_operator: AgentPubKey,

    /// Secondary object operator
    pub secondary_operator: AgentPubKey,

    /// Primary object NORAD ID
    pub primary_norad_id: u32,

    /// Secondary object NORAD ID
    pub secondary_norad_id: u32,

    /// TCA for this conjunction
    pub tca: SpaceTimestamp,

    /// Session status
    pub status: SessionStatus,

    /// Deadline for reaching agreement
    pub deadline: SpaceTimestamp,

    /// Created at
    pub created_at: SpaceTimestamp,
}

/// Session status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SessionStatus {
    /// Waiting for both parties to join
    Pending,
    /// Both parties engaged, exchanging positions
    Active,
    /// Proposals being evaluated
    Proposing,
    /// Agreement reached
    Agreed,
    /// Maneuver executed
    Executed,
    /// Failed to reach agreement
    Failed,
    /// Timed out
    Expired,
}

/// Operator's position/constraints for negotiation
#[hdk_entry_helper]
#[derive(Clone)]
pub struct NegotiationPosition {
    /// Session this position is for
    pub session_id: String,

    /// Operator submitting position
    pub operator: AgentPubKey,

    /// Object they're representing
    pub norad_id: u32,

    /// Maneuver capability
    pub maneuver_capability: ManeuverCapability,

    /// Preferences
    pub preferences: OperatorPreferences,

    /// Submitted at
    pub submitted_at: SpaceTimestamp,
}

/// Operator's maneuver capabilities
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManeuverCapability {
    /// Can this object maneuver?
    pub can_maneuver: bool,

    /// Maximum delta-V available (m/s)
    pub max_delta_v_ms: Option<f64>,

    /// Minimum lead time for maneuver (hours)
    pub min_lead_time_hours: Option<f64>,

    /// Fuel status (0-100%)
    pub fuel_percentage: Option<f64>,

    /// Other constraints
    pub constraints: Vec<String>,
}

/// Operator preferences for resolution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperatorPreferences {
    /// Willingness to maneuver (0-100)
    pub willingness_to_maneuver: u8,

    /// Acceptable risk levels
    pub acceptable_risk: AcceptableRisk,

    /// Preferred maneuver timing
    pub preferred_timing: Option<SpaceTimestamp>,

    /// Maximum acceptable collision probability
    pub max_acceptable_pc: f64,
}

/// Acceptable risk threshold
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum AcceptableRisk {
    /// Conservative (Pc < 1e-6)
    Conservative,
    /// Standard (Pc < 1e-5)
    Standard,
    /// Relaxed (Pc < 1e-4)
    Relaxed,
}

/// A proposed maneuver solution
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ManeuverProposal {
    /// Session this proposal is for
    pub session_id: String,

    /// Who generated this proposal
    pub proposer: AgentPubKey,

    /// Object that would maneuver
    pub maneuvering_object: u32,

    /// Proposed burn time
    pub burn_time: SpaceTimestamp,

    /// Proposed delta-V (m/s)
    pub delta_v_ms: f64,

    /// Direction (unit vector)
    pub direction: [f64; 3],

    /// Resulting miss distance (km)
    pub resulting_miss_km: f64,

    /// Resulting collision probability
    pub resulting_pc: f64,

    /// Cost estimate
    pub cost_estimate: Option<CostEstimate>,

    /// Proposal status
    pub status: ProposalStatus,

    /// Created at
    pub created_at: SpaceTimestamp,
}

/// Cost estimate for a maneuver
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Delta-V cost
    pub delta_v_ms: f64,

    /// Mission impact (days of lifetime lost, etc.)
    pub mission_impact: String,

    /// Monetary cost estimate (optional)
    pub monetary_cost: Option<u64>,

    /// Currency
    pub currency: Option<String>,
}

/// Proposal status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ProposalStatus {
    /// Pending review
    Pending,
    /// Accepted by one party
    PartiallyAccepted,
    /// Accepted by all parties
    Accepted,
    /// Rejected
    Rejected,
    /// Superseded by better proposal
    Superseded,
}

/// A Lambert-computed avoidance maneuver option for a conjunction.
///
/// Generated by `generate_avoidance_options()` — each option represents a
/// different direction (prograde, retrograde, radial, normal) with the ΔV
/// required and the resulting post-maneuver miss distance and Pc.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct AvoidanceOption {
    /// Session this option was generated for
    pub session_id: String,

    /// When the maneuver burn would occur
    pub maneuver_time: SpaceTimestamp,

    /// Delta-V vector [dvx, dvy, dvz] in km/s (TEME)
    pub delta_v: Vec<f64>,

    /// Total ΔV magnitude in m/s
    pub delta_v_magnitude_ms: f64,

    /// Resulting miss distance after maneuver (km)
    pub resulting_miss_distance_km: f64,

    /// Resulting collision probability after maneuver
    pub resulting_pc: f64,

    /// Transfer type: "prograde", "retrograde", "radial", or "normal"
    pub transfer_type: String,
}

/// Agreement between operators
#[hdk_entry_helper]
#[derive(Clone)]
pub struct NegotiationAgreement {
    /// Session this agreement is for
    pub session_id: String,

    /// Accepted proposal
    pub accepted_proposal: ActionHash,

    /// Primary operator signature
    pub primary_signature: Option<AgentPubKey>,

    /// Secondary operator signature
    pub secondary_signature: Option<AgentPubKey>,

    /// Agreement time
    pub agreed_at: SpaceTimestamp,

    /// Execution deadline
    pub execution_deadline: SpaceTimestamp,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::NegotiationSession(session) => validate_session(&session),
            EntryTypes::NegotiationPosition(pos) => validate_position(&pos),
            EntryTypes::ManeuverProposal(prop) => validate_proposal(&prop),
            EntryTypes::NegotiationAgreement(agr) => validate_agreement(&agr),
            EntryTypes::AvoidanceOption(opt) => validate_avoidance_option(&opt),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_session(session: &NegotiationSession) -> ExternResult<ValidateCallbackResult> {
    // Session ID must not be empty
    if session.session_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Session ID cannot be empty".to_string(),
        ));
    }

    // Conjunction ID must not be empty
    if session.conjunction_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Conjunction ID cannot be empty".to_string(),
        ));
    }

    // Both NORAD IDs must be valid
    if session.primary_norad_id == 0 || session.primary_norad_id > 999999 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Invalid primary NORAD ID: {}",
            session.primary_norad_id
        )));
    }
    if session.secondary_norad_id == 0 || session.secondary_norad_id > 999999 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Invalid secondary NORAD ID: {}",
            session.secondary_norad_id
        )));
    }

    // Objects must be different
    if session.primary_norad_id == session.secondary_norad_id {
        return Ok(ValidateCallbackResult::Invalid(
            "Primary and secondary NORAD IDs must be different".to_string(),
        ));
    }

    // Operators must be different
    if session.primary_operator == session.secondary_operator {
        return Ok(ValidateCallbackResult::Invalid(
            "Primary and secondary operators must be different".to_string(),
        ));
    }

    // Deadline must be after TCA (negotiation window)
    if session.deadline.micros < session.tca.micros {
        return Ok(ValidateCallbackResult::Invalid(
            "Deadline cannot be before TCA".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_position(pos: &NegotiationPosition) -> ExternResult<ValidateCallbackResult> {
    // Session ID must not be empty
    if pos.session_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Session ID cannot be empty".to_string(),
        ));
    }

    // NORAD ID must be valid
    if pos.norad_id == 0 || pos.norad_id > 999999 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Invalid NORAD ID: {}",
            pos.norad_id
        )));
    }

    // Validate maneuver capability fields
    if let Some(max_dv) = pos.maneuver_capability.max_delta_v_ms {
        if !max_dv.is_finite() || max_dv < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "max_delta_v_ms must be non-negative and finite, got {}",
                max_dv
            )));
        }
    }

    if let Some(lead_time) = pos.maneuver_capability.min_lead_time_hours {
        if !lead_time.is_finite() || lead_time < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "min_lead_time_hours must be non-negative and finite, got {}",
                lead_time
            )));
        }
    }

    if let Some(fuel) = pos.maneuver_capability.fuel_percentage {
        if !fuel.is_finite() || !(0.0..=100.0).contains(&fuel) {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "fuel_percentage must be between 0 and 100, got {}",
                fuel
            )));
        }
    }

    // Validate preferences
    if pos.preferences.willingness_to_maneuver > 100 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "willingness_to_maneuver must be 0-100, got {}",
            pos.preferences.willingness_to_maneuver
        )));
    }

    if !pos.preferences.max_acceptable_pc.is_finite()
        || pos.preferences.max_acceptable_pc < 0.0
        || pos.preferences.max_acceptable_pc > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "max_acceptable_pc must be between 0 and 1, got {}",
            pos.preferences.max_acceptable_pc
        )));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_proposal(prop: &ManeuverProposal) -> ExternResult<ValidateCallbackResult> {
    // Session ID must not be empty
    if prop.session_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Session ID cannot be empty".to_string(),
        ));
    }

    // NORAD ID must be valid
    if prop.maneuvering_object == 0 || prop.maneuvering_object > 999999 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Invalid maneuvering object NORAD ID: {}",
            prop.maneuvering_object
        )));
    }

    // Delta-V must be positive and finite
    if !prop.delta_v_ms.is_finite() || prop.delta_v_ms <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Delta-V must be a positive finite number, got {}",
            prop.delta_v_ms
        )));
    }

    // Direction must be finite and a unit vector
    for &component in &prop.direction {
        if !component.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "Direction components must be finite numbers".to_string(),
            ));
        }
    }
    let mag_sq: f64 = prop.direction.iter().map(|x| x * x).sum();
    if (mag_sq - 1.0).abs() > 0.01 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Direction must be a unit vector (magnitude squared: {})",
            mag_sq
        )));
    }

    // Resulting miss distance must be non-negative and finite
    if !prop.resulting_miss_km.is_finite() || prop.resulting_miss_km < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Resulting miss distance must be non-negative and finite, got {}",
            prop.resulting_miss_km
        )));
    }

    // Resulting Pc must be in [0, 1]
    if !prop.resulting_pc.is_finite() || prop.resulting_pc < 0.0 || prop.resulting_pc > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Resulting Pc must be between 0 and 1, got {}",
            prop.resulting_pc
        )));
    }

    // Validate cost estimate if present
    if let Some(ref cost) = prop.cost_estimate {
        if !cost.delta_v_ms.is_finite() || cost.delta_v_ms < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Cost estimate delta_v_ms must be non-negative and finite, got {}",
                cost.delta_v_ms
            )));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_agreement(agr: &NegotiationAgreement) -> ExternResult<ValidateCallbackResult> {
    // Session ID must not be empty
    if agr.session_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Session ID cannot be empty".to_string(),
        ));
    }

    // At least the primary signature must be present (creator signs first)
    if agr.primary_signature.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Agreement must have at least a primary signature".to_string(),
        ));
    }

    // If both signatures are present, they must be different
    if let (Some(ref primary), Some(ref secondary)) =
        (&agr.primary_signature, &agr.secondary_signature)
    {
        if primary == secondary {
            return Ok(ValidateCallbackResult::Invalid(
                "Primary and secondary signatures must be from different agents".to_string(),
            ));
        }
    }

    // Execution deadline must be after agreement time
    if agr.execution_deadline.micros < agr.agreed_at.micros {
        return Ok(ValidateCallbackResult::Invalid(
            "Execution deadline cannot be before agreement time".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_avoidance_option(opt: &AvoidanceOption) -> ExternResult<ValidateCallbackResult> {
    if opt.session_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Session ID cannot be empty".to_string(),
        ));
    }

    // Delta-V vector must have exactly 3 elements
    if opt.delta_v.len() != 3 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "delta_v must have 3 elements, got {}",
            opt.delta_v.len()
        )));
    }

    // All delta-V elements must be finite
    for (i, &v) in opt.delta_v.iter().enumerate() {
        if !v.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "delta_v[{}] is not finite: {}",
                i, v
            )));
        }
    }

    // Magnitude must be positive and finite
    if !opt.delta_v_magnitude_ms.is_finite() || opt.delta_v_magnitude_ms <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "delta_v_magnitude_ms must be positive and finite, got {}",
            opt.delta_v_magnitude_ms
        )));
    }

    // Miss distance must be non-negative and finite
    if !opt.resulting_miss_distance_km.is_finite() || opt.resulting_miss_distance_km < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "resulting_miss_distance_km must be non-negative and finite, got {}",
            opt.resulting_miss_distance_km
        )));
    }

    // Pc must be in [0, 1]
    if !opt.resulting_pc.is_finite() || opt.resulting_pc < 0.0 || opt.resulting_pc > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "resulting_pc must be between 0 and 1, got {}",
            opt.resulting_pc
        )));
    }

    // Transfer type must be one of the valid types
    let valid_types = ["prograde", "retrograde", "radial", "normal"];
    if !valid_types.contains(&opt.transfer_type.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "transfer_type must be one of {:?}, got '{}'",
            valid_types, opt.transfer_type
        )));
    }

    Ok(ValidateCallbackResult::Valid)
}
