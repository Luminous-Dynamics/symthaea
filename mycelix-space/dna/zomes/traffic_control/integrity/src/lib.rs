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
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::NegotiationSession(session) => validate_session(&session),
                    EntryTypes::NegotiationPosition(pos) => validate_position(&pos),
                    EntryTypes::ManeuverProposal(prop) => validate_proposal(&prop),
                    EntryTypes::NegotiationAgreement(agr) => validate_agreement(&agr),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_session(session: &NegotiationSession) -> ExternResult<ValidateCallbackResult> {
    // Both NORAD IDs must be valid
    if session.primary_norad_id == 0 || session.primary_norad_id > 999999 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid primary NORAD ID".to_string()
        ));
    }
    if session.secondary_norad_id == 0 || session.secondary_norad_id > 999999 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid secondary NORAD ID".to_string()
        ));
    }

    // Operators must be different
    if session.primary_operator == session.secondary_operator {
        return Ok(ValidateCallbackResult::Invalid(
            "Primary and secondary operators must be different".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_position(pos: &NegotiationPosition) -> ExternResult<ValidateCallbackResult> {
    // NORAD ID must be valid
    if pos.norad_id == 0 || pos.norad_id > 999999 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid NORAD ID".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_proposal(prop: &ManeuverProposal) -> ExternResult<ValidateCallbackResult> {
    // Delta-V must be positive
    if prop.delta_v_ms <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Delta-V must be positive".to_string()
        ));
    }

    // Direction must be unit vector
    let mag_sq: f64 = prop.direction.iter().map(|x| x * x).sum();
    if (mag_sq - 1.0).abs() > 0.01 {
        return Ok(ValidateCallbackResult::Invalid(
            "Direction must be a unit vector".to_string()
        ));
    }

    // Resulting values must be non-negative
    if prop.resulting_miss_km < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resulting miss distance cannot be negative".to_string()
        ));
    }

    if prop.resulting_pc < 0.0 || prop.resulting_pc > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resulting Pc must be between 0 and 1".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_agreement(_agr: &NegotiationAgreement) -> ExternResult<ValidateCallbackResult> {
    // Basic validation - signatures verified separately
    Ok(ValidateCallbackResult::Valid)
}
