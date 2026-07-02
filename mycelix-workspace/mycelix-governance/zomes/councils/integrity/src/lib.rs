// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Councils Integrity Zome
//! Defines entry types and validation for nested governance councils
//!
//! Philosophy: Holonic governance - councils within councils,
//! each reflecting both their own wisdom and that of their constituents.
//! The Mirror shows the fractal health of the whole.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A governance council - a nested group with collective agency
///
/// Councils can contain members and sub-councils, creating
/// a holonic structure where each part contains the whole.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Council {
    /// Unique council identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Council purpose/charter
    pub purpose: String,
    /// Council type (determines powers and processes)
    pub council_type: CouncilType,
    /// Parent council ID (if this is a sub-council)
    pub parent_council_id: Option<String>,
    /// Minimum phi score to participate
    pub phi_threshold: f64,
    /// Decision-making quorum (0-1)
    pub quorum: f64,
    /// Supermajority threshold for major decisions
    pub supermajority: f64,
    /// Can this council create sub-councils?
    pub can_spawn_children: bool,
    /// Maximum delegation depth
    pub max_delegation_depth: u8,
    /// Associated signing committee ID (for threshold-signed decisions)
    #[serde(default)]
    pub signing_committee_id: Option<String>,
    /// Council status
    pub status: CouncilStatus,
    /// Creation timestamp
    pub created_at: Timestamp,
    /// Last activity timestamp
    pub last_activity: Timestamp,
}

/// Types of councils with different powers
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type")]
pub enum CouncilType {
    /// Root governance council (constitutional powers)
    Root,
    /// Domain-specific council (e.g., Treasury, Technical)
    Domain { domain: String },
    /// Geographic/regional council
    Regional { region: String },
    /// Working group (temporary, task-focused)
    WorkingGroup {
        focus: String,
        expires: Option<Timestamp>,
    },
    /// Advisory council (no direct power, provides wisdom)
    Advisory,
    /// Emergency response council (elevated powers, time-limited)
    Emergency { expires: Timestamp },
}

/// Status of a council
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CouncilStatus {
    /// Active and operational
    Active,
    /// Dormant (no recent activity)
    Dormant,
    /// Dissolved
    Dissolved,
    /// Suspended pending review
    Suspended,
}

/// Default membership term: 365 days in microseconds.
/// Members must be re-elected/renewed after this period.
pub const DEFAULT_MEMBERSHIP_TERM_US: i64 = 365 * 24 * 3600 * 1_000_000;

/// Membership in a council
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CouncilMembership {
    /// Unique membership ID
    pub id: String,
    /// Council ID
    pub council_id: String,
    /// Member's DID
    pub member_did: String,
    /// Role in the council
    pub role: MemberRole,
    /// Current phi score (cached)
    pub phi_score: f64,
    /// Voting weight (derived from phi and role)
    pub voting_weight: f64,
    /// Can delegate to others?
    pub can_delegate: bool,
    /// Membership status
    pub status: MembershipStatus,
    /// Joined timestamp
    pub joined_at: Timestamp,
    /// Last participation
    pub last_participation: Timestamp,
    /// When this membership expires.
    /// None = legacy entry (treated as joined_at + DEFAULT_MEMBERSHIP_TERM_US).
    #[serde(default)]
    pub expires_at: Option<Timestamp>,
}

/// Roles within a council
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MemberRole {
    /// Full member with voting rights
    Member,
    /// Council facilitator (process authority, not decision authority)
    Facilitator,
    /// Council steward (administrative duties)
    Steward,
    /// Observer (can witness, cannot vote)
    Observer,
    /// Delegate from another council
    Delegate { from_council: String },
}

/// Membership status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MembershipStatus {
    Active,
    OnLeave,
    Suspended,
    Removed,
}

// ============================================================================
// HOLONIC MIRROR - NESTED GROUP SENSING
// ============================================================================

/// Holonic reflection of a council's collective state
///
/// Philosophy: The Mirror sees not just individual councils,
/// but the fractal health of nested governance.
/// Each holon reflects both itself and its children.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HolonicReflection {
    /// Unique reflection ID
    pub id: String,
    /// Council being reflected
    pub council_id: String,
    /// When this reflection was generated
    pub timestamp: Timestamp,
    /// Holon depth (0 = root)
    pub holon_depth: u8,

    // === MEMBER SENSING ===
    /// Total active members
    pub member_count: u64,
    /// Members with phi > threshold
    pub phi_qualified_count: u64,
    /// Average phi of members
    pub average_phi: f64,
    /// Phi distribution (min, p25, median, p75, max)
    pub phi_distribution: PhiDistribution,
    /// Member participation rate
    pub participation_rate: f64,

    // === DECISION HEALTH ===
    /// Recent decisions count
    pub decisions_last_30_days: u64,
    /// Decision consensus quality (average agreement)
    pub average_consensus: f64,
    /// Contentious decisions ratio
    pub contention_ratio: f64,
    /// Decisions implemented vs proposed
    pub implementation_rate: f64,

    // === HOLONIC HEALTH ===
    /// Child council count
    pub child_count: u64,
    /// Aggregate child health
    pub child_health: Option<AggregateChildHealth>,
    /// Coherence with parent (if sub-council)
    pub parent_coherence: Option<f64>,
    /// Cross-council collaboration score
    pub collaboration_score: f64,

    // === HARMONY ANALYSIS ===
    /// Which harmonies are present in council deliberation
    pub harmony_presence: Vec<HarmonyPresence>,
    /// Overall harmony coverage (0-1)
    pub harmony_coverage: f64,
    /// Absent harmonies (opportunities for growth)
    pub absent_harmonies: Vec<String>,

    // === VITALITY SIGNALS ===
    /// Is the council thriving, stable, or declining?
    pub vitality_trend: VitalityTrend,
    /// Risk factors identified
    pub risk_factors: Vec<RiskFactor>,
    /// Opportunities identified
    pub opportunities: Vec<String>,
    /// Overall council health score (0-1)
    pub health_score: f64,

    /// Human-readable summary
    pub summary: String,
}

/// Distribution of phi scores
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PhiDistribution {
    pub min: f64,
    pub p25: f64,
    pub median: f64,
    pub p75: f64,
    pub max: f64,
}

/// Aggregate health of child councils
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AggregateChildHealth {
    /// Number of healthy children
    pub healthy_count: u64,
    /// Number of struggling children
    pub struggling_count: u64,
    /// Number of dormant children
    pub dormant_count: u64,
    /// Average child health score
    pub average_health: f64,
    /// Children needing attention
    pub children_needing_attention: Vec<String>,
}

/// Presence of a harmony in council deliberation
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HarmonyPresence {
    pub harmony: String,
    pub presence: f64,
    pub trend: TrendDirection,
}

/// Direction of a trend
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Rising,
    Stable,
    Falling,
}

/// Council vitality trend
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VitalityTrend {
    /// Growing in health and activity
    Thriving,
    /// Maintaining steady state
    Stable,
    /// Showing signs of decline
    Declining,
    /// Critically low activity/health
    Critical,
}

/// Risk factor for council health
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RiskFactor {
    pub category: RiskCategory,
    pub severity: f64,
    pub description: String,
}

/// Categories of risk
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RiskCategory {
    /// Low participation
    Participation,
    /// Low phi scores
    Consciousness,
    /// Poor decision outcomes
    DecisionQuality,
    /// Missing harmonies
    HarmonyGap,
    /// Child councils struggling
    ChildHealth,
    /// Disconnection from parent
    ParentDisconnect,
    /// Power concentration
    PowerConcentration,
    /// Stagnation
    Stagnation,
}

// ============================================================================
// COUNCIL ACTIONS AND DECISIONS
// ============================================================================

/// A decision made by a council
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CouncilDecision {
    /// Unique decision ID
    pub id: String,
    /// Council that made the decision
    pub council_id: String,
    /// Linked proposal ID (if from formal proposal)
    pub proposal_id: Option<String>,
    /// Decision title
    pub title: String,
    /// Decision content
    pub content: String,
    /// Decision type
    pub decision_type: DecisionType,
    /// Votes for
    pub votes_for: u64,
    /// Votes against
    pub votes_against: u64,
    /// Abstentions
    pub abstentions: u64,
    /// Phi-weighted result
    pub phi_weighted_result: f64,
    /// Did it pass?
    pub passed: bool,
    /// Decision status
    pub status: DecisionStatus,
    /// Timestamp
    pub created_at: Timestamp,
    /// Executed timestamp
    pub executed_at: Option<Timestamp>,
}

// ============================================================================
// EMERGENCY COUNCIL LIMITS
// Thermodynamic metaphor: Adiabatic isolation has a maximum duration —
// the system must periodically equilibrate with the democratic thermal bath.
// ============================================================================

/// Maximum consecutive emergency sessions before mandatory cooldown.
pub const MAX_CONSECUTIVE_EMERGENCY_SESSIONS: u32 = 3;

/// Maximum duration of a single emergency session: 14 days in microseconds.
pub const MAX_EMERGENCY_SESSION_DURATION_US: i64 = 14 * 24 * 3600 * 1_000_000;

/// Mandatory cooldown after max consecutive sessions: 30 days in microseconds.
pub const EMERGENCY_COOLDOWN_DURATION_US: i64 = 30 * 24 * 3600 * 1_000_000;

/// Tracks an emergency council session for consecutive-session enforcement.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EmergencySession {
    /// Unique session identifier
    pub id: String,
    /// Council ID this session belongs to
    pub council_id: String,
    /// Session number in the consecutive chain (1, 2, or 3)
    pub session_number: u32,
    /// When this session started
    pub started_at: Timestamp,
    /// When this session expires
    pub expires_at: Timestamp,
    /// Previous session ID (for chain verification)
    pub preceding_session_id: Option<String>,
    /// Session status
    pub status: EmergencySessionStatus,
}

/// Status of an emergency session
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EmergencySessionStatus {
    /// Currently active
    Active,
    /// Expired naturally
    Expired,
    /// Extended to the next session (1→2 or 2→3)
    ExtendedToNext,
    /// Cooldown started (after session 3)
    CooldownStarted,
}

/// Mandatory cooldown period after max consecutive emergency sessions.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EmergencyCooldown {
    /// Unique cooldown identifier
    pub id: String,
    /// Council ID (or parent council scope)
    pub council_id: String,
    /// Last emergency session ID that triggered cooldown
    pub last_session_id: String,
    /// When cooldown started
    pub cooldown_started: Timestamp,
    /// When cooldown ends (started + 30 days)
    pub cooldown_ends: Timestamp,
    /// Whether a non-emergency council approved early renewal
    pub override_approved: bool,
}

/// Check emergency session creation invariants
pub fn check_create_emergency_session(session: &EmergencySession) -> Result<(), String> {
    if session.council_id.is_empty() {
        return Err("Council ID cannot be empty".into());
    }
    if session.session_number == 0 || session.session_number > MAX_CONSECUTIVE_EMERGENCY_SESSIONS {
        return Err(format!(
            "Session number must be 1-{}, got {}",
            MAX_CONSECUTIVE_EMERGENCY_SESSIONS, session.session_number
        ));
    }
    if session.session_number > 1 && session.preceding_session_id.is_none() {
        return Err("Sessions 2+ must reference the preceding session".into());
    }
    if session.session_number == 1 && session.preceding_session_id.is_some() {
        return Err("Session 1 must not have a preceding session".into());
    }
    // Verify duration doesn't exceed maximum
    let duration_us = session.expires_at.as_micros() as i64 - session.started_at.as_micros() as i64;
    if duration_us > MAX_EMERGENCY_SESSION_DURATION_US {
        return Err(format!(
            "Emergency session duration exceeds maximum of 14 days ({} us > {} us)",
            duration_us, MAX_EMERGENCY_SESSION_DURATION_US
        ));
    }
    if duration_us <= 0 {
        return Err("Emergency session must have positive duration".into());
    }
    Ok(())
}

/// Check cooldown creation invariants
pub fn check_create_emergency_cooldown(cooldown: &EmergencyCooldown) -> Result<(), String> {
    if cooldown.council_id.is_empty() {
        return Err("Council ID cannot be empty".into());
    }
    if cooldown.last_session_id.is_empty() {
        return Err("Last session ID cannot be empty".into());
    }
    let duration_us =
        cooldown.cooldown_ends.as_micros() as i64 - cooldown.cooldown_started.as_micros() as i64;
    if duration_us < EMERGENCY_COOLDOWN_DURATION_US {
        return Err(format!(
            "Cooldown must be at least 30 days ({} us < {} us)",
            duration_us, EMERGENCY_COOLDOWN_DURATION_US
        ));
    }
    Ok(())
}

/// Types of council decisions
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DecisionType {
    /// Standard operational decision
    Operational,
    /// Policy decision
    Policy,
    /// Resource allocation
    Resource,
    /// Member admission/removal
    Membership,
    /// Sub-council creation
    SubCouncil,
    /// Constitutional matter (requires supermajority)
    Constitutional,
}

/// Status of a decision
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DecisionStatus {
    /// Pending vote
    Pending,
    /// Approved
    Approved,
    /// Rejected
    Rejected,
    /// Executed
    Executed,
    /// Vetoed by parent council
    Vetoed,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Council(Council),
    CouncilMembership(CouncilMembership),
    HolonicReflection(HolonicReflection),
    CouncilDecision(CouncilDecision),
    EmergencySession(EmergencySession),
    EmergencyCooldown(EmergencyCooldown),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Root anchor to all councils
    AllCouncils,
    /// Parent council to child councils
    ParentToChild,
    /// Council to its members
    CouncilToMember,
    /// Member to their councils
    MemberToCouncil,
    /// Council to its decisions
    CouncilToDecision,
    /// Council to its holonic reflections
    CouncilToReflection,
    /// Council type anchor to councils
    TypeToCouncil,
    /// Council to emergency sessions
    CouncilToEmergencySession,
    /// Council to emergency cooldown
    CouncilToCooldown,
}

// ============================================================================
// PURE CHECK FUNCTIONS (testable without HDI host)
// ============================================================================

/// Check council creation invariants (pure function)
pub fn check_create_council(council: &Council) -> Result<(), String> {
    // Validate ID not empty
    if council.id.is_empty() {
        return Err("Council ID cannot be empty".into());
    }

    // Validate name not empty
    if council.name.is_empty() {
        return Err("Council name cannot be empty".into());
    }

    // Validate purpose not empty
    if council.purpose.is_empty() {
        return Err("Council purpose cannot be empty".into());
    }

    // Validate phi threshold in range
    if council.phi_threshold < 0.0 || council.phi_threshold > 1.0 {
        return Err("Phi threshold must be between 0 and 1".into());
    }

    // Validate quorum in range
    if council.quorum < 0.0 || council.quorum > 1.0 {
        return Err("Quorum must be between 0 and 1".into());
    }

    // Validate supermajority in range (typically > 0.5)
    if council.supermajority < 0.5 || council.supermajority > 1.0 {
        return Err("Supermajority must be between 0.5 and 1".into());
    }

    // Root councils cannot have parents
    if matches!(council.council_type, CouncilType::Root) && council.parent_council_id.is_some() {
        return Err("Root council cannot have a parent".into());
    }

    Ok(())
}

/// Check council update invariants (pure function)
pub fn check_update_council(original: &Council, updated: &Council) -> Result<(), String> {
    // Cannot change ID
    if updated.id != original.id {
        return Err("Cannot change council ID".into());
    }

    // Cannot change council type
    if updated.council_type != original.council_type {
        return Err("Cannot change council type".into());
    }

    // Cannot change parent
    if updated.parent_council_id != original.parent_council_id {
        return Err("Cannot change parent council".into());
    }

    Ok(())
}

/// Check membership creation invariants (pure function)
pub fn check_create_membership(membership: &CouncilMembership) -> Result<(), String> {
    // Validate member DID
    if !membership.member_did.starts_with("did:") {
        return Err("Member must be a valid DID".into());
    }

    // Validate council ID not empty
    if membership.council_id.is_empty() {
        return Err("Council ID cannot be empty".into());
    }

    // Validate phi score in range
    if membership.phi_score < 0.0 || membership.phi_score > 1.0 {
        return Err("Phi score must be between 0 and 1".into());
    }

    Ok(())
}

/// Check holonic reflection creation invariants (pure function)
pub fn check_create_reflection(reflection: &HolonicReflection) -> Result<(), String> {
    // Validate council ID not empty
    if reflection.council_id.is_empty() {
        return Err("Council ID cannot be empty".into());
    }

    // Validate health score in range
    if reflection.health_score < 0.0 || reflection.health_score > 1.0 {
        return Err("Health score must be between 0 and 1".into());
    }

    // Validate participation rate in range
    if reflection.participation_rate < 0.0 || reflection.participation_rate > 1.0 {
        return Err("Participation rate must be between 0 and 1".into());
    }

    // Validate harmony coverage in range
    if reflection.harmony_coverage < 0.0 || reflection.harmony_coverage > 1.0 {
        return Err("Harmony coverage must be between 0 and 1".into());
    }

    Ok(())
}

/// Check decision creation invariants (pure function)
pub fn check_create_decision(decision: &CouncilDecision) -> Result<(), String> {
    // Validate council ID not empty
    if decision.council_id.is_empty() {
        return Err("Council ID cannot be empty".into());
    }

    // Validate title not empty
    if decision.title.is_empty() {
        return Err("Decision title cannot be empty".into());
    }

    // Validate phi-weighted result in range
    if decision.phi_weighted_result < 0.0 || decision.phi_weighted_result > 1.0 {
        return Err("Phi-weighted result must be between 0 and 1".into());
    }

    Ok(())
}

/// HDI 0.7 single validation callback
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Council(council) => validate_create_council(action, council),
                EntryTypes::CouncilMembership(membership) => {
                    validate_create_membership(action, membership)
                }
                EntryTypes::HolonicReflection(reflection) => {
                    validate_create_reflection(action, reflection)
                }
                EntryTypes::CouncilDecision(decision) => validate_create_decision(action, decision),
                EntryTypes::EmergencySession(session) => {
                    match check_create_emergency_session(&session) {
                        Ok(()) => Ok(ValidateCallbackResult::Valid),
                        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
                    }
                }
                EntryTypes::EmergencyCooldown(cooldown) => {
                    match check_create_emergency_cooldown(&cooldown) {
                        Ok(()) => Ok(ValidateCallbackResult::Valid),
                        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
                    }
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Council(council) => {
                    validate_update_council(action, council, original_action_hash)
                }
                EntryTypes::CouncilMembership(membership) => {
                    validate_update_membership(action, membership)
                }
                EntryTypes::HolonicReflection(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CouncilDecision(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::EmergencySession(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::EmergencyCooldown(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate council creation
fn validate_create_council(
    _action: Create,
    council: Council,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_council(&council) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

/// Validate council update
fn validate_update_council(
    _action: Update,
    council: Council,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    // Get original council
    let original_record = must_get_valid_record(original_action_hash)?;
    let original_council: Council = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original council not found".into()
        )))?;

    match check_update_council(&original_council, &council) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

/// Validate membership creation
fn validate_create_membership(
    _action: Create,
    membership: CouncilMembership,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_membership(&membership) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

/// Validate membership update
fn validate_update_membership(
    _action: Update,
    _membership: CouncilMembership,
) -> ExternResult<ValidateCallbackResult> {
    // Memberships can be updated (status changes, phi updates, etc.)
    Ok(ValidateCallbackResult::Valid)
}

/// Validate holonic reflection creation
fn validate_create_reflection(
    _action: Create,
    reflection: HolonicReflection,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_reflection(&reflection) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

/// Validate decision creation
fn validate_create_decision(
    _action: Create,
    decision: CouncilDecision,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_decision(&decision) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    fn make_council() -> Council {
        Council {
            id: "council-1".into(),
            name: "Test Council".into(),
            purpose: "Testing governance".into(),
            council_type: CouncilType::Domain {
                domain: "test".into(),
            },
            parent_council_id: None,
            phi_threshold: 0.5,
            quorum: 0.6,
            supermajority: 0.67,
            can_spawn_children: true,
            max_delegation_depth: 3,
            signing_committee_id: None,
            status: CouncilStatus::Active,
            created_at: ts(1000),
            last_activity: ts(2000),
        }
    }

    fn make_membership() -> CouncilMembership {
        CouncilMembership {
            id: "mem-1".into(),
            council_id: "council-1".into(),
            member_did: "did:key:z6Mk123".into(),
            role: MemberRole::Member,
            phi_score: 0.7,
            voting_weight: 1.0,
            can_delegate: true,
            status: MembershipStatus::Active,
            joined_at: ts(1000),
            last_participation: ts(2000),
            expires_at: Some(ts(1000 + DEFAULT_MEMBERSHIP_TERM_US)),
        }
    }

    fn make_reflection() -> HolonicReflection {
        HolonicReflection {
            id: "ref-1".into(),
            council_id: "council-1".into(),
            timestamp: ts(3000),
            holon_depth: 0,
            member_count: 10,
            phi_qualified_count: 8,
            average_phi: 0.6,
            phi_distribution: PhiDistribution {
                min: 0.1,
                p25: 0.4,
                median: 0.6,
                p75: 0.8,
                max: 0.95,
            },
            participation_rate: 0.8,
            decisions_last_30_days: 5,
            average_consensus: 0.75,
            contention_ratio: 0.1,
            implementation_rate: 0.9,
            child_count: 2,
            child_health: None,
            parent_coherence: None,
            collaboration_score: 0.7,
            harmony_presence: vec![],
            harmony_coverage: 0.6,
            absent_harmonies: vec![],
            vitality_trend: VitalityTrend::Stable,
            risk_factors: vec![],
            opportunities: vec![],
            health_score: 0.8,
            summary: "Test reflection".into(),
        }
    }

    fn make_decision() -> CouncilDecision {
        CouncilDecision {
            id: "dec-1".into(),
            council_id: "council-1".into(),
            proposal_id: None,
            title: "Test Decision".into(),
            content: "Decide something".into(),
            decision_type: DecisionType::Operational,
            votes_for: 7,
            votes_against: 2,
            abstentions: 1,
            phi_weighted_result: 0.75,
            passed: true,
            status: DecisionStatus::Approved,
            created_at: ts(4000),
            executed_at: None,
        }
    }

    // --- Council creation tests ---

    #[test]
    fn test_valid_council_accepted() {
        let council = make_council();
        assert!(check_create_council(&council).is_ok());
    }

    #[test]
    fn test_council_id_required() {
        let mut council = make_council();
        council.id = "".into();
        let err = check_create_council(&council).unwrap_err();
        assert!(err.contains("Council ID cannot be empty"));
    }

    #[test]
    fn test_council_name_required() {
        let mut council = make_council();
        council.name = "".into();
        let err = check_create_council(&council).unwrap_err();
        assert!(err.contains("Council name cannot be empty"));
    }

    #[test]
    fn test_council_purpose_required() {
        let mut council = make_council();
        council.purpose = "".into();
        let err = check_create_council(&council).unwrap_err();
        assert!(err.contains("Council purpose cannot be empty"));
    }

    #[test]
    fn test_council_phi_threshold_range() {
        let mut council = make_council();
        council.phi_threshold = 1.5;
        let err = check_create_council(&council).unwrap_err();
        assert!(err.contains("Phi threshold must be between 0 and 1"));

        council.phi_threshold = -0.1;
        let err = check_create_council(&council).unwrap_err();
        assert!(err.contains("Phi threshold must be between 0 and 1"));
    }

    #[test]
    fn test_council_quorum_range() {
        let mut council = make_council();
        council.quorum = 1.1;
        let err = check_create_council(&council).unwrap_err();
        assert!(err.contains("Quorum must be between 0 and 1"));

        council.quorum = -0.01;
        let err = check_create_council(&council).unwrap_err();
        assert!(err.contains("Quorum must be between 0 and 1"));
    }

    #[test]
    fn test_council_supermajority_range() {
        let mut council = make_council();
        council.supermajority = 0.4;
        let err = check_create_council(&council).unwrap_err();
        assert!(err.contains("Supermajority must be between 0.5 and 1"));
    }

    #[test]
    fn test_root_council_no_parent() {
        let mut council = make_council();
        council.council_type = CouncilType::Root;
        council.parent_council_id = Some("parent-1".into());
        let err = check_create_council(&council).unwrap_err();
        assert!(err.contains("Root council cannot have a parent"));
    }

    // --- Council update tests ---

    #[test]
    fn test_council_update_id_immutable() {
        let original = make_council();
        let mut updated = make_council();
        updated.id = "different-id".into();
        let err = check_update_council(&original, &updated).unwrap_err();
        assert!(err.contains("Cannot change council ID"));
    }

    // --- Membership tests ---

    #[test]
    fn test_membership_member_must_be_did() {
        let mut membership = make_membership();
        membership.member_did = "not-a-did".into();
        let err = check_create_membership(&membership).unwrap_err();
        assert!(err.contains("Member must be a valid DID"));
    }

    // --- Reflection tests ---

    #[test]
    fn test_reflection_health_score_range() {
        let mut reflection = make_reflection();
        reflection.health_score = 1.5;
        let err = check_create_reflection(&reflection).unwrap_err();
        assert!(err.contains("Health score must be between 0 and 1"));
    }

    // --- Decision tests ---

    #[test]
    fn test_decision_phi_weighted_result_range() {
        let mut decision = make_decision();
        decision.phi_weighted_result = -0.1;
        let err = check_create_decision(&decision).unwrap_err();
        assert!(err.contains("Phi-weighted result must be between 0 and 1"));
    }

    // --- Emergency session tests ---

    fn make_emergency_session() -> EmergencySession {
        EmergencySession {
            id: "es-1".into(),
            council_id: "council-1".into(),
            session_number: 1,
            started_at: ts(1_000_000),
            expires_at: ts(1_000_000 + 7 * 24 * 3600 * 1_000_000), // 7 days
            preceding_session_id: None,
            status: EmergencySessionStatus::Active,
        }
    }

    #[test]
    fn test_valid_emergency_session() {
        assert!(check_create_emergency_session(&make_emergency_session()).is_ok());
    }

    #[test]
    fn test_emergency_session_number_range() {
        let mut es = make_emergency_session();
        es.session_number = 0;
        assert!(check_create_emergency_session(&es)
            .unwrap_err()
            .contains("Session number must be 1-3"));

        es.session_number = 4;
        assert!(check_create_emergency_session(&es)
            .unwrap_err()
            .contains("Session number must be 1-3"));
    }

    #[test]
    fn test_emergency_session_preceding_chain() {
        // Session 2 must have preceding
        let mut es = make_emergency_session();
        es.session_number = 2;
        es.preceding_session_id = None;
        assert!(check_create_emergency_session(&es)
            .unwrap_err()
            .contains("must reference the preceding"));

        // Session 2 with preceding is OK
        es.preceding_session_id = Some("es-1".into());
        assert!(check_create_emergency_session(&es).is_ok());

        // Session 1 must NOT have preceding
        let mut es1 = make_emergency_session();
        es1.session_number = 1;
        es1.preceding_session_id = Some("bogus".into());
        assert!(check_create_emergency_session(&es1)
            .unwrap_err()
            .contains("must not have a preceding"));
    }

    #[test]
    fn test_emergency_session_max_duration() {
        let mut es = make_emergency_session();
        // 15 days exceeds 14-day max
        es.expires_at = ts(es.started_at.as_micros() as i64 + 15 * 24 * 3600 * 1_000_000);
        assert!(check_create_emergency_session(&es)
            .unwrap_err()
            .contains("exceeds maximum"));
    }

    #[test]
    fn test_emergency_cooldown_duration() {
        let cooldown = EmergencyCooldown {
            id: "ec-1".into(),
            council_id: "council-1".into(),
            last_session_id: "es-3".into(),
            cooldown_started: ts(1_000_000),
            cooldown_ends: ts(1_000_000 + EMERGENCY_COOLDOWN_DURATION_US),
            override_approved: false,
        };
        assert!(check_create_emergency_cooldown(&cooldown).is_ok());

        // Too short cooldown rejected
        let mut short = cooldown.clone();
        short.cooldown_ends = ts(1_000_000 + 29 * 24 * 3600 * 1_000_000); // 29 days < 30
        assert!(check_create_emergency_cooldown(&short)
            .unwrap_err()
            .contains("at least 30 days"));
    }

    #[test]
    fn test_membership_term_is_365_days() {
        assert_eq!(
            DEFAULT_MEMBERSHIP_TERM_US,
            365 * 24 * 3600 * 1_000_000_i64
        );
    }

    #[test]
    fn test_membership_expires_at_default() {
        let m = make_membership();
        assert!(m.expires_at.is_some());
        let expected = ts(1000 + DEFAULT_MEMBERSHIP_TERM_US);
        assert_eq!(m.expires_at.unwrap(), expected);
    }

    #[test]
    fn test_emergency_constants() {
        assert_eq!(MAX_CONSECUTIVE_EMERGENCY_SESSIONS, 3);
        assert_eq!(
            MAX_EMERGENCY_SESSION_DURATION_US,
            14 * 24 * 3600 * 1_000_000
        );
        assert_eq!(EMERGENCY_COOLDOWN_DURATION_US, 30 * 24 * 3600 * 1_000_000);
    }

    // --- Serde round-trip tests (verify TS SDK compatibility) ---

    #[test]
    fn test_council_type_serde_internally_tagged() {
        // Unit variant: TS sends {"type":"Root"}
        let json = serde_json::to_string(&CouncilType::Root).unwrap();
        assert_eq!(json, r#"{"type":"Root"}"#);
        let rt: CouncilType = serde_json::from_str(&json).unwrap();
        assert_eq!(rt, CouncilType::Root);

        // Struct variant: TS sends {"type":"Domain","domain":"water"}
        let json = serde_json::to_string(&CouncilType::Domain {
            domain: "water".into(),
        })
        .unwrap();
        assert!(json.contains(r#""type":"Domain""#));
        assert!(json.contains(r#""domain":"water""#));
        let rt: CouncilType = serde_json::from_str(&json).unwrap();
        assert_eq!(
            rt,
            CouncilType::Domain {
                domain: "water".into()
            }
        );

        // WorkingGroup with optional expires
        let json = serde_json::to_string(&CouncilType::WorkingGroup {
            focus: "testing".into(),
            expires: None,
        })
        .unwrap();
        assert!(json.contains(r#""type":"WorkingGroup""#));
        let rt: CouncilType = serde_json::from_str(&json).unwrap();
        assert_eq!(
            rt,
            CouncilType::WorkingGroup {
                focus: "testing".into(),
                expires: None
            }
        );
    }

    #[test]
    fn test_member_role_serde_externally_tagged() {
        // Unit variant: TS sends bare string "Member"
        let json = serde_json::to_string(&MemberRole::Member).unwrap();
        assert_eq!(json, r#""Member""#);

        // Struct variant: TS sends {"Delegate":{"from_council":"council-1"}}
        let json = serde_json::to_string(&MemberRole::Delegate {
            from_council: "council-1".into(),
        })
        .unwrap();
        assert!(json.contains(r#""Delegate""#));
        assert!(json.contains(r#""from_council":"council-1""#));
    }
}
